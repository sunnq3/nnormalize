import type { BackgroundModel } from './background';

/** A spectral peak as first detected on the averaged spectrum, in bin units. */
export interface RawPeak {
  bx: number; // shifted bin x of the maximum
  by: number;
  u: number; // cycles / pixel, from the bin (refined later)
  v: number;
  z: number; // robust z-score of the excess
  excessDb: number; // 10·log10(power / background) at the maximum
  widthBins: number; // equivalent radius of the connected blob
  area: number; // blob area in bins
}

export interface PeakDetectOptions {
  zThreshold?: number; // default 6
  minRadiusBins?: number; // default 6: ignore the DC neighbourhood
  maxPeaks?: number; // default 64
  localMaxSize?: number; // odd, default 5
}

/**
 * Finds isolated spikes in the Welch spectrum: local maxima whose robust
 * z-score exceeds the threshold, one per connected blob of elevated bins.
 * The spectrum of a real image is conjugate-symmetric, so each spike has a
 * twin at (-u, -v); only the upper-half-plane representative is returned.
 */
export function detectPeaks(model: BackgroundModel, opts: PeakDetectOptions = {}): RawPeak[] {
  const n = model.size;
  const c = n >> 1;
  const zT = opts.zThreshold ?? 6;
  const minR = opts.minRadiusBins ?? 6;
  const maxPeaks = opts.maxPeaks ?? 64;
  const lm = opts.localMaxSize ?? 5;
  const lr = lm >> 1;
  const z = model.z;
  const lp = model.logPower;

  // connected components of z > zT/2 (4-connectivity)
  const labels = new Int32Array(n * n);
  const half = zT / 2;
  let nLabels = 0;
  const stack: number[] = [];
  const areas: number[] = [0];
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== 0 || z[i] <= half) continue;
    nLabels++;
    let area = 0;
    labels[i] = nLabels;
    stack.push(i);
    while (stack.length) {
      const p = stack.pop()!;
      area++;
      const px = p % n;
      const py = (p - px) / n;
      const nb = [p - 1, p + 1, p - n, p + n];
      const ok = [px > 0, px < n - 1, py > 0, py < n - 1];
      for (let k = 0; k < 4; k++) {
        const q = nb[k];
        if (ok[k] && labels[q] === 0 && z[q] > half) {
          labels[q] = nLabels;
          stack.push(q);
        }
      }
    }
    areas.push(area);
  }

  // best candidate per blob
  const best = new Map<number, number>();
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      if (z[i] <= zT) continue;
      const dx = x - c;
      const dy = y - c;
      if (Math.hypot(dx, dy) < minR) continue;
      // local maximum of log power
      let isMax = true;
      const v = lp[i];
      for (let yy = Math.max(0, y - lr); yy <= Math.min(n - 1, y + lr) && isMax; yy++) {
        for (let xx = Math.max(0, x - lr); xx <= Math.min(n - 1, x + lr); xx++) {
          if ((xx !== x || yy !== y) && lp[yy * n + xx] > v) {
            isMax = false;
            break;
          }
        }
      }
      if (!isMax) continue;
      const lab = labels[i];
      if (lab === 0) continue;
      const prev = best.get(lab);
      if (prev === undefined || z[i] > z[prev]) best.set(lab, i);
    }
  }

  const peaks: RawPeak[] = [];
  for (const [lab, i] of best) {
    const bx = i % n;
    const by = (i - bx) / n;
    const area = areas[lab];
    peaks.push({
      bx,
      by,
      u: (bx - c) / n,
      v: (by - c) / n,
      z: z[i],
      excessDb: 10 * model.residual[i],
      widthBins: Math.max(1, Math.sqrt(area / Math.PI)),
      area,
    });
  }
  peaks.sort((a, b) => b.z - a.z);

  // keep one of each conjugate pair: upper half-plane (v < 0), or v == 0 with u > 0
  const kept: RawPeak[] = [];
  const tol = 2;
  for (const p of peaks) {
    const twin = kept.find((q) => Math.abs(q.bx + p.bx - 2 * c) <= tol && Math.abs(q.by + p.by - 2 * c) <= tol);
    const dup = kept.find((q) => Math.abs(q.bx - p.bx) <= tol && Math.abs(q.by - p.by) <= tol);
    if (twin || dup) continue;
    if (p.v > 0 || (p.v === 0 && p.u < 0)) {
      kept.push({ ...p, bx: 2 * c - p.bx, by: 2 * c - p.by, u: -p.u, v: -p.v });
    } else {
      kept.push(p);
    }
    if (kept.length >= maxPeaks) break;
  }
  return kept;
}

export interface HarmonicGroup {
  fundamental: number; // index into the peak list
  members: number[]; // indices, including the fundamental, ordered by harmonic number
  orders: number[]; // harmonic number per member
}

/**
 * Groups peaks that lie at integer multiples of a common fundamental along the
 * same direction (a square wave, a halftone screen's harmonics). Tolerance is
 * in bins and grows with the harmonic order because the fundamental's own
 * position error is multiplied.
 */
export function groupHarmonics(peaks: { u: number; v: number }[], size: number, tolBins = 1.5, maxOrder = 8): HarmonicGroup[] {
  const n = peaks.length;
  const used = new Array<boolean>(n).fill(false);
  // process by increasing |f| so fundamentals are tried first
  const order = peaks.map((p, i) => ({ i, r: Math.hypot(p.u, p.v) })).sort((a, b) => a.r - b.r);
  const groups: HarmonicGroup[] = [];
  for (const { i } of order) {
    if (used[i]) continue;
    const f = peaks[i];
    const members = [i];
    const orders = [1];
    for (let k = 2; k <= maxOrder; k++) {
      const tu = f.u * k;
      const tv = f.v * k;
      let bestJ = -1;
      let bestD = Infinity;
      for (let j = 0; j < n; j++) {
        if (used[j] || j === i) continue;
        const p = peaks[j];
        // the twin of a harmonic may be the one in our half-plane; test both signs
        const d1 = Math.hypot((p.u - tu) * size, (p.v - tv) * size);
        const d2 = Math.hypot((p.u + tu) * size, (p.v + tv) * size);
        const d = Math.min(d1, d2);
        if (d < bestD) {
          bestD = d;
          bestJ = j;
        }
      }
      if (bestJ >= 0 && bestD <= tolBins * Math.sqrt(k)) {
        members.push(bestJ);
        orders.push(k);
        used[bestJ] = true;
      }
    }
    used[i] = true;
    groups.push({ fundamental: i, members, orders });
  }
  return groups;
}
