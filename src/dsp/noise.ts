/**
 * Noise analysis and normalisation.
 *
 * The estimator is the classic Laplacian one: convolve with
 *   [ 1 -2  1 ]
 *   [-2  4 -2 ]
 *   [ 1 -2  1 ]
 * whose response to white noise of sigma s has standard deviation 6 s, and
 * take a robust (MAD) scale of the response in each block so edges and
 * texture, which are sparse, don't inflate it the way a mean would.
 */

import { makePlane, type Plane } from './plane';
import { rng, gaussian } from '../synth/generate';

export interface NoiseMap {
  block: number; // block size in pixels
  bw: number; // blocks across
  bh: number; // blocks down
  sigma: Float32Array; // per-block noise sigma, plane units
  mean: Float32Array; // per-block mean intensity
}

export function noiseMap(p: Plane, block = 32): NoiseMap {
  const { w, h } = p;
  const bw = Math.max(1, Math.floor(w / block));
  const bh = Math.max(1, Math.floor(h / block));
  const sigma = new Float32Array(bw * bh);
  const mean = new Float32Array(bw * bh);
  const d = p.data;
  const buf = new Float64Array(block * block);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const x0 = bx * block;
      const y0 = by * block;
      let n = 0;
      let m = 0;
      const y1 = Math.min(h - 1, y0 + block);
      const x1 = Math.min(w - 1, x0 + block);
      for (let y = Math.max(1, y0); y < y1; y++) {
        for (let x = Math.max(1, x0); x < x1; x++) {
          const i = y * w + x;
          const l =
            d[i - w - 1] - 2 * d[i - w] + d[i - w + 1] - 2 * d[i - 1] + 4 * d[i] - 2 * d[i + 1] + d[i + w - 1] - 2 * d[i + w] + d[i + w + 1];
          buf[n++] = Math.abs(l);
          m += d[i];
        }
      }
      const seg = buf.subarray(0, n);
      seg.sort();
      const med = n ? (n & 1 ? seg[n >> 1] : 0.5 * (seg[(n >> 1) - 1] + seg[n >> 1])) : 0;
      sigma[by * bw + bx] = (1.4826 * med) / 6;
      mean[by * bw + bx] = n ? m / n : 0;
    }
  }
  return { block, bw, bh, sigma, mean };
}

export interface NoiseModel {
  bins: number;
  /** sigma expected at the centre of each intensity bin (0..1) */
  curve: Float32Array;
  globalSigma: number; // median block sigma
}

/** Expected noise as a function of intensity: a robust per-bin median of the block sigmas. */
export function noiseModel(map: NoiseMap, bins = 16): NoiseModel {
  const lists: number[][] = Array.from({ length: bins }, () => []);
  const all: number[] = [];
  for (let i = 0; i < map.sigma.length; i++) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(map.mean[i] * bins)));
    lists[b].push(map.sigma[i]);
    all.push(map.sigma[i]);
  }
  all.sort((a, b) => a - b);
  const globalSigma = all.length ? all[all.length >> 1] : 0;
  const curve = new Float32Array(bins);
  const have = new Uint8Array(bins);
  for (let b = 0; b < bins; b++) {
    const l = lists[b];
    if (l.length >= 4) {
      l.sort((a, c) => a - c);
      curve[b] = l[l.length >> 1];
      have[b] = 1;
    }
  }
  // fill gaps from the nearest populated bin
  for (let b = 0; b < bins; b++) {
    if (have[b]) continue;
    let best = -1;
    let bestD = Infinity;
    for (let c = 0; c < bins; c++) {
      if (have[c] && Math.abs(c - b) < bestD) {
        bestD = Math.abs(c - b);
        best = c;
      }
    }
    curve[b] = best >= 0 ? curve[best] : globalSigma;
  }
  return { bins, curve, globalSigma };
}

export function expectedSigma(model: NoiseModel, intensity: number): number {
  const t = Math.min(model.bins - 1, Math.max(0, intensity * model.bins - 0.5));
  const b0 = Math.floor(t);
  const b1 = Math.min(model.bins - 1, b0 + 1);
  const f = t - b0;
  return model.curve[b0] * (1 - f) + model.curve[b1] * f;
}

export interface NoiseDiscrepancy {
  /** per-block 20·log10(sigma / expected), smoothed with a 3x3 median over blocks */
  ratioDb: Float32Array;
  noisierFraction: number; // blocks above +thresholdDb
  smootherFraction: number; // blocks below -thresholdDb
  thresholdDb: number;
}

export function noiseDiscrepancy(map: NoiseMap, model: NoiseModel, thresholdDb = 3): NoiseDiscrepancy {
  const n = map.sigma.length;
  const raw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const e = Math.max(1e-6, expectedSigma(model, map.mean[i]));
    raw[i] = 20 * Math.log10(Math.max(1e-6, map.sigma[i]) / e);
  }
  const ratioDb = new Float32Array(n);
  const buf: number[] = [];
  for (let by = 0; by < map.bh; by++) {
    for (let bx = 0; bx < map.bw; bx++) {
      buf.length = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const x = bx + dx;
          const y = by + dy;
          if (x >= 0 && x < map.bw && y >= 0 && y < map.bh) buf.push(raw[y * map.bw + x]);
        }
      }
      buf.sort((a, b) => a - b);
      ratioDb[by * map.bw + bx] = buf[buf.length >> 1];
    }
  }
  let up = 0;
  let down = 0;
  for (let i = 0; i < n; i++) {
    if (ratioDb[i] > thresholdDb) up++;
    else if (ratioDb[i] < -thresholdDb) down++;
  }
  return { ratioDb, noisierFraction: up / n, smootherFraction: down / n, thresholdDb };
}

/** Bilinear upsample of a per-block map to a full-resolution plane. */
export function upsampleBlockMap(values: Float32Array, bw: number, bh: number, block: number, w: number, h: number): Plane {
  const out = makePlane(w, h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(bh - 1, Math.max(0, (y + 0.5) / block - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(bh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(bw - 1, Math.max(0, (x + 0.5) / block - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(bw - 1, x0 + 1);
      const tx = fx - x0;
      const a = values[y0 * bw + x0] * (1 - tx) + values[y0 * bw + x1] * tx;
      const b = values[y1 * bw + x0] * (1 - tx) + values[y1 * bw + x1] * tx;
      out.data[y * w + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

/** Separable box blur of radius r (window 2r+1), edge-clamped. */
export function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[row + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = s / (2 * r + 1);
      s += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / (2 * r + 1);
      s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

/**
 * Guided filter (He, Sun, Tang) with the plane as its own guide: an
 * edge-preserving smoother whose strength is set by eps (≈ noise variance).
 */
export function guidedFilter(p: Plane, r: number, eps: number): Plane {
  const { w, h } = p;
  const I = p.data;
  const n = w * h;
  const meanI = boxBlur(I, w, h, r);
  const II = new Float32Array(n);
  for (let i = 0; i < n; i++) II[i] = I[i] * I[i];
  const meanII = boxBlur(II, w, h, r);
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const varI = meanII[i] - meanI[i] * meanI[i];
    a[i] = varI / (varI + eps);
    b[i] = meanI[i] - a[i] * meanI[i];
  }
  const meanA = boxBlur(a, w, h, r);
  const meanB = boxBlur(b, w, h, r);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = meanA[i] * I[i] + meanB[i];
  return { w, h, data: out };
}

export interface NoiseNormaliseOptions {
  thresholdDb?: number; // only act outside ±threshold (default 1.5)
  radius?: number; // guided filter radius (default 2)
}

/**
 * Denoise toward the model where blocks are noisier than expected. Two guided
 * filters of increasing strength are tried; for each block the weakest one
 * that can reach the expected sigma is blended in by exactly the amount that
 * lands there, measured rather than assumed. Returns a delta plane and the
 * total blend map (0..1) that was used.
 */
export function denoiseDelta(p: Plane, map: NoiseMap, model: NoiseModel, opts: NoiseNormaliseOptions = {}): { delta: Plane; blend: Plane } {
  const thr = opts.thresholdDb ?? 1.5;
  const r = opts.radius ?? 2;
  const nBlocks = map.sigma.length;
  const need = new Uint8Array(nBlocks);
  let any = false;
  for (let i = 0; i < nBlocks; i++) {
    const e = Math.max(1e-6, expectedSigma(model, map.mean[i]));
    if (20 * Math.log10(Math.max(1e-6, map.sigma[i]) / e) > thr) {
      need[i] = 1;
      any = true;
    }
  }
  const delta = makePlane(p.w, p.h);
  const blendTotal = new Float32Array(nBlocks);
  if (!any) return { delta, blend: upsampleBlockMap(blendTotal, map.bw, map.bh, map.block, p.w, p.h) };

  const g = model.globalSigma;
  const candidates = [
    { r, eps: (2 * g) ** 2 + 1e-8 },
    { r: r + 1, eps: (6 * g) ** 2 + 1e-8 },
    { r: r + 2, eps: (14 * g) ** 2 + 1e-8 },
  ];
  const remaining = new Uint8Array(need);
  for (let ci = 0; ci < candidates.length; ci++) {
    let left = 0;
    for (let i = 0; i < nBlocks; i++) left += remaining[i];
    if (!left) break;
    const cand = candidates[ci];
    const f = guidedFilter(p, cand.r, cand.eps);
    const fmap = noiseMap(f, map.block);
    const b = new Float32Array(nBlocks);
    const last = ci === candidates.length - 1;
    for (let i = 0; i < nBlocks; i++) {
      if (!remaining[i]) continue;
      const e = Math.max(1e-6, expectedSigma(model, map.mean[i]));
      const s0 = map.sigma[i];
      const sf = fmap.sigma[i];
      if (sf <= e || last) {
        // noise of I + b(f - I) ≈ (1-b)·s0 + b·sf  (the two are correlated)
        const bb = s0 - sf > 1e-9 ? (s0 - e) / (s0 - sf) : 1;
        b[i] = Math.min(1, Math.max(0, bb));
        blendTotal[i] = Math.max(blendTotal[i], b[i]);
        remaining[i] = 0;
      }
    }
    const bm = upsampleBlockMap(b, map.bw, map.bh, map.block, p.w, p.h);
    for (let i = 0; i < delta.data.length; i++) delta.data[i] += (f.data[i] - p.data[i]) * bm.data[i];
  }
  return { delta, blend: upsampleBlockMap(blendTotal, map.bw, map.bh, map.block, p.w, p.h) };
}

/**
 * Add matched grain where blocks are smoother than expected (a previous
 * denoiser's footprint), so the output's noise is level across the frame.
 * Deterministic for a given seed.
 */
export function regrainDelta(p: Plane, map: NoiseMap, model: NoiseModel, opts: NoiseNormaliseOptions & { seed?: number } = {}): { delta: Plane; sigmaAdd: Plane } {
  const thr = opts.thresholdDb ?? 1.5;
  const add = new Float32Array(map.sigma.length);
  let any = false;
  for (let i = 0; i < map.sigma.length; i++) {
    const e = Math.max(1e-6, expectedSigma(model, map.mean[i]));
    const ratioDb = 20 * Math.log10(Math.max(1e-6, map.sigma[i]) / e);
    if (ratioDb < -thr) {
      add[i] = Math.sqrt(Math.max(0, e * e - map.sigma[i] * map.sigma[i]));
      any = true;
    }
  }
  const sigmaAdd = upsampleBlockMap(add, map.bw, map.bh, map.block, p.w, p.h);
  const delta = makePlane(p.w, p.h);
  if (!any) return { delta, sigmaAdd };
  const g = gaussian(rng(opts.seed ?? 0x5eed));
  for (let i = 0; i < delta.data.length; i++) delta.data[i] = sigmaAdd.data[i] > 0 ? sigmaAdd.data[i] * g() : 0;
  return { delta, sigmaAdd };
}

/** Noise sigma implied by the high-frequency floor of a Welch spectrum. */
export function noiseFromSpectrum(power: Float64Array, size: number, winSum: number, winSumSq: number, rMin = 0.4, rMax = 0.5): number {
  const c = size >> 1;
  const vals: number[] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r = Math.hypot(x - c, y - c) / size;
      if (r >= rMin && r <= rMax) vals.push(power[y * size + x]);
    }
  }
  vals.sort((a, b) => a - b);
  // trimmed mean: the top 5 % may hold interference peaks or scene energy
  const keep = Math.max(1, Math.floor(vals.length * 0.95));
  let s = 0;
  for (let i = 0; i < keep; i++) s += vals[i];
  const meanP = s / keep;
  return Math.sqrt((meanP * winSum * winSum) / winSumSq);
}
