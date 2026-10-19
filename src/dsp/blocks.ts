/**
 * JPEG block-grid forensics and deblocking in the image domain.
 */

import { makePlane, type Plane } from './plane';

export interface BlockGrid {
  offsetX: number; // x of the first block boundary (boundary lies between x-1 and x)
  offsetY: number;
  strengthX: number; // excess of boundary differences over the others, 0 = none
  strengthY: number;
  /** the same excess in absolute terms: mean boundary step above the texture, plane units */
  stepX: number;
  stepY: number;
  /** z-like confidence: how far the best phase stands out from the other seven */
  confidenceX: number;
  confidenceY: number;
  detected: boolean;
}

/**
 * For each of the 8 column phases, average the absolute horizontal difference
 * across the whole image; the phase where block boundaries sit is the one
 * with the excess. Same for rows.
 */
export function detectBlockGrid(p: Plane, period = 8): BlockGrid {
  const { w, h } = p;
  const d = p.data;
  const px = new Float64Array(period);
  const nx = new Float64Array(period);
  const py = new Float64Array(period);
  const ny = new Float64Array(period);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 1; x < w; x++) {
      px[x % period] += Math.abs(d[row + x] - d[row + x - 1]);
      nx[x % period]++;
    }
  }
  for (let y = 1; y < h; y++) {
    const row = y * w;
    const prev = row - w;
    let s = 0;
    for (let x = 0; x < w; x++) s += Math.abs(d[row + x] - d[prev + x]);
    py[y % period] += s;
    ny[y % period] += w;
  }
  const analyse = (acc: Float64Array, cnt: Float64Array): { offset: number; strength: number; confidence: number; step: number } => {
    const m = new Float64Array(period);
    for (let k = 0; k < period; k++) m[k] = cnt[k] ? acc[k] / cnt[k] : 0;
    let best = 0;
    for (let k = 1; k < period; k++) if (m[k] > m[best]) best = k;
    let others = 0;
    let n = 0;
    for (let k = 0; k < period; k++) {
      if (k === best) continue;
      others += m[k];
      n++;
    }
    others /= n;
    let v = 0;
    for (let k = 0; k < period; k++) if (k !== best) v += (m[k] - others) ** 2;
    const sd = Math.sqrt(v / n) + 1e-12;
    return { offset: best, strength: others > 0 ? (m[best] - others) / others : 0, confidence: (m[best] - others) / sd, step: Math.max(0, m[best] - others) };
  };
  const ax = analyse(px, nx);
  const ay = analyse(py, ny);
  const detected = ax.strength > 0.03 && ay.strength > 0.03 && ax.confidence > 4 && ay.confidence > 4;
  return {
    offsetX: ax.offset,
    offsetY: ay.offset,
    strengthX: ax.strength,
    strengthY: ay.strength,
    stepX: ax.step,
    stepY: ay.step,
    confidenceX: ax.confidence,
    confidenceY: ay.confidence,
    detected,
  };
}

export interface DeblockOptions {
  period?: number; // default 8
  /** steps larger than this are treated as real edges and left alone (plane units) */
  maxStep?: number; // default 0.08
  /** the boundary step must exceed this multiple of the neighbouring activity */
  activityRatio?: number; // default 1.5
  /** pixels the step is spread over: 4 (two each side) or 8 (the whole block width); 8 removes the grid signature entirely */
  span?: 4 | 8;
}

/**
 * Grid-aligned deblocking: where the step across a block boundary stands out
 * from the texture either side of it, replace the step by a linear ramp over
 * the four pixels that straddle it. Horizontal pass, then vertical on the
 * horizontally corrected plane. Returns the combined delta.
 */
export function deblockDelta(p: Plane, grid: BlockGrid, opts: DeblockOptions = {}): Plane {
  const period = opts.period ?? 8;
  const maxStep = opts.maxStep ?? 0.08;
  const ratio = opts.activityRatio ?? 1.5;
  const span = opts.span ?? 4;
  const { w, h } = p;
  const work = new Float32Array(p.data);
  const delta = makePlane(w, h);
  // a linear ramp across `span` pixels: pixel k of the span moves by
  // d·(k+½)/span on the left and −d·(span−k−½)/span on the right, so the
  // step d becomes d/span per pixel, the same as the gradient inside a block
  const half = span / 2;
  // horizontal boundaries at x ≡ offsetX (mod period), between x-1 and x
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let xb = grid.offsetX; xb < w; xb += period) {
      if (xb < half || xb + half - 1 >= w) continue;
      const a0 = work[row + xb - 2];
      const a1 = work[row + xb - 1];
      const b0 = work[row + xb];
      const b1 = work[row + xb + 1];
      const d = b0 - a1;
      const ad = Math.abs(d);
      if (ad > maxStep || ad < 1e-6) continue;
      const act = 0.5 * (Math.abs(a1 - a0) + Math.abs(b1 - b0));
      if (ad < ratio * act + 1e-4) continue;
      for (let k = 0; k < span; k++) {
        const i = row + xb - half + k;
        const t = k < half ? d * ((k + 0.5) / span) : -d * ((span - k - 0.5) / span);
        work[i] += t;
        delta.data[i] += t;
      }
    }
  }
  for (let x = 0; x < w; x++) {
    for (let yb = grid.offsetY; yb < h; yb += period) {
      if (yb < half || yb + half - 1 >= h) continue;
      const a0 = work[(yb - 2) * w + x];
      const a1 = work[(yb - 1) * w + x];
      const b0 = work[yb * w + x];
      const b1 = work[(yb + 1) * w + x];
      const d = b0 - a1;
      const ad = Math.abs(d);
      if (ad > maxStep || ad < 1e-6) continue;
      const act = 0.5 * (Math.abs(a1 - a0) + Math.abs(b1 - b0));
      if (ad < ratio * act + 1e-4) continue;
      for (let k = 0; k < span; k++) {
        const i = (yb - half + k) * w + x;
        const t = k < half ? d * ((k + 0.5) / span) : -d * ((span - k - 0.5) / span);
        work[i] += t;
        delta.data[i] += t;
      }
    }
  }
  return delta;
}

/** Quality estimate from a luma quantisation table, the libjpeg scaling inverted. */
export function qualityFromQuantTable(table: ArrayLike<number>, isChroma = false): number {
  const std = isChroma ? STD_CHROMA : STD_LUMA;
  let s = 0;
  let n = 0;
  for (let i = 0; i < 64; i++) {
    if (std[i] > 0 && table[i] > 0) {
      s += (100 * table[i]) / std[i];
      n++;
    }
  }
  const scale = s / n; // libjpeg: table = (std * scale + 50) / 100
  const q = scale >= 100 ? 5000 / scale : (200 - scale) / 2;
  return Math.max(1, Math.min(100, q));
}

// Annex K tables in natural (row-major) order
export const STD_LUMA = [
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68,
  109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
export const STD_CHROMA = [
  17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];
