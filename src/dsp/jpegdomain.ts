/**
 * Block settling at the destination's quality.
 *
 * A re-save at quality q rounds every 8×8 block's DCT coefficients to that
 * quality's steps, which can put back a step at the block boundary that the
 * pixel-domain deblocker had smoothed away. Here the luma is written in the
 * destination's own coefficient grid first: each block's coefficients are
 * quantised with the destination's table, then nudged by at most one step
 * each (DC and the nine lowest ACs) wherever that makes the block meet its
 * neighbours more smoothly, with a small penalty for drifting from the
 * pixels we want. The result decodes to pixels that the destination's
 * encoder quantises straight back to these same coefficients, so the
 * boundaries stay settled through its re-save.
 */

import { STD_LUMA } from './blocks';
import { makePlane, type Plane } from './plane';

/** libjpeg's quality scaling of the Annex K luma table, baseline-clamped. */
export function lumaQuantTable(quality: number): Float64Array {
  const q = Math.max(1, Math.min(100, Math.round(quality)));
  const s = q < 50 ? 5000 / q : 200 - 2 * q;
  const out = new Float64Array(64);
  for (let i = 0; i < 64; i++) out[i] = Math.max(1, Math.min(255, Math.floor((STD_LUMA[i] * s + 50) / 100)));
  return out;
}

// 8-point DCT-II basis, orthonormal: this is exactly JPEG's FDCT scaling
const COS = new Float64Array(64);
for (let u = 0; u < 8; u++) {
  const cu = u === 0 ? Math.SQRT1_2 : 1;
  for (let x = 0; x < 8; x++) COS[u * 8 + x] = 0.5 * cu * Math.cos(((2 * x + 1) * u * Math.PI) / 16);
}

/** 2-D DCT of an 8×8 block (row-major, natural order coefficients). */
function fdct8(block: Float64Array, out: Float64Array, tmp: Float64Array): void {
  // rows
  for (let y = 0; y < 8; y++) {
    for (let u = 0; u < 8; u++) {
      let s = 0;
      for (let x = 0; x < 8; x++) s += COS[u * 8 + x] * block[y * 8 + x];
      tmp[y * 8 + u] = s;
    }
  }
  // columns
  for (let u = 0; u < 8; u++) {
    for (let v = 0; v < 8; v++) {
      let s = 0;
      for (let y = 0; y < 8; y++) s += COS[v * 8 + y] * tmp[y * 8 + u];
      out[v * 8 + u] = s;
    }
  }
}

function idct8(coef: Float64Array, out: Float64Array, tmp: Float64Array): void {
  for (let v = 0; v < 8; v++) {
    for (let x = 0; x < 8; x++) {
      let s = 0;
      for (let u = 0; u < 8; u++) s += COS[u * 8 + x] * coef[v * 8 + u];
      tmp[v * 8 + x] = s;
    }
  }
  for (let x = 0; x < 8; x++) {
    for (let y = 0; y < 8; y++) {
      let s = 0;
      for (let v = 0; v < 8; v++) s += COS[v * 8 + y] * tmp[v * 8 + x];
      out[y * 8 + x] = s;
    }
  }
}

// basis patterns of the coefficients we may nudge (v = row frequency, u = column frequency)
const NUDGE: [number, number][] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [0, 2],
  [2, 0],
  [1, 1],
  [0, 3],
  [3, 0],
  [1, 2],
  [2, 1],
];
const BASIS: Float64Array[] = NUDGE.map(([v, u]) => {
  const c = new Float64Array(64);
  c[v * 8 + u] = 1;
  const out = new Float64Array(64);
  idct8(c, out, new Float64Array(64));
  return out;
});

export interface SettleOptions {
  /** weight of staying close to the wanted pixels, against boundary smoothness (default 0.5) */
  lambda?: number;
  /**
   * the photo's grain, levels: on a clean photo a nudge shows as ripple, so
   * the fidelity weight rises as (1.5 / grain)² below 1.5 levels of grain
   */
  grainLevels?: number;
  /** greedy sweeps over all blocks (default 3) */
  iterations?: number;
  /** boundary steps above this (plane units) are real edges and are not smoothed (default 20 levels) */
  maxStep?: number;
}

export interface SettleResult {
  plane: Plane;
  /** blocks whose coefficients were nudged */
  changed: number;
  blocks: number;
  /** mean absolute pixel change, plane units */
  meanChange: number;
}

/**
 * Settle the luma plane (0..1) on the destination's coefficient grid at the
 * given quality. Pixels outside the whole-block area are left as they are.
 */
export function settleBlocks(Y: Plane, quality: number, opts: SettleOptions = {}): SettleResult {
  const grain = Math.max(0.2, opts.grainLevels ?? 1.5);
  const lambda = (opts.lambda ?? 0.5) * Math.max(1, (1.5 / grain) ** 2);
  const iterations = opts.iterations ?? 3;
  const maxStep = (opts.maxStep ?? 20 / 255) * 255;
  const { w, h } = Y;
  const bw = Math.floor(w / 8);
  const bh = Math.floor(h / 8);
  const Q = lumaQuantTable(quality);
  const out = makePlane(w, h);
  out.data.set(Y.data);
  if (bw < 2 || bh < 2) return { plane: out, changed: 0, blocks: bw * bh, meanChange: 0 };

  // target pixels in levels, the quantised reconstruction, and the coefficients
  const W = bw * 8;
  const H = bh * 8;
  const target = new Float64Array(W * H);
  const cur = new Float64Array(W * H);
  const coef = new Int16Array(bw * bh * 64);
  const blk = new Float64Array(64);
  const tmp = new Float64Array(64);
  const c = new Float64Array(64);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        const v = Y.data[(by * 8 + y) * w + bx * 8 + x] * 255;
        target[(by * 8 + y) * W + bx * 8 + x] = v;
        blk[y * 8 + x] = v - 128;
      }
      fdct8(blk, c, tmp);
      const k = (by * bw + bx) * 64;
      for (let i = 0; i < 64; i++) {
        const qv = Math.round(c[i] / Q[i]);
        coef[k + i] = qv;
        c[i] = qv * Q[i];
      }
      idct8(c, blk, tmp);
      for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) cur[(by * 8 + y) * W + bx * 8 + x] = blk[y * 8 + x] + 128;
    }
  }

  // smoothness cost of one boundary between pixel columns/rows a1|b0, with
  // a0 before and b1 after: the step minus the mean inner gradient, squared;
  // real edges (large steps) do not count
  const edgeCost = (a0: number, a1: number, b0: number, b1: number): number => {
    const step = b0 - a1;
    if (Math.abs(step) > maxStep) return 0;
    const e = step - 0.5 * (a1 - a0 + (b1 - b0));
    return e * e;
  };
  const localCost = (bx: number, by: number): number => {
    const x0 = bx * 8;
    const y0 = by * 8;
    let s = 0;
    if (bx > 0) for (let y = y0; y < y0 + 8; y++) s += edgeCost(cur[y * W + x0 - 2], cur[y * W + x0 - 1], cur[y * W + x0], cur[y * W + x0 + 1]);
    if (bx < bw - 1) for (let y = y0; y < y0 + 8; y++) s += edgeCost(cur[y * W + x0 + 6], cur[y * W + x0 + 7], cur[y * W + x0 + 8], cur[y * W + x0 + 9]);
    if (by > 0) for (let x = x0; x < x0 + 8; x++) s += edgeCost(cur[(y0 - 2) * W + x], cur[(y0 - 1) * W + x], cur[y0 * W + x], cur[(y0 + 1) * W + x]);
    if (by < bh - 1) for (let x = x0; x < x0 + 8; x++) s += edgeCost(cur[(y0 + 6) * W + x], cur[(y0 + 7) * W + x], cur[(y0 + 8) * W + x], cur[(y0 + 9) * W + x]);
    return s;
  };
  const fidelity = (bx: number, by: number): number => {
    let s = 0;
    for (let y = 0; y < 8; y++) {
      const row = (by * 8 + y) * W + bx * 8;
      for (let x = 0; x < 8; x++) {
        const d = cur[row + x] - target[row + x];
        s += d * d;
      }
    }
    return s;
  };
  const addPattern = (bx: number, by: number, pattern: Float64Array, amount: number): void => {
    for (let y = 0; y < 8; y++) {
      const row = (by * 8 + y) * W + bx * 8;
      for (let x = 0; x < 8; x++) cur[row + x] += amount * pattern[y * 8 + x];
    }
  };

  // a block whose boundaries are already this smooth is not worth touching
  const skipBelow = 0.5 * 0.5 * 32;
  const touched = new Uint8Array(bw * bh);
  for (let it = 0; it < iterations; it++) {
    let changed = 0;
    for (let by = 0; by < bh; by++) {
      for (let bx = 0; bx < bw; bx++) {
        const smooth0 = localCost(bx, by);
        if (smooth0 < skipBelow) continue;
        const base = smooth0 + lambda * fidelity(bx, by);
        let bestCost = base - 1e-9;
        let bestN = -1;
        let bestD = 0;
        const k = (by * bw + bx) * 64;
        for (let n = 0; n < NUDGE.length; n++) {
          const [v, u] = NUDGE[n];
          const i = v * 8 + u;
          for (const d of [-1, 1]) {
            const amount = d * Q[i];
            addPattern(bx, by, BASIS[n], amount);
            const cost = localCost(bx, by) + lambda * fidelity(bx, by);
            addPattern(bx, by, BASIS[n], -amount);
            if (cost < bestCost) {
              bestCost = cost;
              bestN = n;
              bestD = d;
            }
          }
        }
        if (bestN >= 0) {
          const [v, u] = NUDGE[bestN];
          const i = v * 8 + u;
          coef[k + i] += bestD;
          addPattern(bx, by, BASIS[bestN], bestD * Q[i]);
          touched[by * bw + bx] = 1;
          changed++;
        }
      }
    }
    if (!changed) break;
  }

  let changedBlocks = 0;
  for (let i = 0; i < touched.length; i++) changedBlocks += touched[i];
  let sum = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = cur[y * W + x];
      out.data[y * w + x] = Math.max(0, Math.min(1, v / 255));
      sum += Math.abs(v - target[y * W + x]);
    }
  }
  return { plane: out, changed: changedBlocks, blocks: bw * bh, meanChange: sum / (W * H) / 255 };
}

/** JPEG's RGB → YCbCr (full range, Cb/Cr centred on 0.5), as libjpeg computes it. */
export function rgbToJpegYcc(r: Plane, g: Plane, b: Plane): [Plane, Plane, Plane] {
  const n = r.w * r.h;
  const Y = makePlane(r.w, r.h);
  const Cb = makePlane(r.w, r.h);
  const Cr = makePlane(r.w, r.h);
  for (let i = 0; i < n; i++) {
    const R = r.data[i];
    const G = g.data[i];
    const B = b.data[i];
    Y.data[i] = 0.299 * R + 0.587 * G + 0.114 * B;
    Cb.data[i] = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5;
    Cr.data[i] = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5;
  }
  return [Y, Cb, Cr];
}

export function jpegYccToRgb(Y: Plane, Cb: Plane, Cr: Plane): [Plane, Plane, Plane] {
  const n = Y.w * Y.h;
  const r = makePlane(Y.w, Y.h);
  const g = makePlane(Y.w, Y.h);
  const b = makePlane(Y.w, Y.h);
  for (let i = 0; i < n; i++) {
    const y = Y.data[i];
    const cb = Cb.data[i] - 0.5;
    const cr = Cr.data[i] - 0.5;
    r.data[i] = Math.max(0, Math.min(1, y + 1.402 * cr));
    g.data[i] = Math.max(0, Math.min(1, y - 0.344136 * cb - 0.714136 * cr));
    b.data[i] = Math.max(0, Math.min(1, y + 1.772 * cb));
  }
  return [r, g, b];
}
