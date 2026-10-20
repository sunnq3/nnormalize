/**
 * Banding: visible steps in smooth gradients (skies, walls, shadows), left by
 * 8-bit storage, earlier smoothing, or heavy compression.
 *
 * Found on the stored 8-bit values: a smooth block whose values span at
 * least one level, carry almost no grain to hide the steps, and stay on
 * flat plateaus between single steps. Fixed by an edge-keeping smoother wide
 * enough to turn the staircase back into a ramp (a box mean keeps a linear
 * ramp exactly and keeps any edge stronger than a few levels), plus a fine
 * dither so writing the result back to 8 bits does not cut new steps.
 */

import { correlatedField } from './grain';
import { boxBlur, guidedFilter, upsampleBlockMap, type NoiseMap } from './noise';
import { makePlane, type Plane } from './plane';

export interface BandingMap {
  block: number;
  bw: number;
  bh: number;
  /** 1 where the block is banded */
  banded: Uint8Array;
  /** fraction of blocks banded */
  fraction: number;
  /** typical step between bands, levels */
  stepLevels: number;
}

/**
 * `g` is the channel to inspect on its stored values (green carries most of
 * the brightness), `map` the brightness noise map of the same photo.
 */
export function detectBanding(g: Plane, map: NoiseMap): BandingMap {
  const { w } = g;
  const block = map.block;
  const { bw, bh } = map;
  const banded = new Uint8Array(bw * bh);
  const steps: number[] = [];
  const vals = new Int32Array(block * block);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const k = by * bw + bx;
      // the band edges themselves read as aligned structure, so only the grain level gates here;
      // texture is ruled out below by the steps having to be small and regular
      if (map.sigma[k] * 255 > 0.9) continue;
      let zero = 0;
      let diffs = 0;
      const nz: number[] = [];
      let n = 0;
      for (let y = by * block; y < by * block + block; y++) {
        for (let x = bx * block; x < bx * block + block; x++) {
          const v = Math.round(g.data[y * w + x] * 255);
          vals[n++] = v;
          if (x + 1 < bx * block + block) {
            const d = Math.abs(Math.round(g.data[y * w + x + 1] * 255) - v);
            diffs++;
            if (d === 0) zero++;
            else nz.push(d);
          }
          if (y + 1 < by * block + block) {
            const d = Math.abs(Math.round(g.data[(y + 1) * w + x] * 255) - v);
            diffs++;
            if (d === 0) zero++;
            else nz.push(d);
          }
        }
      }
      const s = vals.subarray(0, n).sort();
      const lo = s[Math.floor(n * 0.05)];
      const hi = s[Math.floor(n * 0.95)];
      const range = hi - lo;
      if (range < 1 || range > 24 || lo <= 1 || hi >= 254) continue;
      if (zero / diffs < 0.6 || nz.length < 8) continue;
      nz.sort((a, b) => a - b);
      const step = nz[nz.length >> 1];
      // the non-zero differences are the steps themselves, not noise
      let regular = 0;
      for (const d of nz) if (d <= 2 * step) regular++;
      if (regular / nz.length < 0.9) continue;
      banded[k] = 1;
      steps.push(step);
    }
  }
  // a lone banded block among unbanded ones is not a visible band
  const keep = new Uint8Array(banded);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const k = by * bw + bx;
      if (!banded[k]) continue;
      let nb = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const x = bx + dx;
          const y = by + dy;
          if (x >= 0 && y >= 0 && x < bw && y < bh && banded[y * bw + x]) nb++;
        }
      if (nb === 0) keep[k] = 0;
    }
  }
  let count = 0;
  for (let i = 0; i < keep.length; i++) count += keep[i];
  steps.sort((a, b) => a - b);
  return { block, bw, bh, banded: keep, fraction: count / Math.max(1, keep.length), stepLevels: steps.length ? steps[steps.length >> 1] : 1 };
}

/**
 * How visible the banding is in a copy of the photo, levels. Bands run
 * across the gradient, so the copy is averaged along the bands (49 px along
 * the block's iso-brightness direction, after a 5×5 box), which keeps the
 * steps and averages grain and dither away; the same is done to the smooth
 * ramp underneath (an edge-keeping wide smoothing), and the rms difference
 * is the banding. A ramp scores ~0, 1-level steps ~0.3, whatever the grain.
 */
export function bandingStrength(p: Plane, banded: Uint8Array, bw: number, bh: number, block: number): number {
  const { w, h } = p;
  const S = boxBlur(p.data, w, h, 2);
  const eps = (3 / 255) ** 2;
  const R = guidedFilter(guidedFilter({ w, h, data: S }, 32, eps), 16, eps).data;
  const L = 32;
  let s = 0;
  let n = 0;
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      if (!banded[by * bw + bx]) continue;
      const x0 = bx * block;
      const y0 = by * block;
      const x1 = Math.min(w - 1, x0 + block);
      const y1 = Math.min(h - 1, y0 + block);
      // the gradient direction of the ramp across this block
      let gx = 0;
      let gy = 0;
      for (let y = Math.max(1, y0); y < y1; y++)
        for (let x = Math.max(1, x0); x < x1; x++) {
          const i = y * w + x;
          gx += R[i + 1] - R[i - 1];
          gy += R[i + w] - R[i - w];
        }
      const gm = Math.hypot(gx, gy);
      if (gm < 1e-9) continue;
      const tx = -gy / gm;
      const ty = gx / gm;
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          // along the bands: keeps the steps; across them: averages the steps
          // into the ramp. Grain is reduced the same in both, so the
          // difference of the two variances is the banding alone.
          let a = 0;
          let m = 0;
          let a2 = 0;
          let m2 = 0;
          for (let k = -L; k <= L; k++) {
            let xx = Math.round(x + k * tx);
            let yy = Math.round(y + k * ty);
            if (xx >= 0 && yy >= 0 && xx < w && yy < h) {
              const j = yy * w + xx;
              a += S[j] - R[j];
              m++;
            }
            xx = Math.round(x + k * ty);
            yy = Math.round(y - k * tx);
            if (xx >= 0 && yy >= 0 && xx < w && yy < h) {
              const j = yy * w + xx;
              a2 += S[j] - R[j];
              m2++;
            }
          }
          if (!m || !m2) continue;
          s += (a / m) ** 2 - (a2 / m2) ** 2;
          n++;
        }
      }
    }
  }
  return n ? Math.sqrt(Math.max(0, s / n)) * 255 : 0;
}

export interface DebandOptions {
  amount: number; // 0..1
  stepLevels: number;
  /** fine dither, levels rms (default 0.7) */
  ditherLevels?: number;
  seed: number;
}

/** Delta for one channel. Use the same seed for all three so the dither carries no colour. */
export function debandDelta(ch: Plane, bands: BandingMap, opts: DebandOptions): Plane {
  const { w, h } = ch;
  const delta = makePlane(w, h);
  const blocks = new Float32Array(bands.banded.length);
  let any = false;
  for (let i = 0; i < blocks.length; i++) {
    blocks[i] = bands.banded[i] * opts.amount;
    if (blocks[i] > 0) any = true;
  }
  if (!any) return delta;
  // spread the blend one block outward so the fix fades in rather than stops
  const spread = new Float32Array(blocks);
  for (let by = 0; by < bands.bh; by++)
    for (let bx = 0; bx < bands.bw; bx++) {
      let m = blocks[by * bands.bw + bx];
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const x = bx + dx;
          const y = by + dy;
          if (x >= 0 && y >= 0 && x < bands.bw && y < bands.bh) m = Math.max(m, 0.5 * blocks[y * bands.bw + x]);
        }
      spread[by * bands.bw + bx] = m;
    }
  const blend = upsampleBlockMap(spread, bands.bw, bands.bh, bands.block, w, h);
  // windows wider than the bands themselves, so a plateau sees its neighbours
  const eps = (Math.max(1.5, 1.5 * opts.stepLevels) / 255) ** 2;
  const f = guidedFilter(guidedFilter(ch, 32, eps), 16, eps);
  const dither = correlatedField(w, h, opts.seed, 1);
  const dl = (opts.ditherLevels ?? 0.7) / 255;
  for (let i = 0; i < delta.data.length; i++) {
    const b = blend.data[i];
    if (b <= 0) continue;
    delta.data[i] = b * (f.data[i] - ch.data[i] + dl * dither[i]);
  }
  return delta;
}
