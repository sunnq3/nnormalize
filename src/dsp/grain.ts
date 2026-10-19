/**
 * Putting grain back. Clean-up leaves smooth areas looking like plastic; a
 * fine, photon-like grain restores the texture of a photograph without
 * being noticed as noise.
 *
 * The grain is shaped like a camera's: slightly stronger in the shadows
 * and fading out near clipping (shot noise seen through the gamma curve),
 * correlated over about a pixel (a sensor's grain after demosaicing is,
 * and pixel-scale noise is the first thing a re-save throws away), with a
 * weaker, coarser colour part. It is added only where the photo is now
 * smoother than the target, block by block, so textured areas keep their
 * own detail and the result is even across the frame.
 */

import { boxBlur, noiseMap, upsampleBlockMap } from './noise';
import { makePlane, type Plane } from './plane';
import { gaussian, rng } from '../synth/generate';

/** Grain strength relative to the midtones, as a function of brightness (0..1). */
export function grainShape(v: number): number {
  const shadow = Math.pow((Math.max(0, v) + 0.06) / 0.56, -0.15); // ×1.4 in deep shadow, 1 at mid grey
  const highlight = 1 - Math.pow(Math.max(0, Math.min(1, v)), 8); // fades out just below clipping
  return shadow * highlight;
}

/** A unit-variance field correlated over about one pixel: white noise through a 3×3 [1 2 1] kernel. */
function correlatedField(w: number, h: number, seed: number, radius: number): Float32Array {
  const g = gaussian(rng(seed));
  const white = new Float32Array(w * h);
  for (let i = 0; i < white.length; i++) white[i] = g();
  if (radius <= 0) return white;
  // a box blur of radius r then renormalised; r = 1 gives ~1.3 px correlation length
  const b = boxBlur(white, w, h, radius);
  let s = 0;
  for (let i = 0; i < b.length; i++) s += b[i] * b[i];
  const scale = 1 / Math.sqrt(s / b.length);
  for (let i = 0; i < b.length; i++) b[i] *= scale;
  return b;
}

export interface GrainOptions {
  targetLevels: number;
  colour: number; // colour grain as a fraction of the luma grain
  seed: number;
  block?: number;
}

/**
 * Deltas for R, G, B that add grain up to `targetLevels` wherever the
 * current grain is below it. `planes` are the cleaned RGB planes.
 */
export function grainDelta(planes: [Plane, Plane, Plane], opts: GrainOptions): [Plane, Plane, Plane] {
  const { w, h } = planes[0];
  const block = opts.block ?? 32;
  const target = opts.targetLevels / 255;
  const luma = makePlane(w, h);
  for (let i = 0; i < luma.data.length; i++) luma.data[i] = 0.299 * planes[0].data[i] + 0.587 * planes[1].data[i] + 0.114 * planes[2].data[i];
  const map = noiseMap(luma, block);
  // how much to add per block: fill up to the target (shaped by the block's
  // brightness), none where the block carries texture of its own
  const add = new Float32Array(map.sigma.length);
  for (let k = 0; k < add.length; k++) {
    const want = target * grainShape(map.mean[k]);
    const have = map.sigma[k];
    add[k] = want > have ? Math.sqrt(want * want - have * have) * (1 - map.texture[k]) : 0;
  }
  const amount = upsampleBlockMap(add, map.bw, map.bh, block, w, h);
  const lumaField = correlatedField(w, h, opts.seed, 1);
  const cbField = opts.colour > 0 ? correlatedField(w, h, opts.seed + 1, 2) : null;
  const crField = opts.colour > 0 ? correlatedField(w, h, opts.seed + 2, 2) : null;
  const out: [Plane, Plane, Plane] = [makePlane(w, h), makePlane(w, h), makePlane(w, h)];
  for (let i = 0; i < w * h; i++) {
    // the block amount carries the brightness shape at block scale; bright
    // pixels inside a block still fade out so nothing is added near clipping
    const v = Math.min(1, Math.max(0, luma.data[i]));
    const a = amount.data[i] * (1 - Math.pow(v, 8));
    const y = a * lumaField[i];
    const cb = cbField ? a * opts.colour * cbField[i] : 0;
    const cr = crField ? a * opts.colour * crField[i] : 0;
    out[0].data[i] = y + 1.402 * cr;
    out[1].data[i] = y - 0.344136 * cb - 0.714136 * cr;
    out[2].data[i] = y + 1.772 * cb;
  }
  return out;
}
