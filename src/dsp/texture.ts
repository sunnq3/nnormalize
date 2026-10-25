/**
 * Texture recovery from the photo's own detail.
 *
 * Noise reduction does not erase fine texture completely; it weakens it.
 * Natural scenes are scale-invariant (power ∝ 1/f², Field 1987; Torralba &
 * Oliva 2003 measure exponents near 2): on average each octave of detail
 * carries the same energy as the next coarser one. Through a good camera the
 * finest octave is a little weaker (lens and sensor softness). So wherever a
 * patch has texture at one scale, the finer scale should carry a predictable
 * share of it. Where it carries less, noise reduction or processing has
 * flattened it, and the photo's own remaining fine detail is strengthened
 * toward that share.
 *
 * Nothing is invented: only detail already there is scaled. Flat areas (no
 * texture at the coarser scale) are left alone, so smooth skies and skin
 * are not roughened; strong straight edges are left to the edge fix; and
 * where noise is what fills the fine scale, the share is already met, so
 * nothing is boosted.
 */

import { edgeZone } from './edges';
import { boxBlur } from './noise';
import { makePlane, type Plane } from './plane';

// --- bands -------------------------------------------------------------------------

function binomial(src: Float32Array, w: number, h: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) tmp[row + x] = 0.25 * src[row + Math.max(0, x - 1)] + 0.5 * src[row + x] + 0.25 * src[row + Math.min(w - 1, x + 1)];
  }
  for (let y = 0; y < h; y++) {
    const up = Math.max(0, y - 1) * w;
    const dn = Math.min(h - 1, y + 1) * w;
    const row = y * w;
    for (let x = 0; x < w; x++) out[row + x] = 0.25 * tmp[up + x] + 0.5 * tmp[row + x] + 0.25 * tmp[dn + x];
  }
  return out;
}
const box3 = (s: Float32Array, w: number, h: number, r: number): Float32Array => boxBlur(boxBlur(boxBlur(s, w, h, r), w, h, r), w, h, r);

/** Three octave-ish detail bands: finest (≈0.18–0.5 cycles/px), next, and the one above. */
function bands(Y: Plane): [Float32Array, Float32Array, Float32Array] {
  const { w, h } = Y;
  const L1 = binomial(Y.data, w, h);
  const L2 = box3(Y.data, w, h, 1);
  const L3 = box3(Y.data, w, h, 2);
  const B1 = new Float32Array(w * h);
  const B2 = new Float32Array(w * h);
  const B3 = new Float32Array(w * h);
  for (let i = 0; i < B1.length; i++) {
    B1[i] = Y.data[i] - L1[i];
    B2[i] = L1[i] - L2[i];
    B3[i] = L2[i] - L3[i];
  }
  return [B1, B2, B3];
}

// --- the reference: band energies of a scale-invariant scene through a good camera --

const CAMERA_SIGMA = 0.6; // px: a good camera's own softness

function lowResponse(level: number, f: number): number {
  if (level === 0) return 1;
  if (level === 1) return Math.cos(Math.PI * f) ** 2;
  const r = level === 2 ? 1 : 2;
  const m = 2 * r + 1;
  const s = Math.sin(Math.PI * f);
  const v = Math.abs(s) < 1e-9 ? 1 : Math.sin(Math.PI * f * m) / (m * s);
  return v * v * v;
}

/** Expected E1/E2 and E2/E3 for power ∝ 1/f² seen through CAMERA_SIGMA blur. */
function referenceRatios(): [number, number] {
  const n = 256;
  const e = [0, 0, 0];
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const fx = (x - n / 2) / n;
      const fy = (y - n / 2) / n;
      const f2 = fx * fx + fy * fy;
      if (f2 < 1e-6) continue;
      const P = (1 / f2) * Math.exp(-4 * Math.PI * Math.PI * CAMERA_SIGMA * CAMERA_SIGMA * f2);
      const H = (l: number): number => lowResponse(l, fx) * lowResponse(l, fy);
      for (let k = 0; k < 3; k++) e[k] += (H(k) - H(k + 1)) ** 2 * P;
    }
  }
  return [Math.sqrt(e[0] / e[1]), Math.sqrt(e[1] / e[2])];
}
const REF = referenceRatios();

// --- measuring and fixing ---------------------------------------------------------

export interface TextureMeasure {
  /** median share of the expected fine texture, over textured areas (1 = natural) */
  share: number;
  /** fraction of the photo that is textured and below 70 % of the expected share */
  weakFraction: number;
}

const STRUCT_MIN = 1.0 / 255; // band-2 energy below this is a flat area

interface Maps {
  B: [Float32Array, Float32Array, Float32Array];
  E: [Float32Array, Float32Array, Float32Array];
}

function maps(Y: Plane): Maps {
  const { w, h } = Y;
  const B = bands(Y);
  const E = B.map((b) => {
    const sq = new Float32Array(b.length);
    for (let i = 0; i < b.length; i++) sq[i] = b[i] * b[i];
    const m = boxBlur(sq, w, h, 6);
    for (let i = 0; i < m.length; i++) m[i] = Math.sqrt(Math.max(0, m[i]));
    return m;
  }) as [Float32Array, Float32Array, Float32Array];
  return { B, E };
}

export function measureTexture(Y: Plane): TextureMeasure {
  const { E } = maps(Y);
  const shares: number[] = [];
  let weak = 0;
  for (let i = 0; i < E[0].length; i += 3) {
    if (E[1][i] < STRUCT_MIN) continue;
    const s = E[0][i] / E[1][i] / REF[0];
    shares.push(s);
    if (s < 0.7) weak++;
  }
  shares.sort((a, b) => a - b);
  return { share: shares.length ? shares[shares.length >> 1] : 1, weakFraction: (weak * 3) / E[0].length };
}

/**
 * The brightness change that brings weakened fine texture back toward the
 * natural share. `amount` 0..1 scales it; the finest band is raised by at
 * most ×2.5 and the next by ×1.6.
 */
export function textureDelta(Y: Plane, amount: number): Plane {
  // two rounds: raising one band shifts the energies the next round compares
  // only big steps are kept out (boosting there would draw an outline)
  const zone = edgeZone(Y, 20 / 255);
  const d1 = textureRound(Y, amount, zone);
  const Y1 = { w: Y.w, h: Y.h, data: Y.data.map((v, i) => v + d1.data[i]) };
  const d2 = textureRound(Y1, amount, zone);
  for (let i = 0; i < d1.data.length; i++) d1.data[i] += d2.data[i];
  return d1;
}

function textureRound(Y: Plane, amount: number, zone: Float32Array): Plane {
  const { w, h } = Y;
  const { B, E } = maps(Y);
  const g1 = new Float32Array(w * h);
  const g2 = new Float32Array(w * h);
  for (let i = 0; i < g1.length; i++) {
    // how much texture is there to go on, fading in above the flat-area floor
    const structure = Math.min(1, Math.max(0, (E[1][i] - STRUCT_MIN) / STRUCT_MIN));
    const keep = structure * (1 - zone[i]);
    const want1 = E[1][i] * REF[0];
    const want2 = E[2][i] * REF[1];
    const r1 = E[0][i] > 1e-6 ? want1 / E[0][i] : 1;
    const r2 = E[1][i] > 1e-6 ? want2 / E[1][i] : 1;
    g1[i] = 1 + keep * amount * (Math.min(2.5, Math.max(1, r1)) - 1);
    g2[i] = 1 + keep * amount * (Math.min(1.6, Math.max(1, r2)) - 1);
  }
  // gains change smoothly across the picture
  const s1 = boxBlur(g1, w, h, 4);
  const s2 = boxBlur(g2, w, h, 4);
  const out = makePlane(w, h);
  for (let i = 0; i < out.data.length; i++) out.data[i] = (s1[i] - 1) * B[0][i] + (s2[i] - 1) * B[1][i];
  return out;
}
