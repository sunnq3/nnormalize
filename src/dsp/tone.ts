/**
 * Contrast and depth: undoing a phone's flattening.
 *
 * Phones like the Pixel blend a dark and a bright exposure to fit a scene's
 * range into the picture (exposure fusion, up to 8× compression), then lift
 * the shadows. Everything ends up packed into the middle: little contrast
 * between neighbouring areas, no deep shadows.
 *
 * The fix has two parts:
 *  - the black point: when the darkest half percent of the picture sits well
 *    above black, shadows are brought back down with a soft toe;
 *  - local contrast, with a local Laplacian filter (Paris, Hasinoff & Kautz
 *    2011; the fast version of Aubry et al. 2014). At every scale, differences
 *    smaller than σr (detail, depth between neighbouring areas) are expanded
 *    (exponent α < 1) and larger ones (edges) are left as they are or scaled
 *    gently (β), so contrast returns without halos or doubled edges.
 *
 * It runs on brightness only. The phone's flattening lives at scales of 8 px
 * and up, so the filter runs on a 4× smaller copy and its change is carried
 * back to full size by guided upsampling (He et al.: the change is fitted
 * locally as a·Y + b and a, b are interpolated), which keeps sharp edges sharp.
 */

import { boxBlur } from './noise';
import { makePlane, type Plane } from './plane';

export interface ToneParams {
  /** detail exponent: 1 = unchanged, < 1 expands differences smaller than σr */
  alpha: number;
  /** edge scale: 1 = unchanged, > 1 expands larger differences (undoes range compression) */
  beta: number;
  /** detail/edge threshold, plane units */
  sigmaR: number;
  /** how far to bring the black point down, 0..1 */
  blackAmount: number;
  /** the picture's grain, plane units: differences below ~2× this are not expanded */
  noise: number;
}

// --- pyramids ------------------------------------------------------------------

function down2(p: Plane): Plane {
  const w = Math.max(1, Math.ceil(p.w / 2));
  const h = Math.max(1, Math.ceil(p.h / 2));
  // [1 4 6 4 1]/16 then decimate
  const tmp = new Float32Array(w * p.h);
  for (let y = 0; y < p.h; y++) {
    const row = y * p.w;
    for (let x = 0; x < w; x++) {
      const cx = 2 * x;
      let s = 0;
      for (let k = -2; k <= 2; k++) {
        const xx = Math.min(p.w - 1, Math.max(0, cx + k));
        s += [1, 4, 6, 4, 1][k + 2] * p.data[row + xx];
      }
      tmp[y * w + x] = s / 16;
    }
  }
  const out = makePlane(w, h);
  for (let y = 0; y < h; y++) {
    const cy = 2 * y;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -2; k <= 2; k++) {
        const yy = Math.min(p.h - 1, Math.max(0, cy + k));
        s += [1, 4, 6, 4, 1][k + 2] * tmp[yy * w + x];
      }
      out.data[y * w + x] = s / 16;
    }
  }
  return out;
}

/** Bilinear expansion of a coarse level to (w, h). */
function up(p: Plane, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(p.h - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(p.h - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(p.w - 1, Math.max(0, (x + 0.5) / 2 - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(p.w - 1, x0 + 1);
      const tx = fx - x0;
      const a = p.data[y0 * p.w + x0] * (1 - tx) + p.data[y0 * p.w + x1] * tx;
      const b = p.data[y1 * p.w + x0] * (1 - tx) + p.data[y1 * p.w + x1] * tx;
      out[y * w + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

function gaussianPyramid(p: Plane, levels: number): Plane[] {
  const g = [p];
  for (let l = 1; l < levels; l++) g.push(down2(g[l - 1]));
  return g;
}

function laplacianPyramid(g: Plane[]): Float32Array[] {
  const L: Float32Array[] = [];
  for (let l = 0; l < g.length - 1; l++) {
    const e = up(g[l + 1], g[l].w, g[l].h);
    const d = new Float32Array(g[l].data.length);
    for (let i = 0; i < d.length; i++) d[i] = g[l].data[i] - e[i];
    L.push(d);
  }
  L.push(new Float32Array(g[g.length - 1].data));
  return L;
}

function collapse(L: Float32Array[], sizes: { w: number; h: number }[]): Plane {
  let cur: Plane = { ...sizes[sizes.length - 1], data: new Float32Array(L[L.length - 1]) };
  for (let l = L.length - 2; l >= 0; l--) {
    const e = up(cur, sizes[l].w, sizes[l].h);
    for (let i = 0; i < e.length; i++) e[i] += L[l][i];
    cur = { ...sizes[l], data: e };
  }
  return cur;
}

// --- the remapping ------------------------------------------------------------

function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Paris et al.'s remapping around reference g, with a noise floor kept linear. */
function remap(i: number, g: number, p: ToneParams): number {
  const d = i - g;
  const a = Math.abs(d);
  const s = d < 0 ? -1 : 1;
  if (a <= p.sigmaR) {
    const x = a / p.sigmaR;
    const x0 = Math.max(0.01, (2 * p.noise) / p.sigmaR);
    const fd = x + smoothstep(x0, 2 * x0, x) * (Math.pow(x, p.alpha) - x);
    return g + s * p.sigmaR * fd;
  }
  return g + s * (p.beta * (a - p.sigmaR) + p.sigmaR);
}

/** Fast local Laplacian filter (Aubry et al. 2014) of a brightness plane in 0..1. */
export function localLaplacian(I: Plane, p: ToneParams, K = 10): Plane {
  let levels = 1;
  while (Math.min(I.w, I.h) >> levels >= 8 && levels < 10) levels++;
  const G = gaussianPyramid(I, levels);
  const sizes = G.map((g) => ({ w: g.w, h: g.h }));
  const out: Float32Array[] = G.map((g) => new Float32Array(g.data.length));
  const delta = 1 / (K - 1);
  for (let k = 0; k < K; k++) {
    const gk = k * delta;
    const R = makePlane(I.w, I.h);
    for (let i = 0; i < R.data.length; i++) R.data[i] = remap(I.data[i], gk, p);
    const Lk = laplacianPyramid(gaussianPyramid(R, levels));
    for (let l = 0; l < levels - 1; l++) {
      const gl = G[l].data;
      const o = out[l];
      const lk = Lk[l];
      for (let i = 0; i < o.length; i++) {
        const wgt = 1 - Math.abs(gl[i] - gk) / delta;
        if (wgt > 0) o[i] += wgt * lk[i];
      }
    }
  }
  // the coarsest level is the input's own: overall brightness stays where it was
  out[levels - 1] = new Float32Array(G[levels - 1].data);
  return collapse(out, sizes);
}

// --- black point ------------------------------------------------------------------

/** Brightness percentile (0..1) of a plane, from a 1024-bin histogram. */
export function percentile(p: Plane, q: number): number {
  const bins = new Uint32Array(1024);
  for (const v of p.data) bins[Math.min(1023, Math.max(0, Math.floor(v * 1024)))]++;
  const target = q * p.data.length;
  let acc = 0;
  for (let b = 0; b < 1024; b++) {
    acc += bins[b];
    if (acc >= target) return (b + 0.5) / 1024;
  }
  return 1;
}

/** Black point b brought to 0 with a soft quadratic toe below 2b (continuous in value and slope). */
export function blackPointCurve(x: number, b: number): number {
  if (b <= 0) return x;
  if (x >= 2 * b) return (x - b) / (1 - b);
  return (x * x) / (4 * b * (1 - b));
}

// --- the whole fix ----------------------------------------------------------------

export interface FlatnessMeasure {
  /** darkest half percent, levels */
  shadowLevels: number;
  /** brightest half percent, levels */
  highlightLevels: number;
  /** contrast between neighbouring areas: rms of (8-px blur − 64-px blur), levels */
  localContrast: number;
}

export function measureFlatness(Y: Plane): FlatnessMeasure {
  const { w, h } = Y;
  const a = boxBlur(boxBlur(Y.data, w, h, 4), w, h, 4);
  const b = boxBlur(boxBlur(Y.data, w, h, 32), w, h, 32);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return { shadowLevels: percentile(Y, 0.005) * 255, highlightLevels: percentile(Y, 0.995) * 255, localContrast: Math.sqrt(s / a.length) * 255 };
}

function areaDown(p: Plane, f: number): Plane {
  const w = Math.max(1, Math.floor(p.w / f));
  const h = Math.max(1, Math.floor(p.h / f));
  const out = makePlane(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let dy = 0; dy < f; dy++) for (let dx = 0; dx < f; dx++) s += p.data[(y * f + dy) * p.w + x * f + dx];
      out.data[y * w + x] = s / (f * f);
    }
  return out;
}

function bilinearUp(p: Plane, f: number, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(p.h - 1, Math.max(0, (y + 0.5) / f - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(p.h - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(p.w - 1, Math.max(0, (x + 0.5) / f - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(p.w - 1, x0 + 1);
      const tx = fx - x0;
      out[y * w + x] = (p.data[y0 * p.w + x0] * (1 - tx) + p.data[y0 * p.w + x1] * tx) * (1 - ty) + (p.data[y1 * p.w + x0] * (1 - tx) + p.data[y1 * p.w + x1] * tx) * ty;
    }
  }
  return out;
}

/**
 * The brightness change the fix makes, full resolution. `shadowLevels` is
 * the picture's darkest half percent (from measureFlatness).
 */
export function toneDelta(Y: Plane, p: ToneParams, shadowLevels: number): Plane {
  const { w, h } = Y;
  // black point: only when the shadows clearly sit above black
  const lift = Math.max(0, shadowLevels - 4) / 255;
  const b = shadowLevels >= 8 ? p.blackAmount * lift : 0;
  const Yg = makePlane(w, h);
  for (let i = 0; i < Yg.data.length; i++) Yg.data[i] = blackPointCurve(Y.data[i], b);
  // local contrast at reduced size
  const f = Math.min(w, h) >= 1200 ? 4 : Math.min(w, h) >= 600 ? 2 : 1;
  const small = f > 1 ? areaDown(Yg, f) : Yg;
  const filtered = localLaplacian(small, p);
  // guided upsampling of the change: D ≈ a·Y + b over 5×5 at the small size
  const r = 2;
  const sw = small.w;
  const sh = small.h;
  const D = new Float32Array(sw * sh);
  for (let i = 0; i < D.length; i++) D[i] = filtered.data[i] - small.data[i];
  const mI = boxBlur(small.data, sw, sh, r);
  const mD = boxBlur(D, sw, sh, r);
  const II = new Float32Array(sw * sh);
  const ID = new Float32Array(sw * sh);
  for (let i = 0; i < II.length; i++) {
    II[i] = small.data[i] * small.data[i];
    ID[i] = small.data[i] * D[i];
  }
  const mII = boxBlur(II, sw, sh, r);
  const mID = boxBlur(ID, sw, sh, r);
  const A = makePlane(sw, sh);
  const B = makePlane(sw, sh);
  const eps = (2 / 255) ** 2;
  for (let i = 0; i < A.data.length; i++) {
    const v = mII[i] - mI[i] * mI[i];
    A.data[i] = (mID[i] - mI[i] * mD[i]) / (v + eps);
    B.data[i] = mD[i] - A.data[i] * mI[i];
  }
  const Am = { w: sw, h: sh, data: boxBlur(A.data, sw, sh, r) };
  const Bm = { w: sw, h: sh, data: boxBlur(B.data, sw, sh, r) };
  const Af = f > 1 ? bilinearUp(Am, f, w, h) : Am.data;
  const Bf = f > 1 ? bilinearUp(Bm, f, w, h) : Bm.data;
  const out = makePlane(w, h);
  for (let i = 0; i < out.data.length; i++) {
    let v = Yg.data[i] + Af[i] * Yg.data[i] + Bf[i];
    // soft shoulders so expanded edges do not clip hard
    if (v > 0.96) v = 0.96 + 0.04 * Math.tanh((v - 0.96) / 0.04);
    if (v < 0) v = 0;
    out.data[i] = v - Y.data[i];
  }
  return out;
}

/** Parameters for a clean-up level and an amount 0..1. */
export function toneParams(level: 'gentle' | 'standard' | 'strong' | 'maximum', amount: number, noise: number): ToneParams {
  const base = { gentle: { alpha: 0.9, beta: 1.05 }, standard: { alpha: 0.8, beta: 1.12 }, strong: { alpha: 0.7, beta: 1.2 }, maximum: { alpha: 0.6, beta: 1.3 } }[level];
  return {
    alpha: 1 - amount * (1 - base.alpha),
    beta: 1 + amount * (base.beta - 1),
    sigmaR: 0.12,
    blackAmount: amount,
    noise,
  };
}
