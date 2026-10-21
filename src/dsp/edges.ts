/**
 * The camera's blur and sharpening, read from the photo's own straight edges.
 *
 * A real edge (a roof against the sky, a window frame, lettering) is a sharp
 * step. Whatever width and whatever bright or dark outline the photo shows
 * there was added by the camera: lens blur, noise reduction, and sharpening.
 * So the reference here is not the photo's detail but the fact that real
 * edges are steps (the slanted-edge idea; Joshi, Szeliski & Kriegman 2008,
 * "PSF estimation using sharp edge prediction").
 *
 * Many isolated, straight, high-contrast edges are found, their profiles
 * sampled across the edge at quarter-pixel steps, normalised to run 0 → 1,
 * aligned on their half-way point and combined with a per-offset median:
 * the camera's edge spread function. It is then fitted with a blurred step
 * plus unsharp masking at three widths (phones sharpen as a sum of Gaussians
 * at several pyramid levels), which is linear and so can be undone exactly,
 * and the correction is the filter that turns that response into a good
 * camera's: a narrow edge with no outline.
 */

import type { Plane } from './plane';
import { boxBlur } from './noise';

export const ESF_RADIUS = 16;
export const ESF_STEP = 0.25;

export interface EdgeProfile {
  /** offsets across the edge, px (dark side negative) */
  t: Float32Array;
  /** normalised profile: 0 on the dark plateau, 1 on the bright one */
  esf: Float32Array;
  /** edges used */
  count: number;
  /** median edge contrast, levels */
  contrastLevels: number;
  /** 10–90 % rise distance, px */
  widthPx: number;
  /** bright outline above the bright side, and dark outline below the dark side, as fractions of the edge's height */
  overshoot: number;
  undershoot: number;
}

function bilinear(d: Float32Array, w: number, h: number, x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) return NaN;
  const tx = x - x0;
  const ty = y - y0;
  const i = y0 * w + x0;
  return (d[i] * (1 - tx) + d[i + 1] * tx) * (1 - ty) + (d[i + w] * (1 - tx) + d[i + w + 1] * tx) * ty;
}

/** Rise distance and outlines of a normalised profile. */
export function profileStats(t: Float32Array, esf: Float32Array): { widthPx: number; overshoot: number; undershoot: number } {
  const cross = (level: number): number => {
    // the crossing nearest the centre
    let best = NaN;
    let bestD = Infinity;
    for (let i = 1; i < esf.length; i++) {
      const a = esf[i - 1];
      const b = esf[i];
      if ((a - level) * (b - level) <= 0 && a !== b) {
        const x = t[i - 1] + ((level - a) / (b - a)) * (t[i] - t[i - 1]);
        if (Math.abs(x) < bestD) {
          bestD = Math.abs(x);
          best = x;
        }
      }
    }
    return best;
  };
  const w = cross(0.9) - cross(0.1);
  let mx = 0;
  let mn = 0;
  for (let i = 0; i < t.length; i++) {
    if (Math.abs(t[i]) > 10) continue;
    if (t[i] > 0) mx = Math.max(mx, esf[i] - 1);
    else mn = Math.max(mn, -esf[i]);
  }
  return { widthPx: Number.isFinite(w) ? Math.abs(w) : NaN, overshoot: mx, undershoot: mn };
}

export interface EdgeOptions {
  /** least edge contrast, levels (default 24) */
  minContrast?: number;
  /** most edges to combine (default 4000) */
  maxEdges?: number;
}

/** The camera's edge profile, or null when the photo has too few clean edges (fewer than 40). */
export function measureEdgeProfile(Y: Plane, opts: EdgeOptions = {}): EdgeProfile | null {
  const { w, h } = Y;
  const d = Y.data;
  const minC = (opts.minContrast ?? 24) / 255;
  const maxEdges = opts.maxEdges ?? 4000;
  const R = ESF_RADIUS;
  const nS = Math.round((2 * R) / ESF_STEP) + 1;

  // Sobel gradients
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const mag = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const a = d[i - w - 1];
      const b = d[i - w];
      const c = d[i - w + 1];
      const e = d[i - 1];
      const f = d[i + 1];
      const g = d[i + w - 1];
      const hh = d[i + w];
      const k = d[i + w + 1];
      gx[i] = (c + 2 * f + k - a - 2 * e - g) / 8;
      gy[i] = (g + 2 * hh + k - a - 2 * b - c) / 8;
      mag[i] = Math.hypot(gx[i], gy[i]);
    }
  }

  // candidates: gradient maxima across the edge, strong enough, on a straight edge,
  // at most a few per 16-px cell so one long edge does not dominate
  const gT = (minC / 5) * 1.0; // a 24-level edge blurred over ~4 px still peaks near 5 levels/px
  const cell = 16;
  const cw = Math.ceil(w / cell);
  const perCell = new Uint8Array(cw * Math.ceil(h / cell));
  const margin = R + 2;
  const profiles: Float32Array[] = [];
  const contrasts: number[] = [];
  const samples = new Float32Array(nS);
  // walk on a coarse lattice first, then the rest, so a cap of maxEdges still spreads over the frame
  for (let pass = 0; pass < 2 && profiles.length < maxEdges; pass++) {
    for (let y = margin; y < h - margin && profiles.length < maxEdges; y += pass === 0 ? 3 : 1) {
      for (let x = margin; x < w - margin && profiles.length < maxEdges; x += pass === 0 ? 3 : 1) {
        if (pass === 1 && y % 3 === 0 && x % 3 === 0) continue;
        const i = y * w + x;
        const m = mag[i];
        if (m < gT) continue;
        const cidx = Math.floor(y / cell) * cw + Math.floor(x / cell);
        if (perCell[cidx] >= 3) continue;
        const nx = gx[i] / m;
        const ny = gy[i] / m;
        // non-maximum along the normal
        const m1 = bilinear(mag, w, h, x + nx, y + ny);
        const m2 = bilinear(mag, w, h, x - nx, y - ny);
        if (!(m >= m1 && m >= m2)) continue;
        // straight: gradients in a 7x7 window agree in direction
        let jxx = 0;
        let jyy = 0;
        let jxy = 0;
        for (let dy = -3; dy <= 3; dy++)
          for (let dx = -3; dx <= 3; dx++) {
            const j = i + dy * w + dx;
            jxx += gx[j] * gx[j];
            jyy += gy[j] * gy[j];
            jxy += gx[j] * gy[j];
          }
        const tr = jxx + jyy;
        if (tr <= 0) continue;
        const coh = ((jxx - jyy) ** 2 + 4 * jxy * jxy) / (tr * tr);
        if (coh < 0.8) continue;
        // sample across the edge, dark side first
        let ok = true;
        for (let s = 0; s < nS; s++) {
          const tt = -R + s * ESF_STEP;
          const v = bilinear(d, w, h, x + nx * tt, y + ny * tt);
          if (Number.isNaN(v)) {
            ok = false;
            break;
          }
          samples[s] = v;
        }
        if (!ok) continue;
        // flat plateaus on both sides, well clear of the edge and its outlines
        const plateau = (a: number, b: number): { mean: number; sd: number } => {
          let sum = 0;
          let sq = 0;
          let n = 0;
          for (let s = 0; s < nS; s++) {
            const tt = -R + s * ESF_STEP;
            if (tt < a || tt > b) continue;
            sum += samples[s];
            sq += samples[s] * samples[s];
            n++;
          }
          const mean = sum / n;
          return { mean, sd: Math.sqrt(Math.max(0, sq / n - mean * mean)) };
        };
        const lo = plateau(-R, -R + 4);
        const hi = plateau(R - 4, R);
        const c = hi.mean - lo.mean;
        if (c < minC || lo.mean < 4 / 255 || hi.mean > 250 / 255) continue;
        if (lo.sd > Math.max(3 / 255, 0.08 * c) || hi.sd > Math.max(3 / 255, 0.08 * c)) continue;
        // no second edge inside the window: the profile must be monotone at the scale of a few pixels once smoothed
        let reversals = 0;
        let prev = 0;
        for (let s = 8; s < nS; s += 8) {
          const step = samples[s] - samples[s - 8];
          if (Math.abs(step) > 0.25 * c && prev && Math.sign(step) !== Math.sign(prev)) reversals++;
          if (Math.abs(step) > 0.25 * c) prev = step;
        }
        if (reversals) continue;
        const prof = new Float32Array(nS);
        for (let s = 0; s < nS; s++) prof[s] = (samples[s] - lo.mean) / c;
        // centre on the half-way crossing nearest the middle
        let t0 = NaN;
        let bestD = Infinity;
        for (let s = 1; s < nS; s++) {
          const a = prof[s - 1];
          const b = prof[s];
          if ((a - 0.5) * (b - 0.5) <= 0 && a !== b) {
            const xx = -R + (s - 1 + (0.5 - a) / (b - a)) * ESF_STEP;
            if (Math.abs(xx) < bestD) {
              bestD = Math.abs(xx);
              t0 = xx;
            }
          }
        }
        if (!Number.isFinite(t0) || Math.abs(t0) > 1.5) continue;
        // resample on the common grid, shifted by t0
        const out = new Float32Array(nS);
        for (let s = 0; s < nS; s++) {
          const tt = -R + s * ESF_STEP + t0;
          const fi = (tt + R) / ESF_STEP;
          const i0 = Math.floor(fi);
          const fr = fi - i0;
          out[s] = i0 < 0 ? prof[0] : i0 >= nS - 1 ? prof[nS - 1] : prof[i0] * (1 - fr) + prof[i0 + 1] * fr;
        }
        profiles.push(out);
        contrasts.push(c * 255);
        perCell[cidx]++;
      }
    }
  }
  if (profiles.length < 40) return null;
  const t = new Float32Array(nS);
  for (let s = 0; s < nS; s++) t[s] = -R + s * ESF_STEP;
  // Most edges in a real scene are soft for real reasons: out of focus,
  // shading, curved surfaces. The camera's own blur is what the sharpest
  // edges show, since nothing in the scene can be sharper than a step. So
  // only the narrowest fifth of the edges (at least 40) make the profile.
  const widths = profiles.map((pr) => {
    const st = profileStats(t, pr);
    return Number.isFinite(st.widthPx) ? st.widthPx : Infinity;
  });
  const order = widths.map((wd, k) => [wd, k] as [number, number]).sort((a, b) => a[0] - b[0]);
  const keep = order.slice(0, Math.max(40, Math.floor(order.length * 0.2))).map(([, k]) => k);
  const esf = new Float32Array(nS);
  const col = new Float32Array(keep.length);
  for (let s = 0; s < nS; s++) {
    keep.forEach((k, j) => (col[j] = profiles[k][s]));
    col.sort();
    esf[s] = col[col.length >> 1];
  }
  contrasts.sort((a, b) => a - b);
  const st = profileStats(t, esf);
  return { t, esf, count: keep.length, contrastLevels: contrasts[contrasts.length >> 1], ...st };
}

// ---------------------------------------------------------------------------
// Camera model: blurred step, then unsharp masking at three widths
// ---------------------------------------------------------------------------

export const USM_SIGMAS = [1, 2.5, 6];

function erf(x: number): number {
  // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
const Phi = (x: number, s: number): number => 0.5 * (1 + erf(x / (s * Math.SQRT2)));

export interface CameraModel {
  /** blur, Gaussian sigma in px */
  blurSigma: number;
  /** unsharp-mask amounts at USM_SIGMAS */
  amounts: number[];
  /** rms misfit of the model to the measured profile */
  rms: number;
}

/** Best blurred-step-plus-sharpening fit to a measured edge profile. */
export function fitCameraModel(p: EdgeProfile): CameraModel {
  let best: CameraModel = { blurSigma: 1, amounts: [0, 0, 0], rms: Infinity };
  const use: number[] = [];
  for (let s = 0; s < p.t.length; s++) if (Math.abs(p.t[s]) <= ESF_RADIUS - 1) use.push(s);
  for (let sb = 0.3; sb <= 4.0001; sb += 0.05) {
    // e(x) = E_b(x) + Σ a_k (E_b(x) - Φ(x / sqrt(sb² + σk²))): linear in a_k
    const base = use.map((s) => Phi(p.t[s], sb));
    const cols = USM_SIGMAS.map((sk) => use.map((s, j) => base[j] - Phi(p.t[s], Math.hypot(sb, sk))));
    const target = use.map((s, j) => p.esf[s] - base[j]);
    let active = [0, 1, 2];
    let a = [0, 0, 0];
    for (let iter = 0; iter < 4; iter++) {
      a = [0, 0, 0];
      if (active.length) {
        const n = active.length;
        const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
        const b = new Array(n).fill(0);
        for (let r = 0; r < n; r++) {
          for (let c = 0; c < n; c++) for (let j = 0; j < use.length; j++) A[r][c] += cols[active[r]][j] * cols[active[c]][j];
          for (let j = 0; j < use.length; j++) b[r] += cols[active[r]][j] * target[j];
          A[r][r] += 1e-6;
        }
        const x = solveSmall(A, b);
        active.forEach((k, r) => (a[k] = x[r]));
      }
      const neg = active.filter((k) => a[k] < 0);
      if (!neg.length) break;
      active = active.filter((k) => a[k] >= 0);
    }
    a = a.map((v) => Math.max(0, Math.min(4, v)));
    let e2 = 0;
    for (let j = 0; j < use.length; j++) {
      let m = base[j];
      for (let k = 0; k < 3; k++) m += a[k] * cols[k][j];
      e2 += (m - p.esf[use[j]]) ** 2;
    }
    const rms = Math.sqrt(e2 / use.length);
    if (rms < best.rms) best = { blurSigma: sb, amounts: a, rms };
  }
  return best;
}

function solveSmall(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]];
    const dd = M[i][i] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === i) continue;
      const k = M[r][i] / dd;
      for (let c = i; c <= n; c++) M[r][c] -= k * M[i][c];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}

/** The camera's frequency response under the model (amplitude). */
export function cameraResponse(m: CameraModel, f: number): number {
  const g = (s: number): number => Math.exp(-2 * Math.PI * Math.PI * s * s * f * f);
  let usm = 1;
  for (let k = 0; k < 3; k++) usm += m.amounts[k] * (1 - g(USM_SIGMAS[k]));
  return g(m.blurSigma) * usm;
}

/** ESF of the model, for display and checking. */
export function modelEsf(m: CameraModel, t: Float32Array): Float32Array {
  const out = new Float32Array(t.length);
  for (let i = 0; i < t.length; i++) {
    const eb = Phi(t[i], m.blurSigma);
    let v = eb;
    for (let k = 0; k < 3; k++) v += m.amounts[k] * (eb - Phi(t[i], Math.hypot(m.blurSigma, USM_SIGMAS[k])));
    out[i] = v;
  }
  return out;
}

/** A good camera's edge: 10–90 % rise of `widthPx` (Gaussian blur, no outline). */
export function targetSigma(widthPx: number): number {
  return widthPx / 2.563;
}

/** Most the correction may raise any frequency (amplitude). */
export const MAX_BOOST = 4;

/**
 * The sharpest edge the correction can reach without overshoot: a Gaussian
 * target whose ratio to the camera's blur stays under MAX_BOOST up to 0.3
 * cycles/px. Going sharper would mean clipping the gain curve, and a
 * clipped curve rings (new outlines), which is what is being removed.
 */
export function reachableSigma(m: CameraModel, wantedSigma: number): number {
  const K = Math.log(MAX_BOOST) / (2 * Math.PI * Math.PI * 0.3 * 0.3);
  return Math.min(m.blurSigma, Math.max(wantedSigma, Math.sqrt(Math.max(0, m.blurSigma * m.blurSigma - K))));
}

/**
 * The correction, as a gain in dB at each frequency: the target response
 * divided by the camera's; boosts are held back (toward no change, not
 * toward zero) where the camera passed little and noise dominates; softly limited
 * to ×4 up and ×0.25 down, with boosts fading out past 0.42 cycles/px.
 */
export function correctionGainDb(m: CameraModel, targetWidthPx: number, f: Float32Array, grainSigma: number, keepCrisp = 0): Float32Array {
  const st = reachableSigma(m, targetSigma(targetWidthPx));
  // part of the camera's sharpening can be kept in the target (1 = all of it:
  // then only the blur is reduced). At edges it makes outlines; in texture it
  // is what reads as detail.
  const crisp = (fr: number): number => {
    let v = 1;
    for (let k = 0; k < 3; k++) v += keepCrisp * m.amounts[k] * (1 - Math.exp(-2 * Math.PI * Math.PI * USM_SIGMAS[k] ** 2 * fr * fr));
    return v;
  };
  const out = new Float32Array(f.length);
  // noise-to-signal: natural scenes ~1/f², grain flat
  const nsr0 = Math.max(0.002, ((grainSigma * 255) / 100) ** 2);
  for (let i = 0; i < f.length; i++) {
    const H = cameraResponse(m, f[i]);
    const T = Math.exp(-2 * Math.PI * Math.PI * st * st * f[i] * f[i]) * crisp(f[i]);
    const eps = nsr0 * (f[i] / 0.25) ** 2;
    // Lowering a frequency (undoing sharpening) needs no care. Raising one
    // (deblurring) is held back where the camera passed little, since what is
    // there is mostly noise; held back toward leaving it as it is, never toward
    // erasing it.
    const ratio = T / Math.max(1e-9, H);
    const C = ratio <= 1 ? ratio : 1 + (ratio - 1) * ((H * H) / (H * H + eps));
    // soft limits keep the curve smooth, so the filter does not ring
    let db = 20 * Math.log10(Math.max(1e-6, C));
    const up = 20 * Math.log10(MAX_BOOST);
    const down = 20 * Math.log10(0.25);
    db = db > 0 ? up * Math.tanh(db / up) : down * Math.tanh(db / down);
    if (db > 0) db *= Math.min(1, Math.max(0, (0.48 - f[i]) / 0.06));
    out[i] = db;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Outline clamp: what the linear correction cannot reach
// ---------------------------------------------------------------------------

function minMax1d(src: Float32Array, w: number, h: number, r: number, horizontal: boolean, max: boolean): Float32Array {
  const out = new Float32Array(w * h);
  const n = horizontal ? w : h;
  const lines = horizontal ? h : w;
  for (let l = 0; l < lines; l++) {
    for (let k = 0; k < n; k++) {
      let v = max ? -Infinity : Infinity;
      for (let j = Math.max(0, k - r); j <= Math.min(n - 1, k + r); j++) {
        const x = horizontal ? src[l * w + j] : src[j * w + l];
        v = max ? (x > v ? x : v) : x < v ? x : v;
      }
      if (horizontal) out[l * w + k] = v;
      else out[k * w + l] = v;
    }
  }
  return out;
}
function morph(src: Float32Array, w: number, h: number, r: number, max: boolean): Float32Array {
  return minMax1d(minMax1d(src, w, h, r, true, max), w, h, r, false, max);
}

/**
 * Outlines beside edges that border a flat area (sky, wall, skin), pulled
 * back to that area's own level. For every point on a straight strong edge
 * the profile across it is read; where the side is flat 5–8 px out, any
 * pixel 0.5–4 px out that dips below it (dark outline) or rises above it
 * (bright outline) is set back to it. Sharpening outlines are often
 * lopsided, which a linear filter cannot undo; this can. Edges with texture
 * on both sides are left to the linear correction alone, so real texture is
 * never flattened.
 */
export function outlineClampDelta(Y: Plane, amount: number): Plane {
  const { w, h } = Y;
  const d = Y.data;
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const mag = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = 0.5 * (d[i + 1] - d[i - 1]);
      gy[i] = 0.5 * (d[i + w] - d[i - w]);
      mag[i] = Math.hypot(gx[i], gy[i]);
    }
  const xx = new Float32Array(w * h);
  const yy = new Float32Array(w * h);
  const xy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    xx[i] = gx[i] * gx[i];
    yy[i] = gy[i] * gy[i];
    xy[i] = gx[i] * gy[i];
  }
  const bxx = boxBlur(xx, w, h, 3);
  const byy = boxBlur(yy, w, h, 3);
  const bxy = boxBlur(xy, w, h, 3);
  const up = new Float32Array(w * h); // raise (dark outline)
  const down = new Float32Array(w * h); // lower (bright outline)
  const tol = 1 / 255;
  const prof = new Float32Array(17);
  for (let y = 9; y < h - 9; y++) {
    for (let x = 9; x < w - 9; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m <= 6 / 255) continue;
      const tr = bxx[i] + byy[i];
      if (tr <= 0 || ((bxx[i] - byy[i]) ** 2 + 4 * bxy[i] * bxy[i]) / (tr * tr) < 0.75) continue;
      const nx = gx[i] / m;
      const ny = gy[i] / m;
      if (!(m >= bilinear(mag, w, h, x + nx, y + ny) && m >= bilinear(mag, w, h, x - nx, y - ny))) continue;
      for (let k = 0; k < 17; k++) prof[k] = bilinear(d, w, h, x + nx * (k - 8), y + ny * (k - 8));
      const side = (a: number, b: number): { mean: number; sd: number } => {
        let s1 = 0;
        let s2 = 0;
        for (let k = a; k <= b; k++) {
          s1 += prof[k];
          s2 += prof[k] * prof[k];
        }
        const n = b - a + 1;
        const mean = s1 / n;
        return { mean, sd: Math.sqrt(Math.max(0, s2 / n - mean * mean)) };
      };
      const dark = side(0, 3); // t = -8..-5
      const bright = side(13, 16); // t = 5..8
      const c = bright.mean - dark.mean;
      if (c < 12 / 255) continue;
      const flatT = Math.max(2 / 255, 0.08 * c);
      for (const [plat, sgn] of [
        [dark, -1],
        [bright, 1],
      ] as [{ mean: number; sd: number }, number][]) {
        if (plat.sd > flatT) continue;
        for (let tt = 0.5; tt <= 4; tt += 0.5) {
          const px = Math.round(x + nx * tt * sgn);
          const py = Math.round(y + ny * tt * sgn);
          const j = py * w + px;
          const v = d[j];
          if (sgn < 0 && v < plat.mean - tol) up[j] = Math.max(up[j], plat.mean - tol - v);
          if (sgn > 0 && v > plat.mean + tol) down[j] = Math.max(down[j], v - plat.mean - tol);
        }
      }
    }
  }
  const out = { w, h, data: new Float32Array(w * h) };
  for (let i = 0; i < w * h; i++) out.data[i] = (up[i] - down[i]) * amount;
  return out;
}

/**
 * Where strong straight edges are, feathered: 1 on and within ~4 px of
 * them, fading to 0 by ~10 px. The full correction (outlines removed) goes
 * there; texture elsewhere keeps the camera's finest crispness.
 */
export function edgeZone(Y: Plane): Float32Array {
  const { w, h } = Y;
  const d = Y.data;
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = 0.5 * (d[i + 1] - d[i - 1]);
      gy[i] = 0.5 * (d[i + w] - d[i - w]);
    }
  const xx = new Float32Array(w * h);
  const yy = new Float32Array(w * h);
  const xy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    xx[i] = gx[i] * gx[i];
    yy[i] = gy[i] * gy[i];
    xy[i] = gx[i] * gy[i];
  }
  const bxx = boxBlur(xx, w, h, 3);
  const byy = boxBlur(yy, w, h, 3);
  const bxy = boxBlur(xy, w, h, 3);
  const edge = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (Math.hypot(gx[i], gy[i]) <= 8 / 255) continue;
    const tr = bxx[i] + byy[i];
    if (tr <= 0) continue;
    if (((bxx[i] - byy[i]) ** 2 + 4 * bxy[i] * bxy[i]) / (tr * tr) >= 0.75) edge[i] = 1;
  }
  const near = morph(edge, w, h, 4, true);
  const soft = boxBlur(boxBlur(near, w, h, 3), w, h, 3);
  for (let i = 0; i < soft.length; i++) soft[i] = Math.min(1, soft[i] * 1.5);
  return soft;
}
