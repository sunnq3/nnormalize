/**
 * Sub-bin frequency measurement by direct evaluation of the windowed DTFT.
 *
 * The FFT only samples the spectrum at bin centres; the true frequency of an
 * interference pattern is almost never on one. Rather than zero-pad, we
 * evaluate S(u, v) = Σ x[y,x]·w[y,x]·e^{-2πi(ux + vy)} at any (u, v) we like.
 * The sum is separable, so each evaluation costs one pass over the tile.
 * A quadratic surface fitted to log|S|² on a 3x3 stencil, with the stencil
 * spacing halved each round, converges to ~1e-3 bin in ten rounds.
 */

import type { Window2D } from './window';

export interface Complex {
  re: number;
  im: number;
}

/** Windowed tile (x·w) as Float64, mean removed before windowing. */
export function windowedTile(tile: Float64Array, win: Window2D | null, w: number, h: number, out?: Float64Array): Float64Array {
  const n = w * h;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += tile[i];
  mean /= n;
  out = out ?? new Float64Array(n);
  if (win) for (let i = 0; i < n; i++) out[i] = (tile[i] - mean) * win.data[i];
  else for (let i = 0; i < n; i++) out[i] = tile[i] - mean;
  return out;
}

/** Row sums R_y = Σ_x tw[y,x]·e^{-2πi u x}, written into re/im of length h. */
function rowSums(tw: Float64Array, w: number, h: number, u: number, re: Float64Array, im: Float64Array): void {
  const cx = new Float64Array(w);
  const sx = new Float64Array(w);
  for (let x = 0; x < w; x++) {
    const a = -2 * Math.PI * u * x;
    cx[x] = Math.cos(a);
    sx[x] = Math.sin(a);
  }
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let r = 0;
    let i = 0;
    for (let x = 0; x < w; x++) {
      const t = tw[row + x];
      r += t * cx[x];
      i += t * sx[x];
    }
    re[y] = r;
    im[y] = i;
  }
}

/** Combine row sums with the column factor e^{-2πi v y}. */
function colCombine(re: Float64Array, im: Float64Array, h: number, v: number): Complex {
  let sr = 0;
  let si = 0;
  for (let y = 0; y < h; y++) {
    const a = -2 * Math.PI * v * y;
    const c = Math.cos(a);
    const s = Math.sin(a);
    sr += re[y] * c - im[y] * s;
    si += re[y] * s + im[y] * c;
  }
  return { re: sr, im: si };
}

/** Single DTFT evaluation of a windowed tile at (u, v) cycles/pixel. */
export function dtft(tw: Float64Array, w: number, h: number, u: number, v: number): Complex {
  const re = new Float64Array(h);
  const im = new Float64Array(h);
  rowSums(tw, w, h, u, re, im);
  return colCombine(re, im, h, v);
}

export interface RefineResult {
  u: number;
  v: number;
  power: number; // |S|² at the refined frequency
  amplitude: number; // peak amplitude of the real sinusoid, in plane units
  phase: number; // radians, referenced to the tile's (0,0)
  iterations: number;
  converged: boolean;
}

export interface RefineOptions {
  maxIter?: number; // default 12
  initialStep?: number; // in cycles/pixel; default 0.5 bin of the tile
  minStep?: number; // stop when the stencil spacing drops below this; default 1e-3 bin
  tileSize?: number; // used to express the defaults in bins
}

/**
 * Refine (u0, v0) to the local maximum of |S(u, v)|². Returns the refined
 * frequency plus the amplitude and phase of the sinusoid there.
 */
export function refinePeak(
  tw: Float64Array,
  w: number,
  h: number,
  winSum: number,
  u0: number,
  v0: number,
  opts: RefineOptions = {},
): RefineResult {
  const size = opts.tileSize ?? Math.max(w, h);
  const maxIter = opts.maxIter ?? 12;
  let d = opts.initialStep ?? 0.5 / size;
  const minStep = opts.minStep ?? 1e-3 / size;
  let u = u0;
  let v = v0;
  const re = [new Float64Array(h), new Float64Array(h), new Float64Array(h)];
  const im = [new Float64Array(h), new Float64Array(h), new Float64Array(h)];
  const f = new Float64Array(9);
  let iter = 0;
  let converged = false;
  let best: Complex = { re: 0, im: 0 };
  for (; iter < maxIter; iter++) {
    for (let i = 0; i < 3; i++) rowSums(tw, w, h, u + (i - 1) * d, re[i], im[i]);
    let bestP = -Infinity;
    let bestI = 1;
    let bestJ = 1;
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 3; i++) {
        const s = colCombine(re[i], im[i], h, v + (j - 1) * d);
        const p = s.re * s.re + s.im * s.im;
        f[j * 3 + i] = Math.log(p + 1e-300);
        if (p > bestP) {
          bestP = p;
          bestI = i;
          bestJ = j;
          best = s;
        }
      }
    }
    // quadratic surface a + bx + cy + dx² + ey² + gxy on x,y ∈ {-1,0,1}
    let A = 0;
    let Sx = 0;
    let Sy = 0;
    let Sxx = 0;
    let Syy = 0;
    let Sxy = 0;
    for (let j = 0; j < 3; j++) {
      const y = j - 1;
      for (let i = 0; i < 3; i++) {
        const x = i - 1;
        const val = f[j * 3 + i];
        A += val;
        Sx += x * val;
        Sy += y * val;
        Sxx += x * x * val;
        Syy += y * y * val;
        Sxy += x * y * val;
      }
    }
    const b = Sx / 6;
    const c = Sy / 6;
    const dd = 0.5 * Sxx - A / 3;
    const ee = 0.5 * Syy - A / 3;
    const g = Sxy / 4;
    const det = 4 * dd * ee - g * g;
    let sx: number;
    let sy: number;
    if (dd < 0 && ee < 0 && det > 0) {
      sx = (-2 * ee * b + g * c) / det;
      sy = (-2 * dd * c + g * b) / det;
    } else {
      sx = bestI - 1;
      sy = bestJ - 1;
    }
    const clampedX = Math.max(-1, Math.min(1, sx));
    const clampedY = Math.max(-1, Math.min(1, sy));
    u += clampedX * d;
    v += clampedY * d;
    // shrink the stencil once the maximum is bracketed inside it
    if (Math.abs(sx) < 1 && Math.abs(sy) < 1) d *= 0.5;
    if (d < minStep) {
      converged = true;
      iter++;
      break;
    }
  }
  const s = dtft(tw, w, h, u, v);
  const power = s.re * s.re + s.im * s.im;
  return {
    u,
    v,
    power,
    amplitude: (2 * Math.sqrt(power)) / winSum,
    phase: Math.atan2(s.im, s.re),
    iterations: iter,
    converged,
  };
}

/** Amplitude and phase of the sinusoid at a known (u, v) on a windowed tile. */
export function measureAt(tw: Float64Array, w: number, h: number, winSum: number, u: number, v: number): { amplitude: number; phase: number; power: number } {
  const s = dtft(tw, w, h, u, v);
  const power = s.re * s.re + s.im * s.im;
  return { amplitude: (2 * Math.sqrt(power)) / winSum, phase: Math.atan2(s.im, s.re), power };
}

/** Convert a phase measured relative to a tile origin into the image frame. */
export function phaseToGlobal(phaseTile: number, u: number, v: number, x0: number, y0: number): number {
  return wrapPhase(phaseTile - 2 * Math.PI * (u * x0 + v * y0));
}

export function wrapPhase(p: number): number {
  p = p % (2 * Math.PI);
  if (p > Math.PI) p -= 2 * Math.PI;
  if (p < -Math.PI) p += 2 * Math.PI;
  return p;
}
