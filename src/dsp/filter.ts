/**
 * Corrections for periodic patterns. Every function returns a delta plane:
 * corrected = original + delta. Deltas are what the compositor scales and
 * masks, so a correction never touches the original and can be dialled from
 * 0 to 100 % (or past it) after the fact.
 */

import { fft2d } from './fft';
import type { PatternMeasure } from './pattern';
import { cropPlane, extractTile, makePlane, reflectPad, type Plane } from './plane';
import { tileGrid, type TileGrid } from './tiles';
import { window1d } from './window';

/**
 * Parametric subtraction of a stationary sinusoid: exact, no leakage, no
 * border effects, touches nothing but that one frequency.
 */
export function parametricDeltaGlobal(w: number, h: number, u: number, v: number, amplitude: number, phase: number): Plane {
  const out = makePlane(w, h);
  const d = out.data;
  const cosx = new Float64Array(w);
  const sinx = new Float64Array(w);
  for (let x = 0; x < w; x++) {
    const a = 2 * Math.PI * u * x;
    cosx[x] = Math.cos(a);
    sinx[x] = Math.sin(a);
  }
  for (let y = 0; y < h; y++) {
    const b = 2 * Math.PI * v * y + phase;
    const cb = Math.cos(b);
    const sb = Math.sin(b);
    const row = y * w;
    // cos(a + b) = cos a cos b - sin a sin b
    for (let x = 0; x < w; x++) d[row + x] = -amplitude * (cosx[x] * cb - sinx[x] * sb);
  }
  return out;
}

/**
 * Parametric subtraction for a pattern whose amplitude or phase drifts across
 * the frame: each tile's own (A, φ) is used, cross-faded with the Hann window
 * so the overlap regions blend smoothly. The blend weight is floored so the
 * outermost image rows and columns, where the first tile's window is zero,
 * still receive the correction.
 */
export function parametricDeltaTiled(w: number, h: number, grid: TileGrid, p: PatternMeasure): Plane {
  const out = makePlane(w, h);
  const wsum = new Float32Array(w * h);
  const n = grid.size;
  const win1 = window1d('hann', n).w;
  const cosx = new Float64Array(w);
  const sinx = new Float64Array(w);
  for (let x = 0; x < w; x++) {
    const a = 2 * Math.PI * p.u * x;
    cosx[x] = Math.cos(a);
    sinx[x] = Math.sin(a);
  }
  for (const t of p.perTile) {
    for (let ty = 0; ty < n; ty++) {
      const y = t.y0 + ty;
      if (y >= h) break;
      const wy = win1[ty];
      const b = 2 * Math.PI * p.v * y + t.phase;
      const cb = Math.cos(b);
      const sb = Math.sin(b);
      const row = y * w;
      for (let tx = 0; tx < n; tx++) {
        const x = t.x0 + tx;
        if (x >= w) break;
        const wt = Math.max(1e-4, wy * win1[tx]);
        out.data[row + x] += -wt * t.amplitude * (cosx[x] * cb - sinx[x] * sb);
        wsum[row + x] += wt;
      }
    }
  }
  for (let i = 0; i < out.data.length; i++) out.data[i] /= wsum[i];
  return out;
}

export interface SpectralTarget {
  u: number;
  v: number;
  widthBins: number; // in bins of the tile size used for detection
}

export interface SpectralOptions {
  mode: 'flatten' | 'notch';
  sigmaScale?: number; // default 3 (times the measured peak width)
  onProgress?: (done: number, total: number) => void;
}

/**
 * Spectral correction by weighted overlap-add. The plane is reflect-padded by
 * half a tile so the image border sits in the interior of the tiling, each
 * tile is analysed with a sqrt-Hann window, corrected, inverse-transformed,
 * resynthesised with the same window, and the result is normalised by the
 * per-pixel sum of analysis·synthesis weights. An identity correction then
 * reproduces the input to rounding, and nothing is amplified near tile edges.
 *
 *   notch   : multiply by (1 - W), W = max of Gaussians at the targets
 *   flatten : pull |F| down to the median of an annulus around each target
 *             (phase untouched, never pushes below the local level)
 */
export function spectralDelta(plane: Plane, grid: TileGrid, targets: SpectralTarget[], opts: SpectralOptions): Plane {
  const n = grid.size;
  const pad = n >> 1;
  const padded = reflectPad(plane, pad);
  const pgrid = tileGrid(padded.w, padded.h, n, n);
  const { w, h } = padded;
  const sigmaScale = opts.sigmaScale ?? 3;

  const hann = window1d('hann', n).w;
  const sq = new Float64Array(n);
  for (let i = 0; i < n; i++) sq[i] = Math.sqrt(hann[i]);

  // weight map and annulus bookkeeping, in unshifted bin coordinates
  const W = new Float64Array(n * n);
  const centres: { bx: number; by: number; sigma: number }[] = [];
  for (const t of targets) {
    const sigma = Math.max(1.5, t.widthBins * sigmaScale);
    for (const sgn of [1, -1]) {
      const bx = ((Math.round(sgn * t.u * n) % n) + n) % n;
      const by = ((Math.round(sgn * t.v * n) % n) + n) % n;
      centres.push({ bx, by, sigma });
      const r = Math.ceil(sigma * 3.5);
      for (let dy = -r; dy <= r; dy++) {
        const yy = (((by + dy) % n) + n) % n;
        for (let dx = -r; dx <= r; dx++) {
          const xx = (((bx + dx) % n) + n) % n;
          const g = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
          const i = yy * n + xx;
          if (g > W[i]) W[i] = g;
        }
      }
    }
  }
  // protect DC and its immediate neighbourhood
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) W[(((dy % n) + n) % n) * n + (((dx % n) + n) % n)] = 0;

  const out = new Float32Array(w * h);
  const wsum = new Float32Array(w * h);
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  const tile = new Float64Array(n * n);
  const total = pgrid.xs.length * pgrid.ys.length;
  let done = 0;
  const annulus: number[] = [];

  for (const y0 of pgrid.ys) {
    for (const x0 of pgrid.xs) {
      extractTile(padded, x0, y0, n, n, tile);
      let mean = 0;
      for (let i = 0; i < tile.length; i++) mean += tile[i];
      mean /= tile.length;
      for (let ty = 0; ty < n; ty++) {
        const wy = sq[ty];
        for (let tx = 0; tx < n; tx++) {
          const i = ty * n + tx;
          re[i] = (tile[i] - mean) * wy * sq[tx];
          im[i] = 0;
        }
      }
      fft2d(re, im, n, n, false);

      if (opts.mode === 'notch') {
        for (let i = 0; i < re.length; i++) {
          const g = 1 - W[i];
          re[i] *= g;
          im[i] *= g;
        }
      } else {
        for (const ct of centres) {
          annulus.length = 0;
          const r1 = Math.ceil(ct.sigma * 3);
          const r2 = Math.ceil(ct.sigma * 5);
          for (let dy = -r2; dy <= r2; dy++) {
            for (let dx = -r2; dx <= r2; dx++) {
              const d = Math.hypot(dx, dy);
              if (d < r1 || d > r2) continue;
              const i = ((((ct.by + dy) % n) + n) % n) * n + ((((ct.bx + dx) % n) + n) % n);
              annulus.push(Math.hypot(re[i], im[i]));
            }
          }
          annulus.sort((a, b) => a - b);
          const target = annulus.length ? annulus[annulus.length >> 1] : 0;
          const r3 = Math.ceil(ct.sigma * 3.5);
          for (let dy = -r3; dy <= r3; dy++) {
            for (let dx = -r3; dx <= r3; dx++) {
              const i = ((((ct.by + dy) % n) + n) % n) * n + ((((ct.bx + dx) % n) + n) % n);
              const g = W[i];
              if (g <= 0) continue;
              const mag = Math.hypot(re[i], im[i]);
              if (mag <= target || mag === 0) continue;
              const newMag = Math.exp(Math.log(mag) + g * (Math.log(target + 1e-30) - Math.log(mag)));
              const ratio = newMag / mag;
              re[i] *= ratio;
              im[i] *= ratio;
            }
          }
        }
      }

      fft2d(re, im, n, n, true);
      for (let ty = 0; ty < n; ty++) {
        const y = y0 + ty;
        const wy = sq[ty];
        for (let tx = 0; tx < n; tx++) {
          const x = x0 + tx;
          const ws = wy * sq[tx];
          const i = ty * n + tx;
          // synthesis-windowed filtered tile minus the identically windowed input
          out[y * w + x] += (re[i] - (tile[i] - mean) * ws) * ws;
          wsum[y * w + x] += ws * ws;
        }
      }
      done++;
      opts.onProgress?.(done, total);
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = wsum[i] > 1e-6 ? out[i] / wsum[i] : 0;
  return cropPlane({ w, h, data: out }, pad, pad, plane.w, plane.h);
}
