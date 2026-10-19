/**
 * Hot, dead and stuck pixels, dust specks: isolated single-pixel outliers
 * against the 3x3 median, judged relative to the local noise level.
 */

import { makePlane, type Plane } from './plane';
import { expectedSigma, type NoiseModel } from './noise';

export interface HotPixel {
  x: number;
  y: number;
  value: number;
  replacement: number;
  zScore: number;
}

export interface HotPixelOptions {
  k?: number; // multiple of local sigma (default 8)
  minAbs?: number; // absolute floor on the deviation, plane units (default 0.03)
  neighbourK?: number; // must differ from every neighbour by this multiple of sigma (default 4)
  max?: number; // cap on the number returned (default 20000)
}

function median9(v: Float32Array, i: number, w: number): number {
  const a = [v[i - w - 1], v[i - w], v[i - w + 1], v[i - 1], v[i], v[i + 1], v[i + w - 1], v[i + w], v[i + w + 1]];
  a.sort((p, q) => p - q);
  return a[4];
}

export function findHotPixels(p: Plane, model: NoiseModel, opts: HotPixelOptions = {}): { pixels: HotPixel[]; delta: Plane } {
  const k = opts.k ?? 8;
  const minAbs = opts.minAbs ?? 0.03;
  const nk = opts.neighbourK ?? 4;
  const cap = opts.max ?? 20000;
  const { w, h } = p;
  const d = p.data;
  const resid = new Float32Array(w * h);
  const med = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const m = median9(d, i, w);
      med[i] = m;
      resid[i] = d[i] - m;
    }
  }
  const pixels: HotPixel[] = [];
  const delta = makePlane(w, h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const r = Math.abs(resid[i]);
      const sigma = Math.max(1e-4, expectedSigma(model, med[i]));
      if (r < k * sigma || r < minAbs) continue;
      // a hot pixel differs from every one of its eight neighbours; an edge or
      // corner pixel agrees with some of them
      let minDiff = Infinity;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const diff = Math.abs(d[i] - d[i + dy * w + dx]);
          if (diff < minDiff) minDiff = diff;
        }
      }
      if (minDiff < nk * sigma || minDiff < minAbs * 0.5) continue;
      pixels.push({ x, y, value: d[i], replacement: med[i], zScore: r / sigma });
      delta.data[i] = med[i] - d[i];
      if (pixels.length >= cap) return { pixels, delta };
    }
  }
  return { pixels, delta };
}
