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
  k?: number; // multiple of local sigma (default 10)
  minAbs?: number; // absolute floor on the deviation, plane units (default 0.06 ≈ 15 levels)
  neighbourK?: number; // must differ from every neighbour by this multiple of sigma (default 5)
  minNeighbourDiff?: number; // and by at least this much, plane units (default 0.03 ≈ 8 levels)
  /** the eight neighbours must agree with each other: their range may be at most this fraction of the deviation (default 0.6) */
  neighbourRange?: number;
  max?: number; // cap on the number returned (default 20000)
}

/** Median of 9 by a fixed sorting network (Paeth); no allocation. */
function median9(v: Float32Array, i: number, w: number): number {
  let p0 = v[i - w - 1];
  let p1 = v[i - w];
  let p2 = v[i - w + 1];
  let p3 = v[i - 1];
  let p4 = v[i];
  let p5 = v[i + 1];
  let p6 = v[i + w - 1];
  let p7 = v[i + w];
  let p8 = v[i + w + 1];
  let t: number;
  // prettier-ignore
  {
    if (p1 < p0) { t = p0; p0 = p1; p1 = t; } if (p4 < p3) { t = p3; p3 = p4; p4 = t; } if (p7 < p6) { t = p6; p6 = p7; p7 = t; }
    if (p2 < p1) { t = p1; p1 = p2; p2 = t; } if (p5 < p4) { t = p4; p4 = p5; p5 = t; } if (p8 < p7) { t = p7; p7 = p8; p8 = t; }
    if (p1 < p0) { t = p0; p0 = p1; p1 = t; } if (p4 < p3) { t = p3; p3 = p4; p4 = t; } if (p7 < p6) { t = p6; p6 = p7; p7 = t; }
    if (p3 < p0) { t = p0; p0 = p3; p3 = t; } if (p5 < p2) { t = p2; p2 = p5; p5 = t; } if (p7 < p4) { t = p4; p4 = p7; p7 = t; }
    if (p6 < p3) { t = p3; p3 = p6; p6 = t; } if (p8 < p5) { t = p5; p5 = p8; p8 = t; } if (p4 < p1) { t = p1; p1 = p4; p4 = t; }
    if (p7 < p4) { t = p4; p4 = p7; p7 = t; } if (p4 < p3) { t = p3; p3 = p4; p4 = t; } if (p5 < p4) { t = p4; p4 = p5; p5 = t; }
    if (p4 < p2) { t = p2; p2 = p4; p4 = t; } if (p6 < p4) { t = p4; p4 = p6; p6 = t; } if (p4 < p2) { t = p2; p2 = p4; p4 = t; }
  }
  return p4;
}

export { median9 };

export function findHotPixels(p: Plane, model: NoiseModel, opts: HotPixelOptions = {}): { pixels: HotPixel[]; delta: Plane } {
  const k = opts.k ?? 10;
  const minAbs = opts.minAbs ?? 0.06;
  const nk = opts.neighbourK ?? 5;
  const minNd = opts.minNeighbourDiff ?? 0.03;
  const rangeFrac = opts.neighbourRange ?? 0.6;
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
      let nbMin = Infinity;
      let nbMax = -Infinity;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nv = d[i + dy * w + dx];
          const diff = Math.abs(d[i] - nv);
          if (diff < minDiff) minDiff = diff;
          if (nv < nbMin) nbMin = nv;
          if (nv > nbMax) nbMax = nv;
        }
      }
      if (minDiff < nk * sigma || minDiff < minNd) continue;
      // ringing next to an edge has neighbours on both sides of the edge
      if (nbMax - nbMin > rangeFrac * r) continue;
      pixels.push({ x, y, value: d[i], replacement: med[i], zScore: r / sigma });
      delta.data[i] = med[i] - d[i];
      if (pixels.length >= cap) return { pixels, delta };
    }
  }
  return { pixels, delta };
}
