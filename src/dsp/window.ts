/**
 * Analysis windows with the constants needed to turn a windowed spectrum
 * back into physical amplitudes.
 *
 *   coherentGain = mean(w)          divide a sinusoid's peak by this
 *   enbw         = N·Σw² / (Σw)²     noise-bandwidth in bins, for noise floors
 */

export type WindowKind = 'hann' | 'blackman-harris' | 'rect';

export interface Window1D {
  kind: WindowKind;
  n: number;
  w: Float64Array;
  sum: number;
  sumSq: number;
  coherentGain: number;
  enbw: number;
}

const cache = new Map<string, Window1D>();

export function window1d(kind: WindowKind, n: number): Window1D {
  const key = `${kind}:${n}`;
  const hit = cache.get(key);
  if (hit) return hit;
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n; // periodic form: Hann at 50 % overlap sums to 1 exactly
    switch (kind) {
      case 'hann':
        w[i] = 0.5 - 0.5 * Math.cos(t);
        break;
      case 'blackman-harris':
        w[i] = 0.35875 - 0.48829 * Math.cos(t) + 0.14128 * Math.cos(2 * t) - 0.01168 * Math.cos(3 * t);
        break;
      case 'rect':
        w[i] = 1;
        break;
    }
  }
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sum += w[i];
    sumSq += w[i] * w[i];
  }
  const out: Window1D = { kind, n, w, sum, sumSq, coherentGain: sum / n, enbw: (n * sumSq) / (sum * sum) };
  cache.set(key, out);
  return out;
}

export interface Window2D {
  kind: WindowKind;
  w: number;
  h: number;
  data: Float64Array; // row-major, w*h
  sum: number; // Σ w(x,y)
  sumSq: number;
  coherentGain: number; // sum / (w*h)
}

const cache2 = new Map<string, Window2D>();

export function window2d(kind: WindowKind, w: number, h: number): Window2D {
  const key = `${kind}:${w}x${h}`;
  const hit = cache2.get(key);
  if (hit) return hit;
  const wx = window1d(kind, w).w;
  const wy = window1d(kind, h).w;
  const data = new Float64Array(w * h);
  let sum = 0;
  let sumSq = 0;
  for (let y = 0; y < h; y++) {
    const a = wy[y];
    for (let x = 0; x < w; x++) {
      const v = a * wx[x];
      data[y * w + x] = v;
      sum += v;
      sumSq += v * v;
    }
  }
  const out: Window2D = { kind, w, h, data, sum, sumSq, coherentGain: sum / (w * h) };
  cache2.set(key, out);
  return out;
}
