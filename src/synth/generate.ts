/**
 * Synthetic test images with known ground truth. Used by the test-suite and by
 * the in-app "test mode" so the tool can report its own measurement error.
 */

import { makePlane, type Plane } from '../dsp/plane';

export interface PlantedSinusoid {
  u: number; // cycles / pixel
  v: number;
  amplitude: number; // peak amplitude, plane units (0..1 scale)
  phase: number; // radians at (0, 0)
}

export interface SynthOptions {
  w: number;
  h: number;
  seed?: number;
  /** Smooth "photo-like" content: band-limited noise plus a gradient and two hard shapes. */
  content?: boolean;
  sinusoids?: PlantedSinusoid[];
  noiseSigma?: number; // white Gaussian noise, plane units
  /** Sinusoids whose amplitude ramps linearly left-to-right (non-stationary). */
  rampSinusoids?: PlantedSinusoid[];
}

/** Deterministic PRNG (mulberry32) so tests and the in-app demo are reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(r: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u: number;
    let v: number;
    let s: number;
    do {
      u = 2 * r() - 1;
      v = 2 * r() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const m = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * m;
    return u * m;
  };
}

/** Separable Gaussian blur, reflect edges. */
export function blurPlane(p: Plane, sigma: number): Plane {
  const r = Math.ceil(sigma * 3);
  const k = new Float64Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) {
    k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
    s += k[i + r];
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  const { w, h } = p;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const refl = (i: number, n: number): number => (i < 0 ? -i - 1 : i >= n ? 2 * n - i - 1 : i);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += p.data[y * w + refl(x + i, w)] * k[i + r];
      tmp[y * w + x] = a;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += tmp[refl(y + i, h) * w + x] * k[i + r];
      out[y * w + x] = a;
    }
  }
  return { w, h, data: out };
}

export function synthesize(o: SynthOptions): { clean: Plane; noisy: Plane } {
  const { w, h } = o;
  const r = rng(o.seed ?? 1);
  const g = gaussian(r);
  let clean: Plane;
  if (o.content ?? true) {
    const base = makePlane(w, h);
    for (let i = 0; i < base.data.length; i++) base.data[i] = g();
    const blurred = blurPlane(base, Math.max(4, Math.min(w, h) / 28));
    // normalise the blurred noise to a known contrast
    let mn = Infinity;
    let mx = -Infinity;
    for (const v of blurred.data) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    clean = makePlane(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let v = 0.35 + 0.3 * ((blurred.data[i] - mn) / (mx - mn)) + 0.15 * (x / w) - 0.05;
        if (x > w * 0.18 && x < w * 0.42 && y > h * 0.22 && y < h * 0.5) v += 0.18;
        const dx = x - w * 0.7;
        const dy = y - h * 0.66;
        if (dx * dx + dy * dy < (Math.min(w, h) * 0.15) ** 2) v -= 0.2;
        clean.data[i] = Math.min(1, Math.max(0, v));
      }
    }
  } else {
    clean = makePlane(w, h, 0.5);
  }
  const noisy = { w, h, data: new Float32Array(clean.data) };
  for (const s of o.sinusoids ?? []) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        noisy.data[y * w + x] += s.amplitude * Math.cos(2 * Math.PI * (s.u * x + s.v * y) + s.phase);
      }
    }
  }
  for (const s of o.rampSinusoids ?? []) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        noisy.data[y * w + x] += (s.amplitude * x) / (w - 1) * Math.cos(2 * Math.PI * (s.u * x + s.v * y) + s.phase);
      }
    }
  }
  if (o.noiseSigma) {
    for (let i = 0; i < noisy.data.length; i++) noisy.data[i] += o.noiseSigma * g();
  }
  return { clean, noisy };
}

export function psnr(a: Plane, b: Plane): number {
  let s = 0;
  for (let i = 0; i < a.data.length; i++) {
    const d = a.data[i] - b.data[i];
    s += d * d;
  }
  return 10 * Math.log10(1 / (s / a.data.length));
}
