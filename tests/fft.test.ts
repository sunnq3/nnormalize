import { describe, expect, it } from 'vitest';
import { FFTPlan, fft2d, fftshift2d, nextPow2, prevPow2 } from '../src/dsp/fft';
import { window1d, window2d } from '../src/dsp/window';

function naiveDft(re: Float64Array, im: Float64Array): { re: Float64Array; im: Float64Array } {
  const n = re.length;
  const or = new Float64Array(n);
  const oi = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let t = 0; t < n; t++) {
      const a = (-2 * Math.PI * k * t) / n;
      const c = Math.cos(a);
      const s = Math.sin(a);
      sr += re[t] * c - im[t] * s;
      si += re[t] * s + im[t] * c;
    }
    or[k] = sr;
    oi[k] = si;
  }
  return { re: or, im: oi };
}

describe('FFT', () => {
  it('matches a naive DFT to 1e-9 relative', () => {
    for (const n of [2, 8, 64, 256]) {
      const re = new Float64Array(n);
      const im = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        re[i] = Math.sin(i * 0.37) + (i % 7) * 0.1;
        im[i] = Math.cos(i * 0.11) * 0.5;
      }
      const ref = naiveDft(re, im);
      const r2 = new Float64Array(re);
      const i2 = new Float64Array(im);
      FFTPlan.get(n).forward(r2, i2);
      let maxErr = 0;
      let maxMag = 0;
      for (let k = 0; k < n; k++) {
        maxErr = Math.max(maxErr, Math.abs(r2[k] - ref.re[k]), Math.abs(i2[k] - ref.im[k]));
        maxMag = Math.max(maxMag, Math.hypot(ref.re[k], ref.im[k]));
      }
      expect(maxErr / maxMag).toBeLessThan(1e-9);
    }
  });

  it('inverse(forward(x)) round-trips', () => {
    const n = 1024;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.random();
      im[i] = Math.random();
    }
    const r2 = new Float64Array(re);
    const i2 = new Float64Array(im);
    const plan = FFTPlan.get(n);
    plan.forward(r2, i2);
    plan.inverse(r2, i2);
    for (let i = 0; i < n; i++) {
      expect(Math.abs(r2[i] - re[i])).toBeLessThan(1e-12);
      expect(Math.abs(i2[i] - im[i])).toBeLessThan(1e-12);
    }
  });

  it('rejects non power-of-two sizes', () => {
    expect(() => new FFTPlan(6)).toThrow();
  });

  it('2-D: a pure sinusoid lands in exactly one conjugate pair of bins', () => {
    const w = 64;
    const h = 32;
    const re = new Float64Array(w * h);
    const im = new Float64Array(w * h);
    const kx = 5;
    const ky = 3;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) re[y * w + x] = Math.cos(2 * Math.PI * ((kx * x) / w + (ky * y) / h));
    fft2d(re, im, w, h);
    const mag = (x: number, y: number): number => Math.hypot(re[y * w + x], im[y * w + x]);
    expect(mag(kx, ky)).toBeCloseTo((w * h) / 2, 6);
    expect(mag(w - kx, h - ky)).toBeCloseTo((w * h) / 2, 6);
    let other = 0;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if (!((x === kx && y === ky) || (x === w - kx && y === h - ky))) other = Math.max(other, mag(x, y));
    expect(other).toBeLessThan(1e-8);
    // and the 2-D inverse restores the input
    fft2d(re, im, w, h, true);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) expect(re[y * w + x]).toBeCloseTo(Math.cos(2 * Math.PI * ((kx * x) / w + (ky * y) / h)), 10);
  });

  it('fftshift2d moves DC to the centre and is its own inverse', () => {
    const w = 8;
    const h = 4;
    const a = new Float64Array(w * h);
    a[0] = 1;
    const s = fftshift2d(a, w, h);
    expect(s[(h / 2) * w + w / 2]).toBe(1);
    const back = fftshift2d(s, w, h);
    expect(back[0]).toBe(1);
  });

  it('pow2 helpers', () => {
    expect(nextPow2(1000)).toBe(1024);
    expect(prevPow2(1000)).toBe(512);
    expect(prevPow2(1024)).toBe(1024);
  });
});

describe('windows', () => {
  it('Hann at 50 % overlap sums to one (periodic form)', () => {
    const n = 256;
    const w = window1d('hann', n).w;
    for (let i = 0; i < n / 2; i++) expect(w[i] + w[i + n / 2]).toBeCloseTo(1, 12);
  });
  it('Hann coherent gain is 0.5 and ENBW is 1.5 bins', () => {
    const w = window1d('hann', 4096);
    expect(w.coherentGain).toBeCloseTo(0.5, 6);
    expect(w.enbw).toBeCloseTo(1.5, 4);
  });
  it('2-D window is the outer product', () => {
    const w2 = window2d('hann', 16, 8);
    const wx = window1d('hann', 16).w;
    const wy = window1d('hann', 8).w;
    expect(w2.data[3 * 16 + 5]).toBeCloseTo(wx[5] * wy[3], 12);
    expect(w2.coherentGain).toBeCloseTo(0.25, 6);
  });
});
