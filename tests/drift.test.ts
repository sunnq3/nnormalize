import { describe, expect, it } from 'vitest';
import { driftDelta, findDriftingPatterns, measureDrift } from '../src/dsp/drift';
import { makePlane, type Plane } from '../src/dsp/plane';
import { gaussian, rng } from '../src/synth/generate';

function gaussBlur(p: Plane, s: number): Plane {
  const r = Math.ceil(s * 3);
  const k: number[] = [];
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp((-i * i) / (2 * s * s));
    k.push(v);
    sum += v;
  }
  const kk = k.map((v) => v / sum);
  const { w, h } = p;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += kk[i + r] * p.data[y * w + Math.min(w - 1, Math.max(0, x + i))];
      tmp[y * w + x] = a;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += kk[i + r] * tmp[Math.min(h - 1, Math.max(0, y + i)) * w + x];
      out[y * w + x] = a;
    }
  return { w, h, data: out };
}

/** A scene with texture at every scale (power ≈ 1/f²), a few hard edges and a bright patch. */
function scene(w: number, h: number, seed: number): Plane {
  const g = gaussian(rng(seed));
  const out = makePlane(w, h, 0.45);
  for (const s of [1, 2, 4, 8, 16, 32]) {
    const n = makePlane(w, h);
    for (let i = 0; i < w * h; i++) n.data[i] = g();
    const b = gaussBlur(n, s);
    for (let i = 0; i < w * h; i++) out.data[i] += 0.035 * s * b.data[i];
  }
  // hard edges: a dark bar and a bright square (straight edges make spectral ridges, not spikes)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (x > w * 0.3 && x < w * 0.34) out.data[y * w + x] -= 0.25;
      if (x > w * 0.6 && x < w * 0.75 && y > h * 0.55 && y < h * 0.7) out.data[y * w + x] += 0.3;
    }
  for (let i = 0; i < w * h; i++) out.data[i] = Math.min(1, Math.max(0, out.data[i]));
  return out;
}

const rms = (a: Float32Array): number => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

describe('patterns that drift across the frame', () => {
  const w = 1024;
  const h = 768;

  it('finds stripes whose spacing and direction drift, and removes them without touching anything else', () => {
    const base = scene(w, h, 3);
    // phase whose gradient (the local frequency) drifts: 0.10→0.14 cycles/px across, 0.01→0.04 down
    const pattern = new Float32Array(w * h);
    const amp = 4 / 255;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const phi = 2 * Math.PI * (0.1 * x + (0.02 * x * x) / w + 0.01 * y + (0.015 * y * y) / h);
        pattern[y * w + x] = amp * Math.cos(phi);
      }
    const Y: Plane = { w, h, data: base.data.map((v, i) => v + pattern[i]) };
    const tracks = findDriftingPatterns(Y, { minAmplitude: 0.5 / 255 });
    expect(tracks.length).toBeGreaterThanOrEqual(1);
    const t = tracks[0];
    // spacing 1/0.1 = 10 px at the left down to 1/0.14 ≈ 7 px at the right; the direction turns by ~15°
    expect(t.periodMin).toBeLessThan(8);
    expect(t.periodMax).toBeGreaterThan(9);
    expect(t.angleSpread).toBeGreaterThan(5);
    expect(t.stationary).toBe(false);
    expect(t.coverage).toBeGreaterThan(0.6);
    expect(t.amplitude * 255).toBeGreaterThan(3);
    expect(t.amplitude * 255).toBeLessThan(5);

    const d = driftDelta([Y], t)[0];
    const cleaned: Plane = { w, h, data: Y.data.map((v, i) => v + d.data[i]) };
    const before = measureDrift(Y, t);
    const after = measureDrift(cleaned, t);
    expect(after.amplitude).toBeLessThan(0.15 * before.amplitude);
    // what was taken out is the planted pattern, little else
    const err = new Float32Array(w * h);
    for (let i = 0; i < err.length; i++) err[i] = d.data[i] + pattern[i];
    expect(rms(err)).toBeLessThan(0.25 * rms(pattern));
    // the scene's own fine detail is untouched: the finest scale (nothing near the band) is identical to 1 %
    const fine = (p: Plane): number => {
      const b = gaussBlur(p, 0.8);
      return rms(p.data.map((v, i) => v - b.data[i]));
    };
    expect(Math.abs(fine(cleaned) - fine(base)) / fine(base)).toBeLessThan(0.01);
    // and nothing changed far from the band: the cleaned photo differs from the clean scene by far less than the pattern
    expect(rms(cleaned.data.map((v, i) => v - base.data[i]))).toBeLessThan(0.3 * rms(pattern));
  }, 60000);

  it('a fixed 2.5-px screen grid is found and removed down to the floor', () => {
    const base = scene(w, h, 5);
    const pattern = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pattern[y * w + x] = (1.5 / 255) * Math.cos(2 * Math.PI * 0.4 * x + 0.3);
    const Y: Plane = { w, h, data: base.data.map((v, i) => v + pattern[i]) };
    const tracks = findDriftingPatterns(Y, { minAmplitude: 0.5 / 255 });
    expect(tracks.length).toBeGreaterThanOrEqual(1);
    const t = tracks[0];
    expect(Math.abs(t.period - 2.5)).toBeLessThan(0.05);
    expect(t.stationary).toBe(true);
    const d = driftDelta([Y], t)[0];
    const cleaned: Plane = { w, h, data: Y.data.map((v, i) => v + d.data[i]) };
    expect(measureDrift(cleaned, t).amplitude * 255).toBeLessThan(0.2);
  }, 60000);

  it('reports nothing on a scene without a pattern, even with straight edges and a small striped patch', () => {
    const base = scene(w, h, 7);
    // a 120-px patch of real striped fabric: a pattern, but a local one
    for (let y = 100; y < 220; y++) for (let x = 600; x < 720; x++) base.data[y * w + x] += (6 / 255) * Math.cos(2 * Math.PI * 0.16 * (x + 0.3 * y));
    const tracks = findDriftingPatterns(base, { minAmplitude: 0.5 / 255 });
    expect(tracks.length).toBe(0);
  }, 60000);
});
