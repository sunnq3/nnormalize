import { describe, expect, it } from 'vitest';
import { bandingStrength, debandDelta, detectBanding } from '../src/dsp/banding';
import { noiseMap } from '../src/dsp/noise';
import { makePlane, type Plane } from '../src/dsp/plane';
import { achievedGainDb, bandGains, detailBalance, detailDelta, radialProfile, wantedGainDb } from '../src/dsp/spectrum';
import { gaussian, rng } from '../src/synth/generate';
import { fft2d } from '../src/dsp/fft';

// a scene with natural statistics: random phases, amplitude ∝ 1/f (power ∝ 1/f²), made in the frequency domain
function naturalScene(w: number, h: number, seed: number): Plane {
  const n = 512;
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  const g = gaussian(rng(seed));
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const fx = (x <= n / 2 ? x : x - n) / n;
      const fy = (y <= n / 2 ? y : y - n) / n;
      const f = Math.hypot(fx, fy);
      const a = f > 0 ? 1 / Math.max(f, 1 / n) : 0;
      re[y * n + x] = a * g();
      im[y * n + x] = a * g();
    }
  fft2d(re, im, n, n, true);
  let s2 = 0;
  for (let i = 0; i < n * n; i++) s2 += re[i] * re[i];
  const k = 0.08 / Math.sqrt(s2 / (n * n));
  const p = makePlane(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) p.data[y * w + x] = 0.5 + k * re[(y % n) * n + (x % n)];
  return p;
}

function blur(p: Plane, passes: number): Plane {
  let d = p.data;
  const { w, h } = p;
  for (let k = 0; k < passes; k++) {
    const o = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        o[i] = (4 * d[i] + d[i - 1] + d[i + 1] + d[i - w] + d[i + w]) / 8;
      }
    d = o;
  }
  return { w, h, data: d };
}

describe('detail balance', () => {
  it('reads a natural scene as balanced, a smoothed one as short of fine detail, and brings it back', () => {
    const w = 512;
    const h = 512;
    const scene = naturalScene(w, h, 3);
    const b0 = detailBalance(radialProfile(scene), 0.002);
    expect(Math.abs(b0.deficitDb)).toBeLessThan(3);
    const smooth = blur(scene, 5);
    const b1 = detailBalance(radialProfile(smooth), 0.002);
    expect(b1.deficitDb).toBeGreaterThan(4);
    const want = wantedGainDb(b1, 'smooth', 8);
    const gains = bandGains(b1.f, want, 1);
    const got = achievedGainDb(b1.f, gains);
    // the bank boosts the fine band
    const at = (f: number): number => got[Math.round(f * 256) - 1];
    expect(at(0.2)).toBeGreaterThan(3);
    expect(Math.abs(at(0.02))).toBeLessThan(1);
    const d = detailDelta(smooth, gains);
    const fixed: Plane = { w, h, data: smooth.data.map((v, i) => v + d.data[i]) };
    const b2 = detailBalance(radialProfile(fixed), 0.002, b1.line);
    expect(b2.deficitDb).toBeLessThan(b1.deficitDb * 0.6);
  });

  it('reads an oversharpened scene as excess and tames it', () => {
    const w = 512;
    const h = 512;
    const scene = naturalScene(w, h, 5);
    // phone-style unsharp mask on a balanced scene
    const sharp: Plane = { w, h, data: new Float32Array(w * h) };
    const bl = blur(scene, 3);
    for (let i = 0; i < w * h; i++) sharp.data[i] = scene.data[i] + 1.5 * (scene.data[i] - bl.data[i]);
    const b1 = detailBalance(radialProfile(sharp), 0.002);
    expect(b1.excessDb).toBeGreaterThan(3);
    const gains = bandGains(b1.f, wantedGainDb(b1, 'sharp', 6), 1);
    const d = detailDelta(sharp, gains);
    const fixed: Plane = { w, h, data: sharp.data.map((v, i) => v + d.data[i]) };
    const b2 = detailBalance(radialProfile(fixed), 0.002, b1.line);
    expect(b2.excessDb).toBeLessThan(b1.excessDb * 0.5);
  });
});

describe('banding', () => {
  it('finds steps in a clean 8-bit gradient, smooths them into a ramp, leaves a hard edge alone', () => {
    const w = 512;
    const h = 256;
    const g = makePlane(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let v = 0.3 + (x / w) * 0.05; // ~13 levels over 512 px: bands ~40 px wide
        if (y >= 200) v += 0.3; // a hard edge
        g.data[y * w + x] = Math.round(v * 255) / 255;
      }
    const map = noiseMap(g, 32);
    const bands = detectBanding(g, map);
    expect(bands.fraction).toBeGreaterThan(0.3);
    expect(bands.stepLevels).toBe(1);
    const before = bandingStrength(g, bands.banded, bands.bw, bands.bh, bands.block);
    expect(before).toBeGreaterThan(0.15);
    const d = debandDelta(g, bands, { amount: 1, stepLevels: bands.stepLevels, seed: 3, ditherLevels: 0 });
    const out: Plane = { w, h, data: g.data.map((v, i) => v + d.data[i]) };
    const after = bandingStrength(out, bands.banded, bands.bw, bands.bh, bands.block);
    expect(after).toBeLessThan(before * 0.4);
    // the hard edge keeps its height
    const x = 256;
    expect(out.data[210 * w + x] - out.data[190 * w + x]).toBeGreaterThan(0.28);
  });

  it('does not call grainy or textured areas banded', () => {
    const w = 512;
    const h = 256;
    const g = makePlane(w, h);
    const n = gaussian(rng(2));
    for (let i = 0; i < w * h; i++) g.data[i] = Math.round((0.4 + ((i % w) / w) * 0.05 + 0.006 * n()) * 255) / 255;
    const bands = detectBanding(g, noiseMap(g, 32));
    expect(bands.fraction).toBeLessThan(0.05);
  });
});
