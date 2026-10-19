import { describe, expect, it } from 'vitest';
import { backgroundModel } from '../src/dsp/background';
import { parametricDeltaGlobal, parametricDeltaTiled, spectralDelta } from '../src/dsp/filter';
import { measurePattern } from '../src/dsp/pattern';
import { detectPeaks } from '../src/dsp/peaks';
import { tileGrid, welchSpectrum } from '../src/dsp/tiles';
import { psnr, synthesize } from '../src/synth/generate';
import type { Plane } from '../src/dsp/plane';

function apply(p: Plane, delta: Plane, strength = 1): Plane {
  const d = new Float32Array(p.data.length);
  for (let i = 0; i < d.length; i++) d[i] = p.data[i] + strength * delta.data[i];
  return { w: p.w, h: p.h, data: d };
}

const w = 700;
const h = 520;

describe('stationary pattern', () => {
  const planted = [
    { u: 0.2137, v: 0.1312, amplitude: 0.055, phase: 0.7 },
    { u: 0.0, v: 0.25, amplitude: 0.035, phase: -1.1 },
  ];
  const { clean, noisy } = synthesize({ w, h, seed: 11, sinusoids: planted, noiseSigma: 0.008 });
  const grid = tileGrid(w, h, 1024);
  const welch = welchSpectrum(noisy, grid);
  const model = backgroundModel(welch.power, welch.size);
  const peaks = detectPeaks(model);

  it('is measured as stationary with the right amplitude and phase from the whole image', () => {
    expect(peaks.length).toBe(2);
    for (const p of planted) {
      const raw = peaks.find((q) => Math.abs(Math.abs(q.u) - Math.abs(p.u)) < 2 / grid.size && Math.abs(Math.abs(q.v) - Math.abs(p.v)) < 2 / grid.size)!;
      const m = measurePattern(noisy, grid, raw.u, raw.v);
      // the detector may hold the conjugate; a conjugate has (-u, -v, -phase)
      const sign = Math.abs(m.u - p.u) < Math.abs(m.u + p.u) || Math.abs(m.v - p.v) < Math.abs(m.v + p.v) ? 1 : -1;
      expect(Math.abs(sign * m.u - p.u)).toBeLessThan(1e-3);
      expect(Math.abs(sign * m.v - p.v)).toBeLessThan(1e-3);
      expect(Math.abs(m.amplitude - p.amplitude) / p.amplitude).toBeLessThan(0.01);
      expect(m.stationary).toBe(true);
      expect(m.phaseCoherence).toBeGreaterThan(0.95);
    }
  });

  it('parametric subtraction removes it to the noise floor (global and tiled agree)', () => {
    const before = psnr(clean, noisy);
    let global = noisy;
    let tiled = noisy;
    for (const raw of peaks) {
      const m = measurePattern(noisy, grid, raw.u, raw.v);
      global = apply(global, parametricDeltaGlobal(w, h, m.u, m.v, m.amplitude, m.phase));
      tiled = apply(tiled, parametricDeltaTiled(w, h, grid, m));
    }
    const noiseOnly = { w, h, data: new Float32Array(clean.data) };
    // the ceiling is the planted white noise alone
    const { noisy: ceilingImg } = synthesize({ w, h, seed: 11, noiseSigma: 0.008 });
    const ceiling = psnr(clean, ceilingImg);
    const afterGlobal = psnr(clean, global);
    const afterTiled = psnr(clean, tiled);
    expect(afterGlobal).toBeGreaterThan(before + 10);
    expect(ceiling - afterGlobal).toBeLessThan(0.3);
    expect(ceiling - afterTiled).toBeLessThan(1.0); // per-tile estimates carry a little more noise
    // and the detector finds nothing afterwards
    const w2 = welchSpectrum(global, grid);
    expect(detectPeaks(backgroundModel(w2.power, w2.size)).length).toBe(0);
    void noiseOnly;
  });

  it('spectral flatten and notch also remove it, with the usual small leakage cost', () => {
    const targets = peaks.map((p) => ({ u: p.u, v: p.v, widthBins: p.widthBins }));
    const before = psnr(clean, noisy);
    const flat = apply(noisy, spectralDelta(noisy, grid, targets, { mode: 'flatten' }));
    const notch = apply(noisy, spectralDelta(noisy, grid, targets, { mode: 'notch' }));
    expect(psnr(clean, flat)).toBeGreaterThan(before + 8);
    expect(psnr(clean, notch)).toBeGreaterThan(before + 8);
    const w2 = welchSpectrum(flat, grid);
    expect(detectPeaks(backgroundModel(w2.power, w2.size)).length).toBe(0);
  });

  it('half strength removes half the amplitude', () => {
    const raw = peaks[0];
    const m = measurePattern(noisy, grid, raw.u, raw.v);
    const halfImg = apply(noisy, parametricDeltaGlobal(w, h, m.u, m.v, m.amplitude, m.phase), 0.5);
    const m2 = measurePattern(halfImg, grid, m.u, m.v, { refine: false });
    expect(Math.abs(m2.amplitude / m.amplitude - 0.5)).toBeLessThan(0.02);
  });
});

describe('non-stationary pattern', () => {
  it('is flagged as such and the tiled subtraction beats the global one', () => {
    const ramp = [{ u: 0.17, v: 0.09, amplitude: 0.08, phase: 0.3 }];
    const { clean, noisy } = synthesize({ w, h, seed: 5, rampSinusoids: ramp, noiseSigma: 0.005 });
    const grid = tileGrid(w, h, 256); // small tiles so the ramp spans several columns of them
    const welch = welchSpectrum(noisy, grid);
    const peaks = detectPeaks(backgroundModel(welch.power, welch.size));
    expect(peaks.length).toBe(1);
    const m = measurePattern(noisy, grid, peaks[0].u, peaks[0].v);
    expect(m.stationary).toBe(false);
    expect(m.amplitudeCv).toBeGreaterThan(0.35);
    const global = apply(noisy, parametricDeltaGlobal(w, h, m.u, m.v, m.amplitude, m.phase));
    const tiled = apply(noisy, parametricDeltaTiled(w, h, grid, m));
    expect(psnr(clean, tiled)).toBeGreaterThan(psnr(clean, global) + 3);
    expect(psnr(clean, tiled)).toBeGreaterThan(psnr(clean, noisy) + 8);
    // presence map rises left to right
    const left = m.presence[0];
    const right = m.presence[m.tilesX - 1];
    expect(right).toBeGreaterThan(left * 2);
  });
});
