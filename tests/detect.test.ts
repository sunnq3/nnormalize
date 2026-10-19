import { describe, expect, it } from 'vitest';
import { backgroundModel } from '../src/dsp/background';
import { detectPeaks, groupHarmonics } from '../src/dsp/peaks';
import { measureAt, phaseToGlobal, refinePeak, windowedTile, wrapPhase } from '../src/dsp/refine';
import { extractTile } from '../src/dsp/plane';
import { tileGrid, welchSpectrum } from '../src/dsp/tiles';
import { window2d } from '../src/dsp/window';
import { synthesize } from '../src/synth/generate';

const planted = [
  { u: 0.2137, v: 0.1312, amplitude: 0.055, phase: 0.7 }, // oblique moire, off-bin
  { u: 0.0, v: 0.25, amplitude: 0.035, phase: -1.1 }, // scan lines, on-bin
  { u: 0.3604, v: -0.3604, amplitude: 0.025, phase: 2.0 }, // fine screen component
];

describe('tile grid', () => {
  it('covers the image with power-of-two tiles at 50 % overlap', () => {
    const g = tileGrid(1500, 900, 1024);
    expect(g.size).toBe(512);
    expect(g.xs[0]).toBe(0);
    expect(g.xs[g.xs.length - 1]).toBe(1500 - 512);
    expect(g.ys[g.ys.length - 1]).toBe(900 - 512);
    for (let i = 1; i < g.xs.length - 1; i++) expect(g.xs[i] - g.xs[i - 1]).toBe(256);
  });
  it('a small image gets one tile', () => {
    const g = tileGrid(300, 200, 1024);
    expect(g.size).toBe(128);
    expect(g.xs.length).toBeGreaterThan(1);
    expect(g.ys.length).toBeGreaterThan(1);
  });
});

describe('detection on a synthetic image', () => {
  const w = 640;
  const h = 480;
  const { noisy } = synthesize({ w, h, seed: 3, sinusoids: planted, noiseSigma: 0.01 });
  const grid = tileGrid(w, h, 1024);
  const welch = welchSpectrum(noisy, grid, 'hann');
  const model = backgroundModel(welch.power, welch.size);
  const peaks = detectPeaks(model, { zThreshold: 6, minRadiusBins: 6 });

  it('finds every planted pattern within one bin and nothing else', () => {
    expect(peaks.length).toBe(planted.length);
    const binTol = 1.0 / welch.size;
    for (const p of planted) {
      const hit = peaks.find(
        (q) =>
          (Math.abs(q.u - p.u) <= binTol && Math.abs(q.v - p.v) <= binTol) || (Math.abs(q.u + p.u) <= binTol && Math.abs(q.v + p.v) <= binTol),
      );
      expect(hit, `peak at (${p.u}, ${p.v})`).toBeDefined();
      expect(hit!.z).toBeGreaterThan(6);
    }
  });

  it('Welch power of an on-bin sinusoid equals (A/2)^2', () => {
    const p = planted[1];
    const c = welch.size >> 1;
    const bx = Math.round(p.u * welch.size) + c;
    const by = Math.round(p.v * welch.size) + c;
    const ratio = welch.power[by * welch.size + bx] / (p.amplitude / 2) ** 2;
    expect(Math.abs(ratio - 1)).toBeLessThan(0.05); // scene content on the axis adds a little
  });

  it('refines frequency to < 1e-3 cycles/px and amplitude to < 5 % from a single tile', () => {
    const n = grid.size;
    const win = window2d('hann', n, n);
    const tile = extractTile(noisy, grid.xs[0], grid.ys[0], n, n);
    const tw = windowedTile(tile, win, n, n);
    for (const p of planted) {
      const raw = peaks.find((q) => Math.abs(Math.abs(q.u) - Math.abs(p.u)) < 2 / n && Math.abs(Math.abs(q.v) - Math.abs(p.v)) < 2 / n)!;
      // detection may hold the conjugate; refine whichever sign matches the planted one
      const sign = Math.abs(raw.u - p.u) + Math.abs(raw.v - p.v) < Math.abs(raw.u + p.u) + Math.abs(raw.v + p.v) ? 1 : -1;
      const r = refinePeak(tw, n, n, win.sum, sign * raw.u, sign * raw.v, { tileSize: n });
      expect(r.converged).toBe(true);
      expect(Math.abs(r.u - p.u)).toBeLessThan(1e-3);
      expect(Math.abs(r.v - p.v)).toBeLessThan(1e-3);
      expect(Math.abs(r.amplitude - p.amplitude) / p.amplitude).toBeLessThan(0.05);
      const phase = phaseToGlobal(r.phase, r.u, r.v, grid.xs[0], grid.ys[0]);
      expect(Math.abs(wrapPhase(phase - p.phase))).toBeLessThan(0.05);
    }
  });

  it('measures the same amplitude and global phase from a different tile', () => {
    const n = grid.size;
    const win = window2d('hann', n, n);
    const x0 = grid.xs[grid.xs.length - 1];
    const y0 = grid.ys[grid.ys.length - 1];
    const tw = windowedTile(extractTile(noisy, x0, y0, n, n), win, n, n);
    const p = planted[0];
    const m = measureAt(tw, n, n, win.sum, p.u, p.v);
    expect(Math.abs(m.amplitude - p.amplitude) / p.amplitude).toBeLessThan(0.02);
    expect(Math.abs(wrapPhase(phaseToGlobal(m.phase, p.u, p.v, x0, y0) - p.phase))).toBeLessThan(0.05);
  });

  it('a clean image yields no peaks', () => {
    const { clean } = synthesize({ w, h, seed: 3, noiseSigma: 0.01 });
    const wl = welchSpectrum(clean, grid, 'hann');
    const m = backgroundModel(wl.power, wl.size);
    expect(detectPeaks(m, { zThreshold: 6 }).length).toBe(0);
  });
});

describe('harmonic grouping', () => {
  it('collects multiples of a fundamental along one direction', () => {
    const size = 1024;
    const f = { u: 0.05, v: 0.02 };
    const peaks = [
      { u: f.u, v: f.v },
      { u: 2 * f.u + 0.0004, v: 2 * f.v },
      { u: 3 * f.u, v: 3 * f.v - 0.0005 },
      { u: 0.3, v: 0.1 }, // unrelated
      { u: -4 * f.u, v: -4 * f.v }, // conjugate of the 4th harmonic
    ];
    const groups = groupHarmonics(peaks, size);
    const g0 = groups.find((g) => g.fundamental === 0)!;
    expect(g0.members).toEqual([0, 1, 2, 4]);
    expect(g0.orders).toEqual([1, 2, 3, 4]);
    expect(groups.find((g) => g.fundamental === 3)!.members).toEqual([3]);
  });
});
