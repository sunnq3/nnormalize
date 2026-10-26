import { describe, expect, it } from 'vitest';
import { Engine, stageSorted } from '../src/engine/engine';
import { lumaPlane } from '../src/dsp/plane';
import { tileGrid, welchSpectrum } from '../src/dsp/tiles';
import { backgroundModel } from '../src/dsp/background';
import { detectPeaks } from '../src/dsp/peaks';

describe('engine end to end (no encode)', () => {
  const engine = new Engine();
  const info = engine.synth({
    w: 900,
    h: 700,
    seed: 21,
    sinusoids: [
      { u: 0.2137, v: 0.1312, amplitude: 0.05, phase: 0.7 },
      { u: 0, v: 0.25, amplitude: 0.03, phase: -1.1 },
    ],
    noiseSigma: 0.01,
    blocks: true,
    hotPixels: 6,
  });

  it('loads a synthetic image with a preview', () => {
    expect(info.w).toBe(900);
    expect(info.preview.scale).toBe(1);
  });

  it('finds the periodic patterns, the block grid, the noise state and the hot pixels', () => {
    const a = engine.runAnalysis({}, () => {});
    const periodic = a.findings.filter((f) => f.kind === 'periodic');
    expect(periodic.length).toBe(2);
    expect(periodic.every((f) => f.enabled)).toBe(true);
    const kinds = new Set(a.findings.map((f) => f.kind));
    expect(kinds.has('jpeg-grid')).toBe(true);
    expect(kinds.has('hotpixels')).toBe(true);
    expect(kinds.has('noise')).toBe(true);
    const hp = a.findings.find((f) => f.kind === 'hotpixels')!;
    expect(hp.kind === 'hotpixels' && hp.details.count).toBeGreaterThanOrEqual(5);
    expect(a.blockGrid.detected).toBe(true);
    expect(a.blockGrid.offsetX).toBe(0);
  }, 30000);

  it('builds layers whose composite at full strength has no residual peaks, and at zero strength equals the source', () => {
    const a = engine.analysis!;
    const layers = [];
    for (const f of stageSorted(a.findings)) {
      if (!f.enabled) continue;
      const stats = engine.buildLayer(f, () => {});
      expect(Number.isFinite(stats.rms)).toBe(true);
      layers.push({ id: f.id, strength: 1 });
    }
    const zero = engine.compose(layers, 0);
    for (let c = 0; c < 3; c++) for (let i = 0; i < 1000; i++) expect(zero[c].data[i]).toBeCloseTo(engine.source!.planes[c].data[i], 6);

    const full = engine.compose(layers, 1);
    const lum = lumaPlane(full[0], full[1], full[2]);
    const grid = tileGrid(lum.w, lum.h, 1024);
    const welch = welchSpectrum(lum, grid);
    const peaks = detectPeaks(backgroundModel(welch.power, welch.size));
    expect(peaks.length).toBe(0);

    // half strength: amplitude halves
    const half = engine.compose(layers, 0.5);
    const lumHalf = lumaPlane(half[0], half[1], half[2]);
    const wh = welchSpectrum(lumHalf, grid);
    const ph = detectPeaks(backgroundModel(wh.power, wh.size));
    expect(ph.length).toBeGreaterThanOrEqual(1);
  }, 30000);

  it('renders preview and crop in every mode', () => {
    const layers = engine.analysis!.findings.filter((f) => f.enabled).map((f) => ({ id: f.id, strength: 1 }));
    for (const mode of ['original', 'result', 'difference', 'mask'] as const) {
      const p = engine.render({ region: { kind: 'preview' }, mode, layers, master: 1, diffGain: 8, maskId: layers[0].id });
      expect(p.rgba.length).toBe(p.w * p.h * 4);
      const c = engine.render({ region: { kind: 'crop', x0: 100, y0: 100, w: 64, h: 48 }, mode, layers, master: 1, diffGain: 8, maskId: null });
      expect(c.w).toBe(64);
      expect(c.rgba[3]).toBe(255);
    }
  });

  it('presence mask follows the per-tile presence', () => {
    const f = engine.analysis!.findings.find((x) => x.kind === 'periodic')!;
    const m = engine.presenceMask(f);
    expect(m.w).toBeGreaterThan(0);
    let s = 0;
    for (const v of m.data) s += v;
    expect(s / m.data.length).toBeGreaterThan(200); // stationary pattern: present everywhere
  });
});
