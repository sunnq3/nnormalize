import { describe, expect, it } from 'vitest';
import { deblockDelta, detectBlockGrid, qualityFromQuantTable, STD_LUMA } from '../src/dsp/blocks';
import { findHotPixels } from '../src/dsp/hotpixels';
import { denoiseDelta, noiseDiscrepancy, noiseFromSpectrum, noiseMap, noiseModel, regrainDelta } from '../src/dsp/noise';
import type { Plane } from '../src/dsp/plane';
import { tileGrid, welchSpectrum } from '../src/dsp/tiles';
import { window2d } from '../src/dsp/window';
import { gaussian, psnr, rng, synthesize } from '../src/synth/generate';

const w = 512;
const h = 384;

describe('noise estimation', () => {
  it('recovers a known white-noise sigma within 5 % on a smooth image', () => {
    const { noisy } = synthesize({ w, h, seed: 2, noiseSigma: 0.02 });
    const map = noiseMap(noisy, 32);
    const model = noiseModel(map);
    expect(Math.abs(model.globalSigma - 0.02) / 0.02).toBeLessThan(0.05);
    // spectral route agrees
    const grid = tileGrid(w, h, 1024);
    const welch = welchSpectrum(noisy, grid);
    const win = window2d('hann', grid.size, grid.size);
    const s = noiseFromSpectrum(welch.power, welch.size, win.sum, win.sumSq);
    expect(Math.abs(s - 0.02) / 0.02).toBeLessThan(0.08);
  });

  it('flags a region that is noisier and one that was smoothed', () => {
    const { noisy } = synthesize({ w, h, seed: 2, noiseSigma: 0.015 });
    const g = gaussian(rng(9));
    // top third: extra noise; bottom third: replace with the clean plane (no noise).
    // Bands run across the image's horizontal gradient so every band spans the
    // same intensities and the intensity model can't absorb the difference.
    const { clean } = synthesize({ w, h, seed: 2 });
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (y < h / 3) noisy.data[i] += 0.03 * g();
        else if (y > (2 * h) / 3) noisy.data[i] = clean.data[i];
      }
    }
    const map = noiseMap(noisy, 32);
    const model = noiseModel(map);
    const disc = noiseDiscrepancy(map, model, 3);
    expect(disc.noisierFraction).toBeGreaterThan(0.2);
    expect(disc.smootherFraction).toBeGreaterThan(0.2);
    const top = disc.ratioDb[1 * map.bw + Math.floor(map.bw / 2)];
    const bottom = disc.ratioDb[(map.bh - 2) * map.bw + Math.floor(map.bw / 2)];
    expect(top).toBeGreaterThan(3);
    expect(bottom).toBeLessThan(-3);

    // normalising brings both regions toward the model
    const dn = denoiseDelta(noisy, map, model);
    const rg = regrainDelta(noisy, map, model, { seed: 1 });
    const out: Plane = { w, h, data: new Float32Array(w * h) };
    for (let i = 0; i < out.data.length; i++) out.data[i] = noisy.data[i] + dn.delta.data[i] + rg.delta.data[i];
    const map2 = noiseMap(out, 32);
    const disc2 = noiseDiscrepancy(map2, model, 3);
    expect(disc2.noisierFraction).toBeLessThan(disc.noisierFraction / 2);
    expect(disc2.smootherFraction).toBeLessThan(disc.smootherFraction / 2);
  });
});

describe('block grid', () => {
  function blockify(p: Plane, offX: number, offY: number, step: number): Plane {
    // add a small random constant per 8x8 block, grid shifted by (offX, offY)
    const out: Plane = { w: p.w, h: p.h, data: new Float32Array(p.data) };
    const r = rng(4);
    const bw = Math.ceil((p.w + 8) / 8);
    const bh = Math.ceil((p.h + 8) / 8);
    const off = new Float32Array(bw * bh);
    for (let i = 0; i < off.length; i++) off[i] = (r() - 0.5) * 2 * step;
    for (let y = 0; y < p.h; y++) {
      for (let x = 0; x < p.w; x++) {
        const bx = Math.floor((x - offX + 8) / 8);
        const by = Math.floor((y - offY + 8) / 8);
        out.data[y * p.w + x] += off[by * bw + bx];
      }
    }
    return out;
  }

  it('finds the grid phase and the deblocker removes most of the damage', () => {
    const { clean } = synthesize({ w, h, seed: 7 });
    const blocked = blockify(clean, 3, 5, 0.02);
    const g = detectBlockGrid(blocked);
    expect(g.detected).toBe(true);
    expect(g.offsetX).toBe(3);
    expect(g.offsetY).toBe(5);
    const none = detectBlockGrid(clean);
    expect(none.detected).toBe(false);
    const delta = deblockDelta(blocked, g);
    const fixed: Plane = { w, h, data: new Float32Array(w * h) };
    for (let i = 0; i < fixed.data.length; i++) fixed.data[i] = blocked.data[i] + delta.data[i];
    // ramping a step is a perceptual fix, not an MSE-optimal one: require the
    // grid to become much less detectable without the error getting worse
    expect(psnr(clean, fixed)).toBeGreaterThan(psnr(clean, blocked) - 1.5);
    const g2 = detectBlockGrid(fixed);
    expect(g2.strengthX).toBeLessThan(g.strengthX / 2);
    expect(g2.strengthY).toBeLessThan(g.strengthY / 2);
  });

  it('inverts the libjpeg quality scaling', () => {
    for (const q of [50, 75, 92, 97]) {
      const scale = q < 50 ? 5000 / q : 200 - 2 * q;
      const table = STD_LUMA.map((v) => Math.max(1, Math.floor((v * scale + 50) / 100)));
      expect(Math.abs(qualityFromQuantTable(table) - q)).toBeLessThan(1.5);
    }
  });
});

describe('hot pixels', () => {
  it('finds planted single-pixel outliers and nothing on edges', () => {
    const { noisy } = synthesize({ w, h, seed: 8, noiseSigma: 0.01 });
    const planted = [
      [50, 60],
      [300, 200],
      [400, 100],
    ];
    for (const [x, y] of planted) noisy.data[y * w + x] = 1.0;
    noisy.data[120 * w + 220] = 0.0; // dead pixel
    const map = noiseMap(noisy, 32);
    const model = noiseModel(map);
    const { pixels, delta } = findHotPixels(noisy, model);
    expect(pixels.length).toBe(4);
    for (const [x, y] of planted) expect(pixels.find((p) => p.x === x && p.y === y)).toBeDefined();
    // the delta restores the pixel to its neighbourhood
    const fixed = noisy.data[60 * w + 50] + delta.data[60 * w + 50];
    expect(fixed).toBeLessThan(0.9);
    expect(fixed).toBeGreaterThan(0.05);
  });
});
