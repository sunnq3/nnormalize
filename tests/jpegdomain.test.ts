import { describe, expect, it } from 'vitest';
import { detectBlockGrid } from '../src/dsp/blocks';
import { jpegYccToRgb, lumaQuantTable, rgbToJpegYcc, settleBlocks } from '../src/dsp/jpegdomain';
import { makePlane, type Plane } from '../src/dsp/plane';
import { synthesize } from '../src/synth/generate';

// quantise a plane on the 8x8 grid at the given quality (a plain re-save), via a naive DCT
function requantise(p: Plane, quality: number): Plane {
  const Q = lumaQuantTable(quality);
  const out = makePlane(p.w, p.h);
  out.data.set(p.data);
  const cos = (k: number, x: number): number => (k === 0 ? Math.SQRT1_2 : 1) * 0.5 * Math.cos(((2 * x + 1) * k * Math.PI) / 16);
  const bw = Math.floor(p.w / 8);
  const bh = Math.floor(p.h / 8);
  const c = new Float64Array(64);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      for (let v = 0; v < 8; v++)
        for (let u = 0; u < 8; u++) {
          let s = 0;
          for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) s += cos(u, x) * cos(v, y) * (p.data[(by * 8 + y) * p.w + bx * 8 + x] * 255 - 128);
          c[v * 8 + u] = Math.round(s / Q[v * 8 + u]) * Q[v * 8 + u];
        }
      for (let y = 0; y < 8; y++)
        for (let x = 0; x < 8; x++) {
          let s = 0;
          for (let v = 0; v < 8; v++) for (let u = 0; u < 8; u++) s += cos(u, x) * cos(v, y) * c[v * 8 + u];
          out.data[(by * 8 + y) * p.w + bx * 8 + x] = Math.max(0, Math.min(1, (s + 128) / 255));
        }
    }
  }
  return out;
}

describe('block settling at the destination quality', () => {
  it('lowers block-boundary steps of a quality-75 copy and holds through another quality-75 re-save', () => {
    const w = 256;
    const h = 192;
    const { noisy } = synthesize({ w, h, seed: 5, noiseSigma: 0.012 });
    const saved = requantise(noisy, 75);
    const g0 = detectBlockGrid(saved);
    const step0 = Math.max(g0.stepX, g0.stepY) * 255;
    expect(step0).toBeGreaterThan(0.3);
    const settled = settleBlocks(saved, 75);
    expect(settled.changed).toBeGreaterThan(settled.blocks * 0.1);
    const g1 = detectBlockGrid(settled.plane);
    const step1 = Math.max(g1.stepX, g1.stepY) * 255;
    console.log('settle: steps', step0.toFixed(2), '->', step1.toFixed(2), 'levels; blocks changed', settled.changed, '/', settled.blocks, 'mean change', (settled.meanChange * 255).toFixed(2));
    expect(step1).toBeLessThan(step0 * 0.6);
    // the settled pixels are a fixed point of the same quantiser
    const again = requantise(settled.plane, 75);
    let maxd = 0;
    for (let i = 0; i < again.data.length; i++) maxd = Math.max(maxd, Math.abs(again.data[i] - settled.plane.data[i]));
    expect(maxd * 255).toBeLessThan(0.05);
    // and the change from what was wanted stays small
    expect(settled.meanChange * 255).toBeLessThan(1.5);
  });

  it('JPEG colour conversion round-trips', () => {
    const w = 64;
    const h = 64;
    const { clean } = synthesize({ w, h, seed: 1 });
    const r = clean;
    const g: Plane = { w, h, data: clean.data.map((v) => v * 0.7) };
    const b: Plane = { w, h, data: clean.data.map((v) => 1 - v * 0.5) };
    const [Y, Cb, Cr] = rgbToJpegYcc(r, g, b);
    const [r2, g2, b2] = jpegYccToRgb(Y, Cb, Cr);
    for (let i = 0; i < w * h; i += 97) {
      expect(Math.abs(r2.data[i] - r.data[i])).toBeLessThan(1e-5);
      expect(Math.abs(g2.data[i] - g.data[i])).toBeLessThan(1e-5);
      expect(Math.abs(b2.data[i] - b.data[i])).toBeLessThan(1e-5);
    }
  });
});
