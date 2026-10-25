import { describe, expect, it } from 'vitest';
import { measureFlatness, toneDelta, toneParams } from '../src/dsp/tone';
import { boxBlur } from '../src/dsp/noise';
import { makePlane, type Plane } from '../src/dsp/plane';
import { synthesize } from '../src/synth/generate';

/** A scene, then flattened like a phone's HDR: large-scale differences halved, shadows lifted. */
function flattened(w: number, h: number): { scene: Plane; flat: Plane } {
  const { clean } = synthesize({ w, h, seed: 4 });
  const scene = makePlane(w, h);
  for (let i = 0; i < w * h; i++) scene.data[i] = Math.min(1, Math.max(0, (clean.data[i] - 0.5) * 1.6 + 0.5));
  // a hard step: a dark building against bright sky
  for (let y = 0; y < h; y++) for (let x = Math.floor(w * 0.7); x < w; x++) scene.data[y * w + x] = 0.12 + 0.04 * Math.sin(y * 0.3);
  const base = boxBlur(boxBlur(scene.data, w, h, 24), w, h, 24);
  const flat = makePlane(w, h);
  for (let i = 0; i < w * h; i++) {
    const detail = scene.data[i] - base[i];
    flat.data[i] = 0.08 + 0.84 * (0.5 + 0.5 * (base[i] - 0.5) + 0.7 * detail);
  }
  return { scene, flat };
}

describe('contrast and depth', () => {
  it('restores depth between areas and brings the shadows down, without halos at a hard edge', () => {
    const w = 640;
    const h = 480;
    const { flat } = flattened(w, h);
    const m0 = measureFlatness(flat);
    expect(m0.shadowLevels).toBeGreaterThan(15);
    const d = toneDelta(flat, toneParams('strong', 1, 0.003), m0.shadowLevels);
    const out: Plane = { w, h, data: flat.data.map((v, i) => v + d.data[i]) };
    const m1 = measureFlatness(out);
    expect(m1.localContrast).toBeGreaterThan(m0.localContrast * 1.15);
    expect(m1.shadowLevels).toBeLessThan(m0.shadowLevels - 5);
    // no halo: along a row, just beside the hard step on the dark side, the fix changes no more than the plateau further in
    const x0 = Math.floor(w * 0.7);
    let near = 0;
    let far = 0;
    for (let y = 40; y < h - 40; y++) {
      near += d.data[y * w + x0 + 3];
      far += d.data[y * w + x0 + 60];
    }
    expect(Math.abs(near - far) / (h - 80) * 255).toBeLessThan(3);
  });
});
