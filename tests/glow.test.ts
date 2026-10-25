import { describe, expect, it } from 'vitest';
import { glowFix } from '../src/dsp/glow';
import { makePlane, type Plane } from '../src/dsp/plane';
import { gaussian, rng } from '../src/synth/generate';

/** Dark building against a sky gradient; optionally the phone's glow along the boundary. */
function scene(w: number, h: number, glow: boolean): Plane {
  const p = makePlane(w, h);
  const g = gaussian(rng(3));
  const bx = Math.floor(w * 0.55);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const sky = 0.55 + 0.25 * (1 - y / h);
      const bld = 0.18;
      const dist = x - bx;
      let v = dist < 0 ? sky : bld;
      if (glow) v += dist < 0 ? 0.035 * Math.exp(dist / 10) : -0.025 * Math.exp(-dist / 10);
      // a soft 2-px edge, light grain
      if (Math.abs(dist) < 1) v = 0.5 * (sky + bld);
      p.data[i] = v + 0.004 * g();
    }
  return p;
}

describe('glow around subjects', () => {
  it('finds the glow along a boundary and flattens it back onto the sky and the wall', () => {
    const w = 400;
    const h = 300;
    const img = scene(w, h, true);
    const r = glowFix(img, 1);
    expect(r.sides).toBeGreaterThan(50);
    expect(r.glowLevels).toBeGreaterThan(3);
    const fixed: Plane = { w, h, data: img.data.map((v, i) => v + r.delta.data[i]) };
    const clean = scene(w, h, false);
    // in the glow band, the fixed picture is much closer to the scene without glow
    const bx = Math.floor(w * 0.55);
    let e0 = 0;
    let e1 = 0;
    for (let y = 60; y < h - 60; y++)
      for (const dx of [-12, -8, -5, 5, 8, 12]) {
        const i = y * w + bx + dx;
        e0 += Math.abs(img.data[i] - clean.data[i]);
        e1 += Math.abs(fixed.data[i] - clean.data[i]);
      }
    expect(e1).toBeLessThan(e0 * 0.4);
  });

  it('leaves a clean boundary alone', () => {
    const r = glowFix(scene(400, 300, false), 1);
    let mx = 0;
    for (const v of r.delta.data) mx = Math.max(mx, Math.abs(v));
    expect(mx * 255).toBeLessThan(1);
  });
});
