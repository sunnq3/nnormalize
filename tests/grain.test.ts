import { describe, expect, it } from 'vitest';
import { grainDelta } from '../src/dsp/grain';
import { noiseMap, noiseModel } from '../src/dsp/noise';
import { makePlane, type Plane } from '../src/dsp/plane';

describe('putting grain back', () => {
  it('adds grain up to the target on smooth areas, little on texture, none near white', () => {
    const w = 384;
    const h = 256;
    const r = makePlane(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let v = 0.45;
        if (x >= 128 && x < 256) v += 0.06 * Math.sin((x + 0.4 * y) * 0.9); // weave
        if (x >= 256) v = 0.985; // near white
        r.data[i] = v;
      }
    const planes: [Plane, Plane, Plane] = [r, { w, h, data: new Float32Array(r.data) }, { w, h, data: new Float32Array(r.data) }];
    const d = grainDelta(planes, { targetLevels: 2.5, colour: 0.3, seed: 7 });
    const rms = (x0: number, x1: number, p: Plane): number => {
      let s = 0;
      let n = 0;
      for (let y = 8; y < h - 8; y++) for (let x = x0 + 8; x < x1 - 8; x++) (s += p.data[y * w + x] ** 2), n++;
      return Math.sqrt(s / n) * 255;
    };
    const smooth = rms(0, 128, d[1]);
    const weave = rms(128, 256, d[1]);
    const white = rms(256, 384, d[1]);
    expect(smooth).toBeGreaterThan(1.8);
    expect(smooth).toBeLessThan(3.5);
    expect(weave).toBeLessThan(smooth * 0.3);
    expect(white).toBeLessThan(0.3);
    // measured as grain, the smooth area now reads close to the target
    const out = makePlane(w, h);
    for (let i = 0; i < w * h; i++) out.data[i] = planes[1].data[i] + d[1].data[i];
    const sub: Plane = { w: 128, h, data: new Float32Array(128 * h) };
    for (let y = 0; y < h; y++) for (let x = 0; x < 128; x++) sub.data[y * 128 + x] = out.data[y * w + x];
    const g = noiseModel(noiseMap(sub, 32)).globalSigma * 255;
    expect(Math.abs(g - 2.5)).toBeLessThan(0.8);
    // the colour part is there but smaller than the brightness part
    let cs = 0;
    let n = 0;
    for (let i = 0; i < w * h; i++) {
      const x = i % w;
      if (x >= 120) continue;
      cs += (d[0].data[i] - d[1].data[i]) ** 2;
      n++;
    }
    expect(Math.sqrt(cs / n) * 255).toBeGreaterThan(0.3);
    expect(Math.sqrt(cs / n) * 255).toBeLessThan(smooth);
  });
});
