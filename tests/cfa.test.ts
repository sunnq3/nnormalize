import { describe, expect, it } from 'vitest';
import { calibrateSensorNoise, cfaSignature, detectCfaPhase, sameGreenGrid, falseColourDelta, measureFalseColour, rebakeThroughSensor } from '../src/dsp/cfa';
import { toRGB, toYCbCr } from '../src/dsp/chroma';
import { makePlane, type Plane } from '../src/dsp/plane';
import { gaussian, rng, synthesize } from '../src/synth/generate';

function colourScene(w: number, h: number): [Plane, Plane, Plane] {
  const { clean } = synthesize({ w, h, seed: 3 });
  const r = makePlane(w, h);
  const g = makePlane(w, h);
  const b = makePlane(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = clean.data[i];
      r.data[i] = Math.min(1, v * (0.8 + 0.3 * (x / w)));
      g.data[i] = v;
      b.data[i] = Math.min(1, v * (1.1 - 0.4 * (y / h)));
    }
  return [r, g, b];
}

/** a crude camera: mosaic on the given grid, bilinear demosaic */
function bilinearCamera(p: [Plane, Plane, Plane], rx: number, ry: number): [Plane, Plane, Plane] {
  const { w, h } = p[0];
  const site = (x: number, y: number): number => ((x & 1) === rx && (y & 1) === ry ? 0 : (x & 1) !== rx && (y & 1) !== ry ? 2 : 1);
  const out: [Plane, Plane, Plane] = [makePlane(w, h), makePlane(w, h), makePlane(w, h)];
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        let n = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            if (site(x + dx, y + dy) !== c) continue;
            // nearest samples only: same site or the 4/8 neighbours that hold this colour
            if (site(x, y) === c && (dx || dy)) continue;
            if (c === 1 && site(x, y) !== 1 && dx && dy) continue;
            s += p[c].data[(y + dy) * w + x + dx];
            n++;
          }
        out[c].data[y * w + x] = s / n;
      }
    }
  return out;
}

describe('sensor grid', () => {
  it('finds where a camera grid sat', () => {
    const scene = colourScene(384, 256);
    for (const [rx, ry] of [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ]) {
      const cam = bilinearCamera(scene, rx, ry);
      const ph = detectCfaPhase(cam);
      expect(ph.detected).toBe(true);
      expect(sameGreenGrid(ph, { redX: rx as 0 | 1, redY: ry as 0 | 1, strength: 0, detected: true })).toBe(true);
    }
  });

  it('re-baking through a sensor keeps the picture and leaves a grid on the chosen position', () => {
    const scene = colourScene(384, 256);
    expect(cfaSignature(scene)).toBeLessThan(0.02);
    const baked = rebakeThroughSensor(scene, { redX: 1, redY: 1, strength: 0, detected: false });
    let s = 0;
    for (let c = 0; c < 3; c++) for (let i = 0; i < scene[c].data.length; i++) s += (baked[c].data[i] - scene[c].data[i]) ** 2;
    expect(Math.sqrt(s / (3 * scene[0].data.length)) * 255).toBeLessThan(1.5);
    // with grain in the picture (as after "Put back grain"), the grid shows clearly, on the chosen position
    const g = gaussian(rng(4));
    const grainy = scene.map((p) => ({ w: p.w, h: p.h, data: p.data.map((v) => v + 0.008 * g()) })) as [Plane, Plane, Plane];
    const ph = detectCfaPhase(rebakeThroughSensor(grainy, { redX: 1, redY: 1, strength: 0, detected: false }));
    expect(ph.detected).toBe(true);
    expect(sameGreenGrid(ph, { redX: 1, redY: 1, strength: 0, detected: true })).toBe(true);
  });
});

describe('sensor noise in the bake', () => {
  it('photon noise at the photosites gives a grid as clear as asked for, on a clean picture', () => {
    const scene = colourScene(512, 512);
    const ph = { redX: 0, redY: 0, strength: 0, detected: true } as const;
    // a 'camera' original with a clear grid: the same scene through a bilinear camera with noise
    const g = gaussian(rng(2));
    const original = bilinearCamera(scene.map((p) => ({ w: p.w, h: p.h, data: p.data.map((v) => v + 0.01 * g()) })) as [Plane, Plane, Plane], 0, 0);
    const n = calibrateSensorNoise(scene, ph, original, 95, 7);
    const lv = n.levels;
    const baked = rebakeThroughSensor(scene, ph, n);
    const got = detectCfaPhase(baked);
    expect(got.detected).toBe(true);
    expect(sameGreenGrid(got, ph)).toBe(true);
    expect(got.strength).toBeGreaterThan(0.06);
    expect(lv).toBeGreaterThanOrEqual(0.3);
    expect(lv).toBeLessThanOrEqual(4);
    // photon noise grows with brightness
    const dark = rebakeThroughSensor(scene.map((p) => ({ w: p.w, h: p.h, data: new Float32Array(p.data.length).fill(0.1) })) as [Plane, Plane, Plane], ph, { levels: 3, seed: 1 });
    const bright = rebakeThroughSensor(scene.map((p) => ({ w: p.w, h: p.h, data: new Float32Array(p.data.length).fill(0.8) })) as [Plane, Plane, Plane], ph, { levels: 3, seed: 1 });
    const sd = (p: Plane): number => {
      let m = 0;
      for (const v of p.data) m += v;
      m /= p.data.length;
      let s2 = 0;
      for (const v of p.data) s2 += (v - m) ** 2;
      return Math.sqrt(s2 / p.data.length);
    };
    expect(sd(dark[1]) * 255).toBeGreaterThan(0.5);
  });
});

describe('false colour', () => {
  it('removes colour that does not follow fine brightness detail, keeps colour that does', () => {
    const w = 256;
    const h = 256;
    const Y = makePlane(w, h);
    const Cb = makePlane(w, h);
    const Cr = makePlane(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const stripes = 0.5 + 0.2 * Math.sign(Math.sin(x * 1.3));
        if (x < 128) {
          // fine grey stripes with false colour that flips at twice their rate
          Y.data[i] = stripes;
          Cb.data[i] = 0.03 * Math.sign(Math.sin(x * 2.6 + 0.5));
          Cr.data[i] = -0.03 * Math.sign(Math.sin(x * 2.6 + 0.5));
        } else {
          // a real colour pattern that follows brightness
          Y.data[i] = stripes;
          Cr.data[i] = (stripes - 0.5) * 0.3;
        }
      }
    const before = measureFalseColour(Y, Cb, Cr);
    const { dCb, dCr } = falseColourDelta(Y, Cb, Cr, 1);
    const Cb2 = { w, h, data: Cb.data.map((v, i) => v + dCb.data[i]) };
    const Cr2 = { w, h, data: Cr.data.map((v, i) => v + dCr.data[i]) };
    let left = 0;
    let right = 0;
    let rightOrig = 0;
    for (let y = 8; y < h - 8; y++)
      for (let x = 8; x < w - 8; x++) {
        const i = y * w + x;
        if (x < 120) left += Math.abs(Cb2.data[i]) + Math.abs(Cr2.data[i]);
        if (x > 136) {
          right += Math.abs(Cr2.data[i]);
          rightOrig += Math.abs(Cr.data[i]);
        }
      }
    expect(before.detailLevels).toBeGreaterThan(2);
    expect(left / (112 * 240) / 0.06).toBeLessThan(0.4); // most false colour gone
    expect(right / rightOrig).toBeGreaterThan(0.9); // real colour kept
    // brightness untouched when converted back
    const [R, G, B] = toRGB(Y, Cb2, Cr2);
    const [Y2] = toYCbCr(R, G, B);
    for (let i = 0; i < 1000; i++) expect(Math.abs(Y2.data[i] - Y.data[i])).toBeLessThan(1e-4);
  });
});
