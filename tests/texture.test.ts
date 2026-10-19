import { describe, expect, it } from 'vitest';
import { chromaDelta, haloBand, measureChromaNoise, measureRinging, ringingDelta, toRGB, toYCbCr } from '../src/dsp/chroma';
import { denoiseDelta, guidedFilter, noiseMap, noiseModel } from '../src/dsp/noise';
import { makePlane, type Plane } from '../src/dsp/plane';
import { gaussian, rng, synthesize } from '../src/synth/generate';

const w = 512;
const h = 384;

describe('texture vs noise', () => {
  it('scores grain low, edges and real texture high, and the smoother leaves texture alone', () => {
    // flat base so the only structure is what we plant
    const { noisy } = synthesize({ w, h, seed: 3, content: false, noiseSigma: 0.015 });
    // right third: isotropic fine texture (sand): random grains a few pixels
    // across, at ~4x the grain's strength
    const sandRaw = makePlane(w, h);
    const sg = gaussian(rng(11));
    for (let i = 0; i < w * h; i++) sandRaw.data[i] = sg();
    const sand = guidedFilter(sandRaw, 1, 1e9); // 3x3 box
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        // middle third: an oriented weave (fabric) with contrast above the grain
        if (x >= w / 3 && x < (2 * w) / 3) noisy.data[i] += 0.06 * Math.sin((x + 0.4 * y) * 0.9);
        if (x >= (2 * w) / 3) noisy.data[i] += 0.06 * 3 * sand.data[i];
      }
    }
    const map = noiseMap(noisy, 32);
    const avg = (x0: number, x1: number): number => {
      let s = 0;
      let n = 0;
      for (let by = 0; by < map.bh; by++)
        for (let bx = 0; bx < map.bw; bx++) {
          const xc = (bx + 0.5) * 32;
          if (xc > x0 + 24 && xc < x1 - 24) {
            s += map.texture[by * map.bw + bx];
            n++;
          }
        }
      return s / n;
    };
    const tNoise = avg(0, w / 3);
    const tWeave = avg(w / 3, (2 * w) / 3);
    const tSand = avg((2 * w) / 3, w);
    expect(tNoise).toBeLessThan(0.15);
    expect(tWeave).toBeGreaterThan(0.5);
    expect(tSand).toBeGreaterThan(0.5);
    const model = noiseModel(map);
    // the curve comes from the plain blocks, not the textured ones
    expect(Math.abs(model.globalSigma - 0.015) / 0.015).toBeLessThan(0.25);
    const dn = denoiseDelta(noisy, map, model, { thresholdDb: 0, targetScale: 0.3 });
    const change = (x0: number, x1: number): number => {
      let s = 0;
      let n = 0;
      for (let y = 0; y < h; y++)
        for (let x = Math.floor(x0) + 24; x < Math.floor(x1) - 24; x++) {
          s += Math.abs(dn.delta.data[y * w + x]);
          n++;
        }
      return s / n;
    };
    const cNoise = change(0, w / 3);
    const cWeave = change(w / 3, (2 * w) / 3);
    const cSand = change((2 * w) / 3, w);
    expect(cNoise).toBeGreaterThan(0.003); // grain reduced
    expect(cWeave).toBeLessThan(cNoise * 0.25); // texture left alone
    expect(cSand).toBeLessThan(cNoise * 0.25);
  });

  it('a region that is merely grainier than the rest is still treated as noise, not texture', () => {
    const { noisy } = synthesize({ w, h, seed: 2, content: false, noiseSigma: 0.015 });
    const g = gaussian(rng(9));
    for (let y = 0; y < h / 3; y++) for (let x = 0; x < w; x++) noisy.data[y * w + x] += 0.018 * g();
    const map = noiseMap(noisy, 32);
    let top = 0;
    let n = 0;
    for (let by = 0; by < Math.floor(map.bh / 3) - 1; by++) for (let bx = 0; bx < map.bw; bx++) top += map.texture[by * map.bw + bx], n++;
    expect(top / n).toBeLessThan(0.3);
  });
});

describe('colour blotches', () => {
  it('finds colour noise, removes it, and leaves brightness untouched', () => {
    const { clean } = synthesize({ w, h, seed: 5 });
    const r: Plane = { w, h, data: new Float32Array(clean.data) };
    const g: Plane = { w, h, data: new Float32Array(clean.data) };
    const b: Plane = { w, h, data: new Float32Array(clean.data) };
    // low-frequency colour blotches: smooth random Cb/Cr fields, zero in luma
    const gn = gaussian(rng(9));
    const blobCb = makePlane(w, h);
    const blobCr = makePlane(w, h);
    for (let i = 0; i < w * h; i++) {
      blobCb.data[i] = 0.02 * gn();
      blobCr.data[i] = 0.02 * gn();
    }
    const [Y0] = toYCbCr(r, g, b);
    const [R, G, B] = toRGB(Y0, blobCb, blobCr);
    const [Y, Cb, Cr] = toYCbCr(R, G, B);
    // brightness is preserved by the colour space round trip
    for (let i = 0; i < 2000; i++) expect(Math.abs(Y.data[i] - Y0.data[i])).toBeLessThan(1e-4);
    const cn = measureChromaNoise(Y, Cb, Cr, 32, 1.0);
    expect(Math.max(cn.sigmaCb, cn.sigmaCr) * 255).toBeGreaterThan(2);
    expect(cn.fraction).toBeGreaterThan(0.9);
    const { dCb, dCr } = chromaDelta(Cb, Cr, cn, 0.9, 1.0);
    const after = makePlane(w, h);
    for (let i = 0; i < w * h; i++) after.data[i] = Cb.data[i] + dCb.data[i];
    const afterCr = makePlane(w, h);
    for (let i = 0; i < w * h; i++) afterCr.data[i] = Cr.data[i] + dCr.data[i];
    const cn2 = measureChromaNoise(Y, after, afterCr, 32, 1.0);
    expect(Math.max(cn2.sigmaCb, cn2.sigmaCr)).toBeLessThan(Math.max(cn.sigmaCb, cn.sigmaCr) * 0.5);
  });
});

describe('edge halos', () => {
  it('detects extra noise beside hard edges and smooths only that band', () => {
    const { clean } = synthesize({ w, h, seed: 7 });
    const img: Plane = { w, h, data: new Float32Array(clean.data) };
    // a hard-edged square
    for (let y = 100; y < 300; y++) for (let x = 150; x < 350; x++) img.data[y * w + x] += 0.4;
    for (let i = 0; i < img.data.length; i++) img.data[i] = Math.min(1, img.data[i]);
    const gn = gaussian(rng(4));
    // light grain everywhere, strong noise in an 8 px band around the square
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = Math.min(Math.abs(x - 150), Math.abs(x - 349));
        const dy = Math.min(Math.abs(y - 100), Math.abs(y - 299));
        const inX = x >= 142 && x <= 357;
        const inY = y >= 92 && y <= 307;
        const near = inX && inY && (dx <= 8 || dy <= 8) && !(dx <= 1 || dy <= 1);
        img.data[y * w + x] += (near ? 0.05 : 0.006) * gn();
      }
    }
    const m = measureRinging(img, 0.12, 8);
    expect(m.ratio).toBeGreaterThan(2);
    const { halo } = haloBand(img, 0.12, 8);
    const d = ringingDelta(img, halo, m.haloSigma, 0.9);
    const fixed: Plane = { w, h, data: new Float32Array(w * h) };
    for (let i = 0; i < w * h; i++) fixed.data[i] = img.data[i] + d.data[i];
    const m2 = measureRinging(fixed, 0.12, 8);
    expect(m2.haloLevels).toBeLessThan(m.haloLevels * 0.6);
    // nothing changed away from any edge
    let far = 0;
    for (let y = 0; y < 60; y++) for (let x = 0; x < w; x++) far = Math.max(far, Math.abs(d.data[y * w + x]));
    expect(far).toBeLessThan(1e-6);
  });
});
