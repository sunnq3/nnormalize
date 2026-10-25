import { describe, expect, it } from 'vitest';
import { correctionGainDb, detailMask, fitCameraModel, measureEdgeProfile, outlineClampDelta } from '../src/dsp/edges';
import { makePlane, type Plane } from '../src/dsp/plane';
import { bandGains, detailDelta } from '../src/dsp/spectrum';
import { gaussian, rng } from '../src/synth/generate';

function gaussBlur(p: Plane, s: number): Plane {
  const r = Math.ceil(s * 4);
  const k: number[] = [];
  let sum = 0;
  for (let i = -r; i <= r; i++) {
    const v = Math.exp((-i * i) / (2 * s * s));
    k.push(v);
    sum += v;
  }
  const kk = k.map((v) => v / sum);
  const { w, h } = p;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += kk[i + r] * p.data[y * w + Math.min(w - 1, Math.max(0, x + i))];
      tmp[y * w + x] = a;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let a = 0;
      for (let i = -r; i <= r; i++) a += kk[i + r] * tmp[Math.min(h - 1, Math.max(0, y + i)) * w + x];
      out[y * w + x] = a;
    }
  return { w, h, data: out };
}

/** Flat shapes with straight and curved edges, at random angles. */
function shapes(w: number, h: number, seed: number): Plane {
  const p = makePlane(w, h, 0.3);
  const r = rng(seed);
  for (let k = 0; k < 40; k++) {
    const cx = r() * w;
    const cy = r() * h;
    const a = r() * Math.PI;
    const hw = 20 + r() * 60;
    const hh = 20 + r() * 60;
    const v = 0.15 + r() * 0.7;
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const u = dx * ca + dy * sa;
        const vv = -dx * sa + dy * ca;
        if (Math.abs(u) < hw && Math.abs(vv) < hh) p.data[y * w + x] = v;
      }
  }
  return p;
}

describe('camera blur and sharpening from edges', () => {
  it('recovers a known blur and sharpening, and the correction gives a clean narrow edge', () => {
    const w = 640;
    const h = 480;
    // supersampled shapes → a real step at pixel scale
    const sharp = shapes(w, h, 4);
    const blurred = gaussBlur(sharp, 1.6);
    const b1 = gaussBlur(blurred, 1);
    const b2 = gaussBlur(blurred, 2.5);
    const cam = makePlane(w, h);
    const g = gaussian(rng(9));
    for (let i = 0; i < w * h; i++) cam.data[i] = blurred.data[i] + 0.8 * (blurred.data[i] - b1.data[i]) + 0.5 * (blurred.data[i] - b2.data[i]) + 0.004 * g();
    const prof = measureEdgeProfile(cam)!;
    expect(prof).not.toBeNull();
    expect(prof.count).toBeGreaterThan(100);
    expect(prof.overshoot).toBeGreaterThan(0.03);
    const m = fitCameraModel(prof);
    expect(Math.abs(m.blurSigma - 1.6)).toBeLessThan(0.35);
    expect(m.amounts[0] + m.amounts[1]).toBeGreaterThan(0.8);
    const f = Float32Array.from({ length: 128 }, (_, i) => (i + 1) / 256);
    const gains = bandGains(f, correctionGainDb(m, 1.6, f, 0.004), 1);
    const d = detailDelta(cam, gains);
    // as the fix runs: the linear step, then the outline clamp beside flat areas
    const lin: Plane = { w, h, data: cam.data.map((v, i) => v + d.data[i]) };
    const c = outlineClampDelta(lin, 1);
    const fixed: Plane = { w, h, data: lin.data.map((v, i) => v + c.data[i]) };
    const after = measureEdgeProfile(fixed)!;
    // outlines mostly gone, edge no wider than before
    expect(after.overshoot).toBeLessThan(prof.overshoot * 0.4);
    // the outlines made the edge look narrower than the blur really is; without them it sits at what deblurring can reach
    expect(after.widthPx).toBeLessThan(prof.widthPx * 1.25);
    console.log('before', prof.widthPx.toFixed(2), prof.overshoot.toFixed(3), 'fit', m.blurSigma.toFixed(2), m.amounts.map((a) => a.toFixed(2)).join(','), m.rms.toFixed(4), 'after', after.widthPx.toFixed(2), after.overshoot.toFixed(3));
  });

  it('a soft camera with no sharpening is sharpened to the target width', () => {
    const w = 640;
    const h = 480;
    const cam = gaussBlur(shapes(w, h, 7), 1.8);
    const prof = measureEdgeProfile(cam)!;
    const m = fitCameraModel(prof);
    expect(m.amounts.reduce((a, b) => a + b, 0)).toBeLessThan(0.3);
    const f = Float32Array.from({ length: 128 }, (_, i) => (i + 1) / 256);
    const gains = bandGains(f, correctionGainDb(m, 1.5, f, 0.002), 1);
    const d = detailDelta(cam, gains);
    const fixed: Plane = { w, h, data: cam.data.map((v, i) => v + d.data[i]) };
    const after = measureEdgeProfile(fixed)!;
    console.log('soft before', prof.widthPx.toFixed(2), 'after', after.widthPx.toFixed(2), after.overshoot.toFixed(3));
    expect(after.widthPx).toBeLessThan(prof.widthPx * 0.9);
    expect(after.overshoot).toBeLessThan(0.06);
  });
});

describe('where sharpening may raise fine detail', () => {
  it('leaves noise on a flat area alone and acts on texture', () => {
    const w = 256;
    const h = 128;
    const g = gaussian(rng(7));
    // left half: flat grey with strong, slightly softened noise (a phone sky);
    // right half: texture with 1/f-like structure (sums of blobs at several sizes)
    const noise = makePlane(w, h);
    for (let i = 0; i < w * h; i++) noise.data[i] = 0.05 * g();
    const soft = gaussBlur(noise, 0.6);
    const tex = makePlane(w, h);
    for (const s of [1, 2, 4, 8]) {
      const n = makePlane(w, h);
      for (let i = 0; i < w * h; i++) n.data[i] = g();
      const b = gaussBlur(n, s);
      for (let i = 0; i < w * h; i++) tex.data[i] += 0.06 * s * b.data[i];
    }
    const Y = makePlane(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        Y.data[i] = 0.5 + (x < w / 2 ? soft.data[i] : tex.data[i]);
      }
    const m = detailMask(Y);
    let left = 0;
    let right = 0;
    let n = 0;
    for (let y = 20; y < h - 20; y++)
      for (let x = 20; x < w / 2 - 20; x++) {
        left += m[y * w + x];
        right += m[y * w + x + w / 2];
        n++;
      }
    // softened noise still leaks a little through; most of it is left alone
    expect(left / n).toBeLessThan(0.2);
    expect(right / n).toBeGreaterThan(0.6);
  });
});
