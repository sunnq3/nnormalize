import { describe, expect, it } from 'vitest';
import { measureTexture, textureDelta } from '../src/dsp/texture';
import { fft2d } from '../src/dsp/fft';
import { makePlane, type Plane } from '../src/dsp/plane';
import { gaussian, rng } from '../src/synth/generate';

function naturalScene(n: number, seed: number): Plane {
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  const g = gaussian(rng(seed));
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const fx = (x <= n / 2 ? x : x - n) / n;
      const fy = (y <= n / 2 ? y : y - n) / n;
      const f = Math.hypot(fx, fy);
      const a = f > 0 ? Math.exp(-2 * Math.PI * Math.PI * 0.36 * f * f) / Math.max(f, 1 / n) : 0;
      re[y * n + x] = a * g();
      im[y * n + x] = a * g();
    }
  fft2d(re, im, n, n, true);
  let s2 = 0;
  for (let i = 0; i < n * n; i++) s2 += re[i] * re[i];
  const k = 0.1 / Math.sqrt(s2 / (n * n));
  const p = makePlane(n, n);
  for (let i = 0; i < n * n; i++) p.data[i] = 0.5 + k * re[i];
  return p;
}

/** what noise reduction does: an edge-blind smoothing that weakens the finest detail */
function smear(p: Plane): Plane {
  const { w, h } = p;
  const o = makePlane(w, h);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      o.data[i] = 0.4 * p.data[i] + 0.15 * (p.data[i - 1] + p.data[i + 1] + p.data[i - w] + p.data[i + w]);
    }
  return o;
}

describe('texture recovery', () => {
  it('reads a natural scene as natural and a smeared one as weak, and brings the fine texture back', () => {
    const scene = naturalScene(512, 2);
    const m0 = measureTexture(scene);
    expect(m0.share).toBeGreaterThan(0.8);
    expect(m0.share).toBeLessThan(1.25);
    const sm = smear(smear(scene));
    const m1 = measureTexture(sm);
    expect(m1.share).toBeLessThan(0.7);
    const d = textureDelta(sm, 1);
    const fixed: Plane = { w: 512, h: 512, data: sm.data.map((v, i) => v + d.data[i]) };
    const m2 = measureTexture(fixed);
    console.log("TEX", m0.share.toFixed(2), m1.share.toFixed(2), m2.share.toFixed(2));
    // part of the way back: heavy smearing removed more than raising what is left can restore
    expect(m2.share).toBeGreaterThan(m1.share + 0.05);
    // and the result is closer to the original scene than the smeared one was
    let e1 = 0;
    let e2 = 0;
    for (let i = 0; i < scene.data.length; i++) {
      e1 += (sm.data[i] - scene.data[i]) ** 2;
      e2 += (fixed.data[i] - scene.data[i]) ** 2;
    }
    expect(e2).toBeLessThan(e1);
  });

  it('leaves a flat, noisy area alone', () => {
    const p = makePlane(256, 256);
    const g = gaussian(rng(1));
    for (let i = 0; i < p.data.length; i++) p.data[i] = 0.5 + 0.002 * g();
    const d = textureDelta(p, 1);
    let mx = 0;
    for (const v of d.data) mx = Math.max(mx, Math.abs(v));
    expect(mx * 255).toBeLessThan(0.3);
  });
});
