import { describe, expect, it } from 'vitest';
import { newPrnuStore, packPrnuStore, prnuAccumulate, prnuApply, prnuCalibrate, prnuFinalize, prnuMeasure, unpackPrnuStore } from '../src/dsp/prnu';
import { lumaPlane, makePlane, type Plane } from '../src/dsp/plane';
import { gaussian, rng } from '../src/synth/generate';

const w = 640;
const h = 512;
const g = gaussian(rng(11));
// the sensor's own gain pattern, 0.5 % rms, and the artefacts every camera of the model shares
const K = new Float32Array(w * h);
for (let i = 0; i < K.length; i++) K[i] = 0.005 * g();
const colStripe = new Float32Array(w);
for (let x = 0; x < w; x++) colStripe[x] = 0.002 * g();

function frame(seed: number, withSignature: boolean, level = 0.55): [Plane, Plane, Plane] {
  const n = gaussian(rng(seed));
  return [0, 1, 2].map((c) => {
    const p = makePlane(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        // a soft lamp fall-off, a little colour, photon-like noise of 0.8 %
        const base = level * (1 - 0.15 * ((x - w / 2) ** 2 + (y - h / 2) ** 2) / (w * w)) * (c === 0 ? 1.03 : c === 2 ? 0.97 : 1);
        let v = base * (1 + (withSignature ? K[i] : 0) + colStripe[x] + 0.0015 * Math.cos((2 * Math.PI * x) / 8)) + 0.008 * Math.sqrt(base) * n();
        p.data[i] = Math.min(1, Math.max(0, v));
      }
    return p;
  }) as [Plane, Plane, Plane];
}

const corr = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < a.length; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= a.length;
  mb /= a.length;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb);
};

describe('the sensor signature (PRNU)', () => {
  const store = newPrnuStore('test|640x512', w, h);
  for (let k = 0; k < 24; k++) prnuAccumulate(store, frame(100 + k, true, 0.4 + 0.3 * ((k * 7) % 10) / 10), lumaPlane(...frame(100 + k, true)));
  const map = prnuFinalize(store);

  it('recovers the planted pattern from flat frames, with the shared artefacts stripped out', () => {
    expect(corr(map.k[1], K)).toBeGreaterThan(0.8);
    expect(map.reliability).toBeGreaterThan(0.6);
    // the column stripe and the 8-px pattern every camera of the model shares are not in the map
    const colOf = new Float32Array(w * h);
    for (let i = 0; i < colOf.length; i++) colOf[i] = colStripe[i % w];
    expect(Math.abs(corr(map.k[1], colOf))).toBeLessThan(0.05);
    const cosOf = new Float32Array(w * h);
    for (let i = 0; i < cosOf.length; i++) cosOf[i] = Math.cos((2 * Math.PI * (i % w)) / 8);
    expect(Math.abs(corr(map.k[1], cosOf))).toBeLessThan(0.05);
  });

  it('tells a photo from this sensor from one without the signature', () => {
    const yes = prnuMeasure(frame(900, true), map);
    const no = prnuMeasure(frame(901, false), map);
    expect(yes.z).toBeGreaterThan(8);
    expect(Math.abs(no.z)).toBeLessThan(4);
  });

  it('puts the signature back to the strength a photo had, scaled by the map’s reliability', () => {
    const original = frame(902, true);
    const target = prnuMeasure(original, map).rho;
    const stripped = frame(902, false);
    const { gain, rho } = prnuCalibrate(stripped, map, target);
    expect(gain).toBeGreaterThan(0.3);
    expect(Math.abs(rho - target)).toBeLessThan(0.1 * target);
    // a photo that never had the signature (another camera) gets none
    expect(prnuCalibrate(stripped, map, 0).gain).toBe(0);
    // applying at gain 0 changes nothing
    const same = prnuApply(stripped, map, 0);
    expect(same[1].data[1234]).toBe(stripped[1].data[1234]);
  });

  it('survives packing to 16 bits and back', () => {
    const { header, buffers } = packPrnuStore(store);
    const back = unpackPrnuStore(header, buffers);
    expect(back.photos).toBe(store.photos);
    const m2 = prnuFinalize(back);
    expect(corr(m2.k[1], map.k[1])).toBeGreaterThan(0.999);
  });
});
