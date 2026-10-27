/**
 * The sensor's own signature: photo-response non-uniformity (PRNU).
 *
 * Every photosite has its own gain, a fraction of a percent off its
 * neighbours', fixed for the life of the sensor. That pattern rides on every
 * photo the camera takes, multiplied by the light at each pixel. Clean-up,
 * and above all a re-save at a low quality, wears it away; nothing can put
 * the real pattern back except the pattern itself, learned from photos of the
 * same camera.
 *
 * The estimate follows the forensics literature (Lukáš, Fridrich & Goljan,
 * IEEE TIFS 2006; Chen, Fridrich, Goljan & Lukáš 2008, the maximum-likelihood
 * form; and the Binghamton reference implementation):
 *
 *   residual   W = I − denoise(I)            (what the scene does not explain)
 *   estimate   K = Σ w·W·I / Σ w·I²          (gain that best explains the residuals,
 *                                             bright flat pixels weighing most)
 *   clean-up   rows and columns zero-meaned, then a Wiener filter in the
 *              frequency domain that suppresses spikes: the JPEG grid, the
 *              colour-filter lattice, column readout, which every camera of the
 *              model shares and which are not the sensor's own pattern.
 *
 * Reliability is measured, not assumed: a split-half pair of estimates on a
 * centre window gives the correlation between two independent halves of the
 * batch, and Spearman–Brown turns that into the reliability of the whole.
 * The pattern is applied scaled by that reliability (the least-error scaling
 * of a noisy estimate) and calibrated so the photo carries as much of the
 * signature as it did before the clean-up.
 */

import { fft2d } from './fft';
import { boxBlur } from './noise';
import { makePlane, type Plane } from './plane';
import { window2d } from './window';

// --- residual and weights ---------------------------------------------------------------

/** Noise level of a plane: 1.5 × the robust spread of its finest detail. */
function noiseSigma(p: Plane): number {
  const { w, h, data } = p;
  const low = boxBlur(data, w, h, 1);
  const n = Math.min(200000, data.length);
  const step = Math.max(1, Math.floor(data.length / n));
  const s = new Float32Array(Math.floor(data.length / step));
  for (let i = 0, k = 0; k < s.length; i += step, k++) s[k] = data[i] - low[i];
  s.sort();
  const med = s[s.length >> 1];
  for (let k = 0; k < s.length; k++) s[k] = Math.abs(s[k] - med);
  s.sort();
  return 1.4826 * s[s.length >> 1] * 1.5;
}

/**
 * The noise residual W = I − denoise(I), with a locally adaptive Wiener
 * (Lee) filter over 5×5: the denoised value keeps the local mean plus the
 * part of the deviation that exceeds the noise.
 */
export function noiseResidual(p: Plane): Float32Array {
  const { w, h, data } = p;
  const s2 = noiseSigma(p) ** 2;
  const m = boxBlur(data, w, h, 2);
  const sq = new Float32Array(data.length);
  for (let i = 0; i < sq.length; i++) sq[i] = data[i] * data[i];
  const m2 = boxBlur(sq, w, h, 2);
  const out = new Float32Array(data.length);
  for (let i = 0; i < out.length; i++) {
    const v = Math.max(0, m2[i] - m[i] * m[i]);
    const keep = v > 1e-12 ? Math.max(0, v - s2) / v : 0;
    const den = m[i] + keep * (data[i] - m[i]);
    out[i] = data[i] - den;
  }
  return out;
}

/**
 * How much each pixel can say about the gain: nothing when clipped, little
 * where the scene itself is busy (texture leaks into the residual), full on
 * smooth, well-lit areas.
 */
export function prnuWeights(luma: Plane): Float32Array {
  const { w, h, data } = luma;
  const sm = boxBlur(data, w, h, 4);
  const sq = new Float32Array(data.length);
  for (let i = 0; i < sq.length; i++) sq[i] = sm[i] * sm[i];
  const m = boxBlur(sm, w, h, 7);
  const m2 = boxBlur(sq, w, h, 7);
  const out = new Float32Array(data.length);
  for (let i = 0; i < out.length; i++) {
    const lv = Math.max(0, m2[i] - m[i] * m[i]);
    const flat = Math.exp(-lv / (0.004 * 0.004));
    out[i] = data[i] < 0.97 && data[i] > 0.03 ? flat : 0;
  }
  return out;
}

// --- the running estimate --------------------------------------------------------------

export const SPLIT_WINDOW = 512;

/** The running sums for one camera at one size, plus a split-half pair on a centre window. */
export interface PrnuStore {
  key: string;
  w: number;
  h: number;
  s1: [Float32Array, Float32Array, Float32Array]; // Σ w·W·I per channel
  s2: [Float32Array, Float32Array, Float32Array]; // Σ w·I²
  photos: number;
  /** split halves on the centre SPLIT_WINDOW² (green channel): photo n goes to half n mod 2 */
  halves: { s1: [Float32Array, Float32Array]; s2: [Float32Array, Float32Array] };
}

export function newPrnuStore(key: string, w: number, h: number): PrnuStore {
  const n = w * h;
  const sw = Math.min(SPLIT_WINDOW, w) * Math.min(SPLIT_WINDOW, h);
  return {
    key,
    w,
    h,
    s1: [new Float32Array(n), new Float32Array(n), new Float32Array(n)],
    s2: [new Float32Array(n), new Float32Array(n), new Float32Array(n)],
    photos: 0,
    halves: { s1: [new Float32Array(sw), new Float32Array(sw)], s2: [new Float32Array(sw), new Float32Array(sw)] },
  };
}

function splitRect(w: number, h: number): { x0: number; y0: number; sw: number; sh: number } {
  const sw = Math.min(SPLIT_WINDOW, w);
  const sh = Math.min(SPLIT_WINDOW, h);
  return { x0: (w - sw) >> 1, y0: (h - sh) >> 1, sw, sh };
}

/**
 * Fold one photo into the store. Returns the share of the photo that was
 * usable (weight > 0.5) so the caller can say how much it taught.
 */
export function prnuAccumulate(store: PrnuStore, planes: [Plane, Plane, Plane], luma: Plane): number {
  const { w, h } = store;
  if (planes[0].w !== w || planes[0].h !== h) throw new Error('photo size does not match the signature');
  const wgt = prnuWeights(luma);
  let usable = 0;
  for (let i = 0; i < wgt.length; i++) if (wgt[i] > 0.5) usable++;
  const half = store.photos & 1;
  const r = splitRect(w, h);
  for (let c = 0; c < 3; c++) {
    const I = planes[c].data;
    const W = noiseResidual(planes[c]);
    const s1 = store.s1[c];
    const s2 = store.s2[c];
    for (let i = 0; i < I.length; i++) {
      const g = wgt[i];
      if (g === 0) continue;
      s1[i] += g * W[i] * I[i];
      s2[i] += g * I[i] * I[i];
    }
    if (c === 1) {
      const h1 = store.halves.s1[half];
      const h2 = store.halves.s2[half];
      for (let y = 0; y < r.sh; y++)
        for (let x = 0; x < r.sw; x++) {
          const i = (r.y0 + y) * w + r.x0 + x;
          const g = wgt[i];
          h1[y * r.sw + x] += g * W[i] * I[i];
          h2[y * r.sw + x] += g * I[i] * I[i];
        }
    }
  }
  store.photos++;
  return usable / wgt.length;
}

// --- finishing the estimate ---------------------------------------------------------------

function zeroMean(k: Float32Array, w: number, h: number): void {
  const col = new Float64Array(w);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) col[x] += k[y * w + x];
  for (let x = 0; x < w; x++) col[x] /= h;
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      k[y * w + x] -= col[x];
      row += k[y * w + x];
    }
    row /= w;
    for (let x = 0; x < w; x++) k[y * w + x] -= row;
  }
}

const WN = 512;

/**
 * Wiener filter in the frequency domain, tile by tile: bins whose
 * neighbourhood carries more power than the flat baseline are spikes (a
 * grid, a lattice, a readout stripe) and are pulled down to it. The sensor's
 * own pattern is white and passes unchanged.
 */
function wienerDft(k: Float32Array, w: number, h: number): Float32Array {
  const n = Math.min(WN, 1 << Math.floor(Math.log2(Math.min(w, h))));
  const hop = n >> 1;
  // reflect-padded by half a window so every pixel of the map sits under
  // full window coverage (at the border alone, dividing by a near-zero
  // window weight would blow the filtered tile up)
  const pad = hop;
  const pw = w + 2 * pad;
  const ph = h + 2 * pad;
  const refl = (i: number, m: number): number => {
    const period = 2 * m - 2;
    if (m === 1) return 0;
    i = ((i % period) + period) % period;
    return i < m ? i : period - i;
  };
  const kp = new Float32Array(pw * ph);
  for (let y = 0; y < ph; y++) {
    const sy = refl(y - pad, h) * w;
    for (let x = 0; x < pw; x++) kp[y * pw + x] = k[sy + refl(x - pad, w)];
  }
  const win = window2d('hann', n, n);
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  const P = new Float64Array(n * n);
  const acc = new Float32Array(pw * ph);
  const norm = new Float32Array(pw * ph);
  const lv = new Float64Array(n * n);
  const tmp = new Float64Array(n * n);
  for (let y0 = 0; y0 + n <= ph + hop - 1; y0 += hop) {
    const ty = Math.min(y0, ph - n);
    for (let x0 = 0; x0 + n <= pw + hop - 1; x0 += hop) {
      const tx = Math.min(x0, pw - n);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) re[y * n + x] = kp[(ty + y) * pw + tx + x] * win.data[y * n + x];
      im.fill(0);
      fft2d(re, im, n, n, false);
      let base = 0;
      for (let i = 0; i < P.length; i++) {
        P[i] = re[i] * re[i] + im[i] * im[i];
        base += P[i];
      }
      base /= P.length;
      // local power: the smallest of the 3×3 … 9×9 neighbourhood means (wrapping)
      lv.fill(Infinity);
      for (const r of [1, 2, 3, 4]) {
        const m = 2 * r + 1;
        // separable wrap-around box mean
        for (let y = 0; y < n; y++)
          for (let x = 0; x < n; x++) {
            let s = 0;
            for (let d = -r; d <= r; d++) s += P[y * n + ((x + d + n) % n)];
            tmp[y * n + x] = s / m;
          }
        for (let y = 0; y < n; y++)
          for (let x = 0; x < n; x++) {
            let s = 0;
            for (let d = -r; d <= r; d++) s += tmp[((y + d + n) % n) * n + x];
            const v = s / m;
            if (v < lv[y * n + x]) lv[y * n + x] = v;
          }
      }
      for (let i = 0; i < P.length; i++) {
        const g = base / Math.max(base, lv[i]);
        re[i] *= g;
        im[i] *= g;
      }
      fft2d(re, im, n, n, true);
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
          const i = (ty + y) * pw + tx + x;
          acc[i] += win.data[y * n + x] * re[y * n + x];
          norm[i] += win.data[y * n + x] * win.data[y * n + x];
        }
      if (tx === pw - n) break;
    }
    if (ty === ph - n) break;
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = (y + pad) * pw + x + pad;
      out[y * w + x] = norm[i] > 1e-3 ? acc[i] / norm[i] : 0;
    }
  return out;
}

export interface PrnuMap {
  key: string;
  w: number;
  h: number;
  k: [Float32Array, Float32Array, Float32Array];
  photos: number;
  /** correlation between the two halves of the batch on the centre window */
  splitHalf: number;
  /** reliability of the whole map (Spearman–Brown), 0..1 */
  reliability: number;
  /** rms of the pattern, as a fraction of the signal */
  rms: number;
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let ma = 0;
  let mb = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma;
    const y = b[i] - mb;
    sab += x * y;
    saa += x * x;
    sbb += y * y;
  }
  return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : 0;
}

/**
 * How reliable the estimate is, from the two halves of the batch on the
 * centre window: their correlation, and the Spearman–Brown reliability of
 * the whole (both halves together). Cheap: the window is small.
 */
export function prnuReliability(store: PrnuStore): { splitHalf: number; reliability: number } {
  const r = splitRect(store.w, store.h);
  const halves = [0, 1].map((hh) => {
    const out = new Float32Array(r.sw * r.sh);
    for (let i = 0; i < out.length; i++) out[i] = store.halves.s2[hh][i] > 1e-4 ? store.halves.s1[hh][i] / store.halves.s2[hh][i] : 0;
    zeroMean(out, r.sw, r.sh);
    return wienerDft(out, r.sw, r.sh);
  });
  const splitHalf = store.photos >= 4 ? Math.max(0, correlation(halves[0], halves[1])) : 0;
  return { splitHalf, reliability: (2 * splitHalf) / (1 + splitHalf) };
}

/** The finished map: the sums divided, cleaned, and their reliability measured. */
export function prnuFinalize(store: PrnuStore, progress: (fraction: number) => void = () => {}): PrnuMap {
  const { w, h } = store;
  const k = [0, 1, 2].map((c) => {
    const out = new Float32Array(w * h);
    for (let i = 0; i < out.length; i++) out[i] = store.s2[c][i] > 1e-4 ? store.s1[c][i] / store.s2[c][i] : 0;
    zeroMean(out, w, h);
    const cleaned = wienerDft(out, w, h);
    progress((c + 1) / 3);
    return cleaned;
  }) as [Float32Array, Float32Array, Float32Array];
  const { splitHalf, reliability } = prnuReliability(store);
  let s = 0;
  for (const v of k[1]) s += v * v;
  return { key: store.key, w, h, k, photos: store.photos, splitHalf, reliability, rms: Math.sqrt(s / k[1].length) };
}

// --- measuring and applying ---------------------------------------------------------------

/**
 * How strongly a photo (or a crop of one at `offset`) carries the signature:
 * the correlation between its residual and the pattern it should carry
 * (K·I), averaged over the channels, and the z-score of that correlation
 * (above ~5 it is unmistakable; 0 means none).
 */
export function prnuMeasure(planes: [Plane, Plane, Plane], map: PrnuMap, offset = { x0: 0, y0: 0 }): { rho: number; z: number } {
  const { w, h } = planes[0];
  let sum = 0;
  let n = 0;
  for (let c = 0; c < 3; c++) {
    const W = noiseResidual(planes[c]);
    const I = planes[c].data;
    const exp = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const kx = offset.x0 + x;
        const ky = offset.y0 + y;
        exp[i] = kx >= 0 && ky >= 0 && kx < map.w && ky < map.h ? map.k[c][ky * map.w + kx] * I[i] : 0;
      }
    sum += correlation(W, exp);
    n++;
  }
  const rho = sum / n;
  return { rho, z: rho * Math.sqrt(w * h) };
}

/** The planes with the signature multiplied in at `gain` (1 = the map as estimated). */
export function prnuApply(planes: [Plane, Plane, Plane], map: PrnuMap, gain: number, offset = { x0: 0, y0: 0 }): [Plane, Plane, Plane] {
  const { w, h } = planes[0];
  return planes.map((p, c) => {
    const out = makePlane(w, h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const kx = offset.x0 + x;
        const ky = offset.y0 + y;
        const k = kx >= 0 && ky >= 0 && kx < map.w && ky < map.h ? map.k[c][ky * map.w + kx] : 0;
        const v = p.data[i] * (1 + gain * k);
        out.data[i] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    return out;
  }) as [Plane, Plane, Plane];
}

/**
 * The gain that brings the result's signature back to a target correlation
 * (normally what the untouched photo carried), found by a few secant steps.
 * Scaled by the map's reliability first, since a noisier map needs less
 * weight to do the same good and more to do harm.
 */
export function prnuCalibrate(planes: [Plane, Plane, Plane], map: PrnuMap, targetRho: number, offset = { x0: 0, y0: 0 }): { gain: number; rho: number } {
  if (targetRho <= 0 || map.reliability <= 0) return { gain: 0, rho: prnuMeasure(planes, map, offset).rho };
  const base = prnuMeasure(planes, map, offset).rho;
  if (base >= targetRho) return { gain: 0, rho: base };
  // the correlation grows roughly linearly with the gain while the pattern is weak
  let g = map.reliability * 1.0;
  let r = prnuMeasure(prnuApply(planes, map, g, offset), map, offset).rho;
  let g0 = 0;
  let r0 = base;
  for (let it = 0; it < 4 && Math.abs(r - targetRho) > 0.02 * targetRho; it++) {
    const slope = (r - r0) / Math.max(1e-6, g - g0);
    if (slope <= 0) break;
    const next = Math.max(0, Math.min(map.reliability * 6, g + (targetRho - r) / slope));
    g0 = g;
    r0 = r;
    g = next;
    r = prnuMeasure(prnuApply(planes, map, g, offset), map, offset).rho;
  }
  return { gain: g, rho: r };
}

// --- storing ------------------------------------------------------------------------------

/** The store packed for IndexedDB or a file: sums quantised to 16 bits per channel. */
export function packPrnuStore(store: PrnuStore): { header: Record<string, unknown>; buffers: ArrayBuffer[] } {
  const buffers: ArrayBuffer[] = [];
  const scales: number[] = [];
  const quant = (a: Float32Array, signed: boolean): void => {
    let mx = 0;
    for (const v of a) if (Math.abs(v) > mx) mx = Math.abs(v);
    const scale = mx > 0 ? mx / (signed ? 32767 : 65535) : 1;
    scales.push(scale);
    const q = signed ? new Int16Array(a.length) : new Uint16Array(a.length);
    for (let i = 0; i < a.length; i++) q[i] = Math.round(a[i] / scale);
    buffers.push(q.buffer);
  };
  for (let c = 0; c < 3; c++) {
    quant(store.s1[c], true);
    quant(store.s2[c], false);
  }
  for (let hh = 0; hh < 2; hh++) {
    quant(store.halves.s1[hh], true);
    quant(store.halves.s2[hh], false);
  }
  return { header: { version: 1, key: store.key, w: store.w, h: store.h, photos: store.photos, scales }, buffers };
}

export function unpackPrnuStore(header: Record<string, unknown>, buffers: ArrayBuffer[]): PrnuStore {
  const store = newPrnuStore(String(header.key), Number(header.w), Number(header.h));
  store.photos = Number(header.photos);
  const scales = header.scales as number[];
  let b = 0;
  const deq = (dst: Float32Array, signed: boolean): void => {
    const scale = scales[b];
    const q = signed ? new Int16Array(buffers[b]) : new Uint16Array(buffers[b]);
    for (let i = 0; i < dst.length; i++) dst[i] = q[i] * scale;
    b++;
  };
  for (let c = 0; c < 3; c++) {
    deq(store.s1[c], true);
    deq(store.s2[c], false);
  }
  for (let hh = 0; hh < 2; hh++) {
    deq(store.halves.s1[hh], true);
    deq(store.halves.s2[hh], false);
  }
  return store;
}

/** The key a photo's signature is filed under: camera make and model, and the frame size it was taken at. */
export function prnuKey(make: string | undefined, model: string | undefined, w: number, h: number): string | null {
  const cam = `${make ?? ''} ${model ?? ''}`.trim();
  if (!cam) return null;
  return `${cam}|${w}x${h}`;
}
