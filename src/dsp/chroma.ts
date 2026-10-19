/**
 * Colour-layer noise ("blotches") and the halo of noise that hugs hard edges
 * after heavy compression ("mosquito noise"). Both are judged automatically
 * and both return delta planes like every other correction.
 */

import { baseGrain, boxBlur, guidedFilter, noiseMap, noiseModel, upsampleBlockMap, type NoiseMap } from './noise';
import { downsamplePlane, makePlane, type Plane } from './plane';

/** RGB (0..1) -> Y, Cb, Cr planes (BT.601, Cb/Cr centred on 0). */
export function toYCbCr(r: Plane, g: Plane, b: Plane): [Plane, Plane, Plane] {
  const n = r.w * r.h;
  const Y = makePlane(r.w, r.h);
  const Cb = makePlane(r.w, r.h);
  const Cr = makePlane(r.w, r.h);
  for (let i = 0; i < n; i++) {
    const R = r.data[i];
    const G = g.data[i];
    const B = b.data[i];
    const y = 0.299 * R + 0.587 * G + 0.114 * B;
    Y.data[i] = y;
    Cb.data[i] = (B - y) * 0.564;
    Cr.data[i] = (R - y) * 0.713;
  }
  return [Y, Cb, Cr];
}

/** Y, Cb, Cr -> RGB planes. */
export function toRGB(Y: Plane, Cb: Plane, Cr: Plane): [Plane, Plane, Plane] {
  const n = Y.w * Y.h;
  const r = makePlane(Y.w, Y.h);
  const g = makePlane(Y.w, Y.h);
  const b = makePlane(Y.w, Y.h);
  for (let i = 0; i < n; i++) {
    const y = Y.data[i];
    const cb = Cb.data[i];
    const cr = Cr.data[i];
    r.data[i] = y + 1.402 * cr;
    g.data[i] = y - 0.344136 * cb - 0.714136 * cr;
    b.data[i] = y + 1.772 * cb;
  }
  return [r, g, b];
}

export interface ChromaNoise {
  /** colour noise in the blocks over the threshold, plane units (×255 for levels of Cb/Cr) */
  sigmaCb: number;
  sigmaCr: number;
  /** fraction of blocks where colour noise exceeds the threshold */
  fraction: number;
  mapCb: NoiseMap;
  mapCr: NoiseMap;
  /** the colour planes with everything brightness does not explain smoothed away */
  smoothCb: Plane;
  smoothCr: Plane;
}

/** Luma-guided smoothing of a colour plane: colour follows brightness where brightness varies, and is averaged where it does not. */
const CHROMA_RADIUS = 16;
const CHROMA_EPS = 0.01 ** 2; // windows whose brightness varies by less than ~2.5 levels count as flat

/**
 * Colour noise is measured on what is left of a colour plane after
 * luma-guided smoothing: colour that brightness does not account for. A red
 * car on a blue sky is explained by its own brightness edges; blotches are
 * not. The residual is measured at two scales, fine (full resolution) and
 * coarse (4× down), and the larger wins per block: blotches are soft blobs
 * several pixels wide that a pixel-scale measure barely sees.
 */
function chromaMap(resid: Plane, block: number): NoiseMap {
  const fine = noiseMap(resid, block);
  if (resid.w < 4 * block || resid.h < 4 * block) return fine;
  const coarse = noiseMap(downsamplePlane(resid, 4), block / 4);
  const out: NoiseMap = { ...fine, sigma: new Float32Array(fine.sigma), texture: new Float32Array(fine.texture) };
  for (let by = 0; by < fine.bh; by++) {
    for (let bx = 0; bx < fine.bw; bx++) {
      const k = by * fine.bw + bx;
      const cx = Math.min(coarse.bw - 1, bx);
      const cy = Math.min(coarse.bh - 1, by);
      const kc = cy * coarse.bw + cx;
      if (coarse.sigma[kc] > out.sigma[k]) out.sigma[k] = coarse.sigma[kc];
      // structure in colour at either scale is a coloured edge or pattern, not noise
      out.texture[k] = Math.max(fine.texture[k], coarse.texture[kc]);
    }
  }
  // the typical level of the merged map, from blocks without colour structure
  const kept: number[] = [];
  for (let k = 0; k < out.sigma.length; k++) if (out.texture[k] < 0.2 && out.sigma[k] >= 0.3 / 255) kept.push(out.sigma[k]);
  kept.sort((a, b) => a - b);
  const base = baseGrain(kept);
  const all = Array.from(out.sigma).sort((a, b) => a - b);
  out.refSigma = base ? base.ref : all[all.length >> 1];
  out.ref = new Float32Array(fine.ref.length).fill(out.refSigma);
  return out;
}

/**
 * Colour noise where it is: sigmaCb/sigmaCr are the medians over the blocks
 * above the threshold (the blocks the fix will touch), or the maps' base
 * level when no block is over it. Blocks with colour structure of their own
 * are never counted as over the threshold.
 */
export function measureChromaNoise(Y: Plane, Cb: Plane, Cr: Plane, block = 32, thresholdLevels = 1.0): ChromaNoise {
  const smoothCb = guidedFilterWithGuide(Cb, Y, CHROMA_RADIUS, CHROMA_EPS);
  const smoothCr = guidedFilterWithGuide(Cr, Y, CHROMA_RADIUS, CHROMA_EPS);
  const residOf = (C: Plane, f: Plane): Plane => ({ w: C.w, h: C.h, data: C.data.map((v, i) => v - f.data[i]) });
  const mapCb = chromaMap(residOf(Cb, smoothCb), block);
  const mapCr = chromaMap(residOf(Cr, smoothCr), block);
  const overCb: number[] = [];
  const overCr: number[] = [];
  for (let i = 0; i < mapCb.sigma.length; i++) {
    if (Math.max(mapCb.texture[i], mapCr.texture[i]) >= 0.5) continue;
    if (Math.max(mapCb.sigma[i], mapCr.sigma[i]) * 255 > thresholdLevels) {
      overCb.push(mapCb.sigma[i]);
      overCr.push(mapCr.sigma[i]);
    }
  }
  const med = (l: number[], fallback: number): number => {
    if (!l.length) return fallback;
    l.sort((a, b) => a - b);
    return l[l.length >> 1];
  };
  const sCb = med(overCb, mapCb.refSigma);
  const sCr = med(overCr, mapCr.refSigma);
  return { sigmaCb: sCb, sigmaCr: sCr, fraction: overCb.length / mapCb.sigma.length, mapCb, mapCr, smoothCb, smoothCr };
}

/**
 * Remove colour noise where it is: the delta toward the luma-guided smooth
 * colour (the same smoothing the measurement used, so what was measured is
 * what goes). `amount` is the fraction removed where colour noise is at or
 * above the threshold, fading in from half of it; blocks with colour
 * structure of their own are left alone.
 */
export function chromaDelta(Cb: Plane, Cr: Plane, noise: ChromaNoise, amount: number, thresholdLevels = 1.0): { dCb: Plane; dCr: Plane } {
  const { w, h } = Cb;
  const out: Plane[] = [];
  for (const [C, f, map] of [
    [Cb, noise.smoothCb, noise.mapCb],
    [Cr, noise.smoothCr, noise.mapCr],
  ] as [Plane, Plane, NoiseMap][]) {
    const blendBlocks = new Float32Array(map.sigma.length);
    let any = false;
    for (let i = 0; i < map.sigma.length; i++) {
      const lv = Math.max(noise.mapCb.sigma[i], noise.mapCr.sigma[i]) * 255;
      const b = Math.min(1, Math.max(0, (lv - thresholdLevels * 0.5) / (thresholdLevels * 0.5)));
      const tex = Math.max(noise.mapCb.texture[i], noise.mapCr.texture[i]);
      blendBlocks[i] = b * amount * (1 - tex) ** 2;
      if (blendBlocks[i] > 0) any = true;
    }
    const delta = makePlane(w, h);
    if (any) {
      const blend = upsampleBlockMap(blendBlocks, map.bw, map.bh, map.block, w, h);
      for (let i = 0; i < delta.data.length; i++) delta.data[i] = (f.data[i] - C.data[i]) * blend.data[i];
    }
    out.push(delta);
  }
  return { dCb: out[0], dCr: out[1] };
}

/** Guided filter of `p` with a separate guide `I` (He, Sun, Tang). */
export function guidedFilterWithGuide(p: Plane, I: Plane, r: number, eps: number): Plane {
  const { w, h } = p;
  const n = w * h;
  const meanI = boxBlur(I.data, w, h, r);
  const meanP = boxBlur(p.data, w, h, r);
  const IP = new Float32Array(n);
  const II = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    IP[i] = I.data[i] * p.data[i];
    II[i] = I.data[i] * I.data[i];
  }
  const meanIP = boxBlur(IP, w, h, r);
  const meanII = boxBlur(II, w, h, r);
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const varI = meanII[i] - meanI[i] * meanI[i];
    const covIP = meanIP[i] - meanI[i] * meanP[i];
    a[i] = covIP / (varI + eps);
    b[i] = meanP[i] - a[i] * meanI[i];
  }
  const meanA = boxBlur(a, w, h, r);
  const meanB = boxBlur(b, w, h, r);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = meanA[i] * I.data[i] + meanB[i];
  return { w, h, data: out };
}

// ---------------------------------------------------------------------------
// Edge halos (mosquito noise)
// ---------------------------------------------------------------------------

export interface RingingMeasure {
  /** rms of the fine residual in the band next to strong edges (clear of the edge itself), levels */
  haloLevels: number;
  /** the same residual in the photo's plain blocks (its ordinary grain), levels */
  grainLevels: number;
  /** haloLevels / grainLevels; > 1.5 means edges carry extra noise */
  ratio: number;
  /** fraction of pixels in the halo band */
  bandFraction: number;
  edgeThreshold: number;
  /** the smoother's strength, plane units; what ringingDelta should use */
  haloSigma: number;
}

/**
 * Strong-edge mask (gradient above `edgeT`) dilated by `band` pixels, minus
 * the edge itself and a 2-px rim: the band the halo fix works on. `measure`
 * is the same band minus a 4-px rim, where a 5×5 residual cannot touch the
 * edge's own transition.
 */
export function haloBand(Y: Plane, edgeT: number, band: number): { halo: Uint8Array; edge: Uint8Array; measure: Uint8Array } {
  const { w, h } = Y;
  const d = Y.data;
  const edge = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = d[i + 1] - d[i - 1];
      const gy = d[i + w] - d[i - w];
      if (Math.hypot(gx, gy) * 0.5 > edgeT) edge[i] = 1;
    }
  }
  const dilate = (src: Uint8Array, r: number): Uint8Array => {
    const tmp = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let m = 0;
        for (let k = -r; k <= r && !m; k++) {
          const xx = x + k;
          if (xx >= 0 && xx < w && src[y * w + xx]) m = 1;
        }
        tmp[y * w + x] = m;
      }
    }
    const out = new Uint8Array(w * h);
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let m = 0;
        for (let k = -r; k <= r && !m; k++) {
          const yy = y + k;
          if (yy >= 0 && yy < h && tmp[yy * w + x]) m = 1;
        }
        out[y * w + x] = m;
      }
    }
    return out;
  };
  const wide = dilate(edge, band);
  const rim2 = dilate(edge, 2);
  const rim4 = dilate(edge, 4);
  const halo = new Uint8Array(w * h);
  const measure = new Uint8Array(w * h);
  for (let i = 0; i < halo.length; i++) {
    halo[i] = wide[i] && !rim2[i] ? 1 : 0;
    measure[i] = wide[i] && !rim4[i] ? 1 : 0;
  }
  return { halo, edge, measure };
}

/** The halo smoother: an edge-preserving filter whose strength never reaches a strong edge's own contrast. */
function haloFilter(Y: Plane, haloSigma: number): Plane {
  return guidedFilter(Y, 2, (4 * Math.min(0.02, Math.max(haloSigma, 0.004))) ** 2);
}

/**
 * Halo next to strong edges: the rms of the fine (5×5) residual in the band
 * beside strong edges, far enough out that the edge's own transition is not
 * in it, against the same residual in the photo's plain blocks (its grain).
 */
export function measureRinging(Y: Plane, edgeT = 0.12, band = 8, grainSigma?: number): RingingMeasure {
  const { w, h } = Y;
  const d = Y.data;
  const { halo, edge, measure } = haloBand(Y, edgeT, band);
  const box5 = boxBlur(d, w, h, 2);
  let sh = 0;
  let nh = 0;
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const i = y * w + x;
      if (!measure[i]) continue;
      const r = d[i] - box5[i];
      sh += r * r;
      nh++;
    }
  }
  const haloLevels = nh ? Math.sqrt(sh / nh) * 255 : 0;
  // the same residual in plain blocks at the photo's base grain, away from any band
  const map = noiseMap(Y, 32);
  let sg = 0;
  let ng = 0;
  for (let by = 0; by < map.bh; by++) {
    for (let bx = 0; bx < map.bw; bx++) {
      const k = by * map.bw + bx;
      const r0 = map.ref[Math.min(15, Math.max(0, Math.floor(map.mean[k] * 16)))];
      if (map.texture[k] >= 0.3 || map.sigma[k] > 1.35 * r0 || map.sigma[k] < r0 / 1.35) continue;
      for (let y = Math.max(2, by * 32); y < Math.min(h - 2, by * 32 + 32); y++) {
        for (let x = Math.max(2, bx * 32); x < Math.min(w - 2, bx * 32 + 32); x++) {
          const i = y * w + x;
          if (halo[i] || edge[i]) continue;
          const r = d[i] - box5[i];
          sg += r * r;
          ng++;
        }
      }
    }
  }
  const grainLevels = ng ? Math.sqrt(sg / ng) * 255 : (grainSigma ?? map.refSigma) * 255;
  const haloSigma = Math.min(0.02, Math.max(haloLevels / 255, 0.004));
  let nb = 0;
  for (let i = 0; i < halo.length; i++) nb += halo[i];
  return { haloLevels, grainLevels, ratio: grainLevels > 1e-6 ? haloLevels / grainLevels : 1, bandFraction: nb / (w * h), edgeThreshold: edgeT, haloSigma };
}

/**
 * Smooth the halo band only, with the edge-preserving halo filter, by
 * `amount`. Returns the delta for the plane given (apply to each RGB channel
 * using the same band computed from luminance).
 */
export function ringingDelta(ch: Plane, halo: Uint8Array, haloSigma: number, amount: number): Plane {
  const f = haloFilter(ch, haloSigma);
  const delta = makePlane(ch.w, ch.h);
  // feather the band by one box blur so the fix fades in
  const soft = boxBlur(Float32Array.from(halo), ch.w, ch.h, 2);
  for (let i = 0; i < delta.data.length; i++) delta.data[i] = (f.data[i] - ch.data[i]) * soft[i] * amount;
  return delta;
}
