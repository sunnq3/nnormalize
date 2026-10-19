/**
 * Noise analysis and normalisation.
 *
 * Grain is measured as the robust (MAD) scale of the residual after a 5×5
 * box blur, per block. That residual holds everything finer than about four
 * pixels, which is where the grain of a JPEG lives: the encoder zeroes the
 * very finest frequencies, so a pixel-scale (Laplacian) estimate reads a
 * compressed photo as almost noise-free while its grain is plainly visible.
 * For white noise of sigma s the residual's sigma is s·sqrt(24/25).
 */

import { makePlane, type Plane } from './plane';
import { rng, gaussian } from '../synth/generate';

export interface NoiseMap {
  block: number; // block size in pixels
  bw: number; // blocks across
  bh: number; // blocks down
  sigma: Float32Array; // per-block noise sigma, plane units
  mean: Float32Array; // per-block mean intensity
  /**
   * How much of the block's fine detail is organised rather than random, 0..1:
   * edges and oriented texture (aligned gradients), fine texture far above the
   * photo's grain, and fine detail whose spatial correlation differs from the
   * grain's. The smoother is held back in proportion to this.
   */
  texture: Float32Array;
  /** the photo's base grain at each of 16 brightness bins (plane units) */
  ref: Float32Array;
  /** the photo's base grain overall (plane units) */
  refSigma: number;
}

export interface GrainCluster {
  ref: number; // median sigma of the cluster
  lo: number;
  hi: number;
  share: number; // fraction of the blocks given
}

/**
 * Distinct clusters of block sigmas (windows of ×1.35), largest first by
 * greedy extraction, returned sorted by sigma. Sorted input.
 */
export function grainClusters(sorted: number[], minShare = 0.12): GrainCluster[] {
  const n = sorted.length;
  const out: GrainCluster[] = [];
  if (n < 4) return out;
  const span = 1.35;
  const taken = new Uint8Array(n);
  for (;;) {
    let bestI = -1;
    let bestC = 0;
    let j = 0;
    for (let i = 0; i < n; i++) {
      if (taken[i]) continue;
      if (j < i) j = i;
      while (j < n && sorted[j] <= sorted[i] * span) j++;
      let c = 0;
      for (let t = i; t < j; t++) if (!taken[t]) c++;
      if (c > bestC) {
        bestC = c;
        bestI = i;
      }
    }
    if (bestI < 0 || bestC < Math.max(4, minShare * n)) break;
    const members: number[] = [];
    for (let t = bestI; t < n && sorted[t] <= sorted[bestI] * span; t++) {
      if (!taken[t]) {
        taken[t] = 1;
        members.push(sorted[t]);
      }
    }
    out.push({ ref: members[members.length >> 1], lo: members[0], hi: members[members.length - 1], share: members.length / n });
  }
  return out.sort((a, b) => a.ref - b.ref);
}

/**
 * The base grain of a set of block sigmas: the lowest cluster that holds a
 * substantial share (40 % of the largest). Blocks that were smoothed earlier
 * form a cluster near zero (excluded by the caller's floor); fine texture
 * forms clusters above the grain and is ignored unless it is nearly all
 * there is.
 */
export function baseGrain(sorted: number[]): GrainCluster | null {
  const cl = grainClusters(sorted);
  if (!cl.length) return null;
  let max = 0;
  for (const c of cl) max = Math.max(max, c.share);
  for (const c of cl) if (c.share >= 0.4 * max) return c;
  return cl[0];
}

/**
 * The base grain within one brightness bin given the photo's overall base
 * grain B: of the clusters within ×3 of B, the one that is both close to B
 * and well populated (cost = 3·|ln(ref/B)| − ln share). A bin with none
 * there holds only texture or smoothed blocks at that brightness, and gets
 * its grain from the neighbouring bins instead (null).
 */
export function binGrain(sorted: number[], B: number): GrainCluster | null {
  const cl = grainClusters(sorted);
  if (!cl.length) return null;
  let best: GrainCluster | null = null;
  let bestCost = Infinity;
  for (const c of cl) {
    const r = c.ref / B;
    if (r < 1 / 3 || r > 3) continue;
    const cost = 3 * Math.abs(Math.log(r)) - Math.log(c.share);
    if (cost < bestCost) {
      bestCost = cost;
      best = c;
    }
  }
  return best;
}

export function noiseMap(p: Plane, block = 32): NoiseMap {
  const { w, h } = p;
  const bw = Math.max(1, Math.floor(w / block));
  const bh = Math.max(1, Math.floor(h / block));
  const nb = bw * bh;
  const sigma = new Float32Array(nb);
  const gMed = new Float32Array(nb); // median gradient magnitude of the lightly smoothed block
  const mean = new Float32Array(nb);
  const texture = new Float32Array(nb);
  const structure = new Float32Array(nb);
  const corr = [new Float32Array(nb), new Float32Array(nb), new Float32Array(nb), new Float32Array(nb)]; // lag-1 x, lag-1 y, lag-2 x, lag-2 y of the fine residual
  const d = p.data;
  const buf = new Float64Array(block * block);
  const buf2 = new Float64Array(block * block);

  // lightly smoothed copy for the structure test: a 3x3 box cuts noise
  // gradients while edges keep most of theirs
  const sm = boxBlur(d, w, h, 1);
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const gmag = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = 0.5 * (sm[i + 1] - sm[i - 1]);
      gy[i] = 0.5 * (sm[i + w] - sm[i - w]);
      gmag[i] = Math.hypot(gx[i], gy[i]);
    }
  }

  const box5 = boxBlur(d, w, h, 2);
  const resid = new Float32Array(w * h);
  for (let i = 0; i < resid.length; i++) resid[i] = d[i] - box5[i];
  const residScale = 1.4826 / Math.sqrt(24 / 25);
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const x0 = bx * block;
      const y0 = by * block;
      let n = 0;
      let m = 0;
      let s2 = 0;
      const sl = [0, 0, 0, 0];
      const y1 = Math.min(h - 2, y0 + block);
      const x1 = Math.min(w - 2, x0 + block);
      for (let y = Math.max(2, y0); y < y1; y++) {
        for (let x = Math.max(2, x0); x < x1; x++) {
          const i = y * w + x;
          const r = resid[i];
          buf[n] = Math.abs(r);
          buf2[n] = gmag[i];
          n++;
          m += d[i];
          s2 += r * r;
          if (x + 1 < x1) sl[0] += r * resid[i + 1];
          if (y + 1 < y1) sl[1] += r * resid[i + w];
          if (x + 2 < x1) sl[2] += r * resid[i + 2];
          if (y + 2 < y1) sl[3] += r * resid[i + 2 * w];
        }
      }
      const k = by * bw + bx;
      const seg = buf.subarray(0, n);
      seg.sort();
      const med = n ? (n & 1 ? seg[n >> 1] : 0.5 * (seg[(n >> 1) - 1] + seg[n >> 1])) : 0;
      sigma[k] = residScale * med;
      const seg2 = buf2.subarray(0, n);
      seg2.sort();
      gMed[k] = n ? seg2[n >> 1] : 0;
      mean[k] = n ? m / n : 0;
      for (let l = 0; l < 4; l++) corr[l][k] = s2 > 1e-12 ? sl[l] / s2 : 0;
    }
  }

  const bins = 16;
  const binOf = (k: number): number => Math.min(bins - 1, Math.max(0, Math.floor(mean[k] * bins)));
  const all = Array.from(sigma).sort((a, c) => a - c);
  const medianAll = all.length ? all[all.length >> 1] : 0;
  const sorted = (l: number[]): number[] => l.sort((a, c) => a - c);

  // First pass at the base grain from the sigmas alone, to learn how steep
  // this photo's grain is: the ratio of a block's median gradient to its
  // sigma, which depends on the grain's spectrum (compression reddens it).
  const base0 = lowestSubstantial(grainClusters(sorted(Array.from(sigma))), 0.5 / 255) ?? { ref: medianAll, lo: medianAll / 1.35, hi: medianAll * 1.35, share: 1 };
  const ratios: number[] = [];
  for (let k = 0; k < nb; k++) if (sigma[k] > 1e-6 && sigma[k] >= base0.lo && sigma[k] <= base0.hi) ratios.push(gMed[k] / sigma[k]);
  ratios.sort((a, c) => a - c);
  const gradPerSigma = ratios.length ? ratios[ratios.length >> 1] : 0.3;

  // structure: pixels whose smoothed gradient is far above what noise of this
  // block's own strength produces (at least the photo's grain, so a block
  // smoothed earlier does not count its gentle gradients), and how aligned
  // those pixels are. Noise alone leaves ~0.02 % of pixels above 3.5× the
  // median gradient; an edge or a weave leaves many more, and aligned.
  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const k = by * bw + bx;
      const gT = 3.5 * gradPerSigma * Math.max(sigma[k], base0.ref);
      let edges = 0;
      let cnt = 0;
      let jxx = 0;
      let jyy = 0;
      let jxy = 0;
      const y1 = Math.min(h - 1, by * block + block);
      const x1 = Math.min(w - 1, bx * block + block);
      for (let y = Math.max(1, by * block); y < y1; y++) {
        for (let x = Math.max(1, bx * block); x < x1; x++) {
          const i = y * w + x;
          cnt++;
          if (gmag[i] > gT) {
            edges++;
            jxx += gx[i] * gx[i];
            jyy += gy[i] * gy[i];
            jxy += gx[i] * gy[i];
          }
        }
      }
      const density = cnt ? edges / cnt : 0;
      const tr = jxx + jyy;
      const coh = tr > 1e-12 && density > 0.02 ? ((jxx - jyy) ** 2 + 4 * jxy * jxy) / (tr * tr) : 0;
      structure[k] = Math.min(1, Math.max(coh * 2.5 * Math.min(1, density / 0.05), (density - 0.03) / 0.12, 0));
    }
  }

  // A reference curve (grain per brightness bin) from a set of blocks, each
  // bin's cluster chosen near the pooled base B, gaps filled from the nearest
  // bin and the curve smoothed with a 3-bin median.
  const curveFrom = (use: (k: number) => boolean, B: number): Float32Array => {
    const lists: number[][] = Array.from({ length: bins }, () => []);
    for (let k = 0; k < nb; k++) if (use(k)) lists[binOf(k)].push(sigma[k]);
    const ref = new Float32Array(bins);
    const have = new Uint8Array(bins);
    for (let b = 0; b < bins; b++) {
      const c = lists[b].length >= 6 ? binGrain(sorted(lists[b]), B) : null;
      if (c) {
        ref[b] = c.ref;
        have[b] = 1;
      }
    }
    for (let b = 0; b < bins; b++) {
      if (have[b]) continue;
      let best = -1;
      let bestD = Infinity;
      for (let c = 0; c < bins; c++) {
        if (have[c] && Math.abs(c - b) < bestD) {
          bestD = Math.abs(c - b);
          best = c;
        }
      }
      ref[b] = best >= 0 ? ref[best] : B;
    }
    const sm = new Float32Array(ref);
    for (let b = 1; b < bins - 1; b++) {
      const a = ref[b - 1];
      const c = ref[b];
      const e = ref[b + 1];
      sm[b] = Math.max(Math.min(a, c), Math.min(Math.max(a, c), e));
    }
    return sm;
  };

  // Pass one: the base grain is the lowest substantial cluster of the
  // structure-free blocks (at least 0.5 levels, so a border or a flattened
  // sky is not it), which is the grain even when most of the photo is fine
  // texture. What the grain looks like (its spatial correlation) is read
  // from the lowest substantial cluster of at least 1 level, where there is
  // grain enough to have a shape.
  const plain = (k: number): boolean => structure[k] < 0.2;
  const sf: number[] = [];
  for (let k = 0; k < nb; k++) if (plain(k)) sf.push(sigma[k]);
  const clusters1 = grainClusters(sorted(sf));
  const fallback = { ref: medianAll, lo: medianAll / 1.35, hi: medianAll * 1.35, share: 1 };
  const B1 = lowestSubstantial(clusters1, 0.5 / 255) ?? largest(clusters1) ?? fallback;
  const cRef = lowestSubstantial(clusters1, 1.0 / 255) ?? B1;
  const ref1 = curveFrom(plain, B1.ref);
  const cl: number[][] = [[], [], [], []];
  for (let k = 0; k < nb; k++) {
    if (!plain(k) || sigma[k] < cRef.lo || sigma[k] > cRef.hi) continue;
    for (let l = 0; l < 4; l++) cl[l].push(corr[l][k]);
  }
  const c0 = cl.map((l) => {
    sorted(l);
    return l.length ? l[l.length >> 1] : 0;
  });

  for (let k = 0; k < nb; k++) {
    const r0 = Math.max(1e-5, ref1[binOf(k)]);
    const ratio = sigma[k] / r0;
    // Sensor noise is consistent at a given brightness within one photo, but
    // a patch can be several times grainier after local processing, and
    // compression lets strong grain through more than weak. Only blocks far
    // above the base grain are called texture on strength alone.
    const far = Math.min(1, Math.max(0, (ratio - 6) / 4));
    // Fine detail above the grain that is more correlated from pixel to pixel
    // than the grain is organised (sand, foliage, fabric): noise that is
    // merely stronger comes through compression whiter, never redder.
    let dc = 0;
    for (let l = 0; l < 4; l++) dc = Math.max(dc, corr[l][k] - Math.max(0, c0[l]));
    const corrTex = Math.min(1, Math.max(0, (dc - 0.05) / 0.1)) * Math.min(1, Math.max(0, (ratio - 1.3) / 0.5));
    texture[k] = Math.max(structure[k], far, corrTex);
  }

  // Pass two: with texture known, the grain the photo has is read from the
  // plain blocks: the lowest substantial cluster, unless a clearly larger
  // one lies nearby (a flattened sky next to a grainier rest, say, where the
  // rest is the photo's grain). Grainier patches sit above it and smoother
  // ones below.
  const plain2 = (k: number): boolean => structure[k] < 0.2 && texture[k] < 0.3;
  const sf2: number[] = [];
  for (let k = 0; k < nb; k++) if (plain2(k)) sf2.push(sigma[k]);
  let B2: GrainCluster = B1;
  let bestCost = Infinity;
  for (const c of grainClusters(sorted(sf2))) {
    const cost = 0.65 * Math.abs(Math.log(c.ref / Math.max(1e-6, B1.ref))) - Math.log(c.share);
    if (cost < bestCost) {
      bestCost = cost;
      B2 = c;
    }
  }
  const ref2 = curveFrom(plain2, B2.ref);
  if (NOISE_DEBUG) NOISE_DEBUG({ sigma, gMed, gradPerSigma, structure, texture, ref: ref2, ref1, corr, c0, B1, B2, mean });
  return { block, bw, bh, sigma, mean, texture, ref: ref2, refSigma: B2.ref };
}

/** The lowest cluster at or above `minRef` holding at least 40 % of what the largest holds. */
function lowestSubstantial(cl: GrainCluster[], minRef: number): GrainCluster | null {
  let max = 0;
  for (const c of cl) max = Math.max(max, c.share);
  for (const c of cl) if (c.ref >= minRef && c.share >= 0.4 * max) return c;
  return null;
}

/** The cluster with the largest share (the lower one on a tie). */
function largest(cl: GrainCluster[]): GrainCluster | null {
  let best: GrainCluster | null = null;
  for (const c of cl) if (!best || c.share > best.share) best = c;
  return best;
}

export let NOISE_DEBUG: ((d: Record<string, unknown>) => void) | null = null;
export function setNoiseDebug(fn: typeof NOISE_DEBUG): void {
  NOISE_DEBUG = fn;
}

export interface NoiseModel {
  bins: number;
  /** sigma expected at the centre of each intensity bin (0..1) */
  curve: Float32Array;
  globalSigma: number; // median block sigma
}

/**
 * Expected noise as a function of intensity: the photo's base grain per
 * brightness bin as the noise map measured it (texture and smoothed regions
 * left out), with the overall base grain as globalSigma.
 */
export function noiseModel(map: NoiseMap): NoiseModel {
  return { bins: map.ref.length, curve: new Float32Array(map.ref), globalSigma: map.refSigma };
}

export function expectedSigma(model: NoiseModel, intensity: number): number {
  const t = Math.min(model.bins - 1, Math.max(0, intensity * model.bins - 0.5));
  const b0 = Math.floor(t);
  const b1 = Math.min(model.bins - 1, b0 + 1);
  const f = t - b0;
  return model.curve[b0] * (1 - f) + model.curve[b1] * f;
}

export interface NoiseDiscrepancy {
  /** per-block 20·log10(sigma / expected), smoothed with a 3x3 median over blocks */
  ratioDb: Float32Array;
  noisierFraction: number; // blocks above +thresholdDb
  smootherFraction: number; // blocks below -thresholdDb
  thresholdDb: number;
}

export function noiseDiscrepancy(map: NoiseMap, model: NoiseModel, thresholdDb = 3): NoiseDiscrepancy {
  const n = map.sigma.length;
  const raw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const e = Math.max(1e-6, expectedSigma(model, map.mean[i]));
    raw[i] = 20 * Math.log10(Math.max(1e-6, map.sigma[i]) / e);
  }
  const ratioDb = new Float32Array(n);
  const buf: number[] = [];
  for (let by = 0; by < map.bh; by++) {
    for (let bx = 0; bx < map.bw; bx++) {
      buf.length = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const x = bx + dx;
          const y = by + dy;
          if (x >= 0 && x < map.bw && y >= 0 && y < map.bh) buf.push(raw[y * map.bw + x]);
        }
      }
      buf.sort((a, b) => a - b);
      ratioDb[by * map.bw + bx] = buf[buf.length >> 1];
    }
  }
  let up = 0;
  let down = 0;
  for (let i = 0; i < n; i++) {
    if (ratioDb[i] > thresholdDb) up++;
    else if (ratioDb[i] < -thresholdDb) down++;
  }
  return { ratioDb, noisierFraction: up / n, smootherFraction: down / n, thresholdDb };
}

/** Bilinear upsample of a per-block map to a full-resolution plane. */
export function upsampleBlockMap(values: Float32Array, bw: number, bh: number, block: number, w: number, h: number): Plane {
  const out = makePlane(w, h);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(bh - 1, Math.max(0, (y + 0.5) / block - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(bh - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(bw - 1, Math.max(0, (x + 0.5) / block - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(bw - 1, x0 + 1);
      const tx = fx - x0;
      const a = values[y0 * bw + x0] * (1 - tx) + values[y0 * bw + x1] * tx;
      const b = values[y1 * bw + x0] * (1 - tx) + values[y1 * bw + x1] * tx;
      out.data[y * w + x] = a * (1 - ty) + b * ty;
    }
  }
  return out;
}

/** Separable box blur of radius r (window 2r+1), edge-clamped. */
export function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[row + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = s / (2 * r + 1);
      s += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / (2 * r + 1);
      s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

/**
 * Guided filter (He, Sun, Tang) with the plane as its own guide: an
 * edge-preserving smoother whose strength is set by eps (≈ noise variance).
 */
export function guidedFilter(p: Plane, r: number, eps: number): Plane {
  const { w, h } = p;
  const I = p.data;
  const n = w * h;
  const meanI = boxBlur(I, w, h, r);
  const II = new Float32Array(n);
  for (let i = 0; i < n; i++) II[i] = I[i] * I[i];
  const meanII = boxBlur(II, w, h, r);
  const a = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const varI = meanII[i] - meanI[i] * meanI[i];
    a[i] = varI / (varI + eps);
    b[i] = meanI[i] - a[i] * meanI[i];
  }
  const meanA = boxBlur(a, w, h, r);
  const meanB = boxBlur(b, w, h, r);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = meanA[i] * I[i] + meanB[i];
  return { w, h, data: out };
}

export interface NoiseNormaliseOptions {
  thresholdDb?: number; // only act outside ±threshold (default 1.5)
  radius?: number; // guided filter radius (default 2)
  /** target = expected sigma × targetScale; 1 levels the noise, 0.3 reduces it to 30 % everywhere (default 1) */
  targetScale?: number;
}

/**
 * Denoise toward the model where blocks are noisier than expected. Five
 * guided filters of increasing strength are tried; for each block the
 * weakest one that can reach the expected sigma is blended in by exactly the
 * amount that lands there, measured rather than assumed. Blocks with texture
 * are only ever given the weakest filters (which take grain-scale variation
 * and keep anything with more contrast than the grain), and less of them.
 * Returns a delta plane and the total blend map (0..1) that was used.
 */
export function denoiseDelta(p: Plane, map: NoiseMap, model: NoiseModel, opts: NoiseNormaliseOptions = {}): { delta: Plane; blend: Plane } {
  const thr = opts.thresholdDb ?? 1.5;
  const r = opts.radius ?? 2;
  const scale = opts.targetScale ?? 1;
  const nBlocks = map.sigma.length;
  const need = new Uint8Array(nBlocks);
  let any = false;
  for (let i = 0; i < nBlocks; i++) {
    const e = Math.max(1e-6, expectedSigma(model, map.mean[i]) * scale);
    if (map.texture[i] < 0.95 && 20 * Math.log10(Math.max(1e-6, map.sigma[i]) / e) > thr) {
      need[i] = 1;
      any = true;
    }
  }
  const delta = makePlane(p.w, p.h);
  const blendTotal = new Float32Array(nBlocks);
  if (!any) return { delta, blend: upsampleBlockMap(blendTotal, map.bw, map.bh, map.block, p.w, p.h) };

  const g = model.globalSigma;
  const candidates = [
    { r, eps: (2 * g) ** 2 + 1e-8 },
    { r: r + 1, eps: (6 * g) ** 2 + 1e-8 },
    { r: r + 2, eps: (14 * g) ** 2 + 1e-8 },
    { r: r + 3, eps: (30 * g) ** 2 + 1e-8 },
    { r: r + 5, eps: (60 * g) ** 2 + 1e-8 },
  ];
  // the strongest candidate a block may use: everything for plain grain, the
  // two weakest where some texture shows, only the weakest where it is clear
  const cap = (t: number): number => (t < 0.3 ? candidates.length - 1 : t < 0.7 ? 1 : 0);
  const remaining = new Uint8Array(need);
  for (let ci = 0; ci < candidates.length; ci++) {
    let left = 0;
    for (let i = 0; i < nBlocks; i++) left += remaining[i];
    if (!left) break;
    const cand = candidates[ci];
    const f = guidedFilter(p, cand.r, cand.eps);
    const fmap = noiseMap(f, map.block);
    const b = new Float32Array(nBlocks);
    for (let i = 0; i < nBlocks; i++) {
      if (!remaining[i]) continue;
      const e = Math.max(1e-6, expectedSigma(model, map.mean[i]) * scale);
      const s0 = map.sigma[i];
      const sf = fmap.sigma[i];
      if (sf <= e || ci >= cap(map.texture[i])) {
        // noise of I + b(f - I) ≈ (1-b)·s0 + b·sf  (the two are correlated)
        const bb = s0 - sf > 1e-9 ? (s0 - e) / (s0 - sf) : 1;
        b[i] = Math.min(1, Math.max(0, bb)) * (1 - map.texture[i]) ** 2;
        blendTotal[i] = Math.max(blendTotal[i], b[i]);
        remaining[i] = 0;
      }
    }
    const bm = upsampleBlockMap(b, map.bw, map.bh, map.block, p.w, p.h);
    for (let i = 0; i < delta.data.length; i++) delta.data[i] += (f.data[i] - p.data[i]) * bm.data[i];
  }
  return { delta, blend: upsampleBlockMap(blendTotal, map.bw, map.bh, map.block, p.w, p.h) };
}

/**
 * Add matched grain where blocks are smoother than expected (a previous
 * denoiser's footprint), so the output's noise is level across the frame.
 * Deterministic for a given seed.
 */
export function regrainDelta(p: Plane, map: NoiseMap, model: NoiseModel, opts: NoiseNormaliseOptions & { seed?: number } = {}): { delta: Plane; sigmaAdd: Plane } {
  const thr = opts.thresholdDb ?? 1.5;
  const add = new Float32Array(map.sigma.length);
  let any = false;
  for (let i = 0; i < map.sigma.length; i++) {
    const e = Math.max(1e-6, expectedSigma(model, map.mean[i]));
    const ratioDb = 20 * Math.log10(Math.max(1e-6, map.sigma[i]) / e);
    if (ratioDb < -thr) {
      add[i] = Math.sqrt(Math.max(0, e * e - map.sigma[i] * map.sigma[i]));
      any = true;
    }
  }
  const sigmaAdd = upsampleBlockMap(add, map.bw, map.bh, map.block, p.w, p.h);
  const delta = makePlane(p.w, p.h);
  if (!any) return { delta, sigmaAdd };
  const g = gaussian(rng(opts.seed ?? 0x5eed));
  for (let i = 0; i < delta.data.length; i++) delta.data[i] = sigmaAdd.data[i] > 0 ? sigmaAdd.data[i] * g() : 0;
  return { delta, sigmaAdd };
}

/** Noise sigma implied by the high-frequency floor of a Welch spectrum. */
export function noiseFromSpectrum(power: Float64Array, size: number, winSum: number, winSumSq: number, rMin = 0.4, rMax = 0.5): number {
  const c = size >> 1;
  const vals: number[] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const r = Math.hypot(x - c, y - c) / size;
      if (r >= rMin && r <= rMax) vals.push(power[y * size + x]);
    }
  }
  vals.sort((a, b) => a - b);
  // trimmed mean: the top 5 % may hold interference peaks or scene energy
  const keep = Math.max(1, Math.floor(vals.length * 0.95));
  let s = 0;
  for (let i = 0; i < keep; i++) s += vals[i];
  const meanP = s / keep;
  return Math.sqrt((meanP * winSum * winSum) / winSumSq);
}
