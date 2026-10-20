/**
 * Detail balance: how the photo's detail falls off from coarse to fine,
 * compared with what a camera photo of the same scene would show.
 *
 * Natural scenes have power falling off as a straight line on log-log axes
 * (power ∝ 1/f^α, α ≈ 2). A camera adds a gentle roll-off near the finest
 * scale and a floor of grain. A photo that was smoothed (AI denoise,
 * upscaling, heavy compression) falls below that line in the fine band;
 * one that was oversharpened rises above it in a hump. The line is fitted
 * to the photo's own coarse-to-mid detail, so the reference is this scene,
 * not some average photo.
 *
 * The correction is a set of detail bands (differences of small blurs),
 * each scaled so the photo's fine-detail curve meets the reference. It runs
 * in the image domain with box blurs, so it is fast on large photos, and
 * only the brightness detail is changed.
 */

import { boxBlur } from './noise';
import { makePlane, type Plane } from './plane';
import { subsampleGrid, tileGrid, welchSpectrum } from './tiles';
import { window2d } from './window';

export interface RadialProfile {
  /** frequency of each bin, cycles per pixel */
  f: Float32Array;
  /** robust (median) power per ring, dB */
  db: Float32Array;
  /** dB of white noise of sigma 1 (plane units) in these power units */
  unitNoiseDb: number;
}

/** Median log power per radius of the averaged spectrum, axes and JPEG block harmonics left out. */
export function radialProfile(p: Plane, maxTiles = 36, size = 256): RadialProfile {
  const grid = subsampleGrid(tileGrid(p.w, p.h, size), maxTiles);
  const welch = welchSpectrum(p, grid, 'hann');
  const n = welch.size;
  const c = n >> 1;
  const rings: number[][] = Array.from({ length: c + 1 }, () => []);
  const near8 = (b: number): boolean => {
    const f = Math.abs(b) / n;
    const k = Math.round(f * 8);
    return k > 0 && Math.abs(f - k / 8) * n < 1.5;
  };
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const dx = x - c;
      const dy = y - c;
      if (Math.abs(dx) <= 1 || Math.abs(dy) <= 1) continue;
      if (near8(dx) && near8(dy)) continue;
      const r = Math.round(Math.hypot(dx, dy));
      if (r < 1 || r > c) continue;
      rings[r].push(welch.power[y * n + x]);
    }
  }
  const f = new Float32Array(c);
  const db = new Float32Array(c);
  for (let r = 1; r <= c; r++) {
    const l = rings[r].sort((a, b) => a - b);
    f[r - 1] = r / n;
    db[r - 1] = l.length ? 10 * Math.log10(Math.max(1e-30, l[l.length >> 1])) : -300;
  }
  // a median of exponentially distributed power sits ln2 below the mean
  for (let i = 0; i < db.length; i++) db[i] += 10 * Math.log10(1 / Math.LN2);
  const win = window2d('hann', n, n);
  const unitNoiseDb = 10 * Math.log10(win.sumSq / (win.sum * win.sum));
  return { f, db, unitNoiseDb };
}

/** The camera's own fall-off near the finest scale that the reference allows, dB. */
function rolloffDb(f: number): number {
  return f <= 0.15 ? 0 : 6 * ((f - 0.15) / 0.35) ** 2;
}

export interface DetailBalance {
  f: Float32Array;
  measured: Float32Array; // dB
  /** reference for scene detail: the coarse-to-mid line extended, with the camera's roll-off */
  reference: Float32Array; // dB
  /** reference with the grain floor added, for judging excess */
  referenceWithGrain: Float32Array; // dB
  /** fitted slope, dB per decade */
  slope: number;
  /** the reference line, to judge other copies of the photo against */
  line: { slope: number; mx: number; my: number };
  /** mean shortfall of fine detail against the reference over 0.10–0.30 cycles/px, dB (positive = too smooth) */
  deficitDb: number;
  /** mean excess over the reference (with grain) over 0.10–0.30 cycles/px, dB (positive = oversharpened) */
  excessDb: number;
}

const BAND_LO = 0.1;
const BAND_HI = 0.3;

export function detailBalance(prof: RadialProfile, grainSigma: number, against?: { slope: number; mx: number; my: number }): DetailBalance {
  const { f, db } = prof;
  // fit a line to dB against log10 f over the coarse-to-mid band
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  let n = 0;
  for (let i = 0; i < f.length; i++) {
    if (f[i] < 0.02 || f[i] > 0.07) continue;
    const x = Math.log10(f[i]);
    sx += x;
    sy += db[i];
    sxx += x * x;
    sxy += x * db[i];
    n++;
  }
  let slope = n > 2 ? (n * sxy - sx * sy) / (n * sxx - sx * sx) : -20;
  slope = Math.max(-35, Math.min(-15, slope)); // power 1/f^1.5 … 1/f^3.5
  let mx = n ? sx / n : -1.3;
  let my = n ? sy / n : 0;
  // a copy of the same photo is judged against the original's line
  if (against) ({ slope, mx, my } = against);
  const grainDb = 10 * Math.log10(Math.max(1e-12, grainSigma * grainSigma)) + prof.unitNoiseDb;
  const reference = new Float32Array(f.length);
  const referenceWithGrain = new Float32Array(f.length);
  let def = 0;
  let exc = 0;
  let nb = 0;
  for (let i = 0; i < f.length; i++) {
    const ref = my + slope * (Math.log10(f[i]) - mx) - rolloffDb(f[i]);
    reference[i] = ref;
    referenceWithGrain[i] = 10 * Math.log10(10 ** (ref / 10) + 10 ** (grainDb / 10));
    if (f[i] >= BAND_LO && f[i] <= BAND_HI) {
      def += ref - db[i];
      exc += db[i] - referenceWithGrain[i];
      nb++;
    }
  }
  return { f, measured: db, reference, referenceWithGrain, slope, line: { slope, mx, my }, deficitDb: nb ? def / nb : 0, excessDb: nb ? exc / nb : 0 };
}

/**
 * The gain wanted at each frequency, dB: down to the reference (with grain)
 * where detail is in excess, up toward the scene reference where it is
 * missing, limited to `maxBoostDb` and to the band a re-save keeps.
 */
export function wantedGainDb(b: DetailBalance, mode: 'smooth' | 'sharp' | 'both', maxBoostDb: number): Float32Array {
  const g = new Float32Array(b.f.length);
  for (let i = 0; i < b.f.length; i++) {
    const f = b.f[i];
    let v = 0;
    const over = b.measured[i] - b.referenceWithGrain[i];
    const under = b.reference[i] - b.measured[i];
    if (over > 1 && mode !== 'smooth') v = -Math.min(12, over);
    else if (under > 1 && mode !== 'sharp') v = Math.min(maxBoostDb, under);
    // leave coarse structure alone; fade boosts out past what a re-save keeps
    const lo = Math.min(1, Math.max(0, (f - 0.04) / 0.03));
    const hi = v > 0 ? Math.min(1, Math.max(0, (0.45 - f) / 0.1)) : 1;
    g[i] = v * lo * hi;
  }
  // smooth the curve over five bins
  const s = new Float32Array(g.length);
  for (let i = 0; i < g.length; i++) {
    let a = 0;
    let m = 0;
    for (let k = -2; k <= 2; k++) {
      const j = i + k;
      if (j >= 0 && j < g.length) {
        a += g[j];
        m++;
      }
    }
    s[i] = a / m;
  }
  return s;
}

// ---------------------------------------------------------------------------
// Band filter bank: lowpasses L0 = identity, then binomial [1 2 1]/4 passes and
// three-box cascades of growing width. Band k = L(k-1) - L(k).
// Output = I + Σ (c_k - 1)·band_k.
// ---------------------------------------------------------------------------

type Low = { kind: 'bin'; passes: number } | { kind: 'box'; r: number };
const LOWS: Low[] = [
  { kind: 'bin', passes: 1 },
  { kind: 'bin', passes: 2 },
  { kind: 'box', r: 1 },
  { kind: 'box', r: 2 },
  { kind: 'box', r: 3 },
  { kind: 'box', r: 5 },
];
const LEVELS = LOWS.length;

function lowpassResponse(level: number, fx: number): number {
  if (level === 0) return 1;
  const L = LOWS[level - 1];
  if (L.kind === 'bin') return Math.cos(Math.PI * fx) ** (2 * L.passes);
  const m = 2 * L.r + 1;
  const s = Math.sin(Math.PI * fx);
  const h = Math.abs(s) < 1e-9 ? 1 : Math.sin(Math.PI * fx * m) / (m * s);
  return h * h * h;
}

/** Radial response of lowpass `level`: the separable 2-D response averaged over axis and diagonal directions. */
function lowpassRadial(level: number, f: number): number {
  const axis = lowpassResponse(level, f);
  const d = f / Math.SQRT2;
  const diag = lowpassResponse(level, d) ** 2;
  return 0.5 * (axis + diag);
}

/**
 * Band gains that reproduce the wanted gain curve (in amplitude): the bands
 * telescope to the identity, so giving each band the wanted gain averaged
 * over where that band lives reproduces a smooth curve closely, with no
 * band fighting another. Clamped to 0.3–3×.
 */
export function bandGains(f: Float32Array, gainDb: Float32Array, amount: number): number[] {
  const out: number[] = [];
  for (let k = 1; k <= LEVELS; k++) {
    let sw = 0;
    let sg = 0;
    for (let i = 0; i < f.length; i++) {
      const b = Math.max(0, lowpassRadial(k - 1, f[i]) - lowpassRadial(k, f[i]));
      const w = b * b;
      sw += w;
      sg += w * 10 ** ((gainDb[i] * amount) / 20);
    }
    out.push(Math.min(3, Math.max(0.3, sw > 0 ? sg / sw : 1)));
  }
  // a few corrections: where the response overshoots or falls short, nudge
  // the bands that live there
  for (let it = 0; it < 6; it++) {
    const got = achievedGainDb(f, out);
    for (let k = 1; k <= LEVELS; k++) {
      let sw = 0;
      let se = 0;
      for (let i = 0; i < f.length; i++) {
        const b = Math.max(0, lowpassRadial(k - 1, f[i]) - lowpassRadial(k, f[i]));
        const w = b * b * (f[i] < 0.07 ? 4 : 1);
        sw += w;
        se += w * (gainDb[i] * amount - got[i]);
      }
      if (sw > 0) out[k - 1] = Math.min(3, Math.max(0.3, out[k - 1] * 10 ** ((0.7 * se) / sw / 20)));
    }
  }
  return out;
}

/** The response the band gains actually give, for display and checking, dB. */
export function achievedGainDb(f: Float32Array, gains: number[]): Float32Array {
  const out = new Float32Array(f.length);
  for (let i = 0; i < f.length; i++) {
    let r = lowpassRadial(LEVELS, f[i]);
    for (let k = 1; k <= LEVELS; k++) r += gains[k - 1] * (lowpassRadial(k - 1, f[i]) - lowpassRadial(k, f[i]));
    out[i] = 20 * Math.log10(Math.max(1e-6, Math.abs(r)));
  }
  return out;
}

function binomial(src: Float32Array, w: number, h: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const a = src[row + Math.max(0, x - 1)];
      const c = src[row + Math.min(w - 1, x + 1)];
      tmp[row + x] = 0.25 * a + 0.5 * src[row + x] + 0.25 * c;
    }
  }
  for (let y = 0; y < h; y++) {
    const up = Math.max(0, y - 1) * w;
    const dn = Math.min(h - 1, y + 1) * w;
    const row = y * w;
    for (let x = 0; x < w; x++) out[row + x] = 0.25 * tmp[up + x] + 0.5 * tmp[row + x] + 0.25 * tmp[dn + x];
  }
  return out;
}

function box3(src: Float32Array, w: number, h: number, r: number): Float32Array {
  return boxBlur(boxBlur(boxBlur(src, w, h, r), w, h, r), w, h, r);
}

/** Delta for one plane: Σ (c_k − 1)·band_k. */
export function detailDelta(p: Plane, gains: number[]): Plane {
  const { w, h } = p;
  const lows: Float32Array[] = [p.data];
  let bin = p.data;
  for (const L of LOWS) {
    if (L.kind === 'bin') {
      while (lows.length - 1 < L.passes) {
        bin = binomial(bin, w, h);
        lows.push(bin);
      }
    } else lows.push(box3(p.data, w, h, L.r));
  }
  const out = makePlane(w, h);
  for (let k = 1; k <= LEVELS; k++) {
    const g = gains[k - 1] - 1;
    if (Math.abs(g) < 1e-4) continue;
    const a = lows[k - 1];
    const b = lows[k];
    for (let i = 0; i < out.data.length; i++) out.data[i] += g * (a[i] - b[i]);
  }
  return out;
}
