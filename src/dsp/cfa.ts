/**
 * The camera sensor's colour filter array (Bayer grid), and its traces.
 *
 * A sensor sees one colour per pixel: red, green or blue in a 2×2 grid
 * (two greens on a diagonal). The camera then fills in the two missing
 * colours at every pixel (demosaicing). That leaves two kinds of traces:
 *
 *  - artefacts: false colour and coloured fringes on fine detail, where the
 *    missing colours were guessed wrong;
 *  - a signature: interpolated values are smoother than measured ones, so
 *    pixels on the measured positions differ from their neighbours more
 *    than pixels on the filled-in positions, in a 2×2 pattern (Popescu &
 *    Farid 2005 use this to tell edited regions apart).
 *
 * Here the false colour is found and removed, the original grid's position
 * is found from the signature, and at the very end the finished image is
 * put back through a sensor: re-sampled to one colour per pixel on that
 * grid, in linear light, and demosaiced again with Malvar–He–Cutler (2004),
 * so the whole photo, edits and grain included, carries one consistent
 * sensor signature.
 */

import { boxBlur } from './noise';
import { guidedFilterWithGuide } from './chroma';
import { makePlane, type Plane } from './plane';
import { gaussian, rng } from '../synth/generate';
import { requantizeRgb } from './jpegdomain';

// ---------------------------------------------------------------------------
// sRGB ↔ linear
// ---------------------------------------------------------------------------

const toLin = new Float32Array(4096);
for (let i = 0; i < 4096; i++) {
  const v = i / 4095;
  toLin[i] = v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}
function srgbToLinear(v: number): number {
  const x = v <= 0 ? 0 : v >= 1 ? 1 : v;
  const f = x * 4095;
  const i = Math.floor(f);
  if (i >= 4095) return toLin[4095];
  return toLin[i] + (toLin[i + 1] - toLin[i]) * (f - i);
}
function linearToSrgb(v: number): number {
  const x = v <= 0 ? 0 : v;
  return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
}

// ---------------------------------------------------------------------------
// The grid's position
// ---------------------------------------------------------------------------

export interface CfaPhase {
  /** red sits at (x % 2, y % 2) = (redX, redY); blue on the opposite corner; greens on the other diagonal */
  redX: 0 | 1;
  redY: 0 | 1;
  /** how clearly the grid shows, 0 (not at all) … 1; under ~0.02 the position is a guess */
  strength: number;
  /** true when the position was read from the photo rather than assumed */
  detected: boolean;
}

/** Mean |value − mean of its 4 neighbours| on each of the four 2×2 positions. */
function siteResiduals(p: Plane): number[] {
  const { w, h } = p;
  const d = p.data;
  const sum = [0, 0, 0, 0];
  const cnt = [0, 0, 0, 0];
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const i = y * w + x;
      const e = Math.abs(d[i] - 0.25 * (d[i - 1] + d[i + 1] + d[i - w] + d[i + w]));
      const k = (y & 1) * 2 + (x & 1);
      sum[k] += e;
      cnt[k]++;
    }
  }
  return sum.map((s, k) => s / Math.max(1, cnt[k]));
}

/**
 * Where the sensor grid sat. Green, half of all sensor pixels, was measured
 * on the diagonal whose pixels stand out more from their neighbours, and
 * that can be read reliably. Which of the other two corners held red and
 * which blue cannot: good demosaicing fills in red at a blue pixel mostly
 * from that blue pixel's own value, so both corners look alike. Red is put
 * on the corner of the same row parity as the top-left green-free corner
 * (RGGB or GRBG), the commonest layouts.
 */
export function detectCfaPhase(planes: [Plane, Plane, Plane]): CfaPhase {
  const g = siteResiduals(planes[1]);
  const diagA = g[0] + g[3]; // (0,0) and (1,1)
  const diagB = g[1] + g[2]; // (1,0) and (0,1)
  const strength = Math.abs(diagA - diagB) / Math.max(1e-9, diagA + diagB);
  const detected = strength >= 0.02;
  if (!detected) return { redX: 0, redY: 0, strength, detected: false }; // RGGB, the most common
  // green on (0,0),(1,1) → red at (1,0) (GRBG); green on (1,0),(0,1) → red at (0,0) (RGGB)
  return diagA > diagB ? { redX: 1, redY: 0, strength, detected: true } : { redX: 0, redY: 0, strength, detected: true };
}

/** True when two grid positions put green on the same diagonal (the part that can be read). */
export function sameGreenGrid(a: CfaPhase, b: CfaPhase): boolean {
  return ((a.redX + a.redY) & 1) === ((b.redX + b.redY) & 1);
}

/** How clearly the 2×2 sensor signature shows in a copy, 0 … 1. */
export function cfaSignature(planes: [Plane, Plane, Plane]): number {
  return detectCfaPhase(planes).strength;
}

// ---------------------------------------------------------------------------
// Re-sampling through a sensor and Malvar–He–Cutler demosaicing
// ---------------------------------------------------------------------------

// kernels ×8, 5×5 row-major
const K_G_AT_RB = [0, 0, -1, 0, 0, 0, 0, 2, 0, 0, -1, 2, 4, 2, -1, 0, 0, 2, 0, 0, 0, 0, -1, 0, 0];
// colour at a green site whose row holds that colour (neighbours left/right)
const K_C_AT_G_ROW = [0, 0, 0.5, 0, 0, 0, -1, 0, -1, 0, -1, 4, 5, 4, -1, 0, -1, 0, -1, 0, 0, 0, 0.5, 0, 0];
// colour at a green site whose column holds that colour (neighbours above/below)
const K_C_AT_G_COL = [0, 0, -1, 0, 0, 0, -1, 4, -1, 0, 0.5, 0, 5, 0, 0.5, 0, -1, 4, -1, 0, 0, 0, -1, 0, 0];
// red at a blue site, blue at a red site
const K_C_AT_OPP = [0, 0, -1.5, 0, 0, 0, 2, 0, 2, 0, -1.5, 0, 6, 0, -1.5, 0, 2, 0, 2, 0, 0, 0, -1.5, 0, 0];
// bilinear: plain neighbour averages (older and simpler cameras; leaves a clearer grid)
const B_G_AT_RB = [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
const B_C_AT_G_ROW = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const B_C_AT_G_COL = [0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0];
const B_C_AT_OPP = [0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0];

export type DemosaicMethod = 'mhc' | 'bilinear';

/**
 * Put RGB through a sensor: linear light, one colour per pixel on the given
 * grid, then Malvar–He–Cutler demosaicing, back to sRGB. Returns new planes.
 */
export interface SensorNoise {
  /** photon (shot) noise, given as the noise it makes at mid grey, in sRGB levels; read noise is a fifth of it */
  levels: number;
  seed: number;
  /** how the missing colours are filled in (default Malvar–He–Cutler) */
  method?: DemosaicMethod;
}

export function rebakeThroughSensor(planes: [Plane, Plane, Plane], phase: CfaPhase, noise?: SensorNoise): [Plane, Plane, Plane] {
  const { w, h } = planes[0];
  const site = (x: number, y: number): 0 | 1 | 2 => {
    const rx = (x & 1) === phase.redX;
    const ry = (y & 1) === phase.redY;
    if (rx && ry) return 0; // red
    if (!rx && !ry) return 2; // blue
    return 1; // green
  };
  // the mosaic, in linear light, with each photosite's own noise: photon
  // noise grows with the light (variance ∝ signal, as Poisson counting
  // gives), plus a small constant read noise
  const M = new Float32Array(w * h);
  const midLin = 0.2140; // sRGB 0.5
  const slope = 1.0826; // d sRGB / d linear at mid grey
  const sigMid = noise ? noise.levels / 255 / slope : 0;
  const k = (sigMid * sigMid) / midLin;
  const read2 = (0.2 * sigMid) ** 2;
  const g = noise ? gaussian(rng(noise.seed)) : null;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = srgbToLinear(planes[site(x, y)].data[i]);
      M[i] = g ? Math.max(0, v + Math.sqrt(k * v + read2) * g()) : v;
    }
  // the mosaic padded by 2 on each side with mirrored values (keeps the grid's parity),
  // so the kernels never need bounds checks
  const pw = w + 4;
  const P = new Float32Array(pw * (h + 4));
  const mirror = (v: number, n: number): number => (v < 0 ? -v : v >= n ? 2 * n - 2 - v : v);
  for (let y = -2; y < h + 2; y++) {
    const sy = mirror(y, h);
    for (let x = -2; x < w + 2; x++) P[(y + 2) * pw + x + 2] = M[sy * w + mirror(x, w)];
  }
  const taps = (k: number[]): { o: Int32Array; c: Float32Array } => {
    const o: number[] = [];
    const c: number[] = [];
    for (let dy = -2; dy <= 2; dy++)
      for (let dx = -2; dx <= 2; dx++) {
        const v = k[(dy + 2) * 5 + dx + 2];
        if (v) {
          o.push(dy * pw + dx);
          c.push(v / 8);
        }
      }
    return { o: Int32Array.from(o), c: Float32Array.from(c) };
  };
  const bil = noise?.method === 'bilinear';
  const TG = taps(bil ? B_G_AT_RB : K_G_AT_RB);
  const TR = taps(bil ? B_C_AT_G_ROW : K_C_AT_G_ROW);
  const TC = taps(bil ? B_C_AT_G_COL : K_C_AT_G_COL);
  const TO = taps(bil ? B_C_AT_OPP : K_C_AT_OPP);
  const conv = (t: { o: Int32Array; c: Float32Array }, j: number): number => {
    let s = 0;
    for (let k = 0; k < t.o.length; k++) s += t.c[k] * P[j + t.o[k]];
    return s;
  };
  const out: [Plane, Plane, Plane] = [makePlane(w, h), makePlane(w, h), makePlane(w, h)];
  for (let y = 0; y < h; y++) {
    const rowHasRed = (y & 1) === phase.redY;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const j = (y + 2) * pw + x + 2;
      const s = site(x, y);
      let R: number;
      let G: number;
      let B: number;
      if (s === 0) {
        R = M[i];
        G = conv(TG, j);
        B = conv(TO, j);
      } else if (s === 2) {
        B = M[i];
        G = conv(TG, j);
        R = conv(TO, j);
      } else {
        G = M[i];
        if (rowHasRed) {
          R = conv(TR, j);
          B = conv(TC, j);
        } else {
          B = conv(TR, j);
          R = conv(TC, j);
        }
      }
      out[0].data[i] = Math.min(1, Math.max(0, linearToSrgb(R)));
      out[1].data[i] = Math.min(1, Math.max(0, linearToSrgb(G)));
      out[2].data[i] = Math.min(1, Math.max(0, linearToSrgb(B)));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// False colour from the camera's demosaicing
// ---------------------------------------------------------------------------

export interface FalseColour {
  /** rms of colour that brightness does not explain, on fine detail, levels */
  detailLevels: number;
  /** the same on smooth areas (ordinary colour noise), levels */
  flatLevels: number;
  /** fraction of pixels that are fine detail */
  detailFraction: number;
}

/** Local standard deviation of brightness over 5×5, plane units. */
function localStd(Y: Plane): Float32Array {
  const { w, h } = Y;
  const m = boxBlur(Y.data, w, h, 2);
  const sq = new Float32Array(w * h);
  for (let i = 0; i < sq.length; i++) sq[i] = Y.data[i] * Y.data[i];
  const m2 = boxBlur(sq, w, h, 2);
  const out = new Float32Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(Math.max(0, m2[i] - m[i] * m[i]));
  return out;
}

/** How much the fix acts at a pixel: by how much fine brightness detail is there (2 → 6 levels local spread). */
function detailWeight(sd: number): number {
  return Math.min(1, Math.max(0, (sd * 255 - 2) / 4));
}

const FC_EPS = (2 / 255) ** 2;

/**
 * False colour is colour that does not follow brightness at the scale of a
 * few pixels, sitting on fine detail: the residual after a 5×5 luma-guided
 * fit of each colour plane, measured where brightness has fine detail.
 */
export function measureFalseColour(Y: Plane, Cb: Plane, Cr: Plane): FalseColour {
  const sd = localStd(Y);
  const fb = guidedFilterWithGuide(Cb, Y, 2, FC_EPS);
  const fr = guidedFilterWithGuide(Cr, Y, 2, FC_EPS);
  let sD = 0;
  let nD = 0;
  let sF = 0;
  let nF = 0;
  for (let i = 0; i < sd.length; i++) {
    const r2 = (Cb.data[i] - fb.data[i]) ** 2 + (Cr.data[i] - fr.data[i]) ** 2;
    const lv = sd[i] * 255;
    if (lv >= 6) {
      sD += r2;
      nD++;
    } else if (lv < 2) {
      sF += r2;
      nF++;
    }
  }
  return {
    detailLevels: nD ? Math.sqrt(sD / nD / 2) * 255 : 0,
    flatLevels: nF ? Math.sqrt(sF / nF / 2) * 255 : 0,
    detailFraction: nD / sd.length,
  };
}

/** Deltas for Cb and Cr: colour on fine detail pulled onto the luma-guided fit. */
export function falseColourDelta(Y: Plane, Cb: Plane, Cr: Plane, amount: number): { dCb: Plane; dCr: Plane } {
  const sd = localStd(Y);
  const out: Plane[] = [];
  for (const C of [Cb, Cr]) {
    const f = guidedFilterWithGuide(C, Y, 2, FC_EPS);
    const d = makePlane(C.w, C.h);
    for (let i = 0; i < d.data.length; i++) d.data[i] = (f.data[i] - C.data[i]) * detailWeight(sd[i]) * amount;
    out.push(d);
  }
  return { dCb: out[0], dCr: out[1] };
}

/** A centred crop, starting on even coordinates so the grid keeps its place. */
function centreCrop(planes: [Plane, Plane, Plane], size: number): [Plane, Plane, Plane] {
  const { w, h } = planes[0];
  const cw = Math.min(w, size);
  const ch = Math.min(h, size);
  const x0 = Math.floor((w - cw) / 4) * 2;
  const y0 = Math.floor((h - ch) / 4) * 2;
  return planes.map((p) => {
    const d = new Float32Array(cw * ch);
    for (let y = 0; y < ch; y++) d.set(p.data.subarray((y0 + y) * w + x0, (y0 + y) * w + x0 + cw), y * cw);
    return { w: cw, h: ch, data: d };
  }) as [Plane, Plane, Plane];
}

/**
 * How the bake should run. The grid has to come through the file being
 * written, whose encoder rounds the finest detail away: so the target is
 * how clearly the original photo's grid shows after that same rounding,
 * and each try is judged after it too. Malvar–He–Cutler filling-in when it
 * can get there, otherwise plain bilinear (older and simpler cameras),
 * then the least photosite noise (0.3–4 levels at mid grey) that reaches
 * the target. Searched on a 512-px crop; the picture's own grain counts.
 */
export type BakeStyle = 'sharp' | 'match';

export function calibrateSensorNoise(planes: [Plane, Plane, Plane], phase: CfaPhase, source: [Plane, Plane, Plane], quality: number, seed: number, style: BakeStyle = 'sharp'): SensorNoise & { target: number } {
  const crop = centreCrop(planes, 512);
  const after = (p: [Plane, Plane, Plane]): number => detectCfaPhase(requantizeRgb(p, quality)).strength;
  const target = Math.min(0.2, after(centreCrop(source, 512)));
  const sig = (lv: number, method: DemosaicMethod): number => after(rebakeThroughSensor(crop, phase, { levels: lv, seed, method }));
  // 'sharp': Malvar–He–Cutler, noise kept subtle (0.3–2 levels at mid grey);
  // the grid may come out fainter than the original's rather than the photo
  // grainier or softer. 'match': whichever filling-in reaches the original's
  // grid (bilinear when Malvar–He–Cutler cannot), with up to 4 levels of noise.
  const lo0 = 0.3;
  const hi0 = style === 'sharp' ? 2 : 4;
  if (target < 0.02) return { levels: lo0, seed, method: 'mhc', target };
  let method: DemosaicMethod = 'mhc';
  if (style === 'match' && sig(hi0, 'mhc') < 0.8 * target) method = 'bilinear';
  let lo = lo0;
  let hi = hi0;
  if (sig(lo, method) >= target) return { levels: lo, seed, method, target };
  if (sig(hi, method) <= target) return { levels: hi, seed, method, target };
  for (let it = 0; it < 6; it++) {
    const mid = Math.sqrt(lo * hi);
    if (sig(mid, method) < target) lo = mid;
    else hi = mid;
  }
  return { levels: Math.sqrt(lo * hi), seed, method, target };
}
