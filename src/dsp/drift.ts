/**
 * Repeating patterns that drift across the frame.
 *
 * A photo of a screen, a scan of a print, the beat between a fabric and the
 * sensor: in any small patch the pattern is a clean set of stripes, but its
 * spacing and direction change smoothly from one part of the frame to another
 * (perspective, the lens, the beat between two grids). The averaged spectrum
 * of the whole photo smears such a pattern into nothing, so it is found and
 * removed window by window instead, the way fringe patterns with a varying
 * frequency are handled in optical metrology (windowed Fourier filtering;
 * Kemao, Applied Optics 2004 and Optics and Lasers in Engineering 2007):
 *
 *  1. 256-px Hann windows overlapping by half. In each, a spike that stands
 *     well above the window's own radial background and is isolated (a
 *     straight edge makes a ridge, not a spike) is a candidate, located to a
 *     fraction of a bin.
 *  2. Candidates in neighbouring windows at nearly the same frequency are
 *     linked. A chain that spans enough windows is a pattern; a lone spike is
 *     texture or chance. The chain is what tells interference from scene
 *     texture: it runs on across skin, cloth and wall alike, and its frequency
 *     changes smoothly rather than stopping at an object's edge.
 *  3. Removal takes out, in each window, only the narrow band of frequencies
 *     around the tracked spike, each bin scaled by how far it stands above the
 *     window's background (the optimum-notch idea of Gonzalez & Woods), and
 *     overlap-adds the windows. Everything outside that sliver of the spectrum
 *     is left exactly as it was: no blur, no smoothing, no softening.
 */

import { fft2d } from './fft';
import { makePlane, type Plane } from './plane';
import { window2d } from './window';

export const DRIFT_N = 256;
export const DRIFT_HOP = 128;
const MIN_R = 4; // bins: periods longer than 64 px are the banding fix's business
const LINK_BINS = 2.5; // same pattern in the next window if within this many bins
const BAND_R = 3; // bins taken out around the tracked spike (the Hann main lobe is ±2)

export interface DriftNode {
  gx: number;
  gy: number;
  u: number; // cycles / pixel
  v: number;
  amplitude: number; // peak amplitude of the stripes here, plane units
  excessDb: number; // how far the spike stands above the window's background
}

export interface DriftTrack {
  nodes: DriftNode[];
  gridW: number;
  gridH: number;
  period: number; // px, median over the nodes
  periodMin: number;
  periodMax: number;
  angleDeg: number; // direction of the wave vector, 0..180, median
  angleSpread: number; // degrees, half-range about the median
  amplitude: number; // median over nodes
  amplitudeP90: number;
  excessDb: number; // median
  coverage: number; // nodes / interior windows (the border ring is not searched)
  stationary: boolean; // drifts by less than a bin: a fixed pattern (or the JPEG grid)
}

export interface DriftOptions {
  /** spikes must stand this far above the window's background (dB) */
  minExcessDb?: number;
  /** and this far above all but the top four bins of their 4–6 bin surroundings (dB) */
  isolationDb?: number;
  /** a chain needs at least this many windows (default: 4 % of the interior ones, at least 6) */
  minNodes?: number;
  /** tracks whose median amplitude is below this are dropped (plane units) */
  minAmplitude?: number;
  maxTracks?: number;
}

// --- geometry ------------------------------------------------------------------------

export interface DriftGeometry {
  gridW: number;
  gridH: number;
  padLeft: number;
  padTop: number;
  pw: number; // padded size
  ph: number;
}

export function driftGeometry(w: number, h: number): DriftGeometry {
  const gridW = Math.ceil(w / DRIFT_HOP) + 1;
  const gridH = Math.ceil(h / DRIFT_HOP) + 1;
  return { gridW, gridH, padLeft: DRIFT_N / 2, padTop: DRIFT_N / 2, pw: (gridW - 1) * DRIFT_HOP + DRIFT_N, ph: (gridH - 1) * DRIFT_HOP + DRIFT_N };
}

/** Reflect-padded copy sized for the window grid. */
function padPlane(p: Plane, g: DriftGeometry): Float64Array {
  const out = new Float64Array(g.pw * g.ph);
  const refl = (i: number, n: number): number => {
    if (n === 1) return 0;
    const period = 2 * n - 2;
    i = ((i % period) + period) % period;
    return i < n ? i : period - i;
  };
  const xs = new Int32Array(g.pw);
  for (let x = 0; x < g.pw; x++) xs[x] = refl(x - g.padLeft, p.w);
  for (let y = 0; y < g.ph; y++) {
    const sy = refl(y - g.padTop, p.h) * p.w;
    const row = y * g.pw;
    for (let x = 0; x < g.pw; x++) out[row + x] = p.data[sy + xs[x]];
  }
  return out;
}

// --- per-window spectrum ------------------------------------------------------------

const N = DRIFT_N;
const NN = N * N;
const hann = window2d('hann', N, N);

/** signed bin coordinate of an unshifted FFT index */
const sbin = (k: number): number => (k < N / 2 ? k : k - N);

interface RingTables {
  ringOf: Int16Array; // integer radius per bin
  rings: Int32Array[]; // bin indices per integer radius
}
let ringTables: RingTables | null = null;
function tables(): RingTables {
  if (ringTables) return ringTables;
  const ringOf = new Int16Array(NN);
  const lists: number[][] = [];
  for (let ky = 0; ky < N; ky++)
    for (let kx = 0; kx < N; kx++) {
      const r = Math.round(Math.hypot(sbin(kx), sbin(ky)));
      ringOf[ky * N + kx] = r;
      (lists[r] ??= []).push(ky * N + kx);
    }
  ringTables = { ringOf, rings: lists.map((l) => Int32Array.from(l ?? [])) };
  return ringTables;
}

/** Hann-windowed, mean-removed tile of the padded plane at (x0, y0), transformed in place into re/im. */
function windowSpectrum(padded: Float64Array, pw: number, x0: number, y0: number, re: Float64Array, im: Float64Array): void {
  let mean = 0;
  for (let y = 0; y < N; y++) {
    const row = (y0 + y) * pw + x0;
    for (let x = 0; x < N; x++) mean += padded[row + x];
  }
  mean /= NN;
  for (let y = 0; y < N; y++) {
    const row = (y0 + y) * pw + x0;
    for (let x = 0; x < N; x++) re[y * N + x] = (padded[row + x] - mean) * hann.data[y * N + x];
  }
  im.fill(0);
  fft2d(re, im, N, N, false);
}

function medianOf(buf: Float64Array, n: number): number {
  const a = buf.subarray(0, n);
  a.sort();
  return n & 1 ? a[n >> 1] : 0.5 * (a[(n >> 1) - 1] + a[n >> 1]);
}

/** log10 power per bin and its radial-median background. */
function ringBackground(P: Float64Array, L: Float64Array, ringMed: Float64Array, scratch: Float64Array): void {
  const { rings } = tables();
  for (let i = 0; i < NN; i++) L[i] = Math.log10(P[i] + 1e-30);
  for (let r = 0; r < rings.length; r++) {
    const list = rings[r];
    if (!list.length) {
      ringMed[r] = r > 0 ? ringMed[r - 1] : 0;
      continue;
    }
    for (let k = 0; k < list.length; k++) scratch[k] = L[list[k]];
    ringMed[r] = medianOf(scratch, list.length);
  }
}

interface Candidate {
  gx: number;
  gy: number;
  u: number;
  v: number;
  amplitude: number;
  excessDb: number;
}

/** Isolated spikes in one window's spectrum, upper half-plane, sub-bin located. */
function windowCandidates(re: Float64Array, im: Float64Array, gx: number, gy: number, minExcess: number, isolation: number, P: Float64Array, L: Float64Array, ringMed: Float64Array, scratch: Float64Array): Candidate[] {
  const { ringOf } = tables();
  for (let i = 0; i < NN; i++) P[i] = re[i] * re[i] + im[i] * im[i];
  ringBackground(P, L, ringMed, scratch);
  const resid = (kx: number, ky: number): number => {
    const i = ((ky + N) % N) * N + ((kx + N) % N);
    return L[i] - ringMed[ringOf[i]];
  };
  const out: Candidate[] = [];
  const half = N / 2;
  for (let ky = -half + 1; ky <= 0; ky++) {
    for (let kx = -half + 1; kx < half; kx++) {
      if (ky === 0 && kx <= 0) continue;
      const r = Math.hypot(kx, ky);
      if (r < MIN_R || r > half) continue;
      const i = ((ky + N) % N) * N + ((kx + N) % N);
      const e = L[i] - ringMed[ringOf[i]];
      if (e < minExcess / 10) continue;
      // local maximum of power in 5×5
      let isMax = true;
      const v0 = L[i];
      for (let dy = -2; dy <= 2 && isMax; dy++)
        for (let dx = -2; dx <= 2; dx++) {
          if (!dx && !dy) continue;
          const j = (((ky + dy) + N) % N) * N + (((kx + dx) + N) % N);
          if (L[j] > v0) {
            isMax = false;
            break;
          }
        }
      if (!isMax) continue;
      // isolated: well above all but the top few of the 120 bins 4–6 away. A
      // straight edge makes a ridge through the origin that is only a few bins
      // wide, so the 5th highest of those bins sits on the ridge and rejects it;
      // a neighbouring harmonic's lobe, weaker than this spike, does not
      let n = 0;
      for (let dy = -6; dy <= 6; dy++)
        for (let dx = -6; dx <= 6; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) < 4) continue;
          scratch[n++] = resid(kx + dx, ky + dy);
        }
      const ann = scratch.subarray(0, n);
      ann.sort();
      if (e - ann[n - 5] < isolation / 10) continue;
      // sub-bin position and height from a parabola through the neighbours
      const lx = resid(kx - 1, ky);
      const rx = resid(kx + 1, ky);
      const ly = resid(kx, ky - 1);
      const ry = resid(kx, ky + 1);
      const dxx = lx - 2 * e + rx;
      const dyy = ly - 2 * e + ry;
      const ox = dxx < -1e-9 ? Math.max(-0.5, Math.min(0.5, (0.5 * (lx - rx)) / dxx)) : 0;
      const oy = dyy < -1e-9 ? Math.max(-0.5, Math.min(0.5, (0.5 * (ly - ry)) / dyy)) : 0;
      const peakL = L[i] - 0.25 * (lx - rx) * ox - 0.25 * (ly - ry) * oy;
      const amplitude = (2 * Math.sqrt(Math.pow(10, peakL))) / hann.sum;
      out.push({ gx, gy, u: (kx + ox) / N, v: (ky + oy) / N, amplitude, excessDb: 10 * e });
    }
  }
  out.sort((a, b) => b.excessDb - a.excessDb);
  return out.slice(0, 8);
}

// --- tracking --------------------------------------------------------------------------

/** distance in bins between two frequencies, either sign of the twin */
function freqDist(a: { u: number; v: number }, b: { u: number; v: number }): number {
  const d1 = Math.hypot((a.u - b.u) * N, (a.v - b.v) * N);
  const d2 = Math.hypot((a.u + b.u) * N, (a.v + b.v) * N);
  return Math.min(d1, d2);
}

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

function summarise(nodes: DriftNode[], gridW: number, gridH: number): DriftTrack {
  // put every node on the same side of the origin as the median one, so
  // angles and drifts are measured without the twin's sign flips
  const ref = nodes[nodes.length >> 1];
  for (const n of nodes) {
    if (n.u * ref.u + n.v * ref.v < 0) {
      n.u = -n.u;
      n.v = -n.v;
    }
  }
  const periods = nodes.map((n) => 1 / Math.hypot(n.u, n.v)).sort((a, b) => a - b);
  const amps = nodes.map((n) => n.amplitude).sort((a, b) => a - b);
  const exc = nodes.map((n) => n.excessDb).sort((a, b) => a - b);
  const angles = nodes.map((n) => (Math.atan2(n.v, n.u) * 180) / Math.PI);
  const sa = [...angles].sort((a, b) => a - b);
  const angMed = percentile(sa, 0.5);
  let spread = 0;
  for (const a of angles) {
    let d = a - angMed;
    while (d > 90) d -= 180;
    while (d < -90) d += 180;
    spread = Math.max(spread, Math.abs(d));
  }
  const um = percentile(nodes.map((n) => n.u).sort((a, b) => a - b), 0.5);
  const vm = percentile(nodes.map((n) => n.v).sort((a, b) => a - b), 0.5);
  // fixed if nine nodes in ten sit within a bin of the median frequency (a
  // strong edge on the same axis can nudge the odd window's estimate)
  const drifts = nodes.map((n) => Math.hypot((n.u - um) * N, (n.v - vm) * N)).sort((a, b) => a - b);
  const drift = percentile(drifts, 0.9);
  return {
    nodes,
    gridW,
    gridH,
    period: percentile(periods, 0.5),
    periodMin: periods[0],
    periodMax: periods[periods.length - 1],
    angleDeg: ((angMed % 180) + 180) % 180,
    angleSpread: spread,
    amplitude: percentile(amps, 0.5),
    amplitudeP90: percentile(amps, 0.9),
    excessDb: percentile(exc, 0.5),
    coverage: nodes.length / (Math.max(1, gridW - 2) * Math.max(1, gridH - 2)),
    stationary: drift < 1,
  };
}

/**
 * Find the drifting patterns in a plane (normally the luma). Tracks come back
 * strongest and widest first.
 */
export function findDriftingPatterns(plane: Plane, opts: DriftOptions = {}, progress: (fraction: number) => void = () => {}): DriftTrack[] {
  const g = driftGeometry(plane.w, plane.h);
  const padded = padPlane(plane, g);
  const minExcess = opts.minExcessDb ?? 12;
  const isolation = opts.isolationDb ?? 6;
  const minNodes = opts.minNodes ?? Math.max(6, Math.round(0.04 * Math.max(1, g.gridW - 2) * Math.max(1, g.gridH - 2)));
  const re = new Float64Array(NN);
  const im = new Float64Array(NN);
  const P = new Float64Array(NN);
  const L = new Float64Array(NN);
  const ringMed = new Float64Array(N);
  const scratch = new Float64Array(NN);
  const perWindow: Candidate[][] = new Array(g.gridW * g.gridH);
  const all: Candidate[] = [];
  const start: number[] = new Array(g.gridW * g.gridH);
  // the border windows reach past the photo into its mirror image, which
  // shifts a pattern's spike and adds a mirrored one; they are left to the
  // fix's reach past the last node instead
  const border = (gx: number, gy: number): boolean => gx === 0 || gy === 0 || gx === g.gridW - 1 || gy === g.gridH - 1;
  for (let gy = 0; gy < g.gridH; gy++) {
    for (let gx = 0; gx < g.gridW; gx++) {
      const wi = gy * g.gridW + gx;
      if (border(gx, gy) && g.gridW > 2 && g.gridH > 2) {
        perWindow[wi] = [];
        start[wi] = all.length;
        continue;
      }
      windowSpectrum(padded, g.pw, gx * DRIFT_HOP, gy * DRIFT_HOP, re, im);
      const c = windowCandidates(re, im, gx, gy, minExcess, isolation, P, L, ringMed, scratch);
      perWindow[wi] = c;
      start[wi] = all.length;
      all.push(...c);
    }
    progress((gy + 1) / g.gridH);
  }
  // link neighbours: union–find over candidates
  const parent = Int32Array.from({ length: all.length }, (_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const union = (a: number, b: number): void => {
    a = find(a);
    b = find(b);
    if (a !== b) parent[a] = b;
  };
  for (let gy = 0; gy < g.gridH; gy++)
    for (let gx = 0; gx < g.gridW; gx++) {
      const wi = gy * g.gridW + gx;
      const here = perWindow[wi];
      for (const [nx, ny] of [
        [gx + 1, gy],
        [gx, gy + 1],
      ]) {
        if (nx >= g.gridW || ny >= g.gridH) continue;
        const wj = ny * g.gridW + nx;
        const there = perWindow[wj];
        for (let a = 0; a < here.length; a++) {
          let best = -1;
          let bestD = LINK_BINS;
          for (let b = 0; b < there.length; b++) {
            const d = freqDist(here[a], there[b]);
            if (d < bestD) {
              bestD = d;
              best = b;
            }
          }
          if (best >= 0) union(start[wi] + a, start[wj] + best);
        }
      }
    }
  const groups = new Map<number, Candidate[]>();
  for (let i = 0; i < all.length; i++) {
    const r = find(i);
    const l = groups.get(r);
    if (l) l.push(all[i]);
    else groups.set(r, [all[i]]);
  }
  const chains: DriftNode[][] = [];
  for (const members of groups.values()) {
    // one node per window: the stronger
    const byWindow = new Map<number, Candidate>();
    for (const c of members) {
      const k = c.gy * g.gridW + c.gx;
      const prev = byWindow.get(k);
      if (!prev || c.excessDb > prev.excessDb) byWindow.set(k, c);
    }
    if (byWindow.size < minNodes) continue;
    const nodes: DriftNode[] = [...byWindow.values()].map((c) => ({ gx: c.gx, gy: c.gy, u: c.u, v: c.v, amplitude: c.amplitude, excessDb: c.excessDb }));
    nodes.sort((a, b) => a.gy - b.gy || a.gx - b.gx);
    chains.push(nodes);
  }
  // a chain that is the same pattern as a bigger one, seen in a few windows
  // where something nudged its position, is folded into it
  chains.sort((a, b) => b.length - a.length);
  const merged: DriftNode[][] = [];
  for (const c of chains) {
    let into: DriftNode[] | null = null;
    for (const m of merged) {
      let close = 0;
      for (const n of c) if (m.some((q) => Math.abs(q.gx - n.gx) <= 2 && Math.abs(q.gy - n.gy) <= 2 && freqDist(q, n) <= LINK_BINS)) close++;
      if (close >= 0.8 * c.length) {
        into = m;
        break;
      }
    }
    if (into) {
      for (const n of c) if (!into.some((q) => q.gx === n.gx && q.gy === n.gy)) into.push(n);
      into.sort((a, b) => a.gy - b.gy || a.gx - b.gx);
    } else merged.push(c);
  }
  const tracks: DriftTrack[] = [];
  for (const nodes of merged) {
    const t = summarise(nodes, g.gridW, g.gridH);
    if (opts.minAmplitude !== undefined && t.amplitudeP90 < opts.minAmplitude) continue;
    tracks.push(t);
  }
  tracks.sort((a, b) => b.amplitude * b.coverage - a.amplitude * a.coverage);
  return tracks.slice(0, opts.maxTracks ?? 8);
}

// --- removal -----------------------------------------------------------------------------

const FILL_RADIUS = 2; // windows: how far past its last node a pattern is still looked for

/**
 * The tracked frequency for every window the fix should act in: the node's
 * own, or for windows up to FILL_RADIUS past the last node, the nodes'
 * distance-weighted average (the field is smooth, and the notch only removes
 * what stands above the background, so reaching a little further is safe).
 */
function frequencyField(track: DriftTrack): { fu: Float64Array; fv: Float64Array; use: Uint8Array } {
  const { gridW, gridH } = track;
  const fu = new Float64Array(gridW * gridH);
  const fv = new Float64Array(gridW * gridH);
  const use = new Uint8Array(gridW * gridH);
  for (const n of track.nodes) {
    const k = n.gy * gridW + n.gx;
    fu[k] = n.u;
    fv[k] = n.v;
    use[k] = 1;
  }
  const fill: [number, number, number][] = [];
  for (let gy = 0; gy < gridH; gy++)
    for (let gx = 0; gx < gridW; gx++) {
      const k = gy * gridW + gx;
      if (use[k]) continue;
      let su = 0;
      let sv = 0;
      let sw = 0;
      for (let dy = -FILL_RADIUS; dy <= FILL_RADIUS; dy++)
        for (let dx = -FILL_RADIUS; dx <= FILL_RADIUS; dx++) {
          const x = gx + dx;
          const y = gy + dy;
          if (x < 0 || y < 0 || x >= gridW || y >= gridH) continue;
          const j = y * gridW + x;
          if (use[j] !== 1) continue;
          const wgt = 1 / (dx * dx + dy * dy);
          su += wgt * fu[j];
          sv += wgt * fv[j];
          sw += wgt;
        }
      if (sw > 0) fill.push([k, su / sw, sv / sw]);
    }
  for (const [k, u, v] of fill) {
    fu[k] = u;
    fv[k] = v;
    use[k] = 2;
  }
  return { fu, fv, use };
}

/** Wiener-style notch of the band around (u, v) and its twin, applied in place to one channel's spectrum. Returns the removed part in dre/dim. */
function notchBand(re: Float64Array, im: Float64Array, u: number, v: number, dre: Float64Array, dim: Float64Array, P: Float64Array, scratch: Float64Array): void {
  const { ringOf, rings } = tables();
  dre.fill(0);
  dim.fill(0);
  for (let i = 0; i < NN; i++) P[i] = re[i] * re[i] + im[i] * im[i];
  const cx = u * N;
  const cy = v * N;
  // this window's background at the band's radius: the ring's median, the
  // band itself and its twin left out
  const r0 = Math.round(Math.hypot(cx, cy));
  let n = 0;
  for (let dr = -1; dr <= 1; dr++) {
    const r = r0 + dr;
    if (r < 0 || r >= rings.length) continue;
    for (const i of rings[r]) {
      const kx = sbin(i % N);
      const ky = sbin((i - (i % N)) / N);
      if (Math.hypot(kx - cx, ky - cy) < BAND_R + 1.5 || Math.hypot(kx + cx, ky + cy) < BAND_R + 1.5) continue;
      scratch[n++] = P[i];
    }
  }
  const bg = n ? medianOf(scratch, n) : 0;
  const take = (sx: number, sy: number, skipNear: [number, number] | null): void => {
    const x0 = Math.round(sx);
    const y0 = Math.round(sy);
    const rr = Math.ceil(BAND_R);
    for (let ky = y0 - rr; ky <= y0 + rr; ky++)
      for (let kx = x0 - rr; kx <= x0 + rr; kx++) {
        if (Math.hypot(kx - sx, ky - sy) > BAND_R) continue;
        if (skipNear && Math.hypot(kx - skipNear[0], ky - skipNear[1]) <= BAND_R) continue;
        const i = ((ky + N) % N) * N + ((kx + N) % N);
        const p = P[i];
        if (p <= bg) continue;
        const g = 1 - bg / p;
        dre[i] = re[i] * g;
        dim[i] = im[i] * g;
      }
  };
  take(cx, cy, null);
  take(-cx, -cy, [cx, cy]);
  void ringOf;
}

/**
 * The change that removes one or more tracked patterns from each plane (all
 * three channels, so coloured moiré goes too), as a delta to add. Two real
 * planes are transformed together as one complex signal; several patterns
 * share each window's transform.
 */
export function driftDelta(planes: Plane[], trackOrTracks: DriftTrack | DriftTrack[], progress: (fraction: number) => void = () => {}): Plane[] {
  const tracks = Array.isArray(trackOrTracks) ? trackOrTracks : [trackOrTracks];
  const w = planes[0].w;
  const h = planes[0].h;
  const g = driftGeometry(w, h);
  for (const t of tracks) if (g.gridW !== t.gridW || g.gridH !== t.gridH) throw new Error('drift track does not match the plane');
  const fields = tracks.map(frequencyField);
  const padded = planes.map((p) => padPlane(p, g));
  const acc = planes.map(() => new Float64Array(g.pw * g.ph));
  const norm = new Float64Array(g.pw * g.ph);
  const re = new Float64Array(NN);
  const im = new Float64Array(NN);
  const ar = new Float64Array(NN);
  const ai = new Float64Array(NN);
  const br = new Float64Array(NN);
  const bi = new Float64Array(NN);
  const dar = new Float64Array(NN);
  const dai = new Float64Array(NN);
  const dbr = new Float64Array(NN);
  const dbi = new Float64Array(NN);
  const P = new Float64Array(NN);
  const scratch = new Float64Array(NN);
  const pairs: [number, number | null][] = [];
  for (let c = 0; c < planes.length; c += 2) pairs.push([c, c + 1 < planes.length ? c + 1 : null]);
  let done = 0;
  const total = g.gridW * g.gridH;
  for (let gy = 0; gy < g.gridH; gy++) {
    for (let gx = 0; gx < g.gridW; gx++, done++) {
      const k = gy * g.gridW + gx;
      const x0 = gx * DRIFT_HOP;
      const y0 = gy * DRIFT_HOP;
      // the normalisation covers every window, so the fix fades out smoothly
      // at the edge of the pattern's reach
      for (let y = 0; y < N; y++) {
        const row = (y0 + y) * g.pw + x0;
        for (let x = 0; x < N; x++) norm[row + x] += hann.data[y * N + x] * hann.data[y * N + x];
      }
      const active = fields.filter((f) => f.use[k]);
      if (!active.length) continue;
      for (const [ca, cb] of pairs) {
        // pack two channels into one complex transform
        let ma = 0;
        let mb = 0;
        for (let y = 0; y < N; y++) {
          const row = (y0 + y) * g.pw + x0;
          for (let x = 0; x < N; x++) {
            ma += padded[ca][row + x];
            if (cb !== null) mb += padded[cb][row + x];
          }
        }
        ma /= NN;
        mb /= NN;
        for (let y = 0; y < N; y++) {
          const row = (y0 + y) * g.pw + x0;
          for (let x = 0; x < N; x++) {
            const i = y * N + x;
            re[i] = (padded[ca][row + x] - ma) * hann.data[i];
            im[i] = cb !== null ? (padded[cb][row + x] - mb) * hann.data[i] : 0;
          }
        }
        fft2d(re, im, N, N, false);
        // unpack: A(k) = (X(k) + X*(−k)) / 2, B(k) = (X(k) − X*(−k)) / 2i
        for (let ky = 0; ky < N; ky++) {
          const my = (N - ky) % N;
          for (let kx = 0; kx < N; kx++) {
            const i = ky * N + kx;
            const j = my * N + ((N - kx) % N);
            ar[i] = 0.5 * (re[i] + re[j]);
            ai[i] = 0.5 * (im[i] - im[j]);
            br[i] = 0.5 * (im[i] + im[j]);
            bi[i] = -0.5 * (re[i] - re[j]);
          }
        }
        // each pattern's band is taken from what the earlier ones left
        re.fill(0);
        im.fill(0);
        for (const f of active) {
          const u = f.fu[k];
          const v = f.fv[k];
          notchBand(ar, ai, u, v, dar, dai, P, scratch);
          if (cb !== null) notchBand(br, bi, u, v, dbr, dbi, P, scratch);
          else {
            dbr.fill(0);
            dbi.fill(0);
          }
          // repack the removed parts (re → channel a, im → channel b) and leave the rest
          for (let i = 0; i < NN; i++) {
            re[i] += dar[i] - dbi[i];
            im[i] += dai[i] + dbr[i];
            ar[i] -= dar[i];
            ai[i] -= dai[i];
            br[i] -= dbr[i];
            bi[i] -= dbi[i];
          }
        }
        fft2d(re, im, N, N, true);
        for (let y = 0; y < N; y++) {
          const row = (y0 + y) * g.pw + x0;
          for (let x = 0; x < N; x++) {
            const i = y * N + x;
            acc[ca][row + x] += hann.data[i] * re[i];
            if (cb !== null) acc[cb][row + x] += hann.data[i] * im[i];
          }
        }
      }
    }
    progress(done / total);
  }
  return planes.map((p, c) => {
    const out = makePlane(w, h);
    for (let y = 0; y < h; y++) {
      const row = (y + g.padTop) * g.pw + g.padLeft;
      for (let x = 0; x < w; x++) {
        const n = norm[row + x];
        out.data[y * w + x] = n > 1e-3 ? -acc[c][row + x] / n : 0;
      }
    }
    return out;
  });
}

// --- measuring (for the check) ---------------------------------------------------------

/**
 * How strong the tracked pattern is in a plane: the amplitude at each node's
 * frequency in that node's window, median and 90th percentile, in plane
 * units. `offset` places the plane within the photo the track was found on
 * (a crop); nodes whose window falls outside are skipped.
 */
export function measureDrift(plane: Plane, track: DriftTrack, offset = { x0: 0, y0: 0 }): { amplitude: number; amplitudeP90: number; count: number } {
  const re = new Float64Array(NN);
  const im = new Float64Array(NN);
  const amps: number[] = [];
  for (const n of track.nodes) {
    const x0 = n.gx * DRIFT_HOP - N / 2 - offset.x0;
    const y0 = n.gy * DRIFT_HOP - N / 2 - offset.y0;
    if (x0 < 0 || y0 < 0 || x0 + N > plane.w || y0 + N > plane.h) continue;
    let mean = 0;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) mean += plane.data[(y0 + y) * plane.w + x0 + x];
    mean /= NN;
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) re[y * N + x] = (plane.data[(y0 + y) * plane.w + x0 + x] - mean) * hann.data[y * N + x];
    im.fill(0);
    fft2d(re, im, N, N, false);
    // the strongest bin within 2 of the node's frequency, parabola-refined
    const cx = Math.round(n.u * N);
    const cy = Math.round(n.v * N);
    let best = -1;
    let bx = cx;
    let by = cy;
    for (let ky = cy - 2; ky <= cy + 2; ky++)
      for (let kx = cx - 2; kx <= cx + 2; kx++) {
        const i = ((ky + N) % N) * N + ((kx + N) % N);
        const p = re[i] * re[i] + im[i] * im[i];
        if (p > best) {
          best = p;
          bx = kx;
          by = ky;
        }
      }
    const lp = (kx: number, ky: number): number => {
      const i = ((ky + N) % N) * N + ((kx + N) % N);
      return Math.log10(re[i] * re[i] + im[i] * im[i] + 1e-30);
    };
    const c0 = lp(bx, by);
    const lx = lp(bx - 1, by);
    const rx = lp(bx + 1, by);
    const ly = lp(bx, by - 1);
    const ry = lp(bx, by + 1);
    const dxx = lx - 2 * c0 + rx;
    const dyy = ly - 2 * c0 + ry;
    const ox = dxx < -1e-9 ? Math.max(-0.5, Math.min(0.5, (0.5 * (lx - rx)) / dxx)) : 0;
    const oy = dyy < -1e-9 ? Math.max(-0.5, Math.min(0.5, (0.5 * (ly - ry)) / dyy)) : 0;
    const peak = c0 - 0.25 * (lx - rx) * ox - 0.25 * (ly - ry) * oy;
    amps.push((2 * Math.sqrt(Math.pow(10, peak))) / hann.sum);
  }
  amps.sort((a, b) => a - b);
  return { amplitude: percentile(amps, 0.5), amplitudeP90: percentile(amps, 0.9), count: amps.length };
}

export function describeDrift(t: DriftTrack): { period: string; angle: string } {
  const p = t.periodMax - t.periodMin > 0.15 * t.period ? `${t.periodMin.toFixed(1)}–${t.periodMax.toFixed(1)} px` : `${t.period.toFixed(1)} px`;
  const a = t.angleSpread > 2 ? `${Math.round(t.angleDeg)}° ± ${Math.round(t.angleSpread)}°` : `${Math.round(t.angleDeg)}°`;
  return { period: p, angle: a };
}
