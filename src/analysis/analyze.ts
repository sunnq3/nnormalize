/**
 * The analysis pass: from planar RGB to a list of findings.
 */

import { backgroundModel } from '../dsp/background';
import { detectBlockGrid } from '../dsp/blocks';
import { findHotPixels } from '../dsp/hotpixels';
import { noiseDiscrepancy, noiseFromSpectrum, noiseMap, noiseModel } from '../dsp/noise';
import { parametricDeltaGlobal, parametricDeltaTiled } from '../dsp/filter';
import { describeFrequency, measurePattern } from '../dsp/pattern';
import { detectPeaks, groupHarmonics, type RawPeak } from '../dsp/peaks';
import { lumaPlane, type Plane } from '../dsp/plane';
import { tileGrid, welchSpectrum, type TileGrid } from '../dsp/tiles';
import { window2d } from '../dsp/window';
import {
  NOT_CHECKED,
  defaultAnalysisOptions,
  type Analysis,
  type AnalysisOptions,
  type Finding,
  type HotPixelFinding,
  type JpegGridFinding,
  type NoiseFinding,
  type PeriodicFinding,
} from './types';

export interface AnalyzeInput {
  planes: [Plane, Plane, Plane];
  dpi: number | null;
  sourceQuality: number | null;
  sourceSubsampling: string | null;
}

export type Progress = (stage: string, fraction: number) => void;

/** sRGB electro-optical transfer: gamma-encoded 0..1 to linear 0..1. */
export function srgbToLinear(p: Plane): Plane {
  const out = new Float32Array(p.data.length);
  for (let i = 0; i < out.length; i++) {
    const c = p.data[i];
    out[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  return { w: p.w, h: p.h, data: out };
}

/** Evenly spaced subset of a grid's tile origins so the Welch pass stays bounded. */
export function subsampleGrid(grid: TileGrid, maxTiles: number): TileGrid {
  const total = grid.xs.length * grid.ys.length;
  if (total <= maxTiles) return grid;
  const ratio = Math.sqrt(maxTiles / total);
  const pick = (arr: number[]): number[] => {
    const k = Math.max(1, Math.round(arr.length * ratio));
    if (k >= arr.length) return arr;
    const out: number[] = [];
    for (let i = 0; i < k; i++) out.push(arr[Math.round((i * (arr.length - 1)) / Math.max(1, k - 1))]);
    return Array.from(new Set(out));
  };
  return { size: grid.size, hop: grid.hop, xs: pick(grid.xs), ys: pick(grid.ys) };
}

const n0 = (n: number): number => n;

export function analyze(input: AnalyzeInput, optsIn: Partial<AnalysisOptions> = {}, progress: Progress = () => {}): Analysis {
  const opts: AnalysisOptions = { ...defaultAnalysisOptions, ...optsIn };
  const timings: Record<string, number> = {};
  const t0 = performance.now();
  const mark = (k: string, from: number): number => {
    const now = performance.now();
    timings[k] = Math.round(now - from);
    return now;
  };

  const [r, g, b] = input.planes;
  const w = r.w;
  const h = r.h;
  let luma = lumaPlane(r, g, b);
  if (opts.linearLight) luma = srgbToLinear(luma);
  let t = mark('luma', t0);

  // --- spectrum -----------------------------------------------------------
  progress('spectrum', 0);
  const grid = tileGrid(w, h, opts.tileSize);
  const welchGrid = subsampleGrid(grid, opts.maxWelchTiles);
  const welch = welchSpectrum(luma, welchGrid, 'hann', (d, n) => progress('spectrum', d / n));
  const model = backgroundModel(welch.power, welch.size);
  t = mark('welch', t);

  // --- peaks ----------------------------------------------------------------
  progress('peaks', 0);
  const rawAll = detectPeaks(model, { zThreshold: opts.zThreshold, minRadiusBins: opts.minRadiusBins, maxPeaks: opts.maxPeaks });
  // Welch power of a sinusoid is (A/2)² (within ~1.4 dB off-bin), so the
  // amplitude is known before any expensive measurement. Peaks below half the
  // floor are invisible and dropped outright.
  const floor = opts.minAmplitudeLevels / 255;
  const raw = rawAll.filter((p) => 2 * Math.sqrt(welch.power[p.by * n0(welch.size) + p.bx]) >= 0.5 * floor);
  const groups = groupHarmonics(raw, welch.size);
  const findings: Finding[] = [];
  const n = welch.size;
  const binTol = 1.2 / n;
  const quick = opts.quick;
  const measureOpts = quick ? { searchGrid: welchGrid, measureGrid: welchGrid } : { searchGrid: welchGrid };
  let gi = 0;
  for (const grp of groups) {
    const fundamental: RawPeak = raw[grp.fundamental];
    const m = measurePattern(luma, grid, fundamental.u, fundamental.v, measureOpts);
    const desc = describeFrequency(m.u, m.v);
    const onAxis = Math.abs(m.u) < binTol || Math.abs(m.v) < binTol;
    const near8 = (f: number): boolean => Math.abs(f * 8 - Math.round(f * 8)) < binTol * 8;
    const gridAligned = near8(m.u) && near8(m.v); // confirmed against the block grid below
    const harmonics = grp.members
      .map((idx, k) => ({ idx, order: grp.orders[k] }))
      .filter((x) => x.order > 1)
      .map(({ idx, order }) => {
        const hp = raw[idx];
        // harmonics only need the refined frequency and a global amplitude
        const hm = measurePattern(luma, grid, hp.u, hp.v, { refine: true, searchGrid: welchGrid, measureGrid: welchGrid });
        return { order, u: hm.u, v: hm.v, amplitude: hm.amplitude, excessDb: hp.excessDb, widthBins: hp.widthBins };
      });
    let category: PeriodicFinding['details']['category'];
    let label: string;
    if (false) {
      category = 'jpeg-harmonic';
      label = '';
    } else if (onAxis && Math.abs(m.u) < binTol) {
      category = 'horizontal-lines';
      label = 'Horizontal lines';
    } else if (onAxis) {
      category = 'vertical-lines';
      label = 'Vertical lines';
    } else if (desc.radius > 0.3) {
      category = 'screen';
      label = 'Fine screen';
    } else {
      category = 'moire';
      label = 'Moiré / oblique pattern';
    }
    const severity = Math.max(0, Math.min(1, (fundamental.excessDb - 6) / 40)) * Math.min(1, m.amplitude / (4 * floor));
    // a pattern confined to part of the frame is judged by what it reaches there
    const visible = Math.max(m.amplitude, m.amplitudeP90) >= floor;
    const strong = fundamental.excessDb >= 12 && fundamental.z >= 8 && visible;
    const confidence = fundamental.z > 15 && m.phaseCoherence > 0.6 ? 'high' : fundamental.z >= 8 ? 'medium' : 'low';
    let note = '';
    if (!visible) note = `Only ${(m.amplitudeP90 * 255).toFixed(2)} levels at most: below the ${opts.minAmplitudeLevels} level floor, not visible. Left unticked.`;
    else if (!strong) note = 'Weak; could be real texture (fabric, screen, brick). Left unticked.';
    else if (m.phaseCoherence < 0.4 && !m.stationary) note = 'Varies across the frame; tiled subtraction will follow it.';

    let extent = 0;
    for (const p of m.presence) if (p > 0.25) extent++;
    extent /= m.presence.length;
    const lpi = input.dpi ? desc.radius * input.dpi : null;
    const f: PeriodicFinding = {
      id: `periodic-${gi++}`,
      kind: 'periodic',
      label,
      description: `${desc.period.toFixed(2)} px period at ${desc.angleDeg.toFixed(1)}°, ${(m.amplitude * 255).toFixed(2)} levels peak, ${fundamental.excessDb.toFixed(1)} dB over background${
        harmonics.length ? `, ${harmonics.length} harmonic${harmonics.length > 1 ? 's' : ''}` : ''
      }`,
      severity,
      extent,
      enabled: strong,
      confidence,
      note,
      strength: 1,
      method: m.stationary ? 'parametric' : 'parametric-tiled',
      details: {
        u: m.u,
        v: m.v,
        period: desc.period,
        angleDeg: desc.angleDeg,
        lpi,
        amplitude: m.amplitude,
        amplitudeP90: m.amplitudeP90,
        phase: m.phase,
        excessDb: fundamental.excessDb,
        z: fundamental.z,
        widthBins: fundamental.widthBins,
        stationary: m.stationary,
        coherence: m.phaseCoherence,
        amplitudeCv: m.amplitudeCv,
        harmonics,
        presence: m.presence,
        tilesX: m.tilesX,
        tilesY: m.tilesY,
        onAxis,
        jpegGridAligned: gridAligned,
        category,
      },
    };
    findings.push(f);
    progress('peaks', gi / groups.length);
  }
  // strongest first
  findings.sort((a, b) => b.severity - a.severity);
  t = mark('peaks', t);

  // --- pattern-free luma --------------------------------------------------------
  // Everything after this point (block grid, noise, hot pixels) is measured on
  // the luma with the strong periodic patterns subtracted, otherwise a moiré's
  // own pixel-to-pixel swings masquerade as block steps, noise and outliers.
  const cleanLuma: Plane = { w, h, data: new Float32Array(luma.data) };
  for (const f of findings) {
    if (f.kind !== 'periodic' || !f.enabled) continue;
    const orders = [{ u: f.details.u, v: f.details.v }, ...f.details.harmonics.map((hm) => ({ u: hm.u, v: hm.v }))];
    for (const o of orders) {
      const stationary = f.details.stationary;
      const m = measurePattern(luma, grid, o.u, o.v, { refine: false, searchGrid: welchGrid, measureGrid: stationary || quick ? welchGrid : grid });
      const d = stationary || quick ? parametricDeltaGlobal(w, h, o.u, o.v, m.amplitude, m.phase) : parametricDeltaTiled(w, h, grid, m);
      for (let i = 0; i < cleanLuma.data.length; i++) cleanLuma.data[i] += d.data[i];
    }
  }
  t = mark('cleanLuma', t);

  // --- block grid ------------------------------------------------------------
  const blockGrid = detectBlockGrid(cleanLuma);
  for (const f of findings) {
    if (f.kind !== 'periodic' || !f.details.jpegGridAligned) continue;
    // a k/8 peak only counts as compression damage when a block grid is really
    // there and the peak is modest; a strong 4-px scan-line pattern also sits
    // on 2/8 and is interference in its own right
    if (!blockGrid.detected || f.details.excessDb >= 25) {
      f.details.jpegGridAligned = false;
      continue;
    }
    f.details.category = 'jpeg-harmonic';
    f.label = 'JPEG block-grid harmonic';
    f.enabled = false;
    f.note = 'Aligned with the 8×8 JPEG grid: compression damage, handled by the deblocker rather than a notch.';
  }
  t = mark('blockGrid', t);

  // --- JPEG grid finding ----------------------------------------------------
  const gridDetails = {
    offsetX: blockGrid.offsetX,
    offsetY: blockGrid.offsetY,
    strengthX: blockGrid.strengthX,
    strengthY: blockGrid.strengthY,
    sourceQuality: input.sourceQuality,
    sourceSubsampling: input.sourceSubsampling,
  };
  if (blockGrid.detected) {
    const s = Math.max(blockGrid.strengthX, blockGrid.strengthY);
    const jf: JpegGridFinding = {
      id: 'jpeg-grid',
      kind: 'jpeg-grid',
      label: 'JPEG block edges',
      description: `8×8 grid at offset (${blockGrid.offsetX}, ${blockGrid.offsetY}); boundary steps ${(s * 100).toFixed(0)} % above texture${
        input.sourceQuality ? `; source quality ≈ ${input.sourceQuality.toFixed(0)}` : ''
      }${input.sourceSubsampling ? `, ${input.sourceSubsampling}` : ''}`,
      severity: Math.min(1, s / 0.5),
      extent: 1,
      enabled: s >= 0.05,
      confidence: Math.min(blockGrid.confidenceX, blockGrid.confidenceY) > 8 ? 'high' : 'medium',
      note: s < 0.05 ? 'Faint grid; deblocking left off by default.' : '',
      strength: 1,
      method: 'deblock',
      details: gridDetails,
    };
    findings.push(jf);
  }
  t = mark('jpegGrid', t);

  // --- noise ----------------------------------------------------------------
  progress('noise', 0);
  const nmap = noiseMap(cleanLuma, opts.noiseBlock);
  const nmodel = noiseModel(nmap);
  const disc = noiseDiscrepancy(nmap, nmodel, opts.noiseThresholdDb);
  const win = window2d('hann', n, n);
  const spectralSigma = noiseFromSpectrum(welch.power, n, win.sum, win.sumSq);
  // spectral tilt of the floor: mean log power in r∈[0.35,0.5] minus r∈[0.15,0.25]
  let hi = 0;
  let nh = 0;
  let mid = 0;
  let nm = 0;
  const c = n >> 1;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const rr = Math.hypot(x - c, y - c) / n;
      const v = model.logPower[y * n + x];
      if (rr >= 0.35 && rr <= 0.5) {
        hi += v;
        nh++;
      } else if (rr >= 0.15 && rr <= 0.25) {
        mid += v;
        nm++;
      }
    }
  }
  const highBandDb = nh && nm ? 10 * (hi / nh - mid / nm) : 0;
  const noiseDetails = {
    block: nmap.block,
    bw: nmap.bw,
    bh: nmap.bh,
    ratioDb: disc.ratioDb,
    noisierFraction: disc.noisierFraction,
    smootherFraction: disc.smootherFraction,
    thresholdDb: disc.thresholdDb,
    globalSigma: nmodel.globalSigma,
    curve: nmodel.curve,
    highBandDb,
    spectralSigma,
  };
  {
    const frac = disc.noisierFraction + disc.smootherFraction;
    const nf: NoiseFinding = {
      id: 'noise',
      kind: 'noise',
      label: 'Uneven noise',
      description: `baseline ${(nmodel.globalSigma * 255).toFixed(2)} levels; ${(disc.noisierFraction * 100).toFixed(0)} % of the frame noisier and ${(
        disc.smootherFraction * 100
      ).toFixed(0)} % smoother than the model by more than ${disc.thresholdDb} dB`,
      severity: Math.min(1, frac / 0.5),
      extent: frac,
      enabled: frac >= 0.05,
      confidence: frac >= 0.15 ? 'high' : frac >= 0.05 ? 'medium' : 'low',
      note: frac < 0.05 ? 'Noise is already level; nothing to normalise.' : '',
      strength: 1,
      method: 'normalise',
      details: noiseDetails,
    };
    findings.push(nf);
  }
  t = mark('noise', t);

  // --- hot pixels ------------------------------------------------------------
  progress('hotpixels', 0);
  const hot = quick ? { pixels: [], delta: cleanLuma } : findHotPixels(cleanLuma, nmodel);
  if (hot.pixels.length) {
    const hf: HotPixelFinding = {
      id: 'hotpixels',
      kind: 'hotpixels',
      label: 'Hot / dead pixels',
      description: `${hot.pixels.length} isolated single-pixel outlier${hot.pixels.length > 1 ? 's' : ''}`,
      severity: Math.min(1, hot.pixels.length / 50),
      extent: hot.pixels.length / (w * h),
      enabled: true,
      confidence: 'high',
      note: '',
      strength: 1,
      method: 'inpaint',
      details: {
        count: hot.pixels.length,
        sample: hot.pixels.slice(0, 500).map((p) => ({ x: p.x, y: p.y, value: p.value, replacement: p.replacement, z: p.zScore })),
        indices: Uint32Array.from(hot.pixels, (p) => p.y * w + p.x),
      },
    };
    findings.push(hf);
  }
  mark('hotpixels', t);
  timings.total = Math.round(performance.now() - t0);

  return {
    w,
    h,
    options: opts,
    tile: { size: grid.size, hop: grid.hop, xs: grid.xs, ys: grid.ys },
    spectrum: {
      size: n,
      logPower: Float32Array.from(model.logPower),
      background: Float32Array.from(model.background),
      z: Float32Array.from(model.z),
      tiles: welch.tiles,
    },
    findings,
    blockGrid: { ...gridDetails, detected: blockGrid.detected },
    noise: noiseDetails,
    timings,
    notChecked: NOT_CHECKED,
  };
}
