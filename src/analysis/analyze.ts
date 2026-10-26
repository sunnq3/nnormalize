/**
 * The analysis pass: from planar RGB to a list of findings.
 */

import { subsampleGrid } from '../dsp/tiles';
import { bandingStrength, detectBanding } from '../dsp/banding';
import { detectCfaPhase, measureFalseColour } from '../dsp/cfa';
import { measureFlatness, toneDelta, toneParams } from '../dsp/tone';
import { glowFix } from '../dsp/glow';
import { describeDrift, driftDelta, findDriftingPatterns, type DriftTrack } from '../dsp/drift';
import { measureTexture, textureDelta } from '../dsp/texture';
import { fitCameraModel, measureEdgeProfile, reachableSigma, targetSigma } from '../dsp/edges';
import { backgroundModel } from '../dsp/background';
import { detectBlockGrid } from '../dsp/blocks';
import { measureChromaNoise, measureRinging, toYCbCr } from '../dsp/chroma';
import { findHotPixels } from '../dsp/hotpixels';
import { noiseDiscrepancy, noiseFromSpectrum, noiseMap, noiseModel } from '../dsp/noise';
import { parametricDeltaGlobal, parametricDeltaTiled } from '../dsp/filter';
import { describeFrequency, measurePattern } from '../dsp/pattern';
import { detectPeaks, groupHarmonics, type RawPeak } from '../dsp/peaks';
import { lumaPlane, type Plane } from '../dsp/plane';
import { tileGrid, welchSpectrum, type TileGrid } from '../dsp/tiles';
import { window2d } from '../dsp/window';
import {
  LEVELS,
  NOT_CHECKED,
  defaultAnalysisOptions,
  type Analysis,
  type AnalysisOptions,
  type ChromaFinding,
  type Finding,
  type RingingFinding,
  type EdgesFinding,
  type FringesFinding,
  type ToneFinding,
  type GlowFinding,
  type TextureFinding,
  type BandingFinding,
  type HotPixelFinding,
  type JpegGridFinding,
  type NoiseFinding,
  type PeriodicFinding,
  type DriftFinding,
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
export { subsampleGrid } from '../dsp/tiles';

const n0 = (n: number): number => n;

export function analyze(input: AnalyzeInput, optsIn: Partial<AnalysisOptions> = {}, progress: Progress = () => {}): Analysis {
  // the clean-up level sets the thresholds; explicit options still win
  const level = LEVELS[optsIn.level ?? defaultAnalysisOptions.level];
  const opts: AnalysisOptions = {
    ...defaultAnalysisOptions,
    zThreshold: level.zThreshold,
    minAmplitudeLevels: level.minAmplitudeLevels,
    noiseThresholdDb: level.noiseThresholdDb,
    ...optsIn,
  };
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
    // the strong levels also tick weak peaks, but only ones a person could see
    const seeable = Math.max(m.amplitude, m.amplitudeP90) * 255 >= Math.max(1, opts.minAmplitudeLevels);
    const strong = (fundamental.excessDb >= 12 && fundamental.z >= 8 && seeable) || (level.enableWeakPeaks && seeable);
    const confidence = fundamental.z > 15 && m.phaseCoherence > 0.6 ? 'high' : fundamental.z >= 8 ? 'medium' : 'low';
    let note = '';
    if (!visible) note = `Only ${(m.amplitudeP90 * 255).toFixed(2)} levels at most: below the ${opts.minAmplitudeLevels} level floor, not visible. Left unticked.`;
    else if (!seeable) note = `Only ${(Math.max(m.amplitude, m.amplitudeP90) * 255).toFixed(2)} levels: real but too faint to see. Left unticked; tick it if you want it gone anyway.`;
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

  // --- patterns that drift across the frame ----------------------------------------
  // A screen photographed at an angle, a scanned print, a fabric beating
  // against the sensor: clean stripes in any small patch, but their spacing
  // and direction change across the frame, so the averaged spectrum above
  // never sees them. Found window by window instead.
  progress('drift', 0);
  const driftTracks: DriftTrack[] = [];
  {
    const tracks = findDriftingPatterns(luma, { minAmplitude: 0.5 * floor }, (fr) => progress('drift', fr));
    let di = 0;
    for (const tr of tracks) {
      const near8 = (f: number): boolean => Math.abs(f * 8 - Math.round(f * 8)) * 32 < 0.5;
      const um = tr.nodes[tr.nodes.length >> 1].u;
      const vm = tr.nodes[tr.nodes.length >> 1].v;
      // a fixed pattern on the 8-px grid of modest strength is compression damage, the deblocker's job
      if (tr.stationary && near8(um) && near8(vm) && tr.excessDb < 25) continue;
      // a fixed pattern the averaged spectrum already found is reported once, there
      if (tr.stationary && findings.some((f) => f.kind === 'periodic' && Math.hypot((f.details.u - um) * 256, (f.details.v - vm) * 256) < 2)) continue;
      const amp = tr.amplitude * 255;
      const p90 = tr.amplitudeP90 * 255;
      const seeable = Math.max(amp, p90) >= Math.max(1, opts.minAmplitudeLevels);
      const visible = Math.max(amp, p90) >= opts.minAmplitudeLevels;
      const enabled = (seeable && tr.excessDb >= 15) || (level.enableWeakPeaks && visible);
      const category: DriftFinding['details']['category'] = tr.period < 3 ? 'fine-grid' : tr.stationary ? 'drifting' : 'screen-moire';
      const d = describeDrift(tr);
      const label = category === 'fine-grid' ? 'Fine screen grid' : category === 'screen-moire' ? 'Drifting stripes (screen moiré)' : 'Faint repeating pattern';
      const df: DriftFinding = {
        id: `drift-${di++}`,
        kind: 'drift',
        label,
        description: `stripes ${d.period} apart at ${d.angle}, ${amp.toFixed(1)} levels (up to ${p90.toFixed(1)}), over ${(tr.coverage * 100).toFixed(0)} % of the photo${tr.stationary ? '' : '; spacing and direction drift across the frame'}`,
        severity: Math.min(1, p90 / 8) * Math.min(1, tr.coverage / 0.3),
        extent: tr.coverage,
        enabled,
        confidence: tr.excessDb >= 20 && tr.nodes.length >= 12 ? 'high' : tr.excessDb >= 15 ? 'medium' : 'low',
        note: !visible ? `Only ${p90.toFixed(2)} levels at most: below the ${opts.minAmplitudeLevels} level floor, not visible. Left unticked.` : !seeable ? `Only ${Math.max(amp, p90).toFixed(2)} levels: real but too faint to see. Left unticked; tick it if you want it gone anyway.` : !enabled ? 'Weak; could be real texture. Left unticked.' : '',
        strength: 1,
        method: 'windowed-notch',
        details: {
          nodes: tr.nodes,
          gridW: tr.gridW,
          gridH: tr.gridH,
          period: tr.period,
          periodMin: tr.periodMin,
          periodMax: tr.periodMax,
          angleDeg: tr.angleDeg,
          angleSpread: tr.angleSpread,
          amplitude: amp,
          amplitudeP90: p90,
          excessDb: tr.excessDb,
          coverage: tr.coverage,
          stationary: tr.stationary,
          category,
          amount: 1,
        },
      };
      findings.push(df);
      if (enabled) driftTracks.push(tr);
    }
  }
  t = mark('drift', t);

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
  if (driftTracks.length) {
    const d = driftDelta([cleanLuma], driftTracks)[0];
    for (let i = 0; i < cleanLuma.data.length; i++) cleanLuma.data[i] += d.data[i];
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
    stepLevels: Math.max(blockGrid.stepX, blockGrid.stepY) * 255,
    sourceQuality: input.sourceQuality,
    sourceSubsampling: input.sourceSubsampling,
    maxStep: level.deblockMaxStep,
    activityRatio: level.deblockActivityRatio,
    span: level.deblockSpan,
  };
  if (blockGrid.detected) {
    const s = Math.max(blockGrid.strengthX, blockGrid.strengthY);
    const stepLv = Math.max(blockGrid.stepX, blockGrid.stepY) * 255;
    const jf: JpegGridFinding = {
      id: 'jpeg-grid',
      kind: 'jpeg-grid',
      label: 'JPEG block edges',
      description: `8×8 grid at offset (${blockGrid.offsetX}, ${blockGrid.offsetY}); boundary steps ${(s * 100).toFixed(0)} % above texture${
        input.sourceQuality ? `; source quality ≈ ${input.sourceQuality.toFixed(0)}` : ''
      }${input.sourceSubsampling ? `, ${input.sourceSubsampling}` : ''}`,
      severity: Math.min(1, stepLv / 4),
      extent: 1,
      enabled: (s >= 0.05 && stepLv >= 0.5) || (level.deblockAlwaysOn && stepLv >= 0.25),
      confidence: Math.min(blockGrid.confidenceX, blockGrid.confidenceY) > 8 ? 'high' : 'medium',
      note: stepLv < 0.5 ? `Block steps of only ${stepLv.toFixed(2)} levels: not visible.` : '',
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
    reduction: level.noiseReduction,
  };
  {
    const frac = disc.noisierFraction + disc.smootherFraction;
    const nf: NoiseFinding = {
      id: 'noise',
      kind: 'noise',
      label: level.noiseReduction > 0 ? 'Grain and noise' : 'Uneven grain',
      description:
        level.noiseReduction > 0
          ? `grain of ${(nmodel.globalSigma * 255).toFixed(1)} levels; reduce it by ${Math.round(level.noiseReduction * 100)} % everywhere`
          : `grain of ${(nmodel.globalSigma * 255).toFixed(1)} levels; ${(disc.noisierFraction * 100).toFixed(0)} % of the photo grainier and ${(
              disc.smootherFraction * 100
            ).toFixed(0)} % smoother than the rest`,
      severity: level.noiseReduction > 0 ? Math.min(1, (nmodel.globalSigma * 255) / 6) : Math.min(1, frac / 0.5),
      extent: level.noiseReduction > 0 ? 1 : frac,
      enabled: frac >= 0.05 || level.noiseAlwaysOn,
      confidence: frac >= 0.15 ? 'high' : frac >= 0.05 ? 'medium' : 'low',
      note: frac < 0.05 && !level.noiseAlwaysOn ? 'Grain is already even across the photo; nothing to level. Raise the clean-up level to reduce it instead.' : '',
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
  t = mark('hotpixels', t);

  // --- colour blotches ---------------------------------------------------------
  if (!quick) {
    progress('colour', 0);
    const [Yc, Cb, Cr] = toYCbCr(r, g, b);
    const cn = measureChromaNoise(Yc, Cb, Cr, opts.noiseBlock, 1.0);
    const lv = Math.max(cn.sigmaCb, cn.sigmaCr) * 255;
    if (lv >= 0.5 || cn.fraction >= 0.05) {
      const amount = level.chromaAmount;
      const cf: ChromaFinding = {
        id: 'chroma',
        kind: 'chroma',
        label: 'Colour noise',
        description: `colour-layer noise of ${lv.toFixed(1)} levels; ${(cn.fraction * 100).toFixed(0)} % of the photo over 1 level`,
        severity: Math.min(1, lv / 4),
        extent: cn.fraction,
        enabled: amount > 0 && cn.fraction >= 0.05,
        confidence: cn.fraction >= 0.15 ? 'high' : 'medium',
        note: amount === 0 ? 'Left alone at the Gentle level.' : cn.fraction < 0.05 ? 'Only traces; left unticked.' : '',
        strength: 1,
        method: 'chroma-smooth',
        details: { sigmaCbLevels: cn.sigmaCb * 255, sigmaCrLevels: cn.sigmaCr * 255, fraction: cn.fraction, thresholdLevels: 1.0, amount: Math.max(amount, 0.6) },
      };
      findings.push(cf);
    }
    t = mark('chroma', t);

    // --- halos next to edges --------------------------------------------------
    progress('halos', 0);
    const rg = measureRinging(cleanLuma, 0.12, 8, nmodel.globalSigma);
    if (rg.ratio >= 1.5 && rg.haloLevels >= 1.0 && rg.bandFraction >= 0.002) {
      const amount = level.ringingAmount;
      const rf: RingingFinding = {
        id: 'ringing',
        kind: 'ringing',
        label: 'Halos next to edges',
        description: `noise next to hard edges is ${rg.ratio.toFixed(1)}× the grain elsewhere (${rg.haloLevels.toFixed(1)} vs ${rg.grainLevels.toFixed(1)} levels)`,
        severity: Math.min(1, (rg.ratio - 1) / 3),
        extent: rg.bandFraction,
        enabled: amount > 0,
        confidence: rg.ratio >= 2.5 ? 'high' : 'medium',
        note: amount === 0 ? 'Left alone at the Gentle level.' : '',
        strength: 1,
        method: 'halo-smooth',
        details: { haloLevels: rg.haloLevels, grainLevels: rg.grainLevels, ratio: rg.ratio, bandFraction: rg.bandFraction, edgeThreshold: rg.edgeThreshold, haloSigma: rg.haloSigma, amount: Math.max(amount, 0.6) },
      };
      findings.push(rf);
    }
    t = mark('ringing', t);

    // --- camera blur and sharpening, from the photo's straight edges -------------
    // (the earlier detail-balance check judged fine detail against the photo's
    // own coarser detail; a camera that softens everything fools that, so the
    // reference here is that real edges are sharp steps)
    progress('edges', 0);
    {
      const levelName = optsIn.level ?? defaultAnalysisOptions.level;
      const prof = measureEdgeProfile(cleanLuma);
      if (prof) {
        const m = fitCameraModel(prof);
        const targetWidthPx = 1.8;
        const reach = reachableSigma(m, targetSigma(targetWidthPx)) * 2.563;
        const outline = Math.max(prof.overshoot, prof.undershoot);
        const softer = reach < prof.widthPx * 0.9;
        const sharpened = outline >= 0.04;
        if (sharpened || softer) {
          const enabled = levelName === 'gentle' ? false : sharpened || levelName === 'strong' || levelName === 'maximum';
          const ef: EdgesFinding = {
            id: 'edges',
            kind: 'edges',
            label: sharpened ? 'Sharpening outlines' : 'Blurry edges',
            description: `edges ${prof.widthPx.toFixed(1)} px wide, outlines ${(outline * 100).toFixed(0)} % of edge height, from ${prof.count} edges`,
            severity: Math.min(1, outline / 0.2 + Math.max(0, prof.widthPx - 2) / 4),
            extent: 1,
            enabled,
            confidence: prof.count >= 150 && m.rms < 0.03 ? 'high' : 'medium',
            note: enabled ? '' : levelName === 'gentle' ? 'Left alone at the Gentle level.' : 'Sharpening soft edges is on from the Strong level. Tick it to try it here.',
            strength: 1,
            method: 'edge-correct',
            details: {
              t: prof.t,
              esf: prof.esf,
              count: prof.count,
              widthPx: prof.widthPx,
              overshoot: prof.overshoot,
              undershoot: prof.undershoot,
              blurSigma: m.blurSigma,
              amounts: m.amounts,
              fitRms: m.rms,
              targetWidthPx,
              reachableWidthPx: reach,
              keepCrisp: 0.45,
              grainSigma: nmodel.globalSigma,
              amount: 1,
            },
          };
          findings.push(ef);
        }
      }
    }
    t = mark('edges', t);

    // --- false colour and fringes from the camera's demosaicing -------------------
    progress('fringes', 0);
    {
      const levelName = optsIn.level ?? defaultAnalysisOptions.level;
      const [Yf, Cbf, Crf] = toYCbCr(r, g, b);
      const fc = measureFalseColour(Yf, Cbf, Crf);
      if (fc.detailLevels >= 1.5 && fc.detailLevels >= 1.5 * fc.flatLevels && fc.detailFraction >= 0.005) {
        const enabled = levelName !== 'gentle';
        const ff: FringesFinding = {
          id: 'fringes',
          kind: 'fringes',
          label: 'False colour on fine detail',
          description: `colour that does not follow brightness on fine detail: ${fc.detailLevels.toFixed(1)} levels (smooth areas ${fc.flatLevels.toFixed(1)})`,
          severity: Math.min(1, fc.detailLevels / 6),
          extent: fc.detailFraction,
          enabled,
          confidence: fc.detailLevels >= 3 ? 'high' : 'medium',
          note: enabled ? '' : 'Left alone at the Gentle level.',
          strength: 1,
          method: 'fringe-fix',
          details: { detailLevels: fc.detailLevels, flatLevels: fc.flatLevels, detailFraction: fc.detailFraction, amount: 1 },
        };
        findings.push(ff);
      }
    }
    t = mark('fringes', t);

    // --- the wide glow around subjects ---------------------------------------------
    progress('glow', 0);
    {
      const levelName = optsIn.level ?? defaultAnalysisOptions.level;
      const gr = glowFix(cleanLuma, 1);
      if (gr.sides > 0 && gr.glowLevels >= 2) {
        const enabled = levelName !== 'gentle';
        const gf: GlowFinding = {
          id: 'glow',
          kind: 'glow',
          label: 'Glow around subjects',
          description: `a ${gr.glowLevels.toFixed(1)}-level glow along ${(gr.share * 100).toFixed(0)} % of smooth edges`,
          severity: Math.min(1, gr.glowLevels / 10),
          extent: gr.coverage,
          enabled,
          confidence: gr.share >= 0.5 ? 'high' : 'medium',
          note: enabled ? '' : 'Left alone at the Gentle level.',
          strength: 1,
          method: 'glow-fix',
          details: { glowLevels: gr.glowLevels, sides: gr.sides, share: gr.share, coverage: gr.coverage, amount: 1 },
        };
        findings.push(gf);
      }
    }
    t = mark('glow', t);

    // --- fine texture weakened by noise reduction ------------------------------------
    progress('texture', 0);
    {
      const levelName = optsIn.level ?? defaultAnalysisOptions.level;
      const tm = measureTexture(cleanLuma);
      if (tm.weakFraction >= 0.05 && tm.share < 0.9) {
        // the effect, estimated on a centred crop
        const cs = Math.min(512, w, h);
        const cx = Math.floor((w - cs) / 2);
        const cy = Math.floor((h - cs) / 2);
        const crop = { w: cs, h: cs, data: new Float32Array(cs * cs) };
        for (let y = 0; y < cs; y++) crop.data.set(cleanLuma.data.subarray((cy + y) * w + cx, (cy + y) * w + cx + cs), y * cs);
        const before = measureTexture(crop);
        const dt = textureDelta(crop, 1);
        const after = measureTexture({ w: cs, h: cs, data: crop.data.map((v, i) => v + dt.data[i]) });
        const expectedShare = Math.max(tm.share, tm.share + (after.share - before.share));
        const enabled = levelName !== 'gentle';
        const xf: TextureFinding = {
          id: 'texture',
          kind: 'texture',
          label: 'Smeared fine texture',
          description: `fine texture at ${(tm.share * 100).toFixed(0)} % of what a scene like this carries, short in ${(tm.weakFraction * 100).toFixed(0)} % of the photo`,
          severity: Math.min(1, (1 - tm.share) * 2),
          extent: tm.weakFraction,
          enabled,
          confidence: tm.weakFraction >= 0.15 ? 'high' : 'medium',
          note: enabled ? '' : 'Left alone at the Gentle level.',
          strength: 1,
          method: 'texture-restore',
          details: { share: tm.share, weakFraction: tm.weakFraction, expectedShare, amount: 1 },
        };
        findings.push(xf);
      }
    }
    t = mark('texture', t);

    // --- contrast and depth (undoing the phone's flattening) ---------------------
    progress('tone', 0);
    {
      const levelName = optsIn.level ?? defaultAnalysisOptions.level;
      const fl = measureFlatness(cleanLuma);
      // estimate the effect on a 4× smaller copy, for the check to compare against
      const f4 = Math.max(1, Math.floor(Math.min(w, h) / 400));
      const sw = Math.floor(w / f4);
      const sh = Math.floor(h / f4);
      const small = { w: sw, h: sh, data: new Float32Array(sw * sh) };
      for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) small.data[y * sw + x] = cleanLuma.data[y * f4 * w + x * f4];
      const pr = toneParams(levelName, 1, nmodel.globalSigma);
      const dsm = toneDelta(small, pr, fl.shadowLevels);
      const after = measureFlatness({ w: sw, h: sh, data: small.data.map((v, i) => v + dsm.data[i]) });
      const before = measureFlatness(small);
      const expectedRatio = before.localContrast > 0.1 ? after.localContrast / before.localContrast : 1;
      const lifted = fl.shadowLevels >= 8;
      const enabled = levelName !== 'gentle';
      const tf: ToneFinding = {
        id: 'tone',
        kind: 'tone',
        label: 'Flat, low contrast',
        description: `darkest shadows at ${fl.shadowLevels.toFixed(0)} levels; contrast between neighbouring areas ${fl.localContrast.toFixed(1)} levels`,
        severity: Math.min(1, Math.max(0, (fl.shadowLevels - 4) / 30) + 0.3),
        extent: 1,
        enabled,
        confidence: lifted ? 'high' : 'medium',
        note: enabled ? '' : 'Left alone at the Gentle level.',
        strength: 1,
        method: 'local-laplacian',
        details: { shadowLevels: fl.shadowLevels, highlightLevels: fl.highlightLevels, localContrast: fl.localContrast, expectedRatio, level: levelName, noise: nmodel.globalSigma, amount: 1 },
      };
      findings.push(tf);
    }
    t = mark('tone', t);

    // --- banding in smooth areas -------------------------------------------------
    progress('banding', 0);
    {
      const levelName = optsIn.level ?? defaultAnalysisOptions.level;
      const bands = detectBanding(g, nmap);
      if (bands.fraction >= 0.02) {
        const strengthLevels = bandingStrength(cleanLuma, bands.banded, bands.bw, bands.bh, bands.block);
        const enabled = levelName !== 'gentle';
        const bf: BandingFinding = {
          id: 'banding',
          kind: 'banding',
          label: 'Banding in smooth areas',
          description: `steps of ${bands.stepLevels} level${bands.stepLevels === 1 ? '' : 's'} in smooth gradients, ${(bands.fraction * 100).toFixed(0)} % of the photo`,
          severity: Math.min(1, strengthLevels / 0.6),
          extent: bands.fraction,
          enabled,
          confidence: bands.fraction >= 0.08 ? 'high' : 'medium',
          note: enabled ? '' : 'Left alone at the Gentle level.',
          strength: 1,
          method: 'deband',
          details: { block: bands.block, bw: bands.bw, bh: bands.bh, banded: bands.banded, fraction: bands.fraction, stepLevels: bands.stepLevels, strengthLevels, amount: 1 },
        };
        findings.push(bf);
      }
    }
    t = mark('banding', t);

  }
  timings.total = Math.round(performance.now() - t0);

  const cfa = detectCfaPhase(input.planes);
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
    cfa,
  };
}
