/**
 * Turn a finding into a delta layer: one Float32 plane per channel such that
 * corrected = original + delta. Every method is applied per channel with the
 * channel's own measurement, because colour moiré and chroma noise differ
 * between channels even when the frequency is shared.
 */

import { deblockDelta } from '../dsp/blocks';
import { chromaDelta, haloBand, measureChromaNoise, ringingDelta, toRGB, toYCbCr } from '../dsp/chroma';
import { grainDelta } from '../dsp/grain';
import { debandDelta } from '../dsp/banding';
import { correctionGainDb, detailMask, edgeZone, outlineClampDelta } from '../dsp/edges';
import { driftDelta, type DriftTrack } from '../dsp/drift';
import { falseColourDelta } from '../dsp/cfa';
import { toneDelta, toneParams } from '../dsp/tone';
import { glowFix } from '../dsp/glow';
import { textureDelta } from '../dsp/texture';
import { bandGains, detailBalance, detailDelta, radialProfile, wantedGainDb } from '../dsp/spectrum';
import { parametricDeltaGlobal, parametricDeltaTiled, spectralDelta, type SpectralTarget } from '../dsp/filter';
import { median9 } from '../dsp/hotpixels';
import { denoiseDelta, noiseMap, noiseModel } from '../dsp/noise';
import { measurePattern } from '../dsp/pattern';
import { makePlane, type Plane } from '../dsp/plane';
import type { TileGrid } from '../dsp/tiles';
import { subsampleGrid } from './analyze';
import type { Analysis, Finding, PeriodicFinding } from './types';

export type Progress = (fraction: number) => void;

function lumaOf(planes: [Plane, Plane, Plane]): Plane {
  const n = planes[0].w * planes[0].h;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = 0.299 * planes[0].data[i] + 0.587 * planes[1].data[i] + 0.114 * planes[2].data[i];
  return { w: planes[0].w, h: planes[0].h, data: out };
}

function addInto(dst: Plane, src: Plane): void {
  for (let i = 0; i < dst.data.length; i++) dst.data[i] += src.data[i];
}

function periodicDelta(ch: Plane, grid: TileGrid, f: PeriodicFinding, progress: Progress): Plane {
  const { w, h } = ch;
  const d = f.details;
  const orders = [{ u: d.u, v: d.v, widthBins: d.widthBins }, ...d.harmonics.map((hm) => ({ u: hm.u, v: hm.v, widthBins: hm.widthBins }))];
  if (f.method === 'flatten' || f.method === 'notch') {
    const targets: SpectralTarget[] = orders;
    return spectralDelta(ch, grid, targets, { mode: f.method, onProgress: (a, b) => progress(a / b) });
  }
  const out = makePlane(w, h);
  const sub = subsampleGrid(grid, 36);
  orders.forEach((o, k) => {
    // a stationary pattern needs one amplitude and phase: a subset of tiles is enough
    const m = measurePattern(ch, grid, o.u, o.v, { refine: false, searchGrid: sub, measureGrid: f.method === 'parametric' ? sub : grid });
    const delta = f.method === 'parametric' ? parametricDeltaGlobal(w, h, o.u, o.v, m.amplitude, m.phase) : parametricDeltaTiled(w, h, grid, m);
    addInto(out, delta);
    progress((k + 1) / orders.length);
  });
  return out;
}

export function computeDelta(planes: [Plane, Plane, Plane], analysis: Analysis, finding: Finding, progress: Progress = () => {}): [Plane, Plane, Plane] {
  const grid: TileGrid = { size: analysis.tile.size, hop: analysis.tile.hop, xs: analysis.tile.xs, ys: analysis.tile.ys };
  const out: Plane[] = [];

  // corrections that work on the colour layer or need the luminance of all
  // three channels at once
  if (finding.kind === 'chroma') {
    const [Y, Cb, Cr] = toYCbCr(planes[0], planes[1], planes[2]);
    progress(0.2);
    const cn = measureChromaNoise(Y, Cb, Cr, 32, finding.details.thresholdLevels);
    const { dCb, dCr } = chromaDelta(Cb, Cr, cn, finding.details.amount, finding.details.thresholdLevels);
    progress(0.8);
    // delta in RGB: convert (Y, Cb+dCb, Cr+dCr) back and subtract the input
    const Cb2 = { w: Cb.w, h: Cb.h, data: new Float32Array(Cb.data) };
    const Cr2 = { w: Cr.w, h: Cr.h, data: new Float32Array(Cr.data) };
    for (let i = 0; i < Cb2.data.length; i++) {
      Cb2.data[i] += dCb.data[i];
      Cr2.data[i] += dCr.data[i];
    }
    const rgb = toRGB(Y, Cb2, Cr2);
    const deltas = rgb.map((p, c) => {
      const d = makePlane(p.w, p.h);
      for (let i = 0; i < d.data.length; i++) d.data[i] = p.data[i] - planes[c].data[i];
      return d;
    });
    progress(1);
    return deltas as [Plane, Plane, Plane];
  }
  if (finding.kind === 'detail') {
    // brightness detail only: the same delta goes to all three channels
    const d = finding.details;
    const Y = lumaOf(planes);
    const prof = radialProfile(Y);
    const bal = detailBalance(prof, d.grainSigma, d.line);
    const gains = bandGains(bal.f, wantedGainDb(bal, d.mode, d.maxBoostDb), d.amount);
    progress(0.5);
    const delta = detailDelta(Y, gains);
    progress(1);
    return [delta, { w: delta.w, h: delta.h, data: new Float32Array(delta.data) }, { w: delta.w, h: delta.h, data: new Float32Array(delta.data) }];
  }
  if (finding.kind === 'drift') {
    // every channel, so coloured moiré goes with the brightness stripes
    const d = finding.details;
    const track: DriftTrack = { nodes: d.nodes, gridW: d.gridW, gridH: d.gridH, period: d.period, periodMin: d.periodMin, periodMax: d.periodMax, angleDeg: d.angleDeg, angleSpread: d.angleSpread, amplitude: d.amplitude / 255, amplitudeP90: d.amplitudeP90 / 255, excessDb: d.excessDb, coverage: d.coverage, stationary: d.stationary };
    const deltas = driftDelta([planes[0], planes[1], planes[2]], track, progress);
    if (d.amount !== 1) for (const p of deltas) for (let i = 0; i < p.data.length; i++) p.data[i] *= d.amount;
    return deltas as [Plane, Plane, Plane];
  }
  if (finding.kind === 'texture') {
    const [Y, Cb, Cr] = toYCbCr(planes[0], planes[1], planes[2]);
    const dT = textureDelta(Y, finding.details.amount);
    progress(0.8);
    const Y2 = { w: Y.w, h: Y.h, data: Y.data.map((v, i) => v + dT.data[i]) };
    const rgb = toRGB(Y2, Cb, Cr);
    progress(1);
    return rgb.map((p, c) => ({ w: p.w, h: p.h, data: p.data.map((v, i) => v - planes[c].data[i]) })) as [Plane, Plane, Plane];
  }
  if (finding.kind === 'glow') {
    const [Y, Cb, Cr] = toYCbCr(planes[0], planes[1], planes[2]);
    const g = glowFix(Y, finding.details.amount);
    progress(0.8);
    const Y2 = { w: Y.w, h: Y.h, data: Y.data.map((v, i) => v + g.delta.data[i]) };
    const rgb = toRGB(Y2, Cb, Cr);
    progress(1);
    return rgb.map((p, c) => ({ w: p.w, h: p.h, data: p.data.map((v, i) => v - planes[c].data[i]) })) as [Plane, Plane, Plane];
  }
  if (finding.kind === 'tone') {
    // brightness only, colour kept as it is
    const d = finding.details;
    const [Y, Cb, Cr] = toYCbCr(planes[0], planes[1], planes[2]);
    const dY = toneDelta(Y, toneParams(d.level, d.amount, d.noise), d.shadowLevels);
    progress(0.8);
    const Y2 = { w: Y.w, h: Y.h, data: Y.data.map((v, i) => v + dY.data[i]) };
    const rgb = toRGB(Y2, Cb, Cr);
    progress(1);
    return rgb.map((p, c) => ({ w: p.w, h: p.h, data: p.data.map((v, i) => v - planes[c].data[i]) })) as [Plane, Plane, Plane];
  }
  if (finding.kind === 'fringes') {
    const [Y, Cb, Cr] = toYCbCr(planes[0], planes[1], planes[2]);
    const { dCb, dCr } = falseColourDelta(Y, Cb, Cr, finding.details.amount);
    progress(0.7);
    const Cb2 = { w: Cb.w, h: Cb.h, data: Cb.data.map((v, i) => v + dCb.data[i]) };
    const Cr2 = { w: Cr.w, h: Cr.h, data: Cr.data.map((v, i) => v + dCr.data[i]) };
    const rgb = toRGB(Y, Cb2, Cr2);
    progress(1);
    return rgb.map((p, c) => ({ w: p.w, h: p.h, data: p.data.map((v, i) => v - planes[c].data[i]) })) as [Plane, Plane, Plane];
  }
  if (finding.kind === 'edges') {
    // brightness only: the same delta goes to all three channels
    const d = finding.details;
    const Y = lumaOf(planes);
    const f = Float32Array.from({ length: 128 }, (_, i) => (i + 1) / 256);
    const model = { blurSigma: d.blurSigma, amounts: d.amounts, rms: d.fitRms };
    // at strong straight edges: the outlines go (keeping only the crispness asked for);
    // in texture: only the wide sharpening goes and the blur is reduced, the fine crispness stays
    const atEdges = detailDelta(Y, bandGains(f, correctionGainDb(model, d.targetWidthPx, f, d.grainSigma, d.keepCrisp ?? 0.5), d.amount));
    progress(0.3);
    const texGain = correctionGainDb(model, d.targetWidthPx, f, d.grainSigma, 1);
    const inTexture = detailDelta(Y, bandGains(f, texGain, d.amount));
    // in flat areas only what lowers a frequency (undoing the camera's wide
    // sharpening): raising fine frequencies there would only raise the noise
    const inFlat = detailDelta(Y, bandGains(f, texGain.map((v) => Math.min(0, v)), d.amount));
    const zone = edgeZone(Y);
    const detail = detailMask(Y, d.grainSigma);
    const delta = makePlane(Y.w, Y.h);
    for (let i = 0; i < delta.data.length; i++) {
      const base = inFlat.data[i] + detail[i] * (inTexture.data[i] - inFlat.data[i]);
      delta.data[i] = base + zone[i] * (atEdges.data[i] - base);
    }
    // a sharpening threshold: what the boost adds is kept where it is large
    // (real edges and texture) and faded where it is the size of noise, so
    // the noise beside edges is not raised and a re-save has less to ring on
    {
      const sample: number[] = [];
      for (let i = 0; i < delta.data.length; i += 3) if (detail[i] > 0.5 || zone[i] > 0.5) sample.push(Math.abs(delta.data[i] - inFlat.data[i]));
      sample.sort((a, b) => a - b);
      const T = sample.length ? 1.4826 * sample[sample.length >> 1] : 0;
      if (T > 0)
        for (let i = 0; i < delta.data.length; i++) {
          const b = delta.data[i] - inFlat.data[i];
          delta.data[i] = inFlat.data[i] + (b * b * b) / (b * b + T * T);
        }
    }
    progress(0.5);
    // then the lopsided outlines the linear step cannot reach, beside flat areas only
    const lin = { w: Y.w, h: Y.h, data: Y.data.map((v, i) => v + delta.data[i]) };
    const clamp = outlineClampDelta(lin, d.amount);
    for (let i = 0; i < delta.data.length; i++) delta.data[i] += clamp.data[i];
    progress(1);
    return [delta, { w: delta.w, h: delta.h, data: new Float32Array(delta.data) }, { w: delta.w, h: delta.h, data: new Float32Array(delta.data) }];
  }
  if (finding.kind === 'banding') {
    const d = finding.details;
    const bands = { block: d.block, bw: d.bw, bh: d.bh, banded: d.banded, fraction: d.fraction, stepLevels: d.stepLevels };
    const deltas = planes.map((p, c) => {
      const r = debandDelta(p, bands, { amount: d.amount, stepLevels: d.stepLevels, seed: 0xbad5 });
      progress((c + 1) / 3);
      return r;
    });
    return deltas as [Plane, Plane, Plane];
  }
  if (finding.kind === 'grain') {
    const d = grainDelta(planes, { targetLevels: finding.details.targetLevels, colour: finding.details.colour, seed: finding.details.seed });
    progress(1);
    return d;
  }
  if (finding.kind === 'ringing') {
    const Y = lumaOf(planes);
    const { halo } = haloBand(Y, finding.details.edgeThreshold, 8);
    const haloSigma = finding.details.haloSigma ?? finding.details.haloLevels / 255;
    const deltas = planes.map((p, c) => {
      const d = ringingDelta(p, halo, haloSigma, finding.details.amount);
      progress((c + 1) / 3);
      return d;
    });
    return deltas as [Plane, Plane, Plane];
  }
  planes.forEach((ch, ci) => {
    const sub = (fr: number): void => progress((ci + fr) / 3);
    switch (finding.kind) {
      case 'periodic':
        out.push(periodicDelta(ch, grid, finding, sub));
        break;
      case 'jpeg-grid': {
        const g = finding.details;
        out.push(
          deblockDelta(
            ch,
            { offsetX: g.offsetX, offsetY: g.offsetY, strengthX: g.strengthX, strengthY: g.strengthY, stepX: 0, stepY: 0, confidenceX: 0, confidenceY: 0, detected: true },
            { maxStep: g.maxStep ?? 0.08, activityRatio: g.activityRatio ?? 1.5, span: g.span ?? 4 },
          ),
        );
        sub(1);
        break;
      }
      case 'noise': {
        const map = noiseMap(ch, finding.details.block);
        const model = noiseModel(map);
        const reduction = Math.min(0.95, Math.max(0, finding.details.reduction ?? 0));
        const thr = Math.max(1.5, finding.details.thresholdDb / 2);
        // grainier patches are brought down; cleaner ones are left clean
        // (no synthetic grain is ever added: it reads as an overlay)
        const dn = denoiseDelta(ch, map, model, { thresholdDb: reduction > 0 ? 0 : thr, targetScale: 1 - reduction });
        out.push(dn.delta);
        sub(1);
        break;
      }
      case 'hotpixels': {
        // replace each outlier found on the luma by the channel's own 3x3 median
        const d = makePlane(ch.w, ch.h);
        const w = ch.w;
        for (const i of finding.details.indices) {
          const x = i % w;
          const y = (i - x) / w;
          if (x < 1 || y < 1 || x >= w - 1 || y >= ch.h - 1) continue;
          d.data[i] = median9(ch.data, i, w) - ch.data[i];
        }
        out.push(d);
        sub(1);
        break;
      }
    }
  });
  return out as [Plane, Plane, Plane];
}
