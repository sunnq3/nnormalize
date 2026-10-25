/**
 * How much of each fix survives a destination's re-save.
 *
 * For every enabled finding we measure its own signature three ways:
 *   source    – the untouched photo
 *   srcDest   – the untouched photo after the destination's re-save
 *   resDest   – our result after the destination's re-save
 * A fix is judged by how far resDest sits from srcDest on the way to "gone":
 *   retained = 1 - resDest / srcDest   (clamped to 0..1)
 * so the destination's own artefacts (which hit both copies alike) do not
 * count against the fix, and a downloader's actual before/after is what is
 * reported.
 */

import { detectBlockGrid } from '../dsp/blocks';
import { measureChromaNoise, measureRinging, toYCbCr } from '../dsp/chroma';
import { median9 } from '../dsp/hotpixels';
import { bandingStrength } from '../dsp/banding';
import { detailBalance, radialProfile } from '../dsp/spectrum';
import { measureEdgeProfile } from '../dsp/edges';
import { measureFalseColour } from '../dsp/cfa';
import { measureFlatness } from '../dsp/tone';
import { glowFix } from '../dsp/glow';
import { measureTexture } from '../dsp/texture';
import { noiseMap, noiseModel } from '../dsp/noise';
import { measurePattern } from '../dsp/pattern';
import { lumaPlane, type Plane } from '../dsp/plane';
import { tileGrid } from '../dsp/tiles';
import { subsampleGrid } from './analyze';
import type { Finding } from './types';

export type Verdict = 'survives' | 'partly' | 'lost' | 'n/a';

export interface FixSurvival {
  id: string;
  kind: Finding['kind'];
  label: string;
  unit: string;
  source: number; // signature in the untouched photo
  sourceOnDest: number; // ... after the destination's re-save
  resultOnDest: number; // our result after the destination's re-save
  retained: number; // 0..1
  verdict: Verdict;
  note: string;
}

export interface SurvivalReport {
  fixes: FixSurvival[];
  /** overall numbers a person can read: block steps and grain, source vs result, both after the destination */
  blockStepLevels: { sourceOnDest: number; resultOnDest: number };
  grainLevels: { sourceOnDest: number; resultOnDest: number };
  pass: boolean;
}

function verdictFor(retained: number): Verdict {
  if (retained >= 0.8) return 'survives';
  if (retained >= 0.35) return 'partly';
  return 'lost';
}

export interface SurvivalOptions {
  /** the planes are a crop of the photo at this origin (multiples of 32), for findings stored in whole-photo coordinates */
  crop?: { x0: number; y0: number; fullW: number };
  /** values for the untouched photo and its re-saved copy, kept between calls on the same crop and destination */
  cache?: Map<string, number>;
}

export function measureSurvival(
  source: [Plane, Plane, Plane],
  srcDest: [Plane, Plane, Plane],
  resDest: [Plane, Plane, Plane],
  findings: Finding[],
  opts: SurvivalOptions = {},
): SurvivalReport {
  const cache = opts.cache ?? new Map<string, number>();
  const memo = (key: string, fn: () => number): number => {
    const v = cache.get(key);
    if (v !== undefined) return v;
    const r = fn();
    cache.set(key, r);
    return r;
  };
  let lumSc: Plane | null = null;
  let lumSDc: Plane | null = null;
  const lumSf = (): Plane => (lumSc ??= lumaPlane(source[0], source[1], source[2]));
  const lumSDf = (): Plane => (lumSDc ??= lumaPlane(srcDest[0], srcDest[1], srcDest[2]));
  const lumRD = lumaPlane(resDest[0], resDest[1], resDest[2]);
  const grid = subsampleGrid(tileGrid(lumRD.w, lumRD.h, 1024), 16);
  const fixes: FixSurvival[] = [];

  const gridR = detectBlockGrid(lumRD);
  const stepS = memo('step:sd', () => {
    const g = detectBlockGrid(lumSDf());
    return Math.max(g.stepX, g.stepY) * 255;
  });
  const stepR = Math.max(gridR.stepX, gridR.stepY) * 255;
  const sigS = memo('sigma:sd', () => noiseModel(noiseMap(lumSDf(), 32)).globalSigma * 255);
  const sigR = noiseModel(noiseMap(lumRD, 32)).globalSigma * 255;

  for (const f of findings) {
    if (!f.enabled) continue;
    switch (f.kind) {
      case 'periodic': {
        const amp = (p: Plane): number => {
          const m = measurePattern(p, grid, f.details.u, f.details.v, { refine: false, searchGrid: grid, measureGrid: grid });
          return Math.max(m.amplitude, m.amplitudeP90) * 255;
        };
        const a0 = memo(`${f.id}:s`, () => amp(lumSf()));
        const a1 = memo(`${f.id}:sd`, () => amp(lumSDf()));
        const a2 = amp(lumRD);
        const retained = a1 > 0.05 ? Math.max(0, Math.min(1, 1 - a2 / a1)) : 1;
        // under one level nobody can see a pattern, whatever share of it that is
        const v = a2 < 1.0 ? 'survives' : verdictFor(retained);
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: a0,
          sourceOnDest: a1,
          resultOnDest: a2,
          retained,
          verdict: v,
          note:
            v === 'survives'
              ? a2 < 1.0 && retained < 0.8
                ? `what remains (${a2.toFixed(1)} levels) is below what can be seen.`
                : 'The pattern stays gone.'
              : v === 'partly'
                ? `part of the pattern shows again after the re-save (${a2.toFixed(1)} of ${a1.toFixed(1)} levels).`
                : `the re-save brings the pattern back (${a2.toFixed(1)} of ${a1.toFixed(1)} levels).`,
        });
        break;
      }
      case 'jpeg-grid': {
        const s0 = memo('step:s', () => {
          const g0 = detectBlockGrid(lumSf());
          return Math.max(g0.stepX, g0.stepY) * 255;
        });
        const retained = stepS > 0.1 ? Math.max(0, Math.min(1, 1 - stepR / stepS)) : 1;
        const v = stepR < 0.5 ? 'survives' : verdictFor(retained);
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: s0,
          sourceOnDest: stepS,
          resultOnDest: stepR,
          retained,
          verdict: v,
          note:
            stepR < 0.5
              ? 'block edges in the re-saved copy are below what can be seen.'
              : v === 'survives'
                ? 'block edges stay down.'
                : `the re-save leaves block edges of ${stepR.toFixed(1)} levels (the untouched photo would have ${stepS.toFixed(1)}); that is the destination’s own compression floor.`,
        });
        break;
      }
      case 'noise': {
        const sig0 = memo('sigma:s', () => noiseModel(noiseMap(lumSf(), 32)).globalSigma * 255);
        // grain deliberately put back afterwards is not grain that came back
        const putBack = findings.find((g) => g.kind === 'grain' && g.enabled);
        const wantBack = putBack && putBack.kind === 'grain' ? putBack.details.targetLevels : 0;
        const target = f.details.reduction > 0 ? Math.max(sigS * (1 - f.details.reduction), wantBack) : sigS;
        // for levelling (reduction 0) the signature is unevenness; for reduction it is the grain itself
        let retained: number;
        let note: string;
        if (f.details.reduction > 0) {
          retained = sigS > 0.05 ? Math.max(0, Math.min(1, (sigS - sigR) / Math.max(1e-6, sigS - target))) : 1;
          if (sigS - target < 0.3) retained = 1; // nothing to remove beyond what is put back
          if (wantBack > 0 && sigR <= wantBack * 1.25) retained = 1; // what is left is the grain put back on purpose
          note = retained >= 0.8 ? (wantBack ? 'grain is reduced, then put back at the amount set.' : 'grain stays reduced.') : retained >= 0.35 ? 'some grain shows again; the re-save adds noise of its own.' : 'the re-save adds about as much noise as was removed.';
        } else {
          retained = 1;
          note = 'levelling only; the re-save adds the same noise everywhere, so evenness is kept.';
        }
        fixes.push({ id: f.id, kind: f.kind, label: f.label, unit: 'levels', source: sig0, sourceOnDest: sigS, resultOnDest: sigR, retained, verdict: verdictFor(retained), note });
        break;
      }
      case 'chroma': {
        const cn = (p: [Plane, Plane, Plane]): number => {
          const [Yc, Cb, Cr] = toYCbCr(p[0], p[1], p[2]);
          const m = measureChromaNoise(Yc, Cb, Cr, 32, f.details.thresholdLevels);
          return Math.max(m.sigmaCb, m.sigmaCr) * 255;
        };
        const c0 = memo(`${f.id}:s`, () => cn(source));
        const c1 = memo(`${f.id}:sd`, () => cn(srcDest));
        const c2 = cn(resDest);
        const target = c1 * (1 - f.details.amount);
        const retained = c1 > 0.05 ? Math.max(0, Math.min(1, (c1 - c2) / Math.max(1e-6, c1 - target))) : 1;
        const cv = c2 < f.details.thresholdLevels ? 'survives' : verdictFor(retained);
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: c0,
          sourceOnDest: c1,
          resultOnDest: c2,
          retained,
          verdict: cv,
          note: cv === 'survives' ? 'colour stays clean.' : 'some colour noise shows again after the re-save.',
        });
        break;
      }
      case 'ringing': {
        const rg = (p: Plane): number => measureRinging(p, f.details.edgeThreshold, 8).haloLevels;
        const h0 = memo(`${f.id}:s`, () => rg(lumSf()));
        const h1 = memo(`${f.id}:sd`, () => rg(lumSDf()));
        const h2 = rg(lumRD);
        const target = h1 * (1 - f.details.amount);
        const retained = h1 > 0.05 ? Math.max(0, Math.min(1, (h1 - h2) / Math.max(1e-6, h1 - target))) : 1;
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: h0,
          sourceOnDest: h1,
          resultOnDest: h2,
          retained,
          verdict: verdictFor(retained),
          note: retained >= 0.8 ? 'edges stay clean.' : 'the re-save puts some halo back next to edges; that is its own ringing.',
        });
        break;
      }
      case 'grain': {
        // the grain that comes through: base grain of the downloader's copy against the target
        const want = f.details.targetLevels;
        const retained = want > 0.05 ? Math.max(0, Math.min(1, sigR / want)) : 1;
        const v = retained >= 0.7 ? 'survives' : retained >= 0.35 ? 'partly' : 'lost';
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: f.details.baseLevels,
          sourceOnDest: sigS,
          resultOnDest: sigR,
          retained,
          verdict: v,
          note:
            v === 'survives'
              ? `the grain comes through at ${sigR.toFixed(1)} levels (${want.toFixed(1)} wanted).`
              : `the re-save keeps only ${sigR.toFixed(1)} of the ${want.toFixed(1)} levels of grain; raise the amount if the copy looks plastic.`,
        });
        break;
      }
      case 'detail': {
        const d = f.details;
        const bal = (p: Plane) => detailBalance(radialProfile(p), d.grainSigma, d.line);
        const key = d.mode === 'sharp' ? 'excessDb' : 'deficitDb';
        const v0 = memo(`${f.id}:s`, () => Math.max(0, bal(lumSf())[key]));
        const v1 = memo(`${f.id}:sd`, () => Math.max(0, bal(lumSDf())[key]));
        const v2 = Math.max(0, bal(lumRD)[key]);
        const retained = v1 > 0.3 ? Math.max(0, Math.min(1, 1 - v2 / v1)) : 1;
        const v: Verdict = v2 < 2 || retained >= 0.6 ? 'survives' : retained >= 0.25 ? 'partly' : 'lost';
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'dB',
          source: v0,
          sourceOnDest: v1,
          resultOnDest: v2,
          retained,
          verdict: v,
          note:
            v === 'survives'
              ? d.mode === 'sharp'
                ? 'detail stays at a natural level.'
                : 'fine detail comes through the re-save.'
              : d.mode === 'sharp'
                ? 'the re-save leaves some of the oversharpening.'
                : 'the re-save smooths away part of the detail brought back; raise the most it may add if the copy looks soft.',
        });
        break;
      }
      case 'texture': {
        const sh = (p: Plane): number => measureTexture(p).share;
        const s0 = memo(`${f.id}:s`, () => sh(lumSf()));
        const s1 = memo(`${f.id}:sd`, () => sh(lumSDf()));
        const s2 = sh(lumRD);
        const want = (f.details.expectedShare - f.details.share) * f.details.amount;
        const retained = want > 0.005 ? Math.max(0, Math.min(1, (s2 - s1) / want)) : 1;
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: '%',
          source: s0 * 100,
          sourceOnDest: s1 * 100,
          resultOnDest: s2 * 100,
          retained,
          verdict: verdictFor(retained),
          note: retained >= 0.8 ? 'the recovered texture comes through.' : 'the re-save smooths away part of the recovered texture.',
        });
        break;
      }
      case 'glow': {
        const gl = (p: Plane): number => {
          const r = glowFix(p, 0);
          return r.share * r.glowLevels;
        };
        const v0 = memo(`${f.id}:s`, () => gl(lumSf()));
        const v1 = memo(`${f.id}:sd`, () => gl(lumSDf()));
        const v2 = gl(lumRD);
        const retained = v1 > 0.05 ? Math.max(0, Math.min(1, 1 - v2 / v1)) : 1;
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: v0,
          sourceOnDest: v1,
          resultOnDest: v2,
          retained,
          verdict: verdictFor(retained),
          note: retained >= 0.8 ? 'the glow stays gone.' : 'some glow shows again after the re-save.',
        });
        break;
      }
      case 'tone': {
        const lc = (p: Plane): number => measureFlatness(p).localContrast;
        const c0 = memo(`${f.id}:s`, () => lc(lumSf()));
        const c1 = memo(`${f.id}:sd`, () => lc(lumSDf()));
        const c2 = lc(lumRD);
        const want = (f.details.expectedRatio - 1) * f.details.amount;
        const retained = want > 0.01 && c1 > 0.1 ? Math.max(0, Math.min(1, (c2 / c1 - 1) / want)) : 1;
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: c0,
          sourceOnDest: c1,
          resultOnDest: c2,
          retained,
          verdict: verdictFor(retained),
          note: retained >= 0.8 ? 'depth and contrast come through.' : 'part of the added contrast is lost in the re-save.',
        });
        break;
      }
      case 'fringes': {
        const fc = (p: [Plane, Plane, Plane]): number => {
          const [Y, Cb, Cr] = toYCbCr(p[0], p[1], p[2]);
          return measureFalseColour(Y, Cb, Cr).detailLevels;
        };
        const v0 = memo(`${f.id}:s`, () => fc(source));
        const v1 = memo(`${f.id}:sd`, () => fc(srcDest));
        const v2 = fc(resDest);
        const retained = v1 > 0.3 ? Math.max(0, Math.min(1, 1 - v2 / v1)) : 1;
        const v: Verdict = v2 <= Math.max(1.2, f.details.flatLevels) || retained >= 0.6 ? 'survives' : retained >= 0.3 ? 'partly' : 'lost';
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: v0,
          sourceOnDest: v1,
          resultOnDest: v2,
          retained,
          verdict: v,
          note: v === 'survives' ? 'fine detail stays free of false colour.' : 'some false colour shows again after the re-save.',
        });
        break;
      }
      case 'edges': {
        const prof = (p: Plane): number => {
          const e = measureEdgeProfile(p, { maxEdges: 2000 });
          return e ? Math.max(e.overshoot, e.undershoot) * 100 : NaN;
        };
        const o0 = memo(`${f.id}:s`, () => prof(lumSf()));
        const o1 = memo(`${f.id}:sd`, () => prof(lumSDf()));
        const o2 = prof(lumRD);
        if (!Number.isFinite(o1) || !Number.isFinite(o2)) break;
        const retained = o1 > 1 ? Math.max(0, Math.min(1, 1 - o2 / o1)) : 1;
        const v: Verdict = o2 <= 4 || retained >= 0.6 ? 'survives' : retained >= 0.3 ? 'partly' : 'lost';
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: '%',
          source: o0,
          sourceOnDest: o1,
          resultOnDest: o2,
          retained,
          verdict: v,
          note: v === 'survives' ? 'edges stay clean, without outlines.' : 'the re-save sharpens or rings some outline back onto the edges.',
        });
        break;
      }
      case 'banding': {
        const d = f.details;
        let banded = d.banded;
        let bw = d.bw;
        let bh = d.bh;
        if (opts.crop) {
          const bx0 = Math.round(opts.crop.x0 / d.block);
          const by0 = Math.round(opts.crop.y0 / d.block);
          bw = Math.floor(lumRD.w / d.block);
          bh = Math.floor(lumRD.h / d.block);
          banded = new Uint8Array(bw * bh);
          for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) if (by0 + y < d.bh && bx0 + x < d.bw) banded[y * bw + x] = d.banded[(by0 + y) * d.bw + bx0 + x];
        }
        let any = 0;
        for (let i = 0; i < banded.length; i++) any += banded[i];
        if (!any) break;
        const str = (p: Plane): number => bandingStrength(p, banded, bw, bh, d.block);
        const b0 = memo(`${f.id}:s`, () => str(lumSf()));
        const b1 = memo(`${f.id}:sd`, () => str(lumSDf()));
        const b2 = str(lumRD);
        const retained = b1 > 0.02 ? Math.max(0, Math.min(1, 1 - b2 / b1)) : 1;
        const v: Verdict = b2 < 0.12 || retained >= 0.7 ? 'survives' : retained >= 0.3 ? 'partly' : 'lost';
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'levels',
          source: b0,
          sourceOnDest: b1,
          resultOnDest: b2,
          retained,
          verdict: v,
          note: v === 'survives' ? 'gradients stay smooth.' : 'the re-save cuts some steps back into the gradients; its own blocks show in smooth areas.',
        });
        break;
      }
      case 'hotpixels': {
        const all = f.details.indices;
        const fullW = opts.crop?.fullW ?? lumRD.w;
        const ox = opts.crop?.x0 ?? 0;
        const oy = opts.crop?.y0 ?? 0;
        const w = lumRD.w;
        const idx: number[] = [];
        for (const i of all) {
          const x = (i % fullW) - ox;
          const y = Math.floor(i / fullW) - oy;
          if (x >= 1 && y >= 1 && x < w - 1 && y < lumRD.h - 1) idx.push(y * w + x);
        }
        const still = (p: Plane): number => {
          let n = 0;
          for (const i of idx) if (Math.abs(p.data[i] - median9(p.data, i, w)) > 0.06) n++;
          return n;
        };
        if (!idx.length) break;
        const n1 = memo(`${f.id}:sd`, () => still(lumSDf()));
        const n2 = still(lumRD);
        const retained = n1 > 0 ? Math.max(0, Math.min(1, 1 - n2 / n1)) : 1;
        fixes.push({
          id: f.id,
          kind: f.kind,
          label: f.label,
          unit: 'pixels',
          source: idx.length,
          sourceOnDest: n1,
          resultOnDest: n2,
          retained,
          verdict: verdictFor(retained),
          note: retained >= 0.8 ? 'they stay fixed.' : `${n2} of ${n1} still stand out after the re-save.`,
        });
        break;
      }
    }
  }
  const pass = fixes.every((x) => x.verdict !== 'lost');
  return { fixes, blockStepLevels: { sourceOnDest: stepS, resultOnDest: stepR }, grainLevels: { sourceOnDest: sigS, resultOnDest: sigR }, pass };
}
