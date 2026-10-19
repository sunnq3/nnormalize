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

export function measureSurvival(source: [Plane, Plane, Plane], srcDest: [Plane, Plane, Plane], resDest: [Plane, Plane, Plane], findings: Finding[]): SurvivalReport {
  const lumS = lumaPlane(source[0], source[1], source[2]);
  const lumSD = lumaPlane(srcDest[0], srcDest[1], srcDest[2]);
  const lumRD = lumaPlane(resDest[0], resDest[1], resDest[2]);
  const grid = subsampleGrid(tileGrid(lumS.w, lumS.h, 1024), 16);
  const fixes: FixSurvival[] = [];

  const gridS = detectBlockGrid(lumSD);
  const gridR = detectBlockGrid(lumRD);
  const stepS = Math.max(gridS.stepX, gridS.stepY) * 255;
  const stepR = Math.max(gridR.stepX, gridR.stepY) * 255;
  const sigS = noiseModel(noiseMap(lumSD, 32)).globalSigma * 255;
  const sigR = noiseModel(noiseMap(lumRD, 32)).globalSigma * 255;

  for (const f of findings) {
    if (!f.enabled) continue;
    switch (f.kind) {
      case 'periodic': {
        const amp = (p: Plane): number => {
          const m = measurePattern(p, grid, f.details.u, f.details.v, { refine: false, searchGrid: grid, measureGrid: grid });
          return Math.max(m.amplitude, m.amplitudeP90) * 255;
        };
        const a0 = amp(lumS);
        const a1 = amp(lumSD);
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
        const g0 = detectBlockGrid(lumS);
        const s0 = Math.max(g0.stepX, g0.stepY) * 255;
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
        const sig0 = noiseModel(noiseMap(lumS, 32)).globalSigma * 255;
        const target = f.details.reduction > 0 ? sigS * (1 - f.details.reduction) : sigS;
        // for levelling (reduction 0) the signature is unevenness; for reduction it is the grain itself
        let retained: number;
        let note: string;
        if (f.details.reduction > 0) {
          retained = sigS > 0.05 ? Math.max(0, Math.min(1, (sigS - sigR) / Math.max(1e-6, sigS - target))) : 1;
          note = retained >= 0.8 ? 'grain stays reduced.' : retained >= 0.35 ? 'some grain shows again; the re-save adds noise of its own.' : 'the re-save adds about as much noise as was removed.';
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
        const c0 = cn(source);
        const c1 = cn(srcDest);
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
        const h0 = rg(lumS);
        const h1 = rg(lumSD);
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
      case 'hotpixels': {
        const idx = f.details.indices;
        const w = lumS.w;
        const still = (p: Plane): number => {
          let n = 0;
          for (const i of idx) {
            const x = i % w;
            const y = (i - x) / w;
            if (x < 1 || y < 1 || x >= w - 1 || y >= p.h - 1) continue;
            if (Math.abs(p.data[i] - median9(p.data, i, w)) > 0.06) n++;
          }
          return n;
        };
        const n1 = still(lumSD);
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
