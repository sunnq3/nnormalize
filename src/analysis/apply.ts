/**
 * Turn a finding into a delta layer: one Float32 plane per channel such that
 * corrected = original + delta. Every method is applied per channel with the
 * channel's own measurement, because colour moiré and chroma noise differ
 * between channels even when the frequency is shared.
 */

import { deblockDelta } from '../dsp/blocks';
import { parametricDeltaGlobal, parametricDeltaTiled, spectralDelta, type SpectralTarget } from '../dsp/filter';
import { findHotPixels } from '../dsp/hotpixels';
import { denoiseDelta, noiseMap, noiseModel, regrainDelta } from '../dsp/noise';
import { measurePattern } from '../dsp/pattern';
import { makePlane, type Plane } from '../dsp/plane';
import type { TileGrid } from '../dsp/tiles';
import type { Analysis, Finding, PeriodicFinding } from './types';

export type Progress = (fraction: number) => void;

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
  orders.forEach((o, k) => {
    const m = measurePattern(ch, grid, o.u, o.v, { refine: false });
    const delta = f.method === 'parametric' ? parametricDeltaGlobal(w, h, o.u, o.v, m.amplitude, m.phase) : parametricDeltaTiled(w, h, grid, m);
    addInto(out, delta);
    progress((k + 1) / orders.length);
  });
  return out;
}

export function computeDelta(planes: [Plane, Plane, Plane], analysis: Analysis, finding: Finding, progress: Progress = () => {}): [Plane, Plane, Plane] {
  const grid: TileGrid = { size: analysis.tile.size, hop: analysis.tile.hop, xs: analysis.tile.xs, ys: analysis.tile.ys };
  const out: Plane[] = [];
  planes.forEach((ch, ci) => {
    const sub = (fr: number): void => progress((ci + fr) / 3);
    switch (finding.kind) {
      case 'periodic':
        out.push(periodicDelta(ch, grid, finding, sub));
        break;
      case 'jpeg-grid': {
        const g = finding.details;
        out.push(
          deblockDelta(ch, {
            offsetX: g.offsetX,
            offsetY: g.offsetY,
            strengthX: g.strengthX,
            strengthY: g.strengthY,
            confidenceX: 0,
            confidenceY: 0,
            detected: true,
          }),
        );
        sub(1);
        break;
      }
      case 'noise': {
        const map = noiseMap(ch, finding.details.block);
        const model = noiseModel(map);
        const dn = denoiseDelta(ch, map, model, { thresholdDb: Math.max(1.5, finding.details.thresholdDb / 2) });
        const rg = regrainDelta(ch, map, model, { thresholdDb: Math.max(1.5, finding.details.thresholdDb / 2), seed: 0x5eed + ci });
        addInto(dn.delta, rg.delta);
        out.push(dn.delta);
        sub(1);
        break;
      }
      case 'hotpixels': {
        const map = noiseMap(ch, 32);
        const model = noiseModel(map);
        out.push(findHotPixels(ch, model).delta);
        sub(1);
        break;
      }
    }
  });
  return out as [Plane, Plane, Plane];
}
