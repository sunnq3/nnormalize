/**
 * Measure one periodic pattern across every tile of the image: its precise
 * frequency, its amplitude and phase per tile (so we know where it lives and
 * whether it is the same pattern everywhere), and a single global estimate.
 */

import { extractTile, type Plane } from './plane';
import { measureAt, phaseToGlobal, refinePeak, windowedTile, wrapPhase, type RefineResult } from './refine';
import type { TileGrid } from './tiles';
import { window2d, type WindowKind } from './window';

export interface TileMeasure {
  x0: number;
  y0: number;
  amplitude: number;
  phase: number; // global-frame phase
  power: number;
}

export interface PatternMeasure {
  u: number;
  v: number;
  amplitude: number; // robust global amplitude (median over tiles)
  phase: number; // amplitude-weighted circular mean of per-tile global phases
  /** Mean resultant length of the phases: 1 = identical everywhere, 0 = random. */
  phaseCoherence: number;
  /** Coefficient of variation of the per-tile amplitudes. */
  amplitudeCv: number;
  stationary: boolean;
  tilesX: number;
  tilesY: number;
  perTile: TileMeasure[]; // row-major over the tile grid
  /** Per-tile amplitude normalised to the maximum, row-major over the grid. */
  presence: Float32Array;
  refine: RefineResult;
  refinedOnTile: number;
}

function measureAll(plane: Plane, grid: TileGrid, u: number, v: number, kind: WindowKind): TileMeasure[] {
  const n = grid.size;
  const win = window2d(kind, n, n);
  const out: TileMeasure[] = [];
  const tile = new Float64Array(n * n);
  const tw = new Float64Array(n * n);
  for (const y0 of grid.ys) {
    for (const x0 of grid.xs) {
      extractTile(plane, x0, y0, n, n, tile);
      windowedTile(tile, win, n, n, tw);
      const m = measureAt(tw, n, n, win.sum, u, v);
      out.push({ x0, y0, amplitude: m.amplitude, phase: phaseToGlobal(m.phase, u, v, x0, y0), power: m.power });
    }
  }
  return out;
}

export interface MeasureOptions {
  window?: WindowKind;
  refine?: boolean; // default true
  coherenceThreshold?: number; // default 0.9
  cvThreshold?: number; // default 0.35
  /** grid for the first (search) pass; defaults to `grid`. A subsampled grid makes the search cheap. */
  searchGrid?: TileGrid;
  /** grid for the second (measurement) pass; defaults to `grid`. */
  measureGrid?: TileGrid;
}

export function measurePattern(plane: Plane, grid: TileGrid, u0: number, v0: number, opts: MeasureOptions = {}): PatternMeasure {
  const kind = opts.window ?? 'hann';
  const n = grid.size;
  const win = window2d(kind, n, n);
  const searchGrid = opts.searchGrid ?? grid;
  const measureGrid = opts.measureGrid ?? grid;

  // pass 1: where is it strongest?
  const first = measureAll(plane, searchGrid, u0, v0, kind);
  let bestI = 0;
  for (let i = 1; i < first.length; i++) if (first[i].power > first[bestI].power) bestI = i;
  const best = first[bestI];

  // refine on that tile
  let refine: RefineResult;
  if (opts.refine ?? true) {
    const tw = windowedTile(extractTile(plane, best.x0, best.y0, n, n), win, n, n);
    refine = refinePeak(tw, n, n, win.sum, u0, v0, { tileSize: n });
  } else {
    refine = { u: u0, v: v0, power: best.power, amplitude: best.amplitude, phase: 0, iterations: 0, converged: true };
  }
  const u = refine.u;
  const v = refine.v;

  // pass 2: amplitude and global phase everywhere at the refined frequency
  const perTile = measureAll(plane, measureGrid, u, v, kind);

  const amps = perTile.map((t) => t.amplitude).sort((a, b) => a - b);
  const medianAmp = amps.length & 1 ? amps[amps.length >> 1] : 0.5 * (amps[(amps.length >> 1) - 1] + amps[amps.length >> 1]);
  let meanAmp = 0;
  for (const a of amps) meanAmp += a;
  meanAmp /= amps.length;
  let varAmp = 0;
  for (const a of amps) varAmp += (a - meanAmp) ** 2;
  const cv = meanAmp > 0 ? Math.sqrt(varAmp / amps.length) / meanAmp : 0;

  let cx = 0;
  let cy = 0;
  let wsum = 0;
  for (const t of perTile) {
    cx += t.amplitude * Math.cos(t.phase);
    cy += t.amplitude * Math.sin(t.phase);
    wsum += t.amplitude;
  }
  const phase = Math.atan2(cy, cx);
  const coherence = wsum > 0 ? Math.hypot(cx, cy) / wsum : 0;

  let maxAmp = 0;
  for (const t of perTile) if (t.amplitude > maxAmp) maxAmp = t.amplitude;
  const presence = new Float32Array(perTile.length);
  for (let i = 0; i < perTile.length; i++) presence[i] = maxAmp > 0 ? perTile[i].amplitude / maxAmp : 0;

  const stationary = coherence >= (opts.coherenceThreshold ?? 0.9) && cv <= (opts.cvThreshold ?? 0.35);

  return {
    u,
    v,
    amplitude: medianAmp,
    phase: wrapPhase(phase),
    phaseCoherence: coherence,
    amplitudeCv: cv,
    stationary,
    tilesX: measureGrid.xs.length,
    tilesY: measureGrid.ys.length,
    perTile,
    presence,
    refine,
    refinedOnTile: bestI,
  };
}

/** Period in pixels and orientation in degrees (0 = vertical stripes, 90 = horizontal stripes). */
export function describeFrequency(u: number, v: number): { radius: number; period: number; angleDeg: number } {
  const radius = Math.hypot(u, v);
  return { radius, period: radius > 0 ? 1 / radius : Infinity, angleDeg: (Math.atan2(-v, u) * 180) / Math.PI };
}
