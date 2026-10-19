/**
 * Message protocol between the UI and the engine worker. The worker owns the
 * full-resolution image and every delta layer; the UI only ever receives
 * small arrays, rendered pixels and plain-data results.
 */

import type { Analysis, AnalysisOptions, Finding } from '../analysis/types';
import type { JpegExportOptions } from '../io/encode';
import type { SourceMeta } from '../io/decode';

export interface LayerSetting {
  id: string;
  strength: number; // 0..2
}

export type ViewMode = 'original' | 'result' | 'difference' | 'mask';

export interface RenderRequest {
  region: { kind: 'preview' } | { kind: 'crop'; x0: number; y0: number; w: number; h: number };
  mode: ViewMode;
  layers: LayerSetting[];
  master: number;
  diffGain: number; // multiplier for the difference view
  maskId: string | null; // which mask to show in 'mask' mode
}

export interface RenderResult {
  w: number;
  h: number;
  rgba: Uint8ClampedArray<ArrayBuffer>;
}

export interface LoadedInfo {
  w: number;
  h: number;
  meta: SourceMeta;
  preview: { w: number; h: number; scale: number };
  maskSize: { w: number; h: number };
}

export interface LayerStats {
  id: string;
  rms: number; // of the luma delta, in 8-bit levels
  max: number;
  ms: number;
}

export interface RoundTripReport {
  /** quantisation error of the encode alone: PSNR between composed float result and decoded JPEG */
  encodePsnr: number;
  encodeRmsLevels: number;
  bytes: number;
  quality: number;
  chroma: string;
  /** findings re-detected on the decoded output */
  residual: Finding[];
  residualPeriodic: number;
  blockGrid: { detected: boolean; strength: number; stepLevels: number };
  noise: { globalSigmaLevels: number; expectedLevels: number };
  /** fixes the verification suggests adding, e.g. a block grid the source analysis could not see under the interference */
  suggested: Finding[];
  pass: boolean;
  reasons: string[];
  budget: Budget;
}

export interface Budget {
  maxPeakDb: number; // no periodic finding above this excess (default 12) ...
  minPeakLevels: number; // ... unless its amplitude is below this many 8-bit levels (default 0.5)
  maxBlockStrength: number; // default 0.05
  minEncodePsnr: number; // default 40 dB (≈ 2.5 levels rms; a q95 encode of a grainy photo lands near here because it discards fine grain)
}

export const defaultBudget: Budget = { maxPeakDb: 12, minPeakLevels: 0.5, maxBlockStrength: 0.05, minEncodePsnr: 40 };

export interface SynthRequest {
  w: number;
  h: number;
  seed: number;
  sinusoids: { u: number; v: number; amplitude: number; phase: number }[];
  noiseSigma: number;
  blocks: boolean; // simulate an 8x8 grid
  hotPixels: number;
}

export type Request =
  | { type: 'load'; id: number; file: Blob }
  | { type: 'synth'; id: number; spec: SynthRequest }
  | { type: 'analyze'; id: number; options: Partial<AnalysisOptions> }
  | { type: 'delta'; id: number; finding: Finding }
  | { type: 'setMask'; id: number; findingId: string; w: number; h: number; data: Uint8Array }
  | { type: 'render'; id: number; req: RenderRequest }
  | { type: 'export'; id: number; layers: LayerSetting[]; master: number; options: JpegExportOptions; budget: Budget; analysisOptions: Partial<AnalysisOptions> }
  | { type: 'verifyResult'; id: number; layers: LayerSetting[]; master: number; analysisOptions: Partial<AnalysisOptions> }
  | { type: 'presenceMask'; id: number; findingId: string }
  | { type: 'addPeak'; id: number; u: number; v: number }
  | { type: 'luminosityMask'; id: number; findingId: string; lo: number; hi: number; soft: number };

export type Response =
  | { type: 'progress'; id: number; stage: string; fraction: number }
  | { type: 'loaded'; id: number; info: LoadedInfo }
  | { type: 'analysis'; id: number; analysis: Analysis }
  | { type: 'delta'; id: number; stats: LayerStats }
  | { type: 'ok'; id: number }
  | { type: 'render'; id: number; result: RenderResult }
  | { type: 'export'; id: number; jpeg: Uint8Array; report: RoundTripReport }
  | { type: 'verify'; id: number; analysis: Analysis }
  | { type: 'mask'; id: number; w: number; h: number; data: Uint8Array }
  | { type: 'finding'; id: number; finding: Finding }
  | { type: 'error'; id: number; message: string };
