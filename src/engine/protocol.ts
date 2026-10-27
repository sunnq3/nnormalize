/**
 * Message protocol between the UI and the engine worker. The worker owns the
 * full-resolution image and every delta layer; the UI only ever receives
 * small arrays, rendered pixels and plain-data results.
 */

import type { Analysis, AnalysisOptions, Finding } from '../analysis/types';
import type { JpegExportOptions } from '../io/encode';
import type { SignatureStatus } from './signature';
import type { SurvivalReport } from '../analysis/survival';
import type { ProfileId } from './profiles';
import type { SourceMeta } from '../io/decode';

export interface LayerSetting {
  id: string;
  strength: number; // 0..2
}

/** 'posted' = our result after the destination's re-save; 'posted-source' = the untouched photo after the same re-save */
export type ViewMode = 'original' | 'result' | 'difference' | 'mask' | 'posted' | 'posted-source';

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
  profile: ProfileId;
  profileLabel: string;
  /** fixed-point encoding: how many passes it took and whether the bytes settled */
  passes: number;
  stable: boolean;
  /** block settling on the destination's coefficient grid, when it was done */
  settled: { quality: number; changedBlocks: number; blocks: number; meanChangeLevels: number } | null;
  sensor: SensorReport;
  /** grain put back was scaled by this so the downloader's copy carries the amount set (1 = no scaling needed) */
  grainScale: number;
  /** what the check simulated */
  checkLabel: string;
  /** per-fix survival through the simulated re-save, plus the before/after a downloader sees */
  survival: SurvivalReport;
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

/** The sensor bake at the end: off, sharp (Malvar–He–Cutler, subtle noise), or matching the original's grid. */
export type BakeSetting = 'off' | 'clean' | 'sharp' | 'match';

/** The sensor's own signature (PRNU) in the untouched photo, the result, the file and the posted copy: correlation with the learned map. */
export interface SignatureReport {
  photos: number;
  reliability: number;
  /** correlation of each copy's noise with the map (0 = none; a photo from this camera reads well above its noise, z ≫ 5) */
  original: number;
  result: number;
  inFile: number;
  posted: number;
  /** z-scores of the same */
  zOriginal: number;
  zPosted: number;
  /** the gain the map was applied at (0 = not applied) */
  gain: number;
  restored: boolean;
}

/** The sensor bake: whether it ran, the grid used, and how clearly the 2×2 sensor pattern shows (0…1). */
export interface SensorReport {
  signature?: SignatureReport;
  baked: boolean;
  /** sensor noise added at the photosites, as its strength at mid grey in levels */
  noiseLevels: number;
  /** 'mhc' (Malvar–He–Cutler) or 'bilinear' */
  method: string;
  layout: string;
  detected: boolean;
  original: number;
  inFile: number;
  posted: number;
}

/** The live check: survival on a crop, written once at the destination's settings. */
export interface LiveReport {
  survival: SurvivalReport;
  checkLabel: string;
  region: { x0: number; y0: number; w: number; h: number; full: boolean };
  /** file size of the whole photo, estimated from the crop */
  bytesEstimate: number;
  grainScale: number;
  sensor: SensorReport;
  ms: number;
}

export interface Budget {
  maxPeakDb: number; // no periodic finding above this excess (default 12) ...
  minPeakLevels: number; // ... unless its amplitude is below this many 8-bit levels (default 1: under one level nobody sees it)
  maxBlockStrength: number; // default 0.05
  minEncodePsnr: number; // default 40 dB (≈ 2.5 levels rms; a q95 encode of a grainy photo lands near here because it discards fine grain)
}

export const defaultBudget: Budget = { maxPeakDb: 12, minPeakLevels: 1.0, maxBlockStrength: 0.05, minEncodePsnr: 40 };

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
  | {
      type: 'export';
      id: number;
      layers: LayerSetting[];
      /** the findings as the person has set them (ticks, strengths, methods); the worker's own copy does not see those edits */
      findings: Finding[];
      master: number;
      profile: ProfileId;
      options: JpegExportOptions;
      budget: Budget;
      analysisOptions: Partial<AnalysisOptions>;
      bake: BakeSetting;
      /** put the sensor's own learned signature back */
      signature: boolean;
    }
  | { type: 'verifyResult'; id: number; layers: LayerSetting[]; master: number; analysisOptions: Partial<AnalysisOptions> }
  | { type: 'liveCheck'; id: number; layers: LayerSetting[]; findings: Finding[]; master: number; profile: ProfileId; options: JpegExportOptions; bake: BakeSetting; signature: boolean }
  | { type: 'signatureLearn'; id: number; file: Blob }
  | { type: 'signatureStatus'; id: number }
  | { type: 'signatureExport'; id: number }
  | { type: 'signatureImport'; id: number; bytes: Uint8Array }
  | { type: 'signatureForget'; id: number }
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
  | { type: 'live'; id: number; report: LiveReport }
  | { type: 'mask'; id: number; w: number; h: number; data: Uint8Array }
  | { type: 'finding'; id: number; finding: Finding }
  | { type: 'signature'; id: number; status: SignatureStatus; learned?: { accepted: boolean; reason: string; usable: number; key: string | null } }
  | { type: 'bytes'; id: number; bytes: Uint8Array | null }
  | { type: 'error'; id: number; message: string };
