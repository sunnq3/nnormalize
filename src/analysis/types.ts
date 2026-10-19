/**
 * Plain-data types shared between the engine worker and the UI. Everything
 * here must survive structured cloning.
 */

export type FindingKind = 'periodic' | 'jpeg-grid' | 'noise' | 'hotpixels';

export type PeriodicMethod = 'parametric' | 'parametric-tiled' | 'flatten' | 'notch';
export type Method = PeriodicMethod | 'deblock' | 'normalise' | 'inpaint';

export interface HarmonicInfo {
  order: number;
  u: number;
  v: number;
  amplitude: number;
  excessDb: number;
  widthBins: number;
}

export interface PeriodicDetails {
  u: number;
  v: number;
  period: number; // pixels
  angleDeg: number; // 0 = vertical stripes, 90 = horizontal stripes
  lpi: number | null; // lines per inch when DPI is known
  amplitude: number; // luma, 0..1, median over tiles
  amplitudeP90: number; // 90th percentile over tiles
  phase: number;
  excessDb: number;
  z: number;
  widthBins: number;
  stationary: boolean;
  coherence: number;
  amplitudeCv: number;
  harmonics: HarmonicInfo[];
  presence: Float32Array; // tilesY x tilesX, 0..1
  tilesX: number;
  tilesY: number;
  onAxis: boolean;
  jpegGridAligned: boolean;
  category: 'moire' | 'horizontal-lines' | 'vertical-lines' | 'screen' | 'jpeg-harmonic';
}

export interface JpegGridDetails {
  offsetX: number;
  offsetY: number;
  strengthX: number;
  strengthY: number;
  /** mean boundary step above the texture, 0..1 plane units (×255 for levels) */
  stepLevels: number;
  sourceQuality: number | null;
  sourceSubsampling: string | null;
  /** deblocker tuning: steps up to this size (0..1) are treated as block edges, if they exceed activityRatio × local texture */
  maxStep: number;
  activityRatio: number;
  span: 4 | 8;
}

export interface NoiseDetails {
  block: number;
  bw: number;
  bh: number;
  ratioDb: Float32Array; // bh x bw
  noisierFraction: number;
  smootherFraction: number;
  thresholdDb: number;
  globalSigma: number; // luma, 0..1 units
  curve: Float32Array; // sigma vs intensity bin
  highBandDb: number; // spectral tilt of the noise floor: ~0 white, negative = previously smoothed, positive = sharpened
  spectralSigma: number; // sigma implied by the spectrum's floor
  /** how much of the photo's own noise to remove everywhere: 0 only levels it, 0.6 removes 60 % */
  reduction: number;
}

export interface HotPixelDetails {
  count: number;
  sample: { x: number; y: number; value: number; replacement: number; z: number }[]; // first N for display
  /** pixel indices (y*w + x) of every detected outlier, for the per-channel fix */
  indices: Uint32Array;
}

export interface FindingBase {
  id: string;
  kind: FindingKind;
  label: string;
  description: string;
  severity: number; // 0..1
  extent: number; // fraction of the frame affected, 0..1
  enabled: boolean;
  confidence: 'high' | 'medium' | 'low';
  note: string;
  strength: number; // 0..2, 1 = as computed
  method: Method;
}

export interface PeriodicFinding extends FindingBase {
  kind: 'periodic';
  method: PeriodicMethod;
  details: PeriodicDetails;
}
export interface JpegGridFinding extends FindingBase {
  kind: 'jpeg-grid';
  method: 'deblock';
  details: JpegGridDetails;
}
export interface NoiseFinding extends FindingBase {
  kind: 'noise';
  method: 'normalise';
  details: NoiseDetails;
}
export interface HotPixelFinding extends FindingBase {
  kind: 'hotpixels';
  method: 'inpaint';
  details: HotPixelDetails;
}

export type Finding = PeriodicFinding | JpegGridFinding | NoiseFinding | HotPixelFinding;

export interface SpectrumView {
  size: number;
  logPower: Float32Array; // shifted, log10
  background: Float32Array;
  z: Float32Array;
  tiles: number;
}

export interface AnalysisOptions {
  zThreshold: number; // default 6
  minRadiusBins: number; // default 6
  maxPeaks: number; // default 40
  tileSize: number; // default 1024
  maxWelchTiles: number; // default 36
  noiseBlock: number; // default 32
  noiseThresholdDb: number; // default 3
  linearLight: boolean; // analyse in linear light (sRGB decode)
  /** peaks whose fitted amplitude is below this (8-bit levels) are left unticked; below half of it they are dropped */
  minAmplitudeLevels: number; // default 0.5
  /** verification mode: fewer tiles, no presence maps, no hot-pixel search */
  quick: boolean;
  /** how far to go: gentle only touches what is clearly wrong, maximum smooths grain and block edges everywhere */
  level: CleanupLevel;
}

export type CleanupLevel = 'gentle' | 'standard' | 'strong' | 'maximum';

export interface LevelProfile {
  zThreshold: number;
  minAmplitudeLevels: number;
  noiseThresholdDb: number;
  noiseReduction: number; // 0..1
  noiseAlwaysOn: boolean;
  deblockMaxStep: number;
  deblockActivityRatio: number;
  deblockSpan: 4 | 8;
  deblockAlwaysOn: boolean;
  enableWeakPeaks: boolean;
}

export const LEVELS: Record<CleanupLevel, LevelProfile> = {
  gentle: { zThreshold: 7, minAmplitudeLevels: 1, noiseThresholdDb: 4, noiseReduction: 0, noiseAlwaysOn: false, deblockMaxStep: 0.06, deblockActivityRatio: 1.8, deblockSpan: 4, deblockAlwaysOn: false, enableWeakPeaks: false },
  standard: { zThreshold: 6, minAmplitudeLevels: 0.5, noiseThresholdDb: 3, noiseReduction: 0, noiseAlwaysOn: false, deblockMaxStep: 0.08, deblockActivityRatio: 1.5, deblockSpan: 4, deblockAlwaysOn: false, enableWeakPeaks: false },
  strong: { zThreshold: 5, minAmplitudeLevels: 0.3, noiseThresholdDb: 2, noiseReduction: 0.5, noiseAlwaysOn: true, deblockMaxStep: 0.15, deblockActivityRatio: 1.2, deblockSpan: 8, deblockAlwaysOn: true, enableWeakPeaks: true },
  maximum: { zThreshold: 4.5, minAmplitudeLevels: 0.2, noiseThresholdDb: 1.5, noiseReduction: 0.8, noiseAlwaysOn: true, deblockMaxStep: 0.25, deblockActivityRatio: 1.0, deblockSpan: 8, deblockAlwaysOn: true, enableWeakPeaks: true },
};

export const defaultAnalysisOptions: AnalysisOptions = {
  zThreshold: 6,
  minRadiusBins: 6,
  maxPeaks: 40,
  tileSize: 1024,
  maxWelchTiles: 36,
  noiseBlock: 32,
  noiseThresholdDb: 3,
  linearLight: false,
  minAmplitudeLevels: 0.5,
  quick: false,
  level: 'standard',
};

export interface Analysis {
  w: number;
  h: number;
  options: AnalysisOptions;
  tile: { size: number; hop: number; xs: number[]; ys: number[] };
  spectrum: SpectrumView;
  findings: Finding[];
  blockGrid: JpegGridDetails & { detected: boolean };
  noise: NoiseDetails;
  timings: Record<string, number>;
  /** What this analysis does NOT look for, stated plainly for the report. */
  notChecked: string[];
}

export const NOT_CHECKED = [
  'dust spots, hairs or scratches larger than a pixel',
  'lens flare, ghosting and chromatic aberration',
  'motion blur, focus and sharpness',
  'blown highlights and crushed shadows',
  'colour casts and white balance',
  'demosaicing errors that are not periodic',
];
