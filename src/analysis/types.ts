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
  amplitude: number; // luma, 0..1
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
  sourceQuality: number | null;
  sourceSubsampling: string | null;
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
}

export interface HotPixelDetails {
  count: number;
  sample: { x: number; y: number; value: number; replacement: number; z: number }[]; // first N for display
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
}

export const defaultAnalysisOptions: AnalysisOptions = {
  zThreshold: 6,
  minRadiusBins: 6,
  maxPeaks: 40,
  tileSize: 1024,
  maxWelchTiles: 36,
  noiseBlock: 32,
  noiseThresholdDb: 3,
  linearLight: false,
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
