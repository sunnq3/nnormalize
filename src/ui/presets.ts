import type { AnalysisOptions } from '../analysis/types';

export interface Preset {
  id: string;
  label: string;
  hint: string;
  options: Partial<AnalysisOptions>;
}

/** Same engine, different thresholds, chosen for where the photo came from. */
export const PRESETS: Preset[] = [
  { id: 'default', label: 'General photo', hint: 'Balanced thresholds.', options: {} },
  {
    id: 'scan',
    label: 'Scanned print or magazine',
    hint: 'Halftone screens are the point: lower the peak threshold and treat faint screens as real.',
    options: { zThreshold: 5, minAmplitudeLevels: 0.3, maxPeaks: 48 },
  },
  {
    id: 'screen',
    label: 'Photo of a screen',
    hint: 'Strong moiré from the pixel grid, usually with harmonics; same floor, more peaks allowed.',
    options: { zThreshold: 6, maxPeaks: 48 },
  },
  {
    id: 'phone',
    label: 'Old phone JPEG',
    hint: 'Compression damage dominates; ignore sub-visible spectral spikes the encoder left behind.',
    options: { zThreshold: 7, minAmplitudeLevels: 0.8 },
  },
  {
    id: 'film',
    label: 'Scanned film',
    hint: 'Grain is signal, not noise; raise the noise-discrepancy threshold so it is not levelled away.',
    options: { noiseThresholdDb: 5, minAmplitudeLevels: 0.5 },
  },
  {
    id: 'linear',
    label: 'Measure in linear light',
    hint: 'Decode sRGB before analysing: interference added before gamma reads truer and harmonics shrink.',
    options: { linearLight: true },
  },
];
