/**
 * Where the photo is going decides how it is written. A destination profile
 * sets the encoder (quality, tables, chroma, fixed-point passes) and the
 * re-save the check simulates ("what a downloader gets").
 *
 * Reddit re-encodes every direct upload at quality 75, 4:2:0, standard
 * tables, same pixel grid, metadata stripped, no resize below 8192 px. Most
 * sharing apps land between quality 67 and 92 at 4:2:0, so a file that is a
 * fixed point at quality 75 comes through nearly all of them unchanged.
 */

import type { JpegExportOptions } from '../io/encode';

export type ProfileId = 'reddit' | 'master' | 'custom';

export interface DestinationProfile {
  id: ProfileId;
  label: string;
  hint: string;
  /** how the file is written */
  encode: JpegExportOptions;
  /** the re-save the check simulates, on both the result and the untouched source */
  check: { label: string; options: JpegExportOptions };
  /** clean-up level the profile suggests, if any */
  suggestedLevel?: 'standard' | 'strong';
}

const redditEncode: JpegExportOptions = {
  quality: 75,
  chroma: '4:2:0',
  progressive: false,
  trellis: false,
  chromaQuality: null,
  tables: 'annex-k',
  fixedPointPasses: 4,
};

export const PROFILES: Record<ProfileId, DestinationProfile> = {
  reddit: {
    id: 'reddit',
    label: 'Reddit and sharing apps',
    hint: 'Written at Reddit’s own quality (75) and settled so their re-save changes nothing. Also holds up through WhatsApp, Instagram, Facebook, Telegram and X.',
    encode: redditEncode,
    check: { label: 'Reddit (quality 75, same grid)', options: { ...redditEncode, fixedPointPasses: 0 } },
    suggestedLevel: 'strong',
  },
  master: {
    id: 'master',
    label: 'Master copy',
    hint: 'Quality 95, full colour detail, standard tables so the finest detail (grain, the sensor grid) is kept. Keep this one; the check still shows what Reddit would do to it.',
    // standard tables and no trellis: trellis and the tuned tables drop the
    // finest detail, which is where a sensor's 2×2 grid and fine grain live
    encode: { quality: 95, chroma: '4:4:4', progressive: true, trellis: false, chromaQuality: null, tables: 'annex-k', fixedPointPasses: 0 },
    check: { label: 'Reddit (quality 75, same grid)', options: { ...redditEncode, fixedPointPasses: 0 } },
  },
  custom: {
    id: 'custom',
    label: 'Custom',
    hint: 'Your own settings. The check simulates a re-save at the same quality on the same grid.',
    encode: { quality: 85, chroma: '4:2:0', progressive: false, trellis: false, chromaQuality: null, tables: 'annex-k', fixedPointPasses: 3 },
    check: { label: 'a re-save at the chosen quality', options: { quality: 85, chroma: '4:2:0', progressive: false, trellis: false, chromaQuality: null, tables: 'annex-k', fixedPointPasses: 0 } },
  },
};

/** For a custom profile the check follows the encode settings. */
export function resolveProfile(id: ProfileId, custom: JpegExportOptions): DestinationProfile {
  if (id !== 'custom') return PROFILES[id];
  return {
    ...PROFILES.custom,
    encode: custom,
    check: { label: `a re-save at quality ${custom.quality}, ${custom.chroma}`, options: { ...custom, trellis: false, tables: 'annex-k', fixedPointPasses: 0 } },
  };
}
