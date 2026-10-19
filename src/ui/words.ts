/**
 * Plain-language wording for findings. The engine keeps its technical text
 * for the JSON report; people see this.
 */

import type { Finding } from '../analysis/types';
import { fmt } from './dom';

export function plainLabel(f: Finding): string {
  switch (f.kind) {
    case 'periodic':
      switch (f.details.category) {
        case 'horizontal-lines':
          return 'Horizontal stripes';
        case 'vertical-lines':
          return 'Vertical stripes';
        case 'screen':
          return 'Fine dot screen';
        case 'jpeg-harmonic':
          return 'Grid left by compression';
        default:
          return f.label === 'Added by hand' ? 'Pattern you added' : 'Repeating pattern (moiré)';
      }
    case 'jpeg-grid':
      return 'Blocky compression';
    case 'noise':
      return f.details.reduction > 0 ? 'Grain and noise' : 'Uneven grain';
    case 'hotpixels':
      return 'Dead or stuck pixels';
  }
}

/** One line a person can act on: what it looks like and how strong it is. */
export function plainDescription(f: Finding): string {
  switch (f.kind) {
    case 'periodic': {
      const d = f.details;
      const where = d.stationary ? 'across the whole photo' : f.extent >= 0.9 ? 'everywhere, varying in strength' : `in about ${fmt(f.extent * 100, 0)} % of the photo`;
      const strength = Math.max(d.amplitude, d.amplitudeP90) * 255;
      const how = strength >= 8 ? 'clearly visible' : strength >= 2 ? 'visible up close' : strength >= 0.5 ? 'faint' : 'too faint to see';
      return `repeats every ${fmt(d.period, 1)} px at ${fmt(d.angleDeg, 0)}°, ${fmt(strength, 1)} levels strong (${how}), ${where}`;
    }
    case 'jpeg-grid': {
      const lv = f.details.stepLevels;
      const how = lv >= 3 ? 'heavy' : lv >= 1 ? 'noticeable' : lv >= 0.5 ? 'faint' : 'invisible';
      return `${how} 8 × 8 block edges, steps of ${fmt(lv, 1)} levels${f.details.sourceQuality ? `; the file was last saved at about quality ${fmt(f.details.sourceQuality, 0)}` : ''}`;
    }
    case 'noise': {
      const d = f.details;
      const g = d.globalSigma * 255;
      const how = g >= 6 ? 'heavy' : g >= 2.5 ? 'moderate' : g >= 1 ? 'light' : 'very light';
      if (d.reduction > 0) return `${how} grain (${fmt(g, 1)} levels); this fix removes ${Math.round(d.reduction * 100)} % of it everywhere`;
      const uneven = d.noisierFraction + d.smootherFraction;
      return `${how} grain (${fmt(g, 1)} levels); ${uneven >= 0.05 ? `${fmt(uneven * 100, 0)} % of the photo is grainier or smoother than the rest` : 'even across the photo'}`;
    }
    case 'hotpixels':
      return `${f.details.count} single pixel${f.details.count === 1 ? '' : 's'} that don't match their surroundings`;
  }
}

/** What the fix does, for the selected-fix panel. */
export function plainFix(f: Finding): string {
  switch (f.kind) {
    case 'periodic':
      switch (f.method) {
        case 'parametric':
          return 'The pattern is measured exactly and subtracted; nothing else in the photo changes.';
        case 'parametric-tiled':
          return 'The pattern is measured and subtracted region by region, following how it changes across the photo.';
        case 'flatten':
          return 'The pattern’s frequency is smoothed down to the level of its surroundings.';
        case 'notch':
          return 'The pattern’s frequency is removed entirely; real detail at that same frequency goes with it.';
      }
      return '';
    case 'jpeg-grid':
      return 'Steps along the 8 × 8 block edges are blended over four pixels; real edges are left alone.';
    case 'noise':
      return f.details.reduction > 0
        ? 'Grain is reduced everywhere with an edge-preserving smoother, by the amount you set.'
        : 'Grain is levelled: reduced where it is heavier than the rest of the photo, matched grain added where an earlier tool scrubbed it.';
    case 'hotpixels':
      return 'Each pixel is replaced by the median of its eight neighbours.';
  }
}

export const LEVEL_WORDS: Record<string, { label: string; hint: string }> = {
  gentle: { label: 'Gentle', hint: 'Only what is clearly wrong. Grain and faint patterns are left alone.' },
  standard: { label: 'Standard', hint: 'Fix patterns, block edges and dead pixels; level the grain without removing it.' },
  strong: { label: 'Strong', hint: 'Everything above, plus faint patterns, firmer deblocking, and half the grain removed everywhere.' },
  maximum: { label: 'Maximum', hint: 'As far as it goes: 80 % of the grain removed, the heaviest deblocking. Expect a smoother look.' },
};
