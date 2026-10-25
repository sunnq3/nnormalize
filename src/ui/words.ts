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
    case 'chroma':
      return 'Colour blotches';
    case 'ringing':
      return 'Halos next to edges';
    case 'grain':
      return 'Put back grain';
    case 'detail':
      return f.details.mode === 'sharp' ? 'Oversharpened' : 'Too smooth (fine detail missing)';
    case 'banding':
      return 'Banding in smooth areas';
    case 'texture':
      return 'Smeared fine texture';
    case 'glow':
      return 'Glow around subjects';
    case 'tone':
      return 'Flat, low contrast';
    case 'fringes':
      return 'False colour on fine detail';
    case 'edges':
      return Math.max(f.details.overshoot, f.details.undershoot) >= 0.04 ? 'Sharpening outlines and blur' : 'Blurry edges';
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
    case 'chroma': {
      const lv = Math.max(f.details.sigmaCbLevels, f.details.sigmaCrLevels);
      const how = lv >= 3 ? 'heavy' : lv >= 1.5 ? 'noticeable' : 'light';
      return `${how} colour noise (${fmt(lv, 1)} levels) in ${fmt(f.details.fraction * 100, 0)} % of the photo; this fix removes ${Math.round(f.details.amount * 100)} % of it there`;
    }
    case 'ringing':
      return `noise next to hard edges is ${fmt(f.details.ratio, 1)}× the grain elsewhere; this fix smooths the band beside edges by ${Math.round(f.details.amount * 100)} %`;
    case 'grain':
      return `fine grain of ${fmt(f.details.targetLevels, 1)} levels added wherever the cleaned photo is smoother than that (the photo had ${fmt(f.details.baseLevels, 1)}), so it doesn't look like plastic`;
    case 'detail': {
      const d = f.details;
      if (d.mode === 'sharp') return `fine detail is ${fmt(10 ** (d.excessDb / 20), 1)}× stronger than a camera photo of this scene would have: outlines and crunchy texture from sharpening; this fix tones it down`;
      return `fine detail is at ${fmt(100 * 10 ** (-d.deficitDb / 20), 0)} % of what a camera photo of this scene would have: smoothed, denoised or upscaled; this fix brings it back by up to ${fmt(10 ** (d.maxBoostDb / 20), 1)}×`;
    }
    case 'texture': {
      const d = f.details;
      return `fine texture (hair, fabric, foliage, skin) is at ${fmt(d.share * 100, 0)} % of what a scene like this carries, clearly short in ${fmt(d.weakFraction * 100, 0)} % of the photo: noise reduction smeared it. This fix strengthens the texture that is still there; nothing is added`;
    }
    case 'glow': {
      const d = f.details;
      return `a soft ${fmt(d.glowLevels, 1)}-level glow hugs ${fmt(d.share * 100, 0)} % of the smooth edges (sky, walls next to subjects): the cut-out, pasted-on look phones' exposure blending leaves. This fix flattens it back onto the area around it`;
    }
    case 'tone': {
      const d = f.details;
      const sh = d.shadowLevels >= 8 ? `the darkest shadows only reach ${fmt(d.shadowLevels, 0)} levels (lifted; a camera's go down to about 0–4)` : 'the shadows reach black';
      return `${sh}; contrast between neighbouring areas is ${fmt(d.localContrast, 1)} levels. This fix brings back depth, about ${fmt((d.expectedRatio - 1) * 100 * d.amount, 0)} % more local contrast${d.shadowLevels >= 8 ? ', and deeper shadows' : ''}`;
    }
    case 'fringes': {
      const d = f.details;
      return `colour speckles and fringes on fine detail that don't follow its brightness (left by the camera filling in colours): ${fmt(d.detailLevels, 1)} levels there, against ${fmt(d.flatLevels, 1)} in smooth areas`;
    }
    case 'edges': {
      const d = f.details;
      const o = Math.max(d.overshoot, d.undershoot);
      const outl = o >= 0.04 ? `bright/dark outlines of ${fmt(o * 100, 0)} % of the edge's height (the pasted-on look)` : 'no outlines';
      const after =
        d.reachableWidthPx < d.widthPx * 0.95
          ? `This fix ${o >= 0.04 ? 'removes the outlines and ' : ''}sharpens the edges to ${fmt(d.reachableWidthPx, 1)} px`
          : o >= 0.04
            ? `This fix removes the outlines; the edges then measure ${fmt(d.reachableWidthPx, 1)} px, because the outlines were making them look narrower than the blur under them`
            : 'The blur is too strong to reduce without new outlines';
      return `measured on ${d.count} of the photo's sharpest edges: they spread over ${fmt(d.widthPx, 1)} px with ${outl}. ${after}`;
    }
    case 'banding': {
      const d = f.details;
      const how = d.strengthLevels >= 0.4 ? 'clearly visible' : d.strengthLevels >= 0.2 ? 'visible' : 'faint';
      return `${how} steps of ${d.stepLevels} level${d.stepLevels === 1 ? '' : 's'} in smooth gradients (skies, walls, shadows), in ${fmt(d.fraction * 100, 0)} % of the photo`;
    }
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
        : 'Grain is levelled: brought down where it is heavier than the rest of the photo. Cleaner areas are left as they are; nothing is added.';
    case 'hotpixels':
      return 'Each pixel is replaced by the median of its eight neighbours.';
    case 'chroma':
      return 'The colour layer is smoothed where it is blotchy, guided by brightness so colour never bleeds across an edge. Brightness detail is untouched.';
    case 'ringing':
      return 'Only the band next to hard edges is smoothed, with an edge-keeping filter; the edge itself and everything else stay as they are.';
    case 'detail':
      return f.details.mode === 'sharp'
        ? 'The detail bands that sharpening pushed up are brought back down to where a camera photo of this scene would sit. Coarse shapes and colour are untouched.'
        : 'The fine detail bands are raised toward where a camera photo of this scene would sit, judged against the photo’s own coarser detail. Only brightness detail changes; nothing past what a re-save keeps is boosted.';
    case 'texture':
      return 'Real scenes carry about as much detail at each scale as at the next coarser one. Where a textured area has much less fine detail than its coarser detail predicts, the fine detail already there is strengthened toward that share (at most ×2.5). Smooth areas, big edges and noise are left as they are, and no noise or pattern is added.';
    case 'glow':
      return 'Along each strong edge the profile across it is read out to 48 px. Where a side is smooth, its real level and gradient are fitted from further out, and the band near the edge that bulges away from it (exaggerating the edge, fading with distance) is set back onto that fit. Sides with texture or other edges are left alone.';
    case 'tone':
      return 'Phones blend a dark and a bright shot and lift the shadows, which packs everything into the middle. A local Laplacian filter expands the differences between neighbouring areas at every scale while leaving real edges at their height, so depth returns without halos; lifted shadows are brought back down with a soft toe. Brightness only; colours are kept.';
    case 'fringes':
      return 'On fine detail only, each pixel’s colour is pulled onto what its brightness predicts within a 5 × 5 neighbourhood: colour that follows brightness (a red letter on white) stays, colour that does not (speckles, zippers, fringes) goes. Brightness is untouched.';
    case 'edges':
      return 'Real edges are sharp steps, so the width and outlines your photo shows at its sharpest edges are what the camera added. The camera’s blur and its sharpening were measured there and fitted; at strong edges the sharpening is undone, and everywhere the blur is reduced as far as it can be without creating new outlines; in texture (hair, fabric, foliage) the camera’s sharpening is kept, since there it reads as detail. Where an edge borders a flat area (sky, wall, skin), any outline left is pulled back to that area’s own level; edges with texture on both sides keep their texture. Brightness only; colour is untouched.';
    case 'banding':
      return 'Inside the banded areas only, the steps are smoothed back into an even ramp with an edge-keeping filter (real edges stay), and a very fine dither is added so saving to 8 bits does not cut new steps.';
    case 'grain':
      return 'Photon-like grain, a little stronger in the shadows and fading out near white, with a touch of colour, is added only where the photo is now smoother than the amount set. Textured areas keep their own detail. Coarse enough to come through a re-save.';
  }
}

export const LEVEL_WORDS: Record<string, { label: string; hint: string }> = {
  gentle: { label: 'Gentle', hint: 'Only what is clearly wrong. Grain and faint patterns are left alone.' },
  standard: { label: 'Standard', hint: 'Fix patterns, block edges, dead pixels, outlines, glow and flat contrast; bring down patches of heavier grain.' },
  strong: { label: 'Strong', hint: 'Everything above, plus faint patterns, firmer deblocking, and half the grain removed everywhere.' },
  maximum: { label: 'Maximum', hint: 'As far as it goes: 80 % of the grain removed, the heaviest deblocking. Expect a smoother look.' },
};
