# nnormalize

A browser tool that reads any photo, finds what is wrong with it — repeating patterns, block edges, uneven grain, colour blotches, halos next to edges, dead pixels — fixes only those things, lets you dial every fix down globally or by area, and writes a JPEG that holds up after the place you post it re-saves it.

Everything runs locally in the browser. Nothing is uploaded.

The hosted copy is at the repository's GitHub Pages address; `nnormalize-single.html` is the same tool in one file that opens from a folder.

## What it does

1. **Reads the photo at full precision.** JPEG through mozjpeg (the same decoder it uses to check its own output), PNG including 16-bit through its own decoder, HEIC through libheif where the browser can't, everything else through the browser. EXIF orientation is applied to the pixels; the camera metadata and colour profile are kept for the output.
2. **Looks for problems, each measured in plain numbers.** "Levels" throughout are brightness steps out of 255: 10 is obvious, 1 is faint, 0.3 is invisible.
   - **Repeating patterns** (moiré, scan lines, screens, banding): spikes in the averaged spectrum of overlapping tiles, measured to a fraction of a bin, then their strength and phase measured on every tile, so the tool knows where in the frame each pattern lives and whether it is the same everywhere. Harmonics are grouped under their fundamental. Line patterns on the spectrum axes are judged against the ring's own spread once they are 12 dB clear, so banding next to straight scene edges is still found.
   - **Block edges** from an earlier JPEG save: the grid phase, the step in levels, and an estimate of the quality it was saved at.
   - **Grain and uneven grain.** Grain is read as the fine residual each 32-px block carries, per brightness. The photo's base grain is the lowest substantial cluster of plain blocks (so a photo that is mostly fine texture still measures its grain from the plain parts), unless a clearly larger cluster lies nearby. Texture is told from grain three ways: aligned gradients (edges, weave), strength far above the grain, and spatial correlation redder than the grain's — stronger noise comes through compression whiter, never redder, so a grainier patch is still grain and sand or fabric is not. Nothing has to be masked: flat areas and grainier patches are found on their own.
   - **Colour blotches:** colour that brightness does not explain (the residual after luma-guided smoothing), measured at two scales so soft blobs count.
   - **Halos next to edges:** the fine residual in the band beside strong edges, clear of the edge's own transition, against the grain in plain blocks.
   - **Dead and stuck pixels.**
   - **Grain to put back.** Clean-up leaves smooth areas looking like plastic, so a fine photon-like grain (a little stronger in the shadows, fading out near white, with a touch of colour, correlated over about a pixel so a re-save keeps it) is added wherever the cleaned photo is smoother than the amount set. On by default at Strong and Maximum, available at the other levels. Textured areas keep their own detail.
3. **Fixes each one as a separate layer, in order.** A stationary pattern is subtracted exactly from its fitted parameters; one that varies is subtracted tile by tile; spectral flatten and notch are there when you want them. Block edges get a grid-aligned deblocker (over four pixels, or the whole block at the stronger levels). Grain is levelled toward the base grain with an edge-preserving smoother whose strength is measured per block, not assumed; textured blocks only ever get the weakest filters, and less of them. Colour blotches are smoothed toward the luma-guided colour. Halos get an edge-preserving smoother capped so strong edges stay sharp. Outliers are replaced by their neighbours. Each later stage is computed on the image with the earlier stages applied, so a moiré is never mistaken for block steps or grain.
4. **Lets you tone it down.** A master strength, a strength per fix, and a brush per fix painted directly on the photo (amber shows where a fix is held back), seeded from where the analysis found the pattern or from a brightness range. Masks are always feathered.
5. **Writes the file for where it is going.** *Reddit and sharing apps*: quality 75, 4:2:0, the standard tables, and the file is encoded repeatedly until a re-save reproduces it byte for byte — so Reddit's own re-encode changes nothing. Block edges are also written on that quality's coefficient grid: each block's coefficients are nudged by at most one step where that makes it meet its neighbours smoothly, so the re-save cannot bring the steps back. Grain put back is measured on the downloader's copy and, if the re-save rounded part of it away, the file is written again with more, so the copy carries the amount set. *Master copy*: quality 95, full colour, mozjpeg's tuned tables. *Custom*: your own settings.
6. **Checks what a downloader gets.** The file it wrote is put through the destination's re-save, and so is the untouched photo; every fix is then measured on both copies and reported as kept / partly / lost, in levels. The *As posted* view shows the downloader's copy, and Split puts the untouched photo through the same re-save beside it.

## Running it

```
npm install
npm run dev          # local development server (dev.html)
npm test             # unit tests: FFT, sub-bin refinement, corrections, noise and texture, block grid and block settling, hot pixels, colour, halos, engine end-to-end
npm run build:pages  # dist/ plus the copy served from the repository root, and the single-file build
```

The site is served from the repository root on the main branch, so `npm run build:pages` must be run before committing a change to the tool; the workflow in `.github/workflows/pages.yml` rebuilds it on every push as well.

## Using it

Drop a photo, or load the test image, which plants three known patterns, an 8×8 grid and a dozen hot pixels so you can see what the numbers look like when the answer is known.

- **Clean-up level** (top bar): *Gentle* and *Standard* fix only what is clearly wrong and leave the grain alone. *Strong* and *Maximum* also remove grain everywhere (50 % / 80 %), blend block edges over the whole block, and smooth colour blotches and halos harder.
- **Going to** (save panel): Reddit and sharing apps, master copy, or custom. "Save JPEG and check it" writes the file and runs the check; "Check only" runs the check without downloading.
- **Views:** before, after, what changed (amplified by the slider), as posted. Split compares before and after, or the downloader's copy against the untouched photo's. Fit, 100 % and 200 % zoom; double-click toggles fit and 100 %.
- **Brush:** with a problem selected, paint on the picture where its fix should be held back. "Only where it was found" and the brightness range seed the mask automatically.
- **Spectrum:** hover for exact frequency, period, angle, lines per inch when the DPI is known, and dB over background. Click a ring to select that finding; "Add a peak by clicking" turns a click into a new finding measured like the rest.
- **Selected fix:** strength, method, brush tools, "Where it was found", a brightness range. Technical numbers sit under *Measurements*. For *Put back grain*: the amount in levels (1.5 barely there, 3 a light film look) and how much colour is in it.
- **Recipes:** save the findings, strengths, methods, masks, level and destination as JSON; load one to apply the same choices to another photo (patterns are matched by frequency). Batch runs a set of photos through analysis, recipe, export and check, downloading each result and a summary report.

## Precision

The claims the tool makes are measured, not assumed, and the test suite holds them: frequency to better than 0.001 cycles per pixel, amplitude to within 1 % from the whole image, phase to within 0.05 rad; a planted stationary pattern is removed to within 0.3 dB of the noise floor; a planted non-stationary one is flagged as such and the tiled subtraction beats the global one; planted grain is recovered within 5 %, texture is scored high and grain low, a grainier patch is still grain, colour blotches are halved, halos are smoothed with nothing changed away from edges, and settled block boundaries are a fixed point of the destination's quantiser. The check then repeats the measurements on the downloader's copy.

## Not checked

Dust, hairs and scratches larger than a pixel; lens flare and chromatic aberration; blur, focus and sharpness; blown highlights and crushed shadows; colour casts; demosaicing errors that are not periodic.

## Layout

```
src/dsp       fft, windows, tiles (Welch), background model, peaks, refine (sub-bin DTFT), pattern (per-tile measurement),
              filter (parametric and spectral corrections), noise (grain, texture, levelling), chroma (colour blotches, halos),
              blocks (grid forensics, deblock), jpegdomain (block settling on the destination's coefficient grid), hotpixels
src/io        decode (mozjpeg, PNG16, HEIC, native), jpegParse (segments, quant tables, metadata splice), exif, png, encode (mozjpeg, fixed point)
src/analysis  types, analyze (findings), apply (delta layers), survival (what a downloader gets)
src/engine    engine (state, layers, compositing, export, check), profiles (destinations), worker protocol, client
src/ui        app, viewer, spectrum panel, presets, words (plain language)
src/synth     synthetic images with known ground truth
tests         vitest
```

## Licence

MIT. mozjpeg via jSquash (Apache-2.0), libheif via libheif-js (LGPL-3.0).
