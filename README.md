# nnormalize

A browser tool that reads any photo, finds what repeats in its spectrum and what is uneven in its noise, lets you dial every fix down globally or by area, and writes a JPEG it has verified.

Everything runs locally in the browser. Nothing is uploaded.

## What it does

1. **Reads the photo at full precision.** JPEG through mozjpeg (the same decoder it uses to verify its own output), PNG including 16-bit through its own decoder, HEIC through libheif where the browser can't, everything else through the browser. EXIF orientation is applied to the pixels; the camera metadata and colour profile are kept for the output.
2. **Looks at it in overlapping tiles.** The averaged spectrum over all tiles shows repeating patterns (moiré, scan lines, halftone screens, banding) as sharp spikes that natural image content never produces. Every spike is measured to a fraction of a bin by evaluating the spectrum directly between bins, then its amplitude and phase are measured on every tile, so the tool knows where in the frame each pattern lives and whether it is the same pattern everywhere.
3. **Lists findings, not guesses.** Each entry carries its exact numbers: period, angle, amplitude in 8-bit levels, dB over the modelled background, where it was found. Findings the tool is unsure about, or that are below the visibility floor, are shown but left unticked. The classes:
   - repeating patterns, with harmonics grouped under their fundamental;
   - JPEG block edges, with the grid phase and an estimate of the source quality from the quantisation tables;
   - uneven noise, as a map of how far each region's grain is from the photo's own model;
   - hot and dead pixels.
4. **Fixes each one as a separate layer.** A stationary pattern is subtracted exactly from its fitted parameters, so nothing but that one frequency changes. A pattern that varies across the frame is subtracted tile by tile. Spectral flatten and notch are there when you want them. Block edges get a grid-aligned deblocker, noise gets levelled toward the model (denoised where it is too strong, matched grain added where a previous tool scrubbed it), outliers get inpainted. Later stages are computed on the image with the earlier stages applied, so a moiré is never mistaken for block steps or grain.
5. **Lets you tone it down.** A master strength, a strength per fix, and a mask per fix painted directly on the photo, seeded from where the analysis found the pattern or from a brightness range. Masks are always feathered.
6. **Exports a JPEG and checks it.** Always a fresh encode through mozjpeg with explicit quality, chroma subsampling, progressive and trellis settings, metadata spliced back in. Then it decodes the file it just wrote and runs the analysis again on those bytes. The report is pass/fail against a budget you set, lists what the encoder re-introduced, offers to raise quality, and proposes fixes the first analysis could not see under the interference. It also states plainly what the tool never looks for.

## Running it

```
npm install
npm run dev          # local development server
npm test             # unit tests: FFT against a naive DFT, sub-bin refinement accuracy, correction PSNR, noise estimator, block grid, hot pixels, engine end-to-end
npm run build        # dist/ for hosting (what GitHub Pages serves)
npm run build:single # dist-single/index.html, one file that opens from a folder with no server
```

The hosted copy is built and deployed by the workflow in `.github/workflows/pages.yml` on every push to `main`; the single-file build is published alongside it as `nnormalize-single.html`.

## Using it

Drop a photo, or load the test image, which plants three known patterns, an 8×8 grid and a dozen hot pixels so you can see what the numbers look like when the answer is known.

- **Views:** original, result, difference (what changed, amplified by the gain slider), mask. Split compares before and after. Fit, 100 % and 200 % zoom; double-click toggles fit and 100 %.
- **Spectrum:** hover for exact frequency, period, angle, lines per inch when the DPI is known, and dB over background. Click a ring to select that finding; "Add a peak by clicking" turns a click into a new finding measured like the rest.
- **Selected fix:** strength, method, brush tools, "Where it was found" to seed the mask from the per-tile presence, a brightness range mask.
- **Analysis:** presets for scans, screens, old phone JPEGs and film; the thresholds behind them; linear-light analysis; re-analyse.
- **Recipes:** save the current findings, strengths, methods, masks and export settings as JSON; load one to apply the same choices to another photo (patterns are matched by frequency). Batch runs a set of photos through analysis, recipe, export and verification, downloading each result and a summary report.

## Precision

The claims the tool makes are measured, not assumed, and the test suite holds them: frequency to better than 0.001 cycles per pixel, amplitude to within 1 % from the whole image, phase to within 0.05 rad; a planted stationary pattern is removed to within 0.3 dB of the noise floor; a planted non-stationary one is flagged as such and the tiled subtraction beats the global one. The verification step then repeats the measurement on the exported bytes.

## Not checked

Dust, hairs and scratches larger than a pixel; lens flare and chromatic aberration; blur, focus and sharpness; blown highlights and crushed shadows; colour casts; demosaicing errors that are not periodic. The report says so every time.

## Layout

```
src/dsp       fft, windows, tiles (Welch), background model, peaks, refine (sub-bin DTFT), pattern (per-tile measurement),
              filter (parametric and spectral corrections), noise, blocks (grid forensics, deblock), hotpixels
src/io        decode (mozjpeg, PNG16, HEIC, native), jpegParse (segments, quant tables, metadata splice), exif, png, encode (mozjpeg)
src/analysis  types, analyze (findings), apply (delta layers)
src/engine    engine (state, layers, compositing, export, round trip), worker protocol, client
src/ui        app, viewer, spectrum panel, presets
src/synth     synthetic images with known ground truth
tests         vitest
```

## Licence

MIT. mozjpeg via jSquash (Apache-2.0), libheif via libheif-js (LGPL-3.0).
