/**
 * JPEG output through mozjpeg (WASM). Always a fresh encode from the composed
 * float planes; metadata is spliced back in afterwards.
 *
 * Two kinds of output:
 *   - a master copy: high quality, mozjpeg's tuned tables, trellis on;
 *   - a destination copy: the destination's own quality and tables (plain
 *     Annex K, no trellis), encoded repeatedly until the file is a fixed
 *     point, so the destination's re-encode has nothing left to change.
 */

import { decode as mozDecode } from '@jsquash/jpeg';
import encodeJpeg, { init as initEncoder } from '@jsquash/jpeg/encode';
import { makePlane, type Plane } from '../dsp/plane';
import { spliceMetadata } from './jpegParse';

export interface JpegExportOptions {
  quality: number; // 1..100
  chroma: '4:4:4' | '4:2:0'; // mozjpeg's wrapper exposes only full and 2x2 chroma
  progressive: boolean;
  trellis: boolean;
  /** separate, usually lower, chroma quality; null = same as luma */
  chromaQuality: number | null;
  /** 'annex-k' = the standard libjpeg tables every platform encoder uses; 'mozjpeg' = its tuned table */
  tables: 'annex-k' | 'mozjpeg';
  /** encode → decode → encode until the bytes stop changing (max this many passes); 0 = single encode */
  fixedPointPasses: number;
}

export const defaultExportOptions: JpegExportOptions = {
  quality: 95,
  chroma: '4:4:4',
  progressive: true,
  trellis: true,
  chromaQuality: null,
  tables: 'mozjpeg',
  fixedPointPasses: 0,
};

let encoderReady: Promise<void> | null = null;
export function ensureEncoder(): Promise<void> {
  if (!encoderReady) encoderReady = initEncoder();
  return encoderReady;
}

/** Float planes (0..1) to interleaved 8-bit RGBA with round-to-nearest. */
export function planesToRgba(planes: [Plane, Plane, Plane]): Uint8ClampedArray {
  const n = planes[0].w * planes[0].h;
  const out = new Uint8ClampedArray(n * 4);
  const r = planes[0].data;
  const g = planes[1].data;
  const b = planes[2].data;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[j] = Math.round(r[i] * 255);
    out[j + 1] = Math.round(g[i] * 255);
    out[j + 2] = Math.round(b[i] * 255);
    out[j + 3] = 255;
  }
  return out;
}

/** Decode a JPEG to float planes with mozjpeg (the same decoder used for verification). */
export async function decodeJpegToPlanes(jpeg: Uint8Array): Promise<[Plane, Plane, Plane]> {
  const img = await mozDecode(jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.byteLength) as ArrayBuffer);
  const { width: w, height: h } = img;
  const d = new Uint8ClampedArray(img.data); // copy out of WASM memory before anything else runs
  const planes: [Plane, Plane, Plane] = [makePlane(w, h), makePlane(w, h), makePlane(w, h)];
  const s = 1 / 255;
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    planes[0].data[i] = d[j] * s;
    planes[1].data[i] = d[j + 1] * s;
    planes[2].data[i] = d[j + 2] * s;
  }
  return planes;
}

async function encodeOnce(rgba: Uint8ClampedArray, w: number, h: number, opts: JpegExportOptions): Promise<Uint8Array> {
  await ensureEncoder();
  const imageData = { data: rgba, width: w, height: h, colorSpace: 'srgb' } as ImageData;
  const subsample = opts.chroma === '4:4:4' ? 1 : 2;
  const buf = await encodeJpeg(imageData, {
    quality: Math.max(1, Math.min(100, Math.round(opts.quality))),
    baseline: !opts.progressive && opts.tables === 'annex-k',
    arithmetic: false,
    progressive: opts.progressive,
    optimize_coding: true,
    smoothing: 0,
    color_space: 3, // YCbCr
    quant_table: opts.tables === 'annex-k' ? 0 : 3,
    trellis_multipass: opts.trellis,
    trellis_opt_zero: opts.trellis,
    trellis_opt_table: opts.trellis,
    trellis_loops: 1,
    auto_subsample: false,
    chroma_subsample: subsample,
    separate_chroma_quality: opts.chromaQuality !== null,
    chroma_quality: opts.chromaQuality ?? opts.quality,
  });
  return new Uint8Array(buf);
}

function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export interface EncodeResult {
  jpeg: Uint8Array;
  passes: number;
  stable: boolean; // the last pass reproduced the previous bytes exactly
}

/**
 * Encode, optionally iterating decode→encode until the bytes stop changing.
 * With the destination's own tables that makes the file a fixed point of the
 * destination's encoder: re-saving it there reproduces it.
 */
export async function encodeToJpeg(
  planes: [Plane, Plane, Plane],
  opts: JpegExportOptions,
  meta: { exif?: Uint8Array | null; icc?: Uint8Array | null; xmp?: Uint8Array | null } = {},
  onPass?: (pass: number) => void,
): Promise<EncodeResult> {
  const w = planes[0].w;
  const h = planes[0].h;
  let rgba = planesToRgba(planes);
  let jpeg = await encodeOnce(rgba, w, h, opts);
  let passes = 1;
  let stable = false;
  onPass?.(1);
  for (let p = 0; p < opts.fixedPointPasses; p++) {
    const img = await mozDecode(jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.byteLength) as ArrayBuffer);
    // the decoder hands back a view into WASM memory, which detaches when the
    // encoder grows that memory: copy it out first
    rgba = new Uint8ClampedArray(img.data);
    const next = await encodeOnce(rgba, w, h, opts);
    passes++;
    onPass?.(passes);
    if (sameBytes(next, jpeg)) {
      stable = true;
      break;
    }
    jpeg = next;
  }
  return { jpeg: spliceMetadata(jpeg, meta), passes, stable };
}

/** One plain encode→decode at the given settings: what a destination's re-save does to these pixels. */
export async function simulateReencode(planes: [Plane, Plane, Plane], opts: JpegExportOptions): Promise<{ planes: [Plane, Plane, Plane]; bytes: number }> {
  const w = planes[0].w;
  const h = planes[0].h;
  const jpeg = await encodeOnce(planesToRgba(planes), w, h, { ...opts, fixedPointPasses: 0 });
  return { planes: await decodeJpegToPlanes(jpeg), bytes: jpeg.length };
}
