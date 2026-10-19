/**
 * JPEG output through mozjpeg (WASM). Always a fresh encode from the composed
 * float planes; metadata is spliced back in afterwards.
 */

import encodeJpeg, { init as initEncoder } from '@jsquash/jpeg/encode';
import type { Plane } from '../dsp/plane';
import { spliceMetadata } from './jpegParse';

export interface JpegExportOptions {
  quality: number; // 1..100
  chroma: '4:4:4' | '4:2:0'; // mozjpeg's wrapper exposes only full and 2x2 chroma
  progressive: boolean;
  trellis: boolean;
  /** separate, usually lower, chroma quality; null = same as luma */
  chromaQuality: number | null;
}

export const defaultExportOptions: JpegExportOptions = {
  quality: 95,
  chroma: '4:4:4',
  progressive: true,
  trellis: true,
  chromaQuality: null,
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

export async function encodeToJpeg(
  planes: [Plane, Plane, Plane],
  opts: JpegExportOptions,
  meta: { exif?: Uint8Array | null; icc?: Uint8Array | null; xmp?: Uint8Array | null } = {},
): Promise<Uint8Array> {
  await ensureEncoder();
  const w = planes[0].w;
  const h = planes[0].h;
  const rgba = planesToRgba(planes);
  const imageData = { data: rgba, width: w, height: h, colorSpace: 'srgb' } as ImageData;
  const subsample = opts.chroma === '4:4:4' ? 1 : 2;
  const buf = await encodeJpeg(imageData, {
    quality: Math.max(1, Math.min(100, Math.round(opts.quality))),
    baseline: false,
    arithmetic: false,
    progressive: opts.progressive,
    optimize_coding: true,
    smoothing: 0,
    color_space: 3, // YCbCr
    quant_table: 3,
    trellis_multipass: opts.trellis,
    trellis_opt_zero: opts.trellis,
    trellis_opt_table: opts.trellis,
    trellis_loops: 1,
    auto_subsample: false,
    chroma_subsample: subsample,
    separate_chroma_quality: opts.chromaQuality !== null,
    chroma_quality: opts.chromaQuality ?? opts.quality,
  });
  const encoded = new Uint8Array(buf);
  return spliceMetadata(encoded, meta);
}
