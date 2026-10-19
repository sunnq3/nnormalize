/**
 * Unified decode: any image the browser (or our own decoders) can read, into
 * planar Float32 RGB in 0..1, as stored (no colour management, no gamma
 * change), with EXIF orientation applied to the pixels and the metadata we
 * will want to carry to the output.
 */

import { decode as mozDecode } from '@jsquash/jpeg';
import { makePlane, type Plane } from '../dsp/plane';
import { qualityFromQuantTable } from '../dsp/blocks';
import { exifDpi, normaliseExif, readExif, type ExifSummary } from './exif';
import { isJpeg, parseJpeg, type JpegInfo } from './jpegParse';
import { decodePng, isPng } from './png';
import { decodeHeif, isHeif } from './heic';

export interface JpegSourceInfo {
  quality: number | null;
  chromaQuality: number | null;
  subsampling: string;
  progressive: boolean;
  quantTables: { id: number; values: number[] }[];
}

export interface SourceMeta {
  fileName: string;
  mime: string;
  byteLength: number;
  bitDepth: number;
  decoder: 'mozjpeg' | 'png' | 'native' | 'libheif';
  exif: Uint8Array | null; // TIFF payload, orientation already normalised to 1
  exifSummary: ExifSummary;
  dpi: number | null;
  icc: Uint8Array | null;
  xmp: Uint8Array | null;
  jpeg: JpegSourceInfo | null;
  orientationApplied: number;
  hasAlpha: boolean;
}

export interface SourceImage {
  w: number;
  h: number;
  planes: [Plane, Plane, Plane];
  meta: SourceMeta;
}

function jpegSourceInfo(info: JpegInfo): JpegSourceInfo {
  const luma = info.quantTables.find((t) => t.id === 0) ?? info.quantTables[0];
  const chroma = info.quantTables.find((t) => t.id === 1);
  return {
    quality: luma ? qualityFromQuantTable(luma.values, false) : null,
    chromaQuality: chroma ? qualityFromQuantTable(chroma.values, true) : null,
    subsampling: info.subsampling,
    progressive: info.progressive,
    quantTables: info.quantTables.map((t) => ({ id: t.id, values: t.values })),
  };
}

/** Rotate / flip planar data according to an EXIF orientation value. */
export function applyOrientation(planes: Plane[], orientation: number): { planes: Plane[]; w: number; h: number } {
  const w = planes[0].w;
  const h = planes[0].h;
  if (orientation <= 1 || orientation > 8) return { planes, w, h };
  const swap = orientation >= 5;
  const ow = swap ? h : w;
  const oh = swap ? w : h;
  const map = (x: number, y: number): [number, number] => {
    // returns source (sx, sy) for destination (x, y)
    switch (orientation) {
      case 2:
        return [w - 1 - x, y];
      case 3:
        return [w - 1 - x, h - 1 - y];
      case 4:
        return [x, h - 1 - y];
      case 5:
        return [y, x];
      case 6:
        return [y, h - 1 - x];
      case 7:
        return [w - 1 - y, h - 1 - x];
      case 8:
        return [w - 1 - y, x];
      default:
        return [x, y];
    }
  };
  const out = planes.map(() => makePlane(ow, oh));
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      const [sx, sy] = map(x, y);
      const si = sy * w + sx;
      const di = y * ow + x;
      for (let c = 0; c < planes.length; c++) out[c].data[di] = planes[c].data[si];
    }
  }
  return { planes: out, w: ow, h: oh };
}

function rgbaToPlanes(data: Uint8ClampedArray | Uint8Array, w: number, h: number): { planes: [Plane, Plane, Plane]; hasAlpha: boolean } {
  const r = makePlane(w, h);
  const g = makePlane(w, h);
  const b = makePlane(w, h);
  let hasAlpha = false;
  const s = 1 / 255;
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    r.data[i] = data[j] * s;
    g.data[i] = data[j + 1] * s;
    b.data[i] = data[j + 2] * s;
    if (data[j + 3] !== 255) hasAlpha = true;
  }
  return { planes: [r, g, b], hasAlpha };
}

async function decodeNative(blob: Blob): Promise<{ planes: [Plane, Plane, Plane]; w: number; h: number; hasAlpha: boolean }> {
  const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image', colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const w = bmp.width;
  const h = bmp.height;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: true });
  if (!ctx) throw new Error('no 2d context');
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const img = ctx.getImageData(0, 0, w, h);
  const { planes, hasAlpha } = rgbaToPlanes(img.data, w, h);
  return { planes, w, h, hasAlpha };
}

export async function decodeImage(file: Blob & { name?: string }): Promise<SourceImage> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const fileName = file.name ?? 'image';
  const mime = file.type || '';
  const base: Omit<SourceMeta, 'bitDepth' | 'decoder' | 'exif' | 'exifSummary' | 'dpi' | 'icc' | 'xmp' | 'jpeg' | 'orientationApplied' | 'hasAlpha'> = {
    fileName,
    mime,
    byteLength: bytes.length,
  };

  if (isJpeg(bytes)) {
    const info = parseJpeg(bytes);
    const summary = readExif(info.exif);
    const img = await mozDecode(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    const { planes } = rgbaToPlanes(img.data, img.width, img.height);
    const o = applyOrientation(planes, summary.orientation);
    return {
      w: o.w,
      h: o.h,
      planes: o.planes as [Plane, Plane, Plane],
      meta: {
        ...base,
        mime: 'image/jpeg',
        bitDepth: 8,
        decoder: 'mozjpeg',
        exif: info.exif ? normaliseExif(info.exif) : null,
        exifSummary: summary,
        dpi: exifDpi(summary),
        icc: info.icc,
        xmp: info.xmp,
        jpeg: jpegSourceInfo(info),
        orientationApplied: summary.orientation,
        hasAlpha: false,
      },
    };
  }

  if (isPng(bytes)) {
    const png = await decodePng(bytes);
    if (png) {
      const { width: w, height: h, channels, samples } = png;
      const r = makePlane(w, h);
      const g = makePlane(w, h);
      const b = makePlane(w, h);
      for (let i = 0; i < w * h; i++) {
        const j = i * channels;
        if (channels <= 2) r.data[i] = g.data[i] = b.data[i] = samples[j];
        else {
          r.data[i] = samples[j];
          g.data[i] = samples[j + 1];
          b.data[i] = samples[j + 2];
        }
      }
      const summary = readExif(png.exif);
      const o = applyOrientation([r, g, b], summary.orientation);
      return {
        w: o.w,
        h: o.h,
        planes: o.planes as [Plane, Plane, Plane],
        meta: {
          ...base,
          mime: 'image/png',
          bitDepth: png.bitDepth,
          decoder: 'png',
          exif: png.exif ? normaliseExif(png.exif) : null,
          exifSummary: summary,
          dpi: exifDpi(summary),
          icc: png.icc,
          xmp: null,
          jpeg: null,
          orientationApplied: summary.orientation,
          hasAlpha: channels === 2 || channels === 4,
        },
      };
    }
  }

  if (isHeif(bytes)) {
    // the browser first (Safari, and Chrome for AVIF), libheif if it refuses
    try {
      const native = await decodeNative(file);
      return nativeResult(base, native);
    } catch {
      const heif = await decodeHeif(bytes);
      if (!heif) throw new Error('HEIC/HEIF file could not be decoded');
      const exif = extractHeifExif(bytes);
      const summary = readExif(exif);
      const o = applyOrientation(heif.planes, summary.orientation);
      return {
        w: o.w,
        h: o.h,
        planes: o.planes as [Plane, Plane, Plane],
        meta: {
          ...base,
          mime: 'image/heic',
          bitDepth: 8,
          decoder: 'libheif',
          exif: exif ? normaliseExif(exif) : null,
          exifSummary: summary,
          dpi: exifDpi(summary),
          icc: null,
          xmp: null,
          jpeg: null,
          orientationApplied: summary.orientation,
          hasAlpha: false,
        },
      };
    }
  }

  const native = await decodeNative(file);
  return nativeResult(base, native);
}

function nativeResult(base: Pick<SourceMeta, 'fileName' | 'mime' | 'byteLength'>, native: { planes: [Plane, Plane, Plane]; w: number; h: number; hasAlpha: boolean }): SourceImage {
  return {
    w: native.w,
    h: native.h,
    planes: native.planes,
    meta: {
      ...base,
      bitDepth: 8,
      decoder: 'native',
      exif: null,
      exifSummary: { orientation: 1, littleEndian: true },
      dpi: null,
      icc: null,
      xmp: null,
      jpeg: null,
      orientationApplied: 0, // handled by the browser
      hasAlpha: native.hasAlpha,
    },
  };
}


/**
 * Best-effort EXIF extraction from a HEIF container: the Exif item payload is
 * a 4-byte offset followed by the TIFF header. We scan for the "Exif\0\0"
 * marker and the TIFF byte-order mark that follows it.
 */
function extractHeifExif(bytes: Uint8Array): Uint8Array | null {
  const n = bytes.length;
  for (let i = 0; i + 10 < n; i++) {
    if (bytes[i] === 0x45 && bytes[i + 1] === 0x78 && bytes[i + 2] === 0x69 && bytes[i + 3] === 0x66 && bytes[i + 4] === 0 && bytes[i + 5] === 0) {
      const t = i + 6;
      const le = bytes[t] === 0x49 && bytes[t + 1] === 0x49;
      const be = bytes[t] === 0x4d && bytes[t + 1] === 0x4d;
      if (le || be) return bytes.slice(t, Math.min(n, t + 65000));
    }
  }
  return null;
}
