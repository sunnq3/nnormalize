/**
 * Minimal TIFF/EXIF walker: read the orientation, and produce a copy of the
 * EXIF payload with orientation reset to 1 and the thumbnail IFD unlinked,
 * so a re-oriented, re-encoded image carries honest metadata.
 */

export interface ExifSummary {
  orientation: number; // 1..8, 1 when absent
  littleEndian: boolean;
  make?: string;
  model?: string;
  xResolution?: number; // pixels per unit
  yResolution?: number;
  resolutionUnit?: number; // 2 = inch, 3 = cm
}

const TAG_ORIENTATION = 0x0112;
const TAG_MAKE = 0x010f;
const TAG_MODEL = 0x0110;
const TAG_XRES = 0x011a;
const TAG_YRES = 0x011b;
const TAG_RESUNIT = 0x0128;

function reader(tiff: Uint8Array): { le: boolean; u16: (o: number) => number; u32: (o: number) => number } | null {
  if (tiff.length < 8) return null;
  const le = tiff[0] === 0x49 && tiff[1] === 0x49;
  const be = tiff[0] === 0x4d && tiff[1] === 0x4d;
  if (!le && !be) return null;
  const dv = new DataView(tiff.buffer, tiff.byteOffset, tiff.byteLength);
  const u16 = (o: number): number => (o + 2 <= tiff.length ? dv.getUint16(o, le) : 0);
  const u32 = (o: number): number => (o + 4 <= tiff.length ? dv.getUint32(o, le) : 0);
  if (u16(2) !== 42) return null;
  return { le, u16, u32 };
}

export function readExif(tiff: Uint8Array | null | undefined): ExifSummary {
  const out: ExifSummary = { orientation: 1, littleEndian: true };
  if (!tiff) return out;
  const r = reader(tiff);
  if (!r) return out;
  out.littleEndian = r.le;
  const ifd0 = r.u32(4);
  if (ifd0 + 2 > tiff.length) return out;
  const n = r.u16(ifd0);
  const readAscii = (off: number, count: number): string => {
    let s = '';
    for (let i = 0; i < count - 1 && off + i < tiff.length; i++) s += String.fromCharCode(tiff[off + i]);
    return s.trim();
  };
  for (let i = 0; i < n; i++) {
    const e = ifd0 + 2 + i * 12;
    if (e + 12 > tiff.length) break;
    const tag = r.u16(e);
    const type = r.u16(e + 2);
    const count = r.u32(e + 4);
    const valOff = e + 8;
    switch (tag) {
      case TAG_ORIENTATION:
        if (type === 3) out.orientation = r.u16(valOff) || 1;
        break;
      case TAG_MAKE:
        if (type === 2) out.make = readAscii(count <= 4 ? valOff : r.u32(valOff), count);
        break;
      case TAG_MODEL:
        if (type === 2) out.model = readAscii(count <= 4 ? valOff : r.u32(valOff), count);
        break;
      case TAG_XRES:
      case TAG_YRES:
        if (type === 5) {
          const o = r.u32(valOff);
          const num = r.u32(o);
          const den = r.u32(o + 4) || 1;
          if (tag === TAG_XRES) out.xResolution = num / den;
          else out.yResolution = num / den;
        }
        break;
      case TAG_RESUNIT:
        if (type === 3) out.resolutionUnit = r.u16(valOff);
        break;
    }
  }
  return out;
}

/**
 * Copy of the TIFF payload with orientation forced to 1 (the pixels have been
 * rotated) and IFD0's "next IFD" pointer zeroed so the stale embedded
 * thumbnail is no longer reachable.
 */
export function normaliseExif(tiff: Uint8Array): Uint8Array {
  const out = new Uint8Array(tiff);
  const r = reader(out);
  if (!r) return out;
  const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
  const ifd0 = r.u32(4);
  if (ifd0 + 2 > out.length) return out;
  const n = r.u16(ifd0);
  for (let i = 0; i < n; i++) {
    const e = ifd0 + 2 + i * 12;
    if (e + 12 > out.length) break;
    if (r.u16(e) === TAG_ORIENTATION && r.u16(e + 2) === 3) dv.setUint16(e + 8, 1, r.le);
  }
  const next = ifd0 + 2 + n * 12;
  if (next + 4 <= out.length) dv.setUint32(next, 0, r.le);
  return out;
}

/** Pixels per inch from the resolution tags, if present. */
export function exifDpi(e: ExifSummary): number | null {
  if (!e.xResolution) return null;
  if (e.resolutionUnit === 3) return e.xResolution * 2.54;
  return e.xResolution;
}
