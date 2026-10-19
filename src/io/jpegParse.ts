/**
 * JPEG container parsing: metadata segments, quantisation tables, frame
 * header. Pixels are decoded elsewhere; this only walks the marker structure.
 */

export interface JpegQuantTable {
  id: number;
  precision: 8 | 16;
  values: number[]; // natural (row-major) order, 64 entries
}

export interface JpegComponent {
  id: number;
  h: number; // horizontal sampling factor
  v: number;
  tq: number; // quant table id
}

export interface JpegInfo {
  width: number;
  height: number;
  progressive: boolean;
  components: JpegComponent[];
  /** "4:4:4", "4:2:2", "4:2:0", "4:4:0", "gray" or the raw factors */
  subsampling: string;
  quantTables: JpegQuantTable[];
  exif: Uint8Array | null; // APP1 payload after "Exif\0\0" (TIFF header onward)
  icc: Uint8Array | null; // concatenated ICC_PROFILE chunks
  xmp: Uint8Array | null; // APP1 payload after the XMP namespace string
  hasJfif: boolean;
  hasAdobe: boolean;
}

const EXIF_HEADER = 'Exif\0\0';
const XMP_HEADER = 'http://ns.adobe.com/xap/1.0/\0';
const ICC_HEADER = 'ICC_PROFILE\0';

function ascii(bytes: Uint8Array, start: number, len: number): string {
  let s = '';
  for (let i = 0; i < len; i++) s += String.fromCharCode(bytes[start + i]);
  return s;
}

export function isJpeg(bytes: Uint8Array): boolean {
  return bytes.length > 3 && bytes[0] === 0xff && bytes[1] === 0xd8;
}

export const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43,
  36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

export function parseJpeg(bytes: Uint8Array): JpegInfo {
  if (!isJpeg(bytes)) throw new Error('not a JPEG');
  const info: JpegInfo = {
    width: 0,
    height: 0,
    progressive: false,
    components: [],
    subsampling: '',
    quantTables: [],
    exif: null,
    icc: null,
    xmp: null,
    hasJfif: false,
    hasAdobe: false,
  };
  const iccChunks: { seq: number; data: Uint8Array }[] = [];
  let pos = 2;
  while (pos + 4 <= bytes.length) {
    if (bytes[pos] !== 0xff) {
      pos++;
      continue;
    }
    const marker = bytes[pos + 1];
    if (marker === 0xff) {
      pos++;
      continue;
    }
    if (marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7) || marker === 0x01) {
      pos += 2;
      continue;
    }
    if (marker === 0xd9) break; // EOI
    const len = (bytes[pos + 2] << 8) | bytes[pos + 3];
    const segStart = pos + 4;
    const segEnd = pos + 2 + len;
    if (segEnd > bytes.length) break;
    const seg = bytes.subarray(segStart, segEnd);
    switch (marker) {
      case 0xe0:
        if (ascii(seg, 0, 4) === 'JFIF') info.hasJfif = true;
        break;
      case 0xe1:
        if (ascii(seg, 0, 6) === EXIF_HEADER) {
          if (!info.exif) info.exif = seg.slice(6);
        } else if (seg.length > XMP_HEADER.length && ascii(seg, 0, XMP_HEADER.length) === XMP_HEADER) {
          if (!info.xmp) info.xmp = seg.slice(XMP_HEADER.length);
        }
        break;
      case 0xe2:
        if (ascii(seg, 0, ICC_HEADER.length) === ICC_HEADER) {
          iccChunks.push({ seq: seg[ICC_HEADER.length], data: seg.slice(ICC_HEADER.length + 2) });
        }
        break;
      case 0xee:
        if (ascii(seg, 0, 5) === 'Adobe') info.hasAdobe = true;
        break;
      case 0xdb: {
        let p = 0;
        while (p < seg.length) {
          const pq = seg[p] >> 4;
          const tq = seg[p] & 15;
          p++;
          const values = new Array<number>(64);
          for (let i = 0; i < 64; i++) {
            let v: number;
            if (pq === 0) v = seg[p++];
            else {
              v = (seg[p] << 8) | seg[p + 1];
              p += 2;
            }
            values[ZIGZAG[i]] = v;
          }
          info.quantTables.push({ id: tq, precision: pq === 0 ? 8 : 16, values });
        }
        break;
      }
      case 0xc0:
      case 0xc1:
      case 0xc2:
      case 0xc3:
      case 0xc5:
      case 0xc6:
      case 0xc7:
      case 0xc9:
      case 0xca:
      case 0xcb:
      case 0xcd:
      case 0xce:
      case 0xcf: {
        info.progressive = marker === 0xc2 || marker === 0xc6 || marker === 0xca || marker === 0xce;
        info.height = (seg[1] << 8) | seg[2];
        info.width = (seg[3] << 8) | seg[4];
        const nc = seg[5];
        for (let i = 0; i < nc; i++) {
          const b = 6 + i * 3;
          info.components.push({ id: seg[b], h: seg[b + 1] >> 4, v: seg[b + 1] & 15, tq: seg[b + 2] });
        }
        break;
      }
      case 0xda:
        // start of scan: entropy-coded data follows; for progressive files more
        // headers come after it, but nothing we need. Stop here.
        pos = bytes.length;
        continue;
      default:
        break;
    }
    pos = segEnd;
  }
  if (iccChunks.length) {
    iccChunks.sort((a, b) => a.seq - b.seq);
    let total = 0;
    for (const c of iccChunks) total += c.data.length;
    const icc = new Uint8Array(total);
    let o = 0;
    for (const c of iccChunks) {
      icc.set(c.data, o);
      o += c.data.length;
    }
    info.icc = icc;
  }
  info.subsampling = describeSubsampling(info.components);
  return info;
}

export function describeSubsampling(components: JpegComponent[]): string {
  if (components.length === 1) return 'gray';
  if (components.length < 3) return 'unknown';
  const y = components[0];
  const c = components[1];
  const hx = y.h / c.h;
  const vy = y.v / c.v;
  if (hx === 1 && vy === 1) return '4:4:4';
  if (hx === 2 && vy === 1) return '4:2:2';
  if (hx === 2 && vy === 2) return '4:2:0';
  if (hx === 1 && vy === 2) return '4:4:0';
  return `${y.h}x${y.v}:${c.h}x${c.v}`;
}

/**
 * Build a JPEG byte stream from an encoder's output plus metadata segments:
 * SOI, [APP1 EXIF], [APP1 XMP], [APP2 ICC chunks], then everything the
 * encoder wrote after its own SOI.
 */
export function spliceMetadata(encoded: Uint8Array, meta: { exif?: Uint8Array | null; icc?: Uint8Array | null; xmp?: Uint8Array | null }): Uint8Array {
  if (!isJpeg(encoded)) throw new Error('encoder output is not a JPEG');
  const parts: Uint8Array[] = [new Uint8Array([0xff, 0xd8])];
  const segment = (marker: number, payload: Uint8Array): Uint8Array => {
    const len = payload.length + 2;
    if (len > 0xffff) throw new Error('segment too long');
    const out = new Uint8Array(payload.length + 4);
    out[0] = 0xff;
    out[1] = marker;
    out[2] = len >> 8;
    out[3] = len & 255;
    out.set(payload, 4);
    return out;
  };
  const str = (s: string): Uint8Array => Uint8Array.from(s, (ch) => ch.charCodeAt(0));
  if (meta.exif && meta.exif.length) {
    const payload = new Uint8Array(6 + meta.exif.length);
    payload.set(str(EXIF_HEADER), 0);
    payload.set(meta.exif, 6);
    if (payload.length + 2 <= 0xffff) parts.push(segment(0xe1, payload));
  }
  if (meta.xmp && meta.xmp.length) {
    const payload = new Uint8Array(XMP_HEADER.length + meta.xmp.length);
    payload.set(str(XMP_HEADER), 0);
    payload.set(meta.xmp, XMP_HEADER.length);
    if (payload.length + 2 <= 0xffff) parts.push(segment(0xe1, payload));
  }
  if (meta.icc && meta.icc.length) {
    const chunkMax = 0xffff - 2 - ICC_HEADER.length - 2;
    const n = Math.ceil(meta.icc.length / chunkMax);
    for (let i = 0; i < n; i++) {
      const piece = meta.icc.subarray(i * chunkMax, Math.min(meta.icc.length, (i + 1) * chunkMax));
      const payload = new Uint8Array(ICC_HEADER.length + 2 + piece.length);
      payload.set(str(ICC_HEADER), 0);
      payload[ICC_HEADER.length] = i + 1;
      payload[ICC_HEADER.length + 1] = n;
      payload.set(piece, ICC_HEADER.length + 2);
      parts.push(segment(0xe2, payload));
    }
  }
  parts.push(encoded.subarray(2));
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}
