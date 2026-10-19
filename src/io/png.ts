/**
 * PNG decoder for the cases the browser's own decoder would flatten to 8-bit:
 * 16-bit grey and RGB(A). 8-bit non-interlaced files also decode here so the
 * eXIf and iCCP chunks can be read. Interlaced and palette images fall back
 * to the native decoder (see decode.ts).
 *
 * Inflate uses the platform DecompressionStream ('deflate' = zlib framing),
 * so there is no dependency.
 */

export interface PngResult {
  width: number;
  height: number;
  bitDepth: 8 | 16;
  channels: 1 | 2 | 3 | 4; // grey, grey+alpha, rgb, rgba
  /** Decoded samples, row-major, interleaved, scaled to 0..1 */
  samples: Float32Array;
  exif: Uint8Array | null;
  icc: Uint8Array | null;
  gamma: number | null;
  srgb: boolean;
}

const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export function isPng(bytes: Uint8Array): boolean {
  if (bytes.length < 8) return false;
  for (let i = 0; i < 8; i++) if (bytes[i] !== SIG[i]) return false;
  return true;
}

export async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const ds = new DecompressionStream('deflate');
  const writer = ds.writable.getWriter();
  void writer.write(new Uint8Array(data) as Uint8Array<ArrayBuffer>);
  void writer.close();
  const chunks: Uint8Array[] = [];
  const reader = ds.readable.getReader();
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/** Returns null when the file needs the native decoder (interlaced, palette, odd depths). */
export async function decodePng(bytes: Uint8Array): Promise<PngResult | null> {
  if (!isPng(bytes)) throw new Error('not a PNG');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = 0;
  let interlace = 0;
  const idat: Uint8Array[] = [];
  let exif: Uint8Array | null = null;
  let iccCompressed: Uint8Array | null = null;
  let gamma: number | null = null;
  let srgb = false;
  while (pos + 8 <= bytes.length) {
    const len = dv.getUint32(pos);
    const type = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    const data = bytes.subarray(pos + 8, pos + 8 + len);
    switch (type) {
      case 'IHDR':
        width = dv.getUint32(pos + 8);
        height = dv.getUint32(pos + 12);
        bitDepth = bytes[pos + 16];
        colorType = bytes[pos + 17];
        interlace = bytes[pos + 20];
        break;
      case 'IDAT':
        idat.push(data);
        break;
      case 'eXIf':
        exif = data.slice();
        break;
      case 'iCCP': {
        let z = 0;
        while (z < data.length && data[z] !== 0) z++;
        iccCompressed = data.slice(z + 2);
        break;
      }
      case 'gAMA':
        gamma = dv.getUint32(pos + 8) / 100000;
        break;
      case 'sRGB':
        srgb = true;
        break;
      case 'IEND':
        pos = bytes.length;
        continue;
    }
    pos += 12 + len;
  }
  if (interlace !== 0 || colorType === 3 || (bitDepth !== 8 && bitDepth !== 16)) return null;
  const channels = ({ 0: 1, 2: 3, 4: 2, 6: 4 } as Record<number, 1 | 2 | 3 | 4>)[colorType];
  if (!channels) return null;

  let total = 0;
  for (const c of idat) total += c.length;
  const z = new Uint8Array(total);
  let o = 0;
  for (const c of idat) {
    z.set(c, o);
    o += c.length;
  }
  const raw = await inflate(z);
  const bpp = (channels * bitDepth) / 8; // bytes per pixel
  const stride = width * bpp;
  if (raw.length < height * (stride + 1)) throw new Error('PNG data truncated');

  const samples = new Float32Array(width * height * channels);
  const prev = new Uint8Array(stride);
  const cur = new Uint8Array(stride);
  const scale = bitDepth === 16 ? 1 / 65535 : 1 / 255;
  for (let y = 0; y < height; y++) {
    const base = y * (stride + 1);
    const filter = raw[base];
    const line = raw.subarray(base + 1, base + 1 + stride);
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev[i];
      const c = i >= bpp ? prev[i - bpp] : 0;
      let v = line[i];
      switch (filter) {
        case 1:
          v += a;
          break;
        case 2:
          v += b;
          break;
        case 3:
          v += (a + b) >> 1;
          break;
        case 4:
          v += paeth(a, b, c);
          break;
      }
      cur[i] = v & 255;
    }
    const rowOut = y * width * channels;
    if (bitDepth === 16) {
      for (let i = 0; i < width * channels; i++) samples[rowOut + i] = ((cur[2 * i] << 8) | cur[2 * i + 1]) * scale;
    } else {
      for (let i = 0; i < width * channels; i++) samples[rowOut + i] = cur[i] * scale;
    }
    prev.set(cur);
  }
  const icc = iccCompressed ? await inflate(iccCompressed) : null;
  return { width, height, bitDepth: bitDepth as 8 | 16, channels, samples, exif, icc, gamma, srgb };
}
