/**
 * HEIC / HEIF decode through libheif (WASM), loaded only when a file needs it.
 * Safari decodes HEIC natively; this path is for the browsers that don't.
 */

import type { Plane } from '../dsp/plane';
import { makePlane } from '../dsp/plane';

interface HeifImage {
  get_width(): number;
  get_height(): number;
  display(target: { data: Uint8ClampedArray; width: number; height: number }, cb: (out: { data: Uint8ClampedArray; width: number; height: number } | null) => void): void;
  free?(): void;
}
interface HeifModule {
  HeifDecoder: new () => { decode(data: Uint8Array): HeifImage[] };
}

let modPromise: Promise<HeifModule> | null = null;

function loadLibheif(): Promise<HeifModule> {
  if (!modPromise) {
    modPromise = import('libheif-js/libheif-wasm/libheif-bundle.mjs').then((m) => {
      const factory = (m as { default: (o?: object) => HeifModule | Promise<HeifModule> }).default;
      return factory({});
    });
  }
  return modPromise;
}

export function isHeif(bytes: Uint8Array): boolean {
  if (bytes.length < 12) return false;
  const brand = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
  const ftyp = String.fromCharCode(bytes[4], bytes[5], bytes[6], bytes[7]);
  return ftyp === 'ftyp' && ['heic', 'heix', 'hevc', 'hevx', 'heim', 'heis', 'mif1', 'msf1', 'avif'].includes(brand);
}

export async function decodeHeif(bytes: Uint8Array): Promise<{ planes: [Plane, Plane, Plane]; w: number; h: number } | null> {
  const lib = await loadLibheif();
  const decoder = new lib.HeifDecoder();
  const images = decoder.decode(bytes);
  if (!images.length) return null;
  const img = images[0];
  const w = img.get_width();
  const h = img.get_height();
  const out = await new Promise<{ data: Uint8ClampedArray; width: number; height: number } | null>((resolve) =>
    img.display({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }, (d) => resolve(d)),
  );
  for (const im of images) im.free?.();
  if (!out) return null;
  const r = makePlane(w, h);
  const g = makePlane(w, h);
  const b = makePlane(w, h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    r.data[i] = out.data[j] / 255;
    g.data[i] = out.data[j + 1] / 255;
    b.data[i] = out.data[j + 2] / 255;
  }
  return { planes: [r, g, b], w, h };
}
