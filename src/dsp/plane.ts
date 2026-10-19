/** A single-channel float image, row-major. Values are nominally 0..1. */
export interface Plane {
  w: number;
  h: number;
  data: Float32Array;
}

export function makePlane(w: number, h: number, fill = 0): Plane {
  const data = new Float32Array(w * h);
  if (fill !== 0) data.fill(fill);
  return { w, h, data };
}

export function clonePlane(p: Plane): Plane {
  return { w: p.w, h: p.h, data: new Float32Array(p.data) };
}

/** Rec. 601 luma from gamma-encoded RGB planes; the detector runs on this. */
export function lumaPlane(r: Plane, g: Plane, b: Plane): Plane {
  const n = r.w * r.h;
  const out = new Float32Array(n);
  const R = r.data;
  const G = g.data;
  const B = b.data;
  for (let i = 0; i < n; i++) out[i] = 0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i];
  return { w: r.w, h: r.h, data: out };
}

/** Copy a w x h window at (x0, y0) into a Float64Array (zero outside the image). */
export function extractTile(p: Plane, x0: number, y0: number, w: number, h: number, out?: Float64Array): Float64Array {
  const dst = out ?? new Float64Array(w * h);
  const src = p.data;
  for (let y = 0; y < h; y++) {
    const sy = y0 + y;
    const row = y * w;
    if (sy < 0 || sy >= p.h) {
      dst.fill(0, row, row + w);
      continue;
    }
    const srow = sy * p.w;
    for (let x = 0; x < w; x++) {
      const sx = x0 + x;
      dst[row + x] = sx >= 0 && sx < p.w ? src[srow + sx] : 0;
    }
  }
  return dst;
}

/** Nearest-neighbour-free box downsample by an integer factor (area average). */
export function downsamplePlane(p: Plane, factor: number): Plane {
  if (factor <= 1) return p;
  const w = Math.floor(p.w / factor);
  const h = Math.floor(p.h / factor);
  const out = new Float32Array(w * h);
  const inv = 1 / (factor * factor);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let dy = 0; dy < factor; dy++) {
        const row = (y * factor + dy) * p.w + x * factor;
        for (let dx = 0; dx < factor; dx++) s += p.data[row + dx];
      }
      out[y * w + x] = s * inv;
    }
  }
  return { w, h, data: out };
}

export function planeStats(p: Plane): { mean: number; min: number; max: number } {
  let s = 0;
  let mn = Infinity;
  let mx = -Infinity;
  const d = p.data;
  for (let i = 0; i < d.length; i++) {
    const v = d[i];
    s += v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return { mean: s / d.length, min: mn, max: mx };
}
