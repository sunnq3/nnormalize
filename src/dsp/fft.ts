/**
 * Float64 power-of-two FFT, in place, split re/im arrays.
 *
 * Only power-of-two sizes are needed: every spectrum in this tool comes from a
 * power-of-two tile. Frequencies between bins are measured by evaluating the
 * DTFT directly (see refine.ts), not by zero-padding, so there is no reason to
 * carry a general mixed-radix implementation.
 */

const planCache = new Map<number, FFTPlan>();

export class FFTPlan {
  readonly n: number;
  private readonly rev: Uint32Array;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;

  constructor(n: number) {
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`);
    this.n = n;
    const bits = Math.log2(n);
    this.rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      let x = i;
      for (let b = 0; b < bits; b++) {
        r = (r << 1) | (x & 1);
        x >>>= 1;
      }
      this.rev[i] = r;
    }
    const half = n >> 1;
    this.cos = new Float64Array(half);
    this.sin = new Float64Array(half);
    for (let i = 0; i < half; i++) {
      const a = (-2 * Math.PI * i) / n;
      this.cos[i] = Math.cos(a);
      this.sin[i] = Math.sin(a);
    }
  }

  static get(n: number): FFTPlan {
    let p = planCache.get(n);
    if (!p) {
      p = new FFTPlan(n);
      planCache.set(n, p);
    }
    return p;
  }

  /** Forward transform, unnormalised: X[k] = sum x[n] e^{-2 pi i k n / N}. */
  forward(re: Float64Array, im: Float64Array): void {
    this.transform(re, im, false);
  }

  /** Inverse transform, normalised by 1/N so inverse(forward(x)) == x. */
  inverse(re: Float64Array, im: Float64Array): void {
    this.transform(re, im, true);
    const n = this.n;
    const s = 1 / n;
    for (let i = 0; i < n; i++) {
      re[i] *= s;
      im[i] *= s;
    }
  }

  private transform(re: Float64Array, im: Float64Array, inverse: boolean): void {
    const n = this.n;
    const rev = this.rev;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    const cos = this.cos;
    const sin = this.sin;
    const sign = inverse ? -1 : 1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        let k = 0;
        for (let j = start; j < start + half; j++) {
          const wr = cos[k];
          const wi = sign * sin[k];
          const l = j + half;
          const tr = re[l] * wr - im[l] * wi;
          const ti = re[l] * wi + im[l] * wr;
          re[l] = re[j] - tr;
          im[l] = im[j] - ti;
          re[j] += tr;
          im[j] += ti;
          k += step;
        }
      }
    }
  }
}

/**
 * In-place 2-D FFT of a w x h complex array stored row-major.
 * Rows first, then columns via a scratch buffer.
 */
export function fft2d(re: Float64Array, im: Float64Array, w: number, h: number, inverse = false): void {
  const rowPlan = FFTPlan.get(w);
  const colPlan = FFTPlan.get(h);
  for (let y = 0; y < h; y++) {
    const r = re.subarray(y * w, (y + 1) * w);
    const i = im.subarray(y * w, (y + 1) * w);
    if (inverse) rowPlan.inverse(r, i);
    else rowPlan.forward(r, i);
  }
  const cr = new Float64Array(h);
  const ci = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      cr[y] = re[y * w + x];
      ci[y] = im[y * w + x];
    }
    if (inverse) colPlan.inverse(cr, ci);
    else colPlan.forward(cr, ci);
    for (let y = 0; y < h; y++) {
      re[y * w + x] = cr[y];
      im[y * w + x] = ci[y];
    }
  }
}

/** Forward 2-D FFT of a real array; returns fresh complex output. */
export function rfft2d(real: ArrayLike<number>, w: number, h: number): { re: Float64Array; im: Float64Array } {
  const re = Float64Array.from(real);
  const im = new Float64Array(w * h);
  fft2d(re, im, w, h, false);
  return { re, im };
}

/**
 * Re-centre a w x h spectrum so DC sits at (w/2, h/2) (like numpy fftshift),
 * returning a new array. Its own inverse for even sizes.
 */
export function fftshift2d<T extends Float64Array | Float32Array>(a: T, w: number, h: number): T {
  const out = new (a.constructor as new (n: number) => T)(a.length);
  const hw = w >> 1;
  const hh = h >> 1;
  for (let y = 0; y < h; y++) {
    const sy = (y + hh) % h;
    for (let x = 0; x < w; x++) {
      const sx = (x + hw) % w;
      out[sy * w + sx] = a[y * w + x];
    }
  }
  return out;
}

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export function prevPow2(n: number): number {
  let p = 1;
  while (p * 2 <= n) p <<= 1;
  return p;
}
