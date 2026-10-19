/**
 * Expected-spectrum model. A natural image's power spectrum falls roughly as
 * 1/f^a and is smooth; interference is anything that sticks up sharply out of
 * that. The model is a radial median profile (captures the 1/f^a shape and
 * any colour of noise floor) plus a small local median of what the radial
 * model leaves behind (captures anisotropy such as the horizontal/vertical
 * emphasis of most photographs). Everything is done in log10 power.
 */

export interface BackgroundModel {
  size: number;
  logPower: Float64Array; // log10(power), shifted
  background: Float64Array; // modelled log10 power
  residual: Float64Array; // logPower - background
  z: Float64Array; // residual / robust sigma
  sigma: number; // robust sigma of the residual (log10 units)
  radialProfile: Float64Array; // ring medians, index = integer radius in bins
  ringSigma: Float64Array; // robust sigma per integer radius
  axisSigma: number; // robust sigma of bins on the two axes
}

function median(buf: Float64Array, len: number): number {
  const a = buf.subarray(0, len);
  a.sort();
  return len & 1 ? a[len >> 1] : 0.5 * (a[(len >> 1) - 1] + a[len >> 1]);
}

/** Separable (row then column) running median of odd width k. Approximate but fast. */
export function medianSeparable(a: Float64Array, w: number, h: number, k: number): Float64Array {
  const r = k >> 1;
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  const buf = new Float64Array(k);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let n = 0;
      for (let d = -r; d <= r; d++) {
        const xx = x + d;
        if (xx >= 0 && xx < w) buf[n++] = a[row + xx];
      }
      tmp[row + x] = median(buf, n);
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let n = 0;
      for (let d = -r; d <= r; d++) {
        const yy = y + d;
        if (yy >= 0 && yy < h) buf[n++] = tmp[yy * w + x];
      }
      out[y * w + x] = median(buf, n);
    }
  }
  return out;
}

export function robustSigma(a: Float64Array, mask?: Uint8Array): { median: number; sigma: number } {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) n++;
  const buf = new Float64Array(n);
  let j = 0;
  for (let i = 0; i < a.length; i++) if (!mask || mask[i]) buf[j++] = a[i];
  buf.sort();
  const med = n & 1 ? buf[n >> 1] : 0.5 * (buf[(n >> 1) - 1] + buf[n >> 1]);
  for (let i = 0; i < n; i++) buf[i] = Math.abs(buf[i] - med);
  buf.sort();
  const mad = n & 1 ? buf[n >> 1] : 0.5 * (buf[(n >> 1) - 1] + buf[n >> 1]);
  return { median: med, sigma: mad > 0 ? 1.4826 * mad : 1e-12 };
}

export interface BackgroundOptions {
  localMedian?: number; // odd width of the local median, bins (default 9)
  excludeRadius?: number; // bins around DC excluded from sigma estimation (default 4)
}

export function backgroundModel(power: Float64Array, size: number, opts: BackgroundOptions = {}): BackgroundModel {
  const n = size;
  const c = n >> 1;
  const localK = opts.localMedian ?? 9;
  const exclude = opts.excludeRadius ?? 4;

  let maxP = 0;
  for (let i = 0; i < power.length; i++) if (power[i] > maxP) maxP = power[i];
  const eps = maxP > 0 ? maxP * 1e-14 : 1e-30;
  const logP = new Float64Array(n * n);
  for (let i = 0; i < logP.length; i++) logP[i] = Math.log10(power[i] + eps);

  // ring medians
  const maxR = Math.ceil(Math.SQRT2 * c) + 1;
  const counts = new Int32Array(maxR + 1);
  const radius = new Int32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const r = Math.round(Math.hypot(x - c, y - c));
      radius[y * n + x] = r;
      counts[r]++;
    }
  }
  const offsets = new Int32Array(maxR + 2);
  for (let r = 0; r <= maxR; r++) offsets[r + 1] = offsets[r] + counts[r];
  const sorted = new Float64Array(n * n);
  const fill = new Int32Array(maxR + 1);
  for (let i = 0; i < logP.length; i++) {
    const r = radius[i];
    sorted[offsets[r] + fill[r]++] = logP[i];
  }
  const profile = new Float64Array(maxR + 1);
  for (let r = 0; r <= maxR; r++) {
    const len = counts[r];
    if (len === 0) {
      profile[r] = r > 0 ? profile[r - 1] : 0;
      continue;
    }
    const seg = sorted.subarray(offsets[r], offsets[r] + len);
    seg.sort();
    profile[r] = len & 1 ? seg[len >> 1] : 0.5 * (seg[(len >> 1) - 1] + seg[len >> 1]);
  }
  // light smoothing of the profile (5-point median) so a strong ring of peaks can't lift it
  const smooth = new Float64Array(profile.length);
  const buf = new Float64Array(5);
  for (let r = 0; r < profile.length; r++) {
    let m = 0;
    for (let d = -2; d <= 2; d++) {
      const rr = r + d;
      if (rr >= 0 && rr < profile.length) buf[m++] = profile[rr];
    }
    smooth[r] = median(buf, m);
  }

  const bgRadial = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const rf = Math.hypot(x - c, y - c);
      const r0 = Math.floor(rf);
      const t = rf - r0;
      const r1 = Math.min(r0 + 1, smooth.length - 1);
      bgRadial[y * n + x] = smooth[r0] * (1 - t) + smooth[r1] * t;
    }
  }

  const resid1 = new Float64Array(n * n);
  for (let i = 0; i < resid1.length; i++) resid1[i] = logP[i] - bgRadial[i];
  const bgLocal = medianSeparable(resid1, n, n, localK);

  const background = new Float64Array(n * n);
  const residual = new Float64Array(n * n);
  for (let i = 0; i < background.length; i++) {
    background[i] = bgRadial[i] + bgLocal[i];
    residual[i] = logP[i] - background[i];
  }

  // The two axes carry a ridge of energy from every straight edge in the
  // scene; a 2-D median across the ridge underestimates it. Model each axis
  // band (3 bins wide) with a 1-D running median along its own direction.
  {
    const half1d = 7;
    const abuf = new Float64Array(2 * half1d + 1);
    for (let band = -1; band <= 1; band++) {
      const x = c + band;
      for (let y = 0; y < n; y++) {
        let m = 0;
        for (let d = -half1d; d <= half1d; d++) {
          const yy = y + d;
          if (yy >= 0 && yy < n) abuf[m++] = logP[yy * n + x];
        }
        const i = y * n + x;
        background[i] = median(abuf, m);
        residual[i] = logP[i] - background[i];
      }
      const yb = c + band;
      for (let x2 = 0; x2 < n; x2++) {
        if (Math.abs(x2 - c) <= 1) continue; // already done (corner of the cross)
        let m = 0;
        for (let d = -half1d; d <= half1d; d++) {
          const xx = x2 + d;
          if (xx >= 0 && xx < n) abuf[m++] = logP[yb * n + xx];
        }
        const i = yb * n + x2;
        background[i] = median(abuf, m);
        residual[i] = logP[i] - background[i];
      }
    }
  }

  const mask = new Uint8Array(n * n);
  for (let i = 0; i < mask.length; i++) mask[i] = radius[i] > exclude ? 1 : 0;
  const { median: med, sigma } = robustSigma(residual, mask);

  // The residual's spread is not uniform: it grows toward high frequencies
  // (fewer photons per bin) and is larger along the two axes, where the
  // straight edges of real scenes put energy. Normalising by a per-ring sigma
  // and a separate axis sigma keeps the z-score honest everywhere.
  const ringSigma = new Float64Array(maxR + 1);
  {
    const absDev = new Float64Array(n * n);
    for (let i = 0; i < absDev.length; i++) absDev[i] = Math.abs(residual[i] - med);
    const scratch = new Float64Array(n * n);
    const fill2 = new Int32Array(maxR + 1);
    for (let i = 0; i < absDev.length; i++) {
      const r = radius[i];
      scratch[offsets[r] + fill2[r]++] = absDev[i];
    }
    for (let r = 0; r <= maxR; r++) {
      const len = counts[r];
      if (len < 16) {
        ringSigma[r] = sigma;
        continue;
      }
      const seg = scratch.subarray(offsets[r], offsets[r] + len);
      seg.sort();
      const mad = len & 1 ? seg[len >> 1] : 0.5 * (seg[(len >> 1) - 1] + seg[len >> 1]);
      ringSigma[r] = Math.max(sigma, 1.4826 * mad);
    }
    // smooth over 9 rings
    const sm = new Float64Array(ringSigma.length);
    for (let r = 0; r < ringSigma.length; r++) {
      let s = 0;
      let m = 0;
      for (let d = -4; d <= 4; d++) {
        const rr = r + d;
        if (rr >= 0 && rr < ringSigma.length) {
          s += ringSigma[rr];
          m++;
        }
      }
      sm[r] = s / m;
    }
    ringSigma.set(sm);
  }
  let axisSigma = sigma;
  {
    const vals: number[] = [];
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const onAxis = Math.abs(x - c) <= 1 || Math.abs(y - c) <= 1;
        const i = y * n + x;
        if (onAxis && radius[i] > exclude) vals.push(Math.abs(residual[i] - med));
      }
    }
    vals.sort((a, b) => a - b);
    const mad = vals.length ? vals[vals.length >> 1] : 0;
    axisSigma = Math.max(sigma, 1.4826 * mad);
  }

  const z = new Float64Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const i = y * n + x;
      const onAxis = Math.abs(x - c) <= 1 || Math.abs(y - c) <= 1;
      const s = Math.max(ringSigma[radius[i]], onAxis ? axisSigma : 0);
      z[i] = (residual[i] - med) / s;
    }
  }

  return { size: n, logPower: logP, background, residual, z, sigma, radialProfile: smooth, ringSigma, axisSigma };
}
