/**
 * The wide glow around subjects.
 *
 * Exposure fusion, which phones use to fit a scene's range into a picture,
 * leaves a soft band along strong brightness boundaries: the bright side
 * gets brighter close to the edge, the dark side darker (the low-frequency
 * halo of exposure fusion; Hessel, "Extended Exposure Fusion", 2019). It is
 * several to tens of pixels wide, too wide for the edge-outline fix, and it
 * is what makes a subject look cut out and pasted on.
 *
 * Along every strong straight edge the profile across the edge is read out
 * to 48 px on each side. Where a side is smooth (sky, wall, skin) its real
 * level and gradient are fitted from 30–48 px out; a band within ~28 px that
 * bulges away from that fit in the direction that exaggerates the edge,
 * and fades with distance, is the glow, and is flattened back onto the fit.
 * Sides with texture or other edges are left alone, so real structure is
 * never touched.
 */

import { boxBlur } from './noise';
import type { Plane } from './plane';

const R = 48;

function bilinear(d: Float32Array, w: number, h: number, x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  if (x0 < 0 || y0 < 0 || x0 >= w - 1 || y0 >= h - 1) return NaN;
  const tx = x - x0;
  const ty = y - y0;
  const i = y0 * w + x0;
  return (d[i] * (1 - tx) + d[i + 1] * tx) * (1 - ty) + (d[i + w] * (1 - tx) + d[i + w + 1] * tx) * ty;
}

export interface GlowResult {
  /** median glow strength next to the edge (mean over 4–12 px), levels */
  glowLevels: number;
  /** edge sides where a glow was found (0 when the photo shows no consistent glow) */
  sides: number;
  /** share of the smooth edge sides that show it */
  share: number;
  /** fraction of pixels the fix touches */
  coverage: number;
  /** the brightness change that removes it */
  delta: Plane;
}

/**
 * Find the glow and the change that removes it (scaled by `amount`). The
 * delta is for brightness; apply it to all three channels or to luma.
 */
export function glowFix(Y: Plane, amount = 1): GlowResult {
  const { w, h } = Y;
  const d = Y.data;
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  const mag = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] = 0.5 * (d[i + 1] - d[i - 1]);
      gy[i] = 0.5 * (d[i + w] - d[i - w]);
      mag[i] = Math.hypot(gx[i], gy[i]);
    }
  const xx = new Float32Array(w * h);
  const yy = new Float32Array(w * h);
  const xy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    xx[i] = gx[i] * gx[i];
    yy[i] = gy[i] * gy[i];
    xy[i] = gx[i] * gy[i];
  }
  const bxx = boxBlur(xx, w, h, 3);
  const byy = boxBlur(yy, w, h, 3);
  const bxy = boxBlur(xy, w, h, 3);

  const sum = new Float32Array(w * h);
  const cnt = new Float32Array(w * h);
  const amps: number[] = [];
  let examined = 0;
  const prof = new Float32Array(2 * R + 1);
  const g = new Float32Array(R + 1);
  for (let y = R + 1; y < h - R - 1; y += 2) {
    for (let x = R + 1; x < w - R - 1; x++) {
      const i = y * w + x;
      const m = mag[i];
      if (m <= 8 / 255) continue;
      const tr = bxx[i] + byy[i];
      if (tr <= 0 || ((bxx[i] - byy[i]) ** 2 + 4 * bxy[i] * bxy[i]) / (tr * tr) < 0.75) continue;
      const nx = gx[i] / m;
      const ny = gy[i] / m;
      if (!(m >= bilinear(mag, w, h, x + nx, y + ny) && m >= bilinear(mag, w, h, x - nx, y - ny))) continue;
      let ok = true;
      for (let k = -R; k <= R; k++) {
        const v = bilinear(d, w, h, x + nx * k, y + ny * k);
        if (Number.isNaN(v)) {
          ok = false;
          break;
        }
        prof[k + R] = v;
      }
      if (!ok) continue;
      // edge height from 4–8 px either side
      let lo = 0;
      let hi = 0;
      for (let k = 4; k <= 8; k++) {
        lo += prof[R - k];
        hi += prof[R + k];
      }
      const c = (hi - lo) / 5;
      if (c < 20 / 255) continue;
      for (const s of [-1, 1]) {
        const v = (t: number): number => prof[R + s * t];
        // the side must be smooth: no other edge between 4 and 48 px
        let maxStep = 0;
        for (let t = 4; t < R; t++) maxStep = Math.max(maxStep, Math.abs(v(t + 1) - v(t)));
        if (maxStep > Math.max(4 / 255, 0.12 * c)) continue;
        // the side's own level and gradient, fitted 30–48 px out
        let st = 0;
        let sv = 0;
        let stt = 0;
        let stv = 0;
        let n = 0;
        for (let t = 30; t <= R; t++) {
          st += t;
          sv += v(t);
          stt += t * t;
          stv += t * v(t);
          n++;
        }
        const b = (n * stv - st * sv) / (n * stt - st * st);
        const a = (sv - b * st) / n;
        let sd = 0;
        for (let t = 30; t <= R; t++) sd += (v(t) - (a + b * t)) ** 2;
        sd = Math.sqrt(sd / n);
        if (sd > Math.max(2 / 255, 0.06 * c)) continue;
        examined++;
        for (let t = 0; t <= R; t++) g[t] = v(t) - (a + b * t);
        // a glow exaggerates the edge (bright side up, dark side down) and fades out
        let near = 0;
        for (let t = 4; t <= 12; t++) near += g[t];
        near /= 9;
        let mid = 0;
        for (let t = 20; t <= 28; t++) mid += g[t];
        mid /= 9;
        const A = s * near;
        // well clear of this side's own noise (a 9-sample mean has a third of it)
        if (A < Math.max(1.5 / 255, (4 * sd) / 3) || Math.abs(mid) > Math.abs(near) * 0.8) continue;
        amps.push(A * 255);
        // flatten the band onto the fit: full from 4 to 26 px, fading in from the edge (2→4) and out (26→30)
        for (let tt = 2; tt <= 30; tt += 0.5) {
          const t0 = Math.floor(tt);
          const gv = g[t0] * (1 - (tt - t0)) + g[Math.min(R, t0 + 1)] * (tt - t0);
          const fade = Math.min(1, (tt - 2) / 2) * Math.min(1, (30 - tt) / 4);
          // only remove the part that exaggerates the edge
          const corr = s * gv > 0 ? -gv * fade : 0;
          const px = Math.round(x + nx * s * tt);
          const py = Math.round(y + ny * s * tt);
          const j = py * w + px;
          sum[j] += corr;
          cnt[j] += 1;
        }
      }
    }
  }
  // average where several edges reach a pixel, then fill the gaps between
  // edge samples (normalised smoothing over the covered area)
  const avg = new Float32Array(w * h);
  const cov = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (cnt[i] > 0) {
      avg[i] = sum[i] / cnt[i];
      cov[i] = 1;
    }
  }
  const sa = boxBlur(boxBlur(avg, w, h, 2), w, h, 2);
  const sc = boxBlur(boxBlur(cov, w, h, 2), w, h, 2);
  const delta: Plane = { w, h, data: new Float32Array(w * h) };
  let touched = 0;
  // a photo either has the phone's glow along its edges or it does not: act
  // only when it shows on a good share of the smooth edge sides
  const consistent = examined > 0 && amps.length / examined >= 0.35 && amps.length >= 20;
  for (let i = 0; i < w * h && consistent; i++) {
    if (sc[i] > 0.15) {
      delta.data[i] = (sa[i] / sc[i]) * Math.min(1, sc[i] / 0.4) * amount;
      if (Math.abs(delta.data[i]) > 0.5 / 255) touched++;
    }
  }
  amps.sort((p, q) => p - q);
  return { glowLevels: consistent && amps.length ? amps[amps.length >> 1] : 0, sides: consistent ? amps.length : 0, share: examined ? amps.length / examined : 0, coverage: touched / (w * h), delta };
}
