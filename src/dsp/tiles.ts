import { fft2d, fftshift2d, prevPow2 } from './fft';
import { extractTile, type Plane } from './plane';
import { window2d, type WindowKind } from './window';

export interface TileGrid {
  size: number; // tile edge, power of two
  hop: number; // size / 2
  xs: number[]; // tile origins along x
  ys: number[]; // tile origins along y
}

/**
 * Square power-of-two tiles at 50 % overlap covering the whole plane. The last
 * tile in each direction is clamped to the edge, so coverage is complete even
 * when the image size is not a multiple of the hop.
 */
export function tileGrid(w: number, h: number, maxSize = 1024, minSize = 64): TileGrid {
  let size = Math.min(maxSize, prevPow2(Math.min(w, h)));
  if (size < minSize) size = minSize;
  const hop = size >> 1;
  const axis = (n: number): number[] => {
    if (n <= size) return [0];
    const out: number[] = [];
    for (let o = 0; o + size < n; o += hop) out.push(o);
    out.push(n - size);
    return out;
  };
  return { size, hop, xs: axis(w), ys: axis(h) };
}

export interface WelchResult {
  size: number;
  /** Averaged power |F|²/(Σw)² per bin, fftshifted so DC is at (size/2, size/2). */
  power: Float64Array;
  tiles: number;
  window: WindowKind;
}

/**
 * Welch-averaged power spectrum over all tiles. Each tile has its mean removed
 * and is windowed before the FFT; the power is normalised by (Σw)², so a
 * sinusoid of amplitude A shows a peak of height (A/2)² at its frequency.
 */
export function welchSpectrum(
  plane: Plane,
  grid: TileGrid,
  kind: WindowKind = 'hann',
  onProgress?: (done: number, total: number) => void,
): WelchResult {
  const n = grid.size;
  const win = window2d(kind, n, n);
  const acc = new Float64Array(n * n);
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  const tile = new Float64Array(n * n);
  const total = grid.xs.length * grid.ys.length;
  let done = 0;
  const norm = 1 / (win.sum * win.sum);
  for (const y0 of grid.ys) {
    for (const x0 of grid.xs) {
      extractTile(plane, x0, y0, n, n, tile);
      let mean = 0;
      for (let i = 0; i < tile.length; i++) mean += tile[i];
      mean /= tile.length;
      for (let i = 0; i < tile.length; i++) {
        re[i] = (tile[i] - mean) * win.data[i];
        im[i] = 0;
      }
      fft2d(re, im, n, n, false);
      for (let i = 0; i < acc.length; i++) acc[i] += (re[i] * re[i] + im[i] * im[i]) * norm;
      done++;
      onProgress?.(done, total);
    }
  }
  for (let i = 0; i < acc.length; i++) acc[i] /= total;
  return { size: n, power: fftshift2d(acc, n, n), tiles: total, window: kind };
}

/** Bin index (shifted) -> frequency in cycles/pixel. */
export function binToFreq(bin: number, size: number): number {
  return (bin - (size >> 1)) / size;
}

/** Frequency in cycles/pixel -> fractional shifted bin index. */
export function freqToBin(f: number, size: number): number {
  return f * size + (size >> 1);
}

/** An evenly spread subset of the tiles, at most `maxTiles`. */
export function subsampleGrid(grid: TileGrid, maxTiles: number): TileGrid {
  const total = grid.xs.length * grid.ys.length;
  if (total <= maxTiles) return grid;
  const ratio = Math.sqrt(maxTiles / total);
  const pick = (arr: number[]): number[] => {
    const k = Math.max(1, Math.round(arr.length * ratio));
    if (k >= arr.length) return arr;
    const out: number[] = [];
    for (let i = 0; i < k; i++) out.push(arr[Math.round((i * (arr.length - 1)) / Math.max(1, k - 1))]);
    return Array.from(new Set(out));
  };
  return { size: grid.size, hop: grid.hop, xs: pick(grid.xs), ys: pick(grid.ys) };
}
