/**
 * The engine: owns the image, runs analysis, builds delta layers, composites
 * on request and exports. Runs inside the worker but has no worker-specific
 * code so it can also be driven directly in tests.
 */

import { analyze } from '../analysis/analyze';
import { jpegYccToRgb, rgbToJpegYcc, settleBlocks } from '../dsp/jpegdomain';
import { noiseMap } from '../dsp/noise';
import { measureSurvival } from '../analysis/survival';
import { computeDelta } from '../analysis/apply';
import type { Analysis, AnalysisOptions, Finding } from '../analysis/types';
import { describeFrequency, measurePattern } from '../dsp/pattern';
import { lumaPlane, makePlane, type Plane } from '../dsp/plane';
import type { TileGrid } from '../dsp/tiles';
import { decodeJpegToPlanes, encodeToJpeg, simulateReencode, type JpegExportOptions } from '../io/encode';
import { resolveProfile, type ProfileId } from './profiles';
import { decodeImage, type SourceImage, type SourceMeta } from '../io/decode';
import { parseJpeg } from '../io/jpegParse';
import { psnr, rng, synthesize } from '../synth/generate';
import type { Budget, LayerSetting, LayerStats, LoadedInfo, RenderRequest, RenderResult, RoundTripReport, SynthRequest } from './protocol';

const DELTA_SCALE = 16384; // Int16 fixed point: ±2.0 range, 1/64 of a level resolution

/** Order in which correction stages are applied; later stages are computed on earlier ones. */
export const STAGE_ORDER: Record<Finding['kind'], number> = { periodic: 0, 'jpeg-grid': 1, hotpixels: 2, ringing: 3, noise: 4, chroma: 5, grain: 6 };

/** Findings in the order their layers should be built. */
export function stageSorted<T extends { kind: Finding['kind'] }>(findings: T[]): T[] {
  return [...findings].sort((a, b) => STAGE_ORDER[a.kind] - STAGE_ORDER[b.kind]);
}

interface Layer {
  id: string;
  full: Int16Array[]; // per channel, w*h
  preview: Float32Array[]; // per channel, pw*ph
}

interface Mask {
  w: number;
  h: number;
  data: Uint8Array; // 0..255
}

export class Engine {
  source: SourceImage | null = null;
  analysis: Analysis | null = null;
  layers = new Map<string, Layer>();
  private manualCount = 0;
  /** after a check: the result and the untouched source as the destination re-saves them, full resolution, 8-bit RGB */
  postedResult: Uint8Array | null = null;
  postedSource: Uint8Array | null = null;
  private postedPreview: { result: Float32Array[]; source: Float32Array[] } | null = null;
  masks = new Map<string, Mask>();
  previewScale = 1;
  pw = 0;
  ph = 0;
  previewSource: Float32Array[] = [];
  maskW = 0;
  maskH = 0;

  // --- loading ---------------------------------------------------------------
  async load(file: Blob & { name?: string }): Promise<LoadedInfo> {
    const img = await decodeImage(file);
    return this.setSource(img);
  }

  synth(spec: SynthRequest): LoadedInfo {
    const { w, h } = spec;
    const { noisy } = synthesize({ w, h, seed: spec.seed, sinusoids: spec.sinusoids, noiseSigma: spec.noiseSigma });
    // a little colour so channels are not identical: tint by a smooth gradient
    const r = makePlane(w, h);
    const g = makePlane(w, h);
    const b = makePlane(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const v = noisy.data[i];
        r.data[i] = v * (0.92 + 0.08 * (x / w));
        g.data[i] = v;
        b.data[i] = v * (0.9 + 0.1 * (y / h));
      }
    }
    if (spec.blocks) {
      const rr = rng(spec.seed + 1);
      const bw = Math.ceil(w / 8) + 1;
      const bh = Math.ceil(h / 8) + 1;
      const off = new Float32Array(bw * bh);
      for (let i = 0; i < off.length; i++) off[i] = (rr() - 0.5) * 0.03;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const o = off[(y >> 3) * bw + (x >> 3)];
          const i = y * w + x;
          r.data[i] += o;
          g.data[i] += o;
          b.data[i] += o;
        }
      }
    }
    if (spec.hotPixels > 0) {
      const rr = rng(spec.seed + 2);
      for (let k = 0; k < spec.hotPixels; k++) {
        const x = 1 + Math.floor(rr() * (w - 2));
        const y = 1 + Math.floor(rr() * (h - 2));
        const i = y * w + x;
        const hot = rr() < 0.5 ? 1 : 0;
        r.data[i] = g.data[i] = b.data[i] = hot;
      }
    }
    for (const p of [r, g, b]) for (let i = 0; i < p.data.length; i++) p.data[i] = Math.min(1, Math.max(0, p.data[i]));
    const meta: SourceMeta = {
      fileName: `synthetic-${w}x${h}-seed${spec.seed}.png`,
      mime: 'image/x-synthetic',
      byteLength: 0,
      bitDepth: 32,
      decoder: 'native',
      exif: null,
      exifSummary: { orientation: 1, littleEndian: true },
      dpi: null,
      icc: null,
      xmp: null,
      jpeg: null,
      orientationApplied: 1,
      hasAlpha: false,
    };
    return this.setSource({ w, h, planes: [r, g, b], meta });
  }

  private setSource(img: SourceImage): LoadedInfo {
    this.source = img;
    this.analysis = null;
    this.layers.clear();
    this.masks.clear();
    this.postedResult = null;
    this.postedSource = null;
    this.postedPreview = null;
    const long = Math.max(img.w, img.h);
    const target = 1600;
    const factor = Math.max(1, Math.ceil(long / target));
    this.previewScale = 1 / factor;
    this.pw = Math.max(1, Math.floor(img.w / factor));
    this.ph = Math.max(1, Math.floor(img.h / factor));
    this.previewSource = img.planes.map((p) => areaDownsample(p.data, p.w, p.h, factor, this.pw, this.ph));
    const mlong = Math.max(this.pw, this.ph);
    const mf = Math.max(1, Math.ceil(mlong / 1024));
    this.maskW = Math.max(1, Math.floor(this.pw / mf));
    this.maskH = Math.max(1, Math.floor(this.ph / mf));
    return {
      w: img.w,
      h: img.h,
      meta: img.meta,
      preview: { w: this.pw, h: this.ph, scale: this.previewScale },
      maskSize: { w: this.maskW, h: this.maskH },
    };
  }

  // --- analysis ------------------------------------------------------------
  runAnalysis(options: Partial<AnalysisOptions>, progress: (stage: string, fraction: number) => void): Analysis {
    if (!this.source) throw new Error('no image loaded');
    const m = this.source.meta;
    const a = analyze(
      { planes: this.source.planes, dpi: m.dpi, sourceQuality: m.jpeg?.quality ?? null, sourceSubsampling: m.jpeg?.subsampling ?? null },
      options,
      progress,
    );
    this.analysis = a;
    return a;
  }

  /** A periodic finding at a user-chosen spectrum position, measured like the rest. */
  addPeak(u: number, v: number): Finding {
    if (!this.source || !this.analysis) throw new Error('analyse first');
    const a = this.analysis;
    const planes = this.source.planes;
    const luma = lumaPlane(planes[0], planes[1], planes[2]);
    const grid: TileGrid = { size: a.tile.size, hop: a.tile.hop, xs: a.tile.xs, ys: a.tile.ys };
    const m = measurePattern(luma, grid, u, v);
    const n = a.spectrum.size;
    const c = n >> 1;
    const bx = Math.min(n - 1, Math.max(0, Math.round(m.u * n) + c));
    const by = Math.min(n - 1, Math.max(0, Math.round(m.v * n) + c));
    const i = by * n + bx;
    const excessDb = 10 * (a.spectrum.logPower[i] - a.spectrum.background[i]);
    const z = a.spectrum.z[i];
    const desc = describeFrequency(m.u, m.v);
    let extent = 0;
    for (const p of m.presence) if (p > 0.25) extent++;
    extent /= m.presence.length;
    const id = `manual-${this.manualCount++}`;
    const f: Finding = {
      id,
      kind: 'periodic',
      label: 'Added by hand',
      description: `${desc.period.toFixed(2)} px period at ${desc.angleDeg.toFixed(1)}°, ${(m.amplitude * 255).toFixed(2)} levels peak, ${excessDb.toFixed(1)} dB over background`,
      severity: Math.max(0, Math.min(1, (excessDb - 6) / 40)),
      extent,
      enabled: true,
      confidence: z > 8 ? 'medium' : 'low',
      note: z < 4 ? 'Nothing stands out of the background here; the subtraction will be tiny.' : '',
      strength: 1,
      method: m.stationary ? 'parametric' : 'parametric-tiled',
      details: {
        u: m.u,
        v: m.v,
        period: desc.period,
        angleDeg: desc.angleDeg,
        lpi: this.source.meta.dpi ? desc.radius * this.source.meta.dpi : null,
        amplitude: m.amplitude,
        amplitudeP90: m.amplitudeP90,
        phase: m.phase,
        excessDb,
        z,
        widthBins: 2,
        stationary: m.stationary,
        coherence: m.phaseCoherence,
        amplitudeCv: m.amplitudeCv,
        harmonics: [],
        presence: m.presence,
        tilesX: m.tilesX,
        tilesY: m.tilesY,
        onAxis: Math.abs(m.u) < 1.2 / n || Math.abs(m.v) < 1.2 / n,
        jpegGridAligned: false,
        category: 'moire',
      },
    };
    a.findings.push(f);
    return f;
  }

  // --- layers ----------------------------------------------------------------
  /**
   * Corrections are staged: periodic → JPEG grid → hot pixels → noise. A later
   * stage is computed on the image with every earlier stage applied in full,
   * otherwise a deblocker sees a moiré's swings as block steps and a denoiser
   * measures the moiré as noise.
   */
  private stageInput(finding: Finding): [Plane, Plane, Plane] {
    const src = this.source!;
    const stage = STAGE_ORDER[finding.kind];
    const earlier = this.analysis!.findings.filter((f) => STAGE_ORDER[f.kind] < stage && this.layers.has(f.id)).map((f) => ({ id: f.id, strength: 1 }));
    if (!earlier.length) return src.planes;
    const out = [0, 1, 2].map((c) => {
      const data = new Float32Array(src.w * src.h);
      this.compositeFull(c, 0, 0, src.w, src.h, earlier, 1, data);
      return { w: src.w, h: src.h, data };
    });
    return out as [Plane, Plane, Plane];
  }

  buildLayer(finding: Finding, progress: (fraction: number) => void): LayerStats {
    if (!this.source || !this.analysis) throw new Error('analyse first');
    const t0 = performance.now();
    const deltas = computeDelta(this.stageInput(finding), this.analysis, finding, progress);
    const { w, h } = this.source;
    const factor = Math.round(1 / this.previewScale);
    const full = deltas.map((d) => {
      const q = new Int16Array(w * h);
      for (let i = 0; i < q.length; i++) {
        const v = Math.round(d.data[i] * DELTA_SCALE);
        q[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : v;
      }
      return q;
    });
    const preview = deltas.map((d) => areaDownsample(d.data, w, h, factor, this.pw, this.ph));
    this.layers.set(finding.id, { id: finding.id, full, preview });
    // stats on the luma delta
    const lum = lumaPlane(deltas[0], deltas[1], deltas[2]);
    let s = 0;
    let mx = 0;
    for (const v of lum.data) {
      s += v * v;
      if (Math.abs(v) > mx) mx = Math.abs(v);
    }
    return { id: finding.id, rms: Math.sqrt(s / lum.data.length) * 255, max: mx * 255, ms: Math.round(performance.now() - t0) };
  }

  setMask(findingId: string, w: number, h: number, data: Uint8Array): void {
    this.masks.set(findingId, { w, h, data });
  }

  /** Mask seeded from the per-tile presence of a periodic finding, feathered. */
  presenceMask(finding: Finding): Mask {
    const w = this.maskW;
    const h = this.maskH;
    const data = new Uint8Array(w * h);
    if (finding.kind !== 'periodic') {
      data.fill(255);
      return { w, h, data };
    }
    const d = finding.details;
    const tx = d.tilesX;
    const ty = d.tilesY;
    // tile centres map to the mask grid; bilinear between them
    for (let y = 0; y < h; y++) {
      const fy = ty > 1 ? Math.min(ty - 1, Math.max(0, ((y + 0.5) / h) * ty - 0.5)) : 0;
      const y0 = Math.floor(fy);
      const y1 = Math.min(ty - 1, y0 + 1);
      const t = fy - y0;
      for (let x = 0; x < w; x++) {
        const fx = tx > 1 ? Math.min(tx - 1, Math.max(0, ((x + 0.5) / w) * tx - 0.5)) : 0;
        const x0 = Math.floor(fx);
        const x1 = Math.min(tx - 1, x0 + 1);
        const s = fx - x0;
        const a = d.presence[y0 * tx + x0] * (1 - s) + d.presence[y0 * tx + x1] * s;
        const b = d.presence[y1 * tx + x0] * (1 - s) + d.presence[y1 * tx + x1] * s;
        const v = a * (1 - t) + b * t;
        // presence is amplitude relative to the strongest tile; treat ≥ 0.6 as full
        data[y * w + x] = Math.round(255 * Math.min(1, Math.max(0, v / 0.6)));
      }
    }
    return { w, h, data };
  }

  /** Mask from the image's own brightness: 1 inside [lo, hi] (0..1 luma), feathered over `soft`. */
  luminosityMask(lo: number, hi: number, soft: number): Mask {
    const w = this.maskW;
    const h = this.maskH;
    const data = new Uint8Array(w * h);
    const [r, g, b] = this.previewSource;
    const pw = this.pw;
    const ph = this.ph;
    const fx = pw / w;
    const fy = ph / h;
    const smooth = (x: number): number => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        // area average of the preview luma under this mask cell
        let s = 0;
        let n = 0;
        const x0 = Math.floor(x * fx);
        const x1 = Math.max(x0 + 1, Math.floor((x + 1) * fx));
        const y0 = Math.floor(y * fy);
        const y1 = Math.max(y0 + 1, Math.floor((y + 1) * fy));
        for (let yy = y0; yy < y1 && yy < ph; yy++) {
          for (let xx = x0; xx < x1 && xx < pw; xx++) {
            const i = yy * pw + xx;
            s += 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
            n++;
          }
        }
        const l = n ? s / n : 0;
        const m = Math.min(smooth((l - lo + soft) / Math.max(1e-6, soft)), smooth((hi - l + soft) / Math.max(1e-6, soft)));
        data[y * w + x] = Math.round(255 * Math.max(0, Math.min(1, m)));
      }
    }
    return { w, h, data };
  }

  // --- compositing -----------------------------------------------------------
  private maskAt(mask: Mask | undefined, x: number, y: number, w: number, h: number): number {
    if (!mask) return 1;
    const fx = Math.min(mask.w - 1, Math.max(0, ((x + 0.5) / w) * mask.w - 0.5));
    const fy = Math.min(mask.h - 1, Math.max(0, ((y + 0.5) / h) * mask.h - 0.5));
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(mask.w - 1, x0 + 1);
    const y1 = Math.min(mask.h - 1, y0 + 1);
    const s = fx - x0;
    const t = fy - y0;
    const a = mask.data[y0 * mask.w + x0] * (1 - s) + mask.data[y0 * mask.w + x1] * s;
    const b = mask.data[y1 * mask.w + x0] * (1 - s) + mask.data[y1 * mask.w + x1] * s;
    return (a * (1 - t) + b * t) / 255;
  }

  /** Composite one channel over a rectangle of the full-resolution image. */
  private compositeFull(ch: number, x0: number, y0: number, cw: number, chh: number, layers: LayerSetting[], master: number, out: Float32Array): void {
    const src = this.source!;
    const { w, h } = src;
    const sp = src.planes[ch].data;
    for (let y = 0; y < chh; y++) {
      const sy = y0 + y;
      for (let x = 0; x < cw; x++) out[y * cw + x] = sp[sy * w + x0 + x];
    }
    for (const ls of layers) {
      const layer = this.layers.get(ls.id);
      if (!layer || ls.strength === 0) continue;
      const k = (master * ls.strength) / DELTA_SCALE;
      const mask = this.masks.get(ls.id);
      const d = layer.full[ch];
      for (let y = 0; y < chh; y++) {
        const sy = y0 + y;
        for (let x = 0; x < cw; x++) {
          const sx = x0 + x;
          const m = mask ? this.maskAt(mask, sx, sy, w, h) : 1;
          out[y * cw + x] += k * m * d[sy * w + sx];
        }
      }
    }
  }

  private compositePreview(ch: number, layers: LayerSetting[], master: number): Float32Array {
    const out = new Float32Array(this.previewSource[ch]);
    const pw = this.pw;
    const ph = this.ph;
    for (const ls of layers) {
      const layer = this.layers.get(ls.id);
      if (!layer || ls.strength === 0) continue;
      const k = master * ls.strength;
      const mask = this.masks.get(ls.id);
      const d = layer.preview[ch];
      for (let y = 0; y < ph; y++) {
        for (let x = 0; x < pw; x++) {
          const i = y * pw + x;
          const m = mask ? this.maskAt(mask, x, y, pw, ph) : 1;
          out[i] += k * m * d[i];
        }
      }
    }
    return out;
  }

  render(req: RenderRequest): RenderResult {
    if (!this.source) throw new Error('no image');
    const region = req.region;
    const isPreview = region.kind === 'preview';
    const w = region.kind === 'preview' ? this.pw : region.w;
    const h = region.kind === 'preview' ? this.ph : region.h;
    const rgba = new Uint8ClampedArray(new ArrayBuffer(w * h * 4));
    const chans: Float32Array[] = [];
    const srcChans: Float32Array[] = [];
    if (req.mode === 'posted' || req.mode === 'posted-source') {
      const buf = req.mode === 'posted' ? this.postedResult : this.postedSource;
      if (!buf) throw new Error('run a check first');
      if (isPreview) {
        const pv = req.mode === 'posted' ? this.postedPreview!.result : this.postedPreview!.source;
        for (let i = 0, j = 0; i < w * h; i++, j += 4) {
          rgba[j] = Math.round(255 * pv[0][i]);
          rgba[j + 1] = Math.round(255 * pv[1][i]);
          rgba[j + 2] = Math.round(255 * pv[2][i]);
          rgba[j + 3] = 255;
        }
      } else {
        const r = region as { x0: number; y0: number; w: number; h: number };
        const W = this.source.w;
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const si = ((r.y0 + y) * W + r.x0 + x) * 3;
            const j = (y * w + x) * 4;
            rgba[j] = buf[si];
            rgba[j + 1] = buf[si + 1];
            rgba[j + 2] = buf[si + 2];
            rgba[j + 3] = 255;
          }
        }
      }
      return { w, h, rgba };
    }
    for (let c = 0; c < 3; c++) {
      let comp: Float32Array;
      let src: Float32Array;
      if (isPreview) {
        src = this.previewSource[c];
        comp = req.mode === 'original' ? src : this.compositePreview(c, req.layers, req.master);
      } else {
        const r = req.region as { x0: number; y0: number; w: number; h: number };
        src = new Float32Array(w * h);
        this.compositeFull(c, r.x0, r.y0, r.w, r.h, [], 1, src);
        if (req.mode === 'original') comp = src;
        else {
          comp = new Float32Array(w * h);
          this.compositeFull(c, r.x0, r.y0, r.w, r.h, req.layers, req.master, comp);
        }
      }
      chans.push(comp);
      srcChans.push(src);
    }
    const mask = req.mode === 'mask' && req.maskId ? this.masks.get(req.maskId) : undefined;
    const fullW = isPreview ? this.pw : this.source.w;
    const fullH = isPreview ? this.ph : this.source.h;
    const rx = isPreview ? 0 : (req.region as { x0: number }).x0;
    const ry = isPreview ? 0 : (req.region as { y0: number }).y0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const j = i * 4;
        if (req.mode === 'difference') {
          for (let c = 0; c < 3; c++) rgba[j + c] = Math.round(255 * (0.5 + req.diffGain * (chans[c][i] - srcChans[c][i])));
        } else if (req.mode === 'mask') {
          const m = req.maskId ? (mask ? this.maskAt(mask, rx + x, ry + y, fullW, fullH) : 1) : 0;
          const lum = 0.299 * srcChans[0][i] + 0.587 * srcChans[1][i] + 0.114 * srcChans[2][i];
          rgba[j] = Math.round(255 * (lum * 0.6 + 0.4 * m));
          rgba[j + 1] = Math.round(255 * (lum * 0.6 + 0.15 * m));
          rgba[j + 2] = Math.round(255 * (lum * 0.6));
        } else {
          for (let c = 0; c < 3; c++) rgba[j + c] = Math.round(255 * chans[c][i]);
        }
        rgba[j + 3] = 255;
      }
    }
    return { w, h, rgba };
  }

  /** Full-resolution composite of all three channels. */
  compose(layers: LayerSetting[], master: number): [Plane, Plane, Plane] {
    const src = this.source!;
    const out = [0, 1, 2].map((c) => {
      const data = new Float32Array(src.w * src.h);
      this.compositeFull(c, 0, 0, src.w, src.h, layers, master, data);
      for (let i = 0; i < data.length; i++) data[i] = data[i] < 0 ? 0 : data[i] > 1 ? 1 : data[i];
      return { w: src.w, h: src.h, data };
    });
    return out as [Plane, Plane, Plane];
  }

  // --- export --------------------------------------------------------------
  async exportJpeg(
    layers: LayerSetting[],
    findings: Finding[],
    master: number,
    profileId: ProfileId,
    customOptions: JpegExportOptions,
    budget: Budget,
    analysisOptions: Partial<AnalysisOptions>,
    progress: (stage: string, fraction: number) => void,
  ): Promise<{ jpeg: Uint8Array; report: RoundTripReport }> {
    if (!this.source || !this.analysis) throw new Error('analyse first');
    const profile = resolveProfile(profileId, customOptions);
    const opts = profile.encode;
    const meta = this.source.meta;
    const grainWant = findings.find((f) => f.kind === 'grain' && f.enabled && layers.some((l) => l.id === f.id && l.strength > 0));
    const want = grainWant && grainWant.kind === 'grain' ? grainWant.details.targetLevels : 0;
    // Grain put back is partly rounded away by the destination's re-save. The
    // file is written once, the downloader's copy measured, and if the grain
    // there is short of the amount set, the grain is scaled up and the file
    // written again, so what was set is what the copy has.
    let grainScale = 1;
    let composed!: [Plane, Plane, Plane];
    let settled: RoundTripReport['settled'] = null;
    let jpeg!: Uint8Array;
    let enc!: Awaited<ReturnType<typeof encodeToJpeg>>;
    let ours!: [Plane, Plane, Plane];
    let resDest!: [Plane, Plane, Plane];
    let srcDest!: [Plane, Plane, Plane];
    let survival!: ReturnType<typeof measureSurvival>;
    for (let round = 0; round < 2; round++) {
      const useLayers = grainScale === 1 ? layers : layers.map((l) => (l.id === grainWant?.id ? { ...l, strength: l.strength * grainScale } : l));
      progress('compose', 0);
      composed = this.compose(useLayers, master);
      // Block settling at the destination's quality: when the file is written
      // with the destination's own tables and the block fix is on, the luma is
      // put on that quality's coefficient grid with block boundaries settled,
      // so the destination's re-save cannot bring the steps back.
      settled = null;
      const blockFixOn = findings.some((f) => f.kind === 'jpeg-grid' && f.enabled) && layers.some((l) => l.id === 'jpeg-grid' && l.strength > 0);
      if (opts.tables === 'annex-k' && !opts.trellis && blockFixOn && master > 0) {
        progress('settle', 0);
        const [Y, Cb, Cr] = rgbToJpegYcc(composed[0], composed[1], composed[2]);
        const grainLevels = noiseMap(Y, 32).refSigma * 255;
        const r = settleBlocks(Y, opts.quality, { grainLevels });
        const rgb = jpegYccToRgb(r.plane, Cb, Cr);
        composed[0] = rgb[0];
        composed[1] = rgb[1];
        composed[2] = rgb[2];
        settled = { quality: opts.quality, changedBlocks: r.changed, blocks: r.blocks, meanChangeLevels: r.meanChange * 255 };
        progress('settle', 1);
      }
      progress('encode', 0);
      enc = await encodeToJpeg(composed, opts, { exif: meta.exif, icc: meta.icc, xmp: meta.xmp }, (pass) =>
        progress('encode', Math.min(1, pass / Math.max(1, opts.fixedPointPasses + 1))),
      );
      jpeg = enc.jpeg;

      // --- what a downloader gets: our file after the destination's re-save,
      // and the untouched photo after the same re-save ------------------------
      progress('check', 0);
      ours = await decodeJpegToPlanes(jpeg);
      resDest = (await simulateReencode(ours, profile.check.options)).planes;
      progress('check', 0.4);
      if (round === 0) srcDest = (await simulateReencode(this.source.planes, profile.check.options)).planes;
      progress('check', 0.7);
      survival = measureSurvival(this.source.planes, srcDest, resDest, findings);
      const g = survival.fixes.find((f) => f.kind === 'grain');
      if (round === 0 && g && want > 0 && g.resultOnDest > 0.05 && g.resultOnDest < 0.85 * want) {
        grainScale = Math.min(2.5, want / g.resultOnDest);
        progress('regrain', 0);
        continue;
      }
      break;
    }
    this.storePosted(resDest, srcDest);

    // encode error of our own file vs the composed float result
    const lumComp = lumaPlane(composed[0], composed[1], composed[2]);
    const lumOurs = lumaPlane(ours[0], ours[1], ours[2]);
    let sq = 0;
    for (let i = 0; i < lumOurs.data.length; i++) sq += (lumOurs.data[i] - lumComp.data[i]) ** 2;
    const encodeRmsLevels = Math.sqrt(sq / lumOurs.data.length) * 255;
    const encodePsnr = psnr(lumComp, lumOurs);
    progress('check', 0.85);

    // residual periodic components on the downloader's copy
    const info = parseJpeg(jpeg);
    const a = analyze(
      { planes: resDest, dpi: this.source.meta.dpi, sourceQuality: profile.check.options.quality, sourceSubsampling: profile.check.options.chroma },
      { ...analysisOptions, quick: true, maxWelchTiles: 16 },
      (st, fr) => progress('check:' + st, fr),
    );
    const residual = a.findings.filter(
      (f) => f.kind === 'periodic' && f.details.excessDb > budget.maxPeakDb && Math.max(f.details.amplitude, f.details.amplitudeP90) * 255 >= budget.minPeakLevels,
    );
    const reasons: string[] = [];
    if (residual.length)
      reasons.push(
        `${residual.length} repeating pattern${residual.length > 1 ? 's are' : ' is'} visible in the downloader's copy (${residual
          .map((f) => (f.kind === 'periodic' ? `every ${f.details.period.toFixed(1)} px, ${(Math.max(f.details.amplitude, f.details.amplitudeP90) * 255).toFixed(1)} levels` : ''))
          .join('; ')})`,
      );
    const bs = Math.max(a.blockGrid.strengthX, a.blockGrid.strengthY);
    const bstep = a.blockGrid.stepLevels;
    const gridVisible = a.blockGrid.detected && bs > budget.maxBlockStrength && bstep >= 0.5;
    // fixes the check can see that the first analysis could not
    const suggested: Finding[] = [];
    const src = { findings };
    const gridAlready = src.findings.some((f) => f.kind === 'jpeg-grid');
    const outGrid = a.findings.find((f) => f.kind === 'jpeg-grid');
    if (outGrid && !gridAlready && gridVisible) suggested.push({ ...outGrid, id: 'jpeg-grid', note: 'Seen in the downloader’s copy; it was hidden under the patterns when the photo was first checked.' });
    for (const f of residual) {
      const tol = 2 / a.spectrum.size;
      const known = src.findings.some(
        (g) => g.kind === 'periodic' && f.kind === 'periodic' && ((Math.abs(g.details.u - f.details.u) < tol && Math.abs(g.details.v - f.details.v) < tol) || (Math.abs(g.details.u + f.details.u) < tol && Math.abs(g.details.v + f.details.v) < tol)),
      );
      if (!known) suggested.push({ ...f, id: `verify-${suggested.length}`, note: 'Seen in the downloader’s copy, not when the photo was first checked.' });
    }
    const report: RoundTripReport = {
      encodePsnr,
      encodeRmsLevels,
      bytes: jpeg.length,
      quality: opts.quality,
      chroma: info.subsampling,
      profile: profile.id,
      profileLabel: profile.label,
      passes: enc.passes,
      stable: enc.stable,
      settled,
      grainScale,
      checkLabel: profile.check.label,
      survival,
      residual,
      residualPeriodic: residual.length,
      blockGrid: { detected: a.blockGrid.detected, strength: bs, stepLevels: bstep },
      noise: { globalSigmaLevels: survival.grainLevels.resultOnDest, expectedLevels: survival.grainLevels.sourceOnDest },
      suggested,
      pass: survival.pass && residual.length === 0,
      reasons,
      budget,
    };
    return { jpeg, report };
  }

  private storePosted(res: [Plane, Plane, Plane], srcd: [Plane, Plane, Plane]): void {
    const w = res[0].w;
    const h = res[0].h;
    const pack = (p: [Plane, Plane, Plane]): Uint8Array => {
      const out = new Uint8Array(w * h * 3);
      for (let i = 0, j = 0; i < w * h; i++, j += 3) {
        out[j] = Math.round(p[0].data[i] * 255);
        out[j + 1] = Math.round(p[1].data[i] * 255);
        out[j + 2] = Math.round(p[2].data[i] * 255);
      }
      return out;
    };
    this.postedResult = pack(res);
    this.postedSource = pack(srcd);
    const factor = Math.round(1 / this.previewScale);
    this.postedPreview = {
      result: res.map((p) => areaDownsample(p.data, w, h, factor, this.pw, this.ph)),
      source: srcd.map((p) => areaDownsample(p.data, w, h, factor, this.pw, this.ph)),
    };
  }

  /** Re-run the analysis on the current composite (no encode), for the residual view. */
  verifyResult(layers: LayerSetting[], master: number, analysisOptions: Partial<AnalysisOptions>, progress: (stage: string, fraction: number) => void): Analysis {
    const composed = this.compose(layers, master);
    return analyze({ planes: composed, dpi: this.source!.meta.dpi, sourceQuality: null, sourceSubsampling: null }, analysisOptions, progress);
  }
}

/** Area-average downsample by an integer factor into a pw x ph buffer. */
export function areaDownsample(src: Float32Array, w: number, _h: number, factor: number, pw: number, ph: number): Float32Array {
  const out = new Float32Array(pw * ph);
  if (factor === 1) {
    for (let y = 0; y < ph; y++) out.set(src.subarray(y * w, y * w + pw), y * pw);
    return out;
  }
  const inv = 1 / (factor * factor);
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      let s = 0;
      for (let dy = 0; dy < factor; dy++) {
        const row = (y * factor + dy) * w + x * factor;
        for (let dx = 0; dx < factor; dx++) s += src[row + dx];
      }
      out[y * pw + x] = s * inv;
    }
  }
  return out;
}
