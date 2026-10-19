/**
 * Spectrum panel: log-power heat map of the averaged spectrum with rings on
 * every finding, a crosshair readout under the cursor, and click-to-add.
 */

import type { Analysis, Finding } from '../analysis/types';

// perceptual-ish ramp: near-black → plum → orange → pale yellow
const STOPS: [number, number, number][] = [
  [8, 6, 20],
  [60, 15, 90],
  [135, 30, 110],
  [205, 70, 80],
  [245, 140, 40],
  [252, 220, 130],
  [255, 250, 220],
];

function ramp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(0.9999, t)) * (STOPS.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = STOPS[i];
  const b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export interface SpectrumCallbacks {
  onHover: (info: string) => void;
  onClick: (u: number, v: number) => void;
  onSelect: (id: string) => void;
}

export class SpectrumPanel {
  readonly el: HTMLDivElement;
  private heat: HTMLCanvasElement;
  private overlay: HTMLCanvasElement;
  private analysis: Analysis | null = null;
  private dpi: number | null = null;
  private selected: string | null = null;
  private lo = 0;
  private hi = 1;
  addMode = false;

  constructor(private cb: SpectrumCallbacks) {
    this.el = document.createElement('div');
    this.el.className = 'spectrum-wrap';
    this.heat = document.createElement('canvas');
    this.overlay = document.createElement('canvas');
    this.overlay.className = 'overlay';
    this.el.append(this.heat, this.overlay);
    new ResizeObserver(() => this.drawOverlay(null)).observe(this.el);
    this.overlay.addEventListener('pointermove', (e) => this.hover(e));
    this.overlay.addEventListener('pointerleave', () => {
      this.drawOverlay(null);
      this.cb.onHover('');
    });
    this.overlay.addEventListener('click', (e) => this.click(e));
  }

  setAnalysis(a: Analysis | null, dpi: number | null): void {
    this.analysis = a;
    this.dpi = dpi;
    if (!a) {
      this.heat.getContext('2d')!.clearRect(0, 0, this.heat.width, this.heat.height);
      this.drawOverlay(null);
      return;
    }
    const n = a.spectrum.size;
    this.heat.width = n;
    this.heat.height = n;
    const lp = a.spectrum.logPower;
    const sorted = Float32Array.from(lp).sort();
    this.lo = sorted[Math.floor(sorted.length * 0.02)];
    this.hi = sorted[Math.floor(sorted.length * 0.999)];
    const img = new ImageData(n, n);
    for (let i = 0; i < n * n; i++) {
      const t = (lp[i] - this.lo) / (this.hi - this.lo);
      const [r, g, b] = ramp(t);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    this.heat.getContext('2d')!.putImageData(img, 0, 0);
    this.drawOverlay(null);
  }

  setSelected(id: string | null): void {
    this.selected = id;
    this.drawOverlay(null);
  }

  private geometry(): { size: number; n: number; scale: number } {
    const r = this.el.getBoundingClientRect();
    const size = Math.min(r.width, r.height);
    const n = this.analysis?.spectrum.size ?? 1;
    return { size, n, scale: size / n };
  }

  private freqToScreen(u: number, v: number): { x: number; y: number } {
    const { n, scale } = this.geometry();
    const c = n >> 1;
    return { x: (u * n + c + 0.5) * scale, y: (v * n + c + 0.5) * scale };
  }

  private screenToFreq(x: number, y: number): { u: number; v: number } {
    const { n, scale } = this.geometry();
    const c = n >> 1;
    return { u: (x / scale - c - 0.5) / n, v: (y / scale - c - 0.5) / n };
  }

  drawOverlay(cursor: { x: number; y: number } | null): void {
    const dpr = window.devicePixelRatio || 1;
    const { size } = this.geometry();
    this.overlay.width = Math.max(1, Math.round(size * dpr));
    this.overlay.height = Math.max(1, Math.round(size * dpr));
    const ctx = this.overlay.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    if (!this.analysis) return;
    const n = this.analysis.spectrum.size;
    const { scale } = this.geometry();
    // axes
    ctx.strokeStyle = 'rgba(228,225,218,0.12)';
    ctx.lineWidth = 1;
    const c = (n >> 1) * scale + 0.5 * scale;
    ctx.beginPath();
    ctx.moveTo(c, 0);
    ctx.lineTo(c, size);
    ctx.moveTo(0, c);
    ctx.lineTo(size, c);
    ctx.stroke();
    // rings on findings and their conjugates
    for (const f of this.analysis.findings) {
      if (f.kind !== 'periodic') continue;
      const sel = f.id === this.selected;
      ctx.strokeStyle = sel ? 'rgba(240,180,69,1)' : f.enabled ? 'rgba(228,225,218,0.85)' : 'rgba(228,225,218,0.35)';
      ctx.lineWidth = sel ? 2 : 1;
      const orders = [{ u: f.details.u, v: f.details.v }, ...f.details.harmonics];
      orders.forEach((o, k) => {
        for (const s of [1, -1]) {
          const p = this.freqToScreen(s * o.u, s * o.v);
          ctx.beginPath();
          ctx.arc(p.x, p.y, Math.max(6, 3 * f.details.widthBins * scale) + (k ? 0 : 2), 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    }
    if (cursor) {
      ctx.strokeStyle = 'rgba(240,180,69,0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cursor.x, 0);
      ctx.lineTo(cursor.x, size);
      ctx.moveTo(0, cursor.y);
      ctx.lineTo(size, cursor.y);
      ctx.stroke();
    }
  }

  private local(e: MouseEvent): { x: number; y: number } {
    const r = this.overlay.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  private hover(e: PointerEvent): void {
    if (!this.analysis) return;
    const p = this.local(e);
    this.drawOverlay(p);
    const { u, v } = this.screenToFreq(p.x, p.y);
    const n = this.analysis.spectrum.size;
    const c = n >> 1;
    const bx = Math.min(n - 1, Math.max(0, Math.round(u * n) + c));
    const by = Math.min(n - 1, Math.max(0, Math.round(v * n) + c));
    const i = by * n + bx;
    const sp = this.analysis.spectrum;
    const r = Math.hypot(u, v);
    const period = r > 0 ? 1 / r : Infinity;
    const angle = (Math.atan2(-v, u) * 180) / Math.PI;
    const dB = 10 * (sp.logPower[i] - sp.background[i]);
    const lpi = this.dpi ? ` · ${(r * this.dpi).toFixed(1)} lpi` : '';
    const amp = 2 * Math.sqrt(Math.pow(10, sp.logPower[i])) * 255;
    const how = dB >= 12 ? 'stands out' : dB >= 6 ? 'slightly raised' : 'nothing special here';
    this.cb.onHover(
      `Here: repeats every <b>${Number.isFinite(period) ? period.toFixed(2) + ' px' : '∞'}</b> at <b>${angle.toFixed(1)}°</b>${lpi}<br><b>${amp.toFixed(2)} levels</b> strong, ${dB.toFixed(1)} dB above its surroundings — ${how}<br><span class="tech">u ${u.toFixed(4)} v ${v.toFixed(4)} cycles/px · z ${sp.z[i].toFixed(1)}</span>${
        this.addMode ? '<br>Click to add a pattern here' : ''
      }`,
    );
  }

  private click(e: MouseEvent): void {
    if (!this.analysis) return;
    const p = this.local(e);
    const { u, v } = this.screenToFreq(p.x, p.y);
    if (this.addMode) {
      this.cb.onClick(u, v);
      return;
    }
    // select the nearest finding (either conjugate)
    const n = this.analysis.spectrum.size;
    let best: Finding | null = null;
    let bestD = 8 / n;
    for (const f of this.analysis.findings) {
      if (f.kind !== 'periodic') continue;
      for (const s of [1, -1]) {
        const d = Math.hypot(s * f.details.u - u, s * f.details.v - v);
        if (d < bestD) {
          bestD = d;
          best = f;
        }
      }
    }
    if (best) this.cb.onSelect(best.id);
  }
}
