/**
 * Image viewer: fit / 100 % / 200 % zoom with panning, four view modes, split
 * compare, and a mask brush that paints directly on screen.
 */

import type { RenderResult, ViewMode } from '../engine/protocol';

export interface MaskBrush {
  size: number; // diameter in image pixels
  hardness: number; // 0..1
  flow: number; // 0..1
  erase: boolean;
}

export interface ViewerCallbacks {
  /** Ask for a full-resolution crop (x0, y0, w, h in image pixels). */
  requestCrop: (x0: number, y0: number, w: number, h: number) => Promise<RenderResult>;
  /** The same crop of the comparison side (the untouched photo, or the untouched photo as posted). */
  requestCropOriginal: (x0: number, y0: number, w: number, h: number) => Promise<RenderResult>;
  /** Ask for the preview-resolution render of the current mode. */
  requestPreview: () => Promise<RenderResult>;
  /** Mask stroke finished; the local mask has changed. */
  maskChanged: () => void;
  onHud: (text: string) => void;
}

export class Viewer {
  readonly el: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private hud: HTMLDivElement;
  private dropHint: HTMLDivElement;

  imageW = 0;
  imageH = 0;
  previewScale = 1;
  mode: ViewMode = 'result';
  split = false;
  zoomMode: 'fit' | 1 | 2 = 'fit';
  private zoom = 1; // image px -> screen css px
  private panX = 0;
  private panY = 0;

  private preview: ImageBitmap | null = null; // current mode at preview res
  private previewOriginal: ImageBitmap | null = null; // for split compare
  private crop: { bmp: ImageBitmap; x0: number; y0: number } | null = null;
  private cropOrig: { bmp: ImageBitmap; x0: number; y0: number } | null = null;
  private cropRequest = 0;

  // mask painting
  maskMode = false;
  brush: MaskBrush = { size: 160, hardness: 0.4, flow: 1, erase: true };
  mask: { w: number; h: number; data: Uint8Array } | null = null;
  private painting = false;
  private lastPaint: { x: number; y: number } | null = null;
  private dragging: { x: number; y: number; panX: number; panY: number } | null = null;

  constructor(private cb: ViewerCallbacks) {
    this.el = document.createElement('div');
    this.el.className = 'stage';
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
    this.el.appendChild(this.canvas);
    this.dropHint = document.createElement('div');
    this.dropHint.className = 'drop';
    this.dropHint.innerHTML = '<div><strong>Drop a photo here</strong>JPEG, PNG (8 or 16-bit), WebP, AVIF, TIFF, HEIC on Safari — or load the test image.</div>';
    this.el.appendChild(this.dropHint);
    this.hud = document.createElement('div');
    this.hud.className = 'hud';
    this.el.appendChild(this.hud);

    new ResizeObserver(() => this.resize()).observe(this.el);
    this.el.addEventListener('pointerdown', (e) => this.pointerDown(e));
    this.el.addEventListener('pointermove', (e) => this.pointerMove(e));
    this.el.addEventListener('pointerup', (e) => this.pointerUp(e));
    this.el.addEventListener('pointercancel', (e) => this.pointerUp(e));
    this.el.addEventListener('wheel', (e) => this.wheel(e), { passive: false });
    this.el.addEventListener('dblclick', () => this.setZoom(this.zoomMode === 'fit' ? 1 : 'fit'));
  }

  // --- geometry -------------------------------------------------------------
  private resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const r = this.el.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    if (this.zoomMode === 'fit') this.fit();
    this.draw();
    this.scheduleCrop();
  }

  setImage(w: number, h: number, previewScale: number): void {
    this.imageW = w;
    this.imageH = h;
    this.previewScale = previewScale;
    this.preview = null;
    this.previewOriginal = null;
    this.crop = null;
    this.cropOrig = null;
    this.mask = null;
    this.dropHint.style.display = w ? 'none' : '';
    this.fit();
    this.draw();
  }

  private fit(): void {
    const r = this.el.getBoundingClientRect();
    if (!this.imageW) return;
    this.zoom = Math.min(r.width / this.imageW, r.height / this.imageH, 1);
    this.panX = (r.width - this.imageW * this.zoom) / 2;
    this.panY = (r.height - this.imageH * this.zoom) / 2;
  }

  setZoom(mode: 'fit' | 1 | 2): void {
    const r = this.el.getBoundingClientRect();
    const cx = r.width / 2;
    const cy = r.height / 2;
    const ix = (cx - this.panX) / this.zoom;
    const iy = (cy - this.panY) / this.zoom;
    this.zoomMode = mode;
    if (mode === 'fit') this.fit();
    else {
      this.zoom = mode;
      this.panX = cx - ix * this.zoom;
      this.panY = cy - iy * this.zoom;
      this.clampPan();
    }
    this.draw();
    this.scheduleCrop();
  }

  private clampPan(): void {
    const r = this.el.getBoundingClientRect();
    const iw = this.imageW * this.zoom;
    const ih = this.imageH * this.zoom;
    if (iw <= r.width) this.panX = (r.width - iw) / 2;
    else this.panX = Math.min(0, Math.max(r.width - iw, this.panX));
    if (ih <= r.height) this.panY = (r.height - ih) / 2;
    else this.panY = Math.min(0, Math.max(r.height - ih, this.panY));
  }

  private toImage(e: PointerEvent): { x: number; y: number } {
    const r = this.el.getBoundingClientRect();
    return { x: (e.clientX - r.left - this.panX) / this.zoom, y: (e.clientY - r.top - this.panY) / this.zoom };
  }

  // --- data ----------------------------------------------------------------
  async refresh(): Promise<void> {
    if (!this.imageW) return;
    const res = await this.cb.requestPreview();
    const img = new ImageData(res.rgba, res.w, res.h);
    const bmp = await createImageBitmap(img);
    this.preview?.close();
    this.preview = bmp;
    this.draw();
    this.scheduleCrop(true);
  }

  async setOriginalPreview(res: RenderResult): Promise<void> {
    const bmp = await createImageBitmap(new ImageData(res.rgba, res.w, res.h));
    this.previewOriginal?.close();
    this.previewOriginal = bmp;
    this.draw();
  }

  private needsCrop(): boolean {
    return this.zoom > this.previewScale + 1e-6;
  }

  private scheduleCrop(force = false): void {
    if (!this.imageW || !this.needsCrop()) {
      if (this.crop && !this.needsCrop()) {
        this.crop.bmp.close();
        this.crop = null;
        this.cropOrig?.bmp.close();
        this.cropOrig = null;
      }
      return;
    }
    const r = this.el.getBoundingClientRect();
    const x0 = Math.max(0, Math.floor(-this.panX / this.zoom));
    const y0 = Math.max(0, Math.floor(-this.panY / this.zoom));
    const x1 = Math.min(this.imageW, Math.ceil((r.width - this.panX) / this.zoom));
    const y1 = Math.min(this.imageH, Math.ceil((r.height - this.panY) / this.zoom));
    if (!force && this.crop && this.crop.x0 === x0 && this.crop.y0 === y0 && this.crop.bmp.width === x1 - x0 && this.crop.bmp.height === y1 - y0) return;
    const id = ++this.cropRequest;
    const cw = Math.max(1, x1 - x0);
    const chh = Math.max(1, y1 - y0);
    void this.cb.requestCrop(x0, y0, cw, chh).then(async (res) => {
      if (id !== this.cropRequest) return;
      const bmp = await createImageBitmap(new ImageData(res.rgba, res.w, res.h));
      this.crop?.bmp.close();
      this.crop = { bmp, x0, y0 };
      this.draw();
    });
    if (this.split) {
      void this.cb.requestCropOriginal(x0, y0, cw, chh).then(async (res) => {
        if (id !== this.cropRequest) return;
        const bmp = await createImageBitmap(new ImageData(res.rgba, res.w, res.h));
        this.cropOrig?.bmp.close();
        this.cropOrig = { bmp, x0, y0 };
        this.draw();
      });
    }
  }

  // --- drawing ---------------------------------------------------------------
  draw(): void {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.imageW) return;
    ctx.imageSmoothingEnabled = this.zoom < 1;
    ctx.imageSmoothingQuality = 'high';
    const r = this.el.getBoundingClientRect();
    const drawBitmap = (bmp: ImageBitmap | null, cropAlt?: { bmp: ImageBitmap; x0: number; y0: number } | null): void => {
      if (cropAlt && this.needsCrop()) {
        ctx.drawImage(cropAlt.bmp, this.panX + cropAlt.x0 * this.zoom, this.panY + cropAlt.y0 * this.zoom, cropAlt.bmp.width * this.zoom, cropAlt.bmp.height * this.zoom);
        return;
      }
      if (!bmp) return;
      ctx.drawImage(bmp, this.panX, this.panY, this.imageW * this.zoom, this.imageH * this.zoom);
    };
    if (this.split && this.previewOriginal && this.mode !== 'difference' && !this.maskMode) {
      const half = r.width / 2;
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, half, r.height);
      ctx.clip();
      drawBitmap(this.previewOriginal, this.cropOrig);
      ctx.restore();
      ctx.save();
      ctx.beginPath();
      ctx.rect(half, 0, r.width - half, r.height);
      ctx.clip();
      drawBitmap(this.preview, this.crop);
      ctx.restore();
      ctx.strokeStyle = 'rgba(240,180,69,0.9)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(half + 0.5, 0);
      ctx.lineTo(half + 0.5, r.height);
      ctx.stroke();
      ctx.fillStyle = 'rgba(43,44,47,0.8)';
      ctx.fillRect(half - 58, 8, 50, 18);
      ctx.fillRect(half + 8, 8, 50, 18);
      ctx.fillStyle = '#e4e1da';
      ctx.font = '12px IBM Plex Sans, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('before', half - 33, 21);
      ctx.fillText('after', half + 33, 21);
    } else {
      drawBitmap(this.preview, this.crop);
    }
    if (this.maskMode) this.drawMaskOverlay();
  }

  /** Tint where the selected fix is held back (mask below full), so an untouched mask shows nothing. */
  private drawMaskOverlay(): void {
    if (!this.mask) return;
    const ctx = this.ctx;
    const { w, h, data } = this.mask;
    const img = new ImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const held = 255 - data[i];
      img.data[i * 4] = 240;
      img.data[i * 4 + 1] = 180;
      img.data[i * 4 + 2] = 69;
      img.data[i * 4 + 3] = Math.round(held * 0.6);
    }
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    tmp.getContext('2d')!.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, this.panX, this.panY, this.imageW * this.zoom, this.imageH * this.zoom);
    // brush cursor
    if (this.lastPaint) {
      ctx.strokeStyle = this.brush.erase ? 'rgba(224,130,118,0.9)' : 'rgba(240,180,69,0.9)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(this.panX + this.lastPaint.x * this.zoom, this.panY + this.lastPaint.y * this.zoom, (this.brush.size / 2) * this.zoom, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // --- interaction -------------------------------------------------------------
  private pointerDown(e: PointerEvent): void {
    if (!this.imageW) return;
    this.el.setPointerCapture(e.pointerId);
    if (this.maskMode && this.mask && e.button === 0 && !e.altKey) {
      this.painting = true;
      const p = this.toImage(e);
      this.lastPaint = p;
      this.paintAt(p.x, p.y);
      this.draw();
      return;
    }
    this.dragging = { x: e.clientX, y: e.clientY, panX: this.panX, panY: this.panY };
  }

  private pointerMove(e: PointerEvent): void {
    if (!this.imageW) return;
    const p = this.toImage(e);
    if (p.x >= 0 && p.y >= 0 && p.x < this.imageW && p.y < this.imageH) this.cb.onHud(`${Math.floor(p.x)}, ${Math.floor(p.y)}  ·  ${Math.round(this.zoom * 100)} %`);
    if (this.painting) {
      const last = this.lastPaint!;
      const d = Math.hypot(p.x - last.x, p.y - last.y);
      const step = Math.max(1, this.brush.size / 6);
      const n = Math.max(1, Math.ceil(d / step));
      for (let i = 1; i <= n; i++) this.paintAt(last.x + ((p.x - last.x) * i) / n, last.y + ((p.y - last.y) * i) / n);
      this.lastPaint = p;
      this.draw();
      return;
    }
    if (this.maskMode) {
      this.lastPaint = p;
      this.draw();
    }
    if (this.dragging) {
      this.panX = this.dragging.panX + (e.clientX - this.dragging.x);
      this.panY = this.dragging.panY + (e.clientY - this.dragging.y);
      this.clampPan();
      this.draw();
    }
  }

  private pointerUp(e: PointerEvent): void {
    if (this.painting) {
      this.painting = false;
      this.cb.maskChanged();
    }
    if (this.dragging) {
      this.dragging = null;
      this.scheduleCrop();
    }
    try {
      this.el.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
  }

  private wheel(e: WheelEvent): void {
    if (!this.imageW) return;
    e.preventDefault();
    if (this.maskMode && e.shiftKey) {
      this.brush.size = Math.max(4, Math.min(2000, this.brush.size * (e.deltaY > 0 ? 0.9 : 1.1)));
      this.draw();
      return;
    }
    if (this.zoomMode === 'fit') return; // scroll zooms only in 100/200 % modes
    this.panX -= e.deltaX;
    this.panY -= e.deltaY;
    this.clampPan();
    this.draw();
    this.scheduleCrop();
  }

  private paintAt(ix: number, iy: number): void {
    const m = this.mask!;
    const sx = m.w / this.imageW;
    const sy = m.h / this.imageH;
    const cx = ix * sx;
    const cy = iy * sy;
    const rad = (this.brush.size / 2) * sx;
    const x0 = Math.max(0, Math.floor(cx - rad));
    const x1 = Math.min(m.w - 1, Math.ceil(cx + rad));
    const y0 = Math.max(0, Math.floor(cy - rad));
    const y1 = Math.min(m.h - 1, Math.ceil(cy + rad));
    const hard = this.brush.hardness;
    const target = this.brush.erase ? 0 : 255;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / Math.max(rad, 0.5);
        if (d > 1) continue;
        // smooth falloff from `hard` to the edge
        const t = d <= hard ? 1 : 1 - (d - hard) / Math.max(1e-6, 1 - hard);
        const a = Math.max(0, Math.min(1, t)) * this.brush.flow * 0.5;
        const i = y * m.w + x;
        m.data[i] = Math.round(m.data[i] + (target - m.data[i]) * a);
      }
    }
  }

  getZoom(): number {
    return this.zoom;
  }

  refreshCrop(): void {
    this.scheduleCrop(true);
  }
}
