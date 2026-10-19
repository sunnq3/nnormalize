/**
 * Application shell: wires the engine client to the viewer, the spectrum
 * panel, the findings list, the detail panel and the export panel.
 */

import type { Analysis, AnalysisOptions, Finding, PeriodicMethod } from '../analysis/types';
import { EngineClient } from '../engine/client';
import { defaultBudget, type Budget, type LayerSetting, type LoadedInfo, type RoundTripReport, type SynthRequest, type ViewMode } from '../engine/protocol';
import { defaultExportOptions, type JpegExportOptions } from '../io/encode';
import { stageSorted } from '../engine/engine';
import { SpectrumPanel } from './spectrum';
import { Viewer } from './viewer';
import { el, fmt } from './dom';

interface LayerState {
  built: boolean;
  building: boolean;
  rms?: number;
}

export class App {
  private client = new EngineClient();
  private info: LoadedInfo | null = null;
  private analysis: Analysis | null = null;
  private layers = new Map<string, LayerState>();
  private masks = new Map<string, { w: number; h: number; data: Uint8Array }>();
  private master = 1;
  private diffGain = 8;
  private selected: string | null = null;
  private exportOptions: JpegExportOptions = { ...defaultExportOptions };
  private budget: Budget = { ...defaultBudget };
  private analysisOptions: Partial<AnalysisOptions> = {};
  private report: RoundTripReport | null = null;
  private jpegUrl: string | null = null;
  private busy = false;
  private renderQueued = false;
  private rendering = false;

  // DOM
  private root: HTMLElement;
  private status!: HTMLElement;
  private progressBar!: HTMLElement;
  private exportBtn!: HTMLButtonElement;
  private viewer!: Viewer;
  private spectrum!: SpectrumPanel;
  private readout!: HTMLElement;
  private findingsEl!: HTMLElement;
  private detailEl!: HTMLElement;
  private exportEl!: HTMLElement;
  private sourceEl!: HTMLElement;
  private modeButtons = new Map<ViewMode, HTMLButtonElement>();
  private zoomButtons = new Map<string, HTMLButtonElement>();
  private splitBtn!: HTMLButtonElement;
  private masterRange!: HTMLInputElement;
  private masterVal!: HTMLElement;
  private diffRange!: HTMLLabelElement;
  private hud!: HTMLElement;
  private fileInput!: HTMLInputElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.build();
  }

  // --- DOM -------------------------------------------------------------------
  private build(): void {
    const r = this.root;
    r.innerHTML = '';

    // top bar
    this.fileInput = el('input', { type: 'file', accept: 'image/*,.heic,.heif,.tif,.tiff', style: 'display:none' }) as HTMLInputElement;
    this.fileInput.addEventListener('change', () => {
      const f = this.fileInput.files?.[0];
      if (f) void this.open(f);
      this.fileInput.value = '';
    });
    this.status = el('div', { class: 'status' }, 'Open a photo to begin.');
    this.progressBar = el('div');
    this.exportBtn = el('button', { class: 'primary', disabled: '' }, 'Export JPEG') as HTMLButtonElement;
    this.exportBtn.addEventListener('click', () => void this.doExport());
    const top = el(
      'div',
      { class: 'top' },
      el('span', { class: 'brand' }, 'nnormalize'),
      el('button', { onclick: () => this.fileInput.click() }, 'Open photo'),
      el('button', { onclick: () => void this.loadTest() }, 'Test image'),
      this.fileInput,
      this.status,
      el('div', { class: 'progress' }, this.progressBar),
      this.exportBtn,
    );

    // viewer
    this.hud = el('div');
    this.viewer = new Viewer({
      requestCrop: (x0, y0, w, h) => this.client.render({ region: { kind: 'crop', x0, y0, w, h }, ...this.renderArgs() }),
      requestPreview: () => this.client.render({ region: { kind: 'preview' }, ...this.renderArgs() }),
      maskChanged: () => void this.pushMask(),
      onHud: (t) => (this.hud.textContent = t),
    });
    this.hud = this.viewer.el.querySelector('.hud') as HTMLElement;
    const modeGroup = el('div', { class: 'group' });
    for (const [mode, label] of [
      ['original', 'Original'],
      ['result', 'Result'],
      ['difference', 'Difference'],
      ['mask', 'Mask'],
    ] as [ViewMode, string][]) {
      const b = el('button', { onclick: () => this.setMode(mode) }, label) as HTMLButtonElement;
      this.modeButtons.set(mode, b);
      modeGroup.appendChild(b);
    }
    const zoomGroup = el('div', { class: 'group' });
    for (const [z, label] of [
      ['fit', 'Fit'],
      ['1', '100 %'],
      ['2', '200 %'],
    ]) {
      const b = el('button', { onclick: () => this.setZoom(z) }, label) as HTMLButtonElement;
      this.zoomButtons.set(z, b);
      zoomGroup.appendChild(b);
    }
    this.splitBtn = el('button', { onclick: () => this.toggleSplit() }, 'Split') as HTMLButtonElement;
    this.masterRange = el('input', { type: 'range', min: '0', max: '150', value: '100' }) as HTMLInputElement;
    this.masterVal = el('span', { class: 'val' }, '100 %');
    this.masterRange.addEventListener('input', () => {
      this.master = Number(this.masterRange.value) / 100;
      this.masterVal.textContent = `${this.masterRange.value} %`;
      this.queueRender();
    });
    const diffInput = el('input', { type: 'range', min: '1', max: '40', value: '8' }) as HTMLInputElement;
    diffInput.addEventListener('input', () => {
      this.diffGain = Number(diffInput.value);
      this.queueRender();
    });
    this.diffRange = el('label', { style: 'display:none' }, 'Gain ', diffInput) as HTMLLabelElement;
    const toolbar = el(
      'div',
      { class: 'toolbar' },
      modeGroup,
      this.splitBtn,
      this.diffRange,
      el('span', { class: 'spacer' }),
      el('label', {}, 'All fixes ', this.masterRange, this.masterVal),
      zoomGroup,
    );
    const view = el('div', { class: 'view' }, toolbar, this.viewer.el);

    // drag and drop
    const stage = this.viewer.el;
    stage.addEventListener('dragover', (e) => {
      e.preventDefault();
      stage.classList.add('dragover');
    });
    stage.addEventListener('dragleave', () => stage.classList.remove('dragover'));
    stage.addEventListener('drop', (e) => {
      e.preventDefault();
      stage.classList.remove('dragover');
      const f = e.dataTransfer?.files?.[0];
      if (f) void this.open(f);
    });

    // side
    this.sourceEl = el('div', { class: 'source' });
    this.spectrum = new SpectrumPanel({
      onHover: (t) => (this.readout.innerHTML = t || 'Hover the spectrum for exact frequency and level; click a ring to select it.'),
      onClick: (u, v) => void this.addPeak(u, v),
      onSelect: (id) => this.select(id),
    });
    this.readout = el('div', { class: 'readout' }, 'Hover the spectrum for exact frequency and level; click a ring to select it.');
    const addBtn = el(
      'button',
      {
        class: 'quiet',
        onclick: () => {
          this.spectrum.addMode = !this.spectrum.addMode;
          addBtn.classList.toggle('on', this.spectrum.addMode);
        },
      },
      'Add a peak by clicking',
    );
    this.findingsEl = el('div', { class: 'findings' });
    this.detailEl = el('div', { class: 'detail' });
    this.exportEl = el('div', { class: 'export' });
    const side = el(
      'div',
      { class: 'side' },
      el('section', {}, el('h2', {}, 'Source'), this.sourceEl),
      el('section', {}, el('h2', {}, 'Spectrum', el('span', { class: 'sub' }, 'averaged over tiles')), this.spectrum.el, this.readout, el('div', { class: 'tools', style: 'margin-top:6px' }, addBtn)),
      el('section', {}, el('h2', {}, 'Findings'), this.findingsEl),
      el('section', {}, el('h2', {}, 'Selected fix'), this.detailEl),
      el('section', {}, el('h2', {}, 'Export'), this.exportEl),
    );

    r.append(top, view, side);
    this.setMode('result');
    this.setZoom('fit');
    this.renderSource();
    this.renderFindings();
    this.renderDetail();
    this.renderExport();
  }

  // --- helpers --------------------------------------------------------------------
  private setStatus(text: string, fraction: number | null = null): void {
    this.status.textContent = text;
    this.progressBar.style.width = fraction === null ? '0' : `${Math.round(fraction * 100)}%`;
  }

  private layerSettings(): LayerSetting[] {
    if (!this.analysis) return [];
    return this.analysis.findings.filter((f) => f.enabled && this.layers.get(f.id)?.built).map((f) => ({ id: f.id, strength: f.strength }));
  }

  private renderArgs(): { mode: ViewMode; layers: LayerSetting[]; master: number; diffGain: number; maskId: string | null } {
    return { mode: this.viewer.mode, layers: this.layerSettings(), master: this.master, diffGain: this.diffGain, maskId: this.selected };
  }

  private queueRender(): void {
    this.renderQueued = true;
    if (this.rendering) return;
    void this.flushRender();
  }

  private async flushRender(): Promise<void> {
    while (this.renderQueued) {
      this.renderQueued = false;
      this.rendering = true;
      try {
        await this.viewer.refresh();
      } finally {
        this.rendering = false;
      }
    }
  }

  private setMode(mode: ViewMode): void {
    this.viewer.mode = mode;
    this.viewer.maskMode = mode === 'mask';
    for (const [m, b] of this.modeButtons) b.classList.toggle('on', m === mode);
    this.diffRange.style.display = mode === 'difference' ? '' : 'none';
    if (mode === 'mask') {
      this.ensureLocalMask();
      this.viewer.draw();
    } else this.queueRender();
  }

  private setZoom(z: string): void {
    for (const [k, b] of this.zoomButtons) b.classList.toggle('on', k === z);
    this.viewer.setZoom(z === 'fit' ? 'fit' : (Number(z) as 1 | 2));
  }

  private toggleSplit(): void {
    this.viewer.split = !this.viewer.split;
    this.splitBtn.classList.toggle('on', this.viewer.split);
    this.viewer.draw();
  }

  // --- loading -----------------------------------------------------------------
  async open(file: File): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      this.setStatus(`Decoding ${file.name}…`, null);
      const info = await this.client.load(file);
      await this.afterLoad(info);
    } catch (e) {
      this.setStatus(`Could not open ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
      this.busy = false;
    }
  }

  async loadTest(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    const spec: SynthRequest = {
      w: 1400,
      h: 1000,
      seed: 7,
      sinusoids: [
        { u: 0.2137, v: 0.1312, amplitude: 0.045, phase: 0.7 },
        { u: 0, v: 0.25, amplitude: 0.03, phase: -1.1 },
        { u: 0.3604, v: -0.3604, amplitude: 0.02, phase: 2.0 },
      ],
      noiseSigma: 0.012,
      blocks: true,
      hotPixels: 12,
    };
    try {
      this.setStatus('Building the test image…');
      const info = await this.client.synth(spec);
      await this.afterLoad(info);
    } catch (e) {
      this.setStatus(`Test image failed: ${e instanceof Error ? e.message : String(e)}`);
      this.busy = false;
    }
  }

  private async afterLoad(info: LoadedInfo): Promise<void> {
    this.info = info;
    this.analysis = null;
    this.layers.clear();
    this.masks.clear();
    this.selected = null;
    this.report = null;
    if (this.jpegUrl) URL.revokeObjectURL(this.jpegUrl);
    this.jpegUrl = null;
    this.exportBtn.disabled = true;
    this.viewer.setImage(info.w, info.h, info.preview.scale);
    this.renderSource();
    this.spectrum.setAnalysis(null, null);
    this.renderFindings();
    this.renderDetail();
    this.renderExport();
    const orig = await this.client.render({ region: { kind: 'preview' }, mode: 'original', layers: [], master: 1, diffGain: 1, maskId: null });
    await this.viewer.setOriginalPreview(orig);
    await this.viewer.refresh();
    await this.runAnalysis();
    this.busy = false;
  }

  private async runAnalysis(): Promise<void> {
    if (!this.info) return;
    const t0 = performance.now();
    this.setStatus('Analysing…', 0);
    const a = await this.client.analyze(this.analysisOptions, (stage, fr) => this.setStatus(`Analysing: ${stage}`, fr));
    this.analysis = a;
    this.spectrum.setAnalysis(a, this.info.meta.dpi ?? null);
    this.renderFindings();
    this.renderSource();
    this.renderExport();
    const first = a.findings.find((f) => f.enabled);
    this.select(first ? first.id : null);
    this.setStatus(`Analysed in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${a.findings.filter((f) => f.enabled).length} fix${a.findings.filter((f) => f.enabled).length === 1 ? '' : 'es'} suggested`);
    await this.buildPending();
    this.exportBtn.disabled = false;
  }

  /** Build layers for every enabled finding that doesn't have one yet, in stage order. */
  private async buildPending(): Promise<void> {
    if (!this.analysis) return;
    for (const f of stageSorted(this.analysis.findings)) {
      if (!f.enabled) continue;
      const st = this.layers.get(f.id);
      if (st?.built || st?.building) continue;
      await this.buildLayer(f);
    }
  }

  private async buildLayer(f: Finding): Promise<void> {
    this.layers.set(f.id, { built: false, building: true });
    this.renderFindings();
    this.setStatus(`Computing ${f.label.toLowerCase()}…`, 0);
    try {
      const stats = await this.client.buildLayer(f, (_s, fr) => this.setStatus(`Computing ${f.label.toLowerCase()}…`, fr));
      this.layers.set(f.id, { built: true, building: false, rms: stats.rms });
      this.setStatus(`${f.label}: ${fmt(stats.rms, 2)} levels rms changed, ${stats.ms} ms`);
    } catch (e) {
      this.layers.set(f.id, { built: false, building: false });
      this.setStatus(`${f.label} failed: ${e instanceof Error ? e.message : String(e)}`);
    }
    this.renderFindings();
    this.renderDetail();
    this.queueRender();
  }

  // --- findings ------------------------------------------------------------------
  private select(id: string | null): void {
    this.selected = id;
    this.spectrum.setSelected(id);
    this.renderFindings();
    this.renderDetail();
    if (this.viewer.mode === 'mask') {
      this.ensureLocalMask();
      this.viewer.draw();
    }
  }

  private async toggleFinding(f: Finding, on: boolean): Promise<void> {
    f.enabled = on;
    this.renderFindings();
    if (on && !this.layers.get(f.id)?.built) {
      // a later stage's layer depends on earlier ones: rebuild everything after it
      await this.rebuildFrom(f);
    } else {
      this.queueRender();
    }
  }

  /** Rebuild this finding's layer and every built layer of a later stage. */
  private async rebuildFrom(f: Finding): Promise<void> {
    if (!this.analysis) return;
    const order = stageSorted(this.analysis.findings);
    const idx = order.findIndex((x) => x.id === f.id);
    for (let i = idx; i < order.length; i++) {
      const g = order[i];
      if (g.id !== f.id && !this.layers.get(g.id)?.built) continue;
      if (!g.enabled) continue;
      this.layers.delete(g.id);
      await this.buildLayer(g);
    }
  }

  private renderFindings(): void {
    const box = this.findingsEl;
    box.innerHTML = '';
    if (!this.analysis) {
      box.appendChild(el('div', { class: 'empty' }, this.info ? 'Analysing…' : 'Nothing analysed yet.'));
      return;
    }
    if (!this.analysis.findings.length) {
      box.appendChild(el('div', { class: 'empty' }, 'Nothing found above the thresholds.'));
      return;
    }
    for (const f of this.analysis.findings) {
      const st = this.layers.get(f.id);
      const cb = el('input', { type: 'checkbox' }) as HTMLInputElement;
      cb.checked = f.enabled;
      cb.addEventListener('click', (e) => e.stopPropagation());
      cb.addEventListener('change', () => void this.toggleFinding(f, cb.checked));
      const row = el(
        'div',
        {
          class: `finding${f.id === this.selected ? ' selected' : ''}${f.enabled ? '' : ' off'}${st?.building ? ' building' : ''}`,
          onclick: () => this.select(f.id),
        },
        cb,
        el('div', { class: 'title' }, f.label),
        el('div', { class: 'sev', title: `severity ${fmt(f.severity * 100, 0)} %` }, el('div', { style: `width:${Math.round(f.severity * 100)}%` })),
        el('div', { class: 'meta' }, f.description),
        f.note ? el('div', { class: 'note' }, f.note) : '',
      );
      box.appendChild(row);
    }
  }

  private renderDetail(): void {
    const box = this.detailEl;
    box.innerHTML = '';
    const f = this.analysis?.findings.find((x) => x.id === this.selected);
    if (!f) {
      box.appendChild(el('div', { class: 'empty' }, 'Select a finding to adjust its fix.'));
      return;
    }
    const st = this.layers.get(f.id);
    // strength
    const range = el('input', { type: 'range', min: '0', max: '150', value: String(Math.round(f.strength * 100)) }) as HTMLInputElement;
    const val = el('span', { class: 'val' }, `${Math.round(f.strength * 100)} %`);
    range.addEventListener('input', () => {
      f.strength = Number(range.value) / 100;
      val.textContent = `${range.value} %`;
      this.queueRender();
    });
    box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Strength'), range, val));

    // facts
    const facts = el('div', { class: 'facts' });
    const fact = (k: string, v: string): void => {
      facts.appendChild(el('span', {}, k));
      facts.appendChild(el('b', {}, v));
    };
    if (f.kind === 'periodic') {
      const d = f.details;
      fact('frequency', `${fmt(d.u, 4)}, ${fmt(d.v, 4)} cyc/px`);
      fact('period', `${fmt(d.period, 3)} px`);
      fact('angle', `${fmt(d.angleDeg, 2)}°`);
      if (d.lpi) fact('screen ruling', `${fmt(d.lpi, 1)} lpi`);
      fact('amplitude', `${fmt(d.amplitude * 255, 2)} levels`);
      fact('above background', `${fmt(d.excessDb, 1)} dB (z ${fmt(d.z, 1)})`);
      fact('behaviour', d.stationary ? 'same everywhere' : 'varies across frame');
      fact('phase coherence', fmt(d.coherence, 2));
      fact('present in', `${fmt(f.extent * 100, 0)} % of tiles`);
      if (d.harmonics.length) fact('harmonics', d.harmonics.map((h) => `${h.order}×`).join(' '));
    } else if (f.kind === 'jpeg-grid') {
      fact('grid offset', `${f.details.offsetX}, ${f.details.offsetY}`);
      fact('boundary excess', `${fmt(f.details.strengthX * 100, 0)} % / ${fmt(f.details.strengthY * 100, 0)} %`);
      if (f.details.sourceQuality) fact('source quality', `≈ ${fmt(f.details.sourceQuality, 0)}`);
      if (f.details.sourceSubsampling) fact('source chroma', f.details.sourceSubsampling);
    } else if (f.kind === 'noise') {
      const d = f.details;
      fact('baseline', `${fmt(d.globalSigma * 255, 2)} levels`);
      fact('from spectrum', `${fmt(d.spectralSigma * 255, 2)} levels`);
      fact('noisier blocks', `${fmt(d.noisierFraction * 100, 0)} %`);
      fact('smoother blocks', `${fmt(d.smootherFraction * 100, 0)} %`);
      fact('noise shape', `${d.highBandDb >= 0 ? '+' : ''}${fmt(d.highBandDb, 1)} dB high band`);
    } else if (f.kind === 'hotpixels') {
      fact('count', String(f.details.count));
      const s = f.details.sample.slice(0, 3).map((p) => `(${p.x}, ${p.y})`).join(' ');
      if (s) fact('first', s);
    }
    if (st?.rms !== undefined) fact('change', `${fmt(st.rms, 2)} levels rms`);
    box.appendChild(facts);

    // method
    if (f.kind === 'periodic') {
      const sel = el('select') as HTMLSelectElement;
      for (const [v, label] of [
        ['parametric', 'Subtract the fitted pattern (exact, same everywhere)'],
        ['parametric-tiled', 'Subtract the fitted pattern, tile by tile'],
        ['flatten', 'Flatten the spectrum at this frequency'],
        ['notch', 'Notch this frequency out'],
      ] as [PeriodicMethod, string][]) {
        const o = el('option', { value: v }, label) as HTMLOptionElement;
        if (v === f.method) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => {
        f.method = sel.value as PeriodicMethod;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row wide' }, el('span', {}, 'Method'), sel));
    }

    // mask tools
    const tools = el('div', { class: 'tools' });
    const maskBtn = el('button', { class: this.viewer.mode === 'mask' ? 'on' : '', onclick: () => this.setMode(this.viewer.mode === 'mask' ? 'result' : 'mask') }, 'Paint where it applies');
    tools.appendChild(maskBtn);
    const eraseBtn = el(
      'button',
      {
        class: this.viewer.brush.erase ? 'on' : '',
        onclick: () => {
          this.viewer.brush.erase = !this.viewer.brush.erase;
          eraseBtn.classList.toggle('on', this.viewer.brush.erase);
        },
      },
      'Erase',
    );
    tools.appendChild(eraseBtn);
    tools.appendChild(el('button', { onclick: () => void this.maskFill(255) }, 'Everywhere'));
    tools.appendChild(el('button', { onclick: () => void this.maskFill(0) }, 'Nowhere'));
    tools.appendChild(el('button', { onclick: () => void this.maskInvert() }, 'Invert'));
    if (f.kind === 'periodic') tools.appendChild(el('button', { onclick: () => void this.maskFromAnalysis() }, 'Where it was found'));
    box.appendChild(tools);
    const size = el('input', { type: 'range', min: '4', max: '600', value: String(this.viewer.brush.size) }) as HTMLInputElement;
    size.addEventListener('input', () => (this.viewer.brush.size = Number(size.value)));
    const hard = el('input', { type: 'range', min: '0', max: '100', value: String(Math.round(this.viewer.brush.hardness * 100)) }) as HTMLInputElement;
    hard.addEventListener('input', () => (this.viewer.brush.hardness = Number(hard.value) / 100));
    box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Brush size'), size, el('span', { class: 'val' }, 'px')));
    box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Brush edge'), hard, el('span', { class: 'val' }, 'soft→hard')));
    box.appendChild(el('div', { style: 'color:var(--ink-faint)' }, 'In mask view: drag to paint, shift-wheel to resize the brush, alt-drag to pan. Amber = fix applies.'));
  }

  // --- masks -------------------------------------------------------------------------
  private ensureLocalMask(): void {
    if (!this.info || !this.selected) {
      this.viewer.mask = null;
      return;
    }
    let m = this.masks.get(this.selected);
    if (!m) {
      const { w, h } = this.info.maskSize;
      m = { w, h, data: new Uint8Array(w * h).fill(255) };
      this.masks.set(this.selected, m);
    }
    this.viewer.mask = m;
  }

  private async pushMask(): Promise<void> {
    if (!this.selected) return;
    const m = this.masks.get(this.selected);
    if (!m) return;
    await this.client.setMask(this.selected, m.w, m.h, m.data);
  }

  private async maskFill(v: number): Promise<void> {
    this.ensureLocalMask();
    const m = this.viewer.mask;
    if (!m) return;
    m.data.fill(v);
    this.viewer.draw();
    await this.pushMask();
    this.queueRender();
  }

  private async maskInvert(): Promise<void> {
    this.ensureLocalMask();
    const m = this.viewer.mask;
    if (!m) return;
    for (let i = 0; i < m.data.length; i++) m.data[i] = 255 - m.data[i];
    this.viewer.draw();
    await this.pushMask();
    this.queueRender();
  }

  private async maskFromAnalysis(): Promise<void> {
    if (!this.selected) return;
    const m = await this.client.presenceMask(this.selected);
    this.masks.set(this.selected, m);
    this.viewer.mask = m;
    this.viewer.draw();
    this.queueRender();
  }

  private async addPeak(u: number, v: number): Promise<void> {
    if (!this.analysis || this.busy) return;
    this.spectrum.addMode = false;
    this.setStatus('Measuring the pattern at that frequency…');
    const f = await this.client.addPeak(u, v);
    this.analysis.findings.push(f);
    this.spectrum.setAnalysis(this.analysis, this.info?.meta.dpi ?? null);
    this.renderFindings();
    this.select(f.id);
    await this.rebuildFrom(f);
  }

  // --- source panel ----------------------------------------------------------------
  private renderSource(): void {
    const box = this.sourceEl;
    box.innerHTML = '';
    if (!this.info) {
      box.appendChild(el('span', { class: 'empty' }, 'No photo loaded.'));
      return;
    }
    const m = this.info.meta;
    const fact = (k: string, v: string): void => {
      box.appendChild(el('span', {}, k));
      box.appendChild(el('b', {}, v));
    };
    fact('file', m.fileName);
    fact('size', `${this.info.w} × ${this.info.h} px`);
    fact('decoded with', m.decoder === 'mozjpeg' ? 'mozjpeg (exact)' : m.decoder === 'png' ? `own PNG decoder, ${m.bitDepth}-bit` : 'browser');
    if (m.jpeg) {
      fact('source quality', m.jpeg.quality ? `≈ ${fmt(m.jpeg.quality, 0)}` : 'unknown');
      fact('chroma', `${m.jpeg.subsampling}${m.jpeg.progressive ? ', progressive' : ''}`);
    }
    if (m.exifSummary.make || m.exifSummary.model) fact('camera', `${m.exifSummary.make ?? ''} ${m.exifSummary.model ?? ''}`.trim());
    if (m.orientationApplied > 1) fact('orientation', `${m.orientationApplied} applied to pixels`);
    if (m.dpi) fact('resolution', `${fmt(m.dpi, 0)} dpi`);
    fact('colour profile', m.icc ? `${(m.icc.length / 1024).toFixed(1)} kB, kept` : 'none');
    if (this.analysis) fact('tiles', `${this.analysis.tile.size} px, ${this.analysis.spectrum.tiles} averaged`);
  }

  // --- export -----------------------------------------------------------------------
  private renderExport(): void {
    const box = this.exportEl;
    box.innerHTML = '';
    const o = this.exportOptions;
    const q = el('input', { type: 'range', min: '60', max: '100', value: String(o.quality) }) as HTMLInputElement;
    const qv = el('span', { class: 'val' }, String(o.quality));
    q.addEventListener('input', () => {
      o.quality = Number(q.value);
      qv.textContent = q.value;
    });
    box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Quality'), q, qv));
    const chroma = el('select') as HTMLSelectElement;
    for (const v of ['4:4:4', '4:2:0']) {
      const opt = el('option', { value: v }, v === '4:4:4' ? '4:4:4 full colour detail' : '4:2:0 halved colour detail') as HTMLOptionElement;
      if (v === o.chroma) opt.selected = true;
      chroma.appendChild(opt);
    }
    chroma.addEventListener('change', () => (o.chroma = chroma.value as JpegExportOptions['chroma']));
    box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Chroma'), chroma, el('span')));
    const prog = el('input', { type: 'checkbox' }) as HTMLInputElement;
    prog.checked = o.progressive;
    prog.addEventListener('change', () => (o.progressive = prog.checked));
    const trellis = el('input', { type: 'checkbox' }) as HTMLInputElement;
    trellis.checked = o.trellis;
    trellis.addEventListener('change', () => (o.trellis = trellis.checked));
    box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Encoding'), el('label', {}, prog, ' progressive'), el('label', {}, trellis, ' trellis')));

    const b = this.budget;
    const peak = el('input', { type: 'number', min: '3', max: '40', step: '1', value: String(b.maxPeakDb) }) as HTMLInputElement;
    peak.addEventListener('change', () => (b.maxPeakDb = Number(peak.value)));
    const psnr = el('input', { type: 'number', min: '30', max: '60', step: '1', value: String(b.minEncodePsnr) }) as HTMLInputElement;
    psnr.addEventListener('change', () => (b.minEncodePsnr = Number(psnr.value)));
    const blk = el('input', { type: 'number', min: '1', max: '50', step: '1', value: String(Math.round(b.maxBlockStrength * 100)) }) as HTMLInputElement;
    blk.addEventListener('change', () => (b.maxBlockStrength = Number(blk.value) / 100));
    const lv = el('input', { type: 'number', min: '0.1', max: '20', step: '0.1', value: String(b.minPeakLevels) }) as HTMLInputElement;
    lv.addEventListener('change', () => (b.minPeakLevels = Number(lv.value)));
    box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Pass if'), el('span', {}, 'no peak over ', peak, ' dB and ', lv, ' levels'), el('span')));
    box.appendChild(el('div', { class: 'row' }, el('span', {}), el('span', {}, 'block grid under ', blk, ' %'), el('span')));
    box.appendChild(el('div', { class: 'row' }, el('span', {}), el('span', {}, 'encode error better than ', psnr, ' dB'), el('span')));

    const actions = el('div', { class: 'actions' });
    actions.appendChild(el('button', { class: 'primary', onclick: () => void this.doExport(), disabled: this.analysis ? undefined : '' }, 'Export and verify'));
    actions.appendChild(el('button', { onclick: () => this.downloadRecipe(), disabled: this.analysis ? undefined : '' }, 'Save recipe'));
    box.appendChild(actions);

    if (this.report) box.appendChild(this.renderReport(this.report));
    if (this.analysis) {
      const nc = el('div', { class: 'notchecked' }, 'Not checked by this tool:');
      const ul = el('ul');
      for (const s of this.analysis.notChecked) ul.appendChild(el('li', {}, s));
      nc.appendChild(ul);
      box.appendChild(nc);
    }
  }

  private renderReport(r: RoundTripReport): HTMLElement {
    const box = el('div', { class: `report ${r.pass ? 'pass' : 'fail'}` });
    box.appendChild(el('div', { class: 'verdict' }, r.pass ? 'Passed on the exported file' : 'Did not pass on the exported file'));
    if (r.reasons.length) {
      const ul = el('ul');
      for (const s of r.reasons) ul.appendChild(el('li', {}, s));
      box.appendChild(ul);
    }
    const facts = el('div', { class: 'facts' });
    const fact = (k: string, v: string): void => {
      facts.appendChild(el('span', {}, k));
      facts.appendChild(el('b', {}, v));
    };
    fact('file', `${(r.bytes / 1024).toFixed(0)} kB, q${r.quality}, ${r.chroma}`);
    fact('encode error', `${fmt(r.encodeRmsLevels, 2)} levels rms (${fmt(r.encodePsnr, 1)} dB)`);
    fact('periodic left', r.residualPeriodic ? r.residual.map((f) => f.kind === 'periodic' ? `${fmt(f.details.period, 2)} px @ ${fmt(f.details.excessDb, 0)} dB` : '').join(', ') : 'none above budget');
    fact('block grid', r.blockGrid.detected ? `${fmt(r.blockGrid.strength * 100, 0)} %` : 'not detected');
    fact('grain kept', r.noise.expectedLevels > 0 ? `${fmt((100 * r.noise.globalSigmaLevels) / r.noise.expectedLevels, 0)} % (${fmt(r.noise.globalSigmaLevels, 2)} of ${fmt(r.noise.expectedLevels, 2)} levels)` : 'n/a');
    box.appendChild(facts);
    const actions = el('div', { class: 'actions' });
    if (this.jpegUrl) {
      const name = (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '-normalized.jpg';
      actions.appendChild(el('a', { href: this.jpegUrl, download: name }, `Download ${name}`));
    }
    if (!r.pass && r.quality < 100 && r.reasons.some((s) => s.includes('encode error') || s.includes('block grid')))
      actions.appendChild(
        el(
          'button',
          {
            onclick: () => {
              this.exportOptions.quality = Math.min(100, this.exportOptions.quality + 2);
              this.renderExport();
              void this.doExport();
            },
          },
          `Raise quality to ${Math.min(100, r.quality + 2)} and retry`,
        ),
      );
    actions.appendChild(el('button', { class: 'quiet', onclick: () => this.downloadReport() }, 'Report as JSON'));
    box.appendChild(actions);
    return box;
  }

  private async doExport(): Promise<void> {
    if (!this.analysis || this.busy) return;
    this.busy = true;
    this.exportBtn.disabled = true;
    try {
      const { jpeg, report } = await this.client.export(this.layerSettings(), this.master, this.exportOptions, this.budget, this.analysisOptions, (stage, fr) =>
        this.setStatus(`Exporting: ${stage}`, fr),
      );
      if (this.jpegUrl) URL.revokeObjectURL(this.jpegUrl);
      this.jpegUrl = URL.createObjectURL(new Blob([jpeg as BlobPart], { type: 'image/jpeg' }));
      this.report = report;
      this.setStatus(report.pass ? `Exported ${(jpeg.length / 1024).toFixed(0)} kB — passed verification` : `Exported ${(jpeg.length / 1024).toFixed(0)} kB — verification found problems`);
      this.renderExport();
      // offer the download right away
      const name = (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '-normalized.jpg';
      const a = document.createElement('a');
      a.href = this.jpegUrl;
      a.download = name;
      a.click();
    } catch (e) {
      this.setStatus(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      this.busy = false;
      this.exportBtn.disabled = false;
    }
  }

  private recipe(): unknown {
    if (!this.analysis) return null;
    return {
      tool: 'nnormalize',
      version: 1,
      source: this.info?.meta.fileName,
      master: this.master,
      analysisOptions: this.analysisOptions,
      exportOptions: this.exportOptions,
      budget: this.budget,
      findings: this.analysis.findings.map((f) => ({
        id: f.id,
        kind: f.kind,
        label: f.label,
        enabled: f.enabled,
        strength: f.strength,
        method: f.method,
        details: f.kind === 'periodic' ? { u: f.details.u, v: f.details.v, period: f.details.period, angleDeg: f.details.angleDeg, amplitude: f.details.amplitude, excessDb: f.details.excessDb } : undefined,
        mask: this.masks.get(f.id) ? { w: this.masks.get(f.id)!.w, h: this.masks.get(f.id)!.h, data: btoa(String.fromCharCode(...this.masks.get(f.id)!.data)) } : undefined,
      })),
    };
  }

  private downloadRecipe(): void {
    const r = this.recipe();
    if (!r) return;
    this.downloadJson(r, (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '.nnormalize.json');
  }

  private downloadReport(): void {
    if (!this.report) return;
    const r = { ...this.report, residual: this.report.residual.map((f) => ({ ...f, details: f.kind === 'periodic' ? { ...f.details, presence: undefined } : f.details })) };
    this.downloadJson({ report: r, recipe: this.recipe(), notChecked: this.analysis?.notChecked }, (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '.report.json');
  }

  private downloadJson(obj: unknown, name: string): void {
    const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
}
