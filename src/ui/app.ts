/**
 * Application shell: wires the engine client to the viewer, the spectrum
 * panel, the findings list, the detail panel and the export panel.
 */

import type { Analysis, AnalysisOptions, CleanupLevel, Finding, PeriodicMethod } from '../analysis/types';
import type { FixSurvival } from '../analysis/survival';
import { EngineClient } from '../engine/client';
import { defaultBudget, type Budget, type BakeSetting, type LayerSetting, type LiveReport, type LoadedInfo, type RoundTripReport, type SynthRequest, type ViewMode } from '../engine/protocol';
import type { JpegExportOptions } from '../io/encode';
import { PROFILES, resolveProfile, type ProfileId } from '../engine/profiles';
import { stageSorted } from '../engine/engine';
import { SpectrumPanel } from './spectrum';
import { Viewer } from './viewer';
import { el, fmt } from './dom';
import { PRESETS } from './presets';
import { LEVEL_WORDS, plainDescription, plainFix, plainLabel } from './words';

interface RecipeFile {
  tool: string;
  version: number;
  name?: string;
  master?: number;
  profile?: ProfileId;
  analysisOptions?: Partial<AnalysisOptions>;
  exportOptions?: Partial<JpegExportOptions>;
  budget?: Partial<Budget>;
  findings: {
    id: string;
    kind: Finding['kind'];
    enabled: boolean;
    strength: number;
    method?: string;
    details?: { u: number; v: number };
    mask?: { w: number; h: number; data: string };
  }[];
}

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
  private exportOptions: JpegExportOptions = { ...PROFILES.custom.encode };
  private profileId: ProfileId = 'reddit';
  private checked = false; // a posted view exists
  private budget: Budget = { ...defaultBudget };
  private analysisOptions: Partial<AnalysisOptions> = { level: 'standard' };
  private report: RoundTripReport | null = null;
  private jpegUrl: string | null = null;
  private busy = false;
  private presetId = 'default';
  private level: CleanupLevel = 'standard';
  private recipe: RecipeFile | null = null;
  private batchQueue: File[] = [];
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
  private settingsEl!: HTMLElement;
  private scrollToReport = false;
  private bake: BakeSetting = 'clean';
  private liveEl!: HTMLElement;
  private liveTimer: ReturnType<typeof setTimeout> | null = null;
  private liveRunning = false;
  private liveDirty = false;
  private liveReport: LiveReport | null = null;
  private liveError = '';
  private liveCollapsed = false;
  private sourceEl!: HTMLElement;
  private analysisEl!: HTMLElement;
  private recipeInput!: HTMLInputElement;
  private batchInput!: HTMLInputElement;
  private modeButtons = new Map<ViewMode, HTMLButtonElement>();
  private zoomButtons = new Map<string, HTMLButtonElement>();
  private splitBtn!: HTMLButtonElement;
  private brushBtn!: HTMLButtonElement;
  private levelSelect!: HTMLSelectElement;
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
    this.exportBtn = el('button', { class: 'primary', disabled: '' }, 'Save JPEG') as HTMLButtonElement;
    this.exportBtn.addEventListener('click', () => void this.doExport(true));
    this.levelSelect = el('select', { title: 'How far the clean-up goes' }) as HTMLSelectElement;
    for (const k of ['gentle', 'standard', 'strong', 'maximum'] as CleanupLevel[]) {
      const o = el('option', { value: k }, LEVEL_WORDS[k].label) as HTMLOptionElement;
      if (k === this.level) o.selected = true;
      this.levelSelect.appendChild(o);
    }
    this.levelSelect.addEventListener('change', () => void this.setLevel(this.levelSelect.value as CleanupLevel));
    const top = el(
      'div',
      { class: 'top' },
      el('span', { class: 'brand' }, 'nnormalize'),
      el('button', { onclick: () => this.fileInput.click() }, 'Open photo'),
      this.fileInput,
      this.status,
      el('div', { class: 'progress' }, this.progressBar),
      this.exportBtn,
    );

    // viewer
    this.hud = el('div');
    this.viewer = new Viewer({
      requestCrop: (x0, y0, w, h) => this.client.render({ region: { kind: 'crop', x0, y0, w, h }, ...this.renderArgs() }),
      requestCropOriginal: (x0, y0, w, h) =>
        this.client.render({ region: { kind: 'crop', x0, y0, w, h }, mode: this.viewer.mode === 'posted' ? 'posted-source' : 'original', layers: [], master: 1, diffGain: 1, maskId: null }),
      requestPreview: () => this.client.render({ region: { kind: 'preview' }, ...this.renderArgs() }),
      maskChanged: () => void this.pushMask().then(() => this.queueRender()),
      onHud: (t) => (this.hud.textContent = t),
    });
    this.hud = this.viewer.el.querySelector('.hud') as HTMLElement;
    const modeGroup = el('div', { class: 'group' });
    for (const [mode, label, title] of [
      ['original', 'Before', 'The photo as opened'],
      ['result', 'After', 'With the ticked fixes'],
      ['difference', 'What changed', 'The difference, amplified'],
      ['posted', 'As posted', 'What a downloader gets after the destination re-saves it; Split shows the untouched photo given the same treatment'],
    ] as [ViewMode, string, string][]) {
      const b = el('button', { onclick: () => this.setMode(mode), title }, label) as HTMLButtonElement;
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
    this.splitBtn = el('button', { onclick: () => this.toggleSplit(), title: 'Before on the left, after on the right' }, 'Split') as HTMLButtonElement;
    this.brushBtn = el('button', { onclick: () => this.toggleBrush(), title: 'Paint where the selected fix should be held back' }, 'Brush') as HTMLButtonElement;
    this.masterRange = el('input', { type: 'range', min: '0', max: '200', value: '100' }) as HTMLInputElement;
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
    this.diffRange = el('label', { style: 'display:none', title: 'How much the change is amplified in this view' }, 'Amplify ', diffInput) as HTMLLabelElement;
    const toolbar = el(
      'div',
      { class: 'toolbar' },
      modeGroup,
      this.splitBtn,
      this.brushBtn,
      this.diffRange,
      el('span', { class: 'spacer' }),
      el('label', { title: 'Scales every fix at once; 100 % is exactly what was measured' }, 'All fixes ', this.masterRange, this.masterVal),
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
    this.sourceEl = el('div', { class: 'sourcebox' });
    this.analysisEl = el('div', { class: 'detail' });
    this.recipeInput = el('input', { type: 'file', accept: '.json,application/json', style: 'display:none' }) as HTMLInputElement;
    this.recipeInput.addEventListener('change', () => {
      const f = this.recipeInput.files?.[0];
      if (f) void this.loadRecipe(f);
      this.recipeInput.value = '';
    });
    this.batchInput = el('input', { type: 'file', multiple: '', accept: 'image/*,.heic,.heif,.tif,.tiff', style: 'display:none' }) as HTMLInputElement;
    this.batchInput.addEventListener('change', () => {
      const files = Array.from(this.batchInput.files ?? []);
      if (files.length) void this.runBatch(files);
      this.batchInput.value = '';
    });
    const readoutIdle = 'Move over the picture for the exact numbers at that spot; click a ring to select that pattern.';
    this.spectrum = new SpectrumPanel({
      onHover: (t) => (this.readout.innerHTML = t || readoutIdle),
      onClick: (u, v) => void this.addPeak(u, v),
      onSelect: (id) => this.select(id),
    });
    this.readout = el('div', { class: 'readout' }, readoutIdle);
    const addBtn = el(
      'button',
      {
        class: 'quiet',
        onclick: () => {
          this.spectrum.addMode = !this.spectrum.addMode;
          addBtn.classList.toggle('on', this.spectrum.addMode);
        },
      },
      'Add a pattern by clicking',
    );
    this.findingsEl = el('div', { class: 'findings' });
    this.detailEl = el('div', { class: 'detail' });
    this.exportEl = el('div', { class: 'export' });
    this.settingsEl = el('div', { class: 'settings' });
    this.liveEl = el('div', { class: 'livecheck' });
    this.exportEl.addEventListener('change', () => this.scheduleLive());
    const step = (n: number, title: string, hint: string, ...body: (HTMLElement | string)[]): HTMLElement =>
      el('section', { class: 'step' }, el('h2', {}, el('span', { class: 'num' }, String(n)), title), hint ? el('div', { class: 'intro' }, hint) : '', ...body);
    const more = el('details', { class: 'more' }, el('summary', {}, 'More: spectrum, analysis settings, recipes and batch'));
    more.appendChild(
      el(
        'div',
        { class: 'moreinner' },
        el('h3', {}, 'What repeats', el('span', { class: 'sub' }, 'the spectrum')),
        el('div', { class: 'intro' }, 'Bright dots away from the centre are stripes or grids in the photo; the further out, the finer. Rings mark the ones found.'),
        this.spectrum.el,
        this.readout,
        el('div', { class: 'tools', style: 'margin:6px 0 14px' }, addBtn),
        el('h3', {}, 'Analysis settings'),
        this.analysisEl,
        this.recipeInput,
        this.batchInput,
      ),
    );
    const side = el(
      'div',
      { class: 'side' },
      step(1, 'Photo', '', this.sourceEl),
      step(2, 'How far, and where it is going', '', this.settingsEl),
      step(3, 'Problems found', 'Ticked problems get fixed. Click one to adjust its fix or hold it back in places.', this.findingsEl),
      step(4, 'Save', '', this.exportEl),
      el('section', {}, more),
    );
    const handle = el('div', { class: 'side-handle', title: 'Drag to resize the panel' });
    let drag: { x: number; w: number } | null = null;
    handle.addEventListener('pointerdown', (e) => {
      drag = { x: e.clientX, w: side.getBoundingClientRect().width };
      handle.classList.add('dragging');
      handle.setPointerCapture(e.pointerId);
    });
    handle.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const w = Math.max(360, Math.min(window.innerWidth * 0.7, drag.w + (drag.x - e.clientX)));
      r.style.setProperty('--side-w', `${Math.round(w)}px`);
    });
    handle.addEventListener('pointerup', () => {
      drag = null;
      handle.classList.remove('dragging');
      try {
        localStorage.setItem('nnormalize.sideWidth', r.style.getPropertyValue('--side-w'));
      } catch {
        /* storage may be unavailable */
      }
    });
    try {
      const saved = localStorage.getItem('nnormalize.sideWidth');
      if (saved) r.style.setProperty('--side-w', saved);
    } catch {
      /* storage may be unavailable */
    }

    r.append(top, view, this.liveEl, side, handle);
    this.setMode('result');
    this.setZoom('fit');
    this.renderSource();
    this.renderSettings();
    this.renderAnalysisPanel();
    this.renderFindings();
    this.renderDetail();
    this.renderExport();
    this.renderLive();
  }

  // --- live check ------------------------------------------------------------------
  /** Re-run the live check a moment after the last change. */
  private scheduleLive(): void {
    if (!this.analysis) return;
    this.liveDirty = true;
    if (this.liveTimer) clearTimeout(this.liveTimer);
    this.liveTimer = setTimeout(() => void this.runLive(), 700);
    this.renderLive();
  }

  private async runLive(): Promise<void> {
    this.liveTimer = null;
    if (!this.analysis || this.liveRunning) return;
    // wait for fixes that are still being computed; their completion schedules another run
    for (const st of this.layers.values()) if (st.building) return;
    for (const f of this.analysis.findings) if (f.enabled && !this.layers.has(f.id)) return; // not started yet
    this.liveDirty = false;
    this.liveRunning = true;
    this.liveError = '';
    this.renderLive();
    try {
      this.liveReport = await this.client.liveCheck(this.layerSettings(), this.analysis.findings, this.master, this.profileId, this.exportOptions, this.bake);
    } catch (e) {
      this.liveError = e instanceof Error ? e.message : String(e);
    } finally {
      this.liveRunning = false;
    }
    this.renderLive();
    if (this.liveDirty && !this.liveTimer) this.liveTimer = setTimeout(() => void this.runLive(), 200);
  }

  private renderLive(): void {
    const box = this.liveEl;
    if (!box) return;
    box.innerHTML = '';
    if (!this.analysis) {
      box.style.display = 'none';
      return;
    }
    box.style.display = '';
    const r = this.liveReport;
    const state = this.liveRunning ? 'checking…' : this.liveDirty ? 'waiting for you to stop…' : this.liveError ? 'check failed' : r ? 'up to date' : '';
    const fixes = r?.survival.fixes ?? [];
    const lost = fixes.filter((f) => f.verdict === 'lost').length;
    const partly = fixes.filter((f) => f.verdict === 'partly').length;
    const verdict = !r ? 'Checking what a downloader gets…' : !fixes.length ? 'Nothing ticked to check.' : lost ? `${lost} fix${lost > 1 ? 'es' : ''} lost after the re-save` : partly ? `${partly} fix${partly > 1 ? 'es' : ''} only partly kept` : 'Every fix holds after the re-save';
    const head = el(
      'div',
      {
        class: 'livehead',
        onclick: () => {
          this.liveCollapsed = !this.liveCollapsed;
          this.renderLive();
        },
        title: 'Click to hide or show the table',
      },
      el('span', { class: `dot ${!r ? '' : lost ? 'lost' : partly ? 'partly' : 'survives'}` }),
      el('b', {}, verdict),
      el('span', { class: `state${this.liveRunning || this.liveDirty ? ' busy' : ''}` }, state),
      el('span', { class: 'chev' }, this.liveCollapsed ? 'show ▴' : 'hide ▾'),
    );
    box.appendChild(head);
    if (this.liveCollapsed) return;
    if (this.liveError) box.appendChild(el('div', { class: 'note' }, this.liveError));
    if (!r) return;
    box.appendChild(el('div', { class: 'livesub' }, `After ${r.checkLabel}: what someone who downloads it gets, next to the untouched photo given the same re-save.`));
    if (fixes.length) box.appendChild(this.survivalTable(fixes));
    const notes = fixes.filter((f) => f.verdict !== 'survives');
    if (notes.length) {
      const ul = el('ul');
      for (const fx of notes) {
        const f = this.analysis.findings.find((x) => x.id === fx.id);
        ul.appendChild(el('li', {}, `${f ? plainLabel(f) : fx.label}: ${fx.note}`));
      }
      box.appendChild(ul);
    }
    const sn = r.sensor;
    const pct = (v: number): string => `${Math.round(v * 100)} %`;
    box.appendChild(
      el(
        'div',
        { class: 'livesub' },
        sn.baked
          ? `Sensor bake (${sn.layout}, ${sn.method === 'bilinear' ? 'bilinear' : 'Malvar–He–Cutler'} fill-in, ${sn.noiseLevels > 0 ? `photon noise ${fmt(sn.noiseLevels, 1)} levels at mid grey` : 'nothing added'}): grid pattern ${pct(sn.inFile)} in the file written for this destination, ${pct(sn.posted)} after its re-save (the original photo: ${pct(sn.original)}).${sn.posted < 0.02 && sn.original >= 0.02 ? ' This re-save erases any camera’s grid, the original’s too; it stays in a Master copy.' : ''}`
          : `Sensor bake is off. Grid pattern in your file ${pct(sn.inFile)}, after the re-save ${pct(sn.posted)} (the original photo: ${pct(sn.original)}).`,
      ),
    );
    const where = r.region.full ? 'Checked on the whole photo' : `Checked on the centre ${r.region.w} × ${r.region.h} px of the photo`;
    box.appendChild(el('div', { class: 'livesub' }, `${where}, in ${(r.ms / 1000).toFixed(1)} s. File about ${Math.round(r.bytesEstimate / 1024)} kB. Saving checks the whole photo.`));
  }

  /** The fix | untouched | yours | kept table, shared by the live check and the save report. */
  private survivalTable(fixes: FixSurvival[]): HTMLElement {
    const tbl = el('table', { class: 'survival' });
    tbl.appendChild(el('tr', {}, el('th', {}, 'fix'), el('th', {}, 'untouched'), el('th', {}, 'yours'), el('th', {}, 'kept')));
    for (const fx of fixes) {
      const f = this.analysis?.findings.find((x) => x.id === fx.id);
      const name = f ? plainLabel(f) : fx.label;
      const unit = fx.unit === 'pixels' ? '' : fx.unit === 'dB' ? ' dB' : fx.unit === '%' ? ' %' : ' lv';
      const digits = fx.unit === 'pixels' || fx.unit === '%' ? 0 : fx.unit === 'dB' ? 1 : 2;
      tbl.appendChild(
        el(
          'tr',
          { class: fx.verdict, title: fx.note, onclick: () => this.select(fx.id) },
          el('td', {}, name),
          el('td', {}, `${fmt(fx.sourceOnDest, digits)}${unit}`),
          el('td', {}, `${fmt(fx.resultOnDest, digits)}${unit}`),
          el('td', {}, fx.verdict === 'survives' ? 'yes' : fx.verdict === 'partly' ? `${Math.round(fx.retained * 100)} %` : 'no'),
        ),
      );
    }
    return tbl;
  }

  /** Step 2: the two choices that drive everything else. */
  private renderSettings(): void {
    const box = this.settingsEl;
    box.innerHTML = '';
    const levelHint = el('div', { class: 'hint' }, LEVEL_WORDS[this.level].hint);
    this.levelSelect.addEventListener('change', () => {
      levelHint.textContent = LEVEL_WORDS[this.levelSelect.value as CleanupLevel].hint;
    });
    box.appendChild(el('div', { class: 'row wide', title: 'How far the clean-up goes' }, el('span', {}, 'Clean-up'), this.levelSelect));
    box.appendChild(levelHint);
    const sel = el('select') as HTMLSelectElement;
    for (const p of Object.values(PROFILES)) {
      const o = el('option', { value: p.id }, p.label) as HTMLOptionElement;
      if (p.id === this.profileId) o.selected = true;
      sel.appendChild(o);
    }
    const profHint = el('div', { class: 'hint' }, PROFILES[this.profileId].hint);
    sel.addEventListener('change', () => {
      this.profileId = sel.value as ProfileId;
      this.checked = false;
      this.report = null;
      profHint.textContent = PROFILES[this.profileId].hint;
      this.renderExport();
      this.scheduleLive();
      if (this.profileId === 'reddit' && (this.level === 'standard' || this.level === 'gentle')) this.setStatus('Tip: Strong clean-up is the better match for Reddit; at Standard the grain looks unchanged after their compression.');
    });
    box.appendChild(el('div', { class: 'row wide', title: 'Decides how the file is written and what the check simulates' }, el('span', {}, 'Going to'), sel));
    box.appendChild(profHint);
  }

  private renderAnalysisPanel(): void {
    const box = this.analysisEl;
    box.innerHTML = '';
    box.appendChild(el('div', { class: 'intro' }, `Clean-up is ${LEVEL_WORDS[this.level].label.toLowerCase()}: ${LEVEL_WORDS[this.level].hint}`));
    const sel = el('select') as HTMLSelectElement;
    for (const p of PRESETS) {
      const o = el('option', { value: p.id }, p.label) as HTMLOptionElement;
      if (p.id === this.presetId) o.selected = true;
      sel.appendChild(o);
    }
    const hint = el('div', { style: 'color:var(--ink-faint);margin:2px 0 6px' }, PRESETS.find((p) => p.id === this.presetId)?.hint ?? '');
    sel.addEventListener('change', () => {
      this.presetId = sel.value;
      this.analysisOptions = { ...(PRESETS.find((p) => p.id === this.presetId)?.options ?? {}), level: this.level };
      this.renderAnalysisPanel();
    });
    box.appendChild(el('div', { class: 'row wide' }, el('span', {}, 'Photo type'), sel));
    box.appendChild(hint);
    const tools = el('div', { class: 'tools' });
    tools.appendChild(el('button', { onclick: () => void this.reanalyse(), disabled: this.info ? undefined : '' }, 'Look again'));
    tools.appendChild(el('button', { onclick: () => this.recipeInput.click() }, this.recipe ? `Settings file: ${this.recipe.name}` : 'Load settings file'));
    if (this.recipe)
      tools.appendChild(
        el(
          'button',
          {
            class: 'quiet',
            onclick: () => {
              this.recipe = null;
              this.renderAnalysisPanel();
            },
          },
          'Forget it',
        ),
      );
    tools.appendChild(el('button', { onclick: () => this.batchInput.click() }, 'Do many photos'));
    box.appendChild(tools);
    box.appendChild(el('div', { style: 'color:var(--ink-faint);margin-top:6px' }, '“Do many photos” runs each one through the same settings, saves it and checks it, and downloads a summary.'));

    const adv = el('details', { class: 'advanced' }, el('summary', {}, 'Thresholds'));
    const o = this.analysisOptions;
    const z = el('input', { type: 'number', min: '3', max: '20', step: '0.5', value: String(o.zThreshold ?? '') , placeholder: 'auto' }) as HTMLInputElement;
    z.addEventListener('change', () => (this.analysisOptions.zThreshold = z.value ? Number(z.value) : undefined));
    const amp = el('input', { type: 'number', min: '0.05', max: '20', step: '0.05', value: String(o.minAmplitudeLevels ?? ''), placeholder: 'auto' }) as HTMLInputElement;
    amp.addEventListener('change', () => (this.analysisOptions.minAmplitudeLevels = amp.value ? Number(amp.value) : undefined));
    const nz = el('input', { type: 'number', min: '1', max: '12', step: '0.5', value: String(o.noiseThresholdDb ?? ''), placeholder: 'auto' }) as HTMLInputElement;
    nz.addEventListener('change', () => (this.analysisOptions.noiseThresholdDb = nz.value ? Number(nz.value) : undefined));
    const lin = el('input', { type: 'checkbox' }) as HTMLInputElement;
    lin.checked = !!o.linearLight;
    lin.addEventListener('change', () => (this.analysisOptions.linearLight = lin.checked));
    adv.appendChild(el('div', { class: 'row' }, el('span', {}, 'Pattern must stand out by'), el('span', {}, z, ' × the usual spread'), el('span')));
    adv.appendChild(el('div', { class: 'row' }, el('span', {}, 'Count it as visible from'), el('span', {}, amp, ' levels'), el('span')));
    adv.appendChild(el('div', { class: 'row' }, el('span', {}, 'Grain counts as uneven from'), el('span', {}, '± ', nz, ' dB'), el('span')));
    adv.appendChild(el('div', { class: 'row' }, el('span', {}, 'Linear light'), el('label', {}, lin, ' measure after undoing the gamma curve'), el('span')));
    adv.appendChild(el('div', { style: 'color:var(--ink-faint)' }, 'Blank means the clean-up level decides. Press “Look again” after changing these.'));
    box.appendChild(adv);
  }

  private async setLevel(level: CleanupLevel): Promise<void> {
    this.level = level;
    this.analysisOptions = { ...this.analysisOptions, level };
    this.setStatus(`Clean-up set to ${LEVEL_WORDS[level].label.toLowerCase()}: ${LEVEL_WORDS[level].hint}`);
    this.renderAnalysisPanel();
    if (this.info && !this.busy) await this.reanalyse();
  }

  private async reanalyse(): Promise<void> {
    if (!this.info || this.busy) return;
    this.busy = true;
    this.layers.clear();
    this.masks.clear();
    this.selected = null;
    await this.runAnalysis();
    this.busy = false;
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
    if (this.viewer.mode !== 'posted' && this.viewer.mode !== 'posted-source') this.checked = false;
    this.scheduleLive();
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
    if (mode === 'posted' && !this.checked) {
      if (!this.analysis || this.busy) return;
      void this.doExport(false).then(() => this.setMode('posted'));
      return;
    }
    this.viewer.mode = mode;
    for (const [m, b] of this.modeButtons) b.classList.toggle('on', m === mode);
    this.diffRange.style.display = mode === 'difference' ? '' : 'none';
    if (mode === 'posted') {
      // split compares against the untouched photo after the same re-save
      void this.client.render({ region: { kind: 'preview' }, mode: 'posted-source', layers: [], master: 1, diffGain: 1, maskId: null }).then((r) => this.viewer.setOriginalPreview(r));
    } else if (this.info) {
      void this.client.render({ region: { kind: 'preview' }, mode: 'original', layers: [], master: 1, diffGain: 1, maskId: null }).then((r) => this.viewer.setOriginalPreview(r));
    }
    this.queueRender();
  }

  private toggleBrush(on?: boolean): void {
    const next = on ?? !this.viewer.maskMode;
    if (next && !this.selected) {
      this.setStatus('Pick a problem in the list first, then paint where its fix should be held back.');
      return;
    }
    this.viewer.maskMode = next;
    this.brushBtn.classList.toggle('on', next);
    if (next) {
      this.ensureLocalMask();
      if (this.viewer.mode !== 'result') this.setMode('result');
    }
    this.viewer.draw();
    this.renderDetail();
  }

  private setZoom(z: string): void {
    for (const [k, b] of this.zoomButtons) b.classList.toggle('on', k === z);
    this.viewer.setZoom(z === 'fit' ? 'fit' : (Number(z) as 1 | 2));
  }

  private toggleSplit(): void {
    this.viewer.split = !this.viewer.split;
    this.splitBtn.classList.toggle('on', this.viewer.split);
    this.viewer.draw();
    this.viewer.refreshCrop();
  }

  // --- loading -----------------------------------------------------------------
  async open(file: File): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      this.setStatus(`Opening ${file.name}…`, null);
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
    this.liveReport = null;
    this.liveDirty = false;
    this.layers.clear();
    this.masks.clear();
    this.selected = null;
    this.report = null;
    this.checked = false;
    if (this.jpegUrl) URL.revokeObjectURL(this.jpegUrl);
    this.jpegUrl = null;
    this.exportBtn.disabled = true;
    if (this.viewer.mode === 'posted') this.setMode('result');
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
    this.setStatus('Looking for problems…', 0);
    const stageWords: Record<string, string> = { spectrum: 'what repeats', peaks: 'measuring patterns', noise: 'grain', hotpixels: 'dead pixels', colour: 'colour noise', halos: 'edge halos' };
    const a = await this.client.analyze({ ...this.analysisOptions, level: this.level }, (stage, fr) => this.setStatus(`Looking for problems: ${stageWords[stage] ?? stage}`, fr));
    this.analysis = a;
    requestAnimationFrame(() => this.findingsEl.parentElement?.scrollIntoView({ block: 'start', behavior: 'smooth' }));
    this.spectrum.setAnalysis(a, this.info.meta.dpi ?? null);
    this.renderFindings();
    this.renderSource();
    this.renderExport();
    if (this.recipe) await this.applyRecipe(this.recipe, a);
    const first = a.findings.find((f) => f.enabled);
    this.select(first ? first.id : null);
    const on = a.findings.filter((f) => f.enabled);
    this.setStatus(
      on.length
        ? `Looked in ${((performance.now() - t0) / 1000).toFixed(1)} s: ${on.map((f) => plainLabel(f).toLowerCase()).join(', ')}. Computing the fixes…`
        : `Looked in ${((performance.now() - t0) / 1000).toFixed(1)} s: nothing to fix at the ${LEVEL_WORDS[this.level].label.toLowerCase()} level. Raise Clean-up to also smooth grain and block edges.`,
    );
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
    this.setStatus(`Computing: ${plainLabel(f).toLowerCase()}…`, 0);
    try {
      const stats = await this.client.buildLayer(f, (_s, fr) => this.setStatus(`Computing: ${plainLabel(f).toLowerCase()}…`, fr));
      this.layers.set(f.id, { built: true, building: false, rms: stats.rms });
      this.setStatus(`${plainLabel(f)}: ready, changes the picture by ${fmt(stats.rms, 1)} levels on average`);
    } catch (e) {
      this.layers.set(f.id, { built: false, building: false });
      this.setStatus(`${plainLabel(f)} failed: ${e instanceof Error ? e.message : String(e)}`);
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
    if (this.viewer.maskMode) {
      if (!id) this.toggleBrush(false);
      else {
        this.ensureLocalMask();
        this.viewer.draw();
      }
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
      box.appendChild(el('div', { class: 'empty' }, this.info ? 'Looking…' : 'Open a photo and the problems it finds will be listed here.'));
      return;
    }
    if (!this.analysis.findings.length) {
      box.appendChild(el('div', { class: 'empty' }, 'Nothing stands out at this clean-up level. Raise it to also smooth grain and block edges.'));
      return;
    }
    const faint = (f: Finding): boolean => (f.kind === 'periodic' && !f.enabled && Math.max(f.details.amplitude, f.details.amplitudeP90) * 255 < 1) || (f.kind === 'drift' && !f.enabled && Math.max(f.details.amplitude, f.details.amplitudeP90) < 1);
    const main = this.analysis.findings.filter((f) => !faint(f) || f.id === this.selected);
    const rest = this.analysis.findings.filter((f) => !main.includes(f));
    const renderRow = (f: Finding): HTMLElement => {
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
        el('div', { class: 'title' }, plainLabel(f)),
        el('div', { class: 'sev', title: `how bad: ${fmt(f.severity * 100, 0)} %` }, el('div', { style: `width:${Math.round(f.severity * 100)}%` })),
        el('div', { class: 'meta' }, plainDescription(f)),
        f.note ? el('div', { class: 'note' }, f.note) : '',
      );
      return row;
    };
    for (const f of main) {
      box.appendChild(renderRow(f));
      if (f.id === this.selected) box.appendChild(this.detailEl);
    }
    if (rest.length) {
      const d = el('details', { class: 'advanced' }, el('summary', {}, `${rest.length} faint pattern${rest.length > 1 ? 's' : ''} under 1 level — not visible, left alone`));
      for (const f of rest) d.appendChild(renderRow(f));
      box.appendChild(d);
    }
  }

  private renderDetail(): void {
    const box = this.detailEl;
    box.innerHTML = '';
    const f = this.analysis?.findings.find((x) => x.id === this.selected);
    if (!f) return;
    const st = this.layers.get(f.id);
    box.appendChild(el('div', { class: 'intro' }, plainFix(f)));

    // strength
    const range = el('input', { type: 'range', min: '0', max: '200', value: String(Math.round(f.strength * 100)) }) as HTMLInputElement;
    const val = el('span', { class: 'val' }, `${Math.round(f.strength * 100)} %`);
    range.addEventListener('input', () => {
      f.strength = Number(range.value) / 100;
      val.textContent = `${range.value} %`;
      this.queueRender();
    });
    box.appendChild(el('div', { class: 'row', title: '100 % is exactly what was measured; more over-corrects' }, el('span', {}, 'Strength'), range, val));

    if (f.kind === 'noise') {
      const red = el('input', { type: 'range', min: '0', max: '95', value: String(Math.round(f.details.reduction * 100)) }) as HTMLInputElement;
      const rv = el('span', { class: 'val' }, `${Math.round(f.details.reduction * 100)} %`);
      red.addEventListener('input', () => (rv.textContent = `${red.value} %`));
      red.addEventListener('change', () => {
        f.details.reduction = Number(red.value) / 100;
        f.enabled = true;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row', title: '0 only levels the grain; higher removes it everywhere' }, el('span', {}, 'Remove grain'), red, rv));
    }
    if (f.kind === 'detail' && f.details.mode === 'smooth') {
      const mb = el('input', { type: 'range', min: '2', max: '12', value: String(Math.round(f.details.maxBoostDb)) }) as HTMLInputElement;
      const label = (db: number): string => `${fmt(10 ** (db / 20), 1)}×`;
      const mv = el('span', { class: 'val' }, label(f.details.maxBoostDb));
      mb.addEventListener('input', () => (mv.textContent = label(Number(mb.value))));
      mb.addEventListener('change', () => {
        f.details.maxBoostDb = Number(mb.value);
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row', title: 'The most the fine detail may be raised. Higher brings back more, and more of whatever noise is there' }, el('span', {}, 'Most it may add'), mb, mv));
    }
    if (f.kind === 'edges') {
      const tw = el('input', { type: 'range', min: '12', max: '35', value: String(Math.round(f.details.targetWidthPx * 10)) }) as HTMLInputElement;
      const tv = el('span', { class: 'val' }, `${fmt(f.details.targetWidthPx, 1)} px`);
      tw.addEventListener('input', () => (tv.textContent = `${fmt(Number(tw.value) / 10, 1)} px`));
      tw.addEventListener('change', () => {
        f.details.targetWidthPx = Number(tw.value) / 10;
        f.enabled = true;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row', title: 'How wide a sharp edge should be. A good camera gives about 1.5–2 px. The fix stops short of this when going further would make new outlines' }, el('span', {}, 'Edge to aim for'), tw, tv));
      const kc = el('input', { type: 'range', min: '0', max: '100', value: String(Math.round((f.details.keepCrisp ?? 0.5) * 100)) }) as HTMLInputElement;
      const kv = el('span', { class: 'val' }, `${Math.round((f.details.keepCrisp ?? 0.5) * 100)} %`);
      kc.addEventListener('input', () => (kv.textContent = `${kc.value} %`));
      kc.addEventListener('change', () => {
        f.details.keepCrisp = Number(kc.value) / 100;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row', title: 'How much of the camera’s sharpening to keep at strong edges, where it shows as an outline. Texture always keeps it, since there it reads as detail. 0 removes it all at edges (cleanest outlines)' }, el('span', {}, 'Keep at edges'), kc, kv));
    }
    if (f.kind === 'chroma' || f.kind === 'ringing' || f.kind === 'detail' || f.kind === 'banding' || f.kind === 'edges' || f.kind === 'fringes' || f.kind === 'tone' || f.kind === 'glow' || f.kind === 'texture' || f.kind === 'drift') {
      const am = el('input', { type: 'range', min: '0', max: '100', value: String(Math.round(f.details.amount * 100)) }) as HTMLInputElement;
      const av = el('span', { class: 'val' }, `${Math.round(f.details.amount * 100)} %`);
      am.addEventListener('input', () => (av.textContent = `${am.value} %`));
      am.addEventListener('change', () => {
        f.details.amount = Number(am.value) / 100;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row' }, el('span', {}, f.kind === 'detail' || f.kind === 'edges' || f.kind === 'tone' || f.kind === 'texture' ? 'Amount' : f.kind === 'banding' ? 'Smooth' : 'Remove'), am, av));
    }
    if (f.kind === 'grain') {
      const am = el('input', { type: 'range', min: '5', max: '60', value: String(Math.round(f.details.targetLevels * 10)) }) as HTMLInputElement;
      const av = el('span', { class: 'val' }, `${f.details.targetLevels.toFixed(1)} lv`);
      am.addEventListener('input', () => (av.textContent = `${(Number(am.value) / 10).toFixed(1)} lv`));
      am.addEventListener('change', () => {
        f.details.targetLevels = Number(am.value) / 10;
        f.enabled = true;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row', title: 'How much grain the photo should end up with everywhere. 1.5 is barely there, 3 is a light film look, 5 is clearly grainy' }, el('span', {}, 'Grain'), am, av));
      const col = el('input', { type: 'range', min: '0', max: '100', value: String(Math.round(f.details.colour * 100)) }) as HTMLInputElement;
      const cv = el('span', { class: 'val' }, `${Math.round(f.details.colour * 100)} %`);
      col.addEventListener('input', () => (cv.textContent = `${col.value} %`));
      col.addEventListener('change', () => {
        f.details.colour = Number(col.value) / 100;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row', title: 'Colour part of the grain, as a share of the brightness part. 0 is pure monochrome grain' }, el('span', {}, 'Colour in it'), col, cv));
    }
    if (f.kind === 'jpeg-grid') {
      const step = el('input', { type: 'range', min: '2', max: '40', value: String(Math.round(f.details.maxStep * 100)) }) as HTMLInputElement;
      const sv = el('span', { class: 'val' }, `${Math.round(f.details.maxStep * 255)} lv`);
      step.addEventListener('input', () => (sv.textContent = `${Math.round((Number(step.value) / 100) * 255)} lv`));
      step.addEventListener('change', () => {
        f.details.maxStep = Number(step.value) / 100;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row', title: 'Steps bigger than this are treated as real edges and left alone' }, el('span', {}, 'Blend steps up to'), step, sv));
      const span = el('select') as HTMLSelectElement;
      for (const [v, label] of [
        ['4', 'over 4 pixels (light touch)'],
        ['8', 'over the whole block (removes the grid)'],
      ]) {
        const o = el('option', { value: v }, label) as HTMLOptionElement;
        if (Number(v) === f.details.span) o.selected = true;
        span.appendChild(o);
      }
      span.addEventListener('change', () => {
        f.details.span = Number(span.value) as 4 | 8;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row wide' }, el('span', {}, 'Spread'), span));
    }

    // method
    if (f.kind === 'periodic') {
      const sel = el('select') as HTMLSelectElement;
      for (const [v, label] of [
        ['parametric', 'Subtract it exactly (same everywhere)'],
        ['parametric-tiled', 'Subtract it region by region (if it varies)'],
        ['flatten', 'Smooth it out of the spectrum'],
        ['notch', 'Cut that frequency out completely'],
      ] as [PeriodicMethod, string][]) {
        const o = el('option', { value: v }, label) as HTMLOptionElement;
        if (v === f.method) o.selected = true;
        sel.appendChild(o);
      }
      sel.addEventListener('change', () => {
        f.method = sel.value as PeriodicMethod;
        void this.rebuildFrom(f);
      });
      box.appendChild(el('div', { class: 'row wide' }, el('span', {}, 'How'), sel));
    }

    // where it applies, tucked away until wanted
    const where = el('details', { class: 'advanced', open: this.viewer.maskMode ? '' : undefined }, el('summary', {}, 'Hold it back in places (brush, brightness range)'));
    box.appendChild(where);
    const tools = el('div', { class: 'tools' });
    const brushOn = this.viewer.maskMode;
    tools.appendChild(el('button', { class: brushOn ? 'on' : '', onclick: () => this.toggleBrush() }, brushOn ? 'Brush is on' : 'Brush'));
    const eraseBtn = el(
      'button',
      {
        class: 'quiet',
        onclick: () => {
          this.viewer.brush.erase = !this.viewer.brush.erase;
          eraseBtn.textContent = this.viewer.brush.erase ? 'Painting: hold the fix back' : 'Painting: put the fix back';
        },
      },
      this.viewer.brush.erase ? 'Painting: hold the fix back' : 'Painting: put the fix back',
    );
    tools.appendChild(eraseBtn);
    tools.appendChild(el('button', { onclick: () => void this.maskFill(255) }, 'Everywhere'));
    tools.appendChild(el('button', { onclick: () => void this.maskFill(0) }, 'Nowhere'));
    tools.appendChild(el('button', { onclick: () => void this.maskInvert() }, 'Swap'));
    if (f.kind === 'periodic' || f.kind === 'drift') tools.appendChild(el('button', { onclick: () => void this.maskFromAnalysis() }, 'Only where it was found'));
    where.appendChild(tools);
    const size = el('input', { type: 'range', min: '4', max: '800', value: String(this.viewer.brush.size) }) as HTMLInputElement;
    const sizeVal = el('span', { class: 'val' }, `${this.viewer.brush.size} px`);
    size.addEventListener('input', () => {
      this.viewer.brush.size = Number(size.value);
      sizeVal.textContent = `${size.value} px`;
    });
    const hard = el('input', { type: 'range', min: '0', max: '100', value: String(Math.round(this.viewer.brush.hardness * 100)) }) as HTMLInputElement;
    hard.addEventListener('input', () => (this.viewer.brush.hardness = Number(hard.value) / 100));
    where.appendChild(el('div', { class: 'row' }, el('span', {}, 'Brush size'), size, sizeVal));
    where.appendChild(el('div', { class: 'row' }, el('span', {}, 'Brush edge'), hard, el('span', { class: 'val' }, 'soft → hard')));
    const lo = el('input', { type: 'range', min: '0', max: '100', value: '0' }) as HTMLInputElement;
    const hi = el('input', { type: 'range', min: '0', max: '100', value: '100' }) as HTMLInputElement;
    const lumVal = el('span', { class: 'val' }, 'dark → bright');
    const applyLum = (): void => {
      lumVal.textContent = `${lo.value}–${hi.value} %`;
      void this.maskFromLuminosity(Number(lo.value) / 100, Number(hi.value) / 100);
    };
    lo.addEventListener('change', applyLum);
    hi.addEventListener('change', applyLum);
    where.appendChild(el('div', { class: 'row', title: 'Apply the fix only to this brightness range' }, el('span', {}, 'By brightness'), el('div', { style: 'display:flex;gap:4px' }, lo, hi), lumVal));
    where.appendChild(el('div', { style: 'color:var(--ink-faint)' }, 'With the brush on, drag over the picture. Amber shows where the fix is held back. Shift + wheel resizes the brush; hold Alt to drag the picture instead.'));

    // technical details, collapsed
    const det = el('details', { class: 'advanced' }, el('summary', {}, 'Measurements'));
    const facts = el('div', { class: 'facts' });
    const fact = (k: string, v: string): void => {
      facts.appendChild(el('span', {}, k));
      facts.appendChild(el('b', {}, v));
    };
    if (f.kind === 'periodic') {
      const d = f.details;
      fact('frequency', `${fmt(d.u, 4)}, ${fmt(d.v, 4)} cycles/px`);
      fact('period', `${fmt(d.period, 3)} px`);
      fact('angle', `${fmt(d.angleDeg, 2)}°`);
      if (d.lpi) fact('screen ruling', `${fmt(d.lpi, 1)} lines/inch`);
      fact('strength', d.stationary ? `${fmt(d.amplitude * 255, 2)} levels` : `${fmt(d.amplitude * 255, 2)} typical, ${fmt(d.amplitudeP90 * 255, 2)} at most`);
      fact('stands out by', `${fmt(d.excessDb, 1)} dB (z ${fmt(d.z, 1)})`);
      fact('same everywhere', d.stationary ? 'yes' : 'no');
      fact('phase agreement', fmt(d.coherence, 2));
      fact('found in', `${fmt(f.extent * 100, 0)} % of the tiles`);
      if (d.harmonics.length) fact('harmonics', d.harmonics.map((h) => `${h.order}×`).join(' '));
    } else if (f.kind === 'jpeg-grid') {
      fact('grid offset', `${f.details.offsetX}, ${f.details.offsetY}`);
      fact('edge steps', `${fmt(f.details.stepLevels, 2)} levels, +${fmt(f.details.strengthX * 100, 0)} % / +${fmt(f.details.strengthY * 100, 0)} % over the texture`);
      if (f.details.sourceQuality) fact('source quality', `≈ ${fmt(f.details.sourceQuality, 0)}`);
      if (f.details.sourceSubsampling) fact('source colour detail', f.details.sourceSubsampling);
    } else if (f.kind === 'noise') {
      const d = f.details;
      fact('grain', `${fmt(d.globalSigma * 255, 2)} levels`);
      fact('grain, from spectrum', `${fmt(d.spectralSigma * 255, 2)} levels`);
      fact('grainier regions', `${fmt(d.noisierFraction * 100, 0)} %`);
      fact('smoother regions', `${fmt(d.smootherFraction * 100, 0)} %`);
      fact('fine vs coarse', `${d.highBandDb >= 0 ? '+' : ''}${fmt(d.highBandDb, 1)} dB`);
    } else if (f.kind === 'hotpixels') {
      fact('count', String(f.details.count));
      const sm = f.details.sample.slice(0, 3).map((p) => `(${p.x}, ${p.y})`).join(' ');
      if (sm) fact('first', sm);
    } else if (f.kind === 'chroma') {
      fact('blue–yellow noise', `${fmt(f.details.sigmaCbLevels, 2)} levels`);
      fact('red–green noise', `${fmt(f.details.sigmaCrLevels, 2)} levels`);
      fact('affected', `${fmt(f.details.fraction * 100, 0)} % of blocks`);
    } else if (f.kind === 'detail') {
      const d = f.details;
      fact(d.mode === 'sharp' ? 'above the reference' : 'below the reference', `${fmt(d.mode === 'sharp' ? d.excessDb : d.deficitDb, 1)} dB in the fine band`);
      fact('scene falls off at', `${fmt(-d.line.slope / 10, 2)} (power ∝ 1/f^this)`);
      const at = (fr: number): string => {
        const i = Math.max(0, Math.min(d.f.length - 1, Math.round(fr * d.f.length * 2) - 1));
        return `${fmt(d.measured[i] - d.reference[i], 1)} dB`;
      };
      fact('fine detail vs reference', `${at(0.12)} · ${at(0.2)} · ${at(0.3)} (at 8, 5, 3 px)`);
    } else if (f.kind === 'texture') {
      fact('fine texture now', `${fmt(f.details.share * 100, 0)} % of a natural scene's`);
      fact('clearly short in', `${fmt(f.details.weakFraction * 100, 0)} % of the photo`);
      fact('expected after the fix', `${fmt(f.details.expectedShare * 100, 0)} %`);
    } else if (f.kind === 'drift') {
      const d = f.details;
      fact('spacing', d.periodMax - d.periodMin > 0.15 * d.period ? `${fmt(d.periodMin, 2)} to ${fmt(d.periodMax, 2)} px (drifts)` : `${fmt(d.period, 2)} px`);
      fact('direction', d.angleSpread > 2 ? `${fmt(d.angleDeg, 1)}°, turning by ± ${fmt(d.angleSpread, 1)}°` : `${fmt(d.angleDeg, 1)}°`);
      fact('strength', `${fmt(d.amplitude, 2)} levels typical, ${fmt(d.amplitudeP90, 2)} at its strongest`);
      fact('above the background by', `${fmt(d.excessDb, 1)} dB`);
      fact('tracked through', `${d.nodes.length} of ${Math.max(1, d.gridW - 2) * Math.max(1, d.gridH - 2)} windows (${fmt(d.coverage * 100, 0)} %)`);
    } else if (f.kind === 'glow') {
      fact('glow beside edges', `${fmt(f.details.glowLevels, 1)} levels`);
      fact('edges with a glow', `${f.details.sides} (${fmt(f.details.share * 100, 0)} % of smooth edge sides)`);
      fact('area touched', `${fmt(f.details.coverage * 100, 1)} % of the photo`);
    } else if (f.kind === 'tone') {
      fact('darkest shadows', `${fmt(f.details.shadowLevels, 0)} levels`);
      fact('brightest highlights', `${fmt(f.details.highlightLevels, 0)} levels`);
      fact('local contrast now', `${fmt(f.details.localContrast, 1)} levels`);
      fact('expected after the fix', `×${fmt(1 + (f.details.expectedRatio - 1) * f.details.amount, 2)}`);
    } else if (f.kind === 'fringes') {
      fact('on fine detail', `${fmt(f.details.detailLevels, 2)} levels`);
      fact('in smooth areas', `${fmt(f.details.flatLevels, 2)} levels`);
      fact('fine detail covers', `${fmt(f.details.detailFraction * 100, 0)} % of the photo`);
    } else if (f.kind === 'edges') {
      const d = f.details;
      fact('edges used', `${d.count} (the sharpest fifth)`);
      fact('edge width now', `${fmt(d.widthPx, 2)} px (10–90 %)`);
      fact('outlines', `${fmt(d.overshoot * 100, 1)} % bright, ${fmt(d.undershoot * 100, 1)} % dark`);
      fact('camera blur', `${fmt(d.blurSigma, 2)} px (Gaussian sigma)`);
      fact('camera sharpening', d.amounts.map((a, k) => `${fmt(a, 2)} at ${[1, 2.5, 6][k]} px`).join(', '));
      fact('model fit', `${fmt(d.fitRms * 100, 1)} % rms`);
      fact('reachable without new outlines', `${fmt(d.reachableWidthPx, 2)} px`);
    } else if (f.kind === 'banding') {
      fact('step', `${f.details.stepLevels} level${f.details.stepLevels === 1 ? '' : 's'}`);
      fact('visible steps', `${fmt(f.details.strengthLevels, 2)} levels`);
      fact('area', `${fmt(f.details.fraction * 100, 0)} % of the photo`);
    } else if (f.kind === 'grain') {
      fact('photo had', `${fmt(f.details.baseLevels, 2)} levels`);
      fact('ends up with', `${fmt(f.details.targetLevels, 1)} levels`);
    } else if (f.kind === 'ringing') {
      fact('next to edges', `${fmt(f.details.haloLevels, 2)} levels`);
      fact('elsewhere', `${fmt(f.details.grainLevels, 2)} levels`);
      fact('band covers', `${fmt(f.details.bandFraction * 100, 1)} % of pixels`);
    }
    if (st?.rms !== undefined) fact('changes the picture by', `${fmt(st.rms, 2)} levels on average`);
    det.appendChild(facts);
    box.appendChild(det);
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

  private async maskFromLuminosity(lo: number, hi: number): Promise<void> {
    if (!this.selected) return;
    const m = await this.client.luminosityMask(this.selected, lo, hi, 0.08);
    this.masks.set(this.selected, m);
    this.viewer.mask = m;
    this.viewer.draw();
    this.queueRender();
  }

  private async addPeak(u: number, v: number): Promise<void> {
    if (!this.analysis || this.busy) return;
    this.spectrum.addMode = false;
    this.setStatus('Measuring the pattern at that spot…');
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
    const buttons = el(
      'div',
      { class: 'tools' },
      el('button', { class: this.info ? '' : 'primary', onclick: () => this.fileInput.click() }, this.info ? 'Open another photo' : 'Open a photo'),
      el('button', { class: 'quiet', onclick: () => void this.loadTest(), title: 'A made-up photo with known stripes, a block grid and dead pixels, to see what the tool does' }, 'Try the test image'),
    );
    if (!this.info) {
      box.appendChild(el('div', { class: 'welcome' }, el('p', {}, 'Open a photo, or drop one on the dark area. Nothing leaves your computer.'), el('p', {}, 'Then: choose how far to go and where the photo is going (2), look over what it found and untick anything you want kept (3), and save; the tool checks that every fix survives the place you post it (4).')));
      box.appendChild(buttons);
      return;
    }
    box.appendChild(buttons);
    const m = this.info.meta;
    const grid = el('div', { class: 'source' });
    const det = el('details', { class: 'advanced' }, el('summary', {}, `${m.fileName}, ${this.info.w} × ${this.info.h} px${m.jpeg?.quality ? `, saved at about quality ${fmt(m.jpeg.quality, 0)}` : ''}`), grid);
    box.appendChild(det);
    const fact = (k: string, v: string): void => {
      grid.appendChild(el('span', {}, k));
      grid.appendChild(el('b', {}, v));
    };
    fact('file', m.fileName);
    fact('size', `${this.info.w} × ${this.info.h} px`);
    if (m.jpeg) {
      fact('last saved at', m.jpeg.quality ? `quality ≈ ${fmt(m.jpeg.quality, 0)}` : 'unknown quality');
      fact('colour detail', m.jpeg.subsampling === '4:4:4' ? 'full' : `reduced (${m.jpeg.subsampling})`);
    }
    if (m.exifSummary.make || m.exifSummary.model) fact('camera', `${m.exifSummary.make ?? ''} ${m.exifSummary.model ?? ''}`.trim());
    if (m.orientationApplied > 1) fact('rotation', 'applied from the camera tag');
    if (m.dpi) fact('resolution', `${fmt(m.dpi, 0)} dpi`);
    fact('colour profile', m.icc ? 'kept' : 'none');
    fact('read with', m.decoder === 'mozjpeg' ? 'mozjpeg (exact)' : m.decoder === 'png' ? `own PNG reader, ${m.bitDepth}-bit` : m.decoder === 'libheif' ? 'libheif' : 'the browser');
  }

  // --- export -----------------------------------------------------------------------
  private renderExport(): void {
    const box = this.exportEl;
    box.innerHTML = '';
    const prof = resolveProfile(this.profileId, this.exportOptions);
    if (this.profileId === 'custom') {
      const o = this.exportOptions;
      const q = el('input', { type: 'range', min: '50', max: '100', value: String(o.quality) }) as HTMLInputElement;
      const qv = el('span', { class: 'val' }, String(o.quality));
      q.addEventListener('input', () => {
        o.quality = Number(q.value);
        qv.textContent = q.value;
      });
      box.appendChild(el('div', { class: 'row', title: 'Match the quality of the place the photo is going; the file is settled at that quality so a re-save there changes nothing' }, el('span', {}, 'Quality'), q, qv));
      const chroma = el('input', { type: 'checkbox' }) as HTMLInputElement;
      chroma.checked = o.chroma === '4:4:4';
      chroma.addEventListener('change', () => (o.chroma = chroma.checked ? '4:4:4' : '4:2:0'));
      box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Colour'), el('label', { title: 'Off halves the colour resolution, like most apps do; on keeps it' }, chroma, ' keep full colour detail'), el('span')));
      const fp = el('input', { type: 'checkbox' }) as HTMLInputElement;
      fp.checked = o.fixedPointPasses > 0;
      fp.addEventListener('change', () => (o.fixedPointPasses = fp.checked ? 3 : 0));
      box.appendChild(el('div', { class: 'row' }, el('span', {}, 'Settle'), el('label', { title: 'Re-encode until the file stops changing, so the destination’s own re-save is a no-op' }, fp, ' make the file a fixed point of this quality'), el('span')));
      const adv = el('details', { class: 'advanced' }, el('summary', {}, 'File options'));
      const prog = el('input', { type: 'checkbox' }) as HTMLInputElement;
      prog.checked = o.progressive;
      prog.addEventListener('change', () => (o.progressive = prog.checked));
      const trellis = el('input', { type: 'checkbox' }) as HTMLInputElement;
      trellis.checked = o.trellis;
      trellis.addEventListener('change', () => (o.trellis = trellis.checked));
      const tables = el('input', { type: 'checkbox' }) as HTMLInputElement;
      tables.checked = o.tables === 'annex-k';
      tables.addEventListener('change', () => (o.tables = tables.checked ? 'annex-k' : 'mozjpeg'));
      adv.appendChild(el('div', { class: 'row' }, el('span', {}, 'Progressive'), el('label', {}, prog, ' loads in passes on the web; same quality'), el('span')));
      adv.appendChild(el('div', { class: 'row' }, el('span', {}, 'Trellis'), el('label', {}, trellis, ' slower encode, smaller file; off when settling'), el('span')));
      adv.appendChild(el('div', { class: 'row' }, el('span', {}, 'Standard tables'), el('label', {}, tables, ' use the same rounding tables as other apps (needed to settle)'), el('span')));
      box.appendChild(adv);
    } else {
      const e = prof.encode;
      box.appendChild(
        el(
          'div',
          { class: 'facts' },
          el('span', {}, 'written as'),
          el('b', {}, `quality ${e.quality}, ${e.chroma === '4:4:4' ? 'full' : 'reduced'} colour detail${e.fixedPointPasses ? ', settled' : ''}`),
          el('span', {}, 'check simulates'),
          el('b', {}, prof.check.label),
        ),
      );
    }

    const b = this.budget;
    const check = el('details', { class: 'advanced' }, el('summary', {}, 'What counts as a problem in the downloader’s copy'));
    const lv = el('input', { type: 'number', min: '0.1', max: '20', step: '0.1', value: String(b.minPeakLevels) }) as HTMLInputElement;
    lv.addEventListener('change', () => (b.minPeakLevels = Number(lv.value)));
    check.appendChild(el('div', { class: 'row' }, el('span', {}, 'A pattern'), el('span', {}, 'still visible at ', lv, ' levels or more'), el('span')));
    check.appendChild(el('div', { style: 'color:var(--ink-faint)' }, 'Each fix is also judged on its own: how much of it is left in the copy, compared with the untouched photo given the same re-save.'));
    box.appendChild(check);

    const layout = this.analysis ? (this.analysis.cfa.detected ? `the photo's own sensor grid (green found; ${['RGGB', 'GRBG', 'GBRG', 'BGGR'][this.analysis.cfa.redY * 2 + this.analysis.cfa.redX]} layout)` : 'a standard RGGB sensor grid (the photo shows no grid of its own)') : 'the photo’s sensor grid';
    const bakeSel = el('select') as HTMLSelectElement;
    for (const [v, label] of [
      ['clean', 'Clean: re-sampled and filled in again, nothing added'],
      ['sharp', 'With subtle sensor noise (a clearer sensor grid)'],
      ['match', 'Match the original’s grid: softer and grainier'],
      ['off', 'Off'],
    ] as [BakeSetting, string][]) {
      const o = el('option', { value: v }, label) as HTMLOptionElement;
      if (v === this.bake) o.selected = true;
      bakeSel.appendChild(o);
    }
    bakeSel.addEventListener('change', () => {
      this.bake = bakeSel.value as BakeSetting;
      this.checked = false;
      this.scheduleLive();
    });
    box.appendChild(
      el(
        'div',
        { class: 'row wide', title: 'At the very end the finished picture is re-sampled to one colour per pixel, the way a camera sensor sees light, in linear light with photon noise at each photosite, and the missing colours are filled in again. Everything, edits and grain included, then carries one consistent sensor signature. The After view does not show this step; As posted and the saved file do.' },
        el('span', {}, 'Sensor bake'),
        bakeSel,
      ),
    );
    box.appendChild(el('div', { class: 'intro' }, `Bakes the finished picture through a camera sensor on ${layout}.`));

    const actions = el('div', { class: 'actions' });
    actions.appendChild(el('button', { class: 'primary', onclick: () => void this.doExport(true), disabled: this.analysis ? undefined : '' }, 'Save JPEG and check it'));
    actions.appendChild(el('button', { onclick: () => void this.doExport(false), disabled: this.analysis ? undefined : '', title: 'Check the whole photo (the live check below looks at the centre) and show the As posted view, without downloading' }, 'Full check'));
    actions.appendChild(el('button', { class: 'quiet', onclick: () => this.downloadRecipe(), disabled: this.analysis ? undefined : '', title: 'Everything you set here, as a file you can load for another photo' }, 'Save settings file'));
    box.appendChild(actions);

    if (this.report) {
      const rep = this.renderReport(this.report);
      box.appendChild(rep);
      if (this.scrollToReport) {
        this.scrollToReport = false;
        requestAnimationFrame(() => rep.scrollIntoView({ block: 'start', behavior: 'smooth' }));
      }
    }
    if (this.analysis) {
      const nc = el('details', { class: 'advanced notchecked' }, el('summary', {}, 'Things this tool does not look for'));
      const ul = el('ul');
      for (const t of this.analysis.notChecked) ul.appendChild(el('li', {}, t));
      nc.appendChild(ul);
      box.appendChild(nc);
    }
  }

  private renderReport(r: RoundTripReport): HTMLElement {
    const box = el('div', { class: `report ${r.pass ? 'pass' : 'fail'}` });
    box.appendChild(el('div', { class: 'verdict' }, r.pass ? `After ${r.checkLabel}: every fix holds` : `After ${r.checkLabel}: not everything holds`));
    box.appendChild(el('div', { class: 'intro' }, 'What a downloader gets, compared with the untouched photo given the same re-save. Use the As posted view and Split to see it.'));
    const tbl = this.survivalTable(r.survival.fixes);
    box.appendChild(tbl);
    const notes: string[] = [];
    for (const fx of r.survival.fixes) {
      if (fx.verdict === 'survives') continue;
      const f = this.analysis?.findings.find((x) => x.id === fx.id);
      const name = f ? plainLabel(f) : fx.label;
      notes.push(`${name}: ${fx.note}`);
    }
    notes.push(...r.reasons);
    if (notes.length) {
      const ul = el('ul');
      for (const t of notes) ul.appendChild(el('li', {}, t));
      box.appendChild(ul);
    }
    const facts = el('div', { class: 'facts' });
    const fact = (k: string, v: string): void => {
      facts.appendChild(el('span', {}, k));
      facts.appendChild(el('b', {}, v));
    };
    fact('file', `${(r.bytes / 1024).toFixed(0)} kB, quality ${r.quality}, ${r.chroma === '4:4:4' ? 'full colour detail' : 'reduced colour detail'}`);
    if (r.passes > 1) fact('settled', r.stable ? `yes, in ${r.passes} passes: a re-save there reproduces it` : `not fully after ${r.passes} passes`);
    if (r.grainScale && r.grainScale !== 1) fact('grain put back', `added at ${fmt(r.grainScale, 1)}× the amount set, because the re-save rounds part of it away; the copy carries about the amount set`);
    if (r.sensor) fact('sensor bake', r.sensor.baked ? `${r.sensor.layout} grid, ${r.sensor.method === 'bilinear' ? 'bilinear' : 'Malvar–He–Cutler'} fill-in, photon noise ${fmt(r.sensor.noiseLevels, 1)} lv; pattern ${Math.round(r.sensor.inFile * 100)} % in the file, ${Math.round(r.sensor.posted * 100)} % after the re-save (original ${Math.round(r.sensor.original * 100)} %)` : 'off');
    if (r.settled)
      fact(
        'block edges written on their grid',
        `${Math.round((100 * r.settled.changedBlocks) / Math.max(1, r.settled.blocks))} % of blocks adjusted at quality ${r.settled.quality} so a re-save keeps the boundaries smooth (moved pixels by ${fmt(r.settled.meanChangeLevels, 2)} lv)`,
      );
    fact('block edges, downloader’s copy', `${fmt(r.survival.blockStepLevels.resultOnDest, 2)} lv (untouched photo: ${fmt(r.survival.blockStepLevels.sourceOnDest, 2)})`);
    fact('grain, downloader’s copy', `${fmt(r.survival.grainLevels.resultOnDest, 2)} lv (untouched photo: ${fmt(r.survival.grainLevels.sourceOnDest, 2)})`);
    fact('patterns visible', r.residualPeriodic ? r.residual.map((f) => (f.kind === 'periodic' ? `every ${fmt(f.details.period, 1)} px, ${fmt(Math.max(f.details.amplitude, f.details.amplitudeP90) * 255, 1)} lv` : '')).join('; ') : 'none');
    fact('saving changed pixels by', `${fmt(r.encodeRmsLevels, 2)} lv on average`);
    const det = el('details', { class: 'advanced' }, el('summary', {}, 'Details of the file and the check'));
    det.appendChild(facts);
    const actions = el('div', { class: 'actions' });
    if (this.jpegUrl) {
      const name = (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '-normalized.jpg';
      actions.appendChild(el('a', { href: this.jpegUrl, download: name }, `Download ${name}`));
    }
    actions.appendChild(el('button', { onclick: () => this.setMode('posted') }, 'Show as posted'));
    for (const sf of r.suggested) actions.appendChild(el('button', { onclick: () => void this.adoptSuggested(sf) }, `Add fix: ${plainLabel(sf).toLowerCase()}`));
    actions.appendChild(el('button', { class: 'quiet', onclick: () => this.downloadReport() }, 'Full report (JSON)'));
    box.appendChild(actions);
    box.appendChild(det);
    return box;
  }

  /** Take a fix the verification proposed into the findings list and build it. */
  private async adoptSuggested(f: Finding): Promise<void> {
    if (!this.analysis || this.busy) return;
    if (this.analysis.findings.some((x) => x.id === f.id)) return;
    const copy: Finding = { ...f, enabled: true, strength: 1 };
    this.analysis.findings.push(copy);
    this.report = null;
    this.renderFindings();
    this.renderExport();
    this.select(copy.id);
    await this.rebuildFrom(copy);
    this.setStatus(`${plainLabel(copy)} added — save again to check it`);
  }

  private async doExport(download = true): Promise<void> {
    if (!this.analysis || this.busy) return;
    this.busy = true;
    this.exportBtn.disabled = true;
    try {
      const { jpeg, report } = await this.client.export(this.layerSettings(), this.analysis.findings, this.master, this.profileId, this.exportOptions, this.budget, this.analysisOptions, (stage, fr) =>
        this.setStatus(`${download ? 'Saving' : 'Checking'}: ${stage.replace('check:', 'looking at the downloader’s copy — ').replace(/^settle$/, 'settling block edges on their grid').replace(/^regrain$/, 'writing again with more grain so the copy carries the amount set').replace(/^bake$/, 'baking through a camera sensor').replace(/^encode$/, 'writing and settling the file').replace(/^check$/, 'simulating the re-save')}`, fr),
        this.bake,
      );
      if (this.jpegUrl) URL.revokeObjectURL(this.jpegUrl);
      this.jpegUrl = URL.createObjectURL(new Blob([jpeg as BlobPart], { type: 'image/jpeg' }));
      this.report = report;
      this.checked = true;
      this.scrollToReport = true;
      const fixes = report.survival.fixes;
      const kept = fixes.filter((f) => f.verdict === 'survives').length;
      const lost = fixes.filter((f) => f.verdict === 'lost').length;
      const how =
        fixes.length === 0
          ? 'nothing to check'
          : kept === fixes.length
            ? `every fix holds through ${report.checkLabel}`
            : lost === 0
              ? `${kept} of ${fixes.length} fixes hold fully through ${report.checkLabel}, the rest partly`
              : `${lost} of ${fixes.length} fixes lost through ${report.checkLabel} — see the report`;
      this.setStatus(`${download ? 'Saved' : 'Checked'} ${(jpeg.length / 1024).toFixed(0)} kB: ${how}`);
      this.renderExport();
      if (download) {
        const name = (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '-normalized.jpg';
        const a = document.createElement('a');
        a.href = this.jpegUrl;
        a.download = name;
        a.click();
      }
    } catch (e) {
      this.setStatus(`${download ? 'Saving' : 'Checking'} failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      this.busy = false;
      this.exportBtn.disabled = false;
    }
  }

  private async loadRecipe(file: File): Promise<void> {
    try {
      const r = JSON.parse(await file.text()) as RecipeFile;
      if (r.tool !== 'nnormalize') throw new Error('not an nnormalize recipe');
      r.name = file.name;
      this.recipe = r;
      if (r.exportOptions) this.exportOptions = { ...this.exportOptions, ...r.exportOptions };
      if (r.profile && r.profile in PROFILES) this.profileId = r.profile;
      if (r.budget) this.budget = { ...this.budget, ...r.budget };
      if (r.analysisOptions) this.analysisOptions = { ...r.analysisOptions };
      if (typeof r.master === 'number') {
        this.master = r.master;
        this.masterRange.value = String(Math.round(r.master * 100));
        this.masterVal.textContent = `${Math.round(r.master * 100)} %`;
      }
      this.renderAnalysisPanel();
      this.renderExport();
      if (this.analysis && !this.busy) {
        this.busy = true;
        await this.applyRecipe(r, this.analysis);
        this.renderFindings();
        this.renderDetail();
        await this.buildPending();
        this.queueRender();
        this.busy = false;
      }
      this.setStatus(`Settings file ${file.name} loaded`);
    } catch (e) {
      this.setStatus(`Could not load that settings file: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  /**
   * Apply a recipe to a fresh analysis: periodic entries match by frequency
   * (within two bins, either conjugate), other kinds by kind. Unmatched
   * recipe entries are ignored; unmatched findings keep their defaults.
   */
  private async applyRecipe(r: RecipeFile, a: Analysis): Promise<void> {
    const tol = 2 / a.spectrum.size;
    for (const entry of r.findings) {
      let target: Finding | undefined;
      if (entry.kind === 'periodic' && entry.details) {
        const d = entry.details;
        target = a.findings.find(
          (f) => f.kind === 'periodic' && ((Math.abs(f.details.u - d.u) < tol && Math.abs(f.details.v - d.v) < tol) || (Math.abs(f.details.u + d.u) < tol && Math.abs(f.details.v + d.v) < tol)),
        );
      } else target = a.findings.find((f) => f.kind === entry.kind);
      if (!target) continue;
      target.enabled = entry.enabled;
      target.strength = entry.strength;
      if (entry.method && target.kind === 'periodic' && ['parametric', 'parametric-tiled', 'flatten', 'notch'].includes(entry.method)) target.method = entry.method as PeriodicMethod;
      if (entry.mask) {
        const bin = atob(entry.mask.data);
        const data = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) data[i] = bin.charCodeAt(i);
        const m = { w: entry.mask.w, h: entry.mask.h, data };
        this.masks.set(target.id, m);
        await this.client.setMask(target.id, m.w, m.h, m.data);
      }
    }
  }

  private async runBatch(files: File[]): Promise<void> {
    if (this.busy) return;
    this.batchQueue = files;
    const results: { file: string; pass: boolean; reasons: string[]; bytes: number }[] = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      this.setStatus(`Batch ${i + 1}/${files.length}: ${f.name}`);
      try {
        await this.open(f);
        if (!this.analysis) throw new Error('analysis failed');
        this.busy = true;
        const { jpeg, report } = await this.client.export(this.layerSettings(), this.analysis.findings, this.master, this.profileId, this.exportOptions, this.budget, this.analysisOptions, (stage, fr) =>
          this.setStatus(`Batch ${i + 1}/${files.length}: ${f.name} — ${stage}`, fr),
          this.bake,
        );
        this.report = report;
        this.renderExport();
        const name = f.name.replace(/\.[^.]+$/, '') + '-normalized.jpg';
        this.downloadBlob(new Blob([jpeg as BlobPart], { type: 'image/jpeg' }), name);
        results.push({ file: f.name, pass: report.pass, reasons: report.reasons, bytes: jpeg.length });
      } catch (e) {
        results.push({ file: f.name, pass: false, reasons: [e instanceof Error ? e.message : String(e)], bytes: 0 });
      } finally {
        this.busy = false;
      }
    }
    this.downloadJson({ tool: 'nnormalize', batch: results, recipe: this.recipe?.name ?? null, exportOptions: this.exportOptions, budget: this.budget }, 'nnormalize-batch-report.json');
    const passed = results.filter((r) => r.pass).length;
    this.setStatus(`Batch done: ${passed}/${results.length} passed verification`);
  }

  private downloadBlob(blob: Blob, name: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }

  private recipeJson(): unknown {
    if (!this.analysis) return null;
    return {
      tool: 'nnormalize',
      version: 1,
      source: this.info?.meta.fileName,
      master: this.master,
      profile: this.profileId,
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
    const r = this.recipeJson();
    if (!r) return;
    this.downloadJson(r, (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '.nnormalize.json');
  }

  private downloadReport(): void {
    if (!this.report) return;
    const r = { ...this.report, residual: this.report.residual.map((f) => ({ ...f, details: f.kind === 'periodic' ? { ...f.details, presence: undefined } : f.details })) };
    this.downloadJson({ report: r, recipe: this.recipeJson(), notChecked: this.analysis?.notChecked }, (this.info?.meta.fileName ?? 'image').replace(/\.[^.]+$/, '') + '.report.json');
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
