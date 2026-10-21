/**
 * Promise wrapper around the engine worker.
 */

import type { Analysis, AnalysisOptions, Finding } from '../analysis/types';
import { createEngineWorker } from './workerFactory';
import type { ProfileId } from './profiles';
import type { JpegExportOptions } from '../io/encode';
import type { BakeSetting, Budget, LayerSetting, LayerStats, LiveReport, LoadedInfo, RenderRequest, RenderResult, Request, Response, RoundTripReport, SynthRequest } from './protocol';

type RequestBody = Request extends infer R ? (R extends { id: number } ? Omit<R, 'id'> : never) : never;

type Pending = { resolve: (v: Response) => void; reject: (e: Error) => void; onProgress?: (stage: string, fraction: number) => void };

export class EngineClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();

  constructor() {
    this.worker = createEngineWorker();
    this.worker.onmessage = (ev: MessageEvent<Response>) => {
      const msg = ev.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.type === 'progress') {
        p.onProgress?.(msg.stage, msg.fraction);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else p.resolve(msg);
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message));
      this.pending.clear();
    };
  }

  private send(req: RequestBody, onProgress?: (stage: string, fraction: number) => void, transfer: Transferable[] = []): Promise<Response> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ ...req, id } as Request, transfer);
    });
  }

  async load(file: File): Promise<LoadedInfo> {
    const r = await this.send({ type: 'load', file });
    if (r.type !== 'loaded') throw new Error('unexpected reply');
    return r.info;
  }

  async synth(spec: SynthRequest): Promise<LoadedInfo> {
    const r = await this.send({ type: 'synth', spec });
    if (r.type !== 'loaded') throw new Error('unexpected reply');
    return r.info;
  }

  async analyze(options: Partial<AnalysisOptions>, onProgress?: (stage: string, fraction: number) => void): Promise<Analysis> {
    const r = await this.send({ type: 'analyze', options }, onProgress);
    if (r.type !== 'analysis') throw new Error('unexpected reply');
    return r.analysis;
  }

  async buildLayer(finding: Finding, onProgress?: (stage: string, fraction: number) => void): Promise<LayerStats> {
    const r = await this.send({ type: 'delta', finding }, onProgress);
    if (r.type !== 'delta') throw new Error('unexpected reply');
    return r.stats;
  }

  async setMask(findingId: string, w: number, h: number, data: Uint8Array): Promise<void> {
    await this.send({ type: 'setMask', findingId, w, h, data: new Uint8Array(data) });
  }

  async presenceMask(findingId: string): Promise<{ w: number; h: number; data: Uint8Array }> {
    const r = await this.send({ type: 'presenceMask', findingId });
    if (r.type !== 'mask') throw new Error('unexpected reply');
    return { w: r.w, h: r.h, data: r.data };
  }

  async luminosityMask(findingId: string, lo: number, hi: number, soft: number): Promise<{ w: number; h: number; data: Uint8Array }> {
    const r = await this.send({ type: 'luminosityMask', findingId, lo, hi, soft });
    if (r.type !== 'mask') throw new Error('unexpected reply');
    return { w: r.w, h: r.h, data: r.data };
  }

  async addPeak(u: number, v: number): Promise<Finding> {
    const r = await this.send({ type: 'addPeak', u, v });
    if (r.type !== 'finding') throw new Error('unexpected reply');
    return r.finding;
  }

  async render(req: RenderRequest): Promise<RenderResult> {
    const r = await this.send({ type: 'render', req });
    if (r.type !== 'render') throw new Error('unexpected reply');
    return r.result;
  }

  async export(
    layers: LayerSetting[],
    findings: Finding[],
    master: number,
    profile: ProfileId,
    options: JpegExportOptions,
    budget: Budget,
    analysisOptions: Partial<AnalysisOptions>,
    onProgress?: (stage: string, fraction: number) => void,
    bake: BakeSetting = 'sharp',
  ): Promise<{ jpeg: Uint8Array; report: RoundTripReport }> {
    const r = await this.send({ type: 'export', layers, findings, master, profile, options, budget, analysisOptions, bake }, onProgress);
    if (r.type !== 'export') throw new Error('unexpected reply');
    return { jpeg: r.jpeg, report: r.report };
  }

  async liveCheck(layers: LayerSetting[], findings: Finding[], master: number, profile: ProfileId, options: JpegExportOptions, bake: BakeSetting = 'sharp'): Promise<LiveReport> {
    const r = await this.send({ type: 'liveCheck', layers, findings, master, profile, options, bake });
    if (r.type !== 'live') throw new Error('unexpected reply');
    return r.report;
  }

  async verifyResult(layers: LayerSetting[], master: number, analysisOptions: Partial<AnalysisOptions>, onProgress?: (stage: string, fraction: number) => void): Promise<Analysis> {
    const r = await this.send({ type: 'verifyResult', layers, master, analysisOptions }, onProgress);
    if (r.type !== 'verify') throw new Error('unexpected reply');
    return r.analysis;
  }
}
