import { Engine } from '../engine/engine';
import type { Request, Response } from '../engine/protocol';

const engine = new Engine();
const ctx = self as unknown as Worker;

function post(msg: Response, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
}

function transferablesOf(obj: unknown, acc: Transferable[] = [], seen = new Set<unknown>()): Transferable[] {
  if (!obj || typeof obj !== 'object' || seen.has(obj)) return acc;
  seen.add(obj);
  if (ArrayBuffer.isView(obj)) {
    if (!acc.includes(obj.buffer as ArrayBuffer)) acc.push(obj.buffer as ArrayBuffer);
    return acc;
  }
  if (Array.isArray(obj)) for (const v of obj) transferablesOf(v, acc, seen);
  else for (const v of Object.values(obj as Record<string, unknown>)) transferablesOf(v, acc, seen);
  return acc;
}

ctx.onmessage = async (ev: MessageEvent<Request>) => {
  const msg = ev.data;
  const id = msg.id;
  const progress = (stage: string, fraction: number): void => post({ type: 'progress', id, stage, fraction });
  try {
    switch (msg.type) {
      case 'load': {
        const info = await engine.load(msg.file as Blob & { name?: string });
        post({ type: 'loaded', id, info });
        break;
      }
      case 'synth': {
        const info = engine.synth(msg.spec);
        post({ type: 'loaded', id, info });
        break;
      }
      case 'analyze': {
        const analysis = engine.runAnalysis(msg.options, progress);
        post({ type: 'analysis', id, analysis }, transferablesOf(analysis));
        break;
      }
      case 'delta': {
        const stats = engine.buildLayer(msg.finding, (f) => progress('delta', f));
        post({ type: 'delta', id, stats });
        break;
      }
      case 'setMask': {
        engine.setMask(msg.findingId, msg.w, msg.h, msg.data);
        post({ type: 'ok', id });
        break;
      }
      case 'presenceMask': {
        const f = engine.analysis?.findings.find((x) => x.id === msg.findingId);
        if (!f) throw new Error('unknown finding');
        const m = engine.presenceMask(f);
        engine.setMask(msg.findingId, m.w, m.h, m.data);
        post({ type: 'mask', id, w: m.w, h: m.h, data: new Uint8Array(m.data) });
        break;
      }
      case 'luminosityMask': {
        const m = engine.luminosityMask(msg.lo, msg.hi, msg.soft);
        engine.setMask(msg.findingId, m.w, m.h, m.data);
        post({ type: 'mask', id, w: m.w, h: m.h, data: new Uint8Array(m.data) });
        break;
      }
      case 'addPeak': {
        const finding = engine.addPeak(msg.u, msg.v);
        post({ type: 'finding', id, finding }, transferablesOf(finding));
        break;
      }
      case 'render': {
        const result = engine.render(msg.req);
        post({ type: 'render', id, result }, [result.rgba.buffer]);
        break;
      }
      case 'export': {
        const { jpeg, report } = await engine.exportJpeg(msg.layers, msg.master, msg.options, msg.budget, msg.analysisOptions, progress);
        post({ type: 'export', id, jpeg, report }, [jpeg.buffer, ...transferablesOf(report)]);
        break;
      }
      case 'verifyResult': {
        const analysis = engine.verifyResult(msg.layers, msg.master, msg.analysisOptions, progress);
        post({ type: 'verify', id, analysis }, transferablesOf(analysis));
        break;
      }
    }
  } catch (e) {
    post({ type: 'error', id, message: e instanceof Error ? e.message : String(e) });
  }
};
