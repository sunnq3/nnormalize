import { Engine } from '../engine/engine';
import type { Request, Response } from '../engine/protocol';

const engine = new Engine();
const ctx = self as unknown as Worker;

function post(msg: Response, transfer: Transferable[] = []): void {
  ctx.postMessage(msg, transfer);
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
        post({ type: 'analysis', id, analysis }); // cloned, not transferred: the engine keeps using these arrays
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
        post({ type: 'finding', id, finding });
        break;
      }
      case 'render': {
        const result = engine.render(msg.req);
        post({ type: 'render', id, result }, [result.rgba.buffer]);
        break;
      }
      case 'export': {
        const { jpeg, report } = await engine.exportJpeg(msg.layers, msg.findings, msg.master, msg.profile, msg.options, msg.budget, msg.analysisOptions, progress, msg.bake, msg.signature);
        post({ type: 'export', id, jpeg, report }, [jpeg.buffer]);
        break;
      }
      case 'liveCheck': {
        const report = await engine.liveCheck(msg.layers, msg.findings, msg.master, msg.profile, msg.options, msg.bake, msg.signature);
        post({ type: 'live', id, report });
        break;
      }
      case 'signatureLearn': {
        const learned = await engine.signatureLearn(msg.file as Blob & { name?: string });
        await engine.signature.save();
        post({ type: 'signature', id, status: await engine.signatureStatus(), learned });
        break;
      }
      case 'signatureStatus': {
        post({ type: 'signature', id, status: await engine.signatureStatus() });
        break;
      }
      case 'signatureExport': {
        await engine.signature.save();
        const bytes = engine.signature.exportBlob();
        post({ type: 'bytes', id, bytes }, bytes ? [bytes.buffer] : []);
        break;
      }
      case 'signatureImport': {
        await engine.signature.importBlob(msg.bytes);
        post({ type: 'signature', id, status: await engine.signatureStatus() });
        break;
      }
      case 'signatureForget': {
        await engine.signature.forget();
        post({ type: 'signature', id, status: await engine.signatureStatus() });
        break;
      }
      case 'verifyResult': {
        const analysis = engine.verifyResult(msg.layers, msg.master, msg.analysisOptions, progress);
        post({ type: 'verify', id, analysis });
        break;
      }
    }
  } catch (e) {
    post({ type: 'error', id, message: e instanceof Error ? e.message : String(e) });
  }
};
