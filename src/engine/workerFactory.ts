/** Normal build: the worker is a separate module file next to the app. */
export function createEngineWorker(): Worker {
  return new Worker(new URL('../workers/engine.worker.ts', import.meta.url), { type: 'module' });
}
