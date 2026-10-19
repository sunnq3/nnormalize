/** Single-file build: the worker is inlined into the page as a blob. */
import InlineWorker from '../workers/engine.worker.ts?worker&inline';

export function createEngineWorker(): Worker {
  return new InlineWorker();
}
