import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';
import renameEntry from './scripts/rename-entry.mjs';

// One self-contained HTML file: scripts, styles, the engine worker and every
// WASM binary inlined, so it opens from a local folder without a server.
export default defineConfig({
  base: './',
  plugins: [viteSingleFile({ removeViteModuleLoader: true }), renameEntry()],
  resolve: {
    alias: {
      './workerFactory': fileURLToPath(new URL('./src/engine/workerFactory.single.ts', import.meta.url)),
    },
  },
  build: {
    target: 'es2022',
    outDir: 'dist-single',
    rollupOptions: { input: 'dev.html' },
    assetsInlineLimit: () => true,
    sourcemap: false,
    cssCodeSplit: false,
  },
  worker: {
    format: 'iife',
    rollupOptions: { output: { codeSplitting: false } },
  },
});
