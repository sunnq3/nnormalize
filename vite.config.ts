import { defineConfig } from 'vitest/config';
import renameEntry from './scripts/rename-entry.mjs';

// `base: './'` keeps every asset path relative, so the same build works on
// GitHub Pages under /nnormalize/, on any sub-path, and from a local folder.
// The dev entry is dev.html: the root index.html is the committed production
// build that GitHub Pages serves when its source is the main branch.
export default defineConfig({
  base: './',
  plugins: [renameEntry()],
  server: { open: '/dev.html' },
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: { input: 'dev.html' },
  },
  worker: {
    format: 'es',
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
