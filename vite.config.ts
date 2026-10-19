import { defineConfig } from 'vitest/config';

// `base: './'` keeps every asset path relative, so the same build works on
// GitHub Pages under /nnormalize/, on any sub-path, and from a local folder.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: true,
  },
  worker: {
    format: 'es',
  },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
