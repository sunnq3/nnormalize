// Vite names the built page after its input file; the dev entry is dev.html
// (the root index.html is the committed build), so rename the output.
import { renameSync, existsSync } from 'node:fs';
import { join } from 'node:path';

export default function renameEntry() {
  let outDir = 'dist';
  return {
    name: 'rename-entry',
    configResolved(c) {
      outDir = c.build.outDir;
    },
    closeBundle() {
      const from = join(outDir, 'dev.html');
      if (existsSync(from)) renameSync(from, join(outDir, 'index.html'));
    },
  };
}
