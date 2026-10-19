// Copy the production build to the repository root so GitHub Pages can serve
// it when its source is "Deploy from a branch" (main, root). The hosted copy
// through the Actions deploy serves the same files from dist/.
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';

if (!existsSync('dist/index.html')) throw new Error('run `npm run build` first');
rmSync('assets', { recursive: true, force: true });
mkdirSync('assets');
cpSync('dist/assets', 'assets', { recursive: true });
cpSync('dist/index.html', 'index.html');
if (existsSync('dist-single/index.html')) cpSync('dist-single/index.html', 'nnormalize-single.html');
writeFileSync('.nojekyll', '');
console.log('copied build to repository root');
