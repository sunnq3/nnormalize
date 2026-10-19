import { App } from './ui/app';

const root = document.getElementById('app');
if (!root) throw new Error('missing #app');
new App(root);
