/** Tiny element builder: el('button', { class: 'x', onclick }, 'label', childEl). */
export function el(tag: string, attrs: Record<string, string | undefined | ((e: Event) => void)> = {}, ...children: (Node | string)[]): HTMLElement {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined) continue;
    if (typeof v === 'function') e.addEventListener(k.replace(/^on/, ''), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c === '') continue;
    e.append(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return e;
}

export function fmt(v: number, digits: number): string {
  if (!Number.isFinite(v)) return '∞';
  return v.toFixed(digits);
}
