/**
 * The sensor-signature bank: one running PRNU estimate per camera and frame
 * size, learned from the photos that pass through the tool, kept in the
 * browser's own storage, and put back into finished pictures.
 */

import { newPrnuStore, packPrnuStore, prnuAccumulate, prnuFinalize, prnuReliability, unpackPrnuStore, type PrnuMap, type PrnuStore } from '../dsp/prnu';
import type { Plane } from '../dsp/plane';

export interface SignatureStatus {
  /** the photo currently loaded: its camera key, or null when the file names no camera */
  photoKey: string | null;
  /** the signature in hand: key, how many photos taught it, how reliable it is */
  key: string | null;
  camera: string;
  size: string;
  photos: number;
  splitHalf: number;
  reliability: number;
  /** the signature in hand belongs to the loaded photo's camera and size */
  matchesPhoto: boolean;
  /** reliable enough to put back (4+ photos and reliability ≥ 0.3) */
  ready: boolean;
  /** why the last save failed, if it did */
  saveError: string | null;
  /** other signatures kept in this browser */
  others: { key: string; photos: number }[];
}

const DB = 'nnormalize';
const STORE = 'signatures';

function openDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'key' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
}

function idbGet<T>(db: IDBDatabase, key: string): Promise<T | undefined> {
  return new Promise((resolve) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result as T | undefined);
    req.onerror = () => resolve(undefined);
  });
}

function idbPut(db: IDBDatabase, value: unknown): Promise<void> {
  return new Promise((resolve, reject) => {
    let tx: IDBTransaction;
    try {
      tx = db.transaction(STORE, 'readwrite');
      tx.objectStore(STORE).put(value);
    } catch (e) {
      reject(e instanceof Error ? e : new Error(String(e)));
      return;
    }
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(new Error(tx.error?.message ?? 'storage error'));
    tx.onabort = () => reject(new Error(tx.error?.message ?? 'storage refused the write (out of space?)'));
  });
}

function idbDelete(db: IDBDatabase, key: string): Promise<void> {
  return new Promise((resolve) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

function idbAll(db: IDBDatabase): Promise<{ key: string; header: Record<string, unknown> }[]> {
  return new Promise((resolve) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve((req.result as { key: string; header: Record<string, unknown> }[]) ?? []);
    req.onerror = () => resolve([]);
  });
}

interface Record_ {
  key: string;
  header: Record<string, unknown>;
  buffers: ArrayBuffer[];
}

const MAGIC = 'NNPRNU1\n';

export class SignatureBank {
  private store: PrnuStore | null = null;
  private dirty = false;
  private map: PrnuMap | null = null;
  private seen = new Set<string>();
  private db: IDBDatabase | null | undefined;
  photoKey: string | null = null;

  private async database(): Promise<IDBDatabase | null> {
    if (this.db === undefined) this.db = await openDb();
    return this.db;
  }

  /** Switch to the signature filed under `key` (saving the current one first), or to none. */
  async select(key: string | null): Promise<void> {
    if (this.store && this.store.key === key) return;
    await this.save();
    this.store = null;
    this.map = null;
    this.seen.clear();
    if (!key) return;
    const db = await this.database();
    const rec = db ? await idbGet<Record_>(db, key) : undefined;
    if (rec) {
      try {
        this.store = unpackPrnuStore(rec.header, rec.buffers);
        const seen = (rec.header.seen as string[] | undefined) ?? [];
        // the list of files learned from can never be longer than the count of photos learned
        if (seen.length <= this.store.photos) for (const s of seen) this.seen.add(s);
      } catch {
        this.store = null;
        if (db) await idbDelete(db, key);
      }
    }
  }

  /** What the last save did, for the panel: null when it went through, else why not. */
  lastSaveError: string | null = null;

  /** Fold a photo in. Returns what happened in plain words. */
  async learn(key: string, planes: [Plane, Plane, Plane], luma: Plane, fileId: string): Promise<{ accepted: boolean; reason: string; usable: number }> {
    await this.select(key);
    if (this.seen.has(fileId)) return { accepted: false, reason: 'already learned from this file', usable: 0 };
    if (!this.store) this.store = newPrnuStore(key, planes[0].w, planes[0].h);
    if (this.store.w !== planes[0].w || this.store.h !== planes[0].h) return { accepted: false, reason: 'different frame size', usable: 0 };
    const usable = prnuAccumulate(this.store, planes, luma);
    this.seen.add(fileId);
    this.dirty = true;
    this.map = null;
    return { accepted: true, reason: usable < 0.05 ? 'accepted, but almost nothing in it was smooth and well lit' : 'accepted', usable };
  }

  async save(): Promise<void> {
    if (!this.store || !this.dirty) return;
    const db = await this.database();
    if (!db) {
      this.lastSaveError = 'this browser offers no storage (private window?); the signature lives only until the page is closed';
      return;
    }
    const { header, buffers } = packPrnuStore(this.store);
    header.seen = [...this.seen];
    try {
      await idbPut(db, { key: this.store.key, header, buffers } satisfies Record_);
      this.dirty = false;
      this.lastSaveError = null;
    } catch (e) {
      this.lastSaveError = `could not be saved (${e instanceof Error ? e.message : String(e)}); it lives only until the page is closed. Export it to a file to keep it`;
    }
  }

  async forget(): Promise<void> {
    const key = this.store?.key;
    this.store = null;
    this.map = null;
    this.seen.clear();
    this.dirty = false;
    const db = await this.database();
    if (db && key) await idbDelete(db, key);
  }

  /** The finished map for the loaded photo's camera, or null when there is none worth using. */
  async mapFor(key: string | null, progress: (fraction: number) => void = () => {}): Promise<PrnuMap | null> {
    if (!key) return null;
    await this.select(key);
    if (!this.store || this.store.photos < 4) return null;
    if (!this.map) this.map = prnuFinalize(this.store, progress);
    return this.map.reliability >= 0.3 ? this.map : null;
  }

  async status(): Promise<SignatureStatus> {
    const db = await this.database();
    const all = db ? await idbAll(db) : [];
    const s = this.store;
    const rel = s ? prnuReliability(s) : { splitHalf: 0, reliability: 0 };
    const [camera, size] = s ? s.key.split('|') : ['', ''];
    return {
      photoKey: this.photoKey,
      key: s?.key ?? null,
      camera,
      size,
      photos: s?.photos ?? 0,
      splitHalf: rel.splitHalf,
      reliability: rel.reliability,
      matchesPhoto: !!s && s.key === this.photoKey,
      ready: !!s && s.photos >= 4 && rel.reliability >= 0.3,
      saveError: this.lastSaveError,
      others: all.filter((r) => r.key !== s?.key).map((r) => ({ key: r.key, photos: Number(r.header.photos) })),
    };
  }

  /** The signature in hand as one file. */
  exportBlob(): Uint8Array | null {
    if (!this.store) return null;
    const { header, buffers } = packPrnuStore(this.store);
    header.seen = [...this.seen];
    header.lengths = buffers.map((b) => b.byteLength);
    const head = new TextEncoder().encode(JSON.stringify(header));
    const total = MAGIC.length + 4 + head.length + buffers.reduce((s, b) => s + b.byteLength, 0);
    const out = new Uint8Array(total);
    let o = 0;
    out.set(new TextEncoder().encode(MAGIC), o);
    o += MAGIC.length;
    new DataView(out.buffer).setUint32(o, head.length, true);
    o += 4;
    out.set(head, o);
    o += head.length;
    for (const b of buffers) {
      out.set(new Uint8Array(b), o);
      o += b.byteLength;
    }
    return out;
  }

  async importBlob(bytes: Uint8Array): Promise<string> {
    const magic = new TextDecoder().decode(bytes.subarray(0, MAGIC.length));
    if (magic !== MAGIC) throw new Error('not a signature file');
    let o = MAGIC.length;
    const len = new DataView(bytes.buffer, bytes.byteOffset).getUint32(o, true);
    o += 4;
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(o, o + len))) as Record<string, unknown>;
    o += len;
    const buffers: ArrayBuffer[] = [];
    for (const l of header.lengths as number[]) {
      buffers.push(bytes.slice(o, o + l).buffer);
      o += l;
    }
    await this.save();
    this.store = unpackPrnuStore(header, buffers);
    this.seen = new Set((header.seen as string[] | undefined) ?? []);
    this.map = null;
    this.dirty = true;
    await this.save();
    return this.store.key;
  }
}
