// SPDX-License-Identifier: MIT OR Apache-2.0
// browseML's search from WebAssembly: the Rust crate in rust/, compiled to search.wasm, behind the same API as
// search.js — BM25 (scores, rank), fuse, cosine — and the same numbers, bit for bit (search-wasm.test.mjs measures it).
// search.wasm (wasm32-unknown-unknown) imports nothing. A wasm32-wasip1 build also loads: its four WASI imports are
// answered with stubs here, since the search never calls them. Nothing is sent anywhere.
//
//   import { load } from "./search-wasm.js";
//   const { BM25, fuse, cosine } = await load();               // or load(bytes | Response | URL)
//   const idx = new BM25(texts); idx.rank("does my typing leave the device"); idx.free();

const WASI = {
  environ_sizes_get: () => 0, environ_get: () => 0, fd_write: () => 52 /* ENOSYS */,
  proc_exit: (c) => { throw new Error(`search.wasm exited (${c})`); },
};

export async function load(source = new URL("./search.wasm", import.meta.url)) {
  let bytes = source;
  if (source instanceof URL || typeof source === "string") {
    const url = new URL(source, import.meta.url);
    bytes = url.protocol === "file:" ? (await import("node:fs")).readFileSync(url) : await fetch(url);
  }
  if (typeof Response !== "undefined" && bytes instanceof Response) bytes = await bytes.arrayBuffer();
  const module = await WebAssembly.compile(bytes), imports = {};
  for (const i of WebAssembly.Module.imports(module)) {
    if (i.module !== "wasi_snapshot_preview1" || !(i.name in WASI)) throw new Error(`search.wasm wants ${i.module}.${i.name}`);
    (imports[i.module] ??= {})[i.name] = WASI[i.name];
  }
  const w = (await WebAssembly.instantiate(module, imports)).exports;
  const enc = new TextEncoder();
  const u8 = () => new Uint8Array(w.memory.buffer);          // fresh views: memory may grow on any call
  const put = (s) => { const b = enc.encode(String(s)), p = w.bml_alloc(b.length); u8().set(b, p); return [p, b.length]; };
  const gone = new FinalizationRegistry((h) => w.bml_bm25_free(h));

  class BM25 {
    constructor(texts = [], { k1 = 1.2, b = 0.75 } = {}) {
      this.h = w.bml_bm25_new(k1, b); gone.register(this, this.h, this);
      for (const t of texts) this.add(t);
    }
    add(text) { const [p, n] = put(text); try { return w.bml_bm25_add(this.h, p, n); } finally { w.bml_free(p, n); } }
    get length() { return w.bml_bm25_len(this.h); }
    /** Each document's score for the query, in the order added. */
    scores(query) {
      const len = this.length, [p, n] = put(query), out = w.bml_alloc(8 * len);
      try { w.bml_bm25_scores(this.h, p, n, out); return Array.from(new Float64Array(w.memory.buffer, out, len)); }
      finally { w.bml_free(out, 8 * len); w.bml_free(p, n); }
    }
    /** Document indices, best first (only those that match). */
    rank(query, limit = 10) {
      const cap = Math.min(limit, this.length), [p, n] = put(query), out = w.bml_alloc(4 * cap);
      try { const c = w.bml_bm25_rank(this.h, p, n, cap, out); return Array.from(new Uint32Array(w.memory.buffer, out, c)); }
      finally { w.bml_free(out, 4 * cap); w.bml_free(p, n); }
    }
    free() { if (this.h) { gone.unregister(this); w.bml_bm25_free(this.h); this.h = 0; } }
  }

  /** Reciprocal rank fusion: [[index, score], …] best first, as search.js's fuse. */
  function fuse(rankings, k = 60) {
    const total = rankings.reduce((s, r) => s + r.length, 0), cnt = rankings.length;
    const ids = w.bml_alloc(4 * total), lens = w.bml_alloc(4 * cnt), oi = w.bml_alloc(4 * total), os = w.bml_alloc(8 * total);
    try {
      new Uint32Array(w.memory.buffer, ids, total).set(rankings.flat());
      new Uint32Array(w.memory.buffer, lens, cnt).set(rankings.map((r) => r.length));
      const c = w.bml_fuse(ids, lens, cnt, k, oi, os);
      const I = new Uint32Array(w.memory.buffer, oi, c), S = new Float64Array(w.memory.buffer, os, c);
      return Array.from(I, (i, j) => [i, S[j]]);
    } finally { w.bml_free(ids, 4 * total); w.bml_free(lens, 4 * cnt); w.bml_free(oi, 4 * total); w.bml_free(os, 8 * total); }
  }

  /** Cosine similarity of two unit-length vectors; Float32Arrays stay f32 on the way in, summed in f64. */
  function cosine(a, b) {
    const f32 = a instanceof Float32Array, sz = f32 ? 4 : 8, n = a.length, T = f32 ? Float32Array : Float64Array;
    const pa = w.bml_alloc(sz * n), pb = w.bml_alloc(sz * n);
    try {
      new T(w.memory.buffer, pa, n).set(a); new T(w.memory.buffer, pb, n).set(b);
      return (f32 ? w.bml_cosine_f32 : w.bml_cosine)(pa, pb, n);
    } finally { w.bml_free(pa, sz * n); w.bml_free(pb, sz * n); }
  }

  return { BM25, fuse, cosine, ln: w.bml_ln };
}
