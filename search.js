// SPDX-License-Identifier: MIT OR Apache-2.0
// Search in the browser, as bankML's Savante searches its .history: BM25 over words (k1 = 1.2, b = 0.75, Lucene's
// idf), and reciprocal rank fusion (k = 60, Cormack, Clarke and Büttcher 2009) to combine it with a second ranking —
// meaning (embeddings) when a model for them runs here. The scores are Savante's, number for number (tested against
// its formula). Nothing is sent anywhere.

/** Savante's words: lowercase runs of letters and digits. */
export const tokens = (t) => (String(t).toLowerCase().match(/[a-z0-9]+/g) ?? []);

/** A BM25 index over texts. */
export class BM25 {
  constructor(texts = [], { k1 = 1.2, b = 0.75 } = {}) {
    this.k1 = k1; this.b = b; this.docs = []; this.df = new Map();
    for (const t of texts) this.add(t);
  }
  add(text) {
    const ws = tokens(text), tf = new Map();
    for (const w of ws) tf.set(w, (tf.get(w) ?? 0) + 1);
    for (const w of tf.keys()) this.df.set(w, (this.df.get(w) ?? 0) + 1);
    this.docs.push({ tf, len: ws.length });
    return this.docs.length - 1;
  }
  /** Each document's score for the query, in the order added. */
  scores(query) {
    const n = this.docs.length, avg = n ? this.docs.reduce((s, d) => s + d.len, 0) / n : 1, q = tokens(query);
    return this.docs.map((d) => {
      let sc = 0;
      for (const w of q) {
        const f = d.tf.get(w);
        if (!f) continue;
        const df = this.df.get(w);
        const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
        sc += idf * f * (this.k1 + 1) / (f + this.k1 * (1 - this.b + this.b * d.len / avg));
      }
      return sc;
    });
  }
  /** Document indices, best first (only those that match). */
  rank(query, limit = 10) {
    return this.scores(query).map((s, i) => [i, s]).filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]).slice(0, limit).map(([i]) => i);
  }
}

/** Reciprocal rank fusion of rankings (lists of document indices, best first): score = Σ 1 / (k + rank). */
export function fuse(rankings, k = 60) {
  const sc = new Map();
  for (const r of rankings) r.forEach((i, at) => sc.set(i, (sc.get(i) ?? 0) + 1 / (k + at + 1)));
  return [...sc.entries()].sort((a, b) => b[1] - a[1]);
}

/** Cosine similarity of two unit-length vectors (an embedding model's output). */
export const cosine = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
