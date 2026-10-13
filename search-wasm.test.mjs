// SPDX-License-Identifier: MIT OR Apache-2.0
// The Rust search (search.wasm, and the wasm32-wasip1 build when present) against expected.json and against
// search.js itself: the same scores bit for bit, the same rankings, the same fusion, the same logarithm as V8.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import * as js from "./search.js";
import { load } from "./search-wasm.js";
const ex = JSON.parse(readFileSync(new URL("./expected.json", import.meta.url), "utf8"));
const wasi = new URL("./rust/target/wasm32-wasip1/release/browseml_embeddings.wasm", import.meta.url);
const builds = [["search.wasm (wasm32-unknown-unknown)", undefined]];
if (existsSync(wasi)) builds.push(["wasm32-wasip1", readFileSync(wasi)]);

// a deterministic generator (xorshift64*), so every run checks the same corpus
function rng(seed) { let s = BigInt(seed); return () => { s ^= s >> 12n; s ^= (s << 25n) & 0xffffffffffffffffn; s ^= s >> 27n; return Number(((s * 0x2545f4914f6cdd1dn) & 0xffffffffffffffffn) >> 11n) / 2 ** 53; }; }
function corpus(n, seed = 7) {
  const r = rng(seed), vocab = Array.from({ length: 3000 }, (_, i) => i.toString(36) + ["", "ing", "ed", "s"][i % 4]);
  const extra = ["İSTANBUL", "Kelvin", "café", "Bonsai-1.7B", "248MB!", "ΣΊΣΥΦΟΣ", "naïve", "ﬁle"];
  const word = () => (r() < 0.02 ? extra[Math.floor(r() * extra.length)] : vocab[Math.floor(vocab.length * r() ** 3)]);
  return Array.from({ length: n }, () => Array.from({ length: 5 + Math.floor(r() * 120) }, word).join(r() < 0.5 ? " " : ", "));
}

for (const [name, src] of builds) {
  test(`${name}: BM25 scores are Savante's (expected.json), and repeated words count once`, async () => {
    const w = await load(src), idx = new w.BM25(ex.docs);
    let worst = 0, n = 0;
    for (const [q, want] of Object.entries(ex.queries)) idx.scores(q).forEach((s, i) => { worst = Math.max(worst, Math.abs(s - want[i])); n++; });
    assert.ok(worst < 1e-12, `max |error| ${worst}`);
    assert.equal(n, 15);
    idx.scores("browser BROWSER cpu").forEach((s, i) => assert.ok(Math.abs(s - ex.queries["browser cpu"][i]) < 1e-12));
    assert.equal(idx.rank("does my typing leave the device")[0], 4);
    assert.equal(idx.rank("Bonsai model size in MB")[0], 2);
    const f = w.fuse([[2, 0, 1], [0, 3]]);
    assert.equal(f[0][0], 0);
    assert.ok(Math.abs(f[0][1] - (1 / 62 + 1 / 61)) < 1e-15);
    idx.free();
  });

  test(`${name}: ln is V8's Math.log, bit for bit`, async () => {
    const { ln } = await load(src), r = rng(11), buf = new Float64Array(1), bits = new BigUint64Array(buf.buffer);
    let differ = 0, n = 0;
    for (let i = 0; i < 200000; i++) {
      bits[0] = BigInt(Math.floor(r() * 2 ** 31)) << 32n | BigInt(Math.floor(r() * 2 ** 32)); // any positive double
      for (const x of [buf[0], 1 + r() * 4096, 1 + r() * 1e-6]) { if (!Object.is(ln(x), Math.log(x))) differ++; n++; }
    }
    assert.equal(differ, 0, `${differ} of ${n} differ`);
  });

  test(`${name}: the same scores and rankings as search.js on a generated corpus`, async () => {
    const w = await load(src), docs = corpus(2000), a = new js.BM25(docs), b = new w.BM25(docs), r = rng(3);
    let bitwise = 0, total = 0;
    const queries = Array.from({ length: 200 }, (_, i) => docs[Math.floor(r() * docs.length)].split(/[ ,]+/).slice(i % 7, (i % 7) + 1 + (i % 5)).join(" "));
    queries.push("İ kelvin caf", "ﬁle file", "σίσυφος", "nothing-matches-zzzz", "");
    for (const q of queries) {
      const sa = a.scores(q), sb = b.scores(q);
      sa.forEach((s, i) => { total++; if (Object.is(s, sb[i])) bitwise++; });
      assert.deepEqual(b.rank(q, 25), a.rank(q, 25), q);
    }
    assert.equal(bitwise, total, `${total - bitwise} of ${total} scores differ`);
    b.free();
  });

  test(`${name}: fusion and cosine as search.js`, async () => {
    const w = await load(src), r = rng(5);
    for (let t = 0; t < 200; t++) {
      const rk = Array.from({ length: 1 + (t % 4) }, () => Array.from({ length: Math.floor(r() * 40) }, () => Math.floor(r() * 60)));
      assert.deepEqual(w.fuse(rk, 60), js.fuse(rk, 60));
      const d = 384 + (t % 3) * 320, f = Float32Array.from({ length: d }, () => r() - 0.5), g = Float32Array.from({ length: d }, () => r() - 0.5);
      assert.ok(Object.is(w.cosine(f, g), js.cosine(f, g)));
      const F = Array.from(f, (x) => x * 1.0000001), G = Array.from(g);
      assert.ok(Object.is(w.cosine(F, G), js.cosine(F, G)));
    }
  });
}
