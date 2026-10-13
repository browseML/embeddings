// SPDX-License-Identifier: MIT OR Apache-2.0
// Timing of index build and BM25 scoring: search.js and search.wasm under Node, on a corpus (one document per line),
// the same queries as examples/bench.rs. Best of 5.   node rust/bench.mjs corpus.txt [queries=200]
import { readFileSync } from "node:fs";
import * as js from "../search.js";
import { load } from "../search-wasm.js";
const [path, nq = "200"] = process.argv.slice(2);
const docs = readFileSync(path, "utf8").split("\n").filter((l, i, a) => i < a.length - 1 || l);
const step = Math.max(1, Math.floor(docs.length / +nq));
const queries = Array.from({ length: +nq }, (_, i) => docs[(i * step) % docs.length].split(" ").slice(3, 6).join(" "));
const w = await load();
for (const [name, BM25] of [["search.js", js.BM25], ["search.wasm", w.BM25]]) {
  let build = Infinity, q = Infinity, idx, check = 0;
  for (let r = 0; r < 5; r++) { idx?.free?.(); const t = performance.now(); idx = new BM25(docs); build = Math.min(build, performance.now() - t); }
  for (let r = 0; r < 5; r++) {
    const t = performance.now();
    for (const s of queries) for (const x of idx.scores(s)) check += x;
    q = Math.min(q, performance.now() - t);
  }
  console.log(`${name.padEnd(11)}: ${docs.length} docs; build ${build.toFixed(1)} ms; ${queries.length} queries ${q.toFixed(1)} ms (${(q * 1000 / queries.length).toFixed(1)} µs/query) [checksum ${(check / 5).toFixed(6)}]`);
}
