// SPDX-License-Identifier: MIT OR Apache-2.0
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { BM25, fuse, tokens } from "./search.js";
const ex = JSON.parse(readFileSync(new URL("./expected.json", import.meta.url), "utf8"));
test("Savante's words", () => assert.deepEqual(tokens("Bonsai-1.7B, 248 MB!"), ["bonsai", "1", "7b", "248", "mb"]));
test("BM25 scores are Savante's, number for number (its formula, computed in Python)", () => {
  const idx = new BM25(ex.docs);
  for (const [q, want] of Object.entries(ex.queries)) {
    const got = idx.scores(q);
    got.forEach((s, i) => assert.ok(Math.abs(s - want[i]) < 1e-12, `${q} doc ${i}: ${s} vs ${want[i]}`));
  }
});
test("a repeated query word counts once, as in Savante (set(toks(q)))", () => {
  const idx = new BM25(ex.docs), want = ex.queries["browser cpu"];
  idx.scores("browser BROWSER cpu").forEach((s, i) => assert.ok(Math.abs(s - want[i]) < 1e-12, `doc ${i}: ${s} vs ${want[i]}`));
});
test("the right document first", () => {
  const idx = new BM25(ex.docs);
  assert.equal(idx.rank("does my typing leave the device")[0], 4);
  assert.equal(idx.rank("Bonsai model size in MB")[0], 2);   // "which model and how big" matches only "model": the shorter document wins, as BM25 should
});
test("reciprocal rank fusion, as Savante's fuse()", () => {
  const f = fuse([[2, 0, 1], [0, 3]]);
  assert.equal(f[0][0], 0);                      // ranked well in both
  assert.ok(Math.abs(f[0][1] - (1 / 62 + 1 / 61)) < 1e-15);
});
