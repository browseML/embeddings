<div align="center">

<img src="logo.png" width="140" alt="browseML">

### browseML's search — words now, meaning next, all in the browser

</div>

Finding things in what browseML knows — its conversations, its notes, its documentation — on the visitor's own
computer. It is bankML's Savante search, brought to the browser: **BM25** over words, and **reciprocal rank fusion**
to combine it with a ranking by meaning when an embedding model runs here.

<table>
<tr>
<td width="33%" valign="top">

**Savante's numbers**

BM25 with k1 = 1.2, b = 0.75 and Lucene's idf, over Savante's words, each query word counted once: the same scores as Savante's formula, to 1e-12.

</td>
<td width="33%" valign="top">

**Fusion ready**

Reciprocal rank fusion (k = 60), as Savante fuses BM25 with bge-m3: give it any second ranking.

</td>
<td width="33%" valign="top">

**On the device**

No index leaves the browser. No dependency — in JavaScript, or in Rust compiled to WebAssembly.

</td>
</tr>
</table>

## Use

```js
import { BM25, fuse, cosine } from "./search.js";
const idx = new BM25(texts);
const words = idx.rank("does my typing leave the device");      // indices, best first
const both = fuse([words, meaningRanking]);                      // [[index, score], …]
```

```sh
node --test search.test.mjs      # tokens, BM25 against Savante's formula (expected.json, computed in Python), ranking, fusion
```

The same API from WebAssembly (the Rust crate below, `search.wasm`, 58 KiB, imports nothing):

```js
import { load } from "./search-wasm.js";
const { BM25, fuse, cosine } = await load();                    // fetches ./search.wasm beside it
const idx = new BM25(texts);
idx.rank("does my typing leave the device");                    // the same indices as search.js, the same scores bit for bit
idx.free();                                                     // or leave it to the garbage collector
```

## Rust

`rust/` is the same search as a Rust crate: Savante's words, BM25, reciprocal rank fusion and cosine. No dependency,
`no_std` + `alloc` (the `std` feature only gives the `.wasm` its allocator), native and WebAssembly —
`wasm32-unknown-unknown` (what `search-wasm.js` loads) and `wasm32-wasip1` (browseML's target), both with SIMD128 as
browseML builds. The toolchain is browseML's, pinned in `rust/rust-toolchain.toml` (1.99.0, with both wasm targets).

```sh
rust/build.sh                    # cargo test, the no_std build, both wasm builds, search.wasm, then node --test
cd rust && cargo test --release  # expected.json to 1e-12, the words, fusion, the logarithm
```

**Exactness first.** Every score is computed with the operations `search.js` uses, in the order it uses them, and
the logarithm is fdlibm's (`rust/src/ln.rs`) — the routine V8 ports for `Math.log` — on every target, `no_std`
included, rather than whichever libm the platform links. What that buys, measured:

| check | result |
|---|---|
| `cargo test` against `expected.json` (Savante's formula, computed in Python) | 15 of 15 scores, max \|error\| **0** (bit-equal) |
| `rust/src/ln.rs` against V8's `Math.log` (`search-wasm.test.mjs`) | 600,000 inputs, **0** differ |
| `search.wasm` against `search.js`: 2,000 generated documents (Unicode included: İ, K, ﬁ, Σ), 205 queries | 410,000 scores, **0** differ bitwise; every top-25 ranking identical |
| fusion and cosine (f32 and f64 vectors, 384–1,024 dims), 200 cases each | identical to `search.js` |
| the same against the `wasm32-wasip1` build | the same |
| `rust/src/ln.rs` against the platform's `f64::ln` (x86-64 glibc), 1,000,000 doubles | never more than 1 ulp apart (fdlibm and glibc round differently, rarely) |

Ties are broken as `search.js` breaks them — the order documents were added, and the order fused documents were first
seen. Savante sorts its `(score, source, text)` tuples in reverse, so where two scores are exactly equal Savante puts
the later document first; only the order of exact ties differs.

**Speed, measured** on a modest corpus — 12,082 paragraphs (541,184 words, 4.4 MiB) of mindX's documentation, one
line each, 200 three-word queries, every document scored for every query, best of 5, `nice`d on a loaded 4-core laptop
(load average 4.7–5.7, so read these as ranges; `rust/bench.mjs` and `rust/examples/bench.rs`):

| | build the index | score all documents, per query |
|---|---|---|
| Rust, native | 61–76 ms | 23–52 µs |
| `search.wasm` in Node 24 | 105–114 ms | 0.92–1.08 ms |
| `search.js` in Node 24 | 304–568 ms | 9.2–26.7 ms |

All three return the same checksum over every score. The gap is mostly the algorithm, not the language: the crate
keeps an inverted index (each query word visits only the documents that contain it), while `search.js` walks every
document for every query word; and most of the wasm time per query is copying 12,082 scores out into a JavaScript
array. `rank()`, which returns only the top indices, copies little. No browser has been timed yet.

**Fixed while porting:** `search.js` counted a repeated query word once per repetition ("browser browser cpu" scored
`browser` twice), where Savante iterates `set(toks(q))`. It now counts each word once, as Savante does, and both test
suites check it.

## Meaning next

**What the server side holds.** mindX embeds with **bge-m3, 1,024 dimensions**, through the local Ollama
(`/api/embed`, inputs cut at 4,000 characters, chunks of 512 words with no overlap), into PostgreSQL `vector(1024)`
columns — `doc_embeddings` (about 48,000 chunks) and `memories` — searched by cosine distance (`<=>`) through a
pgvectorscale **StreamingDiskANN** (`diskann`) index, HNSW where vectorscale is missing. Documentation search there is
vector-only; its event catalogue is hybrid, PostgreSQL full-text rank and cosine fused by reciprocal rank with
k = 60, as here. bankML's Savante uses the same model and width (`bankml_exchanges.embedding vector(1024)`). On CPU a
bge-m3 embedding costs tens of seconds per 512-word chunk on that two-core server (mindX's own comment says ~57 s).

**The mismatch problem.** A vector is only comparable with vectors from the *same model* — not merely the same width.
mxbai-embed-large is also 1,024 wide and lives in a different space; mindX's own notes record a time when bge-m3 was
missing on the server and mxbai stood in. So width checks are not enough: every vector should carry the model's name,
its digest and its dimension (as Savante's `.emb` cache already records the digest), and a search should refuse to
compare vectors whose provenance differs.

**What fits a page.** bge-m3 is a 568M-parameter XLM-RoBERTa (1.16 GB of F16 weights): browseML already loads a
248 MB model, so a 4-bit bge-m3 is conceivable as an opt-in, not a default. The models that fit a page are 384-wide:
all-MiniLM-L6-v2 (22.7M parameters, ~23 MB at int8) and bge-small-en-v1.5 (33M, ~33 MB at int8), run by candle compiled
to wasm32 (Rust, in browseML's idiom) or ONNX Runtime Web. Their vectors cannot query a bge-m3 index, and a learned
384→1,024 projection would be a lossy guess presented as an answer.

**The design, in three steps:**

1. **Fuse rankings, not vectors.** In the browser: BM25 here, plus a small 384-d model over the page's own `.history`
   and docs, cosine-ranked here (`cosine_f32`; a few thousand vectors is a brute-force scan, no index needed). Against
   the server: send the query *text* to mindX, which embeds it with bge-m3 and searches its DiskANN index, and returns
   a ranking. Reciprocal rank fusion combines rankings, never scores, so rankings from different spaces fuse honestly —
   it is why the fusion here takes "any second ranking".
2. **A second, small column on the server, if the browser should query it by vector.** mindX's model registry already
   lists all-minilm (384-d) as needing a `vector(384)` migration. A parallel `embedding_small vector(384)` with its own
   `diskann` index, filled by the *same* model and weights the page runs, lets a browser vector (no text sent) query
   the server. Costs to measure before promising: re-embedding ~48,000 chunks on the server's CPU, and the index
   size; and the same weights must produce the same vectors in candle-wasm and on the server to within a stated
   cosine tolerance — measured, as everything here is.
3. **bge-m3 in the browser, opt-in, for the full space.** Then a browser vector queries the 1,024-d index directly.
   Quantized weights will not reproduce the server's F16 vectors bit for bit; the agreement (cosine between the two
   vectors of the same text) must be measured and stated beside the digest.

Until a model runs here, search is BM25 alone, and says so.

- **The history and the docs:** browseML's `.history` and its own documentation searched from the field.

Used by [browseML](https://github.com/browseML/browseML). Licence: `MIT OR Apache-2.0`.
