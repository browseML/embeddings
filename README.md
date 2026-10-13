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

BM25 with k1 = 1.2, b = 0.75 and Lucene's idf, over Savante's words: the same scores as Savante's formula, to 1e-12.

</td>
<td width="33%" valign="top">

**Fusion ready**

Reciprocal rank fusion (k = 60), as Savante fuses BM25 with bge-m3: give it any second ranking.

</td>
<td width="33%" valign="top">

**On the device**

No index leaves the browser. No dependency.

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

## Next

- **Meaning, in the browser:** a small embedding model on WebGPU or WebAssembly (bankML embeds with bge-m3 through
  Ollama natively; in the browser, a model that fits a page), its vectors fused with BM25 here.
- **The history and the docs:** browseML's `.history` and its own documentation searched from the field.

Used by [browseML](https://github.com/browseML/browseML). Licence: `MIT OR Apache-2.0`.
