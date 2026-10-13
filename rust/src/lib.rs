// SPDX-License-Identifier: MIT OR Apache-2.0
//! browseML's search, in Rust: the same as `search.js`, number for number.
//!
//! - [`tokens`]: Savante's words — the text lowercased (Unicode, as JavaScript's `toLowerCase` and Python's `lower`),
//!   then the runs of `a-z0-9`.
//! - [`Bm25`]: BM25 with k1 = 1.2, b = 0.75 and Lucene's idf `ln(1 + (n − df + 0.5) / (df + 0.5))`, each query word
//!   counted once (as Savante's `set(toks(q))`). Every score is computed with the same operations in the same order
//!   as `search.js`, and the logarithm is fdlibm's — the one V8's `Math.log` uses — so a score here is the score
//!   there, bit for bit (measured by `search-wasm.test.mjs`, not assumed).
//! - [`fuse`]: reciprocal rank fusion (k = 60, Cormack, Clarke and Büttcher 2009), ties in first-seen order, as the
//!   JavaScript `Map` keeps them.
//! - [`cosine`] / [`cosine_f32`]: the dot product of two unit-length vectors, summed in order in f64.
//!
//! `no_std` + `alloc`; no dependency. The `std` feature (default) only supplies the allocator for the `.wasm`.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

mod ln;
pub use ln::ln;

#[cfg(feature = "std")]
pub mod ffi;

/// Calls `f` with each of Savante's words in `text`, as lowercase ASCII bytes.
pub fn for_each_token(text: &str, mut f: impl FnMut(&[u8])) {
    let mut word: Vec<u8> = Vec::new();
    for c in text.chars() {
        if c.is_ascii() {
            let l = c.to_ascii_lowercase() as u8;
            if l.is_ascii_lowercase() || l.is_ascii_digit() {
                word.push(l);
            } else if !word.is_empty() {
                f(&word);
                word.clear();
            }
        } else {
            // Unicode lowercasing can yield ASCII (K KELVIN SIGN → k; İ → i + U+0307): keep what JavaScript keeps.
            for l in c.to_lowercase() {
                if l.is_ascii_lowercase() || l.is_ascii_digit() {
                    word.push(l as u8);
                } else if !word.is_empty() {
                    f(&word);
                    word.clear();
                }
            }
        }
    }
    if !word.is_empty() {
        f(&word);
    }
}

/// Savante's words: lowercase runs of letters and digits.
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    // every byte pushed is ASCII, so the bytes are UTF-8
    for_each_token(text, |w| out.push(String::from_utf8(w.to_vec()).unwrap_or_default()));
    out
}

/// Words to ids: an open-addressing table over one byte arena (FNV-1a, linear probing).
#[derive(Default, Clone)]
struct Vocab {
    arena: Vec<u8>,
    spans: Vec<(u32, u32)>, // id -> (start, len) in arena
    table: Vec<u32>,        // slot -> id + 1, 0 = empty
}

impl Vocab {
    fn hash(w: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in w {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    fn word(&self, id: u32) -> &[u8] {
        let (s, l) = self.spans[id as usize];
        &self.arena[s as usize..(s + l) as usize]
    }
    fn get(&self, w: &[u8]) -> Option<u32> {
        if self.table.is_empty() {
            return None;
        }
        let mask = self.table.len() - 1;
        let mut i = Self::hash(w) as usize & mask;
        loop {
            match self.table[i] {
                0 => return None,
                e if self.word(e - 1) == w => return Some(e - 1),
                _ => i = (i + 1) & mask,
            }
        }
    }
    fn intern(&mut self, w: &[u8]) -> u32 {
        if (self.spans.len() + 1) * 2 > self.table.len() {
            self.grow();
        }
        let mask = self.table.len() - 1;
        let mut i = Self::hash(w) as usize & mask;
        loop {
            match self.table[i] {
                0 => {
                    let id = self.spans.len() as u32;
                    self.spans.push((self.arena.len() as u32, w.len() as u32));
                    self.arena.extend_from_slice(w);
                    self.table[i] = id + 1;
                    return id;
                }
                e if self.word(e - 1) == w => return e - 1,
                _ => i = (i + 1) & mask,
            }
        }
    }
    fn grow(&mut self) {
        let cap = (self.table.len() * 2).max(64);
        self.table = alloc::vec![0; cap];
        let mask = cap - 1;
        for id in 0..self.spans.len() as u32 {
            let mut i = Self::hash(self.word(id)) as usize & mask;
            while self.table[i] != 0 {
                i = (i + 1) & mask;
            }
            self.table[i] = id + 1;
        }
    }
}

/// A BM25 index over texts — `new BM25(texts, { k1, b })` in `search.js`.
#[derive(Clone)]
pub struct Bm25 {
    k1: f64,
    b: f64,
    vocab: Vocab,
    postings: Vec<Vec<(u32, u32)>>, // word id -> (document, term frequency), documents ascending
    lens: Vec<u32>,
    total: u64,
    scratch: Vec<u32>,
}

impl Default for Bm25 {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25 {
    /// k1 = 1.2, b = 0.75: Savante's.
    pub fn new() -> Self {
        Self::with_params(1.2, 0.75)
    }
    pub fn with_params(k1: f64, b: f64) -> Self {
        Bm25 { k1, b, vocab: Vocab::default(), postings: Vec::new(), lens: Vec::new(), total: 0, scratch: Vec::new() }
    }
    pub fn from_texts<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        let mut idx = Self::new();
        for t in texts {
            idx.add(t);
        }
        idx
    }
    /// Adds a document; returns its index.
    pub fn add(&mut self, text: &str) -> usize {
        let mut ids = core::mem::take(&mut self.scratch);
        ids.clear();
        let (vocab, postings) = (&mut self.vocab, &mut self.postings);
        for_each_token(text, |w| {
            let id = vocab.intern(w);
            if id as usize == postings.len() {
                postings.push(Vec::new());
            }
            ids.push(id);
        });
        let doc = self.lens.len() as u32;
        self.lens.push(ids.len() as u32);
        self.total += ids.len() as u64;
        ids.sort_unstable();
        let mut i = 0;
        while i < ids.len() {
            let mut j = i + 1;
            while j < ids.len() && ids[j] == ids[i] {
                j += 1;
            }
            self.postings[ids[i] as usize].push((doc, (j - i) as u32));
            i = j;
        }
        self.scratch = ids;
        doc as usize
    }
    pub fn len(&self) -> usize {
        self.lens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lens.is_empty()
    }
    /// The query's words found in the index, each once, in first-seen order.
    fn query_ids(&self, query: &str) -> Vec<u32> {
        let mut q: Vec<u32> = Vec::new();
        for_each_token(query, |w| {
            if let Some(id) = self.vocab.get(w) {
                if !q.contains(&id) {
                    q.push(id);
                }
            }
        });
        q
    }
    /// Each document's score, in the order added, written into `out` (resized to the number of documents).
    pub fn scores_into(&self, query: &str, out: &mut Vec<f64>) {
        let n = self.lens.len();
        out.clear();
        out.resize(n, 0.0);
        // search.js: avg = n ? Σ len / n : 1 (the sum is exact: integers below 2^53)
        let avg = if n > 0 { self.total as f64 / n as f64 } else { 1.0 };
        let (k1, b) = (self.k1, self.b);
        for id in self.query_ids(query) {
            let p = &self.postings[id as usize];
            let df = p.len() as f64;
            let idf = ln(1.0 + (n as f64 - df + 0.5) / (df + 0.5));
            for &(d, f) in p {
                let f = f as f64;
                let len = self.lens[d as usize] as f64;
                // idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len / avg)), evaluated as JavaScript evaluates it
                out[d as usize] += idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * len / avg));
            }
        }
    }
    pub fn scores(&self, query: &str) -> Vec<f64> {
        let mut out = Vec::new();
        self.scores_into(query, &mut out);
        out
    }
    /// Document indices, best first, only those that match; ties in the order added.
    pub fn rank(&self, query: &str, limit: usize) -> Vec<usize> {
        let mut s: Vec<(usize, f64)> = self.scores(query).into_iter().enumerate().filter(|&(_, s)| s > 0.0).collect();
        s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal)); // stable, as V8's sort
        s.truncate(limit);
        s.into_iter().map(|(i, _)| i).collect()
    }
}

/// Reciprocal rank fusion of rankings (document indices, best first): score = Σ 1 / (k + rank), best first, ties in
/// the order first seen — `fuse(rankings, k)` in `search.js`.
pub fn fuse<R: AsRef<[usize]>>(rankings: &[R], k: f64) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64)> = Vec::new();
    let mut at: BTreeMap<usize, usize> = BTreeMap::new();
    for r in rankings {
        for (pos, &i) in r.as_ref().iter().enumerate() {
            let add = 1.0 / (k + pos as f64 + 1.0);
            match at.get(&i) {
                Some(&j) => out[j].1 += add,
                None => {
                    at.insert(i, out.len());
                    out.push((i, 0.0 + add));
                }
            }
        }
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    out
}

/// Cosine similarity of two unit-length vectors: their dot product, summed in order in f64 (as `search.js`).
pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |s, (x, y)| s + x * y)
}

/// The same for f32 vectors (an embedding model's usual output), widened to f64 as a `Float32Array` is in
/// JavaScript — so the answer equals `cosine(a, b)` over the same `Float32Array`s in `search.js`.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |s, (&x, &y)| s + x as f64 * y as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn words() {
        assert_eq!(tokens("Bonsai-1.7B, 248 MB!"), ["bonsai", "1", "7b", "248", "mb"]);
        assert_eq!(tokens("İSTANBUL \u{212A}elvin café"), ["i", "stanbul", "kelvin", "caf"]);
        assert!(tokens("").is_empty());
    }
    #[test]
    fn vocab_grows() {
        let mut v = Vocab::default();
        let words: Vec<String> = (0..5000).map(|i| alloc::format!("w{i}")).collect();
        for (i, w) in words.iter().enumerate() {
            assert_eq!(v.intern(w.as_bytes()), i as u32);
        }
        for (i, w) in words.iter().enumerate() {
            assert_eq!(v.get(w.as_bytes()), Some(i as u32));
        }
        assert_eq!(v.get(b"absent"), None);
    }
    #[test]
    fn empty_index() {
        let idx = Bm25::new();
        assert!(idx.scores("anything").is_empty());
        assert!(idx.rank("anything", 10).is_empty());
    }
    #[test]
    fn fusion() {
        let f = fuse(&[vec![2, 0, 1], vec![0, 3]], 60.0);
        assert_eq!(f[0].0, 0);
        assert!((f[0].1 - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-15);
        assert_eq!(f.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 2, 3, 1]); // 2 and 3 tie at 1/61: first seen first
    }
    #[test]
    fn cos() {
        assert_eq!(cosine(&[0.6, 0.8], &[0.6, 0.8]), 0.6 * 0.6 + 0.8 * 0.8);
        assert_eq!(cosine_f32(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }
}
