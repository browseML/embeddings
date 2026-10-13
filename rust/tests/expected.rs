// SPDX-License-Identifier: MIT OR Apache-2.0
// The same checks as search.test.mjs, against the same expected.json (BM25 by Savante's formula, computed in Python).
use browseml_embeddings::{fuse, tokens, Bm25};

// A JSON reader for expected.json's shape (objects, arrays, strings, numbers) — no dependency.
#[derive(Debug)]
enum J {
    O(Vec<(String, J)>),
    A(Vec<J>),
    S(String),
    N(f64),
}
struct P<'a>(&'a [u8], usize);
impl P<'_> {
    fn ws(&mut self) {
        while self.1 < self.0.len() && self.0[self.1].is_ascii_whitespace() {
            self.1 += 1;
        }
    }
    fn val(&mut self) -> J {
        self.ws();
        match self.0[self.1] {
            b'{' => {
                self.1 += 1;
                let mut o = Vec::new();
                loop {
                    self.ws();
                    if self.0[self.1] == b'}' {
                        self.1 += 1;
                        return J::O(o);
                    }
                    let J::S(k) = self.val() else { panic!("key") };
                    self.ws();
                    assert_eq!(self.0[self.1], b':');
                    self.1 += 1;
                    o.push((k, self.val()));
                    self.ws();
                    if self.0[self.1] == b',' {
                        self.1 += 1;
                    }
                }
            }
            b'[' => {
                self.1 += 1;
                let mut a = Vec::new();
                loop {
                    self.ws();
                    if self.0[self.1] == b']' {
                        self.1 += 1;
                        return J::A(a);
                    }
                    a.push(self.val());
                    self.ws();
                    if self.0[self.1] == b',' {
                        self.1 += 1;
                    }
                }
            }
            b'"' => {
                self.1 += 1;
                let mut s = Vec::new();
                while self.0[self.1] != b'"' {
                    assert_ne!(self.0[self.1], b'\\', "escapes not needed by expected.json");
                    s.push(self.0[self.1]);
                    self.1 += 1;
                }
                self.1 += 1;
                J::S(String::from_utf8(s).unwrap())
            }
            _ => {
                let st = self.1;
                while self.1 < self.0.len() && b"+-.eE0123456789".contains(&self.0[self.1]) {
                    self.1 += 1;
                }
                J::N(std::str::from_utf8(&self.0[st..self.1]).unwrap().parse().unwrap())
            }
        }
    }
}

fn expected() -> (Vec<String>, Vec<(String, Vec<f64>)>) {
    let src = include_str!("../../expected.json");
    let J::O(top) = P(src.as_bytes(), 0).val() else { panic!() };
    let mut docs = Vec::new();
    let mut queries = Vec::new();
    for (k, v) in top {
        match (k.as_str(), v) {
            ("docs", J::A(a)) => docs = a.into_iter().map(|d| if let J::S(s) = d { s } else { panic!() }).collect(),
            ("queries", J::O(q)) => {
                queries = q
                    .into_iter()
                    .map(|(q, v)| {
                        let J::A(a) = v else { panic!() };
                        (q, a.into_iter().map(|n| if let J::N(x) = n { x } else { panic!() }).collect())
                    })
                    .collect()
            }
            _ => panic!("unexpected key {k}"),
        }
    }
    (docs, queries)
}

#[test]
fn savantes_words() {
    assert_eq!(tokens("Bonsai-1.7B, 248 MB!"), ["bonsai", "1", "7b", "248", "mb"]);
}

#[test]
fn bm25_scores_are_savantes_number_for_number() {
    let (docs, queries) = expected();
    assert_eq!(docs.len(), 5);
    assert_eq!(queries.len(), 3);
    let idx = Bm25::from_texts(docs.iter().map(|s| s.as_str()));
    let (mut checked, mut worst) = (0, 0.0f64);
    for (q, want) in &queries {
        let got = idx.scores(q);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let e = (g - w).abs();
            assert!(e < 1e-12, "{q} doc {i}: {g} vs {w}");
            worst = worst.max(e);
            checked += 1;
        }
    }
    println!("{checked} scores checked against expected.json, max |error| = {worst:e}");
}

#[test]
fn a_repeated_query_word_counts_once_as_in_savante() {
    // Savante iterates set(toks(q)): "browser browser cpu" scores as "browser cpu".
    let (docs, queries) = expected();
    let idx = Bm25::from_texts(docs.iter().map(|s| s.as_str()));
    let want = &queries.iter().find(|(q, _)| q == "browser cpu").unwrap().1;
    for (g, w) in idx.scores("browser BROWSER cpu").iter().zip(want) {
        assert!((g - w).abs() < 1e-12);
    }
}

#[test]
fn the_right_document_first() {
    let (docs, _) = expected();
    let idx = Bm25::from_texts(docs.iter().map(|s| s.as_str()));
    assert_eq!(idx.rank("does my typing leave the device", 10)[0], 4);
    assert_eq!(idx.rank("Bonsai model size in MB", 10)[0], 2);
}

#[test]
fn reciprocal_rank_fusion_as_savantes_fuse() {
    let f = fuse(&[vec![2usize, 0, 1], vec![0, 3]], 60.0);
    assert_eq!(f[0].0, 0);
    assert!((f[0].1 - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-15);
}
