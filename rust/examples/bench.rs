// SPDX-License-Identifier: MIT OR Apache-2.0
// Native timing of index build and BM25 scoring over a corpus (one document per line), for comparison with
// rust/bench.mjs (search.js and search.wasm under Node) on the same file.
//   cargo run --release --example bench -- corpus.txt [queries=200]
use browseml_embeddings::Bm25;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("corpus.txt (one document per line)");
    let nq: usize = std::env::args().nth(2).map_or(200, |s| s.parse().unwrap());
    let text = std::fs::read_to_string(path).unwrap();
    let docs: Vec<&str> = text.lines().collect();
    let queries = queries(&docs, nq);
    let mut best_build = f64::MAX;
    let mut idx = Bm25::new();
    for _ in 0..5 {
        let t = Instant::now();
        idx = Bm25::from_texts(docs.iter().copied());
        best_build = best_build.min(t.elapsed().as_secs_f64());
    }
    let mut out = Vec::new();
    let mut best_q = f64::MAX;
    let mut check = 0.0;
    for _ in 0..5 {
        let t = Instant::now();
        for q in &queries {
            idx.scores_into(q, &mut out);
            check += out.iter().sum::<f64>();
        }
        best_q = best_q.min(t.elapsed().as_secs_f64());
    }
    println!("rust native: {} docs; build {:.1} ms; {} queries {:.1} ms ({:.1} µs/query) [checksum {:.6}]",
        docs.len(), best_build * 1e3, queries.len(), best_q * 1e3, best_q * 1e6 / queries.len() as f64, check / 5.0);
}

// the same queries as bench.mjs: words 3..6 of every (len/n)-th document
fn queries(docs: &[&str], n: usize) -> Vec<String> {
    let step = (docs.len() / n).max(1);
    (0..n).map(|i| docs[(i * step) % docs.len()].split(' ').skip(3).take(3).collect::<Vec<_>>().join(" ")).collect()
}
