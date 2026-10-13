#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# Build the Rust search to WebAssembly beside search.js: search.wasm (wasm32-unknown-unknown, no imports, SIMD) —
# and check the wasm32-wasip1 build too (browseML's target). Then the tests: Rust's against expected.json, the
# no_std build, and Node's against search.js.
#   rust/build.sh           build + test
set -eu
cd "$(dirname "$0")"
cargo test --release
cargo rustc --release --lib --no-default-features --crate-type rlib --target wasm32-unknown-unknown   # no_std + alloc
cargo build --release --target wasm32-unknown-unknown
cargo build --release --target wasm32-wasip1
cp target/wasm32-unknown-unknown/release/browseml_embeddings.wasm ../search.wasm
cd .. && node --test search.test.mjs search-wasm.test.mjs
