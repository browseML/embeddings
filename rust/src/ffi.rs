// SPDX-License-Identifier: MIT OR Apache-2.0
//! The C ABI the `.wasm` exports, for `search-wasm.js` (no wasm-bindgen: no dependency). Pointers are offsets into
//! the module's memory; the page copies text in (UTF-8) and reads numbers out. Each call is independent; nothing is
//! sent anywhere — the module imports nothing.
#![allow(clippy::missing_safety_doc)]

use crate::{cosine, cosine_f32, fuse, ln, Bm25};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::slice;

unsafe fn text<'a>(p: *const u8, n: usize) -> &'a str {
    if n == 0 {
        return "";
    }
    // SAFETY: the caller wrote n bytes of UTF-8 (TextEncoder) at p.
    unsafe { core::str::from_utf8_unchecked(slice::from_raw_parts(p, n)) }
}

fn layout(n: usize) -> core::alloc::Layout {
    // 8-byte aligned, so a block can hold f64s; never zero-sized
    core::alloc::Layout::from_size_align(n.max(1), 8).expect("layout")
}

/// n bytes of the module's memory, for the page to write into.
#[no_mangle]
pub extern "C" fn bml_alloc(n: usize) -> *mut u8 {
    // SAFETY: layout() is never zero-sized.
    unsafe { alloc::alloc::alloc(layout(n)) }
}

#[no_mangle]
pub unsafe extern "C" fn bml_free(p: *mut u8, n: usize) {
    // SAFETY: p came from bml_alloc(n).
    unsafe { alloc::alloc::dealloc(p, layout(n)) }
}

#[no_mangle]
pub extern "C" fn bml_bm25_new(k1: f64, b: f64) -> *mut Bm25 {
    Box::into_raw(Box::new(Bm25::with_params(k1, b)))
}

#[no_mangle]
pub unsafe extern "C" fn bml_bm25_free(idx: *mut Bm25) {
    // SAFETY: idx came from bml_bm25_new.
    drop(unsafe { Box::from_raw(idx) });
}

#[no_mangle]
pub unsafe extern "C" fn bml_bm25_add(idx: *mut Bm25, p: *const u8, n: usize) -> u32 {
    unsafe { (*idx).add(text(p, n)) as u32 }
}

#[no_mangle]
pub unsafe extern "C" fn bml_bm25_len(idx: *const Bm25) -> u32 {
    unsafe { (*idx).len() as u32 }
}

/// Writes each document's score (f64, in the order added) at `out`, which holds `bml_bm25_len` of them.
#[no_mangle]
pub unsafe extern "C" fn bml_bm25_scores(idx: *const Bm25, p: *const u8, n: usize, out: *mut f64) {
    let idx = unsafe { &*idx };
    let s = idx.scores(unsafe { text(p, n) });
    unsafe { core::ptr::copy_nonoverlapping(s.as_ptr(), out, s.len()) };
}

/// Writes up to `limit` document indices (u32), best first, at `out`; returns how many.
#[no_mangle]
pub unsafe extern "C" fn bml_bm25_rank(idx: *const Bm25, p: *const u8, n: usize, limit: u32, out: *mut u32) -> u32 {
    let idx = unsafe { &*idx };
    let r = idx.rank(unsafe { text(p, n) }, limit as usize);
    for (j, &i) in r.iter().enumerate() {
        unsafe { *out.add(j) = i as u32 };
    }
    r.len() as u32
}

/// Fuses `count` rankings laid end to end at `ids` (u32), the length of each at `lens` (u32). Writes the fused
/// indices (u32) at `out_ids` and scores (f64) at `out_scores` — room for Σ lens of each — and returns how many.
#[no_mangle]
pub unsafe extern "C" fn bml_fuse(
    ids: *const u32,
    lens: *const u32,
    count: u32,
    k: f64,
    out_ids: *mut u32,
    out_scores: *mut f64,
) -> u32 {
    let lens = unsafe { slice::from_raw_parts(lens, count as usize) };
    let mut rankings: Vec<Vec<usize>> = Vec::with_capacity(lens.len());
    let mut at = 0usize;
    for &l in lens {
        let r = unsafe { slice::from_raw_parts(ids.add(at), l as usize) };
        rankings.push(r.iter().map(|&i| i as usize).collect());
        at += l as usize;
    }
    let f = fuse(&rankings, k);
    for (j, &(i, s)) in f.iter().enumerate() {
        unsafe {
            *out_ids.add(j) = i as u32;
            *out_scores.add(j) = s;
        }
    }
    f.len() as u32
}

#[no_mangle]
pub unsafe extern "C" fn bml_cosine(a: *const f64, b: *const f64, n: usize) -> f64 {
    unsafe { cosine(slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) }
}

#[no_mangle]
pub unsafe extern "C" fn bml_cosine_f32(a: *const f32, b: *const f32, n: usize) -> f64 {
    unsafe { cosine_f32(slice::from_raw_parts(a, n), slice::from_raw_parts(b, n)) }
}

/// fdlibm's ln, exported so the page (and the test) can hold it to `Math.log`.
#[no_mangle]
pub extern "C" fn bml_ln(x: f64) -> f64 {
    ln(x)
}
