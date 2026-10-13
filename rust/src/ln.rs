// SPDX-License-Identifier: MIT OR Apache-2.0
//! The natural logarithm as fdlibm computes it (Sun Microsystems, 1993: e_log.c, `__ieee754_log`), the routine V8
//! ports for `Math.log` (src/base/ieee754.cc). One logarithm on every target — native, wasm32, no_std — so an idf
//! here is the idf in the browser's JavaScript, bit for bit; `search-wasm.test.mjs` measures that against V8.
//!
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================

#![allow(clippy::excessive_precision)] // the constants as fdlibm prints them

const LN2_HI: f64 = 6.93147180369123816490e-01; // 3fe62e42 fee00000
const LN2_LO: f64 = 1.90821492927058770002e-10; // 3dea39ef 35793c76
const TWO54: f64 = 1.80143985094819840000e+16; // 43500000 00000000
const LG1: f64 = 6.666666666666735130e-01; // 3FE55555 55555593
const LG2: f64 = 3.999999999940941908e-01; // 3FD99999 9997FA04
const LG3: f64 = 2.857142874366239149e-01; // 3FD24924 94229359
const LG4: f64 = 2.222219843214978396e-01; // 3FCC71C5 1D8E78AF
const LG5: f64 = 1.818357216161805012e-01; // 3FC74664 96CB03DE
const LG6: f64 = 1.531383769920937332e-01; // 3FC39A09 D078C69F
const LG7: f64 = 1.479819860511658591e-01; // 3FC2F112 DF3E5244

#[inline]
fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as u32 as i32
}
#[inline]
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}
#[inline]
fn with_hi(x: f64, h: i32) -> f64 {
    f64::from_bits(((h as u32 as u64) << 32) | lo(x) as u64)
}

/// ln(x), fdlibm's algorithm and operation order.
pub fn ln(mut x: f64) -> f64 {
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
    if hx < 0x0010_0000 {
        // x < 2^-1022
        if ((hx & 0x7fff_ffff) as u32 | lx) == 0 {
            return f64::NEG_INFINITY; // ln(±0)
        }
        if hx < 0 {
            return f64::NAN; // ln(negative)
        }
        k -= 54;
        x *= TWO54; // subnormal: scale up
        hx = hi(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    let i = (hx + 0x95f64) & 0x10_0000;
    x = with_hi(x, hx | (i ^ 0x3ff0_0000)); // normalize x or x/2
    k += i >> 20;
    let f = x - 1.0;
    if (0x000f_ffff & (2 + hx)) < 3 {
        // -2^-20 <= f < 2^-20
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = k as f64;
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.333_333_333_333_333_33 * f);
        if k == 0 {
            return f - r;
        }
        let dk = k as f64;
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let mut i = hx - 0x6147a;
    let w = z * z;
    let j = 0x6b851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (hfsq - s * (hfsq + r))
        } else {
            dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        f - s * (f - r)
    } else {
        dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::ln;
    #[test]
    fn specials() {
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert_eq!(ln(f64::INFINITY), f64::INFINITY);
        assert!(ln(f64::NAN).is_nan());
    }
    #[test]
    fn within_one_ulp_of_the_platform() {
        // fdlibm is accurate to < 1 ulp; the platform's libm may round the other way. Never further than one ulp.
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut worst = 0u64;
        for _ in 0..1_000_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f64::from_bits((s >> 2) | 0x0010_0000_0000_0000) ; // positive, normal
            let (a, b) = (ln(x).to_bits() as i64, x.ln().to_bits() as i64);
            worst = worst.max(a.abs_diff(b));
        }
        assert!(worst <= 1, "worst {worst} ulp");
        for x in [0.5, 2.0, 10.0, 1e-300, 5e-324, 1.0 + 1e-9, 1.7976931348623157e308] {
            assert!((ln(x).to_bits() as i64).abs_diff(x.ln().to_bits() as i64) <= 1, "{x}");
        }
    }
}
