//! Deterministic transcendental functions for building filter tables (Gaussian weights,
//! gamma and exposure curves, rotation tables).
//!
//! Platform `exp`, `ln`, `powf`, `sin` and `cos` are not correctly rounded and may
//! differ between the native libm and the wasm one; these use only IEEE basic
//! operations (`+ − × ÷`, `floor`) and bit manipulation, in a fixed order, so every
//! target computes the same bits. Accuracy is about 1e-15 relative, far more than an
//! 8-bit table needs. They run once per table entry, never per pixel.

const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-01;
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
const LOG2E: f64 = 1.442_695_040_888_963_407_36;
const PIO2_HI: f64 = 1.570_796_326_734_125_614_17;
const PIO2_LO: f64 = 6.077_100_506_506_192_249_30e-11;

pub const PI: f64 = std::f64::consts::PI;

/// `2^k` for an integer `k` (saturating to 0 or infinity outside the normal range).
fn exp2i(k: i64) -> f64 {
    if k > 1023 {
        f64::INFINITY
    } else if k < -1022 {
        0.0
    } else {
        f64::from_bits(((k + 1023) as u64) << 52)
    }
}

/// `e^x`.
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    if x < -745.0 {
        return 0.0;
    }
    let k = (x * LOG2E + 0.5).floor();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    // Taylor series to r^14 / 14! (|r| <= 0.35: the remainder is below 1e-17).
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..=14 {
        term = term * r / n as f64;
        sum += term;
    }
    let k = k as i64;
    // Split the scale so results near the subnormal edge don't underflow early.
    if k < -1000 {
        sum * exp2i(k + 60) * exp2i(-60)
    } else {
        sum * exp2i(k)
    }
}

/// Natural logarithm (`NaN` for negative input, `-inf` at 0).
pub fn ln(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    let (mut x, mut e) = (x, 0i64);
    if x < f64::MIN_POSITIVE {
        x *= exp2i(54);
        e -= 54;
    }
    let bits = x.to_bits();
    e += ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    // ln m = 2 atanh(s), s = (m − 1) / (m + 1), |s| <= 0.1716.
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = s;
    for n in 1..=12 {
        term *= s2;
        sum += term / (2 * n + 1) as f64;
    }
    e as f64 * LN2_HI + (e as f64 * LN2_LO + 2.0 * sum)
}

/// `x^y` for `x >= 0` (0 for `x == 0` and `y > 0`, 1 for `y == 0`).
pub fn pow(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return if y > 0.0 { 0.0 } else { f64::INFINITY };
    }
    if x == 1.0 {
        return 1.0;
    }
    exp(y * ln(x))
}

/// `2^x`.
pub fn exp2(x: f64) -> f64 {
    exp(x * std::f64::consts::LN_2)
}

/// `(sin x, cos x)`.
pub fn sin_cos(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let k = (x / (PI / 2.0) + 0.5).floor();
    let r = (x - k * PIO2_HI) - k * PIO2_LO;
    let r2 = r * r;
    // Taylor series on |r| <= π/4, to r^19 (remainder below 1e-17).
    let (mut s, mut st) = (r, r);
    let (mut c, mut ct) = (1.0, 1.0);
    for n in 1..=9 {
        st = -st * r2 / ((2 * n) * (2 * n + 1)) as f64;
        s += st;
        ct = -ct * r2 / ((2 * n - 1) * (2 * n)) as f64;
        c += ct;
    }
    match (k as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// Degrees to radians.
pub fn radians(deg: f64) -> f64 {
    deg * (PI / 180.0)
}

/// sRGB-encoded 0..1 to linear light.
pub fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        pow((v + 0.055) / 1.055, 2.4)
    }
}

/// Linear light to sRGB-encoded 0..1.
pub fn linear_to_srgb(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * pow(v, 1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    // The platform functions are fine as an oracle here: tests only run natively.
    #[test]
    fn matches_libm_closely() {
        let mut x = -30.0;
        while x < 30.0 {
            assert!(close(exp(x), f64::exp(x), 1e-14), "exp({x})");
            let (s, c) = sin_cos(x);
            assert!((s - f64::sin(x)).abs() < 1e-14, "sin({x})");
            assert!((c - f64::cos(x)).abs() < 1e-14, "cos({x})");
            x += 0.0137;
        }
        let mut x = 1e-6;
        while x < 1e6 {
            assert!(close(ln(x), f64::ln(x), 1e-14), "ln({x})");
            assert!(
                close(pow(x, 0.4545), f64::powf(x, 0.4545), 1e-13),
                "pow({x})"
            );
            x *= 1.37;
        }
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(1.0), 0.0);
        assert!(close(exp2(3.0), 8.0, 1e-15));
        for v in 0..=255 {
            let e = v as f64 / 255.0;
            assert!((linear_to_srgb(srgb_to_linear(e)) - e).abs() < 1e-12);
        }
    }
}
