//! JS number-semantics helpers shared by the parser/measurer ports.
//!
//! These mirror the exact JS coercions the TS sources rely on:
//! `parseFloat`, `parseInt(s, 10)`, and the `formatNumber` used by
//! `utils/richtext/typographyNormalizer.ts:42-45`.

/// JS `parseFloat(s)`: skip leading whitespace, parse the longest valid
/// decimal-literal prefix (sign, digits, '.', exponent). Returns None for NaN.
pub fn js_parse_float(s: &str) -> Option<f64> {
    let t = s.trim_start_matches(|c: char| c.is_whitespace());
    let bytes = t.as_bytes();
    let mut i = 0usize;
    let mut seen_digit = false;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        seen_digit = true;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            seen_digit = true;
        }
    }
    if !seen_digit {
        // JS also accepts "Infinity"; none of our inputs use it.
        return None;
    }
    // optional exponent
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let mut exp_digit = false;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
            exp_digit = true;
        }
        if exp_digit {
            i = j;
        }
    }
    t[..i].parse::<f64>().ok()
}

/// JS `parseInt(s, 10)`: skip whitespace, optional sign, digits prefix.
pub fn js_parse_int(s: &str) -> Option<i64> {
    let t = s.trim_start_matches(|c: char| c.is_whitespace());
    let bytes = t.as_bytes();
    let mut i = 0usize;
    let mut sign = 1i64;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        if bytes[i] == b'-' {
            sign = -1;
        }
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return None;
    }
    t[start..i].parse::<i64>().ok().map(|v| sign * v)
}

/// JS `Math.round` — half-up toward +∞ (differs from Rust `f64::round` for
/// negative halves: JS Math.round(-0.5) = -0, Rust (-0.5).round() = -1).
pub fn js_math_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

/// `formatNumber` (`utils/richtext/typographyNormalizer.ts:42-45`):
/// `Number(value.toFixed(6)).toString()` — i.e. round to 6 decimals.
/// We only need the numeric value (callers re-parse with parseFloat), so we
/// return the rounded f64. toFixed rounds half away from zero.
pub fn js_to_fixed6_value(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0; // Object.is(value, -0) → '0'
    }
    let scaled = value * 1e6;
    let rounded = if scaled >= 0.0 {
        (scaled + 0.5).floor()
    } else {
        (scaled - 0.5).ceil()
    };
    rounded / 1e6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_float_prefixes() {
        // cited: parseFloat semantics used by parser.ts:17/36 regex captures
        assert_eq!(js_parse_float("1.5px"), Some(1.5));
        assert_eq!(js_parse_float("  -2.25"), Some(-2.25));
        assert_eq!(js_parse_float(".5"), Some(0.5));
        assert_eq!(js_parse_float("px"), None);
    }

    #[test]
    fn parse_int_prefixes() {
        assert_eq!(js_parse_int("700"), Some(700));
        assert_eq!(js_parse_int("700.9"), Some(700));
        assert_eq!(js_parse_int("bold"), None);
        assert_eq!(js_parse_int(" -3x"), Some(-3));
    }

    #[test]
    fn to_fixed6() {
        assert_eq!(js_to_fixed6_value(0.1 + 0.2), 0.3);
        assert_eq!(js_to_fixed6_value(-0.0), 0.0);
    }
}
