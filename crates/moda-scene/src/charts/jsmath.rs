//! JS numeric semantics for the chart port (M4c).
//!
//! The chart generators are a line-for-line port of TS that leans on JS
//! number behavior: `undefined` arithmetic → NaN, `Math.max(NaN, x)` → NaN,
//! `Number.prototype.toFixed/toExponential` (round-half-away-from-zero on the
//! EXACT decimal expansion), and ECMA `Number::toString` shortest-round-trip
//! rendering. Rust's defaults differ on every one of those, so this module
//! provides faithful equivalents ("copy the bugs", PLAN §0 rule 3).

/// `Math.max` — NaN-propagating (Rust `f64::max` ignores NaN).
pub fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a > b {
        a
    } else {
        b
    }
}

/// `Math.min` — NaN-propagating.
pub fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        a
    } else {
        b
    }
}

/// `Math.round` — half toward +∞, exact on the 0.49999999999999994 edge case
/// (naive `floor(x + 0.5)` would round that up; JS does not).
pub fn js_round(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// `Number.isFinite`.
pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

/// `Number.isInteger`.
pub fn is_integer(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0
}

// ---------------------------------------------------------------------------
// Exact decimal expansion + rounding (the core under toFixed/toExponential)
// ---------------------------------------------------------------------------

/// Exact decimal digits of a finite positive f64: returns `(digits, e)` where
/// the value equals `0.d1 d2 … × 10^(e+1)` — i.e. `digits[0]` is the first
/// significant digit and the value is `d1.d2… × 10^e`. Trailing zeros
/// trimmed. Uses Rust's correctly-rounded `{:.P e}` formatting with enough
/// precision (770 fraction digits) to capture every nonzero digit any f64 can
/// have, so subsequent string rounding is rounding of the EXACT value.
fn exact_digits(x: f64) -> (Vec<u8>, i32) {
    debug_assert!(x.is_finite() && x > 0.0);
    let s = format!("{:.*e}", 770, x);
    // form: d.dddd...e<exp> (Rust e-format: exponent without '+' for positives)
    let epos = s.rfind('e').unwrap();
    let exp: i32 = s[epos + 1..].parse().unwrap();
    let mantissa = &s[..epos];
    let mut digits: Vec<u8> = Vec::with_capacity(mantissa.len());
    for c in mantissa.bytes() {
        if c.is_ascii_digit() {
            digits.push(c - b'0');
        }
    }
    while digits.len() > 1 && *digits.last().unwrap() == 0 {
        digits.pop();
    }
    (digits, exp)
}

/// Round exact digits to `p` significant digits, half away from zero (the
/// ECMA `toFixed`/`toExponential` tie rule on magnitudes). Returns
/// `(digits, e)` in the same convention as `exact_digits`. `p >= 1`.
fn round_significant(digits: &[u8], e: i32, p: usize) -> (Vec<u8>, i32) {
    if digits.len() <= p {
        let mut d = digits.to_vec();
        while d.len() < p {
            d.push(0);
        }
        return (d, e);
    }
    let mut d: Vec<u8> = digits[..p].to_vec();
    let round_up = digits[p] >= 5;
    let mut e = e;
    if round_up {
        let mut i = p;
        loop {
            if i == 0 {
                d.insert(0, 1);
                d.pop();
                e += 1;
                break;
            }
            i -= 1;
            if d[i] == 9 {
                d[i] = 0;
            } else {
                d[i] += 1;
                break;
            }
        }
    }
    (d, e)
}

fn digits_to_string(d: &[u8]) -> String {
    d.iter().map(|&x| (b'0' + x) as char).collect()
}

/// JS `Number.prototype.toExponential()` / `(p)` digits, as used by d3's
/// `formatDecimalParts`: returns `(coefficient_digits, exponent)` for a
/// positive finite x. `p` is the number of SIGNIFICANT digits (d3 passes
/// `p` ⇒ `toExponential(p-1)`); `None` = shortest round-trip.
pub fn format_decimal_parts(x: f64, p: Option<usize>) -> Option<(String, i32)> {
    if !x.is_finite() || x == 0.0 {
        return None;
    }
    debug_assert!(x > 0.0, "d3 always passes Math.abs");
    match p {
        None => {
            // shortest round-trip, same digits as JS toExponential()
            let s = format!("{x:e}");
            let epos = s.rfind('e').unwrap();
            let exp: i32 = s[epos + 1..].parse().unwrap();
            let digits: String = s[..epos].chars().filter(|c| c.is_ascii_digit()).collect();
            let digits = digits.trim_end_matches('0');
            let digits = if digits.is_empty() { "0" } else { digits };
            Some((digits.to_string(), exp))
        }
        Some(p) => {
            let p = p.max(1);
            let (d, e) = exact_digits(x);
            let (d, e) = round_significant(&d, e, p);
            Some((digits_to_string(&d), e))
        }
    }
}

/// JS `x.toFixed(p)` for finite non-negative x (|x| < 1e21).
pub fn to_fixed(x: f64, p: usize) -> String {
    debug_assert!(x.is_finite());
    if x < 0.0 {
        let m = to_fixed(-x, p);
        // JS toFixed keeps "-0.00" sign only when the rounded value is nonzero?
        // (-0.001).toFixed(2) === "-0.00" in JS. Keep the sign.
        return format!("-{m}");
    }
    if x == 0.0 {
        return if p == 0 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(p))
        };
    }
    let (digits, e) = exact_digits(x);
    fixed_from_digits(&digits, e, p)
}

/// `Intl.NumberFormat`'s rounding of `x × 10^-shift` to `p` fraction digits, for finite non-negative x. ICU
/// rounds the shortest decimal that round-trips x (the digits JS prints), half away from zero; `toFixed` rounds the
/// exact binary value, so 334650 / 1000 (stored as 334.6499…) is "334.6" there and "334.7" in Intl, and 1.005 is
/// "1.00" there and "1.01" in Intl. `shift` moves the decimal point exactly (compact notation's 10^3 units).
pub fn intl_fixed(x: f64, shift: i32, p: usize) -> String {
    debug_assert!(x.is_finite() && x >= 0.0);
    let Some((digits, e)) = format_decimal_parts(x, None) else {
        return if p == 0 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(p))
        };
    };
    let digits: Vec<u8> = digits.bytes().map(|b| b - b'0').collect();
    fixed_from_digits(&digits, e - shift, p)
}

/// `d1.d2… × 10^e` (a positive value's digits) rounded half away from zero to `p` fraction digits.
fn fixed_from_digits(digits: &[u8], e: i32, p: usize) -> String {
    // Round at decimal position p ⇒ significant digits = e + 1 + p.
    let sig = e + 1 + p as i32;
    if sig <= 0 {
        // Rounds to zero unless the first dropped digit forces a carry into
        // position. sig == 0 means the first significant digit sits exactly
        // one place below the rounding position.
        if sig == 0 && digits[0] >= 5 {
            // carries to 1 at the last kept fraction digit
            let mut s = String::from("0.");
            for _ in 0..p.saturating_sub(1) {
                s.push('0');
            }
            s.push('1');
            if p == 0 {
                return "1".to_string();
            }
            return s;
        }
        return if p == 0 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(p))
        };
    }
    let (d, e) = round_significant(&digits, e, sig as usize);
    let ds = digits_to_string(&d);
    render_fixed(&ds, e, p)
}

/// Render digit string `d1.d2… × 10^e` as fixed-point with exactly `p`
/// fraction digits (digits already rounded to e+1+p significant places).
fn render_fixed(digits: &str, e: i32, p: usize) -> String {
    let mut int_part = String::new();
    let mut frac_part = String::new();
    if e < 0 {
        int_part.push('0');
        for _ in 0..(-e - 1) {
            frac_part.push('0');
        }
        frac_part.push_str(digits);
    } else {
        let ilen = (e + 1) as usize;
        if digits.len() >= ilen {
            int_part.push_str(&digits[..ilen]);
            frac_part.push_str(&digits[ilen..]);
        } else {
            int_part.push_str(digits);
            for _ in 0..(ilen - digits.len()) {
                int_part.push('0');
            }
        }
    }
    while frac_part.len() < p {
        frac_part.push('0');
    }
    frac_part.truncate(p);
    if p == 0 {
        int_part
    } else {
        format!("{int_part}.{frac_part}")
    }
}

/// JS `x.toPrecision(p)` for finite non-negative x (d3 'g' type).
pub fn to_precision(x: f64, p: usize) -> String {
    debug_assert!(x.is_finite() && x >= 0.0);
    let p = p.max(1);
    if x == 0.0 {
        return if p == 1 {
            "0".to_string()
        } else {
            format!("0.{}", "0".repeat(p - 1))
        };
    }
    let (digits, e) = exact_digits(x);
    let (d, e) = round_significant(&digits, e, p);
    let ds = digits_to_string(&d);
    if e < -6 || e >= p as i32 {
        // exponential form: d.ddd e±k (JS prints explicit sign)
        let mut m = String::new();
        m.push(ds.as_bytes()[0] as char);
        if ds.len() > 1 {
            m.push('.');
            m.push_str(&ds[1..]);
        }
        let sign = if e < 0 { '-' } else { '+' };
        format!("{m}e{sign}{}", e.abs())
    } else {
        // fixed with p significant digits
        let frac = p as i32 - 1 - e;
        render_fixed(&ds, e, frac.max(0) as usize)
    }
}

/// ECMA `Number::toString(10)` — the exact string JS produces for a number
/// (template-literal interpolation, `String(x)`).
pub fn js_num_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x < 0.0 {
        return format!("-{}", js_num_to_string(-x));
    }
    // shortest round-trip digits via Rust's {:e}
    let s = format!("{x:e}");
    let epos = s.rfind('e').unwrap();
    let exp: i32 = s[epos + 1..].parse().unwrap();
    let digits: String = s[..epos].chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let k = digits.len() as i32; // digit count
    let n = exp + 1; // value = 0.digits × 10^n
    if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        if n >= k {
            digits.to_string()
        } else {
            format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
        }
    } else if -6 < n && n <= 0 {
        format!("0.{}{}", "0".repeat((-n) as usize), digits)
    } else {
        // exponential
        let mantissa = if k == 1 {
            digits.to_string()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let e = n - 1;
        let sign = if e < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{}", e.abs())
    }
}

/// JS strict `Number(string)` for d3's `+value === 0` check — only plain
/// decimal forms; anything with stray characters → NaN (caller treats as
/// nonzero).
pub fn js_number_strict(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        return 0.0;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

/// en-US thousands grouping of an ASCII digit string (d3 formatGroup with
/// grouping [3], width ∞).
pub fn group_thousands(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut groups: Vec<&str> = Vec::new();
    let mut i = bytes.len();
    while i > 0 {
        let start = i.saturating_sub(3);
        groups.push(&value[start..i]);
        i = start;
    }
    groups.reverse();
    groups.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_to_string_matches_js() {
        // verified against node -e 'console.log(String(x))'
        assert_eq!(js_num_to_string(12.0), "12");
        assert_eq!(js_num_to_string(0.5), "0.5");
        assert_eq!(js_num_to_string(45.599999999999994), "45.599999999999994");
        assert_eq!(js_num_to_string(1e21), "1e+21");
        assert_eq!(js_num_to_string(1e-7), "1e-7");
        assert_eq!(js_num_to_string(0.000001), "0.000001");
        assert_eq!(js_num_to_string(-3.25), "-3.25");
        assert_eq!(js_num_to_string(1234567890123.0), "1234567890123");
    }

    #[test]
    fn to_fixed_matches_js() {
        // verified against node: (x).toFixed(p)
        assert_eq!(to_fixed(45.599999999999994, 2), "45.60");
        assert_eq!(to_fixed(0.5, 2), "0.50");
        assert_eq!(to_fixed(1.005, 2), "1.00"); // 1.005 is < 1.005 exactly in binary
        assert_eq!(to_fixed(2.5, 0), "3"); // exact tie rounds away
        assert_eq!(to_fixed(1234.0, 0), "1234");
        assert_eq!(to_fixed(0.125, 2), "0.13"); // exact tie
        assert_eq!(to_fixed(0.004, 2), "0.00");
        assert_eq!(to_fixed(0.006, 2), "0.01");
        assert_eq!(to_fixed(0.005, 2), "0.01"); // 0.005 binary is slightly above? node says "0.01"
    }

    #[test]
    fn to_precision_matches_js() {
        assert_eq!(to_precision(1234.0, 12), "1234.00000000");
        assert_eq!(to_precision(0.5, 12), "0.500000000000");
        assert_eq!(to_precision(1e21, 12), "1.00000000000e+21");
    }

    #[test]
    fn decimal_parts_match_d3() {
        // formatDecimalParts(1.23) -> ["123", 0]
        assert_eq!(
            format_decimal_parts(1.23, None),
            Some(("123".to_string(), 0))
        );
        // formatDecimalParts(4567, 3) -> toExponential(2) = "4.57e+3"
        assert_eq!(
            format_decimal_parts(4567.0, Some(3)),
            Some(("457".to_string(), 3))
        );
        assert_eq!(format_decimal_parts(0.0, None), None);
    }

    #[test]
    fn grouping() {
        assert_eq!(group_thousands("1234567"), "1,234,567");
        assert_eq!(group_thousands("123"), "123");
        assert_eq!(group_thousands("1234"), "1,234");
    }

    #[test]
    fn js_round_edges() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
    }
}
