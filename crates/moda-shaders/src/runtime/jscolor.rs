//! Port of the production color parsing used by shader instance packing.
//!
//! `colorToNormalizedRgba` (`utils/colorHelpers.ts:377-399`) tries hex first
//! (`hexToRgba`, `:324-371`: 3/4/6/8-digit, leading `#` optional), then
//! `rgb()`/`rgba()` strings (`parseRgbaColor`, `:285-317`). Named CSS colors
//! do NOT parse here — they fall through to `None`, which packing turns into
//! opaque white (copy the bug; cf. `writeColorAt`, `mesh/colorPacking.ts:26-33`).
//!
//! JS `parseInt(s, 16)` parses the longest leading hex prefix ("5g" → 5,
//! "g5" → NaN); mirrored by `js_parse_int_hex`.

/// Straight (non-premultiplied) rgba, all channels normalized 0..1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormRgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

/// JS `parseInt(s, 16)`: longest leading hex-digit prefix; None for NaN.
fn js_parse_int_hex_chars(chars: impl IntoIterator<Item = char>) -> Option<u32> {
    let mut value: Option<u32> = None;
    for c in chars {
        let Some(digit) = c.to_digit(16) else {
            break;
        };
        value = Some(value.unwrap_or(0).saturating_mul(16).saturating_add(digit));
    }
    value
}

/// `hexToRgba` (`utils/colorHelpers.ts:324-371`). Removes only the FIRST `#`
/// (JS `String.replace` with a string pattern), then dispatches on length.
fn hex_to_rgba(hex: &str) -> Option<NormRgba> {
    let clean = hex.replacen('#', "", 1);
    let chars: Vec<char> = clean.chars().collect();
    let pair = |i: usize| js_parse_int_hex_chars([chars[i], chars[i + 1]]);
    let dup = |c: char| js_parse_int_hex_chars([c, c]);
    // Use collected chars for 6/8-char forms too; shader params can contain
    // multi-byte UTF-8 and JS substring would not panic on those inputs.
    let (r, g, b, a) = match chars.len() {
        3 => {
            let r = dup(chars[0])?;
            let g = dup(chars[1])?;
            let b = dup(chars[2])?;
            (r, g, b, 1.0)
        }
        4 => {
            let r = dup(chars[0])?;
            let g = dup(chars[1])?;
            let b = dup(chars[2])?;
            let a = dup(chars[3])? as f64 / 255.0;
            (r, g, b, a)
        }
        6 => {
            let r = pair(0)?;
            let g = pair(2)?;
            let b = pair(4)?;
            (r, g, b, 1.0)
        }
        8 => {
            let r = pair(0)?;
            let g = pair(2)?;
            let b = pair(4)?;
            let a = pair(6)? as f64 / 255.0;
            (r, g, b, a)
        }
        _ => return None,
    };
    Some(NormRgba {
        r: r as f64,
        g: g as f64,
        b: b as f64,
        a,
    })
}

/// `parseRgbaColor` (`utils/colorHelpers.ts:285-317`):
/// `^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)$`,
/// alpha via JS `parseFloat` over a `[\d.]+` token (longest valid prefix).
fn parse_rgba_color(s: &str) -> Option<NormRgba> {
    let clean = s.trim();
    let rest = clean
        .strip_prefix("rgba")
        .or_else(|| clean.strip_prefix("rgb"))?;
    let rest = rest.strip_prefix('(')?;
    let rest = rest.strip_suffix(')')?;

    // Split into 3 or 4 comma-separated tokens.
    let parts: Vec<&str> = rest.split(',').map(str::trim).collect();
    if parts.len() != 3 && parts.len() != 4 {
        return None;
    }
    // (\d+): digits only.
    let int_part = |t: &str| -> Option<u64> {
        if t.is_empty() || !t.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        t.parse::<u64>().ok()
    };
    let r = int_part(parts[0])?;
    let g = int_part(parts[1])?;
    let b = int_part(parts[2])?;
    let a = if parts.len() == 4 {
        let t = parts[3];
        // Regex token [\d.]+ (digits and dots only), then JS parseFloat
        // (longest valid leading float prefix; NaN fails the range check).
        if t.is_empty() || !t.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return None;
        }
        js_parse_float(t)?
    } else {
        1.0
    };
    if r > 255 || g > 255 || b > 255 {
        return None;
    }
    if !(0.0..=1.0).contains(&a) {
        return None;
    }
    Some(NormRgba {
        r: r as f64,
        g: g as f64,
        b: b as f64,
        a,
    })
}

/// JS `parseFloat`: longest valid leading float prefix; None when no digits.
fn js_parse_float(s: &str) -> Option<f64> {
    let mut end = 0usize;
    let mut seen_digit = false;
    let mut seen_dot = false;
    for (i, c) in s.char_indices() {
        match c {
            '0'..='9' => {
                seen_digit = true;
                end = i + 1;
            }
            '.' if !seen_dot => {
                seen_dot = true;
                end = i + 1;
            }
            _ => break,
        }
    }
    if !seen_digit {
        return None;
    }
    s[..end]
        .trim_end_matches('.')
        .parse::<f64>()
        .ok()
        .or_else(|| {
            // e.g. "." alone — no digits case is already handled above.
            None
        })
}

/// `colorToNormalizedRgba` (`utils/colorHelpers.ts:377-399`): hex first, then
/// rgb()/rgba(); channels normalized to 0..1. `None` = unparsable (named
/// colors, garbage) — callers substitute their own fallback.
pub fn color_to_normalized_rgba(color: &str) -> Option<NormRgba> {
    if let Some(c) = hex_to_rgba(color) {
        return Some(NormRgba {
            r: c.r / 255.0,
            g: c.g / 255.0,
            b: c.b / 255.0,
            a: c.a,
        });
    }
    parse_rgba_color(color).map(|c| NormRgba {
        r: c.r / 255.0,
        g: c.g / 255.0,
        b: c.b / 255.0,
        a: c.a,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_forms() {
        let c = color_to_normalized_rgba("#FF006E").unwrap();
        assert_eq!((c.r, c.g, c.b, c.a), (1.0, 0.0, 110.0 / 255.0, 1.0));
        // 8-digit with alpha (shape-grid default background "#00000000").
        let c = color_to_normalized_rgba("#00000000").unwrap();
        assert_eq!((c.r, c.g, c.b, c.a), (0.0, 0.0, 0.0, 0.0));
        // 3-digit.
        let c = color_to_normalized_rgba("fff").unwrap();
        assert_eq!((c.r, c.g, c.b, c.a), (1.0, 1.0, 1.0, 1.0));
        // 4-digit with alpha.
        let c = color_to_normalized_rgba("#f00c").unwrap();
        assert_eq!(c.a, 204.0 / 255.0);
    }

    #[test]
    fn rgba_strings() {
        let c = color_to_normalized_rgba("rgba(255, 0, 110, 0.5)").unwrap();
        assert_eq!((c.r, c.b, c.a), (1.0, 110.0 / 255.0, 0.5));
        let c = color_to_normalized_rgba("rgb(0,0,0)").unwrap();
        assert_eq!((c.r, c.a), (0.0, 1.0));
    }

    #[test]
    fn named_colors_do_not_parse() {
        // Copy the bug: named CSS colors are not parsed by the packing path.
        assert!(color_to_normalized_rgba("red").is_none());
        assert!(color_to_normalized_rgba("").is_none());
    }

    #[test]
    fn js_parseint_prefix_semantics() {
        // parseInt("5g", 16) === 5 in JS — the second nibble char of a 6-digit
        // hex parses its prefix.
        let c = hex_to_rgba("12345g").unwrap();
        assert_eq!((c.r, c.g, c.b), (0x12 as f64, 0x34 as f64, 5.0));
        assert!(hex_to_rgba("zzzzzz").is_none());
    }

    #[test]
    fn multibyte_hex_does_not_panic() {
        assert!(color_to_normalized_rgba("ф23456").is_none());
        assert!(color_to_normalized_rgba("1a2b3cф0").is_none());

        // JS parseInt prefix semantics still apply when the invalid code point
        // is the second character in a two-character hex slice.
        let c = color_to_normalized_rgba("1ф3456").unwrap();
        assert_eq!(
            (c.r, c.g, c.b, c.a),
            (1.0 / 255.0, 0x34 as f64 / 255.0, 0x56 as f64 / 255.0, 1.0)
        );
    }
}
