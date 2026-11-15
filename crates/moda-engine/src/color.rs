//! color.rs — port of the renderer's color parsing
//! (`renderers/webgpu/utils/WebGPUColorCache.parseColor` →
//! `utils/colorHelpers.ts colorToNormalizedRgba` = hexToRgba → parseRgbaColor).
//!
//! COPY THE BUGS:
//! - Named CSS colors (other than 'transparent') are NOT supported on this
//!   path: `colorToNormalizedRgba("red")` → hex parse of "red" fails → rgba
//!   regex fails → null → the renderer falls back to DEFAULT_FILL_COLOR grey
//!   {0.5, 0.5, 0.5, 1} (`WebGPUConstants.ts:145`,
//!   `WebGPUPathRenderer.ts:1267`). The named-color table in colorHelpers is
//!   used by OTHER code paths, not by the path renderer.
//! - Non-string fills (objects, numbers, null) parse to None → grey.
//! - rgb()/rgba() accepts INTEGER rgb components only (regex `\d+`), alpha
//!   `[\d.]+`; out-of-range values reject the whole string → grey.
//! - hex strips a single leading '#' (JS `String.replace` replaces the first
//!   occurrence only) and accepts 3/4/6/8 digit forms.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

/// `PATH_RENDERER_DEFAULTS.DEFAULT_FILL_COLOR` (`WebGPUConstants.ts:145`).
pub const DEFAULT_FILL_COLOR: Rgba = Rgba {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
};

/// `WebGPUColorCache.parseColor` (cache-free port). Returns None exactly when
/// the production parser returns null.
pub fn parse_color(color: Option<&str>) -> Option<Rgba> {
    let color = color?;
    let normalized = color.trim().to_lowercase();
    if normalized == "transparent" {
        return Some(Rgba {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        });
    }
    // colorToNormalizedRgba(color) || colorToNormalizedRgba(normalized)
    color_to_normalized_rgba(color).or_else(|| color_to_normalized_rgba(&normalized))
}

/// `colorToNormalizedRgba` (`utils/colorHelpers.ts:377-401`) — pub(crate) for
/// M3b's gradient-stop parsing, which uses THIS function directly (no
/// 'transparent' keyword, no lowercase retry — unlike `parse_color`).
pub(crate) fn color_to_normalized_rgba(color: &str) -> Option<Rgba> {
    if let Some((r, g, b, a)) = hex_to_rgba(color) {
        return Some(Rgba {
            r: r / 255.0,
            g: g / 255.0,
            b: b / 255.0,
            a,
        });
    }
    if let Some((r, g, b, a)) = parse_rgba_color(color) {
        return Some(Rgba {
            r: r / 255.0,
            g: g / 255.0,
            b: b / 255.0,
            a,
        });
    }
    None
}

/// `colorToNormalizedRgb` (`utils/colorHelpers.ts:256-279`) — the PAGE-QUAD
/// parser (`WebGPUPageSystem.ts:427`): `hexToRgb` (3/6-digit hex ONLY — 4/8
/// digit forms are REJECTED, unlike `colorToNormalizedRgba`'s `hexToRgba`)
/// then `parseRgbColor` (integer rgb()/rgba(); the alpha term only has to be
/// `[\d.]+`-shaped and is otherwise IGNORED — `rgba(0,0,0,9.5)` parses here
/// while `parseRgbaColor` rejects it). No 'transparent' keyword, no lowercase
/// retry. Callers fall back to WHITE on None (`|| { r: 1, g: 1, b: 1 }`).
pub fn color_to_normalized_rgb(color: &str) -> Option<(f64, f64, f64)> {
    if let Some((r, g, b)) = hex_to_rgb(color) {
        return Some((r / 255.0, g / 255.0, b / 255.0));
    }
    if let Some((r, g, b)) = parse_rgb_color(color) {
        return Some((r / 255.0, g / 255.0, b / 255.0));
    }
    None
}

/// `hexToRgb` (`utils/colorHelpers.ts:218-246`): strip the FIRST '#', expand
/// 3-digit, validate `^[0-9a-fA-F]{6}$`. Returns (r,g,b 0-255).
fn hex_to_rgb(hex: &str) -> Option<(f64, f64, f64)> {
    if hex.is_empty() {
        return None;
    }
    let clean = hex.replacen('#', "", 1);
    let full: String = if clean.chars().count() == 3 {
        clean.chars().flat_map(|c| [c, c]).collect()
    } else {
        clean
    };
    if full.len() != 6 || !full.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let d = |s: &str| -> Option<f64> { u32::from_str_radix(s, 16).ok().map(|v| v as f64) };
    Some((d(&full[0..2])?, d(&full[2..4])?, d(&full[4..6])?))
}

/// `parseRgbColor` (`utils/colorHelpers.ts:285-312`): regex
/// `^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*[\d.]+)?\s*\)$` — the
/// optional alpha is non-capturing and its VALUE is never validated.
fn parse_rgb_color(s: &str) -> Option<(f64, f64, f64)> {
    let t = s.trim();
    let rest = t.strip_prefix("rgba").or_else(|| t.strip_prefix("rgb"))?;
    let inner = rest.strip_prefix('(')?.strip_suffix(')')?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 3 && parts.len() != 4 {
        return None;
    }
    let int_part = |p: &str| -> Option<f64> {
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        p.parse::<f64>().ok()
    };
    let r = int_part(parts[0])?;
    let g = int_part(parts[1])?;
    let b = int_part(parts[2])?;
    if parts.len() == 4 {
        let p = parts[3];
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
            return None;
        }
    }
    if !(0.0..=255.0).contains(&r) || !(0.0..=255.0).contains(&g) || !(0.0..=255.0).contains(&b) {
        return None;
    }
    Some((r, g, b))
}

/// `hexToRgba` (`utils/colorHelpers.ts:324-369`). Returns (r,g,b 0-255, a 0-1).
fn hex_to_rgba(hex: &str) -> Option<(f64, f64, f64, f64)> {
    if hex.is_empty() {
        return None;
    }
    // JS `hex.replace('#', '')` — first occurrence only.
    let clean = hex.replacen('#', "", 1);
    let chars: Vec<char> = clean.chars().collect();
    let byte = |hi: char, lo: char| -> Option<f64> {
        let hi = hi.to_digit(16)?;
        let lo = lo.to_digit(16)?;
        Some((hi * 16 + lo) as f64)
    };
    let dup = |c: char| -> Option<f64> {
        let d = c.to_digit(16)?;
        Some((d * 17) as f64)
    };
    // Index the collected chars, not byte ranges over `clean`: color-shaped
    // user strings can contain multi-byte UTF-8 and must return None, not panic.
    let (r, g, b, a) = match chars.len() {
        3 => (dup(chars[0])?, dup(chars[1])?, dup(chars[2])?, 1.0),
        4 => (
            dup(chars[0])?,
            dup(chars[1])?,
            dup(chars[2])?,
            dup(chars[3])? / 255.0,
        ),
        6 => (
            byte(chars[0], chars[1])?,
            byte(chars[2], chars[3])?,
            byte(chars[4], chars[5])?,
            1.0,
        ),
        8 => (
            byte(chars[0], chars[1])?,
            byte(chars[2], chars[3])?,
            byte(chars[4], chars[5])?,
            byte(chars[6], chars[7])? / 255.0,
        ),
        _ => return None,
    };
    Some((r, g, b, a))
}

/// `parseRgbaColor` (`utils/colorHelpers.ts:285-317`): strict regex
/// `^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)$`.
fn parse_rgba_color(s: &str) -> Option<(f64, f64, f64, f64)> {
    let t = s.trim();
    let rest = t.strip_prefix("rgba").or_else(|| t.strip_prefix("rgb"))?;
    let inner = rest.strip_prefix('(')?.strip_suffix(')')?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 3 && parts.len() != 4 {
        return None;
    }
    let int_part = |p: &str| -> Option<f64> {
        if p.is_empty() || !p.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        p.parse::<f64>().ok()
    };
    let r = int_part(parts[0])?;
    let g = int_part(parts[1])?;
    let b = int_part(parts[2])?;
    let a = if parts.len() == 4 {
        let p = parts[3];
        // [\d.]+ — digits and dots only; JS parseFloat of e.g. "1.2.3" → 1.2,
        // but the regex itself permits it; mirror parseFloat's prefix parse.
        if p.is_empty() || !p.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return None;
        }
        js_parse_float(p)?
    } else {
        1.0
    };
    if !(0.0..=255.0).contains(&r) || !(0.0..=255.0).contains(&g) || !(0.0..=255.0).contains(&b) {
        return None;
    }
    if !(0.0..=1.0).contains(&a) {
        return None;
    }
    Some((r, g, b, a))
}

/// JS parseFloat: longest valid numeric prefix.
fn js_parse_float(s: &str) -> Option<f64> {
    let mut end = 0;
    let mut seen_dot = false;
    for (i, c) in s.char_indices() {
        if c.is_ascii_digit() {
            end = i + 1;
        } else if c == '.' && !seen_dot {
            seen_dot = true;
            end = i + 1;
        } else {
            break;
        }
    }
    s[..end]
        .trim_end_matches('.')
        .parse::<f64>()
        .ok()
        .or_else(|| {
            if end > 0 {
                s[..end].parse::<f64>().ok()
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_forms() {
        let c = parse_color(Some("#3a86ff")).unwrap();
        assert_eq!(
            (c.r, c.g, c.b, c.a),
            (58.0 / 255.0, 134.0 / 255.0, 255.0 / 255.0, 1.0)
        );
        assert_eq!(parse_color(Some("#fff")).unwrap().r, 1.0);
        assert!((parse_color(Some("#ffffff80")).unwrap().a - 128.0 / 255.0).abs() < 1e-12);
        assert!(parse_color(Some("abc")).is_some()); // no-# 3-digit parses (copy)
    }

    #[test]
    fn rgba_forms() {
        let c = parse_color(Some("rgba(255, 0, 110, 0.6)")).unwrap();
        assert_eq!((c.r, c.g, c.b), (1.0, 0.0, 110.0 / 255.0));
        assert!((c.a - 0.6).abs() < 1e-12);
        assert_eq!(parse_color(Some("rgb(0,0,0)")).unwrap().a, 1.0);
        assert!(parse_color(Some("rgb(256,0,0)")).is_none());
        assert!(parse_color(Some("rgb(1.5, 0, 0)")).is_none()); // ints only
    }

    #[test]
    fn multibyte_hex_does_not_panic() {
        // Regression for WEB-NEXTJS-1YH: a color-shaped value that is exactly
        // 6 or 8 Unicode scalars but contains multi-byte UTF-8 used to make the
        // 6/8-digit arms byte-slice mid-character and panic
        // ("end byte index 4 is not a char boundary; it is inside 'ф'"). These
        // must now return None (→ DEFAULT_FILL_COLOR grey) without panicking,
        // matching the JS parser's null path.
        assert!(parse_color(Some("абвгде")).is_none()); // 6 chars, all 2-byte
        assert!(parse_color(Some("ф23456")).is_none()); // 6 chars, leading 2-byte
        assert!(parse_color(Some("abcфde")).is_none()); // the reported shape
        assert!(parse_color(Some("1a2b3cф0")).is_none()); // 8 chars w/ multibyte
        assert!(parse_color(Some("абвгдежз")).is_none()); // 8 chars, all 2-byte
                                                          // Valid ASCII hex still parses exactly as before — byte-identical.
        let c = parse_color(Some("#1a2b3c")).unwrap();
        assert_eq!(
            (c.r, c.g, c.b, c.a),
            (26.0 / 255.0, 43.0 / 255.0, 60.0 / 255.0, 1.0)
        );
        assert_eq!(parse_color(Some("1a2b3c")), parse_color(Some("#1a2b3c")));
        assert!((parse_color(Some("1a2b3c80")).unwrap().a - 128.0 / 255.0).abs() < 1e-12);
        assert_eq!(parse_color(Some("fff")).unwrap().r, 1.0);
    }

    #[test]
    fn quirks() {
        // Named colors fall through to None → grey at the call site.
        assert!(parse_color(Some("red")).is_none());
        assert_eq!(
            parse_color(Some("transparent")),
            Some(Rgba {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0
            })
        );
        assert!(parse_color(None).is_none());
    }
}
