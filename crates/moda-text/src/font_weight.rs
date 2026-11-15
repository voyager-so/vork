//! Ports of `utils/richtext/fontWeight.ts` and the subset of
//! `utils/richtext/typographyNormalizer.ts` the rich-text parser consumes
//! (letter-spacing normalization, font-weight/font-style normalization).
//! Hand-rolled scanners replace the JS regexes 1:1; each function cites its
//! TS source.

use crate::jsnum::{js_parse_float, js_to_fixed6_value};

/// `FONT_WEIGHT_ALIASES` (`fontWeight.ts:1-18`).
fn alias_for(key: &str) -> Option<&'static str> {
    Some(match key {
        "normal" => "400",
        "regular" => "400",
        "book" => "400",
        "medium" => "500",
        "bold" => "700",
        "semibold" => "600",
        "demibold" => "600",
        "extrabold" => "800",
        "ultrabold" => "800",
        "black" => "900",
        "heavy" => "900",
        "light" => "300",
        "extralight" => "200",
        "ultralight" => "200",
        "thin" => "100",
        "hairline" => "100",
        _ => return None,
    })
}

const RELATIVE_WEIGHT_STEPS: [i64; 4] = [100, 400, 700, 900];

/// `normalizeWeightKey` (`fontWeight.ts:22-27`): lowercase, strip quotes,
/// collapse `[\s_-]+` to nothing.
fn normalize_weight_key(value: &str) -> String {
    let mut out = String::new();
    for c in value.to_lowercase().chars() {
        if c == '\'' || c == '"' {
            continue;
        }
        if c.is_whitespace() || c == '_' || c == '-' {
            continue;
        }
        out.push(c);
    }
    out
}

/// A complete finite CSS weight number, preserving fractional precision.
fn parse_numeric_weight_str(raw: &str) -> Option<f64> {
    let value = raw.trim().parse::<f64>().ok()?;
    (value.is_finite() && (1.0..=1000.0).contains(&value)).then_some(value)
}

/// `resolveRelativeWeight` (`fontWeight.ts:47-62`).
fn resolve_relative_weight(base_weight: f64, keyword: &str) -> i64 {
    let min_step = RELATIVE_WEIGHT_STEPS[0];
    let max_step = RELATIVE_WEIGHT_STEPS[3];
    let clamped = base_weight.clamp(min_step as f64, max_step as f64);
    if keyword == "bolder" {
        for step in RELATIVE_WEIGHT_STEPS {
            if step as f64 > clamped {
                return step;
            }
        }
        return max_step;
    }
    for step in RELATIVE_WEIGHT_STEPS.iter().rev() {
        if (*step as f64) < clamped {
            return *step;
        }
    }
    min_step
}

/// `resolveBaseWeight` (`fontWeight.ts:64-71`), string input.
fn resolve_base_weight(value: Option<&str>) -> Option<f64> {
    let value = value?;
    if let Some(n) = parse_numeric_weight_str(value) {
        return Some(n);
    }
    let key = normalize_weight_key(value);
    alias_for(&key).and_then(|a| a.parse::<f64>().ok())
}

/// `normalizeFontWeightValue` (`fontWeight.ts:73-88`), string input.
/// Returns the normalized numeric-string, or None (TS `undefined`).
pub fn normalize_font_weight_value(value: &str, base_weight: Option<&str>) -> Option<String> {
    if let Some(n) = parse_numeric_weight_str(value) {
        return Some(n.to_string());
    }
    let raw = value.trim();
    if raw.is_empty() {
        return None;
    }
    let key = normalize_weight_key(raw);
    if key == "bolder" || key == "lighter" {
        let base = resolve_base_weight(base_weight).unwrap_or(400.0);
        return Some(resolve_relative_weight(base, &key).to_string());
    }
    alias_for(&key).map(|s| s.to_string())
}

/// `parseFontWeightNumber` (`fontWeight.ts:90-96`).
pub fn parse_font_weight_number(value: &str, base_weight: Option<&str>) -> Option<f64> {
    normalize_font_weight_value(value, base_weight).and_then(|s| s.parse::<f64>().ok())
}

// ---------------------------------------------------------------------------
// typographyNormalizer.ts subset
// ---------------------------------------------------------------------------

/// `normalizeFontStyleToken` (`typographyNormalizer.ts:56-59`).
fn normalize_font_style_token(value: &str) -> Option<&'static str> {
    match value.trim().to_lowercase().as_str() {
        "normal" => Some("normal"),
        "italic" => Some("italic"),
        "oblique" => Some("oblique"),
        _ => None,
    }
}

/// `normalizeFontWeightNumber` (`typographyNormalizer.ts:69-74`).
fn normalize_font_weight_number(value: &str) -> Option<f64> {
    normalize_font_weight_value(value, None).and_then(|s| s.parse::<f64>().ok())
}

#[derive(Debug, Default, PartialEq)]
pub struct CombinedFontStyle {
    pub font_weight: Option<f64>,
    pub font_style: Option<&'static str>,
    pub unresolved_weight: bool,
}

/// `normalizeCombinedFontStyle` for string input
/// (`typographyNormalizer.ts:76-121`).
pub fn normalize_combined_font_style(value: &str, normal_as_weight: bool) -> CombinedFontStyle {
    let raw = value.trim();
    if raw.is_empty() {
        return CombinedFontStyle::default();
    }
    let lower = raw.to_lowercase();
    if lower == "bolder" || lower == "lighter" {
        return CombinedFontStyle {
            unresolved_weight: true,
            ..Default::default()
        };
    }

    if !raw.chars().any(|c| c.is_whitespace()) {
        let direct_weight = normalize_font_weight_number(raw);
        let direct_style = normalize_font_style_token(raw);
        if direct_weight.is_some() && direct_style.is_none() {
            return CombinedFontStyle {
                font_weight: direct_weight,
                ..Default::default()
            };
        }
        if direct_style.is_some() && direct_weight.is_none() {
            return CombinedFontStyle {
                font_style: direct_style,
                ..Default::default()
            };
        }
    }

    let mut style_parts: Vec<&'static str> = Vec::new();
    let mut weight_parts: Vec<String> = Vec::new();
    for part in raw.split_whitespace() {
        let style = normalize_font_style_token(part);
        match style {
            Some(s) if !(normal_as_weight && s == "normal") => style_parts.push(s),
            _ => weight_parts.push(part.to_string()),
        }
    }

    let font_weight = if !weight_parts.is_empty() {
        normalize_font_weight_number(&weight_parts.join(" "))
    } else {
        None
    };
    let font_style = style_parts
        .last()
        .and_then(|s| normalize_font_style_token(s));
    CombinedFontStyle {
        font_weight,
        font_style,
        unresolved_weight: !weight_parts.is_empty() && font_weight.is_none(),
    }
}

/// Result of `normalizeTypographyStyleValues({fontWeight})`
/// (`typographyNormalizer.ts:288-312`) — styles only; warnings dropped.
pub struct NormalizedWeightStyles {
    pub font_weight: Option<f64>,
    pub font_style: Option<&'static str>,
}

pub fn normalize_typography_font_weight(value: &str) -> NormalizedWeightStyles {
    let raw_weight = value.trim().to_lowercase();
    let direct_weight = if raw_weight == "bolder"
        || raw_weight == "lighter"
        || raw_weight.chars().any(|c| c.is_whitespace())
    {
        None
    } else {
        normalize_font_weight_number(value)
    };
    if let Some(w) = direct_weight {
        return NormalizedWeightStyles {
            font_weight: Some(w),
            font_style: None,
        };
    }
    let combined = normalize_combined_font_style(value, true);
    NormalizedWeightStyles {
        font_weight: combined.font_weight,
        font_style: combined.font_style,
    }
}

/// `normalizeTypographyStyleValues({fontStyle})` core
/// (`typographyNormalizer.ts:314-335`) — style result only.
pub fn normalize_typography_font_style(value: &str) -> Option<&'static str> {
    normalize_combined_font_style(value, false).font_style
}

/// `normalizeLetterSpacingValue` for string input
/// (`typographyNormalizer.ts:123-185`). Returns the normalized CSS string
/// (e.g. `"1.5px"`, `"0.2em"`) or None on invalid.
pub fn normalize_letter_spacing_value(value: &str) -> Option<String> {
    let raw = value.trim();
    // LETTER_SPACING_RE: ^([-+]?(?:\d+\.?\d*|\.\d+))(px|em|pt|%)?$ (i)
    let (amount_str, unit) =
        match_css_number_with_unit(raw, &["px", "em", "pt", "pc", "in", "cm", "mm", "q", "%"])?;
    let amount = js_parse_float(amount_str)?;
    let unit = unit.map(|u| u.to_lowercase());
    match unit.as_deref() {
        Some("%") => Some(format!("{}em", format_number(amount / 100.0))),
        Some("pt") => Some(format!("{}px", format_number(amount))),
        Some("pc") => Some(format!("{}px", format_number(amount * 12.0))),
        Some("in") => Some(format!("{}px", format_number(amount * 72.0))),
        Some("cm") => Some(format!("{}px", format_number(amount * 72.0 / 2.54))),
        Some("mm") => Some(format!("{}px", format_number(amount * 72.0 / 25.4))),
        Some("q") => Some(format!("{}px", format_number(amount * 72.0 / 101.6))),
        Some(u) => Some(format!("{}{}", format_number(amount), u)),
        None => Some(format!("{}px", format_number(amount))),
    }
}

/// `formatNumber` (`typographyNormalizer.ts:42-45`) — JS shortest repr of the
/// toFixed(6)-rounded value. We need a string parseFloat round-trips to the
/// same f64; Rust's `{}` for f64 is shortest round-trip like JS.
fn format_number(value: f64) -> String {
    let v = js_to_fixed6_value(value);
    if v == 0.0 {
        return "0".to_string();
    }
    format!("{}", v)
}

/// Match `^(NUMBER)(unit)?$` where NUMBER is `[-+]?(\d+\.?\d*|\.\d+)` and
/// unit ∈ allowed (case-insensitive). Returns (number_str, unit).
pub fn match_css_number_with_unit<'a>(
    raw: &'a str,
    allowed_units: &[&str],
) -> Option<(&'a str, Option<&'a str>)> {
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i > digits_start {
        // \d+\.?\d*
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
    } else if i < bytes.len() && bytes[i] == b'.' {
        // \.\d+
        i += 1;
        let frac_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return None;
        }
    } else {
        return None;
    }
    let number = &raw[..i];
    let rest = &raw[i..];
    if rest.is_empty() {
        return Some((number, None));
    }
    for u in allowed_units {
        if rest.eq_ignore_ascii_case(u) {
            return Some((number, Some(rest)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_aliases_and_relative() {
        // fontWeight.ts:1-18, 47-62
        assert_eq!(
            normalize_font_weight_value("bold", None).as_deref(),
            Some("700")
        );
        assert_eq!(
            normalize_font_weight_value("Semi Bold", None).as_deref(),
            Some("600")
        );
        assert_eq!(
            normalize_font_weight_value("bolder", Some("400")).as_deref(),
            Some("700")
        );
        assert_eq!(
            normalize_font_weight_value("lighter", Some("700")).as_deref(),
            Some("400")
        );
        assert_eq!(
            normalize_font_weight_value("450.4", None).as_deref(),
            Some("450.4")
        );
        assert_eq!(normalize_font_weight_value("oblique", None), None);
    }

    #[test]
    fn weight_tokens_are_complete_and_preserve_fractional_css_numbers() {
        for invalid in [
            "junk600", "600junk", "0", "1000.1", "NaN", "Infinity", "1e309",
        ] {
            assert_eq!(
                normalize_font_weight_value(invalid, None),
                None,
                "{invalid}"
            );
        }
        assert_eq!(
            normalize_font_weight_value("449.9", None).as_deref(),
            Some("449.9")
        );
        assert_eq!(parse_font_weight_number("4.5e2", None), Some(450.0));
    }

    #[test]
    fn letter_spacing_norm() {
        // typographyNormalizer.ts:123-185
        assert_eq!(
            normalize_letter_spacing_value("1.5px").as_deref(),
            Some("1.5px")
        );
        assert_eq!(
            normalize_letter_spacing_value("0.2em").as_deref(),
            Some("0.2em")
        );
        assert_eq!(
            normalize_letter_spacing_value("50%").as_deref(),
            Some("0.5em")
        );
        assert_eq!(normalize_letter_spacing_value("6").as_deref(), Some("6px"));
        assert_eq!(
            normalize_letter_spacing_value("6pt").as_deref(),
            Some("6px")
        );
        assert_eq!(normalize_letter_spacing_value("1pc").as_deref(), Some("12px"));
        assert_eq!(normalize_letter_spacing_value("1in").as_deref(), Some("72px"));
        assert_eq!(normalize_letter_spacing_value("1pcjunk"), None);
        assert_eq!(normalize_letter_spacing_value("abc"), None);
    }

    #[test]
    fn combined_font_styles() {
        // typographyNormalizer.ts:76-121
        let c = normalize_combined_font_style("italic 700", false);
        assert_eq!(c.font_weight, Some(700.0));
        assert_eq!(c.font_style, Some("italic"));
        let c = normalize_typography_font_weight("bold");
        assert_eq!(c.font_weight, Some(700.0));
    }
}
