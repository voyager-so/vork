//! E6.3 variable-font axes on a text run: CSS `font-variation-settings`
//! (`"wght" 650, "wdth" 80`), stored in the run's span style like every other
//! run style. The parser reads it into
//! [`crate::types::InlineStyle::font_variations`]; the vector lane instances
//! the face at those coordinates (`vector::measure::VectorFontStore`, over the
//! catalog variant's own coordinates and the automatic `opsz`), so the drawn
//! outlines, the advances and HarfBuzz shaping all use the same instance.
//!
//! Mirrors `utils/richtext/fontVariations.ts`; keep the two in step.

use crate::vector::varfont::Variation;

fn valid_tag(tag: &str) -> bool {
    tag.len() == 4 && tag.bytes().all(|b| (0x20..=0x7e).contains(&b))
}

/// Parse a CSS `font-variation-settings` value. `normal` (or anything
/// unparseable) is the empty list. Entries are `"tag" <number>`,
/// comma-separated, single or double quotes; a bad entry is skipped and a tag
/// set twice keeps its last value, as in CSS.
pub fn parse_font_variation_settings(value: &str) -> Vec<Variation> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("normal") {
        return Vec::new();
    }
    let mut out: Vec<Variation> = Vec::new();
    for entry in trimmed.split(',') {
        let entry = entry.trim();
        let Some(q) = entry.chars().next().filter(|c| *c == '"' || *c == '\'') else {
            continue;
        };
        let Some(end) = entry[1..].find(q) else {
            continue;
        };
        let tag = &entry[1..1 + end];
        if !valid_tag(tag) {
            continue;
        }
        let Ok(v) = entry[2 + end..].trim().parse::<f32>() else {
            continue;
        };
        if !v.is_finite() {
            continue;
        }
        out.retain(|(t, _)| t != tag);
        out.push((tag.to_string(), v));
    }
    out
}

/// Serialize back to canonical CSS (`"wght" 650, "wdth" 80`); empty → `normal`.
pub fn serialize_font_variation_settings(variations: &[Variation]) -> String {
    if variations.is_empty() {
        return "normal".to_string();
    }
    variations
        .iter()
        .map(|(t, v)| format!("\"{t}\" {v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_serialize() {
        assert!(parse_font_variation_settings("normal").is_empty());
        assert_eq!(
            parse_font_variation_settings(
                r#""wght" 650, 'wdth' 80.5, "bad" 1, "opsz" x, "wght" 700"#
            ),
            vec![("wdth".to_string(), 80.5), ("wght".to_string(), 700.0)]
        );
        assert_eq!(
            serialize_font_variation_settings(&[("wght".into(), 650.0), ("wdth".into(), 80.5)]),
            r#""wght" 650, "wdth" 80.5"#
        );
    }
}
