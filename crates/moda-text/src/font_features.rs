//! OpenType feature settings on a text run (F7.3).
//!
//! A run stores its features the way it stores every other run style: in the
//! rich-text HTML, as the CSS `font-feature-settings` property on the run's
//! `<span style>` (`"smcp" 1, "ss01" 1, "liga" 0`). The browser's editing DOM
//! renders that natively, SVG export copies it, and here the parser reads it
//! into [`crate::types::InlineStyle::font_features`], which the vector lane
//! hands to rustybuzz (`vector::hb::shape_features_hb`).
//!
//! Mirrors `utils/richtext/fontFeatures.ts` (`parseFontFeatureSettings`,
//! `serializeFontFeatureSettings`); keep the two in step.

/// One `font-feature-settings` entry: a four-byte OpenType tag and its value
/// (0 = off, 1 = on, n = alternate index for `salt`/`cvXX`/`swsh`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontFeature {
    pub tag: [u8; 4],
    pub value: u32,
}

impl FontFeature {
    pub fn tag_str(&self) -> &str {
        std::str::from_utf8(&self.tag).unwrap_or("????")
    }
}

fn is_tag_char(b: u8) -> bool {
    // OpenType tags are printable ASCII (0x20..=0x7E); CSS requires exactly
    // four of them inside the quotes.
    (0x20..=0x7e).contains(&b)
}

/// Parse a CSS `font-feature-settings` value. `normal` (or anything
/// unparseable) is the empty list. Each entry is `"tag"` (value 1), `"tag" on`,
/// `"tag" off` or `"tag" <integer>`, comma-separated; single or double quotes.
/// An invalid entry is skipped rather than failing the whole declaration (the
/// browser drops the whole declaration; skipping is kinder to hand-written
/// markup and never changes a valid one). A tag set twice keeps the last value,
/// as in CSS.
pub fn parse_font_feature_settings(value: &str) -> Vec<FontFeature> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("normal") {
        return Vec::new();
    }
    let mut out: Vec<FontFeature> = Vec::new();
    for entry in trimmed.split(',') {
        let entry = entry.trim();
        let bytes = entry.as_bytes();
        let Some(&q) = bytes.first() else { continue };
        if q != b'"' && q != b'\'' {
            continue;
        }
        if bytes.len() < 6 || bytes[5] != q {
            continue;
        }
        let tag_bytes = &bytes[1..5];
        if !tag_bytes.iter().all(|b| is_tag_char(*b)) {
            continue;
        }
        let rest = entry[6..].trim();
        let value = if rest.is_empty() || rest.eq_ignore_ascii_case("on") {
            Some(1)
        } else if rest.eq_ignore_ascii_case("off") {
            Some(0)
        } else {
            rest.parse::<u32>().ok()
        };
        let Some(value) = value else { continue };
        let tag = [tag_bytes[0], tag_bytes[1], tag_bytes[2], tag_bytes[3]];
        out.retain(|f| f.tag != tag);
        out.push(FontFeature { tag, value });
    }
    out
}

/// Serialize back to CSS (`"smcp" 1, "liga" 0`); empty → `normal`.
pub fn serialize_font_feature_settings(features: &[FontFeature]) -> String {
    if features.is_empty() {
        return "normal".to_string();
    }
    features
        .iter()
        .map(|f| format!("\"{}\" {}", f.tag_str(), f.value))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The feature tags a font's GSUB table declares (any script), deduplicated.
pub fn gsub_feature_tags(bytes: &[u8]) -> Vec<[u8; 4]> {
    let Ok(face) = rustybuzz::ttf_parser::Face::parse(bytes, 0) else {
        return Vec::new();
    };
    let mut tags: Vec<[u8; 4]> = Vec::new();
    if let Some(gsub) = face.tables().gsub {
        for feature in gsub.features {
            let t = feature.tag.to_bytes();
            if !tags.contains(&t) {
                tags.push(t);
            }
        }
    }
    tags
}

/// The feature tags a font's GSUB and GPOS tables declare (any script,
/// deduplicated, sorted). The Type details popover lists only these.
pub fn font_feature_tags(bytes: &[u8]) -> Vec<String> {
    let Ok(face) = rustybuzz::ttf_parser::Face::parse(bytes, 0) else {
        return Vec::new();
    };
    let mut tags: Vec<String> = Vec::new();
    for table in [face.tables().gsub, face.tables().gpos]
        .into_iter()
        .flatten()
    {
        for feature in table.features {
            let t = feature.tag.to_bytes();
            if let Ok(s) = std::str::from_utf8(&t) {
                if !tags.iter().any(|x| x == s) {
                    tags.push(s.to_string());
                }
            }
        }
    }
    tags.sort();
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(tag: &str, value: u32) -> FontFeature {
        let b = tag.as_bytes();
        FontFeature {
            tag: [b[0], b[1], b[2], b[3]],
            value,
        }
    }

    #[test]
    fn parses_css_forms() {
        assert_eq!(parse_font_feature_settings("normal"), vec![]);
        assert_eq!(parse_font_feature_settings(""), vec![]);
        assert_eq!(
            parse_font_feature_settings(r#""smcp", 'ss01' 1, "liga" off, "tnum" on, "salt" 3"#),
            vec![
                f("smcp", 1),
                f("ss01", 1),
                f("liga", 0),
                f("tnum", 1),
                f("salt", 3)
            ]
        );
    }

    #[test]
    fn skips_bad_entries_and_keeps_the_last_duplicate() {
        assert_eq!(
            parse_font_feature_settings(r#"smcp, "toolong" 1, "tnum" x, "liga" 0, "liga" 1"#),
            vec![f("liga", 1)]
        );
    }

    #[test]
    fn serializes_round_trip() {
        let feats = vec![f("smcp", 1), f("liga", 0)];
        let css = serialize_font_feature_settings(&feats);
        assert_eq!(css, r#""smcp" 1, "liga" 0"#);
        assert_eq!(parse_font_feature_settings(&css), feats);
        assert_eq!(serialize_font_feature_settings(&[]), "normal");
    }

    #[test]
    fn lists_inter_features() {
        let bytes =
            std::fs::read(crate::test_fixtures::font_path("inter/Inter_24pt-Regular.ttf"))
                .unwrap();
        let tags = font_feature_tags(&bytes);
        for want in ["kern", "dlig", "tnum", "zero", "frac", "ss01"] {
            assert!(
                tags.iter().any(|t| t == want),
                "{want} missing from {tags:?}"
            );
        }
        assert!(!tags.iter().any(|t| t == "smcp"), "{tags:?}");
        assert!(font_feature_tags(b"not a font").is_empty());
    }
}
