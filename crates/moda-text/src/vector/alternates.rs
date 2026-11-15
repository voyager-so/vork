//! The glyph-alternates query (E8.22, Illustrator's Glyphs panel "alternates for the selected glyph"): which other
//! glyphs a font offers for a character, and the OpenType feature that picks each one, so an agent or a panel can
//! swap it by setting that feature on the character (`font-feature-settings`, `crate::font_features`).
//!
//! The font's GSUB features are tried one at a time on the text, the way the shaper will apply them: a feature that
//! turns the text's glyphs into different ones is an alternate (`swsh` 1 → `Q.swsh`). A feature that takes an index
//! (`salt`, `swsh`, `cvXX` over an alternate lookup) is tried at 1, 2, 3, … until it stops giving new glyphs. The
//! shaping-internal features (`ccmp`, `locl`, `rlig`, …), the default-on ones (`liga`, `calt`) and `aalt` (a
//! catalogue of the others) are not tried.

use crate::catalog::Catalog;
use crate::font_features::FontFeature;
use crate::vector::hb::shape_features_hb;
use crate::vector::measure::VectorFontStore;

/// Features that aren't a user's choice of glyph.
const SKIPPED: &[&[u8; 4]] = &[
    b"aalt", b"ccmp", b"locl", b"rlig", b"rvrn", b"liga", b"clig", b"calt", b"kern", b"mark",
    b"mkmk", b"abvs", b"blws", b"abvf", b"blwf", b"akhn", b"rphf", b"pref", b"half", b"pstf",
    b"vatu", b"cjct", b"nukt", b"pres", b"psts", b"haln", b"curs", b"dist", b"ljmo", b"vjmo",
    b"tjmo", b"vert", b"vrt2", b"rtlm", b"ltra", b"ltrm", b"rtla", b"isol",
];

/// Highest index tried for an indexed feature.
const MAX_INDEX: u32 = 20;

/// One alternate: the feature (and value) that selects it, and the glyphs it gives.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphAlternate {
    pub feature: String,
    pub value: u32,
    pub glyph_ids: Vec<u32>,
    pub glyph_names: Vec<Option<String>>,
    /// The glyphs' outline, em × 1000, y-down from the baseline (an SVG path `d`), for a panel's preview.
    pub path: String,
    /// The glyphs' advance, em × 1000.
    pub advance: f64,
}

/// The text's default glyphs (with the run's `features`) and every alternate the font offers for it.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphAlternates {
    pub default: GlyphAlternate,
    pub alternates: Vec<GlyphAlternate>,
}

fn feature_rank(tag: &[u8; 4]) -> (u8, [u8; 4]) {
    let rank = match tag {
        b"swsh" | b"cswh" => 0,
        b"salt" => 1,
        [b's', b's', _, _] => 2,
        [b'c', b'v', _, _] => 3,
        _ => 4,
    };
    (rank, *tag)
}

fn outline_path(
    font: &crate::vector::otjs::OtjsFont,
    glyphs: &[crate::vector::measure::ShapedGlyph],
) -> (String, f64) {
    let upm = if font.units_per_em != 0 {
        font.units_per_em as f64
    } else {
        1000.0
    };
    let mut d = String::new();
    let mut pen = 0.0f64;
    for g in glyphs {
        let x0 = pen + g.x_offset / upm;
        let y0 = -g.y_offset / upm;
        if let Ok(rings) = font.glyph_em_contours(g.glyph) {
            for ring in rings.iter().filter(|r| r.len() > 1) {
                for (i, p) in ring.iter().enumerate() {
                    let cmd = if i == 0 { 'M' } else { 'L' };
                    d.push_str(&format!(
                        "{cmd}{} {}",
                        crate::character::css_number(((x0 + p[0]) * 1000.0 * 10.0).round() / 10.0),
                        crate::character::css_number(((y0 + p[1]) * 1000.0 * 10.0).round() / 10.0)
                    ));
                }
                d.push('Z');
            }
        }
        pen += g.x_advance / upm;
    }
    (d, pen * 1000.0)
}

/// Every alternate `font` offers for `text` (usually one character) at `family` / `weight` / `style`, starting from
/// the run's own `features`. `None` when the variant has no font or no shaping face.
pub fn glyph_alternates(
    store: &VectorFontStore,
    catalog: &Catalog,
    family: &str,
    weight: i64,
    style: &str,
    text: &str,
    features: &[FontFeature],
) -> Option<GlyphAlternates> {
    let size: Option<f64> = None;
    let font = store.font_for_variant(catalog, family, weight, style, size)?;
    let hb = store.hb_face_for_variant(catalog, family, weight, style, size)?;
    let describe = |feature: &str, value: u32, shaped: &[crate::vector::measure::ShapedGlyph]| {
        let (path, advance) = outline_path(&font, shaped);
        GlyphAlternate {
            feature: feature.to_string(),
            value,
            glyph_ids: shaped.iter().map(|g| g.glyph).collect(),
            glyph_names: shaped.iter().map(|g| hb.glyph_name(g.glyph)).collect(),
            path,
            advance,
        }
    };
    let base = shape_features_hb(&hb, &font, text, features).ok()?;
    let base_ids: Vec<u32> = base.iter().map(|g| g.glyph).collect();
    let default = describe("", 0, &base);
    let mut seen: Vec<Vec<u32>> = vec![base_ids.clone()];
    let mut tags: Vec<[u8; 4]> = hb
        .gsub_feature_tags()
        .into_iter()
        .filter(|t| !SKIPPED.contains(&t))
        .collect();
    tags.sort_by_key(feature_rank);
    let mut alternates = Vec::new();
    for tag in tags {
        let mut prev: Option<Vec<u32>> = None;
        for value in 1..=MAX_INDEX {
            let mut feats: Vec<FontFeature> =
                features.iter().filter(|f| f.tag != tag).copied().collect();
            feats.push(FontFeature { tag, value });
            let Ok(shaped) = shape_features_hb(&hb, &font, text, &feats) else {
                break;
            };
            let ids: Vec<u32> = shaped.iter().map(|g| g.glyph).collect();
            // No change, or the index ran past the lookup's alternates (it repeats the last, or falls back).
            if ids == base_ids || prev.as_ref() == Some(&ids) {
                break;
            }
            if !seen.contains(&ids) {
                seen.push(ids.clone());
                let name = std::str::from_utf8(&tag).unwrap_or("????");
                alternates.push(describe(name, value, &shaped));
            }
            prev = Some(ids);
        }
    }
    Some(GlyphAlternates {
        default,
        alternates,
    })
}

impl GlyphAlternate {
    pub fn to_json(&self) -> serde_json::Value {
        let mut v = serde_json::json!({
            "glyphIds": self.glyph_ids,
            "glyphNames": self.glyph_names,
            "advance": self.advance,
            "path": self.path,
        });
        if !self.feature.is_empty() {
            v["feature"] = serde_json::json!(self.feature);
            v["value"] = serde_json::json!(self.value);
            // The markup spelling (`font-features`): `swsh`, `salt=2`.
            v["fontFeatures"] = serde_json::json!(if self.value == 1 {
                self.feature.clone()
            } else {
                format!("{}={}", self.feature, self.value)
            });
        }
        v
    }
}

impl GlyphAlternates {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "default": self.default.to_json(),
            "alternates": self.alternates.iter().map(GlyphAlternate::to_json).collect::<Vec<_>>(),
        })
    }
}
