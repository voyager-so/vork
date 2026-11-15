//! Character attributes of a text run, as Illustrator's Character panel has them (E8.22): baseline shift, horizontal
//! and vertical scale, character rotation, and the kerning method (metrics, optical or none).
//!
//! A run stores them the way it stores its gradient (`crate::run_fill`): as `--nf-*` custom properties in the
//! rich-text HTML's `<span style>` (or on the `<p>` for the whole paragraph), inherited like any other run style:
//!
//! ```text
//! --nf-baseline-shift: 6px          px, positive raises the glyphs (em is accepted and stored as px)
//! --nf-horizontal-scale: 120%       glyph width and advance (1%–10000%; a bare number is a multiple: 1.2)
//! --nf-vertical-scale: 80%          glyph height, from the baseline
//! --nf-character-rotation: 15deg    each glyph about its own centre, counter-clockwise (a bare number is degrees)
//! --nf-kerning: optical             metrics (the font's kerning, the default) | optical | none
//! ```
//!
//! What each one does in the vector lane (`crate::vector`):
//! - kerning `none` shapes with `kern` off; `optical` shapes with `kern` off and then spaces each pair of glyphs from
//!   their outlines ([`optical_kerning_em`]). Both are advance changes, so wrap, alignment, the draw, the layout dump
//!   (caret) and every export agree ([`adjust_shaped`] runs wherever a segment is shaped);
//! - horizontal scale multiplies every advance and offset ([`adjust_shaped`]) and widens the outline;
//! - vertical scale and rotation change the outline only: the line box, the baseline and the advances stay (as in
//!   Illustrator, where neither changes leading or spacing). Both ride the per-glyph transform the renderer already
//!   applies ([`CharacterStyle::glyph_xform`], composed with KineticText and synthetic oblique);
//! - baseline shift moves the glyphs off the line's baseline; the line box and leading stay.
//!
//! A run with the defaults is untouched on every path, so existing text lays out and draws byte-identically.
//!
//! Mirrors `utils/richtext/characterStyle.ts` (parse and serialize); keep the two in step.

use std::borrow::Cow;

use crate::font_features::FontFeature;
use crate::vector::measure::ShapedGlyph;
use crate::vector::otjs::OtjsFont;

pub const BASELINE_SHIFT_PROPERTY: &str = "--nf-baseline-shift";
pub const HORIZONTAL_SCALE_PROPERTY: &str = "--nf-horizontal-scale";
pub const VERTICAL_SCALE_PROPERTY: &str = "--nf-vertical-scale";
pub const CHARACTER_ROTATION_PROPERTY: &str = "--nf-character-rotation";
pub const KERNING_PROPERTY: &str = "--nf-kerning";

/// Illustrator's scale range: 1% to 10000%.
pub const MIN_SCALE: f64 = 0.01;
pub const MAX_SCALE: f64 = 100.0;

/// How a run's glyph pairs are spaced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kerning {
    /// The font's own kerning (GPOS `kern`, or the legacy `kern` table): what every run had before.
    #[default]
    Metrics,
    /// Spacing from the glyph outlines ([`optical_kerning_em`]), the font's kerning off.
    Optical,
    /// No kerning: the bare advances.
    None,
}

impl Kerning {
    pub fn parse(value: &str) -> Option<Kerning> {
        match value.trim().to_ascii_lowercase().as_str() {
            "metrics" | "auto" | "normal" => Some(Kerning::Metrics),
            "optical" => Some(Kerning::Optical),
            "none" | "0" | "off" => Some(Kerning::None),
            _ => None,
        }
    }

    pub fn css(self) -> &'static str {
        match self {
            Kerning::Metrics => "metrics",
            Kerning::Optical => "optical",
            Kerning::None => "none",
        }
    }
}

/// A run's character attributes. [`CharacterStyle::DEFAULT`] is every run without them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterStyle {
    /// px, positive raises the glyphs above the baseline.
    pub baseline_shift: f64,
    /// Multiple of the glyph width and advance (1.0 = 100%).
    pub horizontal_scale: f64,
    /// Multiple of the glyph height, from the baseline (1.0 = 100%).
    pub vertical_scale: f64,
    /// Degrees, counter-clockwise on screen (Illustrator's sign), each glyph about its own centre.
    pub rotation: f64,
    pub kerning: Kerning,
}

impl Default for CharacterStyle {
    fn default() -> Self {
        CharacterStyle::DEFAULT
    }
}

fn split_number(raw: &str) -> Option<(f64, String)> {
    let raw = raw.trim();
    let end = raw
        .char_indices()
        .find(|(i, c)| !(c.is_ascii_digit() || *c == '.' || ((*c == '-' || *c == '+') && *i == 0)))
        .map(|(i, _)| i)
        .unwrap_or(raw.len());
    let n: f64 = raw[..end].parse().ok()?;
    if !n.is_finite() {
        return None;
    }
    Some((n, raw[end..].trim().to_ascii_lowercase()))
}

/// `6px`, `-2`, `0.25em` → px (`em` against `font_size_px`).
pub fn parse_baseline_shift_px(value: &str, font_size_px: f64) -> Option<f64> {
    let (n, unit) = split_number(value)?;
    match unit.as_str() {
        "" | "px" => Some(n),
        "em" => Some(n * font_size_px),
        "pt" => Some(n), // a point is a px (72 per inch)
        _ => None,
    }
}

/// `120%` or `1.2` → 1.2, clamped to [`MIN_SCALE`, `MAX_SCALE`].
pub fn parse_scale(value: &str) -> Option<f64> {
    let (n, unit) = split_number(value)?;
    let m = match unit.as_str() {
        "%" => n / 100.0,
        "" => n,
        _ => return None,
    };
    if m <= 0.0 {
        return None;
    }
    Some(m.clamp(MIN_SCALE, MAX_SCALE))
}

/// `15deg`, `-30`, `0.5turn`, `1rad`, `100grad` → degrees in (-360, 360).
pub fn parse_rotation_deg(value: &str) -> Option<f64> {
    let (n, unit) = split_number(value)?;
    let deg = match unit.as_str() {
        "" | "deg" => n,
        "turn" => n * 360.0,
        "rad" => n.to_degrees(),
        "grad" => n * 0.9,
        _ => return None,
    };
    Some(deg % 360.0)
}

/// Canonical CSS number: no trailing zeros, at most 4 decimals (the TS twin writes the same).
pub fn css_number(n: f64) -> String {
    let r = (n * 10000.0).round() / 10000.0;
    let r = if r == 0.0 { 0.0 } else { r };
    let s = format!("{r:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

impl CharacterStyle {
    pub const DEFAULT: CharacterStyle = CharacterStyle {
        baseline_shift: 0.0,
        horizontal_scale: 1.0,
        vertical_scale: 1.0,
        rotation: 0.0,
        kerning: Kerning::Metrics,
    };

    pub fn is_default(&self) -> bool {
        *self == CharacterStyle::DEFAULT
    }

    /// Whether the glyph outlines change shape (scale or rotation): these need a per-glyph transform in the draw,
    /// and can't be written as plain live text in SVG.
    pub fn transforms_glyphs(&self) -> bool {
        self.horizontal_scale != 1.0 || self.vertical_scale != 1.0 || self.rotation != 0.0
    }

    /// Whether shaping changes (kerning method or horizontal scale).
    pub fn changes_advances(&self) -> bool {
        self.horizontal_scale != 1.0 || self.kerning != Kerning::Metrics
    }

    /// Apply one CSS declaration. `font_size_px` resolves `em`. Returns false when `prop` isn't one of ours; an
    /// invalid value for one of ours is ignored (the inherited value stays), like the browser does.
    pub fn apply_declaration(&mut self, prop: &str, value: &str, font_size_px: f64) -> bool {
        match prop {
            BASELINE_SHIFT_PROPERTY => {
                if let Some(px) = parse_baseline_shift_px(value, font_size_px) {
                    self.baseline_shift = px;
                }
            }
            HORIZONTAL_SCALE_PROPERTY => {
                if let Some(s) = parse_scale(value) {
                    self.horizontal_scale = s;
                }
            }
            VERTICAL_SCALE_PROPERTY => {
                if let Some(s) = parse_scale(value) {
                    self.vertical_scale = s;
                }
            }
            CHARACTER_ROTATION_PROPERTY => {
                if let Some(d) = parse_rotation_deg(value) {
                    self.rotation = d;
                }
            }
            KERNING_PROPERTY => {
                if let Some(k) = Kerning::parse(value) {
                    self.kerning = k;
                }
            }
            _ => return false,
        }
        true
    }

    /// The non-default attributes as CSS declarations (`--nf-baseline-shift: 6px; …`), for exporters and dumps.
    pub fn css_declarations(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if self.baseline_shift != 0.0 {
            out.push((
                BASELINE_SHIFT_PROPERTY,
                format!("{}px", css_number(self.baseline_shift)),
            ));
        }
        if self.horizontal_scale != 1.0 {
            out.push((
                HORIZONTAL_SCALE_PROPERTY,
                format!("{}%", css_number(self.horizontal_scale * 100.0)),
            ));
        }
        if self.vertical_scale != 1.0 {
            out.push((
                VERTICAL_SCALE_PROPERTY,
                format!("{}%", css_number(self.vertical_scale * 100.0)),
            ));
        }
        if self.rotation != 0.0 {
            out.push((
                CHARACTER_ROTATION_PROPERTY,
                format!("{}deg", css_number(self.rotation)),
            ));
        }
        if self.kerning != Kerning::Metrics {
            out.push((KERNING_PROPERTY, self.kerning.css().to_string()));
        }
        out
    }

    /// The per-glyph transform for the scale and rotation, in the renderer's convention (`GlyphXform`: a 2×2 `lin`
    /// applied about the glyph's em-box centre `C`, then `off`): `p ↦ R·S·p + (I − R)·S·C`, i.e. the outline scaled
    /// from the glyph's origin (the pen on the baseline: the left edge keeps the pen, the baseline stays) and then
    /// turned about the scaled glyph's centre. `em_box` is in em (y-down), `font_size_px` its scale.
    /// Identity for a run with neither.
    pub fn glyph_xform(
        &self,
        em_box: [f32; 4],
        font_size_px: f32,
    ) -> crate::vector::ops::GlyphXform {
        use crate::vector::ops::GlyphXform;
        if !self.transforms_glyphs() {
            return GlyphXform::IDENTITY;
        }
        let (sx, sy) = (self.horizontal_scale, self.vertical_scale);
        let (s, c) = self.rotation.to_radians().sin_cos();
        // Counter-clockwise on a y-down screen: x' = c·x + s·y, y' = −s·x + c·y.
        let (ra, rc, rb, rd) = (c, s, -s, c);
        // lin = R·S (row-major [a, c, b, d]: row0 = (a, c), row1 = (b, d)).
        let lin = [ra * sx, rc * sy, rb * sx, rd * sy];
        let cx = (em_box[0] as f64 + em_box[2] as f64) * 0.5 * font_size_px as f64;
        let cy = (em_box[1] as f64 + em_box[3] as f64) * 0.5 * font_size_px as f64;
        // off = (S − I)·C: about C, lin·(p − C) + C + off = R·S·p + (I − R)·S·C.
        let off = [(sx - 1.0) * cx, (sy - 1.0) * cy];
        GlyphXform {
            lin: [lin[0] as f32, lin[1] as f32, lin[2] as f32, lin[3] as f32],
            off: [off[0] as f32, off[1] as f32],
            opacity: 1.0,
        }
    }
}

/// The features a run shapes with: its own, plus `kern` off when it doesn't use the font's kerning (unless the run
/// sets `kern` itself). Borrowed, and unchanged, for every run with metrics kerning.
pub fn shaping_features<'a>(
    features: &'a [FontFeature],
    ch: &CharacterStyle,
) -> Cow<'a, [FontFeature]> {
    if ch.kerning == Kerning::Metrics || features.iter().any(|f| &f.tag == b"kern") {
        return Cow::Borrowed(features);
    }
    let mut out = features.to_vec();
    out.push(FontFeature {
        tag: *b"kern",
        value: 0,
    });
    Cow::Owned(out)
}

/// Optical kerning's reach into open counters (em): a band's gap counts at most this much more than the pair's
/// tightest band, so an open side (the arms of a T, the bowl of a C) pulls the neighbour in only part of the way.
const OPTICAL_DEPTH_EM: f64 = 0.12;
/// Optical kerning never moves a pair more than this (em).
const OPTICAL_LIMIT_EM: f64 = 0.3;
/// Horizontal bands the outlines are sampled in, over the em from 1 em above the baseline to 0.3 em below.
const OPTICAL_BANDS: usize = 40;
const OPTICAL_TOP_EM: f64 = -1.0;
const OPTICAL_BOTTOM_EM: f64 = 0.3;

/// A glyph's horizontal extent in each band (em, pen-relative), `None` where the band has no ink.
fn glyph_profile(font: &OtjsFont, gid: u32) -> Option<Vec<Option<(f64, f64)>>> {
    let rings = font.glyph_em_contours(gid).ok()?;
    if rings.iter().all(|r| r.len() < 2) {
        return None;
    }
    let mut out = Vec::with_capacity(OPTICAL_BANDS);
    let mut any = false;
    for b in 0..OPTICAL_BANDS {
        let y = OPTICAL_TOP_EM
            + (b as f64 + 0.5) * (OPTICAL_BOTTOM_EM - OPTICAL_TOP_EM) / OPTICAL_BANDS as f64;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for ring in rings.iter() {
            let n = ring.len();
            for i in 0..n {
                let a = ring[i];
                let b2 = ring[(i + 1) % n];
                let (y0, y1) = (a[1], b2[1]);
                if (y0 <= y && y < y1) || (y1 <= y && y < y0) {
                    let t = (y - y0) / (y1 - y0);
                    let x = a[0] + t * (b2[0] - a[0]);
                    lo = lo.min(x);
                    hi = hi.max(x);
                }
            }
        }
        if lo.is_finite() && hi.is_finite() {
            any = true;
            out.push(Some((lo, hi)));
        } else {
            out.push(None);
        }
    }
    any.then_some(out)
}

/// The optical gap between two glyphs set side by side at `left_advance_em` (em): the mean, over the bands where
/// both have ink, of each band's white space, each capped at the tightest band plus [`OPTICAL_DEPTH_EM`]. Pairs that
/// share no band (a comma after a T) use their outline extremes. `None` when either glyph has no ink.
fn optical_gap_em(font: &OtjsFont, left: u32, right: u32, left_advance_em: f64) -> Option<f64> {
    let l = glyph_profile(font, left)?;
    let r = glyph_profile(font, right)?;
    let gaps: Vec<f64> = l
        .iter()
        .zip(&r)
        .filter_map(|(a, b)| match (a, b) {
            (Some((_, lhi)), Some((rlo, _))) => Some((left_advance_em - lhi) + rlo),
            _ => None,
        })
        .collect();
    if gaps.is_empty() {
        let lhi = l
            .iter()
            .flatten()
            .map(|p| p.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let rlo = r
            .iter()
            .flatten()
            .map(|p| p.0)
            .fold(f64::INFINITY, f64::min);
        return Some((left_advance_em - lhi) + rlo);
    }
    let min = gaps.iter().copied().fold(f64::INFINITY, f64::min);
    let cap = min + OPTICAL_DEPTH_EM;
    Some(gaps.iter().map(|g| g.min(cap)).sum::<f64>() / gaps.len() as f64)
}

/// The gap a pair is spaced to: the font's own "nn" (its designed lowercase rhythm), else "HH", else 0.08 em.
fn optical_target_em(font: &OtjsFont) -> f64 {
    let upm = if font.units_per_em != 0 {
        font.units_per_em as f64
    } else {
        1000.0
    };
    for ch in ['n', 'H'] {
        let g = font.char_to_glyph_index(ch);
        if g != 0 {
            if let Some(gap) = optical_gap_em(font, g, g, font.advance_width(g) / upm) {
                return gap;
            }
        }
    }
    0.08
}

/// Optical kerning for one pair (em): how much to add to the left glyph's advance so the pair's optical gap
/// ([`optical_gap_em`]) matches the font's own ([`optical_target_em`]), within ±[`OPTICAL_LIMIT_EM`]. Zero for a pair
/// with a blank glyph (a space).
pub fn optical_kerning_em(
    font: &OtjsFont,
    left: u32,
    right: u32,
    left_advance_em: f64,
    target_em: f64,
) -> f64 {
    match optical_gap_em(font, left, right, left_advance_em) {
        Some(gap) => (target_em - gap).clamp(-OPTICAL_LIMIT_EM, OPTICAL_LIMIT_EM),
        None => 0.0,
    }
}

/// Apply a run's advance-changing attributes to its shaped glyphs (font units): optical kerning between clusters,
/// then the horizontal scale on every advance and offset. A no-op for a run with neither, so every caller can run it
/// unconditionally. Shaping must already have used [`shaping_features`] (kerning off for optical and none).
pub fn adjust_shaped(font: &OtjsFont, shaped: &mut [ShapedGlyph], ch: &CharacterStyle) {
    if !ch.changes_advances() {
        return;
    }
    let upm = if font.units_per_em != 0 {
        font.units_per_em as f64
    } else {
        1000.0
    };
    if ch.kerning == Kerning::Optical && shaped.len() > 1 {
        let target = optical_target_em(font);
        for i in 0..shaped.len() - 1 {
            if shaped[i + 1].cluster == shaped[i].cluster {
                continue;
            }
            let (l, r) = (shaped[i].glyph, shaped[i + 1].glyph);
            let k = optical_kerning_em(font, l, r, shaped[i].x_advance / upm, target);
            shaped[i].x_advance += k * upm;
        }
    }
    if ch.horizontal_scale != 1.0 {
        for g in shaped.iter_mut() {
            g.x_advance *= ch.horizontal_scale;
            g.x_offset *= ch.horizontal_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_each_attribute_and_ignores_bad_values() {
        let mut c = CharacterStyle::DEFAULT;
        assert!(c.apply_declaration(BASELINE_SHIFT_PROPERTY, "6px", 20.0));
        assert_eq!(c.baseline_shift, 6.0);
        c.apply_declaration(BASELINE_SHIFT_PROPERTY, "-0.25em", 20.0);
        assert_eq!(c.baseline_shift, -5.0);
        c.apply_declaration(BASELINE_SHIFT_PROPERTY, "lots", 20.0);
        assert_eq!(
            c.baseline_shift, -5.0,
            "an invalid value keeps the inherited one"
        );
        c.apply_declaration(HORIZONTAL_SCALE_PROPERTY, "120%", 20.0);
        assert!((c.horizontal_scale - 1.2).abs() < 1e-12);
        c.apply_declaration(VERTICAL_SCALE_PROPERTY, "0.5", 20.0);
        assert_eq!(c.vertical_scale, 0.5);
        c.apply_declaration(VERTICAL_SCALE_PROPERTY, "0%", 20.0);
        assert_eq!(c.vertical_scale, 0.5, "zero scale is refused");
        c.apply_declaration(HORIZONTAL_SCALE_PROPERTY, "50000%", 20.0);
        assert_eq!(c.horizontal_scale, MAX_SCALE);
        c.apply_declaration(CHARACTER_ROTATION_PROPERTY, "375deg", 20.0);
        assert_eq!(c.rotation, 15.0);
        c.apply_declaration(KERNING_PROPERTY, "Optical", 20.0);
        assert_eq!(c.kerning, Kerning::Optical);
        assert!(!c.apply_declaration("letter-spacing", "2px", 20.0));
    }

    #[test]
    fn css_round_trips_the_non_defaults() {
        assert!(CharacterStyle::DEFAULT.css_declarations().is_empty());
        let c = CharacterStyle {
            baseline_shift: -3.5,
            horizontal_scale: 1.25,
            vertical_scale: 0.8,
            rotation: 12.5,
            kerning: Kerning::None,
        };
        let decls = c.css_declarations();
        assert_eq!(
            decls,
            vec![
                (BASELINE_SHIFT_PROPERTY, "-3.5px".to_string()),
                (HORIZONTAL_SCALE_PROPERTY, "125%".to_string()),
                (VERTICAL_SCALE_PROPERTY, "80%".to_string()),
                (CHARACTER_ROTATION_PROPERTY, "12.5deg".to_string()),
                (KERNING_PROPERTY, "none".to_string()),
            ]
        );
        let mut back = CharacterStyle::DEFAULT;
        for (p, v) in &decls {
            back.apply_declaration(p, v, 16.0);
        }
        assert_eq!(back, c);
    }

    #[test]
    fn glyph_xform_scales_from_the_origin_and_turns_about_the_centre() {
        // A 0.5 × 0.7 em box above the baseline at 100 px: centre (25, -35) px from the pen.
        let em_box = [0.0f32, -0.7, 0.5, 0.0];
        let apply = |x: &crate::vector::ops::GlyphXform, p: (f64, f64)| {
            let (cx, cy) = (25.0, -35.0);
            let (dx, dy) = (p.0 - cx, p.1 - cy);
            let [a, c, b, d] = x.lin.map(|v| v as f64);
            (
                a * dx + c * dy + cx + x.off[0] as f64,
                b * dx + d * dy + cy + x.off[1] as f64,
            )
        };
        let scaled = CharacterStyle {
            horizontal_scale: 2.0,
            vertical_scale: 0.5,
            ..CharacterStyle::DEFAULT
        };
        let x = scaled.glyph_xform(em_box, 100.0);
        // The origin stays; the top-right corner (50, -70) goes to (100, -35).
        let o = apply(&x, (0.0, 0.0));
        assert!(o.0.abs() < 1e-4 && o.1.abs() < 1e-4, "{o:?}");
        let tr = apply(&x, (50.0, -70.0));
        assert!(
            (tr.0 - 100.0).abs() < 1e-4 && (tr.1 + 35.0).abs() < 1e-4,
            "{tr:?}"
        );
        // 90° counter-clockwise: the right-hand midpoint (50, -35) goes to the top (25, -60).
        let turned = CharacterStyle {
            rotation: 90.0,
            ..CharacterStyle::DEFAULT
        };
        let r = apply(&turned.glyph_xform(em_box, 100.0), (50.0, -35.0));
        assert!(
            (r.0 - 25.0).abs() < 1e-4 && (r.1 + 60.0).abs() < 1e-4,
            "{r:?}"
        );
        assert!(CharacterStyle::DEFAULT
            .glyph_xform(em_box, 100.0)
            .is_identity());
    }

    #[test]
    fn shaping_features_turn_kern_off_only_when_asked() {
        let feats = vec![FontFeature {
            tag: *b"smcp",
            value: 1,
        }];
        assert!(matches!(
            shaping_features(&feats, &CharacterStyle::DEFAULT),
            Cow::Borrowed(_)
        ));
        let none = CharacterStyle {
            kerning: Kerning::None,
            ..CharacterStyle::DEFAULT
        };
        let out = shaping_features(&feats, &none);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[1],
            FontFeature {
                tag: *b"kern",
                value: 0
            }
        );
        let own = vec![FontFeature {
            tag: *b"kern",
            value: 1,
        }];
        assert_eq!(
            shaping_features(&own, &none).len(),
            1,
            "the run's own kern setting wins"
        );
    }
}
