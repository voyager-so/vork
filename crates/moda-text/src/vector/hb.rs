//! Phase 2 hybrid shaping (FIX r4, task #128, quirk Q-122): the
//! `Canvas2dParity` lane's HarfBuzz-class shaper for `letterSpacing == 0`
//! runs — rustybuzz behind the same `ShapedGlyph` seam the opentype path
//! emits into.
//!
//! WHY: the hybrid arm's comparison target is the default-config golden,
//! where fallback-class nodes are Canvas2D rasters. Chromium's Canvas2D
//! `fillText`/`measureText` shape with HarfBuzz — kerning through EVERY GPOS
//! lookup shape (incl. the type-9 extension lookups opentype.js is blind to,
//! `font.js:252-261` — the d31a991e p3 drift) plus the default GSUB feature
//! set (`liga`, `calt`, `ccmp`, … — the 11cf37a6 `bracketleft.case` calt the
//! otjs lane never applies), and they resolve UAX #9 bidi per drawn string
//! with the canvas's LTR base direction. The golden's fallback raster draws
//! ONE `fillText` per layout segment (`utils/canvas2dRichTextRenderer.ts`
//! via `drawTextWithLetterSpacing`'s ls==0 branch), so the parity seam is
//! PER-SEGMENT shaping. (The Canvas2D engine now also reorders segments at
//! line assembly for RTL lines — `computeVisualLine` in
//! `utils/richtext/bidiLayout.ts`, LTR base per its "base direction is
//! always LTR" note — but no fixture golden carries RTL content (Q-118), and
//! complex-script nodes decline to the Degraded lane before reaching this
//! hybrid arm, so the arm stays per-segment.)
//!
//! Letter-spaced runs stay on the Q-121 per-char lane (the Canvas2D pipeline
//! draws those one `fillText` per character — unkerned, no cross-char GSUB);
//! `--text-engine vector` (`VectorParity`) never reaches this module.
//!
//! Oracle mirror: `dump-linebox.mjs --engine hybrid` shapes the same
//! segments through harfbuzzjs (the browser pipeline's own WASM shaper,
//! `renderers/webgpu/text/vector/harfbuzzShaper.ts::shapeWithHarfBuzz`) —
//! rustybuzz and HarfBuzz agree on these fonts to well below the gate's
//! 1e-3 px tolerance (advances are integer font units).

#[cfg(test)]
use std::rc::Rc;

use crate::font_features::FontFeature;
use crate::vector::measure::{normalize_text_for_shaping, ShapedGlyph};
use crate::vector::otjs::{OtjsFont, ShapeError};

/// A rustybuzz face bound to the font bytes that back it. The face borrows
/// from the `std::sync::Arc<Vec<u8>>` heap buffer; the struct keeps that buffer alive for
/// exactly as long as the face, making the `'static` lifetime extension
/// below sound (see SAFETY).
pub struct HbFace {
    face: rustybuzz::Face<'static>,
    /// Order matters: `face` (above) drops before the buffer it borrows.
    _bytes: std::sync::Arc<Vec<u8>>,
}

impl HbFace {
    /// Build a shaping face over the SAME vendored bytes the otjs parse used
    /// (mirror of `harfbuzzShaper.ts::getHbFont` over
    /// `registerShapingFontData`'s binary). `None` mirrors the browser's
    /// failed-face cache entry — callers fall back to the opentype path.
    pub fn from_bytes(bytes: std::sync::Arc<Vec<u8>>) -> Option<HbFace> {
        Self::from_bytes_with_variations(bytes, &[])
    }

    /// Desktop fork: the same face at a variable-font instance (the
    /// coordinates the otjs font for this variant was parsed at).
    pub fn from_bytes_with_variations(
        bytes: std::sync::Arc<Vec<u8>>,
        variations: &[crate::vector::varfont::Variation],
    ) -> Option<HbFace> {
        // SAFETY: `face` borrows the heap allocation owned by `bytes`. The
        // allocation is stable (Arc — never moved, never mutated; no &mut API
        // exists on the stored buffer) and outlives `face` because `_bytes`
        // is a field of the same struct declared AFTER `face` (struct fields
        // drop in declaration order). The slice is never handed out with the
        // extended lifetime.
        let slice: &'static [u8] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) };
        let mut face = rustybuzz::Face::from_slice(slice, 0)?;
        if !variations.is_empty() {
            let vars: Vec<rustybuzz::Variation> = variations
                .iter()
                .filter(|(tag, _)| tag.len() == 4)
                .map(|(tag, value)| rustybuzz::Variation {
                    tag: rustybuzz::ttf_parser::Tag::from_bytes_lossy(tag.as_bytes()),
                    value: *value,
                })
                .collect();
            face.set_variations(&vars);
        }
        Some(HbFace {
            face,
            _bytes: bytes,
        })
    }
}

impl HbFace {
    /// E8.22 (the glyph-alternates query): the GSUB feature tags the font declares, deduplicated, in table order.
    pub fn gsub_feature_tags(&self) -> Vec<[u8; 4]> {
        let mut out: Vec<[u8; 4]> = Vec::new();
        if let Some(gsub) = self.face.tables().gsub {
            for feature in gsub.features {
                let tag = feature.tag.to_bytes();
                if !out.contains(&tag) {
                    out.push(tag);
                }
            }
        }
        out
    }

    /// The glyph's name (`post` or CFF), when the font has one.
    pub fn glyph_name(&self, gid: u32) -> Option<String> {
        let gid = u16::try_from(gid).ok()?;
        self.face
            .glyph_name(rustybuzz::ttf_parser::GlyphId(gid))
            .map(str::to_string)
    }
}

/// rustybuzz emits UTF-8 BYTE-offset clusters; KineticText addresses
/// `unitTransforms` by CODE-POINT index, so both HB shapers convert through this.
/// Returns the SORTED byte offset of each code point (index `i` = code point `i`);
/// `byte_offsets.binary_search(&byte)` then yields the code-point index in
/// O(log n) with no hashing. A ligature survivor's first-component byte maps to
/// its first code point. Layout-neutral: byte offsets are strictly monotonic on
/// char boundaries, so `place_glyphs` letter-spacing (cluster inequality only) is
/// unchanged. A non-boundary byte (never emitted by rustybuzz) → `binary_search`
/// Err → `.unwrap_or(len)` past-end → the mapper reads identity.
fn byte_to_codepoint_clusters(text: &str) -> Vec<usize> {
    text.char_indices().map(|(byte, _)| byte).collect()
}

/// Shape one layout segment the way Chromium's Canvas2D `fillText` does
/// (HYBRID arm, `letterSpacing == 0` only — Q-122):
///
/// 1. whitespace normalization shared with the opentype lane
///    (`normalizeTextForShaping` — tab → space, unmapped whitespace →
///    space; Chromium's canvas text-preparation replaces ASCII whitespace
///    with U+0020 the same way);
/// 2. UAX #9 resolution over the segment text with an LTR base level (the
///    canvas `direction` is 'inherit' → ltr on the export surface — one
///    `fillText` call is one bidi paragraph);
/// 3. each level run shaped by rustybuzz in VISUAL order with the run's
///    direction set explicitly (script/language guessed from content —
///    `harfbuzzShaper.ts:170-172` semantics) and the font's DEFAULT feature
///    set (empty feature list == HB defaults: kern/liga/calt/ccmp/…).
///
/// Glyphs come back in visual order with font-unit advances and GPOS
/// offsets; `place_glyphs` consumes them unchanged (ls is 0 on this lane, so
/// cluster boundaries are inert). An out-of-range glyph id mirrors the
/// browser shaper's "HB face and opentype font disagree" guard
/// (`shapeWithHarfBuzz`, harfbuzzShaper.ts:178-181) — there the run falls
/// back to the opentype path; here it unwinds as a `ShapeError` (the node
/// declines to the Degraded tier, engine-truth conservative).
pub fn shape_canvas2d_hb(
    hb: &HbFace,
    font: &OtjsFont,
    text: &str,
) -> Result<Vec<ShapedGlyph>, ShapeError> {
    shape_features_hb(hb, font, text, &[])
}

/// F7.3: [`shape_canvas2d_hb`] with a run's `font-feature-settings` applied on
/// top of HarfBuzz's defaults (a `liga` 0 turns a default off; `smcp` 1 turns
/// an optional one on), each over the whole run. Every lane routes a run that
/// carries features here (`measure::shape_featured`), so the draw, the measure
/// and the text-geometry export all see the same glyphs. An empty list is
/// exactly `shape_canvas2d_hb`.
pub fn shape_features_hb(
    hb: &HbFace,
    font: &OtjsFont,
    text: &str,
    features: &[FontFeature],
) -> Result<Vec<ShapedGlyph>, ShapeError> {
    let hb_features: Vec<rustybuzz::Feature> = features
        .iter()
        .map(|f| {
            rustybuzz::Feature::new(rustybuzz::ttf_parser::Tag::from_bytes(&f.tag), f.value, ..)
        })
        .collect();
    let norm = normalize_text_for_shaping(font, text);
    if norm.is_empty() {
        return Ok(Vec::new());
    }
    // Byte → code-point clusters, so a kinetic node that reaches this lane via
    // the Hybrid VectorParity-decline fallback maps `unitTransforms` correctly
    // for non-ASCII text (ls==0 here, so this is layout-inert either way).
    let cp_index = byte_to_codepoint_clusters(&norm);
    let mut out: Vec<ShapedGlyph> = Vec::new();
    for (run, rtl) in visual_runs_ltr_base(&norm) {
        let mut buf = rustybuzz::UnicodeBuffer::new();
        buf.push_str(&norm[run.clone()]);
        buf.set_direction(if rtl {
            rustybuzz::Direction::RightToLeft
        } else {
            rustybuzz::Direction::LeftToRight
        });
        // Fills only the UNSET properties (script, language) — the
        // direction above is the bidi pass's, not HB's guess, matching
        // the browser shaper (harfbuzzShaper.ts:168-172).
        buf.guess_segment_properties();
        let shaped = rustybuzz::shape(&hb.face, &hb_features, buf);
        let infos = shaped.glyph_infos();
        let positions = shaped.glyph_positions();
        for (info, pos) in infos.iter().zip(positions) {
            if info.glyph_id >= u32::from(font.num_glyphs) {
                return Err(ShapeError::Other(format!(
                    "harfbuzz glyph id {} out of range (numGlyphs {})",
                    info.glyph_id, font.num_glyphs
                )));
            }
            out.push(ShapedGlyph {
                glyph: info.glyph_id,
                x_advance: f64::from(pos.x_advance),
                x_offset: f64::from(pos.x_offset),
                // HB y_offset is y-up font units — same convention as
                // `ShapedGlyph::y_offset` (glyphPlacement.ts:13-28).
                y_offset: f64::from(pos.y_offset),
                // Per-run byte offset rebased to the segment, then mapped to a
                // code-point index (KineticText). Inert for letter-spacing (ls==0).
                cluster: cp_index
                    .binary_search(&(run.start + info.cluster as usize))
                    .unwrap_or(cp_index.len()),
            });
        }
    }
    Ok(out)
}

/// The VECTOR lane's HarfBuzz reroute for otjs-incompatible fonts
/// (Lato's type-7 GSUB extension lookups — the exact `lookupType: 7` throw
/// that used to decline these nodes whole to MSDF). Mirror of the browser
/// dispatch (`VectorGlyphShaper.shape` → `shapeWithHarfBuzz`): normalize with
/// the shared whitespace rules, shape the WHOLE run in ONE buffer with an
/// explicit LTR direction (script/language guessed from content) and the
/// font's default feature set. No bidi split — the vector lane is LTR-only
/// here (complex/RTL chunks fail coverage upstream, Q-118), matching the
/// browser where the run's direction is resolved BEFORE shaping and Latin
/// runs arrive `ltr`. Out-of-range glyph ids unwind as `ShapeError`, same as
/// `shape_canvas2d_hb`.
pub fn shape_vector_hb(
    hb: &HbFace,
    font: &OtjsFont,
    text: &str,
) -> Result<Vec<ShapedGlyph>, ShapeError> {
    let norm = normalize_text_for_shaping(font, text);
    if norm.is_empty() {
        return Ok(Vec::new());
    }
    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str(&norm);
    buf.set_direction(rustybuzz::Direction::LeftToRight);
    // Fills only the UNSET properties (script, language) — the direction
    // above is explicit, matching shapeWithHarfBuzz (harfbuzzShaper.ts).
    buf.guess_segment_properties();
    let shaped = rustybuzz::shape(&hb.face, &[], buf);
    let infos = shaped.glyph_infos();
    let positions = shaped.glyph_positions();
    // Byte → code-point clusters (KineticText): a ligature survivor's
    // first-component byte → its first code point. Layout-neutral (ls==0 here).
    let cp_index = byte_to_codepoint_clusters(&norm);
    let mut out: Vec<ShapedGlyph> = Vec::with_capacity(infos.len());
    for (info, pos) in infos.iter().zip(positions) {
        if info.glyph_id >= u32::from(font.num_glyphs) {
            return Err(ShapeError::Other(format!(
                "harfbuzz glyph id {} out of range (numGlyphs {})",
                info.glyph_id, font.num_glyphs
            )));
        }
        out.push(ShapedGlyph {
            glyph: info.glyph_id,
            x_advance: f64::from(pos.x_advance),
            x_offset: f64::from(pos.x_offset),
            y_offset: f64::from(pos.y_offset),
            // Byte cluster → code-point index (fallback past-end for a non-boundary
            // cluster, which rustybuzz never emits → the mapper reads identity).
            cluster: cp_index
                .binary_search(&(info.cluster as usize))
                .unwrap_or(cp_index.len()),
        });
    }
    Ok(out)
}

/// UAX #9 resolution of one drawn string with an LTR base level — the bidi
/// pass Chromium runs inside a single `fillText`/`measureText` call on the
/// (dir-less, ltr) export surface. Returns the level runs in VISUAL order,
/// each tagged rtl. Font-free so the run split itself is unit-pinnable (no
/// vendored font covers an RTL script — Q-118's coverage note still holds
/// for the fixture catalog).
pub(crate) fn visual_runs_ltr_base(text: &str) -> Vec<(std::ops::Range<usize>, bool)> {
    let bidi = unicode_bidi::BidiInfo::new(text, Some(unicode_bidi::Level::ltr()));
    let mut out = Vec::new();
    // Segment text is paragraph-free by construction (sanitize + paragraph
    // split upstream), but iterate defensively — same output either way.
    for para in &bidi.paragraphs {
        let (levels, runs) = bidi.visual_runs(para, para.range.clone());
        for run in runs {
            let rtl = levels[run.start].is_rtl();
            out.push((run, rtl));
        }
    }
    out
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use crate::test_fixtures::font_path;

    fn load(rel: &str) -> (Rc<OtjsFont>, HbFace) {
        load_from(&font_path(rel))
    }

    fn load_from(path: &std::path::Path) -> (Rc<OtjsFont>, HbFace) {
        let bytes = std::sync::Arc::new(std::fs::read(path).unwrap());
        let font = Rc::new(OtjsFont::parse(&bytes).unwrap());
        let hb = HbFace::from_bytes(bytes).unwrap();
        (font, hb)
    }

    /// Q-122 pin 1 — extension-lookup kerning (the d31a991e p3 mechanism):
    /// Lato routes ALL kern pairs through GPOS type-9 extension lookups,
    /// invisible to the otjs lane (`get_kerning_value` direct-type-2 only).
    /// HB must kern A/V; the otjs lane must (still) not — that blindness is
    /// browser-vector truth, pinned by `vector_oracle_parity`.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn lato_extension_kerning_visible_to_hb_invisible_to_otjs() {
        let (font, hb) = load("lato/Lato-Regular.ttf");
        let hb_shaped = shape_canvas2d_hb(&hb, &font, "AV").unwrap();
        assert_eq!(hb_shaped.len(), 2);
        let a_gid = font.char_to_glyph_index('A');
        assert_eq!(hb_shaped[0].glyph, a_gid);
        let bare = font.advance_width(a_gid);
        let kerned = hb_shaped[0].x_advance;
        assert!(
            kerned < bare - 50.0,
            "extension kern must fold into A's advance: hb {kerned} vs bare {bare}"
        );
        // otjs: the type-9 lookups are invisible — kerning value is 0.
        assert_eq!(
            font.get_kerning_value(a_gid, hb_shaped[1].glyph).unwrap(),
            0.0
        );
    }

    /// Q-122 pin 2 — default GSUB (`liga`): Lato's liga lives in type-7
    /// extension substs, the exact `lookupType: 7` otjs throw that used to
    /// decline these nodes; HB applies it ("ffi" → one ligature glyph).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn lato_liga_applies_where_otjs_threw() {
        let (font, hb) = load("lato/Lato-Regular.ttf");
        let shaped = shape_canvas2d_hb(&hb, &font, "Office").unwrap();
        assert!(
            shaped.len() < "Office".chars().count(),
            "ffi must ligate: got {} glyphs",
            shaped.len()
        );
        // The opentype lane still throws — eligibility comes from HB now.
        assert!(crate::vector::measure::shape(&font, "Office").is_err());
    }

    /// Q-122 pin 3 — `calt` (the 11cf37a6 p0 mechanism class): Inter's
    /// contextual alternates fire on "3x9" under default features.
    #[test]
    fn inter_calt_contextual_alternates_apply() {
        let (font, hb) = load("inter/Inter_24pt-Regular.ttf");
        let plain: Vec<u32> = "3x9".chars().map(|c| font.char_to_glyph_index(c)).collect();
        let shaped = shape_canvas2d_hb(&hb, &font, "3x9").unwrap();
        let hb_gids: Vec<u32> = shaped.iter().map(|s| s.glyph).collect();
        assert_eq!(hb_gids.len(), 3);
        assert_ne!(hb_gids, plain, "calt must substitute the x in 3x9");
    }

    /// Direct (type-2) kerning agrees across HB and otjs — Fraunces' A/V
    /// pair is visible to both lanes, so total advances match exactly.
    #[test]
    fn direct_kerning_agrees_with_otjs_lane() {
        let (font, hb) = load("fraunces/Fraunces[SOFT,WONK,opsz,wght]-Regular.ttf");
        let hb_shaped = shape_canvas2d_hb(&hb, &font, "AV").unwrap();
        let ot_shaped = crate::vector::measure::shape(&font, "AV").unwrap();
        let total = |s: &[ShapedGlyph]| s.iter().map(|g| g.x_advance + g.x_offset).sum::<f64>();
        assert_eq!(hb_shaped.len(), ot_shaped.len());
        assert!(
            (total(&hb_shaped) - total(&ot_shaped)).abs() < 1e-9,
            "hb {} != otjs {}",
            total(&hb_shaped),
            total(&ot_shaped)
        );
    }

    fn feat(tag: &str, value: u32) -> FontFeature {
        let b = tag.as_bytes();
        FontFeature {
            tag: [b[0], b[1], b[2], b[3]],
            value,
        }
    }

    fn gids(shaped: &[ShapedGlyph]) -> Vec<u32> {
        shaped.iter().map(|s| s.glyph).collect()
    }

    /// F7.3: `liga` is on by default ("fi" → one ligature glyph) and a run's
    /// `"liga" 0` turns it off (two glyphs, the cmap's own f and i).
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn features_turn_the_fi_ligature_off() {
        let (font, hb) = load("lato/Lato-Regular.ttf");
        let cmap = vec![font.char_to_glyph_index('f'), font.char_to_glyph_index('i')];
        let on = shape_features_hb(&hb, &font, "fi", &[]).unwrap();
        assert_eq!(on.len(), 1, "default liga must ligate fi: {:?}", gids(&on));
        assert!(!cmap.contains(&on[0].glyph));
        let off = shape_features_hb(&hb, &font, "fi", &[feat("liga", 0)]).unwrap();
        assert_eq!(gids(&off), cmap);
        // No features is exactly the default shaping.
        assert_eq!(
            gids(&shape_canvas2d_hb(&hb, &font, "fi").unwrap()),
            gids(&on)
        );
    }

    /// F7.3: `tnum` gives every digit the same advance; default figures are
    /// proportional ("1" is narrower than "0").
    #[test]
    fn tnum_makes_digit_advances_equal() {
        let (font, hb) = load("inter/Inter_24pt-Regular.ttf");
        let adv = |s: &[ShapedGlyph]| s.iter().map(|g| g.x_advance).collect::<Vec<_>>();
        let prop = adv(&shape_features_hb(&hb, &font, "10", &[]).unwrap());
        assert_ne!(prop[0], prop[1], "proportional 1 and 0 differ: {prop:?}");
        let tab = adv(&shape_features_hb(&hb, &font, "10", &[feat("tnum", 1)]).unwrap());
        assert_eq!(tab[0], tab[1], "tabular figures share one advance: {tab:?}");
    }

    /// F7.3: small caps — `smcp` swaps lowercase, `c2sc` swaps capitals, and
    /// neither touches the other case.
    #[test]
    fn smcp_and_c2sc_substitute_small_caps() {
        let (font, hb) = load("crates/moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf");
        let a = font.char_to_glyph_index('a');
        let cap_a = font.char_to_glyph_index('A');
        let smcp = gids(&shape_features_hb(&hb, &font, "aA", &[feat("smcp", 1)]).unwrap());
        assert_ne!(smcp[0], a);
        assert_eq!(smcp[1], cap_a);
        let both = gids(
            &shape_features_hb(&hb, &font, "aA", &[feat("smcp", 1), feat("c2sc", 1)]).unwrap(),
        );
        assert_eq!(both[0], smcp[0]);
        assert_ne!(both[1], cap_a);
        // A capital to small cap is the same glyph as the lowercase one.
        assert_eq!(both[1], both[0]);
    }

    /// F7.3: `ss01` swaps Montserrat's two-storey a for the single-storey
    /// one; `frac` turns "1/2" into numerator, fraction slash and denominator
    /// glyphs (none of them the cmap's own).
    #[test]
    #[ignore = "needs unported Moda fixture montserrat/Montserrat-Regular.ttf"]
    fn ss01_and_frac_substitute_glyphs() {
        let (font, hb) = load_from(&crate::test_fixtures::oracle_fonts_root().join("montserrat/Montserrat-Regular.ttf"));
        let a = font.char_to_glyph_index('a');
        let ss01 = gids(&shape_features_hb(&hb, &font, "a", &[feat("ss01", 1)]).unwrap());
        assert_ne!(ss01, vec![a]);
        let cmap: Vec<u32> = "1/2".chars().map(|c| font.char_to_glyph_index(c)).collect();
        let plain = gids(&shape_features_hb(&hb, &font, "1/2", &[]).unwrap());
        assert_eq!(plain, cmap);
        let frac = gids(&shape_features_hb(&hb, &font, "1/2", &[feat("frac", 1)]).unwrap());
        assert!(
            frac.iter().all(|g| !cmap.contains(g)),
            "frac must replace every glyph: {frac:?} vs {cmap:?}"
        );
    }

    /// UAX #9 with the LTR base level (one Chromium `fillText` of the same
    /// string): RTL content forms a level-1 run; mixed strings keep their
    /// LTR anchors in place; neutrals between same-direction RTL words join
    /// the RTL run (N1). Font-free pin — no vendored font covers an RTL
    /// script (Q-118 coverage note), so the split logic is pinned directly
    /// and Arabic placement truth comes from the IB53 measurement.
    #[test]
    fn bidi_visual_runs_ltr_base() {
        // Pure LTR → one non-rtl run.
        assert_eq!(visual_runs_ltr_base("hello kern"), vec![(0..10, false)]);
        // Pure Arabic (incl. the inter-word space, N1) → one rtl run.
        let ar = "\u{0645}\u{0631}\u{062D} \u{0628}\u{0643}";
        assert_eq!(visual_runs_ltr_base(ar), vec![(0..ar.len(), true)]);
        // Mixed: "ab <heb> cd" — Latin anchors stay put, the Hebrew island
        // is a single rtl run between them (visual order == this order under
        // an LTR base).
        let mixed = "ab \u{05D0}\u{05D1} cd";
        let runs = visual_runs_ltr_base(mixed);
        assert_eq!(runs.len(), 3, "{runs:?}");
        assert!(!runs[0].1 && runs[1].1 && !runs[2].1, "{runs:?}");
        assert_eq!(&mixed[runs[1].0.clone()], "\u{05D0}\u{05D1}");
    }

    /// The RTL run reaches rustybuzz with an explicit RTL direction — glyphs
    /// come back in reversed (visual) order. Pinned with Latin glyphs forced
    /// through the RTL path via U+202E (RLO override), which Inter maps; the
    /// override makes 'ab' a level-1 run without needing an RTL-script font.
    #[test]
    fn rtl_run_emits_visual_order() {
        let (font, hb) = load("inter/Inter_24pt-Regular.ttf");
        let text = "\u{202E}ab\u{202C}";
        let runs = visual_runs_ltr_base(text);
        assert!(runs.iter().any(|(_, rtl)| *rtl), "{runs:?}");
        let shaped = shape_canvas2d_hb(&hb, &font, text).unwrap();
        let a = font.char_to_glyph_index('a');
        let b = font.char_to_glyph_index('b');
        let gids: Vec<u32> = shaped.iter().map(|s| s.glyph).collect();
        let pa = gids.iter().position(|g| *g == a).expect("a shaped");
        let pb = gids.iter().position(|g| *g == b).expect("b shaped");
        assert!(pb < pa, "RTL-overridden run must reverse: {gids:?}");
    }
}
