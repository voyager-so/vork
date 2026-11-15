//! Desktop fork: outlines and advances for the faces the opentype.js port in
//! `otjs.rs` cannot read — CFF/CFF2 outlines (no `glyf` table) and variable
//! fonts instanced away from their default coordinates.
//!
//! Both come from ttf-parser (already in the tree through rustybuzz, re-exported
//! as `rustybuzz::ttf_parser`), over the SAME bytes the otjs parse and the
//! HarfBuzz face read. Static `glyf` fonts never reach this module, so their
//! outlines stay on the byte-exact opentype.js port.
//!
//! Instances are fixed coordinates chosen by the font catalog (a variable
//! font's named instances) plus `opsz`, which follows the font size the way
//! CSS `font-optical-sizing: auto` does (see `VectorFontStore`). Known
//! deviation: GPOS kerning stays at the default instance (the otjs kerning walk
//! ignores device/variation tables), exactly as it ignores them for static
//! fonts.

use std::cell::RefCell;
use std::collections::HashMap;

use rustybuzz::ttf_parser;

use crate::ttf::TtfMetrics;
use crate::vector::curves::PathCommand;

/// A variation axis coordinate, e.g. `("wght", 700.0)`.
pub type Variation = (String, f32);

/// One variation axis of a face (`fvar`).
#[derive(Debug, Clone, PartialEq)]
pub struct AxisInfo {
    pub tag: String,
    pub min: f32,
    pub default: f32,
    pub max: f32,
}

fn tag_of(s: &str) -> Option<ttf_parser::Tag> {
    let b = s.as_bytes();
    (b.len() == 4).then(|| ttf_parser::Tag::from_bytes(&[b[0], b[1], b[2], b[3]]))
}

/// The `fvar` axes of `data` (empty for a static font or unparseable bytes).
pub fn variation_axes(data: &[u8]) -> Vec<AxisInfo> {
    let Ok(face) = ttf_parser::Face::parse(data, 0) else {
        return Vec::new();
    };
    face.variation_axes()
        .into_iter()
        .map(|a| AxisInfo {
            tag: String::from_utf8_lossy(&a.tag.to_bytes()).into_owned(),
            min: a.min_value,
            default: a.def_value,
            max: a.max_value,
        })
        .collect()
}

/// A ttf-parser face bound to the font bytes that back it, with variation
/// coordinates applied.
pub(crate) struct VarFace {
    face: ttf_parser::Face<'static>,
    /// Order matters: `face` (above) drops before the buffer it borrows.
    _bytes: std::sync::Arc<Vec<u8>>,
    /// True when the coordinates differ from the default instance, i.e. the
    /// otjs tables (hmtx, glyf, OS/2) no longer describe this instance.
    pub(crate) instanced: bool,
    advances: RefCell<HashMap<u32, f64>>,
}

impl VarFace {
    pub(crate) fn new(bytes: std::sync::Arc<Vec<u8>>, variations: &[Variation]) -> Result<VarFace, String> {
        // SAFETY: same argument as `HbFace::from_bytes` — `face` borrows the
        // heap allocation owned by `bytes`, which is never mutated or moved and
        // outlives `face` because `_bytes` is declared after it.
        let slice: &'static [u8] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) };
        let mut face = ttf_parser::Face::parse(slice, 0).map_err(|e| format!("ttf-parser: {e}"))?;
        for (tag, value) in variations {
            if let Some(tag) = tag_of(tag) {
                let _ = face.set_variation(tag, *value);
            }
        }
        let instanced = face.has_non_default_variation_coordinates();
        Ok(VarFace {
            face,
            _bytes: bytes,
            instanced,
            advances: RefCell::new(HashMap::new()),
        })
    }

    /// Horizontal advance at this instance (HVAR, or gvar phantom points).
    pub(crate) fn advance(&self, gid: u32) -> Option<f64> {
        if let Some(a) = self.advances.borrow().get(&gid) {
            return Some(*a);
        }
        let id = u16::try_from(gid).ok()?;
        let a = self.face.glyph_hor_advance(ttf_parser::GlyphId(id))? as f64;
        self.advances.borrow_mut().insert(gid, a);
        Some(a)
    }

    /// Font-unit, y-up path commands for `gid` at this instance. A glyph with
    /// no outline (space) yields an empty list, like the glyf reader.
    pub(crate) fn glyph_commands(&self, gid: u32) -> Result<Vec<PathCommand>, String> {
        let id = u16::try_from(gid).map_err(|_| format!("glyph id {gid} out of range"))?;
        let mut b = CommandBuilder(Vec::new());
        let _ = self.face.outline_glyph(ttf_parser::GlyphId(id), &mut b);
        Ok(b.0)
    }

    /// `metrics` with this instance's MVAR deltas applied (same source table
    /// `choose_baseline_metrics` picked).
    pub(crate) fn vary_metrics(&self, m: &TtfMetrics) -> TtfMetrics {
        if !self.instanced {
            return m.clone();
        }
        let f = &self.face;
        let (ascent, descent, line_gap) = if m.source == "os2" {
            (
                f.typographic_ascender().map(f64::from).unwrap_or(m.ascent),
                f.typographic_descender()
                    .map(f64::from)
                    .unwrap_or(m.descent),
                f.typographic_line_gap()
                    .map(f64::from)
                    .unwrap_or(m.line_gap),
            )
        } else {
            (
                f64::from(f.ascender()),
                f64::from(f.descender()),
                f64::from(f.line_gap()),
            )
        };
        TtfMetrics {
            ascent,
            descent,
            line_gap,
            x_height: m.x_height.map(|x| f.x_height().map(f64::from).unwrap_or(x)),
            cap_height: m
                .cap_height
                .map(|c| f.capital_height().map(f64::from).unwrap_or(c)),
            ..m.clone()
        }
    }
}

struct CommandBuilder(Vec<PathCommand>);

impl ttf_parser::OutlineBuilder for CommandBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.push(PathCommand::M {
            x: x as f64,
            y: y as f64,
        });
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.push(PathCommand::L {
            x: x as f64,
            y: y as f64,
        });
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.0.push(PathCommand::Q {
            x1: x1 as f64,
            y1: y1 as f64,
            x: x as f64,
            y: y as f64,
        });
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.0.push(PathCommand::C {
            x1: x1 as f64,
            y1: y1 as f64,
            x2: x2 as f64,
            y2: y2 as f64,
            x: x as f64,
            y: y as f64,
        });
    }
    fn close(&mut self) {
        self.0.push(PathCommand::Z);
    }
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use crate::vector::otjs::OtjsFont;

    fn read(path: &str) -> Option<std::sync::Arc<Vec<u8>>> {
        std::fs::read(path).ok().map(std::sync::Arc::new)
    }

    /// A CFF (`.otf`) face — no glyf table — yields cubic outlines through
    /// ttf-parser. Use the shipped two-glyph Noto CJK subset.
    #[test]
    fn cff_outlines_render() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../moda-text/tests/fonts/HanTest.otf"
        );
        let bytes = read(path).expect("bundled CJK otf");
        let font = OtjsFont::parse(&bytes).expect("parse");
        let gid = font.char_to_glyph_index('字');
        assert_ne!(gid, 0);
        let q = font.glyph_quadratics(gid).expect("cff outline");
        assert!(q.curve_count > 4, "curves: {}", q.curve_count);
        assert!(q.max_y > q.min_y && q.max_x > q.min_x);
        let han = font
            .glyph_quadratics(font.char_to_glyph_index('漢'))
            .unwrap();
        assert!(han.curve_count > q.curve_count);
    }

    /// A variable face instanced at a named instance's coordinates: advances,
    /// outlines and the HarfBuzz face all follow the instance.
    #[test]
    #[cfg(vork_private_tests)] // Reads a private Greenlight scene font.
    fn variable_instance_changes_advances_and_outlines() {
        let bytes = read(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../scenes/greenlight/text/assets/NotoSans.ttf")).unwrap();
        let regular = OtjsFont::parse_instance(&bytes, &[]).unwrap();
        let bold = OtjsFont::parse_instance(&bytes, &[("wght".into(), 700.0)]).unwrap();
        let condensed = OtjsFont::parse_instance(&bytes, &[("wdth".into(), 60.0)]).unwrap();
        let h = regular.char_to_glyph_index('H');
        assert!(bold.advance_width(h) > regular.advance_width(h));
        assert!(condensed.advance_width(h) < regular.advance_width(h));
        let (r, b) = (
            regular.glyph_quadratics(h).unwrap(),
            bold.glyph_quadratics(h).unwrap(),
        );
        assert_ne!(r.points, b.points);
        // Default coordinates stay on the opentype.js glyf port.
        let same = OtjsFont::parse_instance(&bytes, &[("wght".into(), 400.0)]).unwrap();
        assert_eq!(same.advance_width(h), regular.advance_width(h));
        assert_eq!(same.glyph_quadratics(h).unwrap().points, r.points);
        let hb = crate::vector::hb::HbFace::from_bytes_with_variations(
            bytes.clone(),
            &[("wght".into(), 700.0)],
        );
        assert!(hb.is_some());
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads a private Greenlight scene font.
    fn axes_are_listed_for_variable_fonts_only() {
        let bytes = read(concat!(env!("CARGO_MANIFEST_DIR"),
            "/../../scenes/greenlight/text/assets/NotoSans.ttf")).unwrap();
        let axes = super::variation_axes(&bytes);
        let weight = axes.iter().find(|a| a.tag == "wght").expect("weight axis");
        assert!(weight.min < weight.max);
        let inter = read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../moda-engine/tests/fixtures/fonts/inter/Inter_24pt-Regular.ttf"
        ))
        .unwrap();
        assert!(super::variation_axes(&inter).is_empty());
    }

    /// A GSUB/GPOS 1.1 table (FeatureVariations: most variable fonts, 18 of the bundled faces) parses. The version
    /// check read 1.1 as minor 0x1000, so these faces failed to parse and their text fell back to the raster tier,
    /// which draws no text on a path.
    #[test]
    #[ignore = "needs unported Moda fixture Archivo[wdth,wght].ttf with GSUB 1.1"]
    fn layout_tables_version_1_1_parse() {
        let bytes = std::sync::Arc::new(
            read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../packages/editor/public/fonts/google/archivo/Archivo[wdth,wght].ttf"
            ))
            .unwrap(),
        );
        let gsub = bytes
            .windows(4)
            .position(|w| w == b"GSUB")
            .expect("GSUB in the table directory");
        let off = u32::from_be_bytes(bytes[gsub + 8..gsub + 12].try_into().unwrap()) as usize;
        assert_eq!(
            &bytes[off..off + 4],
            &[0, 1, 0, 1],
            "the fixture's GSUB is version 1.1"
        );
        assert!(crate::vector::OtjsFont::parse(&bytes).is_ok());
        let wght: [super::Variation; 1] = [("wght".into(), 600.0)];
        assert!(crate::vector::OtjsFont::parse_instance(&bytes, &wght).is_ok());
    }
}
