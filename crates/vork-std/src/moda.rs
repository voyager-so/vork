//! Moda's text engine (crate `moda-text`) hosted on a scene's own font files: the one font host and
//! layout helpers that the 2D canvas's text (vork-render) and std/text's 3D outlines share.
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One font face: the name messages give it (its path) and its bytes.
pub struct Face {
    pub label: String,
    pub bytes: Vec<u8>,
}

/// The CSS `font-family` name of the `index`th font list, unique per list.
pub fn family(index: usize) -> String {
    format!("vork-font-{index}")
}

/// Each list's font files, read.
pub fn read(lists: &[Vec<PathBuf>]) -> Result<Vec<Vec<Face>>> {
    lists
        .iter()
        .map(|list| {
            list.iter()
                .map(|path| {
                    let bytes = std::fs::read(path).with_context(|| {
                        format!("font {}: restore a readable .ttf or .otf file or correct its path", path.display())
                    })?;
                    Ok(Face { label: path.display().to_string(), bytes })
                })
                .collect()
        })
        .collect()
}

/// A Moda catalog, manifest and families index for the faces, installed with the faces themselves in
/// a host-owned immutable in-memory overlay under a directory named by their
/// bytes, and loaded as a text host. Each list is one family (`family(index)`); its faces are the ones
/// font-weight and font-style choose from, by each face's OS/2 weight class and style. The manifest's
/// `keys` give Moda's degraded tier (`TextHost::degraded_engine`, the lane that shapes right-to-left and
/// complex scripts) the same faces under the same families, and `systemFonts: false` keeps the
/// machine's own fonts out of it, so a scene draws the same everywhere.
pub fn host(lists: &[Vec<Face>]) -> Result<moda_text::TextHost> {
    let mut identity = blake3::Hasher::new();
    for face in lists.iter().flatten() {
        identity.update(blake3::hash(&face.bytes).as_bytes());
    }
    for list in lists {
        identity.update(&(list.len() as u64).to_le_bytes());
    }
    let root = PathBuf::from(format!("/vork-fonts/{}", identity.finalize().to_hex()));
    let key = |name: &str| root.join(name).to_string_lossy().into_owned();
    let mut overlay = HashMap::new();
    let mut families = Vec::new();
    let mut files = serde_json::Map::new();
    let mut keys = serde_json::Map::new();
    for (index, list) in lists.iter().enumerate() {
        let mut variants = Vec::new();
        for (number, face) in list.iter().enumerate() {
            let parsed = ttf_parser::Face::parse(&face.bytes, 0)
                .map_err(|error| anyhow::anyhow!("font {}: not a TrueType or OpenType font ({error})", face.label))?;
            let style = if parsed.is_italic() {
                "italic"
            } else if parsed.is_oblique() {
                "oblique"
            } else {
                "normal"
            };
            let url = format!("vork-font:{index}:{number}");
            let file = format!("/vork-font-data/{}.ttf", blake3::hash(&face.bytes).to_hex());
            variants.push(serde_json::json!({"weight": parsed.weight().to_number(), "style": style, "msdf_ready": true,
                "atlas_urls": ["vork:vector-lane-only"], "ttf_url": url}));
            // Moda's store reads a manifest path joined to the host's directory, overlay first.
            files.insert(url, serde_json::json!({"path": file}));
            keys.insert(format!("{index}-{number}"), serde_json::json!({"family": family(index), "ttfPath": file}));
            overlay.insert(key(&file), face.bytes.clone());
        }
        families.push(serde_json::json!({"family_name": family(index), "category": "", "variants": variants}));
    }
    overlay.insert(key("catalog.json"), serde_json::to_vec(&serde_json::json!({"platform_fonts": families}))?);
    overlay.insert(key("MANIFEST.json"), serde_json::to_vec(&serde_json::json!({"files": files, "keys": keys, "systemFonts": false}))?);
    overlay.insert(key("families-index.json"), serde_json::to_vec(&serde_json::json!({"families": {}}))?);
    moda_text::TextHost::load_with_files(Path::new(&root), overlay).map_err(|error| anyhow::anyhow!("fonts: {error}"))
}

/// The widest line of a Moda layout (`paragraphs[].lines[].segments[]`), px.
pub fn widest_line(layout: &serde_json::Value) -> f64 {
    let mut widest: f64 = 0.;
    for paragraph in layout["paragraphs"].as_array().into_iter().flatten() {
        for line in paragraph["lines"].as_array().into_iter().flatten() {
            let (mut left, mut right) = (f64::INFINITY, f64::NEG_INFINITY);
            for segment in line["segments"].as_array().into_iter().flatten() {
                let (Some(x), Some(width)) = (segment["x"].as_f64(), segment["width"].as_f64()) else { continue };
                left = left.min(x);
                right = right.max(x + width);
            }
            if right > left {
                widest = widest.max(right - left);
            }
        }
    }
    widest
}

/// The HTML Moda's parser reads for text, with HTML's three special characters escaped.
pub fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Moda's path commands (`moda_engine::tess::moda_engine_path_types::PathCmd`), the model Moda's tessellator,
/// strokes, dashes and extrusion read.
pub use moda_engine::tess::moda_engine_path_types::PathCmd;

/// SVG 2 path data (`d`) as Moda's path commands: absolute move, line, cubic, quadratic and arc commands, one
/// subpath per move, and whether each subpath was closed with `Z`. Relative, horizontal, vertical and smooth
/// commands become their absolute forms; arcs stay arcs (Moda converts them to cubics as SVG defines). A command
/// after `Z` starts a new subpath at the closed one's start, as SVG says.
pub fn path_commands(d: &str) -> Result<(Vec<PathCmd>, Vec<bool>)> {
    path_segments(svgtypes::PathParser::from(d).map(|segment|segment.map_err(|e|anyhow::anyhow!("d: {e}"))))
}

/// Normalize parsed SVG segments, including interpolated animation values, through the same path rules.
pub fn path_segments(segments: impl IntoIterator<Item=Result<svgtypes::PathSegment>>) -> Result<(Vec<PathCmd>, Vec<bool>)> {
    use svgtypes::PathSegment::*;
    let mut cmds = Vec::new();
    let mut closed: Vec<bool> = Vec::new();
    let [mut x0, mut y0, mut sx, mut sy] = [0f64; 4];
    // The previous segment's control point, for S and T reflection.
    let mut cubic: Option<(f64, f64)> = None;
    let mut quadratic: Option<(f64, f64)> = None;
    let mut open = false;
    let finite = |values: &[f64]| -> Result<()> {
        anyhow::ensure!(values.iter().all(|v| v.is_finite()), "d: coordinates must be finite");
        Ok(())
    };
    for segment in segments {
        let segment = segment?;
        let at = |abs: bool, x: f64, y: f64| if abs { (x, y) } else { (x0 + x, y0 + y) };
        if !open && !matches!(segment, MoveTo { .. }) {
            // After Z (or at the start, which svgtypes refuses), the next subpath starts where the last one did.
            cmds.push(PathCmd::Move { x: sx, y: sy });
            closed.push(false);
            open = true;
        }
        let (mut next_cubic, mut next_quadratic) = (None, None);
        match segment {
            MoveTo { abs, x, y } => {
                let (x, y) = at(abs, x, y);
                finite(&[x, y])?;
                cmds.push(PathCmd::Move { x, y });
                closed.push(false);
                (x0, y0, sx, sy) = (x, y, x, y);
                open = true;
            }
            LineTo { abs, x, y } => {
                let (x, y) = at(abs, x, y);
                finite(&[x, y])?;
                cmds.push(PathCmd::Line { x, y });
                (x0, y0) = (x, y);
            }
            HorizontalLineTo { abs, x } => {
                let x = if abs { x } else { x0 + x };
                finite(&[x])?;
                cmds.push(PathCmd::Line { x, y: y0 });
                x0 = x;
            }
            VerticalLineTo { abs, y } => {
                let y = if abs { y } else { y0 + y };
                finite(&[y])?;
                cmds.push(PathCmd::Line { x: x0, y });
                y0 = y;
            }
            CurveTo { abs, x1, y1, x2, y2, x, y } => {
                let (cp1, cp2, (x, y)) = (at(abs, x1, y1), at(abs, x2, y2), at(abs, x, y));
                finite(&[cp1.0, cp1.1, cp2.0, cp2.1, x, y])?;
                cmds.push(PathCmd::Curve { x, y, cp1, cp2 });
                next_cubic = Some(cp2);
                (x0, y0) = (x, y);
            }
            SmoothCurveTo { abs, x2, y2, x, y } => {
                let cp1 = cubic.map_or((x0, y0), |(cx, cy)| (2. * x0 - cx, 2. * y0 - cy));
                let (cp2, (x, y)) = (at(abs, x2, y2), at(abs, x, y));
                finite(&[cp2.0, cp2.1, x, y])?;
                cmds.push(PathCmd::Curve { x, y, cp1, cp2 });
                next_cubic = Some(cp2);
                (x0, y0) = (x, y);
            }
            Quadratic { abs, x1, y1, x, y } => {
                let (cp, (x, y)) = (at(abs, x1, y1), at(abs, x, y));
                finite(&[cp.0, cp.1, x, y])?;
                cmds.push(PathCmd::Quadratic { x, y, cp });
                next_quadratic = Some(cp);
                (x0, y0) = (x, y);
            }
            SmoothQuadratic { abs, x, y } => {
                let cp = quadratic.map_or((x0, y0), |(cx, cy)| (2. * x0 - cx, 2. * y0 - cy));
                let (x, y) = at(abs, x, y);
                finite(&[x, y])?;
                cmds.push(PathCmd::Quadratic { x, y, cp });
                next_quadratic = Some(cp);
                (x0, y0) = (x, y);
            }
            EllipticalArc { abs, rx, ry, x_axis_rotation, large_arc, sweep, x, y } => {
                let (x, y) = at(abs, x, y);
                finite(&[rx, ry, x_axis_rotation, x, y])?;
                cmds.push(PathCmd::Arc { x, y, rx, ry, x_axis_rotation, large_arc, sweep });
                (x0, y0) = (x, y);
            }
            ClosePath { .. } => {
                if let Some(last) = closed.last_mut() {
                    *last = true;
                }
                (x0, y0) = (sx, sy);
                open = false;
            }
        }
        cubic = next_cubic;
        quadratic = next_quadratic;
    }
    Ok((cmds, closed))
}

/// The geometric bounds of the outline, including curve and arc extrema.
pub use moda_engine::tess::path_bounds;

/// A CSS colour (SVG 2 / CSS Color 3: names, `#rgb[a]`, `#rrggbb[aa]`, `rgb()`, `rgba()`, `hsl()`, `hsla()`,
/// `transparent`) as straight sRGB-encoded RGBA in 0..1.
pub fn css_color(text: &str) -> Result<[f32; 4]> {
    let color: svgtypes::Color = text
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("{text:?} is not a CSS colour; write a name, #rrggbb, rgb(), rgba(), hsl() or hsla()"))?;
    Ok([color.red, color.green, color.blue, color.alpha].map(|v| f32::from(v) / 255.))
}

/// Straight sRGB-encoded RGBA in 0..1 as the `#rrggbbaa` string Moda's colour parser reads.
pub fn moda_color([r, g, b, a]: [f32; 4]) -> String {
    let byte = |v: f32| (v.clamp(0., 1.) * 255.).round() as u8;
    format!("#{:02x}{:02x}{:02x}{:02x}", byte(r), byte(g), byte(b), byte(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_text::{measure::TextMeasurer, parser::parse_rich_text_to_paragraphs, types::InlineStyle};

    #[test]
    fn extrusion_bounds_include_arc_bulges_and_curve_extrema() {
        for d in ["M0 0 A1000 1000 0 1 1 1 0 Z", "M0 0 Q10 20 20 0 Z", "M0 0 C0 40 20 40 20 0 Z"] {
            let (cmds, _) = path_commands(d).unwrap();
            let b = path_bounds(&cmds).unwrap();
            let expected = if d.contains('A') { 2000. } else if d.contains('Q') { 10. } else { 30. };
            assert!(((b[3] - b[1]) - expected).abs() < 0.001, "{d}: {b:?}");
        }
    }

    #[test]
    fn retired_font_namespace_is_released_after_lazy_store() {
        let faces = vec![vec![Face { label: "test".into(),
            bytes: [include_bytes!("../tests/fonts/LiberationSans-Regular.ttf").as_slice(), b"retirement-test"].concat() }]];
        let mut identity = blake3::Hasher::new();
        identity.update(blake3::hash(&faces[0][0].bytes).as_bytes());
        identity.update(&1u64.to_le_bytes());
        let path = PathBuf::from(format!("/vork-fonts/{}", identity.finalize().to_hex())).join("catalog.json");
        let host = host(&faces).unwrap();
        let catalog = moda_text::catalog::Catalog::from_catalog_json(&serde_json::from_slice(&moda_text::host::read_file_bytes(&path).unwrap()).unwrap());
        let store = host.vector_font_store_lenient();
        drop(host);
        assert!(store.font_for_variant(&catalog, &family(0), 400, "normal", None).is_some());
        assert!(moda_text::host::read_file_bytes(&path).is_ok());
        drop(store);
        assert!(moda_text::host::read_file_bytes(&path).is_err(), "retired font namespace remains rooted");
    }

    #[test]
    fn font_bytes_are_shared_between_distinct_live_sets() {
        let bytes = include_bytes!("../tests/fonts/LiberationSans-Regular.ttf");
        let face = || Face { label: "shared".into(), bytes: bytes.to_vec() };
        let first = host(&[vec![face()]]).unwrap();
        let path = PathBuf::from(format!("/vork-font-data/{}.ttf", blake3::hash(bytes).to_hex()));
        let a = moda_text::host::read_file_shared(&path).unwrap();
        let second = host(&[vec![face()], vec![face()]]).unwrap();
        let b = moda_text::host::read_file_shared(&path).unwrap();
        assert!(std::sync::Arc::ptr_eq(&a, &b));
        drop(first);
        assert!(second.vector_font_store_lenient().font_for_variant(&second.catalog, &family(1), 400, "normal", None).is_some());
        let _ = second.degraded_engine();
    }

    struct Measure;
    impl TextMeasurer for Measure {
        fn measure(&self, text: &str, _: &InlineStyle) -> f64 { text.chars().count() as f64 * 10. }
    }

    #[test]
    fn normal_wrapping_preserves_words_and_nbsp() {
        for text in ["WWWWWWWW", "A\u{a0}B\u{a0}C"] {
            let paragraphs = parse_rich_text_to_paragraphs(&format!("<p>{text}</p>"));
            let lines = moda_text::wrap::layout_paragraph(&paragraphs[0], 25., &Measure);
            assert_eq!(lines.len(), 1, "{text:?}");
        }
        let paragraphs = parse_rich_text_to_paragraphs("<p>AA BB</p>");
        assert_eq!(moda_text::wrap::layout_paragraph(&paragraphs[0], 25., &Measure).len(), 2);
    }

    #[test]
    fn bidi_override_routes_ltr_paragraphs_to_shaping() {
        for attrs in ["", " style=\"direction: ltr\""] {
            assert!(moda_text::paragraph_direction::html_needs_rtl_reorder(
                &format!("<p{attrs}>abc \u{202e}DEF\u{202c} xyz</p>")));
        }
        assert!(!moda_text::paragraph_direction::html_needs_rtl_reorder("<p>abc DEF xyz</p>"));
    }

    #[test]
    fn oblique_survives_font_resolution_and_synthesizes_slant() {
        let host = host(&[vec![Face { label: "LiberationSans-Regular.ttf".into(),
            bytes: include_bytes!("../tests/fonts/LiberationSans-Regular.ttf").to_vec() }]]).unwrap();
        let html = "<p style=\"font-family: vork-font-0; font-style: oblique\">ABC</p>";
        let runs = moda_text::runs::extract_msdf_runs_by_paragraph(html, &host.catalog);
        assert_eq!(runs[0].font_style, "oblique");
        let store = host.vector_font_store_lenient();
        let spec = moda_text::ops::TextNodeSpec { html: html.into(), ..Default::default() };
        let moda_text::vector::ops::VectorEngineDecision::Vector(ops) =
            moda_text::vector::ops::build_vector_node_ops(&spec, &store, &host.catalog, 1., moda_text::vector::VectorPlacement::VectorParity)
            else { panic!("oblique must draw outlined glyphs"); };
        let records: Vec<_> = ops.ops.iter().flat_map(|op| &op.records).collect();
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|record| record.glyph_lin[1] != 0.), "oblique must slant every glyph");
    }
}
