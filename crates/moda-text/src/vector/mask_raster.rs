//! Vector-lane text-mask rasterization (mask:text-alpha better tier).
//!
//! The native text-as-mask coverage source historically came from the
//! cosmic-text Degraded raster (`crate::degraded`), whose shaping/metrics
//! deviate from Chromium's Canvas2D letterforms (measured IoU 0.605 on the
//! `masks/text-alpha` fixture vs 0.919 for the vector lane against the same
//! browser golden — and the same node rendered as VISIBLE text takes the
//! vector lane, so one export drew two different letterform sets for one
//! string). This module closes that gap: it renders the exact per-glyph
//! quadratic outlines the GA vector text lane draws (`build_vector_node_ops`)
//! into a CPU alpha raster with the SAME texture semantics as the Degraded
//! raster (`DegradedRaster`: qualityScale-2 supersample, top-left anchored,
//! `logicalSize` = texture px / qualityScale), so the mask seam
//! (`moda_engine::effects::mask`) consumes it unchanged.
//!
//! Q-68 pin: the mask silhouette is TOP-LEFT anchored — `verticalAlign` moves
//! the display quad, never the texture content — so the caller-facing entry
//! forces `vertical_align: top` before layout, and the raster is laid out in
//! TEXTURE-local coordinates (wrap width = texture content width, exactly like
//! `canvas2dRichTextRenderer.ts:165-198` and `degraded.rs:388`), not node-box
//! coordinates.
//!
//! Scope: silhouette-only sources. Anything whose Canvas2D texture alpha is
//! not the plain glyph/decoration silhouette (drop shadow / legacy
//! stroke, structured effects, gradients, kinetic per-glyph transforms)
//! declines with a reason and the caller falls back to the Degraded raster —
//! byte-identical to the pre-existing behavior for those nodes.

use crate::catalog::Catalog;
use crate::degraded::{DegradedRaster, QUALITY_SCALE};
use crate::ops::TextNodeSpec;
use crate::vector::measure::VectorFontStore;
use crate::vector::ops::{
    build_vector_node_ops, VectorEngineDecision, VectorNodeOps, VectorOp, VectorOpPipeline,
};
use resvg::tiny_skia;

/// Same texture budget as the Degraded raster (`degraded.rs`): the browser
/// standalone texture is a GPU texture with the identical ceiling.
const MAX_TEXTURE_PX: u32 = 8192;

/// Contour-chaining tolerance in em units: quadratic segments come out of the
/// outline extractor in contour order, endpoint-to-startpoint exact, so this
/// only has to absorb f32 rounding.
const CONTOUR_EPS: f32 = 1e-4;

/// Render the vector-lane mask raster for one richtext mask source, or
/// decline (`Err(reason)`) so the caller falls back to the Degraded raster.
///
/// `logical_w`/`logical_h` are the texture logical size the Degraded lane
/// computed for this node (`DegradedEngine::canvas_logical_size` — the port of
/// `getRichTextCanvasLogicalSize`): reusing them keeps the coverage-quad
/// geometry byte-identical to the previous tier (only the pixels change) and
/// preserves the Q-68 top-left anchor and horizontal texture-local alignment.
pub fn render_mask_raster(
    spec: &TextNodeSpec,
    logical_w: f64,
    logical_h: f64,
    store: &VectorFontStore,
    catalog: &Catalog,
) -> Result<DegradedRaster, String> {
    // --- Silhouette-only gates (decline → Degraded fallback, the prior tier).
    if spec.drop_shadows().next().is_some() {
        return Err("drop-shadow".to_string());
    }
    if spec.stroke_color.is_some() {
        return Err("legacy-stroke".to_string());
    }
    if spec.gradient.is_some() {
        // Gradient stops can carry per-stop alpha the solid rasterization
        // below would flatten to opaque.
        return Err("gradient-fill".to_string());
    }
    if !spec.effects.is_empty() {
        return Err("structured-effects".to_string());
    }
    if !spec.unit_transforms.is_empty() {
        return Err("unit-transforms".to_string());
    }

    // --- Texture-local layout: wrap at the texture content width, top
    // anchored (Q-68) — the same box the Canvas2D/Degraded draw walk uses.
    let mut tex_spec = spec.clone();
    tex_spec.width = Some(logical_w);
    tex_spec.height = Some(logical_h);
    tex_spec.vertical_align = None;
    // The mask samples texture alpha; the display-quad concerns don't apply.
    tex_spec.distort = None;
    // The browser's standalone mask texture (`utils/canvas2dRichTextRenderer
    // .ts::renderRichTextNodeToCanvasWithFixedSize`) draws ONLY htmlContent's
    // per-run colors — it has no fillPriority handling at all, so the
    // node-level 'color' fill override never reaches the texture (the
    // Degraded mask fallback clears it too: moda-engine `effects/mask.rs`).
    // Clear it here so the vector ops carry the same per-run colors/alphas
    // as both siblings.
    tex_spec.fill_override = None;

    let ops = match build_vector_node_ops(
        &tex_spec,
        store,
        catalog,
        QUALITY_SCALE,
        super::VectorPlacement::Canvas2dParity,
    ) {
        VectorEngineDecision::Vector(ops) => ops,
        VectorEngineDecision::Fallback { reason, .. } => return Err(reason),
    };
    if !ops.emojis.is_empty() {
        // This raster draws outlines only; the degraded raster draws colour emoji.
        return Err("emoji-run".to_string());
    }

    // Defense in depth: the spec gates above should leave only plain fill
    // (Analytic) + solid primitive (Tess: decoration rects, list bullets)
    // passes. Anything else means an unmodeled pass — decline.
    for op in &ops.ops {
        let plain = match op.pipeline {
            VectorOpPipeline::Tess | VectorOpPipeline::Analytic => {
                op.blur_px == 0.0 && op.stroke_px == 0.0 && !op.additive
            }
            VectorOpPipeline::AnalyticGradient | VectorOpPipeline::AnalyticShaderFill => false,
        };
        if !plain {
            return Err("effect-pass".to_string());
        }
        if op.records.iter().any(|r| {
            r.glyph_lin != [1.0, 0.0, 0.0, 1.0]
                || r.glyph_off != [0.0, 0.0]
                || r.glyph_opacity != 1.0
        }) {
            return Err("glyph-transform".to_string());
        }
    }

    // Q-69 parity (`detectMixedAlpha`, alpha.ts:29-39): uniform-alpha TEXT
    // rasterizes OPAQUE in the browser's standalone Canvas2D texture —
    // aggregate translucency ships in `fill_alpha`, a display-quad concern
    // masks never sample — while mixed-alpha runs bake their per-run alpha
    // into the texture. Detect over the PARSED runs (whitespace-only runs
    // included: a styled space is a layout segment the browser's detector
    // sees, but it emits no glyph op — detecting over ops would miss it and
    // wrongly force translucent glyphs opaque). Always from the parsed HTML:
    // the Canvas2D standalone path draws only htmlContent (fill_override was
    // cleared above to match).
    let mixed = {
        let mut first_alpha: Option<f64> = None;
        let mut mixed = false;
        for para in crate::parser::parse_rich_text_to_paragraphs(&tex_spec.html) {
            for run in &para.runs {
                if run.text.is_empty() {
                    continue;
                }
                let a = crate::color::css_color_to_normalized_rgba(&run.style.color)
                    .map(|c| c[3].clamp(0.0, 1.0))
                    .unwrap_or(1.0);
                match first_alpha {
                    None => first_alpha = Some(a),
                    Some(f) if (a - f).abs() > crate::degraded::MIXED_ALPHA_TOLERANCE => {
                        mixed = true;
                    }
                    Some(_) => {}
                }
            }
        }
        mixed
    };

    rasterize(&ops, logical_w, logical_h, mixed)
}

fn rasterize(
    ops: &VectorNodeOps,
    logical_w: f64,
    logical_h: f64,
    mixed: bool,
) -> Result<DegradedRaster, String> {
    // Same canvas sizing as `DegradedEngine::render_node`.
    let scale = QUALITY_SCALE;
    let canvas_w = ((logical_w * scale).floor() as i64).max(1) as u32;
    let canvas_h = ((logical_h * scale).floor() as i64).max(1) as u32;
    if canvas_w > MAX_TEXTURE_PX || canvas_h > MAX_TEXTURE_PX {
        return Err(format!(
            "vector mask raster {canvas_w}x{canvas_h} exceeds texture budget"
        ));
    }
    let mut pixmap = tiny_skia::Pixmap::new(canvas_w, canvas_h)
        .ok_or_else(|| "pixmap allocation failed".to_string())?;
    let transform = tiny_skia::Transform::from_scale(scale as f32, scale as f32);

    for op in &ops.ops {
        let path = match op.pipeline {
            VectorOpPipeline::Tess => tess_path(op),
            VectorOpPipeline::Analytic => analytic_path(op, &ops.curve_points),
            // Filtered out by the caller; unreachable by construction.
            _ => None,
        };
        let Some(path) = path else { continue };
        let mut paint = tiny_skia::Paint::default();
        paint.anti_alias = true;
        // NEVER bake node opacity (`op.opacity` = params.y for the plain fill
        // pass): the browser's standalone mask texture is drawn without node
        // opacity — that multiplies into the visible DISPLAY quad only — and
        // the Degraded raster likewise excludes it, so a 50%-opacity node
        // still clips at full mask strength on every arm.
        let a = if matches!(op.pipeline, VectorOpPipeline::Analytic) && !mixed {
            // Uniform-alpha glyphs rasterize opaque (Q-69).
            1.0
        } else {
            op.color[3].clamp(0.0, 1.0)
        };
        paint.set_color(
            tiny_skia::Color::from_rgba(
                op.color[0].clamp(0.0, 1.0),
                op.color[1].clamp(0.0, 1.0),
                op.color[2].clamp(0.0, 1.0),
                a,
            )
            .ok_or_else(|| "invalid op color".to_string())?,
        );
        pixmap.fill_path(&path, &paint, tiny_skia::FillRule::Winding, transform, None);
    }

    // tiny-skia stores premultiplied RGBA; `DegradedRaster.rgba` is straight.
    let mut rgba = pixmap.take();
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a > 0 && a < 255 {
            px[0] = ((px[0] as u32 * 255 + a / 2) / a).min(255) as u8;
            px[1] = ((px[1] as u32 * 255 + a / 2) / a).min(255) as u8;
            px[2] = ((px[2] as u32 * 255 + a / 2) / a).min(255) as u8;
        }
    }

    Ok(DegradedRaster {
        rgba,
        width: canvas_w,
        height: canvas_h,
        logical_w,
        logical_h,
        quality_scale: scale,
        pad_top: 0.0,
        pad_bottom: 0.0,
        // Masks ignore `fill_alpha` (display-quad concern) — see the seam doc
        // on `MaskText`.
        fill_alpha: 1.0,
    })
}

/// One path from a Tess op's flat `[x0,y0, x1,y1, x2,y2, …]` triangle list.
/// All triangles go into a single path so shared edges cancel in the winding
/// accumulation (no AA seams between the two halves of a rectangle).
fn tess_path(op: &VectorOp) -> Option<tiny_skia::Path> {
    let mut pb = tiny_skia::PathBuilder::new();
    for tri in op.tris.chunks_exact(6) {
        pb.move_to(tri[0], tri[1]);
        pb.line_to(tri[2], tri[3]);
        pb.line_to(tri[4], tri[5]);
        pb.close();
    }
    pb.finish()
}

/// One path from an Analytic op's glyph instances: each record's quadratic
/// segments (em units, y-down, pen origin) mapped to texture-local px via
/// `pen + em·fontSize` — the identity-instance fast path of the analytic
/// vertex shader — and chained back into contours (the extractor emits
/// segments in contour order).
fn analytic_path(op: &VectorOp, curve_points: &[f32]) -> Option<tiny_skia::Path> {
    let mut pb = tiny_skia::PathBuilder::new();
    for rec in &op.records {
        let fs = rec.font_size_px;
        let (px, py) = (rec.pen_x, rec.baseline_y);
        let mut cur: Option<(f32, f32)> = None;
        let first = rec.first_curve as usize;
        let count = rec.curve_count as usize;
        for i in 0..count {
            let base = (first + i) * 6;
            let Some(seg) = curve_points.get(base..base + 6) else {
                break;
            };
            let p0 = (px + seg[0] * fs, py + seg[1] * fs);
            let p1 = (px + seg[2] * fs, py + seg[3] * fs);
            let p2 = (px + seg[4] * fs, py + seg[5] * fs);
            let eps = CONTOUR_EPS * fs.max(1.0);
            let start_new = match cur {
                Some((cx, cy)) => (cx - p0.0).abs() > eps || (cy - p0.1).abs() > eps,
                None => true,
            };
            if start_new {
                pb.move_to(p0.0, p0.1);
            }
            pb.quad_to(p1.0, p1.1, p2.0, p2.1);
            cur = Some(p2);
        }
    }
    pb.finish()
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use crate::test_fixtures::store_and_catalog;

    fn spec(html: &str) -> TextNodeSpec {
        TextNodeSpec {
            node_id: "m".into(),
            html: html.into(),
            width: Some(920.0),
            height: Some(320.0),
            opacity: 1.0,
            ..Default::default()
        }
    }

    const MODA_HTML: &str =
        "<p style=\"font-family: Inter; font-size: 160px; font-weight: 800; color: #000000\">MODA</p>";

    fn alpha_stats(r: &DegradedRaster) -> (usize, u32, u32, u32, u32) {
        // (count>0, min_x, min_y, max_x, max_y) over alpha>16 pixels.
        let (mut n, mut x0, mut y0, mut x1, mut y1) = (0usize, u32::MAX, u32::MAX, 0u32, 0u32);
        for y in 0..r.height {
            for x in 0..r.width {
                let a = r.rgba[((y * r.width + x) * 4 + 3) as usize];
                if a > 16 {
                    n += 1;
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        (n, x0, y0, x1, y1)
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn plain_text_renders_vector_alpha_raster() {
        let (store, catalog) = store_and_catalog();
        let s = spec(MODA_HTML);
        let r = render_mask_raster(&s, 700.0, 200.0, &store, &catalog).expect("vector raster");
        assert_eq!(r.width, 1400);
        assert_eq!(r.height, 400);
        assert_eq!(r.logical_w, 700.0);
        assert_eq!(r.logical_h, 200.0);
        assert_eq!(r.quality_scale, QUALITY_SCALE);
        let (n, x0, y0, x1, y1) = alpha_stats(&r);
        // Substantial ink, starting near the texture origin (top-left anchor),
        // spanning most of a 160px 'MODA' at 2x supersampling.
        assert!(n > 10_000, "ink pixels: {n}");
        assert!(x0 < 60 && y0 < 120, "ink starts at ({x0},{y0})");
        assert!(x1 > 700 && y1 > 200, "ink extends to ({x1},{y1})");
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn vertical_align_never_moves_the_silhouette() {
        // Q-68: the mask texture is TOP-LEFT anchored — verticalAlign moves
        // the display quad, never the raster content.
        let (store, catalog) = store_and_catalog();
        let top = render_mask_raster(&spec(MODA_HTML), 700.0, 200.0, &store, &catalog).unwrap();
        for align in ["middle", "bottom"] {
            let mut s = spec(MODA_HTML);
            s.vertical_align = Some(align.to_string());
            let r = render_mask_raster(&s, 700.0, 200.0, &store, &catalog).unwrap();
            assert_eq!(
                r.rgba, top.rgba,
                "verticalAlign {align} moved the mask silhouette"
            );
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn silhouette_gates_decline_to_degraded() {
        let (store, catalog) = store_and_catalog();
        let cases: Vec<(&str, Box<dyn Fn(&mut TextNodeSpec)>)> = vec![
            (
                "drop-shadow",
                Box::new(|s| {
                    s.effects = vec![crate::ops::TextEffectSpec::DropShadow(crate::ops::TextDropShadow {
                        color: "#000000".into(),
                        offset_x: 0.0,
                        offset_y: 4.0,
                        blur: 4.0,
                        opacity: 1.0,
                    })]
                }),
            ),
            (
                "legacy-stroke",
                Box::new(|s| s.stroke_color = Some("#000000".into())),
            ),
            (
                "unit-transforms",
                Box::new(|s| s.unit_transforms = vec![crate::vector::ops::GlyphXform::IDENTITY].into()),
            ),
        ];
        for (want, mutate) in cases {
            let mut s = spec(MODA_HTML);
            mutate(&mut s);
            let err = render_mask_raster(&s, 700.0, 200.0, &store, &catalog)
                .expect_err("gate must decline");
            assert_eq!(err, want);
        }
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn non_vector_content_declines() {
        let (store, catalog) = store_and_catalog();
        // Emoji runs decline the vector routing (`resolve_vector_runs`
        // emoji-run fallback) → Degraded raster, exactly like visible text.
        let s = spec("<p style=\"font-family: Inter\">emoji \u{1F600}</p>");
        assert!(render_mask_raster(&s, 700.0, 200.0, &store, &catalog).is_err());
        // Complex-script content declines to the cosmic shaper.
        let s = spec("<p style=\"font-family: Inter\">مرحبا</p>");
        assert!(render_mask_raster(&s, 700.0, 200.0, &store, &catalog).is_err());
    }

    /// Node opacity is a DISPLAY-quad concern: the browser's standalone mask
    /// texture and the Degraded raster both exclude it, so a 50%-opacity node
    /// must clip at full mask strength — the raster is byte-identical to the
    /// opaque node's.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn node_opacity_never_dims_the_mask() {
        let (store, catalog) = store_and_catalog();
        let opaque = render_mask_raster(&spec(MODA_HTML), 700.0, 200.0, &store, &catalog).unwrap();
        let mut s = spec(MODA_HTML);
        s.opacity = 0.5;
        let dimmed = render_mask_raster(&s, 700.0, 200.0, &store, &catalog).unwrap();
        assert_eq!(
            opaque.rgba, dimmed.rgba,
            "node opacity leaked into the mask raster"
        );
    }

    /// Q-69: uniform-alpha text rasterizes OPAQUE (translucency ships in
    /// `fill_alpha`, which masks ignore) — rgba(...,0.5) text must produce the
    /// same mask pixels as opaque text.
    #[test]
    fn uniform_alpha_rasterizes_opaque() {
        let (store, catalog) = store_and_catalog();
        let opaque = render_mask_raster(&spec(MODA_HTML), 700.0, 200.0, &store, &catalog).unwrap();
        let translucent_html =
            "<p style=\"font-family: Inter; font-size: 160px; font-weight: 800; color: rgba(0, 0, 0, 0.5)\">MODA</p>";
        let translucent =
            render_mask_raster(&spec(translucent_html), 700.0, 200.0, &store, &catalog).unwrap();
        assert_eq!(
            opaque.rgba, translucent.rgba,
            "uniform 0.5-alpha text must clip at full coverage (Q-69)"
        );
    }

    /// Styled WHITESPACE participates in `detectMixedAlpha` (it is a layout
    /// segment in the browser) even though it emits no glyph op: translucent
    /// visible text + an opaque styled space is MIXED, so the glyphs keep
    /// their translucent alpha instead of being forced opaque.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn styled_whitespace_counts_toward_mixed_alpha() {
        let (store, catalog) = store_and_catalog();
        let html = "<p style=\"font-family: Inter; font-size: 160px; font-weight: 800\">\
                    <span style=\"color: rgba(0, 0, 0, 0.5)\">MODA</span>\
                    <span style=\"color: #000000\"> </span></p>";
        let r = render_mask_raster(&spec(html), 900.0, 200.0, &store, &catalog).unwrap();
        let max_a = r.rgba.chunks_exact(4).map(|px| px[3]).max().unwrap_or(0);
        assert!(
            (110..=145).contains(&max_a),
            "translucent glyphs must keep ~0.5 alpha under mixed detection \
             (opaque styled space); max alpha {max_a}"
        );
    }

    /// The node-level 'color' fill override never reaches the standalone mask
    /// texture (the Canvas2D path draws only htmlContent's per-run colors and
    /// the Degraded mask fallback clears it too): mixed-alpha HTML
    /// plus a fill override must still bake PER-RUN alpha — byte-identical to
    /// the same node without the override.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn fill_override_never_reaches_the_mask() {
        let (store, catalog) = store_and_catalog();
        let html = "<p style=\"font-family: Inter; font-size: 160px; font-weight: 800\">\
                    <span style=\"color: #000000\">MO</span>\
                    <span style=\"color: rgba(0, 0, 0, 0.4)\">DA</span></p>";
        let plain = render_mask_raster(&spec(html), 900.0, 200.0, &store, &catalog).unwrap();
        let mut s = spec(html);
        s.fill_override = Some("#ff0000".into());
        let overridden = render_mask_raster(&s, 900.0, 200.0, &store, &catalog).unwrap();
        assert_eq!(
            plain.rgba, overridden.rgba,
            "fill_override leaked into the mask raster (browser Canvas2D ignores it)"
        );
    }

    /// Mixed-alpha runs DO bake per-run alpha into the texture (the browser
    /// skips the opaque-force when `detectMixedAlpha` fires): an opaque run
    /// and a 0.4-alpha run must land at distinct coverage strengths.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn mixed_alpha_bakes_per_run_alpha() {
        let (store, catalog) = store_and_catalog();
        let html = "<p style=\"font-family: Inter; font-size: 160px; font-weight: 800\">\
                    <span style=\"color: #000000\">MO</span>\
                    <span style=\"color: rgba(0, 0, 0, 0.4)\">DA</span></p>";
        let r = render_mask_raster(&spec(html), 900.0, 200.0, &store, &catalog).unwrap();
        let mut max_a = 0u8;
        let mut mid = 0usize; // pixels in the 0.4-alpha neighborhood
        for px in r.rgba.chunks_exact(4) {
            max_a = max_a.max(px[3]);
            if (70..=135).contains(&px[3]) {
                mid += 1;
            }
        }
        assert!(max_a >= 250, "opaque run missing (max alpha {max_a})");
        assert!(
            mid > 100,
            "0.4-alpha run not baked at per-run alpha ({mid} mid-alpha px)"
        );
    }
}
