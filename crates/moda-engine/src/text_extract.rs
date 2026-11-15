//! Read-only page text-run extraction — the PDF text layer's engine seam
//! (flatten=False sandwich, `moda-cli --pdf-text-layer`).
//!
//! Walks one page's display list and, for every richtext item, re-runs the
//! SAME vector layout the draw path shapes with
//! (`moda_text::vector::extract::extract_vector_text_runs` — the structured
//! twin of the `dump_rich_text_node_vector` walk), then attaches each node's
//! world affine re-based to the page origin. The result is everything the
//! invisible PDF overlay needs: glyph ids + placed origins (engine truth, so
//! the layer aligns with the raster by construction), per-run text + byte
//! ranges (ToUnicode), and the font variant to embed.
//!
//! Deliberately READ-ONLY: no plan-builder (`build_plan_retained` /
//! camera-lane) code is touched — byte-exact render plans are unaffected.
//! Nodes the vector lane would not draw (MSDF/canvas2d routing, complex-script
//! routing, resolution fallback, shape error) are OMITTED — the raster still
//! shows their pixels — never an error; extraction itself degrades to an empty
//! layer if the text host is unavailable. Also omitted, because the plain
//! vector layout cannot mirror their raster placement or visibility:
//! clip-mask SOURCE nodes (the renderer never draws them — frame.rs skips
//! `text-mask-source` items, so their text must not be searchable either),
//! KineticText nodes carrying per-glyph `unitTransforms`, and nodes with an
//! active `distort` homography. Scope (v1): richtext nodes; shape/connector
//! labels stay raster-only. Accepted v1 spillover: a richtext node PARTIALLY
//! clipped by a container ancestor contributes its full runs (clipping is a
//! draw-time concern; the common case is fully-visible text in a container,
//! and per-line clip intersection is a follow-up).

use moda_scene::transform;
use moda_scene::types::PaintKind;
use moda_scene::Scene;
use moda_text::node_layout::RichTextNodeSpec;
use moda_text::vector::extract::{extract_vector_text_runs, ExtractedGlyph};

/// One extracted run in PAGE space: node-local glyph placements plus the
/// node-local→page-px affine (rotation/scale ride the transform).
#[derive(Debug, Clone)]
pub struct PageTextRun {
    pub family: String,
    pub weight: i64,
    /// "normal" | "italic".
    pub style: String,
    pub font_size_px: f64,
    /// Baseline y in node-local px (pair with `transform`).
    pub baseline_y_px: f64,
    /// Final pen x after the last glyph (node-local px) — the last glyph's
    /// PDF advance, so its selection box isn't zero-width.
    pub end_x_px: f64,
    /// The run's shaping-normalized text (`glyphs[].text_range` indexes it).
    pub text: String,
    /// Node-local px → page canvas px: `[a, b, c, d, e, f]` mapping
    /// `(x, y) → (a·x + c·y + e, b·x + d·y + f)`.
    pub transform: [f64; 6],
    pub glyphs: Vec<ExtractedGlyph>,
}

/// Extract every vector-lane text run of one page, in display-list order.
/// Returns an empty vec when the page is out of range, the text host is
/// unavailable, or no node routes through the vector lane.
pub fn extract_page_text_runs(scene: &Scene, page_index: usize) -> Vec<PageTextRun> {
    let Some(page) = scene.doc().pages.get(page_index) else {
        return Vec::new();
    };
    let page_id = page.id().to_string();
    let (page_x, page_y) = scene.page_position(page_index);

    // The same store/catalog the vector DRAW lane uses (thread-local store over
    // the installed/discovered TextHost) — extraction truth = draw truth.
    let Ok((store, host)) = crate::text_draw::vector_store_and_host() else {
        return Vec::new();
    };
    let catalog = &host.catalog;

    let dl = moda_scene::build_display_list(scene, &page_id);
    let mut out: Vec<PageTextRun> = Vec::new();
    for item in &dl.items {
        if item.kind != PaintKind::Richtext || !item.is_primary_node_item {
            continue;
        }
        // Clip-mask SOURCE nodes are never drawn (frame.rs skips them), so
        // their text must not be searchable/copyable either.
        if item.fallback_reasons.contains(&"text-mask-source") {
            continue;
        }
        let Some(node) = scene.effective_node(&item.node_id) else {
            continue;
        };
        // KineticText per-glyph transforms and the distort homography displace
        // glyphs at draw time; the plain vector layout cannot mirror them, so a
        // text layer here would be misaligned. Omit (raster shows the pixels).
        if node.array("unitTransforms").is_some_and(|a| !a.is_empty())
            || crate::paint::node_distort(node).is_some()
            || crate::text_draw::node_text_path(node).is_some()
        {
            continue;
        }
        // Area type (E8.14): the lines this frame draws; a later frame of a thread shows its head's story.
        let area = crate::text_area::node_area_spec(scene, node);
        let html = if area.as_ref().is_some_and(|a| a.frames.len() > 1) {
            crate::text_area::story_html(scene, node)
        } else {
            node.str("htmlContent").unwrap_or("")
        };
        let spec = RichTextNodeSpec {
            node_id: node.id().to_string(),
            html_content: moda_text::lang::html_with_document_lang(html, crate::text_draw::document_locale(scene)),
            width: node.f64("width"),
            height: node.f64("height"),
            paragraph_spacing: node.f64("paragraphSpacing").unwrap_or(0.0),
            line_height_policy: Default::default(),
            vertical_align: node.str("verticalAlign").map(str::to_string),
            truncation: crate::text_draw::node_truncation(node),
            auto_width: crate::text_draw::node_auto_width(node),
            area,
        };
        let Some(runs) = extract_vector_text_runs(&spec, &store, catalog) else {
            continue; // node not on the vector lane — raster-only
        };
        // Node-local → page px: world affine re-based to the page origin.
        let world = item.world_affine.unwrap_or_else(transform::identity);
        let a = transform::compose(transform::translate(-page_x, -page_y), world);
        let xf = [a.a, a.b, a.c, a.d, a.e, a.f];
        for r in runs {
            out.push(PageTextRun {
                family: r.family,
                weight: r.weight,
                style: r.style,
                font_size_px: r.font_size_px,
                baseline_y_px: r.baseline_y_px,
                end_x_px: r.end_x_px,
                text: r.text,
                transform: xf,
                glyphs: r.glyphs,
            });
        }
    }
    out
}

/// Raw TTF bytes for a variant an extracted run references (the bytes the
/// vector lane itself parsed) — the caller's font-embedding table source.
pub fn vector_font_bytes(family: &str, weight: i64, style: &str) -> Option<std::sync::Arc<Vec<u8>>> {
    let (store, host) = crate::text_draw::vector_store_and_host().ok()?;
    store.bytes_for_variant(&host.catalog, family, weight, style)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn scene_with_text(nodes: moda_scene::value::Value, node_ids: Vec<&str>) -> Scene {
        Scene::load(&json!({
            "nodes": nodes,
            "pages": [{"id":"p1","name":"P","width":800.0,"height":600.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],
                       "nodeIds": node_ids}]
        })).unwrap()
    }

    /// Fixture-backed: runs against the vendored oracle fonts dir (the same
    /// discovery the engine's draw lane uses).
    #[test]
    fn extracts_richtext_runs_in_page_space() {
        let scene = scene_with_text(
            json!([
                {"id":"t1","type":"richtext","x":40.0,"y":30.0,"width":300.0,"height":100.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px;\">Hello</span></p>"}
            ]),
            vec!["t1"],
        );
        let runs = extract_page_text_runs(&scene, 0);
        assert!(!runs.is_empty(), "vector richtext must extract");
        let r = &runs[0];
        assert_eq!(r.family, "Inter");
        assert!(r.text.contains("Hello"));
        assert!(!r.glyphs.is_empty());
        // Page-space transform carries the node offset (page at origin in
        // page-render space, node at (40, 30)).
        assert!((r.transform[4] - 40.0).abs() < 1e-9);
        assert!((r.transform[5] - 30.0).abs() < 1e-9);
        // Font bytes resolve for embedding.
        assert!(vector_font_bytes(&r.family, r.weight, &r.style).is_some());
    }

    #[test]
    fn out_of_range_page_and_non_text_scene_are_empty() {
        let scene = scene_with_text(
            json!([
                {"id":"r1","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]}
            ]),
            vec!["r1"],
        );
        assert!(extract_page_text_runs(&scene, 0).is_empty());
        assert!(extract_page_text_runs(&scene, 5).is_empty());
    }

    #[test]
    fn clip_mask_source_text_is_omitted() {
        // "t1" is a group's clipMaskNodeId → the renderer never draws it
        // (frame.rs text-mask-source skip), so its text must not extract.
        let scene = Scene::load(&json!({
            "nodes": [
                {"id":"g","type":"group","x":0.0,"y":0.0,"children":["t1","c"],
                 "clipMaskNodeId":"t1"},
                {"id":"t1","type":"richtext","x":10.0,"y":10.0,"width":300.0,"height":100.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px;\">Masked secret</span></p>"},
                {"id":"c","type":"rectangle","x":0.0,"y":0.0,"width":50.0,"height":50.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]}
            ],
            "pages": [{"id":"p1","name":"P","width":800.0,"height":600.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],
                       "nodeIds":["g"]}]
        })).unwrap();
        assert!(extract_page_text_runs(&scene, 0).is_empty());
    }

    #[test]
    fn kinetic_unit_transforms_node_is_omitted() {
        // Per-glyph unitTransforms displace glyphs at draw time — the plain
        // layout can't mirror them, so the node is omitted (raster-only).
        let scene = scene_with_text(
            json!([
                {"id":"t1","type":"richtext","x":40.0,"y":30.0,"width":300.0,"height":100.0,
                 "rotation":0,"scale":1,
                 "unitTransforms":[{"dx":4.0,"dy":-2.0}],
                 "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px;\">Kinetic</span></p>"}
            ]),
            vec!["t1"],
        );
        assert!(extract_page_text_runs(&scene, 0).is_empty());
        // Empty array = no per-glyph transforms → still extracts.
        let scene = scene_with_text(
            json!([
                {"id":"t1","type":"richtext","x":40.0,"y":30.0,"width":300.0,"height":100.0,
                 "rotation":0,"scale":1,
                 "unitTransforms":[],
                 "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px;\">Static</span></p>"}
            ]),
            vec!["t1"],
        );
        assert!(!extract_page_text_runs(&scene, 0).is_empty());
    }

    #[test]
    fn complex_script_node_is_omitted() {
        // The draw walk routes complex scripts (bidi/joining/reordering) to the
        // cosmic-text lane; a vector-layout text layer would misalign. Omit.
        let scene = scene_with_text(
            json!([
                {"id":"t1","type":"richtext","x":40.0,"y":30.0,"width":300.0,"height":100.0,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px;\">שלום עולם</span></p>"}
            ]),
            vec!["t1"],
        );
        assert!(extract_page_text_runs(&scene, 0).is_empty());
    }

    #[test]
    fn rotated_richtext_carries_rotation_in_transform() {
        let scene = scene_with_text(
            json!([
                {"id":"t1","type":"richtext","x":100.0,"y":100.0,"width":200.0,"height":80.0,
                 "rotation":90.0,"scale":1,
                 "htmlContent":"<p><span style=\"font-family: Inter; font-size: 24px;\">Spin</span></p>"}
            ]),
            vec!["t1"],
        );
        let runs = extract_page_text_runs(&scene, 0);
        assert!(!runs.is_empty());
        let t = runs[0].transform;
        // 90° rotation: linear part ~ [0, 1, -1, 0].
        assert!(t[0].abs() < 1e-9 && (t[1] - 1.0).abs() < 1e-9);
        assert!((t[2] + 1.0).abs() < 1e-9 && t[3].abs() < 1e-9);
    }
}
