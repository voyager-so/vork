//! CSS 2.1 §10.8: line-height is a CSS-pixel length, not a rounded device-pixel height.
//! Exercise the rendered glyph positions as well as the layout used by animation and bounds.
use moda_text::{
    catalog::Catalog,
    ops::TextNodeSpec,
    types::LineHeightPolicy,
    vector::{
        layout_rich_text_node_vector,
        ops::{build_vector_node_ops, VectorEngineDecision},
        VectorFontStore, VectorPlacement,
    },
    RichTextNodeSpec,
};

fn fonts() -> (VectorFontStore, Catalog) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fonts");
    let url = "test://fractional/fraunces.ttf";
    let catalog = Catalog::from_catalog_json(&serde_json::json!({"platform_fonts":[{
        "family_name":"Fraunces", "variants":[{"weight":400,"style":"normal","ttf_url":url}]
    }]}));
    let store = VectorFontStore::load_from_manifest_for_tests(
        &root,
        &serde_json::json!({"files":{url:{"path":"Fraunces-SpacingSubset.ttf"}}}),
    );
    (store, catalog)
}

#[test]
fn fractional_lines_keep_layout_and_draw_in_step_without_changing_moda_default() {
    let (store, catalog) = fonts();
    // Soft wraps, hard breaks, paragraph boundaries, and empty paragraphs share the same convention.
    for height in [24.25_f64, 24.5, 24.75, 25.125] {
        let style = format!("font-family: Fraunces; font-size: 16px; line-height: {height}px");
        for (html, width, expected_lines, glyph_lines) in [
            (
                format!("<p style='{style}'>A<br>A<br>A</p>"),
                200.,
                3,
                vec![0, 1, 2],
            ),
            (
                format!("<p style='{style}'>A A A</p>"),
                15.,
                3,
                vec![0, 1, 2],
            ),
            (
                format!("<p style='{style}'>A</p><p style='{style}'></p><p style='{style}'>A</p>"),
                200.,
                3,
                vec![0, 2],
            ),
        ] {
            for policy in [LineHeightPolicy::Figma, LineHeightPolicy::Css] {
                let line_height = if policy == LineHeightPolicy::Figma {
                    height.round()
                } else {
                    height
                };
                let rich = RichTextNodeSpec {
                    html_content: html.clone(),
                    width: Some(width),
                    height: Some(200.),
                    line_height_policy: policy,
                    ..Default::default()
                };
                let layout = layout_rich_text_node_vector(&rich, &store, &catalog);
                assert_eq!(layout["engine"], "vector", "{layout}");
                let lines: Vec<_> = layout["paragraphs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|p| p["lines"].as_array().unwrap())
                    .collect();
                assert_eq!(lines.len(), expected_lines, "{html}: {layout}");
                for (i, line) in lines.iter().enumerate() {
                    assert!(
                        (line["lineHeightPx"].as_f64().unwrap() - line_height).abs() < 1e-8,
                        "{line}"
                    );
                    assert!(
                        (line["lineTopY"].as_f64().unwrap() - i as f64 * line_height).abs() < 1e-8,
                        "{line}"
                    );
                }
                let spec = TextNodeSpec {
                    html: html.clone().into(),
                    width: Some(width),
                    height: Some(200.),
                    line_height_policy: policy,
                    opacity: 1.,
                    ..Default::default()
                };
                let VectorEngineDecision::Vector(ops) = build_vector_node_ops(
                    &spec,
                    &store,
                    &catalog,
                    1.,
                    VectorPlacement::VectorParity,
                ) else {
                    panic!("expected vector draw")
                };
                let records: Vec<_> = ops.ops.iter().flat_map(|op| &op.records).collect();
                assert_eq!(records.len(), glyph_lines.len());
                for (record, line) in records.iter().zip(&glyph_lines) {
                    assert!(
                        (f64::from(record.baseline_y)
                            - lines[*line]["baselineY"].as_f64().unwrap())
                        .abs()
                            < 1e-4
                    );
                }
                for pair in records.windows(2).zip(glyph_lines.windows(2)) {
                    let (records, indexes) = pair;
                    assert!(
                        (f64::from(records[1].baseline_y - records[0].baseline_y)
                            - (indexes[1] - indexes[0]) as f64 * line_height)
                            .abs()
                            < 1e-4
                    );
                }
                if policy == LineHeightPolicy::Figma {
                    let default = TextNodeSpec {
                        line_height_policy: Default::default(),
                        ..spec
                    };
                    let VectorEngineDecision::Vector(default_ops) = build_vector_node_ops(
                        &default,
                        &store,
                        &catalog,
                        1.,
                        VectorPlacement::VectorParity,
                    ) else {
                        panic!("expected default vector draw")
                    };
                    assert_eq!(ops, default_ops);
                }
            }
        }
    }
}
