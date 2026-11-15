use moda_text::{
    catalog::Catalog,
    ops::TextNodeSpec,
    vector::{
        ops::{build_vector_node_ops, VectorEngineDecision, VectorNodeOps},
        VectorFontStore, VectorPlacement,
    },
};

fn draw_with(css: &str, stroke: bool, prepare: impl FnOnce(&mut TextNodeSpec)) -> VectorNodeOps {
    let root = std::path::Path::new("/rmbugs-vector-fonts");
    moda_text::host::install_vfs(std::collections::HashMap::from([(
        root.join("variable.ttf").to_string_lossy().into_owned(),
        include_bytes!("fonts/NotoSans-WeightSubset.ttf").to_vec(),
    )]));
    let url = "test://rmbugs/variable.ttf";
    let catalog = Catalog::from_catalog_json(&serde_json::json!({"platform_fonts":[{
        "family_name":"Variable", "variants":[{"weight":400,"style":"normal","msdf_ready":true,
            "atlas_urls":["test://vector-only"],"ttf_url":url}]
    }]}));
    let store = VectorFontStore::load_from_manifest_for_tests(
        root,
        &serde_json::json!({"files":{url:{"path":"variable.ttf"}}}),
    );
    let mut spec = TextNodeSpec {
        html: format!("<p><span style=\"font-family: Variable; font-size: 80px; color: black; {css}\">Hamburgefonts</span></p>").into(),
        width: Some(1000.), height: Some(200.),
        stroke_color: stroke.then(|| "red".into()), stroke_width: stroke.then_some(10.),
        ..Default::default()
    };
    prepare(&mut spec);
    match build_vector_node_ops(&spec, &store, &catalog, 1., VectorPlacement::VectorParity) {
        VectorEngineDecision::Vector(ops) => ops,
        other => panic!("{other:?}"),
    }
}

fn draw(css: &str, stroke: bool) -> VectorNodeOps {
    draw_with(css, stroke, |_| {})
}

#[test]
fn font_weight_instances_variable_outlines_and_layout() {
    let regular = draw("", false);
    for weight in [100., 400., 650.5, 800., 900., 1000.] {
        let automatic = draw(&format!("font-weight: {weight}"), false);
        let explicit = draw(&format!("font-variation-settings: 'wght' {weight}"), false);
        assert!(
            automatic == explicit,
            "font-weight {weight} must select the same outlines and positions as wght"
        );
        if weight != 400. {
            assert_ne!(automatic.curve_points, regular.curve_points);
        }
    }
    assert_eq!(
        draw(
            "font-weight: 800; font-variation-settings: 'wght' 400",
            false
        ),
        regular
    );
    assert_eq!(
        draw("font-weight: 1000", false),
        draw("font-weight: 900", false),
        "axis limits clamp"
    );
}

#[test]
fn text_stroke_paints_over_fill() {
    let ops = draw("", true);
    let fill = ops
        .ops
        .iter()
        .position(|op| op.stroke_px == 0. && !op.records.is_empty())
        .unwrap();
    let stroke = ops.ops.iter().position(|op| op.stroke_px == 10.).unwrap();
    assert!(stroke > fill, "CSS/SVG stroke paints over fill");
}

#[test]
fn unit_ranges_keep_full_layout_and_shared_pivots() {
    let draw_range = |range| draw_with("", true, |spec| {
        let count = moda_text::plain_text::parse_plain_text(&spec.html).chars().count();
        spec.unit_transforms = vec![moda_text::vector::ops::GlyphXform::resolve(4., 7., 1.5, 30., 1.); count].into();
        spec.unit_groups = vec![0; count].into();
        spec.unit_range = range;
    });
    let full = draw_range(None);
    let selected = draw_range(Some(1..3));
    assert_eq!(full.ops.len(), selected.ops.len());
    for (full_op, selected_op) in full.ops.iter().zip(&selected.ops) {
        assert_eq!(selected_op.records.len(), 2);
        for (expected, actual) in full_op.records[1..3].iter().zip(&selected_op.records) {
            let curves = |ops: &VectorNodeOps, record: &moda_text::vector::ops::VectorAnalyticRecord| {
                let start = record.first_curve as usize * 6;
                ops.curve_points[start..start + record.curve_count as usize * 6].to_vec()
            };
            assert_eq!(curves(&full, expected), curves(&selected, actual));
            // The selected batch packs only its glyphs' curves, so their buffer indices differ.
            let mut expected = *expected;
            expected.first_curve = actual.first_curve;
            assert_eq!(expected, *actual);
        }
    }
}
