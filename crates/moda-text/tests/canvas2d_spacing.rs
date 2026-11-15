//! Chrome 141 Canvas 2D with Fraunces Regular and --font-render-hinting=none.
//! At 100px: A=71.5, V=70.35, space=22.6, kern(AV)=-4.5. Tracking
//! preserves the kern and adds one advance per character, including the space.
//! These browser references exercise both production drawing and measurement.
use moda_text::{
    catalog::Catalog,
    measure::TextMeasurer,
    ops::TextNodeSpec,
    types::{default_style, InlineStyle},
    vector::{
        measure::VectorMeasurer,
        ops::{build_vector_node_ops, VectorEngineDecision},
        VectorFontStore, VectorPlacement,
    },
};

#[test]
fn canvas2d_parity_letter_spacing_keeps_kerning_and_tracks_word_gaps() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fonts");
    let url = "test://fraunces.ttf";
    let catalog = Catalog::from_catalog_json(&serde_json::json!({"platform_fonts":[{
        "family_name":"Fraunces", "variants":[{"weight":400,"style":"normal","ttf_url":url}]
    }]}));
    let store = VectorFontStore::load_from_manifest_for_tests(
        &root,
        &serde_json::json!({"files":{url:{"path":"Fraunces-SpacingSubset.ttf"}}}),
    );
    for size in [50., 100.] {
        for spacing in [0., 4., -2.] {
            let scale = size / 100.;
            let spec = TextNodeSpec {
                html: format!("<p><span style=\"font-family: Fraunces; font-size: {size}px; letter-spacing: {spacing}px\">AV AV</span></p>").into(),
                width: Some(1000.), height: Some(400.), opacity: 1.,
                ..Default::default()
            };
            let style = InlineStyle {
                font_family: "Fraunces".into(),
                font_size: size,
                letter_spacing: spacing,
                ..default_style()
            };
            for placement in [
                VectorPlacement::Canvas2dParity,
                VectorPlacement::VectorParity,
            ] {
                let VectorEngineDecision::Vector(ops) =
                    build_vector_node_ops(&spec, &store, &catalog, 1., placement)
                else {
                    panic!("expected vector drawing: {placement:?}");
                };
                assert_eq!(ops.ops.len(), 1);
                let records = &ops.ops[0].records;
                assert_eq!(records.len(), 4); // The space advances without drawing ink.
                for (record, (base, gaps)) in
                    records
                        .iter()
                        .zip([(0., 0.), (67., 1.), (159.95, 3.), (226.95, 4.)])
                {
                    let expected = base * scale + gaps * spacing;
                    assert!(
                        (f64::from(record.pen_x) - expected).abs() < 1e-3,
                        "{placement:?}, size={size}, spacing={spacing}: {} != {expected}",
                        record.pen_x
                    );
                }
                let measurer = VectorMeasurer::with_placement(&store, &catalog, placement);
                for (text, base, gaps) in
                    [("AV AV", 297.3, 5.), ("AV", 137.35, 2.), (" ", 22.6, 1.)]
                {
                    let expected = base * scale + gaps * spacing;
                    let measured = measurer.measure(text, &style);
                    assert!((measured - expected).abs() < 1e-3,
                        "{placement:?} {text:?}, size={size}, spacing={spacing}: {measured} != {expected}");
                }
                assert!(measurer.take_poison().is_none());
            }
        }
    }
}
