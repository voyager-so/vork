//! Independent HarfBuzz advances guard the production shaping dispatch, not just
//! the rustybuzz wrapper. The JSON records HarfBuzz version and font SHA-256.
//! Reference generation: uharfbuzz Face(font_bytes), Font(face), Buffer.add_str(text),
//! Buffer.guess_segment_properties(), shape(font, buffer, feature_dict); record
//! glyph_infos.codepoint/cluster and glyph_positions.x_advance at the default UPEM scale.
//! Buffer.add_str reports source code-point clusters, matching the canvas contract.
use moda_text::{
    catalog::Catalog,
    font_features::FontFeature,
    measure::TextMeasurer,
    types::InlineStyle,
    vector::{
        measure::{shape_segment, VectorMeasurer},
        VectorFontStore, VectorPlacement,
    },
};

#[test]
fn default_and_explicit_features_match_harfbuzz() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("harfbuzz-advances.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let path = case["font"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let settings = case["features"].as_str().unwrap();
        let url = "test://harfbuzz/font.ttf";
        let catalog = Catalog::from_catalog_json(&serde_json::json!({"platform_fonts":[{
            "family_name":"Reference", "variants":[{"weight":400,"style":"normal","ttf_url":url}]
        }]}));
        let store = VectorFontStore::load_from_manifest_for_tests(
            root,
            &serde_json::json!({"files":{url:{"path":path}}}),
        );
        let mut style = InlineStyle {
            font_family: "Reference".into(),
            font_size: 80.,
            ..moda_text::types::default_style()
        };
        if !settings.is_empty() {
            style.font_features = settings
                .split(',')
                .map(|setting| {
                    let (tag, value) = setting.split_once('=').unwrap();
                    FontFeature {
                        tag: tag.as_bytes().try_into().unwrap(),
                        value: value.parse().unwrap(),
                    }
                })
                .collect();
        }
        let font = store
            .font_for_variant(&catalog, "Reference", 400, "normal", Some(80.))
            .unwrap();
        let shaped = shape_segment(
            &store,
            &catalog,
            "Reference",
            400,
            "normal",
            Some(80.),
            &font,
            text,
            &style,
            VectorPlacement::VectorParity,
        )
        .unwrap();
        let advances: Vec<f64> = case["advances"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let glyphs: Vec<u32> = case["glyphs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let reference_clusters: Vec<usize> = case["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            shaped.iter().map(|g| g.cluster).collect::<Vec<_>>(),
            reference_clusters,
            "source code-point clusters: {path} {text:?} {settings}"
        );
        assert_eq!(font.units_per_em as u64, case["upem"].as_u64().unwrap());
        assert_eq!(
            shaped.iter().map(|g| g.glyph).collect::<Vec<_>>(),
            glyphs,
            "{path} {text:?} {settings}"
        );
        assert_eq!(
            shaped.iter().map(|g| g.x_advance).collect::<Vec<_>>(),
            advances,
            "{path} {text:?} {settings}"
        );
        for size in [45., 80.] {
            for spacing in [0., 2.] {
                style.font_size = size;
                style.letter_spacing = spacing;
                let spaced = shape_segment(
                    &store,
                    &catalog,
                    "Reference",
                    400,
                    "normal",
                    Some(size),
                    &font,
                    text,
                    &style,
                    VectorPlacement::VectorParity,
                )
                .unwrap();
                assert_eq!(
                    spaced.iter().map(|g| g.x_advance).collect::<Vec<_>>(),
                    advances,
                    "letter spacing must retain OpenType advances: {path} {text:?} {settings}"
                );
                // Canvas layout adds tracking after each source cluster, including
                // the final one; ligature components share one cluster.
                assert_eq!(
                    spaced.iter().map(|g| g.cluster).collect::<Vec<_>>(),
                    reference_clusters,
                    "tracked source clusters: {path} {text:?} {settings}"
                );
                let clusters = reference_clusters
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                let expected_width = advances.iter().sum::<f64>() * size
                    / f64::from(font.units_per_em)
                    + spacing * clusters as f64;
                let measurer = VectorMeasurer::new(&store, &catalog);
                let width = measurer.measure(text, &style);
                assert!(measurer.take_poison().is_none());
                assert!((width - expected_width).abs() < 1e-8,
                    "{path} {text:?} {settings} size={size} spacing={spacing}: {width} != {expected_width}");
            }
        }
    }
}

#[test]
fn complex_scripts_keep_the_shaping_fallback() {
    use moda_text::{
        ops::TextNodeSpec,
        vector::ops::{build_vector_node_ops, VectorEngineDecision},
    };
    let catalog = Catalog::from_catalog_json(&serde_json::json!({"platform_fonts":[]}));
    let store = VectorFontStore::load_from_manifest_for_tests(
        std::path::Path::new("/"),
        &serde_json::json!({"files":{}}),
    );
    for text in ["مرحبا", "বাংলা", "देवनागरी", "Hello שלום"] {
        let spec = TextNodeSpec {
            html: format!("<p>{text}</p>").into(),
            ..Default::default()
        };
        assert!(
            matches!(build_vector_node_ops(&spec, &store, &catalog, 1., VectorPlacement::VectorParity),
            VectorEngineDecision::Fallback { reason, .. } if reason == "complex-script"),
            "{text}"
        );
    }
}
