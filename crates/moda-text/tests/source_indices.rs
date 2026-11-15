use moda_text::{catalog::Catalog, ops::TextNodeSpec, parser::parse_rich_text_to_paragraphs,
    vector::{VectorFontStore, VectorPlacement, ops::{build_vector_node_ops, GlyphXform, VectorEngineDecision, VectorNodeOps}}};

fn fonts() -> (Catalog, VectorFontStore) {
    let root = std::path::Path::new("/source-index-contract-fonts");
    moda_text::host::install_vfs(std::collections::HashMap::from([(
        root.join("font.ttf").to_string_lossy().into_owned(),
        include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf").to_vec(),
    )]));
    let url = "source-index-font";
    let catalog = Catalog::from_catalog_json(&serde_json::json!({"platform_fonts":[{
        "family_name":"Source", "variants":[{"weight":400,"style":"normal","msdf_ready":true,
            "atlas_urls":["vector-only"],"ttf_url":url}]
    }]}));
    let store = VectorFontStore::load_from_manifest_for_tests(root, &serde_json::json!({"files":{url:{"path":"font.ttf"}}}));
    (catalog, store)
}

fn html(body: &str) -> String { format!("<div style=\"font-family: Source; font-size: 40px\">{body}</div>") }

fn draw(body: &str, selected: std::ops::Range<usize>) -> VectorNodeOps {
    let (catalog, store) = fonts();
    let spec = TextNodeSpec { html: html(body).into(), width: Some(400.), auto_width: true,
        unit_transforms: vec![GlyphXform::IDENTITY; 128].into(), unit_range: Some(selected), ..Default::default() };
    match build_vector_node_ops(&spec, &store, &catalog, 1., VectorPlacement::VectorParity) {
        VectorEngineDecision::Vector(ops) => ops,
        other => panic!("{other:?}"),
    }
}
fn records(ops: &VectorNodeOps) -> usize { ops.ops.iter().map(|op| op.records.len()).sum() }

#[test]
fn unicode_spaces_do_not_steal_the_following_glyph_index() {
    assert_eq!(records(&draw("A\u{a0}\u{a0}B", 2..3)), 0);
    assert_eq!(records(&draw("A\u{a0}\u{a0}B", 3..4)), 1);
    for space in ['\u{a0}', '\u{1680}', '\u{2007}', '\u{202f}', '\u{3000}'] {
        // Parser + wrapping may decline uncovered space glyphs in individual fonts, so
        // also pin the source map independently of font coverage.
        let parsed = parse_rich_text_to_paragraphs(&format!("<p>{space}A{space}</p>"));
        assert_eq!(parsed[0].runs[0].style.source.values(), vec![Some(0),Some(1),Some(2)]);
    }
    assert_eq!(records(&draw("\u{2007}A\u{202f}", 1..2)), 1);
}

#[test]
fn blank_paragraphs_and_removed_controls_keep_original_origins() {
    let parsed = parse_rich_text_to_paragraphs("<p>A</p><p></p><p>B</p>");
    assert_eq!(parsed[2].runs[0].style.source.values(), vec![Some(3)]);
    assert_eq!(records(&draw("<p>A</p><p></p><p>B</p>", 3..4)), 1);
    let text = format!("A{}B B", "\u{2060}".repeat(16));
    assert_eq!(records(&draw(&text, 17..18)), 1);
    assert_eq!(records(&draw(&text, 19..20)), 1);
}

#[test]
fn case_expansion_and_contraction_preserve_following_origins() {
    for (body, expected_text, expected_map) in [
        ("<p style='text-transform:uppercase'>α\u{301}B</p>", "ΑB", vec![Some(0),Some(2)]),
        ("<p lang='tr' style='text-transform:lowercase'>I\u{307}B</p>", "ib", vec![Some(0),Some(2)]),
        ("<p style='text-transform:uppercase'>ßB</p>", "SSB", vec![Some(0),Some(0),Some(1)]),
    ] {
        let parsed = parse_rich_text_to_paragraphs(body);
        assert_eq!(parsed[0].runs[0].text, expected_text);
        assert_eq!(parsed[0].runs[0].style.source.values(), expected_map);
    }
}

#[test]
fn wrapping_and_styled_repeated_words_retain_exact_origins() {
    let (catalog, store) = fonts();
    let spec = moda_text::RichTextNodeSpec { html_content: html("<p>AB <span style='color:red'>AB</span> AB</p>"), width: Some(65.), ..Default::default() };
    let runs = moda_text::vector::extract::extract_vector_text_runs(&spec, &store, &catalog).unwrap();
    let source: Vec<_> = runs.iter().flat_map(|r| r.glyphs.iter().filter_map(|g| g.source_code_point)).collect();
    assert_eq!(source, vec![0,1,3,4,6,7]);
}

#[test]
fn one_decoded_source_preserves_blank_lines_and_sanitized_holes() {
    for source in ["A\u{a0}\u{a0}B", "A\n\nB", "A\n\u{2060}\nB", "A\u{2060}\u{2060}"] {
        let html: String = source.split('\n').map(|line| format!("<p>{line}</p>")).collect();
        let parsed = moda_text::parser::parse_rich_text_with_source(&html);
        assert_eq!(parsed.source_text, source);
        assert_eq!(moda_text::parser::source_code_point_count(&html), source.chars().count());
    }
}

#[test]
fn cjk_spacing_splits_preserve_their_source_ranges() {
    let parsed = parse_rich_text_to_paragraphs("<p style='text-autospace:ideograph-alpha'>漢A漢B</p>");
    let spaced = moda_text::cjk_spacing::apply_char_spacing(&parsed[0]).unwrap();
    assert!(spaced.runs.len() > 1);
    let origins: Vec<_> = spaced.runs.iter().flat_map(|r| r.style.source.values()).collect();
    assert_eq!(origins, vec![Some(0),Some(1),Some(2),Some(3)]);
}

#[test]
fn emoji_origins_span_all_styled_parts_of_the_cluster() {
    let (catalog, store) = fonts();
    let content = html("<p>👩<span>\u{200d}🚀</span>X</p>");
    let runs = moda_text::runs::extract_msdf_runs_by_paragraph(&content, &catalog);
    let emoji = runs.iter().find(|r| r.kind == moda_text::runs::RunKind::Emoji).unwrap();
    assert_eq!(emoji.source.values(), vec![Some(0),Some(1),Some(2)]);
    let spec = moda_text::RichTextNodeSpec { html_content: content, width: Some(400.), ..Default::default() };
    let dump = moda_text::vector::layout_rich_text_node_detail(&spec, &store, &catalog);
    let segment = dump["paragraphs"].as_array().unwrap().iter()
        .flat_map(|p| p["lines"].as_array().unwrap())
        .flat_map(|l| l["segments"].as_array().unwrap())
        .find(|s| s["emoji"] == true).unwrap();
    assert_eq!(segment["sourceIndices"], serde_json::json!([0,1,2]));
}
