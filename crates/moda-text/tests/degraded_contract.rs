use moda_text::{degraded::DegradedEngine, ops::TextNodeSpec};

fn engine() -> DegradedEngine {
    let root = std::path::Path::new("/rev1770-contract-fonts");
    moda_text::host::install_vfs(std::collections::HashMap::from([
        (root.join("MANIFEST.json").to_string_lossy().into_owned(), br#"{"systemFonts":false,"keys":{"han":{"family":"Noto Sans CJK JP","ttfPath":"han.otf"},"latin":{"family":"Liberation Sans","ttfPath":"latin.ttf"},"arabic":{"family":"Noto Sans Arabic","ttfPath":"arabic.ttf"}}}"#.to_vec()),
        (root.join("han.otf").to_string_lossy().into_owned(), include_bytes!("fonts/HanTest.otf").to_vec()),
        (root.join("latin.ttf").to_string_lossy().into_owned(), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf").to_vec()),
        (root.join("arabic.ttf").to_string_lossy().into_owned(), include_bytes!("../../vork-std/tests/fonts/NotoSansArabic-Regular.ttf").to_vec()),
    ]));
    DegradedEngine::load(None, Some(root))
}

fn spec(html: &str) -> TextNodeSpec {
    TextNodeSpec {
        html: html.into(),
        width: Some(400.),
        height: Some(160.),
        ..Default::default()
    }
}

fn refuses(html: &str, diagnostic: &str) {
    let engine = engine();
    let spec = spec(html);
    let layout = engine.layout_json(&spec).expect_err("layout must refuse");
    let raster = engine.render_node(&spec).expect_err("raster must refuse");
    let outlines = engine
        .glyph_outlines(&spec)
        .expect_err("outlines must refuse");
    for error in [layout, raster, outlines] {
        assert!(error.contains(diagnostic), "{error}");
        assert_eq!(error.lines().count(), 1, "{error}");
    }
}

#[test]
fn missing_glyphs_name_uncovered_characters() {
    refuses("<p style='font-family: Liberation Sans'>你好</p>", "你好");
    refuses(
        "<p style='font-family: Noto Sans Arabic'>سلام <span>你好</span></p>",
        "你好",
    );
}

#[test]
fn variation_settings_refuse_on_paragraphs_and_runs() {
    for html in [
        "<p style=\"font-family: Noto Sans Arabic; font-variation-settings: 'wght' 900\">سلام</p>",
        "<p style='font-family: Noto Sans Arabic'>سلام <span style=\"font-variation-settings: 'wght' 900\">سلام</span></p>",
    ] {
        refuses(html, "font-variation-settings");
    }
}

#[test]
fn contextual_letter_spacing_refuses_on_paragraphs_and_runs() {
    for spacing in [-1, 1] {
        refuses(
            &format!(
                "<p style='font-family: Noto Sans Arabic; letter-spacing: {spacing}px'>سلام</p>"
            ),
            "letter-spacing",
        );
        refuses(&format!("<p style='font-family: Noto Sans Arabic'>سلام <span style='letter-spacing: {spacing}px'>سلام</span></p>"), "letter-spacing");
    }
}

#[test]
fn covered_joined_arabic_and_spaced_latin_still_draw() {
    let engine = engine();
    for html in [
        "<p style='font-family: Noto Sans Arabic; font-size: 64px' dir='rtl'>سلام</p>",
        "<p style='font-family: Liberation Sans; letter-spacing: 1px'>Latin</p>",
        "<p style='font-family: Noto Sans Arabic'>س\u{200c}لام</p>",
        "<p style='font-family: Liberation Sans'>Latin سلام</p>",
    ] {
        let spec = spec(html);
        engine.layout_json(&spec).unwrap();
        let raster = engine.render_node(&spec).unwrap();
        assert!(raster.rgba.chunks_exact(4).any(|p| p[3] != 0));
    }
    assert_eq!(
        engine
            .glyph_outlines(&spec("<p style='font-family: Noto Sans Arabic'>سلام</p>"))
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn shaped_paragraph_preserves_cjk_wrap_opportunities() {
    let engine = engine();
    for html in [
        "<p style='font-family: Noto Sans CJK JP; font-size: 40px'>漢字漢字 مرحبا</p>",
        "<p style='font-family: Noto Sans CJK JP; font-size: 40px'><span style='color:blue'>漢</span>字漢字 مرحبا</p>",
        "<ul><li style='font-family: Noto Sans CJK JP; font-size: 40px'>漢字漢字 مرحبا</li></ul>",
    ] {
        let mut spec = spec(html);
        spec.width = Some(80.);
        let layout = engine.layout_json(&spec).unwrap();
        // Intrinsic measurement must use the same breaks as the drawn lines. Arabic may overflow intact.
        if html.starts_with("<p") {
            assert!(layout["logicalW"].as_f64().unwrap() < 100., "{layout}");
        }
        let lines = layout["paragraphs"][0]["lines"].as_array().unwrap();
        let han_lines: Vec<String> = lines.iter().map(|line| {
            line["segments"].as_array().unwrap().iter()
                .map(|seg| seg["text"].as_str().unwrap()).collect::<String>()
        }).filter(|line| line.contains('漢') || line.contains('字')).collect();
        assert!(han_lines.len() >= 2, "{layout}");
        assert_eq!(han_lines.concat(), "漢字漢字");
        for line in &han_lines {
            assert!(line.chars().count() <= 2, "{layout}");
        }
    }
}

#[test]
fn css_fractional_line_height_matches_shaped_layout_and_raster() {
    let engine = engine();
    for height in [24.25_f64, 25.5, 26.75] {
        let style = format!("font-family: Noto Sans Arabic; font-size: 16px; line-height: {height}px");
        for html in [
            format!("<p style='{style}'>سلام</p><p style='{style}'>سلام</p><p style='{style}'>سلام</p>"),
            format!("<p style='{style}'>سلام</p><p style='{style}'></p><p style='{style}'>سلام</p>"),
        ] {
            let mut spec = spec(&html);
            spec.line_height_policy = moda_text::types::LineHeightPolicy::Css;
            let layout = engine.layout_json(&spec).unwrap();
            let lines: Vec<_> = layout["paragraphs"].as_array().unwrap().iter()
                .flat_map(|p| p["lines"].as_array().unwrap()).collect();
            assert_eq!(lines.len(), 3, "{layout}");
            let origin = lines[0]["lineTopY"].as_f64().unwrap();
            for (i, line) in lines.iter().enumerate() {
                assert!((line["lineHeightPx"].as_f64().unwrap()-height).abs() < 1e-8, "{line}");
                assert!((line["lineTopY"].as_f64().unwrap()-origin-i as f64*height).abs() < 1e-8, "{line}");
            }
            engine.render_node(&spec).expect("fractional shaping raster");
            engine.glyph_outlines(&spec).expect("fractional shaping outlines");
        }
    }
}

#[test]
fn css_normal_empty_lines_allocate_the_same_font_metrics_as_layout() {
    let engine = engine();
    for family in ["Noto Sans Arabic", "Liberation Sans"] {
        for size in [16.25, 31.5] {
            let paragraph = format!("<p style='font-family: {family}; font-size: {size}px; line-height: normal'></p>");
            let mut spec = spec(&paragraph.repeat(3));
            spec.line_height_policy = moda_text::types::LineHeightPolicy::Css;
            let layout = engine.layout_json(&spec).unwrap();
            let lines: Vec<_> = layout["paragraphs"].as_array().unwrap().iter()
                .flat_map(|p| p["lines"].as_array().unwrap()).collect();
            assert_eq!(lines.len(), 3, "{layout}");
            let line_height = lines[0]["lineHeightPx"].as_f64().unwrap();
            let (_, allocated_height) = engine.canvas_logical_size(&spec);
            assert_eq!(allocated_height, (3. * line_height).ceil(), "{family} {size}: {layout}");
            let origin = lines[0]["lineTopY"].as_f64().unwrap();
            let last = lines.last().unwrap();
            let bottom = last["lineTopY"].as_f64().unwrap() + last["lineHeightPx"].as_f64().unwrap() - origin;
            assert!(allocated_height >= bottom && allocated_height < bottom + 1., "{layout}");
            engine.render_node(&spec).expect("normal empty lines raster");
        }
    }
}
