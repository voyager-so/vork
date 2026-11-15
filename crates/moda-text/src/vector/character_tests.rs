//! E8.22 layout tests: each character attribute (`crate::character`) and the OpenType features, through the vector
//! lane's layout dump (what the caret and exports see) and its draw records (what the canvas and PNG draw).

use serde_json::Value;

use crate::catalog::Catalog;
use crate::node_layout::RichTextNodeSpec;
use crate::ops::TextNodeSpec;
use crate::test_fixtures::store_and_catalog;
use crate::vector::measure::VectorFontStore;
use crate::vector::ops::{
    build_vector_node_ops, VectorAnalyticRecord, VectorEngineDecision, VectorPlacement,
};

fn span(style: &str, text: &str) -> String {
    format!("<p><span style=\"font-family: Inter; font-size: 40px;{style}\">{text}</span></p>")
}

fn dump(html: &str, store: &VectorFontStore, catalog: &Catalog) -> Value {
    let spec = RichTextNodeSpec {
        node_id: "t".into(),
        html_content: html.into(),
        width: Some(2000.0),
        height: Some(400.0),
        ..Default::default()
    };
    let d = crate::vector::layout_rich_text_node_vector(&spec, store, catalog);
    assert_eq!(d["engine"], "vector", "{d}");
    d
}

fn records(html: &str, store: &VectorFontStore, catalog: &Catalog) -> Vec<VectorAnalyticRecord> {
    let spec = TextNodeSpec {
        node_id: "t".into(),
        html: html.into(),
        width: Some(2000.0),
        height: Some(400.0),
        opacity: 1.0,
        ..Default::default()
    };
    match build_vector_node_ops(&spec, store, catalog, 1.0, VectorPlacement::VectorParity) {
        VectorEngineDecision::Vector(ops) => {
            ops.ops.iter().flat_map(|o| o.records.clone()).collect()
        }
        other => panic!("not vector: {other:?}"),
    }
}

/// The first line's text segments (whitespace-only ones left out).
fn segments(d: &Value) -> Vec<Value> {
    d["paragraphs"][0]["lines"][0]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["whitespaceOnly"] != true)
        .cloned()
        .collect()
}

fn glyph_xs(d: &Value) -> Vec<f64> {
    segments(d)
        .iter()
        .flat_map(|s| {
            s["glyphs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|g| g["x"].as_f64().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn width(d: &Value) -> f64 {
    d["paragraphs"][0]["lines"][0]["totalWidth"]
        .as_f64()
        .unwrap()
}

fn line(d: &Value, key: &str) -> f64 {
    d["paragraphs"][0]["lines"][0][key].as_f64().unwrap()
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn baseline_shift_moves_the_glyphs_not_the_line() {
    let (store, catalog) = store_and_catalog();
    let plain = span("", "Hx");
    let shifted = span(" --nf-baseline-shift: 8px", "Hx");
    let (a, b) = (
        dump(&plain, &store, &catalog),
        dump(&shifted, &store, &catalog),
    );
    // The line box, baseline and advances are the same.
    for key in ["lineHeightPx", "baselineY", "totalWidth"] {
        assert_eq!(line(&a, key), line(&b, key), "{key}");
    }
    assert_eq!(glyph_xs(&a), glyph_xs(&b));
    assert_eq!(segments(&b)[0]["character"]["--nf-baseline-shift"], "8px");
    // The glyphs draw 8 px higher.
    let (ra, rb) = (
        records(&plain, &store, &catalog),
        records(&shifted, &store, &catalog),
    );
    assert_eq!(ra.len(), rb.len());
    for (x, y) in ra.iter().zip(&rb) {
        assert_eq!(x.pen_x, y.pen_x);
        assert!(
            (x.baseline_y - 8.0 - y.baseline_y).abs() < 1e-4,
            "{} vs {}",
            x.baseline_y,
            y.baseline_y
        );
        assert_eq!(y.glyph_lin, [1.0, 0.0, 0.0, 1.0]);
    }
    // em resolves against the run's size (0.25 × 40 = 10 px); negative lowers.
    let em = records(
        &span(" --nf-baseline-shift: -0.25em", "Hx"),
        &store,
        &catalog,
    );
    assert!((em[0].baseline_y - ra[0].baseline_y - 10.0).abs() < 1e-4);
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn horizontal_scale_widens_advances_and_outlines() {
    let (store, catalog) = store_and_catalog();
    let plain = dump(&span("", "Hello"), &store, &catalog);
    let wide = dump(
        &span(" --nf-horizontal-scale: 150%", "Hello"),
        &store,
        &catalog,
    );
    assert!(
        (width(&wide) - 1.5 * width(&plain)).abs() < 1e-6,
        "{} vs {}",
        width(&wide),
        width(&plain)
    );
    let (xa, xb) = (glyph_xs(&plain), glyph_xs(&wide));
    for (a, b) in xa.iter().zip(&xb) {
        assert!((b - 1.5 * a).abs() < 1e-6, "glyph x {a} → {b}");
    }
    let r = records(
        &span(" --nf-horizontal-scale: 150%", "Hello"),
        &store,
        &catalog,
    );
    assert!(r
        .iter()
        .all(|g| (g.glyph_lin[0] - 1.5).abs() < 1e-6 && g.glyph_lin[3] == 1.0));
    // The outline scales from its origin: off.x = (sx − 1)·Cx.
    let g = r[0];
    let cx = (g.em_box[0] + g.em_box[2]) * 0.5 * g.font_size_px;
    assert!((g.glyph_off[0] - 0.5 * cx).abs() < 1e-3);
    // Wrapping sees the wider text: at a width the plain text fits, the scaled text wraps.
    let spec = |html: &str| RichTextNodeSpec {
        node_id: "t".into(),
        html_content: html.into(),
        width: Some(width(&plain) * 2.0 + 30.0),
        height: Some(400.0),
        ..Default::default()
    };
    let lines = |html: &str| {
        crate::vector::layout_rich_text_node_vector(&spec(html), &store, &catalog)["paragraphs"][0]
            ["lines"]
            .as_array()
            .unwrap()
            .len()
    };
    assert_eq!(lines(&span("", "Hello Hello")), 1);
    assert_eq!(
        lines(&span(" --nf-horizontal-scale: 150%", "Hello Hello")),
        2
    );
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn vertical_scale_keeps_advances_and_the_baseline() {
    let (store, catalog) = store_and_catalog();
    let html = span(" --nf-vertical-scale: 50%", "Hx");
    let (a, b) = (
        dump(&span("", "Hx"), &store, &catalog),
        dump(&html, &store, &catalog),
    );
    assert_eq!(glyph_xs(&a), glyph_xs(&b));
    assert_eq!(line(&a, "lineHeightPx"), line(&b, "lineHeightPx"));
    for g in records(&html, &store, &catalog) {
        assert_eq!(g.glyph_lin, [1.0, 0.0, 0.0, 0.5]);
        // Scaled from the baseline: off.y = (sy − 1)·Cy, so y = 0 stays put.
        let cy = (g.em_box[1] + g.em_box[3]) * 0.5 * g.font_size_px;
        assert!((g.glyph_off[1] + 0.5 * cy).abs() < 1e-3);
        let baseline_after = g.glyph_lin[3] * (0.0 - cy) + cy + g.glyph_off[1];
        assert!(baseline_after.abs() < 1e-3);
    }
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn rotation_turns_each_glyph_about_its_centre_and_keeps_spacing() {
    let (store, catalog) = store_and_catalog();
    let html = span(" --nf-character-rotation: 90deg", "Hx");
    let (a, b) = (
        dump(&span("", "Hx"), &store, &catalog),
        dump(&html, &store, &catalog),
    );
    assert_eq!(glyph_xs(&a), glyph_xs(&b));
    for g in records(&html, &store, &catalog) {
        // Counter-clockwise on screen (y-down): [cos, sin; −sin, cos].
        let [a, c, b, d] = g.glyph_lin;
        assert!(
            a.abs() < 1e-6 && (c - 1.0).abs() < 1e-6 && (b + 1.0).abs() < 1e-6 && d.abs() < 1e-6
        );
        assert_eq!(g.glyph_off, [0.0, 0.0], "about the centre, no offset");
    }
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn kerning_metrics_optical_and_none() {
    let (store, catalog) = store_and_catalog();
    let w = |family: &str, kerning: &str, text: &str| {
        let style = if kerning.is_empty() {
            String::new()
        } else {
            format!(" --nf-kerning: {kerning}")
        };
        let html = format!(
            "<p><span style=\"font-family: {family}; font-size: 40px;{style}\">{text}</span></p>"
        );
        width(&dump(&html, &store, &catalog))
    };
    // Lato kerns "AV" in the lane's metrics kerning: turning kerning off widens it.
    let (metrics, none) = (w("Lato", "", "AV"), w("Lato", "none", "AV"));
    assert!(none > metrics + 1.0, "none {none} vs metrics {metrics}");
    // Optical kerning closes the open diagonals, on a font whose kerning the lane doesn't apply too (Inter).
    for family in ["Lato", "Inter"] {
        assert!(
            w(family, "optical", "AV") < w(family, "none", "AV") - 1.0,
            "{family} AV"
        );
        assert!(
            w(family, "optical", "To") < w(family, "none", "To") - 1.0,
            "{family} To"
        );
        // Straight-sided pairs are where the font already wants them: "nn" is the reference.
        let (o, n) = (w(family, "optical", "nnn"), w(family, "none", "nnn"));
        assert!((o - n).abs() < 0.05, "{family} nnn {o} {n}");
    }
    let w = |kerning: &str, text: &str| w("Inter", kerning, text);
    // Spaces are never kerned against.
    let with_space = dump(&span(" --nf-kerning: optical", "A V"), &store, &catalog);
    let plain_space = dump(&span(" --nf-kerning: none", "A V"), &store, &catalog);
    assert_eq!(width(&with_space), width(&plain_space));
    // The dump tells the edit overlay how much each cluster moved from the font's bare advances (no kerning), which
    // is what the overlay's unkerned browser text has: nothing for no kerning, even where the font kerns the pair
    // (Lato "AV"); optical kerning's change after the A.
    let lato = |kerning: &str| {
        format!("<p><span style=\"font-family: Lato; font-size: 40px; --nf-kerning: {kerning}\">AV</span></p>")
    };
    let none = dump(&lato("none"), &store, &catalog);
    let extra = segments(&none)[0]["clusterExtraPx"]
        .as_array()
        .unwrap()
        .clone();
    assert!(
        extra.is_empty(),
        "no kerning is the bare advances: {extra:?}"
    );
    let optical = dump(&lato("optical"), &store, &catalog);
    let extra = segments(&optical)[0]["clusterExtraPx"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(extra.len(), 1);
    assert_eq!(extra[0][0], 0);
    let px = extra[0][1].as_f64().unwrap();
    assert!(px < -1.0);
    assert!((width(&none) + px - width(&optical)).abs() < 1e-6);
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn horizontal_scale_cluster_extras_sum_to_the_width_change() {
    let (store, catalog) = store_and_catalog();
    // Over the bare advances (no kerning): with a kerned base, the pairs' kerning would be missing from the extras.
    let a = dump(&span(" --nf-kerning: none", "AVAWAYTo"), &store, &catalog);
    let b = dump(
        &span(" --nf-horizontal-scale: 80%", "AVAWAYTo"),
        &store,
        &catalog,
    );
    let extra: f64 = segments(&b)[0]["clusterExtraPx"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e[1].as_f64().unwrap())
        .sum();
    assert!((width(&a) + extra - width(&b)).abs() < 1e-6);
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn defaults_written_out_change_nothing() {
    let (store, catalog) = store_and_catalog();
    let text = "Type AV To, 0123 fi";
    let plain = span("", text);
    let explicit = span(
        " --nf-baseline-shift: 0px; --nf-horizontal-scale: 100%; --nf-vertical-scale: 1; \
         --nf-character-rotation: 0deg; --nf-kerning: metrics",
        text,
    );
    assert_eq!(
        records(&plain, &store, &catalog),
        records(&explicit, &store, &catalog)
    );
    assert_eq!(
        dump(&plain, &store, &catalog),
        dump(&explicit, &store, &catalog)
    );
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn attributes_combine_and_inherit_from_the_paragraph() {
    let (store, catalog) = store_and_catalog();
    let html = "<p style=\"--nf-baseline-shift: 4px; --nf-horizontal-scale: 120%\"><span style=\"font-family: Inter; \
                font-size: 40px; --nf-character-rotation: 10deg\">Hi</span></p>";
    let seg = &segments(&dump(html, &store, &catalog))[0];
    assert_eq!(seg["character"]["--nf-baseline-shift"], "4px");
    assert_eq!(seg["character"]["--nf-horizontal-scale"], "120%");
    assert_eq!(seg["character"]["--nf-character-rotation"], "10deg");
    let r = records(html, &store, &catalog);
    let (s, c) = 10f32.to_radians().sin_cos();
    assert!(
        (r[0].glyph_lin[0] - c * 1.2).abs() < 1e-5 && (r[0].glyph_lin[2] + s * 1.2).abs() < 1e-5
    );
}

/// The OpenType features the Character and OpenType panels offer, each on a font that has it: turning it on changes
/// the glyphs the run lays out (and `liga` off undoes a ligature).
#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn opentype_features_pick_other_glyphs() {
    let (store, catalog) = store_and_catalog();
    let glyph_ids = |family: &str, features: &str, text: &str| -> Vec<u64> {
        let style = if features.is_empty() {
            String::new()
        } else {
            format!(" font-feature-settings: {features};")
        };
        let html = format!(
            "<p><span style=\"font-family: {family}; font-size: 40px;{style}\">{text}</span></p>"
        );
        segments(&dump(&html, &store, &catalog))
            .iter()
            .flat_map(|s| {
                s["glyphs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|g| g["g"].as_u64().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let cases = [
        ("Lato", "'onum' 1", "0123"),    // oldstyle figures
        ("Inter", "'tnum' 1", "1111"),   // tabular figures
        ("Inter", "'frac' 1", "1/2"),    // fractions
        ("Inter", "'ss01' 1", "a6"),     // a stylistic set
        ("Inter", "'salt' 1", "a"),      // stylistic alternates
        ("Noto Sans", "'smcp' 1", "ab"), // small caps (the font's own)
        ("Lato", "'dlig' 1", "st"),      // discretionary ligatures
    ];
    for (family, features, text) in cases {
        assert_ne!(
            glyph_ids(family, "", text),
            glyph_ids(family, features, text),
            "{family} {features} {text}"
        );
    }
    // Ligatures off: Lato's "fi" ligature splits into two glyphs.
    assert_eq!(glyph_ids("Lato", "", "fi").len(), 1);
    assert_eq!(glyph_ids("Lato", "'liga' 0", "fi").len(), 2);
}

/// EB Garamond Italic (the app's bundled Google Fonts copy, OFL) has swash capitals and more: a store over it.
fn garamond() -> (VectorFontStore, Catalog) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .join("packages/editor/public/fonts/google/ebgaramond");
    let url = "test://ebgaramond/EBGaramond-Italic[wght].ttf";
    let catalog = Catalog::from_catalog_json(&serde_json::json!({
        "platform_fonts": [{
            "family_name": "EB Garamond",
            "variants": [{ "weight": 400, "style": "italic", "ttf_url": url, "variations": { "wght": 400 } }]
        }]
    }));
    let manifest =
        serde_json::json!({ "files": { url: { "path": "EBGaramond-Italic[wght].ttf" } } });
    (
        VectorFontStore::load_from_manifest_for_tests(&root, &manifest),
        catalog,
    )
}

#[test]
#[ignore = "needs unported Moda fixture EBGaramond-Italic[wght].ttf"]
fn swash_alternates_draw_and_are_listed() {
    let (store, catalog) = garamond();
    let html = |style: &str| {
        format!(
            "<p><span style=\"font-family: 'EB Garamond'; font-style: italic; font-size: 60px;{style}\">Quill</span></p>"
        )
    };
    let first = |d: &Value| segments(d)[0]["glyphs"][0]["g"].as_u64().unwrap();
    let plain = dump(&html(""), &store, &catalog);
    let swash = dump(&html(" font-feature-settings: 'swsh' 1"), &store, &catalog);
    assert_ne!(first(&plain), first(&swash), "swsh picks Q.swsh");

    let alts = crate::vector::alternates::glyph_alternates(
        &store,
        &catalog,
        "EB Garamond",
        400,
        "italic",
        "Q",
        &[],
    )
    .expect("alternates");
    let swsh = alts
        .alternates
        .iter()
        .find(|a| a.feature == "swsh")
        .expect("a swash Q");
    assert_eq!(swsh.glyph_ids, vec![first(&swash) as u32]);
    assert_eq!(swsh.glyph_names, vec![Some("Q.swsh".to_string())]);
    assert!(swsh.path.starts_with('M') && swsh.advance > 0.0);
    assert_eq!(alts.default.glyph_names, vec![Some("Q".to_string())]);
    // Swashes rank first; small caps (caps to small caps, for a capital) are offered too.
    assert_eq!(alts.alternates[0].feature, "swsh");
    assert!(alts.alternates.iter().any(|a| a.feature == "c2sc"));
    // Every entry is distinct, and none is the default.
    let mut ids: Vec<_> = alts
        .alternates
        .iter()
        .map(|a| a.glyph_ids.clone())
        .collect();
    ids.push(alts.default.glyph_ids.clone());
    let n = ids.len();
    ids.dedup();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), n);
    // Starting from a run that already has the swash, the plain Q is not "the default" any more.
    let from_swash = crate::vector::alternates::glyph_alternates(
        &store,
        &catalog,
        "EB Garamond",
        400,
        "italic",
        "Q",
        &[crate::font_features::FontFeature {
            tag: *b"swsh",
            value: 1,
        }],
    )
    .unwrap();
    assert_eq!(
        from_swash.default.glyph_names,
        vec![Some("Q.swsh".to_string())]
    );
}

#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn indexed_alternates_are_walked() {
    let (store, catalog) = store_and_catalog();
    let alts = crate::vector::alternates::glyph_alternates(
        &store,
        &catalog,
        "Inter",
        400,
        "normal",
        "a",
        &[],
    )
    .unwrap();
    assert!(!alts.alternates.is_empty());
    let json = alts.to_json();
    let first = &json["alternates"][0];
    assert!(first["fontFeatures"].is_string() && first["glyphIds"].is_array());
}

/// What exporters writing live text get (`vector::extract`, the SVG `<text>` and PDF text-layer source): the shifted
/// baseline in each glyph's y offset, and a turned glyph's origin.
#[test]
#[cfg(vork_private_tests)] // Reads the private world-type fonts.
fn extraction_carries_the_shift_and_the_turned_origins() {
    let (store, catalog) = store_and_catalog();
    let spec = |html: &str| RichTextNodeSpec {
        node_id: "t".into(),
        html_content: html.into(),
        width: Some(2000.0),
        height: Some(400.0),
        ..Default::default()
    };
    let runs = crate::vector::extract::extract_vector_text_runs(
        &spec(&span(
            " --nf-baseline-shift: 6px; --nf-character-rotation: 90deg",
            "Hi",
        )),
        &store,
        &catalog,
    )
    .unwrap();
    let plain =
        crate::vector::extract::extract_vector_text_runs(&spec(&span("", "Hi")), &store, &catalog)
            .unwrap();
    assert_eq!(runs[0].character.rotation, 90.0);
    for (g, p) in runs[0].glyphs.iter().zip(&plain[0].glyphs) {
        assert_eq!(g.x_px, p.x_px);
        assert_eq!(g.y_offset_px, p.y_offset_px + 6.0);
        // 90° about the centre C moves the origin to (Cx + Cy, Cy − Cx) (y-down): right of and below the pen for a
        // glyph above the baseline.
        assert!(g.rotation_origin_px.0 > 0.0, "{:?}", g.rotation_origin_px);
    }
    assert_eq!(plain[0].glyphs[0].rotation_origin_px, (0.0, 0.0));
}
