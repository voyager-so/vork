//! Brush sizing: by default the stroke width is the width of the mark (as in
//! Figma); `scaleWithWeight: true` is Illustrator's (the weight multiplies the
//! designed size).

use crate::value::{json, Value};

use super::common::Placement;
use super::tests_art::{rect_art, straight};

/// `v` without the test helpers' `scaleWithWeight: true`.
fn absolute(mut v: Value) -> Value {
    v.as_object_mut().unwrap().remove("scaleWithWeight");
    v
}

fn height(p: &Placement) -> f64 {
    p.bounds.max.y - p.bounds.min.y
}

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn art_height_is_the_stroke_width() {
    // Artwork 100 × 20 on a straight 300 px line.
    let art = rect_art(100.0, 20.0);
    for width in [1.0, 6.0, 30.0] {
        let node = super::tests_art::line_node(straight(300.0), json!({"width": width}));
        let abs = super::tests_art::place(&node, &absolute(super::tests_art::brush(art.clone(), json!({}))));
        assert!(near(height(&abs), width, 1e-6), "width {width}: {}", height(&abs));
        let ill = super::tests_art::place(&node, &super::tests_art::brush(art.clone(), json!({})));
        assert!(near(height(&ill), 20.0 * width, 1e-6), "scaleWithWeight: {}", height(&ill));
    }
}

#[test]
fn pattern_side_tile_height_is_the_stroke_width() {
    let tiles = json!({"side": super::tests_pattern::tile("s", 10.0, "#3366cc")});
    let node = super::tests_scatter::line_node(straight(300.0), json!({"width": 8}));
    let abs = super::tests_pattern::place(&node, &absolute(super::tests_pattern::pattern(tiles.clone(), json!({}))));
    assert!(near(height(&abs), 8.0, 1e-6), "{}", height(&abs));
    let ill = super::tests_pattern::place(&node, &super::tests_pattern::pattern(tiles, json!({})));
    assert!(near(height(&ill), 80.0, 1e-6), "{}", height(&ill));
}

#[test]
fn scatter_copy_and_spacing_follow_the_stroke_width() {
    // A 10 × 10 square at size 1, spacing 2: absolute copies are `width`
    // across, stepped 2 × width apart.
    let art = super::tests_scatter::tile_art();
    let brush = super::tests_scatter::scatter(art, json!({"spacing": 2}));
    let node = super::tests_scatter::line_node(straight(120.0), json!({"width": 6}));
    let abs = super::tests_scatter::place(&node, &absolute(brush.clone()));
    let squares = abs.paths.iter().filter(|q| q.source_id == "sq").count();
    // Copies at 0, 12, …, 120.
    assert_eq!(squares, 11);
    assert!(near(height(&abs), 6.0, 1e-6), "{}", height(&abs));
    let ill = super::tests_scatter::place(&node, &brush);
    assert!(near(height(&ill), 60.0, 1e-6), "{}", height(&ill));
}

#[test]
fn bristle_band_follows_the_stroke_width_not_the_size() {
    let line = super::tests_bristle::polyline(&super::tests_bristle::LINE);
    let band = |brush: &Value, width: f64| height(&super::tests_bristle::place(&super::tests_bristle::node(line.clone(), width, json!({})), brush));
    let small = absolute(super::tests_bristle::bristle(json!({"size": 2})));
    let big = absolute(super::tests_bristle::bristle(json!({"size": 9})));
    // The size doesn't enter; the band grows with the width (not exactly in
    // proportion on a fixed path: the ends and corner lag depend on its
    // length in diameters) and is about as wide as the stroke.
    assert!(near(band(&small, 20.0), band(&big, 20.0), 1e-9));
    let ratio = band(&small, 40.0) / band(&small, 20.0);
    assert!((1.7..=2.3).contains(&ratio), "twice the width: {ratio}× the band");
    let w = band(&small, 20.0);
    assert!((14.0..=28.0).contains(&w), "band {w} for a 20 px stroke");
    // Illustrator's sizing: size (mm) × weight.
    let ill = band(&super::tests_bristle::bristle(json!({"size": 9})), 1.0);
    assert!(ill > 25.0, "9 mm at weight 1: {ill}");
}
