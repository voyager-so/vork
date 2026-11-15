//! Bristle brush tests: Illustrator's measured rules (strand count, strand
//! opacity, extents and side bias of the stock presets), pen channels, trim /
//! dash windows, budgets — and an ignored gallery / side-by-side comparison
//! (`cargo test -p moda-scene --lib stroke_brush::tests_bristle -- --ignored`).

use crate::value::{json, Value};

use super::bristle::*;
use super::common::Placement;
use super::raster::{polygons, Canvas};
use crate::path_geom::vec2::V2;
use crate::stroke_outline::ArcWindow;

/// Illustrator 30.8's stock "Bristle Brush Library", measured on the path
/// (40,100) → (360,100) at weight 1 after Expand Appearance: name, shape,
/// size (mm), length, density, thickness, stiffness, strands, strand opacity
/// (%), y extent, x extent.
#[allow(clippy::type_complexity)]
pub const PRESETS: [(&str, &str, f64, f64, f64, f64, f64, usize, f64, (f64, f64), (f64, f64)); 14] = [
    ("Round", "round-point", 4.5, 1.0, 0.33, 0.5, 0.5, 12, 10.91, (98.1, 113.2), (38.8, 372.2)),
    ("Flat", "flat-blunt", 3.0, 1.0, 0.2, 0.3, 0.5, 7, 17.97, (96.8, 109.2), (37.0, 366.5)),
    ("Fan", "flat-fan", 3.0, 1.5, 0.33, 0.5, 0.5, 12, 10.91, (95.9, 114.8), (36.1, 369.6)),
    ("Mop", "round-fan", 6.0, 1.5, 0.33, 0.75, 0.5, 12, 10.91, (95.5, 127.4), (34.3, 384.9)),
    ("Dome", "round-curve", 1.0, 0.75, 0.33, 0.5, 0.5, 12, 10.91, (99.4, 102.6), (39.6, 362.4)),
    ("Spotter", "round-point", 1.0, 0.5, 0.2, 0.5, 0.5, 7, 17.97, (99.8, 102.2), (40.0, 362.1)),
    ("Liner", "round-point", 1.0, 2.0, 0.2, 0.5, 0.5, 7, 17.97, (99.5, 104.9), (39.7, 364.4)),
    ("Angle", "flat-angle", 3.0, 1.0, 0.3, 0.5, 0.5, 11, 11.84, (101.1, 109.2), (37.8, 366.5)),
    ("Dagger", "flat-angle", 2.0, 2.0, 0.25, 0.5, 0.5, 9, 14.28, (100.6, 109.7), (38.2, 368.6)),
    ("Stippler", "round-blunt", 3.0, 1.0, 0.33, 0.75, 0.75, 12, 10.91, (96.9, 107.6), (38.1, 366.3)),
    ("Deerfoot", "round-angle", 4.0, 1.0, 0.33, 0.5, 0.75, 12, 10.91, (99.4, 111.1), (38.6, 370.3)),
    ("Bright", "flat-blunt", 2.0, 0.5, 0.2, 0.3, 0.5, 7, 17.97, (98.3, 104.4), (38.4, 362.8)),
    ("Cat's Tongue", "flat-point", 3.0, 1.5, 0.33, 0.5, 0.5, 12, 10.91, (98.0, 113.7), (38.0, 370.4)),
    ("Filbert", "flat-curve", 3.0, 1.5, 0.3, 0.5, 0.5, 11, 11.84, (97.8, 111.8), (37.9, 368.8)),
];

pub fn preset_brush(i: usize) -> Value {
    let (name, shape, size, length, density, thickness, stiffness, ..) = PRESETS[i];
    json!({"scaleWithWeight": true, "id": "b", "name": name, "type": "bristle", "shape": shape, "size": size, "length": length,
        "density": density, "thickness": thickness, "opacity": 0.75, "stiffness": stiffness})
}

pub fn bristle(extra: Value) -> Value {
    let mut b = json!({"scaleWithWeight": true, "id": "b", "name": "B", "type": "bristle", "shape": "round-point", "size": 3,
        "length": 1, "density": 0.33, "thickness": 0.5, "opacity": 0.75, "stiffness": 0.5});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

pub fn node(points: Value, width: f64, extra: Value) -> Value {
    let mut n = json!({"id": "n", "type": "path", "x": 0, "y": 0, "points": points, "closed": false,
        "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#231f20"}, "width": width,
            "brush": {"brushId": "b", "seed": 7}}]});
    for (k, v) in extra.as_object().unwrap() {
        n[k] = v.clone();
    }
    n
}

pub fn polyline(pts: &[(f64, f64)]) -> Value {
    Value::Array(
        pts.iter()
            .enumerate()
            .map(|(i, (x, y))| json!({"type": if i == 0 { "move" } else { "line" }, "x": x, "y": y}))
            .collect(),
    )
}

pub const LINE: [(f64, f64); 2] = [(40.0, 100.0), (360.0, 100.0)];
pub const ZIG: [(f64, f64); 5] = [(40.0, 150.0), (120.0, 40.0), (200.0, 100.0), (280.0, 160.0), (360.0, 50.0)];

pub fn s_curve() -> Value {
    json!([{"type": "move", "x": 40, "y": 100},
        {"type": "curve", "x": 200, "y": 100, "cp1": {"x": 80, "y": 20}, "cp2": {"x": 160, "y": 20}},
        {"type": "curve", "x": 360, "y": 100, "cp1": {"x": 240, "y": 180}, "cp2": {"x": 320, "y": 180}}])
}

pub fn place_w(node: &Value, brush: &Value, windows: Option<&[ArcWindow]>) -> Placement {
    let b = parse_bristle_brush(brush).expect("bristle brush");
    bristle_brush_for_node(node.as_object().unwrap(), &b, windows)
}

pub fn place(node: &Value, brush: &Value) -> Placement {
    place_w(node, brush, None)
}

/// Flattened outline points of every strand.
fn all_points(p: &Placement) -> Vec<V2> {
    p.paths
        .iter()
        .flat_map(|s| polygons(&s.points).into_iter().flatten())
        .collect()
}

fn extent(p: &Placement) -> ((f64, f64), (f64, f64)) {
    let pts = all_points(p);
    let f = |g: fn(&V2) -> f64| {
        pts.iter()
            .map(g)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)))
    };
    (f(|p| p.x), f(|p| p.y))
}

/// Mean y of the strand outlines, weighted by nothing (a cheap centroid).
fn mean_y(p: &Placement) -> f64 {
    let pts = all_points(p);
    pts.iter().map(|p| p.y).sum::<f64>() / pts.len() as f64
}

#[test]
fn strand_count_and_opacity_follow_illustrator() {
    for (i, preset) in PRESETS.iter().enumerate() {
        let b = parse_bristle_brush(&preset_brush(i)).unwrap();
        assert_eq!(b.strand_count(), preset.7, "{}", preset.0);
        assert!(
            (b.strand_opacity() * 100.0 - preset.8).abs() < 0.01,
            "{}: {}",
            preset.0,
            b.strand_opacity() * 100.0
        );
        let placed = place(&node(polyline(&LINE), 1.0, json!({})), &preset_brush(i));
        assert!(placed.too_complex.is_none());
        assert_eq!(placed.paths.len(), preset.7, "{}", preset.0);
        for s in &placed.paths {
            assert!(s.closed);
            assert_eq!(s.paint["fills"][0]["paint"]["color"], "#231f20");
            assert!((s.paint["opacity"].as_f64().unwrap() * 100.0 - preset.8).abs() < 0.01);
        }
    }
    // The stack of all n strands is exactly the paint opacity.
    for (density, opacity) in [(0.01, 0.4), (0.5, 0.75), (1.0, 1.0)] {
        let b = parse_bristle_brush(&bristle(json!({"density": density, "opacity": opacity}))).unwrap();
        let a = b.strand_opacity();
        let stack = 1.0 - (1.0 - a).powi(b.strand_count() as i32);
        assert!((stack - opacity).abs() < 1e-9, "{density} {opacity}: {stack}");
    }
}

#[test]
fn parse_clamps_and_defaults() {
    let b = parse_bristle_brush(&json!({"id": "x", "type": "bristle", "shape": "nope", "size": 40,
        "length": 0, "density": 2, "thickness": -1, "opacity": "a"}))
    .unwrap();
    assert_eq!(b.shape, BristleShape::RoundPoint);
    assert_eq!((b.size, b.length, b.density, b.thickness), (10.0, 0.25, 1.0, 0.01));
    assert_eq!((b.opacity, b.stiffness), (0.75, 0.5));
    assert_eq!(b.strand_count(), 36);
    assert!(parse_bristle_brush(&json!({"id": "x", "type": "art"})).is_none());
    assert!(parse_bristle_brush(&json!({"id": "", "type": "bristle"})).is_none());
    for s in BristleShape::ALL {
        assert_eq!(BristleShape::parse(s.id()), Some(s));
    }
}

#[test]
fn extents_land_in_illustrators_measured_ranges() {
    for (i, preset) in PRESETS.iter().enumerate().rev() {
        let placed = place(&node(polyline(&LINE), 1.0, json!({})), &preset_brush(i));
        let ((x0, x1), (y0, y1)) = extent(&placed);
        let ((ax0, ax1), (ay0, ay1)) = (preset.10, preset.9);
        let span = ay1 - ay0;
        let d = preset.2 * MM;
        eprintln!("{:14} x {x0:6.1} {x1:6.1} (AI {ax0} {ax1})  y {y0:6.1} {y1:6.1} (AI {ay0} {ay1})", preset.0);
        // Across: each edge within a quarter of Illustrator's spread (+ 1 px).
        let tol = span / 4.0 + 1.0;
        assert!((y0 - ay0).abs() < tol, "{}: y0 {y0:.1} vs {ay0}", preset.0);
        assert!((y1 - ay1).abs() < tol, "{}: y1 {y1:.1} vs {ay1}", preset.0);
        let ours = y1 - y0;
        assert!(ours > span * 0.7 && ours < span * 1.35, "{}: spread {ours:.1} vs {span:.1}", preset.0);
        // Along: starts near the path's start, overshoots its end like Illustrator's.
        assert!((x0 - ax0).abs() < 0.4 * d + 2.0, "{}: x0 {x0:.1} vs {ax0}", preset.0);
        assert!((x1 - ax1).abs() < 0.4 * d + 2.0, "{}: x1 {x1:.1} vs {ax1}", preset.0);
        assert!(x1 > 360.0, "{}: overshoots the end", preset.0);
    }
}

#[test]
fn strands_sit_right_of_the_direction_of_travel() {
    let b = preset_brush(0);
    let ltr = place(&node(polyline(&LINE), 1.0, json!({})), &b);
    assert!(mean_y(&ltr) > 103.0, "{}", mean_y(&ltr));
    let rtl = place(&node(polyline(&[(360.0, 100.0), (40.0, 100.0)]), 1.0, json!({})), &b);
    assert!(mean_y(&rtl) < 97.0, "{}", mean_y(&rtl));
    let down = place(&node(polyline(&[(200.0, 20.0), (200.0, 180.0)]), 1.0, json!({})), &b);
    let ((x0, x1), _) = extent(&down);
    assert!(x0 < 190.0 && x1 < 203.0, "{x0} {x1}");
}

#[test]
fn stroke_weight_scales_everything() {
    let b = preset_brush(0);
    let one = place(&node(polyline(&LINE), 1.0, json!({})), &b);
    let two = place(&node(polyline(&LINE), 2.0, json!({})), &b);
    let (_, (a0, a1)) = extent(&one);
    let (_, (b0, b1)) = extent(&two);
    let r = ((b1 - 100.0) / (a1 - 100.0), (100.0 - b0) / (100.0 - a0));
    assert!((r.0 - 2.0).abs() < 0.05, "{r:?}");
    assert!((r.1 - 2.0).abs() < 0.2, "{r:?}");
    let zero = place(&node(polyline(&LINE), 0.0, json!({})), &b);
    assert!(zero.paths.is_empty());
}

#[test]
fn deterministic_per_seed() {
    let b = preset_brush(3);
    let n = node(json!(s_curve()), 1.0, json!({}));
    let ids = |p: &Placement| p.paths.iter().map(|s| format!("{:?}", s.points)).collect::<Vec<_>>();
    assert_eq!(ids(&place(&n, &b)), ids(&place(&n, &b)));
    let mut other = n.clone();
    other["strokes"][0]["brush"]["seed"] = json!(8);
    assert_ne!(ids(&place(&n, &b)), ids(&place(&other, &b)));
}

#[test]
fn strands_drag_round_corners() {
    // A right-angle corner: a lagging strand cuts inside it instead of
    // following the sharp turn.
    let b = bristle(json!({"stiffness": 0.05, "length": 2}));
    let corner = polyline(&[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0)]);
    let placed = place(&node(corner, 1.0, json!({})), &b);
    // How deep a strand cuts into the corner: the largest distance from
    // both legs at once (an offset curve without lag reaches ≈ the band).
    let depth = |pl: &Placement| {
        all_points(pl)
            .iter()
            .filter(|p| p.x <= 200.0 && p.y >= 0.0)
            .map(|p| (200.0 - p.x).min(p.y))
            .fold(0.0, f64::max)
    };
    let stiff = place(&node(polyline(&[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0)]), 1.0, json!({})),
        &bristle(json!({"stiffness": 1.0, "length": 2})));
    assert!(depth(&placed) > depth(&stiff) + 1.0, "{} {}", depth(&placed), depth(&stiff));
}

fn ink(samples: usize, f: impl Fn(f64) -> (f64, f64, f64, f64)) -> Value {
    let t: Vec<f64> = (0..samples).map(|i| i as f64 / (samples - 1) as f64).collect();
    let vals: Vec<(f64, f64, f64, f64)> = t.iter().map(|t| f(*t)).collect();
    json!({"t": t, "pressure": vals.iter().map(|v| v.0).collect::<Vec<_>>(),
        "tiltX": vals.iter().map(|v| v.1).collect::<Vec<_>>(),
        "tiltY": vals.iter().map(|v| v.2).collect::<Vec<_>>(),
        "twist": vals.iter().map(|v| v.3).collect::<Vec<_>>()})
}

#[test]
fn pressure_narrows_and_lightens() {
    let b = preset_brush(0);
    let light = place(&node(polyline(&LINE), 1.0, json!({"inkChannels": ink(5, |_| (0.15, 0.0, 0.0, 0.0))})), &b);
    let firm = place(&node(polyline(&LINE), 1.0, json!({"inkChannels": ink(5, |_| (1.0, 0.0, 0.0, 0.0))})), &b);
    let span = |p: &Placement| {
        let (_, (a, b)) = extent(p);
        b - a
    };
    assert!(span(&light) < span(&firm) * 0.7, "{} {}", span(&light), span(&firm));
    // Lifted strands: the light stroke's mid-path area is thinner overall.
    let area = |p: &Placement| {
        p.paths
            .iter()
            .map(|s| {
                let poly = &polygons(&s.points)[0];
                let n = poly.len();
                (0..n).map(|i| poly[i].cross(poly[(i + 1) % n])).sum::<f64>().abs() / 2.0
            })
            .sum::<f64>()
    };
    assert!(area(&light) < area(&firm) * 0.5, "{} {}", area(&light), area(&firm));
}

#[test]
fn tilt_leans_the_band_to_the_pens_side() {
    let b = preset_brush(0);
    // Upright-ish pen leaning towards −y (screen up), i.e. LEFT of travel on
    // a left-to-right path: tiltY < 0 leans towards −y.
    let left = place(&node(polyline(&LINE), 1.0, json!({"inkChannels": ink(3, |_| (1.0, 0.0, -50.0, 0.0))})), &b);
    assert!(mean_y(&left) < 99.0, "{}", mean_y(&left));
    let right = place(&node(polyline(&LINE), 1.0, json!({"inkChannels": ink(3, |_| (1.0, 0.0, 50.0, 0.0))})), &b);
    assert!(mean_y(&right) > 101.0, "{}", mean_y(&right));
}

#[test]
fn rotation_turns_a_flat_tip() {
    let flat = preset_brush(1);
    let span = |twist: f64, b: &Value| {
        let p = place(&node(polyline(&LINE), 1.0, json!({"inkChannels": ink(3, |_| (1.0, 0.0, 0.0, twist))})), b);
        let (_, (a, c)) = extent(&p);
        c - a
    };
    assert!(span(90.0, &flat) < span(0.0, &flat) * 0.6);
    // A round tip ignores rotation.
    let round = preset_brush(0);
    assert!((span(90.0, &round) - span(0.0, &round)).abs() < 1e-6);
}

#[test]
fn trim_and_dash_windows_clip_the_strands() {
    let b = preset_brush(0);
    let n = node(polyline(&LINE), 1.0, json!({}));
    let w = |a: f64, e: f64| ArcWindow { subpath: 0, start: a, end: e, taper_extent: None };
    // First half: nothing past x = 200 (+ a lagging strand's width).
    let half = place_w(&n, &b, Some(&[w(0.0, 160.0)]));
    let ((x0, x1), _) = extent(&half);
    assert!(x1 < 205.0 && x0 < 45.0, "{x0} {x1}");
    assert_eq!(half.paths.len(), 12);
    // Second half keeps the overshoot past the end.
    let tail = place_w(&n, &b, Some(&[w(160.0, 320.0)]));
    let ((t0, t1), _) = extent(&tail);
    assert!(t0 > 190.0 && t1 > 362.0, "{t0} {t1}");
    // Dashes: a piece per strand per dash.
    let dashes = place_w(&n, &b, Some(&[w(0.0, 60.0), w(120.0, 180.0), w(240.0, 300.0)]));
    assert_eq!(dashes.paths.len(), 36);
    assert!(dashes.paths.iter().all(|p| p.id.matches('.').count() == 2));
    // No window on this subpath: nothing.
    assert!(place_w(&n, &b, Some(&[])).paths.is_empty());
    // A full window is the whole placement.
    let full = place_w(&n, &b, Some(&[w(0.0, 320.0)]));
    assert_eq!(extent(&full), extent(&place(&n, &b)));
}

#[test]
fn closed_and_compound_paths() {
    let b = preset_brush(1);
    let ring = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
        {"type": "line", "x": 100, "y": 100}, {"type": "line", "x": 0, "y": 100}, {"type": "close"},
        {"type": "move", "x": 200, "y": 0}, {"type": "line", "x": 300, "y": 0}]);
    let placed = place(&node(ring, 1.0, json!({})), &b);
    assert_eq!(placed.paths.len(), 14);
    assert!(placed.paths.iter().any(|p| p.at.subpath == 1));
    let ((x0, x1), (y0, y1)) = extent(&placed);
    assert!(x0 > -15.0 && x1 < 330.0 && y0 > -15.0 && y1 < 115.0, "{x0} {x1} {y0} {y1}");
}

#[test]
fn huge_paths_stay_in_budget() {
    // 4 000 zig-zag points over 400 000 px at the densest, biggest brush.
    let pts: Vec<(f64, f64)> = (0..4000).map(|i| (i as f64 * 100.0, if i % 2 == 0 { 0.0 } else { 80.0 })).collect();
    let b = bristle(json!({"density": 1, "size": 10, "length": 3}));
    let t = std::time::Instant::now();
    let placed = place(&node(polyline(&pts), 1.0, json!({})), &b);
    assert!(t.elapsed().as_secs_f64() < 5.0);
    let segs: usize = placed.paths.iter().map(|p| p.points.len()).sum();
    assert!(placed.too_complex.is_some() || segs <= super::common::MAX_OUTPUT_SEGMENTS + placed.paths.len());
    // Degenerate input: no panic, nothing placed.
    for pts in [json!([]), json!([{"type": "move", "x": 1, "y": 1}]), polyline(&[(5.0, 5.0), (5.0, 5.0)])] {
        assert!(place(&node(pts, 1.0, json!({})), &b).paths.is_empty());
    }
}

/// The agent-art benchmark's probe (landscape, sumi-e notes): a short vertical
/// stroke next to a big brush (flat-blunt 5 mm at weight 5, a 70 px ferrule on
/// a 140 px path) with pressure 1 → 0.8 → 1.
fn short_pressure_stroke(seed: u64, x: f64) -> (Value, Value) {
    let pts: Vec<(f64, f64)> = (0..=8).map(|k| (x, 30.0 + 140.0 * k as f64 / 8.0)).collect();
    let mut n = node(
        polyline(&pts),
        5.0,
        json!({"inkChannels": ink(9, |t| (1.0 - 0.2 * (std::f64::consts::PI * t).sin(), 0.0, 0.0, 0.0))}),
    );
    n["strokes"][0]["brush"]["seed"] = json!(seed);
    let b = bristle(json!({"shape": "flat-blunt", "size": 5, "density": 0.22, "thickness": 0.7, "opacity": 0.85,
        "stiffness": 0.6}));
    (n, b)
}

/// The farthest any strand's outline strays from the same strand sampled
/// `fine` times (symmetric Hausdorff over the flattened outlines).
fn drift_from_fine(n: &Value, b: &Value, fine: usize) -> f64 {
    let outlines = |p: &Placement| -> Vec<Vec<V2>> { p.paths.iter().map(|s| polygons(&s.points).concat()).collect() };
    let ours = outlines(&place(n, b));
    MIN_SAMPLES_OVERRIDE.with(|c| c.set(Some(fine)));
    let reference = outlines(&place(n, b));
    MIN_SAMPLES_OVERRIDE.with(|c| c.set(None));
    assert_eq!(ours.len(), reference.len());
    let one_way = |a: &[V2], b: &[V2]| {
        a.iter()
            .map(|p| b.iter().map(|q| p.dist(*q)).fold(f64::INFINITY, f64::min))
            .fold(0.0, f64::max)
    };
    ours.iter()
        .zip(&reference)
        .map(|(a, b)| one_way(a, b).max(one_way(b, a)))
        .fold(0.0, f64::max)
}

#[test]
fn a_short_stroke_under_a_big_brush_keeps_smooth_strands() {
    // However short the stroke is next to the brush, each strand is sampled
    // finely enough that it is (nearly) the strand a much finer sampling
    // draws. Before, a third of the diameter per sample (21 px here) left
    // 7-10 samples per strand: its ends came out as polygonal chevrons and
    // pressure changes as blocky steps.
    let dia = 5.0 * MM * 5.0;
    for seed in [1, 7, 23, 99] {
        let (n, b) = short_pressure_stroke(seed, 100.0);
        let drift = drift_from_fine(&n, &b, 400);
        assert!(drift < 0.1 * dia, "seed {seed}: strands {drift:.1} px off the finely sampled ones (D = {dia:.0})");
    }
}

/// The probe side by side over a few seeds (`BRISTLE_GALLERY_DIR`, or the
/// gallery folder): `bristle-short-pressure.png`.
#[test]
#[ignore]
fn gallery_short_pressure_strokes() {
    let mut c = Canvas::new(800, 220, 0.0, 0.0, 1.0);
    for (i, seed) in [1u64, 7, 23, 99, 5, 11, 17, 31].into_iter().enumerate() {
        let (n, b) = short_pressure_stroke(seed, 50.0 + i as f64 * 95.0);
        c.draw(&place(&n, &b).paths);
    }
    c.write_png(&gallery_dir().join("bristle-short-pressure.png"));
}

#[test]
fn json_envelope_dispatches_bristle() {
    let req = json!({"node": node(polyline(&LINE), 1.0, json!({})), "brush": preset_brush(0)});
    let out: Value = crate::value::from_str(&super::stroke_brush_json(&req.to_string())).unwrap();
    assert_eq!(out["status"], "ok");
    assert_eq!(out["nodes"].as_array().unwrap().len(), 12);
    assert_eq!(out["nodes"][0]["sourceId"], "bristle");
    assert_eq!(out["nodes"][0]["closed"], true);
    let bad = json!({"node": node(polyline(&LINE), 1.0, json!({})), "brush": {"id": "b", "type": "bristlez"}});
    let out: Value = crate::value::from_str(&super::stroke_brush_json(&bad.to_string())).unwrap();
    assert_eq!(out["status"], "error");
}

// ---------------------------------------------------------------------------
// Gallery and the side-by-side comparison with Illustrator (ignored).
// ---------------------------------------------------------------------------

fn gallery_dir() -> std::path::PathBuf {
    let dir = std::env::var("BRISTLE_GALLERY_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/roadmap/brushes/gallery")
    });
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Illustrator's expanded strands (`<dir>/<prefix><name>.txt`: first line
/// `paths:N`, then per strand `…;opacity%;… | x,y,inX,inY,outX,outY …`).
fn ai_strands(dir: &str, file: &str) -> Option<Vec<(f64, Vec<V2>)>> {
    let text = std::fs::read_to_string(std::path::Path::new(dir).join(file)).ok()?;
    let mut out = Vec::new();
    for line in text.split(['\r', '\n']).filter(|l| !l.is_empty()).skip(1) {
        let (meta, pts) = line.split_once(" | ")?;
        let op: f64 = meta.split(';').nth(2)?.parse().ok()?;
        let anchors: Vec<[f64; 6]> = pts
            .split(' ')
            .filter_map(|a| {
                let v: Vec<f64> = a.split(',').filter_map(|x| x.parse().ok()).collect();
                (v.len() == 6).then(|| [v[0], v[1], v[2], v[3], v[4], v[5]])
            })
            .collect();
        let m = anchors.len();
        let mut poly = Vec::new();
        for i in 0..m {
            let (a, b) = (anchors[i], anchors[(i + 1) % m]);
            let c = crate::path_geom::vec2::Cubic([
                crate::path_geom::vec2::v2(a[0], a[1]),
                crate::path_geom::vec2::v2(a[4], a[5]),
                crate::path_geom::vec2::v2(b[2], b[3]),
                crate::path_geom::vec2::v2(b[0], b[1]),
            ]);
            for k in 0..12 {
                poly.push(c.eval(k as f64 / 12.0));
            }
        }
        out.push((op / 100.0, poly));
    }
    Some(out)
}

const INK: [f32; 3] = [35.0, 31.0, 32.0];

fn draw_ours(c: &mut Canvas, pts: &Value, brush: &Value, dx: f64, dy: f64) {
    let mut moved = pts.clone();
    for p in moved.as_array_mut().unwrap() {
        for k in ["", "cp1", "cp2"] {
            let o = if k.is_empty() { Some(&mut *p) } else { p.get_mut(k) };
            if let Some(o) = o {
                if let (Some(x), Some(y)) = (o["x"].as_f64(), o["y"].as_f64()) {
                    o["x"] = json!(x + dx);
                    o["y"] = json!(y + dy);
                }
            }
        }
    }
    c.draw(&place(&node(moved, 1.0, json!({})), brush).paths);
}

fn draw_ai(c: &mut Canvas, strands: &[(f64, Vec<V2>)], dx: f64, dy: f64) {
    for (op, poly) in strands {
        let poly: Vec<V2> = poly.iter().map(|p| crate::path_geom::vec2::v2(p.x + dx, p.y + dy)).collect();
        let cov = c.coverage(&[poly], false);
        c.paint(&cov, INK, *op as f32);
    }
}

/// Every preset on a line, an S-curve and the zig-zag
/// (`docs/roadmap/brushes/gallery/bristle-presets.png`, or `BRISTLE_GALLERY_DIR`).
#[test]
#[ignore]
fn gallery_presets() {
    let (cw, ch) = (400.0, 200.0);
    let mut c = Canvas::new(3 * cw as usize, PRESETS.len() * 110, 0.0, 0.0, 1.0);
    for i in 0..PRESETS.len() {
        let b = preset_brush(i);
        let y = i as f64 * 110.0 - 45.0;
        draw_ours(&mut c, &polyline(&LINE), &b, 0.0, y);
        draw_ours(&mut c, &s_curve(), &b, cw, y);
        // The zig-zag squeezed into the row.
        let zig: Vec<(f64, f64)> = ZIG.iter().map(|(x, yy)| (*x, 100.0 + (yy - 100.0) * 0.45)).collect();
        draw_ours(&mut c, &polyline(&zig), &b, 2.0 * cw, y);
    }
    let _ = ch;
    c.write_png(&gallery_dir().join("bristle-presets.png"));
}

/// Side by side with Illustrator (needs its expanded strands:
/// `BRISTLE_AI_DIR=<dir with line-<Preset>.txt, zig-<Preset>.txt>`): each row
/// ours (left) and Illustrator's (right) at 2×.
#[test]
#[ignore]
fn compare_with_illustrator() {
    let Ok(dir) = std::env::var("BRISTLE_AI_DIR") else {
        eprintln!("BRISTLE_AI_DIR not set: skipped");
        return;
    };
    let scale = 2.0;
    let row = 70.0;
    let zig_rows = ["Round", "Flat", "Fan", "Mop"];
    let h = (PRESETS.len() as f64 * row + zig_rows.len() as f64 * 180.0) * scale;
    let mut c = Canvas::new((800.0 * scale) as usize, h as usize, 0.0, 0.0, 1.0 / scale);
    let mut y = 0.0;
    for (i, p) in PRESETS.iter().enumerate() {
        let dy = y - 100.0 + row * 0.4;
        draw_ours(&mut c, &polyline(&LINE), &preset_brush(i), 0.0, dy);
        if let Some(s) = ai_strands(&dir, &format!("line-{}.txt", p.0)) {
            draw_ai(&mut c, &s, 400.0, dy);
        }
        y += row;
    }
    for name in zig_rows {
        let i = PRESETS.iter().position(|p| p.0 == name).unwrap();
        let dy = y - 30.0;
        draw_ours(&mut c, &polyline(&ZIG), &preset_brush(i), 0.0, dy);
        if let Some(s) = ai_strands(&dir, &format!("zig-{name}.txt")) {
            draw_ai(&mut c, &s, 400.0, dy);
        }
        y += 180.0;
    }
    c.write_png(&gallery_dir().join("bristle.png"));
}

/// A placed shape built as a node (`PlacedPath::to_node`) is the node its JSON text parses to: every preset, on a
/// straight path, a corner and a curve, with pressure.
#[test]
fn placed_nodes_match_their_json() {
    let pressure = json!({"t": [0, 0.5, 1], "pressure": [0.3, 1, 0.6]});
    for i in 0..PRESETS.len() {
        for (pts, extra) in [
            (polyline(&LINE), json!({})),
            (polyline(&ZIG), json!({})),
            (s_curve(), json!({"inkChannels": pressure.clone()})),
        ] {
            let p = place(&node(pts, 1.3, extra), &preset_brush(i));
            assert!(!p.paths.is_empty());
            for (k, path) in p.paths.iter().enumerate() {
                let id = format!("n::brush::{k}");
                let mut want = path.to_node_json();
                want["id"] = json!(id.clone());
                let got = path.to_node(id);
                assert_eq!(Value::Object(got.to_json_map()), want, "preset {i} shape {k}");
            }
        }
    }
}
