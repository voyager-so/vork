//! Calligraphic nib, ink-channel `Dyn` variation and the dynamic wobble.

use super::nib::{ink_at, Dyn, InkAt, Nib, Vary};
use super::tests::{coverage, node, shape};
use super::*;
use crate::path_geom::fit::tests::Rng;
use crate::path_geom::vec2::v2;
use crate::path_geom::vec2::V2;
use crate::stroke_model::parse_ink_channels;
use crate::value::json;

fn run(v: Value, brush: Option<Value>) -> Outline {
    stroke_outline(
        &node(v),
        brush.as_ref(),
        &OutlineOptions {
            tolerance: 0.02,
            samples: true,
            ..Default::default()
        },
    )
    .unwrap()
}

fn samples(o: &Outline) -> &WidthSamples {
    match o {
        Outline::Shape {
            samples: Some(s), ..
        } => s,
        _ => panic!("expected samples"),
    }
}

fn calligraphic(angle: Value, roundness: Value, size: Value) -> Value {
    json!({"id": "nib", "name": "Nib", "type": "calligraphic", "angle": angle, "roundness": roundness, "size": size})
}

#[test]
fn nib_width_depends_on_direction() {
    // Flat nib along +x: thin going right, full going down.
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}, {"type": "line", "x": 100, "y": 100}]);
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}]});
    let o = run(v, Some(calligraphic(json!(0), json!(0.2), json!(10))));
    let ws = samples(&o);
    let (l, r) = ws.width_at(0.25);
    assert!(
        (l - 1.0).abs() < 1e-9 && (r - 1.0).abs() < 1e-9,
        "horizontal: {l} {r}"
    );
    let (l, _) = ws.width_at(0.75);
    assert!((l - 5.0).abs() < 1e-9, "vertical: {l}");
    // The stroke weight scales the nib (Illustrator: size is at weight 1).
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 2, "brush": {"brushId": "nib"}}]});
    let o = run(v, Some(calligraphic(json!(0), json!(0.2), json!(10))));
    assert!((samples(&o).width_at(0.75).0 - 10.0).abs() < 1e-9);
    // 45° nib: a diagonal stroke along the nib is thin, across it is thick.
    let nib = Nib::from_brush(&calligraphic(json!(45), json!(0.0), json!(10))).unwrap();
    let along = V2 { x: 1.0, y: -1.0 }.unit().unwrap(); // screen up-right
    let n_along = along.left_normal();
    let n_across = V2 { x: 1.0, y: 1.0 }.unit().unwrap().left_normal();
    let (a, b, u) = nib.ellipse(&InkAt::default(), [0.0; 3], None);
    // Extent along a normal = the support point's normal component.
    assert!(super::nib::support(a, b, u, n_along).dot(n_along) < 1e-2);
    assert!((super::nib::support(a, b, u, n_across).dot(n_across) - 5.0).abs() < 1e-9);
}

#[test]
fn brush_must_match_the_stroke() {
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    let nib = calligraphic(json!(0), json!(0.2), json!(10));
    // No strokeBrush → the plain stroke; wrong id → the plain stroke.
    let o = run(json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4}]}), Some(nib.clone()));
    assert_eq!(samples(&o).width_at(0.5), (2.0, 2.0));
    let o = run(
        json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "brush": {"brushId": "other"}}]}),
        Some(nib),
    );
    assert_eq!(samples(&o).width_at(0.5), (2.0, 2.0));
    // Non-calligraphic definitions are not nibs.
    let art = json!({"id": "a", "type": "art"});
    let o = run(
        json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "brush": {"brushId": "a"}}]}),
        Some(art),
    );
    assert_eq!(samples(&o).width_at(0.5), (2.0, 2.0));
}

#[test]
fn pressure_drives_nib_size() {
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 200, "y": 0}]);
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}],
                   "inkChannels": {"t": [0, 0.5, 1], "pressure": [0, 1, 0.5]}});
    let brush = calligraphic(
        json!(90),
        json!(1),
        json!({"value": 6, "vary": "pressure", "min": 2, "max": 10}),
    );
    let o = run(v, Some(brush.clone()));
    let ws = samples(&o);
    for (t, half) in [(0.0, 1.0), (0.25, 3.0), (0.5, 5.0), (1.0, 3.0)] {
        let (l, r) = ws.width_at(t);
        assert!(
            (l - half).abs() < 0.02 && (r - half).abs() < 0.02,
            "t {t}: {l} {r} vs {half}"
        );
    }
    // No channel recorded → the fixed value.
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}]});
    let o = run(v, Some(brush));
    assert!((samples(&o).width_at(0.5).0 - 3.0).abs() < 1e-9);
}

#[test]
fn random_variation_is_seeded_per_stroke() {
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    let brush = calligraphic(
        json!(90),
        json!(1),
        json!({"value": 6, "vary": "random", "min": 2, "max": 10}),
    );
    let width = |seed: u32| {
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib", "seed": seed}}]});
        samples(&run(v, Some(brush.clone()))).width_at(0.5).0
    };
    assert_eq!(width(7), width(7));
    let ws: Vec<f64> = (0..8).map(width).collect();
    assert!(ws.iter().all(|w| (1.0..=5.0).contains(w)), "{ws:?}");
    assert!(
        ws.windows(2).any(|w| (w[0] - w[1]).abs() > 0.1),
        "seeds differ: {ws:?}"
    );
    // Constant along the stroke (random is per stroke, not per point).
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib", "seed": 3}}]});
    let o = run(v, Some(brush));
    let s = samples(&o);
    assert!(s.left.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12));
}

#[test]
fn ink_tilt_bearing_rotation() {
    let ink = parse_ink_channels(
        &json!({"t": [0, 1], "tiltX": [0, 45], "tiltY": [0, 0], "twist": [0, 180]}),
    )
    .unwrap();
    let a = ink_at(&ink, 1.0);
    assert!((a.tilt.unwrap() - 45.0).abs() < 1e-9);
    assert!(a.bearing.unwrap().abs() < 1e-9, "leaning right is 0°");
    assert!((a.rotation.unwrap() - 180.0).abs() < 1e-9);
    // Pointer Events are clockwise on the y-down screen; the model is CCW:
    // leaning toward the user (+tiltY, screen down) is a bearing of 270°.
    let ink = parse_ink_channels(&json!({"t": [0], "tiltX": [0], "tiltY": [30]})).unwrap();
    assert!((ink_at(&ink, 0.5).bearing.unwrap() - 270.0).abs() < 1e-9);
    // One tilt axis only: the other reads as 0.
    let ink = parse_ink_channels(&json!({"t": [0], "tiltY": [30]})).unwrap();
    let at = ink_at(&ink, 0.5);
    assert!((at.tilt.unwrap() - 30.0).abs() < 1e-9 && (at.bearing.unwrap() - 270.0).abs() < 1e-9);
    // Twist is clockwise too (90° CW = 270° CCW), and wraps the short way:
    // 350° → 10° passes through 0°.
    let ink = parse_ink_channels(&json!({"t": [0, 1], "twist": [90, 90]})).unwrap();
    assert!((ink_at(&ink, 0.5).rotation.unwrap() - 270.0).abs() < 1e-9);
    let ink = parse_ink_channels(&json!({"t": [0, 1], "twist": [350, 10]})).unwrap();
    let r = ink_at(&ink, 0.5).rotation.unwrap();
    assert!(r < 1e-9 || (360.0 - r) < 1e-9, "shortest arc: {r}");
    let r = ink_at(&ink, 0.25).rotation.unwrap();
    assert!((r - 5.0).abs() < 1e-9, "CCW of 355° CW: {r}");
    let d = Dyn {
        value: 1.0,
        vary: Some(Vary::Tilt),
        min: 0.0,
        max: 90.0,
    };
    assert!(
        (d.eval(
            &InkAt {
                tilt: Some(45.0),
                ..Default::default()
            },
            0.0
        ) - 45.0)
            .abs()
            < 1e-9
    );
    assert_eq!(d.eval(&InkAt::default(), 0.0), 1.0);
    // Parsing clamps and defaults.
    let r = Dyn::parse(Some(&json!({"value": 3, "min": -1})), 1.0, 0.0, 1.0);
    assert_eq!((r.value, r.min, r.max), (1.0, 0.0, 1.0));
    assert_eq!(Dyn::parse(Some(&json!(0.4)), 1.0, 0.0, 1.0).value, 0.4);
}

#[test]
fn dynamic_wobble_is_seeded_and_bounded() {
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 300, "y": 0}]);
    let bounds_of = |dynamic: Value| {
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "dynamic": dynamic}]});
        match run(v, None) {
            Outline::Shape { bounds, points, .. } => (bounds, points.len()),
            _ => panic!(),
        }
    };
    let (plain, _) = bounds_of(json!({"wiggle": 0}));
    assert_eq!((plain.min_y, plain.max_y), (-2.0, 2.0));
    let (a, _) = bounds_of(json!({"wiggle": 6, "frequency": 3, "seed": 1}));
    let (b, _) = bounds_of(json!({"wiggle": 6, "frequency": 3, "seed": 1}));
    let (c, _) = bounds_of(json!({"wiggle": 6, "frequency": 3, "seed": 2}));
    assert_eq!(a, b, "deterministic");
    assert_ne!(a, c, "seed matters");
    assert!(a.min_y >= -8.0 - 1e-6 && a.max_y <= 8.0 + 1e-6, "{a:?}");
    assert!(a.max_y - a.min_y > 6.0, "it actually wobbles: {a:?}");
    // Smoothen removes the finer, irregular octave: fewer crests.
    let rough: f64 = (0..4).map(|seed| wobbles_per_100px(3.0, 0.0, seed)).sum();
    let smooth: f64 = (0..4).map(|seed| wobbles_per_100px(3.0, 1.0, seed)).sum();
    assert!(smooth < rough, "smooth {smooth} vs rough {rough}");
}

#[test]
fn dynamic_closed_path_is_seamless() {
    let circle = json!([{"type": "move", "x": 50, "y": 0}, {"type": "arc", "x": -50, "y": 0, "rx": 50, "ry": 50, "largeArc": false, "sweep": true},
                        {"type": "arc", "x": 50, "y": 0, "rx": 50, "ry": 50, "largeArc": false, "sweep": true}]);
    let v = json!({"points": circle, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 6, "dynamic": {"wiggle": 5, "frequency": 4, "seed": 9}}]});
    let o = run(v.clone(), None);
    let s = samples(&o);
    let n = s.x.len();
    // Last sample and first sample are neighbours on the ring: no jump.
    let gap = ((s.x[n - 1] - s.x[0]).powi(2) + (s.y[n - 1] - s.y[0]).powi(2)).sqrt();
    let typical = ((s.x[1] - s.x[0]).powi(2) + (s.y[1] - s.y[0]).powi(2)).sqrt();
    assert!(
        gap < 3.0 * typical + 1e-6,
        "seam gap {gap} vs step {typical}"
    );
    // Two rings (plus fan pieces where a tight crest clamps the inner side).
    assert!(shape(&o).1 >= 2);
    let c = coverage(
        &json!({"points": circle, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 6, "join": "round", "dynamic": {"wiggle": 5, "frequency": 4, "seed": 9}}]}),
        0.5,
    );
    assert_eq!(c.spikes, 0);
    assert!(
        c.holes * 500 < c.inside.max(1),
        "{} of {}",
        c.holes,
        c.inside
    );
}

#[test]
fn nib_and_wobble_cover_the_exact_sweep() {
    // Random cubic paths, random fixed nibs, with and without wobble: the
    // outline equals the Minkowski sum of the nib and the (displaced)
    // centerline it was built from, within 1 % of the area.
    let mut rng = Rng::new(0xCA11);
    let mut worst = 0.0f64;
    for case in 0..12 {
        let mut pts = vec![json!({"type": "move", "x": 20, "y": 20})];
        for _ in 0..3 {
            pts.push(json!({"type": "curve", "x": rng.f(20.0, 200.0), "y": rng.f(20.0, 200.0),
                "cp1": {"x": rng.f(20.0, 200.0), "y": rng.f(20.0, 200.0)}, "cp2": {"x": rng.f(20.0, 200.0), "y": rng.f(20.0, 200.0)}}));
        }
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1.5, "dynamic": {"wiggle": (case % 3) as f64 * 2.0, "frequency": 2, "seed": case}, "brush": {"brushId": "nib", "seed": case}}]});
        let brush = nib_def(rng.f(0.0, 180.0), rng.f(0.1, 1.0), 12.0);
        let o = run(v.clone(), Some(brush.clone()));
        let ws = samples(&o);
        let centre: Vec<V2> = (0..ws.x.len()).map(|i| v2(ws.x[i], ws.y[i])).collect();
        let e = nib_sweep_error(v, brush, &dense_polyline(&centre, false));
        worst = worst.max(e);
        assert!(e < 0.01, "case {case}: {:.3} %", e * 100.0);
    }
    eprintln!("worst symmetric difference {:.3} %", worst * 100.0);
}

#[test]
fn brush_and_dynamic_fuzz_no_panic() {
    let mut rng = Rng::new(0xB0B);
    let nums: [f64; 8] = [0.0, -1.0, 1e-9, 0.5, 3.0, 1e6, 1e12, 360.0];
    for case in 0..150 {
        let n = 1 + (rng.next_u64() % 5) as usize;
        let mut pts =
            vec![json!({"type": "move", "x": rng.f(-10.0, 10.0), "y": rng.f(-10.0, 10.0)})];
        for _ in 0..n {
            pts.push(json!({"type": "line", "x": rng.f(-10.0, 10.0) * nums[case % nums.len()].abs().min(1e6), "y": rng.f(-10.0, 10.0)}));
        }
        let pick = |k: usize| nums[(case / (k + 1)) % nums.len()];
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": (pick(0).abs()), "dynamic": {"wiggle": (pick(1)), "frequency": (pick(2)), "smoothen": (pick(3)), "seed": (pick(4))}, "brush": {"brushId": "nib", "seed": case}}], "closed": case % 3 == 0,
            "inkChannels": {"t": [0, 0.5, 1], "pressure": [(pick(5)), 0.5, 1], "tiltX": [0, (pick(6)), 90], "tiltY": [0, 0, (pick(7))], "twist": [0, 1, (pick(2))]}});
        let vary = ["random", "pressure", "tilt", "bearing", "rotation", "bogus"][case % 6];
        let brush = json!({"id": "nib", "type": "calligraphic",
            "angle": {"value": (pick(1)), "vary": vary, "min": (pick(2)), "max": (pick(3))},
            "roundness": {"value": (pick(4)), "vary": vary},
            "size": {"value": (pick(5)), "vary": vary, "min": (pick(6)), "max": (pick(7))}});
        match stroke_outline(
            &node(v.clone()),
            Some(&brush),
            &OutlineOptions {
                samples: case % 2 == 0,
                ..Default::default()
            },
        ) {
            Err(e) => assert_ne!(e.code, "solver_failed", "case {case}: {v}"),
            Ok(Outline::Shape { points, .. }) => {
                assert!(
                    !points
                        .iter()
                        .any(|p| p.to_json().to_string().contains("null")),
                    "case {case}: {v}"
                )
            }
            Ok(Outline::Empty) => {}
        }
    }
}

/// The exact reference for a constant nib: the Minkowski sum of the path and
/// the nib ellipse, rasterised as the union of the nib stamped at dense
/// centerline positions (≤ 0.1 px apart).
fn minkowski_mask(g: &super::raster::Grid, centre: &[V2], a: f64, b: f64, u: V2) -> Vec<bool> {
    let mut m = vec![false; g.w * g.h];
    let v = super::nib::minor_axis(u);
    let r = a.max(b);
    for c in centre {
        let x0 = (((c.x - r - g.x0) / g.px).floor().max(0.0)) as usize;
        let x1 = (((c.x + r - g.x0) / g.px).ceil().max(0.0) as usize).min(g.w);
        let y0 = (((c.y - r - g.y0) / g.px).floor().max(0.0)) as usize;
        let y1 = (((c.y + r - g.y0) / g.px).ceil().max(0.0) as usize).min(g.h);
        for yy in y0..y1 {
            for xx in x0..x1 {
                let p = V2 {
                    x: g.x0 + (xx as f64 + 0.5) * g.px,
                    y: g.y0 + (yy as f64 + 0.5) * g.px,
                } - *c;
                let (du, dv) = (p.dot(u) / a, p.dot(v) / b);
                if du * du + dv * dv <= 1.0 {
                    m[yy * g.w + xx] = true;
                }
            }
        }
    }
    m
}

fn dense_polyline(pts: &[V2], closed: bool) -> Vec<V2> {
    let mut out = Vec::new();
    let n = pts.len();
    let segs = if closed { n } else { n - 1 };
    for i in 0..segs {
        let (p, q) = (pts[i], pts[(i + 1) % n]);
        let k = ((p.dist(q) / 0.1).ceil() as usize).max(1);
        for j in 0..=k {
            out.push(p.lerp(q, j as f64 / k as f64));
        }
    }
    out
}

/// Symmetric difference between our outline (filled nonzero) and the exact
/// nib sweep, as a fraction of the sweep's area.
fn nib_sweep_error(v: Value, brush: Value, centre: &[V2]) -> f64 {
    let nib = Nib::from_brush(&brush).unwrap();
    let w = v["strokes"][0]["width"].as_f64().unwrap();
    let (a, b, u) = nib.ellipse(&InkAt::default(), [0.0; 3], None);
    let (a, b) = (a * w, b * w);
    let o = stroke_outline(
        &node(v),
        Some(&brush),
        &OutlineOptions {
            tolerance: 0.05,
            ..Default::default()
        },
    )
    .unwrap();
    let Outline::Shape { points, bounds, .. } = &o else {
        panic!()
    };
    let g = super::raster::Grid::around(
        bounds.min_x,
        bounds.min_y,
        bounds.max_x,
        bounds.max_y,
        0.25,
        3.0,
    );
    let ours = g.fill_nonzero(&super::raster::polygons(points));
    let truth = minkowski_mask(&g, centre, a, b.max(a * 1e-3), u);
    let diff = ours.iter().zip(&truth).filter(|(x, y)| x != y).count();
    let area = truth.iter().filter(|x| **x).count().max(1);
    diff as f64 / area as f64
}

fn nib_def(angle: f64, roundness: f64, size: f64) -> Value {
    json!({"id": "nib", "type": "calligraphic", "angle": angle, "roundness": roundness, "size": size})
}

#[test]
fn nib_outline_is_the_minkowski_sweep() {
    let mut report = Vec::new();
    // L corner, flat-ish nibs at 0° and 45° (lineJoin must not matter).
    let l_pts = [v2(20.0, 20.0), v2(120.0, 20.0), v2(120.0, 120.0)];
    for (angle, join) in [
        (0.0, "miter"),
        (0.0, "round"),
        (45.0, "miter"),
        (30.0, "bevel"),
    ] {
        let v = json!({"points": [{"type": "move", "x": 20, "y": 20}, {"type": "line", "x": 120, "y": 20}, {"type": "line", "x": 120, "y": 120}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "join": join, "brush": {"brushId": "nib"}}]});
        let e = nib_sweep_error(v, nib_def(angle, 0.2, 24.0), &dense_polyline(&l_pts, false));
        report.push((format!("L corner, nib {angle}°, join {join}"), e));
    }
    // A straight line with a 45° nib: the ends are half-nibs whatever the cap.
    for cap in ["butt", "round", "square"] {
        let v = json!({"points": [{"type": "move", "x": 10, "y": 50}, {"type": "line", "x": 150, "y": 90}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 2, "cap": cap, "brush": {"brushId": "nib"}}]});
        let e = nib_sweep_error(
            v,
            nib_def(45.0, 0.3, 10.0),
            &dense_polyline(&[v2(10.0, 50.0), v2(150.0, 90.0)], false),
        );
        report.push((format!("line, 45° nib, cap {cap}"), e));
    }
    // Circle R20 with 16 px and 36 px nibs (the 36 px nib is wider than the hole).
    let circle = json!([{"type": "move", "x": 70, "y": 50}, {"type": "arc", "x": 30, "y": 50, "rx": 20, "ry": 20, "largeArc": false, "sweep": true},
                        {"type": "arc", "x": 70, "y": 50, "rx": 20, "ry": 20, "largeArc": false, "sweep": true}]);
    let ring: Vec<V2> = (0..4000)
        .map(|i| {
            let t = i as f64 / 4000.0 * std::f64::consts::TAU;
            v2(50.0 + 20.0 * t.cos(), 50.0 + 20.0 * t.sin())
        })
        .collect();
    for (size, round) in [(16.0, 0.05), (16.0, 0.3), (36.0, 0.05), (36.0, 0.3)] {
        let v = json!({"points": circle, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}]});
        let e = nib_sweep_error(v, nib_def(30.0, round, size), &ring);
        report.push((format!("circle R20, nib {size} px, roundness {round}"), e));
    }
    for (name, e) in &report {
        eprintln!("{name}: symmetric difference {:.3} %", e * 100.0);
    }
    for (name, e) in &report {
        assert!(*e < 0.01, "{name}: {:.3} % of the sweep area", e * 100.0);
    }
}

/// Local maxima of the wobble per 100 px (one full wobble = one crest).
fn wobbles_per_100px(frequency: f64, smoothen: f64, seed: u32) -> f64 {
    let len = 4000.0;
    let v = json!({"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": len, "y": 0}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "dynamic": {"wiggle": 5, "frequency": frequency, "smoothen": smoothen, "seed": seed}}]});
    let o = run(v, None);
    let s = samples(&o);
    let ys: Vec<f64> = s.y.clone();
    let crests = (1..ys.len() - 1)
        .filter(|&i| ys[i] < ys[i - 1] && ys[i] <= ys[i + 1])
        .count();
    crests as f64 / (len / 100.0)
}

#[test]
fn wobble_frequency_is_wobbles_per_100px() {
    for f in [0.5, 1.0, 2.0, 5.0] {
        let mean: f64 = (0..6)
            .map(|seed| wobbles_per_100px(f, 1.0, seed))
            .sum::<f64>()
            / 6.0;
        eprintln!("frequency {f}: {mean:.2} wobbles / 100 px");
        assert!(
            (mean / f - 1.0).abs() < 0.2,
            "frequency {f}: {mean:.2} wobbles per 100 px"
        );
    }
}

#[test]
fn flat_nibs_keep_a_half_unit_hairline() {
    // Roundness 0.001 × 20 units would be 0.02 units thick moving along the
    // nib; like Illustrator it draws a 0.5-unit hairline (page units at 1×).
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}]});
    let o = run(v, Some(calligraphic(json!(0), json!(0.001), json!(20))));
    let (l, r) = samples(&o).width_at(0.5);
    assert!((l + r - 0.5).abs() < 1e-9, "{l} + {r}");
}

#[test]
fn flat_nib_hairline_still_tapers_to_zero_at_the_tip() {
    // The hairline floor applies to the nib, before the width profile: a
    // wedge still closes to a point.
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "profile": {"preset": "wedge"}, "brush": {"brushId": "nib"}}]});
    let o = run(v, Some(calligraphic(json!(0), json!(0.001), json!(20))));
    let s = samples(&o);
    let (l0, r0) = s.width_at(0.0);
    let (l1, r1) = s.width_at(1.0);
    let (lm, rm) = s.width_at(0.5);
    assert!(
        ((l0 + r0).max(l1 + r1) - 0.5).abs() < 1e-6,
        "full end: {l0}+{r0} / {l1}+{r1}"
    );
    assert!((l0 + r0).min(l1 + r1) < 1e-6, "tip: {l0}+{r0} / {l1}+{r1}");
    assert!((lm + rm - 0.25).abs() < 0.01, "middle: {lm}+{rm}");
}

#[test]
fn nibs_under_the_hairline_stay_circular() {
    // A 0.3-unit nib at roundness 0.1: the floor is capped by the major
    // semi-axis, so the nib becomes a 0.3 circle (not a 0.5 blob), the same
    // width along and across its angle.
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    for angle in [0, 90] {
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}]});
        let o = run(v, Some(calligraphic(json!(angle), json!(0.1), json!(0.3))));
        let (l, r) = samples(&o).width_at(0.5);
        assert!((l + r - 0.3).abs() < 1e-6, "angle {angle}: {l} + {r}");
    }
}

#[test]
fn path_relative_nib_turns_with_the_path() {
    // Flat nib (size 10, roundness 0.2) at angle 0: along the direction of travel when path-relative.
    let run_dir = |pts: Value, relative: bool| {
        let mut brush = calligraphic(json!(0), json!(0.2), json!(10));
        if relative {
            brush["angleRelativeTo"] = json!("path");
        }
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}]});
        samples(&run(v, Some(brush))).width_at(0.5).0
    };
    let right = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    let down = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 0, "y": 100}]);
    let diag = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 70, "y": -70}]);
    assert!((run_dir(right.clone(), false) - 1.0).abs() < 1e-9);
    assert!((run_dir(down.clone(), false) - 5.0).abs() < 1e-9, "page: the nib stays horizontal");
    for pts in [right, down, diag] {
        assert!((run_dir(pts, true) - 1.0).abs() < 1e-9, "path: always along the travel");
    }
    // Unit: the heading adds to the angle (screen CCW); without a heading the page.
    let mut def = calligraphic(json!(30), json!(0.5), json!(4));
    def["angleRelativeTo"] = json!("path");
    let nib = Nib::from_brush(&def).unwrap();
    assert!(nib.angle_relative_to_path);
    let up = V2 { x: 0.0, y: -1.0 };
    let (_, _, u) = nib.ellipse(&InkAt::default(), [0.0; 3], Some(up));
    // 30° + 90° (travel straight up) = 120° CCW: (cos 120°, −sin 120°) on the y-down canvas.
    assert!((u.x - (-0.5)).abs() < 1e-9 && (u.y - (-(3f64.sqrt()) / 2.0)).abs() < 1e-9, "{u:?}");
    let (_, _, u) = nib.ellipse(&InkAt::default(), [0.0; 3], None);
    assert!((u.x - 30f64.to_radians().cos()).abs() < 1e-9);
    assert!(!Nib::from_brush(&calligraphic(json!(0), json!(1), json!(1))).unwrap().angle_relative_to_path);
}

#[test]
fn path_relative_nib_turns_in_place_at_corners() {
    // A zigzag drawn with a flat nib at 45° to the path: each leg is the nib
    // swept at its own orientation, and at each corner the nib turns in place
    // from one to the other. Compared with that swept region, rasterised.
    let (a, b, rel) = (15.0, 2.25, 45f64.to_radians()); // size 30, roundness 0.15
    let pts = [v2(0.0, 200.0), v2(120.0, 0.0), v2(240.0, 200.0), v2(360.0, 60.0)];
    let mut brush = calligraphic(json!(45), json!(0.15), json!(30));
    brush["angleRelativeTo"] = json!("path");
    let path: Vec<Value> = pts
        .iter()
        .enumerate()
        .map(|(i, p)| json!({"type": if i == 0 { "move" } else { "line" }, "x": p.x, "y": p.y}))
        .collect();
    let v = json!({"points": path, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "brush": {"brushId": "nib"}}]});
    let Outline::Shape { points, bounds, .. } = run(v, Some(brush)) else {
        panic!("expected a shape");
    };
    // The nib's major axis for a travel direction `d` (screen-CCW `rel` from it).
    let axis = |d: V2| {
        let t = (-d.y).atan2(d.x) + rel;
        v2(t.cos(), -t.sin())
    };
    let ellipse = |c: V2, u: V2| -> Vec<V2> {
        let w = v2(-u.y, u.x);
        (0..48)
            .map(|k| {
                let phi = std::f64::consts::TAU * k as f64 / 48.0;
                c + u * (a * phi.cos()) + w * (b * phi.sin())
            })
            .collect()
    };
    let mut ideal: Vec<Vec<V2>> = Vec::new();
    for w in pts.windows(2) {
        let d = (w[1] - w[0]).unit().unwrap();
        let len = w[0].dist(w[1]);
        let steps = (len / 0.25).ceil() as usize;
        for k in 0..=steps {
            ideal.push(ellipse(w[0] + d * (len * k as f64 / steps as f64), axis(d)));
        }
    }
    for j in 1..pts.len() - 1 {
        let (d0, d1) = ((pts[j] - pts[j - 1]).unit().unwrap(), (pts[j + 1] - pts[j]).unit().unwrap());
        let turn = d0.cross(d1).atan2(d0.dot(d1));
        for k in 0..=64 {
            let t = d0.y.atan2(d0.x) + turn * k as f64 / 64.0;
            ideal.push(ellipse(pts[j], axis(v2(t.cos(), t.sin()))));
        }
    }
    let g = super::raster::Grid::around(bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y, 0.5, 4.0);
    let out = g.fill_nonzero(&super::raster::polygons(&points));
    let want = g.fill_union(&ideal);
    let (deep, wide) = (g.erode(&want, 2), g.dilate(&want, 2));
    let holes = (0..out.len()).filter(|&i| deep[i] && !out[i]).count();
    let spikes = (0..out.len()).filter(|&i| out[i] && !wide[i]).count();
    let inside = deep.iter().filter(|x| **x).count();
    assert!(holes * 1000 < inside, "holes {holes} of {inside}");
    assert!(spikes * 1000 < inside, "spikes {spikes} of {inside}");
}

#[test]
fn per_stroke_overrides_replace_the_nib_options() {
    // Flat horizontal nib (size 10, roundness 0.2): 1 px going right.
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    let with = |overrides: Value| {
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1,
            "brush": {"brushId": "nib", "overrides": overrides}}]});
        samples(&run(v, Some(calligraphic(json!(0), json!(0.2), json!(10))))).width_at(0.5).0
    };
    assert!((with(json!({})) - 1.0).abs() < 1e-9);
    // Angle 90: the nib stands across the path → half its size.
    assert!((with(json!({"angle": 90})) - 5.0).abs() < 1e-9);
    // Roundness 1 → a circle of radius 5.
    assert!((with(json!({"roundness": 1})) - 5.0).abs() < 1e-9);
    // Size 20 → minor half-axis 2.
    assert!((with(json!({"size": 20})) - 2.0).abs() < 1e-9);
    // A varying override reads the ink (pressure 0 → 1: size 10 → 30 at the end).
    let v = json!({"points": pts, "inkChannels": {"t": [0, 1], "pressure": [0, 1]},
        "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1,
        "brush": {"brushId": "nib", "overrides": {"angle": 90, "size": {"value": 10, "vary": "pressure", "min": 10, "max": 30}}}}]});
    let o = run(v, Some(calligraphic(json!(0), json!(0.2), json!(10))));
    let ws = samples(&o);
    assert!((ws.width_at(0.0).0 - 5.0).abs() < 1e-6 && (ws.width_at(1.0).0 - 15.0).abs() < 1e-6);
    // Unit: with_overrides keeps what is not overridden, clamps what is.
    let nib = Nib::from_brush(&calligraphic(json!(30), json!(0.5), json!(4)))
        .unwrap()
        .with_overrides(json!({"roundness": 3, "size": {"value": 2, "vary": "wheel", "min": 1, "max": 3}}).as_object());
    assert_eq!(nib.angle, Dyn::fixed(30.0));
    assert_eq!(nib.roundness, Dyn::fixed(1.0));
    assert_eq!(nib.size.vary, Some(Vary::Wheel));
}
