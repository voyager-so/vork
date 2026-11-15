//! Simplify: fewer anchors, never further than the tolerance from the original.

use super::*;
use std::f64::consts::PI;

fn mv(x: f64, y: f64) -> BlendPathPoint {
    BlendPathPoint::Move { x, y }
}
fn ln(x: f64, y: f64) -> BlendPathPoint {
    BlendPathPoint::Line { x, y }
}

/// A densely sampled sine wave as a polyline (a traced / pencil path).
fn wave(n: usize) -> Vec<BlendPathPoint> {
    let mut pts = vec![mv(0.0, 0.0)];
    for i in 1..=n {
        let x = i as f64 * 400.0 / n as f64;
        pts.push(ln(x, 40.0 * (x / 400.0 * 2.0 * PI).sin()));
    }
    pts
}

/// A circle as a 120-gon (closed).
fn polygon_circle(r: f64, n: usize) -> Vec<BlendPathPoint> {
    (0..n)
        .map(|i| {
            let a = i as f64 / n as f64 * 2.0 * PI;
            let (x, y) = (r * a.cos(), r * a.sin());
            if i == 0 {
                mv(x, y)
            } else {
                ln(x, y)
            }
        })
        .collect()
}

/// Independent check: max distance from dense samples of `a` to dense samples of `b`, both ways.
fn deviation(a: &[BlendPathPoint], b: &[BlendPathPoint], closed: bool) -> f64 {
    let pa: Vec<V2> = split_subpaths(a)
        .iter()
        .flat_map(|s| flatten(&subpath_cubics(s, closed), 0.001))
        .collect();
    let pb: Vec<V2> = split_subpaths(b)
        .iter()
        .flat_map(|s| flatten(&subpath_cubics(s, closed), 0.001))
        .collect();
    let (da, _) = densify(&pa, 0.2);
    let (db, _) = densify(&pb, 0.2);
    let one = |from: &[V2], to: &[V2]| {
        from.iter()
            .map(|p| {
                to.windows(2)
                    .map(|w| seg_dist(*p, w[0], w[1]))
                    .fold(f64::MAX, f64::min)
            })
            .fold(0.0, f64::max)
    };
    one(&da, &db).max(one(&db, &da))
}

#[test]
fn open_wave_gets_far_fewer_anchors_within_tolerance() {
    let src = wave(400);
    for tol in [0.25, 1.0, 4.0] {
        let out = simplify_path(
            &src,
            false,
            &SimplifyOptions {
                tolerance: tol,
                ..Default::default()
            },
        );
        assert!(out.simplified);
        assert!(
            out.anchors_after < 20,
            "tol {tol}: {} anchors",
            out.anchors_after
        );
        assert!(out.anchors_after < out.anchors_before);
        let dev = deviation(&src, &out.points, false);
        assert!(dev <= tol * 1.02, "tol {tol}: deviation {dev}");
        assert!(out.max_deviation <= tol);
        // Ends stay put.
        let last = out.points.last().map(anchor).unwrap();
        assert!(last.dist(v2(400.0, 0.0)) < 1e-6);
    }
}

#[test]
fn looser_tolerance_means_fewer_anchors() {
    let src = wave(400);
    let counts: Vec<usize> = [0.05, 0.5, 5.0]
        .iter()
        .map(|&t| {
            simplify_path(
                &src,
                false,
                &SimplifyOptions {
                    tolerance: t,
                    ..Default::default()
                },
            )
            .anchors_after
        })
        .collect();
    assert!(
        counts[0] >= counts[1] && counts[1] >= counts[2],
        "{counts:?}"
    );
}

#[test]
fn closed_polygon_circle_becomes_a_few_curves() {
    let src = polygon_circle(100.0, 120);
    let out = simplify_path(
        &src,
        true,
        &SimplifyOptions {
            tolerance: 0.5,
            ..Default::default()
        },
    );
    assert!(out.simplified);
    assert!(out.anchors_after <= 10, "{} anchors", out.anchors_after);
    let dev = deviation(&src, &out.points, true);
    assert!(dev <= 0.51, "deviation {dev}");
}

#[test]
fn corners_stay_corners() {
    // An L traced densely: the corner must survive as an anchor.
    let mut src = vec![mv(0.0, 0.0)];
    for i in 1..=50 {
        src.push(ln(i as f64 * 4.0, 0.0));
    }
    for i in 1..=50 {
        src.push(ln(200.0, i as f64 * 4.0));
    }
    let out = simplify_path(
        &src,
        false,
        &SimplifyOptions {
            tolerance: 0.5,
            ..Default::default()
        },
    );
    assert!(out.anchors_after <= 4, "{} anchors", out.anchors_after);
    assert!(out
        .points
        .iter()
        .any(|p| anchor(p).dist(v2(200.0, 0.0)) < 1e-6));
    assert!(deviation(&src, &out.points, false) <= 0.51);
}

#[test]
fn already_simple_paths_are_left_alone() {
    let square = vec![
        mv(0.0, 0.0),
        ln(100.0, 0.0),
        ln(100.0, 100.0),
        ln(0.0, 100.0),
    ];
    let out = simplify_path(&square, true, &SimplifyOptions::default());
    assert!(!out.simplified);
    assert_eq!(out.anchors_after, 4);
    assert_eq!(out.max_deviation, 0.0);
}

#[test]
fn straight_lines_make_a_polygon_within_tolerance() {
    let src = wave(400);
    let out = simplify_path(
        &src,
        false,
        &SimplifyOptions {
            tolerance: 1.0,
            straight_lines: true,
            ..Default::default()
        },
    );
    assert!(out.simplified);
    assert!(out
        .points
        .iter()
        .all(|p| !matches!(p, BlendPathPoint::Curve { .. })));
    assert!(out.anchors_after < 60, "{} anchors", out.anchors_after);
    assert!(deviation(&src, &out.points, false) <= 1.02);
}

#[test]
fn compound_paths_simplify_per_subpath() {
    let mut src = wave(200);
    src.extend(vec![mv(0.0, 200.0), ln(50.0, 200.0), ln(50.0, 250.0)]);
    let out = simplify_path(&src, false, &SimplifyOptions::default());
    assert_eq!(out.subpaths.len(), 2);
    assert!(out.subpaths[0].simplified);
    assert!(!out.subpaths[1].simplified);
    // The second subpath is untouched.
    assert_eq!(&out.points[out.points.len() - 3..].len(), &3);
    let s = out.subpaths[0].sample_s.as_ref().unwrap();
    let t = out.subpaths[0].sample_t.as_ref().unwrap();
    assert_eq!(s.len(), t.len());
    assert!(t.windows(2).all(|w| w[1] >= w[0]));
}

#[test]
fn curves_and_arcs_in_the_input() {
    // An S made of many short cubics (the result of some other tool) plus an arc.
    let mut src = vec![mv(0.0, 0.0)];
    for i in 0..40 {
        let x = i as f64 * 10.0;
        src.push(BlendPathPoint::Curve {
            x: x + 10.0,
            y: 30.0 * ((x + 10.0) / 400.0 * 2.0 * PI).sin(),
            cp1: (x + 3.3, 30.0 * ((x + 3.3) / 400.0 * 2.0 * PI).sin()),
            cp2: (x + 6.7, 30.0 * ((x + 6.7) / 400.0 * 2.0 * PI).sin()),
        });
    }
    src.push(BlendPathPoint::Arc {
        x: 500.0,
        y: 0.0,
        rx: 50.0,
        ry: 50.0,
        x_axis_rotation: 0.0,
        large_arc: false,
        sweep: true,
    });
    let out = simplify_path(
        &src,
        false,
        &SimplifyOptions {
            tolerance: 0.5,
            ..Default::default()
        },
    );
    assert!(out.simplified);
    assert!(out.anchors_after < 15, "{} anchors", out.anchors_after);
    assert!(deviation(&src, &out.points, false) <= 0.51);
}

#[test]
fn the_sample_budget_covers_all_subpaths_together() {
    // 12 long subpaths each hit the per-subpath cap alone; together they share the path's budget (25,000
    // subpaths at the per-subpath cap took 40 min, docs/qa/robustness.md). Each still stays within tolerance.
    let mut pts = Vec::new();
    for k in 0..12 {
        pts.extend(wave(60).iter().map(|p| match *p {
            BlendPathPoint::Move { x, y } => mv(x, y + k as f64 * 100.0),
            BlendPathPoint::Line { x, y } => ln(x, y + k as f64 * 100.0),
            ref other => other.clone(),
        }));
    }
    let opts = SimplifyOptions {
        tolerance: 1.0,
        ..SimplifyOptions::default()
    };
    let count = |out: &SimplifyOutput| -> usize {
        out.subpaths
            .iter()
            .filter_map(|s| s.sample_s.as_ref())
            .map(Vec::len)
            .sum()
    };
    let unbudgeted = count(&simplify_path(&pts, false, &opts));
    let out = simplify_path_budgeted(&pts, false, &opts, 1_200);
    let samples = count(&out);
    // The spacing grows to fit the budget; the input's own vertices are always samples too.
    assert!(samples <= 1_200 + pts.len(), "{samples} samples");
    assert!(samples < unbudgeted / 2, "{samples} of {unbudgeted}");
    assert!(out.max_deviation <= 1.0, "{}", out.max_deviation);
    assert!(out.simplified);
}

#[test]
fn json_envelope() {
    let req = json!({
        "points": wave(100).iter().map(BlendPathPoint::to_json).collect::<Vec<_>>(),
        "closed": false,
        "tolerance": 1.0,
    });
    let v: Value = crate::value::from_str(&simplify_path_json(&req.to_string())).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["simplified"], true);
    assert_eq!(v["anchorsBefore"], 101);
    assert!(v["anchorsAfter"].as_u64().unwrap() < 20);
    assert!(v["subpaths"][0]["sampleT"].is_array());
    for bad in ["x", "{}", r#"{"points":[{"type":"?"}]}"#] {
        let v: Value = crate::value::from_str(&simplify_path_json(bad)).unwrap();
        assert_eq!(v["status"], "error", "{bad}");
    }
    let v: Value = crate::value::from_str(&simplify_path_json(r#"{"points":[]}"#)).unwrap();
    assert_eq!(v["status"], "empty");
}

#[test]
fn simplify_fuzz_no_panic() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for case in 0..200 {
        let n = 1 + (rnd() * 40.0) as usize;
        let mut pts = vec![mv(rnd() * 100.0, rnd() * 100.0)];
        for _ in 0..n {
            let r = rnd();
            let (x, y) = if r < 0.1 {
                (f64::NAN, 1.0)
            } else if r < 0.2 {
                (1e12, 0.0)
            } else {
                (rnd() * 100.0, rnd() * 100.0)
            };
            pts.push(if rnd() < 0.3 {
                BlendPathPoint::Curve {
                    x,
                    y,
                    cp1: (rnd() * 100.0, 0.0),
                    cp2: (0.0, rnd() * 100.0),
                }
            } else if rnd() < 0.1 {
                mv(x, y)
            } else {
                ln(x, y)
            });
        }
        let tol = [0.0, 0.01, 1.0, 1e6, f64::NAN][case % 5];
        let out = simplify_path(
            &pts,
            case % 2 == 0,
            &SimplifyOptions {
                tolerance: tol,
                straight_lines: case % 3 == 0,
                ..Default::default()
            },
        );
        assert!(out.anchors_after <= out.anchors_before);
    }
}

// ---------------------------------------------------------------------------
// The Illustrator parity cases (scripts/compare-illustrator/parity-cases.mjs,
// docs/qa/illustrator-parity.md): polylines Illustrator's automatic Simplify
// takes to 9 / 51 / 6 anchors within 0.19 / 0.19 / 0.13 px.
// ---------------------------------------------------------------------------

fn polyline(pts: &[(f64, f64)]) -> Vec<BlendPathPoint> {
    pts.iter()
        .enumerate()
        .map(|(i, &(x, y))| if i == 0 { mv(x, y) } else { ln(x, y) })
        .collect()
}

/// (name, points, closed, Illustrator's anchors, Illustrator's deviation).
fn parity_cases() -> Vec<(&'static str, Vec<BlendPathPoint>, bool, usize, f64)> {
    let sine: Vec<(f64, f64)> = (0..=60)
        .map(|i| (15.0 + i as f64 * 3.5, 100.0 + 45.0 * (i as f64 / 6.0).sin()))
        .collect();
    let lp: Vec<(f64, f64)> = (0..80)
        .map(|i| {
            let a = i as f64 / 80.0 * 2.0 * PI;
            let r = 70.0 + 4.0 * (7.0 * a).sin() + 2.0 * (13.0 * a).cos();
            (120.0 + r * a.cos(), 110.0 + r * a.sin())
        })
        .collect();
    let mut mixed = vec![(20.0, 160.0), (50.0, 60.0), (80.0, 160.0), (110.0, 110.0)];
    for i in 1..=30 {
        let a = PI - i as f64 / 30.0 * PI;
        mixed.push((160.0 + 50.0 * a.cos(), 110.0 + 50.0 * a.sin()));
    }
    vec![
        ("sine", polyline(&sine), false, 9, 0.19),
        ("wobbly-loop", polyline(&lp), true, 51, 0.19),
        ("zigzag-arc", polyline(&mixed), false, 6, 0.13),
    ]
}

#[test]
#[ignore = "probe: prints anchors / deviation per tolerance (cargo test -p moda-scene parity_probe -- --ignored --nocapture)"]
fn parity_probe() {
    for (name, pts, closed, ai_n, ai_dev) in parity_cases() {
        let mut row = format!(
            "{name:12} in {:3}; Illustrator {ai_n} ({ai_dev})",
            pts.len()
        );
        for tol in [3.0, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05] {
            let out = simplify_path(
                &pts,
                closed,
                &SimplifyOptions {
                    tolerance: tol,
                    ..Default::default()
                },
            );
            let dev = deviation(&pts, &out.points, closed);
            row += &format!(" | {tol}: {} ({dev:.2})", out.anchors_after);
        }
        println!("{row}");
    }
}

/// Anchors in `points` (a closed fit repeats its first point at the end).
fn anchors(points: &[BlendPathPoint], closed: bool) -> usize {
    let n = points.len();
    if closed && n > 1 && anchor(&points[0]).dist(anchor(&points[n - 1])) < 1e-6 {
        n - 1
    } else {
        n
    }
}

#[test]
fn matches_illustrators_automatic_simplify() {
    // Simplify's default (0.2, `DEFAULT_SIMPLIFY_FIDELITY` in core/geometry/fidelity.ts) lands where Illustrator's
    // automatic Simplify does, with no more anchors; the finest setting (0.1) gets closer still.
    for (name, pts, closed, ai_anchors, ai_dev) in parity_cases() {
        let run = |tol: f64| {
            let out = simplify_path(
                &pts,
                closed,
                &SimplifyOptions {
                    tolerance: tol,
                    ..Default::default()
                },
            );
            (
                anchors(&out.points, closed),
                deviation(&pts, &out.points, closed),
            )
        };
        let (n, dev) = run(0.2);
        assert!(
            n <= ai_anchors && dev <= ai_dev.max(0.2) * 1.02,
            "{name} at 0.2: {n} anchors, {dev:.3} px (Illustrator {ai_anchors}, {ai_dev})"
        );
        let (n, dev) = run(0.1);
        assert!(
            dev <= 0.1 * 1.02 && n < pts.len(),
            "{name} at 0.1: {n} anchors, {dev:.3} px"
        );
    }
}


#[test]
fn flatten_keeps_collinear_excursions_in_either_direction() {
    for sign in [-1.0, 1.0] {
        let c = Cubic([
            v2(0.0, 0.0),
            v2(sign * 100.0, 0.0),
            v2(sign * 100.0, 0.0),
            v2(sign, 0.0),
        ]);
        let poly = flatten(&[c], 0.001);
        let reach = poly.iter().map(|p| sign * p.x).fold(0.0, f64::max);
        assert!((reach - 75.125).abs() < 0.01, "lost excursion: {reach}");
    }
}
