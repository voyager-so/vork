use super::*;
use crate::value::Value;

/// xorshift64*; deterministic.
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub(crate) fn f(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
}

fn segs_of(points: &[BlendPathPoint]) -> Vec<FitSeg> {
    let mut out = Vec::new();
    let mut cur = V2::default();
    for p in points {
        match p {
            BlendPathPoint::Move { x, y } => cur = v2(*x, *y),
            BlendPathPoint::Line { x, y } => {
                out.push(FitSeg::Line(cur, v2(*x, *y)));
                cur = v2(*x, *y);
            }
            BlendPathPoint::Curve { x, y, cp1, cp2 } => {
                let end = v2(*x, *y);
                out.push(FitSeg::Cubic(Cubic([
                    cur,
                    v2(cp1.0, cp1.1),
                    v2(cp2.0, cp2.1),
                    end,
                ])));
                cur = end;
            }
            _ => panic!("fit emits only move/line/curve"),
        }
    }
    out
}

fn flatten(segs: &[FitSeg], per: usize) -> Vec<V2> {
    let mut pts = Vec::new();
    for s in segs {
        let c = s.as_cubic();
        for i in 0..=per {
            pts.push(c.eval(i as f64 / per as f64));
        }
    }
    pts
}

fn dist_to_polyline(p: V2, poly: &[V2]) -> f64 {
    let mut best = f64::INFINITY;
    for w in poly.windows(2) {
        let d = w[1] - w[0];
        let t = if d.len2() > 0.0 {
            ((p - w[0]).dot(d) / d.len2()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        best = best.min((w[0] + d * t).dist(p));
    }
    best
}

fn max_deviation(input: &[V2], out: &FitOutput) -> f64 {
    let poly = flatten(&segs_of(&out.points), 256);
    input
        .iter()
        .map(|p| dist_to_polyline(*p, &poly))
        .fold(0.0, f64::max)
}

fn circle(n: usize, r: f64, closed_repeat: bool) -> Vec<V2> {
    let mut pts: Vec<V2> = (0..n)
        .map(|i| {
            let a = i as f64 / n as f64 * std::f64::consts::TAU;
            v2(100.0 + r * a.cos(), 100.0 + r * a.sin())
        })
        .collect();
    if closed_repeat {
        pts.push(pts[0]);
    }
    pts
}

#[test]
fn straight_samples_fit_one_segment() {
    let input: Vec<V2> = (0..50).map(|i| v2(i as f64 * 2.0, i as f64)).collect();
    let out = fit_points(&input, &FitOptions::default());
    assert_eq!(out.points.len(), 2, "{:?}", out.points);
    assert!(max_deviation(&input, &out) < 1e-6);
    // Evenly spaced samples map to evenly spaced arc-length fractions.
    for (i, t) in out.sample_t.iter().enumerate() {
        assert!((t - i as f64 / 49.0).abs() < 1e-3, "sample {i}: {t}");
    }
}

#[test]
fn circle_fits_within_tolerance_with_few_segments() {
    for tol in [0.1, 0.5, 2.0] {
        let input = circle(400, 80.0, false);
        let out = fit_points(
            &input,
            &FitOptions {
                tolerance: tol,
                ..Default::default()
            },
        );
        let dev = max_deviation(&input, &out);
        assert!(dev <= tol * 1.05, "tol {tol}: deviation {dev}");
        assert!(
            out.points.len() <= 12,
            "tol {tol}: {} points",
            out.points.len()
        );
        assert!(out.corners.is_empty(), "a smooth arc has no corners");
    }
}

#[test]
fn closed_circle_returns_to_start() {
    let input = circle(200, 50.0, true);
    let out = fit_points(
        &input,
        &FitOptions {
            tolerance: 0.25,
            closed: true,
            ..Default::default()
        },
    );
    assert!(out.closed);
    let segs = segs_of(&out.points);
    assert!(
        segs.first()
            .unwrap()
            .start()
            .dist(segs.last().unwrap().end())
            < 1e-9
    );
    assert!(max_deviation(&input, &out) <= 0.26);
    assert_eq!(*out.sample_t.last().unwrap(), 1.0);
}

#[test]
fn v_shape_keeps_a_sharp_corner() {
    // Dense samples down to an apex and back up: the apex must be an anchor.
    let mut input = Vec::new();
    for i in 0..=40 {
        input.push(v2(i as f64, i as f64 * 2.0));
    }
    for i in 1..=40 {
        input.push(v2(40.0 + i as f64, 80.0 - i as f64 * 2.0));
    }
    let out = fit_points(
        &input,
        &FitOptions {
            tolerance: 1.0,
            ..Default::default()
        },
    );
    assert_eq!(out.corners, vec![40]);
    let segs = segs_of(&out.points);
    assert!(segs.iter().any(|s| s.end().dist(v2(40.0, 80.0)) < 1e-9));
    assert!(max_deviation(&input, &out) < 1.0);
}

#[test]
fn sparse_polyline_corners_are_vertices() {
    // Sparse samples with straight runs: the turn is concentrated at the vertex.
    let input = vec![
        v2(0.0, 0.0),
        v2(50.0, 0.0),
        v2(100.0, 0.0),
        v2(100.0, 50.0),
        v2(100.0, 100.0),
    ];
    let out = fit_points(&input, &FitOptions::default());
    assert_eq!(out.corners, vec![2]);
    assert!(max_deviation(&input, &out) < 1e-9);
    for (a, e) in out.sample_t.iter().zip([0.0, 0.25, 0.5, 0.75, 1.0]) {
        assert!((a - e).abs() < 1e-9, "{:?}", out.sample_t);
    }
    // Three points: a lone vertex is a corner.
    assert_eq!(
        fit_points(
            &[v2(0.0, 0.0), v2(20.0, 40.0), v2(40.0, 0.0)],
            &FitOptions::default()
        )
        .corners,
        vec![1]
    );
}

#[test]
fn sample_t_is_monotone_and_spans_zero_to_one() {
    let mut rng = Rng::new(7);
    let mut p = v2(0.0, 0.0);
    let mut dir: f64 = 0.0;
    let mut input = Vec::new();
    for _ in 0..500 {
        dir += rng.f(-0.3, 0.3);
        p = p + v2(dir.cos(), dir.sin()) * rng.f(0.5, 3.0);
        input.push(p + v2(rng.f(-0.3, 0.3), rng.f(-0.3, 0.3)));
    }
    let out = fit_points(
        &input,
        &FitOptions {
            tolerance: 1.5,
            ..Default::default()
        },
    );
    assert_eq!(out.sample_t.len(), input.len());
    assert_eq!(out.sample_t[0], 0.0);
    assert!((out.sample_t[input.len() - 1] - 1.0).abs() < 1e-9);
    assert!(out.sample_t.windows(2).all(|w| w[1] >= w[0]));
    assert!(max_deviation(&input, &out) <= 1.5 * 1.05);
    assert!(
        out.points.len() < 120,
        "{} points for 500 noisy samples",
        out.points.len()
    );
}

#[test]
fn degenerate_inputs() {
    let empty = fit_points(&[], &FitOptions::default());
    assert!(empty.points.is_empty() && empty.sample_t.is_empty());

    let one = fit_points(&[v2(3.0, 4.0)], &FitOptions::default());
    assert_eq!(one.points.len(), 1);
    assert_eq!(one.sample_t, vec![0.0]);

    let same = fit_points(&[v2(1.0, 1.0); 5], &FitOptions::default());
    assert_eq!(same.points.len(), 1);
    assert_eq!(same.sample_t, vec![0.0; 5]);

    let nan = fit_points(
        &[
            v2(f64::NAN, 0.0),
            v2(0.0, 0.0),
            v2(f64::INFINITY, 1.0),
            v2(10.0, 0.0),
        ],
        &FitOptions::default(),
    );
    assert_eq!(nan.sample_t, vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(nan.points.len(), 2);
}

#[test]
fn fit_polyline_honours_breaks_and_tolerance_zero() {
    let pts: Vec<V2> = (0..=20)
        .map(|i| {
            let a = i as f64 / 20.0 * std::f64::consts::PI;
            v2(a.cos() * 10.0, a.sin() * 10.0)
        })
        .collect();
    let mut breaks = vec![false; pts.len()];
    breaks[10] = true;
    let segs = fit_polyline(&pts, &breaks, false, 0.05);
    assert!(
        segs.iter().any(|s| s.end().dist(pts[10]) < 1e-12),
        "break point is an anchor"
    );
    let lines = fit_polyline(&pts, &breaks, false, 0.0);
    assert_eq!(lines.len(), 20);
    let ring = fit_polyline(&pts[..20], &[], true, 0.0);
    assert_eq!(ring.len(), 20);
    assert!(ring.last().unwrap().end().dist(pts[0]) < 1e-12);
}

#[test]
fn json_envelope() {
    let ok: Value = crate::value::from_str(&fit_points_json(
        r#"{"points": [[0,0],[5,1],[10,0]], "tolerance": 0.5}"#,
    ))
    .unwrap();
    assert_eq!(ok["status"], "ok");
    assert_eq!(ok["sampleT"].as_array().unwrap().len(), 3);
    let empty: Value = crate::value::from_str(&fit_points_json(r#"{"points": []}"#)).unwrap();
    assert_eq!(empty["status"], "empty");
    for bad in ["nope", "[]", r#"{"points": 3}"#] {
        let e: Value = crate::value::from_str(&fit_points_json(bad)).unwrap();
        assert_eq!(e["status"], "error", "{bad}");
    }
}

/// No panics and finite output on hostile input (wasm is panic = abort).
#[test]
fn fit_fuzz_no_panic() {
    let mut rng = Rng::new(0xF17);
    for case in 0..3000 {
        let n = (rng.next_u64() % 60) as usize;
        let scale = [1e-9, 1e-3, 1.0, 1e3, 1e8][case % 5];
        let mut input: Vec<V2> = (0..n)
            .map(|_| match rng.next_u64() % 12 {
                0 => v2(f64::NAN, 1.0),
                1 => v2(0.0, 0.0),
                2 => v2(1e300, -1e300),
                _ => v2(rng.f(-1.0, 1.0) * scale, rng.f(-1.0, 1.0) * scale),
            })
            .collect();
        if case % 7 == 0 && n > 2 {
            // repeated points and back-tracking
            let p = input[1];
            input.insert(2, p);
            input.push(input[0]);
        }
        let opts = FitOptions {
            tolerance: [0.0, 1e-6, 0.5, 5.0, f64::NAN, 1e9][case % 6],
            corner_angle: [0.0, 30.0, 70.0, 179.0, f64::INFINITY][case % 5],
            closed: case % 3 == 0,
        };
        let out = fit_points(&input, &opts);
        assert_eq!(out.sample_t.len(), input.len(), "case {case}");
        assert!(
            out.sample_t
                .iter()
                .all(|t| t.is_finite() && (0.0..=1.0).contains(t)),
            "case {case}"
        );
        for p in &out.points {
            let j = p.to_json();
            assert!(
                j.to_string().find("null").is_none(),
                "case {case}: non-finite output {j}"
            );
        }
        let breaks: Vec<bool> = (0..n).map(|_| rng.next_u64() % 5 == 0).collect();
        for s in fit_polyline(&input, &breaks, case % 2 == 0, opts.tolerance) {
            assert!(s.start().is_finite() && s.end().is_finite(), "case {case}");
        }
    }
}

fn ring(n: usize, r: f64, from: f64, to: f64, closed: bool) -> Vec<V2> {
    let count = if closed { n } else { n - 1 };
    (0..n)
        .map(|i| {
            let a = from + (to - from) * i as f64 / count as f64;
            v2(r * a.cos(), r * a.sin())
        })
        .collect()
}

#[test]
fn sparse_polygons_keep_corners_sparse_arcs_stay_smooth() {
    use std::f64::consts::{PI, TAU};
    let open = FitOptions::default();
    let closed = FitOptions {
        closed: true,
        ..Default::default()
    };
    // Agents send sparse polygons and expect corners.
    let square = [
        v2(0.0, 0.0),
        v2(100.0, 0.0),
        v2(100.0, 100.0),
        v2(0.0, 100.0),
    ];
    assert_eq!(
        fit_points(&square, &open).corners,
        vec![1, 2],
        "open square"
    );
    assert_eq!(
        fit_points(&square, &closed).corners,
        vec![0, 1, 2, 3],
        "closed square"
    );
    assert_eq!(
        fit_points(&ring(5, 50.0, 0.0, TAU, true), &closed).corners,
        vec![0, 1, 2, 3, 4],
        "pentagon"
    );
    let tri = [v2(0.0, 0.0), v2(100.0, 0.0), v2(50.0, 80.0)];
    assert_eq!(
        fit_points(&tri, &closed).corners,
        vec![0, 1, 2],
        "closed triangle"
    );
    assert_eq!(fit_points(&tri, &open).corners, vec![1], "open triangle");
    // Gentle, evenly spread turns stay smooth.
    assert!(
        fit_points(&ring(8, 50.0, 0.0, TAU, true), &closed)
            .corners
            .is_empty(),
        "8-point circle"
    );
    assert!(
        fit_points(&ring(6, 50.0, 0.0, PI, false), &open)
            .corners
            .is_empty(),
        "6-point U"
    );
    let wave: Vec<V2> = (0..12)
        .map(|i| v2(i as f64 * 20.0, 15.0 * (i as f64 * PI / 3.0).sin()))
        .collect();
    assert!(fit_points(&wave, &open).corners.is_empty(), "12-point wave");
    // Dense samples round a tight bend: the stencil turn is large but spread
    // over short chords, so no corner (the review's pointy-arc-top case).
    let tight = ring(40, 3.0, 0.0, PI, false);
    assert!(
        fit_points(&tight, &open).corners.is_empty(),
        "dense tight arc"
    );
    // A dense V is still a corner.
    let mut v = Vec::new();
    for i in 0..=40 {
        v.push(v2(i as f64, i as f64 * 2.0));
    }
    for i in 1..=40 {
        v.push(v2(40.0 + i as f64, 80.0 - i as f64 * 2.0));
    }
    assert_eq!(fit_points(&v, &open).corners, vec![40], "dense V");
}

#[test]
fn closed_fit_starts_at_the_first_sample() {
    // A teardrop: its corner is at sample 30, but the loop starts at sample 0.
    let mut pts: Vec<V2> = (0..30)
        .map(|i| {
            let a = std::f64::consts::PI * i as f64 / 30.0;
            v2(50.0 * a.cos(), 50.0 * a.sin())
        })
        .collect();
    // Two straight legs down to a point at (0, -80) and back.
    for i in 0..30 {
        let f = i as f64 / 30.0;
        pts.push(if f < 0.5 {
            v2(-50.0 + 100.0 * f, -160.0 * f)
        } else {
            v2(100.0 * (f - 0.5), -80.0 + 160.0 * (f - 0.5))
        });
    }
    let out = fit_points(
        &pts,
        &FitOptions {
            closed: true,
            ..Default::default()
        },
    );
    let BlendPathPoint::Move { x, y } = out.points[0] else {
        panic!()
    };
    assert!(
        (x - 50.0).abs() < 1e-9 && y.abs() < 1e-9,
        "starts at input[0]: ({x}, {y})"
    );
    assert_eq!(out.sample_t[0], 0.0);
}

#[test]
fn sample_t_uses_the_shared_arc_length() {
    // On a corner the sample's t is the fitted path's anchor fraction from
    // `path_length::anchor_fractions` (the one arc length markers use).
    let mut input = Vec::new();
    for i in 0..=40 {
        input.push(v2(i as f64, (i as f64 * 0.2).sin() * 10.0 + i as f64));
    }
    for i in 1..=40 {
        input.push(v2(40.0 + i as f64, 40.0 + (0.0 - i as f64) * 2.0));
    }
    let out = fit_points(
        &input,
        &FitOptions {
            tolerance: 0.5,
            ..Default::default()
        },
    );
    let json: Vec<Value> = out.points.iter().map(BlendPathPoint::to_json).collect();
    let fr = crate::path_length::anchor_fractions(&json, false);
    for &c in &out.corners {
        let p = input[c];
        let k = out
            .points
            .iter()
            .position(|q| {
                let j = q.to_json();
                (j["x"].as_f64().unwrap() - p.x).abs() < 1e-9
                    && (j["y"].as_f64().unwrap() - p.y).abs() < 1e-9
            })
            .expect("corner is an anchor");
        assert!(
            (out.sample_t[c] - fr.anchor_t[k]).abs() < 1e-12,
            "{} vs {}",
            out.sample_t[c],
            fr.anchor_t[k]
        );
    }
    assert!(!out.corners.is_empty());
}

#[test]
fn corner_walk_is_bounded() {
    // 100k samples inside half a pixel: every walk would otherwise run long.
    let mut rng = Rng::new(3);
    let blob: Vec<V2> = (0..100_000)
        .map(|_| v2(rng.f(0.0, 0.5), rng.f(0.0, 0.5)))
        .collect();
    let out = fit_points(
        &blob,
        &FitOptions {
            tolerance: 5.0,
            ..Default::default()
        },
    );
    assert_eq!(out.sample_t.len(), blob.len());
}

#[test]
fn a_straight_run_takes_in_a_break_a_hair_off_its_line() {
    // A line sampled every 0.25, then a forced break a hair off it (a closed stroke's seam, offset along its own
    // slightly kinked bisector), then a quarter circle bending away.
    let run = |off: f64| {
        let mut pts: Vec<V2> = (0..=40).map(|i| v2(0.25 * i as f64, 0.0)).collect();
        pts.push(v2(10.25, off));
        let c = v2(10.25, 5.0 + off);
        pts.extend((1..=32).map(|i| {
            let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_2 * i as f64 / 32.0;
            c + v2(a.cos(), a.sin()) * 5.0
        }));
        let mut breaks = vec![false; pts.len()];
        breaks[41] = true;
        fit_polyline(&pts, &breaks, false, 0.05)
    };
    // 2e-5 off: one line to the break, not a 0.25 line of its own after it.
    let segs = run(2e-5);
    assert!(
        matches!(segs[0], FitSeg::Line(a, b) if a == v2(0.0, 0.0) && b == v2(10.25, 2e-5)),
        "{segs:?}"
    );
    assert!(matches!(segs[1], FitSeg::Cubic(_)), "{segs:?}");
    // Visibly off (0.01, a fifth of the tolerance): the break keeps its own short line.
    let segs = run(0.01);
    assert!(
        matches!(segs[0], FitSeg::Line(_, b) if b == v2(10.0, 0.0)),
        "{segs:?}"
    );
    assert!(
        matches!(segs[1], FitSeg::Line(_, b) if b == v2(10.25, 0.01)),
        "{segs:?}"
    );
}
