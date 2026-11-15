//! Offset Path: areas and bounds against the analytic answers, insets, open
//! paths, compound paths, clean-up of self-intersections, and a pixel check.

use super::*;
use crate::path_bool::Bounds;
use crate::path_geom::vec2::{v2, V2};
use crate::stroke_outline::raster::{polygons, Grid};
use std::f64::consts::PI;

/// Every result is within the default tolerance of the exact offset.
const TOL: f64 = DEFAULT_OFFSET_TOLERANCE;

fn mv(x: f64, y: f64) -> BlendPathPoint {
    BlendPathPoint::Move { x, y }
}
fn ln(x: f64, y: f64) -> BlendPathPoint {
    BlendPathPoint::Line { x, y }
}

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<BlendPathPoint> {
    vec![mv(x0, y0), ln(x1, y0), ln(x1, y1), ln(x0, y1)]
}

/// A circle of four kappa cubics (what shapes convert to).
fn circle(cx: f64, cy: f64, r: f64) -> Vec<BlendPathPoint> {
    let k = 0.552_284_749_830_793_4 * r;
    let c = |x: f64, y: f64, a: (f64, f64), b: (f64, f64)| BlendPathPoint::Curve {
        x,
        y,
        cp1: a,
        cp2: b,
    };
    vec![
        mv(cx + r, cy),
        c(cx, cy + r, (cx + r, cy + k), (cx + k, cy + r)),
        c(cx - r, cy, (cx - k, cy + r), (cx - r, cy + k)),
        c(cx, cy - r, (cx - r, cy - k), (cx - k, cy - r)),
        c(cx + r, cy, (cx + k, cy - r), (cx + r, cy - k)),
    ]
}

fn shoelace(poly: &[V2]) -> f64 {
    let n = poly.len();
    let mut a = 0.0;
    for i in 0..n {
        let (p, q) = (poly[i], poly[(i + 1) % n]);
        a += p.x * q.y - q.x * p.y;
    }
    a / 2.0
}

fn inside(poly: &[V2], p: V2) -> bool {
    let n = poly.len();
    let mut c = false;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (b.x - a.x) * (p.y - a.y) / (b.y - a.y) {
            c = !c;
        }
    }
    c
}

/// Even-odd area: each contour counts positive at even nesting depth, negative at odd.
fn evenodd_area(points: &[BlendPathPoint]) -> f64 {
    let polys = polygons(points);
    let mut total = 0.0;
    for (i, p) in polys.iter().enumerate() {
        let probe = p[0];
        let depth = polys
            .iter()
            .enumerate()
            .filter(|(j, q)| *j != i && inside(q, probe))
            .count();
        let a = shoelace(p).abs();
        total += if depth % 2 == 0 { a } else { -a };
    }
    total
}

fn run(req: &OffsetRequest) -> (Vec<BlendPathPoint>, usize, Bounds) {
    match offset_path(req).expect("offset solves") {
        BooleanOutcome::Shape {
            points,
            subpath_count,
            bounds,
        } => (points, subpath_count, bounds),
        BooleanOutcome::Empty => panic!("offset came back empty"),
    }
}

fn req(points: Vec<BlendPathPoint>, closed: bool, d: f64, join: OffsetJoin) -> OffsetRequest {
    OffsetRequest {
        join,
        ..OffsetRequest::new(points, closed, d)
    }
}

fn assert_close(what: &str, got: f64, want: f64, tol: f64) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (±{tol})"
    );
}

fn assert_bounds(b: &Bounds, want: [f64; 4], tol: f64) {
    assert_close("minX", b.min_x, want[0], tol);
    assert_close("minY", b.min_y, want[1], tol);
    assert_close("maxX", b.max_x, want[2], tol);
    assert_close("maxY", b.max_y, want[3], tol);
}

#[test]
fn square_outward_miter_is_a_bigger_square() {
    let (points, n, b) = run(&req(
        rect(0.0, 0.0, 100.0, 100.0),
        true,
        10.0,
        OffsetJoin::Miter,
    ));
    assert_eq!(n, 1);
    assert_bounds(&b, [-10.0, -10.0, 110.0, 110.0], TOL);
    assert_close("area", evenodd_area(&points), 120.0 * 120.0, 1.0);
    // Sharp corners stay one anchor each: four lines, no curves.
    assert!(points
        .iter()
        .all(|p| !matches!(p, BlendPathPoint::Curve { .. })));
}

#[test]
fn square_outward_round_and_bevel() {
    let (points, _, b) = run(&req(
        rect(0.0, 0.0, 100.0, 100.0),
        true,
        10.0,
        OffsetJoin::Round,
    ));
    assert_bounds(&b, [-10.0, -10.0, 110.0, 110.0], TOL);
    // Minkowski sum with a disc: square + four side strips + one full disc.
    assert_close(
        "round area",
        evenodd_area(&points),
        10_000.0 + 4_000.0 + PI * 100.0,
        (400.0 + 20.0 * PI) * TOL,
    );

    let (points, _, b) = run(&req(
        rect(0.0, 0.0, 100.0, 100.0),
        true,
        10.0,
        OffsetJoin::Bevel,
    ));
    assert_bounds(&b, [-10.0, -10.0, 110.0, 110.0], TOL);
    // The miter square minus four corner triangles (legs 10).
    assert_close(
        "bevel area",
        evenodd_area(&points),
        14_400.0 - 4.0 * 50.0,
        1.0,
    );
}

#[test]
fn square_inward_and_too_far_inward() {
    for join in [OffsetJoin::Miter, OffsetJoin::Round, OffsetJoin::Bevel] {
        let (points, n, b) = run(&req(rect(0.0, 0.0, 100.0, 100.0), true, -10.0, join));
        assert_eq!(n, 1);
        // An inset's corners are sharp whatever the join (the join is on the outside).
        assert_bounds(&b, [10.0, 10.0, 90.0, 90.0], TOL);
        assert_close("inset area", evenodd_area(&points), 6_400.0, 1.0);
    }
    let gone = offset_path(&req(
        rect(0.0, 0.0, 100.0, 100.0),
        true,
        -60.0,
        OffsetJoin::Miter,
    ))
    .unwrap();
    assert!(matches!(gone, BooleanOutcome::Empty));
}

#[test]
fn circle_outward_and_inward() {
    for (d, r) in [(10.0, 60.0), (-10.0, 40.0), (25.0, 75.0)] {
        let (points, n, b) = run(&req(circle(0.0, 0.0, 50.0), true, d, OffsetJoin::Round));
        assert_eq!(n, 1);
        // The kappa circle itself is 0.03% off a true circle; the offset keeps that.
        assert_bounds(&b, [-r, -r, r, r], TOL);
        let area = evenodd_area(&points);
        // Within the tolerance everywhere: the area is off by at most perimeter × TOL.
        assert_close("circle area", area, PI * r * r, 2.0 * PI * r * TOL);
    }
}

#[test]
fn open_path_becomes_a_closed_outline() {
    let line = vec![mv(0.0, 0.0), ln(100.0, 0.0)];
    let (points, n, b) = run(&req(line.clone(), false, 10.0, OffsetJoin::Miter));
    assert_eq!(n, 1);
    assert_bounds(&b, [0.0, -10.0, 100.0, 10.0], TOL);
    assert_close("butt area", evenodd_area(&points), 2_000.0, 0.5);

    // Round joins give round ends; a negative distance offsets an open path the same way.
    let (points, _, b) = run(&req(line, false, -10.0, OffsetJoin::Round));
    assert_bounds(&b, [-10.0, -10.0, 110.0, 10.0], TOL);
    assert_close(
        "round area",
        evenodd_area(&points),
        2_000.0 + PI * 100.0,
        (200.0 + 20.0 * PI) * TOL,
    );
}

#[test]
fn open_polyline_self_overlap_is_cleaned_up() {
    // A tight zig-zag whose outline overlaps itself: one clean contour, no double cover.
    let zig = vec![mv(0.0, 0.0), ln(50.0, 0.0), ln(0.0, 6.0), ln(50.0, 12.0)];
    let (points, n, _) = run(&req(zig, false, 5.0, OffsetJoin::Round));
    assert_eq!(n, 1, "one contour after clean-up");
    let area = evenodd_area(&points);
    // Bounded by the three legs' capsules without overlap counted twice.
    assert!(area > 0.0 && area < 3.0 * (50.5 * 10.0 + PI * 25.0));
}

#[test]
fn compound_path_hole_shrinks_as_the_outside_grows() {
    let mut pts = rect(0.0, 0.0, 100.0, 100.0);
    pts.extend(rect(40.0, 40.0, 60.0, 60.0));
    let mut r = req(pts.clone(), true, 5.0, OffsetJoin::Miter);
    r.fill_rule = FillRule::EvenOdd;
    let (points, n, b) = run(&r);
    assert_eq!(n, 2);
    assert_bounds(&b, [-5.0, -5.0, 105.0, 105.0], TOL);
    assert_close(
        "area",
        evenodd_area(&points),
        110.0 * 110.0 - 10.0 * 10.0,
        1.0,
    );

    // Grown past the hole's half-width, the hole closes.
    let mut r = req(pts.clone(), true, 12.0, OffsetJoin::Miter);
    r.fill_rule = FillRule::EvenOdd;
    let (_, n, _) = run(&r);
    assert_eq!(n, 1);

    // Inset: the hole grows.
    let mut r = req(pts, true, -5.0, OffsetJoin::Miter);
    r.fill_rule = FillRule::EvenOdd;
    let (points, n, _) = run(&r);
    assert_eq!(n, 2);
    assert_close(
        "inset area",
        evenodd_area(&points),
        90.0 * 90.0 - 30.0 * 30.0,
        1.0,
    );
}

#[test]
fn concave_notch_fills_in_when_grown() {
    // A U: a 100×100 square with a 6-wide slot cut 80 deep from the top.
    let u = vec![
        mv(0.0, 0.0),
        ln(47.0, 0.0),
        ln(47.0, 80.0),
        ln(53.0, 80.0),
        ln(53.0, 0.0),
        ln(100.0, 0.0),
        ln(100.0, 100.0),
        ln(0.0, 100.0),
    ];
    let (points, n, b) = run(&req(u, true, 5.0, OffsetJoin::Miter));
    assert_eq!(n, 1);
    assert_bounds(&b, [-5.0, -5.0, 105.0, 105.0], TOL);
    // The slot (narrower than 2·5) closes completely: a plain 110 square.
    assert_close("area", evenodd_area(&points), 110.0 * 110.0, 1.0);
}

#[test]
fn miter_limit_bevels_sharp_corners() {
    // A thin wedge: a ~11° tip. Its miter ratio 1/sin(θ/2) ≈ 10.
    let wedge = vec![mv(0.0, 0.0), ln(100.0, 10.0), ln(100.0, -10.0)];
    let long = run(&req(wedge.clone(), true, 5.0, OffsetJoin::Miter)).2;
    let mut limited = req(wedge, true, 5.0, OffsetJoin::Miter);
    limited.miter_limit = 2.0;
    let short = run(&limited).2;
    // With the default limit (4) the tip is beveled too; at 20 it would reach ~-50. Both stop short of that.
    assert!(long.min_x > -60.0);
    assert!(
        short.min_x > long.min_x - 1e-6,
        "limit 2 bevels at least as early as 4"
    );
    let mut sharp = req(
        vec![mv(0.0, 0.0), ln(100.0, 10.0), ln(100.0, -10.0)],
        true,
        5.0,
        OffsetJoin::Miter,
    );
    sharp.miter_limit = 20.0;
    let spike = run(&sharp).2;
    assert!(
        spike.min_x < short.min_x - 20.0,
        "a high limit keeps the long miter: {} vs {}",
        spike.min_x,
        short.min_x
    );
}

#[test]
fn a_miter_over_the_limit_is_bevelled_not_clipped() {
    // Illustrator 30 (scripts/compare-illustrator/run.mjs --parity, offset-star-out-miter2): past the limit the
    // corner is BEVELLED — cut between the two offset edges' ends, right by the corner — not clipped at
    // `limit × distance` out along the miter.
    let wedge = vec![mv(0.0, 0.0), ln(100.0, 10.0), ln(100.0, -10.0)]; // tip ratio ≈ 10
    let tip = run(&req(wedge.clone(), true, 5.0, OffsetJoin::Miter))
        .2
        .min_x;
    // The bevel's ends sit d along each edge's normal: x = -5·sin(5.7°) ≈ -0.5 (a clip would reach ≈ -19.5).
    assert_close("bevelled tip", tip, -5.0 * 10.0 / 10_100.0_f64.sqrt(), 0.1);
    let bevel = run(&req(wedge.clone(), true, 5.0, OffsetJoin::Bevel))
        .2
        .min_x;
    assert_close("same as a bevel join", tip, bevel, 1e-6);
    // Within the limit the miter stays: a 60° star-like tip (ratio 2) at limit 4.
    let tri = vec![mv(0.0, 0.0), ln(100.0, 57.735), ln(100.0, -57.735)];
    let tip = run(&req(tri.clone(), true, 5.0, OffsetJoin::Miter)).2.min_x;
    assert_close("mitered tip", tip, -10.0, 0.1);
    // Past a lower limit the same tip bevels: the ends sit 5·sin 30° out.
    let mut limited = req(tri, true, 5.0, OffsetJoin::Miter);
    limited.miter_limit = 1.5;
    let cut = run(&limited).2.min_x;
    assert_close("bevelled at limit 1.5", cut, -2.5, 0.1);

    // Insets: an L's reflex corner, shrunk by 10. Its miter (ratio √2) cuts a 10×10 triangle more than a bevel.
    let l = vec![
        mv(0.0, 0.0),
        ln(100.0, 0.0),
        ln(100.0, 40.0),
        ln(40.0, 40.0),
        ln(40.0, 100.0),
        ln(0.0, 100.0),
    ];
    let area = |join, limit| {
        let mut r = req(l.clone(), true, -10.0, join);
        r.miter_limit = limit;
        evenodd_area(&run(&r).0)
    };
    let bevel = area(OffsetJoin::Bevel, 4.0);
    assert_close(
        "inset miter within the limit",
        area(OffsetJoin::Miter, 4.0),
        bevel - 50.0,
        0.5,
    );
    assert_close(
        "inset miter past the limit",
        area(OffsetJoin::Miter, 1.2),
        bevel,
        0.5,
    );
}

#[test]
fn open_path_miter_bevels_past_the_limit() {
    // A hairpin (ratio ≈ 10): the outer side of the turn is bevelled right by the corner.
    let hairpin = vec![mv(100.0, 10.0), ln(0.0, 0.0), ln(100.0, -10.0)];
    let b = run(&req(hairpin.clone(), false, 5.0, OffsetJoin::Miter)).2;
    let bevel = run(&req(hairpin, false, 5.0, OffsetJoin::Bevel)).2;
    assert_close("open tip", b.min_x, bevel.min_x, 1e-6);
    // A right angle keeps its miter.
    let l = vec![mv(0.0, 100.0), ln(0.0, 0.0), ln(100.0, 0.0)];
    let b = run(&req(l, false, 5.0, OffsetJoin::Miter)).2;
    assert_bounds(&b, [-5.0, -5.0, 100.0, 100.0], TOL);
}

#[test]
fn zero_distance_is_the_shape() {
    let (points, _, b) = run(&req(
        rect(0.0, 0.0, 100.0, 50.0),
        true,
        0.0,
        OffsetJoin::Miter,
    ));
    assert_bounds(&b, [0.0, 0.0, 100.0, 50.0], 1e-6);
    assert_close("area", evenodd_area(&points), 5_000.0, 1e-6);
}

#[test]
fn json_envelopes() {
    let ok: Value = crate::value::from_str(&offset_path_json(
        r#"{"points":[{"type":"move","x":0,"y":0},{"type":"line","x":10,"y":0},{"type":"line","x":10,"y":10},{"type":"line","x":0,"y":10}],
            "closed":true,"distance":2,"join":"round"}"#,
    ))
    .unwrap();
    assert_eq!(ok["status"], "ok");
    assert_eq!(ok["fillRule"], "evenodd");
    assert!((ok["bounds"]["minX"].as_f64().unwrap() + 2.0).abs() < TOL);
    let empty: Value = crate::value::from_str(&offset_path_json(
        r#"{"points":[{"type":"move","x":0,"y":0},{"type":"line","x":10,"y":0},{"type":"line","x":10,"y":10}],"closed":true,"distance":-20}"#,
    ))
    .unwrap();
    assert_eq!(empty["status"], "empty");
    for bad in [
        "nope",
        "[]",
        r#"{"points":[]}"#,
        r#"{"points":[],"distance":"x"}"#,
        r#"{"points":[{"type":"zz"}],"distance":1}"#,
        r#"{"points":[],"distance":1,"join":"sharp"}"#,
        r#"{"points":[],"distance":1e300}"#,
    ] {
        let v: Value = crate::value::from_str(&offset_path_json(bad)).unwrap();
        assert!(
            v["status"] == "error" || v["status"] == "empty",
            "{bad}: {v}"
        );
    }
}

#[test]
fn offset_fuzz_no_panic() {
    // Hostile shapes: random self-intersecting curves, repeated points,
    // zero-length segments, tiny distances. Every case must come back as an
    // envelope, never `solver_failed` (a panic, which aborts the wasm engine).
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut fails = Vec::new();
    for case in 0..150 {
        let n = 2 + (rnd() * 8.0) as usize;
        let mut pts = vec![mv(rnd() * 100.0, rnd() * 100.0)];
        for _ in 0..n {
            let (x, y) = (rnd() * 100.0, rnd() * 100.0);
            if rnd() < 0.2 {
                if let Some(BlendPathPoint::Line { x, y } | BlendPathPoint::Move { x, y }) =
                    pts.last().cloned()
                {
                    pts.push(ln(x, y));
                    continue;
                }
            }
            if rnd() < 0.4 {
                pts.push(BlendPathPoint::Curve {
                    x,
                    y,
                    cp1: (rnd() * 100.0, rnd() * 100.0),
                    cp2: (rnd() * 100.0, rnd() * 100.0),
                });
            } else {
                pts.push(ln(x, y));
            }
        }
        let d = [0.5, 3.0, -3.0, 20.0, -20.0, 1e-7][case % 6];
        let join = [OffsetJoin::Miter, OffsetJoin::Round, OffsetJoin::Bevel][case % 3];
        let r = req(pts, case % 2 == 0, d, join);
        if let Err(e) = offset_path(&r) {
            if e.code == "solver_failed" {
                fails.push((case, r.closed, d));
            }
        }
    }
    assert!(fails.is_empty(), "{} failures: {fails:?}", fails.len());
}

/// Headless pixel check: rasterise the offset and compare it with the exact
/// Minkowski sum (every pixel centre within `d` of the square). Only pixels on
/// the boundary may differ.
#[test]
fn pixels_match_the_exact_offset_region() {
    let d = 12.0;
    let (points, _, _) = run(&req(rect(0.0, 0.0, 60.0, 40.0), true, d, OffsetJoin::Round));
    let grid = Grid::around(-d, -d, 60.0 + d, 40.0 + d, 0.5, 4.0);
    let got = grid.fill_nonzero(&polygons(&points));
    let mut wrong = 0;
    let mut boundary_band = 0;
    for j in 0..grid.h {
        for i in 0..grid.w {
            let p = v2(
                grid.x0 + (i as f64 + 0.5) * grid.px,
                grid.y0 + (j as f64 + 0.5) * grid.px,
            );
            let dx = (0.0 - p.x).max(p.x - 60.0).max(0.0);
            let dy = (0.0 - p.y).max(p.y - 40.0).max(0.0);
            let dist = dx.hypot(dy);
            let want = dist <= d;
            if (dist - d).abs() < grid.px {
                boundary_band += 1;
                continue;
            }
            if got[j * grid.w + i] != want {
                wrong += 1;
            }
        }
    }
    assert!(boundary_band > 0);
    assert_eq!(wrong, 0, "{wrong} pixels off the exact offset region");
}

/// Large offsets on the shapes people offset: smooth blobs, stars, rounded
/// letters-like outlines. Never a panic; the offset contains the shape.
#[test]
fn large_offsets_on_realistic_shapes_never_panic() {
    let mut seed = 0x51f1_5eed_0bad_cafeu64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for case in 0..40 {
        // A star-shaped blob: radius varies around the centre, smooth or spiky.
        let n = 5 + (rnd() * 12.0) as usize;
        let spiky = case % 2 == 0;
        let rs: Vec<f64> = (0..n)
            .map(|i| {
                if spiky && i % 2 == 1 {
                    20.0 + rnd() * 10.0
                } else {
                    50.0 + rnd() * 30.0
                }
            })
            .collect();
        let at = |i: usize| {
            let a = i as f64 / n as f64 * 2.0 * PI;
            v2(rs[i % n] * a.cos(), rs[i % n] * a.sin())
        };
        let mut pts = vec![mv(at(0).x, at(0).y)];
        for i in 1..=n {
            let (p0, p1) = (at(i - 1), at(i));
            if spiky {
                pts.push(ln(p1.x, p1.y));
            } else {
                let (t0, t1) = (
                    (at(i) - at((i + n - 2) % n)) * 0.2,
                    (at(i + 1) - at(i - 1)) * 0.2,
                );
                pts.push(BlendPathPoint::Curve {
                    x: p1.x,
                    y: p1.y,
                    cp1: (p0.x + t0.x, p0.y + t0.y),
                    cp2: (p1.x - t1.x, p1.y - t1.y),
                });
            }
        }
        for d in [2.0, 15.0, 40.0, -5.0, -15.0] {
            for join in [OffsetJoin::Miter, OffsetJoin::Round, OffsetJoin::Bevel] {
                match offset_path(&req(pts.clone(), true, d, join)) {
                    Ok(BooleanOutcome::Shape { bounds, .. }) => {
                        if d > 0.0 {
                            assert!(bounds.max_x - bounds.min_x > 2.0 * d, "case {case} d {d}");
                        }
                    }
                    Ok(BooleanOutcome::Empty) => assert!(d < 0.0, "case {case} d {d}: empty"),
                    Err(e) => assert_ne!(
                        e.code, "solver_failed",
                        "case {case} d {d} {join:?}: {}",
                        e.message
                    ),
                }
            }
        }
    }
}

/// A mitred offset is one contour with one anchor per corner, as Illustrator makes it: the miter wedges are merged
/// into the outline. They used to stay separate subpaths meeting it along each bevel, so the copy's stroke (Offset
/// Path copies the source's) drew a bevel line across every corner.
#[test]
fn mitred_offsets_merge_their_wedges() {
    let tri = vec![mv(150.0, 50.0), ln(250.0, 250.0), ln(50.0, 250.0)];
    let mut star = Vec::new();
    for i in 0..10 {
        let r = if i % 2 == 0 { 100.0 } else { 45.0 };
        let a = (i as f64) * PI / 5.0 - PI / 2.0;
        let (x, y) = (150.0 + r * a.cos(), 150.0 + r * a.sin());
        star.push(if i == 0 { mv(x, y) } else { ln(x, y) });
    }
    let ell = vec![
        mv(0.0, 0.0),
        ln(200.0, 0.0),
        ln(200.0, 80.0),
        ln(80.0, 80.0),
        ln(80.0, 200.0),
        ln(0.0, 200.0),
    ];
    for (name, shape, anchors) in [("triangle", tri, 3), ("star", star, 10), ("L", ell, 6)] {
        for d in [3.0, 10.0, -3.0, -10.0] {
            let BooleanOutcome::Shape {
                points,
                subpath_count,
                ..
            } = offset_path(&OffsetRequest::new(shape.clone(), true, d)).unwrap()
            else {
                panic!("{name} {d}: empty");
            };
            assert_eq!(subpath_count, 1, "{name} offset {d}: {points:?}");
            // Distinct anchors (the refit may write the closing segment out).
            let mut at: Vec<(i64, i64)> = points
                .iter()
                .map(|p| match *p {
                    BlendPathPoint::Move { x, y }
                    | BlendPathPoint::Line { x, y }
                    | BlendPathPoint::Curve { x, y, .. }
                    | BlendPathPoint::Quadratic { x, y, .. }
                    | BlendPathPoint::Arc { x, y, .. } => {
                        ((x * 1e3).round() as i64, (y * 1e3).round() as i64)
                    }
                })
                .collect();
            at.sort_unstable();
            at.dedup();
            assert_eq!(at.len(), anchors, "{name} offset {d}: {points:?}");
        }
    }
}

/// A small offset stays within a small fraction of its distance: the fixed 0.1 tolerance moved a 0.1-unit square's
/// 0.005 outset by 4× the distance (its box came back 0.11 × 0.144) and a 1-unit circle's 0.05 outset by 0.04.
#[test]
fn small_offsets_are_accurate_relative_to_their_distance() {
    for (side, d) in [(0.1, 0.005), (1.0, 0.05), (1.0, -0.05), (24.0, 0.5)] {
        for shape in [
            rect(0.0, 0.0, side, side),
            circle(side / 2.0, side / 2.0, side / 2.0),
        ] {
            let BooleanOutcome::Shape { bounds, .. } =
                offset_path(&OffsetRequest::new(shape, true, d)).unwrap()
            else {
                panic!("empty offset of {side} by {d}");
            };
            for (got, want) in [
                (bounds.min_x, -d),
                (bounds.min_y, -d),
                (bounds.max_x, side + d),
                (bounds.max_y, side + d),
            ] {
                assert!(
                    (got - want).abs() <= d.abs() * 0.05,
                    "side {side} d {d}: {got} vs {want}"
                );
            }
        }
    }
    // A large circle offset by a hair still flattens within the vertex budget.
    let big = offset_path(&OffsetRequest::new(circle(0.0, 0.0, 50_000.0), true, 0.01));
    assert!(matches!(big, Ok(BooleanOutcome::Shape { .. })), "{big:?}");
}


#[test]
fn output_range_failure_refuses_the_whole_offset() {
    for sign in [-1.0, 1.0] {
        for compound in [false, true] {
            let mut points = rect(sign * 9_999_800.0, 0.0, sign * 9_999_900.0, 100.0);
            if compound {
                points.extend(rect(0.0, 0.0, 100.0, 100.0));
            }
            let result = offset_path(&req(points, true, 1000.0, OffsetJoin::Miter));
            assert_eq!(
                result
                    .expect_err("must not return empty or partial geometry")
                    .code,
                "out_of_range"
            );
        }
    }
}
