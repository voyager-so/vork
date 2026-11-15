//! Geometry tests for the live effects, against Illustrator's documented
//! behaviour (Effect ▸ Distort & Transform, Path, Stylize, Warp). The cases a
//! live Illustrator session should confirm are listed in the E8.1 PR.

use super::*;
use crate::value::json;

fn raw(v: Value) -> RawNode {
    RawNode::new(v.as_object().unwrap().clone())
}

fn square(size: f64) -> Shape {
    let pts = crate::shape_outline::rectangle_outline(size, size, None);
    contours_from_points(&pts, true)
}

fn effect(v: Value) -> Effect {
    crate::value::from_value(v).expect("effect")
}

fn run(shape: &Shape, effects: &[Value]) -> Geometry {
    let effects: Vec<Effect> = effects.iter().cloned().map(effect).collect();
    let node = raw(json!({"id": "n", "type": "path"}));
    apply_effects(
        &points_from_contours(shape),
        true,
        false,
        &effects,
        &node,
        None,
    )
}

fn anchors(shape: &Shape) -> Vec<V2> {
    shape.iter().flat_map(|c| c.anchors()).collect()
}

fn close(a: V2, b: V2, eps: f64) -> bool {
    a.dist(b) <= eps
}

/// Distance from `p` to the unit square outline scaled to `size` (axis-aligned, origin 0).
fn dist_to_square(p: V2, size: f64) -> f64 {
    let inside = p.x >= 0.0 && p.x <= size && p.y >= 0.0 && p.y <= size;
    let dx = (0.0 - p.x).max(p.x - size).max(0.0);
    let dy = (0.0 - p.y).max(p.y - size).max(0.0);
    if inside {
        p.x.min(size - p.x).min(p.y).min(size - p.y)
    } else {
        dx.hypot(dy)
    }
}

// --- conversion ---

#[test]
fn contours_round_trip_a_rectangle() {
    let s = square(100.0);
    assert_eq!(s.len(), 1);
    assert!(s[0].closed);
    // The rectangle converter ends on its start: 4 sides, no extra closing segment.
    assert_eq!(s[0].segs.len(), 4);
    assert!(s[0].segs.iter().all(|seg| seg.line));
    let back = contours_from_points(&points_from_contours(&s), true);
    assert_eq!(back, s);
}

// --- Roughen ---

fn roughen_fx(size: f64, mode: &str, detail: f64, points: &str, seed: f64) -> Value {
    json!({"id": "r", "type": "roughen", "size": size, "sizeMode": mode, "detail": detail, "points": points, "seed": seed})
}

#[test]
fn roughen_adds_detail_points_per_inch() {
    // Illustrator: Detail is anchors added per inch. A 100 px side at 10 / in gets
    // floor(100 × 10 / 72) = 13 extra anchors, so the square has 4 × 14 anchors.
    let g = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 10.0, "corner", 1.0)],
    );
    assert_eq!(anchors(&g.copies[0]).len(), 56);
    // Detail 0 moves only the original anchors.
    let g0 = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 0.0, "corner", 1.0)],
    );
    assert_eq!(anchors(&g0.copies[0]).len(), 4);
}

#[test]
fn roughen_moves_each_anchor_at_most_size() {
    for seed in 0..20 {
        let g = run(
            &square(100.0),
            &[roughen_fx(6.0, "absolute", 20.0, "corner", seed as f64)],
        );
        for p in anchors(&g.copies[0]) {
            // Each anchor moves at most `size` along x and along y.
            assert!(
                dist_to_square(p, 100.0) <= 6.0 * std::f64::consts::SQRT_2 + 1e-9,
                "{p:?}"
            );
        }
    }
}

#[test]
fn roughen_relative_size_is_a_percentage_of_the_larger_side() {
    // 200 × 100: 5 % relative = 10 px.
    let pts = crate::shape_outline::rectangle_outline(200.0, 100.0, None);
    let shape = contours_from_points(&pts, true);
    let g = run(&shape, &[roughen_fx(5.0, "relative", 0.0, "corner", 3.0)]);
    let corners = [
        v2(200.0, 0.0),
        v2(200.0, 100.0),
        v2(0.0, 100.0),
        v2(0.0, 0.0),
    ];
    let mut max_move: f64 = 0.0;
    for (a, b) in anchors(&g.copies[0]).iter().zip(corners.iter()) {
        max_move = max_move.max((a.x - b.x).abs()).max((a.y - b.y).abs());
    }
    assert!(max_move <= 10.0 + 1e-9);
    assert!(
        max_move > 2.0,
        "a 10 px roughen should move something: {max_move}"
    );
}

#[test]
fn roughen_is_seeded_and_deterministic() {
    let a = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 10.0, "smooth", 7.0)],
    );
    let b = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 10.0, "smooth", 7.0)],
    );
    let c = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 10.0, "smooth", 8.0)],
    );
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn roughen_statistics_match_a_uniform_displacement() {
    // Seeded Roughen can't be matched point for point against Illustrator; its
    // statistics can. An absolute size's offsets are uniform in
    // [-size / 2, size / 2] per axis (Illustrator: 10 px moves points at most
    // 5 px): mean ~0, mean |offset| ~ size / 4, max |offset| ≤ size / 2.
    let g = run(
        &square(1000.0),
        &[roughen_fx(10.0, "absolute", 50.0, "corner", 42.0)],
    );
    let pts = anchors(&g.copies[0]);
    // Offsets perpendicular to each side (points away from the corners).
    let mut perp = Vec::new();
    for p in &pts {
        let d_top = p.y.abs();
        let d_bottom = (p.y - 1000.0).abs();
        let d_left = p.x.abs();
        let d_right = (p.x - 1000.0).abs();
        let m = d_top.min(d_bottom).min(d_left).min(d_right);
        if p.x > 20.0 && p.x < 980.0 && d_top.min(d_bottom) == m {
            perp.push(if d_top < d_bottom { p.y } else { p.y - 1000.0 });
        }
    }
    assert!(perp.len() > 500);
    let mean = perp.iter().sum::<f64>() / perp.len() as f64;
    let mean_abs = perp.iter().map(|v| v.abs()).sum::<f64>() / perp.len() as f64;
    assert!(mean.abs() < 1.0, "mean {mean}");
    assert!((mean_abs - 2.5).abs() < 0.4, "mean |offset| {mean_abs}");
    let max_abs = perp.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        max_abs <= 5.0 + 1e-9 && max_abs > 4.5,
        "max |offset| {max_abs}"
    );
}

#[test]
fn roughen_relative_size_is_not_halved() {
    // Illustrator: relative 10 % on 200 × 50 moves points up to 20 px (10 % of
    // the larger side); only the absolute mode halves.
    let pts = crate::shape_outline::rectangle_outline(200.0, 50.0, None);
    let shape = contours_from_points(&pts, true);
    let mut max_perp: f64 = 0.0;
    for seed in 0..5 {
        let g = run(
            &shape,
            &[roughen_fx(10.0, "relative", 30.0, "corner", seed as f64)],
        );
        for p in anchors(&g.copies[0]) {
            if p.x > 30.0 && p.x < 170.0 {
                max_perp = max_perp.max(p.y.abs().min((p.y - 50.0).abs()));
            }
        }
    }
    assert!(max_perp <= 20.0 + 1e-9 && max_perp > 15.0, "{max_perp}");
}

#[test]
fn roughen_points_smooth_or_corner() {
    let smooth = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 5.0, "smooth", 1.0)],
    );
    assert!(smooth.copies[0][0].segs.iter().all(|s| !s.line));
    let corner = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 5.0, "corner", 1.0)],
    );
    assert!(corner.copies[0][0].segs.iter().all(|s| s.line));
}

// --- Zig Zag ---

fn rect(w: f64, h: f64) -> Shape {
    contours_from_points(&crate::shape_outline::rectangle_outline(w, h, None), true)
}

#[test]
fn zig_zag_relative_size_is_a_percentage_of_the_larger_side() {
    // Illustrator 30.8 (`relAmount`, absoluteness 0): 20 % of 100 × 100 → 20 px,
    // 20 % of 160 × 60 → 32 (the larger side); 10 % of 160 × 60 → 16.
    for (w, h, pct, want) in [
        (100.0, 100.0, 20.0, 20.0),
        (160.0, 60.0, 10.0, 16.0),
        (60.0, 160.0, 10.0, 16.0),
    ] {
        let fx = json!({"id": "z", "type": "zig-zag", "size": pct, "sizeMode": "relative", "ridges": 4, "points": "corner"});
        let g = run(&rect(w, h), &[fx]);
        // Ridge points away from the corners on the top and left sides sit
        // `want` off their side.
        let mut checked = 0;
        for p in anchors(&g.copies[0]) {
            if p.x.min(w - p.x) > want + 1.0 && p.y.abs() < want + 1.0 {
                assert!((p.y.abs() - want).abs() < 1e-9, "{w}×{h}: {p:?}");
                checked += 1;
            } else if p.y.min(h - p.y) > want + 1.0 && p.x.abs() < want + 1.0 {
                assert!((p.x.abs() - want).abs() < 1e-9, "{w}×{h}: {p:?}");
                checked += 1;
            }
        }
        assert!(checked > 0, "{w}×{h}");
    }
}

#[test]
fn zig_zag_smooth_handles_are_half_the_spacing() {
    // Illustrator: 100 px square, 4 ridges → points 20 px apart along each side,
    // handles 10 px along the path.
    let g = run(&square(100.0), &[zig_zag_fx(10.0, 4.0, "smooth")]);
    let c = &g.copies[0][0];
    assert_eq!(c.segs.len(), 20);
    for s in &c.segs {
        let [p0, p1, p2, p3] = s.c.0;
        assert!((p0.dist(p1) - 10.0).abs() < 1e-9, "{s:?}");
        assert!((p3.dist(p2) - 10.0).abs() < 1e-9, "{s:?}");
    }
}

fn zig_zag_fx(size: f64, ridges: f64, points: &str) -> Value {
    json!({"id": "z", "type": "zig-zag", "size": size, "sizeMode": "absolute", "ridges": ridges, "points": points})
}

#[test]
fn zig_zag_ridges_per_segment_alternate_sides() {
    // An open horizontal line, 100 px, 4 ridges: anchors every 20 px, alternately
    // 10 px above (left of travel on a y-down canvas) and below.
    let shape = contours_from_points(
        &[
            BlendPathPoint::Move { x: 0.0, y: 0.0 },
            BlendPathPoint::Line { x: 100.0, y: 0.0 },
        ],
        false,
    );
    let effects = vec![effect(zig_zag_fx(10.0, 4.0, "corner"))];
    let node = raw(json!({"id": "n", "type": "path"}));
    let g = apply_effects(
        &points_from_contours(&shape),
        false,
        false,
        &effects,
        &node,
        None,
    );
    let a = anchors(&g.copies[0]);
    assert_eq!(a.len(), 6);
    for (i, p) in a.iter().enumerate() {
        assert!((p.x - 20.0 * i as f64).abs() < 1e-9);
        let expected = if i % 2 == 0 { -10.0 } else { 10.0 };
        assert!((p.y - expected).abs() < 1e-9, "{i}: {p:?}");
    }
    assert!(g.copies[0][0].segs.iter().all(|s| s.line));
}

#[test]
fn zig_zag_square_count_and_smooth_wave() {
    let g = run(&square(100.0), &[zig_zag_fx(5.0, 3.0, "smooth")]);
    // 4 sides × (1 anchor + 3 ridges).
    assert_eq!(anchors(&g.copies[0]).len(), 16);
    let c = &g.copies[0][0];
    assert!(c.segs.iter().all(|s| !s.line));
    // Smooth ridges keep handles along the path: the first ridge on the top side
    // (a point, not a corner) has handles parallel to the side.
    for s in &c.segs {
        let [p0, p1, _, _] = s.c.0;
        let d = p1 - p0;
        if (p0.y.abs() - 5.0).abs() < 1e-9 && p0.x > 1.0 && p0.x < 99.0 {
            assert!(
                d.y.abs() < 1e-9,
                "handle leaves the side's direction: {d:?}"
            );
        }
    }
}

// --- Pucker & Bloat ---

#[test]
fn bloat_100_turns_a_square_into_a_clover() {
    let g = run(
        &square(100.0),
        &[json!({"id": "p", "type": "pucker-bloat", "amount": 100})],
    );
    let c = &g.copies[0][0];
    // Every anchor lands on the centre…
    for p in c.anchors() {
        assert!(close(p, v2(50.0, 50.0), 1e-9));
    }
    // …and each side becomes a petal reaching 1.5 × the half side past the centre,
    // pointing at the side's midpoint.
    let tips: Vec<V2> = c.segs.iter().map(|s| s.c.eval(0.5)).collect();
    let mut expected = vec![
        v2(125.0, 50.0),
        v2(50.0, 125.0),
        v2(-25.0, 50.0),
        v2(50.0, -25.0),
    ];
    for t in tips {
        let i = expected
            .iter()
            .position(|e| close(*e, t, 1e-9))
            .expect("petal tip");
        expected.remove(i);
    }
}

#[test]
fn pucker_100_turns_a_square_into_a_star() {
    let g = run(
        &square(100.0),
        &[json!({"id": "p", "type": "pucker-bloat", "amount": -100})],
    );
    let c = &g.copies[0][0];
    // Anchors move out to twice their distance from the centre; handles meet at it.
    for p in c.anchors() {
        let d = p - v2(50.0, 50.0);
        assert!((d.x.abs() - 100.0).abs() < 1e-9 && (d.y.abs() - 100.0).abs() < 1e-9);
    }
    for s in &c.segs {
        assert!(close(s.c.0[1], v2(50.0, 50.0), 1e-9));
        assert!(close(s.c.0[2], v2(50.0, 50.0), 1e-9));
        // The concave side passes inside the original square's edge.
        assert!(dist_to_square(s.c.eval(0.5), 100.0) > 0.0);
    }
}

#[test]
fn pucker_bloat_zero_keeps_the_outline() {
    let g = run(
        &square(100.0),
        &[json!({"id": "p", "type": "pucker-bloat", "amount": 0})],
    );
    let c = &g.copies[0][0];
    assert_eq!(c.anchors(), square(100.0)[0].anchors());
    for s in &c.segs {
        // Straight: both handles on the chord.
        let [p0, p1, p2, p3] = s.c.0;
        assert!((p1 - p0).cross(p3 - p0).abs() < 1e-9 && (p2 - p0).cross(p3 - p0).abs() < 1e-9);
    }
}

// --- Twist ---

#[test]
fn twist_turns_the_centre_and_pins_the_corners() {
    let g = run(
        &square(100.0),
        &[json!({"id": "t", "type": "twist", "angle": 90})],
    );
    let a = anchors(&g.copies[0]);
    for corner in [
        v2(100.0, 0.0),
        v2(100.0, 100.0),
        v2(0.0, 100.0),
        v2(0.0, 0.0),
    ] {
        assert!(
            a.iter().any(|p| close(*p, corner, 1e-9)),
            "corner {corner:?} moved"
        );
    }
    // A side's midpoint (distance 50 of a 70.7 half diagonal) turns by 90 × (1 − 50/70.7).
    let f = 90f64.to_radians() * (1.0 - 50.0 / (50.0 * std::f64::consts::SQRT_2));
    let expected = v2(50.0 + 50.0 * f.cos(), 50.0 + 50.0 * f.sin());
    assert!(
        a.iter().any(|p| close(*p, expected, 1e-6)),
        "no anchor at {expected:?}"
    );
}

// --- Transform ---

#[test]
fn transform_without_copies_moves_the_art_itself() {
    let g = run(
        &square(100.0),
        &[json!({"id": "t", "type": "transform", "moveX": 10, "moveY": -5})],
    );
    assert_eq!(g.copies.len(), 1);
    assert!(close(g.copies[0][0].start, v2(110.0, -5.0), 1e-9));
}

#[test]
fn transform_copies_keep_the_original_and_add_n() {
    let g = run(
        &square(10.0),
        &[json!({"id": "t", "type": "transform", "moveX": 20, "copies": 3})],
    );
    assert_eq!(g.copies.len(), 4);
    for (k, shape) in g.copies.iter().enumerate() {
        assert!(close(shape[0].start, v2(10.0 + 20.0 * k as f64, 0.0), 1e-9));
    }
}

#[test]
fn the_repeat_trick_closes_a_ring() {
    // Move 50 px + rotate 30°, 12 copies: 12 squares round a ring, and the 12th
    // copy (T¹²) lands back on the original, as the famous Illustrator repeat does.
    let fx = json!({"id": "t", "type": "transform", "moveX": 50, "rotate": 30, "copies": 12});
    let g = run(&square(20.0), &[fx]);
    assert_eq!(g.copies.len(), 13);
    let first = anchors(&g.copies[0]);
    let last = anchors(&g.copies[12]);
    for (a, b) in first.iter().zip(last.iter()) {
        assert!(close(*a, *b, 1e-6), "{a:?} vs {b:?}");
    }
    // The copies' centres lie on one circle.
    let centres: Vec<V2> = g.copies[..12]
        .iter()
        .map(|s| bounds_of(std::iter::once(s)).unwrap().center())
        .collect();
    let mid = centres.iter().fold(v2(0.0, 0.0), |acc, c| acc + *c) * (1.0 / 12.0);
    let r0 = centres[0].dist(mid);
    assert!(r0 > 50.0);
    for c in &centres {
        assert!((c.dist(mid) - r0).abs() < 1e-6);
    }
}

#[test]
fn transform_scale_reflect_and_reference_point() {
    // Scale 50 % about the top-left corner.
    let g = run(
        &square(100.0),
        &[
            json!({"id": "t", "type": "transform", "scaleX": 50, "scaleY": 50, "origin": "top-left"}),
        ],
    );
    let b = bounds_of(g.copies.iter()).unwrap();
    assert!(close(b.min, v2(0.0, 0.0), 1e-9) && close(b.max, v2(50.0, 50.0), 1e-9));
    // Reflect X mirrors x about the reference point (Illustrator 30.8,
    // `transform-pin0-reflectx`: scale 50 % about the top-left corner plus
    // Reflect X puts the copy LEFT of the corner).
    let r = run(
        &square(100.0),
        &[
            json!({"id": "t", "type": "transform", "reflectX": true, "scaleX": 50, "scaleY": 50, "origin": "top-left"}),
        ],
    );
    let b = bounds_of(r.copies.iter()).unwrap();
    assert!(close(b.min, v2(-50.0, 0.0), 1e-9) && close(b.max, v2(0.0, 50.0), 1e-9));
    // Reflect Y mirrors y.
    let r = run(
        &square(100.0),
        &[
            json!({"id": "t", "type": "transform", "reflectY": true, "scaleX": 50, "scaleY": 50, "origin": "top-left"}),
        ],
    );
    let b = bounds_of(r.copies.iter()).unwrap();
    assert!(close(b.min, v2(0.0, -50.0), 1e-9) && close(b.max, v2(50.0, 0.0), 1e-9));
    // Rotation is clockwise on screen (y down): 90° takes the right edge's
    // midpoint to the bottom edge's.
    let m = transform_matrix(
        &crate::value::from_value(json!({"rotate": 90})).unwrap(),
        bounds_of(std::iter::once(&square(100.0))),
    );
    assert!(close(
        apply_affine(&m, v2(100.0, 50.0)),
        v2(50.0, 100.0),
        1e-9
    ));
}

// --- Round Corners, Offset Path, Outline Stroke ---

#[test]
fn round_corners_fillets_every_corner() {
    let g = run(
        &square(100.0),
        &[json!({"id": "r", "type": "round-corners", "radius": 10})],
    );
    let c = &g.copies[0][0];
    assert_eq!(c.segs.iter().filter(|s| !s.line).count(), 4);
    let b = bounds_of(g.copies.iter()).unwrap();
    assert!(close(b.min, v2(0.0, 0.0), 1e-6) && close(b.max, v2(100.0, 100.0), 1e-6));
    // A corner is cut: nothing lands on (0, 0) any more.
    assert!(anchors(&g.copies[0])
        .iter()
        .all(|p| p.dist(v2(0.0, 0.0)) > 5.0));
}

fn round_fx(r: f64) -> Value {
    json!({"id": "r", "type": "round-corners", "radius": r})
}

/// The corner joins of a rounded contour: (from anchor, handle out, handle in, to anchor).
fn joins(c: &Contour) -> Vec<[V2; 4]> {
    c.segs.iter().filter(|s| !s.line).map(|s| s.c.0).collect()
}

#[test]
fn round_corners_sets_anchors_back_by_the_radius_on_a_square() {
    // Illustrator 30.8, 100 px square (70..170), r = 30: anchors 30 from each
    // corner, handles 16.5 pointing at the corner.
    let pts = crate::shape_outline::rectangle_outline(100.0, 100.0, None);
    let shape: Shape = contours_from_points(&pts, true)
        .iter()
        .map(|c| Contour {
            start: c.start + v2(70.0, 70.0),
            segs: c
                .segs
                .iter()
                .map(|s| Seg::line(s.start() + v2(70.0, 70.0), s.end() + v2(70.0, 70.0)))
                .collect(),
            closed: true,
        })
        .collect();
    let g = run(&shape, &[round_fx(30.0)]);
    let c = &g.copies[0][0];
    assert_eq!(c.segs.len(), 8);
    let mut got: Vec<(i64, i64)> = anchors(&g.copies[0])
        .iter()
        .map(|p| (p.x.round() as i64, p.y.round() as i64))
        .collect();
    got.sort();
    let mut want = vec![
        (70, 100),
        (100, 70),
        (140, 70),
        (170, 100),
        (170, 140),
        (140, 170),
        (100, 170),
        (70, 140),
    ];
    want.sort();
    assert_eq!(got, want);
    let corners = [
        v2(70.0, 70.0),
        v2(170.0, 70.0),
        v2(170.0, 170.0),
        v2(70.0, 170.0),
    ];
    let js = joins(c);
    assert_eq!(js.len(), 4);
    for [a, h1, h2, b] in js {
        let k = *corners
            .iter()
            .min_by(|p, q| p.dist(a).total_cmp(&q.dist(a)))
            .unwrap();
        assert!((a.dist(k) - 30.0).abs() < 1e-9 && (b.dist(k) - 30.0).abs() < 1e-9);
        assert!((h1.dist(a) - 16.5).abs() < 1e-9 && (h2.dist(b) - 16.5).abs() < 1e-9);
        // Handles point at the corner.
        assert!(((k - a).unit().unwrap().dot((h1 - a).unit().unwrap()) - 1.0).abs() < 1e-9);
        assert!(((k - b).unit().unwrap().dot((h2 - b).unit().unwrap()) - 1.0).abs() < 1e-9);
    }
}

fn star(cx: f64, cy: f64, outer: f64, inner: f64, n: usize) -> Shape {
    let pts: Vec<BlendPathPoint> = (0..2 * n)
        .map(|i| {
            let r = if i % 2 == 0 { outer } else { inner };
            let a = (-90.0 + 180.0 * i as f64 / n as f64).to_radians();
            let (x, y) = (cx + r * a.cos(), cy + r * a.sin());
            if i == 0 {
                BlendPathPoint::Move { x, y }
            } else {
                BlendPathPoint::Line { x, y }
            }
        })
        .collect();
    contours_from_points(&pts, true)
}

#[test]
fn round_corners_matches_illustrator_on_a_star() {
    // Illustrator 30.8 golden (fx-round-corners-12-star): 5-point star, centre
    // (120, 125), R 70, r 30, r = 12 → every corner (tips and concave) becomes
    // anchors 12 along each edge with 6.6 handles. Top tip's anchors:
    // (115.683, 66.196) and (124.317, 66.196), handles (118.057, 60.038) and
    // (121.943, 60.038).
    let g = run(&star(120.0, 125.0, 70.0, 30.0, 5), &[round_fx(12.0)]);
    let c = &g.copies[0][0];
    let js = joins(c);
    assert_eq!(js.len(), 10);
    assert_eq!(anchors(&g.copies[0]).len(), 20);
    let tip = js
        .iter()
        .find(|j| j[0].dist(v2(124.317, 66.196)) < 0.01 || j[0].dist(v2(115.683, 66.196)) < 0.01)
        .expect("top tip join");
    let (mut ends, mut handles) = (vec![tip[0], tip[3]], vec![tip[1], tip[2]]);
    ends.sort_by(|a, b| a.x.total_cmp(&b.x));
    handles.sort_by(|a, b| a.x.total_cmp(&b.x));
    assert!(close(ends[0], v2(115.683, 66.196), 2e-3), "{ends:?}");
    assert!(close(ends[1], v2(124.317, 66.196), 2e-3), "{ends:?}");
    assert!(close(handles[0], v2(118.057, 60.038), 2e-3), "{handles:?}");
    assert!(close(handles[1], v2(121.943, 60.038), 2e-3), "{handles:?}");
    for [a, h1, h2, b] in &js {
        assert!((h1.dist(*a) - 6.6).abs() < 1e-9 && (h2.dist(*b) - 6.6).abs() < 1e-9);
    }
    // A concave corner: (133.316, 89.533) → (149.616, 101.376) about (137.63, 100.73).
    assert!(
        js.iter()
            .any(|j| close(j[0], v2(133.316, 89.533), 2e-3)
                && close(j[3], v2(149.616, 101.376), 2e-3))
    );
}

#[test]
fn round_corners_clamps_to_half_a_segment_and_skips_smooth_anchors() {
    // A 20 px square with r = 30: setback clamps to 10, the sides vanish, four
    // joins remain.
    let g = run(&square(20.0), &[round_fx(30.0)]);
    let c = &g.copies[0][0];
    assert!(c.segs.iter().all(|s| !s.line));
    assert_eq!(c.segs.len(), 4);
    for [a, h1, _, _] in joins(c) {
        assert!((h1.dist(a) - 5.5).abs() < 1e-9);
    }
    // An ellipse has no corners: unchanged.
    let e = contours_from_points(
        &crate::shape_outline::ellipse_outline(100.0, 60.0, None, None, None),
        true,
    );
    let g = run(&e, &[round_fx(10.0)]);
    assert_eq!(g.copies[0], e);
}

#[test]
fn offset_path_grows_by_the_offset_and_fills_even_odd() {
    let g = run(
        &square(100.0),
        &[json!({"id": "o", "type": "offset-path", "offset": 10, "join": "miter"})],
    );
    assert!(g.even_odd);
    let b = bounds_of(g.copies.iter()).unwrap();
    assert!(
        close(b.min, v2(-10.0, -10.0), 0.1) && close(b.max, v2(110.0, 110.0), 0.1),
        "{b:?}"
    );
    // A shrink past half the width erases it.
    let gone = run(
        &square(100.0),
        &[json!({"id": "o", "type": "offset-path", "offset": -60})],
    );
    assert!(gone.copies[0].is_empty());
}

#[test]
fn outline_stroke_outlines_the_stroke_as_a_closed_shape() {
    let node = raw(json!({
        "id": "n", "type": "path",
        "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10}],
        "strokeCap": "butt"
    }));
    let pts = [
        BlendPathPoint::Move { x: 0.0, y: 0.0 },
        BlendPathPoint::Line { x: 100.0, y: 0.0 },
    ];
    let effects = vec![effect(json!({"id": "o", "type": "outline-stroke"}))];
    let g = apply_effects(&pts, false, false, &effects, &node, None);
    assert!(g.closed);
    assert!(!g.even_odd);
    let b = bounds_of(g.copies.iter()).unwrap();
    assert!((b.height() - 10.0).abs() < 0.2, "{b:?}");
    assert!((b.width() - 100.0).abs() < 0.2, "{b:?}");
    // No stroke: the outline is left as it is.
    let bare = raw(json!({"id": "n", "type": "path"}));
    let g2 = apply_effects(&pts, false, false, &effects, &bare, None);
    assert!(!g2.closed);
}

// --- Warp ---

#[test]
fn every_warp_is_the_identity_at_bend_zero() {
    for style in WARP_STYLES {
        let g = run(
            &square(100.0),
            &[json!({"id": "w", "type": "warp", "style": style, "bend": 0})],
        );
        for s in &g.copies[0] {
            for seg in &s.segs {
                assert!(dist_to_square(seg.c.eval(0.5), 100.0) < 1e-9, "{style}");
            }
        }
    }
}

#[test]
fn arc_bend_100_bends_the_bottom_edge_into_a_half_circle() {
    // Illustrator's Arc is a fan: the bottom edge keeps its chord and bends by
    // bend × 180°, the rows above are concentric. At 100 % a 200 × 2 bar's
    // bottom is a half circle of radius 100, its top one of radius 102.
    let pts = crate::shape_outline::rectangle_outline(200.0, 2.0, None);
    let shape = contours_from_points(&pts, true);
    let g = run(
        &shape,
        &[json!({"id": "w", "type": "warp", "style": "arc", "bend": 100})],
    );
    let b = bounds_of(g.copies.iter()).unwrap();
    assert!((b.width() - 204.0).abs() < 0.5, "{b:?}");
    assert!((b.height() - 102.0).abs() < 0.5, "{b:?}");
}

#[test]
fn arch_keeps_the_sides_vertical_and_lifts_the_middle() {
    let g = run(
        &square(100.0),
        &[json!({"id": "w", "type": "warp", "style": "arch", "bend": 50})],
    );
    for s in &g.copies[0] {
        for seg in &s.segs {
            for i in 0..=8 {
                let p = seg.c.eval(i as f64 / 8.0);
                // The side edges stay at x = 0 and x = 100.
                assert!(p.x >= -1e-6 && p.x <= 100.0 + 1e-6);
            }
        }
    }
    let b = bounds_of(g.copies.iter()).unwrap();
    // The top edge is a 90° circular arc on its 100 px chord: its middle
    // rises 50·(√2 − 1) = 20.7 px.
    assert!((b.min.y + 20.71).abs() < 0.1, "{b:?}");
}

#[test]
fn bulge_swells_top_and_bottom_symmetrically() {
    let g = run(
        &square(100.0),
        &[json!({"id": "w", "type": "warp", "style": "bulge", "bend": 50})],
    );
    let b = bounds_of(g.copies.iter()).unwrap();
    assert!(
        (b.min.y + 20.71).abs() < 0.1 && (b.max.y - 120.71).abs() < 0.1,
        "{b:?}"
    );
    assert!(b.min.x.abs() < 1e-6 && (b.max.x - 100.0).abs() < 1e-6);
}

#[test]
fn vertical_orientation_transposes_the_warp() {
    let h = run(
        &square(100.0),
        &[json!({"id": "w", "type": "warp", "style": "arch", "bend": 50})],
    );
    let v = run(
        &square(100.0),
        &[
            json!({"id": "w", "type": "warp", "style": "arch", "bend": 50, "orientation": "vertical"}),
        ],
    );
    let bh = bounds_of(h.copies.iter()).unwrap();
    let bv = bounds_of(v.copies.iter()).unwrap();
    assert!((bh.height() - bv.width()).abs() < 1e-6 && (bh.width() - bv.height()).abs() < 1e-6);
}

#[test]
fn warp_distortion_tapers_the_art() {
    let g = run(
        &square(100.0),
        &[json!({"id": "w", "type": "warp", "style": "arc", "bend": 0, "horizontal": 50})],
    );
    // Horizontal distortion 50 %: the right edge 1.5× as tall, the left 0.5×.
    let a = anchors(&g.copies[0]);
    let right: Vec<f64> = a
        .iter()
        .filter(|p| (p.x - 100.0).abs() < 1e-6)
        .map(|p| p.y)
        .collect();
    let span = right.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - right.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!((span - 150.0).abs() < 1e-6, "{span}");
}

/// Points along a closed outline: `n` per segment.
fn outline_samples(shape: &Shape, n: usize) -> Vec<V2> {
    let mut out = Vec::new();
    for c in shape {
        for s in &c.segs {
            for i in 0..n {
                out.push(s.c.eval(i as f64 / n as f64));
            }
        }
    }
    out
}

fn dist_to_polygon(p: V2, poly: &[V2]) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let ab = b - a;
        let t = if ab.len2() > 0.0 {
            ((p - a).dot(ab) / ab.len2()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        best = best.min(p.dist(a + ab * t));
    }
    best
}

/// Every Warp preset against Illustrator's expanded output (Effect ▸ Warp on a
/// 200 × 100 rectangle, plus distortion, vertical and non-rectangular cases):
/// the outlines agree within 0.5 px (symmetric Hausdorff of dense samples).
#[test]
fn warp_presets_match_illustrator() {
    let fixture: Value =
        crate::value::from_str(include_str!("../tests/fixtures/illustrator_warp.json")).unwrap();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let pt = |v: &Value| v2(v[0].as_f64().unwrap(), v[1].as_f64().unwrap());
        let src: Vec<V2> = case["source"].as_array().unwrap().iter().map(pt).collect();
        let segs = (0..src.len())
            .map(|i| Seg::line(src[i], src[(i + 1) % src.len()]))
            .collect();
        let shape: Shape = vec![Contour {
            start: src[0],
            segs,
            closed: true,
        }];
        let mut fx = case["effect"].clone();
        fx["id"] = json!("w");
        let g = run(&shape, &[fx]);
        let ours = outline_samples(&g.copies[0], 16);
        // Illustrator's anchors [x, y, inX, inY, outX, outY] → cubics.
        let ai: Vec<Vec<f64>> = case["illustrator"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                q.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect()
            })
            .collect();
        let mut theirs = Vec::new();
        for i in 0..ai.len() {
            let (a, b) = (&ai[i], &ai[(i + 1) % ai.len()]);
            let c = Cubic([
                v2(a[0], a[1]),
                v2(a[4], a[5]),
                v2(b[2], b[3]),
                v2(b[0], b[1]),
            ]);
            for s in 0..64 {
                theirs.push(c.eval(s as f64 / 64.0));
            }
        }
        let d1 = ours
            .iter()
            .map(|&p| dist_to_polygon(p, &theirs))
            .fold(0.0, f64::max);
        let d2 = theirs
            .iter()
            .map(|&p| dist_to_polygon(p, &ours))
            .fold(0.0, f64::max);
        let d = d1.max(d2);
        if d > 0.5 {
            failures.push(format!("{name}: {d:.2} px"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

// --- The pipeline ---

#[test]
fn effects_apply_in_list_order() {
    // Transform (move 100) then Round Corners, vs the other way round: same bounds,
    // but Roughen then Transform (with copies) copies the same roughened art.
    let g = run(
        &square(50.0),
        &[
            roughen_fx(5.0, "absolute", 5.0, "corner", 9.0),
            json!({"id": "t", "type": "transform", "moveX": 100, "copies": 1}),
        ],
    );
    let a0 = anchors(&g.copies[0]);
    let a1 = anchors(&g.copies[1]);
    for (p, q) in a0.iter().zip(a1.iter()) {
        assert!(close(*p + v2(100.0, 0.0), *q, 1e-9));
    }
}

#[test]
fn resolve_node_turns_a_shape_into_an_effected_path() {
    let node = raw(json!({
        "id": "r1", "type": "rectangle", "x": 10, "y": 20, "width": 100, "height": 60,
        "cornerRadius": 8, "fills": [{"id": "f", "type": "solid", "color": "#f00"}],
        "effects": [
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 4, "offset": {"x": 2, "y": 2}},
            {"id": "z", "type": "zig-zag", "size": 4, "sizeMode": "absolute", "ridges": 2, "points": "corner"}
        ]
    }));
    let r = resolve_node(&node, None).expect("resolved");
    assert_eq!(r.node_type(), "path");
    assert_eq!(r.f64("x"), Some(10.0));
    assert_eq!(r.f64("width"), Some(100.0));
    assert!(r.json_field("cornerRadius").is_none());
    assert_eq!(r.bool("closed"), Some(true));
    let effects = r.json_field("effects").unwrap().as_array().unwrap();
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0]["type"], json!("drop-shadow"));
    assert!(r.array("points").unwrap().len() > 10);
    assert!(r.json_field(EFFECT_COPIES_KEY).is_none());
}

#[test]
fn resolve_node_skips_hidden_effects_and_unsupported_types() {
    let hidden = raw(json!({
        "id": "r", "type": "rectangle", "width": 10, "height": 10,
        "effects": [{"id": "t", "type": "twist", "angle": 30, "visible": false}]
    }));
    assert!(resolve_node(&hidden, None).is_none());
    let text = raw(json!({
        "id": "t", "type": "richtext", "width": 10, "height": 10,
        "effects": [{"id": "t", "type": "twist", "angle": 30}]
    }));
    assert!(resolve_node(&text, None).is_none());
}

#[test]
fn resolved_copies_split_for_the_body_lane() {
    let node = raw(json!({
        "id": "e", "type": "ellipse", "width": 20, "height": 20,
        "effects": [{"id": "t", "type": "transform", "moveX": 30, "copies": 2}]
    }));
    let r = resolve_node(&node, None).unwrap();
    let parts = split_copies(&r).expect("three copies");
    assert_eq!(parts.len(), 3);
    let total: usize = parts.iter().map(|p| p.array("points").unwrap().len()).sum();
    assert_eq!(total, r.array("points").unwrap().len());
    assert!(parts.iter().all(|p| p.json_field(EFFECT_COPIES_KEY).is_none()));
}

#[test]
fn every_shape_type_resolves() {
    for (ty, extra) in [
        ("rectangle", json!({})),
        ("ellipse", json!({})),
        ("star", json!({"numPoints": 5, "innerRadiusRatio": 0.5})),
        ("polygon", json!({"sides": 6})),
        (
            "path",
            json!({"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 40, "y": 40}], "closed": false}),
        ),
    ] {
        let mut v = json!({"id": "n", "type": ty, "width": 40, "height": 40,
            "effects": [{"id": "p", "type": "pucker-bloat", "amount": 40}]});
        for (k, val) in extra.as_object().unwrap() {
            v[k] = val.clone();
        }
        let r = resolve_node(&raw(v), None).unwrap_or_else(|| panic!("{ty}"));
        assert!(r.array("points").unwrap().len() >= 2, "{ty}");
    }
}

#[test]
fn runaway_effects_are_skipped_not_hung() {
    // 1000 copies of a 100-detail roughen stays under the segment cap or is dropped.
    let g = run(
        &square(1000.0),
        &[
            roughen_fx(5.0, "absolute", 100.0, "smooth", 1.0),
            json!({"id": "t", "type": "transform", "moveX": 1, "copies": 1000}),
        ],
    );
    assert!(g.segment_count() <= MAX_SEGMENTS);
}

#[test]
fn a_roughen_or_zig_zag_past_the_cap_is_refused_before_it_samples() {
    // A 6e10 px side at 10 points per inch asked for 8e9 samples: an 8 GB allocation that aborted
    // (docs/qa/robustness.md). The effect is skipped, as one past the cap after the fact is.
    let huge = square(6e10);
    let g = run(&huge, &[roughen_fx(5.0, "absolute", 10.0, "smooth", 1.0)]);
    assert_eq!(
        g.segment_count(),
        huge.iter().map(|c| c.segs.len()).sum::<usize>()
    );
    // 100 ridges on each of 3,000 segments is 303,000 points, past the 250,000 cap.
    let pts: Vec<BlendPathPoint> = (0..3000)
        .filter_map(|i| {
            BlendPathPoint::from_json(
                &json!({"type": if i == 0 { "move" } else { "line" }, "x": i, "y": i % 2}),
            )
        })
        .collect();
    let wavy = contours_from_points(&pts, false);
    let g = run(
        &wavy,
        &[json!({"id": "z", "type": "zig-zag", "size": 1, "ridges": 100})],
    );
    assert_eq!(g.segment_count(), 3000, "the closed outline, unchanged");
    // Under the cap it still applies.
    let g = run(
        &square(100.0),
        &[roughen_fx(5.0, "absolute", 10.0, "smooth", 1.0)],
    );
    assert!(g.segment_count() > 4);
}

#[path = "live_effects_distort_tests.rs"]
mod distort_tests;

#[test]
fn json_entry_point() {
    let req = json!({"node": {
        "id": "r", "type": "rectangle", "width": 10, "height": 10,
        "effects": [{"id": "t", "type": "transform", "moveX": 20, "copies": 1}]
    }});
    let out: Value = crate::value::from_str(&geometry_effects_json(&req.to_string())).unwrap();
    assert_eq!(out["status"], json!("ok"));
    assert_eq!(out["copies"].as_array().unwrap().len(), 2);
    assert_eq!(out["node"]["type"], json!("path"));
    assert!((out["bounds"]["maxX"].as_f64().unwrap() - 30.0).abs() < 1e-9);
    let none: Value = crate::value::from_str(&geometry_effects_json(
        &json!({"node": {"id": "r", "type": "rectangle", "width": 1, "height": 1}}).to_string(),
    ))
    .unwrap();
    assert_eq!(none["status"], json!("none"));
    let bad: Value = crate::value::from_str(&geometry_effects_json("{")).unwrap();
    assert_eq!(bad["status"], json!("error"));
}

fn offset_fx(offset: f64, join: &str) -> Value {
    json!({"id": "o", "type": "offset-path", "offset": offset, "join": join})
}

/// Largest distance from a point of `a`'s outline (sampled) to `b`'s outline (as chords, 32 per segment).
fn outline_gap(a: &Shape, b: &Shape) -> f64 {
    let samples = |s: &Shape, n: usize| -> Vec<(V2, V2)> {
        s.iter()
            .flat_map(|c| c.segs.iter())
            .flat_map(|seg| {
                (0..n).map(move |t| {
                    (
                        seg.c.eval(t as f64 / n as f64),
                        seg.c.eval((t + 1) as f64 / n as f64),
                    )
                })
            })
            .collect()
    };
    let to_chord = |p: V2, (a, b): (V2, V2)| {
        let (ab, ap) = (v2(b.x - a.x, b.y - a.y), v2(p.x - a.x, p.y - a.y));
        let len2 = ab.x * ab.x + ab.y * ab.y;
        let t = if len2 > 0.0 {
            ((ap.x * ab.x + ap.y * ab.y) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        p.dist(v2(a.x + t * ab.x, a.y + t * ab.y))
    };
    let chords = samples(b, 32);
    samples(a, 8)
        .iter()
        .map(|&(p, _)| {
            chords
                .iter()
                .map(|&c| to_chord(p, c))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

#[test]
fn consecutive_round_offsets_are_one_offset_by_their_sum() {
    // Grow, and shrink: a run of round offsets of one sign is offset once, by the sum.
    for (step, n) in [(4.0, 5), (-3.0, 4)] {
        let run_of: Vec<Value> = (0..n).map(|_| offset_fx(step, "round")).collect();
        let merged = run(&square(100.0), &run_of);
        let once = run(&square(100.0), &[offset_fx(step * n as f64, "round")]);
        assert_eq!(merged.copies, once.copies, "{n} × {step}");
        assert!(merged.even_odd && merged.closed);
        // The step-by-step result, each offset on the last one's outline, is the same shape within the
        // offsetter's tolerance per step (the merged one is closer to the exact offset: one tolerance, not n).
        let mut g = contours_from_points(&points_from_contours(&square(100.0)), true);
        let mut even_odd = false;
        for _ in 0..n {
            let o = OffsetPathEffect {
                offset: step,
                join: Some("round".into()),
                miter_limit: None,
            };
            g = offset(&g, &o, true, even_odd).unwrap();
            even_odd = true;
        }
        let gap = outline_gap(&merged.copies[0], &g).max(outline_gap(&g, &merged.copies[0]));
        assert!(
            gap < 0.1 * n as f64,
            "{n} × {step}: step-by-step result {gap} px away"
        );
    }
}

#[test]
fn offsets_that_do_not_compose_are_not_merged() {
    let one_by_one = |effects: &[Value]| {
        let mut g = run(&square(100.0), &effects[..1]);
        for e in &effects[1..] {
            g = apply_effects(
                &points_from_contours(&g.copies[0]),
                g.closed,
                g.even_odd,
                &[effect(e.clone())],
                &raw(json!({"id": "n", "type": "path"})),
                None,
            );
        }
        g.copies
    };
    for effects in [
        // Miter and bevel joins: a corner's miter or bevel is not the sum of two.
        vec![offset_fx(4.0, "miter"), offset_fx(4.0, "miter")],
        vec![offset_fx(4.0, "round"), offset_fx(4.0, "bevel")],
        // Opposite signs: a grow then a shrink rounds concave corners off (a closing), not a no-op.
        vec![offset_fx(4.0, "round"), offset_fx(-4.0, "round")],
    ] {
        assert_eq!(
            run(&square(100.0), &effects).copies,
            one_by_one(&effects),
            "{effects:?}"
        );
    }
    // An open path's first offset is a band on both sides: a shrink after it is not a longer shrink.
    let o = |d: f64| OffsetPathEffect {
        offset: d,
        join: Some("round".into()),
        miter_limit: None,
    };
    let rest = [effect(offset_fx(-2.0, "round"))];
    assert_eq!(merged_offsets(&o(-4.0), &rest, false).1, 1);
    assert_eq!(merged_offsets(&o(-4.0), &rest, true).1, 2);
    let grow = [effect(offset_fx(2.0, "round"))];
    assert_eq!(merged_offsets(&o(4.0), &grow, false), (o(6.0), 2));
    // The sum stays within the offsetter's limit.
    let huge = [effect(offset_fx(60_000.0, "round"))];
    assert_eq!(merged_offsets(&o(60_000.0), &huge, true).1, 1);
}

#[test]
fn a_stack_past_the_cap_stops_there_and_says_so() {
    let twist = json!({"id": "t", "type": "twist", "angle": 10});
    let at_cap = run(&square(100.0), &vec![twist.clone(); MAX_GEOMETRY_EFFECTS]);
    assert!(at_cap.warnings.is_empty());
    let over = run(&square(100.0), &vec![twist; MAX_GEOMETRY_EFFECTS + 5]);
    assert_eq!(over.copies, at_cap.copies);
    assert_eq!(
        over.warnings,
        vec![StackWarning::TooManyEffects {
            evaluated: MAX_GEOMETRY_EFFECTS,
            total: MAX_GEOMETRY_EFFECTS + 5
        }]
    );
    let node = raw(json!({"id": "n", "type": "rectangle", "width": 10, "height": 10}));
    let m = resolved_map(&node, &over);
    assert_eq!(
        m[EFFECT_WARNINGS_KEY][0]["code"],
        json!("effect_stack_truncated")
    );
    assert!(!resolved_map(&node, &at_cap).contains_key(EFFECT_WARNINGS_KEY));
}

#[test]
fn an_effect_past_max_segments_stops_the_stack() {
    // 1,000 copies of 1,000 copies of a square: the second Transform would make 4 million segments. It is refused
    // before its copies are made, and the Round Corners after it is not evaluated.
    let copies = json!({"id": "t", "type": "transform", "moveX": 1, "copies": 1000});
    let g = run(
        &square(100.0),
        &[
            copies.clone(),
            copies,
            json!({"id": "r", "type": "round-corners", "radius": 4}),
        ],
    );
    assert_eq!(g.copies.len(), 1001);
    assert_eq!(
        g.warnings,
        vec![StackWarning::TooComplex { at: 1, total: 3 }]
    );
}

#[test]
fn a_stack_past_its_segment_budget_stops_there() {
    // Each Transform keeps the art under MAX_SEGMENTS (12 × 1,001 squares: 48,048 segments), but the stack remakes
    // it until the budget is spent.
    let mut effects = vec![
        json!({"id": "t", "type": "transform", "moveX": 1, "copies": 1000}),
        json!({"id": "t", "type": "transform", "moveY": 1, "copies": 11}),
    ];
    effects.extend(vec![
        json!({"id": "t", "type": "transform", "moveX": -1});
        40
    ]);
    let g = run(&square(100.0), &effects);
    let per = g.segment_count();
    assert_eq!(per, 48_048);
    let at = match g.warnings.as_slice() {
        [StackWarning::OverBudget { at, total: 42 }] => *at,
        other => panic!("{other:?}"),
    };
    // `made` counts every result: 4,004 for the first effect, then 48,048 for each after it.
    let made = |n: usize| 4_004 + (n - 1) * per;
    assert!(
        made(at) > MAX_STACK_SEGMENTS && made(at - 1) <= MAX_STACK_SEGMENTS,
        "{at}"
    );
}

#[test]
fn json_entry_point_reports_a_stopped_stack() {
    let req = json!({"node": {
        "id": "r", "type": "rectangle", "width": 10, "height": 10,
        "effects": vec![json!({"id": "t", "type": "twist", "angle": 5}); MAX_GEOMETRY_EFFECTS + 1]
    }});
    let out: Value = crate::value::from_str(&geometry_effects_json(&req.to_string())).unwrap();
    assert_eq!(out["status"], json!("ok"));
    assert_eq!(out["warnings"][0]["code"], json!("effect_stack_truncated"));
    assert_eq!(out["warnings"][0]["total"], json!(MAX_GEOMETRY_EFFECTS + 1));
    assert!(out["node"].get(EFFECT_WARNINGS_KEY).is_none());
    // No key at all when the whole stack ran.
    let req = json!({"node": {
        "id": "r", "type": "rectangle", "width": 10, "height": 10,
        "effects": [{"id": "t", "type": "twist", "angle": 5}]
    }});
    let out: Value = crate::value::from_str(&geometry_effects_json(&req.to_string())).unwrap();
    assert!(out.get("warnings").is_none());
}
