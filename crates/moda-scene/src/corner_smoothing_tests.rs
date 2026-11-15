use super::*;

const EPS: f64 = 1e-9;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn pts_close(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
    close(a.0, b.0, tol) && close(a.1, b.1, tol)
}

/// Reference distances for R = 100 from the article's construction, worked by
/// hand (independently of `corner_params`):
///
/// ξ = 0.6 (the iOS-like setting): p = 160, arc measure 36°,
///   L = sin 18° · 100 · √2 = 43.7016, α = 27°, |P3P4| = 100 tan 13.5° = 24.0079,
///   β = 27°, c = 24.0079 cos 27° = 21.3912, d = c tan 27° = 10.8993,
///   b = (160 − 43.7016 − 21.3912 − 10.8993) / 3 = 28.0026, a = 56.0053.
/// ξ = 1: p = 200, no arc, α = 45°, c = d = 100 tan 22.5° cos 45° = 29.2893,
///   b = (200 − 58.5786) / 3 = 47.1405, a = 94.2809.
#[test]
fn article_reference_distances() {
    let c = corner_params(100.0, 0.6, 1000.0);
    assert!(close(c.p, 160.0, 1e-9));
    assert!(close(c.arc_measure_deg, 36.0, 1e-9));
    assert!(close(c.arc_section_length, 43.7016, 1e-4));
    assert!(close(c.c, 21.3912, 1e-4));
    assert!(close(c.d, 10.8993, 1e-4));
    assert!(close(c.b, 28.0026, 1e-4));
    assert!(close(c.a, 56.0053, 1e-4));
    // "two thirds of the interval to a and the remaining third to b".
    assert!(close(c.a, 2.0 * c.b, EPS));

    let f = corner_params(100.0, 1.0, 1000.0);
    assert!(close(f.p, 200.0, EPS));
    assert!(close(f.arc_section_length, 0.0, EPS));
    assert!(close(f.c, 29.2893, 1e-4));
    assert!(close(f.d, 29.2893, 1e-4));
    assert!(close(f.b, 47.1405, 1e-4));
    assert!(close(f.a, 94.2809, 1e-4));

    // ξ = 0 is a plain circular corner: p = R, the arc is the whole quarter.
    let z = corner_params(100.0, 0.0, 1000.0);
    assert!(close(z.p, 100.0, EPS));
    assert!(close(z.arc_section_length, 100.0, 1e-9));
    assert!(close(z.a + z.b + z.c + z.d, 0.0, 1e-9));
}

/// Reference points on the top-right corner of a 400×400 square, R = 100,
/// ξ = 0.6 (corner at (400, 0); arc centre at (300, 100)).
#[test]
fn article_reference_points_on_the_outline() {
    let segs = smoothed_rect_path(400.0, 400.0, [100.0; 4], 0.6);
    // P1: the corner leaves the top edge p = 160 before the corner.
    assert_eq!(segs[0], Seg::Move(240.0, 0.0));
    let Seg::Cubic { cp1, cp2, to } = segs[1] else {
        panic!("edge→arc cubic")
    };
    assert!(pts_close(cp1, (296.0053, 0.0), 1e-4)); // P2 = P1 + a
    assert!(pts_close(cp2, (324.0079, 0.0), 1e-4)); // P3 = P1 + a + b
    assert!(pts_close(to, (345.3991, 10.8993), 1e-4)); // P4 = P3 + (c, d)
                                                       // P4 lies on the radius-100 circle, 27° (= 45°·ξ) past the top of it.
    let ang = (to.0 - 300.0).atan2(100.0 - to.1).to_degrees();
    assert!(close(
        ((to.0 - 300.0).powi(2) + (to.1 - 100.0).powi(2)).sqrt(),
        100.0,
        1e-9
    ));
    assert!(close(ang, 27.0, 1e-9));
    // The arc ends symmetrically at 63°, and the corner ends p = 160 down the right edge.
    let Seg::Cubic { to: e2, .. } = segs[2] else {
        panic!("arc")
    };
    assert!(pts_close(e2, (389.1007, 54.6009), 1e-4));
    let Seg::Cubic { to: end, .. } = segs[3] else {
        panic!("arc→edge cubic")
    };
    assert!(pts_close(end, (400.0, 160.0), 1e-9));
    // The 45° midpoint of the corner sits on the circle: (300 + 100/√2, 100 − 100/√2).
    let mid = cubic_at(
        to,
        match segs[2] {
            Seg::Cubic { cp1, .. } => cp1,
            _ => unreachable!(),
        },
        match segs[2] {
            Seg::Cubic { cp2, .. } => cp2,
            _ => unreachable!(),
        },
        e2,
        0.5,
    );
    assert!(pts_close(mid, (370.7107, 29.2893), 1e-3));
}

/// G1 at both junctions, and the curvature of the edge→arc cubic runs from 0
/// on the edge up to the order of 1/R where it meets the arc.
#[test]
fn tangent_and_curvature_continuity() {
    for &s in &[0.2, 0.6, 0.9] {
        let segs = smoothed_rect_path(1000.0, 1000.0, [100.0; 4], s);
        let Seg::Move(x0, y0) = segs[0] else { panic!() };
        let Seg::Cubic { cp1, cp2, to } = segs[1] else {
            panic!()
        };
        // Start: P1, P2, P3 on the edge → zero curvature.
        assert!(close(cp1.1, y0, EPS) && close(cp2.1, y0, EPS));
        // End tangent (P3→P4) matches the arc tangent at P4.
        let centre = (900.0, 100.0);
        let radial = (to.0 - centre.0, to.1 - centre.1);
        let tan = (cp2.0 - to.0, cp2.1 - to.1);
        assert!(close(radial.0 * tan.0 + radial.1 * tan.1, 0.0, 1e-6));
        // Curvature at t = 1: κ = (2/3)|(P2 − P3) × (P3 − P4)| / |P4 − P3|³, near 1/R.
        let v = (to.0 - cp2.0, to.1 - cp2.1);
        let w = (cp2.0 - cp1.0, cp2.1 - cp1.1);
        let k = (2.0 / 3.0) * (w.0 * v.1 - w.1 * v.0).abs() / (v.0 * v.0 + v.1 * v.1).powf(1.5);
        // The article's fixed 2:1 split of a and b (rather than solving for G2)
        // lands the join curvature at 1.3–1.6 × 1/R across ξ, the same as the
        // widely used figma-squircle port; pin that range.
        assert!(k * 100.0 > 1.2 && k * 100.0 < 1.7, "s={s} κR={}", k * 100.0);
        let _ = x0;
    }
}

#[test]
fn zero_smoothing_is_the_plain_rounded_rect() {
    let segs = smoothed_rect_path(200.0, 100.0, [20.0; 4], 0.0);
    assert_eq!(segs[0], Seg::Move(180.0, 0.0));
    let Seg::Cubic { cp1, cp2, to } = segs[1] else {
        panic!()
    };
    assert!(pts_close(cp1, (180.0 + 20.0 * KAPPA, 0.0), EPS));
    assert!(pts_close(cp2, (200.0, 20.0 - 20.0 * KAPPA), EPS));
    assert!(pts_close(to, (200.0, 20.0), EPS));
    assert!(!is_smoothed([20.0; 4], 0.0));
    assert!(!is_smoothed([0.0; 4], 0.6));
    assert!(is_smoothed([0.0, 4.0, 0.0, 0.0], 0.6));
}

#[test]
fn full_smoothing_meets_at_45_degrees_without_an_arc() {
    let segs = smoothed_rect_path(1000.0, 1000.0, [100.0; 4], 1.0);
    // Per corner: two cubics only. 1 move + 4 × (2 cubics + 1 edge line).
    let cubics = segs
        .iter()
        .filter(|s| matches!(s, Seg::Cubic { .. }))
        .count();
    assert_eq!(cubics, 8);
}

#[test]
fn smoothing_gives_way_when_there_is_no_room() {
    // Pill: R = h/2 leaves no room to smooth, the ends stay semicircles.
    let c = corner_params(50.0, 1.0, 50.0);
    assert!(close(c.smoothing, 0.0, EPS));
    assert!(close(c.p, 50.0, EPS));
    // Partial room: R = 40, budget 50 → ξ ≤ 0.25.
    let c = corner_params(40.0, 0.6, 50.0);
    assert!(close(c.smoothing, 0.25, EPS));
    assert!(close(c.p, 50.0, EPS));
    // Every outline point stays inside the box for any radius/smoothing.
    for &r in &[5.0, 30.0, 50.0, 80.0, 500.0] {
        for &s in &[0.0, 0.3, 0.6, 1.0] {
            for p in flatten(&smoothed_rect_path(100.0, 60.0, [r; 4], s), 8) {
                assert!(
                    p.0 >= -1e-9 && p.0 <= 100.0 + 1e-9 && p.1 >= -1e-9 && p.1 <= 60.0 + 1e-9,
                    "r={r} s={s} {p:?}"
                );
            }
        }
    }
}

#[test]
fn per_corner_radii() {
    // Uniform radii: every budget is half the short side.
    assert_eq!(corner_budgets(200.0, 100.0, [10.0; 4]), [50.0; 4]);
    // A zero neighbour leaves the whole edge to the rounded corner (up to the other edge).
    let b = corner_budgets(200.0, 100.0, [0.0, 30.0, 0.0, 0.0]);
    assert_eq!(b[1], 100.0);
    // Largest first: tl=40 takes its proportional 40/(40+10) of the top edge (80),
    // tr then gets the rest of the top (20) but only 10/(10+10) of the right edge.
    let b = corner_budgets(100.0, 100.0, [40.0, 10.0, 10.0, 10.0]);
    assert!(close(b[0], 80.0, EPS));
    assert!(close(b[1], 20.0, EPS));
    // Paths with mixed radii: a sharp corner stays sharp, each corner consumes (1+ξ)R.
    let segs = smoothed_rect_path(300.0, 200.0, [0.0, 40.0, 10.0, 60.0], 0.5);
    assert_eq!(segs[0], Seg::Move(300.0 - 60.0, 0.0)); // tr: p = 1.5 × 40
    assert!(segs.contains(&Seg::Line(0.0, 0.0))); // tl sharp
                                                  // Closed: last point returns to the start.
    let last = match *segs.last().unwrap() {
        Seg::Line(x, y) | Seg::Move(x, y) => (x, y),
        Seg::Cubic { to, .. } => to,
    };
    assert!(pts_close(last, (240.0, 0.0), EPS));
    // Radii larger than the box are normalized first, as every rounded rect.
    assert_eq!(
        normalize_radii(100.0, 100.0, [100.0, 100.0, 0.0, 0.0]),
        [50.0, 50.0, 0.0, 0.0]
    );
}

#[test]
fn contains_point_follows_the_smoothed_outline() {
    // Both outlines pass through the same 45° point (P4 sits on the circle), but the
    // squircle leaves the top edge earlier and stays flatter: at x = 325 the plain
    // R=100 corner is at y ≈ 3.2 and the ξ=1 one at y ≈ 6.3.
    let (x, y) = (325.0, 4.5);
    assert!(contains_point(400.0, 400.0, [100.0; 4], 0.0, x, y));
    assert!(!contains_point(400.0, 400.0, [100.0; 4], 1.0, x, y));
    assert!(contains_point(400.0, 400.0, [100.0; 4], 1.0, 200.0, 200.0));
    assert!(!contains_point(400.0, 400.0, [100.0; 4], 1.0, 401.0, 200.0));
}
