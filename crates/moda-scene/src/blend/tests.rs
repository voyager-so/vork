use super::geometry::{correspond, Pt, Shape};
use super::paint_lerp::{lerp_fills, lerp_paint, lerp_strokes};
use super::steps::{ease, smooth_steps};
use super::*;
use crate::paint::{Fill, Paint, Stroke};
use crate::scene::Scene;

fn scene_with(nodes: Value, node_ids: Vec<&str>) -> Scene {
    let doc = json!({
        "nodes": nodes,
        "pages": [{
            "id": "page-1", "width": 1280.0, "height": 800.0,
            "nodeIds": node_ids,
        }],
        "variables": [],
    });
    Scene::load(&doc).unwrap()
}

fn poly(pts: &[Pt], closed: bool) -> Shape {
    let mut v = vec![BlendPathPoint::Move {
        x: pts[0].0,
        y: pts[0].1,
    }];
    v.extend(
        pts[1..]
            .iter()
            .map(|p| BlendPathPoint::Line { x: p.0, y: p.1 }),
    );
    Shape::from_points(&v, closed)
}

fn square(x: f64, y: f64, s: f64) -> Shape {
    poly(&[(x, y), (x + s, y), (x + s, y + s), (x, y + s)], true)
}

/// A circle as four quarter arcs.
fn circle(cx: f64, cy: f64, r: f64) -> Shape {
    let arc = |x: f64, y: f64| BlendPathPoint::Arc {
        x,
        y,
        rx: r,
        ry: r,
        x_axis_rotation: 0.0,
        large_arc: false,
        sweep: true,
    };
    Shape::from_points(
        &[
            BlendPathPoint::Move { x: cx + r, y: cy },
            arc(cx, cy + r),
            arc(cx - r, cy),
            arc(cx, cy - r),
            arc(cx + r, cy),
        ],
        true,
    )
}

/// A regular star with `n` points (2n line segments).
fn star(cx: f64, cy: f64, outer: f64, inner: f64, n: usize) -> Shape {
    let pts: Vec<Pt> = (0..2 * n)
        .map(|i| {
            let a = std::f64::consts::PI * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
            let r = if i % 2 == 0 { outer } else { inner };
            (cx + r * a.cos(), cy + r * a.sin())
        })
        .collect();
    poly(&pts, true)
}

fn close(a: Pt, b: Pt, tol: f64) -> bool {
    (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
}

/// Every anchor and control point of `a` equals `b`'s (same topology).
fn same_points(a: &Shape, b: &Shape, tol: f64) -> bool {
    a.contours.len() == b.contours.len()
        && a.contours.iter().zip(&b.contours).all(|(x, y)| {
            x.segs.len() == y.segs.len()
                && x.segs.iter().zip(&y.segs).all(|(p, q)| {
                    close(p.p0, q.p0, tol)
                        && close(p.c1, q.c1, tol)
                        && close(p.c2, q.c2, tol)
                        && close(p.p3, q.p3, tol)
                })
        })
}

/// Sample points along an outline, `n` per segment.
fn samples_n(s: &Shape, n: usize) -> Vec<Pt> {
    s.contours
        .iter()
        .flat_map(|c| {
            c.segs
                .iter()
                .flat_map(move |g| (0..n).map(move |i| g.point_at(i as f64 / n as f64)))
        })
        .collect()
}

fn samples(s: &Shape) -> Vec<Pt> {
    samples_n(s, 8)
}

/// Largest distance from any sample of `a` to the outline of `b` (densely sampled): a one-sided Hausdorff estimate,
/// good to about 0.15 px on these shapes.
fn hausdorff(a: &Shape, b: &Shape) -> f64 {
    let sb = samples_n(b, 400);
    samples(a)
        .iter()
        .map(|p| {
            sb.iter()
                .map(|q| (p.0 - q.0).hypot(p.1 - q.1))
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

fn source(shape: Shape, style: Value) -> BlendSource {
    BlendSource::new(shape, style.as_object().unwrap())
}

fn solid(color: &str) -> Value {
    json!({"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": color}}]})
}

fn steps_opts(n: usize) -> BlendOptions {
    BlendOptions {
        spacing: Spacing::Steps(n),
        ..BlendOptions::default()
    }
}

fn centers(steps: &[BlendStep]) -> Vec<Pt> {
    steps.iter().map(|s| s.shape.center().unwrap()).collect()
}

fn fill_color(step: &BlendStep) -> String {
    match &step.fills[0].paint {
        Paint::Solid(s) => s.color.as_str().unwrap().to_string(),
        other => panic!("not solid: {other:?}"),
    }
}

// --- geometry ------------------------------------------------------------------------------------------------------

#[test]
fn quadratic_and_arc_segments_become_cubics_through_the_same_points() {
    let q = Shape::from_points(
        &[
            BlendPathPoint::Move { x: 0.0, y: 0.0 },
            BlendPathPoint::Quadratic {
                x: 100.0,
                y: 0.0,
                cp: (50.0, 100.0),
            },
        ],
        false,
    );
    let seg = q.contours[0].segs[0];
    // A quadratic's midpoint is (p0 + 2 cp + p2) / 4.
    assert!(close(seg.point_at(0.5), (50.0, 50.0), 1e-9));
    assert!(close(seg.c1, (100.0 / 3.0, 200.0 / 3.0), 1e-9));

    let c = circle(0.0, 0.0, 50.0);
    assert_eq!(c.contours[0].segs.len(), 4);
    for p in samples(&c) {
        assert!(
            ((p.0.hypot(p.1)) - 50.0).abs() < 0.2,
            "{p:?} off the circle"
        );
    }
}

#[test]
fn correspondence_equalises_segments_and_keeps_the_end_shapes() {
    let a = square(-50.0, -50.0, 100.0); // 4 segments
    let b = star(0.0, 0.0, 60.0, 25.0, 5); // 10 segments
    let (ca, cb) = correspond(&a, &b);
    // Each is split at the other's anchor positions: the same count, at least the richer one's.
    assert_eq!(ca.contours[0].segs.len(), cb.contours[0].segs.len());
    assert!(ca.contours[0].segs.len() >= 10);
    // t = 0 and t = 1 are the objects themselves, to the point.
    assert!(hausdorff(&ca.lerp(&cb, 0.0), &a) < 0.2 && hausdorff(&a, &ca.lerp(&cb, 0.0)) < 0.2);
    assert!(same_points(&ca.lerp(&cb, 1.0), &cb, 1e-9));
    assert!(hausdorff(&cb, &b) < 0.2 && hausdorff(&b, &cb) < 0.2);
    // Straight segments stay straight.
    assert!(ca.lerp(&cb, 0.5).contours[0].segs.iter().all(|s| s.line));
}

#[test]
fn correspondence_aligns_start_point_and_direction() {
    // The same square, drawn from another corner and the other way round, blends to itself: no twist.
    let a = square(0.0, 0.0, 100.0);
    let b = poly(
        &[(100.0, 100.0), (100.0, 0.0), (0.0, 0.0), (0.0, 100.0)],
        true,
    );
    let (ca, cb) = correspond(&a, &b);
    assert!(same_points(&ca.lerp(&cb, 0.5), &ca, 1e-9));
}

#[test]
fn square_to_circle_passes_between_them() {
    let (ca, cb) = correspond(&square(-50.0, -50.0, 100.0), &circle(0.0, 0.0, 50.0));
    let mid = ca.lerp(&cb, 0.5);
    for p in samples(&mid) {
        let r = p.0.hypot(p.1);
        // Between the circle (50; the cubic arcs are a hair inside) and the square's corners (70.7): points pair at
        // like angles, so nothing dents inside the circle.
        assert!((49.8..=70.8).contains(&r), "{p:?} at radius {r}");
    }
    // Corners are pulled in but stay on the diagonals (no twist): the farthest points sit at 45°.
    let diag = samples(&mid)
        .into_iter()
        .map(|p| p.0.hypot(p.1))
        .fold(0.0, f64::max);
    assert!(diag > 55.0 && diag < 65.0, "max radius {diag}");
    let far = samples(&mid)
        .into_iter()
        .max_by(|p, q| p.0.hypot(p.1).total_cmp(&q.0.hypot(q.1)))
        .unwrap();
    assert!(
        (far.0.abs() - far.1.abs()).abs() < 1.0,
        "farthest point {far:?} is off the diagonal"
    );
}

#[test]
fn square_to_circle_steps_stay_convex() {
    // The chain in the visual-verify scenario: a 40 px square to a 40 px circle. Every step is convex (no dents
    // where a side's midpoint meets the circle).
    let (ca, cb) = correspond(&square(-20.0, -20.0, 40.0), &circle(0.0, 0.0, 20.0));
    for t in [0.25, 0.5, 0.75] {
        let pts = samples_n(&ca.lerp(&cb, t), 32);
        let n = pts.len();
        let mut signs = (0, 0);
        for i in 0..n {
            let (a, b, c) = (pts[i], pts[(i + 1) % n], pts[(i + 2) % n]);
            let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
            if cross > 1e-6 {
                signs.0 += 1;
            } else if cross < -1e-6 {
                signs.1 += 1;
            }
        }
        assert!(
            signs.0 == 0 || signs.1 == 0,
            "t = {t}: {} left turns, {} right turns",
            signs.0,
            signs.1
        );
    }
}

#[test]
fn contours_without_a_partner_grow_from_a_point() {
    let ring = {
        let mut s = square(0.0, 0.0, 100.0);
        s.contours.extend(square(25.0, 25.0, 50.0).contours);
        s
    };
    let (ca, cb) = correspond(&square(300.0, 0.0, 100.0), &ring);
    assert_eq!(ca.contours.len(), 2);
    // The extra contour starts as a point at the hole's place relative to the centre.
    let hole = &ca.contours[1];
    assert!(hole.segs.iter().all(|s| close(s.p0, (350.0, 50.0), 1e-9)));
    assert_eq!(hole.segs.len(), cb.contours[1].segs.len());
}

// --- paint ---------------------------------------------------------------------------------------------------------

fn fills(v: Value) -> Vec<Fill> {
    crate::paint::parse_fills(Some(&v))
}

#[test]
fn solid_colours_interpolate_per_channel() {
    let red = Paint::Solid(crate::paint::SolidPaint {
        color: json!("#ff0000"),
    });
    let blue = Paint::Solid(crate::paint::SolidPaint {
        color: json!("rgba(0, 0, 255, 0.5)"),
    });
    let mid = lerp_paint(&red, &blue, 0.5);
    assert_eq!(
        crate::paint::solid_color(&mid),
        Some("rgba(128, 0, 128, 0.75)")
    );
    assert_eq!(
        crate::paint::solid_color(&lerp_paint(&red, &blue, 0.0)),
        Some("rgba(255, 0, 0, 1)")
    );
}

#[test]
fn gradients_interpolate_placement_and_the_union_of_stops() {
    let a = fills(
        json!([{"id": "f", "paint": {"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50},
        "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}]}}]),
    );
    let b = fills(
        json!([{"id": "f", "paint": {"type": "linear", "start": {"x": 50, "y": 0}, "end": {"x": 50, "y": 100},
        "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 0.5, "color": "#00ff00"}, {"offset": 1, "color": "#0000ff"}]}}]),
    );
    let mid = lerp_fills(&a, &b, 0.5);
    let Paint::Linear(g) = &mid[0].paint else {
        panic!("{mid:?}")
    };
    assert_eq!(
        (g.start.x, g.start.y, g.end.x, g.end.y),
        (25.0, 25.0, 75.0, 75.0)
    );
    let stops: Vec<(f64, &str)> = g
        .stops
        .iter()
        .map(|s| (s.offset, s.color.as_str().unwrap()))
        .collect();
    // a at 0.5 is mid grey (127.5); b there is green.
    assert_eq!(
        stops,
        vec![
            (0.0, "rgba(128, 0, 0, 1)"),
            (0.5, "rgba(64, 191, 64, 1)"),
            (1.0, "rgba(128, 128, 255, 1)")
        ]
    );
}

#[test]
fn solid_to_radial_takes_the_radial_placement() {
    let a = fills(json!([{"id": "f", "paint": {"type": "solid", "color": "#ffffff"}}]));
    let b = fills(
        json!([{"id": "f", "paint": {"type": "radial", "center": {"x": 30, "y": 40}, "radiusX": 20, "radiusY": 10, "rotation": 0,
        "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#000000"}]}}]),
    );
    let q = lerp_fills(&a, &b, 0.25);
    let Paint::Radial(r) = &q[0].paint else {
        panic!("{q:?}")
    };
    assert_eq!((r.center.x, r.center.y, r.radius_x), (30.0, 40.0, 20.0));
    assert_eq!(r.stops[0].color, json!("rgba(191, 191, 191, 1)"));
}

#[test]
fn unpaired_layers_fade_and_images_switch_at_half_way() {
    let a = fills(
        json!([{"id": "f", "paint": {"type": "solid", "color": "#000000"}},
                         {"id": "g", "paint": {"type": "image", "src": "a.png"}}]),
    );
    let b = fills(json!([{"id": "f", "paint": {"type": "image", "src": "b.png"}}]));
    let q = lerp_fills(&a, &b, 0.25);
    assert_eq!(q.len(), 2);
    assert!(
        matches!(&q[0].paint, Paint::Solid(_)),
        "image vs solid: nearer wins"
    );
    assert_eq!(q[1].opacity, Some(0.75));
    assert!(
        matches!(&lerp_fills(&a, &b, 0.75)[0].paint, Paint::Image(p) if p.src.as_deref() == Some("b.png"))
    );
}

#[test]
fn strokes_interpolate_width_paint_and_dashes() {
    let s = |w: f64, c: &str, dash: Value| -> Vec<Stroke> {
        crate::paint::parse_strokes(Some(
            &json!([{"id": "s", "paint": {"type": "solid", "color": c}, "width": w,
            "cap": "round", "dash": dash}]),
        ))
    };
    let q = lerp_strokes(
        &s(2.0, "#000000", json!([4, 4])),
        &s(10.0, "#ffffff", json!([8, 0])),
        0.5,
    );
    assert_eq!(q[0].width, 6.0);
    assert_eq!(q[0].dash, Some(vec![6.0, 2.0]));
    assert_eq!(q[0].cap.as_deref(), Some("round"));
    assert_eq!(
        crate::paint::solid_color(&q[0].paint),
        Some("rgba(128, 128, 128, 1)")
    );
}

// --- spacing, easing, chains -------------------------------------------------------------------------------------

#[test]
fn specified_steps_are_evenly_spaced_between_two_objects() {
    let srcs = [
        source(square(0.0, 0.0, 20.0), solid("#ff0000")),
        source(square(200.0, 100.0, 40.0), solid("#0000ff")),
    ];
    let steps = blend_steps(&srcs, None, &steps_opts(3));
    assert_eq!(steps.len(), 3);
    let c = centers(&steps);
    for (i, p) in c.iter().enumerate() {
        let t = (i + 1) as f64 / 4.0;
        assert!(
            close(*p, (10.0 + 210.0 * t, 10.0 + 110.0 * t), 1e-9),
            "{p:?}"
        );
        let b = steps[i].shape.bounds().unwrap();
        assert!((b.width() - (20.0 + 20.0 * t)).abs() < 1e-9);
    }
    assert_eq!(fill_color(&steps[1]), "rgba(128, 0, 128, 1)");
    assert!(steps.iter().all(|s| s.pair == 0));
}

#[test]
fn a_chain_of_three_blends_each_pair_in_order() {
    let srcs = [
        source(square(0.0, 0.0, 10.0), solid("#ff0000")),
        source(circle(105.0, 5.0, 5.0), solid("#00ff00")),
        source(square(100.0, 100.0, 10.0), solid("#0000ff")),
    ];
    let steps = blend_steps(&srcs, None, &steps_opts(4));
    assert_eq!(steps.len(), 8);
    assert_eq!(
        steps.iter().map(|s| s.pair).collect::<Vec<_>>(),
        vec![0, 0, 0, 0, 1, 1, 1, 1]
    );
    let c = centers(&steps);
    // First leg along y = 5, second along x = 105.
    assert!(c[..4].iter().all(|p| (p.1 - 5.0).abs() < 1e-9));
    assert!(c[4..].iter().all(|p| (p.0 - 105.0).abs() < 1e-9));
    assert_eq!(fill_color(&steps[4]), "rgba(0, 204, 51, 1)");
    // Reverse Front to Back stacks the same steps the other way.
    let rev = blend_steps(
        &srcs,
        None,
        &BlendOptions {
            reverse_stack: true,
            ..steps_opts(4)
        },
    );
    assert_eq!(centers(&rev), c.into_iter().rev().collect::<Vec<_>>());
}

#[test]
fn specified_distance_places_steps_every_distance_short_of_the_next_object() {
    let srcs = [
        source(square(0.0, 0.0, 10.0), solid("#000")),
        source(square(100.0, 0.0, 10.0), solid("#000")),
    ];
    let opts = BlendOptions {
        spacing: Spacing::Distance(30.0),
        ..BlendOptions::default()
    };
    let xs: Vec<f64> = centers(&blend_steps(&srcs, None, &opts))
        .iter()
        .map(|p| p.0)
        .collect();
    assert_eq!(xs, vec![35.0, 65.0, 95.0]);
    // Exactly divisible: no step on top of the next object.
    let opts = BlendOptions {
        spacing: Spacing::Distance(25.0),
        ..opts
    };
    assert_eq!(blend_steps(&srcs, None, &opts).len(), 3);
}

fn rect(x: f64, y: f64, w: f64, h: f64) -> Shape {
    poly(&[(x, y), (x + w, y), (x + w, y + h), (x, y + h)], true)
}

#[test]
fn smooth_color_steps_follow_the_colour_difference() {
    let a = source(square(0.0, 0.0, 10.0), solid("#ff0000"));
    let b = source(square(100.0, 0.0, 10.0), solid("#0000ff"));
    // A full ramp stops at 254, as in Illustrator 30.8.
    assert_eq!(smooth_steps(&a, &b), 254);
    let grey = source(square(100.0, 0.0, 10.0), solid("#f00a00"));
    assert_eq!(smooth_steps(&a, &grey), 15);
    let near = source(square(100.0, 0.0, 10.0), solid("#fc0000"));
    assert_eq!(smooth_steps(&a, &near), 3);
    let one = source(square(100.0, 0.0, 10.0), solid("#fe0000"));
    assert_eq!(smooth_steps(&a, &one), 1);
    let red_55 = source(square(100.0, 0.0, 10.0), solid("#c80000"));
    assert_eq!(smooth_steps(&a, &red_55), 55);
}

/// Steps between two same-colour objects: boxes `a` and `b`, measured in Illustrator 30.8 (Blend ▸ Make + Expand).
fn same_colour(a: Shape, b: Shape) -> usize {
    smooth_steps(&source(a, solid("#3366cc")), &source(b, solid("#3366cc")))
}

#[test]
fn smooth_color_same_colour_steps_follow_the_gap_in_object_sizes() {
    // 30×30 squares side by side: (gap, Illustrator's steps).
    for (gap, want) in [
        (20.0, 1),
        (44.0, 1),
        (45.0, 2),
        (46.0, 2),
        (70.0, 2),
        (75.0, 3),
        (105.0, 4),
        (160.0, 5),
        (300.0, 10),
        (600.0, 20),
        (900.0, 30),
    ] {
        assert_eq!(
            same_colour(square(0.0, 0.0, 30.0), square(30.0 + gap, 0.0, 30.0)),
            want,
            "30×30, gap {gap}"
        );
    }
    // Size along the blend axis sets the unit.
    assert_eq!(
        same_colour(square(0.0, 0.0, 100.0), square(260.0, 0.0, 100.0)),
        2
    );
    assert_eq!(
        same_colour(rect(0.0, 0.0, 60.0, 20.0), rect(220.0, 0.0, 60.0, 20.0)),
        3
    );
    assert_eq!(
        same_colour(rect(0.0, 0.0, 20.0, 60.0), rect(180.0, 0.0, 20.0, 60.0)),
        8
    );
    // Stacked vertically, and diagonally.
    assert_eq!(
        same_colour(square(0.0, 0.0, 30.0), square(0.0, 190.0, 30.0)),
        5
    );
    assert_eq!(
        same_colour(square(10.0, 10.0, 30.0), square(200.0, 200.0, 30.0)),
        5
    );
    // Different sizes: the mean extent.
    assert_eq!(
        same_colour(square(0.0, 0.0, 30.0), square(190.0, 0.0, 60.0)),
        4
    );
    assert_eq!(
        same_colour(square(0.0, 0.0, 30.0), square(190.0, 0.0, 90.0)),
        3
    );
    // Overlapping: one step.
    assert_eq!(
        same_colour(square(0.0, 0.0, 100.0), square(50.0, 0.0, 100.0)),
        1
    );
}

#[test]
fn smooth_color_same_colour_ignores_a_replaced_spine() {
    let srcs = vec![
        source(square(0.0, 0.0, 30.0), solid("#3366cc")),
        source(square(190.0, 0.0, 30.0), solid("#3366cc")),
    ];
    // A long detour between the two: still 5 steps, from the boxes.
    let spine = poly(
        &[(15.0, 15.0), (15.0, 400.0), (205.0, 400.0), (205.0, 15.0)],
        false,
    );
    let opts = BlendOptions::default();
    assert_eq!(blend_steps(&srcs, Some(&spine), &opts).len(), 5);
    assert_eq!(blend_steps(&srcs, None, &opts).len(), 5);
}

#[test]
fn easing_bunches_steps_toward_an_object() {
    assert_eq!(ease(0.3, 0.0, 0.0), 0.3);
    assert!(
        (ease(0.5, 0.5, 0.5) - 0.5).abs() < 1e-9,
        "symmetric ease keeps the middle"
    );
    assert!(ease(0.25, 0.8, 0.0) < 0.25, "ease-in starts slow");
    assert!(ease(0.75, 0.0, 0.8) > 0.75, "ease-out ends slow");
    let mut prev = 0.0;
    for i in 1..100 {
        let e = ease(i as f64 / 100.0, 0.7, 0.3);
        assert!(e > prev, "monotone");
        prev = e;
    }
    let srcs = [
        source(square(0.0, 0.0, 10.0), solid("#000000")),
        source(square(100.0, 0.0, 10.0), solid("#ffffff")),
    ];
    let steps = blend_steps(
        &srcs,
        None,
        &BlendOptions {
            ease_in: 1.0,
            ..steps_opts(5)
        },
    );
    let gaps: Vec<f64> = centers(&steps)
        .windows(2)
        .map(|w| w[1].0 - w[0].0)
        .collect();
    assert!(
        gaps.windows(2).all(|w| w[1] > w[0]),
        "gaps widen away from the first object: {gaps:?}"
    );
    // Colour follows the eased parameter.
    assert!(steps[0].t < 1.0 / 6.0);
    let expected = crate::blend::color::js_round(255.0 * steps[0].t);
    assert_eq!(
        fill_color(&steps[0]),
        format!("rgba({expected}, {expected}, {expected}, 1)")
    );
}

// --- spines --------------------------------------------------------------------------------------------------------

/// A semicircle spine of radius 100 around (100, 0), from (0, 0) over the top to (200, 0).
fn arc_spine() -> Shape {
    Shape::from_points(
        &[
            BlendPathPoint::Move { x: 0.0, y: 0.0 },
            BlendPathPoint::Arc {
                x: 200.0,
                y: 0.0,
                rx: 100.0,
                ry: 100.0,
                x_axis_rotation: 0.0,
                large_arc: false,
                sweep: true,
            },
        ],
        false,
    )
}

#[test]
fn steps_follow_a_replaced_spine_by_arc_length() {
    let srcs = [
        source(square(-5.0, -5.0, 10.0), solid("#000")),
        source(square(195.0, -5.0, 10.0), solid("#000")),
    ];
    let steps = blend_steps(&srcs, Some(&arc_spine()), &steps_opts(7));
    assert_eq!(steps.len(), 7);
    let c = centers(&steps);
    for (i, p) in c.iter().enumerate() {
        let r = (p.0 - 100.0).hypot(p.1);
        assert!(
            (r - 100.0).abs() < 0.05,
            "step {i} at {p:?} is off the arc (r = {r})"
        );
        // Evenly spaced by arc length: the angle from the start grows by 180 / 8 degrees a step.
        let angle = (-p.1).atan2(100.0 - p.0).to_degrees();
        assert!(
            (angle - 22.5 * (i + 1) as f64).abs() < 0.1,
            "step {i} at {angle} degrees"
        );
    }
    // Align to Page: every step stays an upright 10 px square.
    for s in &steps {
        let b = s.shape.bounds().unwrap();
        assert!((b.width() - 10.0).abs() < 1e-6 && (b.height() - 10.0).abs() < 1e-6);
    }
}

#[test]
fn align_to_path_turns_steps_with_the_spine_and_meets_both_objects() {
    // Two identical upright bars at both ends of the arc, where the spine runs straight up and straight down.
    let bar = |cx: f64| {
        poly(
            &[
                (cx - 2.0, -10.0),
                (cx + 2.0, -10.0),
                (cx + 2.0, 10.0),
                (cx - 2.0, 10.0),
            ],
            true,
        )
    };
    let srcs = [
        source(bar(0.0), solid("#000")),
        source(bar(200.0), solid("#000")),
    ];
    let opts = BlendOptions {
        orientation: Orientation::Path,
        ..steps_opts(3)
    };
    let steps = blend_steps(&srcs, Some(&arc_spine()), &opts);
    // Half way the spine runs horizontally, so the bar lies down: wide and short.
    let mid = steps[1].shape.bounds().unwrap();
    assert!(
        (mid.width() - 20.0).abs() < 0.1 && (mid.height() - 4.0).abs() < 0.1,
        "{mid:?}"
    );
    // Page orientation leaves it upright.
    let page = blend_steps(&srcs, Some(&arc_spine()), &steps_opts(3));
    let b = page[1].shape.bounds().unwrap();
    assert!(
        (b.width() - 4.0).abs() < 0.1 && (b.height() - 20.0).abs() < 0.1,
        "{b:?}"
    );
}

#[test]
fn objects_off_the_spine_carry_their_offset() {
    // The second object sits 30 px below the spine's end: steps drift to meet it.
    let srcs = [
        source(square(-5.0, -5.0, 10.0), solid("#000")),
        source(square(195.0, 25.0, 10.0), solid("#000")),
    ];
    let steps = blend_steps(&srcs, Some(&arc_spine()), &steps_opts(1));
    let p = centers(&steps)[0];
    assert!(close(p, (100.0, -100.0 + 15.0), 0.05), "{p:?}");
}

// --- scene and JSON ------------------------------------------------------------------------------------------------

fn rect_node(id: &str, x: f64, color: &str) -> Value {
    json!({"id": id, "type": "rectangle", "x": x, "y": 10.0, "width": 20.0, "height": 20.0, "rotation": 0.0, "scale": 1.0,
           "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": color}}]})
}

fn rect_abs(scene: &Scene, n: &RawNode) -> Option<Vec<BlendPathPoint>> {
    let w = n.f64("width")?;
    let h = n.f64("height")?;
    let t = scene.node_canvas_affine(n.id())?;
    let pts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let mut v = vec![BlendPathPoint::Move { x: 0.0, y: 0.0 }];
    v.extend(
        pts[1..]
            .iter()
            .map(|p| BlendPathPoint::Line { x: p.0, y: p.1 }),
    );
    Some(v.iter().map(|p| p.apply_affine(t)).collect())
}

#[test]
fn expand_blend_node_chains_scene_objects() {
    let scene = scene_with(
        json!([
            rect_node("a", 0.0, "#ff0000"),
            rect_node("b", 100.0, "#00ff00"),
            rect_node("c", 200.0, "#0000ff"),
            {"id": "bl", "type": "blend", "sourceIds": ["a", "b", "c", "missing"], "spacing": "steps", "steps": 1,
             "distance": 20, "orientation": "page"},
        ]),
        vec!["a", "b", "c", "bl"],
    );
    let node = scene.node("bl").unwrap();
    let BlendExpansion::Paths { page_index, nodes } =
        expand_blend_node(&scene, node, &|n| rect_abs(&scene, n))
    else {
        panic!("skipped");
    };
    assert_eq!(page_index, Some(0));
    assert_eq!(
        nodes.iter().map(|n| n.id().to_string()).collect::<Vec<_>>(),
        vec!["bl-s0", "bl-s1"]
    );
    let first = nodes[0].array("points").unwrap();
    assert_eq!(first[0], json!({"type": "move", "x": 50.0, "y": 10.0}));
    assert_eq!(nodes[0].json_field("closed"), Some(&json!(true)));
    assert_eq!(dependency_ids(node), vec!["a", "b", "c", "missing"]);
}

#[test]
fn a_blend_with_fewer_than_two_objects_draws_nothing() {
    let scene = scene_with(
        json!([rect_node("a", 0.0, "#ff0000"), {"id": "bl", "type": "blend", "sourceIds": ["a", "gone"]}]),
        vec!["a", "bl"],
    );
    let node = scene.node("bl").unwrap();
    assert!(matches!(
        expand_blend_node(&scene, node, &|n| rect_abs(&scene, n)),
        BlendExpansion::Skip(_)
    ));
}

#[test]
fn blend_steps_json_round_trips() {
    let sq = |x: f64| {
        json!([{"type": "move", "x": x, "y": 0}, {"type": "line", "x": x + 10.0, "y": 0},
               {"type": "line", "x": x + 10.0, "y": 10}, {"type": "line", "x": x, "y": 10}])
    };
    let req = json!({
        "sources": [
            {"points": sq(0.0), "closed": true, "fills": [{"id": "f", "paint": {"type": "solid", "color": "#000000"}}], "opacity": 0.5},
            {"points": sq(100.0), "closed": true, "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ffffff"}}]},
        ],
        "options": {"spacing": "steps", "steps": 1},
    });
    let reply: Value = crate::value::from_str(&blend_steps_json(&req.to_string())).unwrap();
    assert_eq!(reply["status"], "ok");
    let step = &reply["steps"][0];
    assert_eq!(step["pair"], 0);
    assert_eq!(step["t"], 0.5);
    assert_eq!(step["opacity"], 0.75);
    assert_eq!(
        step["points"][0],
        json!({"type": "move", "x": 50.0, "y": 0.0})
    );
    assert_eq!(step["fills"][0]["paint"]["color"], "rgba(128, 128, 128, 1)");
    let bad: Value = crate::value::from_str(&blend_steps_json(r#"{"sources": []}"#)).unwrap();
    assert_eq!(bad["code"], "too_few_sources");
}

#[test]
fn unrounded_path_absolute_points_match_the_rounder() {
    let scene = scene_with(
        json!([
            {"id": "pa", "type": "path", "x": 20.0, "y": 30.0, "closed": true,
             "points": [
                 {"type": "move", "x": 0.0, "y": 0.0},
                 {"type": "line", "x": 100.0, "y": 0.0},
                 {"type": "line", "x": 100.0, "y": 100.0}
             ]},
        ]),
        vec!["pa"],
    );
    let node = scene.node("pa").unwrap();
    let got = path_absolute_points(&scene, node).unwrap();
    let parsed: Vec<BlendPathPoint> = node
        .points()
        .unwrap()
        .iter()
        .filter_map(BlendPathPoint::from_ref)
        .collect();
    let t = scene.node_canvas_affine("pa").unwrap();
    let expected: Vec<BlendPathPoint> =
        crate::path_geom::round_corners::round_path_corners(&parsed, true, 0.0)
            .iter()
            .map(|p| p.apply_affine(t))
            .collect();
    assert_eq!(format!("{got:?}"), format!("{expected:?}"));
}

#[test]
fn a_blend_of_a_many_anchor_outline_is_linear_in_its_anchors() {
    // Matching anchors scanned every taken fraction per cut: 16k anchors took 2.8 s in release (a minute in a debug
    // build), 100k about two minutes (docs/qa/robustness.md). Now it is about linear.
    let ring = |n: usize, cx: f64| -> Value {
        let pts: Vec<Value> = (0..n)
            .map(|i| {
                let a = i as f64 / n as f64 * std::f64::consts::TAU;
                json!({"type": if i == 0 { "move" } else { "line" }, "x": cx + 10.0 * a.cos(), "y": 10.0 * a.sin()})
            })
            .collect();
        Value::Array(pts)
    };
    let req = json!({
        "sources": [{"points": ring(16_000, 0.0), "closed": true}, {"points": ring(4, 100.0), "closed": true}],
        "options": {"spacing": "steps", "steps": 2},
    });
    let started = std::time::Instant::now();
    let reply: Value = crate::value::from_str(&blend_steps_json(&req.to_string())).unwrap();
    assert_eq!(reply["status"], "ok");
    assert_eq!(reply["steps"].as_array().unwrap().len(), 2);
    assert!(started.elapsed().as_secs() < 20, "{:?}", started.elapsed());
}
