//! Scribble, Tweak, Free Distort, Convert to Shape and the Transform effect
//! against Illustrator 30.8's expanded output (measured through its scripting
//! DOM: `applyEffect` with the plug-in's own keys, then Expand Appearance).
//! Coordinates are the probes': a 100 × 100 square at (70, 70) on a 240 px
//! artboard, y down.

use super::*;

fn rect_at(x: f64, y: f64, w: f64, h: f64) -> Shape {
    let pts = [
        BlendPathPoint::Move { x, y },
        BlendPathPoint::Line { x: x + w, y },
        BlendPathPoint::Line { x: x + w, y: y + h },
        BlendPathPoint::Line { x, y: y + h },
    ];
    contours_from_points(&pts, true)
}

fn circle_at(cx: f64, cy: f64, r: f64) -> Shape {
    let k = r * distort_kappa();
    let pts = [
        BlendPathPoint::Move { x: cx - r, y: cy },
        BlendPathPoint::Curve {
            x: cx,
            y: cy - r,
            cp1: (cx - r, cy - k),
            cp2: (cx - k, cy - r),
        },
        BlendPathPoint::Curve {
            x: cx + r,
            y: cy,
            cp1: (cx + k, cy - r),
            cp2: (cx + r, cy - k),
        },
        BlendPathPoint::Curve {
            x: cx,
            y: cy + r,
            cp1: (cx + r, cy + k),
            cp2: (cx + k, cy + r),
        },
        BlendPathPoint::Curve {
            x: cx - r,
            y: cy,
            cp1: (cx - k, cy + r),
            cp2: (cx - r, cy + k),
        },
    ];
    contours_from_points(&pts, true)
}

fn distort_kappa() -> f64 {
    0.552_284_749_830_793_4
}

fn scribble_fx(extra: Value) -> ScribbleEffectJson {
    let mut v = json!({
        "angle": 0, "pathOverlap": 0, "pathOverlapVariation": 0, "strokeWidth": 2,
        "curviness": 0, "curvinessVariation": 0, "spacing": 10, "spacingVariation": 0
    });
    for (k, val) in extra.as_object().unwrap() {
        v[k] = val.clone();
    }
    crate::value::from_value(v).unwrap()
}

type ScribbleEffectJson = crate::effect::ScribbleEffect;

fn lines(shape: &Shape, e: &ScribbleEffectJson) -> Vec<Contour> {
    let mut rng = distort::scribble_rng(e);
    distort::scribble_centrelines(shape, e, false, &mut rng)
}

// --- Scribble ---

#[test]
fn scribble_hatches_at_zero_degrees_like_illustrator() {
    // `s-a0-sp10`: one path, 20 anchors: (70, 70.53) → (169.97, 70.53) → (70, 80.53) → …
    let runs = lines(&rect_at(70.0, 70.0, 100.0, 100.0), &scribble_fx(json!({})));
    assert_eq!(runs.len(), 1);
    let a = runs[0].anchors();
    assert_eq!(a.len(), 20);
    for (k, pair) in a.chunks(2).enumerate() {
        let y = 70.53 + 10.0 * k as f64;
        assert!(close(pair[0], v2(70.0, y), 0.05), "{k}: {:?}", pair[0]);
        assert!(close(pair[1], v2(170.0, y), 0.05), "{k}: {:?}", pair[1]);
    }
    assert!(
        runs[0].segs.iter().all(|s| s.line),
        "curviness 0 draws straight strokes"
    );
}

#[test]
fn scribble_first_line_sits_a_quarter_stroke_in() {
    // `sq-w1-sp10` / `sq-w8-sp10`: first lines at 70.28 and 72.03.
    for (w, y) in [(1.0, 70.28), (8.0, 72.03)] {
        let runs = lines(
            &rect_at(70.0, 70.0, 100.0, 100.0),
            &scribble_fx(json!({"strokeWidth": w})),
        );
        assert!(
            (runs[0].start.y - y).abs() < 0.02,
            "{w}: {}",
            runs[0].start.y
        );
    }
}

#[test]
fn scribble_angle_turns_the_hatch_and_where_it_starts() {
    // `s-a90-sp10`: vertical lines from the left, each drawn bottom to top.
    let runs = lines(
        &rect_at(70.0, 70.0, 100.0, 100.0),
        &scribble_fx(json!({"angle": 90})),
    );
    let a = runs[0].anchors();
    assert!(
        close(a[0], v2(70.53, 170.0), 0.05) && close(a[1], v2(70.53, 70.0), 0.05),
        "{:?}",
        &a[..2]
    );
    assert!(close(a[2], v2(80.53, 170.0), 0.05));
    // `s-a30-sp10`: 28 anchors, starting in the top-left corner; along the left edge
    // the lines are 10 / cos 30° = 11.55 apart.
    let runs = lines(
        &rect_at(70.0, 70.0, 100.0, 100.0),
        &scribble_fx(json!({"angle": 30})),
    );
    let a = runs[0].anchors();
    assert_eq!(a.len(), 28);
    assert!(close(a[2], v2(70.0, 82.15), 0.1), "{:?}", a[2]);
    assert!(close(a[4], v2(70.0, 93.70), 0.1), "{:?}", a[4]);
    assert!(close(a[3], v2(91.04, 70.0), 0.1), "{:?}", a[3]);
    // `sq-a180`: from the bottom, drawn right to left.
    let runs = lines(
        &rect_at(70.0, 70.0, 100.0, 100.0),
        &scribble_fx(json!({"angle": 180})),
    );
    let a = runs[0].anchors();
    assert!(
        close(a[0], v2(170.0, 169.47), 0.05) && close(a[1], v2(70.0, 169.47), 0.05),
        "{:?}",
        &a[..2]
    );
}

#[test]
fn scribble_overlap_pushes_the_ends_past_the_edge() {
    for (o, x0, x1) in [(5.0, 65.0, 175.0), (-5.0, 75.0, 165.0)] {
        let runs = lines(
            &rect_at(70.0, 70.0, 100.0, 100.0),
            &scribble_fx(json!({"pathOverlap": o})),
        );
        let a = runs[0].anchors();
        assert!(
            (a[0].x - x0).abs() < 0.05 && (a[1].x - x1).abs() < 0.05,
            "{o}: {:?}",
            &a[..2]
        );
    }
}

#[test]
fn scribble_curviness_handles_run_across_the_lines() {
    // `s-a0-sp10-c20`: handles ±10 px across (the square is 100 wide);
    // `wide-c20-sp10`: ±16 on a 160-wide box (curviness × half the crossing).
    for (w, h) in [(100.0, 10.0), (160.0, 16.0)] {
        let runs = lines(
            &rect_at(40.0, 90.0, w, 60.0),
            &scribble_fx(json!({"curviness": 20})),
        );
        let s = runs[0].segs[0];
        let [p0, p1, p2, p3] = s.c.0;
        assert!(close(p1, p0 + v2(0.0, h), 1e-6), "{w}: {p1:?}");
        assert!(close(p2, p3 - v2(0.0, h), 1e-6), "{w}: {p2:?}");
        // The diagonal back leaves downward too.
        let [q0, q1, ..] = runs[0].segs[1].c.0;
        assert!(close(q1, q0 + v2(0.0, h), 1e-6));
    }
}

#[test]
fn scribble_splits_a_star_into_a_path_per_leg() {
    // `s-star-a0`: 26 anchors (13 lines, continuing down the left leg) and 6 (the right leg).
    let runs = lines(&star(120.0, 125.0, 70.0, 30.0, 5), &scribble_fx(json!({})));
    let counts: Vec<usize> = runs.iter().map(|r| r.anchors().len()).collect();
    assert_eq!(counts, vec![26, 6]);
    let a = runs[0].anchors();
    // The last crossing of the first path is the left leg's.
    assert!(a[25].x < 120.0 && a[24].x < 120.0, "{:?}", &a[24..]);
    // The line through the upper arms spans them both (56.2 … 183.8 in Illustrator).
    assert!(
        (a[10].x - 56.2).abs() < 0.6 && (a[11].x - 183.8).abs() < 0.6,
        "{:?}",
        &a[10..12]
    );
}

#[test]
fn scribble_follows_the_edge_of_a_circle() {
    // `s-ellipse-a0` ends with 80.45 … 159.55 at y 150.53.
    let runs = lines(&circle_at(120.0, 120.0, 50.0), &scribble_fx(json!({})));
    let a = runs[0].anchors();
    let pair = a
        .chunks(2)
        .find(|p| (p[0].y - 150.53).abs() < 0.01)
        .expect("line at 150.53");
    assert!(
        (pair[0].x - 80.45).abs() < 0.2 && (pair[1].x - 159.55).abs() < 0.2,
        "{pair:?}"
    );
}

#[test]
fn scribble_variations_are_seeded() {
    let fx = |seed: f64| {
        scribble_fx(
            json!({"spacingVariation": 5, "pathOverlapVariation": 5, "curviness": 5, "curvinessVariation": 3, "seed": seed}),
        )
    };
    let a = lines(&rect_at(0.0, 0.0, 100.0, 100.0), &fx(1.0));
    let b = lines(&rect_at(0.0, 0.0, 100.0, 100.0), &fx(1.0));
    let c = lines(&rect_at(0.0, 0.0, 100.0, 100.0), &fx(2.0));
    assert_eq!(a, b);
    assert_ne!(a, c);
    // Every gap stays within spacing ± variation.
    let ys: Vec<f64> = a[0].anchors().chunks(2).map(|p| p[0].y).collect();
    for w in ys.windows(2) {
        let gap = w[1] - w[0];
        assert!((5.0 - 1e-9..=15.0 + 1e-9).contains(&gap), "{gap}");
    }
}

#[test]
fn scribble_splits_fill_and_stroke_into_their_own_paints() {
    // Illustrator scribbles a stroked shape's fill and, over it, the stroke's band.
    let node = raw(json!({
        "id": "n", "type": "path", "closed": true,
        "points": points_from_contours(&rect_at(70.0, 70.0, 100.0, 100.0)).iter().map(BlendPathPoint::to_json).collect::<Vec<_>>(),
        "fills": [{"id": "f", "paint": {"type": "solid", "color": "#264653"}}],
        "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#e63946"}, "width": 6, "opacity": 0.5}],
        "effects": [{"id": "sc", "type": "scribble", "angle": 0, "pathOverlap": 0, "pathOverlapVariation": 0,
            "strokeWidth": 2, "curviness": 0, "curvinessVariation": 0, "spacing": 10, "spacingVariation": 0}]
    }));
    let r = resolve_node(&node, None).unwrap();
    let parts = split_copies(&r).expect("fill and stroke parts");
    assert_eq!(parts.len(), 2);
    assert!(parts[0].json_field("strokes").is_none());
    assert_eq!(
        parts[0].array("fills").unwrap()[0]["paint"]["color"],
        json!("#264653")
    );
    assert!(parts[1].json_field("strokes").is_none());
    let f = &parts[1].array("fills").unwrap()[0];
    assert_eq!(
        (f["paint"]["color"].clone(), f["opacity"].clone()),
        (json!("#e63946"), json!(0.5))
    );
    // The stroke band's scribble reaches past the square by the half stroke (3 px).
    let pts: Vec<BlendPathPoint> = parts[1]
        .array("points")
        .unwrap()
        .iter()
        .filter_map(BlendPathPoint::from_json)
        .collect();
    let b = bounds_of(std::iter::once(&contours_from_points(&pts, true))).unwrap();
    assert!(
        (b.min.x - 66.0).abs() < 0.2 && (b.max.x - 174.0).abs() < 0.2,
        "{b:?}"
    );
    // The JSON entry point reports the roles.
    let out: Value =
        crate::value::from_str(&geometry_effects_json(&json!({"node": node.to_json_map()}).to_string())).unwrap();
    assert_eq!(out["copyRoles"], json!(["fill", "stroke"]));
}

#[test]
fn scribble_outline_is_the_stroke_width_wide() {
    let fx = scribble_fx(json!({}));
    let mut rng = distort::scribble_rng(&fx);
    let out = distort::scribble(&rect_at(70.0, 70.0, 100.0, 100.0), &fx, false, &mut rng);
    let b = bounds_of(std::iter::once(&out)).unwrap();
    // Round caps: the first line (y 70.53) reaches 1 px above and past each end.
    assert!(
        (b.min.y - 69.53).abs() < 0.05 && (b.min.x - 69.0).abs() < 0.05,
        "{b:?}"
    );
}

// --- Tweak ---

fn tweak_fx(extra: Value) -> Value {
    let mut v = json!({"id": "t", "type": "tweak", "horizontal": 10, "vertical": 0, "sizeMode": "relative", "seed": 3});
    for (k, val) in extra.as_object().unwrap() {
        v[k] = val.clone();
    }
    v
}

#[test]
fn tweak_relative_amounts_are_percentages_of_width_and_height() {
    // Illustrator: 10 % horizontal on a 160 × 60 box moves points up to 16 px across, none down.
    let src = rect_at(40.0, 90.0, 160.0, 60.0);
    let mut max_dx: f64 = 0.0;
    for seed in 0..40 {
        let g = run(
            &src,
            &[tweak_fx(
                json!({"seed": seed, "inHandles": false, "outHandles": false}),
            )],
        );
        for (a, b) in anchors(&g.copies[0]).iter().zip(anchors(&src)) {
            assert!((a.y - b.y).abs() < 1e-9);
            assert!((a.x - b.x).abs() <= 16.0 + 1e-9);
            max_dx = max_dx.max((a.x - b.x).abs());
        }
    }
    assert!(
        max_dx > 14.0,
        "the offsets reach toward the amount: {max_dx}"
    );
    let g = run(&src, &[tweak_fx(json!({"horizontal": 0, "vertical": 10}))]);
    for (a, b) in anchors(&g.copies[0]).iter().zip(anchors(&src)) {
        assert!((a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() <= 6.0 + 1e-9);
    }
}

#[test]
fn tweak_anchors_only_leaves_the_handles_on_the_old_anchors() {
    // `B-rel-anch`: the anchors move; the (retracted) handles stay where the anchors were.
    let src = rect_at(0.0, 0.0, 100.0, 100.0);
    let g = run(
        &src,
        &[tweak_fx(json!({"inHandles": false, "outHandles": false}))],
    );
    let c = &g.copies[0][0];
    for (i, s) in c.segs.iter().enumerate() {
        let [_, p1, p2, _] = s.c.0;
        let orig = &src[0].segs[i];
        assert!(close(p1, orig.start(), 1e-9) && close(p2, orig.end(), 1e-9));
    }
    // Handles only: the anchors stay.
    let g = run(&src, &[tweak_fx(json!({"anchors": false}))]);
    assert_eq!(anchors(&g.copies[0]), anchors(&src));
}

#[test]
fn tweak_absolute_is_px_and_seeded() {
    let src = rect_at(0.0, 0.0, 400.0, 400.0);
    let g = run(
        &src,
        &[tweak_fx(
            json!({"sizeMode": "absolute", "horizontal": 5, "vertical": 5}),
        )],
    );
    for (a, b) in anchors(&g.copies[0]).iter().zip(anchors(&src)) {
        assert!((a.x - b.x).abs() <= 5.0 + 1e-9 && (a.y - b.y).abs() <= 5.0 + 1e-9);
    }
    let again = run(
        &src,
        &[tweak_fx(
            json!({"sizeMode": "absolute", "horizontal": 5, "vertical": 5}),
        )],
    );
    assert_eq!(g.copies, again.copies);
    // Turning the handles off keeps the anchors' draws.
    let anchors_only = run(
        &src,
        &[tweak_fx(
            json!({"sizeMode": "absolute", "horizontal": 5, "vertical": 5, "inHandles": false}),
        )],
    );
    assert_eq!(anchors(&anchors_only.copies[0]), anchors(&g.copies[0]));
}

// --- Free Distort ---

#[test]
fn free_distort_is_bilinear_on_anchors_and_handles() {
    // `K-trapezoid-ellipse`: top corners pulled in to 0.3 / 0.7. The circle keeps 4
    // anchors; the left one lands on (85, 120) with handles (76.716, 147.614) and
    // (93.284, 92.386), the images of the originals under the bilinear map.
    let fx =
        json!({"id": "f", "type": "free-distort", "corners": [[0.3, 0], [0.7, 0], [1, 1], [0, 1]]});
    let g = run(&circle_at(120.0, 120.0, 50.0), &[fx]);
    let c = &g.copies[0][0];
    assert_eq!(c.segs.len(), 4);
    assert!(close(c.start, v2(85.0, 120.0), 1e-3), "{:?}", c.start);
    let last = c.segs[3].c.0;
    assert!(close(last[2], v2(76.716, 147.614), 1e-3), "{:?}", last[2]);
    let first = c.segs[0].c.0;
    assert!(close(first[1], v2(93.284, 92.386), 1e-3), "{:?}", first[1]);
    // `J-trapezoid`: a square's corners land on the targets; edges stay straight.
    let g = run(
        &rect_at(70.0, 70.0, 100.0, 100.0),
        &[
            json!({"id": "f", "type": "free-distort", "corners": [[0.3, 0], [0.7, 0], [1, 1], [0, 1]]}),
        ],
    );
    assert_eq!(
        anchors(&g.copies[0]),
        vec![
            v2(100.0, 70.0),
            v2(140.0, 70.0),
            v2(170.0, 170.0),
            v2(70.0, 170.0)
        ]
    );
    assert!(g.copies[0][0].segs.iter().all(|s| s.line));
}

#[test]
fn free_distort_corners_default_to_no_change() {
    let src = rect_at(0.0, 0.0, 50.0, 20.0);
    let g = run(
        &src,
        &[json!({"id": "f", "type": "free-distort", "corners": [[0, 0], [1, 0]]})],
    );
    assert_eq!(g.copies[0], src);
}

// --- Convert to Shape ---

fn to_shape(extra: Value) -> Geometry {
    let mut v = json!({"id": "c", "type": "convert-to-shape"});
    for (k, val) in extra.as_object().unwrap() {
        v[k] = val.clone();
    }
    run(&rect_at(70.0, 70.0, 100.0, 100.0), &[v])
}

#[test]
fn convert_to_shape_matches_illustrator() {
    // `shape-rect-abs`: 50 × 30 about the centre.
    let b = bounds_of(
        to_shape(json!({"shape": "rectangle", "sizeMode": "absolute", "width": 50, "height": 30}))
            .copies
            .iter(),
    )
    .unwrap();
    assert!(close(b.min, v2(95.0, 105.0), 1e-9) && close(b.max, v2(145.0, 135.0), 1e-9));
    // `shape-round-rel`: 10 / 20 extra each side, radius 9 (circular arcs).
    let g = to_shape(
        json!({"shape": "rounded-rectangle", "extraWidth": 10, "extraHeight": 20, "cornerRadius": 9}),
    );
    let b = bounds_of(g.copies.iter()).unwrap();
    assert!(close(b.min, v2(60.0, 50.0), 1e-9) && close(b.max, v2(180.0, 190.0), 1e-9));
    let arc = g.copies[0][0].segs.iter().find(|s| !s.line).unwrap();
    let [p0, p1, ..] = arc.c.0;
    assert!((p0.dist(p1) - 9.0 * distort_kappa()).abs() < 1e-9);
    // `shape-ellipse-rel`: through the corners of the grown box (√2 × its half sides).
    let b = bounds_of(
        to_shape(json!({"shape": "ellipse", "extraWidth": 10, "extraHeight": 20}))
            .copies
            .iter(),
    )
    .unwrap();
    assert!(
        (b.min.x - 35.147).abs() < 0.01 && (b.min.y - 21.005).abs() < 0.01,
        "{b:?}"
    );
    // `shape-ellipse-abs`: exactly 50 × 30.
    let b = bounds_of(
        to_shape(json!({"shape": "ellipse", "sizeMode": "absolute", "width": 50, "height": 30}))
            .copies
            .iter(),
    )
    .unwrap();
    assert!(
        close(b.min, v2(95.0, 105.0), 0.01) && close(b.max, v2(145.0, 135.0), 0.01),
        "{b:?}"
    );
    // `shape-round-rel-bigradius`: the radius clamps to half the side: a circle.
    let g = to_shape(json!({"shape": "rounded-rectangle", "cornerRadius": 80}));
    assert!(g.copies[0][0].segs.iter().all(|s| !s.line));
}

// --- Transform (the parity case) ---

#[test]
fn transform_matches_illustrator_scale_rotate_move() {
    // `transform-factor`: scale 50 % × 80 %, rotate 30° (counter-clockwise in
    // Illustrator = −30 here), move 30 right and 10 up, 2 copies about the centre.
    let g = run(
        &rect_at(70.0, 70.0, 100.0, 100.0),
        &[
            json!({"id": "t", "type": "transform", "scaleX": 50, "scaleY": 80, "rotate": -30, "moveX": 30, "moveY": -10, "copies": 2}),
        ],
    );
    assert_eq!(g.copies.len(), 3);
    // Copy 1's top-left and copy 2's top-left, from Illustrator.
    assert!(
        close(g.copies[1][0].start, v2(108.349, 87.859), 1e-3),
        "{:?}",
        g.copies[1][0].start
    );
    assert!(
        close(g.copies[2][0].start, v2(132.099, 90.645), 1e-3),
        "{:?}",
        g.copies[2][0].start
    );
}

#[test]
fn transform_random_stays_between_none_and_the_value() {
    let fx = |seed: f64| json!({"id": "t", "type": "transform", "moveX": 40, "copies": 5, "random": true, "seed": seed});
    let g = run(&rect_at(0.0, 0.0, 10.0, 10.0), &[fx(1.0)]);
    let starts: Vec<f64> = g.copies.iter().map(|c| c[0].start.x).collect();
    for w in starts.windows(2) {
        let step = w[1] - w[0];
        assert!((0.0..=40.0).contains(&step), "{step}");
    }
    assert_ne!(
        starts,
        run(&rect_at(0.0, 0.0, 10.0, 10.0), &[fx(2.0)])
            .copies
            .iter()
            .map(|c| c[0].start.x)
            .collect::<Vec<_>>()
    );
}
