use super::raster::{polygons, Grid};
use super::*;
use crate::path_geom::fit::tests::Rng;
use crate::path_geom::vec2::{v2, V2};
use crate::value::json;

pub(crate) fn node(v: Value) -> RawNode {
    let mut v = v;
    v["type"] = json!("path");
    RawNode::from_value(v).unwrap()
}

pub(crate) fn outline(v: Value, opts: OutlineOptions) -> Outline {
    stroke_outline(&node(v), None, &opts).expect("outline")
}

pub(crate) fn shape(o: &Outline) -> (&Vec<PathPoint>, usize) {
    match o {
        Outline::Shape {
            points,
            subpath_count,
            ..
        } => (points, *subpath_count),
        Outline::Empty => panic!("expected a shape"),
    }
}

fn line_pts(x0: f64, y0: f64, x1: f64, y1: f64) -> Value {
    json!([{"type": "move", "x": x0, "y": y0}, {"type": "line", "x": x1, "y": y1}])
}

/// Vertical extent of the outline at `x`: (min y, max y) over polygon crossings.
fn extent_at_x(polys: &[Vec<V2>], x: f64) -> Option<(f64, f64)> {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for poly in polys {
        let n = poly.len();
        for i in 0..n {
            let (a, b) = (poly[i], poly[(i + 1) % n]);
            if (a.x <= x) != (b.x <= x) {
                let y = a.y + (x - a.x) / (b.x - a.x) * (b.y - a.y);
                lo = lo.min(y);
                hi = hi.max(y);
            }
        }
    }
    lo.is_finite().then_some((lo, hi))
}

fn all_finite(points: &[PathPoint]) -> bool {
    !points
        .iter()
        .any(|p| p.to_json().to_string().contains("null"))
}

#[test]
fn plain_line_is_a_rectangle() {
    let o = outline(
        json!({"points": line_pts(0.0, 0.0, 100.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10}]}),
        OutlineOptions::default(),
    );
    let (pts, n) = shape(&o);
    assert_eq!(n, 1);
    let polys = polygons(pts);
    for x in [1.0, 25.0, 50.0, 99.0] {
        let (lo, hi) = extent_at_x(&polys, x).unwrap();
        assert!(
            (lo + 5.0).abs() < 1e-6 && (hi - 5.0).abs() < 1e-6,
            "x {x}: {lo} {hi}"
        );
    }
    // Butt caps: nothing beyond the ends.
    assert!(extent_at_x(&polys, -0.5).is_none() && extent_at_x(&polys, 100.5).is_none());
    // A rectangle needs few points.
    assert!(pts.len() <= 6, "{pts:?}");
    if let Outline::Shape {
        bounds,
        max_half_width,
        ..
    } = &o
    {
        assert_eq!(*max_half_width, 5.0);
        assert!((bounds.min_y + 5.0).abs() < 1e-9 && (bounds.max_x - 100.0).abs() < 1e-9);
    }
}

#[test]
fn width_matches_profile_at_samples() {
    // Asymmetric markers incl. a corner marker and a discontinuity; straight path
    // along +x (y-down): left of travel is −y.
    let profile = json!({"markers": [
        {"id": "a", "t": 0, "left": 0.5, "right": 1},
        {"id": "b", "t": 0.3, "left": 2, "right": 1, "corner": true},
        {"id": "c", "t": 0.6, "left": 1, "right": 0.25},
        {"id": "d", "t": 0.6, "left": 3, "right": 3},
        {"id": "e", "t": 1, "left": 1, "right": 2}
    ]});
    let v = json!({"points": line_pts(0.0, 0.0, 200.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 8, "profile": profile}]});
    let tol = 0.05;
    let o = outline(
        v.clone(),
        OutlineOptions {
            tolerance: tol,
            ..Default::default()
        },
    );
    let polys = polygons(shape(&o).0);
    let model = ProfileCurve::from_model(&node(v).stroke_profile().unwrap(), &[], false);
    for i in 1..200 {
        let x = i as f64 + 0.37;
        let t = x / 200.0;
        let (l, r) = model.eval(t, false);
        let (lo, hi) = extent_at_x(&polys, x).unwrap();
        assert!(
            (lo + l * 4.0).abs() < 0.1,
            "x {x}: top {lo} vs {}",
            -l * 4.0
        );
        assert!(
            (hi - r * 4.0).abs() < 0.1,
            "x {x}: bottom {hi} vs {}",
            r * 4.0
        );
    }
    // The step at t = 0.6 (x = 120) is sharp: just either side differs.
    let (lo_a, _) = extent_at_x(&polys, 119.9).unwrap();
    let (lo_b, _) = extent_at_x(&polys, 120.1).unwrap();
    assert!(lo_a > -4.5 && lo_b < -11.5, "{lo_a} {lo_b}");
}

#[test]
fn symmetric_profiles_give_symmetric_outlines() {
    for preset in ["eye", "mirrored-taper", "uniform"] {
        let v = json!({"points": line_pts(0.0, 0.0, 300.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 20, "profile": {"preset": preset}}]});
        let o = outline(
            v,
            OutlineOptions {
                tolerance: 0.02,
                ..Default::default()
            },
        );
        let polys = polygons(shape(&o).0);
        for i in 1..60 {
            let x = i as f64 * 5.0 + 0.5;
            let (lo, hi) = extent_at_x(&polys, x).unwrap_or((0.0, 0.0));
            let (lo2, hi2) = extent_at_x(&polys, 300.0 - x).unwrap_or((0.0, 0.0));
            assert!((lo + hi).abs() < 0.05, "{preset} across at {x}: {lo} {hi}");
            assert!(
                (lo - lo2).abs() < 0.05 && (hi - hi2).abs() < 0.05,
                "{preset} along at {x}"
            );
        }
    }
}

#[test]
fn widths_come_from_the_shared_profile_evaluator() {
    // The outliner samples `stroke_model`'s evaluator (the Width tool's and TS's),
    // so a preset / marker width here IS the model's width.
    for preset in [
        "uniform",
        "wedge",
        "taper",
        "quarter-taper",
        "eye",
        "mirrored-taper",
    ] {
        let v = json!({"points": line_pts(0.0, 0.0, 100.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "profile": {"preset": preset, "flipAlong": true}}]});
        let profile = node(v.clone()).stroke_profile().unwrap();
        let Outline::Shape {
            samples: Some(ws), ..
        } = outline(
            v,
            OutlineOptions {
                samples: true,
                ..Default::default()
            },
        )
        else {
            panic!()
        };
        for (i, t) in ws.t.iter().enumerate() {
            let (l, r) = crate::stroke_model::width_at(&profile, *t, None, Some((0.0, 1.0)));
            assert!(
                (ws.left[i] - 5.0 * l).abs() < 1e-9 && (ws.right[i] - 5.0 * r).abs() < 1e-9,
                "{preset} at {t}"
            );
        }
    }
}

#[test]
fn anchored_markers_follow_their_anchor() {
    // End-to-end: the anchor index is resolved on the node's points.
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 80, "y": 0}, {"type": "line", "x": 100, "y": 0}]);
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "profile": {"markers": [
        {"id": "a", "t": 0, "left": 1, "right": 1},
        {"id": "b", "t": 0.5, "left": 3, "right": 3, "anchor": 1, "corner": true},
        {"id": "c", "t": 1, "left": 1, "right": 1}
    ]}}]});
    let o = outline(
        v,
        OutlineOptions {
            tolerance: 0.01,
            ..Default::default()
        },
    );
    let polys = polygons(shape(&o).0);
    let (lo, _) = extent_at_x(&polys, 80.0).unwrap();
    assert!(
        (lo + 15.0).abs() < 0.05,
        "peak follows anchor 1 at x=80: {lo}"
    );
}

#[test]
fn samples_on_anchors_have_the_anchor_t() {
    // Sample t is measured with the shared `path_length::anchor_fractions`, so
    // a width marker bound to an anchor lands exactly on it.
    let pts = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 100, "y": 0, "cp1": {"x": 30, "y": -60}, "cp2": {"x": 70, "y": 60}},
        {"type": "arc", "x": 150, "y": 50, "rx": 50, "ry": 50, "largeArc": false, "sweep": true},
        {"type": "line", "x": 150, "y": 120}]);
    let fr = crate::path_length::anchor_fractions(pts.as_array().unwrap(), true);
    let o = outline(
        json!({"points": pts, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4}]}),
        OutlineOptions {
            samples: true,
            ..Default::default()
        },
    );
    let Outline::Shape {
        samples: Some(ws), ..
    } = &o
    else {
        panic!()
    };
    for (k, anchor) in [
        (1usize, v2(100.0, 0.0)),
        (2, v2(150.0, 50.0)),
        (3, v2(150.0, 120.0)),
    ] {
        let i = (0..ws.x.len())
            .find(|&i| v2(ws.x[i], ws.y[i]).dist(anchor) < 1e-9)
            .expect("a sample on the anchor");
        assert!(
            (ws.t[i] - fr.anchor_t[k]).abs() < 1e-12,
            "anchor {k}: {} vs {}",
            ws.t[i],
            fr.anchor_t[k]
        );
    }
}

#[test]
fn path_without_leading_move_starts_at_its_first_point() {
    // `anchor_fractions` and the outliner agree: no segment from (0, 0).
    let pts = json!([{"type": "line", "x": 30, "y": 40}, {"type": "line", "x": 60, "y": 0}]);
    let fr = crate::path_length::anchor_fractions(pts.as_array().unwrap(), false);
    assert_eq!(fr.total_length, 50.0);
    let o = outline(
        json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4}]}),
        OutlineOptions {
            samples: true,
            ..Default::default()
        },
    );
    let Outline::Shape {
        samples: Some(ws),
        bounds,
        ..
    } = &o
    else {
        panic!()
    };
    assert!(
        bounds.min_x > 25.0,
        "nothing drawn from the origin: {bounds:?}"
    );
    assert_eq!((ws.x[0], ws.y[0], ws.t[0]), (30.0, 40.0, 0.0));
}

#[test]
fn closed_ring_profile_has_no_v_at_the_seam() {
    // A single closed subpath evaluates its profile periodically: the eye's
    // zero-width point at the seam is a smooth minimum (quadratic), not a V.
    let ring = json!([{"type": "move", "x": 50, "y": 0}, {"type": "arc", "x": -50, "y": 0, "rx": 50, "ry": 50, "largeArc": false, "sweep": true},
                      {"type": "arc", "x": 50, "y": 0, "rx": 50, "ry": 50, "largeArc": false, "sweep": true}]);
    let widths = |closed: bool| {
        let o = outline(
            json!({"points": ring, "closed": closed, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 20, "profile": {"preset": "eye"}}]}),
            OutlineOptions {
                samples: true,
                ..Default::default()
            },
        );
        let Outline::Shape {
            samples: Some(ws), ..
        } = o
        else {
            panic!()
        };
        (
            ws.width_at(0.01).0,
            ws.width_at(0.02).0,
            ws.width_at(0.99).0,
        )
    };
    let (a, b, c) = widths(true);
    assert!(a / b < 0.35, "quadratic near the seam: {a} vs {b}");
    assert!(
        (a - c).abs() < 0.02 * b.max(1e-9) + 1e-6,
        "symmetric across the seam: {a} vs {c}"
    );
    // The same ring left open keeps the one-sided (linear) ends.
    let (a, b, _) = widths(false);
    assert!(a / b > 0.4, "open ring: linear ends: {a} vs {b}");
}

#[test]
fn taper_and_wave() {
    let v = json!({"points": line_pts(0.0, 0.0, 200.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "taper": {"units": "px", "startLength": 50, "endLength": 20, "startWidth": 0, "endWidth": 0.5}}]});
    let o = outline(
        v,
        OutlineOptions {
            tolerance: 0.01,
            ..Default::default()
        },
    );
    let polys = polygons(shape(&o).0);
    let (lo, _) = extent_at_x(&polys, 25.0).unwrap();
    assert!((lo + 2.5).abs() < 0.05, "linear taper halfway: {lo}");
    let (lo, _) = extent_at_x(&polys, 100.0).unwrap();
    assert!((lo + 5.0).abs() < 0.05);
    let (lo, _) = extent_at_x(&polys, 199.99).unwrap();
    assert!((lo + 2.5).abs() < 0.05, "end width 0.5: {lo}");
    let (lo, _) = extent_at_x(&polys, 0.01).unwrap();
    assert!(lo.abs() < 0.05, "start tapers to a point: {lo}");

    let v = json!({"points": line_pts(0.0, 0.0, 200.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "wave": {"amount": 0.5, "units": "px", "wavelength": 40, "phase": 90}}]});
    let o = outline(
        v,
        OutlineOptions {
            tolerance: 0.01,
            ..Default::default()
        },
    );
    let polys = polygons(shape(&o).0);
    let (lo, _) = extent_at_x(&polys, 40.0).unwrap();
    assert!((lo + 7.5).abs() < 0.05, "crest: {lo}");
    let (lo, _) = extent_at_x(&polys, 60.0).unwrap();
    assert!((lo + 2.5).abs() < 0.05, "trough: {lo}");
}

#[test]
fn caps_and_joins() {
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}, {"type": "line", "x": 100, "y": 100}]);
    for (cap, join, beyond) in [
        ("round", "round", 5.0),
        ("square", "miter", 5.0),
        ("butt", "bevel", 0.0),
    ] {
        let o = outline(
            json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "cap": cap, "join": join}]}),
            OutlineOptions {
                tolerance: 0.01,
                ..Default::default()
            },
        );
        let Outline::Shape { bounds, .. } = &o else {
            panic!()
        };
        assert!(
            (bounds.min_x + beyond).abs() < 0.02,
            "{cap}: start cap reaches {}",
            bounds.min_x
        );
        assert!(
            (bounds.max_y - (100.0 + beyond)).abs() < 0.02,
            "{cap}: end cap reaches {}",
            bounds.max_y
        );
        let polys = polygons(shape(&o).0);
        // The outer corner is at (105, -5): miter reaches it, bevel/round don't.
        let (lo, _) = extent_at_x(&polys, 104.9).unwrap();
        match join {
            "miter" => assert!((lo + 5.0).abs() < 0.05, "{join}: {lo}"),
            "round" => assert!(lo > -1.5, "{join}: {lo}"),
            _ => assert!(lo > -0.2, "{join}: {lo}"),
        }
    }
    // Miter limit: a hairpin exceeds 4 and bevels.
    let hairpin = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}, {"type": "line", "x": 0, "y": 10}]);
    let o = outline(
        json!({"points": hairpin, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "join": "miter"}]}),
        OutlineOptions::default(),
    );
    let Outline::Shape { bounds, .. } = &o else {
        panic!()
    };
    assert!(
        bounds.max_x < 106.0,
        "miter limit bevels the hairpin: {}",
        bounds.max_x
    );
}

#[test]
fn round_cap_takes_the_end_width_and_taper_ends_in_a_point() {
    let v = json!({"points": line_pts(0.0, 0.0, 100.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "cap": "round", "profile": {"markers": [{"id": "a", "t": 0, "left": 2, "right": 2}, {"id": "b", "t": 1, "left": 0.5, "right": 0.5}]}}]});
    let Outline::Shape { bounds, .. } = outline(
        v,
        OutlineOptions {
            tolerance: 0.01,
            ..Default::default()
        },
    ) else {
        panic!()
    };
    assert!(
        (bounds.min_x + 10.0).abs() < 0.05,
        "start cap radius 10: {}",
        bounds.min_x
    );
    assert!(
        (bounds.max_x - 102.5).abs() < 0.05,
        "end cap radius 2.5: {}",
        bounds.max_x
    );
    let v = json!({"points": line_pts(0.0, 0.0, 100.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "cap": "round", "profile": {"preset": "wedge"}}]});
    let Outline::Shape { bounds, .. } = outline(v, OutlineOptions::default()) else {
        panic!()
    };
    assert!(
        (bounds.max_x - 100.0).abs() < 1e-6,
        "tapered end has no cap: {}",
        bounds.max_x
    );
}

#[test]
fn degenerate_paths() {
    let opts = OutlineOptions::default();
    // No stroke weight / no points / lone move.
    assert!(matches!(
        outline(json!({"points": line_pts(0.0, 0.0, 10.0, 0.0)}), opts),
        Outline::Empty
    ));
    assert!(matches!(
        outline(json!({"points": [], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4}]}), opts),
        Outline::Empty
    ));
    assert!(matches!(
        outline(
            json!({"points": [{"type": "move", "x": 1, "y": 1}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "cap": "round"}]}),
            opts
        ),
        Outline::Empty
    ));
    // Zero-length segment: a dot with round / square caps, nothing with butt.
    let zero = json!([{"type": "move", "x": 5, "y": 5}, {"type": "line", "x": 5, "y": 5}]);
    let Outline::Shape { bounds, .. } = outline(
        json!({"points": zero, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "cap": "round"}]}),
        opts,
    ) else {
        panic!()
    };
    assert!((bounds.min_x - 3.0).abs() < 0.01 && (bounds.max_x - 7.0).abs() < 0.01);
    assert!(matches!(
        outline(
            json!({"points": zero, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "cap": "square"}]}),
            opts
        ),
        Outline::Shape { .. }
    ));
    assert!(matches!(
        outline(json!({"points": zero, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4}]}), opts),
        Outline::Empty
    ));
    // All-zero profile.
    let v = json!({"points": line_pts(0.0, 0.0, 10.0, 0.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "profile": {"markers": [{"id": "a", "t": 0, "left": 0, "right": 0}]}}]});
    assert!(matches!(outline(v, opts), Outline::Empty));
    // Garbage points are skipped, not fatal.
    let v = json!({"points": [{"type": "move", "x": 0, "y": 0}, {"type": "wat"}, {"type": "line", "x": "a", "y": 1},
                              {"type": "line", "x": 1e300, "y": 0}, {"type": "line", "x": 10, "y": 0}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 2}]});
    assert!(matches!(outline(v, opts), Outline::Shape { .. }));
}

#[test]
fn closed_paths_are_rings() {
    // A closed square: an outer and an inner ring, opposite windings.
    let sq = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                    {"type": "line", "x": 100, "y": 100}, {"type": "line", "x": 0, "y": 100}]);
    let o = outline(
        json!({"points": sq, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10}]}),
        OutlineOptions::default(),
    );
    let (pts, n) = shape(&o);
    assert_eq!(n, 2);
    let polys = polygons(pts);
    let g = Grid::around(-10.0, -10.0, 110.0, 110.0, 1.0, 0.0);
    let m = g.fill_nonzero(&polys);
    let at = |x: f64, y: f64| m[((y + 10.0) as usize) * g.w + (x + 10.0) as usize];
    assert!(
        at(0.0, 50.0) && at(50.0, 2.0) && at(103.0, 103.0),
        "band + mitered corner"
    );
    assert!(!at(50.0, 50.0), "the middle is a hole");
    // A small closed circle with a huge width fills solid (inner ring collapses).
    let circle = json!([{"type": "move", "x": 10, "y": 0}, {"type": "arc", "x": -10, "y": 0, "rx": 10, "ry": 10, "largeArc": false, "sweep": true},
                        {"type": "arc", "x": 10, "y": 0, "rx": 10, "ry": 10, "largeArc": false, "sweep": true}]);
    let o = outline(
        json!({"points": circle, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 40}]}),
        OutlineOptions::default(),
    );
    let polys = polygons(shape(&o).0);
    let g = Grid::around(-30.0, -30.0, 30.0, 30.0, 1.0, 0.0);
    let m = g.fill_nonzero(&polys);
    let covered = m.iter().filter(|b| **b).count();
    // Disc of radius 30 ≈ 2827 px.
    assert!(
        (covered as f64 - 2827.0).abs() < 60.0,
        "solid disc: {covered}"
    );
    assert!(m[30 * g.w + 30], "centre filled");
}

#[test]
fn compound_paths_outline_each_subpath_with_global_t() {
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                     {"type": "move", "x": 0, "y": 50}, {"type": "line", "x": 100, "y": 50}]);
    // Markers over the TOTAL length: the first subpath gets 0..0.5.
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "profile": {"markers": [
        {"id": "a", "t": 0, "left": 0, "right": 0, "corner": true}, {"id": "b", "t": 1, "left": 2, "right": 2, "corner": true}]}}]});
    let o = outline(
        v,
        OutlineOptions {
            tolerance: 0.01,
            ..Default::default()
        },
    );
    let (pts_out, n) = shape(&o);
    assert_eq!(n, 2);
    let polys = polygons(pts_out);
    let top = polys.iter().find(|p| p.iter().all(|q| q.y < 25.0)).unwrap();
    let bot = polys.iter().find(|p| p.iter().all(|q| q.y > 25.0)).unwrap();
    let (lo, _) = extent_at_x(std::slice::from_ref(top), 99.9).unwrap();
    assert!(
        (lo + 5.0).abs() < 0.05,
        "end of subpath 1 is t=0.5 → width 1: {lo}"
    );
    let (lo, _) = extent_at_x(std::slice::from_ref(bot), 99.9).unwrap();
    assert!(
        (lo - 40.0).abs() < 0.05,
        "end of subpath 2 is t=1 → width 2: {lo}"
    );
    // Presets are per subpath: both subpaths wedge the same way.
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                     {"type": "move", "x": 0, "y": 50}, {"type": "line", "x": 100, "y": 50}]);
    let o = outline(
        json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "profile": {"preset": "wedge"}}]}),
        OutlineOptions::default(),
    );
    let polys = polygons(shape(&o).0);
    let top = polys.iter().find(|p| p.iter().all(|q| q.y < 25.0)).unwrap();
    let bot = polys.iter().find(|p| p.iter().all(|q| q.y > 25.0)).unwrap();
    let (a, _) = extent_at_x(std::slice::from_ref(top), 50.0).unwrap();
    let (b, _) = extent_at_x(std::slice::from_ref(bot), 50.0).unwrap();
    assert!((a + 2.5).abs() < 0.05 && (b - 47.5).abs() < 0.05, "{a} {b}");
}

#[test]
fn outline_winding_is_consistent() {
    // Every contour of an open stroke winds positive (band on the right, y-down).
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "curve", "x": 100, "y": 0, "cp1": {"x": 30, "y": -80}, "cp2": {"x": 70, "y": 80}}]);
    for dir in [false, true] {
        let mut p = pts.clone();
        if dir {
            p = json!([{"type": "move", "x": 100, "y": 0}, {"type": "curve", "x": 0, "y": 0, "cp1": {"x": 70, "y": 80}, "cp2": {"x": 30, "y": -80}}]);
        }
        let o = outline(
            json!({"points": p, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 12, "cap": "round"}]}),
            OutlineOptions::default(),
        );
        for poly in polygons(shape(&o).0) {
            let a: f64 = (0..poly.len())
                .map(|i| poly[i].cross(poly[(i + 1) % poly.len()]))
                .sum();
            assert!(a > 0.0, "reversed={dir}");
        }
    }
}

/// The band as the union of per-chord trapezoids (independent of the
/// outliner's side construction), from the returned width samples.
fn band_pieces(ws: &WidthSamples) -> Vec<Vec<V2>> {
    let mut starts = ws.subpath_starts.clone();
    starts.push(ws.t.len());
    let mut out = Vec::new();
    for w in starts.windows(2) {
        for i in w[0]..w[1].saturating_sub(1) {
            let (a, b) = (v2(ws.x[i], ws.y[i]), v2(ws.x[i + 1], ws.y[i + 1]));
            let Some(d) = (b - a).unit() else { continue };
            let n = d.left_normal();
            out.push(vec![
                a + n * ws.left[i],
                b + n * ws.left[i + 1],
                b - n * ws.right[i + 1],
                a - n * ws.right[i],
            ]);
        }
    }
    out
}

/// Discs at every sample (radius = larger side): with round caps and joins
/// the stroke never reaches past `band ∪ discs`.
fn disc_pieces(ws: &WidthSamples, min_r: f64) -> Vec<Vec<V2>> {
    (0..ws.t.len())
        .map(|i| {
            let r = ws.left[i].max(ws.right[i]).max(min_r);
            (0..24)
                .map(|k| {
                    let a = k as f64 / 24.0 * std::f64::consts::TAU;
                    v2(
                        ws.x[i] + r * a.cos() / (std::f64::consts::PI / 24.0).cos(),
                        ws.y[i] + r * a.sin() / (std::f64::consts::PI / 24.0).cos(),
                    )
                })
                .collect()
        })
        .collect()
}

pub(crate) struct Coverage {
    pub holes: usize,
    pub spikes: usize,
    pub inside: usize,
}

/// Holes: pixels well inside the band that the outline leaves empty. Spikes:
/// outline pixels well outside band ∪ discs. (Round caps/joins only.)
pub(crate) fn coverage(v: &Value, px: f64) -> Coverage {
    let o = outline(
        v.clone(),
        OutlineOptions {
            samples: true,
            tolerance: 0.1,
            ..Default::default()
        },
    );
    let Outline::Shape {
        points,
        bounds,
        samples: Some(ws),
        ..
    } = &o
    else {
        return Coverage {
            holes: 0,
            spikes: 0,
            inside: 0,
        };
    };
    assert!(all_finite(points));
    let g = Grid::around(
        bounds.min_x,
        bounds.min_y,
        bounds.max_x,
        bounds.max_y,
        px,
        3.0 * px,
    );
    let out = g.fill_nonzero(&polygons(points));
    let band = g.fill_union(&band_pieces(ws));
    let mut outer_pieces = band_pieces(ws);
    outer_pieces.extend(disc_pieces(ws, 1.5 * px));
    let outer = g.dilate(&g.fill_union(&outer_pieces), 2);
    let deep = g.erode(&band, 2);
    let holes = (0..out.len()).filter(|&i| deep[i] && !out[i]).count();
    let spikes = (0..out.len()).filter(|&i| out[i] && !outer[i]).count();
    Coverage {
        holes,
        spikes,
        inside: deep.iter().filter(|b| **b).count(),
    }
}

fn random_path(rng: &mut Rng, segs: usize, scale: f64) -> Value {
    let mut pts = vec![json!({"type": "move", "x": rng.f(0.0, scale), "y": rng.f(0.0, scale)})];
    for _ in 0..segs {
        if rng.next_u64() % 3 == 0 {
            pts.push(json!({"type": "line", "x": rng.f(0.0, scale), "y": rng.f(0.0, scale)}));
        } else {
            pts.push(json!({"type": "curve", "x": rng.f(0.0, scale), "y": rng.f(0.0, scale),
                "cp1": {"x": rng.f(0.0, scale), "y": rng.f(0.0, scale)}, "cp2": {"x": rng.f(0.0, scale), "y": rng.f(0.0, scale)}}));
        }
    }
    Value::Array(pts)
}

fn random_profile(rng: &mut Rng) -> Value {
    let n = 2 + (rng.next_u64() % 5) as usize;
    let mut ms = Vec::new();
    for i in 0..n {
        let t = if i == 0 {
            0.0
        } else if i == n - 1 {
            1.0
        } else {
            rng.f(0.0, 1.0)
        };
        ms.push(json!({"id": format!("m{i}"), "t": t, "left": rng.f(0.0, 2.0), "right": rng.f(0.0, 2.0), "corner": rng.next_u64() % 4 == 0}));
    }
    json!({"markers": ms})
}

#[test]
fn property_no_holes_or_spikes_on_random_strokes() {
    let mut rng = Rng::new(0x5_7120);
    let mut worst = 0.0f64;
    for case in 0..60 {
        let segs = 1 + (rng.next_u64() % 4) as usize;
        let path = random_path(&mut rng, segs, 200.0);
        let width = [4.0, 12.0, 30.0][case % 3];
        let mut v = json!({"points": path, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": "round", "join": "round"}],
                           "closed": case % 5 == 0});
        if case % 2 == 1 {
            v["strokes"][0]["profile"] = random_profile(&mut rng);
        }
        if case % 7 == 3 {
            v["strokes"][0]["taper"] =
                json!({"startLength": 30, "endLength": 30, "startEase": rng.f(-1.0, 1.0)});
        }
        let c = coverage(&v, 1.0);
        let frac = c.holes as f64 / c.inside.max(1) as f64;
        worst = worst.max(frac);
        assert!(c.spikes == 0, "case {case}: {} spike pixels\n{v}", c.spikes);
        // At 1 px a hairline's boundary can flip a pixel or two.
        assert!(
            c.holes <= 2 || frac < 0.002,
            "case {case}: {} hole pixels of {}\n{v}",
            c.holes,
            c.inside
        );
    }
    eprintln!("worst hole fraction {worst}");
}

#[test]
fn tight_turns_have_no_holes() {
    // Width far beyond the curvature radius: U-turn, hairpin curve, loop.
    let cases = [
        json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
               {"type": "curve", "x": 100, "y": 10, "cp1": {"x": 107, "y": 0}, "cp2": {"x": 107, "y": 10}},
               {"type": "line", "x": 0, "y": 10}]),
        json!([{"type": "move", "x": 0, "y": 0}, {"type": "curve", "x": 0, "y": 5, "cp1": {"x": 200, "y": -40}, "cp2": {"x": 200, "y": 45}}]),
        json!([{"type": "move", "x": 0, "y": 50}, {"type": "curve", "x": 100, "y": 50, "cp1": {"x": 150, "y": -30}, "cp2": {"x": -50, "y": -30}}]),
        json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 60, "y": 0},
               {"type": "curve", "x": 60, "y": 20, "cp1": {"x": 75, "y": 0}, "cp2": {"x": 75, "y": 20}},
               {"type": "line", "x": 50, "y": 20}]),
    ];
    for (k, pts) in cases.iter().enumerate() {
        for width in [10.0, 40.0, 90.0] {
            let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": "round", "join": "round"}]});
            let c = coverage(&v, 0.5);
            assert!(c.spikes == 0, "case {k} w {width}: spikes {}", c.spikes);
            assert!(
                (c.holes as f64) < 0.002 * c.inside as f64 + 4.0,
                "case {k} w {width}: holes {} of {}",
                c.holes,
                c.inside
            );
        }
    }
}

#[test]
fn json_envelope_and_samples() {
    let ok: Value = crate::value::from_str(&stroke_outline_json(
        r##"{"points": [{"type":"move","x":0,"y":0},{"type":"line","x":50,"y":0}], "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 6}], "samples": true, "junk": 1}"##,
    ))
    .unwrap();
    assert_eq!(ok["status"], "ok");
    assert_eq!(ok["fillRule"], "nonzero");
    assert_eq!(ok["maxHalfWidth"], 3.0);
    let s = &ok["samples"];
    assert_eq!(s["t"][0], 0.0);
    assert_eq!(s["subpathStarts"], json!([0]));
    let empty: Value = crate::value::from_str(&stroke_outline_json(r#"{"points": []}"#)).unwrap();
    assert_eq!(empty["status"], "empty");
    for bad in ["{", "[]", r#"{"points": {}}"#] {
        let e: Value = crate::value::from_str(&stroke_outline_json(bad)).unwrap();
        assert_eq!(e["status"], "error", "{bad}");
    }
    // width_at interpolates.
    let ws = WidthSamples {
        t: vec![0.0, 1.0],
        left: vec![1.0, 3.0],
        right: vec![2.0, 2.0],
        ..Default::default()
    };
    assert_eq!(ws.width_at(0.5), (2.0, 2.0));
    assert_eq!(ws.width_at(-1.0), (1.0, 2.0));
}

#[test]
fn vertex_counts_are_bounded() {
    // A 500-sample hand-drawn stroke, fitted, then outlined with a profile.
    let mut rng = Rng::new(42);
    let mut p = v2(0.0, 0.0);
    let mut dir: f64 = 0.0;
    let mut samples = Vec::new();
    for _ in 0..500 {
        dir += rng.f(-0.25, 0.25);
        p = p + v2(dir.cos(), dir.sin()) * 2.0;
        samples.push(p);
    }
    let fit = crate::path_geom::fit::fit_points(&samples, &Default::default());
    let pts: Vec<Value> = fit.points.iter().map(PathPoint::to_json).collect();
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 8, "cap": "round", "profile": {"preset": "eye"}}]});
    let o = outline(v, OutlineOptions::default());
    let (out, _) = shape(&o);
    assert!(all_finite(out));
    assert!(
        out.len() < 600,
        "{} outline points for {} fitted points",
        out.len(),
        fit.points.len()
    );
}

/// No panics / non-finite output on hostile input (wasm is panic = abort).
#[test]
fn stroke_outline_fuzz_no_panic() {
    let mut rng = Rng::new(0xDEAD);
    let nums: [f64; 11] = [0.0, -0.0, 1e-12, 1e-6, 0.5, 1.0, 7.0, 1e3, 1e8, 1e12, -5.0];
    for case in 0..1500 {
        let n = (rng.next_u64() % 8) as usize;
        let mut pts = Vec::new();
        for i in 0..n {
            let pick = |rng: &mut Rng| -> Value {
                match rng.next_u64() % 10 {
                    0 => json!(nums[(rng.next_u64() % nums.len() as u64) as usize]),
                    1 => json!(0),
                    2 => json!(5),
                    _ => json!(rng.f(-50.0, 50.0)),
                }
            };
            let ty = ["move", "line", "curve", "quadratic", "arc", "bogus"]
                [(rng.next_u64() % 6) as usize];
            let ty = if i == 0 && rng.next_u64() % 2 == 0 {
                "move"
            } else {
                ty
            };
            pts.push(json!({"type": ty, "x": pick(&mut rng), "y": pick(&mut rng),
                "cp1": {"x": pick(&mut rng), "y": pick(&mut rng)}, "cp2": {"x": pick(&mut rng), "y": pick(&mut rng)},
                "cp": {"x": pick(&mut rng), "y": pick(&mut rng)}, "rx": pick(&mut rng), "ry": pick(&mut rng),
                "largeArc": rng.next_u64() % 2 == 0, "sweep": rng.next_u64() % 2 == 0}));
        }
        let mut v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": nums[case % nums.len()].abs() + (case % 3) as f64, "cap": (["butt", "round", "square"][case % 3]), "join": (["miter", "round", "bevel"][(case / 3) % 3])}],
            "closed": case % 4 == 0, "cornerRadius": ([0.0, 3.0, 1e9][case % 3])});
        if case % 2 == 0 {
            v["strokes"][0]["profile"] = if case % 6 == 0 {
                json!({"preset": "eye", "flipAlong": true})
            } else {
                random_profile(&mut rng)
            };
        }
        if case % 3 == 0 {
            v["strokes"][0]["taper"] = json!({"units": (["px", "percent"][case % 2]), "startLength": (nums[case % nums.len()]), "endLength": 80, "startEase": -1, "endEase": 1});
        }
        if case % 5 == 0 {
            v["strokes"][0]["wave"] = json!({"amount": 3, "units": (["px", "cycles"][case % 2]), "wavelength": (nums[(case / 5) % nums.len()])});
        }
        let tol = [0.0, 0.25, 1e-9, 50.0][case % 4];
        let o = stroke_outline(
            &node(v.clone()),
            None,
            &OutlineOptions {
                tolerance: tol,
                miter_limit: [4.0, 1.0, 1e9][case % 3],
                samples: case % 2 == 0,
            },
        );
        match o {
            Err(e) => assert_ne!(e.code, "solver_failed", "case {case}: {v}"),
            Ok(Outline::Shape {
                points,
                bounds,
                max_half_width,
                ..
            }) => {
                assert!(all_finite(&points), "case {case}: {v}");
                assert!(
                    bounds.min_x.is_finite() && max_half_width.is_finite(),
                    "case {case}"
                );
                assert!(points.len() < 200_000, "case {case}");
            }
            Ok(Outline::Empty) => {}
        }
    }
}

/// Debug aid: `STROKE_CASE='<request json>' cargo test -p moda-scene --lib
/// stroke_outline::tests::render_case -- --ignored` writes
/// `target/stroke-outline-gallery/case.png` (outline dark, holes mid-grey,
/// centerline light).
#[test]
#[ignore]
fn render_case() {
    let Ok(src) = std::env::var("STROKE_CASE") else {
        return;
    };
    let v: Value = crate::value::from_str(&src).unwrap();
    let px: f64 = std::env::var("STROKE_PX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let tol: f64 = std::env::var("STROKE_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    let o = outline(
        v.clone(),
        OutlineOptions {
            samples: true,
            tolerance: tol,
            ..Default::default()
        },
    );
    let Outline::Shape {
        points,
        bounds,
        samples: Some(ws),
        ..
    } = &o
    else {
        panic!("empty")
    };
    let g = Grid::around(
        bounds.min_x,
        bounds.min_y,
        bounds.max_x,
        bounds.max_y,
        px,
        4.0,
    );
    let out = g.fill_nonzero(&polygons(points));
    let band = g.erode(&g.fill_union(&band_pieces(ws)), 2);
    let mut outer_pieces = band_pieces(ws);
    outer_pieces.extend(disc_pieces(ws, 1.5 * px));
    let outer = g.dilate(&g.fill_union(&outer_pieces), 2);
    let spikes: Vec<(usize, usize)> = (0..out.len())
        .filter(|&i| out[i] && !outer[i])
        .map(|i| (i % g.w, i / g.w))
        .collect();
    if !spikes.is_empty() {
        eprintln!(
            "spikes at {:?}",
            spikes
                .iter()
                .take(10)
                .map(|(x, y)| (g.x0 + *x as f64 * px, g.y0 + *y as f64 * px))
                .collect::<Vec<_>>()
        );
    }
    let holes: Vec<bool> = (0..out.len())
        .map(|i| (band[i] && !out[i]) || (out[i] && !outer[i]))
        .collect();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/stroke-outline-gallery");
    std::fs::create_dir_all(&dir).unwrap();
    g.write_png(&dir.join("case.png"), &out, Some(&holes));
    eprintln!(
        "holes {} / {}; {} path points",
        holes.iter().filter(|b| **b).count(),
        band.iter().filter(|b| **b).count(),
        points.len()
    );
}

/// `cargo test -p moda-scene --release --lib stroke_outline::tests::perf -- --ignored --nocapture`
#[test]
#[ignore]
fn perf_500_sample_stroke() {
    let mut rng = Rng::new(42);
    let mut p = v2(0.0, 0.0);
    let mut dir: f64 = 0.0;
    let mut raw = Vec::new();
    let mut pressure = Vec::new();
    for i in 0..500 {
        dir += rng.f(-0.25, 0.25);
        p = p + v2(dir.cos(), dir.sin()) * 2.0;
        raw.push(p);
        pressure.push(0.5 + 0.5 * (i as f64 / 40.0).sin());
    }
    // Best of 15 batches: the shared machine is noisy, the minimum is the cost.
    let time = |label: &str, f: &mut dyn FnMut()| {
        for _ in 0..20 {
            f();
        }
        let mut best = f64::INFINITY;
        for _ in 0..15 {
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            best = best.min(t0.elapsed().as_secs_f64() / 20.0);
        }
        eprintln!("{label}: {:.1} µs", best * 1e6);
    };
    time("fit_points 500 samples", &mut || {
        std::hint::black_box(crate::path_geom::fit::fit_points(&raw, &Default::default()));
    });
    let fit = crate::path_geom::fit::fit_points(&raw, &Default::default());
    let fitted: Vec<Value> = fit.points.iter().map(PathPoint::to_json).collect();
    let poly: Vec<Value> = raw
        .iter()
        .enumerate()
        .map(|(i, q)| json!({"type": if i == 0 { "move" } else { "line" }, "x": q.x, "y": q.y}))
        .collect();
    let ts: Vec<f64> = fit.sample_t.clone();
    let cases = [
        (
            "plain, fitted path",
            json!({"points": fitted, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 8, "cap": "round"}]}),
            None,
        ),
        (
            "eye + taper + wave, fitted path",
            json!({"points": fitted, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 8, "cap": "round", "profile": {"preset": "eye"}, "taper": {"startLength": 10, "endLength": 20}, "wave": {"amount": 0.2, "wavelength": 30}}]}),
            None,
        ),
        (
            "500-point polyline, markers",
            json!({"points": poly, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 8, "join": "round", "profile": {"markers": [{"id": "a", "t": 0, "left": 0.2, "right": 0.2}, {"id": "b", "t": 0.5, "left": 2, "right": 1}, {"id": "c", "t": 1, "left": 0, "right": 0}]}}]}),
            None,
        ),
        (
            "calligraphic + pressure + wobble",
            json!({"points": fitted, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1, "dynamic": {"wiggle": 2, "frequency": 2, "seed": 3}, "brush": {"brushId": "n", "seed": 1}}], "inkChannels": {"t": ts, "pressure": pressure}}),
            Some(
                json!({"id": "n", "type": "calligraphic", "angle": 30, "roundness": 0.3,
                "size": {"value": 10, "vary": "pressure", "min": 4, "max": 14}}),
            ),
        ),
    ];
    for (label, v, brush) in cases {
        let nd = node(v);
        let o = stroke_outline(&nd, brush.as_ref(), &OutlineOptions::default()).unwrap();
        let npts = shape(&o).0.len();
        time(
            &format!("stroke_outline: {label} ({npts} out pts)"),
            &mut || {
                std::hint::black_box(
                    stroke_outline(&nd, brush.as_ref(), &OutlineOptions::default()).unwrap(),
                );
            },
        );
        time(&format!("  same, tolerance 0 (no fit)"), &mut || {
            std::hint::black_box(
                stroke_outline(
                    &nd,
                    brush.as_ref(),
                    &OutlineOptions {
                        tolerance: 0.0,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        });
    }
}

/// Reference band for a uniform stroke of half-width `h` along the path's own
/// curves: normal quads between 4,000 samples per segment, plus a disc at each
/// joint (round-join truth). Independent of the outliner's sampling.
fn dense_truth(points: &Value, h: f64) -> Vec<Vec<V2>> {
    let mut cur = v2(0.0, 0.0);
    let mut segs: Vec<crate::path_geom::vec2::Cubic> = vec![];
    for p in points.as_array().unwrap() {
        let e = v2(p["x"].as_f64().unwrap(), p["y"].as_f64().unwrap());
        let cp = |k: &str| v2(p[k]["x"].as_f64().unwrap(), p[k]["y"].as_f64().unwrap());
        match p["type"].as_str().unwrap() {
            "line" => segs.push(crate::path_geom::vec2::Cubic::line(cur, e)),
            "curve" => segs.push(crate::path_geom::vec2::Cubic([
                cur,
                cp("cp1"),
                cp("cp2"),
                e,
            ])),
            _ => {}
        }
        cur = e;
    }
    let disc = |c: V2, r: f64| -> Vec<V2> {
        (0..64)
            .map(|k| {
                let a = k as f64 / 64.0 * std::f64::consts::TAU;
                c + v2(a.cos(), a.sin()) * r
            })
            .collect()
    };
    let mut out = vec![];
    for (k, c) in segs.iter().enumerate() {
        let mut prev: Option<(V2, V2)> = None;
        for i in 0..=4000 {
            let u = i as f64 / 4000.0;
            let p = c.eval(u);
            let Some(t) = c.tangent(u) else { continue };
            let nl = t.left_normal();
            if let Some((pp, pn)) = prev {
                out.push(vec![pp + pn * h, p + nl * h, p - nl * h, pp - pn * h]);
            }
            prev = Some((p, nl));
        }
        if k > 0 {
            out.push(disc(c.0[0], h));
        }
    }
    out
}

/// `(extra, missing)` in px² against `truth` (dilated / eroded by 1 cell).
fn area_diff(v: &Value, tol: f64, truth: &[Vec<V2>], px: f64) -> (f64, f64) {
    let o = outline(
        v.clone(),
        OutlineOptions {
            tolerance: tol,
            ..Default::default()
        },
    );
    let Outline::Shape {
        points, bounds: b, ..
    } = &o
    else {
        panic!("empty")
    };
    assert!(all_finite(points));
    let g = Grid::around(b.min_x, b.min_y, b.max_x, b.max_y, px, 6.0);
    let out = g.fill_nonzero(&polygons(points));
    let t = g.fill_union(truth);
    let (tdil, tero) = (g.dilate(&t, 1), g.erode(&t, 1));
    let extra = (0..out.len()).filter(|&i| out[i] && !tdil[i]).count() as f64 * px * px;
    let missing = (0..out.len()).filter(|&i| tero[i] && !out[i]).count() as f64 * px * px;
    (extra, missing)
}

#[test]
fn tight_smooth_turns_are_round_whatever_the_join() {
    // Smooth U-turns much tighter than the stroke: no miter arrowheads, no
    // bevel facets (review r18), and a hook at an open end has no hole (r19).
    let mut report = vec![];
    for gap in [4.0, 10.0, 20.0, 40.0] {
        for join in ["miter", "bevel", "round"] {
            let pts = json!([{"type":"move","x":0,"y":0},{"type":"curve","x":0,"y":gap,"cp1":{"x":150,"y":0},"cp2":{"x":150,"y":gap}}]);
            let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 40, "join": join}]});
            report.push((
                format!("u-turn gap {gap} {join}"),
                area_diff(&v, 0.25, &dense_truth(&pts, 20.0), 0.25),
            ));
        }
    }
    let hook = json!([{"type":"move","x":0,"y":0},{"type":"curve","x":100,"y":0,"cp1":{"x":60,"y":80},"cp2":{"x":104,"y":-3}}]);
    for tol in [0.0, 0.25] {
        let v = json!({"points": hook, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 40}]});
        report.push((
            format!("end hook tol {tol}"),
            area_diff(&v, tol, &dense_truth(&hook, 20.0), 0.25),
        ));
    }
    let lp = json!([{"type":"move","x":0,"y":60},{"type":"curve","x":120,"y":60,"cp1":{"x":170,"y":-20},"cp2":{"x":-50,"y":-20}}]);
    let v = json!({"points": lp, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 30}]});
    report.push((
        "self-crossing loop, miter".to_string(),
        area_diff(&v, 0.25, &dense_truth(&lp, 15.0), 0.25),
    ));
    let mut rng = Rng::new(0xC0FFEE);
    for case in 0..30 {
        let mut pts = vec![json!({"type": "move", "x": rng.f(0.0, 150.0), "y": rng.f(0.0, 150.0)})];
        for _ in 0..(1 + case % 3) {
            pts.push(json!({"type": "curve", "x": rng.f(0.0, 150.0), "y": rng.f(0.0, 150.0),
                "cp1": {"x": rng.f(0.0, 150.0), "y": rng.f(0.0, 150.0)}, "cp2": {"x": rng.f(0.0, 150.0), "y": rng.f(0.0, 150.0)}}));
        }
        let w = [8.0, 30.0, 80.0][case % 3];
        let pts = Value::Array(pts);
        let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": w, "join": "round"}]});
        report.push((
            format!("random cubics {case} w {w}"),
            area_diff(&v, 0.25, &dense_truth(&pts, w / 2.0), 0.25),
        ));
    }
    for (name, (e, m)) in &report {
        if *e > 0.5 || *m > 0.5 {
            eprintln!("{name}: extra {e:.2} px², missing {m:.2} px²");
        }
    }
    for (name, (e, m)) in &report {
        assert!(
            *e <= 4.0 && *m <= 4.0,
            "{name}: extra {e:.2} px², missing {m:.2} px²"
        );
    }
}

#[test]
fn width_step_at_a_corner_makes_one_join() {
    // Review r05: a discontinuous marker on the corner anchor (width 1 → 4 at
    // the L's vertex) draws ONE join at the outgoing width, not two skewed ones.
    let pts = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0}, {"type": "line", "x": 100, "y": 100}]);
    let markers = json!({"markers": [
        {"id": "a", "t": 0, "left": 1, "right": 1, "corner": true},
        {"id": "b", "t": 0.5, "left": 1, "right": 1, "anchor": 1, "corner": true},
        {"id": "c", "t": 0.5, "left": 4, "right": 4, "anchor": 1, "corner": true},
        {"id": "d", "t": 1, "left": 4, "right": 4, "corner": true}]});
    let rect =
        |x0: f64, y0: f64, x1: f64, y1: f64| vec![v2(x0, y0), v2(x1, y0), v2(x1, y1), v2(x0, y1)];
    let legs = vec![rect(0.0, -5.0, 100.0, 5.0), rect(80.0, 0.0, 120.0, 100.0)];
    // Outer corner (the path turns toward +y, so the outer side is -y / +x).
    let mut quarter = vec![v2(100.0, 0.0)];
    quarter.extend((0..=32).map(|k| {
        let a = -std::f64::consts::FRAC_PI_2 + k as f64 / 32.0 * std::f64::consts::FRAC_PI_2;
        v2(100.0 + 20.0 * a.cos(), 20.0 * a.sin())
    }));
    for (join, corner) in [
        ("round", quarter),
        ("miter", rect(100.0, -20.0, 120.0, 0.0)),
    ] {
        let v =
            json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "join": join, "profile": markers}]});
        let mut truth = legs.clone();
        truth.push(corner);
        let (e, m) = area_diff(&v, 0.05, &truth, 0.25);
        assert!(
            e <= 2.0 && m <= 2.0,
            "{join}: extra {e:.2} px², missing {m:.2} px²"
        );
    }
}

#[test]
fn sample_budget_bounds_memory() {
    // Review r20: 20,000 wiggly cubics must not explode the sample count (the
    // turn-based counts used to reach ~850k samples / 2.5 GB).
    let mut rng = Rng::new(1);
    let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
    for _ in 0..20_000 {
        pts.push(json!({"type": "curve", "x": rng.f(0.0, 500.0), "y": rng.f(0.0, 500.0),
            "cp1": {"x": rng.f(0.0, 500.0), "y": rng.f(0.0, 500.0)}, "cp2": {"x": rng.f(0.0, 500.0), "y": rng.f(0.0, 500.0)}}));
    }
    let o = outline(
        json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 3}]}),
        OutlineOptions {
            samples: true,
            ..Default::default()
        },
    );
    let Outline::Shape {
        samples: Some(ws),
        points,
        ..
    } = &o
    else {
        panic!()
    };
    assert!(
        ws.t.len() <= 2 * MAX_SAMPLES + 100,
        "{} samples",
        ws.t.len()
    );
    assert!(points.len() < 400_000, "{} outline points", points.len());
}

/// Loop cutting (offset.rs `trim_loops`), deterministically, beyond the random property test: on each
/// case the outline must cover the stroke band with no holes and nothing outside it (an uncut loop or a
/// wrong cut shows as one or the other under the nonzero rule), at widths from modest to far beyond
/// the curvature. Coverage alone can't see a missing cut (an uncut overlap still fills the band), so
/// each case also asserts, through a test-only counter, that `trim_loops` actually cut.
///
/// How many loops the outline of `v` cut.
fn loop_cuts(v: &Value) -> usize {
    super::offset::LOOP_CUTS.with(|c| c.set(0));
    let _ = outline(v.clone(), OutlineOptions::default());
    super::offset::LOOP_CUTS.with(|c| c.get())
}

fn assert_clean(name: &str, v: Value) {
    let c = coverage(&v, 0.5);
    assert!(c.inside > 0, "{name}: nothing drawn\n{v}");
    assert!(c.spikes == 0, "{name}: {} spike pixels\n{v}", c.spikes);
    assert!(
        (c.holes as f64) < 0.002 * c.inside as f64 + 4.0,
        "{name}: {} hole pixels of {}\n{v}",
        c.holes,
        c.inside
    );
}

fn p(x: f64, y: f64) -> Value {
    json!({"x": x, "y": y})
}

#[test]
fn loop_cutting_figure_eight() {
    // One self-crossing cubic (open), and a closed figure eight made of two lobes.
    let open = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 100, "y": 0, "cp1": p(160.0, 120.0), "cp2": p(-60.0, 120.0)}]);
    let closed = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 0, "y": 0, "cp1": p(120.0, -80.0), "cp2": p(120.0, 80.0)},
        {"type": "curve", "x": 0, "y": 0, "cp1": p(-120.0, 80.0), "cp2": p(-120.0, -80.0)}]);
    for width in [6.0, 24.0, 60.0] {
        for (name, pts, is_closed) in [
            ("open figure eight", &open, false),
            ("closed figure eight", &closed, true),
        ] {
            for (join, cap) in [("round", "round"), ("miter", "butt"), ("bevel", "square")] {
                let v = json!({"points": pts, "closed": is_closed, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": cap, "join": join}]});
                if join == "round" {
                    assert_clean(&format!("{name} w {width}"), v.clone());
                }
                if !is_closed && width >= 24.0 {
                    // The crossing's inner fold is cut, whatever the join.
                    assert!(
                        loop_cuts(&v) >= 1,
                        "{name} w {width} {join}: the fold was not cut"
                    );
                }
                // Every join and cap: a finite, non-empty outline.
                let o = outline(v, OutlineOptions::default());
                assert!(all_finite(shape(&o).0), "{name} w {width} {join}");
            }
        }
    }
}

#[test]
fn loop_cutting_cusp() {
    // A cubic whose control polygon crosses so the curve turns back on itself in a point (a cusp):
    // the offset on the inside folds into a swallowtail that must be cut.
    let cusp = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 100, "y": 0, "cp1": p(100.0, 75.0), "cp2": p(0.0, 75.0)}]);
    // The same with a width profile (the fold moves with the width).
    for width in [8.0, 30.0, 70.0] {
        let v =
            json!({"points": cusp, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": "round", "join": "round"}]});
        assert!(
            loop_cuts(&v) >= 1,
            "cusp w {width}: the swallowtail was not cut"
        );
        assert_clean(&format!("cusp w {width}"), v);
        assert_clean(
            &format!("cusp eye w {width}"),
            json!({"points": cusp, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": "round", "join": "round", "profile": {"preset": "eye"}}]}),
        );
    }
}

#[test]
fn loop_cutting_tight_inner_loop_at_high_width() {
    // A path that swings through a small loop (radius ~10) drawn far wider than the loop: the inside
    // offset of the loop collapses entirely and is cut away.
    let looped = json!([{"type": "move", "x": 0, "y": 40},
        {"type": "line", "x": 80, "y": 40},
        {"type": "curve", "x": 80, "y": 20, "cp1": p(106.0, 40.0), "cp2": p(106.0, 20.0)},
        {"type": "curve", "x": 80, "y": 40, "cp1": p(54.0, 20.0), "cp2": p(54.0, 40.0)},
        {"type": "line", "x": 160, "y": 40}]);
    for width in [10.0, 40.0, 80.0, 120.0] {
        assert_clean(
            &format!("inner loop w {width}"),
            json!({"points": looped, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": "round", "join": "round"}]}),
        );
    }
    // Near the loop's own size the collapsed inside is cut (far wider, the band swallows it whole).
    let at_loop_size =
        json!({"points": looped, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 60, "cap": "round", "join": "round"}]});
    assert!(
        loop_cuts(&at_loop_size) >= 1,
        "inner loop w 60: nothing cut"
    );
    // With a taper ending inside the loop.
    assert_clean(
        "inner loop tapered",
        json!({"points": looped, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 60, "cap": "round", "join": "round", "taper": {"units": "percent", "startLength": 0, "endLength": 60}}]}),
    );
}

#[test]
fn loop_cutting_closed_paths() {
    // Closed rings whose inner offset inverts (width beyond the inner radius / half the side), a thin
    // closed sliver, and a closed ring with a profile (periodic markers across the seam).
    let square = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 20, "y": 0},
        {"type": "line", "x": 20, "y": 20}, {"type": "line", "x": 0, "y": 20}]);
    let k = 0.5523 * 10.0;
    let circle = json!([{"type": "move", "x": 10, "y": 0},
        {"type": "curve", "x": 0, "y": 10, "cp1": p(10.0, k), "cp2": p(k, 10.0)},
        {"type": "curve", "x": -10, "y": 0, "cp1": p(-k, 10.0), "cp2": p(-10.0, k)},
        {"type": "curve", "x": 0, "y": -10, "cp1": p(-10.0, -k), "cp2": p(-k, -10.0)},
        {"type": "curve", "x": 10, "y": 0, "cp1": p(k, -10.0), "cp2": p(10.0, -k)}]);
    let sliver = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 200, "y": 0}, {"type": "line", "x": 200, "y": 4}]);
    // The square's inner sides meet at their true corners (Illustrator's inner square), not through the pivots.
    let v =
        json!({"points": square, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 6, "join": "miter"}]});
    let Outline::Shape { points, .. } = outline(v, OutlineOptions::default()) else {
        panic!("empty")
    };
    let has = |x: f64, y: f64| {
        points.iter().any(|q| match q {
            PathPoint::Move { x: a, y: b } | PathPoint::Line { x: a, y: b } => {
                (a - x).abs() < 1e-6 && (b - y).abs() < 1e-6
            }
            _ => false,
        })
    };
    for (x, y) in [(3.0, 3.0), (17.0, 3.0), (17.0, 17.0), (3.0, 17.0)] {
        assert!(has(x, y), "closed square w 6: no inner corner at ({x}, {y})");
    }
    for (x, y) in [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)] {
        assert!(!has(x, y), "closed square w 6: routed through the pivot ({x}, {y})");
    }
    for width in [4.0, 16.0, 40.0, 60.0] {
        for (name, pts) in [
            ("square", &square),
            ("circle", &circle),
            ("sliver", &sliver),
        ] {
            assert_clean(
                &format!("closed {name} w {width}"),
                json!({"points": pts, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "join": "round"}]}),
            );
        }
    }
    assert_clean(
        "closed circle, profiled",
        json!({"points": circle, "closed": true, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 30, "join": "round", "profile": {"markers": [{"id": "a", "t": 0, "left": 1, "right": 1}, {"id": "b", "t": 0.5, "left": 2.5, "right": 0.3}, {"id": "c", "t": 1, "left": 1, "right": 1}]}}]}),
    );
}

#[test]
fn loop_cutting_leaves_smooth_strokes_alone() {
    // No false cuts: a straight line and a gentle arc have no loops to cut at any width.
    let straight = json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 200, "y": 0}]);
    let gentle = json!([{"type": "move", "x": 0, "y": 0},
        {"type": "curve", "x": 200, "y": 0, "cp1": p(60.0, -40.0), "cp2": p(140.0, -40.0)}]);
    for (name, pts) in [("straight", &straight), ("gentle arc", &gentle)] {
        for width in [6.0, 24.0, 60.0, 120.0] {
            for join in ["round", "miter"] {
                let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": "round", "join": join}]});
                assert_eq!(loop_cuts(&v), 0, "{name} w {width} {join}");
            }
        }
    }
}

#[test]
fn node_miter_limit_wins_over_the_default() {
    let v = |limit: Option<f64>| {
        let mut n = json!({"points": [{"type":"move","x":100.6,"y":140},{"type":"line","x":120,"y":30},{"type":"line","x":139.4,"y":140}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 14, "join": "miter", "profile": {"preset": "uniform"}}]});
        if let Some(l) = limit {
            n["strokes"][0]["miterLimit"] = json!(l);
        }
        match outline(n, OutlineOptions::default()) {
            Outline::Shape { bounds, .. } => bounds.min_y,
            _ => panic!(),
        }
    };
    assert!(v(None) > 20.0, "default 4 bevels a 20° apex");
    assert!(v(Some(10.0)) < 0.0, "miterLimit 10 miters it (tip ≈ −10.3)");
}

#[test]
fn request_stroke_miter_limit_wins_over_the_option() {
    // The JSON entry point builds its node from NODE_KEYS only: the stroke's
    // own `miterLimit` travels inside `strokes`, and the request's top-level
    // `miterLimit` is only the option (used when the stroke has none).
    assert!(super::NODE_KEYS.contains(&"strokes"));
    let min_y = |stroke_limit: Option<f64>| {
        let mut req = json!({"points": [{"type":"move","x":100.6,"y":140},{"type":"line","x":120,"y":30},{"type":"line","x":139.4,"y":140}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 14, "join": "miter"}],
            "miterLimit": 4});
        if let Some(l) = stroke_limit {
            req["strokes"][0]["miterLimit"] = json!(l);
        }
        let out: Value = crate::value::from_str(&stroke_outline_json(&req.to_string())).unwrap();
        out["bounds"]["minY"].as_f64().unwrap()
    };
    assert!(min_y(None) > 20.0, "option 4 bevels a 20° apex");
    assert!(min_y(Some(10.0)) < 0.0, "the stroke's miterLimit 10 miters it");
}
