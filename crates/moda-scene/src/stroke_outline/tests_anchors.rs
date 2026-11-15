//! Outline anchor counts and accuracy: a round-capped, round-joined stroke's outline uses about as few anchors as
//! Illustrator's Outline Stroke (a stroked circle: at most 4 cubics per side; the lucide bell ~30 anchors) while
//! it stays within the fit tolerance of the true edge — the points `width / 2` from the centreline — and covers
//! exactly the true stroke away from that edge.
use super::centerline::{node_segments, Seg};
use super::raster::Grid;
use super::tests::node;
use super::*;
use crate::path_geom::vec2::{v2, Cubic, V2};
use crate::value::json;

/// lucide bell, house, settings and heart as the SVG importer brings them in (arcs, cubics, lines).
const LUCIDE: &str = include_str!("lucide_icons.json");

/// The editor's `outlineFitTolerance`: 2.5% of the width, 0.01–0.25.
fn tol_for(width: f64) -> f64 {
    (width * 0.025).clamp(0.01, 0.25)
}

/// The centreline as a polyline per subpath, `per` points per cubic (closed rings repeat their first point).
fn centreline(v: &Value, per: usize) -> Vec<Vec<V2>> {
    let pts = v["points"].as_array().cloned().unwrap_or_default();
    let closed = v["closed"].as_bool().unwrap_or(false);
    let r = v["cornerRadius"].as_f64().unwrap_or(0.0);
    let (paths, _, _) = node_segments(&pts, closed, &crate::path_geom::round_corners::PathCorners::uniform(r));
    paths
        .iter()
        .map(|sp| {
            let mut out = vec![sp.start];
            for s in &sp.segs {
                match s {
                    Seg::Line(_, b) => out.push(*b),
                    Seg::Cubic(c) => {
                        (1..=per).for_each(|i| out.push(c.eval(i as f64 / per as f64)))
                    }
                }
            }
            if sp.closed {
                out.push(sp.start);
            }
            out
        })
        .collect()
}

fn dist_to_polylines(p: V2, lines: &[Vec<V2>]) -> f64 {
    let mut best = f64::INFINITY;
    for l in lines {
        if l.len() == 1 {
            best = best.min(p.dist(l[0]));
        }
        for w in l.windows(2) {
            let d = w[1] - w[0];
            let len2 = d.dot(d);
            let t = if len2 > 0.0 {
                ((p - w[0]).dot(d) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            best = best.min(p.dist(w[0] + d * t));
        }
    }
    best
}

#[derive(Debug)]
struct Measure {
    /// `points.len()` of the outline: what the editor reports as its anchors (a `move` per ring, and a closed
    /// ring's last segment ends back on it).
    points: usize,
    subpaths: usize,
    /// Farthest any point of the outline strays OUTSIDE the true stroke (distance to the centreline − half width).
    stray: f64,
    /// Farthest any point of the outline is from the true edge either way (points inside the stroke by more than
    /// the tolerance are overlap, invisible under the nonzero fill).
    edge: f64,
    /// Grid cells more than the tolerance from the true edge whose nonzero coverage differs from the true stroke.
    coverage_misses: usize,
}

fn measure(v: &Value) -> Measure {
    let width = v["strokes"][0]["width"].as_f64().unwrap();
    let tol = tol_for(width);
    let half = width / 2.0;
    let opts = OutlineOptions {
        tolerance: tol,
        ..OutlineOptions::default()
    };
    let o = stroke_outline(&node(v.clone()), None, &opts).expect("outline");
    let Outline::Shape {
        points,
        subpath_count,
        bounds,
        ..
    } = o
    else {
        panic!("empty")
    };
    let fine = centreline(v, 400);
    let (mut stray, mut edge): (f64, f64) = (0.0, 0.0);
    let mut cur = V2::default();
    let mut check = |q: V2| {
        let d = dist_to_polylines(q, &fine) - half;
        stray = stray.max(d);
        edge = edge.max(d.abs());
    };
    for p in &points {
        match p {
            PathPoint::Move { x, y } => cur = v2(*x, *y),
            PathPoint::Line { x, y } => {
                let b = v2(*x, *y);
                (0..=16).for_each(|i| check(cur.lerp(b, i as f64 / 16.0)));
                cur = b;
            }
            PathPoint::Curve { x, y, cp1, cp2 } => {
                let c = Cubic([cur, v2(cp1.0, cp1.1), v2(cp2.0, cp2.1), v2(*x, *y)]);
                (0..=32).for_each(|i| check(c.eval(i as f64 / 32.0)));
                cur = v2(*x, *y);
            }
            _ => {}
        }
    }
    let coarse = centreline(v, 96);
    let px = (width / 10.0).max(0.02);
    let g = Grid::around(
        bounds.min_x,
        bounds.min_y,
        bounds.max_x,
        bounds.max_y,
        px,
        2.0 * px,
    );
    let filled = g.fill_nonzero(&polygons(&points));
    let mut coverage_misses = 0;
    for row in 0..g.h {
        for col in 0..g.w {
            let q = v2(
                g.x0 + (col as f64 + 0.5) * px,
                g.y0 + (row as f64 + 0.5) * px,
            );
            let d = dist_to_polylines(q, &coarse) - half;
            if d.abs() > tol && (d < 0.0) != filled[row * g.w + col] {
                coverage_misses += 1;
            }
        }
    }
    Measure {
        points: points.len(),
        subpaths: subpath_count,
        stray,
        edge,
        coverage_misses,
    }
}

/// The outline's rings flattened finely (64 chords a cubic: the chords' own sag stays far under the tolerance).
fn polygons(points: &[PathPoint]) -> Vec<Vec<V2>> {
    let mut out: Vec<Vec<V2>> = Vec::new();
    let mut cur = V2::default();
    for p in points {
        match p {
            PathPoint::Move { x, y } => {
                cur = v2(*x, *y);
                out.push(vec![cur]);
            }
            PathPoint::Line { x, y } => {
                cur = v2(*x, *y);
                out.last_mut().unwrap().push(cur);
            }
            PathPoint::Curve { x, y, cp1, cp2 } => {
                let c = Cubic([cur, v2(cp1.0, cp1.1), v2(cp2.0, cp2.1), v2(*x, *y)]);
                (1..=64).for_each(|i| out.last_mut().unwrap().push(c.eval(i as f64 / 64.0)));
                cur = v2(*x, *y);
            }
            _ => {}
        }
    }
    out
}

fn stroke(width: f64) -> Value {
    json!([{"id": "s", "paint": {"type": "solid", "color": "#000"}, "width": width, "cap": "round", "join": "round"}])
}

/// A circle as the ellipse tool draws it: four κ-handled quarter cubics.
fn circle(r: f64, width: f64) -> Value {
    let k = 0.5522847498 * r;
    json!({"points": [
        {"type": "move", "x": r, "y": 0.0},
        {"type": "curve", "cp1": {"x": r, "y": k}, "cp2": {"x": k, "y": r}, "x": 0.0, "y": r},
        {"type": "curve", "cp1": {"x": -k, "y": r}, "cp2": {"x": -r, "y": k}, "x": -r, "y": 0.0},
        {"type": "curve", "cp1": {"x": -r, "y": -k}, "cp2": {"x": -k, "y": -r}, "x": 0.0, "y": -r},
        {"type": "curve", "cp1": {"x": k, "y": -r}, "cp2": {"x": r, "y": -k}, "x": r, "y": 0.0}
    ], "closed": true, "strokes": stroke(width)})
}

fn rounded_rect(w: f64, h: f64, r: f64, width: f64) -> Value {
    json!({"points": [
        {"type": "move", "x": 0.0, "y": 0.0}, {"type": "line", "x": w, "y": 0.0},
        {"type": "line", "x": w, "y": h}, {"type": "line", "x": 0.0, "y": h}
    ], "closed": true, "cornerRadius": r, "strokes": stroke(width)})
}

/// The icon's paths with lucide's own stroke: 2 wide, round caps and joins.
fn lucide(name: &str) -> Vec<Value> {
    let all: Value = crate::value::from_str(LUCIDE).unwrap();
    all[name]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let mut p = p.clone();
            p["strokes"] = stroke(2.0);
            p
        })
        .collect()
}

/// Total outline points over the paths, after checking each one's accuracy.
fn check_accurate(name: &str, paths: &[Value]) -> usize {
    let mut total = 0;
    for (i, v) in paths.iter().enumerate() {
        let tol = tol_for(v["strokes"][0]["width"].as_f64().unwrap());
        let m = measure(v);
        println!("{name}#{i}: {m:?}");
        assert!(
            m.stray <= tol,
            "{name}#{i} strays {} outside the stroke (tolerance {tol})",
            m.stray
        );
        assert_eq!(
            m.coverage_misses, 0,
            "{name}#{i} covers the wrong cells: {m:?}"
        );
        total += m.points;
    }
    total
}

#[test]
fn stroked_circle_outlines_as_at_most_four_cubics_per_side() {
    // A big one too: r400 took a fifth cubic a side (edge 0.21) while chord-length parameters alone judged its
    // quarters, 2 px off before Newton; reparameterised, a quarter fits within 0.09 and the edge is 0.07.
    for (r, w, most) in [
        (10.0, 2.0, 10),
        (100.0, 8.0, 10),
        (3.0, 1.0, 10),
        (24.0, 2.0, 10),
        (400.0, 40.0, 10),
    ] {
        let m = measure(&circle(r, w));
        let tol = tol_for(w);
        println!("circle r{r} w{w}: {m:?}");
        assert_eq!(m.subpaths, 2);
        // Two rings of `move` + ≤ 4 cubics (Illustrator: 4 anchors a side).
        assert!(m.points <= most, "circle r{r} w{w}: {m:?}");
        assert!(
            m.edge <= tol,
            "circle r{r} w{w} strays {} from its edge (tolerance {tol})",
            m.edge
        );
        assert_eq!(m.coverage_misses, 0);
    }
}

#[test]
fn rounded_rect_outline_is_its_sides_and_corner_arcs() {
    let m = measure(&rounded_rect(20.0, 14.0, 3.0, 2.0));
    println!("{m:?}");
    // Per ring: a move, 4 sides and 4 corner arcs.
    assert_eq!(m.points, 18, "{m:?}");
    assert!(m.edge <= 0.05);
    assert_eq!(m.coverage_misses, 0);
}

#[test]
fn lucide_bell_outlines_in_about_as_many_anchors_as_illustrator() {
    // Was 289 (56 of its 58 rings were slivers the radius-1 arcs clamped by rounding noise); Illustrator gives ~30.
    let bell = lucide("bell");
    let total = check_accurate("bell", &bell);
    assert!(total <= 30, "bell outlines to {total} points");
    assert_eq!(
        measure(&bell[1]).subpaths,
        2,
        "the body is one outer and one inner ring"
    );
    for v in &bell {
        assert!(measure(v).edge <= 0.05);
    }
    // The body's flat bottom (y 9 under a 2-wide stroke: y 10 outside) is one straight line, not a cubic bowing to
    // take in the round corners beside it.
    let o = stroke_outline(
        &node(bell[1].clone()),
        None,
        &OutlineOptions {
            tolerance: 0.05,
            ..OutlineOptions::default()
        },
    );
    let Ok(Outline::Shape { points, .. }) = o else {
        panic!("empty")
    };
    let mut cur = (0.0, 0.0);
    let mut flat = false;
    for p in &points {
        match p {
            PathPoint::Move { x, y } => cur = (*x, *y),
            PathPoint::Line { x, y } => {
                flat |= (cur.1 - 10.0).abs() < 1e-6
                    && (y - 10.0).abs() < 1e-6
                    && (x - cur.0).abs() > 15.0;
                cur = (*x, *y);
            }
            PathPoint::Curve { x, y, .. } => cur = (*x, *y),
            _ => {}
        }
    }
    assert!(flat, "the bottom edge is a line");
}

#[test]
fn lucide_icons_outline_compactly_and_accurately() {
    // Before: house 121, settings 60.
    let house = check_accurate("house", &lucide("house"));
    assert!(house <= 40, "house outlines to {house} points");
    let settings = check_accurate("settings", &lucide("settings"));
    assert!(settings <= 50, "settings outlines to {settings} points");
    // The heart's inner corners (the dip at the top, the tip at the bottom) meet where the inner offsets cross, as
    // Illustrator's do: no point of the outline strays from the edge (it was 1.0, through the centreline).
    // Each ring's seam (the centreline starts at 17,11) is one anchor: the straight side runs right into it, with no
    // 0.25 line of its own in between. Before: 24, then 22.
    let heart = lucide("heart");
    let n = check_accurate("heart", &heart);
    assert!(n <= 20, "heart outlines to {n} points");
    let m = measure(&heart[0]);
    assert!(m.edge <= tol_for(2.0), "heart: {m:?}");
}
