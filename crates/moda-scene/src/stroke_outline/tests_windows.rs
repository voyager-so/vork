//! Arc-length windows (trim / dash on outlines, plan P8).
use super::raster::polygons;
use super::tests::{node, shape};
use super::*;
use crate::path_geom::fit::tests::Rng;
use crate::path_geom::vec2::V2;
use crate::value::json;

fn line(len: f64) -> Value {
    json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": len, "y": 0}])
}

fn win(subpath: usize, start: f64, end: f64, retaper: bool) -> ArcWindow {
    ArcWindow {
        subpath,
        start,
        end,
        taper_extent: retaper.then_some((start, end)),
    }
}

fn run(v: Value, brush: Option<Value>, ws: &[ArcWindow]) -> Outline {
    stroke_outline_windows(&node(v), brush.as_ref(), &OutlineOptions::default(), ws)
        .expect("outline")
}

/// (min y, max y) of the outline at `x`.
fn extent_at_x(polys: &[Vec<V2>], x: f64) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
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

fn bounds(o: &Outline) -> Bounds {
    match o {
        Outline::Shape { bounds, .. } => *bounds,
        Outline::Empty => panic!("empty"),
    }
}

fn wedge() -> Value {
    // Width grows linearly from 0 to 2× over the line.
    json!({"markers": [{"id": "a", "t": 0, "left": 0.0, "right": 0.0, "corner": true},
                       {"id": "b", "t": 1, "left": 2.0, "right": 2.0, "corner": true}]})
}

#[test]
fn window_cuts_the_outline_with_butt_caps() {
    let v = json!({"points": line(100.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10}]});
    let o = run(v, None, &[win(0, 25.0, 75.0, false)]);
    let b = bounds(&o);
    assert!(
        (b.min_x - 25.0).abs() < 1e-6 && (b.max_x - 75.0).abs() < 1e-6,
        "{b:?}"
    );
    assert!((b.max_y - 5.0).abs() < 1e-6);
}

#[test]
fn dash_windows_keep_full_path_widths() {
    let v = json!({"points": line(100.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "profile": wedge()}]});
    let full = stroke_outline(&node(v.clone()), None, &OutlineOptions::default()).unwrap();
    let full_polys = polygons(shape(&full).0);
    let o = run(
        v,
        None,
        &[win(0, 10.0, 30.0, false), win(0, 60.0, 90.0, false)],
    );
    let (pts, n) = shape(&o);
    assert_eq!(n, 2);
    let polys = polygons(pts);
    for x in [15.0, 25.0, 65.0, 85.0] {
        let (lo, hi) = extent_at_x(&polys, x).unwrap();
        let (flo, fhi) = extent_at_x(&full_polys, x).unwrap();
        assert!(
            (lo - flo).abs() < 0.3 && (hi - fhi).abs() < 0.3,
            "x {x}: {lo},{hi} vs {flo},{fhi}"
        );
    }
    assert!(extent_at_x(&polys, 45.0).is_none());
}

#[test]
fn trim_windows_retaper_to_their_own_ends() {
    let taper = json!({"units": "px", "startLength": 0, "endLength": 20, "startWidth": 1, "endWidth": 0,
                       "startEase": 0, "endEase": 0});
    let v = json!({"points": line(100.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "taper": taper}]});
    // Trim (retaper): the tip at 50 tapers to a point.
    let o = run(v.clone(), None, &[win(0, 0.0, 50.0, true)]);
    let polys = polygons(shape(&o).0);
    let (lo, hi) = extent_at_x(&polys, 49.0).unwrap();
    assert!(hi - lo < 1.5, "trim tip {lo} {hi}");
    let (lo, hi) = extent_at_x(&polys, 20.0).unwrap();
    assert!((hi - lo - 10.0).abs() < 0.3, "trim body {lo} {hi}");
    // Dash (no retaper): the same window is a slice of the untapered middle.
    let o = run(v.clone(), None, &[win(0, 0.0, 50.0, false)]);
    let polys = polygons(shape(&o).0);
    let (lo, hi) = extent_at_x(&polys, 49.0).unwrap();
    assert!((hi - lo - 10.0).abs() < 0.3, "dash end {lo} {hi}");
    // A dash of the trimmed run [0, 50] tapers with the run, not per dash.
    let dash = ArcWindow {
        subpath: 0,
        start: 20.0,
        end: 49.5,
        taper_extent: Some((0.0, 50.0)),
    };
    let o = run(v, None, &[dash]);
    let polys = polygons(shape(&o).0);
    let (lo, hi) = extent_at_x(&polys, 25.0).unwrap();
    assert!((hi - lo - 10.0).abs() < 0.6, "dash body {lo} {hi}");
    let (lo, hi) = extent_at_x(&polys, 49.0).unwrap();
    assert!(hi - lo < 1.5, "dash at the run's tip {lo} {hi}");
}

fn square(closed: bool) -> Value {
    json!({"points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                      {"type": "line", "x": 100, "y": 100}, {"type": "line", "x": 0, "y": 100}],
           "closed": closed, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 6, "profile": {"preset": "eye"}}]})
}

#[test]
fn closed_windows_ring_and_seam() {
    let spans = subpath_spans(&node(square(true)));
    assert_eq!(spans.len(), 1);
    assert!((spans[0].length - 400.0).abs() < 1e-6 && spans[0].closed);
    // Whole ring: two rings (outer + inner), like the untrimmed outline.
    let o = run(square(true), None, &[win(0, 0.0, 400.0, false)]);
    assert_eq!(shape(&o).1, 2);
    // Through the seam: one open piece around (0,0).
    let o = run(square(true), None, &[win(0, 350.0, 450.0, false)]);
    assert_eq!(shape(&o).1, 1);
    let b = bounds(&o);
    assert!(
        b.min_x < 0.0 && b.max_x > 45.0 && b.max_y > 45.0 && b.max_y < 60.0,
        "{b:?}"
    );
    // Open path: windows clamp to the ends.
    let o = run(square(false), None, &[win(0, -10.0, 1000.0, false)]);
    assert_eq!(shape(&o).1, 1);
}

#[test]
fn nib_windows_get_nib_caps() {
    let brush = json!({"id": "nib", "name": "Nib", "type": "calligraphic", "angle": {"value": 45},
                       "roundness": {"value": 0.2}, "size": {"value": 1}});
    let v = json!({"points": line(100.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "cap": "butt", "brush": {"brushId": "nib"}}]});
    let o = run(v, Some(brush), &[win(0, 30.0, 60.0, false)]);
    let b = bounds(&o);
    // A butt cap would stop at 30 / 60; the nib's half-ellipse reaches past.
    assert!(b.min_x < 29.0 && b.max_x > 61.0, "{b:?}");
}

#[test]
fn empty_and_hostile_windows_never_panic() {
    let v = json!({"points": line(100.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "profile": wedge(), "dynamic": {"frequency": 5, "wiggle": 3, "smoothen": 0.5}}]});
    assert!(matches!(run(v.clone(), None, &[]), Outline::Empty));
    assert!(matches!(
        run(v.clone(), None, &[win(3, 0.0, 10.0, false)]),
        Outline::Empty
    ));
    assert!(matches!(
        run(v.clone(), None, &[win(0, 50.0, 50.0, false)]),
        Outline::Empty
    ));
    let mut rng = Rng::new(7);
    for _ in 0..300 {
        let a = rng.f(-150.0, 250.0);
        let b = a + rng.f(-50.0, 250.0);
        let ws = [
            win(0, a, b, rng.f(0.0, 1.0) < 0.5),
            win(0, f64::NAN, b, false),
        ];
        let _ = run(v.clone(), None, &ws);
        let _ = run(square(true), None, &ws);
    }
}

#[test]
fn open_windows_clamp_without_moving_the_taper_extent() {
    let taper = json!({"units": "px", "startLength": 0, "endLength": 20, "startWidth": 1, "endWidth": 0,
                       "startEase": 0, "endEase": 0});
    let v = json!({"points": line(100.0), "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 10, "taper": taper}]});
    // A trim run starting before the path (clamped to 0) keeps its extent [-10, 50]:
    // the tip at 50 is still pointed.
    let w = ArcWindow {
        subpath: 0,
        start: -10.0,
        end: 50.0,
        taper_extent: Some((-10.0, 50.0)),
    };
    let o = run(v, None, &[w]);
    let polys = polygons(shape(&o).0);
    let (lo, hi) = extent_at_x(&polys, 49.0).unwrap();
    assert!(hi - lo < 1.5, "tip {lo} {hi}");
}

#[test]
fn many_dashes_on_a_long_path_stay_fast() {
    let mut pts = vec![json!({"type": "move", "x": 0, "y": 0})];
    for i in 1..400 {
        pts.push(
            json!({"type": "line", "x": i as f64 * 10.0, "y": if i % 2 == 0 { 0.0 } else { 8.0 }}),
        );
    }
    let v = json!({"points": pts, "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4, "profile": {"preset": "eye"}}]});
    let n = node(v);
    let len = subpath_spans(&n)[0].length;
    let ws: Vec<ArcWindow> = (0..(len / 4.0) as usize)
        .map(|i| ArcWindow {
            subpath: 0,
            start: i as f64 * 4.0,
            end: i as f64 * 4.0 + 2.0,
            taper_extent: None,
        })
        .collect();
    let t = std::time::Instant::now();
    let o = stroke_outline_windows(&n, None, &OutlineOptions::default(), &ws).unwrap();
    let ms = t.elapsed().as_millis();
    assert!(shape(&o).1 > 500);
    assert!(ms < 1500, "{} dashes took {ms} ms", ws.len());
}
