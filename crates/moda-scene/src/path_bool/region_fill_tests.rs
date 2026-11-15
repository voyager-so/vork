//! Region fill (`region_fill.rs`): nested regions, regions formed by open
//! strokes, gaps just under and over the tolerance, self-intersections,
//! compound holes, and the 500-path perf budget.

use std::time::Instant;

use crate::value::{json, Value};

use super::region_fill::{parse_source, Arrangement, Drawing, FilledRegion, RegionAt, RegionSource};
use super::region_fill_json;

fn stroke(width: f64, cap: &str) -> Value {
    json!([{ "id": "s", "paint": { "type": "solid", "color": "#000000" }, "width": width, "cap": cap }])
}

fn polyline(pts: &[(f64, f64)]) -> Value {
    Value::Array(
        pts.iter()
            .enumerate()
            .map(|(i, (x, y))| json!({ "type": if i == 0 { "move" } else { "line" }, "x": x, "y": y }))
            .collect(),
    )
}

/// A circle of four cubics.
fn circle(cx: f64, cy: f64, r: f64) -> Vec<Value> {
    let k = 0.552_284_749_8 * r;
    vec![
        json!({ "type": "move", "x": cx + r, "y": cy }),
        json!({ "type": "curve", "x": cx, "y": cy + r, "cp1": { "x": cx + r, "y": cy + k }, "cp2": { "x": cx + k, "y": cy + r } }),
        json!({ "type": "curve", "x": cx - r, "y": cy, "cp1": { "x": cx - k, "y": cy + r }, "cp2": { "x": cx - r, "y": cy + k } }),
        json!({ "type": "curve", "x": cx, "y": cy - r, "cp1": { "x": cx - r, "y": cy - k }, "cp2": { "x": cx - k, "y": cy - r } }),
        json!({ "type": "curve", "x": cx + r, "y": cy, "cp1": { "x": cx + k, "y": cy - r }, "cp2": { "x": cx + r, "y": cy - k } }),
    ]
}

fn line(a: (f64, f64), b: (f64, f64), width: f64) -> Value {
    json!({ "points": polyline(&[a, b]), "strokes": stroke(width, "butt") })
}

fn src(v: &Value) -> RegionSource {
    parse_source(v.as_object().unwrap()).expect("source parses")
}

fn drawing(sources: &[Value], gap: f64, overlap: f64) -> Drawing {
    let s: Vec<RegionSource> = sources.iter().map(src).collect();
    Drawing::new(s, gap, overlap).expect("a drawing")
}

/// The whole arrangement (every region).
fn build(sources: &[Value], gap: f64, overlap: f64) -> Arrangement {
    drawing(sources, gap, overlap).arrange(None).expect("builds")
}

/// The region at `p`, through the growing window a single pick uses.
fn region_at(sources: &[Value], p: (f64, f64), gap: f64) -> Option<FilledRegion> {
    match drawing(sources, gap, 0.0).at(p, None).expect("solves").0 {
        RegionAt::Region(r) => Some(r),
        _ => None,
    }
}

/// Area of a region's points: its largest ring less the others (the holes
/// in these tests nest one deep).
fn area_of(r: &FilledRegion) -> f64 {
    let areas: Vec<f64> = crate::stroke_outline::raster::polygons(&r.points)
        .iter()
        .map(|p| {
            let n = p.len();
            (0..n).map(|i| p[i].x * p[(i + 1) % n].y - p[(i + 1) % n].x * p[i].y).sum::<f64>().abs() / 2.0
        })
        .collect();
    let outer = areas.iter().copied().fold(0.0, f64::max);
    2.0 * outer - areas.iter().sum::<f64>()
}

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// A square drawn as four open strokes that cross at the corners (width 2).
fn square_of_lines(x0: f64, y0: f64, s: f64) -> Vec<Value> {
    let o = 10.0;
    vec![
        line((x0 - o, y0), (x0 + s + o, y0), 2.0),
        line((x0 - o, y0 + s), (x0 + s + o, y0 + s), 2.0),
        line((x0, y0 - o), (x0, y0 + s + o), 2.0),
        line((x0 + s, y0 - o), (x0 + s, y0 + s + o), 2.0),
    ]
}

#[test]
fn open_strokes_that_cross_enclose_a_region() {
    let lines = square_of_lines(0.0, 0.0, 100.0);
    let r = region_at(&lines, (50.0, 50.0), 0.0).expect("a region");
    // Inside the ink: 98 × 98 (each stroke is 2 wide, centred on the square).
    assert!(near(r.area, 98.0 * 98.0, 2.0), "face area {}", r.area);
    assert!(near(area_of(&r), 98.0 * 98.0, 4.0), "fill area {}", area_of(&r));
    assert!(near(r.bounds.min_x, 1.0, 0.1) && near(r.bounds.max_x, 99.0, 0.1), "{:?}", r.bounds);
    assert_eq!(r.subpath_count, 1);
    // The overhangs past the corners enclose nothing; outside is outside.
    let arr = build(&lines, 0.0, 0.0);
    assert!(matches!(arr.at((-5.0, 50.0)), RegionAt::Outside));
    assert!(matches!(arr.at((150.0, 150.0)), RegionAt::Outside));
    assert!(matches!(arr.at((0.0, 50.0)), RegionAt::OnLine));
    assert_eq!(arr.region_count(), 1);
}

#[test]
fn overlap_reaches_under_the_ink_but_never_past_it() {
    let lines = square_of_lines(0.0, 0.0, 100.0);
    let arr = build(&lines, 0.0, 0.5);
    let RegionAt::Region(r) = arr.at((50.0, 50.0)) else { panic!("a region") };
    assert!(near(r.bounds.min_x, 0.5, 0.1) && near(r.bounds.max_y, 99.5, 0.1), "{:?}", r.bounds);
    // Overlap wider than the stroke stops at the ink's far side (x = -1 and
    // 101 along the sides; only at the corners does it run on under the
    // overhanging lines, which are ink too).
    let arr = build(&lines, 0.0, 3.0);
    let RegionAt::Region(r) = arr.at((50.0, 50.0)) else { panic!("a region") };
    // Where the region's outline crosses y = 50.
    let mut side: Vec<f64> = Vec::new();
    for poly in crate::stroke_outline::raster::polygons(&r.points) {
        for k in 0..poly.len() {
            let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
            if (a.y - 50.0) * (b.y - 50.0) < 0.0 {
                side.push(a.x + (50.0 - a.y) / (b.y - a.y) * (b.x - a.x));
            }
        }
    }
    let (lo, hi) = side.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    assert!(near(lo, -1.0, 0.1) && near(hi, 101.0, 0.1), "{lo} {hi}");
    assert!(r.bounds.min_x >= -2.1 && r.bounds.max_x <= 102.1, "{:?}", r.bounds);
}

#[test]
fn nested_regions_are_separate_and_the_ring_has_a_hole() {
    // Two concentric stroked circles.
    let outer = json!({ "points": circle(0.0, 0.0, 100.0), "closed": true, "strokes": stroke(4.0, "butt") });
    let inner = json!({ "points": circle(0.0, 0.0, 50.0), "closed": true, "strokes": stroke(4.0, "butt") });
    let arr = build(&[outer, inner], 0.0, 0.0);
    assert_eq!(arr.region_count(), 2);
    let RegionAt::Region(disc) = arr.at((0.0, 0.0)) else { panic!("the disc") };
    let RegionAt::Region(ring) = arr.at((75.0, 0.0)) else { panic!("the ring") };
    assert_eq!(disc.subpath_count, 1);
    assert_eq!(ring.subpath_count, 2, "the ring's hole");
    let pi = std::f64::consts::PI;
    assert!(near(disc.area, pi * 48.0 * 48.0, 30.0), "disc {}", disc.area);
    assert!(near(ring.area, pi * (98.0 * 98.0 - 52.0 * 52.0), 60.0), "ring {}", ring.area);
    assert!(near(area_of(&ring), ring.area, 60.0), "ring points {}", area_of(&ring));
}

#[test]
fn a_gap_just_under_the_tolerance_closes_and_just_over_stays_open() {
    // The left line stops 3 px short of the top line's ink (butt caps, width 2:
    // the top line's ink ends at y = 1, the left line ends at y = 4).
    let mut lines = square_of_lines(0.0, 0.0, 100.0);
    lines[2] = line((0.0, 110.0), (0.0, 4.0), 2.0);
    assert!(region_at(&lines, (50.0, 50.0), 0.0).is_none(), "open with no gap closing");
    assert!(region_at(&lines, (50.0, 50.0), 2.9).is_none(), "2.9 < 3: stays open");
    let r = region_at(&lines, (50.0, 50.0), 3.1).expect("3.1 ≥ 3: closed");
    assert!(near(r.area, 98.0 * 98.0, 10.0), "{}", r.area);
    let arr = build(&lines, 3.1, 0.0);
    assert_eq!(arr.bridges.len(), 1, "{:?}", arr.bridges);
    let b = arr.bridges[0];
    assert!(near(b[0], 0.0, 1e-9) && near(b[1], 4.0, 1e-9) && near(b[3], 0.0, 1e-9), "{b:?}");
}

#[test]
fn round_caps_count_their_reach_when_measuring_a_gap() {
    // Round caps reach a half width (1) past the end: a 3 px gap between end
    // points is 2 px of visible gap.
    let mut lines = square_of_lines(0.0, 0.0, 100.0);
    lines[2] = json!({ "points": polyline(&[(0.0, 110.0), (0.0, 4.0)]), "strokes": stroke(2.0, "round") });
    assert!(region_at(&lines, (50.0, 50.0), 1.9).is_none());
    assert!(region_at(&lines, (50.0, 50.0), 2.1).is_some());
}

#[test]
fn a_nearly_closed_stroke_closes_on_itself() {
    // A "C": a circle with a 4-degree bite, as one open stroke.
    let n = 90;
    let pts: Vec<(f64, f64)> = (0..=n)
        .map(|i| {
            let t = 0.035 + (2.0 * std::f64::consts::PI - 0.07) * i as f64 / n as f64;
            (100.0 * t.cos(), 100.0 * t.sin())
        })
        .collect();
    let c = json!({ "points": polyline(&pts), "strokes": stroke(2.0, "butt") });
    // The ends are 2·100·sin(0.035) ≈ 7 px apart.
    assert!(region_at(&[c.clone()], (0.0, 0.0), 5.0).is_none());
    let r = region_at(&[c], (0.0, 0.0), 8.0).expect("closed on itself");
    assert!(near(r.area, std::f64::consts::PI * 99.0 * 99.0, 200.0), "{}", r.area);
}

#[test]
fn a_self_intersecting_stroke_makes_one_region_per_lobe() {
    // A bowtie: one closed stroked path that crosses itself.
    let bow = json!({
        "points": polyline(&[(0.0, 0.0), (200.0, 100.0), (200.0, 0.0), (0.0, 100.0)]),
        "closed": true,
        "strokes": stroke(2.0, "butt"),
    });
    let arr = build(&[bow], 0.0, 0.0);
    assert_eq!(arr.region_count(), 2);
    let RegionAt::Region(left) = arr.at((20.0, 50.0)) else { panic!("left lobe") };
    let RegionAt::Region(right) = arr.at((180.0, 50.0)) else { panic!("right lobe") };
    assert!(left.bounds.max_x < 101.0 && right.bounds.min_x > 99.0);
    assert!(near(left.area, right.area, 1.0));
    // Each lobe is a triangle of base 100 and height 100, less the ink.
    assert!(left.area > 4000.0 && left.area < 5000.0, "{}", left.area);
}

#[test]
fn a_compound_path_with_a_hole_bounds_the_ring_and_the_hole() {
    // A stroked compound path: a square with a square hole (both subpaths stroked).
    let mut pts = polyline(&[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)]).as_array().unwrap().clone();
    pts.extend(polyline(&[(50.0, 50.0), (50.0, 150.0), (150.0, 150.0), (150.0, 50.0)]).as_array().unwrap().clone());
    let frame = json!({ "points": pts, "closed": true, "fillRule": "evenodd", "strokes": stroke(2.0, "butt") });
    let arr = build(&[frame.clone()], 0.0, 0.0);
    assert_eq!(arr.region_count(), 2);
    let RegionAt::Region(ring) = arr.at((20.0, 100.0)) else { panic!("ring") };
    let RegionAt::Region(hole) = arr.at((100.0, 100.0)) else { panic!("hole") };
    assert_eq!(ring.subpath_count, 2);
    assert!(near(ring.area, 198.0 * 198.0 - 102.0 * 102.0, 5.0), "{}", ring.area);
    assert!(near(hole.area, 98.0 * 98.0, 5.0), "{}", hole.area);

    // Unstroked, the same compound path bounds by its edges and its fill rule.
    let bare = json!({ "points": frame["points"], "closed": true, "fillRule": "evenodd" });
    let arr = build(&[bare], 0.0, 0.0);
    let RegionAt::Region(ring) = arr.at((20.0, 100.0)) else { panic!("edge ring") };
    assert!(near(ring.area, 200.0 * 200.0 - 100.0 * 100.0, 5.0), "{}", ring.area);
    let RegionAt::Region(hole) = arr.at((100.0, 100.0)) else { panic!("edge hole") };
    assert!(near(hole.area, 100.0 * 100.0, 5.0), "{}", hole.area);
}

#[test]
fn an_unstroked_closed_shape_splits_the_region_it_sits_in() {
    let mut lines = square_of_lines(0.0, 0.0, 100.0);
    lines.push(json!({ "points": circle(50.0, 50.0, 20.0), "closed": true }));
    let arr = build(&lines, 0.0, 0.0);
    assert_eq!(arr.region_count(), 2);
    let RegionAt::Region(outer) = arr.at((10.0, 10.0)) else { panic!("outer") };
    assert_eq!(outer.subpath_count, 2, "the circle is a hole in it");
    let RegionAt::Region(disc) = arr.at((50.0, 50.0)) else { panic!("disc") };
    assert!(near(disc.area, std::f64::consts::PI * 400.0, 2.0), "{}", disc.area);
}

#[test]
fn unstroked_open_paths_bound_nothing_and_are_reported() {
    let mut lines = square_of_lines(0.0, 0.0, 100.0);
    lines.push(json!({ "points": polyline(&[(0.0, 50.0), (100.0, 50.0)]) }));
    let d = drawing(&lines, 0.0, 0.0);
    assert_eq!(d.ignored(), vec![4]);
    assert_eq!(d.arrange(None).unwrap().region_count(), 1, "the hairline does not split the square");
}

#[test]
fn transforms_map_the_stroke_as_drawn() {
    // The unit square's lines in their own space, scaled 100× (stroke 0.02 → 2 px).
    let lines: Vec<Value> = [((-0.1, 0.0), (1.1, 0.0)), ((-0.1, 1.0), (1.1, 1.0)), ((0.0, -0.1), (0.0, 1.1)), ((1.0, -0.1), (1.0, 1.1))]
        .iter()
        .map(|&(a, b)| json!({ "points": polyline(&[a, b]), "strokes": stroke(0.02, "butt"), "transform": [100, 0, 0, 100, 500, 500] }))
        .collect();
    let r = region_at(&lines, (550.0, 550.0), 0.0).expect("a region");
    assert!(near(r.area, 98.0 * 98.0, 2.0), "{}", r.area);
    assert!(near(r.bounds.min_x, 501.0, 0.1), "{:?}", r.bounds);
}

#[test]
fn every_region_of_a_grid_in_reading_order() {
    // Tic-tac-toe in a border: nine cells.
    let mut lines = square_of_lines(0.0, 0.0, 300.0);
    for k in [100.0, 200.0] {
        lines.push(line((k, -5.0), (k, 305.0), 2.0));
        lines.push(line((-5.0, k), (305.0, k), 2.0));
    }
    let regions = build(&lines, 0.0, 0.5).regions();
    assert_eq!(regions.len(), 9);
    for (i, r) in regions.iter().enumerate() {
        let (row, col) = (i / 3, i % 3);
        let (cx, cy) = (col as f64 * 100.0 + 50.0, row as f64 * 100.0 + 50.0);
        assert!(r.bounds.min_x < cx && cx < r.bounds.max_x && r.bounds.min_y < cy && cy < r.bounds.max_y, "cell {i}: {:?}", r.bounds);
    }
}

#[test]
fn json_contract() {
    let lines = square_of_lines(0.0, 0.0, 100.0);
    let out: Value = crate::value::from_str(&region_fill_json(&json!({ "op": "at", "sources": lines, "at": [50, 50] }).to_string())).unwrap();
    assert_eq!(out["status"], "ok", "{out}");
    assert_eq!(out["region"]["fillRule"], "evenodd");
    assert!(out["region"]["points"].as_array().unwrap().len() >= 4);

    let out: Value = crate::value::from_str(&region_fill_json(&json!({ "op": "at", "sources": lines, "at": [500, 50] }).to_string())).unwrap();
    assert_eq!(out["status"], "empty");
    assert_eq!(out["reason"], "outside");
    let out: Value = crate::value::from_str(&region_fill_json(&json!({ "op": "at", "sources": lines, "at": [0, 50] }).to_string())).unwrap();
    assert_eq!(out["reason"], "on_line");
    let out: Value = crate::value::from_str(&region_fill_json(&json!({ "op": "regions", "sources": lines }).to_string())).unwrap();
    assert_eq!(out["regions"].as_array().unwrap().len(), 1);
    let out: Value = crate::value::from_str(&region_fill_json(&json!({ "op": "regions", "sources": [] }).to_string())).unwrap();
    assert_eq!(out["reason"], "no_lines");

    for bad in [
        json!({ "op": "nope", "sources": [] }),
        json!({ "op": "at", "sources": [] }),
        json!({ "op": "regions", "sources": [{ "points": 3 }] }),
        json!({ "op": "regions", "sources": [], "gap": -1 }),
        json!({ "op": "regions", "sources": [{ "points": [], "transform": [0, 0, 0, 0, 0, 0] }] }),
    ] {
        let out: Value = crate::value::from_str(&region_fill_json(&bad.to_string())).unwrap();
        assert_eq!(out["status"], "error", "{bad} → {out}");
    }
    let out: Value = crate::value::from_str(&region_fill_json("{")).unwrap();
    assert_eq!(out["code"], "invalid_json");
}

#[test]
fn within_caps_the_search() {
    // A 2000 px square: its region doesn't close inside a 100 px box.
    let lines = square_of_lines(0.0, 0.0, 2000.0);
    let d = drawing(&lines, 0.0, 0.0);
    assert!(matches!(d.at((1000.0, 1000.0), Some([950.0, 950.0, 1050.0, 1050.0])).unwrap().0, RegionAt::Outside));
    assert!(matches!(d.at((1000.0, 1000.0), None).unwrap().0, RegionAt::Region(_)));
    let out: Value = crate::value::from_str(&region_fill_json(
        &json!({ "op": "at", "sources": lines, "at": [1000, 1000], "within": [950, 950, 1050, 1050] }).to_string(),
    ))
    .unwrap();
    assert_eq!(out["reason"], "outside");
}

#[test]
fn hostile_inputs_do_not_panic() {
    let huge = json!({ "points": polyline(&[(0.0, 0.0), (1e6, 1e6)]), "strokes": stroke(1e5, "round") });
    let tiny = json!({ "points": polyline(&[(0.0, 0.0), (1e-9, 0.0)]), "strokes": stroke(1.0, "butt") });
    let dup = json!({ "points": polyline(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]), "closed": true });
    let nan = json!({ "points": [{ "type": "move", "x": 0, "y": 0 }, { "type": "line", "x": 1e300, "y": 0 }], "strokes": stroke(1.0, "butt") });
    for sources in [vec![huge.clone(), tiny.clone(), dup.clone()], vec![nan], vec![dup.clone(), dup]] {
        for gap in [0.0, 5.0, 100.0] {
            let out: Value = crate::value::from_str(&region_fill_json(&json!({ "op": "regions", "sources": sources, "gap": gap }).to_string())).unwrap();
            assert_ne!(out["code"], "solver_failed", "{out}");
        }
    }
}

/// A 500-path line drawing: a grid of wavy strokes and stroked circles, like
/// a colouring page. Each region must fill in under 200 ms (release build;
/// debug only reports).
fn drawing_500() -> Vec<Value> {
    let mut sources = Vec::new();
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f64) / (1u64 << 31) as f64
    };
    // 100 wavy strokes: 50 across, 50 down, each with 12 curve segments.
    for axis in 0..2 {
        for k in 0..50 {
            let c = 20.0 + k as f64 * 40.0;
            let mut pts = vec![];
            for i in 0..=24 {
                let s = -20.0 + i as f64 * 85.0;
                let w = 6.0 * (s / 60.0 + k as f64).sin() + 2.0 * rnd();
                pts.push(if axis == 0 { (s, c + w) } else { (c + w, s) });
            }
            let mut v = vec![json!({ "type": "move", "x": pts[0].0, "y": pts[0].1 })];
            for w in pts.windows(2) {
                let (a, b) = (w[0], w[1]);
                v.push(json!({ "type": "curve", "x": b.0, "y": b.1,
                    "cp1": { "x": a.0 + (b.0 - a.0) / 3.0, "y": a.1 + (b.1 - a.1) / 3.0 + 3.0 },
                    "cp2": { "x": a.0 + 2.0 * (b.0 - a.0) / 3.0, "y": a.1 + 2.0 * (b.1 - a.1) / 3.0 - 3.0 } }));
            }
            sources.push(json!({ "points": v, "strokes": stroke(2.0 + rnd(), "round") }));
        }
    }
    // 400 stroked circles, some with gaps left by open arcs.
    for k in 0..400 {
        let (cx, cy) = (40.0 + (k % 20) as f64 * 100.0 + rnd() * 20.0, 40.0 + (k / 20) as f64 * 100.0 + rnd() * 20.0);
        let r = 8.0 + rnd() * 10.0;
        if k % 5 == 0 {
            let pts: Vec<(f64, f64)> = (0..=24).map(|i| {
                let t = 0.2 + 6.0 * i as f64 / 24.0;
                (cx + r * t.cos(), cy + r * t.sin())
            }).collect();
            sources.push(json!({ "points": polyline(&pts), "strokes": stroke(1.5, "round") }));
        } else {
            sources.push(json!({ "points": circle(cx, cy, r), "closed": true, "strokes": stroke(1.5, "butt") }));
        }
    }
    assert_eq!(sources.len(), 500);
    sources
}

#[test]
fn a_500_path_drawing_fills_a_region_in_under_200_ms() {
    let sources = drawing_500();
    // Every region at once (fill_regions).
    let request = json!({ "op": "regions", "sources": sources, "gap": 3.0 }).to_string();
    let t = Instant::now();
    let out: Value = crate::value::from_str(&region_fill_json(&request)).unwrap();
    let all_ms = t.elapsed().as_secs_f64() * 1000.0;
    let regions = out["regions"].as_array().expect("regions").clone();
    let n = regions.len();
    assert!(n > 1000, "{n} regions");
    eprintln!("every region: {n} in {all_ms:.1} ms ({:.2} ms per region)", all_ms / n as f64);

    // One region at a time (fill_region), at five regions' sample points:
    // every request parses the drawing again, as the tool call does.
    let mut worst = 0.0f64;
    for k in 0..5 {
        let r = &regions[k * (n - 1) / 4];
        let p = (r["sample"]["x"].as_f64().unwrap(), r["sample"]["y"].as_f64().unwrap());
        let request = json!({ "op": "at", "sources": sources, "at": [p.0, p.1], "gap": 3.0 }).to_string();
        let t = Instant::now();
        let out: Value = crate::value::from_str(&region_fill_json(&request)).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(out["status"], "ok", "region at {p:?}: {out}");
        assert!((out["region"]["area"].as_f64().unwrap() - r["area"].as_f64().unwrap()).abs() < 1.0, "the same region either way");
        worst = worst.max(ms);
        eprintln!("region at ({:.0}, {:.0}): {ms:.1} ms", p.0, p.1);
    }
    // A miss has to look at the whole drawing before it can say so.
    let request = json!({ "op": "at", "sources": sources, "at": [1990.0, 10.0], "gap": 3.0 }).to_string();
    let t = Instant::now();
    let out: Value = crate::value::from_str(&region_fill_json(&request)).unwrap();
    assert_eq!(out["reason"], "outside");
    eprintln!("a miss (outside every region): {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    eprintln!("500-path drawing: worst {worst:.1} ms per region");
    if !cfg!(debug_assertions) {
        assert!(worst < 200.0, "{worst:.1} ms per region (budget 200 ms)");
        assert!(all_ms / (n as f64) < 200.0);
    }
}

#[test]
#[ignore]
fn profile_500() {
    let sources = drawing_500();
    let t = Instant::now();
    let d = drawing(&sources, 3.0, 0.5);
    eprintln!("parse {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    let t = Instant::now();
    let _ = d.at((50.0, 50.0), None).unwrap();
    eprintln!("at (cold) {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    let t = Instant::now();
    let a = d.arrange(None).unwrap();
    eprintln!("arrange everything {:.1} ms, {} faces", t.elapsed().as_secs_f64() * 1000.0, a.region_count());
    let t = Instant::now();
    let r = a.regions();
    eprintln!("refit every region {:.1} ms ({})", t.elapsed().as_secs_f64() * 1000.0, r.len());
}
