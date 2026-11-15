use crate::value::json;

use super::*;
use crate::blend::BlendPathPoint as PathPoint;
use crate::path_bool::FillRule;

const TOL: f64 = DEFAULT_TOLERANCE;

pub(super) fn v(x: f64, y: f64) -> Vector {
    Vector::new(x, y)
}

/// A network of straight segments from a vertex list and index pairs.
pub(super) fn lines(verts: &[(f64, f64)], segs: &[(usize, usize)]) -> VectorNetwork {
    let mut n = VectorNetwork::new();
    for &(x, y) in verts {
        n.add_vertex(v(x, y));
    }
    for &(a, b) in segs {
        n.add_line(a, b).unwrap();
    }
    n
}

/// Sum of face areas and the outer boundary area implied by them.
pub(super) fn face_areas(n: &VectorNetwork) -> Vec<f64> {
    let mut a: Vec<f64> = n.faces(TOL).iter().map(|f| f.area).collect();
    a.sort_by(f64::total_cmp);
    a
}

fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps * (1.0 + a.abs().max(b.abs()))
}

fn vertex_at(n: &VectorNetwork, p: Vector) -> Option<usize> {
    (0..n.vertices.len()).find(|&i| n.vertex_pos(i).dist(p) < 1e-6)
}

/// Exact area of a closed chain of oriented segments.
fn loop_area(n: &VectorNetwork, lp: &[(usize, bool)]) -> f64 {
    lp.iter()
        .map(|&(s, r)| n.oriented_cubic(s, r).signed_area())
        .sum()
}

fn circle(n: &mut VectorNetwork, cx: f64, cy: f64, r: f64) -> Vec<usize> {
    // Four-arc cubic circle, counter-clockwise (in y-up terms).
    let k = 0.552_284_749_830_793_4 * r;
    let p = [v(cx + r, cy), v(cx, cy + r), v(cx - r, cy), v(cx, cy - r)];
    let t = [v(0.0, k), v(-k, 0.0), v(0.0, -k), v(k, 0.0)];
    let ids: Vec<usize> = p.iter().map(|&q| n.add_vertex(q)).collect();
    (0..4)
        .map(|i| {
            let j = (i + 1) % 4;
            n.add_segment(ids[i], ids[j], t[i], -t[j]).unwrap()
        })
        .collect()
}

// -------------------------------------------------------------------------
// JSON
// -------------------------------------------------------------------------

#[test]
fn json_round_trips_figma_plugin_shape() {
    // Floats throughout: serde_json compares 4 and 4.0 as different numbers.
    let src = json!({
        "vertices": [
            { "x": 0.0, "y": 0.0, "strokeCap": "ROUND", "strokeJoin": "BEVEL",
              "cornerRadius": 4.0, "handleMirroring": "ANGLE_AND_LENGTH" },
            { "x": 100.0, "y": 0.0, "strokeCap": "ARROW_EQUILATERAL" },
            { "x": 50.0, "y": 80.0, "strokeCap": "WASHI_TAPE_3" }
        ],
        "segments": [
            { "start": 0, "end": 1, "tangentStart": { "x": 0.0, "y": 0.0 }, "tangentEnd": { "x": 0.0, "y": 0.0 } },
            { "start": 1, "end": 2, "tangentStart": { "x": 0.0, "y": 30.0 }, "tangentEnd": { "x": 20.0, "y": 0.0 } },
            { "start": 2, "end": 0, "tangentStart": { "x": 0.0, "y": 0.0 }, "tangentEnd": { "x": 0.0, "y": 0.0 } }
        ],
        "regions": [
            { "windingRule": "EVENODD", "loops": [[0, 1, 2]],
              "fills": [{ "type": "SOLID", "color": { "r": 1, "g": 0, "b": 0 } }],
              "fillStyleId": "S:abc" }
        ]
    });
    let n = VectorNetwork::from_json_value(&src).unwrap();
    assert_eq!(n.vertices[0].stroke_cap, Some(StrokeCap::Round));
    assert_eq!(n.vertices[0].stroke_join, Some(StrokeJoin::Bevel));
    assert_eq!(n.vertices[0].corner_radius, Some(4.0));
    assert_eq!(
        n.vertices[0].handle_mirroring,
        Some(HandleMirroring::AngleAndLength)
    );
    assert_eq!(
        n.vertices[2].stroke_cap,
        Some(StrokeCap::Other("WASHI_TAPE_3".into()))
    );
    assert_eq!(n.segments[1].tangent_start, v(0.0, 30.0));
    assert_eq!(n.regions[0].winding_rule, WindingRule::EvenOdd);
    let out = n.to_json_value();
    assert_eq!(out, src);
    assert_eq!(VectorNetwork::from_json(&n.to_json()).unwrap(), n);
}

#[test]
fn json_defaults_missing_tangents_and_regions() {
    let n = VectorNetwork::from_json(
        r#"{"vertices":[{"x":0,"y":0},{"x":1,"y":1}],"segments":[{"start":0,"end":1}]}"#,
    )
    .unwrap();
    assert!(n.segments[0].is_straight());
    assert!(n.regions.is_empty());
    assert_eq!(n.regions.len(), 0);
    // Regions default to NONZERO when the rule is missing.
    let n = VectorNetwork::from_json(
        r#"{"vertices":[{"x":0,"y":0},{"x":1,"y":1}],"segments":[{"start":0,"end":1}],"regions":[{"loops":[[0]]}]}"#,
    )
    .unwrap();
    assert_eq!(n.regions[0].winding_rule, WindingRule::NonZero);
}

#[test]
fn json_rejects_bad_indices_and_garbage() {
    assert!(matches!(
        VectorNetwork::from_json(
            r#"{"vertices":[{"x":0,"y":0}],"segments":[{"start":0,"end":3}]}"#
        ),
        Err(NetworkError::DanglingSegment {
            segment: 0,
            vertex: 3
        })
    ));
    assert!(matches!(
        VectorNetwork::from_json(
            r#"{"vertices":[{"x":0,"y":0},{"x":1,"y":0}],"segments":[{"start":0,"end":1}],"regions":[{"windingRule":"NONZERO","loops":[[0,7]]}]}"#
        ),
        Err(NetworkError::DanglingLoop {
            region: 0,
            segment: 7
        })
    ));
    assert!(matches!(
        VectorNetwork::from_json(r#"{"vertices":[{"x":"a","y":0}],"segments":[]}"#),
        Err(NetworkError::Json(_))
    ));
    assert!(matches!(
        VectorNetwork::from_json(
            r#"{"vertices":[{"x":0,"y":0}],"segments":[{"start":-1,"end":0}]}"#
        ),
        Err(NetworkError::Json(_))
    ));
}

// -------------------------------------------------------------------------
// Editing
// -------------------------------------------------------------------------

#[test]
fn add_and_remove_keep_indices_and_regions_consistent() {
    let mut n = lines(
        &[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (20.0, 20.0),
        ],
        &[(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)],
    );
    n.regions.push(VectorRegion::new(
        WindingRule::NonZero,
        vec![vec![0, 1, 2, 3]],
    ));
    assert_eq!(n.degree(2), 3);
    // Removing the spoke leaves the square region intact.
    n.remove_segment(4).unwrap();
    assert_eq!(n.regions[0].loops, vec![vec![0, 1, 2, 3]]);
    // Removing vertex 4 (now isolated) compacts nothing else.
    n.remove_vertex(4).unwrap();
    assert_eq!(n.vertices.len(), 4);
    n.validate().unwrap();
    // Deleting a boundary edge unfills the square.
    n.remove_segment(1).unwrap();
    assert!(n.regions.is_empty());
    assert_eq!(n.segments.len(), 3);
    // Deleting a vertex deletes its segments and compacts indices.
    n.remove_vertex(0).unwrap();
    assert_eq!(n.vertices.len(), 3);
    assert_eq!(n.segments.len(), 1);
    assert_eq!(n.vertex_pos(n.segments[0].start), v(10.0, 10.0));
    assert_eq!(n.vertex_pos(n.segments[0].end), v(0.0, 10.0));
    n.validate().unwrap();
    assert!(n.add_line(0, 9).is_err());
    assert!(n.remove_segment(5).is_err());
}

#[test]
fn split_segment_preserves_geometry_and_region_order() {
    let mut n = VectorNetwork::new();
    let a = n.add_vertex(v(0.0, 0.0));
    let b = n.add_vertex(v(30.0, 0.0));
    let c = n.add_vertex(v(15.0, 25.0));
    n.add_segment(a, b, v(10.0, -10.0), v(-10.0, -10.0))
        .unwrap();
    n.add_line(b, c).unwrap();
    n.add_line(c, a).unwrap();
    // The loop traverses segment 0 backwards.
    n.regions
        .push(VectorRegion::new(WindingRule::NonZero, vec![vec![2, 0, 1]]));
    let before = n.segment_cubic(0);
    let area_before = {
        let (o, _) = n.chain_loop(&n.regions[0].loops[0]).unwrap();
        loop_area(&n, &o)
    };
    let m = n.split_segment(0, 0.3).unwrap();
    assert!(n.vertex_pos(m).dist(before.eval(0.3)) < 1e-12);
    // Pieces trace the original.
    let p0 = n.segment_cubic(0);
    let p1 = n.segment_cubic(3);
    for k in 0..=10 {
        let t = k as f64 / 10.0;
        assert!(p0.eval(t).dist(before.eval(0.3 * t)) < 1e-9);
        assert!(p1.eval(t).dist(before.eval(0.3 + 0.7 * t)) < 1e-9);
    }
    // c→a, then a→m→b means reversed pieces: [2, 0, 3, 1]? No: the loop goes
    // c → a → b → c, so segment 0 runs forward here: a→m (0), m→b (3).
    let lp = &n.regions[0].loops[0];
    assert_eq!(lp, &vec![2, 0, 3, 1]);
    let (o, closed) = n.chain_loop(lp).unwrap();
    assert!(closed);
    assert!(close(loop_area(&n, &o), area_before, 1e-12));

    // Reversed traversal gets the pieces reversed.
    let mut r = VectorNetwork::new();
    let a = r.add_vertex(v(0.0, 0.0));
    let b = r.add_vertex(v(10.0, 0.0));
    let c = r.add_vertex(v(0.0, 10.0));
    r.add_line(a, b).unwrap();
    r.add_line(b, c).unwrap();
    r.add_line(a, c).unwrap(); // stored a→c, walked c→a
    r.regions
        .push(VectorRegion::new(WindingRule::NonZero, vec![vec![0, 1, 2]]));
    r.split_segment(2, 0.5).unwrap();
    assert_eq!(r.regions[0].loops[0], vec![0, 1, 3, 2]);
    assert!(r.chain_loop(&r.regions[0].loops[0]).unwrap().1);
    assert!(r.segments[3].is_straight());

    assert!(r.split_segment(0, 0.0).is_err());
    assert!(r.split_segment(0, 1.0).is_err());
    assert!(r.split_segment(0, f64::NAN).is_err());
}

#[test]
fn merge_vertices_rewires_segments_and_drops_collapsed_ones() {
    let mut n = lines(
        &[(0.0, 0.0), (10.0, 0.0), (10.0, 0.5), (20.0, 0.0)],
        &[(0, 1), (1, 2), (2, 3)],
    );
    let keep = n.merge_vertices(1, 2).unwrap();
    assert_eq!(keep, 1);
    assert_eq!(n.vertices.len(), 3);
    // 1–2 collapsed; 2–3 now starts at 1.
    assert_eq!(n.segments.len(), 2);
    assert_eq!(n.degree(1), 2);
    n.validate().unwrap();
    // Merging into a later index returns the shifted index.
    let mut n = lines(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], &[(0, 1), (1, 2)]);
    assert_eq!(n.merge_vertices(2, 0).unwrap(), 1);
    assert_eq!(n.vertices.len(), 2);
}

// -------------------------------------------------------------------------
// Planarisation and faces
// -------------------------------------------------------------------------

#[test]
fn figure_eight_of_lines_splits_into_two_triangles() {
    let mut n = lines(
        &[(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)],
        &[(0, 1), (1, 2), (2, 3), (3, 0)],
    );
    assert!(!n.is_planar(TOL));
    let rep = n.planarize(TOL);
    assert_eq!(rep.vertices_added, 1);
    assert_eq!(n.vertices.len(), 5);
    assert_eq!(n.segments.len(), 6);
    assert!(n.is_planar(TOL));
    let centre = vertex_at(&n, v(5.0, 5.0)).expect("crossing vertex");
    assert_eq!(n.degree(centre), 4);
    let areas = face_areas(&n);
    assert_eq!(areas.len(), 2);
    assert!(
        close(areas[0], 25.0, 1e-9) && close(areas[1], 25.0, 1e-9),
        "{areas:?}"
    );
}

#[test]
fn figure_eight_of_one_looping_cubic() {
    // A cubic that loops over itself, closed by a straight base.
    let mut n = VectorNetwork::new();
    let a = n.add_vertex(v(0.0, 0.0));
    let b = n.add_vertex(v(10.0, 0.0));
    n.add_segment(a, b, v(20.0, 10.0), v(-20.0, 10.0)).unwrap();
    n.add_line(b, a).unwrap();
    let whole = {
        let c = n.segment_cubic(0);
        (c.signed_area() + n.segment_cubic(1).signed_area()).abs()
    };
    n.planarize(TOL);
    assert!(n.is_planar(TOL));
    let crossing = n
        .vertices
        .iter()
        .position(|p| p.pos().dist(v(5.0, 3.0)) < 1e-6)
        .expect("self-crossing vertex at (5, 3)");
    assert_eq!(n.degree(crossing), 4);
    let faces = n.faces(TOL);
    assert_eq!(faces.len(), 2, "{faces:?}");
    // The loop and the lens under it; the signed areas of the two lobes of
    // the original closed curve have opposite signs, so |sum of faces| is not
    // |net area|, but each lobe is recovered.
    let total: f64 = faces.iter().map(|f| f.area).sum();
    assert!(total > whole, "{total} vs net {whole}");
    for f in &faces {
        assert!(f.area > 0.0);
        assert_eq!(f.loops.len(), 1);
    }
}

#[test]
fn plus_sign_of_crossing_lines_has_a_degree_four_vertex_and_no_faces() {
    let mut n = lines(
        &[(-5.0, 0.0), (5.0, 0.0), (0.0, -5.0), (0.0, 5.0)],
        &[(0, 1), (2, 3)],
    );
    n.planarize(TOL);
    assert_eq!(n.vertices.len(), 5);
    assert_eq!(n.segments.len(), 4);
    let c = vertex_at(&n, v(0.0, 0.0)).unwrap();
    assert_eq!(n.degree(c), 4);
    assert!(n.faces(TOL).is_empty());
    // Already-planar plus sign (one hub vertex) is left alone.
    let mut hub = lines(
        &[(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (-5.0, 0.0), (0.0, -5.0)],
        &[(0, 1), (0, 2), (0, 3), (0, 4)],
    );
    let before = hub.clone();
    assert!(!hub.planarize(TOL).changed());
    assert_eq!(hub, before);
    assert!(hub.faces(TOL).is_empty());
}

#[test]
fn plus_sign_inside_a_square_makes_four_quarters() {
    // Arms end exactly on the square's edges: T-junctions.
    let mut n = lines(
        &[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (5.0, 0.0),
            (5.0, 10.0),
            (0.0, 5.0),
            (10.0, 5.0),
        ],
        &[(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (6, 7)],
    );
    n.planarize(TOL);
    assert!(n.is_planar(TOL));
    assert_eq!(n.vertices.len(), 9);
    assert_eq!(n.segments.len(), 12);
    let areas = face_areas(&n);
    assert_eq!(areas.len(), 4);
    for a in areas {
        assert!(close(a, 25.0, 1e-9));
    }
}

#[test]
fn triangle_with_inner_spoke_is_one_face_without_the_spoke() {
    let mut n = lines(
        &[(0.0, 0.0), (12.0, 0.0), (6.0, 9.0), (6.0, 3.0)],
        &[(0, 1), (1, 2), (2, 0), (0, 3)],
    );
    n.planarize(TOL);
    let faces = n.faces(TOL);
    assert_eq!(faces.len(), 1);
    assert!(close(faces[0].area, 54.0, 1e-9));
    assert_eq!(faces[0].loops.len(), 1);
    let mut segs = faces[0].segment_loops()[0].clone();
    segs.sort();
    assert_eq!(segs, vec![0, 1, 2]);
    // A free-floating spoke inside (its own component) is a zero-area hole:
    // nothing to fill.
    let mut n2 = lines(
        &[(0.0, 0.0), (12.0, 0.0), (6.0, 9.0), (5.0, 3.0), (7.0, 3.0)],
        &[(0, 1), (1, 2), (2, 0), (3, 4)],
    );
    n2.planarize(TOL);
    let f2 = n2.faces(TOL);
    assert_eq!(f2.len(), 1);
    assert!(close(f2[0].area, 54.0, 1e-9));
    assert_eq!(f2[0].loops.len(), 1);
    // A spoke that reaches the far side splits the triangle.
    let mut n3 = lines(
        &[(0.0, 0.0), (12.0, 0.0), (6.0, 9.0)],
        &[(0, 1), (1, 2), (2, 0)],
    );
    let apex = 2;
    let base_mid = n3.add_vertex(v(6.0, 0.0));
    n3.add_line(apex, base_mid).unwrap();
    n3.planarize(TOL);
    let a3 = face_areas(&n3);
    assert_eq!(a3.len(), 2);
    assert!(close(a3[0], 27.0, 1e-9) && close(a3[1], 27.0, 1e-9));
}

#[test]
fn square_with_square_hole_and_island() {
    let mut n = lines(
        &[
            (0.0, 0.0),
            (30.0, 0.0),
            (30.0, 30.0),
            (0.0, 30.0),
            (10.0, 10.0),
            (20.0, 10.0),
            (20.0, 20.0),
            (10.0, 20.0),
        ],
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
        ],
    );
    n.planarize(TOL);
    let faces = n.faces(TOL);
    assert_eq!(faces.len(), 2);
    let ring = faces
        .iter()
        .find(|f| f.loops.len() == 2)
        .expect("ring face");
    assert!(close(ring.area, 800.0, 1e-9));
    assert!(ring.contains(&n, v(5.0, 5.0), TOL));
    assert!(!ring.contains(&n, v(15.0, 15.0), TOL));
    let island = faces.iter().find(|f| f.loops.len() == 1).unwrap();
    assert!(close(island.area, 100.0, 1e-9));
    assert!(island.contains(&n, v(15.0, 15.0), TOL));
}

#[test]
fn circle_cut_by_a_chord_and_a_curve() {
    let mut n = VectorNetwork::new();
    circle(&mut n, 0.0, 0.0, 10.0);
    let whole: f64 = (0..4).map(|i| n.segment_cubic(i).signed_area()).sum();
    let a = n.add_vertex(v(-15.0, 2.0));
    let b = n.add_vertex(v(15.0, -3.0));
    n.add_line(a, b).unwrap();
    // A wavy curve crossing the circle and the chord.
    let c = n.add_vertex(v(-3.0, -15.0));
    let d = n.add_vertex(v(4.0, 15.0));
    n.add_segment(c, d, v(20.0, 10.0), v(-20.0, -10.0)).unwrap();
    n.planarize(TOL);
    assert!(n.is_planar(TOL));
    let faces = n.faces(TOL);
    assert!(faces.len() >= 4, "{}", faces.len());
    let sum: f64 = faces.iter().map(|f| f.area).sum();
    assert!(close(sum, whole.abs(), 1e-7), "{sum} vs {whole}");
}

#[test]
fn overlapping_cubics_merge_into_shared_segments() {
    // Identical curves on distinct vertices collapse to one.
    let mut n = VectorNetwork::new();
    let t0 = v(10.0, 20.0);
    let t1 = v(-10.0, 20.0);
    for _ in 0..2 {
        let a = n.add_vertex(v(0.0, 0.0));
        let b = n.add_vertex(v(30.0, 0.0));
        n.add_segment(a, b, t0, t1).unwrap();
    }
    n.planarize(TOL);
    assert_eq!(n.vertices.len(), 2);
    assert_eq!(n.segments.len(), 1);

    // A sub-curve [0.25, 0.75] drawn separately: the long curve is split at
    // the short one's ends and the middle pieces merge.
    let mut n = VectorNetwork::new();
    let a = n.add_vertex(v(0.0, 0.0));
    let b = n.add_vertex(v(30.0, 0.0));
    n.add_segment(a, b, t0, t1).unwrap();
    let full = n.segment_cubic(0);
    let sub = full.subsegment(0.25, 0.75);
    let c = n.add_vertex(sub.p0);
    let d = n.add_vertex(sub.p3);
    n.add_segment(c, d, sub.p1 - sub.p0, sub.p2 - sub.p3)
        .unwrap();
    n.regions
        .push(VectorRegion::new(WindingRule::NonZero, vec![vec![1]]));
    n.planarize(TOL);
    assert!(n.is_planar(TOL));
    assert_eq!(n.vertices.len(), 4);
    assert_eq!(n.segments.len(), 3);
    // The region's reference followed the merge.
    assert_eq!(n.regions.len(), 1);
    let s = n.regions[0].loops[0][0];
    let seg = &n.segments[s];
    let ends = [n.vertex_pos(seg.start), n.vertex_pos(seg.end)];
    assert!(ends.iter().any(|p| p.dist(sub.p0) < 1e-6));
    assert!(ends.iter().any(|p| p.dist(sub.p3) < 1e-6));

    // Partially overlapping curves (the same cubic, [0, 0.6] and [0.4, 1]).
    let mut n = VectorNetwork::new();
    for (t0, t1) in [(0.0, 0.6), (0.4, 1.0)] {
        let piece = full.subsegment(t0, t1);
        let a = n.add_vertex(piece.p0);
        let b = n.add_vertex(piece.p3);
        n.add_segment(a, b, piece.p1 - piece.p0, piece.p2 - piece.p3)
            .unwrap();
    }
    n.planarize(TOL);
    assert!(n.is_planar(TOL));
    assert_eq!(n.vertices.len(), 4);
    assert_eq!(n.segments.len(), 3);

    // Collinear overlapping lines.
    let mut n = lines(
        &[(0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (15.0, 0.0)],
        &[(0, 1), (2, 3)],
    );
    n.planarize(TOL);
    assert_eq!(n.vertices.len(), 4);
    assert_eq!(n.segments.len(), 3);
}

#[test]
fn two_curves_crossing_twice_make_a_lens() {
    let mut n = VectorNetwork::new();
    let a = n.add_vertex(v(0.0, 0.0));
    let b = n.add_vertex(v(30.0, 0.0));
    n.add_segment(a, b, v(10.0, 20.0), v(-10.0, 20.0)).unwrap();
    let c = n.add_vertex(v(0.0, 10.0));
    let d = n.add_vertex(v(30.0, 10.0));
    n.add_segment(c, d, v(10.0, -20.0), v(-10.0, -20.0))
        .unwrap();
    n.planarize(TOL);
    assert!(n.is_planar(TOL));
    assert_eq!(n.vertices.len(), 6);
    assert_eq!(n.segments.len(), 6);
    let faces = n.faces(TOL);
    assert_eq!(faces.len(), 1);
    assert_eq!(faces[0].loops[0].len(), 2);
}

#[test]
fn degenerate_input_is_scrubbed() {
    // A zero-length straight segment, a vertex pair closer than the
    // tolerance, and a curve whose control points sit on its vertex.
    let mut n = lines(
        &[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (10.0, 10.0 + 1e-5),
            (0.0, 10.0),
        ],
        &[(0, 1), (1, 2), (3, 4), (4, 0), (2, 2)],
    );
    let z = n.add_vertex(v(50.0, 50.0));
    n.add_segment(z, z, v(1e-5, 0.0), v(0.0, 1e-5)).unwrap();
    n.regions.push(VectorRegion::new(
        WindingRule::NonZero,
        vec![vec![0, 1, 4, 2, 3]],
    ));
    let rep = n.planarize(TOL);
    assert_eq!(rep.vertices_merged, 1);
    assert_eq!(rep.segments_removed, 2);
    assert_eq!(n.segments.len(), 4);
    // Vertex 5 (the tiny loop's) stays as an isolated vertex.
    assert_eq!(n.vertices.len(), 5);
    // The region survives and still chains as a closed square.
    assert_eq!(n.regions[0].loops[0].len(), 4);
    assert!(n.chain_loop(&n.regions[0].loops[0]).unwrap().1);
    assert!(close(face_areas(&n)[0], 100.0, 1e-9));
    n.validate().unwrap();

    // A curved self-loop (a teardrop) is split so it has a face.
    let mut t = VectorNetwork::new();
    let a = t.add_vertex(v(0.0, 0.0));
    t.add_segment(a, a, v(20.0, 20.0), v(-20.0, 20.0)).unwrap();
    t.planarize(TOL);
    assert_eq!(t.segments.len(), 2);
    let f = t.faces(TOL);
    assert_eq!(f.len(), 1);
    let exact = Cubic::new(v(0.0, 0.0), v(20.0, 20.0), v(-20.0, 20.0), v(0.0, 0.0))
        .signed_area()
        .abs();
    assert!(close(f[0].area, exact, 1e-9));

    // Non-finite coordinates are rejected, not planarised.
    let mut bad = lines(&[(0.0, 0.0), (f64::NAN, 1.0)], &[(0, 1)]);
    assert!(bad.validate().is_err());
    assert!(!bad.planarize(TOL).changed());

    // Empty and vertex-only networks.
    let mut e = VectorNetwork::new();
    e.planarize(TOL);
    assert!(e.faces(TOL).is_empty());
    e.add_vertex(v(1.0, 1.0));
    e.planarize(TOL);
    assert!(e.faces(TOL).is_empty());
    assert_eq!(e.toggle_fill_at(v(1.0, 1.0), TOL), FillToggle::NoFace);
}

#[test]
fn tangent_edges_leaving_a_vertex_are_ordered_by_curvature() {
    // Two curves leave (0,0) in the same direction and rejoin at (20,0): a
    // thin crescent. Only the second-order ordering gets the face right.
    let mut n = VectorNetwork::new();
    let a = n.add_vertex(v(0.0, 0.0));
    let b = n.add_vertex(v(20.0, 0.0));
    n.add_segment(a, b, v(5.0, 5.0), v(-5.0, 5.0)).unwrap();
    n.add_segment(a, b, v(5.0, 5.0), v(-5.0, 10.0)).unwrap();
    n.add_line(a, b).unwrap();
    n.planarize(TOL);
    let faces = n.faces(TOL);
    assert_eq!(faces.len(), 2);
    let c0 = n.segment_cubic(0);
    let c1 = n.segment_cubic(1);
    let chord = n.segment_cubic(2);
    let lens = (c1.signed_area() - c0.signed_area()).abs();
    let low = (c0.signed_area() - chord.signed_area()).abs();
    let mut want = [lens, low];
    want.sort_by(f64::total_cmp);
    let got = face_areas(&n);
    assert!(
        close(got[0], want[0], 1e-9) && close(got[1], want[1], 1e-9),
        "{got:?} {want:?}"
    );
}

// -------------------------------------------------------------------------
// Fill toggle
// -------------------------------------------------------------------------

#[test]
fn clicking_faces_fills_and_unfills_them() {
    // Two squares sharing an edge.
    let mut n = lines(
        &[
            (0.0, 0.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (20.0, 10.0),
            (10.0, 10.0),
            (0.0, 10.0),
        ],
        &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)],
    );
    let left = v(5.0, 5.0);
    let right = v(15.0, 5.0);
    assert_eq!(
        n.toggle_fill_at(left, TOL),
        FillToggle::Filled { region: 0 }
    );
    assert!(n.region_contains(0, left, TOL));
    assert!(!n.region_contains(0, right, TOL));
    assert_eq!(
        n.toggle_fill_at(right, TOL),
        FillToggle::Filled { region: 1 }
    );
    assert_eq!(n.toggle_fill_at(left, TOL), FillToggle::Unfilled);
    assert_eq!(n.regions.len(), 1);
    assert!(n.region_contains(0, right, TOL));
    assert_eq!(n.toggle_fill_at(v(50.0, 50.0), TOL), FillToggle::NoFace);

    // A region drawn round the outside covers both faces; clicking one
    // unfills only that one, keeping the region's paint on the other.
    let mut n2 = n.clone();
    n2.regions = vec![VectorRegion {
        winding_rule: WindingRule::NonZero,
        loops: vec![vec![0, 1, 2, 3, 4, 5]],
        fills: Some(vec![json!({"type": "SOLID"})]),
        fill_style_id: None,
    }];
    assert!(n2.region_contains(0, left, TOL) && n2.region_contains(0, right, TOL));
    assert_eq!(n2.toggle_fill_at(left, TOL), FillToggle::Unfilled);
    assert_eq!(n2.regions.len(), 1);
    assert!(!n2.region_contains(0, left, TOL));
    assert!(n2.region_contains(0, right, TOL));
    assert_eq!(n2.regions[0].fills, Some(vec![json!({"type": "SOLID"})]));
}

#[test]
fn clicking_crossing_strokes_fills_the_new_face() {
    // A triangle crossed by a line: the click planarises first.
    let mut n = lines(
        &[
            (0.0, 0.0),
            (20.0, 0.0),
            (10.0, 20.0),
            (-5.0, 5.0),
            (25.0, 5.0),
        ],
        &[(0, 1), (1, 2), (2, 0), (3, 4)],
    );
    let r = n.toggle_fill_at(v(10.0, 2.0), TOL);
    assert_eq!(r, FillToggle::Filled { region: 0 });
    // The filled face is the trapezoid under y = 5: (20 + 15) / 2 * 5.
    let paths = n.to_paths();
    assert_eq!(paths.regions.len(), 1);
    let face = n.face_at(v(10.0, 2.0), TOL).unwrap();
    assert!(close(face.area, 87.5, 1e-9));
    assert!(n.region_contains(0, v(10.0, 2.0), TOL));
    assert!(!n.region_contains(0, v(10.0, 10.0), TOL));
}

#[test]
fn fill_all_faces_fills_every_enclosed_area() {
    let mut n = lines(
        &[(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)],
        &[(0, 1), (1, 2), (2, 3), (3, 0)],
    );
    n.fill_all_faces(TOL);
    assert_eq!(n.regions.len(), 2);
    for p in [v(2.0, 5.0), v(8.0, 5.0)] {
        assert_eq!((0..2).filter(|&r| n.region_contains(r, p, TOL)).count(), 1);
    }
    assert!(!(0..2).any(|r| n.region_contains(r, v(5.0, 2.0), TOL)));
}

// -------------------------------------------------------------------------
// Paths
// -------------------------------------------------------------------------

fn pts_eq(a: &[PathPoint], b: &[PathPoint]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            let (jx, jy) = (x.to_json(), y.to_json());
            if jx["type"] != jy["type"] {
                return false;
            }
            let f = |j: &crate::value::Value, k: &str| j[k].as_f64().unwrap_or(0.0);
            let cp = |j: &crate::value::Value, k: &str| {
                (
                    j[k]["x"].as_f64().unwrap_or(0.0),
                    j[k]["y"].as_f64().unwrap_or(0.0),
                )
            };
            let near = |p: f64, q: f64| (p - q).abs() < 1e-9;
            near(f(&jx, "x"), f(&jy, "x"))
                && near(f(&jx, "y"), f(&jy, "y"))
                && ["cp1", "cp2"].iter().all(|k| {
                    let (p, q) = (cp(&jx, k), cp(&jy, k));
                    near(p.0, q.0) && near(p.1, q.1)
                })
        })
}

#[test]
fn closed_path_round_trips_through_a_network() {
    let square = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Line { x: 10.0, y: 0.0 },
        PathPoint::Line { x: 10.0, y: 10.0 },
        PathPoint::Line { x: 0.0, y: 10.0 },
    ];
    let n = VectorNetwork::from_path(&square, true, FillRule::NonZero, TOL);
    assert_eq!(n.vertices.len(), 4);
    assert_eq!(n.segments.len(), 4);
    assert_eq!(n.regions.len(), 1);
    let p = n.to_paths();
    assert_eq!(p.regions.len(), 1);
    assert!(p.open_runs.is_empty());
    assert!(pts_eq(&p.regions[0].points, &square));

    // Curves, closing on a curve, plus a hole, under even-odd.
    let blob = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Curve {
            x: 30.0,
            y: 0.0,
            cp1: (10.0, -10.0),
            cp2: (20.0, -10.0),
        },
        PathPoint::Line { x: 30.0, y: 30.0 },
        PathPoint::Curve {
            x: 0.0,
            y: 0.0,
            cp1: (20.0, 40.0),
            cp2: (-10.0, 20.0),
        },
        PathPoint::Move { x: 10.0, y: 5.0 },
        PathPoint::Line { x: 20.0, y: 5.0 },
        PathPoint::Line { x: 20.0, y: 15.0 },
    ];
    let n = VectorNetwork::from_path(&blob, true, FillRule::EvenOdd, TOL);
    assert_eq!(n.vertices.len(), 6);
    assert_eq!(n.segments.len(), 6);
    assert_eq!(n.regions[0].winding_rule, WindingRule::EvenOdd);
    assert_eq!(n.regions[0].loops.len(), 2);
    let p = n.to_paths();
    assert_eq!(p.regions[0].fill_rule, FillRule::EvenOdd);
    assert!(
        pts_eq(&p.regions[0].points, &blob),
        "{:?}",
        p.regions[0].points
    );
    // Its JSON round-trips too.
    assert_eq!(VectorNetwork::from_json(&n.to_json()).unwrap(), n);
}

#[test]
fn open_path_round_trips_as_a_run() {
    let wave = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Curve {
            x: 10.0,
            y: 0.0,
            cp1: (3.0, 5.0),
            cp2: (7.0, 5.0),
        },
        PathPoint::Line { x: 20.0, y: 0.0 },
    ];
    let n = VectorNetwork::from_path(&wave, false, FillRule::NonZero, TOL);
    assert!(n.regions.is_empty());
    let p = n.to_paths();
    assert!(p.regions.is_empty());
    assert_eq!(p.open_runs.len(), 1);
    assert!(!p.open_runs[0].closed);
    assert!(pts_eq(&p.open_runs[0].points, &wave));
}

#[test]
fn quadratics_and_arcs_become_cubics() {
    let pts = vec![
        PathPoint::Move { x: 0.0, y: 0.0 },
        PathPoint::Quadratic {
            x: 10.0,
            y: 0.0,
            cp: (5.0, 10.0),
        },
        PathPoint::Arc {
            x: 0.0,
            y: 0.0,
            rx: 5.0,
            ry: 5.0,
            x_axis_rotation: 0.0,
            large_arc: false,
            sweep: true,
        },
    ];
    let n = VectorNetwork::from_path(&pts, true, FillRule::NonZero, TOL);
    // Quadratic: exact degree elevation.
    let q = n.segment_cubic(0);
    for k in 0..=8 {
        let t = k as f64 / 8.0;
        let mt = 1.0 - t;
        let want = v(10.0 * t * t + 2.0 * mt * t * 5.0, 2.0 * mt * t * 10.0);
        assert!(q.eval(t).dist(want) < 1e-9);
    }
    // The half-circle arc closes back onto vertex 0.
    assert!(n.segments.len() >= 2);
    let last = n.segments.last().unwrap();
    assert_eq!(last.end, 0);
    let mid = n.segment_cubic(1).eval(1.0);
    assert!((mid.dist(v(5.0, 0.0)) - 5.0).abs() < 1e-6);
}

#[test]
fn network_to_paths_to_network_keeps_topology_after_planarize() {
    // Plus sign inside a square, with the left quarter filled.
    let mut n = lines(
        &[
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (5.0, 0.0),
            (5.0, 10.0),
            (0.0, 5.0),
            (10.0, 5.0),
        ],
        &[(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (6, 7)],
    );
    n.planarize(TOL);
    n.toggle_fill_at(v(2.0, 2.0), TOL);
    let paths = n.to_paths();
    assert_eq!(paths.regions.len(), 1);
    let runs = n.stroke_runs();

    let mut back = VectorNetwork::new();
    for r in &runs {
        let piece = VectorNetwork::from_path(&r.points, r.closed, FillRule::NonZero, TOL);
        let off = back.vertices.len();
        back.vertices.extend(piece.vertices);
        for mut s in piece.segments {
            s.start += off;
            s.end += off;
            back.segments.push(s);
        }
    }
    back.planarize(TOL);
    assert_eq!(back.vertices.len(), n.vertices.len());
    assert_eq!(back.segments.len(), n.segments.len());
    let (fa, fb) = (face_areas(&back), face_areas(&n));
    assert_eq!(fa.len(), fb.len());
    for (a, b) in fa.iter().zip(&fb) {
        assert!(close(*a, *b, 1e-9));
    }
}

#[test]
fn unchainable_region_loops_are_skipped_in_paths() {
    let mut n = lines(
        &[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
        &[(0, 1), (2, 3)],
    );
    n.regions
        .push(VectorRegion::new(WindingRule::NonZero, vec![vec![0, 1]]));
    let p = n.to_paths();
    assert_eq!(p.skipped_loops, 1);
    assert!(p.regions.is_empty());
    assert_eq!(p.open_runs.len(), 2);
}


#[test]
fn extreme_finite_grid_keys_do_not_overflow_neighbors() {
    for coord in [-1e20, 1e20, -1e300, 1e300] {
        for tolerance in [1e-20, 0.01] {
            let request = crate::value::json!({
                "op": "planarize", "tolerance": tolerance,
                "network": {"vertices": [{"x":coord,"y":coord}], "segments": []}
            });
            let reply: crate::value::Value =
                crate::value::from_str(&vector_network_json(&request.to_string())).unwrap();
            assert_eq!(reply["status"], "ok");
            assert_eq!(reply["network"]["vertices"].as_array().unwrap().len(), 1);
        }
    }
}
