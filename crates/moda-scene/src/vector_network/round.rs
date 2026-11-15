//! Corner rounding on a vector network (Figma's vertex `cornerRadius`): the
//! network rewritten with its rounded vertices filleted, which every lane then
//! draws like any other network (fills per region, strokes per segment).
//!
//! A vertex rounds with its own `cornerRadius`, else the node's (a path's
//! `cornerRadius`). At a rounded vertex the segment ends that meet there are
//! sorted by the direction they leave in, and every pair of neighbours is
//! filleted:
//!
//! - **Degree 2** rounds exactly as `roundPathCorners` rounds a path's corner
//!   (`path_geom::round_corners`, same constants and arc-length sampling): each
//!   segment is cut back by `d = r·tan(φ/2)` (φ the turn, `d` clamped to half
//!   of either segment) and a cubic arc with handles `4/3·tan(φ/4)·r` joins the
//!   cut points. The vertex itself drops out.
//! - **A junction (3+ segment ends)** fillets each pair of neighbours whose
//!   gap is under a half turn, so a plus sign's hub becomes four arcs, each
//!   spoke flowing into both of its neighbours. A segment end cut for two
//!   fillets is cut at the larger of the two distances (the arcs stay tangent;
//!   they are circular when the distances agree). A pair a half turn or more
//!   apart (the straight side of a T) is not rounded: both ends run on to the
//!   vertex as they did, through short stubs, which keep it.
//! - Degree 1 (an open end) and a vertex whose segments do not turn are left
//!   as drawn.
//!
//! Regions follow: a region loop through a rounded vertex runs through the
//! fillet between its two segments (or the stubs, when that pair is not
//! rounded), so a filled face gets the same rounded corner its stroke does.
//! Segment `i` of the result is what is left of segment `i`; the fillets and
//! stubs come after.
//!
//! The TypeScript mirror is `roundVectorNetwork` in
//! `packages/editor/src/core/geometry/vectorNetwork.ts`; both are pinned to
//! `fixtures/vector-network-rounding/cases.json`.

use std::collections::HashMap;

use super::geom::{Cubic, Vector};
use super::{VectorNetwork, VectorRegion, VectorSegment, VectorVertex};

/// Below this a length is zero and a direction undefined.
const EPS: f64 = 1e-9;
/// A turn smaller than this (radians) is a smooth join, not a corner.
const MIN_TURN: f64 = 1e-4;
/// A turn this close to a full reversal is a cusp; no fillet fits.
const MAX_TURN: f64 = std::f64::consts::PI - 1e-4;
/// Uniform samples used to measure a cubic's arc length and invert it.
const CUBIC_LENGTH_SAMPLES: usize = 32;

/// One segment end: segment `seg` at its start (`at_start`) or its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct End {
    seg: usize,
    at_start: bool,
}

#[derive(Debug, Clone, Copy)]
struct Fillet {
    a: End,
    b: End,
    turn: f64,
    /// `d / tan(turn / 2)` for each end's own cut `d`.
    radius_a: f64,
    radius_b: f64,
}

fn cubic_cumulative(c: &Cubic) -> [f64; CUBIC_LENGTH_SAMPLES + 1] {
    let mut cum = [0.0; CUBIC_LENGTH_SAMPLES + 1];
    let mut prev = c.p0;
    for (k, slot) in cum.iter_mut().enumerate().skip(1) {
        let p = c.eval(k as f64 / CUBIC_LENGTH_SAMPLES as f64);
        *slot = prev.dist(p);
        prev = p;
    }
    for k in 1..=CUBIC_LENGTH_SAMPLES {
        cum[k] += cum[k - 1];
    }
    cum
}

/// The parameter at arc length `s` (sampled, linearly inverted).
fn param_at_length(c: &Cubic, s: f64) -> f64 {
    let cum = cubic_cumulative(c);
    let total = cum[CUBIC_LENGTH_SAMPLES];
    if total < EPS || s <= 0.0 {
        return 0.0;
    }
    if s >= total {
        return 1.0;
    }
    let mut k = 1;
    while k < CUBIC_LENGTH_SAMPLES && cum[k] < s {
        k += 1;
    }
    let span = cum[k] - cum[k - 1];
    let frac = if span < EPS {
        0.0
    } else {
        (s - cum[k - 1]) / span
    };
    (k as f64 - 1.0 + frac) / CUBIC_LENGTH_SAMPLES as f64
}

fn unit(v: Vector) -> Option<Vector> {
    let l = v.len();
    (l >= EPS).then(|| v * (1.0 / l))
}

impl VectorNetwork {
    fn seg_straight(&self, s: usize) -> bool {
        self.segments[s].is_straight()
    }

    fn seg_length(&self, s: usize) -> f64 {
        let c = self.segment_cubic(s);
        if self.seg_straight(s) {
            c.p0.dist(c.p3)
        } else {
            cubic_cumulative(&c)[CUBIC_LENGTH_SAMPLES]
        }
    }

    /// The segment as a cubic leaving `end`'s vertex.
    fn leaving(&self, e: End) -> Cubic {
        let c = self.segment_cubic(e.seg);
        if e.at_start {
            c
        } else {
            c.reversed()
        }
    }

    /// The unit direction a segment end leaves its vertex in (the first
    /// control point that is not on it, as `roundPathCorners` reads it).
    fn leave_direction(&self, e: End) -> Option<Vector> {
        let c = self.leaving(e);
        if self.seg_straight(e.seg) {
            return unit(c.p3 - c.p0);
        }
        unit(c.p1 - c.p0)
            .or_else(|| unit(c.p2 - c.p0))
            .or_else(|| unit(c.p3 - c.p0))
    }

    /// The parameter (on the segment as stored) `d` along it from `e`'s vertex.
    fn param_from_end(&self, e: End, d: f64) -> f64 {
        let t = if self.seg_straight(e.seg) {
            let len = self.seg_length(e.seg);
            if len < EPS {
                0.0
            } else {
                (d / len).clamp(0.0, 1.0)
            }
        } else {
            param_at_length(&self.leaving(e), d)
        };
        if e.at_start {
            t
        } else {
            1.0 - t
        }
    }

    /// The network with its rounded vertices filleted: each vertex rounds with
    /// its own `cornerRadius`, else `default_radius`. A network with nothing to
    /// round comes back unchanged.
    pub fn rounded(&self, default_radius: f64) -> VectorNetwork {
        let radii: Vec<f64> = self
            .vertices
            .iter()
            .map(|v| v.corner_radius.unwrap_or(default_radius))
            .map(|r| if r.is_finite() && r > 0.0 { r } else { 0.0 })
            .collect();
        if radii.iter().all(|&r| r <= 0.0) {
            return self.clone();
        }

        let nv = self.vertices.len();
        let mut incident: Vec<Vec<End>> = vec![Vec::new(); nv];
        for (i, s) in self.segments.iter().enumerate() {
            incident[s.start].push(End {
                seg: i,
                at_start: true,
            });
            incident[s.end].push(End {
                seg: i,
                at_start: false,
            });
        }
        let lengths: Vec<f64> = (0..self.segments.len())
            .map(|s| self.seg_length(s))
            .collect();

        // Plan: the fillets and each end's cut distance.
        let mut cut: HashMap<End, f64> = HashMap::new();
        let mut fillets: Vec<Fillet> = Vec::new();
        // Vertices that stay: a junction with an unrounded side, or ends left uncut.
        let mut rounded_vertex = vec![false; nv];
        // Ends that meet an unrounded neighbour at a junction: they keep a stub.
        let mut stub_ends: Vec<End> = Vec::new();
        for v in 0..nv {
            let r = radii[v];
            let ends = &incident[v];
            if r <= 0.0 || ends.len() < 2 {
                continue;
            }
            let mut dirs: Vec<(f64, End, Vector)> = Vec::with_capacity(ends.len());
            let mut degenerate = false;
            for &e in ends {
                match self.leave_direction(e) {
                    Some(d) => dirs.push((d.y.atan2(d.x), e, d)),
                    None => degenerate = true,
                }
            }
            if degenerate {
                continue;
            }
            dirs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let n = dirs.len();
            // Two ends are one corner; a junction has one per neighbouring pair.
            let pairs: Vec<(usize, usize)> = if n == 2 {
                vec![(0, 1)]
            } else {
                (0..n).map(|k| (k, (k + 1) % n)).collect()
            };
            let mut planned: Vec<(End, End, f64, f64)> = Vec::new();
            let mut unrounded: Vec<(End, End)> = Vec::new();
            for (i, j) in pairs {
                let (ai, ea, da) = dirs[i];
                let (aj, eb, db) = dirs[j];
                let gap = if n == 2 {
                    da.dot(db).clamp(-1.0, 1.0).acos()
                } else {
                    let mut g = aj - ai;
                    if j == 0 {
                        g += std::f64::consts::TAU;
                    }
                    g
                };
                let turn = std::f64::consts::PI - gap;
                if !(MIN_TURN..=MAX_TURN).contains(&turn) {
                    unrounded.push((ea, eb));
                    continue;
                }
                let wanted = r * (turn / 2.0).tan();
                let d = wanted.min(lengths[ea.seg] / 2.0).min(lengths[eb.seg] / 2.0);
                if d < EPS {
                    unrounded.push((ea, eb));
                    continue;
                }
                planned.push((ea, eb, turn, d));
            }
            if planned.is_empty() {
                continue;
            }
            rounded_vertex[v] = true;
            for &(ea, eb, _, d) in &planned {
                for e in [ea, eb] {
                    let c = cut.entry(e).or_insert(0.0);
                    *c = c.max(d);
                }
            }
            for &(ea, eb, turn, _) in &planned {
                let half = (turn / 2.0).tan();
                fillets.push(Fillet {
                    a: ea,
                    b: eb,
                    turn,
                    radius_a: cut[&ea] / half,
                    radius_b: cut[&eb] / half,
                });
            }
            for (ea, eb) in unrounded {
                stub_ends.push(ea);
                stub_ends.push(eb);
            }
        }
        if fillets.is_empty() {
            return self.clone();
        }

        let mut out = VectorNetwork {
            vertices: self
                .vertices
                .iter()
                .map(|v| VectorVertex {
                    corner_radius: None,
                    ..v.clone()
                })
                .collect(),
            segments: Vec::with_capacity(self.segments.len() + fillets.len()),
            regions: Vec::new(),
        };
        // A new vertex at every cut end.
        let mut cut_vertex: HashMap<End, usize> = HashMap::new();
        let mut cut_param: HashMap<End, f64> = HashMap::new();
        let mut ends: Vec<End> = cut.keys().copied().collect();
        ends.sort_by_key(|e| (e.seg, !e.at_start));
        for e in ends {
            let t = self.param_from_end(e, cut[&e]);
            let p = self.segment_cubic(e.seg).eval(t);
            let p = if self.seg_straight(e.seg) {
                let s = &self.segments[e.seg];
                self.vertex_pos(s.start).lerp(self.vertex_pos(s.end), t)
            } else {
                p
            };
            cut_param.insert(e, t);
            cut_vertex.insert(e, out.vertices.len());
            out.vertices.push(VectorVertex::at(p.x, p.y));
        }
        let end_vertex = |e: End| {
            let s = &self.segments[e.seg];
            if e.at_start {
                s.start
            } else {
                s.end
            }
        };
        // Write `c` (control points absolute) as a segment `a → b`.
        let push = |out: &mut VectorNetwork, a: usize, b: usize, c: &Cubic, straight: bool| {
            let seg = if straight {
                VectorSegment::line(a, b)
            } else {
                VectorSegment {
                    start: a,
                    end: b,
                    tangent_start: c.p1 - out.vertex_pos(a),
                    tangent_end: c.p2 - out.vertex_pos(b),
                }
            };
            out.segments.push(seg);
            out.segments.len() - 1
        };
        // Cores: segment i, cut back at either end.
        for (i, s) in self.segments.iter().enumerate() {
            let sa = End {
                seg: i,
                at_start: true,
            };
            let sb = End {
                seg: i,
                at_start: false,
            };
            let t0 = cut_param.get(&sa).copied().unwrap_or(0.0);
            let t1 = cut_param.get(&sb).copied().unwrap_or(1.0);
            let a = cut_vertex.get(&sa).copied().unwrap_or(s.start);
            let b = cut_vertex.get(&sb).copied().unwrap_or(s.end);
            let c = self.segment_cubic(i);
            let piece = if t1 > t0 {
                c.subsegment(t0, t1)
            } else {
                Cubic::line(c.eval(t0), c.eval(t0))
            };
            push(&mut out, a, b, &piece, s.is_straight());
        }
        // Stubs: from a cut end on to its vertex, kept where a junction side
        // does not round (and where a region needs one, below).
        let mut stubs: HashMap<End, usize> = HashMap::new();
        let mut stub = |out: &mut VectorNetwork, e: End| -> Option<usize> {
            let cv = *cut_vertex.get(&e)?;
            if let Some(&s) = stubs.get(&e) {
                return Some(s);
            }
            let t = cut_param[&e];
            let c = self.segment_cubic(e.seg);
            // From the cut point to the vertex.
            let piece = if e.at_start {
                c.subsegment(0.0, t).reversed()
            } else {
                c.subsegment(t, 1.0)
            };
            let idx = push(out, cv, end_vertex(e), &piece, self.seg_straight(e.seg));
            stubs.insert(e, idx);
            Some(idx)
        };
        let mut fillet_index: HashMap<(End, End), usize> = HashMap::new();
        for f in &fillets {
            let (ca, cb) = (cut_vertex[&f.a], cut_vertex[&f.b]);
            let (pa, pb) = (out.vertex_pos(ca), out.vertex_pos(cb));
            let v = self.vertex_pos(end_vertex(f.a));
            // Heading into the corner at each cut point: along the cut segment
            // toward the vertex.
            let toward = |e: End, p: Vector| {
                let c = self.segment_cubic(e.seg);
                let t = cut_param[&e];
                let d = if self.seg_straight(e.seg) {
                    v - p
                } else if e.at_start {
                    -c.deriv(t)
                } else {
                    c.deriv(t)
                };
                unit(d).or_else(|| unit(v - p)).unwrap_or(Vector::ZERO)
            };
            let (ta, tb) = (toward(f.a, pa), toward(f.b, pb));
            let k = (4.0 / 3.0) * (f.turn / 4.0).tan();
            let c = Cubic::new(
                pa,
                pa + ta * (k * f.radius_a),
                pb + tb * (k * f.radius_b),
                pb,
            );
            let idx = push(&mut out, ca, cb, &c, false);
            fillet_index.insert((f.a, f.b), idx);
            fillet_index.insert((f.b, f.a), idx);
        }
        for e in std::mem::take(&mut stub_ends) {
            stub(&mut out, e);
        }

        // Regions: each loop runs through the fillet (or stubs) at every
        // rounded vertex it turns at.
        for region in &self.regions {
            let mut loops = Vec::with_capacity(region.loops.len());
            for lp in &region.loops {
                let Some((orient, closed)) = self.chain_loop(lp) else {
                    loops.push(lp.clone());
                    continue;
                };
                let n = orient.len();
                let mut new_loop = Vec::with_capacity(n * 2);
                for k in 0..n {
                    let (si, rev) = orient[k];
                    new_loop.push(si);
                    if k + 1 == n && !closed {
                        break;
                    }
                    let (sj, rev_j) = orient[(k + 1) % n];
                    let ei = End {
                        seg: si,
                        at_start: rev,
                    };
                    let ej = End {
                        seg: sj,
                        at_start: !rev_j,
                    };
                    if let Some(&f) = fillet_index.get(&(ei, ej)) {
                        new_loop.push(f);
                        continue;
                    }
                    if let Some(s) = stub(&mut out, ei) {
                        new_loop.push(s);
                    }
                    if let Some(s) = stub(&mut out, ej) {
                        new_loop.push(s);
                    }
                }
                loops.push(new_loop);
            }
            out.regions.push(VectorRegion {
                loops,
                ..region.clone()
            });
        }

        // A rounded vertex nothing reaches any more drops out.
        let mut used = vec![false; out.vertices.len()];
        for s in &out.segments {
            used[s.start] = true;
            used[s.end] = true;
        }
        let drop: Vec<bool> = (0..out.vertices.len())
            .map(|v| v < nv && rounded_vertex[v] && !used[v])
            .collect();
        out.remove_vertices_masked(&drop);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn net(v: crate::value::Value) -> VectorNetwork {
        VectorNetwork::from_json_value(&v).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// A plus sign with a hub vertex: four spokes of 50 from (50, 50).
    fn plus(radius: f64) -> VectorNetwork {
        net(json!({
            "vertices": [{"x": 50, "y": 50, "cornerRadius": radius}, {"x": 0, "y": 50}, {"x": 100, "y": 50}, {"x": 50, "y": 0}, {"x": 50, "y": 100}],
            "segments": [{"start": 0, "end": 1}, {"start": 0, "end": 2}, {"start": 0, "end": 3}, {"start": 0, "end": 4}],
        }))
    }

    #[test]
    fn nothing_to_round_is_unchanged() {
        let n = plus(0.0);
        assert_eq!(n.rounded(0.0), n);
        assert_eq!(n.rounded(-3.0), n);
    }

    #[test]
    fn a_plus_hub_becomes_four_quarter_arcs() {
        let r = plus(10.0).rounded(0.0);
        // The hub drops out; four spokes cut back by r (a right angle's tangent length is r), four fillets.
        assert_eq!(r.segments.len(), 8);
        assert_eq!(r.vertices.len(), 8);
        for i in 0..4 {
            let c = r.segment_cubic(i);
            let hub_end = c.p0;
            assert!(
                close(hub_end.dist(Vector::new(50.0, 50.0)), 10.0),
                "{hub_end:?}"
            );
        }
        // Every fillet is a quarter circle of radius 10 about the corner of the square its cut points span
        // with the hub (the hub's diagonal opposite).
        for i in 4..8 {
            let c = r.segment_cubic(i);
            let (a, b) = (c.p0, c.p3);
            // Centre: the corner of the square spanned by the two cut points and the hub.
            let centre = Vector::new(a.x + b.x - 50.0, a.y + b.y - 50.0);
            let mid = c.eval(0.5);
            assert!(
                (mid.dist(centre) - 10.0).abs() < 0.01,
                "{}",
                mid.dist(centre)
            );
            // Tangent to the spokes: the handles run along them toward the hub.
            assert!(close((c.p1 - a).cross(Vector::new(50.0, 50.0) - a), 0.0));
        }
        // Each cut vertex joins its spoke and two fillets.
        for v in 0..r.vertices.len() {
            let d = r.degree(v);
            assert!(d == 1 || d == 3, "vertex {v} has degree {d}");
        }
    }

    #[test]
    fn a_y_junction_rounds_each_pair_of_neighbours() {
        // Three spokes 120° apart: each turn is 60°, so the cut is r·tan(30°).
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0, "cornerRadius": 12}, {"x": 0, "y": -60}, {"x": 51.96152422706632, "y": 30}, {"x": -51.96152422706632, "y": 30}],
            "segments": [{"start": 0, "end": 1}, {"start": 0, "end": 2}, {"start": 0, "end": 3}],
        }));
        let r = n.rounded(0.0);
        assert_eq!(r.segments.len(), 6);
        let want = 12.0 * (std::f64::consts::PI / 6.0).tan();
        for i in 0..3 {
            assert!(close(r.segment_cubic(i).p0.len(), want));
        }
    }

    #[test]
    fn a_t_junction_keeps_its_straight_side_through_the_vertex() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0, "cornerRadius": 5}, {"x": -40, "y": 0}, {"x": 40, "y": 0}, {"x": 0, "y": 40}],
            "segments": [{"start": 0, "end": 1}, {"start": 0, "end": 2}, {"start": 0, "end": 3}],
        }));
        let r = n.rounded(0.0);
        // 3 cores, 2 fillets into the stem, 2 stubs along the top back to the vertex, which stays.
        assert_eq!(r.segments.len(), 7);
        assert_eq!(r.vertices.len(), 4 + 3);
        assert_eq!(r.degree(0), 2);
    }

    #[test]
    fn degree_two_matches_path_corner_rounding() {
        use crate::blend::BlendPathPoint;
        use crate::path_geom::round_corners::round_path_corners;
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 100, "y": 60}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
        }));
        let r = n.rounded(8.0);
        let path = round_path_corners(
            &[
                BlendPathPoint::Move { x: 0.0, y: 0.0 },
                BlendPathPoint::Line { x: 100.0, y: 0.0 },
                BlendPathPoint::Line { x: 100.0, y: 60.0 },
            ],
            false,
            8.0,
        );
        let BlendPathPoint::Curve { x, y, cp1, cp2 } = path[2].clone() else {
            panic!("{path:?}")
        };
        let mut f = r.segment_cubic(2);
        if !close(f.p3.x, x) {
            f = f.reversed();
        }
        assert!(close(f.p3.x, x) && close(f.p3.y, y));
        assert!(close(f.p1.x, cp1.0) && close(f.p1.y, cp1.1));
        assert!(close(f.p2.x, cp2.0) && close(f.p2.y, cp2.1));
        // The corner vertex is gone; the two ends and two cut points stay.
        assert_eq!(r.vertices.len(), 4);
    }

    #[test]
    fn a_rounded_region_runs_through_its_fillets() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 40, "y": 40}, {"x": 0, "y": 40}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 3}, {"start": 3, "end": 0}],
            "regions": [{"windingRule": "NONZERO", "loops": [[0, 1, 2, 3]]}],
        }));
        let r = n.rounded(6.0);
        assert_eq!(r.regions[0].loops[0].len(), 8);
        let (_, closed) = r.chain_loop(&r.regions[0].loops[0]).unwrap();
        assert!(closed);
        // A rounded square of 40 with r = 6 loses 4·(1 − π/4)·36 of its area.
        let paths = r.to_paths();
        let (orient, _) = r.chain_loop(&r.regions[0].loops[0]).unwrap();
        let area: f64 = orient
            .iter()
            .map(|&(s, rev)| r.oriented_cubic(s, rev).signed_area())
            .sum::<f64>()
            .abs();
        let want = 1600.0 - 4.0 * (1.0 - std::f64::consts::PI / 4.0) * 36.0;
        assert!((area - want).abs() < 0.05, "{area} vs {want}");
        assert_eq!(paths.regions.len(), 1);
    }

    #[test]
    fn a_region_through_a_junction_uses_the_matching_fillet() {
        // A square split by a vertical divider: two faces meeting the divider at T junctions.
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 80, "y": 0}, {"x": 80, "y": 40}, {"x": 40, "y": 40}, {"x": 0, "y": 40}],
            "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 3}, {"start": 3, "end": 4},
                         {"start": 4, "end": 5}, {"start": 5, "end": 0}, {"start": 1, "end": 4}],
            "regions": [{"windingRule": "NONZERO", "loops": [[0, 6, 4, 5]]}, {"windingRule": "NONZERO", "loops": [[1, 2, 3, 6]]}],
        }));
        let r = n.rounded(5.0);
        for region in &r.regions {
            let (_, closed) = r.chain_loop(&region.loops[0]).unwrap();
            assert!(closed);
            // Four corners, each one fillet: no stubs in a face.
            assert_eq!(region.loops[0].len(), 8);
        }
    }

    #[test]
    fn curved_segments_are_cut_by_arc_length() {
        let n = net(json!({
            "vertices": [{"x": 0, "y": 0}, {"x": 100, "y": 0, "cornerRadius": 10}, {"x": 100, "y": 100}],
            "segments": [{"start": 0, "end": 1, "tangentStart": {"x": 30, "y": -30}, "tangentEnd": {"x": -30, "y": 0}},
                         {"start": 1, "end": 2}],
        }));
        let r = n.rounded(0.0);
        let core = r.segment_cubic(0);
        assert!(!r.segments[0].is_straight());
        // The fillet leaves the curved core tangent to it.
        let mut f = r.segment_cubic(2);
        if f.p0.dist(core.p3) > 1e-9 {
            f = f.reversed();
        }
        assert!(f.p0.dist(core.p3) < 1e-9);
        let along = core.deriv(1.0);
        assert!(close(
            unit(along).unwrap().cross(unit(f.p1 - f.p0).unwrap()),
            0.0
        ));
    }
}
