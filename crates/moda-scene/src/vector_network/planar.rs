//! Planarisation: make the network a planar graph whose segments meet only at
//! vertices, which is what the face walk (`faces.rs`) needs.
//!
//! Each pass:
//! 1. **Clean up**: merge vertices closer than the tolerance, drop zero-length
//!    segments, split closed self-loop segments in two (the face walk needs
//!    distinct endpoints), and merge coincident segments.
//! 2. **Touches**: split a segment wherever some vertex lies on its interior
//!    (T-junctions, and the ends of an overlapping stretch). Clean up, which
//!    merges the now-identical overlapping pieces.
//! 3. **Crossings**: split segments at every crossing with another segment and
//!    at every self-crossing. Both curves of a crossing land on one shared
//!    vertex. Clean up.
//!
//! Passes repeat until one changes nothing (at most [`MAX_PASSES`]); a second
//! pass only ever finds work when snapping moved a vertex onto a curve.

use std::collections::HashMap;

use super::geom::{Cubic, Rect, Vector};
use super::intersect::{intersect, self_intersections, Hits};
use super::VectorNetwork;

const MAX_PASSES: usize = 4;

/// What [`VectorNetwork::planarize`] changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanarizeReport {
    /// Vertices merged into a coincident neighbour.
    pub vertices_merged: usize,
    /// Zero-length or duplicate segments deleted.
    pub segments_removed: usize,
    /// Vertices inserted at crossings, touches and self-loop midpoints.
    pub vertices_added: usize,
    /// Segment splits performed (one per split point).
    pub splits: usize,
}

impl PlanarizeReport {
    pub fn changed(&self) -> bool {
        *self != PlanarizeReport::default()
    }
}

/// Spatial hash of vertices for "find a vertex within `tol`, else add one".
struct VertexGrid {
    cell: f64,
    tol: f64,
    buckets: HashMap<(i64, i64), Vec<usize>>,
}

impl VertexGrid {
    fn new(net: &VectorNetwork, tol: f64) -> VertexGrid {
        let mut g = VertexGrid {
            cell: tol.max(1e-12) * 2.0,
            tol,
            buckets: HashMap::new(),
        };
        for (i, v) in net.vertices.iter().enumerate() {
            g.insert(i, v.pos());
        }
        g
    }
    fn key(&self, p: Vector) -> (i64, i64) {
        (
            (p.x / self.cell).floor() as i64,
            (p.y / self.cell).floor() as i64,
        )
    }
    fn insert(&mut self, i: usize, p: Vector) {
        let k = self.key(p);
        self.buckets.entry(k).or_default().push(i);
    }
    fn nearest(&self, net: &VectorNetwork, p: Vector) -> Option<usize> {
        let (kx, ky) = self.key(p);
        let mut best: Option<(usize, f64)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                // Float-to-integer casts saturate: extreme finite coordinates share
                // an edge bucket. Distance checks still decide membership; never
                // wrap a neighbour key across the integer boundary.
                let (Some(x), Some(y)) = (kx.checked_add(dx), ky.checked_add(dy)) else {
                    continue;
                };
                let Some(b) = self.buckets.get(&(x, y)) else {
                    continue;
                };
                for &i in b {
                    let d = net.vertex_pos(i).dist(p);
                    if d < self.tol && best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((i, d));
                    }
                }
            }
        }
        best.map(|(i, _)| i)
    }
    fn find_or_add(&mut self, net: &mut VectorNetwork, p: Vector, added: &mut usize) -> usize {
        if let Some(i) = self.nearest(net, p) {
            return i;
        }
        let i = net.add_vertex(p);
        self.insert(i, p);
        *added += 1;
        i
    }
}

fn sanitize_tol(tol: f64) -> f64 {
    if tol.is_finite() && tol > 0.0 {
        tol
    } else {
        super::DEFAULT_TOLERANCE
    }
}

impl VectorNetwork {
    /// Make the network planar (see the module doc). Existing regions are kept
    /// in step: a loop through a split segment gets its pieces, and a loop
    /// through a merged duplicate gets the survivor. Vertex attributes of a
    /// merged vertex are the lower-indexed vertex's.
    pub fn planarize(&mut self, tol: f64) -> PlanarizeReport {
        let tol = sanitize_tol(tol);
        let mut rep = PlanarizeReport::default();
        if self.validate().is_err() {
            return rep;
        }
        self.cleanup(tol, &mut rep);
        for _ in 0..MAX_PASSES {
            let before = rep;
            let touches = self.touch_splits(tol);
            self.apply_splits(touches, tol, &mut rep);
            self.cleanup(tol, &mut rep);
            let crossings = self.crossing_splits(tol);
            self.apply_splits(crossings, tol, &mut rep);
            self.cleanup(tol, &mut rep);
            if rep == before {
                break;
            }
        }
        rep
    }

    /// True when no segment crosses or touches another away from a shared
    /// vertex (what [`planarize`](Self::planarize) establishes).
    pub fn is_planar(&self, tol: f64) -> bool {
        let tol = sanitize_tol(tol);
        if self.segments.iter().any(|s| s.start == s.end) {
            return false;
        }
        let touches = self.touch_splits(tol);
        let crossings = self.crossing_splits(tol);
        touches.iter().chain(crossings.iter()).all(Vec::is_empty)
    }

    fn cleanup(&mut self, tol: f64, rep: &mut PlanarizeReport) {
        let (merged, dropped) = self.merge_coincident_vertices(tol);
        rep.vertices_merged += merged;
        rep.segments_removed += dropped;
        rep.segments_removed += self.remove_degenerate_segments(tol);
        self.split_self_loops(tol, rep);
        rep.segments_removed += self.dedupe_segments(tol);
    }

    /// Merge each vertex into the first earlier vertex within `tol`. Returns
    /// (vertices merged, segments that collapsed to nothing).
    fn merge_coincident_vertices(&mut self, tol: f64) -> (usize, usize) {
        let mut grid = VertexGrid {
            cell: tol * 2.0,
            tol,
            buckets: HashMap::new(),
        };
        let mut map: Vec<usize> = (0..self.vertices.len()).collect();
        let mut merged = 0;
        for (i, slot) in map.iter_mut().enumerate() {
            let p = self.vertex_pos(i);
            if let Some(j) = grid.nearest(self, p) {
                *slot = j;
                merged += 1;
            } else {
                grid.insert(i, p);
            }
        }
        let dropped = if merged > 0 {
            self.remap_vertices(&map)
        } else {
            0
        };
        (merged, dropped)
    }

    /// Delete segments with no extent: a straight segment from a vertex to
    /// itself, or a curve whose control points all sit on its vertex.
    fn remove_degenerate_segments(&mut self, tol: f64) -> usize {
        let mask: Vec<bool> = (0..self.segments.len())
            .map(|i| {
                let s = &self.segments[i];
                if s.start != s.end {
                    return false;
                }
                if s.is_straight() {
                    return true;
                }
                let c = self.segment_cubic(i);
                c.p1.dist(c.p0) < tol && c.p2.dist(c.p0) < tol
            })
            .collect();
        let n = mask.iter().filter(|&&m| m).count();
        if n > 0 {
            self.remove_segments_masked(&mask, false);
        }
        n
    }

    /// A segment from a vertex back to itself is split at its farthest point.
    fn split_self_loops(&mut self, tol: f64, rep: &mut PlanarizeReport) {
        let mut grid = VertexGrid::new(self, tol);
        for i in 0..self.segments.len() {
            let s = &self.segments[i];
            if s.start != s.end {
                continue;
            }
            let c = self.segment_cubic(i);
            let mut best = (0.5, 0.0);
            for k in 1..64 {
                let t = k as f64 / 64.0;
                let d = c.eval(t).dist(c.p0);
                if d > best.1 {
                    best = (t, d);
                }
            }
            let mut added = 0;
            let v = grid.find_or_add(self, c.eval(best.0), &mut added);
            rep.vertices_added += added;
            if v == self.segments[i].start {
                continue;
            }
            self.split_segment_multi(i, &[(best.0, v)]);
            rep.splits += 1;
        }
    }

    /// Merge segments that join the same two vertices along the same curve.
    fn dedupe_segments(&mut self, tol: f64) -> usize {
        let mut by_ends: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, s) in self.segments.iter().enumerate() {
            by_ends
                .entry((s.start.min(s.end), s.start.max(s.end)))
                .or_default()
                .push(i);
        }
        let mut mask = vec![false; self.segments.len()];
        let mut groups: Vec<Vec<usize>> = by_ends.into_values().filter(|g| g.len() > 1).collect();
        groups.sort();
        for g in groups {
            for a in 0..g.len() {
                if mask[g[a]] {
                    continue;
                }
                for b in (a + 1)..g.len() {
                    if mask[g[b]] {
                        continue;
                    }
                    if self.coincident(g[a], g[b], tol) {
                        mask[g[b]] = true;
                        self.replace_segment_refs(g[b], g[a]);
                    }
                }
            }
        }
        let n = mask.iter().filter(|&&m| m).count();
        if n > 0 {
            self.remove_segments_masked(&mask, false);
        }
        n
    }

    /// Two segments with the same end vertices trace the same curve (checked
    /// by projecting samples of each onto the other).
    fn coincident(&self, i: usize, j: usize, tol: f64) -> bool {
        let a = self.segment_cubic(i);
        let b = self.segment_cubic(j);
        let near = |x: &Cubic, y: &Cubic| {
            (1..8).all(|k| {
                let p = x.eval(k as f64 / 8.0);
                y.closest_param(p).1 < tol * 2.0
            })
        };
        near(&a, &b) && near(&b, &a)
    }

    fn segment_bounds(&self) -> (Vec<Cubic>, Vec<Rect>) {
        let cubics: Vec<Cubic> = (0..self.segments.len())
            .map(|i| self.segment_cubic(i))
            .collect();
        let bounds = cubics.iter().map(Cubic::hull_bounds).collect();
        (cubics, bounds)
    }

    /// For each segment, the parameters where some vertex other than its own
    /// ends lies on its interior.
    fn touch_splits(&self, tol: f64) -> Vec<Vec<f64>> {
        let (cubics, bounds) = self.segment_bounds();
        let mut out = vec![Vec::new(); self.segments.len()];
        for (i, s) in self.segments.iter().enumerate() {
            let c = &cubics[i];
            for (v, vert) in self.vertices.iter().enumerate() {
                if v == s.start || v == s.end {
                    continue;
                }
                let p = vert.pos();
                if !bounds[i].contains(p, tol) {
                    continue;
                }
                let (t, d) = c.closest_param(p);
                if d < tol && c.eval(t).dist(c.p0) >= tol && c.eval(t).dist(c.p3) >= tol {
                    out[i].push(t);
                }
            }
        }
        out
    }

    /// For each segment, the parameters where it crosses another segment or
    /// itself, away from any segment end.
    fn crossing_splits(&self, tol: f64) -> Vec<Vec<f64>> {
        let (cubics, bounds) = self.segment_bounds();
        let n = self.segments.len();
        let mut out = vec![Vec::new(); n];
        for i in 0..n {
            if !self.segments[i].is_straight() {
                for (t1, t2) in self_intersections(&cubics[i], tol) {
                    out[i].push(t1);
                    out[i].push(t2);
                }
            }
        }
        // Sweep on x so far-apart pairs are never compared.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| bounds[a].min.x.total_cmp(&bounds[b].min.x));
        for (oi, &i) in order.iter().enumerate() {
            for &j in &order[oi + 1..] {
                if bounds[j].min.x > bounds[i].max.x + tol {
                    break;
                }
                if !bounds[i].overlaps(&bounds[j], tol) {
                    continue;
                }
                let (ci, cj) = (&cubics[i], &cubics[j]);
                let Hits::Points(hits) = intersect(ci, cj, tol) else {
                    continue;
                };
                for (s, t) in hits {
                    let p = ci.eval(s);
                    if [ci.p0, ci.p3, cj.p0, cj.p3].iter().any(|e| e.dist(p) < tol) {
                        continue;
                    }
                    out[i].push(s);
                    out[j].push(t);
                }
            }
        }
        out
    }

    /// Split each segment at its parameters. Split points within `tol` of an
    /// existing vertex land on that vertex, so both curves of a crossing share
    /// one.
    fn apply_splits(&mut self, splits: Vec<Vec<f64>>, tol: f64, rep: &mut PlanarizeReport) {
        if splits.iter().all(Vec::is_empty) {
            return;
        }
        let mut grid = VertexGrid::new(self, tol);
        for (seg, mut ts) in splits.into_iter().enumerate() {
            if ts.is_empty() {
                continue;
            }
            ts.retain(|t| t.is_finite() && *t > 0.0 && *t < 1.0);
            ts.sort_by(f64::total_cmp);
            let c = self.segment_cubic(seg);
            let (start, end) = (self.segments[seg].start, self.segments[seg].end);
            let mut pts: Vec<(f64, usize)> = Vec::new();
            for t in ts {
                let p = c.eval(t);
                if p.dist(c.p0) < tol || p.dist(c.p3) < tol {
                    continue;
                }
                if pts.last().is_some_and(|&(lt, _)| t - lt < 1e-9) {
                    continue;
                }
                let mut added = 0;
                let v = grid.find_or_add(self, p, &mut added);
                rep.vertices_added += added;
                if v == start || v == end {
                    continue;
                }
                // The same vertex twice is a self-crossing (the piece between
                // is the loop's lobe, split later as a self-loop) unless the
                // two parameters are just one point found twice.
                if let Some(&(lt, lv)) = pts.last() {
                    if lv == v && c.subsegment(lt, t).hull_bounds().diag() < 4.0 * tol {
                        continue;
                    }
                }
                pts.push((t, v));
            }
            if pts.is_empty() {
                continue;
            }
            rep.splits += pts.len();
            self.split_segment_multi(seg, &pts);
        }
    }
}
