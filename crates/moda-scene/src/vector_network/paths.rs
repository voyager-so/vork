//! Conversion between a network and Moda's `PathNode` point lists
//! (`BlendPathPoint`, the TS `PathPoint`).
//!
//! Network → paths gives one compound path per region (each loop a subpath;
//! the node is `closed: true`, which Moda applies to every subpath) plus the
//! segments no region uses, chained into runs for stroking. This is the shape
//! Figma paste already produces from a decoded network
//! (`lib/figmaClipboard/convert/vector.ts`), so the renderer can draw a network
//! before it understands one natively.
//!
//! Path → network makes a vertex per anchor and a segment per command; a
//! closed subpath closes back onto its first vertex and becomes a loop of one
//! region under the path's fill rule. Coincident anchors of different subpaths
//! stay separate vertices until [`VectorNetwork::planarize`] merges them.

use crate::blend::BlendPathPoint as PathPoint;
use crate::path_bool::FillRule;
use crate::path_geom::arc::{arc_to_beziers, ArcParams};

use super::geom::Vector;
use super::{OrientedLoop, VectorNetwork, VectorRegion, WindingRule};

/// One region as a compound path. Draw with `closed: true`.
#[derive(Debug, Clone)]
pub struct RegionPath {
    /// Index into `VectorNetwork::regions`.
    pub region: usize,
    pub points: Vec<PathPoint>,
    pub fill_rule: FillRule,
}

/// A chain of segments, for stroking.
#[derive(Debug, Clone)]
pub struct PathRun {
    pub points: Vec<PathPoint>,
    /// The run returns to its first vertex; draw it `closed: true`.
    pub closed: bool,
}

#[derive(Debug, Clone)]
pub struct NetworkPaths {
    pub regions: Vec<RegionPath>,
    /// Segments not in any region loop.
    pub open_runs: Vec<PathRun>,
    /// Region loops whose indices do not chain (dropped).
    pub skipped_loops: usize,
}

impl From<WindingRule> for FillRule {
    fn from(w: WindingRule) -> FillRule {
        match w {
            WindingRule::NonZero => FillRule::NonZero,
            WindingRule::EvenOdd => FillRule::EvenOdd,
        }
    }
}

impl From<FillRule> for WindingRule {
    fn from(f: FillRule) -> WindingRule {
        match f {
            FillRule::NonZero => WindingRule::NonZero,
            FillRule::EvenOdd => WindingRule::EvenOdd,
        }
    }
}

impl VectorNetwork {
    /// Regions as compound paths, plus the region-less segments as runs.
    pub fn to_paths(&self) -> NetworkPaths {
        let mut covered = vec![false; self.segments.len()];
        let mut regions = Vec::new();
        let mut skipped = 0;
        for (ri, region) in self.regions.iter().enumerate() {
            let mut points = Vec::new();
            for lp in &region.loops {
                let Some((orient, closed)) = self.chain_loop(lp) else {
                    skipped += 1;
                    continue;
                };
                for &s in lp {
                    covered[s] = true;
                }
                self.emit_run(&orient, closed, &mut points);
            }
            if !points.is_empty() {
                regions.push(RegionPath {
                    region: ri,
                    points,
                    fill_rule: region.winding_rule.into(),
                });
            }
        }
        NetworkPaths {
            regions,
            open_runs: self.chain_runs(&covered),
            skipped_loops: skipped,
        }
    }

    /// Every segment exactly once, chained into runs: the network's stroke
    /// (Figma strokes every segment, filled or not).
    pub fn stroke_runs(&self) -> Vec<PathRun> {
        self.chain_runs(&vec![false; self.segments.len()])
    }

    /// Append a loop or run as `move` + one command per segment. A closed run
    /// whose last segment is straight leaves that segment to the implicit
    /// close, the way a hand-drawn closed path is stored.
    fn emit_run(&self, orient: &[(usize, bool)], closed: bool, out: &mut Vec<PathPoint>) {
        for (k, &(s, rev)) in orient.iter().enumerate() {
            let c = self.oriented_cubic(s, rev);
            if k == 0 {
                out.push(PathPoint::Move {
                    x: c.p0.x,
                    y: c.p0.y,
                });
            }
            let straight = self.segments[s].is_straight();
            if closed && k + 1 == orient.len() && straight && k > 0 {
                break;
            }
            if straight {
                out.push(PathPoint::Line {
                    x: c.p3.x,
                    y: c.p3.y,
                });
            } else {
                out.push(PathPoint::Curve {
                    x: c.p3.x,
                    y: c.p3.y,
                    cp1: (c.p1.x, c.p1.y),
                    cp2: (c.p2.x, c.p2.y),
                });
            }
        }
    }

    /// Chain the uncovered segments into runs, endpoint-first and greedily
    /// (the port of `chainOpenRuns` in the fig decoder). Each segment is
    /// emitted once; runs start at degree-1 vertices so a polyline comes out
    /// whole, and what remains are cycles.
    fn chain_runs(&self, covered: &[bool]) -> Vec<PathRun> {
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); self.vertices.len()];
        for (i, s) in self.segments.iter().enumerate() {
            if covered[i] {
                continue;
            }
            incident[s.start].push(i);
            incident[s.end].push(i);
        }
        let mut used = vec![false; self.segments.len()];
        let mut runs = Vec::new();
        let walk = |from: usize, used: &mut Vec<bool>| -> OrientedLoop {
            let mut out = Vec::new();
            let mut cursor = from;
            while let Some(&next) = incident[cursor].iter().find(|&&i| !used[i]) {
                used[next] = true;
                let s = &self.segments[next];
                let rev = s.start != cursor;
                cursor = if rev { s.start } else { s.end };
                out.push((next, rev));
            }
            out
        };
        let emit = |orient: OrientedLoop, runs: &mut Vec<PathRun>| {
            if orient.is_empty() {
                return;
            }
            let first = orient[0];
            let last = orient[orient.len() - 1];
            let s0 = &self.segments[first.0];
            let sl = &self.segments[last.0];
            let start_v = if first.1 { s0.end } else { s0.start };
            let end_v = if last.1 { sl.start } else { sl.end };
            let closed = start_v == end_v;
            let mut points = Vec::new();
            self.emit_run(&orient, closed, &mut points);
            runs.push(PathRun { points, closed });
        };
        for (v, inc) in incident.iter().enumerate() {
            if inc.len() == 1 {
                let o = walk(v, &mut used);
                emit(o, &mut runs);
            }
        }
        for i in 0..self.segments.len() {
            if covered[i] || used[i] {
                continue;
            }
            let o = walk(self.segments[i].start, &mut used);
            emit(o, &mut runs);
        }
        runs
    }

    /// Build a network from `PathNode` points. With `closed` (Moda closes
    /// every subpath of a closed path) each subpath gets a closing segment
    /// unless its last anchor already sits on its first (within `tol`), and
    /// the closed subpaths form one region under `fill_rule`. An open path's
    /// subpath that ends on its own start is closed and filled too.
    /// Quadratics are degree-elevated and arcs converted to cubics.
    pub fn from_path(
        points: &[PathPoint],
        closed: bool,
        fill_rule: FillRule,
        tol: f64,
    ) -> VectorNetwork {
        let tol = if tol.is_finite() && tol > 0.0 {
            tol
        } else {
            super::DEFAULT_TOLERANCE
        };
        let mut net = VectorNetwork::new();
        let mut loops: Vec<Vec<usize>> = Vec::new();

        // Subpath: first anchor + pieces (end anchor, optional control points).
        type Piece = (Vector, Option<(Vector, Vector)>);
        let mut subpaths: Vec<(Vector, Vec<Piece>)> = Vec::new();
        let mut cursor: Option<Vector> = None;
        for p in points {
            let (x, y) = match p {
                PathPoint::Move { x, y }
                | PathPoint::Line { x, y }
                | PathPoint::Curve { x, y, .. }
                | PathPoint::Quadratic { x, y, .. }
                | PathPoint::Arc { x, y, .. } => (*x, *y),
            };
            let end = Vector::new(x, y);
            if !end.is_finite() {
                continue;
            }
            let Some(cur) = cursor.filter(|_| !matches!(p, PathPoint::Move { .. })) else {
                subpaths.push((end, Vec::new()));
                cursor = Some(end);
                continue;
            };
            let pieces = &mut subpaths.last_mut().expect("subpath open").1;
            match p {
                PathPoint::Move { .. } => unreachable!(),
                PathPoint::Line { .. } => pieces.push((end, None)),
                PathPoint::Curve { cp1, cp2, .. } => pieces.push((
                    end,
                    Some((Vector::new(cp1.0, cp1.1), Vector::new(cp2.0, cp2.1))),
                )),
                PathPoint::Quadratic { cp, .. } => {
                    let q = Vector::new(cp.0, cp.1);
                    pieces.push((
                        end,
                        Some((cur + (q - cur) * (2.0 / 3.0), end + (q - end) * (2.0 / 3.0))),
                    ));
                }
                PathPoint::Arc {
                    rx,
                    ry,
                    x_axis_rotation,
                    large_arc,
                    sweep,
                    ..
                } => {
                    let bz = arc_to_beziers(
                        cur.x,
                        cur.y,
                        &ArcParams {
                            rx: *rx,
                            ry: *ry,
                            x_axis_rotation: *x_axis_rotation,
                            large_arc: *large_arc,
                            sweep: *sweep,
                            x,
                            y,
                        },
                    );
                    if bz.is_empty() {
                        pieces.push((end, None));
                    } else {
                        for b in bz {
                            pieces.push((
                                Vector::new(b.end[0], b.end[1]),
                                Some((
                                    Vector::new(b.cp1[0], b.cp1[1]),
                                    Vector::new(b.cp2[0], b.cp2[1]),
                                )),
                            ));
                        }
                    }
                }
            }
            cursor = Some(end);
        }

        for (first, pieces) in subpaths {
            let v0 = net.add_vertex(first);
            let n = pieces.len();
            let returns = n > 0 && pieces[n - 1].0.dist(first) < tol;
            let is_closed = n > 0 && (closed || returns);
            let mut seg_ids = Vec::new();
            let mut cur_v = v0;
            for (k, (end, ctrl)) in pieces.into_iter().enumerate() {
                let from = net.vertex_pos(cur_v);
                let last = k + 1 == n;
                if ctrl.is_none() && end.dist(from) < tol && !(last && returns) {
                    // A zero-length line adds nothing.
                    continue;
                }
                let to_v = if last && returns {
                    v0
                } else {
                    net.add_vertex(end)
                };
                let to = net.vertex_pos(to_v);
                let (ts, te) = match ctrl {
                    None => (Vector::ZERO, Vector::ZERO),
                    Some((c1, c2)) => (c1 - from, c2 - to),
                };
                if to_v == cur_v && ts.is_zero() && te.is_zero() {
                    continue;
                }
                seg_ids.push(net.add_segment(cur_v, to_v, ts, te).expect("valid"));
                cur_v = to_v;
            }
            if is_closed && !seg_ids.is_empty() {
                if cur_v != v0 {
                    seg_ids.push(net.add_line(cur_v, v0).expect("valid"));
                }
                loops.push(seg_ids);
            }
        }
        if !loops.is_empty() {
            net.regions.push(VectorRegion::new(fill_rule.into(), loops));
        }
        net
    }
}
