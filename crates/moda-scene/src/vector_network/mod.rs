//! Vector networks (roadmap E7.9): Figma's generalisation of a path, where an
//! anchor may join any number of segments and any enclosed area can be filled.
//!
//! A `path` node's optional `vectorNetwork` field holds one (authoritative
//! when present; the node's `points` / `closed` / `fillRule` are derived from
//! it by the editor). The engine draws it in `moda_engine::tess::network`,
//! and [`api::vector_network_json`] is the JSON entry the editor reaches
//! through `moda-wasm`.
//!
//! # Data model
//!
//! Matches Figma's plugin API `VectorNetwork` field for field, and serialises to
//! the same JSON:
//!
//! ```jsonc
//! {
//!   "vertices": [{ "x": 0, "y": 0, "strokeCap": "ROUND", "strokeJoin": "MITER",
//!                  "cornerRadius": 4, "handleMirroring": "ANGLE" }],
//!   "segments": [{ "start": 0, "end": 1,
//!                  "tangentStart": { "x": 0, "y": 0 }, "tangentEnd": { "x": 0, "y": 0 } }],
//!   "regions":  [{ "windingRule": "NONZERO", "loops": [[0, 1, 2]],
//!                  "fills": [/* Paint */], "fillStyleId": "S:…" }]
//! }
//! ```
//!
//! * A segment's tangents are **relative to their own vertex**: the cubic is
//!   `start, start + tangentStart, end + tangentEnd, end`. Both zero means a
//!   straight segment.
//! * A region loop lists segment indices in traversal order but not their
//!   direction; the direction of each is fixed by chaining (the first segment
//!   is tried both ways). A loop need not return to its start: Figma closes a
//!   region loop implicitly when filling it, as SVG's `Z` does.
//! * `fills` is Figma's `Paint[]`, carried as opaque JSON so a round-trip loses
//!   nothing; mapping it to Moda's paint stack is the model integration's job.
//!
//! # Geometry
//!
//! * [`VectorNetwork::planarize`] snaps coincident vertices, drops zero-length
//!   segments, splits every segment where it crosses or touches another (or
//!   itself), and merges coincident segments, so the result is a planar graph.
//! * [`VectorNetwork::faces`] walks the planar graph's half-edges (a DCEL face
//!   walk on cubic segments) and returns every bounded face with its holes.
//! * [`VectorNetwork::toggle_fill_at`] is the "click an enclosed area" action.
//! * [`VectorNetwork::to_paths`] / [`VectorNetwork::from_path`] convert to and
//!   from Moda's `PathNode` point lists.
//!
//! All tolerances are absolute, in network units ([`DEFAULT_TOLERANCE`]).

pub mod api;
mod faces;
pub mod geom;
mod intersect;
mod paths;
mod planar;
mod round;

#[cfg(test)]
mod prop_tests;
#[cfg(test)]
mod tests;

use std::fmt;

use serde::{Deserialize, Serialize};

pub use api::{derived_points, network_from_node, vector_network_json};
pub use faces::{Face, FillToggle};
pub use geom::{Cubic, Rect, Vector};
pub use paths::{NetworkPaths, PathRun, RegionPath};
pub use planar::PlanarizeReport;

/// Default snapping tolerance, in network units (px for a Moda path).
pub const DEFAULT_TOLERANCE: f64 = 1e-3;

/// Figma's `StrokeCap`. Unknown future values survive a round-trip as `Other`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrokeCap {
    #[serde(rename = "NONE")]
    None,
    #[serde(rename = "ROUND")]
    Round,
    #[serde(rename = "SQUARE")]
    Square,
    #[serde(rename = "ARROW_LINES")]
    ArrowLines,
    #[serde(rename = "ARROW_EQUILATERAL")]
    ArrowEquilateral,
    #[serde(rename = "DIAMOND_FILLED")]
    DiamondFilled,
    #[serde(rename = "TRIANGLE_FILLED")]
    TriangleFilled,
    #[serde(rename = "CIRCLE_FILLED")]
    CircleFilled,
    #[serde(untagged)]
    Other(String),
}

/// Figma's `StrokeJoin`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrokeJoin {
    #[serde(rename = "MITER")]
    Miter,
    #[serde(rename = "BEVEL")]
    Bevel,
    #[serde(rename = "ROUND")]
    Round,
    #[serde(untagged)]
    Other(String),
}

/// Figma's `HandleMirroring` (how the pen tool couples a vertex's handles).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandleMirroring {
    #[serde(rename = "NONE")]
    None,
    #[serde(rename = "ANGLE")]
    Angle,
    #[serde(rename = "ANGLE_AND_LENGTH")]
    AngleAndLength,
    #[serde(untagged)]
    Other(String),
}

/// Figma's `WindingRule` for a region.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindingRule {
    #[default]
    #[serde(rename = "NONZERO")]
    NonZero,
    #[serde(rename = "EVENODD")]
    EvenOdd,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorVertex {
    pub x: f64,
    pub y: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_cap: Option<StrokeCap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_join: Option<StrokeJoin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corner_radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handle_mirroring: Option<HandleMirroring>,
}

impl VectorVertex {
    pub fn at(x: f64, y: f64) -> VectorVertex {
        VectorVertex {
            x,
            y,
            ..VectorVertex::default()
        }
    }
    pub fn pos(&self) -> Vector {
        Vector::new(self.x, self.y)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSegment {
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub tangent_start: Vector,
    #[serde(default)]
    pub tangent_end: Vector,
}

impl VectorSegment {
    pub fn line(start: usize, end: usize) -> VectorSegment {
        VectorSegment {
            start,
            end,
            ..VectorSegment::default()
        }
    }
    /// Both tangents zero: a straight segment.
    pub fn is_straight(&self) -> bool {
        self.tangent_start.is_zero() && self.tangent_end.is_zero()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorRegion {
    #[serde(default)]
    pub winding_rule: WindingRule,
    pub loops: Vec<Vec<usize>>,
    /// Figma `Paint[]` override for this region, carried opaquely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fills: Option<Vec<crate::value::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_style_id: Option<String>,
}

impl VectorRegion {
    pub fn new(winding_rule: WindingRule, loops: Vec<Vec<usize>>) -> VectorRegion {
        VectorRegion {
            winding_rule,
            loops,
            fills: None,
            fill_style_id: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VectorNetwork {
    pub vertices: Vec<VectorVertex>,
    pub segments: Vec<VectorSegment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<VectorRegion>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NetworkError {
    Json(String),
    NonFinite,
    VertexIndex(usize),
    SegmentIndex(usize),
    /// A segment references a vertex that does not exist.
    DanglingSegment {
        segment: usize,
        vertex: usize,
    },
    /// A region loop references a segment that does not exist.
    DanglingLoop {
        region: usize,
        segment: usize,
    },
    /// A split parameter outside the open interval (0, 1).
    Parameter(f64),
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::Json(m) => write!(f, "invalid vector network JSON: {m}"),
            NetworkError::NonFinite => write!(f, "non-finite coordinate or tangent"),
            NetworkError::VertexIndex(i) => write!(f, "no vertex {i}"),
            NetworkError::SegmentIndex(i) => write!(f, "no segment {i}"),
            NetworkError::DanglingSegment { segment, vertex } => {
                write!(f, "segment {segment} references missing vertex {vertex}")
            }
            NetworkError::DanglingLoop { region, segment } => {
                write!(f, "region {region} references missing segment {segment}")
            }
            NetworkError::Parameter(t) => write!(f, "parameter {t} is not in (0, 1)"),
        }
    }
}

impl std::error::Error for NetworkError {}

/// A region loop's segments oriented for traversal: `(segment, reversed)`.
pub type OrientedLoop = Vec<(usize, bool)>;

impl VectorNetwork {
    pub fn new() -> VectorNetwork {
        VectorNetwork::default()
    }

    /// Parse Figma plugin-API JSON and validate it.
    pub fn from_json(s: &str) -> Result<VectorNetwork, NetworkError> {
        let net: VectorNetwork =
            crate::value::from_str(s).map_err(|e| NetworkError::Json(e.to_string()))?;
        net.validate()?;
        Ok(net)
    }

    pub fn from_json_value(v: &crate::value::Value) -> Result<VectorNetwork, NetworkError> {
        let net: VectorNetwork =
            crate::value::from_value(v.clone()).map_err(|e| NetworkError::Json(e.to_string()))?;
        net.validate()?;
        Ok(net)
    }

    pub fn to_json(&self) -> String {
        crate::value::to_string(self).expect("vector network serialises")
    }

    pub fn to_json_value(&self) -> crate::value::Value {
        crate::value::to_value(self).expect("vector network serialises")
    }

    /// Every index in range and every number finite.
    pub fn validate(&self) -> Result<(), NetworkError> {
        for v in &self.vertices {
            if !v.x.is_finite() || !v.y.is_finite() {
                return Err(NetworkError::NonFinite);
            }
            if let Some(r) = v.corner_radius {
                if !r.is_finite() {
                    return Err(NetworkError::NonFinite);
                }
            }
        }
        for (i, s) in self.segments.iter().enumerate() {
            for vertex in [s.start, s.end] {
                if vertex >= self.vertices.len() {
                    return Err(NetworkError::DanglingSegment { segment: i, vertex });
                }
            }
            if !s.tangent_start.is_finite() || !s.tangent_end.is_finite() {
                return Err(NetworkError::NonFinite);
            }
        }
        for (r, region) in self.regions.iter().enumerate() {
            for lp in &region.loops {
                for &segment in lp {
                    if segment >= self.segments.len() {
                        return Err(NetworkError::DanglingLoop { region: r, segment });
                    }
                }
            }
        }
        Ok(())
    }

    pub fn vertex_pos(&self, i: usize) -> Vector {
        self.vertices[i].pos()
    }

    /// Segment `i` as a cubic from its start vertex to its end vertex.
    pub fn segment_cubic(&self, i: usize) -> Cubic {
        let s = &self.segments[i];
        let a = self.vertex_pos(s.start);
        let b = self.vertex_pos(s.end);
        if s.is_straight() {
            Cubic::line(a, b)
        } else {
            Cubic::new(a, a + s.tangent_start, b + s.tangent_end, b)
        }
    }

    /// Segment `i` as traversed: reversed runs end → start.
    pub fn oriented_cubic(&self, i: usize, reversed: bool) -> Cubic {
        let c = self.segment_cubic(i);
        if reversed {
            c.reversed()
        } else {
            c
        }
    }

    /// Bounds of every vertex and control point.
    pub fn bounds(&self) -> Option<Rect> {
        let mut r: Option<Rect> = None;
        for v in &self.vertices {
            let p = Rect::from_points(&[v.pos()]);
            r = Some(r.map_or(p, |q| q.union(p)));
        }
        for i in 0..self.segments.len() {
            let b = self.segment_cubic(i).hull_bounds();
            r = Some(r.map_or(b, |q| q.union(b)));
        }
        r
    }

    /// Number of segments incident to vertex `v` (a self-loop counts twice).
    pub fn degree(&self, v: usize) -> usize {
        self.segments
            .iter()
            .map(|s| usize::from(s.start == v) + usize::from(s.end == v))
            .sum()
    }

    // ---------------------------------------------------------------------
    // Editing
    // ---------------------------------------------------------------------

    pub fn add_vertex(&mut self, p: Vector) -> usize {
        self.vertices.push(VectorVertex::at(p.x, p.y));
        self.vertices.len() - 1
    }

    /// Add a segment; tangents are relative to their own vertex, as in Figma.
    pub fn add_segment(
        &mut self,
        start: usize,
        end: usize,
        tangent_start: Vector,
        tangent_end: Vector,
    ) -> Result<usize, NetworkError> {
        for v in [start, end] {
            if v >= self.vertices.len() {
                return Err(NetworkError::VertexIndex(v));
            }
        }
        if !tangent_start.is_finite() || !tangent_end.is_finite() {
            return Err(NetworkError::NonFinite);
        }
        self.segments.push(VectorSegment {
            start,
            end,
            tangent_start,
            tangent_end,
        });
        Ok(self.segments.len() - 1)
    }

    pub fn add_line(&mut self, start: usize, end: usize) -> Result<usize, NetworkError> {
        self.add_segment(start, end, Vector::ZERO, Vector::ZERO)
    }

    /// Delete segment `i`. A region loop through it no longer bounds anything,
    /// so the loop is dropped (and the region, if that was its last loop), the
    /// way deleting an edge in Figma unfills the areas it closed.
    pub fn remove_segment(&mut self, i: usize) -> Result<(), NetworkError> {
        if i >= self.segments.len() {
            return Err(NetworkError::SegmentIndex(i));
        }
        let mut mask = vec![false; self.segments.len()];
        mask[i] = true;
        self.remove_segments_masked(&mask, true);
        Ok(())
    }

    /// Delete vertex `v` and every segment incident to it.
    pub fn remove_vertex(&mut self, v: usize) -> Result<(), NetworkError> {
        if v >= self.vertices.len() {
            return Err(NetworkError::VertexIndex(v));
        }
        let mask: Vec<bool> = self
            .segments
            .iter()
            .map(|s| s.start == v || s.end == v)
            .collect();
        self.remove_segments_masked(&mask, true);
        let mut vmask = vec![false; self.vertices.len()];
        vmask[v] = true;
        self.remove_vertices_masked(&vmask);
        Ok(())
    }

    /// Split segment `i` at parameter `t ∈ (0, 1)`, inserting a vertex there.
    /// The two pieces trace exactly the original curve; region loops through
    /// the segment get both pieces, in traversal order. Returns the new vertex.
    pub fn split_segment(&mut self, i: usize, t: f64) -> Result<usize, NetworkError> {
        if i >= self.segments.len() {
            return Err(NetworkError::SegmentIndex(i));
        }
        if !(t > 0.0 && t < 1.0) {
            return Err(NetworkError::Parameter(t));
        }
        let p = self.segment_cubic(i).eval(t);
        let v = self.add_vertex(p);
        self.split_segment_multi(i, &[(t, v)]);
        Ok(v)
    }

    /// Merge vertex `remove` into `keep`: every segment on `remove` now ends on
    /// `keep` (its tangents, being relative, move with it), segments that
    /// collapse to a zero-length straight edge are dropped, and `remove` is
    /// deleted. Returns `keep`'s index after the deletion.
    pub fn merge_vertices(&mut self, keep: usize, remove: usize) -> Result<usize, NetworkError> {
        for v in [keep, remove] {
            if v >= self.vertices.len() {
                return Err(NetworkError::VertexIndex(v));
            }
        }
        if keep == remove {
            return Ok(keep);
        }
        let mut map: Vec<usize> = (0..self.vertices.len()).collect();
        map[remove] = keep;
        self.remap_vertices(&map);
        Ok(if keep > remove { keep - 1 } else { keep })
    }

    // ---------------------------------------------------------------------
    // Internal bookkeeping
    // ---------------------------------------------------------------------

    /// Point every segment at `map[v]`, drop zero-length straight segments that
    /// creates, then delete vertices no longer mapped to themselves. Returns
    /// the number of segments dropped.
    pub(crate) fn remap_vertices(&mut self, map: &[usize]) -> usize {
        for s in &mut self.segments {
            s.start = map[s.start];
            s.end = map[s.end];
        }
        let collapsed: Vec<bool> = self
            .segments
            .iter()
            .map(|s| s.start == s.end && s.is_straight())
            .collect();
        let dropped = collapsed.iter().filter(|&&c| c).count();
        if dropped > 0 {
            self.remove_segments_masked(&collapsed, false);
        }
        let vmask: Vec<bool> = (0..self.vertices.len()).map(|v| map[v] != v).collect();
        self.remove_vertices_masked(&vmask);
        dropped
    }

    /// Delete the masked segments and compact indices. With `drop_loops`, a
    /// loop that referenced a deleted segment is dropped; otherwise the index
    /// is just removed from the loop (right for zero-length and duplicate
    /// segments, whose removal keeps the chain intact).
    pub(crate) fn remove_segments_masked(&mut self, mask: &[bool], drop_loops: bool) {
        let mut new_index = vec![usize::MAX; self.segments.len()];
        let mut next = 0;
        for (i, &m) in mask.iter().enumerate() {
            if !m {
                new_index[i] = next;
                next += 1;
            }
        }
        let mut i = 0;
        self.segments.retain(|_| {
            let keep = !mask[i];
            i += 1;
            keep
        });
        for region in &mut self.regions {
            region.loops.retain_mut(|lp| {
                if drop_loops && lp.iter().any(|&s| mask[s]) {
                    return false;
                }
                lp.retain(|&s| !mask[s]);
                for s in lp.iter_mut() {
                    *s = new_index[*s];
                }
                !lp.is_empty()
            });
        }
        self.regions.retain(|r| !r.loops.is_empty());
    }

    /// Delete the masked vertices (which must have no segments) and compact.
    pub(crate) fn remove_vertices_masked(&mut self, mask: &[bool]) {
        if !mask.iter().any(|&m| m) {
            return;
        }
        let mut new_index = vec![usize::MAX; self.vertices.len()];
        let mut next = 0;
        for (i, &m) in mask.iter().enumerate() {
            if !m {
                new_index[i] = next;
                next += 1;
            }
        }
        let mut i = 0;
        self.vertices.retain(|_| {
            let keep = !mask[i];
            i += 1;
            keep
        });
        for s in &mut self.segments {
            debug_assert!(new_index[s.start] != usize::MAX && new_index[s.end] != usize::MAX);
            s.start = new_index[s.start];
            s.end = new_index[s.end];
        }
    }

    /// Chain a loop's segments, fixing each one's direction. The first segment
    /// is tried both ways; a chain that returns to its start wins, otherwise
    /// the first chain that connects. `None` if the indices do not chain.
    /// Returns the orientation and whether the chain closes.
    pub fn chain_loop(&self, lp: &[usize]) -> Option<(OrientedLoop, bool)> {
        if lp.is_empty() {
            return None;
        }
        let mut fallback = None;
        for first_reversed in [false, true] {
            let mut out: OrientedLoop = Vec::with_capacity(lp.len());
            let mut cursor = usize::MAX;
            let mut origin = usize::MAX;
            let mut ok = true;
            for (k, &si) in lp.iter().enumerate() {
                let s = self.segments.get(si)?;
                let reversed = if k == 0 {
                    origin = if first_reversed { s.end } else { s.start };
                    first_reversed
                } else if s.start == cursor {
                    false
                } else if s.end == cursor {
                    true
                } else {
                    ok = false;
                    break;
                };
                cursor = if reversed { s.start } else { s.end };
                out.push((si, reversed));
            }
            if !ok {
                continue;
            }
            let closed = cursor == origin;
            if closed {
                return Some((out, true));
            }
            if fallback.is_none() {
                fallback = Some((out, false));
            }
        }
        fallback
    }

    /// Write a cubic piece into segment slot `idx`, from vertex `start` to
    /// vertex `end`; control points are kept absolute.
    pub(crate) fn set_segment_from_cubic(
        &mut self,
        idx: usize,
        start: usize,
        end: usize,
        c: &Cubic,
        straight: bool,
    ) {
        let (ts, te) = if straight {
            (Vector::ZERO, Vector::ZERO)
        } else {
            (c.p1 - self.vertex_pos(start), c.p2 - self.vertex_pos(end))
        };
        let seg = VectorSegment {
            start,
            end,
            tangent_start: ts,
            tangent_end: te,
        };
        if idx == self.segments.len() {
            self.segments.push(seg);
        } else {
            self.segments[idx] = seg;
        }
    }

    /// Split segment `seg` at the sorted parameters `pts`, each landing on the
    /// given (existing) vertex. The first piece keeps index `seg`; the rest are
    /// appended. Region loops through `seg` get all pieces in traversal order.
    /// Returns the piece indices from the segment's start to its end.
    pub(crate) fn split_segment_multi(&mut self, seg: usize, pts: &[(f64, usize)]) -> Vec<usize> {
        if pts.is_empty() {
            return vec![seg];
        }
        // Orientation of every loop occurrence, before the geometry changes.
        let mut occurrences: Vec<(usize, usize, usize, bool)> = Vec::new();
        for (ri, region) in self.regions.iter().enumerate() {
            for (li, lp) in region.loops.iter().enumerate() {
                if !lp.contains(&seg) {
                    continue;
                }
                let orient = self.chain_loop(lp).map(|(o, _)| o);
                for (pos, &s) in lp.iter().enumerate() {
                    if s == seg {
                        let rev = orient.as_ref().is_some_and(|o| o[pos].1);
                        occurrences.push((ri, li, pos, rev));
                    }
                }
            }
        }

        let orig = self.segments[seg].clone();
        let straight = orig.is_straight();
        let full = self.segment_cubic(seg);
        let mut rest = full;
        let mut prev_t = 0.0;
        let mut from = orig.start;
        let mut pieces = Vec::with_capacity(pts.len() + 1);
        for (k, &(t, v)) in pts.iter().enumerate() {
            let local = if prev_t >= 1.0 {
                0.0
            } else {
                ((t - prev_t) / (1.0 - prev_t)).clamp(0.0, 1.0)
            };
            let (left, right) = rest.split(local);
            let idx = if k == 0 { seg } else { self.segments.len() };
            self.set_segment_from_cubic(idx, from, v, &left, straight);
            pieces.push(idx);
            rest = right;
            prev_t = t;
            from = v;
        }
        let idx = self.segments.len();
        self.set_segment_from_cubic(idx, from, orig.end, &rest, straight);
        pieces.push(idx);

        // Later positions first, so earlier ones stay valid.
        occurrences.sort_by_key(|o| std::cmp::Reverse(o.2));
        for (ri, li, pos, rev) in occurrences {
            let lp = &mut self.regions[ri].loops[li];
            let mut ins = pieces.clone();
            if rev {
                ins.reverse();
            }
            lp.splice(pos..=pos, ins);
        }
        pieces
    }

    /// Replace every loop reference to segment `from` with `to`.
    pub(crate) fn replace_segment_refs(&mut self, from: usize, to: usize) {
        for region in &mut self.regions {
            for lp in &mut region.loops {
                for s in lp.iter_mut() {
                    if *s == from {
                        *s = to;
                    }
                }
            }
        }
    }
}
