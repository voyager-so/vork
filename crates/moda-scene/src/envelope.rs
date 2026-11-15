//! Envelope distort (E8.17, Illustrator's Object ▸ Envelope Distort): a group whose `envelope` field bends
//! everything inside it (paths, shapes, text, images, frames, masks, booleans, other envelopes, and art with live
//! effects of its own) through one smooth map of the content's bounds:
//!
//! - `{"type": "warp", "style", "bend", "horizontal"?, "vertical"?, "orientation"?}`: Make with Warp, the Warp
//!   effect's 15 presets over the whole group (`crate::live_effects::warp_map`; Wave ripples the box's edges too);
//! - `{"type": "mesh", "rows", "columns", "points"?}`: Make with Mesh, a rows × columns grid of points, each with
//!   up to four Bézier handles; each cell is the Coons patch of its four edge curves;
//! - `{"type": "top-object", "points", "closed"?}`: Make with Top Object, the content's bounds mapped onto a
//!   closed path (the Coons patch of the path split into four sides at the corners nearest the path's bounds).
//!
//! Mesh points and top-object points are stored in the content box's own units (0..1 across and down), so the
//! envelope follows its content when the group is moved, scaled or its contents edited.
//!
//! **How it draws.** At scene resolution (after live effects and booleans, innermost envelope first) the group's
//! content resolves to TRANSIENT nodes that the group draws in place of its children ([`Scene::visit_z`]): each
//! vector leaf becomes a `path` whose outline is the warped fill and the warped outline of each stroke (so strokes
//! bend with the art, as Illustrator's "Distort Appearance"); text becomes its warped glyph outlines; an image fill
//! becomes a quadtree of small rectangles, each a projective (corner-pin `distort`) patch of the texture, fine
//! enough that no texel is more than [`IMAGE_TOLERANCE_PX`] from where the smooth map puts it (mesh texture
//! mapping); groups, frames and their clip masks become transient groups whose mask is the warped mask. Every
//! lane (canvas, PNG export, masks, shadows, isolation) draws the transients like any other node.
//!
//! **Exactness.** Curves are mapped exactly to a flatness tolerance: each source segment is split adaptively until
//! a cubic Hermite fit of the mapped curve is within [`envelope_tolerance`] of it at every check point. The
//! tolerance follows the device scale: [`tolerance_for_scale`] is a fifth of a device pixel, and the scene uses
//! the top of the editor's zoom range ([`MAX_PX_PER_UNIT`]), so the result is within a fifth of a device pixel at
//! every zoom and export scale up to that; the tessellator then flattens the cubics per zoom band as it does every
//! curve. A document without envelopes never enters this module.

use std::collections::HashMap;

use crate::value::{json, Map, Value};

use crate::blend::BlendPathPoint;
use crate::effect::WarpEffect;
use crate::live_effects::{
    bounds_of, contours_from_points, outline_stroke, points_from_contours, split_copies, warp_map, Bounds,
    Contour, Seg, Shape, EFFECT_COPIES_KEY, EFFECT_COPY_FILLS_KEY, EFFECT_COPY_ROLES_KEY,
};
use crate::path_geom::vec2::{v2, Cubic, V2};
use crate::scene::Scene;
use crate::shape_outline::node_outline;
use crate::text_warp::TextOutlines;
use crate::transform;
use crate::types::{Affine, NodeId, RawNode};

/// The group field that makes a group an envelope.
pub const ENVELOPE_KEY: &str = "envelope";
/// On every transient an envelope makes: the envelope's id.
pub const ENVELOPE_OF_KEY: &str = "_envelopeOf";
/// On every transient an envelope makes: the id of the node it was made from.
pub const ENVELOPE_SOURCE_KEY: &str = "_envelopeSource";
/// On an image cell: its quad in the envelope's space (8 numbers), so an outer envelope can bend it again.
pub const ENVELOPE_CELL_KEY: &str = "_envelopeCell";

/// A fifth of a device pixel, the tolerance [`tolerance_for_scale`] keeps at every scale.
pub const DEVICE_TOLERANCE_PX: f64 = 0.2;
/// The top of the editor's zoom range on a 2× display (25,600 %): 512 device px per canvas unit.
pub const MAX_PX_PER_UNIT: f64 = 512.0;
/// How far a texel of a warped image may land from where the smooth map puts it, canvas units. Images are
/// sampled, not flattened: a tenth of a unit is invisible at 100 % and stays under a device pixel to 1,000 %.
pub const IMAGE_TOLERANCE_PX: f64 = 0.1;
/// Most segments one envelope's warped outlines may hold (the adaptive split stops refining past it).
pub const MAX_ENVELOPE_SEGMENTS: usize = 400_000;
/// Most image cells one envelope may draw.
pub const MAX_IMAGE_CELLS: usize = 16_384;
/// Deepest adaptive split of one source segment.
const MAX_SPLIT_DEPTH: u32 = 16;
/// Deepest image quadtree.
const MAX_CELL_DEPTH: u32 = 9;

/// The flatness tolerance, canvas units, that keeps the mapped curves within [`DEVICE_TOLERANCE_PX`] at a device
/// scale of `px_per_unit`.
pub fn tolerance_for_scale(px_per_unit: f64) -> f64 {
    DEVICE_TOLERANCE_PX / px_per_unit.clamp(1.0 / 16.0, MAX_PX_PER_UNIT)
}

/// The tolerance scene resolution uses: exact at every scale the editor reaches.
pub fn envelope_tolerance() -> f64 {
    tolerance_for_scale(MAX_PX_PER_UNIT)
}

// ---------------------------------------------------------------------------
// The envelope spec
// ---------------------------------------------------------------------------

/// One point of an envelope mesh, in content-box units; handles are offsets from the point (`None`: a third of
/// the way to the neighbour, a straight edge).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshPoint {
    pub p: V2,
    pub left: Option<V2>,
    pub right: Option<V2>,
    pub up: Option<V2>,
    pub down: Option<V2>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub rows: usize,
    pub columns: usize,
    /// Row-major, `(rows + 1) × (columns + 1)`.
    pub points: Vec<MeshPoint>,
}

/// Most rows or columns a mesh may have.
pub const MAX_MESH_DIVISIONS: usize = 50;

#[derive(Debug, Clone)]
pub enum EnvelopeSpec {
    Warp(WarpEffect),
    Mesh(Mesh),
    TopObject(Shape),
    Deform(Deform),
}

/// A deformer about a point, in the group's own coordinates (px): inside `radius` of `center`, `bulge` moves points
/// out from the centre (`amount` > 0) or in toward it (< 0), `twist` turns them about it (`amount` degrees at the
/// centre); both fade smoothly to nothing at `radius`, outside which nothing moves.
///
/// Bulge: a point at distance `r` goes to `r · (1 + a · (1 − (r / R)²)²)`, which never folds for `a` in
/// [−0.95, 1.2] (the range `amount` is clamped to). Twist: it turns by `amount · (1 − r / R)²` degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deform {
    pub kind: DeformKind,
    pub center: V2,
    pub radius: f64,
    pub amount: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeformKind {
    Bulge,
    Twist,
}

/// The bulge factor's range: outside it the map would fold over itself.
pub const BULGE_RANGE: (f64, f64) = (-0.95, 1.2);

impl Deform {
    pub fn map(&self, p: V2) -> V2 {
        let d = p - self.center;
        let r = (d.x * d.x + d.y * d.y).sqrt();
        if !(self.radius > 0.0) || r >= self.radius || r == 0.0 {
            return p;
        }
        let x = r / self.radius;
        match self.kind {
            DeformKind::Bulge => {
                let a = self.amount.clamp(BULGE_RANGE.0, BULGE_RANGE.1);
                let k = 1.0 - x * x;
                self.center + d * (1.0 + a * k * k)
            }
            DeformKind::Twist => {
                let th = (self.amount * (1.0 - x) * (1.0 - x)).to_radians();
                let (s, c) = th.sin_cos();
                self.center + v2(d.x * c - d.y * s, d.x * s + d.y * c)
            }
        }
    }
}

fn num(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|x| x.is_finite())
}

fn vec_of(v: Option<&Value>) -> Option<V2> {
    let v = v?;
    if let Some(a) = v.as_array() {
        return Some(v2(num(a.first())?, num(a.get(1))?));
    }
    Some(v2(num(v.get("x"))?, num(v.get("y"))?))
}

impl Mesh {
    /// The identity mesh: points evenly spaced over the unit box, straight edges.
    pub fn identity(rows: usize, columns: usize) -> Mesh {
        let mut points = Vec::with_capacity((rows + 1) * (columns + 1));
        for j in 0..=rows {
            for i in 0..=columns {
                points.push(MeshPoint {
                    p: v2(i as f64 / columns as f64, j as f64 / rows as f64),
                    left: None,
                    right: None,
                    up: None,
                    down: None,
                });
            }
        }
        Mesh { rows, columns, points }
    }

    fn at(&self, i: usize, j: usize) -> &MeshPoint {
        &self.points[j * (self.columns + 1) + i]
    }

    fn p(&self, i: usize, j: usize) -> V2 {
        self.at(i, j).p
    }

    /// The absolute control point of the handle leaving `(i, j)` toward its neighbour `(ni, nj)`.
    fn handle(&self, i: usize, j: usize, ni: usize, nj: usize) -> V2 {
        let m = self.at(i, j);
        let h = if ni > i {
            m.right
        } else if ni < i {
            m.left
        } else if nj > j {
            m.down
        } else {
            m.up
        };
        m.p + h.unwrap_or_else(|| (self.p(ni, nj) - m.p) * (1.0 / 3.0))
    }

    fn edge(&self, a: (usize, usize), b: (usize, usize)) -> Cubic {
        Cubic([
            self.p(a.0, a.1),
            self.handle(a.0, a.1, b.0, b.1),
            self.handle(b.0, b.1, a.0, a.1),
            self.p(b.0, b.1),
        ])
    }

    /// The mesh map in unit space: `(u, v)` across the content box → where it lands (box units). Each cell is the
    /// Coons patch of its four edge curves; with straight edges and even handles that is the bilinear map, so the
    /// identity mesh maps every point to itself.
    pub fn eval(&self, u: f64, v: f64) -> V2 {
        let (cols, rows) = (self.columns as f64, self.rows as f64);
        let su = u * cols;
        let sv = v * rows;
        let ci = (su.floor().max(0.0) as usize).min(self.columns - 1);
        let cj = (sv.floor().max(0.0) as usize).min(self.rows - 1);
        let s = su - ci as f64;
        let t = sv - cj as f64;
        let top = self.edge((ci, cj), (ci + 1, cj)).eval(s);
        let bottom = self.edge((ci, cj + 1), (ci + 1, cj + 1)).eval(s);
        let left = self.edge((ci, cj), (ci, cj + 1)).eval(t);
        let right = self.edge((ci + 1, cj), (ci + 1, cj + 1)).eval(t);
        let (p00, p10, p01, p11) = (self.p(ci, cj), self.p(ci + 1, cj), self.p(ci, cj + 1), self.p(ci + 1, cj + 1));
        top * (1.0 - t) + bottom * t + left * (1.0 - s) + right * s
            - (p00 * ((1.0 - s) * (1.0 - t)) + p10 * (s * (1.0 - t)) + p01 * ((1.0 - s) * t) + p11 * (s * t))
    }
}

/// Parse a group's `envelope` field. `None` for anything that isn't a complete, valid envelope.
pub fn parse_spec(v: &Value) -> Option<EnvelopeSpec> {
    let o = v.as_object()?;
    match o.get("type").and_then(Value::as_str)? {
        "warp" => {
            let e: WarpEffect = crate::value::from_value(v.clone()).ok()?;
            crate::live_effects::WARP_STYLES
                .contains(&e.style.as_str())
                .then_some(EnvelopeSpec::Warp(e))
        }
        "mesh" => {
            let rows = o.get("rows").and_then(Value::as_u64)? as usize;
            let columns = o.get("columns").and_then(Value::as_u64)? as usize;
            if !(1..=MAX_MESH_DIVISIONS).contains(&rows) || !(1..=MAX_MESH_DIVISIONS).contains(&columns) {
                return None;
            }
            let mut mesh = Mesh::identity(rows, columns);
            if let Some(points) = o.get("points").and_then(Value::as_array) {
                if points.len() != mesh.points.len() {
                    return None;
                }
                for (slot, p) in mesh.points.iter_mut().zip(points) {
                    *slot = MeshPoint {
                        p: vec_of(Some(p))?,
                        left: vec_of(p.get("left")),
                        right: vec_of(p.get("right")),
                        up: vec_of(p.get("up")),
                        down: vec_of(p.get("down")),
                    };
                }
            }
            Some(EnvelopeSpec::Mesh(mesh))
        }
        "top-object" => {
            let points: Vec<BlendPathPoint> = o
                .get("points")?
                .as_array()?
                .iter()
                .filter_map(BlendPathPoint::from_json)
                .collect();
            let shape = contours_from_points(&points, true);
            // The largest ring is the envelope's shape.
            let ring = shape
                .into_iter()
                .filter(|c| c.closed && c.segs.len() >= 2)
                .max_by(|a, b| ring_area(a).abs().total_cmp(&ring_area(b).abs()))?;
            (ring_area(&ring).abs() > 1e-9).then(|| EnvelopeSpec::TopObject(vec![ring]))
        }
        "deform" => {
            let kind = match o.get("kind").and_then(Value::as_str)? {
                "bulge" => DeformKind::Bulge,
                "twist" => DeformKind::Twist,
                _ => return None,
            };
            let radius = num(o.get("radius"))?;
            (radius > 0.0).then_some(())?;
            Some(EnvelopeSpec::Deform(Deform { kind, center: vec_of(o.get("center"))?, radius, amount: num(o.get("amount")).unwrap_or(0.0) }))
        }
        _ => None,
    }
}

fn ring_area(c: &Contour) -> f64 {
    let mut a = 0.0;
    let mut prev = c.start;
    for s in &c.segs {
        for k in 1..=8 {
            let p = s.eval(k as f64 / 8.0);
            a += prev.cross(p);
            prev = p;
        }
    }
    a / 2.0
}

/// Whether `node` is an envelope: a group with a valid `envelope` field.
pub fn is_envelope(node: &RawNode) -> bool {
    node.node_type() == "group" && node.json_field(ENVELOPE_KEY).is_some_and(|v| parse_spec(v).is_some())
}

// ---------------------------------------------------------------------------
// Maps
// ---------------------------------------------------------------------------

/// One side of a top object: an arc-length-parametrised run along its ring.
struct Ring {
    segs: Vec<Seg>,
    /// Cumulative arc length at each segment's start, then the total.
    cum: Vec<f64>,
    /// Per segment: (arc length, t) samples, for the arc-length → t lookup.
    table: Vec<Vec<(f64, f64)>>,
}

impl Ring {
    fn new(c: &Contour) -> Ring {
        let segs: Vec<Seg> = c.segs.iter().copied().filter(|s| s.len() > 1e-12).collect();
        let mut cum = Vec::with_capacity(segs.len() + 1);
        let mut table = Vec::with_capacity(segs.len());
        let mut at = 0.0;
        for s in &segs {
            cum.push(at);
            const N: usize = 32;
            let mut rows = Vec::with_capacity(N + 1);
            let mut len = 0.0;
            let mut prev = s.eval(0.0);
            rows.push((0.0, 0.0));
            for k in 1..=N {
                let t = k as f64 / N as f64;
                let p = s.eval(t);
                len += prev.dist(p);
                prev = p;
                rows.push((len, t));
            }
            table.push(rows);
            at += len;
        }
        cum.push(at);
        Ring { segs, cum, table }
    }

    fn total(&self) -> f64 {
        *self.cum.last().unwrap_or(&0.0)
    }

    /// The point at arc length `s` (wrapped into the ring).
    fn at(&self, s: f64) -> V2 {
        let total = self.total();
        if self.segs.is_empty() || !(total > 0.0) {
            return self.segs.first().map(|s| s.start()).unwrap_or(v2(0.0, 0.0));
        }
        let s = s.rem_euclid(total);
        let k = match self.cum.binary_search_by(|c| c.total_cmp(&s)) {
            Ok(k) => k.min(self.segs.len() - 1),
            Err(k) => k.saturating_sub(1).min(self.segs.len() - 1),
        };
        let local = s - self.cum[k];
        let rows = &self.table[k];
        let r = rows.partition_point(|(l, _)| *l < local).clamp(1, rows.len() - 1);
        let (l0, t0) = rows[r - 1];
        let (l1, t1) = rows[r];
        let t = if l1 > l0 { t0 + (t1 - t0) * ((local - l0) / (l1 - l0)).clamp(0.0, 1.0) } else { t0 };
        self.segs[k].eval(t)
    }

    /// The arc length of the ring point where `score` is largest (the first, on a tie).
    fn extreme(&self, score: impl Fn(V2) -> f64) -> f64 {
        let mut best = (f64::NEG_INFINITY, 0.0);
        for (k, s) in self.segs.iter().enumerate() {
            for (l, t) in &self.table[k] {
                let v = score(s.eval(*t));
                if v > best.0 + 1e-12 {
                    best = (v, self.cum[k] + l);
                }
            }
        }
        best.1
    }
}

/// One side of a top object, between two corners, parametrised by how far it has come along the side's own
/// direction (down for the left and right sides, across for the top and bottom), with a little arc length mixed
/// in so a stretch that runs across the direction still advances. Rows of the content then run level between the
/// left and right sides (text fitted into a bottle stays level on its shoulders, where an arc-length Coons patch
/// folds).
struct Side {
    /// Arc positions along the ring, in the side's order.
    s: Vec<f64>,
    /// The side's measure at each position, 0 → 1.
    m: Vec<f64>,
}

impl Side {
    fn new(ring: &Ring, from: f64, to: f64, dir: f64, vertical: bool) -> Side {
        const N: usize = 1024;
        let total = ring.total();
        let len = ((to - from) * dir).rem_euclid(total);
        let mut s = Vec::with_capacity(N + 1);
        let mut m = Vec::with_capacity(N + 1);
        let mut acc = 0.0;
        let mut prev = ring.at(from);
        for k in 0..=N {
            let at = from + dir * len * k as f64 / N as f64;
            let p = ring.at(at);
            let d = p - prev;
            let (along, across) = if vertical { (d.y.abs(), d.x.abs()) } else { (d.x.abs(), d.y.abs()) };
            acc += along + 0.05 * across;
            prev = p;
            s.push(at);
            m.push(acc);
        }
        let last = *m.last().unwrap_or(&0.0);
        if last > 0.0 {
            for v in &mut m {
                *v /= last;
            }
        } else {
            for (k, v) in m.iter_mut().enumerate() {
                *v = k as f64 / N as f64;
            }
        }
        Side { s, m }
    }

    fn at(&self, ring: &Ring, f: f64) -> V2 {
        let f = f.clamp(0.0, 1.0);
        let r = self.m.partition_point(|v| *v < f).clamp(1, self.m.len() - 1);
        let (m0, m1) = (self.m[r - 1], self.m[r]);
        let t = if m1 > m0 { (f - m0) / (m1 - m0) } else { 0.0 };
        ring.at(self.s[r - 1] + (self.s[r] - self.s[r - 1]) * t)
    }
}

/// A top object's Coons patch in unit space.
struct TopObject {
    ring: Ring,
    /// Top-left → top-right, top-right → bottom-right, bottom-left → bottom-right, top-left → bottom-left.
    top: Side,
    right: Side,
    bottom: Side,
    left: Side,
    /// The corners nearest the ring's bounds' top-left, top-right, bottom-right, bottom-left.
    corners: [V2; 4],
}

impl TopObject {
    fn new(shape: &Shape) -> Option<TopObject> {
        let c = shape.first()?;
        let ring = Ring::new(c);
        if !(ring.total() > 0.0) {
            return None;
        }
        let b = bounds_of(std::iter::once(shape))?;
        // The corners are the ring's extremes along the box's diagonals (in box units, so the aspect doesn't
        // matter): the neck's corners of a bottle, not the shoulders that are nearer the box's corners.
        let (w, h) = (b.width().max(1e-12), b.height().max(1e-12));
        let at = [
            ring.extreme(|p| -((p.x - b.min.x) / w + (p.y - b.min.y) / h)),
            ring.extreme(|p| (p.x - b.min.x) / w - (p.y - b.min.y) / h),
            ring.extreme(|p| (p.x - b.min.x) / w + (p.y - b.min.y) / h),
            ring.extreme(|p| (p.y - b.min.y) / h - (p.x - b.min.x) / w),
        ];
        let total = ring.total();
        let fwd = |a: f64, b: f64| (b - a).rem_euclid(total);
        // Forward from the top-left corner, the top-right comes before the bottom-left on a ring that runs
        // clockwise on screen.
        let dir = if fwd(at[0], at[1]) <= fwd(at[0], at[3]) { 1.0 } else { -1.0 };
        let top = Side::new(&ring, at[0], at[1], dir, false);
        let right = Side::new(&ring, at[1], at[2], dir, true);
        // The bottom and left run against the ring.
        let bottom = Side::new(&ring, at[3], at[2], -dir, false);
        let left = Side::new(&ring, at[0], at[3], -dir, true);
        let corners = at.map(|s| ring.at(s));
        Some(TopObject { ring, top, right, bottom, left, corners })
    }

    fn eval(&self, u: f64, v: f64) -> V2 {
        let r = &self.ring;
        let top = self.top.at(r, u);
        let bottom = self.bottom.at(r, u);
        let left = self.left.at(r, v);
        let right = self.right.at(r, v);
        let [tl, tr, br, bl] = self.corners;
        top * (1.0 - v) + bottom * v + left * (1.0 - u) + right * u
            - (tl * ((1.0 - u) * (1.0 - v)) + tr * (u * (1.0 - v)) + bl * ((1.0 - u) * v) + br * (u * v))
    }
}

/// An envelope's point map over the content box `b`, envelope-local coordinates in and out.
pub struct EnvelopeMap {
    f: Box<dyn Fn(V2) -> V2>,
    /// A length that sets the scale of the map's features (the box's larger side): the initial split of long
    /// segments.
    pub feature: f64,
}

impl EnvelopeMap {
    pub fn map(&self, p: V2) -> V2 {
        (self.f)(p)
    }
}

/// Build the map of `spec` over the content box `b`. `None` for an empty box.
pub fn envelope_map(spec: &EnvelopeSpec, b: Bounds) -> Option<EnvelopeMap> {
    let (w, h) = (b.width(), b.height());
    if !(w.is_finite() && h.is_finite()) || (w <= 0.0 && h <= 0.0) {
        return None;
    }
    let feature = w.max(h);
    let (min, bw, bh) = (b.min, w.max(1e-9), h.max(1e-9));
    let unit = move |p: V2| v2(if w > 0.0 { (p.x - min.x) / bw } else { 0.5 }, if h > 0.0 { (p.y - min.y) / bh } else { 0.5 });
    let back = move |q: V2| v2(min.x + q.x * bw, min.y + q.y * bh);
    let f: Box<dyn Fn(V2) -> V2> = match spec {
        EnvelopeSpec::Warp(e) => Box::new(warp_map(e, b, true)?),
        EnvelopeSpec::Mesh(mesh) => {
            let mesh = mesh.clone();
            Box::new(move |p| {
                let q = unit(p);
                back(mesh.eval(q.x, q.y))
            })
        }
        EnvelopeSpec::TopObject(shape) => {
            let t = TopObject::new(shape)?;
            Box::new(move |p| {
                let q = unit(p);
                back(t.eval(q.x, q.y))
            })
        }
        EnvelopeSpec::Deform(d) => {
            let d = *d;
            // Its feature is its radius: segments are split finely enough to follow it.
            return Some(EnvelopeMap { f: Box::new(move |p| d.map(p)), feature: feature.min(d.radius * 2.0).max(1e-3) });
        }
    };
    Some(EnvelopeMap { f, feature })
}

// ---------------------------------------------------------------------------
// Exact curve mapping
// ---------------------------------------------------------------------------

/// Map `shape` through `f` to within `tol`: each segment is split until the cubic Hermite fit of its mapped curve
/// (the mapped end points and end tangents) is within `tol` of the mapped curve at 7 check points. Long segments
/// are first cut to at most `feature / 8`, so a map feature narrower than a segment is never stepped over.
/// `budget` counts down the segments still allowed; once it runs out, segments are no longer split.
pub fn map_shape_exact(shape: &Shape, f: &dyn Fn(V2) -> V2, tol: f64, feature: f64, budget: &mut usize) -> Shape {
    shape
        .iter()
        .map(|c| {
            let mut segs = Vec::with_capacity(c.segs.len());
            for s in &c.segs {
                let len = s.len();
                if !(len > 0.0) || !len.is_finite() {
                    let p = f(s.start());
                    segs.push(Seg::line(p, p));
                    continue;
                }
                let n = if feature > 0.0 { ((len / (feature / 8.0)).ceil() as usize).clamp(1, 64) } else { 1 };
                for k in 0..n {
                    let (t0, t1) = (k as f64 / n as f64, (k + 1) as f64 / n as f64);
                    fit(s, f, t0, t1, tol, 0, budget, &mut segs);
                }
            }
            let start = segs.first().map(|s| s.start()).unwrap_or_else(|| f(c.start));
            Contour { start, segs, closed: c.closed }
        })
        .collect()
}

/// The mapped curve's derivative at `t` (by `t`), a second-order difference (one-sided at the ends).
fn mapped_d1(s: &Seg, f: &dyn Fn(V2) -> V2, t: f64) -> V2 {
    let h = 1e-5;
    let g = |t: f64| f(s.eval(t));
    if t - h < 0.0 {
        (g(t) * -3.0 + g(t + h) * 4.0 - g(t + 2.0 * h)) * (1.0 / (2.0 * h))
    } else if t + h > 1.0 {
        (g(t) * 3.0 - g(t - h) * 4.0 + g(t - 2.0 * h)) * (1.0 / (2.0 * h))
    } else {
        (g(t + h) - g(t - h)) * (1.0 / (2.0 * h))
    }
}

#[allow(clippy::too_many_arguments)]
fn fit(s: &Seg, f: &dyn Fn(V2) -> V2, t0: f64, t1: f64, tol: f64, depth: u32, budget: &mut usize, out: &mut Vec<Seg>) {
    let dt = t1 - t0;
    let a = f(s.eval(t0));
    let b = f(s.eval(t1));
    let da = mapped_d1(s, f, t0) * (dt / 3.0);
    let db = mapped_d1(s, f, t1) * (dt / 3.0);
    let c = Cubic([a, a + da, b - db, b]);
    let mut err: f64 = 0.0;
    for k in 1..8 {
        let u = k as f64 / 8.0;
        err = err.max(c.eval(u).dist(f(s.eval(t0 + u * dt))));
    }
    let finite = c.0.iter().all(|p| p.is_finite());
    if (err > tol || !finite) && depth < MAX_SPLIT_DEPTH && *budget > 1 {
        *budget -= 1;
        let m = 0.5 * (t0 + t1);
        fit(s, f, t0, m, tol, depth + 1, budget, out);
        fit(s, f, m, t1, tol, depth + 1, budget, out);
        return;
    }
    if !finite {
        out.push(Seg::line(a, b));
    } else {
        out.push(Seg::curve(c.0[0], c.0[1], c.0[2], c.0[3]));
    }
}

// ---------------------------------------------------------------------------
// Image cells (mesh texture mapping)
// ---------------------------------------------------------------------------

/// The projective map of the unit square onto `q` (TL, TR, BR, BL), row-major `[h11 … h33]`, `h33 = 1`
/// (Heckbert's square-to-quad).
pub fn unit_square_to_quad(q: [V2; 4]) -> Option<[f64; 9]> {
    let [p0, p1, p2, p3] = q;
    let d1 = p1 - p2;
    let d2 = p3 - p2;
    let d3 = p0 - p1 + p2 - p3;
    let (g, h) = if d3.x.abs() < 1e-12 && d3.y.abs() < 1e-12 {
        (0.0, 0.0)
    } else {
        let det = d1.x * d2.y - d2.x * d1.y;
        if det.abs() < 1e-18 {
            return None;
        }
        ((d3.x * d2.y - d2.x * d3.y) / det, (d1.x * d3.y - d3.x * d1.y) / det)
    };
    let m = [
        p1.x - p0.x + g * p1.x,
        p3.x - p0.x + h * p3.x,
        p0.x,
        p1.y - p0.y + g * p1.y,
        p3.y - p0.y + h * p3.y,
        p0.y,
        g,
        h,
        1.0,
    ];
    m.iter().all(|v| v.is_finite()).then_some(m)
}

pub fn apply_projective(m: &[f64; 9], u: f64, v: f64) -> V2 {
    let w = m[6] * u + m[7] * v + m[8];
    v2((m[0] * u + m[1] * v + m[2]) / w, (m[3] * u + m[4] * v + m[5]) / w)
}

/// A convex quad with a consistent winding and no degenerate corner (the renderer's `distort` rule).
fn convex(q: &[V2; 4]) -> bool {
    let mut sign = 0.0;
    let mut area = 0.0;
    for i in 0..4 {
        let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
        area += a.cross(b);
        let turn = (b - a).cross(c - b);
        if !turn.is_finite() || turn.abs() < 1e-10 {
            return false;
        }
        if sign == 0.0 {
            sign = turn.signum();
        } else if turn.signum() != sign {
            return false;
        }
    }
    area.abs() > 1e-10
}

/// One image cell: a source rectangle (the image leaf's own coordinates) and the quad it lands on (envelope space).
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub src: [f64; 4],
    pub quad: [V2; 4],
}

/// Split the source box `[x, y, w, h]` into cells whose projective patch is within `tol` of `g` (source → envelope
/// space) at 9 check points. Cells that fold (not convex) are split to the depth limit and then dropped.
pub fn image_cells(bx: [f64; 4], g: &dyn Fn(V2) -> V2, tol: f64, budget: &mut usize) -> Vec<Cell> {
    let mut out = Vec::new();
    let [x, y, w, h] = bx;
    if !(w > 0.0 && h > 0.0) {
        return out;
    }
    // Start from roughly square cells.
    let (nx, ny) = if w >= h {
        (((w / h).round() as usize).clamp(1, 16), 1)
    } else {
        (1, ((h / w).round() as usize).clamp(1, 16))
    };
    for j in 0..ny {
        for i in 0..nx {
            let cw = w / nx as f64;
            let ch = h / ny as f64;
            cell(x + i as f64 * cw, y + j as f64 * ch, cw, ch, g, tol, 0, budget, &mut out);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cell(x: f64, y: f64, w: f64, h: f64, g: &dyn Fn(V2) -> V2, tol: f64, depth: u32, budget: &mut usize, out: &mut Vec<Cell>) {
    let quad = [g(v2(x, y)), g(v2(x + w, y)), g(v2(x + w, y + h)), g(v2(x, y + h))];
    let ok = quad.iter().all(|p| p.is_finite()) && convex(&quad);
    let err = match (ok, unit_square_to_quad(quad)) {
        (true, Some(m)) => {
            let mut e: f64 = 0.0;
            for (u, v) in [(0.5, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5), (0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)] {
                e = e.max(apply_projective(&m, u, v).dist(g(v2(x + u * w, y + v * h))));
            }
            e
        }
        _ => f64::INFINITY,
    };
    if err > tol && depth < MAX_CELL_DEPTH && *budget >= 4 {
        *budget -= 3;
        let (hw, hh) = (w / 2.0, h / 2.0);
        cell(x, y, hw, hh, g, tol, depth + 1, budget, out);
        cell(x + hw, y, hw, hh, g, tol, depth + 1, budget, out);
        cell(x + hw, y + hh, hw, hh, g, tol, depth + 1, budget, out);
        cell(x, y + hh, hw, hh, g, tol, depth + 1, budget, out);
        return;
    }
    if ok {
        out.push(Cell { src: [x, y, w, h], quad });
    }
}

// ---------------------------------------------------------------------------
// Content
// ---------------------------------------------------------------------------

/// How a leaf's local coordinates reach the envelope's space.
#[derive(Clone)]
enum ToEnv {
    Affine(Affine),
    /// An image cell of an inner envelope: its projective patch (cell box `[x, y, w, h]` → the quad, inner
    /// envelope space), then the inner envelope's affine.
    Cell { m: [f64; 9], bx: [f64; 4], then: Affine },
}

impl ToEnv {
    fn apply(&self, p: V2) -> V2 {
        match self {
            ToEnv::Affine(a) => {
                let (x, y) = transform::apply_point((p.x, p.y), *a);
                v2(x, y)
            }
            ToEnv::Cell { m, bx, then } => {
                let q = apply_projective(m, (p.x - bx[0]) / bx[2], (p.y - bx[1]) / bx[3]);
                let (x, y) = transform::apply_point((q.x, q.y), *then);
                v2(x, y)
            }
        }
    }
}

fn affine_shape(shape: &Shape, a: &Affine) -> Shape {
    let f = |p: V2| {
        let (x, y) = transform::apply_point((p.x, p.y), *a);
        v2(x, y)
    };
    shape
        .iter()
        .map(|c| Contour {
            start: f(c.start),
            segs: c
                .segs
                .iter()
                .map(|s| {
                    let [p0, p1, p2, p3] = s.c.0;
                    if s.line {
                        Seg::line(f(p0), f(p3))
                    } else {
                        Seg::curve(f(p0), f(p1), f(p2), f(p3))
                    }
                })
                .collect(),
            closed: c.closed,
        })
        .collect()
}

/// One filled outline of a vector leaf, in the envelope's space before the map.
#[derive(Clone)]
struct Piece {
    shape: Shape,
    /// The `fills` stack it fills with.
    fills: Value,
    even_odd: bool,
}

enum Part {
    /// A vector leaf: its drawn record (the template of the transient) and its outlines.
    Vector { src: RawNode, suffix: &'static str, pieces: Vec<Piece> },
    /// One image fill layer of a leaf.
    Image { src: RawNode, box_: [f64; 4], to_env: ToEnv, fill: Value, suffix: String },
    /// A group of parts: a canonical group, a frame, a leaf with an image fill, or an inner envelope.
    Group { src: RawNode, suffix: &'static str, template: Map, mask: Option<usize>, children: Vec<Part> },
}

/// The keys a transient keeps from the node it was made from: paint, compositing, drawing effects, naming.
const KEPT_KEYS: &[&str] = &[
    "opacity", "blendMode", "effects", "name", "metadata", "maskType", "maskInvert", "maskFeather", "fills", "strokes",
];

fn template_of(node: &RawNode) -> Map {
    let mut m = Map::new();
    for k in KEPT_KEYS {
        if let Some(v) = node.json_field(k) {
            m.insert((*k).into(), v.clone());
        }
    }
    // The geometry effects are applied already; the drawing ones (shadows, glows, blurs) stay.
    if let Some(Value::Array(fx)) = m.get("effects") {
        let kept: Vec<Value> = fx
            .iter()
            .filter(|e| {
                !e.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|t| crate::effect::GEOMETRY_EFFECT_TYPES.contains(&t))
            })
            .cloned()
            .collect();
        if kept.is_empty() {
            m.remove("effects");
        } else {
            m.insert("effects".into(), Value::Array(kept));
        }
    }
    m
}

fn shown_layers(v: Option<&Value>) -> Vec<Value> {
    v.and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter(|l| {
                    l.get("visible").and_then(Value::as_bool) != Some(false)
                        && l.get("opacity").and_then(Value::as_f64) != Some(0.0)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn is_image_layer(l: &Value) -> bool {
    l.get("paint").and_then(|p| p.get("type")).and_then(Value::as_str) == Some("image")
}

/// The drawn stroke of one stroke layer as a fill piece (its outline, filled with its paint).
fn stroke_piece(outline: &Shape, closed: bool, node: &RawNode, stroke: &Value, brush: Option<&Value>) -> Option<(Shape, Value)> {
    let mut m = node.to_json_map();
    m.insert("strokes".into(), Value::Array(vec![stroke.clone()]));
    let band = outline_stroke(outline, &RawNode::new(m), closed, brush)?;
    let mut fill = Map::new();
    for k in ["id", "visible", "opacity", "blendMode", "paint"] {
        if let Some(v) = stroke.get(k) {
            fill.insert(k.into(), v.clone());
        }
    }
    Some((band, Value::Array(vec![Value::Object(fill)])))
}

/// The resolved envelope (see the module docs): the transient tree to draw, in envelope-local coordinates.
#[derive(Debug, Clone, Default)]
pub struct Resolved {
    /// `(parent id, transient)` in drawing order per parent; the envelope's own children have its id as parent.
    pub transients: Vec<(NodeId, RawNode)>,
    /// Every warped outline and image cell, as one compound path record (the envelope used as a clip mask).
    pub outline: Option<RawNode>,
    /// The warped art's bounds, envelope-local.
    pub bounds: Option<Bounds>,
    /// The content's bounds before the map (what the envelope's map spans), envelope-local.
    pub content_bounds: Option<Bounds>,
    /// Text whose fonts aren't loaded yet: left out for now (retried when fonts arrive).
    pub pending: bool,
    /// Ids of transients that are clip-mask sources of a transient group.
    pub mask_sources: Vec<NodeId>,
    /// What couldn't be bent and was left out, as `id (reason)`.
    pub skipped: Vec<String>,
}

struct Walk<'a> {
    scene: &'a Scene,
    /// Envelope-local → page space, inverted: page → envelope-local.
    to_local: Affine,
    pending: bool,
    skipped: Vec<String>,
}

fn invert(a: &Affine) -> Option<Affine> {
    let det = a.a * a.d - a.b * a.c;
    if !det.is_finite() || det.abs() < 1e-18 {
        return None;
    }
    let (ia, ib, ic, id) = (a.d / det, -a.b / det, -a.c / det, a.a / det);
    Some(Affine { a: ia, b: ib, c: ic, d: id, e: -(ia * a.e + ic * a.f), f: -(ib * a.e + id * a.f) })
}

fn visible(node: &RawNode) -> bool {
    !node.truthy("hidden") && node.f64("opacity") != Some(0.0)
}

impl Walk<'_> {
    /// A node's local → envelope-local affine.
    fn to_env(&self, id: &str) -> Option<Affine> {
        let (a, _) = self.scene.accumulated_affine(id)?;
        Some(transform::compose(self.to_local, a))
    }

    fn children(&self, id: &str, node: &RawNode) -> Vec<NodeId> {
        self.scene
            .children_by_parent
            .get(id)
            .cloned()
            .or_else(|| node.array("children").map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()))
            .unwrap_or_default()
    }

    fn walk_ids(&mut self, ids: &[NodeId], out: &mut Vec<Part>) {
        for id in ids {
            if let Some(p) = self.walk(id) {
                out.push(p);
            }
        }
    }

    fn walk(&mut self, id: &str) -> Option<Part> {
        let canonical = self.scene.canonical_or_transient(id)?;
        if !visible(canonical) {
            return None;
        }
        let node = self.scene.effective_node(id)?.clone();
        match canonical.node_type() {
            "group" => {
                // An inner envelope: its transients (already bent) are its content.
                if self.scene.envelope_transients.contains_key(id) {
                    let ids = self.scene.transient_by_parent.get(id).cloned().unwrap_or_default();
                    let mut children = Vec::new();
                    self.walk_ids(&ids, &mut children);
                    return Some(Part::Group { src: node.clone(), suffix: "", template: template_of(&node), mask: None, children });
                }
                let ids = self.children(id, canonical);
                let mut children = Vec::new();
                let mut mask = None;
                let mask_id = canonical.str("clipMaskNodeId").filter(|m| !m.is_empty()).map(str::to_string);
                for cid in &ids {
                    if let Some(p) = self.walk(cid) {
                        if mask_id.as_deref() == Some(cid.as_str()) {
                            // The mask clips its siblings and never draws: only a single outline can clip here.
                            if !matches!(p, Part::Vector { .. }) {
                                continue;
                            }
                            mask = Some(children.len());
                        }
                        children.push(p);
                    }
                }
                let mut template = template_of(&node);
                template.remove("fills");
                template.remove("strokes");
                if let Some(m) = &mask_id {
                    if mask.is_none() {
                        self.skipped.push(format!("{m} (a clip mask that is not a single shape)"));
                    }
                }
                for k in ["clipMode", "clipInherit"] {
                    if let Some(v) = canonical.json_field(k) {
                        template.insert(k.into(), v.clone());
                    }
                }
                Some(Part::Group { src: node, suffix: "", template, mask, children })
            }
            "boolean" => self.vector_leaf(id, &node),
            "container" | "section" => {
                let a = self.to_env(id)?;
                let mut template = template_of(&node);
                template.remove("fills");
                template.remove("strokes");
                let mut box_node = node.to_json_map();
                box_node.insert("type".into(), json!("rectangle"));
                let box_node = RawNode::new(box_node);
                let mut children = Vec::new();
                // The frame's own fill and stroke, under its children.
                let mut body = node.to_json_map();
                body.insert("opacity".into(), json!(1));
                body.remove("blendMode");
                body.remove("effects");
                if let Some(p) = self.shape_leaf(&RawNode::new(body.clone()), &box_node, &a, "~body") {
                    children.push(p);
                }
                let mut mask = None;
                if node.truthy("clipContent") {
                    if let Some((pts, closed)) = node_outline(&box_node) {
                        let shape = affine_shape(&contours_from_points(&pts, closed), &a);
                        mask = Some(children.len());
                        children.push(Part::Vector {
                            src: node.clone(),
                            suffix: "~clip",
                            pieces: vec![Piece { shape, fills: json!([{"id": "clip", "paint": {"type": "solid", "color": "#000000"}}]), even_odd: false }],
                        });
                    }
                }
                if !(node.node_type() == "section" && node.truthy("contentsHidden")) {
                    let ids = self.children(id, canonical);
                    self.walk_ids(&ids, &mut children);
                }
                Some(Part::Group { src: node, suffix: "", template, mask, children })
            }
            "table" | "generated" | "generator" => {
                let ids = self.scene.transient_by_parent.get(id).cloned().unwrap_or_default();
                let mut children = Vec::new();
                self.walk_ids(&ids, &mut children);
                let mut template = Map::new();
                template.insert("opacity".into(), json!(1));
                Some(Part::Group { src: node, suffix: "", template, mask: None, children })
            }
            "richtext" => {
                if node.node_type() != "richtext" {
                    // Warped text resolves to a path already.
                    return self.vector_leaf(id, &node);
                }
                let a = self.to_env(id)?;
                let Some(outliner) = crate::text_warp::outliner() else {
                    self.pending = true;
                    return None;
                };
                match outliner(self.scene, &node) {
                    TextOutlines::Pending => {
                        self.pending = true;
                        None
                    }
                    TextOutlines::Ready(groups) => {
                        let pieces = groups
                            .iter()
                            .map(|g| Piece { shape: affine_shape(&contours_from_points(&g.points, true), &a), fills: g.fills.clone(), even_odd: false })
                            .collect();
                        let mut src = node.to_json_map();
                        src.remove("textFills");
                        Some(Part::Vector { src: RawNode::new(src), suffix: "", pieces })
                    }
                }
            }
            "line" => self.line_leaf(id, &node),
            "rectangle" | "ellipse" | "star" | "polygon" | "path" => {
                if node.json_field(ENVELOPE_CELL_KEY).is_some() {
                    return self.cell_leaf(id, &node);
                }
                self.vector_leaf(id, &node)
            }
            other => {
                self.skipped.push(format!("{id} ({other} layers don't bend yet)"));
                None
            }
        }
    }

    /// A shape, path or boolean result: its drawn copies (live effects applied), each a leaf.
    fn vector_leaf(&mut self, id: &str, node: &RawNode) -> Option<Part> {
        let a = self.to_env(id)?;
        let copies = split_copies(node).unwrap_or_else(|| vec![node.clone()]);
        if copies.len() == 1 {
            return self.shape_leaf(&copies[0], &copies[0], &a, "");
        }
        let mut children = Vec::new();
        // Copies that each resolve to plain outlines draw as one path of pieces (a warped text layer's run
        // colours, an inner envelope's fill and stroke), keeping one transient per leaf.
        let mut merged = Vec::new();
        for c in &copies {
            match self.shape_leaf(c, c, &a, "") {
                Some(Part::Vector { pieces, .. }) if pieces.iter().all(|p| !p.even_odd) => merged.extend(pieces),
                None => {}
                _ => {
                    merged.clear();
                    break;
                }
            }
        }
        if !merged.is_empty() {
            return Some(Part::Vector { src: node.clone(), suffix: "", pieces: merged });
        }
        for (k, c) in copies.iter().enumerate() {
            let mut c = c.to_json_map();
            c.insert("opacity".into(), json!(1));
            c.remove("blendMode");
            c.remove("effects");
            let c = RawNode::new(c);
            if let Some(mut p) = self.shape_leaf(&c, &c, &a, "") {
                if let Part::Vector { suffix, .. } | Part::Group { suffix, .. } = &mut p {
                    *suffix = COPY_SUFFIXES.get(k).copied().unwrap_or("~copy");
                }
                children.push(p);
            }
        }
        let mut template = template_of(node);
        template.remove("fills");
        template.remove("strokes");
        Some(Part::Group { src: node.clone(), suffix: "", template, mask: None, children })
    }

    /// One shape: `paint` gives the fills and strokes, `geom` the outline (a frame's body draws with the frame's
    /// paint on its box).
    fn shape_leaf(&mut self, paint: &RawNode, geom: &RawNode, a: &Affine, suffix: &'static str) -> Option<Part> {
        let (pts, closed) = node_outline(geom)?;
        let local = contours_from_points(&pts, closed);
        let even_odd = geom.str("fillRule") == Some("evenodd");
        let fills = shown_layers(paint.json_field("fills"));
        let strokes = shown_layers(paint.json_field("strokes"));
        let images: Vec<Value> = fills.iter().filter(|l| is_image_layer(l)).cloned().collect();
        let plain: Vec<Value> = fills.iter().filter(|l| !is_image_layer(l)).cloned().collect();
        let brush = paint
            .json_field("strokes")
            .and_then(Value::as_array)
            .and_then(|s| s.last())
            .and_then(|s| s.get("brush"))
            .and_then(|b| b.get("brushId"))
            .and_then(Value::as_str)
            .and_then(|b| self.scene.doc.brush(b));
        let env = affine_shape(&local, a);
        let mut fill_pieces = Vec::new();
        if !plain.is_empty() && closed_enough(&local, closed) {
            fill_pieces.push(Piece { shape: env.clone(), fills: Value::Array(plain), even_odd });
        }
        let mut stroke_pieces = Vec::new();
        for s in &strokes {
            if s.get("dash").and_then(Value::as_array).is_some_and(|d| d.iter().any(|v| v.as_f64().is_some_and(|x| x > 0.0))) {
                self.skipped.push(format!("{} (its stroke's dashes: drawn solid)", paint.str(ENVELOPE_SOURCE_KEY).unwrap_or(paint.id())));
            }
            if s.get("width").and_then(Value::as_f64).is_some_and(|w| w > 0.0) {
                if let Some((band, fills)) = stroke_piece(&local, closed, paint, s, brush) {
                    stroke_pieces.push(Piece { shape: affine_shape(&band, a), fills, even_odd: false });
                }
            }
        }
        if images.is_empty() && !(even_odd && !stroke_pieces.is_empty() && !fill_pieces.is_empty()) {
            let mut pieces = fill_pieces;
            pieces.extend(stroke_pieces);
            if pieces.is_empty() {
                return None;
            }
            return Some(Part::Vector { src: paint.clone(), suffix, pieces });
        }
        // An image fill, or an even-odd fill under a stroke: a group of parts.
        let mut children = Vec::new();
        if !fill_pieces.is_empty() {
            children.push(Part::Vector { src: paint.clone(), suffix: "~fill", pieces: fill_pieces });
        }
        if !images.is_empty() {
            let box_ = crate::paint::node_image_box(geom)
                .or_else(|| bounds_of(std::iter::once(&local)).map(|b| [b.min.x, b.min.y, b.width(), b.height()]))?;
            let mut cells = Vec::new();
            for (k, fill) in images.iter().enumerate() {
                cells.push(Part::Image { src: paint.clone(), box_, to_env: ToEnv::Affine(*a), fill: fill.clone(), suffix: format!("~img{k}") });
            }
            let plain_box = geom.node_type() == "rectangle"
                && square_corners(geom)
                && crate::paint::node_image_box(geom).is_some();
            if plain_box {
                children.extend(cells);
            } else {
                let mut inner = vec![Part::Vector {
                    src: paint.clone(),
                    suffix: "~imgclip",
                    pieces: vec![Piece { shape: env.clone(), fills: json!([{"id": "clip", "paint": {"type": "solid", "color": "#000000"}}]), even_odd }],
                }];
                inner.extend(cells);
                let mut t = Map::new();
                t.insert("opacity".into(), json!(1));
                children.push(Part::Group { src: paint.clone(), suffix: "~imgs", template: t, mask: Some(0), children: inner });
            }
        }
        if !stroke_pieces.is_empty() {
            children.push(Part::Vector { src: paint.clone(), suffix: "~stroke", pieces: stroke_pieces });
        }
        let mut template = template_of(paint);
        template.remove("fills");
        template.remove("strokes");
        Some(Part::Group { src: paint.clone(), suffix, template, mask: None, children })
    }

    /// A straight or elbowed line: its stroke as a polyline path.
    fn line_leaf(&mut self, id: &str, node: &RawNode) -> Option<Part> {
        let a = self.to_env(id)?;
        let pts = node.array("points")?;
        let n = |i: usize| pts.get(i).and_then(Value::as_f64);
        let (x1, y1, x2, y2) = (n(0)?, n(1)?, n(2)?, n(3)?);
        let mut anchors = vec![v2(x1, y1)];
        if node.str("routing").unwrap_or("straight") != "straight" {
            if let Some(w) = node.array("waypoints") {
                anchors.extend(w.iter().filter_map(|p| vec_of(Some(p))));
            }
        }
        anchors.push(v2(x2, y2));
        if node.str("startArrow").is_some() || node.str("endArrow").is_some() || stroke_has_arrows(node) {
            self.skipped.push(format!("{id} (its arrowheads)"));
        }
        let mut m = node.to_json_map();
        m.insert("type".into(), json!("path"));
        let points: Vec<Value> = anchors
            .iter()
            .enumerate()
            .map(|(k, p)| json!({"type": if k == 0 { "move" } else { "line" }, "x": p.x, "y": p.y}))
            .collect();
        m.insert("points".into(), Value::Array(points));
        m.insert("closed".into(), json!(false));
        m.remove("fills");
        let path = RawNode::new(m);
        self.shape_leaf(&path, &path, &a, "")
    }

    /// An image cell of an inner envelope: bend it again.
    fn cell_leaf(&mut self, id: &str, node: &RawNode) -> Option<Part> {
        let then = self.to_env(id)?;
        let q: Vec<f64> = node.array(ENVELOPE_CELL_KEY)?.iter().filter_map(Value::as_f64).collect();
        if q.len() != 8 {
            return None;
        }
        let quad = [v2(q[0], q[1]), v2(q[2], q[3]), v2(q[4], q[5]), v2(q[6], q[7])];
        let (w, h) = (node.f64("width")?, node.f64("height")?);
        let m = unit_square_to_quad(quad)?;
        let fill = shown_layers(node.json_field("fills")).into_iter().find(is_image_layer)?;
        let mut src = node.to_json_map();
        src.remove(ENVELOPE_CELL_KEY);
        src.remove("distort");
        Some(Part::Image {
            src: RawNode::new(src),
            box_: [0.0, 0.0, w, h],
            to_env: ToEnv::Cell { m, bx: [0.0, 0.0, w, h], then },
            fill,
            suffix: String::new(),
        })
    }
}

/// A rectangle whose outline is its box: no corner radius (a number, or per-corner radii all 0).
fn square_corners(node: &RawNode) -> bool {
    match node.json_field("cornerRadius") {
        None | Some(Value::Null) => true,
        Some(Value::Number(n)) => n.as_f64() == Some(0.0),
        Some(Value::Array(a)) => a.iter().all(|v| v.as_f64() == Some(0.0)),
        Some(Value::Object(o)) => o.values().all(|v| v.as_f64() == Some(0.0)),
        _ => false,
    }
}

const COPY_SUFFIXES: &[&str] = &["~c0", "~c1", "~c2", "~c3", "~c4", "~c5", "~c6", "~c7", "~c8", "~c9"];

fn stroke_has_arrows(node: &RawNode) -> bool {
    node.array("strokes").is_some_and(|s| {
        s.iter().any(|s| {
            s.get("startArrow").and_then(Value::as_str).is_some_and(|a| a != "none")
                || s.get("endArrow").and_then(Value::as_str).is_some_and(|a| a != "none")
        })
    })
}

/// An open path fills too (as the renderer closes it), unless it is a single straight segment.
fn closed_enough(shape: &Shape, closed: bool) -> bool {
    closed || shape.iter().any(|c| c.segs.len() >= 2 || c.segs.iter().any(|s| !s.line))
}

/// The content box: the union of every piece's outline and every image box's corners, before the map.
fn union_into(acc: &mut Option<Bounds>, b: Bounds) {
    *acc = Some(match *acc {
        None => b,
        Some(a) => Bounds { min: v2(a.min.x.min(b.min.x), a.min.y.min(b.min.y)), max: v2(a.max.x.max(b.max.x), a.max.y.max(b.max.y)) },
    });
}

fn content_bounds(parts: &[Part], acc: &mut Option<Bounds>) {
    let add = |acc: &mut Option<Bounds>, b: Bounds| union_into(acc, b);
    for p in parts {
        match p {
            Part::Vector { pieces, .. } => {
                if let Some(b) = bounds_of(pieces.iter().map(|p| &p.shape)) {
                    add(acc, b);
                }
            }
            Part::Image { box_, to_env, .. } => {
                let [x, y, w, h] = *box_;
                for c in [v2(x, y), v2(x + w, y), v2(x + w, y + h), v2(x, y + h)] {
                    let q = to_env.apply(c);
                    if q.is_finite() {
                        add(acc, Bounds { min: q, max: q });
                    }
                }
            }
            Part::Group { children, .. } => content_bounds(children, acc),
        }
    }
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

struct Emit<'a> {
    env_id: &'a str,
    map: &'a EnvelopeMap,
    tol: f64,
    segment_budget: usize,
    cell_budget: usize,
    out: Resolved,
    /// What the envelope draws, as outlines (its clip shape when it is a clip mask).
    outline: Shape,
    /// Whether the parts being emitted add to [`Emit::outline`] (not inside a clip-masked group: the mask does).
    collect: bool,
    /// The last vector part's warped outlines.
    last: Shape,
}

fn base_transient(env_id: &str, id: String, src: &RawNode, ty: &str) -> Map {
    let mut m = Map::new();
    m.insert("id".into(), json!(id));
    m.insert("type".into(), json!(ty));
    for (k, v) in [("x", 0.0), ("y", 0.0), ("rotation", 0.0), ("scaleX", 1.0), ("scaleY", 1.0)] {
        m.insert(k.into(), json!(v));
    }
    m.insert(ENVELOPE_OF_KEY.into(), json!(env_id));
    m.insert(ENVELOPE_SOURCE_KEY.into(), json!(src.str(ENVELOPE_SOURCE_KEY).unwrap_or(src.id())));
    m
}

impl Emit<'_> {
    fn id_for(&self, src: &RawNode, suffix: &str) -> String {
        // An inner envelope's transient keeps its own path of ids under ours.
        format!("{}~{}{}", self.env_id, src.id(), suffix)
    }

    fn emit(&mut self, parent: &str, part: &Part) -> Option<String> {
        match part {
            Part::Vector { src, suffix, pieces } => {
                let id = self.id_for(src, suffix);
                let f = |p: V2| self.map.map(p);
                let mut all = Vec::new();
                let mut counts = Vec::new();
                let mut fills = Vec::new();
                let mut even_odd = false;
                let mut shapes: Shape = Vec::new();
                for piece in pieces {
                    let warped = map_shape_exact(&piece.shape, &f, self.tol, self.map.feature, &mut self.segment_budget);
                    let pts = points_from_contours(&warped);
                    counts.push(pts.len());
                    all.extend(pts.iter().map(BlendPathPoint::to_json));
                    fills.push(piece.fills.clone());
                    even_odd |= piece.even_odd;
                    shapes.extend(warped);
                }
                if self.collect {
                    self.outline.extend(shapes.iter().cloned());
                }
                self.last = shapes;
                let mut m = base_transient(self.env_id, id.clone(), src, "path");
                for (k, v) in template_of(src) {
                    m.insert(k, v);
                }
                m.remove("strokes");
                m.insert("fills".into(), fills.first().cloned().unwrap_or(json!([])));
                m.insert("points".into(), Value::Array(all));
                m.insert("closed".into(), json!(true));
                m.insert("fillRule".into(), json!(if even_odd { "evenodd" } else { "nonzero" }));
                if counts.len() > 1 {
                    m.insert(EFFECT_COPIES_KEY.into(), json!(counts));
                    m.insert(EFFECT_COPY_ROLES_KEY.into(), json!(vec!["fill"; counts.len()]));
                    if fills.iter().any(|f| *f != fills[0]) {
                        m.insert(EFFECT_COPY_FILLS_KEY.into(), Value::Array(fills));
                    }
                }
                self.out.transients.push((parent.to_string(), RawNode::new(m)));
                Some(id)
            }
            Part::Image { src, box_, to_env, fill, suffix } => {
                let id = self.id_for(src, suffix);
                let g = |p: V2| self.map.map(to_env.apply(p));
                let cells = image_cells(*box_, &g, IMAGE_TOLERANCE_PX, &mut self.cell_budget);
                let placement = resolved_placement(fill, *box_);
                let mut group = base_transient(self.env_id, id.clone(), src, "group");
                group.insert("opacity".into(), json!(1));
                let mut kids = Vec::with_capacity(cells.len());
                for (k, c) in cells.iter().enumerate() {
                    let Some(node) = cell_node(self.env_id, format!("{id}~{k}"), src, c, fill, &placement) else {
                        continue;
                    };
                    kids.push(json!(node.id()));
                    if self.collect {
                        self.outline.push(Contour {
                        start: c.quad[0],
                        segs: (0..4).map(|i| Seg::line(c.quad[i], c.quad[(i + 1) % 4])).collect(),
                        closed: true,
                        });
                    }
                    self.out.transients.push((id.clone(), node));
                }
                group.insert("children".into(), Value::Array(kids));
                // The group goes before its cells in the list; `Scene` indexes by parent.
                self.out.transients.push((parent.to_string(), RawNode::new(group)));
                Some(id)
            }
            Part::Group { src, suffix, template, mask, children } => {
                let id = self.id_for(src, if suffix.is_empty() { "" } else { suffix });
                let mut m = base_transient(self.env_id, id.clone(), src, "group");
                for (k, v) in template {
                    m.insert(k.clone(), v.clone());
                }
                let mut kids = Vec::with_capacity(children.len());
                let mut mask_id = None;
                let collect = self.collect;
                // A clip-masked group draws within its mask: the mask's outline stands for it.
                let mut mask_shape = None;
                if mask.is_some() {
                    self.collect = false;
                }
                for (k, c) in children.iter().enumerate() {
                    if let Some(cid) = self.emit(&id, c) {
                        if *mask == Some(k) {
                            mask_id = Some(cid.clone());
                            mask_shape = Some(std::mem::take(&mut self.last));
                        }
                        kids.push(json!(cid));
                    }
                }
                self.collect = collect;
                if let (true, Some(shape)) = (collect, mask_shape) {
                    self.outline.extend(shape);
                }
                m.insert("children".into(), Value::Array(kids));
                if let Some(mid) = mask_id {
                    m.insert("clipMaskNodeId".into(), json!(mid));
                    self.out.mask_sources.push(mid);
                }
                self.out.transients.push((parent.to_string(), RawNode::new(m)));
                Some(id)
            }
        }
    }
}

/// An image fill's placement resolved against its box (`[x, y, w, h]`): `(offset, scale, rotation, tile repeat)`.
fn resolved_placement(fill: &Value, box_: [f64; 4]) -> (V2, V2, f64, Option<&'static str>) {
    let paint: Option<crate::paint::ImagePaint> = fill.get("paint").and_then(|p| crate::value::from_value(p.clone()).ok());
    let placement = paint.map(|p| p.placement).unwrap_or_default();
    let r = placement.resolve(box_, None);
    let tile = (placement.mode() == "tile").then_some(r.repeat);
    (v2(r.offset.0, r.offset.1), v2(r.scale.0, r.scale.1), r.rotation, tile)
}

/// One cell: a rectangle over the cell's source box, corner-pinned (`distort`) onto its quad, filled with the
/// image placed as the leaf placed it.
fn cell_node(env_id: &str, id: String, src: &RawNode, c: &Cell, fill: &Value, placement: &(V2, V2, f64, Option<&'static str>)) -> Option<RawNode> {
    let [x, y, w, h] = c.src;
    let unit_quad = c.quad.map(|q| v2(q.x / w, q.y / h));
    let m = unit_square_to_quad(unit_quad)?;
    let (offset, scale, rotation, tile) = placement;
    let mut paint = fill.get("paint")?.as_object()?.clone();
    let mut pl = Map::new();
    // The cell's local origin is the source box's (x, y): the image's placement moves by that much.
    pl.insert("fit".into(), json!(if tile.is_some() { "tile" } else { "crop" }));
    if let Some(r) = tile {
        pl.insert("repeat".into(), json!(r));
    }
    pl.insert("offset".into(), json!({"x": offset.x - x, "y": offset.y - y}));
    pl.insert("scale".into(), json!({"x": scale.x, "y": scale.y}));
    if *rotation != 0.0 {
        pl.insert("rotation".into(), json!(rotation));
    }
    paint.insert("placement".into(), Value::Object(pl));
    let mut layer = Map::new();
    layer.insert("id".into(), fill.get("id").cloned().unwrap_or(json!("fill")));
    for k in ["opacity", "blendMode"] {
        if let Some(v) = fill.get(k) {
            layer.insert(k.into(), v.clone());
        }
    }
    layer.insert("paint".into(), Value::Object(paint));
    let mut n = base_transient(env_id, id, src, "rectangle");
    n.insert("width".into(), json!(w));
    n.insert("height".into(), json!(h));
    n.insert("fills".into(), json!([Value::Object(layer)]));
    n.insert("distort".into(), json!({"type": "matrix", "matrix": m.to_vec()}));
    n.insert(ENVELOPE_CELL_KEY.into(), json!(c.quad.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<_>>()));
    Some(RawNode::new(n))
}

/// Resolve envelope `env_id` of `scene` (see the module docs). Inner envelopes must be resolved already.
pub fn resolve(scene: &Scene, env_id: &str) -> Resolved {
    resolve_with(scene, env_id, envelope_tolerance())
}

/// [`resolve`] at a chosen flatness tolerance (canvas units).
pub fn resolve_with(scene: &Scene, env_id: &str, tol: f64) -> Resolved {
    let mut out = Resolved::default();
    let Some(node) = scene.node(env_id) else {
        return out;
    };
    let Some(spec) = node.json_field(ENVELOPE_KEY).and_then(parse_spec) else {
        return out;
    };
    let Some(to_local) = scene.accumulated_affine(env_id).and_then(|(a, _)| invert(&a)) else {
        return out;
    };
    let mut walk = Walk { scene, to_local, pending: false, skipped: Vec::new() };
    let ids = walk.children(env_id, node);
    let mut parts = Vec::new();
    walk.walk_ids(&ids, &mut parts);
    out.pending = walk.pending;
    out.skipped = walk.skipped;
    let mut cb = None;
    content_bounds(&parts, &mut cb);
    out.content_bounds = cb;
    let Some(map) = cb.and_then(|b| envelope_map(&spec, b)) else {
        return out;
    };
    let mut emit = Emit {
        env_id,
        map: &map,
        tol,
        segment_budget: MAX_ENVELOPE_SEGMENTS,
        cell_budget: MAX_IMAGE_CELLS,
        out,
        outline: Vec::new(),
        collect: true,
        last: Vec::new(),
    };
    for p in &parts {
        emit.emit(env_id, p);
    }
    let mut out = emit.out;
    let outline = emit.outline;
    out.bounds = bounds_of(std::iter::once(&outline));
    if !outline.is_empty() {
        let mut m = Map::new();
        m.insert("id".into(), json!(env_id));
        m.insert("type".into(), json!("path"));
        m.insert("closed".into(), json!(true));
        m.insert("fillRule".into(), json!("nonzero"));
        m.insert("points".into(), Value::Array(points_from_contours(&outline).iter().map(BlendPathPoint::to_json).collect()));
        for k in ["x", "y", "rotation", "scaleX", "scaleY", "skewX", "maskType", "maskInvert", "maskFeather"] {
            if let Some(v) = node.json_field(k) {
                m.insert(k.into(), v.clone());
            }
        }
        m.insert("fills".into(), json!([{"id": "mask", "paint": {"type": "solid", "color": "#000000"}}]));
        out.outline = Some(RawNode::new(m));
    }
    out
}

/// Ids of the envelopes and repeats (`crate::repeat`) in `scene`, innermost first (deepest nesting first, then
/// document order): each resolves after the live groups inside it.
pub fn envelope_ids(scene: &Scene) -> Vec<NodeId> {
    let mut envs: Vec<(usize, usize, NodeId)> = scene
        .node_order
        .iter()
        .enumerate()
        .filter(|(_, id)| scene.node(id).is_some_and(|n| is_envelope(n) || crate::repeat::is_repeat(n)))
        .map(|(k, id)| {
            let mut depth = 0;
            let mut cur = id.as_str();
            while let Some(p) = scene.parent_map.get(cur) {
                depth += 1;
                cur = p;
                if depth > 512 {
                    break;
                }
            }
            (depth, k, id.clone())
        })
        .collect();
    envs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    envs.into_iter().map(|(_, _, id)| id).collect()
}

/// Every transient id an envelope made, keyed by envelope (for [`Scene`]'s bookkeeping).
pub type EnvelopeTransients = HashMap<NodeId, Vec<NodeId>>;

// ---------------------------------------------------------------------------
// JSON entry point (the `envelope_geometry` wasm export)
// ---------------------------------------------------------------------------

/// `{"nodes": [the envelope group and every node under it], "envelopeId", "pxPerUnit"?, "brushes"?}` →
/// `{"status": "ok", "bounds", "contentBounds", "items": [tree], "pending", "skipped"}` |
/// `{"status": "none"}` (not an envelope) | `{"status": "error", "code", "message"}`.
///
/// `items` is the transient tree the renderer draws, envelope-local: `{"node": record, "children"?: [...]}` per
/// transient, in drawing order. A `rectangle` with `_envelopeCell` is an image cell. `pxPerUnit` (default: the
/// editor's top zoom) sets the flatness tolerance ([`tolerance_for_scale`]). Contract:
/// `packages/editor/src/core/geometry/envelope.ts`.
pub fn envelope_geometry_json(request_json: &str) -> String {
    let err = |code: &str, message: &str| json!({"status": "error", "code": code, "message": message}).to_string();
    let value: Value = match crate::value::from_str(request_json) {
        Ok(v) => v,
        Err(e) => return err("invalid_json", &format!("request is not valid JSON: {e}")),
    };
    let Some(env_id) = value.get("envelopeId").and_then(Value::as_str) else {
        return err("invalid_request", "request.envelopeId must be a string");
    };
    let Some(nodes) = value.get("nodes").and_then(Value::as_array) else {
        return err("invalid_request", "request.nodes must be an array");
    };
    // The envelope alone at the origin: its own transform leaves the local geometry as it is.
    let nodes: Vec<Value> = nodes
        .iter()
        .map(|n| {
            if n.get("id").and_then(Value::as_str) == Some(env_id) {
                let mut n = n.clone();
                if let Some(o) = n.as_object_mut() {
                    for (k, v) in [("x", 0.0), ("y", 0.0), ("rotation", 0.0), ("scaleX", 1.0), ("scaleY", 1.0), ("skewX", 0.0)] {
                        o.insert(k.into(), json!(v));
                    }
                    o.remove("parentId");
                }
                n
            } else {
                n.clone()
            }
        })
        .collect();
    let mut doc = json!({
        "nodes": nodes,
        "pages": [{"id": "__envelope", "width": 1, "height": 1, "nodeIds": [env_id]}],
    });
    // The document's brushes: a brushed stroke inside bends as its brush draws it.
    if let Some(b) = value.get("brushes").filter(|b| b.is_array()) {
        doc["brushes"] = b.clone();
    }
    let scene = match Scene::load(&doc) {
        Ok(scene) => scene,
        Err(e) => return err("invalid_request", &e.to_string()),
    };
    let Some(node) = scene.node(env_id) else {
        return err("invalid_request", "request.nodes has no node with the envelope's id");
    };
    if !is_envelope(node) {
        return json!({"status": "none"}).to_string();
    }
    let px = value.get("pxPerUnit").and_then(Value::as_f64).filter(|v| v.is_finite() && *v > 0.0);
    let r = match px {
        // The scene resolved at the default tolerance; re-resolve at the one asked for.
        Some(px) => resolve_with(&scene, env_id, tolerance_for_scale(px)),
        None => resolve(&scene, env_id),
    };
    let b = |b: Option<Bounds>| b.map(|b| json!({"minX": b.min.x, "minY": b.min.y, "maxX": b.max.x, "maxY": b.max.y}));
    let mut by_parent: HashMap<&str, Vec<&RawNode>> = HashMap::new();
    for (p, n) in &r.transients {
        by_parent.entry(p.as_str()).or_default().push(n);
    }
    fn tree(id: &str, by_parent: &HashMap<&str, Vec<&RawNode>>) -> Vec<Value> {
        let Some(kids) = by_parent.get(id) else {
            return Vec::new();
        };
        let order: Vec<&RawNode> = kids.to_vec();
        order
            .iter()
            .map(|n| {
                let mut item = Map::new();
                item.insert("node".into(), Value::Object(n.to_json_map()));
                if n.node_type() == "group" {
                    // The group's own `children` gives the order.
                    let ids: Vec<&str> = n.array("children").map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                    let sub = tree(n.id(), by_parent);
                    let mut ordered = Vec::with_capacity(sub.len());
                    for cid in ids {
                        if let Some(v) = sub.iter().find(|v| v["node"]["id"] == json!(cid)) {
                            ordered.push(v.clone());
                        }
                    }
                    item.insert("children".into(), Value::Array(ordered));
                }
                Value::Object(item)
            })
            .collect()
    }
    json!({
        "status": "ok",
        "bounds": b(r.bounds),
        "contentBounds": b(r.content_bounds),
        "items": tree(env_id, &by_parent),
        "pending": r.pending,
        "skipped": r.skipped,
    })
    .to_string()
}

#[cfg(test)]
#[path = "envelope_tests.rs"]
mod tests;
