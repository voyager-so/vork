//! Region fill: a static Live Paint (illustrator-gap-audit.md §21, E8.19).
//!
//! Given line art (paths, open or closed, stroked or not) and a point, find the
//! enclosed region at that point: the area the lines' outlines bound where they
//! cross, with gaps up to a tolerance closed. The region comes back as a new
//! closed path for the caller to place beneath the line art. It is static: it
//! does not follow the lines if they move later (Illustrator's Live Paint
//! groups do; this is their "expanded" result, which is what agents and the
//! Paint Bucket want).
//!
//! What bounds a region:
//! - a **stroked** path counts by its stroke outline (its ink), built in the
//!   path's own space and mapped to canvas space by its `transform`, so caps,
//!   joins, the miter limit, inside / outside alignment and non-uniform scales
//!   are the drawn ones. A plain stroke is outlined by `i_overlay`'s stroker on
//!   the flattened centre line; a variable-width, tapered, wavy, dynamic or
//!   calligraphic one by the stroke outliner (`stroke_outline`). Dashes are
//!   ignored: a dashed line still bounds regions, as in Illustrator;
//! - an **unstroked closed** path counts by its outline, as a zero-width edge
//!   (its fill rule decides inside and outside);
//! - an unstroked open path draws nothing and bounds nothing (`ignored`).
//!
//! **Gaps** are closed geometrically, never by rasterising. Every open end of
//! a stroked path looks for other line art within `gap` of its ink: the
//! nearest point of each other path (and of its own path, far enough along it
//! that it isn't the end's own neighbourhood), measured from the end of its
//! ink to the other ink's edge (a butt cap ends at the end point; round and
//! square caps reach a half width beyond). Each one within `gap` gets a bridge:
//! an invisible capsule from the end to that point, as wide as the thinner of
//! the two strokes, which then bounds regions like ink. A gap wider than `gap`
//! stays open, and the area that leaks out through it is not enclosed.
//!
//! The arrangement is solved on the polygon lane (`i_overlay`'s integer
//! booleans, as `overlay.rs`): the free space is a frame minus every ink and
//! bridge, split by the zero-width edges; its connected faces are the regions,
//! and a face touching the frame is outside. For one point the frame starts as
//! a window around it that grows until the face there closes inside it (only
//! the line art near a small region is ever outlined), and ends as a frame
//! around everything. The picked face is grown by `overlap` under the
//! surrounding ink (never past it: the growth is clipped to the face plus the
//! ink), so a fill placed beneath the lines has no antialiasing seam, then
//! refitted to cubics within `FIT_TOLERANCE`. Curves are flattened within
//! `FLATTEN_TOLERANCE`, so a region is within 0.1 path units of the exact ink
//! edge.
//!
//! Contract:
//!
//! ```jsonc
//! // request
//! { "op": "at" | "regions",
//!   "sources": [ { "points": PathPoint[],        // the path, in its own space
//!                  "closed"?: false, "fillRule"?: "nonzero" | "evenodd",
//!                  "cornerRadius"?, "strokes"?: [Stroke], "inkChannels"?, "brush"?,
//!                  "transform"?: [a, b, c, d, e, f] } ],   // own space → canvas
//!   "at": [x, y],                              // op "at", canvas space
//!   "within"?: [minX, minY, maxX, maxY],       // op "at": give up (outside) past this box
//!   "gap"?: 0,                                 // px; ≤ 100
//!   "overlap"?: 0.5 }                          // px the region reaches under the ink; ≤ 4
//! // response, at
//! { "status": "ok", "region": Region,
//!   "bridges": [[x0, y0, x1, y1], …],          // the gaps closed around the region
//!   "ignored": [3] }                           // sources that bound nothing
//! { "status": "empty", "reason": "outside" | "on_line" | "no_lines", "bridges", "ignored" }
//! // (outside: not enclosed, or with `within`, not enclosed inside that box)
//! // response, regions (every enclosed region, in reading order: top to bottom, then left to right)
//! { "status": "ok", "regions": [Region], "bridges", "ignored" }
//! { "status": "empty", "reason": "no_regions" | "no_lines", "bridges", "ignored" }
//! // Region
//! { "points", "fillRule": "evenodd", "closed": true, "subpathCount", "bounds": {minX, minY, maxX, maxY},
//!   "sample": {x, y},     // a point well inside it
//!   "area" }
//! { "status": "error", "code", "message", "sourceIndex"? }
//! ```

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};

use i_overlay::core::fill_rule::FillRule as OverlayFill;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::simplify::SimplifyShape;
use i_overlay::float::single::SingleFloatOverlay;
use i_overlay::mesh::float::outline::offset::OutlineOffset;
use i_overlay::mesh::float::stroke::offset::StrokeOffset;
use i_overlay::mesh::float::style::{LineCap, LineJoin, OutlineStyle, StrokeStyle};
use crate::value::{json, Map, Value};

use super::convert::points_to_subpaths;
use super::overlay::{FIT_TOLERANCE, FLATTEN_TOLERANCE};
use super::solve::err;
use super::{BooleanError, Bounds, FillRule, PathPoint};
use crate::blend::flatten_path_to_polyline;
use crate::path_geom::fit::{fit_polyline, FitSeg};
use crate::path_geom::vec2::{v2, V2};
use crate::stroke_outline::{stroke_outline, Outline, OutlineOptions};
use crate::types::{Affine, RawNode};

/// Most sources one request may carry.
pub const MAX_SOURCES: usize = 20_000;
/// Largest `gap` (px).
pub const MAX_GAP: f64 = 100.0;
/// Largest `overlap` (px).
pub const MAX_OVERLAP: f64 = 4.0;
/// Default `overlap` (px): enough to hide the antialiasing seam between a fill
/// and the stroke over it at 1×.
pub const DEFAULT_OVERLAP: f64 = 0.5;
/// Budget on the flattened ink and edges of one arrangement.
const MAX_FLAT_VERTICES: usize = 4_000_000;
/// Budget on bridges (a pathological scribble of ends).
const MAX_BRIDGES: usize = 20_000;
/// Chords per curve are capped, so a huge curve cannot allocate without bound.
const MAX_CHORDS_PER_CURVE: usize = 512;
/// The outliner's polylines are thinned to within this (px) of themselves.
const DECIMATE: f64 = 0.01;
/// Narrowest bridge (half width, px), so a hairline's bridge still seals.
const MIN_BRIDGE_HALF: f64 = 0.25;
/// Bridges one open end may make (to different paths), nearest first.
const MAX_BRIDGES_PER_END: usize = 4;
/// A face narrower on average than this (area / perimeter, px) is a sliver
/// between two edges that coincide, not a region.
const SLIVER: f64 = 0.02;
/// A vertex that turns more than this (radians) is a corner in the refit.
const CORNER_TURN: f64 = 0.5;
/// Largest coordinate accepted.
const MAX_COORD: f64 = 1e7;
/// Half the side of the first window around a point (px).
const FIRST_WINDOW: f64 = 200.0;

type Contour = Vec<[f64; 2]>;
type Contours = Vec<Contour>;
/// One connected face: its outer contour, then its holes.
type Shape = Vec<Contour>;
type Box4 = [f64; 4];

/// How a source's ink is built.
#[derive(Debug, Clone)]
enum Ink {
    /// A plain stroke: `i_overlay`'s stroker on the flattened centre line.
    Plain { width: f64, cap: String, join: String, miter_limit: f64, align: String },
    /// A variable stroke: the stroke outliner.
    Outline { node: RawNode, brush: Option<Value> },
}

/// One piece of line art. Its ink is built on first use.
#[derive(Debug)]
pub struct RegionSource {
    /// The path in its own space.
    local: Vec<PathPoint>,
    affine: Option<Affine>,
    /// Length scale of the affine (√|det|).
    scale: f64,
    /// The centre line in canvas space, one polyline per subpath.
    lines: Vec<Vec<V2>>,
    closed: bool,
    fill_rule: FillRule,
    ink: Option<Ink>,
    /// Largest half width of the stroke in canvas px (0 when unstroked).
    half_width: f64,
    /// How far the ink reaches past an open end: the half width for round and
    /// square caps, 0 for butt.
    cap_extent: f64,
    /// Canvas bounds of everything it can draw.
    reach: Box4,
    /// Its ink (or edge), resolved: outers counter-clockwise, holes clockwise.
    built: OnceCell<Result<Contours, String>>,
}

impl RegionSource {
    fn is_edge(&self) -> bool {
        self.ink.is_none() && self.closed
    }

    fn bounds_nothing(&self) -> bool {
        self.ink.is_none() && !self.closed
    }

    /// Its ink (or, for an edge, its resolved outline), in canvas space.
    fn contours(&self) -> Result<&Contours, String> {
        self.built.get_or_init(|| self.build()).as_ref().map_err(Clone::clone)
    }

    fn build(&self) -> Result<Contours, String> {
        let to_canvas = |rings: Contours| -> Contours {
            match self.affine {
                Some(m) => rings
                    .into_iter()
                    .map(|r| r.into_iter().map(|p| [m.a * p[0] + m.c * p[1] + m.e, m.b * p[0] + m.d * p[1] + m.f]).collect())
                    .collect(),
                None => rings,
            }
        };
        let resolve = |rings: Contours, rule: OverlayFill| -> Contours {
            if rings.is_empty() {
                return rings;
            }
            flat(rings.simplify_shape(rule))
        };
        match &self.ink {
            None if self.closed => {
                // Mapped before resolving, so a mirroring transform can't flip
                // the winding the arrangement reads.
                let rings = to_canvas(flatten_closed(&self.local, FLATTEN_TOLERANCE / self.scale));
                Ok(resolve(rings, overlay_fill(self.fill_rule)))
            }
            None => Ok(Vec::new()),
            Some(Ink::Plain { width, cap, join, miter_limit, align }) => {
                let tol = FLATTEN_TOLERANCE / self.scale;
                let half = width / 2.0;
                // An arc's chord step within the tolerance, as `L / R`.
                let step = (2.0 * (1.0 - (tol / half.max(tol)).min(1.0)).acos()).clamp(0.02, 1.0);
                let cap = match cap.as_str() {
                    "round" => LineCap::Round(step),
                    "square" => LineCap::Square,
                    _ => LineCap::Butt,
                };
                let join = match join.as_str() {
                    "round" => LineJoin::Round(step),
                    "bevel" => LineJoin::Bevel,
                    // SVG's limit L keeps the miter while 1 / sin(θ / 2) ≤ L.
                    _ => LineJoin::Miter(2.0 * (1.0 / miter_limit.max(1.0)).asin()),
                };
                let paths = local_polylines(&self.local, tol);
                // Inside / outside alignment on a closed path: the stroke is
                // twice as wide, cut to the fill region or away from it.
                let aligned = self.closed && (align == "inside" || align == "outside");
                let w = if aligned { 2.0 * width } else { *width };
                let style = StrokeStyle::new(w).start_cap(cap.clone()).end_cap(cap).line_join(join);
                let mut rings: Contours = Vec::new();
                for path in &paths {
                    if path.len() >= 2 {
                        rings.extend(flat(path.stroke(style.clone(), self.closed)));
                    }
                }
                if aligned && !rings.is_empty() {
                    let fill = flatten_closed(&self.local, tol);
                    let rule = if align == "inside" { OverlayRule::Intersect } else { OverlayRule::Difference };
                    rings = flat(rings.overlay(&fill, rule, OverlayFill::NonZero));
                }
                let mut rings = to_canvas(rings);
                if paths.len() > 1 || aligned {
                    // Subpaths' strokes may overlap: one resolved piece.
                    Ok(resolve(rings, OverlayFill::NonZero))
                } else {
                    // The stroker's answer is resolved already; a mirroring
                    // transform turns it inside out, so wind it back.
                    if self.affine.is_some_and(|m| m.a * m.d - m.b * m.c < 0.0) {
                        rings.iter_mut().for_each(|r| r.reverse());
                    }
                    Ok(rings)
                }
            }
            Some(Ink::Outline { node, brush }) => {
                // Polylines (tolerance 0): the outliner's own samples (≤ ~4° of
                // turn apart), unfitted, thinned within DECIMATE.
                let opts = OutlineOptions { tolerance: 0.0, ..OutlineOptions::default() };
                match stroke_outline(node, brush.as_ref(), &opts).map_err(|e| e.message)? {
                    Outline::Shape { points, .. } => {
                        let rings: Contours = flatten_closed(&points, FLATTEN_TOLERANCE / self.scale)
                            .iter()
                            .map(|r| decimate(r, DECIMATE / self.scale))
                            .filter(|r| r.len() >= 3)
                            .collect();
                        Ok(resolve(to_canvas(rings), OverlayFill::NonZero))
                    }
                    Outline::Empty => Ok(Vec::new()),
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct FilledRegion {
    pub points: Vec<PathPoint>,
    pub subpath_count: usize,
    pub bounds: Bounds,
    pub sample: (f64, f64),
    pub area: f64,
}

/// A gap closed: the bridge's centre line.
pub type BridgeLine = [f64; 4];

#[derive(Debug, Clone)]
pub enum RegionAt {
    Region(FilledRegion),
    /// The point is outside every enclosed region.
    Outside,
    /// The point is on a line (its ink, or a bridge).
    OnLine,
    /// Nothing bounds any region.
    NoLines,
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

fn parse_affine(v: Option<&Value>) -> Result<Option<Affine>, String> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let arr = v.as_array().filter(|a| a.len() == 6).ok_or("transform must be [a, b, c, d, e, f]")?;
    let n: Vec<f64> = arr.iter().filter_map(Value::as_f64).filter(|x| x.is_finite()).collect();
    if n.len() != 6 {
        return Err("transform must be six finite numbers".into());
    }
    Ok(Some(Affine { a: n[0], b: n[1], c: n[2], d: n[3], e: n[4], f: n[5] }))
}

fn parse_points(v: Option<&Value>, what: &str) -> Result<Vec<PathPoint>, String> {
    let arr = v.and_then(Value::as_array).ok_or_else(|| format!("{what} must be an array"))?;
    arr.iter()
        .map(|p| PathPoint::from_json(p).ok_or_else(|| format!("{what} contains a malformed PathPoint")))
        .collect()
}

/// The subpaths of `points` as canvas polylines, flattened within
/// `FLATTEN_TOLERANCE` (the gap measure reads them).
fn centrelines(points: &[PathPoint], m: Option<Affine>, scale: f64) -> Vec<Vec<V2>> {
    local_polylines(points, FLATTEN_TOLERANCE / scale)
        .into_iter()
        .map(|line| {
            line.into_iter()
                .map(|[x, y]| match m {
                    Some(m) => v2(m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f),
                    None => v2(x, y),
                })
                .collect()
        })
        .collect()
}

fn has(v: &Option<Value>) -> bool {
    v.as_ref().is_some_and(|v| !v.is_null())
}

/// One source from its request object (its ink is built on first use).
pub fn parse_source(o: &Map) -> Result<RegionSource, String> {
    let local = parse_points(o.get("points"), "points")?;
    if local.len() > crate::stroke_outline::MAX_POINTS {
        return Err("at most 100000 points per source".into());
    }
    let closed = o.get("closed").and_then(Value::as_bool).unwrap_or(false);
    let fill_rule = match o.get("fillRule").and_then(Value::as_str) {
        None => FillRule::NonZero,
        Some(s) => FillRule::from_token(s).ok_or("fillRule must be nonzero or evenodd")?,
    };
    let affine = parse_affine(o.get("transform"))?;
    // Lengths scale by the transform's area scale (exact for a similarity).
    let scale = affine.map_or(1.0, |m| (m.a * m.d - m.b * m.c).abs().sqrt());
    if !(scale.is_finite() && scale > 1e-9) {
        return Err("transform is singular".into());
    }
    let lines = centrelines(&local, affine, scale);
    if lines.iter().flatten().any(|p| !(p.is_finite() && p.x.abs() <= MAX_COORD && p.y.abs() <= MAX_COORD)) {
        return Err("points must be finite and within ±1e7 on the canvas".into());
    }

    let mut node = Map::new();
    node.insert("type".into(), json!("path"));
    node.insert("points".into(), Value::Array(local.iter().map(PathPoint::to_json).collect()));
    node.insert("closed".into(), json!(closed));
    for k in ["cornerRadius", "strokes", "inkChannels"] {
        if let Some(v) = o.get(k) {
            node.insert(k.into(), v.clone());
        }
    }
    let raw = RawNode::new(node);
    let stroke = crate::paint::model_stroke(&raw).filter(|s| s.width.is_finite() && s.width > 0.0);
    let brush = o.get("brush").filter(|b| b.is_object()).cloned();
    let (ink, half_width, cap_reaches) = match stroke {
        None => (None, 0.0, false),
        Some(s) => {
            let variable = has(&s.profile)
                || has(&s.taper)
                || has(&s.wave)
                || has(&s.dynamic)
                || brush.is_some()
                || raw.json_field("inkChannels").is_some_and(|v| !v.is_null());
            let cap = s.cap.clone().unwrap_or_else(|| "butt".into());
            let reaches = cap == "round" || cap == "square";
            if variable {
                // A variable stroke can be wider than its weight: the
                // outliner's largest half width decides its reach.
                let opts = OutlineOptions { tolerance: 0.0, ..OutlineOptions::default() };
                let hw = match stroke_outline(&raw, brush.as_ref(), &opts) {
                    Ok(Outline::Shape { max_half_width, .. }) => max_half_width,
                    _ => s.width / 2.0,
                };
                (Some(Ink::Outline { node: raw, brush }), hw * scale, reaches)
            } else {
                let align = s.align.clone().unwrap_or_else(|| "center".into());
                let aligned = closed && (align == "inside" || align == "outside");
                let ink = Ink::Plain {
                    width: s.width,
                    cap,
                    join: s.join.clone().unwrap_or_else(|| "miter".into()),
                    miter_limit: s.miter_limit.filter(|m| m.is_finite()).unwrap_or(4.0).clamp(1.0, 500.0),
                    align,
                };
                (Some(ink), s.width / 2.0 * scale * if aligned { 2.0 } else { 1.0 }, reaches)
            }
        }
    };
    // Reach: the centre line's box, padded by the widest the ink can be (a
    // miter can reach the limit times the half width).
    let mut reach = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in lines.iter().flatten() {
        reach = [reach[0].min(p.x), reach[1].min(p.y), reach[2].max(p.x), reach[3].max(p.y)];
    }
    let miter = match &ink {
        Some(Ink::Plain { miter_limit, .. }) => *miter_limit,
        _ => 4.0,
    };
    let pad = half_width * miter.max(2.0) + 1.0;
    let reach = [reach[0] - pad, reach[1] - pad, reach[2] + pad, reach[3] + pad];
    Ok(RegionSource {
        local,
        affine,
        scale,
        lines,
        closed,
        fill_rule,
        cap_extent: if cap_reaches { half_width } else { 0.0 },
        half_width,
        ink,
        reach,
        built: OnceCell::new(),
    })
}

// ---------------------------------------------------------------------------
// Flattening
// ---------------------------------------------------------------------------

type P2 = (f64, f64);

/// Chords that keep a cubic within `tol`: n equal-t chords are within
/// (3/4)·L/n² of it, L the larger second difference of its control polygon.
fn cubic_chords(p0: P2, c1: P2, c2: P2, p3: P2, tol: f64) -> usize {
    let l = (p0.0 - 2.0 * c1.0 + c2.0).hypot(p0.1 - 2.0 * c1.1 + c2.1).max((c1.0 - 2.0 * c2.0 + p3.0).hypot(c1.1 - 2.0 * c2.1 + p3.1));
    ((0.75 * l / tol).sqrt().ceil() as usize).clamp(1, MAX_CHORDS_PER_CURVE)
}

fn cubic_at(p0: P2, c1: P2, c2: P2, p3: P2, t: f64) -> [f64; 2] {
    let mt = 1.0 - t;
    let (a, b, c, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
    [a * p0.0 + b * c1.0 + c * c2.0 + d * p3.0, a * p0.1 + b * c1.1 + c * c2.1 + d * p3.1]
}

/// Closed subpaths of `points` flattened within `tol`.
fn flatten_closed(points: &[PathPoint], tol: f64) -> Contours {
    points_to_subpaths(points, 1e-12)
        .iter()
        .map(|(start, segs)| {
            let mut out = vec![[start.0, start.1]];
            let mut p0 = (start.0, start.1);
            for (c1, c2, p3) in segs {
                let (c1, c2, p3) = ((c1.0, c1.1), (c2.0, c2.1), (p3.0, p3.1));
                let n = cubic_chords(p0, c1, c2, p3, tol);
                for i in 1..n {
                    out.push(cubic_at(p0, c1, c2, p3, i as f64 / n as f64));
                }
                out.push([p3.0, p3.1]);
                p0 = p3;
            }
            if out.len() > 1 && out[0] == out[out.len() - 1] {
                out.pop();
            }
            out
        })
        .filter(|c| c.len() >= 3)
        .collect()
}

/// Every subpath of `points` (open) flattened within `tol`, for the stroker.
fn local_polylines(points: &[PathPoint], tol: f64) -> Contours {
    let mut out: Contours = Vec::new();
    let mut cur: Contour = Vec::new();
    let mut at: P2 = (0.0, 0.0);
    let push = |c: &mut Contour, p: [f64; 2]| {
        if c.last().is_none_or(|q| (q[0] - p[0]).hypot(q[1] - p[1]) > 1e-12) {
            c.push(p);
        }
    };
    let curve = |cur: &mut Contour, at: P2, c1: P2, c2: P2, end: P2| {
        let n = cubic_chords(at, c1, c2, end, tol);
        for i in 1..=n {
            push(cur, cubic_at(at, c1, c2, end, i as f64 / n as f64));
        }
    };
    for p in points {
        match p {
            PathPoint::Move { x, y } => {
                if cur.len() >= 2 {
                    out.push(std::mem::take(&mut cur));
                }
                cur.clear();
                at = (*x, *y);
                cur.push([*x, *y]);
            }
            PathPoint::Line { x, y } => {
                at = (*x, *y);
                push(&mut cur, [*x, *y]);
            }
            PathPoint::Curve { x, y, cp1, cp2 } => {
                curve(&mut cur, at, *cp1, *cp2, (*x, *y));
                at = (*x, *y);
            }
            PathPoint::Quadratic { x, y, cp } => {
                let c1 = (at.0 + 2.0 / 3.0 * (cp.0 - at.0), at.1 + 2.0 / 3.0 * (cp.1 - at.1));
                let c2 = (*x + 2.0 / 3.0 * (cp.0 - *x), *y + 2.0 / 3.0 * (cp.1 - *y));
                curve(&mut cur, at, c1, c2, (*x, *y));
                at = (*x, *y);
            }
            PathPoint::Arc { x, y, .. } => {
                // Arcs through the shared flattener (8 chords per quadrant).
                let seg = [PathPoint::Move { x: at.0, y: at.1 }, p.clone()];
                for (px, py) in flatten_path_to_polyline(&seg, false).into_iter().skip(1) {
                    push(&mut cur, [px, py]);
                }
                at = (*x, *y);
            }
        }
    }
    if cur.len() >= 2 {
        out.push(cur);
    }
    out
}

/// A closed ring thinned by Douglas–Peucker to within `tol` of itself.
fn decimate(ring: &Contour, tol: f64) -> Contour {
    let n = ring.len();
    if n <= 8 {
        return ring.clone();
    }
    // Split the loop at its first vertex and the vertex farthest from it.
    let d0 = |k: usize| (ring[k][0] - ring[0][0]).hypot(ring[k][1] - ring[0][1]);
    let far = (1..n).max_by(|&i, &j| d0(i).total_cmp(&d0(j))).unwrap_or(n / 2);
    let mut keep = vec![false; n + 1];
    keep[0] = true;
    keep[far] = true;
    keep[n] = true;
    let at = |i: usize| ring[i % n];
    let seg = |p: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        (p[0] - a[0] - dx * t).hypot(p[1] - a[1] - dy * t)
    };
    let mut stack = vec![(0usize, far), (far, n)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (mut worst, mut idx) = (0.0, a);
        for i in a + 1..b {
            let d = seg(at(i), at(a), at(b));
            if d > worst {
                worst = d;
                idx = i;
            }
        }
        if worst > tol {
            keep[idx] = true;
            stack.push((a, idx));
            stack.push((idx, b));
        }
    }
    (0..n).filter(|&i| keep[i]).map(|i| ring[i]).collect()
}

fn overlay_fill(rule: FillRule) -> OverlayFill {
    match rule {
        FillRule::NonZero => OverlayFill::NonZero,
        FillRule::EvenOdd => OverlayFill::EvenOdd,
    }
}

fn flat(shapes: Vec<Shape>) -> Contours {
    shapes.into_iter().flatten().filter(|c| c.len() >= 3).collect()
}

/// A capsule from `a` to `b` of half width `r`.
fn capsule(a: V2, b: V2, r: f64) -> Contour {
    let d = (b - a).unit().unwrap_or(v2(1.0, 0.0));
    let n = d.left_normal();
    const HALF: usize = 8;
    let mut out = Vec::with_capacity(2 * HALF + 2);
    // Around b from +n to -n through +d, then around a from -n to +n through -d.
    for k in 0..=HALF {
        let t = std::f64::consts::PI * k as f64 / HALF as f64;
        let p = b + (n * t.cos() + d * t.sin()) * r;
        out.push([p.x, p.y]);
    }
    for k in 0..=HALF {
        let t = std::f64::consts::PI * k as f64 / HALF as f64;
        let p = a + (-n * t.cos() - d * t.sin()) * r;
        out.push([p.x, p.y]);
    }
    out
}

// ---------------------------------------------------------------------------
// Gap bridges
// ---------------------------------------------------------------------------

struct Line<'a> {
    pts: &'a [V2],
    /// Cumulative arc length at each vertex.
    cum: Vec<f64>,
    closed: bool,
    half_width: f64,
    cap_extent: f64,
    inked: bool,
}

/// The nearest point to `p` on segment `a`–`b`: (point, t).
fn nearest_on(a: V2, b: V2, p: V2) -> (V2, f64) {
    let ab = b - a;
    let l2 = ab.len2();
    let t = if l2 <= 1e-18 { 0.0 } else { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) };
    (a + ab * t, t)
}

/// The gaps up to `gap` between the open ends of `sources[used]` and the line
/// art near them: `(end, target point, bridge half width)`.
fn bridges(sources: &[RegionSource], used: &[usize], gap: f64) -> Result<Vec<(V2, V2, f64)>, BooleanError> {
    let mut lines: Vec<Line> = Vec::new();
    for &i in used {
        let s = &sources[i];
        if s.bounds_nothing() {
            continue;
        }
        let inked = s.ink.is_some();
        for pts in &s.lines {
            if pts.len() < 2 {
                continue;
            }
            let mut cum = Vec::with_capacity(pts.len());
            let mut acc = 0.0;
            cum.push(0.0);
            for w in pts.windows(2) {
                acc += w[0].dist(w[1]);
                cum.push(acc);
            }
            lines.push(Line {
                pts,
                cum,
                closed: s.closed,
                half_width: if inked { s.half_width } else { 0.0 },
                cap_extent: if inked { s.cap_extent } else { 0.0 },
                inked,
            });
        }
    }
    let max_hw = lines.iter().map(|l| l.half_width).fold(0.0, f64::max);
    // Segments in a uniform grid; every cell a segment passes through lists
    // it (sampled at a quarter cell, so a query's 5 × 5 neighbourhood finds
    // any segment within one cell of it).
    let cell = (gap + 2.0 * max_hw).max(1.0);
    let key = |p: V2| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<(usize, usize)>> = HashMap::new();
    for (li, l) in lines.iter().enumerate() {
        let n = l.pts.len();
        let segs = if l.closed { n } else { n - 1 };
        for si in 0..segs {
            let (a, b) = (l.pts[si], l.pts[(si + 1) % n]);
            let steps = ((a.dist(b) / (cell / 4.0)).ceil() as usize).clamp(1, 1 << 16);
            let mut last = None;
            for k in 0..=steps {
                let c = key(a.lerp(b, k as f64 / steps as f64));
                if last != Some(c) {
                    grid.entry(c).or_default().push((li, si));
                    last = Some(c);
                }
            }
        }
    }

    let mut out = Vec::new();
    for (li, l) in lines.iter().enumerate() {
        if l.closed || !l.inked {
            continue;
        }
        let n = l.pts.len();
        for (e, e_arc) in [(l.pts[0], 0.0), (l.pts[n - 1], l.cum[n - 1])] {
            // The nearest point of each line.
            let mut best: HashMap<usize, (f64, V2, f64)> = HashMap::new();
            let (kx, ky) = key(e);
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            for gx in kx - 2..=kx + 2 {
                for gy in ky - 2..=ky + 2 {
                    let Some(cands) = grid.get(&(gx, gy)) else { continue };
                    for &(tj, si) in cands {
                        if !seen.insert((tj, si)) {
                            continue;
                        }
                        let t = &lines[tj];
                        let m = t.pts.len();
                        let (a, b) = (t.pts[si], t.pts[(si + 1) % m]);
                        let (q, u) = nearest_on(a, b, e);
                        let d = q.dist(e);
                        if tj == li {
                            // Its own line: only far enough along it that the
                            // point is not the end's own neighbourhood.
                            let q_arc = t.cum[si] + (t.cum[(si + 1).min(m - 1)] - t.cum[si]) * u;
                            if (q_arc - e_arc).abs() <= 2.0 * d + 4.0 * l.half_width + 1e-6 {
                                continue;
                            }
                        }
                        // How far the target's ink reaches from its centre line
                        // here: past an open end only its cap, else its half width.
                        let at_end = !t.closed && ((si == 0 && u <= 1e-9) || (si + 2 == m && u >= 1.0 - 1e-9));
                        let reach_t = if at_end { t.cap_extent } else { t.half_width };
                        let visible = d - reach_t - l.cap_extent;
                        if d <= 1e-9 || visible > gap {
                            continue;
                        }
                        let entry = best.entry(tj).or_insert((f64::INFINITY, q, 0.0));
                        if d < entry.0 {
                            *entry = (d, q, visible);
                        }
                    }
                }
            }
            let mut found: Vec<(usize, (f64, V2, f64))> = best.into_iter().collect();
            found.sort_by(|a, b| a.1 .0.total_cmp(&b.1 .0).then(a.0.cmp(&b.0)));
            for (tj, (_, q, visible)) in found.into_iter().take(MAX_BRIDGES_PER_END) {
                // Already touching: the inks meet, nothing to close.
                if visible <= 0.0 {
                    continue;
                }
                let t = &lines[tj];
                let hw_t = if t.half_width > 0.0 { t.half_width } else { l.half_width };
                let r = l.half_width.min(hw_t).max(MIN_BRIDGE_HALF);
                out.push((e, q, r));
                if out.len() > MAX_BRIDGES {
                    return Err(err("too_complex", format!("more than {MAX_BRIDGES} gaps to close")));
                }
            }
        }
    }
    // Deterministic order (the maps above are not).
    out.sort_by(|a, b| (a.0.x, a.0.y, a.1.x, a.1.y).partial_cmp(&(b.0.x, b.0.y, b.1.x, b.1.y)).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Polygon helpers
// ---------------------------------------------------------------------------

fn contour_bounds(c: &Contour) -> Box4 {
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in c {
        b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
    }
    b
}

fn bounds_of(rings: &Contours) -> Box4 {
    rings.iter().map(contour_bounds).fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |a, c| {
        [a[0].min(c[0]), a[1].min(c[1]), a[2].max(c[2]), a[3].max(c[3])]
    })
}

fn signed_area(c: &Contour) -> f64 {
    let n = c.len();
    (0..n).map(|i| c[i][0] * c[(i + 1) % n][1] - c[(i + 1) % n][0] * c[i][1]).sum::<f64>() / 2.0
}

fn perimeter(c: &Contour) -> f64 {
    let n = c.len();
    (0..n).map(|i| (c[(i + 1) % n][0] - c[i][0]).hypot(c[(i + 1) % n][1] - c[i][1])).sum()
}

/// Even-odd containment over every contour of `rings`.
fn contains(rings: &[Contour], p: (f64, f64)) -> bool {
    let mut inside = false;
    for ring in rings {
        let n = ring.len();
        let mut j = n - 1;
        for i in 0..n {
            let (a, b) = (ring[i], ring[j]);
            if (a[1] > p.1) != (b[1] > p.1) && p.0 < (b[0] - a[0]) * (p.1 - a[1]) / (b[1] - a[1]) + a[0] {
                inside = !inside;
            }
            j = i;
        }
    }
    inside
}

/// Sutherland–Hodgman: `ring` cut to the box. Keeps the ring's winding at
/// every point inside the box (the cut may leave zero-area runs along the
/// box's sides, which fill nothing).
fn clip_to_box(ring: &Contour, b: &Box4) -> Contour {
    let mut out = ring.clone();
    // (axis, bound, keep ≥ bound)
    for (axis, bound, above) in [(0usize, b[0], true), (0, b[2], false), (1, b[1], true), (1, b[3], false)] {
        if out.is_empty() {
            break;
        }
        let inside = |p: &[f64; 2]| if above { p[axis] >= bound } else { p[axis] <= bound };
        let cut = |p: &[f64; 2], q: &[f64; 2]| {
            let t = (bound - p[axis]) / (q[axis] - p[axis]);
            let mut r = [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
            r[axis] = bound;
            r
        };
        let input = std::mem::take(&mut out);
        let n = input.len();
        for k in 0..n {
            let (p, q) = (&input[(k + n - 1) % n], &input[k]);
            match (inside(p), inside(q)) {
                (true, true) => out.push(*q),
                (true, false) => out.push(cut(p, q)),
                (false, true) => {
                    out.push(cut(p, q));
                    out.push(*q);
                }
                (false, false) => {}
            }
        }
    }
    out
}

fn overlaps(a: &Box4, b: &Box4) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

fn in_box(b: &Box4, p: (f64, f64)) -> bool {
    p.0 >= b[0] && p.0 <= b[2] && p.1 >= b[1] && p.1 <= b[3]
}

/// Where a horizontal (`axis` 1: vertical) line at `at` crosses `rings`, sorted.
fn crossings(rings: &[Contour], axis: usize, at: f64) -> Vec<f64> {
    let o = 1 - axis;
    let mut xs = Vec::new();
    for ring in rings {
        let n = ring.len();
        for i in 0..n {
            let (p, q) = (ring[i], ring[(i + 1) % n]);
            if (p[axis] <= at) != (q[axis] <= at) {
                xs.push(p[o] + (at - p[axis]) / (q[axis] - p[axis]) * (q[o] - p[o]));
            }
        }
    }
    xs.sort_by(f64::total_cmp);
    xs
}

/// The inside span (even-odd) of the line at `at` along `axis` that holds `v`.
fn span_at(rings: &[Contour], axis: usize, at: f64, v: f64) -> Option<(f64, f64)> {
    crossings(rings, axis, at).chunks_exact(2).map(|p| (p[0], p[1])).find(|&(a, b)| a <= v && v <= b)
}

/// A point well inside `rings` (even-odd), clear of the outline both across and
/// along: on a few scanlines, each span's middle, re-centred on the vertical
/// span through it; the one farthest from the outline in both directions wins.
fn interior_sample(rings: &[Contour], b: &Box4) -> Option<(f64, f64)> {
    let h = b[3] - b[1];
    if !(h > 0.0) {
        return None;
    }
    let mut best: Option<((f64, f64), f64)> = None;
    const LINES: usize = 15;
    for k in 0..LINES {
        let y = b[1] + h * ((k as f64 + 0.5) / LINES as f64 + 0.000_137);
        for pair in crossings(rings, 1, y).chunks_exact(2) {
            if !(pair[1] > pair[0]) {
                continue;
            }
            let x = (pair[0] + pair[1]) / 2.0;
            let Some((top, bottom)) = span_at(rings, 0, x, y) else { continue };
            let cy = (top + bottom) / 2.0;
            let Some((left, right)) = span_at(rings, 1, cy, x) else { continue };
            let clear = ((right - left) / 2.0).min((x - left).min(right - x)).min((bottom - top) / 2.0);
            if best.is_none_or(|(_, s)| clear > s) {
                best = Some(((x, cy), clear));
            }
        }
    }
    best.map(|(p, _)| p)
}

// ---------------------------------------------------------------------------
// The arrangement
// ---------------------------------------------------------------------------

struct Face {
    shape: Shape,
    bounds: Box4,
    area: f64,
}

impl Face {
    fn new(shape: Shape) -> Face {
        let bounds = contour_bounds(&shape[0]);
        let area = signed_area(&shape[0]).abs() - shape[1..].iter().map(|c| signed_area(c).abs()).sum::<f64>();
        Face { shape, bounds, area }
    }
}

/// The solved arrangement inside a frame: every face that closes inside it.
pub struct Arrangement {
    faces: Vec<Face>,
    /// The free space that reaches the frame: outside every region.
    outside: Vec<Face>,
    /// Every ink and bridge, each resolved on its own (outers counter-clockwise,
    /// holes clockwise, so over all of them the winding counts coverage), with
    /// its bounds.
    ink: Vec<(Contours, Box4)>,
    pub bridges: Vec<BridgeLine>,
    overlap: f64,
    has_lines: bool,
}

/// The line art of a request and the options it is solved with.
pub struct Drawing {
    sources: Vec<RegionSource>,
    gap: f64,
    overlap: f64,
    /// Canvas bounds of every source's reach.
    all: Option<Box4>,
}

impl Drawing {
    pub fn new(sources: Vec<RegionSource>, gap: f64, overlap: f64) -> Result<Drawing, BooleanError> {
        if sources.len() > MAX_SOURCES {
            return Err(err("too_complex", format!("at most {MAX_SOURCES} sources are supported")));
        }
        let gap = if gap.is_finite() { gap.clamp(0.0, MAX_GAP) } else { 0.0 };
        let overlap = if overlap.is_finite() { overlap.clamp(0.0, MAX_OVERLAP) } else { DEFAULT_OVERLAP };
        let all = sources
            .iter()
            .filter(|s| !s.bounds_nothing() && !s.lines.is_empty())
            .map(|s| s.reach)
            .reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]);
        Ok(Drawing { sources, gap, overlap, all })
    }

    /// Sources that bound nothing: unstroked open paths, and strokes or shapes
    /// with no area. Builds every source's ink.
    pub fn ignored(&self) -> Vec<usize> {
        (0..self.sources.len())
            .filter(|&i| {
                let s = &self.sources[i];
                s.bounds_nothing() || s.lines.is_empty() || s.contours().map_or(true, |c| c.is_empty())
            })
            .collect()
    }

    /// [`Drawing::ignored`] without building anything new: unstroked open
    /// paths, and built sources with no area.
    pub fn ignored_built(&self) -> Vec<usize> {
        (0..self.sources.len())
            .filter(|&i| {
                let s = &self.sources[i];
                s.bounds_nothing() || s.lines.is_empty() || s.built.get().is_some_and(|c| c.as_ref().map_or(true, |c| c.is_empty()))
            })
            .collect()
    }

    /// The arrangement inside `window` (everything when `None`).
    pub fn arrange(&self, window: Option<Box4>) -> Result<Arrangement, BooleanError> {
        let Some(all) = self.all else {
            return Ok(Arrangement { faces: Vec::new(), outside: Vec::new(), ink: Vec::new(), bridges: Vec::new(), overlap: self.overlap, has_lines: false });
        };
        let m = 8.0 + self.gap;
        let frame = window.unwrap_or([all[0] - m, all[1] - m, all[2] + m, all[3] + m]);
        // Everything that can reach into the frame, bridges included.
        let widest = self.sources.iter().map(|s| s.half_width).fold(0.0, f64::max);
        let pad = self.gap + 2.0 * widest + 1.0;
        let load = [frame[0] - pad, frame[1] - pad, frame[2] + pad, frame[3] + pad];
        let used: Vec<usize> = (0..self.sources.len())
            .filter(|&i| {
                let s = &self.sources[i];
                !s.bounds_nothing() && !s.lines.is_empty() && overlaps(&s.reach, &load)
            })
            .collect();

        let mut pieces: Vec<(Contours, Box4)> = Vec::new();
        let mut edges: Vec<(Contours, Box4)> = Vec::new();
        let mut vertices = 0usize;
        for &i in &used {
            let s = &self.sources[i];
            let rings = s.contours().map_err(|m| BooleanError { code: "invalid_request".into(), message: m, operand_index: Some(i) })?;
            if rings.is_empty() {
                continue;
            }
            vertices += rings.iter().map(Vec::len).sum::<usize>();
            if vertices > MAX_FLAT_VERTICES {
                return Err(err("too_complex", format!("the line art flattens to more than {MAX_FLAT_VERTICES} vertices")));
            }
            let b = bounds_of(rings);
            if s.is_edge() {
                edges.push((rings.clone(), b));
            } else {
                pieces.push((rings.clone(), b));
            }
        }
        let bridge_list = if self.gap > 0.0 { bridges(&self.sources, &used, self.gap)? } else { Vec::new() };
        for &(a, b, r) in &bridge_list {
            let resolved = flat(vec![capsule(a, b, r)].simplify_shape(OverlayFill::NonZero));
            if !resolved.is_empty() {
                let bb = bounds_of(&resolved);
                pieces.push((resolved, bb));
            }
        }
        let bridges: Vec<BridgeLine> = bridge_list.iter().map(|(a, b, _)| [a.x, a.y, b.x, b.y]).collect();
        let has_lines = !pieces.is_empty() || !edges.is_empty();

        // The free space: the frame minus every piece of ink at once (each is
        // resolved and wound positively, so `NonZero` reads their union).
        let frame_ring: Contour = vec![[frame[0], frame[1]], [frame[0], frame[3]], [frame[2], frame[3]], [frame[2], frame[1]]];
        // Each cut to (just past) the frame first: a long line through a small
        // window costs its local run, not its length. The cut keeps every
        // point's winding inside the frame, which is all the difference reads.
        let cut = [frame[0] - 1.0, frame[1] - 1.0, frame[2] + 1.0, frame[3] + 1.0];
        let near: Contours = pieces
            .iter()
            .filter(|(_, b)| overlaps(b, &frame))
            .flat_map(|(c, _)| c.iter())
            .filter(|c| overlaps(&contour_bounds(c), &cut))
            .map(|c| {
                let b = contour_bounds(c);
                if b[0] >= cut[0] && b[1] >= cut[1] && b[2] <= cut[2] && b[3] <= cut[3] {
                    c.clone()
                } else {
                    clip_to_box(c, &cut)
                }
            })
            .filter(|c| c.len() >= 3)
            .collect();
        let mut shapes: Vec<Shape> = if near.is_empty() {
            vec![frame_ring].simplify_shape(OverlayFill::NonZero)
        } else {
            vec![frame_ring].overlay(&near, OverlayRule::Difference, OverlayFill::NonZero)
        };
        // Zero-width edges split the faces they cross.
        for (edge, eb) in &edges {
            let mut next = Vec::with_capacity(shapes.len());
            for shape in shapes {
                let sb = shape.first().map(contour_bounds).unwrap_or([0.0; 4]);
                if !overlaps(&sb, eb) {
                    next.push(shape);
                    continue;
                }
                next.extend(shape.overlay(edge, OverlayRule::Intersect, OverlayFill::NonZero));
                next.extend(shape.overlay(edge, OverlayRule::Difference, OverlayFill::NonZero));
            }
            shapes = next;
        }

        let eps = 1e-6 * (1.0 + frame.iter().fold(0.0f64, |a, v| a.max(v.abs())));
        let mut faces = Vec::new();
        let mut outside = Vec::new();
        for shape in shapes {
            if shape.is_empty() || shape[0].len() < 3 {
                continue;
            }
            let face = Face::new(shape);
            let b = face.bounds;
            // A face that reaches the frame is outside (or not closed yet).
            if b[0] <= frame[0] + eps || b[1] <= frame[1] + eps || b[2] >= frame[2] - eps || b[3] >= frame[3] - eps {
                outside.push(face);
                continue;
            }
            let per: f64 = face.shape.iter().map(perimeter).sum();
            if face.area <= SLIVER * per {
                continue;
            }
            faces.push(face);
        }
        Ok(Arrangement { faces, outside, ink: pieces, bridges, overlap: self.overlap, has_lines })
    }

    /// The region at canvas point `p`: solved in a window around it that grows
    /// until the face there closes inside it, else over everything. Also the
    /// bridges near the answer. `within` caps the window: a region that does
    /// not close inside it answers `Outside` (a hover can't wait for the
    /// whole drawing).
    pub fn at(&self, p: (f64, f64), within: Option<Box4>) -> Result<(RegionAt, Vec<BridgeLine>), BooleanError> {
        let Some(all) = self.all else {
            return Ok((RegionAt::NoLines, Vec::new()));
        };
        let mut half = FIRST_WINDOW.max(4.0 * self.gap);
        loop {
            let mut window = [p.0 - half, p.1 - half, p.0 + half, p.1 + half];
            let capped = within.is_some_and(|w| window[0] <= w[0] && window[1] <= w[1] && window[2] >= w[2] && window[3] >= w[3]);
            if let (true, Some(w)) = (capped, within) {
                window = w;
            }
            let covers = window[0] <= all[0] && window[1] <= all[1] && window[2] >= all[2] && window[3] >= all[3];
            let arr = self.arrange(if covers { None } else { Some(window) })?;
            match arr.at(p) {
                RegionAt::Region(r) => {
                    let near = arr.bridges_near(&r.bounds);
                    return Ok((RegionAt::Region(r), near));
                }
                RegionAt::OnLine => return Ok((RegionAt::OnLine, Vec::new())),
                // Over everything, no lines at all is an answer; in a window
                // it only means none are near yet.
                RegionAt::NoLines if covers => return Ok((RegionAt::NoLines, arr.bridges)),
                _ if covers || capped => return Ok((RegionAt::Outside, arr.bridges)),
                _ => half *= 4.0,
            }
        }
    }
}

impl Arrangement {
    pub fn region_count(&self) -> usize {
        self.faces.len()
    }

    /// The region at canvas point `p`.
    pub fn at(&self, p: (f64, f64)) -> RegionAt {
        if !self.has_lines {
            return RegionAt::NoLines;
        }
        if let Some(i) = self.faces.iter().position(|f| in_box(&f.bounds, p) && contains(&f.shape, p)) {
            return RegionAt::Region(self.region(i));
        }
        if self.ink.iter().any(|(c, b)| in_box(b, p) && contains(c, p)) {
            return RegionAt::OnLine;
        }
        RegionAt::Outside
    }

    /// Every region, in reading order (top to bottom, then left to right).
    pub fn regions(&self) -> Vec<FilledRegion> {
        let mut out: Vec<FilledRegion> = (0..self.faces.len()).map(|i| self.region(i)).collect();
        out.sort_by(|a, b| (a.sample.1.round(), a.sample.0).partial_cmp(&(b.sample.1.round(), b.sample.0)).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Bridges that touch the box `b` (padded by 2 px).
    pub fn bridges_near(&self, b: &Bounds) -> Vec<BridgeLine> {
        let pad = [b.min_x - 2.0, b.min_y - 2.0, b.max_x + 2.0, b.max_y + 2.0];
        self.bridges
            .iter()
            .filter(|l| overlaps(&[l[0].min(l[2]), l[1].min(l[3]), l[0].max(l[2]), l[1].max(l[3])], &pad))
            .copied()
            .collect()
    }

    fn region(&self, i: usize) -> FilledRegion {
        let face = &self.faces[i];
        let center = ((face.bounds[0] + face.bounds[2]) / 2.0, (face.bounds[1] + face.bounds[3]) / 2.0);
        let sample = interior_sample(&face.shape, &face.bounds).unwrap_or(center);
        let rings: Contours = if self.overlap > 0.0 && !self.ink.is_empty() {
            // Grown under the ink, never past it: what the growth reaches of
            // any other free space (a neighbouring region, the outside) is cut
            // away again, so it stays within the face plus the ink.
            let grown = flat(vec![face.shape.clone()].outline(&OutlineStyle::new(self.overlap)));
            let o = self.overlap + 1.0;
            let reach = [face.bounds[0] - o, face.bounds[1] - o, face.bounds[2] + o, face.bounds[3] + o];
            // Each cut to the reach box first (a face's winding inside the box
            // is kept), so a huge neighbour costs its length, not an overlay.
            let others: Contours = self
                .faces
                .iter()
                .enumerate()
                .filter(|&(j, f)| j != i && overlaps(&f.bounds, &reach))
                .map(|(_, f)| f)
                .chain(self.outside.iter().filter(|f| overlaps(&f.bounds, &reach)))
                .flat_map(|f| f.shape.iter())
                .filter(|c| overlaps(&contour_bounds(c), &reach))
                .map(|c| clip_to_box(c, &reach))
                .filter(|c| c.len() >= 3)
                .collect();
            let rings = if others.is_empty() { grown } else { flat(grown.overlay(&others, OverlayRule::Difference, OverlayFill::NonZero)) };
            if rings.is_empty() {
                face.shape.clone()
            } else {
                rings
            }
        } else {
            face.shape.clone()
        };
        let mut points = Vec::new();
        let mut count = 0;
        let mut bounds = Bounds { min_x: f64::INFINITY, min_y: f64::INFINITY, max_x: f64::NEG_INFINITY, max_y: f64::NEG_INFINITY };
        for ring in &rings {
            if refit_ring(ring, &mut points, &mut bounds) {
                count += 1;
            }
        }
        FilledRegion { points, subpath_count: count, bounds, sample, area: face.area }
    }
}

/// One ring refitted to cubics within `FIT_TOLERANCE`, breaking at corners.
fn refit_ring(ring: &Contour, out: &mut Vec<PathPoint>, bounds: &mut Bounds) -> bool {
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let pts: Vec<V2> = ring.iter().map(|p| v2(p[0], p[1])).collect();
    let breaks: Vec<bool> = (0..n)
        .map(|i| {
            let (a, b, c) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
            match ((b - a).unit(), (c - b).unit()) {
                (Some(u), Some(v)) => u.cross(v).atan2(u.dot(v)).abs() > CORNER_TURN,
                _ => true,
            }
        })
        .collect();
    let segs = fit_polyline(&pts, &breaks, true, FIT_TOLERANCE);
    let Some(first) = segs.first() else { return false };
    let start = first.start();
    out.push(PathPoint::Move { x: start.x, y: start.y });
    // The ring's own extent (the fit stays within FIT_TOLERANCE of it; its
    // control points may not).
    for p in &pts {
        bounds.include(p.x, p.y);
    }
    let last = segs.len() - 1;
    for (k, s) in segs.iter().enumerate() {
        match s {
            FitSeg::Line(_, b) => {
                // The closing line back to the start is implied by `closed`.
                if k == last && k >= 2 && b.dist(start) <= 1e-9 {
                    break;
                }
                out.push(PathPoint::Line { x: b.x, y: b.y });
            }
            FitSeg::Cubic(c) => {
                let [_, c1, c2, p] = c.0;
                out.push(PathPoint::Curve { x: p.x, y: p.y, cp1: (c1.x, c1.y), cp2: (c2.x, c2.y) });
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

fn region_json(r: &FilledRegion) -> Value {
    json!({
        "points": r.points.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
        "fillRule": "evenodd",
        "closed": true,
        "subpathCount": r.subpath_count,
        "bounds": { "minX": r.bounds.min_x, "minY": r.bounds.min_y, "maxX": r.bounds.max_x, "maxY": r.bounds.max_y },
        "sample": { "x": r.sample.0, "y": r.sample.1 },
        "area": r.area,
    })
}

fn error_json(e: BooleanError) -> String {
    let mut v = json!({ "status": "error", "code": e.code, "message": e.message });
    if let Some(i) = e.operand_index {
        v["sourceIndex"] = json!(i);
    }
    v.to_string()
}

/// JSON in, JSON out (module doc). Always a valid envelope. Never panics on
/// native (the solve runs under `catch_unwind`).
pub fn region_fill_json(request_json: &str) -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| solve_json(request_json))) {
            Ok(s) => s,
            Err(_) => error_json(err("solver_failed", "the region solver panicked on this input")),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        solve_json(request_json)
    }
}

fn invalid_at(message: impl Into<String>, i: usize) -> BooleanError {
    BooleanError { code: "invalid_request".into(), message: message.into(), operand_index: Some(i) }
}

/// A request's sources and options.
pub fn parse_drawing(obj: &Map) -> Result<Drawing, BooleanError> {
    let num = |k: &str, d: f64| -> Result<f64, BooleanError> {
        match obj.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_f64()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| err("invalid_request", format!("{k} must be a number ≥ 0"))),
        }
    };
    let gap = num("gap", 0.0)?;
    let overlap = num("overlap", DEFAULT_OVERLAP)?;
    let raw = obj.get("sources").and_then(Value::as_array).ok_or_else(|| err("invalid_request", "sources must be an array"))?;
    if raw.len() > MAX_SOURCES {
        return Err(err("too_complex", format!("at most {MAX_SOURCES} sources are supported")));
    }
    let mut sources = Vec::with_capacity(raw.len());
    for (i, s) in raw.iter().enumerate() {
        let o = s.as_object().ok_or_else(|| invalid_at("source must be an object", i))?;
        sources.push(parse_source(o).map_err(|m| invalid_at(m, i))?);
    }
    Drawing::new(sources, gap, overlap)
}

fn solve_json(request_json: &str) -> String {
    if request_json.len() > 256 * 1024 * 1024 {
        return error_json(err("request_too_large", "request exceeds 256 MiB"));
    }
    let value: Value = match crate::value::from_str(request_json) {
        Ok(v) => v,
        Err(e) => return error_json(err("invalid_json", format!("request is not valid JSON: {e}"))),
    };
    let Some(obj) = value.as_object() else {
        return error_json(err("invalid_request", "request must be an object"));
    };
    let op = obj.get("op").and_then(Value::as_str).unwrap_or("");
    if op != "at" && op != "regions" {
        return error_json(err("invalid_request", "op must be at or regions"));
    }
    let at = if op == "at" {
        match obj.get("at").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>()) {
            Some(v) if v.len() == 2 && v.iter().all(|x| x.is_finite()) => Some((v[0], v[1])),
            _ => return error_json(err("invalid_request", "at must be [x, y]")),
        }
    } else {
        None
    };
    let within = match obj.get("within") {
        None | Some(Value::Null) => None,
        Some(v) => match v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>()) {
            Some(b) if b.len() == 4 && b.iter().all(|x| x.is_finite()) && b[0] < b[2] && b[1] < b[3] => Some([b[0], b[1], b[2], b[3]]),
            _ => return error_json(err("invalid_request", "within must be [minX, minY, maxX, maxY]")),
        },
    };
    let drawing = match parse_drawing(obj) {
        Ok(d) => d,
        Err(e) => return error_json(e),
    };
    let lines = |b: &[BridgeLine]| json!(b.iter().map(|l| json!(l)).collect::<Vec<_>>());
    match at {
        Some(p) => {
            let (found, bridges) = match drawing.at(p, within) {
                Ok(v) => v,
                Err(e) => return error_json(e),
            };
            match found {
                // Only the line art near the region was built: `ignored` lists
                // what is known without building the rest.
                RegionAt::Region(r) => json!({ "status": "ok", "region": region_json(&r), "bridges": lines(&bridges), "ignored": drawing.ignored_built() }),
                other => {
                    let reason = match other {
                        RegionAt::OnLine => "on_line",
                        RegionAt::NoLines => "no_lines",
                        _ => "outside",
                    };
                    json!({ "status": "empty", "reason": reason, "bridges": lines(&bridges), "ignored": drawing.ignored() })
                }
            }
            .to_string()
        }
        None => {
            let arr = match drawing.arrange(None) {
                Ok(a) => a,
                Err(e) => return error_json(e),
            };
            let ignored = json!(drawing.ignored());
            let regions = arr.regions();
            if regions.is_empty() {
                let reason = if arr.has_lines { "no_regions" } else { "no_lines" };
                return json!({ "status": "empty", "reason": reason, "bridges": lines(&arr.bridges), "ignored": ignored }).to_string();
            }
            json!({
                "status": "ok",
                "regions": regions.iter().map(region_json).collect::<Vec<_>>(),
                "bridges": lines(&arr.bridges),
                "ignored": ignored,
            })
            .to_string()
        }
    }
}
