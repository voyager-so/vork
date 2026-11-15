//! Path corner rounding for an arbitrary `BlendPathPoint` list — the native
//! half of the "live corners" primitive: a node-level radius and style, and
//! each anchor's own (`PathPoint.cornerRadius` / `cornerStyle`).
//!
//! This is a line-for-line port of `core/geometry/roundPathCorners.ts`. Every
//! constant (`EPS`, `MIN_TURN`, `MAX_TURN`, `CUBIC_LENGTH_SAMPLES`), the
//! cumulative-chord arc-length sampling and its linear inversion, the subpath
//! split, the ring detection, the corner plan, the cubic handle lengths and the
//! emission order are the TS algorithm's, not a re-derivation: the two lanes are
//! pinned against ONE shared fixture (`fixtures/path-corner-parity/cases.json`,
//! read by `tests/round_corners_parity.rs` here and by
//! `packages/editor/test/roundPathCornersParity.test.ts`), so any divergence is a
//! test failure rather than a silent export/canvas mismatch.
//!
//! Deliberately does NOT reuse `moda-engine`'s `tess/dash.rs` arc-length walk:
//! that one flattens adaptively for stroking and would land on different split
//! parameters, breaking parity by more than the fixture's 1e-9 tolerance.
//!
//! Corner styles (Illustrator's Corners dialog, Absolute rounding): every style
//! trims the two edges at the same points, the tangent points of a circle of the
//! corner's radius (`r · tan(θ/2)` from the vertex for a turn θ, clamped to half
//! the shorter neighbour), and joins them with
//! - `Round`: that circle's arc (convex, tangent to both edges);
//! - `Inverted`: a concave circular arc centred on the vertex;
//! - `Chamfer`: a straight line.
//!
//! Conventions inherited from the TS:
//! - Nothing to round (node radius `<= 0` or non-finite, and no anchor with a
//!   positive radius of its own) returns the input UNCHANGED, so a caller can
//!   invoke this unconditionally without perturbing an existing path.
//! - An anchor's own radius overrides the node's; `0` keeps it sharp.
//! - A subpath is a ring when `closed` is true or its last anchor lands on its
//!   first; an open subpath never rounds its endpoints.
//! - `quadratic`/`arc` are normalized to cubics first, so the output only ever
//!   contains `Move`/`Line`/`Curve`.

use crate::blend::BlendPathPoint;
use crate::path_geom::arc::{arc_to_beziers, ArcParams};

/// Below this a segment is a point and a tangent is undefined; the vertex is left as authored.
const EPS: f64 = 1e-9;
/// A turn smaller than this (radians) is a smooth join, not a corner.
const MIN_TURN: f64 = 1e-4;
/// A turn this close to a full reversal is a cusp; there is no fillet that fits.
const MAX_TURN: f64 = std::f64::consts::PI - 1e-4;
/// Uniform samples used to measure a cubic's arc length and invert it.
const CUBIC_LENGTH_SAMPLES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Pt {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy)]
enum Seg {
    Line { a: Pt, b: Pt },
    Cubic { a: Pt, c1: Pt, c2: Pt, b: Pt },
}

/// How a corner is cut: Illustrator's corner types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CornerStyle {
    /// A convex circular fillet tangent to both edges (the default).
    #[default]
    Round,
    /// A concave circular arc centred on the vertex (Illustrator's Inverted Round).
    Inverted,
    /// A straight cut between the trim points.
    Chamfer,
}

impl CornerStyle {
    /// `'round' | 'inverted' | 'chamfer'` (anything else: `None`).
    pub fn parse(s: &str) -> Option<CornerStyle> {
        match s {
            "round" => Some(CornerStyle::Round),
            "inverted" => Some(CornerStyle::Inverted),
            "chamfer" => Some(CornerStyle::Chamfer),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CornerStyle::Round => "round",
            CornerStyle::Inverted => "inverted",
            CornerStyle::Chamfer => "chamfer",
        }
    }
}

/// An anchor's own corner settings (`PathPoint.cornerRadius` / `cornerStyle`); `None` inherits the node's.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PointCorner {
    pub radius: Option<f64>,
    pub style: Option<CornerStyle>,
}

impl PointCorner {
    /// The settings a point carries: a finite `cornerRadius` (a negative one reads as 0, sharp) and a known
    /// `cornerStyle`.
    pub fn from_ref(p: crate::path_points::PathPointRef<'_>) -> PointCorner {
        PointCorner {
            radius: p.f64("cornerRadius").filter(|r| r.is_finite()).map(|r| r.max(0.0)),
            style: p.str("cornerStyle").and_then(CornerStyle::parse),
        }
    }

    pub fn from_json(v: &crate::value::Value) -> PointCorner {
        PointCorner::from_ref(crate::path_points::PathPointRef::Json(v))
    }

    fn is_set(&self) -> bool {
        self.radius.is_some() || self.style.is_some()
    }
}

/// A path's corner settings: the node's radius and style, and each point's own (`points[i]` for input point `i`;
/// shorter than the points, or empty, when the rest carry nothing).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathCorners {
    pub radius: f64,
    pub style: CornerStyle,
    pub points: Vec<PointCorner>,
}

impl PathCorners {
    /// One radius, round, on every corner (the uniform `cornerRadius`).
    pub fn uniform(radius: f64) -> PathCorners {
        PathCorners {
            radius,
            style: CornerStyle::Round,
            points: Vec::new(),
        }
    }

    /// A node's settings: `cornerRadius` (`|| 0`, as JS reads it) and `cornerStyle`; its points' own go in
    /// [`PathCorners::points`] (see [`parse_path_points`]).
    pub fn of_node(node: &crate::types::RawNode) -> PathCorners {
        PathCorners {
            radius: node.f64("cornerRadius").filter(|v| v.is_finite()).unwrap_or(0.0),
            style: node.str("cornerStyle").and_then(CornerStyle::parse).unwrap_or_default(),
            points: Vec::new(),
        }
    }

    /// [`PathCorners::of_node`] for a node held as a JSON object.
    pub fn of_json(node: &crate::value::Map) -> PathCorners {
        PathCorners {
            radius: node
                .get("cornerRadius")
                .and_then(crate::value::Value::as_f64)
                .filter(|v| v.is_finite())
                .unwrap_or(0.0),
            style: node
                .get("cornerStyle")
                .and_then(crate::value::Value::as_str)
                .and_then(CornerStyle::parse)
                .unwrap_or_default(),
            points: Vec::new(),
        }
    }

    /// Whether any corner could round: the node's radius or an anchor's own is positive. When not, the rounder
    /// hands its input back.
    pub fn rounds_anything(&self) -> bool {
        positive(self.radius) || self.points.iter().any(|p| p.radius.is_some_and(positive))
    }

    /// The settings at the vertex on anchor `at`; `seam` is a second anchor on the same vertex (a ring's last one,
    /// landing back on its first), whose settings count where `at`'s are absent.
    fn at(&self, at: Option<usize>, seam: Option<usize>) -> (f64, CornerStyle) {
        let own = |i: Option<usize>| i.and_then(|i| self.points.get(i)).copied().unwrap_or_default();
        let (a, b) = (own(at), own(seam));
        let radius = a.radius.or(b.radius).unwrap_or(self.radius);
        let style = a.style.or(b.style).unwrap_or(self.style);
        (radius, style)
    }
}

fn positive(r: f64) -> bool {
    r.is_finite() && r > 0.0
}

/// Each point's own corner settings, in step with the points `BlendPathPoint::from_ref` keeps (the unparseable
/// ones dropped); empty when none has any. Compact points never carry any.
pub fn point_corners(points: crate::path_points::PathPoints<'_>) -> Vec<PointCorner> {
    let mut corners: Vec<PointCorner> = Vec::new();
    if !matches!(points, crate::path_points::PathPoints::Json(_)) {
        return corners;
    }
    let mut kept = 0usize;
    for p in points.iter() {
        if BlendPathPoint::from_ref(p).is_none() {
            continue;
        }
        let c = PointCorner::from_ref(p);
        if c.is_set() {
            corners.resize(kept, PointCorner::default());
            corners.push(c);
        }
        kept += 1;
    }
    corners
}

/// [`point_corners`] for points held as JSON values (`BlendPathPoint::from_json` keeps the same ones).
pub fn point_corners_json(points: &[crate::value::Value]) -> Vec<PointCorner> {
    point_corners(crate::path_points::PathPoints::Json(points))
}

impl PathCorners {
    /// A path node's whole corner settings: the node's and its points' own.
    pub fn read(node: &crate::types::RawNode) -> PathCorners {
        let mut c = PathCorners::of_node(node);
        if let Some(points) = node.points() {
            c.points = point_corners(points);
        }
        c
    }

    /// [`PathCorners::read`] for a node held as a JSON object.
    pub fn read_json(node: &crate::value::Map) -> PathCorners {
        let mut c = PathCorners::of_json(node);
        if let Some(points) = node.get("points").and_then(crate::value::Value::as_array) {
            c.points = point_corners_json(points);
        }
        c
    }
}

/// One cut vertex: how far to trim each neighbour, the fillet's effective radius, and how to join the trim points.
#[derive(Debug, Clone, Copy)]
struct Fillet {
    /// Trim distance along the incoming segment, measured back from the vertex.
    d_in: f64,
    /// Trim distance along the outgoing segment, measured forward from the vertex.
    d_out: f64,
    /// Turning angle at the vertex (radians).
    turn: f64,
    /// Radius after clamping `d` to half the shorter neighbour.
    radius: f64,
    style: CornerStyle,
    /// The vertex (the inverted arc's centre).
    vertex: Pt,
    /// +1 when the outline turns counter-clockwise (y up) at the vertex, else -1.
    side: f64,
}

/// `roundPathCorners(points, closed, radius)`: one round radius on every corner.
pub fn round_path_corners(
    points: &[BlendPathPoint],
    closed: bool,
    radius: f64,
) -> Vec<BlendPathPoint> {
    round_path_corners_with(points, closed, &PathCorners::uniform(radius))
}

/// `roundPathCorners(points, closed, radius, style)` with the points' own settings in `corners.points`.
pub fn round_path_corners_with(
    points: &[BlendPathPoint],
    closed: bool,
    corners: &PathCorners,
) -> Vec<BlendPathPoint> {
    if !corners.rounds_anything() {
        return points.to_vec();
    }
    if points.len() < 2 {
        return points.to_vec();
    }

    let mut out: Vec<BlendPathPoint> = Vec::new();
    for (offset, subpath) in split_into_subpaths(points) {
        out.extend(round_subpath(subpath, offset, closed, corners));
    }
    out
}

/// The radius beyond which `round_path_corners` stops changing the shape: the MAX over
/// corners of `min(lenIn, lenOut)/2 / tan(turn/2)`, i.e. the radius at which the last
/// corner hits its half-neighbour clamp. Larger values are a visual no-op, so this is the
/// top of the radius slider (mirror of TS `maxUniformCornerRadius`). It is the largest
/// per-corner limit, not the smallest: a star's acute tips saturate long before its
/// concave notches, and the notches must stay roundable. `f64::INFINITY` when the path
/// has no corner to round (a circle, a straight run, a single segment).
pub fn max_uniform_corner_radius(points: &[BlendPathPoint], closed: bool) -> f64 {
    if points.len() < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::NEG_INFINITY;
    for (_, subpath) in split_into_subpaths(points) {
        let Some((segs, _, ring)) = ring_segments(subpath, closed) else {
            continue;
        };
        let n = segs.len();
        let lengths: Vec<f64> = segs.iter().map(seg_length).collect();
        for i in (if ring { 0 } else { 1 })..n {
            let prev = (i + n - 1) % n;
            let Some(turn) = corner_turn(&segs[prev], &segs[i], lengths[prev], lengths[i]) else {
                continue;
            };
            let limit = (lengths[prev].min(lengths[i]) / 2.0) / (turn / 2.0).tan();
            if limit > best {
                best = limit;
            }
        }
    }
    if best == f64::NEG_INFINITY {
        f64::INFINITY
    } else {
        best
    }
}

/// `splitPathIntoSubpaths` (`core/geometry/pathHitTest.ts:30-50`): a `move`
/// starts a new run; a leading non-move is kept in the first run regardless.
/// Each run comes with the index of its first point.
fn split_into_subpaths(points: &[BlendPathPoint]) -> Vec<(usize, &[BlendPathPoint])> {
    let mut out: Vec<(usize, &[BlendPathPoint])> = Vec::new();
    let mut start = 0usize;
    for (i, p) in points.iter().enumerate() {
        if matches!(p, BlendPathPoint::Move { .. }) {
            if i > start {
                out.push((start, &points[start..i]));
            }
            start = i;
        }
    }
    if start < points.len() {
        out.push((start, &points[start..]));
    }
    out
}

/// Normalized segments plus, for each, the subpath index of the anchor it ends
/// on (`None` inside an arc split into several cubics), plus ring-ness, with the
/// implicit closing line already appended (ending on anchor 0). `None` when
/// there is nothing to round (no segments, or fewer than the two a vertex
/// needs). Shared by the rounder and the max-radius probe so they can never
/// disagree about where the corners are.
#[allow(clippy::type_complexity)]
fn ring_segments(
    subpath: &[BlendPathPoint],
    closed: bool,
) -> Option<(Vec<Seg>, Vec<Option<usize>>, bool)> {
    let first = anchor(&subpath[0]);
    let (mut segs, mut ends) = normalize_segments(subpath);
    if segs.is_empty() {
        return None;
    }

    let last_end = seg_b(&segs[segs.len() - 1]);
    let returns_to_start = same_point(last_end, first);
    let ring = closed || returns_to_start;
    if ring && !returns_to_start {
        segs.push(Seg::Line {
            a: last_end,
            b: first,
        });
        ends.push(Some(0));
    }

    // A ring needs two segments to have a vertex; an open run needs two to have an interior one.
    if segs.len() < 2 {
        return None;
    }
    Some((segs, ends, ring))
}

fn round_subpath(
    subpath: &[BlendPathPoint],
    offset: usize,
    closed: bool,
    settings: &PathCorners,
) -> Vec<BlendPathPoint> {
    let Some((segs, ends, ring)) = ring_segments(subpath, closed) else {
        return subpath.to_vec();
    };
    let n = segs.len();

    // Corner i sits at the START of segs[i] (between segs[i-1] and segs[i]), on the anchor
    // segs[i-1] ends on. For an open run, corner 0 (the first anchor) does not exist. A ring's
    // corner 0 is its seam: its first anchor, or also its last when that one lands back on it.
    let lengths: Vec<f64> = segs.iter().map(seg_length).collect();
    let mut corners: Vec<Option<Fillet>> = vec![None; n];
    for i in (if ring { 0 } else { 1 })..n {
        let prev = (i + n - 1) % n;
        let at = ends[prev].map(|k| k + offset);
        let (radius, style) = if i == 0 {
            settings.at(Some(offset), at)
        } else {
            settings.at(at, None)
        };
        corners[i] = plan_fillet(&segs[prev], &segs[i], lengths[prev], lengths[i], radius, style);
    }
    if corners.iter().all(Option::is_none) {
        return subpath.to_vec();
    }

    let mut emitted: Vec<BlendPathPoint> = Vec::new();
    let start_corner = if ring { corners[0] } else { None };
    // Segment i is trimmed by the fillet at its start (vertex i, `dOut`) and the fillet at
    // its end (vertex i+1, `dIn`); the last segment of a ring ends at vertex 0.
    let first_piece = trim_segment(
        &segs[0],
        start_corner.map_or(0.0, |c| c.d_out),
        corners[1].map_or(0.0, |c| c.d_in),
    );
    let head = seg_a(&first_piece);
    emitted.push(BlendPathPoint::Move {
        x: head.x,
        y: head.y,
    });
    emit_segment(&mut emitted, &first_piece);

    for i in 1..n {
        let corner_here = corners[i];
        let next_corner = if i + 1 < n {
            corners[i + 1]
        } else {
            start_corner
        };
        let piece = trim_segment(
            &segs[i],
            corner_here.map_or(0.0, |c| c.d_out),
            next_corner.map_or(0.0, |c| c.d_in),
        );
        if let Some(corner) = corner_here {
            let from = prev_piece_end(&emitted);
            emit_corner(
                &mut emitted,
                from,
                seg_a(&piece),
                corner,
                tangent_at_start(&piece),
            );
        }
        emit_segment(&mut emitted, &piece);
    }
    if ring {
        if let Some(corner) = start_corner {
            // Close the ring with the fillet at vertex 0, which ends exactly on the move point.
            let from = prev_piece_end(&emitted);
            emit_corner(
                &mut emitted,
                from,
                seg_a(&first_piece),
                corner,
                tangent_at_start(&first_piece),
            );
        }
    }
    emitted
}

// ---------------------------------------------------------------------------
// Fillet planning
// ---------------------------------------------------------------------------

/// The turning angle at the vertex between two segments, or `None` when the
/// vertex is degenerate, smooth, or a cusp — the shared front half of
/// `planFillet`.
// The two-sided comparison is the TS guard verbatim; a `RangeInclusive::contains`
// rewrite would read further from the source this file is diffed against.
#[allow(clippy::manual_range_contains)]
fn corner_turn(seg_in: &Seg, seg_out: &Seg, len_in: f64, len_out: f64) -> Option<f64> {
    corner_turn_side(seg_in, seg_out, len_in, len_out).map(|(turn, _)| turn)
}

/// [`corner_turn`] and the side the outline turns to (+1 counter-clockwise with y up, else -1).
#[allow(clippy::manual_range_contains)]
fn corner_turn_side(seg_in: &Seg, seg_out: &Seg, len_in: f64, len_out: f64) -> Option<(f64, f64)> {
    if len_in < EPS || len_out < EPS {
        return None;
    }
    let t_in = tangent_at_end(seg_in)?;
    let t_out = tangent_at_start(seg_out)?;
    let cos = clamp(t_in.x * t_out.x + t_in.y * t_out.y, -1.0, 1.0);
    let turn = cos.acos();
    if turn < MIN_TURN || turn > MAX_TURN {
        return None;
    }
    let side = if t_in.x * t_out.y - t_in.y * t_out.x < 0.0 {
        -1.0
    } else {
        1.0
    };
    Some((turn, side))
}

fn plan_fillet(
    seg_in: &Seg,
    seg_out: &Seg,
    len_in: f64,
    len_out: f64,
    radius: f64,
    style: CornerStyle,
) -> Option<Fillet> {
    // A sharp corner (this anchor's own 0, or no radius at all).
    if !positive(radius) {
        return None;
    }
    let (turn, side) = corner_turn_side(seg_in, seg_out, len_in, len_out)?;

    // Tangent length of a circular fillet of radius r between two edges turning by θ.
    let wanted = radius * (turn / 2.0).tan();
    let d = wanted.min(len_in / 2.0).min(len_out / 2.0);
    if d < EPS {
        return None;
    }
    let effective = d / (turn / 2.0).tan();
    Some(Fillet {
        d_in: d,
        d_out: d,
        turn,
        radius: effective,
        style,
        vertex: seg_b(seg_in),
        side,
    })
}

/// Join the trim points `from` (on the incoming edge) and `to` (on the outgoing one) as the corner's style says.
fn emit_corner(
    out: &mut Vec<BlendPathPoint>,
    from: Pt,
    to: Pt,
    fillet: Fillet,
    tangent_out: Option<Pt>,
) {
    match fillet.style {
        CornerStyle::Round => emit_fillet(out, from, to, fillet, tangent_out),
        CornerStyle::Chamfer => {
            if dist(from, to) >= EPS {
                out.push(BlendPathPoint::Line { x: to.x, y: to.y });
            }
        }
        CornerStyle::Inverted => emit_inverted(out, from, to, fillet),
    }
}

/// A concave circular arc centred on the vertex from `from` to `to` (the short way round, into the shape), as one
/// cubic: each handle is perpendicular to its radius, `4/3·tan(φ/4)·ρ` long for the sweep φ.
fn emit_inverted(out: &mut Vec<BlendPathPoint>, from: Pt, to: Pt, fillet: Fillet) {
    let v = fillet.vertex;
    let (r1, r2) = (dist(v, from), dist(v, to));
    let (Some(u1), Some(u2)) = (direction(v, from), direction(v, to)) else {
        // A trim point on the vertex: nothing to arc around.
        if dist(from, to) >= EPS {
            out.push(BlendPathPoint::Line { x: to.x, y: to.y });
        }
        return;
    };
    let sweep = clamp(u1.x * u2.x + u1.y * u2.y, -1.0, 1.0).acos();
    // The arc turns against the outline (a notch), so from u1 towards u2 the short way; on a near reversal of the
    // radii, against the outline's turn.
    let cross = u1.x * u2.y - u1.y * u2.x;
    let rot = if cross.abs() > EPS {
        cross.signum()
    } else {
        -fillet.side
    };
    let perp = |u: Pt| Pt {
        x: -u.y * rot,
        y: u.x * rot,
    };
    let (t1, t2) = (perp(u1), perp(u2));
    let h = (4.0 / 3.0) * (sweep / 4.0).tan();
    out.push(BlendPathPoint::Curve {
        x: to.x,
        y: to.y,
        cp1: (from.x + t1.x * h * r1, from.y + t1.y * h * r1),
        cp2: (to.x - t2.x * h * r2, to.y - t2.y * h * r2),
    });
}

fn emit_fillet(
    out: &mut Vec<BlendPathPoint>,
    from: Pt,
    to: Pt,
    fillet: Fillet,
    tangent_out: Option<Pt>,
) {
    // Tangent leaving the trimmed incoming piece is the direction of its last emitted
    // segment at its end; derive it from the emitted geometry so a curved neighbour's split
    // tangent is what the fillet joins to.
    let chord = direction(from, to);
    let t_in = last_emitted_tangent(out).or(chord);
    let t_out = tangent_out.or(chord);
    // Coincident trim points (a fully clamped corner on a degenerate edge) have no arc to draw.
    let (Some(t_in), Some(t_out)) = (t_in, t_out) else {
        return;
    };
    // Cubic approximation of a circular arc of sweep θ and radius r: handle = 4/3·tan(θ/4)·r.
    let k = (4.0 / 3.0) * (fillet.turn / 4.0).tan() * fillet.radius;
    out.push(BlendPathPoint::Curve {
        x: to.x,
        y: to.y,
        cp1: (from.x + t_in.x * k, from.y + t_in.y * k),
        cp2: (to.x - t_out.x * k, to.y - t_out.y * k),
    });
}

// ---------------------------------------------------------------------------
// Segment normalization and trimming
// ---------------------------------------------------------------------------

/// Turn a subpath into line/cubic segments, dropping zero-length ones; with each, the subpath index of the anchor
/// it ends on (`None` for the inner pieces of an arc split into several cubics).
fn normalize_segments(subpath: &[BlendPathPoint]) -> (Vec<Seg>, Vec<Option<usize>>) {
    let mut segs: Vec<Seg> = Vec::new();
    let mut ends: Vec<Option<usize>> = Vec::new();
    let mut current = anchor(&subpath[0]);
    for (k, p) in subpath.iter().enumerate().skip(1) {
        let end = anchor(p);
        let pushed = segs.len();
        match p {
            BlendPathPoint::Move { .. } => {
                // splitPathIntoSubpaths guarantees no interior moves; treat defensively as a jump.
                current = end;
                continue;
            }
            BlendPathPoint::Line { .. } => push_seg(&mut segs, Seg::Line { a: current, b: end }),
            BlendPathPoint::Curve { cp1, cp2, .. } => push_seg(
                &mut segs,
                Seg::Cubic {
                    a: current,
                    c1: Pt { x: cp1.0, y: cp1.1 },
                    c2: Pt { x: cp2.0, y: cp2.1 },
                    b: end,
                },
            ),
            BlendPathPoint::Quadratic { cp, .. } => {
                // Degree elevation: exact.
                let c1 = Pt {
                    x: current.x + (2.0 / 3.0) * (cp.0 - current.x),
                    y: current.y + (2.0 / 3.0) * (cp.1 - current.y),
                };
                let c2 = Pt {
                    x: end.x + (2.0 / 3.0) * (cp.0 - end.x),
                    y: end.y + (2.0 / 3.0) * (cp.1 - end.y),
                };
                push_seg(
                    &mut segs,
                    Seg::Cubic {
                        a: current,
                        c1,
                        c2,
                        b: end,
                    },
                );
            }
            BlendPathPoint::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => {
                let params = ArcParams {
                    rx: *rx,
                    ry: *ry,
                    x_axis_rotation: *x_axis_rotation,
                    large_arc: *large_arc,
                    sweep: *sweep,
                    x: *x,
                    y: *y,
                };
                let beziers = arc_to_beziers(current.x, current.y, &params);
                if beziers.is_empty() {
                    push_seg(&mut segs, Seg::Line { a: current, b: end });
                } else {
                    let mut a = current;
                    for bz in &beziers {
                        let b = Pt {
                            x: bz.end[0],
                            y: bz.end[1],
                        };
                        push_seg(
                            &mut segs,
                            Seg::Cubic {
                                a,
                                c1: Pt {
                                    x: bz.cp1[0],
                                    y: bz.cp1[1],
                                },
                                c2: Pt {
                                    x: bz.cp2[0],
                                    y: bz.cp2[1],
                                },
                                b,
                            },
                        );
                        a = b;
                    }
                }
            }
        }
        // Every piece this point added ends inside it, but the last, which ends on its anchor.
        ends.resize(segs.len(), None);
        if segs.len() > pushed {
            ends[segs.len() - 1] = Some(k);
        }
        current = end;
    }
    (segs, ends)
}

fn push_seg(segs: &mut Vec<Seg>, seg: Seg) {
    if seg_length(&seg) < EPS {
        return;
    }
    segs.push(seg);
}

/// Trim `trim_start` off the start and `trim_end` off the end of a segment, by arc length.
fn trim_segment(seg: &Seg, trim_start: f64, trim_end: f64) -> Seg {
    let mut piece = *seg;
    if trim_start > EPS {
        piece = split_at_length(&piece, trim_start).1;
    }
    if trim_end > EPS {
        let len = seg_length(&piece);
        piece = split_at_length(&piece, (len - trim_end).max(0.0)).0;
    }
    piece
}

/// Split a segment at arc length `s` from its start into (before, after).
fn split_at_length(seg: &Seg, s: f64) -> (Seg, Seg) {
    match seg {
        Seg::Line { a, b } => {
            let len = seg_length(seg);
            let t = if len < EPS {
                0.0
            } else {
                clamp(s / len, 0.0, 1.0)
            };
            let m = lerp(*a, *b, t);
            (Seg::Line { a: *a, b: m }, Seg::Line { a: m, b: *b })
        }
        Seg::Cubic { .. } => split_cubic(seg, cubic_param_at_length(seg, s)),
    }
}

fn split_cubic(seg: &Seg, t: f64) -> (Seg, Seg) {
    let Seg::Cubic { a, c1, c2, b } = *seg else {
        return (*seg, *seg);
    };
    let p01 = lerp(a, c1, t);
    let p12 = lerp(c1, c2, t);
    let p23 = lerp(c2, b, t);
    let p012 = lerp(p01, p12, t);
    let p123 = lerp(p12, p23, t);
    let m = lerp(p012, p123, t);
    (
        Seg::Cubic {
            a,
            c1: p01,
            c2: p012,
            b: m,
        },
        Seg::Cubic {
            a: m,
            c1: p123,
            c2: p23,
            b,
        },
    )
}

fn cubic_point(a: Pt, c1: Pt, c2: Pt, b: Pt, t: f64) -> Pt {
    let mt = 1.0 - t;
    let w0 = mt * mt * mt;
    let w1 = 3.0 * mt * mt * t;
    let w2 = 3.0 * mt * t * t;
    let w3 = t * t * t;
    Pt {
        x: w0 * a.x + w1 * c1.x + w2 * c2.x + w3 * b.x,
        y: w0 * a.y + w1 * c1.y + w2 * c2.y + w3 * b.y,
    }
}

/// Cumulative chord lengths at uniform `t` samples; index k is the length up to t = k/N.
fn cubic_cumulative_lengths(seg: &Seg) -> Vec<f64> {
    let Seg::Cubic { a, c1, c2, b } = *seg else {
        return vec![0.0; CUBIC_LENGTH_SAMPLES + 1];
    };
    let mut cum = Vec::with_capacity(CUBIC_LENGTH_SAMPLES + 1);
    cum.push(0.0);
    let mut prev = a;
    for k in 1..=CUBIC_LENGTH_SAMPLES {
        let p = cubic_point(a, c1, c2, b, k as f64 / CUBIC_LENGTH_SAMPLES as f64);
        cum.push(cum[k - 1] + dist(prev, p));
        prev = p;
    }
    cum
}

/// Invert arc length → t by linear interpolation inside the sampled polyline.
fn cubic_param_at_length(seg: &Seg, s: f64) -> f64 {
    let cum = cubic_cumulative_lengths(seg);
    let total = cum[CUBIC_LENGTH_SAMPLES];
    if total < EPS {
        return 0.0;
    }
    if s <= 0.0 {
        return 0.0;
    }
    if s >= total {
        return 1.0;
    }
    let mut k = 1usize;
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

fn seg_length(seg: &Seg) -> f64 {
    match seg {
        Seg::Line { a, b } => dist(*a, *b),
        Seg::Cubic { .. } => cubic_cumulative_lengths(seg)[CUBIC_LENGTH_SAMPLES],
    }
}

// ---------------------------------------------------------------------------
// Tangents and emission
// ---------------------------------------------------------------------------

fn tangent_at_start(seg: &Seg) -> Option<Pt> {
    match seg {
        Seg::Line { a, b } => direction(*a, *b),
        Seg::Cubic { a, c1, c2, b } => direction(*a, *c1)
            .or_else(|| direction(*a, *c2))
            .or_else(|| direction(*a, *b)),
    }
}

fn tangent_at_end(seg: &Seg) -> Option<Pt> {
    match seg {
        Seg::Line { a, b } => direction(*a, *b),
        Seg::Cubic { a, c1, c2, b } => direction(*c2, *b)
            .or_else(|| direction(*c1, *b))
            .or_else(|| direction(*a, *b)),
    }
}

fn emit_segment(out: &mut Vec<BlendPathPoint>, seg: &Seg) {
    // Two fillets clamped to half the same edge meet at its midpoint and leave nothing to draw.
    if seg_length(seg) < EPS {
        return;
    }
    match seg {
        Seg::Line { b, .. } => out.push(BlendPathPoint::Line { x: b.x, y: b.y }),
        Seg::Cubic { c1, c2, b, .. } => out.push(BlendPathPoint::Curve {
            x: b.x,
            y: b.y,
            cp1: (c1.x, c1.y),
            cp2: (c2.x, c2.y),
        }),
    }
}

fn prev_piece_end(out: &[BlendPathPoint]) -> Pt {
    anchor(&out[out.len() - 1])
}

/// Direction the most recently emitted segment is heading at its end.
fn last_emitted_tangent(out: &[BlendPathPoint]) -> Option<Pt> {
    if out.len() < 2 {
        return None;
    }
    let last = &out[out.len() - 1];
    let prev = anchor(&out[out.len() - 2]);
    let end = anchor(last);
    if let BlendPathPoint::Curve { cp1, cp2, .. } = last {
        let c1 = Pt { x: cp1.0, y: cp1.1 };
        let c2 = Pt { x: cp2.0, y: cp2.1 };
        return direction(c2, end)
            .or_else(|| direction(c1, end))
            .or_else(|| direction(prev, end));
    }
    direction(prev, end)
}

// ---------------------------------------------------------------------------
// Small vector helpers
// ---------------------------------------------------------------------------

fn anchor(p: &BlendPathPoint) -> Pt {
    match p {
        BlendPathPoint::Move { x, y }
        | BlendPathPoint::Line { x, y }
        | BlendPathPoint::Curve { x, y, .. }
        | BlendPathPoint::Quadratic { x, y, .. }
        | BlendPathPoint::Arc { x, y, .. } => Pt { x: *x, y: *y },
    }
}

fn seg_a(seg: &Seg) -> Pt {
    match seg {
        Seg::Line { a, .. } | Seg::Cubic { a, .. } => *a,
    }
}

fn seg_b(seg: &Seg) -> Pt {
    match seg {
        Seg::Line { b, .. } | Seg::Cubic { b, .. } => *b,
    }
}

fn direction(a: Pt, b: Pt) -> Option<Pt> {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = dx.hypot(dy);
    if len < EPS {
        return None;
    }
    Some(Pt {
        x: dx / len,
        y: dy / len,
    })
}

fn dist(a: Pt, b: Pt) -> f64 {
    (b.x - a.x).hypot(b.y - a.y)
}

fn lerp(a: Pt, b: Pt, t: f64) -> Pt {
    Pt {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
    }
}

fn same_point(a: Pt, b: Pt) -> bool {
    (a.x - b.x).abs() < EPS && (a.y - b.y).abs() < EPS
}

/// TS `clamp` (ternary), not `f64::clamp`: NaN bounds must not panic here.
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(x: f64, y: f64) -> BlendPathPoint {
        BlendPathPoint::Move { x, y }
    }
    fn ln(x: f64, y: f64) -> BlendPathPoint {
        BlendPathPoint::Line { x, y }
    }

    fn square_100() -> Vec<BlendPathPoint> {
        vec![
            mv(0.0, 0.0),
            ln(100.0, 0.0),
            ln(100.0, 100.0),
            ln(0.0, 100.0),
        ]
    }

    fn kinds(points: &[BlendPathPoint]) -> Vec<&'static str> {
        points
            .iter()
            .map(|p| match p {
                BlendPathPoint::Move { .. } => "move",
                BlendPathPoint::Line { .. } => "line",
                BlendPathPoint::Curve { .. } => "curve",
                BlendPathPoint::Quadratic { .. } => "quadratic",
                BlendPathPoint::Arc { .. } => "arc",
            })
            .collect()
    }

    #[test]
    fn zero_or_negative_radius_returns_the_input_unchanged() {
        let pts = square_100();
        for r in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let out = round_path_corners(&pts, true, r);
            assert_eq!(kinds(&out), kinds(&pts), "radius {r} must pass through");
            for (a, b) in out.iter().zip(pts.iter()) {
                assert_eq!(a.to_json(), b.to_json());
            }
        }
    }

    #[test]
    fn square_rounds_to_four_fillets_with_kappa_handles() {
        let radius = 10.0;
        let out = round_path_corners(&square_100(), true, radius);
        assert_eq!(
            kinds(&out),
            vec!["move", "line", "curve", "line", "curve", "line", "curve", "line", "curve"],
        );

        // A 90° turn's fillet is the exact circular quarter-arc cubic, so its
        // handles are KAPPA·r — the same constant `rectangleToPathPoints` uses.
        let kappa = (4.0 / 3.0) * (std::f64::consts::SQRT_2 - 1.0);
        let mut curves = 0;
        let mut prev = (0.0, 0.0);
        for p in &out {
            if let BlendPathPoint::Curve { x, y, cp1, cp2 } = p {
                let h1 = (cp1.0 - prev.0).hypot(cp1.1 - prev.1);
                let h2 = (x - cp2.0).hypot(y - cp2.1);
                assert!(
                    (h1 - kappa * radius).abs() < 1e-9 && (h2 - kappa * radius).abs() < 1e-9,
                    "handle lengths {h1}/{h2} != {}",
                    kappa * radius
                );
                curves += 1;
            }
            let a = anchor(p);
            prev = (a.x, a.y);
        }
        assert_eq!(curves, 4);
    }

    #[test]
    fn open_polyline_keeps_its_endpoints() {
        let pts = vec![mv(0.0, 0.0), ln(50.0, 0.0), ln(50.0, 50.0)];
        let out = round_path_corners(&pts, false, 10.0);
        // Only the interior vertex rounds: move, shortened line, fillet, shortened line.
        assert_eq!(kinds(&out), vec!["move", "line", "curve", "line"]);
        let first = anchor(&out[0]);
        let last = anchor(&out[out.len() - 1]);
        assert!((first.x - 0.0).abs() < 1e-12 && (first.y - 0.0).abs() < 1e-12);
        assert!((last.x - 50.0).abs() < 1e-12 && (last.y - 50.0).abs() < 1e-12);
    }

    #[test]
    fn pentagon_max_radius_is_the_half_edge_tangent_limit() {
        let circum = 100.0;
        let pts: Vec<BlendPathPoint> = (0..5)
            .map(|i| {
                let a =
                    -std::f64::consts::FRAC_PI_2 + (i as f64) * 2.0 * std::f64::consts::PI / 5.0;
                let (x, y) = (circum * a.cos(), circum * a.sin());
                if i == 0 {
                    mv(x, y)
                } else {
                    ln(x, y)
                }
            })
            .collect();

        let edge = 2.0 * circum * (std::f64::consts::PI / 5.0).sin();
        let turn = 2.0 * std::f64::consts::PI / 5.0; // exterior angle
        let expected = (edge / 2.0) / (turn / 2.0).tan();
        let got = max_uniform_corner_radius(&pts, true);
        assert!(
            (got - expected).abs() < 1e-9,
            "pentagon max radius {got} != {expected}"
        );

        // At exactly that radius every corner is on the clamp, so the fillets meet
        // at the edge midpoints and no straight run survives.
        let out = round_path_corners(&pts, true, got);
        assert_eq!(
            out.iter()
                .filter(|p| matches!(p, BlendPathPoint::Curve { .. }))
                .count(),
            5
        );
    }

    #[test]
    fn point_corners_follow_the_points_the_parser_keeps() {
        let pts = crate::value::json!([
            {"type": "move", "x": 0, "y": 0},
            {"type": "bogus", "x": 1, "y": 1, "cornerRadius": 99},
            {"type": "line", "x": 10, "y": 0, "cornerRadius": 5, "cornerStyle": "chamfer"},
            {"type": "line", "x": 10, "y": 10, "cornerRadius": -2, "cornerStyle": "nope"}
        ]);
        let c = point_corners_json(pts.as_array().unwrap());
        assert_eq!(
            c,
            vec![
                PointCorner::default(),
                PointCorner {
                    radius: Some(5.0),
                    style: Some(CornerStyle::Chamfer)
                },
                PointCorner {
                    radius: Some(0.0),
                    style: None
                },
            ]
        );
        let plain = crate::value::json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 1, "y": 0}]);
        assert!(point_corners_json(plain.as_array().unwrap()).is_empty());
    }

    #[test]
    fn inverted_corner_is_a_concave_arc_about_the_vertex() {
        let r = 20.0;
        let corners = PathCorners {
            radius: r,
            style: CornerStyle::Inverted,
            points: Vec::new(),
        };
        let out = round_path_corners_with(&square_100(), true, &corners);
        // The first fillet: from (80, 0) to (100, 20), about (100, 0).
        let BlendPathPoint::Curve { x, y, cp1, cp2 } = out[2].clone() else {
            panic!("{out:?}")
        };
        let from = anchor(&out[1]);
        let mid = cubic_point(
            from,
            Pt { x: cp1.0, y: cp1.1 },
            Pt { x: cp2.0, y: cp2.1 },
            Pt { x, y },
            0.5,
        );
        assert!(((mid.x - 100.0).hypot(mid.y) - r).abs() < 0.01, "{mid:?}");
        assert!(mid.x < 100.0 && mid.y > 0.0, "inside the square: {mid:?}");
        // Chamfer: lines only; an anchor's own 0 keeps that corner sharp.
        let chamfer = PathCorners {
            radius: 10.0,
            style: CornerStyle::Chamfer,
            points: vec![PointCorner::default(), PointCorner { radius: Some(0.0), style: None }],
        };
        let out = round_path_corners_with(&square_100(), true, &chamfer);
        assert!(out.iter().all(|p| matches!(p, BlendPathPoint::Move { .. } | BlendPathPoint::Line { .. })));
        assert!(out.iter().any(|p| anchor(p) == Pt { x: 100.0, y: 0.0 }));
    }

    #[test]
    fn nothing_to_round_is_a_pass_through() {
        let corners = PathCorners {
            radius: 0.0,
            style: CornerStyle::Chamfer,
            points: vec![PointCorner { radius: Some(0.0), style: Some(CornerStyle::Inverted) }],
        };
        assert!(!corners.rounds_anything());
        let pts = square_100();
        let out = round_path_corners_with(&pts, true, &corners);
        assert_eq!(kinds(&out), kinds(&pts));
    }

    #[test]
    fn no_corner_path_has_infinite_max_radius() {
        let straight = vec![mv(0.0, 0.0), ln(10.0, 0.0), ln(20.0, 0.0)];
        assert_eq!(max_uniform_corner_radius(&straight, false), f64::INFINITY);
        assert_eq!(
            max_uniform_corner_radius(&[mv(0.0, 0.0)], false),
            f64::INFINITY
        );
    }
}
