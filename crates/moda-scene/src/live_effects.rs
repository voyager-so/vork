//! Live geometry effects (E8.1, Illustrator's Effect menu): Roughen, Zig Zag,
//! Pucker & Bloat, Twist, Transform (with copies), Round Corners, Offset Path,
//! Outline Stroke and the Warp presets, as entries of a node's `effects` list.
//!
//! **Where they run.** At scene resolution ([`crate::scene::Scene`]): a shape
//! (rectangle, ellipse, star, polygon, path) with a visible geometry effect
//! resolves to a `path` node whose `points` are the effected outline
//! ([`resolve_node`]). Every render lane reads resolved nodes, so the body,
//! the stroke, shadows, masks, backdrop coverage and bounds all see the same
//! geometry, before any fill or stroke draws. The TS side reaches the same
//! code through the `geometry_effects` wasm export (SVG / PDF export, Expand
//! Appearance), so there is one implementation.
//!
//! **Order.** Geometry effects apply in list order (`effects[0]` first), before
//! every drawing effect (shadows, blurs, …), which then see the result. In
//! Illustrator's Appearance panel the order among all effects matters; here a
//! drop shadow always shadows the effected outline (Illustrator's result when
//! the geometry effect sits above the shadow).
//!
//! **Copies.** The Transform effect with copies turns one outline into several.
//! The resolved `points` hold every copy (a compound path, for the lanes that
//! need one outline: shadows, masks, hit bounds) and `__effectCopies` lists
//! how many points each copy has, so the body lane draws the copies one after
//! the other, each with its own fill and stroke, as Illustrator does (a later
//! copy covers an earlier one's stroke).
//!
//! **Determinism.** Everything here is plain `f64` math with no platform
//! calls beyond `sin` / `cos` / `atan2`; Roughen draws from a SplitMix64
//! stream seeded by the effect's `seed`, so a seed always gives the same shape
//! on every machine and in every lane.
//!
//! **Sizes.** Canvas px throughout; an inch is 72 px (Illustrator's points at
//! 100 %). A "relative" size is a percentage of the larger side of the art's
//! bounds at the time the effect runs.

use crate::value::{json, Map, Value};

use crate::blend::BlendPathPoint;
use crate::effect::{
    parse_effects, Effect, EffectKind, OffsetPathEffect, PuckerBloat, Roughen, TransformEffect,
    Twist, WarpEffect, ZigZag,
};

#[path = "live_effects_distort.rs"]
pub mod distort;
use crate::path_bool::BooleanOutcome;
use crate::path_geom::offset_path::{offset_path, OffsetJoin, OffsetRequest};
use crate::path_geom::vec2::{v2, Cubic, V2};
use crate::shape_outline::node_outline;
use crate::stroke_outline::{stroke_outline, Outline, OutlineOptions};
use crate::types::RawNode;

/// The resolved node's per-copy point counts (see the module docs). Internal to
/// the render lanes: never stored in a document.
pub const EFFECT_COPIES_KEY: &str = "__effectCopies";
/// The resolved node's per-copy paint roles (`fill` / `stroke`, see
/// [`PaintRole`]), present only when a copy paints one of them alone (Scribble
/// splits a copy into its fill's scribble and its stroke's). Never stored.
pub const EFFECT_COPY_ROLES_KEY: &str = "__effectCopyRoles";
/// The resolved node's stack warnings ([`StackWarning`] as JSON), present only when the stack stopped before its
/// end. Never stored; `geometry_effects_json` answers them as `warnings`, `assess` as manifest warnings.
pub const EFFECT_WARNINGS_KEY: &str = "__effectWarnings";
/// The resolved node's per-copy fill stacks, present only when copies fill with different paints (a warped text
/// layer with multi-colour runs: one outline group per run fill, `crate::text_warp`). Never stored.
pub const EFFECT_COPY_FILLS_KEY: &str = "__effectCopyFills";

/// Which of the node's paints a copy draws with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaintRole {
    /// Its fills and its strokes (every copy but Scribble's).
    Both,
    /// Its fills only.
    Fill,
    /// Its strokes' paints, filling the copy (a stroke band already outlined).
    Stroke,
}

impl PaintRole {
    pub fn as_str(self) -> &'static str {
        match self {
            PaintRole::Both => "both",
            PaintRole::Fill => "fill",
            PaintRole::Stroke => "stroke",
        }
    }
    fn parse(s: &str) -> PaintRole {
        match s {
            "fill" => PaintRole::Fill,
            "stroke" => PaintRole::Stroke,
            _ => PaintRole::Both,
        }
    }
}

/// Points per inch.
const INCH: f64 = 72.0;
/// Most copies a Transform effect makes.
pub const MAX_COPIES: usize = 1000;
/// Most segments an effected outline may hold; an effect that would exceed it
/// is skipped (the outline so far is kept), so a slider dragged to an extreme
/// never hangs the renderer.
pub const MAX_SEGMENTS: usize = 250_000;
/// Most pieces one segment is split into before a non-linear map (twist, warp).
const MAX_SPLITS_PER_SEGMENT: usize = 64;
/// Most geometry effects evaluated on one node, in list order; the rest are not evaluated and a
/// [`StackWarning::TooManyEffects`] reports them. Far beyond any stack an artist builds by hand (Illustrator's
/// Appearance panel stacks run to a handful), it bounds the stack's cost: each effect's own cost is bounded (by
/// [`MAX_SEGMENTS`], [`MAX_COPIES`] and the offsetter's limits), but a stack of 2,000 compounds it. 2,000 Twists
/// grow the outline towards `MAX_SEGMENTS` and re-split all of it at every step (400 took 7 s); 2,000 round Offset
/// Paths each flatten and refit an ever larger outline (400 of them took 4 minutes).
pub const MAX_GEOMETRY_EFFECTS: usize = 64;
/// Most segments a stack's effects may make between them (the sum of every effect's result): four outlines at
/// [`MAX_SEGMENTS`]. Each effect costs about as much as the outline it reshapes, so this bounds a stack whose
/// effects each stay under `MAX_SEGMENTS` but rework a huge outline many times over (64 Roughens at 249,000
/// segments took 9 s). The effects after the one that crosses it are not evaluated; a
/// [`StackWarning::OverBudget`] reports them.
pub const MAX_STACK_SEGMENTS: usize = 4 * MAX_SEGMENTS;

// ---------------------------------------------------------------------------
// Geometry model
// ---------------------------------------------------------------------------

/// One segment: a cubic, flagged when it is a straight line (emitted as `line`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seg {
    pub c: Cubic,
    pub line: bool,
}

impl Seg {
    pub(crate) fn line(a: V2, b: V2) -> Seg {
        Seg {
            c: Cubic::line(a, b),
            line: true,
        }
    }
    pub(crate) fn curve(a: V2, c1: V2, c2: V2, b: V2) -> Seg {
        Seg {
            c: Cubic([a, c1, c2, b]),
            line: false,
        }
    }
    pub(crate) fn start(&self) -> V2 {
        self.c.0[0]
    }
    pub(crate) fn end(&self) -> V2 {
        self.c.0[3]
    }
    pub(crate) fn len(&self) -> f64 {
        if self.line {
            self.start().dist(self.end())
        } else {
            self.c.arc_len()
        }
    }
    /// Split at parameter `t` (de Casteljau).
    pub(crate) fn split(&self, t: f64) -> (Seg, Seg) {
        if self.line {
            let m = self.start().lerp(self.end(), t);
            return (Seg::line(self.start(), m), Seg::line(m, self.end()));
        }
        let [p0, p1, p2, p3] = self.c.0;
        let a = p0.lerp(p1, t);
        let b = p1.lerp(p2, t);
        let c = p2.lerp(p3, t);
        let d = a.lerp(b, t);
        let e = b.lerp(c, t);
        let f = d.lerp(e, t);
        (Seg::curve(p0, a, d, f), Seg::curve(f, e, c, p3))
    }
    /// The parameter at arc length `s` from the start.
    pub(crate) fn t_at_len(&self, s: f64, total: f64) -> f64 {
        if self.line || total <= 0.0 {
            return if total > 0.0 {
                (s / total).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if self.c.arc_len_to(mid) < s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
    pub(crate) fn eval(&self, t: f64) -> V2 {
        if self.line {
            self.start().lerp(self.end(), t)
        } else {
            self.c.eval(t)
        }
    }
    /// Unit tangent at `t` (the chord for a degenerate curve).
    fn tangent(&self, t: f64) -> Option<V2> {
        let d = if self.line {
            self.end() - self.start()
        } else {
            self.c.d1(t)
        };
        d.unit().or_else(|| (self.end() - self.start()).unit())
    }
}

/// One subpath.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Contour {
    pub start: V2,
    pub segs: Vec<Seg>,
    /// A ring: the last segment ends on `start`.
    pub closed: bool,
}

impl Contour {
    /// The anchors, a ring's start once.
    pub fn anchors(&self) -> Vec<V2> {
        let mut a = vec![self.start];
        for s in &self.segs {
            a.push(s.end());
        }
        if self.closed {
            a.pop();
        }
        a
    }
}

/// One copy of the art: its subpaths.
pub type Shape = Vec<Contour>;

/// The effected geometry: one shape per copy, and the fill rule they fill with.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub copies: Vec<Shape>,
    /// One per copy: which paints it draws with.
    pub roles: Vec<PaintRole>,
    /// One per copy: the index of the input outline it came from ([`apply_effects_to_copies`] takes several, each
    /// with its own paint; [`apply_effects`] takes one, so every entry is 0).
    pub groups: Vec<usize>,
    /// `false` = nonzero, `true` = evenodd.
    pub even_odd: bool,
    /// Whether the outline is closed (Moda paths carry one flag for every subpath).
    pub closed: bool,
    /// Why the stack stopped before its end (empty when every effect was evaluated).
    pub warnings: Vec<StackWarning>,
}

/// Why a geometry effect stack stopped before its end. Past either cap the effects that follow are not evaluated
/// (the outline so far is drawn), so a pathological stack costs a bounded amount instead of hanging the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackWarning {
    /// More than [`MAX_GEOMETRY_EFFECTS`] effects: the first `evaluated` of `total` were.
    TooManyEffects { evaluated: usize, total: usize },
    /// The effect at index `at` would have made more than [`MAX_SEGMENTS`] segments: it and the `total - at - 1`
    /// after it were not applied. (Before, each of them was computed in full and then refused one by one: 64
    /// stacked Outline Strokes took 13 s, 64 Transforms with copies ran for minutes.)
    TooComplex { at: usize, total: usize },
    /// The effects before index `at` made more than [`MAX_STACK_SEGMENTS`] segments between them: the effect at
    /// `at` and the `total - at - 1` after it were not evaluated.
    OverBudget { at: usize, total: usize },
}

impl StackWarning {
    pub fn code(self) -> &'static str {
        match self {
            StackWarning::TooManyEffects { .. } => "effect_stack_truncated",
            StackWarning::TooComplex { .. } => "effect_too_complex",
            StackWarning::OverBudget { .. } => "effect_stack_over_budget",
        }
    }

    /// `{"code", "message", …}`: the `warnings` entries of `geometry_effects_json`.
    pub fn to_json(self) -> Value {
        match self {
            StackWarning::TooManyEffects { evaluated, total } => json!({
                "code": self.code(),
                "message": format!("only the first {evaluated} of {total} geometry effects are applied"),
                "evaluated": evaluated,
                "total": total,
            }),
            StackWarning::TooComplex { at, total } => json!({
                "code": self.code(),
                "message": format!(
                    "geometry effect {} of {total} would make more than {MAX_SEGMENTS} segments; it and the effects after it are not applied",
                    at + 1
                ),
                "effect": at,
                "total": total,
            }),
            StackWarning::OverBudget { at, total } => json!({
                "code": self.code(),
                "message": format!(
                    "geometry effects 1 to {at} of {total} made more than {MAX_STACK_SEGMENTS} segments between them; the effects after them are not applied"
                ),
                "effect": at,
                "total": total,
            }),
        }
    }
}

impl Geometry {
    fn segment_count(&self) -> usize {
        self.copies
            .iter()
            .flat_map(|s| s.iter())
            .map(|c| c.segs.len())
            .sum()
    }
}

fn anchor_of(p: &BlendPathPoint) -> V2 {
    match p {
        BlendPathPoint::Move { x, y }
        | BlendPathPoint::Line { x, y }
        | BlendPathPoint::Curve { x, y, .. }
        | BlendPathPoint::Quadratic { x, y, .. }
        | BlendPathPoint::Arc { x, y, .. } => v2(*x, *y),
    }
}

/// Path points → contours. `closed` closes every subpath (Moda's one flag);
/// an open subpath whose last anchor lands on its first is a ring too.
pub fn contours_from_points(points: &[BlendPathPoint], closed: bool) -> Shape {
    use crate::path_geom::arc::{arc_to_beziers, ArcParams};
    let mut out: Shape = Vec::new();
    let mut cur: Option<Contour> = None;
    let mut pen = v2(0.0, 0.0);
    let finish = |c: Contour, out: &mut Shape| {
        let mut c = c;
        let end = c.segs.last().map(Seg::end).unwrap_or(c.start);
        if closed && end.dist(c.start) > 1e-9 {
            c.segs.push(Seg::line(end, c.start));
        }
        c.closed = !c.segs.is_empty() && (closed || end.dist(c.start) <= 1e-9);
        if !c.segs.is_empty() {
            out.push(c);
        }
    };
    for p in points {
        let end = anchor_of(p);
        if !end.is_finite() {
            continue;
        }
        match p {
            BlendPathPoint::Move { .. } => {
                if let Some(c) = cur.take() {
                    finish(c, &mut out);
                }
                cur = Some(Contour {
                    start: end,
                    ..Default::default()
                });
                pen = end;
                continue;
            }
            _ => {
                if cur.is_none() {
                    // A path that does not start with a move starts at its first anchor.
                    cur = Some(Contour {
                        start: end,
                        ..Default::default()
                    });
                    pen = end;
                    continue;
                }
            }
        }
        let c = cur.as_mut().expect("contour");
        match p {
            BlendPathPoint::Line { .. } => c.segs.push(Seg::line(pen, end)),
            BlendPathPoint::Curve { cp1, cp2, .. } => {
                c.segs
                    .push(Seg::curve(pen, v2(cp1.0, cp1.1), v2(cp2.0, cp2.1), end))
            }
            BlendPathPoint::Quadratic { cp, .. } => {
                let q = v2(cp.0, cp.1);
                c.segs.push(Seg::curve(
                    pen,
                    pen + (q - pen) * (2.0 / 3.0),
                    end + (q - end) * (2.0 / 3.0),
                    end,
                ))
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
                let arcs = arc_to_beziers(
                    pen.x,
                    pen.y,
                    &ArcParams {
                        rx: *rx,
                        ry: *ry,
                        x_axis_rotation: *x_axis_rotation,
                        large_arc: *large_arc,
                        sweep: *sweep,
                        x: *x,
                        y: *y,
                    },
                );
                if arcs.is_empty() {
                    c.segs.push(Seg::line(pen, end));
                } else {
                    let mut a = pen;
                    for b in arcs {
                        let e = v2(b.end[0], b.end[1]);
                        c.segs.push(Seg::curve(
                            a,
                            v2(b.cp1[0], b.cp1[1]),
                            v2(b.cp2[0], b.cp2[1]),
                            e,
                        ));
                        a = e;
                    }
                }
            }
            BlendPathPoint::Move { .. } => unreachable!(),
        }
        pen = end;
    }
    if let Some(c) = cur.take() {
        finish(c, &mut out);
    }
    out
}

/// Contours → path points (`move`, then `line` / `curve`; a ring ends on its start).
pub fn points_from_contours(shape: &[Contour]) -> Vec<BlendPathPoint> {
    let mut out = Vec::new();
    for c in shape {
        out.push(BlendPathPoint::Move {
            x: c.start.x,
            y: c.start.y,
        });
        for s in &c.segs {
            let [_, c1, c2, e] = s.c.0;
            out.push(if s.line {
                BlendPathPoint::Line { x: e.x, y: e.y }
            } else {
                BlendPathPoint::Curve {
                    x: e.x,
                    y: e.y,
                    cp1: (c1.x, c1.y),
                    cp2: (c2.x, c2.y),
                }
            });
        }
    }
    out
}

/// Axis-aligned bounds of a set of shapes: anchors plus 8 samples per curve
/// (tight to well under a pixel for the sizes effects work at).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: V2,
    pub max: V2,
}

impl Bounds {
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }
    pub fn center(&self) -> V2 {
        self.min.lerp(self.max, 0.5)
    }
    /// The larger side: what a relative size is a percentage of.
    pub fn size(&self) -> f64 {
        self.width().max(self.height())
    }
}

pub fn bounds_of<'a>(shapes: impl IntoIterator<Item = &'a Shape>) -> Option<Bounds> {
    let mut min = v2(f64::INFINITY, f64::INFINITY);
    let mut max = v2(f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut add = |p: V2| {
        min = v2(min.x.min(p.x), min.y.min(p.y));
        max = v2(max.x.max(p.x), max.y.max(p.y));
    };
    for shape in shapes {
        for c in shape {
            add(c.start);
            for s in &c.segs {
                add(s.end());
                if !s.line {
                    for i in 1..8 {
                        add(s.c.eval(i as f64 / 8.0));
                    }
                }
            }
        }
    }
    (min.x <= max.x && min.y <= max.y).then_some(Bounds { min, max })
}

/// Split every segment into pieces no longer than `max_len` (at most
/// `MAX_SPLITS_PER_SEGMENT` each), by equal parameter steps.
fn subdivide(shape: &Shape, max_len: f64) -> Shape {
    shape
        .iter()
        .map(|c| {
            let mut segs = Vec::with_capacity(c.segs.len());
            for s in &c.segs {
                let len = s.len();
                let n = if max_len > 0.0 && len.is_finite() {
                    ((len / max_len).ceil() as usize).clamp(1, MAX_SPLITS_PER_SEGMENT)
                } else {
                    1
                };
                let mut rest = *s;
                for i in 0..n - 1 {
                    // Split the remainder at 1/(pieces left) → equal parameter steps.
                    let (a, b) = rest.split(1.0 / (n - i) as f64);
                    segs.push(a);
                    rest = b;
                }
                segs.push(rest);
            }
            Contour {
                start: c.start,
                segs,
                closed: c.closed,
            }
        })
        .collect()
}

/// Map every anchor and handle through `f`. Lines become curves (a non-linear
/// map bends them) unless `affine`.
fn map_shape(shape: &Shape, f: &dyn Fn(V2) -> V2, affine: bool) -> Shape {
    shape
        .iter()
        .map(|c| Contour {
            start: f(c.start),
            segs: c
                .segs
                .iter()
                .map(|s| {
                    if s.line && affine {
                        Seg::line(f(s.start()), f(s.end()))
                    } else {
                        let [p0, p1, p2, p3] = s.c.0;
                        Seg::curve(f(p0), f(p1), f(p2), f(p3))
                    }
                })
                .collect(),
            closed: c.closed,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Point sequences (Roughen, Zig Zag)
// ---------------------------------------------------------------------------

/// A point on the contour with the path direction there.
#[derive(Debug, Clone, Copy)]
struct Sample {
    p: V2,
    /// Unit direction of travel (the bisector of the two sides at a corner).
    dir: V2,
}

/// The contour's anchors plus `extra(seg)` evenly spaced (by arc length)
/// points inside each segment, in order. A ring does not repeat its start.
/// Whether sampling `shape` with `extra` points per segment would make more than [`MAX_SEGMENTS`] segments:
/// the effect's result would be refused after the fact, so it is refused before paying for it (a 6e10 px
/// segment roughened at 10 points per inch asked for 8e9 samples, an 8 GB allocation that aborts the wasm
/// engine).
fn samples_exceed_budget(shape: &Shape, extra: &dyn Fn(f64) -> usize) -> bool {
    let mut total = 0usize;
    for c in shape.iter() {
        // A contour through its samples has one segment fewer than it has points when open, as many closed.
        total = total.saturating_sub(1);
        for s in &c.segs {
            total = total.saturating_add(1).saturating_add(extra(s.len()));
        }
        total = total.saturating_add(1);
        if total > MAX_SEGMENTS {
            return true;
        }
    }
    false
}

fn samples(c: &Contour, extra: &dyn Fn(f64) -> usize) -> Vec<Sample> {
    let n = c.segs.len();
    let mut out = Vec::new();
    for (i, s) in c.segs.iter().enumerate() {
        // The anchor at the segment's start: direction = bisector of the incoming
        // and outgoing tangents (an open contour's first anchor has no incoming).
        let t_out = s.tangent(0.0).unwrap_or(v2(1.0, 0.0));
        let incoming = if i > 0 {
            c.segs[i - 1].tangent(1.0)
        } else if c.closed {
            c.segs[n - 1].tangent(1.0)
        } else {
            None
        };
        let dir = incoming
            .and_then(|t_in| (t_in + t_out).unit())
            .unwrap_or(t_out);
        out.push(Sample { p: s.start(), dir });
        let len = s.len();
        let k = extra(len);
        for j in 1..=k {
            let t = s.t_at_len(len * j as f64 / (k + 1) as f64, len);
            out.push(Sample {
                p: s.eval(t),
                dir: s.tangent(t).unwrap_or(t_out),
            });
        }
    }
    if !c.closed {
        if let Some(last) = c.segs.last() {
            out.push(Sample {
                p: last.end(),
                dir: last.tangent(1.0).unwrap_or(v2(1.0, 0.0)),
            });
        }
    }
    out
}

/// A contour through `pts`: straight (`corner`) or smooth. A smooth contour's
/// handles follow `dirs` when given (Zig Zag: along the path), else the
/// Catmull-Rom tangent (Roughen), with length `handle` × `spans[i]` (the
/// along-path distance to the next point) when given, else × the chord.
fn contour_through(
    pts: &[V2],
    dirs: Option<&[V2]>,
    spans: Option<&[f64]>,
    closed: bool,
    smooth: bool,
    handle: f64,
) -> Contour {
    let n = pts.len();
    let mut c = Contour {
        start: pts[0],
        segs: Vec::with_capacity(n),
        closed,
    };
    let seg_count = if closed { n } else { n - 1 };
    let at = |i: usize| pts[i % n];
    // Unit tangent at point i.
    let tangent = |i: usize| -> V2 {
        if let Some(d) = dirs {
            return d[i % n];
        }
        let prev = if i == 0 {
            if closed {
                pts[n - 1]
            } else {
                pts[0]
            }
        } else {
            pts[i - 1]
        };
        let next = if i + 1 >= n {
            if closed {
                pts[0]
            } else {
                pts[n - 1]
            }
        } else {
            pts[i + 1]
        };
        (next - prev).unit().unwrap_or(v2(0.0, 0.0))
    };
    for i in 0..seg_count {
        let a = at(i);
        let b = at(i + 1);
        if !smooth {
            c.segs.push(Seg::line(a, b));
            continue;
        }
        let d = spans
            .and_then(|s| s.get(i).copied())
            .unwrap_or_else(|| a.dist(b))
            * handle;
        c.segs
            .push(Seg::curve(a, a + tangent(i) * d, b - tangent(i + 1) * d, b));
    }
    c
}

/// SplitMix64: a small, fast, well-mixed stream; identical on every platform.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub fn signed_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn is_relative(mode: &Option<String>) -> bool {
    mode.as_deref() != Some("absolute")
}

fn is_smooth(points: &Option<String>) -> bool {
    points.as_deref() != Some("corner")
}

/// Canvas px from a size and its mode.
fn size_px(size: f64, mode: &Option<String>, bounds: Option<Bounds>) -> f64 {
    let size = if size.is_finite() { size.max(0.0) } else { 0.0 };
    if is_relative(mode) {
        size / 100.0 * bounds.map(|b| b.size()).unwrap_or(0.0)
    } else {
        size
    }
}

// ---------------------------------------------------------------------------
// The effects
// ---------------------------------------------------------------------------

/// Roughen. Each point moves up to the amplitude along x and along y: a
/// relative size's amplitude is size % of the larger side, an absolute size's
/// is half the size (Illustrator 30.8: absolute 10 px moves points at most 5 px,
/// relative 10 % of a 200 × 50 up to 20 px). Handle length for smooth points:
/// 1/3 of the neighbour distance along the Catmull-Rom tangent (half the chord
/// to each side), which keeps the curve inside the displaced points' hull.
pub fn roughen(shape: &Shape, e: &Roughen, bounds: Option<Bounds>, rng_state: &mut u64) -> Shape {
    let size =
        size_px(e.size, &e.size_mode, bounds) * if is_relative(&e.size_mode) { 1.0 } else { 0.5 };
    let detail = if e.detail.is_finite() {
        e.detail.clamp(0.0, 100.0)
    } else {
        0.0
    };
    let smooth = is_smooth(&e.points);
    let extra = |len: f64| (len * detail / INCH).floor() as usize;
    if samples_exceed_budget(shape, &extra) {
        return shape.clone();
    }
    let mut rng = SplitMix64(*rng_state);
    let out = shape
        .iter()
        .filter_map(|c| {
            let pts: Vec<V2> = samples(c, &extra)
                .into_iter()
                .map(|s| s.p + v2(rng.signed_unit() * size, rng.signed_unit() * size))
                .collect();
            (pts.len() >= 2).then(|| contour_through(&pts, None, None, c.closed, smooth, 1.0 / 3.0))
        })
        .collect();
    *rng_state = rng.0;
    out
}

/// Zig Zag. The first point of each subpath moves to the left of travel, the
/// next to the right, and so on; a corner moves along its bisector's normal.
/// Smooth ridges keep handles along the path (a wave); their length is half the
/// along-path spacing between successive points (Illustrator 30.8). A relative
/// size is a percentage of the art's larger side (Illustrator 30.8, its
/// `relAmount`: 20 % of a 100 × 100 square → 20 px, of a 160 × 60 box → 32).
pub fn zig_zag(shape: &Shape, e: &ZigZag, bounds: Option<Bounds>) -> Shape {
    let size = size_px(e.size, &e.size_mode, bounds);
    let ridges = if e.ridges.is_finite() {
        e.ridges.clamp(0.0, 100.0).floor() as usize
    } else {
        0
    };
    let smooth = is_smooth(&e.points);
    if samples_exceed_budget(shape, &|_| ridges) {
        return shape.clone();
    }
    shape
        .iter()
        .filter_map(|c| {
            let ss = samples(c, &|_| ridges);
            if ss.len() < 2 {
                return None;
            }
            let pts: Vec<V2> = ss
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    s.p + s.dir.left_normal() * (size * sign)
                })
                .collect();
            let dirs: Vec<V2> = ss.iter().map(|s| s.dir).collect();
            // Along-path spacing from each point to the next: a segment's length
            // over its ridges + 1 (one entry per output segment).
            let spans: Vec<f64> = c
                .segs
                .iter()
                .flat_map(|s| std::iter::repeat(s.len() / (ridges + 1) as f64).take(ridges + 1))
                .collect();
            Some(contour_through(
                &pts,
                Some(&dirs),
                Some(&spans),
                c.closed,
                smooth,
                0.5,
            ))
        })
        .collect()
}

/// Pucker & Bloat about the centre of the art's bounds: anchors move toward the
/// centre by `amount` (away when negative) and handles the other way. A
/// straight segment's handles start on its anchors, as in Illustrator, so a
/// square bloats into a four-leaf clover and puckers into a four-point star.
pub fn pucker_bloat(shape: &Shape, e: &PuckerBloat, bounds: Option<Bounds>) -> Shape {
    let Some(b) = bounds else {
        return shape.clone();
    };
    let a = if e.amount.is_finite() {
        e.amount.clamp(-200.0, 200.0) / 100.0
    } else {
        0.0
    };
    let c0 = b.center();
    let anchor = |p: V2| p + (c0 - p) * a;
    let handle = |h: V2| h + (h - c0) * a;
    shape
        .iter()
        .map(|c| Contour {
            start: anchor(c.start),
            segs: c
                .segs
                .iter()
                .map(|s| {
                    let [p0, p1, p2, p3] = s.c.0;
                    let (h1, h2) = if s.line { (p0, p3) } else { (p1, p2) };
                    Seg::curve(anchor(p0), handle(h1), handle(h2), anchor(p3))
                })
                .collect(),
            closed: c.closed,
        })
        .collect()
}

/// Twist about the centre of the art's bounds: a point at distance `r` turns by
/// `angle × (1 − r / R)`, R the half diagonal (the corners stay put).
/// Segments are split to a 64th of R first so the curves follow the spiral.
pub fn twist(shape: &Shape, e: &Twist, bounds: Option<Bounds>) -> Shape {
    let Some(b) = bounds else {
        return shape.clone();
    };
    let angle = if e.angle.is_finite() {
        e.angle.clamp(-3600.0, 3600.0).to_radians()
    } else {
        0.0
    };
    let c0 = b.center();
    let r_max = (b.max - c0).len();
    if angle == 0.0 || r_max <= 0.0 {
        return shape.clone();
    }
    let f = |p: V2| {
        let d = p - c0;
        let th = angle * (1.0 - (d.len() / r_max).min(1.0));
        let (s, co) = th.sin_cos();
        c0 + v2(d.x * co - d.y * s, d.x * s + d.y * co)
    };
    map_shape(&subdivide(shape, r_max / 64.0), &f, false)
}

/// A reference point of `b`: `top-left` … `bottom-right`, else the centre.
fn reference_point(b: &Bounds, origin: Option<&str>) -> V2 {
    let (fx, fy) = match origin.unwrap_or("center") {
        "top-left" => (0.0, 0.0),
        "top" => (0.5, 0.0),
        "top-right" => (1.0, 0.0),
        "left" => (0.0, 0.5),
        "right" => (1.0, 0.5),
        "bottom-left" => (0.0, 1.0),
        "bottom" => (0.5, 1.0),
        "bottom-right" => (1.0, 1.0),
        _ => (0.5, 0.5),
    };
    v2(b.min.x + b.width() * fx, b.min.y + b.height() * fy)
}

/// The Transform effect's affine map `p ↦ R + L·(p − R) + move`, as
/// `[a, b, c, d, e, f]` with `x' = a·x + c·y + e`, `y' = b·x + d·y + f`.
/// L = rotate · scale · reflect.
pub fn transform_matrix(e: &TransformEffect, bounds: Option<Bounds>) -> [f64; 6] {
    let fin = |v: Option<f64>, d: f64| v.filter(|v| v.is_finite()).unwrap_or(d);
    let mut sx = fin(e.scale_x, 100.0) / 100.0;
    let mut sy = fin(e.scale_y, 100.0) / 100.0;
    // Illustrator 30.8: Reflect X mirrors x (left and right swap about the
    // reference point), Reflect Y mirrors y.
    if e.reflect_x == Some(true) {
        sx = -sx;
    }
    if e.reflect_y == Some(true) {
        sy = -sy;
    }
    let (s, co) = fin(e.rotate, 0.0).to_radians().sin_cos();
    let (a, b, c, d) = (co * sx, s * sx, -s * sy, co * sy);
    let r = bounds
        .map(|b| reference_point(&b, e.origin.as_deref()))
        .unwrap_or(v2(0.0, 0.0));
    let (mx, my) = (fin(e.move_x, 0.0), fin(e.move_y, 0.0));
    [
        a,
        b,
        c,
        d,
        r.x - (a * r.x + c * r.y) + mx,
        r.y - (b * r.x + d * r.y) + my,
    ]
}

fn apply_affine(m: &[f64; 6], p: V2) -> V2 {
    v2(
        m[0] * p.x + m[2] * p.y + m[4],
        m[1] * p.x + m[3] * p.y + m[5],
    )
}

/// Transform with copies: `copies = 0` transforms the art itself; `n > 0` keeps
/// the original and adds n copies, copy k transformed k times about the
/// ORIGINAL's reference point (so move + rotate walks the copies round a ring,
/// the "repeat" trick).
pub fn transform_copies(
    copies_in: &[Shape],
    e: &TransformEffect,
    bounds: Option<Bounds>,
) -> Vec<Shape> {
    let n = transform_copy_count(e);
    // Random: each step draws its own fraction of every value (Illustrator's
    // Random checkbox: between none and the value), from `seed`.
    let mut rng = SplitMix64(
        (e.seed
            .filter(|s| s.is_finite())
            .unwrap_or(0.0)
            .abs()
            .floor() as u64)
            ^ 0x6D6F_6461_5472_6E73,
    );
    let random = e.random == Some(true);
    let mut step_matrix = || -> [f64; 6] {
        if !random {
            return transform_matrix(e, bounds);
        }
        let mut r = || (rng.signed_unit() + 1.0) / 2.0;
        let lerp =
            |v: Option<f64>, d: f64, t: f64| v.filter(|v| v.is_finite()).map(|v| d + (v - d) * t);
        let jittered = TransformEffect {
            scale_x: lerp(e.scale_x, 100.0, r()),
            scale_y: lerp(e.scale_y, 100.0, r()),
            move_x: lerp(e.move_x, 0.0, r()),
            move_y: lerp(e.move_y, 0.0, r()),
            rotate: lerp(e.rotate, 0.0, r()),
            ..e.clone()
        };
        transform_matrix(&jittered, bounds)
    };
    let mut out = Vec::new();
    for base in copies_in {
        if n == 0 {
            let m = step_matrix();
            out.push(map_shape(base, &|p| apply_affine(&m, p), true));
            continue;
        }
        let mut cur = base.clone();
        out.push(cur.clone());
        for _ in 0..n {
            let m = step_matrix();
            cur = map_shape(&cur, &|p| apply_affine(&m, p), true);
            out.push(cur.clone());
        }
    }
    out
}

/// How many copies a Transform effect adds (0 = it transforms the art itself).
fn transform_copy_count(e: &TransformEffect) -> usize {
    e.copies
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(0.0, MAX_COPIES as f64).floor() as usize)
        .unwrap_or(0)
}

/// Handle length of a Round Corners join, as a fraction of its setback.
const ROUND_CORNERS_HANDLE: f64 = 0.55;

/// Round Corners, Illustrator's way (not a circular fillet): each corner anchor
/// is replaced by two anchors set back `radius` px along each adjacent segment
/// (arc length on a curve), whatever the corner's angle, joined by a cubic whose
/// handles run from each new anchor toward the corner along the segment's
/// tangent there, 0.55 × the setback long. The setback is clamped to half of
/// each adjacent segment. Smooth anchors (tangent in = tangent out) and an open
/// path's ends are left alone. So acute tips stay sharper than a fillet and
/// concave corners round as much as convex ones. (Corner radius on shapes uses
/// the true fillet in `path_geom::round_corners`; this is the effect only.)
pub fn round_corners(shape: &Shape, radius: f64) -> Shape {
    if !(radius > 0.0) || !radius.is_finite() {
        return shape.clone();
    }
    shape.iter().map(|c| round_contour(c, radius)).collect()
}

fn round_contour(c: &Contour, radius: f64) -> Contour {
    let n = c.segs.len();
    if n == 0 {
        return c.clone();
    }
    // Whether the anchor at the END of segment i (the start of segment i + 1) is
    // a corner.
    let corner_after = |i: usize| -> bool {
        if !c.closed && i + 1 >= n {
            return false;
        }
        let a = &c.segs[i];
        let b = &c.segs[(i + 1) % n];
        match (a.tangent(1.0), b.tangent(0.0)) {
            (Some(t_in), Some(t_out)) => t_in.cross(t_out).abs() > 1e-6 || t_in.dot(t_out) < 0.0,
            _ => false,
        }
    };
    let corners: Vec<bool> = (0..n).map(corner_after).collect();
    // Each segment trimmed at a cornered end: (piece, setback at start, setback
    // at end, forward tangent at the piece's start, at its end).
    struct Piece {
        seg: Seg,
        s0: f64,
        s1: f64,
        dir0: V2,
        dir1: V2,
    }
    let pieces: Vec<Piece> = (0..n)
        .map(|i| {
            let s = c.segs[i];
            let len = s.len();
            let start_corner = if i == 0 {
                c.closed && corners[n - 1]
            } else {
                corners[i - 1]
            };
            let s0 = if start_corner {
                radius.min(len / 2.0)
            } else {
                0.0
            };
            let s1 = if corners[i] {
                radius.min(len / 2.0)
            } else {
                0.0
            };
            let t0 = if s0 > 0.0 { s.t_at_len(s0, len) } else { 0.0 };
            let t1 = if s1 > 0.0 {
                s.t_at_len(len - s1, len)
            } else {
                1.0
            };
            let fallback = (s.end() - s.start()).unit().unwrap_or(v2(1.0, 0.0));
            let dir0 = s.tangent(t0).unwrap_or(fallback);
            let dir1 = s.tangent(t1).unwrap_or(fallback);
            // Cut [t0, t1] out of the segment.
            let seg = if t0 <= 0.0 && t1 >= 1.0 {
                s
            } else if s.line {
                Seg::line(s.eval(t0), s.eval(t1))
            } else {
                let (head, _) = s.split(t1);
                if t0 > 0.0 && t1 > 0.0 {
                    head.split(t0 / t1).1
                } else {
                    head
                }
            };
            Piece {
                seg,
                s0,
                s1,
                dir0,
                dir1,
            }
        })
        .collect();
    let mut out = Contour {
        start: pieces[0].seg.start(),
        segs: Vec::with_capacity(2 * n),
        closed: c.closed,
    };
    for i in 0..n {
        let p = &pieces[i];
        if p.seg.len() > 1e-9 {
            out.segs.push(p.seg);
        }
        if corners[i] {
            let q = &pieces[(i + 1) % n];
            let a = p.seg.end();
            let b = q.seg.start();
            out.segs.push(Seg::curve(
                a,
                a + p.dir1 * (ROUND_CORNERS_HANDLE * p.s1),
                b - q.dir0 * (ROUND_CORNERS_HANDLE * q.s0),
                b,
            ));
        }
    }
    out
}

/// Offset Path through the shared offsetter (`path_geom::offset_path`); the
/// result fills even-odd. `None` when the offset erases the art (a shrink past
/// its width) or the solver refuses it.
pub fn offset(shape: &Shape, e: &OffsetPathEffect, closed: bool, even_odd: bool) -> Option<Shape> {
    let mut req = OffsetRequest::new(points_from_contours(shape), closed, e.offset);
    if even_odd {
        req.fill_rule = crate::path_bool::FillRule::EvenOdd;
    }
    if let Some(j) = e.join.as_deref().and_then(OffsetJoin::from_token) {
        req.join = j;
    }
    if let Some(m) = e.miter_limit.filter(|m| m.is_finite() && *m >= 1.0) {
        req.miter_limit = m;
    }
    req.tolerance = 0.05;
    match offset_path(&req) {
        Ok(BooleanOutcome::Shape { points, .. }) => Some(contours_from_points(&points, true)),
        Ok(BooleanOutcome::Empty) => Some(Vec::new()),
        Err(_) => None,
    }
}

/// The run of Offset Path effects `first` starts (with the effects at the head of `rest`) that composes into one
/// offset exactly: every one of them has round joins and an offset of the same sign (and positive when the outline is open, whose first offset
/// is a band on both sides). Growing by `a` then by `b` with round joins is the Minkowski sum with a disc of
/// radius `a`, then one of radius `b`, which is the sum with one disc of radius `a + b`; shrinking is the
/// Minkowski difference, which composes the same way. So the run is one offset by the summed distance (within
/// the offsetter's tolerance of the step-by-step result), at the cost of one offset rather than one per step.
/// Answers the merged effect and how many entries it stands for (1 when nothing merges); the sum stays within
/// the offsetter's [`MAX_OFFSET`](crate::path_geom::offset_path::MAX_OFFSET).
fn merged_offsets(
    first: &OffsetPathEffect,
    rest: &[Effect],
    closed: bool,
) -> (OffsetPathEffect, usize) {
    let round = |o: &OffsetPathEffect| o.join.as_deref() == Some("round");
    let d = first.offset;
    if !round(first) || !d.is_finite() || d == 0.0 || (!closed && d < 0.0) {
        return (first.clone(), 1);
    }
    let mut merged = first.clone();
    let mut used = 1;
    for e in rest {
        let EffectKind::OffsetPath(o) = &e.kind else {
            break;
        };
        let sum = merged.offset + o.offset;
        if !round(o)
            || !o.offset.is_finite()
            || o.offset.signum() != d.signum()
            || o.offset == 0.0
            || sum.abs() > crate::path_geom::offset_path::MAX_OFFSET
        {
            break;
        }
        merged.offset = sum;
        used += 1;
    }
    (merged, used)
}

/// Outline Stroke: the node's stroke (weight, caps, joins, width profile) as a
/// closed nonzero outline. `None` when the node draws no stroke (the geometry
/// is then left as it is).
pub fn outline_stroke(
    shape: &Shape,
    node: &RawNode,
    closed: bool,
    brush: Option<&Value>,
) -> Option<Shape> {
    let points = points_from_contours(shape);
    // The outliner's own refusal, before the node is built: a stack of Outline Strokes past the cap built a
    // 200k-point node per effect only to have it refused (0.2 s each).
    if points.len() > crate::stroke_outline::MAX_POINTS {
        return None;
    }
    let mut m = node.to_json_map();
    m.insert("type".into(), json!("path"));
    m.insert(
        "points".into(),
        Value::Array(points.iter().map(BlendPathPoint::to_json).collect()),
    );
    m.insert("closed".into(), json!(closed));
    m.remove("cornerRadius");
    let opts = OutlineOptions {
        tolerance: 0.1,
        ..OutlineOptions::default()
    };
    match stroke_outline(&RawNode::new(m), brush, &opts) {
        Ok(Outline::Shape { points, .. }) => Some(contours_from_points(&points, true)),
        _ => None,
    }
}

// --- Warp ---

/// The warp presets.
pub const WARP_STYLES: &[&str] = &[
    "arc",
    "arc-lower",
    "arc-upper",
    "arch",
    "bulge",
    "shell-lower",
    "shell-upper",
    "flag",
    "wave",
    "fish",
    "rise",
    "fisheye",
    "inflate",
    "squeeze",
    "twist",
];

/// A warp's control mesh: `rows[j][i]`, row `j` down the box, column `i`
/// across, in box-local px (origin at the box's top-left). Illustrator's warps
/// ("Adobe Deform") are single bicubic Bézier patches over the art's bounds;
/// these meshes reproduce its presets (matched against expanded Illustrator
/// output: `warp_presets_match_illustrator`, `tests/fixtures/illustrator_warp.json`).
type WarpMesh = [[V2; 4]; 4];

fn warp_line_row(a: V2, b: V2) -> [V2; 4] {
    [a, a.lerp(b, 1.0 / 3.0), a.lerp(b, 2.0 / 3.0), b]
}

/// The cubic of a circular arc on the chord `a → b` spanning `2θ`, bulging
/// toward the unit normal `n` (away from it for θ < 0).
fn warp_arc_row(a: V2, b: V2, theta: f64, n: V2) -> [V2; 4] {
    let d = b - a;
    let c = d.len();
    if theta.abs() < 1e-9 || c <= 0.0 {
        return warp_line_row(a, b);
    }
    let e = d * (1.0 / c);
    // Handle length of a cubic arc: 4/3·tan(angle / 4)·radius, radius = (c/2)/sin θ.
    let l = 4.0 / 3.0 * (theta / 2.0).tan() * (c / 2.0) / theta.sin();
    let (st, ct) = theta.sin_cos();
    let t0 = e * ct + n * st;
    let t3 = e * ct - n * st;
    [a, a + t0 * l, b - t3 * l, b]
}

/// Rows linear between `top` and `bottom` (a ruled patch).
fn warp_ruled(top: [V2; 4], bottom: [V2; 4]) -> WarpMesh {
    std::array::from_fn(|j| std::array::from_fn(|i| top[i].lerp(bottom[i], j as f64 / 3.0)))
}

fn warp_identity(w: f64, h: f64) -> WarpMesh {
    warp_ruled(
        warp_line_row(v2(0.0, 0.0), v2(w, 0.0)),
        warp_line_row(v2(0.0, h), v2(w, h)),
    )
}

/// The mesh upside down (y → h − y, rows reversed).
fn warp_flip_v(m: WarpMesh, h: f64) -> WarpMesh {
    std::array::from_fn(|j| m[3 - j].map(|p| v2(p.x, h - p.y)))
}

/// Illustrator's Arc for bend `k` > 0: a fan. The bottom edge keeps its
/// chord and bends into an arc of `k·180°` about a centre below; every row is
/// a concentric arc of the same angle, the sides are radii (so the top edge is
/// the longer, outer arc).
fn warp_arc_fan(k: f64, w: f64, h: f64) -> WarpMesh {
    let th = k * std::f64::consts::FRAC_PI_2;
    let (st, ct) = th.sin_cos();
    let rb = (w / 2.0) / st;
    let c = v2(w / 2.0, h + rb * ct);
    let row = |r: f64| -> [V2; 4] {
        let l = 4.0 / 3.0 * (th / 2.0).tan() * r;
        let p0 = c + v2(-st, -ct) * r;
        let p3 = c + v2(st, -ct) * r;
        [p0, p0 + v2(ct, -st) * l, p3 - v2(ct, st) * l, p3]
    };
    warp_ruled(row(rb + h), row(rb))
}

/// The mesh of a preset over a `w × h` box at bend `k` (−1..1), horizontal
/// orientation; `None` for the presets drawn as a closed form
/// ([`warp_closed_form`]).
fn warp_mesh(style: &str, k: f64, w: f64, h: f64) -> Option<WarpMesh> {
    let th = k * std::f64::consts::FRAC_PI_2;
    let up = v2(0.0, -1.0);
    let down = v2(0.0, 1.0);
    let (tl, tr, bl, br) = (v2(0.0, 0.0), v2(w, 0.0), v2(0.0, h), v2(w, h));
    let top = warp_line_row(tl, tr);
    let bottom = warp_line_row(bl, br);
    // Rows offset vertically by `d[i]` at the columns.
    let shift =
        |row: [V2; 4], d: [f64; 4]| -> [V2; 4] { std::array::from_fn(|i| row[i] + v2(0.0, d[i])) };
    // Flag, fish and rise move the rows by up to twice the bend × the height.
    let d = 2.0 * k * h;
    let m = match style {
        "arc" | "shell-upper" | "shell-lower" => {
            if k.abs() < 1e-9 {
                return Some(warp_identity(w, h));
            }
            // Arc bends up about a centre below; a negative bend is the same
            // fan upside down (the top edge keeps its chord).
            let fan = |k: f64| {
                if k > 0.0 {
                    warp_arc_fan(k, w, h)
                } else {
                    warp_flip_v(warp_arc_fan(-k, w, h), h)
                }
            };
            let id = warp_identity(w, h);
            // Shell Upper: Arc's upper half rows with the lower two rows left
            // straight; Shell Lower: the upside-down fan with the upper two
            // rows straight. A negative bend swaps them (Arc's flip).
            let upper = (style == "shell-upper") == (k > 0.0);
            match style {
                "arc" => fan(k),
                _ if upper => {
                    let mut m = fan(k.abs());
                    m[2] = id[2];
                    m[3] = id[3];
                    m
                }
                _ => {
                    let mut m = fan(-k.abs());
                    m[0] = id[0];
                    m[1] = id[1];
                    m
                }
            }
        }
        // Circular arcs of `k·180°` on the edges' chords.
        "arc-lower" => warp_ruled(top, warp_arc_row(bl, br, th, down)),
        "arc-upper" => warp_ruled(warp_arc_row(tl, tr, th, up), bottom),
        "arch" => warp_ruled(warp_arc_row(tl, tr, th, up), warp_arc_row(bl, br, th, up)),
        "bulge" => warp_ruled(warp_arc_row(tl, tr, th, up), warp_arc_row(bl, br, th, down)),
        "flag" => warp_ruled(
            shift(top, [0.0, d, -d, 0.0]),
            shift(bottom, [0.0, d, -d, 0.0]),
        ),
        "fish" => warp_ruled(
            shift(top, [0.0, -d, d, 0.0]),
            shift(bottom, [0.0, d, -d, 0.0]),
        ),
        "rise" => warp_ruled(
            shift(top, [d, d, 0.0, 0.0]),
            shift(bottom, [d, d, 0.0, 0.0]),
        ),
        _ => return None,
    };
    Some(m)
}

fn bernstein3(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t]
}

fn warp_mesh_eval(m: &WarpMesh, u: f64, v: f64) -> V2 {
    let bu = bernstein3(u);
    let bv = bernstein3(v);
    let mut p = v2(0.0, 0.0);
    for j in 0..4 {
        for i in 0..4 {
            p = p + m[j][i] * (bv[j] * bu[i]);
        }
    }
    p
}

/// The presets whose edges Illustrator leaves in place (Wave, Fisheye, Twist
/// move only the art inside the box) or whose mesh is a plain bulge of every
/// side (Inflate, Squeeze), as closed forms in unit space: `s` across (−1 left
/// … 1 right), `t` down (−1 top … 1 bottom), box aspect `h / w`.
fn warp_closed_form(style: &str, s: f64, t: f64, k: f64, aspect: f64) -> (f64, f64) {
    use std::f64::consts::PI;
    let hump = 1.0 - s * s;
    let vhump = 1.0 - t * t;
    match style {
        // Every side bows out (a parabola: the Illustrator cubic's shape).
        "inflate" => (s * (1.0 + 0.5 * k * vhump), t * (1.0 + 0.5 * k * hump)),
        // The sides bow in, the top and bottom out.
        "squeeze" => (s * (1.0 - 0.5 * k * vhump), t * (1.0 + 0.5 * k * hump)),
        "fisheye" => {
            let r2 = (s * s + t * t).min(1.0);
            let f = 1.0 + 0.5 * k * (1.0 - r2);
            (s * f, t * f)
        }
        // The inside ripples one period across; the box's edges stay put.
        "wave" => (s, t - k * (PI * s).sin() * vhump),
        "twist" => {
            let th = k * PI / 2.0 * hump * vhump;
            let (sn, co) = th.sin_cos();
            // Rotate in true proportions (t is in half-height units).
            let (x, y) = (s, t * aspect);
            let (x2, y2) = (x * co - y * sn, x * sn + y * co);
            (x2, y2 / aspect)
        }
        _ => (s, t),
    }
}

/// Warp over the art's bounds, as Illustrator's Effect ▸ Warp. The art's
/// bounding box maps through the preset's mesh (vertical orientation runs it
/// transposed). Distortion, in the warp's own frame: the across one (Horizontal
/// for a horizontal warp) tapers the box's height across its width before the
/// mesh (a perspective of the parameter space); the down one scales each row
/// of the warped art about its own baseline, wider at one end.
pub fn warp(shape: &Shape, e: &WarpEffect, bounds: Option<Bounds>) -> Shape {
    let Some(b) = bounds else {
        return shape.clone();
    };
    let (w, h) = (b.width(), b.height());
    let Some(f) = warp_map(e, b, false) else {
        return shape.clone();
    };
    map_shape(&subdivide(shape, (w.max(h)) / 48.0), &f, false)
}

/// The Warp preset's point map over the box `b` (see [`warp`]), or `None` when it leaves the art as it is (an empty
/// box, an unknown preset). `edge_wave`: Wave ripples the box's top and bottom edges too (the envelope's Wave,
/// `crate::envelope`), instead of only the art inside them (the Warp effect's, matched to Illustrator's goldens).
pub(crate) fn warp_map(e: &WarpEffect, b: Bounds, edge_wave: bool) -> Option<impl Fn(V2) -> V2> {
    let (w, h) = (b.width(), b.height());
    if !(w > 0.0) && !(h > 0.0) {
        return None;
    }
    let pct = |v: f64| {
        if v.is_finite() {
            v.clamp(-100.0, 100.0) / 100.0
        } else {
            0.0
        }
    };
    let k = pct(e.bend);
    let dh = pct(e.horizontal.unwrap_or(0.0));
    let dv = pct(e.vertical.unwrap_or(0.0));
    let vertical = e.orientation.as_deref() == Some("vertical");
    let style = e.style.clone();
    if !WARP_STYLES.contains(&style.as_str()) {
        return None;
    }
    // The warp's frame: transposed for a vertical warp.
    let tr = move |p: V2| if vertical { v2(p.y, p.x) } else { p };
    let o = b.min;
    let (fw, fh) = if vertical { (h, w) } else { (w, h) };
    let (fw, fh) = (fw.max(1e-9), fh.max(1e-9));
    let (d_across, d_down) = if vertical { (dv, dh) } else { (dh, dv) };
    let mesh = warp_mesh(&style, k, fw, fh);
    let f = move |p: V2| -> V2 {
        let q = tr(p - o);
        let u = q.x / fw;
        let v = q.y / fh;
        let v_in = 0.5 + (v - 0.5) * (1.0 + d_across * (2.0 * u - 1.0));
        let m = match &mesh {
            Some(m) => warp_mesh_eval(m, u, v_in),
            None => {
                let (s2, t2) = if edge_wave && style == "wave" {
                    // The whole box ripples: every row moves by the same sine across.
                    let s = 2.0 * u - 1.0;
                    (s, 2.0 * v_in - 1.0 - k * (std::f64::consts::PI * s).sin())
                } else {
                    warp_closed_form(&style, 2.0 * u - 1.0, 2.0 * v_in - 1.0, k, fh / fw)
                };
                v2((s2 + 1.0) * fw / 2.0, (t2 + 1.0) * fh / 2.0)
            }
        };
        let g = 1.0 + d_down * (2.0 * v - 1.0);
        let base_y = v * fh;
        let r = v2(fw / 2.0 + (m.x - fw / 2.0) * g, base_y + (m.y - base_y) * g);
        tr(r) + o
    };
    Some(f)
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

/// The node's visible geometry effects, in list order.
pub fn geometry_effects(node: &RawNode) -> Vec<Effect> {
    parse_effects(node.json_field("effects"))
        .into_iter()
        .filter(|e| e.is_visible() && e.is_geometry())
        .collect()
}

/// Whether the node has a visible geometry effect (JSON check, no parse of the
/// other effects' payloads).
pub fn has_geometry_effects(node: &RawNode) -> bool {
    crate::effect::node_effects_json(node).iter().any(|e| {
        crate::effect::is_visible_json(e)
            && e.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| crate::effect::GEOMETRY_EFFECT_TYPES.contains(&t))
    })
}

/// Node types a geometry effect reshapes.
pub const GEOMETRY_EFFECT_NODE_TYPES: &[&str] =
    &["rectangle", "ellipse", "star", "polygon", "path"];

/// Apply `effects` (geometry effects, in order) to an outline.
pub fn apply_effects(
    points: &[BlendPathPoint],
    closed: bool,
    even_odd: bool,
    effects: &[Effect],
    node: &RawNode,
    brush: Option<&Value>,
) -> Geometry {
    apply_effects_to_copies(
        vec![contours_from_points(points, closed)],
        closed,
        even_odd,
        effects,
        node,
        brush,
    )
}

/// [`apply_effects`] over several outlines treated as ONE piece of art: every effect measures the bounds of them
/// all (a warp bends a whole text layer, not each run on its own), and [`Geometry::groups`] says which input each
/// resulting copy came from.
pub fn apply_effects_to_copies(
    copies: Vec<Shape>,
    closed: bool,
    even_odd: bool,
    effects: &[Effect],
    node: &RawNode,
    brush: Option<&Value>,
) -> Geometry {
    let n = copies.len();
    let mut g = Geometry {
        copies,
        roles: vec![PaintRole::Both; n],
        groups: (0..n).collect(),
        even_odd,
        closed,
        warnings: Vec::new(),
    };
    let total = effects.len();
    let effects = &effects[..total.min(MAX_GEOMETRY_EFFECTS)];
    if effects.len() < total {
        g.warnings.push(StackWarning::TooManyEffects {
            evaluated: effects.len(),
            total,
        });
    }
    let mut next = 0;
    let mut made = 0usize;
    // Roughen streams: one per Roughen entry, seeded by its `seed`.
    while let Some(e) = effects.get(next) {
        let at = next;
        next += 1;
        if made > MAX_STACK_SEGMENTS {
            g.warnings.push(StackWarning::OverBudget { at, total });
            break;
        }
        let bounds = bounds_of(g.copies.iter());
        let before = g.clone();
        match &e.kind {
            EffectKind::Roughen(r) => {
                let seed = r.seed.filter(|s| s.is_finite()).unwrap_or(0.0);
                let mut state = (seed.abs().floor() as u64) ^ 0x6D6F_6461_5275_6768; // "modaRugh"
                g.copies = g
                    .copies
                    .iter()
                    .map(|s| roughen(s, r, bounds, &mut state))
                    .collect();
            }
            EffectKind::ZigZag(z) => {
                g.copies = g.copies.iter().map(|s| zig_zag(s, z, bounds)).collect();
            }
            EffectKind::PuckerBloat(p) => {
                g.copies = g
                    .copies
                    .iter()
                    .map(|s| pucker_bloat(s, p, bounds))
                    .collect();
            }
            EffectKind::Twist(t) => {
                g.copies = g.copies.iter().map(|s| twist(s, t, bounds)).collect();
            }
            EffectKind::Transform(t) => {
                let per = transform_copy_count(t) + 1;
                // Every copy has the segments of its source, so the refusal below is known before the copies are made.
                if g.segment_count().saturating_mul(per) > MAX_SEGMENTS {
                    g.warnings.push(StackWarning::TooComplex { at, total });
                    break;
                }
                g.copies = transform_copies(&g.copies, t, bounds);
                g.roles = g
                    .roles
                    .iter()
                    .flat_map(|r| std::iter::repeat(*r).take(per))
                    .collect();
                g.groups = g
                    .groups
                    .iter()
                    .flat_map(|k| std::iter::repeat(*k).take(per))
                    .collect();
            }
            EffectKind::RoundCorners(r) => {
                g.copies = g
                    .copies
                    .iter()
                    .map(|s| round_corners(s, r.radius))
                    .collect();
            }
            EffectKind::OffsetPath(o) => {
                let (o, used) = merged_offsets(o, &effects[next..], g.closed);
                next += used - 1;
                let res: Option<Vec<Shape>> = g
                    .copies
                    .iter()
                    .map(|s| offset(s, &o, g.closed, g.even_odd))
                    .collect();
                if let Some(copies) = res {
                    g.copies = copies;
                    g.even_odd = true;
                    g.closed = true;
                }
            }
            EffectKind::OutlineStroke(_) => {
                let res: Option<Vec<Shape>> = g
                    .copies
                    .iter()
                    .map(|s| outline_stroke(s, node, g.closed, brush))
                    .collect();
                if let Some(copies) = res {
                    g.copies = copies;
                    g.even_odd = false;
                    g.closed = true;
                }
            }
            EffectKind::Warp(w) => {
                g.copies = g.copies.iter().map(|s| warp(s, w, bounds)).collect();
            }
            EffectKind::Tweak(t) => {
                let mut rng = distort::tweak_rng(t);
                g.copies = g
                    .copies
                    .iter()
                    .map(|s| distort::tweak(s, t, bounds, &mut rng))
                    .collect();
            }
            EffectKind::FreeDistort(f) => {
                g.copies = g
                    .copies
                    .iter()
                    .map(|s| distort::free_distort(s, f, bounds))
                    .collect();
            }
            EffectKind::ConvertToShape(c) => {
                g.copies = g
                    .copies
                    .iter()
                    .map(|s| distort::convert_to_shape(s, c))
                    .collect();
                g.closed = true;
                g.even_odd = false;
            }
            EffectKind::Scribble(sc) => {
                // Illustrator scribbles the fill and, over it, the stroke's band,
                // each in its own paint.
                let mut rng = distort::scribble_rng(sc);
                let mut copies = Vec::with_capacity(g.copies.len() * 2);
                let mut roles = Vec::with_capacity(g.copies.len() * 2);
                let mut groups = Vec::with_capacity(g.copies.len() * 2);
                for ((s, role), group) in g.copies.iter().zip(&g.roles).zip(&g.groups) {
                    match role {
                        PaintRole::Both => {
                            copies.push(distort::scribble(s, sc, g.even_odd, &mut rng));
                            roles.push(PaintRole::Fill);
                            groups.push(*group);
                            if let Some(band) = outline_stroke(s, node, g.closed, brush) {
                                copies.push(distort::scribble(&band, sc, false, &mut rng));
                                roles.push(PaintRole::Stroke);
                                groups.push(*group);
                            }
                        }
                        other => {
                            copies.push(distort::scribble(s, sc, g.even_odd, &mut rng));
                            roles.push(*other);
                            groups.push(*group);
                        }
                    }
                }
                g.copies = copies;
                g.roles = roles;
                g.groups = groups;
                g.closed = true;
                g.even_odd = false;
            }
            _ => {}
        }
        let finite = g.copies.iter().flatten().all(|c| {
            c.start.is_finite() && c.segs.iter().all(|s| s.c.0.iter().all(|p| p.is_finite()))
        });
        let too_complex = g.segment_count() > MAX_SEGMENTS;
        if too_complex
            || !finite
            || g.roles.len() != g.copies.len()
            || g.groups.len() != g.copies.len()
        {
            g = before;
        }
        if too_complex {
            g.warnings.push(StackWarning::TooComplex { at, total });
            break;
        }
        made = made.saturating_add(g.segment_count());
    }
    g
}

/// Split every curve whose ends meet (a loop: Pucker & Bloat at ±100 % pulls
/// both anchors of a side onto the centre) into its two halves. Tessellators
/// and path consumers read a segment that starts where it ends as empty, so
/// the loop would vanish; the halves draw the same curve.
fn split_loops(shape: &Shape) -> Shape {
    shape
        .iter()
        .map(|c| {
            let mut segs = Vec::with_capacity(c.segs.len());
            for s in &c.segs {
                if !s.line && s.start().dist(s.end()) <= 1e-9 && s.len() > 1e-9 {
                    let (a, b) = s.split(0.5);
                    segs.push(a);
                    segs.push(b);
                } else {
                    segs.push(*s);
                }
            }
            Contour {
                start: c.start,
                segs,
                closed: c.closed,
            }
        })
        .collect()
}

/// Keys that describe a shape's own geometry, dropped when it resolves to a path.
const SHAPE_GEOMETRY_KEYS: &[&str] = &[
    "cornerRadius",
    "startAngle",
    "sweepAngle",
    "innerRadiusRatio",
    "numPoints",
    "sides",
    "distort",
];

/// The node with its visible geometry effects applied: a `path` node (same id,
/// transform, paint and drawing effects) whose `points` are the effected
/// outline, or `None` when the node has no visible geometry effect or its type
/// takes none. `brush` is the definition the node's stroke brush names (Outline
/// Stroke outlines a calligraphic stroke with its nib).
///
/// Width and height stay (paint that maps to the node box keeps its box); a
/// shape's own geometry keys, its corner radius and a 4-corner `distort` go
/// (the effected outline replaces them).
pub fn resolve_node(node: &RawNode, brush: Option<&Value>) -> Option<RawNode> {
    if !GEOMETRY_EFFECT_NODE_TYPES.contains(&node.node_type()) || !has_geometry_effects(node) {
        return None;
    }
    let effects = geometry_effects(node);
    let (points, closed) = node_outline(node)?;
    let even_odd = node.node_type() == "path" && node.str("fillRule") == Some("evenodd");
    let g = apply_effects(&points, closed, even_odd, &effects, node, brush);
    Some(RawNode::new(resolved_map(node, &g)))
}

/// [`resolve_node`] with the stroke brush looked up in the document's brush table.
pub fn resolve_node_in(node: &RawNode, doc: &crate::types::SceneDocument) -> Option<RawNode> {
    if !has_geometry_effects(node) {
        return None;
    }
    let brush = crate::paint::model_stroke_style(node)
        .brush
        .as_ref()
        .and_then(|b| b.get("brushId"))
        .and_then(Value::as_str)
        .and_then(|id| doc.brush(id));
    resolve_node(node, brush)
}

/// The resolved node's JSON for geometry `g` (see [`resolve_node`]).
pub fn resolved_map(node: &RawNode, g: &Geometry) -> Map {
    let mut m = node.fields().clone();
    m.insert("type".into(), json!("path"));
    for k in SHAPE_GEOMETRY_KEYS {
        m.remove(*k);
    }
    let mut all = Vec::new();
    let mut counts = Vec::with_capacity(g.copies.len());
    for shape in &g.copies {
        let pts = points_from_contours(&split_loops(shape));
        counts.push(pts.len());
        all.extend(pts.iter().map(BlendPathPoint::to_json));
    }
    m.insert("points".into(), Value::Array(all));
    m.insert("closed".into(), json!(g.closed));
    m.insert(
        "fillRule".into(),
        json!(if g.even_odd { "evenodd" } else { "nonzero" }),
    );
    let plain = g.roles.iter().all(|r| *r == PaintRole::Both);
    if counts.len() > 1 || !plain {
        m.insert(EFFECT_COPIES_KEY.into(), json!(counts));
    } else {
        m.remove(EFFECT_COPIES_KEY);
    }
    if g.warnings.is_empty() {
        m.remove(EFFECT_WARNINGS_KEY);
    } else {
        let warnings = g.warnings.iter().map(|w| w.to_json()).collect();
        m.insert(EFFECT_WARNINGS_KEY.into(), Value::Array(warnings));
    }
    if plain {
        m.remove(EFFECT_COPY_ROLES_KEY);
    } else {
        let roles: Vec<&str> = g.roles.iter().map(|r| r.as_str()).collect();
        m.insert(EFFECT_COPY_ROLES_KEY.into(), json!(roles));
    }
    // The drawing effects stay; the geometry ones are baked.
    let kept: Vec<Value> = crate::effect::node_effects_json(node)
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
    m
}

/// A resolved node's copies as separate path nodes (the body lane draws them in
/// turn), or `None` when it has one copy. Each copy keeps the node's id.
pub fn split_copies(node: &RawNode) -> Option<Vec<RawNode>> {
    let counts: Vec<usize> = node
        .json_field(EFFECT_COPIES_KEY)?
        .as_array()?
        .iter()
        .filter_map(Value::as_u64)
        .map(|n| n as usize)
        .collect();
    let roles = copy_roles(node);
    if counts.len() < 2 && roles.iter().all(|r| *r == PaintRole::Both) {
        return None;
    }
    let points = node.points()?.json();
    let fills = node.array(EFFECT_COPY_FILLS_KEY);
    // Every copy's record, less its points: cloned once per copy, so it must not hold every copy's points (a
    // Transform stack's 4,096 copies cloned 20,480 points each, 40 s).
    let mut template = node.fields().clone();
    template.remove(EFFECT_COPIES_KEY);
    template.remove(EFFECT_COPY_ROLES_KEY);
    template.remove(EFFECT_COPY_FILLS_KEY);
    template.remove(EFFECT_WARNINGS_KEY);
    template.insert("points".into(), Value::Array(Vec::new()));
    let mut out = Vec::with_capacity(counts.len());
    let mut at = 0usize;
    for (k, n) in counts.into_iter().enumerate() {
        let end = (at + n).min(points.len());
        let mut m = template.clone();
        m.insert("points".into(), Value::Array(points[at..end].to_vec()));
        if let Some(f) = fills.and_then(|f| f.get(k)) {
            m.insert("fills".into(), f.clone());
        }
        apply_paint_role(&mut m, roles.get(k).copied().unwrap_or(PaintRole::Both));
        out.push(RawNode::new(m));
        at = end;
    }
    Some(out)
}

/// A resolved node's per-copy paint roles (empty when every copy draws both).
pub fn copy_roles(node: &RawNode) -> Vec<PaintRole> {
    node.json_field(EFFECT_COPY_ROLES_KEY)
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|v| PaintRole::parse(v.as_str().unwrap_or("")))
                .collect()
        })
        .unwrap_or_default()
}

/// A copy's paint for its role: `Fill` drops the strokes; `Stroke` fills the
/// copy (the stroke's band, already outlined) with the strokes' paints, the
/// top stroke on top, and drops the strokes. Mirrored by `applyCopyRole`
/// (`core/geometry/liveEffects.ts`).
pub fn apply_paint_role(m: &mut Map, role: PaintRole) {
    match role {
        PaintRole::Both => {}
        PaintRole::Fill => {
            m.remove("strokes");
        }
        PaintRole::Stroke => {
            let fills: Vec<Value> = m
                .get("strokes")
                .and_then(Value::as_array)
                .map(|strokes| {
                    strokes
                        .iter()
                        .filter_map(|s| {
                            let s = s.as_object()?;
                            let mut f = Map::new();
                            for k in ["id", "visible", "opacity", "blendMode", "paint"] {
                                if let Some(v) = s.get(k) {
                                    f.insert(k.into(), v.clone());
                                }
                            }
                            f.contains_key("paint").then_some(Value::Object(f))
                        })
                        .collect()
                })
                .unwrap_or_default();
            m.remove("strokes");
            if fills.is_empty() {
                m.remove("fills");
            } else {
                m.insert("fills".into(), Value::Array(fills));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON entry point (the `geometry_effects` wasm export)
// ---------------------------------------------------------------------------

/// `{"node": Node, "brush"?: BrushDefinition, "effects"?: Effect[]}` →
/// `{"status":"ok", "node": resolved path node, "copies": [path points per copy],
/// "fillRule", "closed", "bounds": {minX,minY,maxX,maxY}}`, `{"status":"none"}`
/// (no visible geometry effect, or a node type that takes none) or
/// `{"status":"error","code","message"}`. `effects`, when given, replaces the
/// node's list (a preview of an edit). An `ok` reply for a stack that stopped
/// before its end carries `"warnings": [{"code", "message", …}]` ([`StackWarning`]:
/// `effect_stack_truncated` past [`MAX_GEOMETRY_EFFECTS`], `effect_too_complex`
/// past [`MAX_SEGMENTS`]); other replies have no `warnings` key. Contract:
/// `packages/editor/src/core/geometry/liveEffects.ts`.
pub fn geometry_effects_json(request_json: &str) -> String {
    let err = |code: &str, message: &str| {
        json!({"status": "error", "code": code, "message": message}).to_string()
    };
    let value: Value = match crate::value::from_str(request_json) {
        Ok(v) => v,
        Err(e) => return err("invalid_json", &format!("request is not valid JSON: {e}")),
    };
    let Some(node) = value.get("node").and_then(Value::as_object) else {
        return err("invalid_request", "request.node must be an object");
    };
    let mut node = RawNode::new(node.clone());
    if let Some(effects) = value.get("effects") {
        node.set("effects".into(), effects.clone());
    }
    let brush = value.get("brush").filter(|b| b.is_object());
    let resolved = if node.node_type() == "richtext" {
        // A text layer: its glyph outlines, effected (`crate::text_warp`), laid out on its own.
        let area = value.get("area").filter(|a| a.is_object());
        match crate::text_warp::resolve_standalone(&node, area) {
            crate::text_warp::TextWarp::Resolved(r) => r,
            crate::text_warp::TextWarp::None => return json!({"status": "none"}).to_string(),
            crate::text_warp::TextWarp::Pending => {
                return err(
                    "text_pending",
                    "the text cannot be outlined yet (its fonts are not loaded)",
                )
            }
        }
    } else {
        let Some(resolved) = resolve_node(&node, brush) else {
            return json!({"status": "none"}).to_string();
        };
        resolved
    };
    // Per-copy fills (a warped text layer's run fills), one entry per copy; absent when every copy wears the node's.
    let copy_fills = resolved.json_field(EFFECT_COPY_FILLS_KEY).cloned();
    let warnings = resolved.json_field(EFFECT_WARNINGS_KEY).cloned();
    let copies: Vec<Value> = match split_copies(&resolved) {
        Some(parts) => parts
            .iter()
            .map(|p| {
                p.points()
                    .map_or(json!([]), |a| Value::Array(a.json().into_owned()))
            })
            .collect(),
        None => vec![resolved
            .points()
            .map_or(json!([]), |a| Value::Array(a.json().into_owned()))],
    };
    let roles: Vec<&str> = {
        let r = copy_roles(&resolved);
        if r.is_empty() {
            vec!["both"; copies.len()]
        } else {
            r.iter().map(|r| r.as_str()).collect()
        }
    };
    let all: Vec<BlendPathPoint> = resolved
        .points()
        .map(|a| a.iter().filter_map(BlendPathPoint::from_ref).collect())
        .unwrap_or_default();
    let closed = resolved.bool("closed").unwrap_or(false);
    let shape = contours_from_points(&all, closed);
    let bounds = bounds_of(std::iter::once(&shape))
        .map(|b| json!({"minX": b.min.x, "minY": b.min.y, "maxX": b.max.x, "maxY": b.max.y}));
    let mut reply = json!({
        "status": "ok",
        "fillRule": resolved.str("fillRule").unwrap_or("nonzero"),
        "closed": closed,
        "copies": copies,
        "copyRoles": roles,
        "copyFills": copy_fills,
        "bounds": bounds,
        "node": Value::Object({
            let mut m = resolved.into_json_map();
            m.remove(EFFECT_COPY_FILLS_KEY);
            m.remove(EFFECT_WARNINGS_KEY);
            m
        }),
    });
    if let Some(w) = warnings {
        reply["warnings"] = w;
    }
    reply.to_string()
}

#[cfg(test)]
#[path = "live_effects_tests.rs"]
mod tests;
