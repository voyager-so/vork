//! Bristle brush (Illustrator's fifth brush type): a paintbrush of `n`
//! bristles whose stroke is `n` overlapping, filled, translucent strand paths
//! — thin closed ribbons that follow the stroke, drag round its corners and
//! overshoot its end.
//!
//! # The rules measured in Illustrator (30.8, stock Bristle Brush Library,
//! mouse input, Expand Appearance)
//!
//! - **Strand count** `n = round(36 × density)`, whatever the size, length or
//!   shape.
//! - **Strand opacity** `a = 1 − (1 − paintOpacity)^(1/n)`: where every strand
//!   overlaps the stack reaches exactly the paint opacity. Strands are plain
//!   filled paths (normal blend) in the stroke's colour, so one stroke never
//!   builds up on itself beyond that, separate strokes do.
//! - **The stroke weight scales everything**; the brush's own unit is its
//!   diameter at the ferrule `D = size (mm) × 72 / 25.4 × weight`. (That is
//!   Illustrator's sizing, `scaleWithWeight`; by default the band `spread × D`
//!   is as wide as the stroke instead, `D = weight / spread`.)
//! - **Side bias**: with a mouse the brush leans, so the strands sit mostly to
//!   the RIGHT of the direction of travel: the band runs from `−0.15 S` to
//!   `+0.85 S` across the path, `S` the spread (`≈ 1.2 D` at bristle length
//!   100 %, growing with length; flat and fan tips are wider, angled ones
//!   narrower and fully to one side).
//! - **Ends**: strands start 0–0.6 D after the path's start and overshoot its
//!   end by up to `D × (0.35 + 0.55 × length)`; a pointed tip tapers them,
//!   blunt and fan tips round them off.
//! - Strands **lag at corners** (bristles drag round a turn): each strand is
//!   an exponential moving average of its raw offset point along the path,
//!   with a lag length that grows with the bristle length and shrinks with
//!   stiffness.
//!
//! Exact geometry parity is not the goal (Illustrator's model is a physical
//! simulation); the counts and opacities are exact, the extents land in the
//! measured ranges (tests below), the look is close.
//!
//! # Pen channels (`inkChannels`)
//!
//! - **pressure** spreads the bristles (low pressure = a narrower band) and
//!   lifts the outer strands off the paper first, so the stack thins out and
//!   the stroke gets lighter;
//! - **tilt + bearing** replace the mouse's fixed lean: the band leans to the
//!   side the pen leans towards (across the path), more with more tilt, and
//!   splays wider;
//! - **rotation** turns a flat tip: the band narrows to 30 % when the flat
//!   side runs along the path.
//!
//! Everything is deterministic (the stroke's `seed`), total (no panics) and
//! budgeted: at most [`MAX_FEED_SAMPLES`] samples per placement (longer paths
//! sample more coarsely) and [`MAX_OUTPUT_SEGMENTS`] output segments.

use crate::value::{json, Map, Value};

use super::common::{
    error, node_stroke, parse_request, placement_json, subpath_length, with_json_cap, BrushStroke,
    PlacedAt, Placement, Request, MAX_OUTPUT_SEGMENTS,
};
use super::dyn_value::{draw, ink_at, InkAt};
use super::geom::{Bounds, Subpath};
use super::spine::Spine;
use crate::path_geom::vec2::{v2, Cubic, V2};
use crate::stroke_outline::ArcWindow;

/// Points per millimetre (1 px = 1 pt at 100 %).
pub const MM: f64 = 72.0 / 25.4;
/// Most strands a brush has (`density` 100 %).
pub const MAX_STRANDS: usize = 36;
/// Most centerline samples per placement, over every strand and subpath.
pub const MAX_FEED_SAMPLES: usize = 150_000;

const SALT_POS: u64 = 0xB1;
const SALT_POS2: u64 = 0xB2;
const SALT_WIDTH: u64 = 0xB3;
const SALT_START: u64 = 0xB4;
const SALT_END: u64 = 0xB5;
const SALT_LAG: u64 = 0xB6;
const SALT_LIFT: u64 = 0xB7;
const SALT_WOBBLE: u64 = 0xB8;
const SALT_PHASE: u64 = 0xB9;
const SALT_TAIL: u64 = 0xBA;
/// Fewest samples along one strand (budget permitting).
const MIN_STRAND_SAMPLES: usize = 48;

#[cfg(test)]
thread_local! {
    /// Tests: sample every strand at least this many times instead.
    pub(crate) static MIN_SAMPLES_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn min_strand_samples() -> usize {
    #[cfg(test)]
    if let Some(n) = MIN_SAMPLES_OVERRIDE.with(|c| c.get()) {
        return n;
    }
    MIN_STRAND_SAMPLES
}
/// The strand's position lag, as a fraction of its heading lag.
const POS_LAG: f64 = 0.2;
/// How far the tips trail the ferrule, as a fraction of the heading lag.
const TRAIL: f64 = 0.3;

/// The ten tip shapes, as `BristleShape` ids (nodes.ts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BristleShape {
    RoundPoint,
    FlatPoint,
    RoundBlunt,
    FlatBlunt,
    RoundCurve,
    FlatCurve,
    RoundAngle,
    FlatAngle,
    RoundFan,
    FlatFan,
}

/// How a tip's bristles are cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Point,
    Blunt,
    Curve,
    Angle,
    Fan,
}

impl BristleShape {
    pub const ALL: [BristleShape; 10] = [
        BristleShape::RoundPoint,
        BristleShape::FlatPoint,
        BristleShape::RoundBlunt,
        BristleShape::FlatBlunt,
        BristleShape::RoundCurve,
        BristleShape::FlatCurve,
        BristleShape::RoundAngle,
        BristleShape::FlatAngle,
        BristleShape::RoundFan,
        BristleShape::FlatFan,
    ];

    pub fn id(self) -> &'static str {
        match self {
            BristleShape::RoundPoint => "round-point",
            BristleShape::FlatPoint => "flat-point",
            BristleShape::RoundBlunt => "round-blunt",
            BristleShape::FlatBlunt => "flat-blunt",
            BristleShape::RoundCurve => "round-curve",
            BristleShape::FlatCurve => "flat-curve",
            BristleShape::RoundAngle => "round-angle",
            BristleShape::FlatAngle => "flat-angle",
            BristleShape::RoundFan => "round-fan",
            BristleShape::FlatFan => "flat-fan",
        }
    }

    pub fn parse(s: &str) -> Option<BristleShape> {
        Self::ALL.into_iter().find(|b| b.id() == s)
    }

    pub fn flat(self) -> bool {
        matches!(
            self,
            BristleShape::FlatPoint
                | BristleShape::FlatBlunt
                | BristleShape::FlatCurve
                | BristleShape::FlatAngle
                | BristleShape::FlatFan
        )
    }

    pub fn profile(self) -> Profile {
        match self {
            BristleShape::RoundPoint | BristleShape::FlatPoint => Profile::Point,
            BristleShape::RoundBlunt | BristleShape::FlatBlunt => Profile::Blunt,
            BristleShape::RoundCurve | BristleShape::FlatCurve => Profile::Curve,
            BristleShape::RoundAngle | BristleShape::FlatAngle => Profile::Angle,
            BristleShape::RoundFan | BristleShape::FlatFan => Profile::Fan,
        }
    }
}

/// `BristleBrushDefinition` (nodes.ts), normalised like
/// `normalizeBrushDefinition`: size in mm 1..10, length 0.25..3, the rest
/// 0.01..1.
#[derive(Debug, Clone, PartialEq)]
pub struct BristleBrush {
    pub id: String,
    pub name: String,
    pub shape: BristleShape,
    pub size: f64,
    pub length: f64,
    pub density: f64,
    pub thickness: f64,
    pub opacity: f64,
    pub stiffness: f64,
    /// `scaleWithWeight`: Illustrator's sizing — the stroke weight multiplies
    /// the brush's size (mm). Without it (the default, as in Figma) the
    /// stroke width is the width of the mark: the design is scaled to it.
    pub scale_with_weight: bool,
}

/// Defaults of a new bristle brush (Illustrator's).
pub const DEFAULT_SHAPE: BristleShape = BristleShape::RoundPoint;
pub const DEFAULT_SIZE: f64 = 3.0;
pub const DEFAULT_LENGTH: f64 = 1.0;
pub const DEFAULT_DENSITY: f64 = 0.33;
pub const DEFAULT_THICKNESS: f64 = 0.5;
pub const DEFAULT_OPACITY: f64 = 0.75;
pub const DEFAULT_STIFFNESS: f64 = 0.5;

fn field(o: &Map, k: &str, fallback: f64, lo: f64, hi: f64) -> f64 {
    o.get(k)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .unwrap_or(fallback)
        .clamp(lo, hi)
}

pub fn parse_bristle_brush(raw: &Value) -> Option<BristleBrush> {
    let o = raw.as_object()?;
    let id = o
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    if o.get("type").and_then(Value::as_str) != Some("bristle") {
        return None;
    }
    Some(BristleBrush {
        id: id.to_string(),
        name: o
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .to_string(),
        shape: o
            .get("shape")
            .and_then(Value::as_str)
            .and_then(BristleShape::parse)
            .unwrap_or(DEFAULT_SHAPE),
        size: field(o, "size", DEFAULT_SIZE, 1.0, 10.0),
        length: field(o, "length", DEFAULT_LENGTH, 0.25, 3.0),
        density: field(o, "density", DEFAULT_DENSITY, 0.01, 1.0),
        thickness: field(o, "thickness", DEFAULT_THICKNESS, 0.01, 1.0),
        opacity: field(o, "opacity", DEFAULT_OPACITY, 0.01, 1.0),
        stiffness: field(o, "stiffness", DEFAULT_STIFFNESS, 0.01, 1.0),
        scale_with_weight: super::common::scale_with_weight(o),
    })
}

impl BristleBrush {
    /// `round(36 × density)`, at least one.
    pub fn strand_count(&self) -> usize {
        ((MAX_STRANDS as f64 * self.density).round() as usize).clamp(1, MAX_STRANDS)
    }

    /// Each strand's opacity: `n` of them stack to exactly the paint opacity.
    pub fn strand_opacity(&self) -> f64 {
        let n = self.strand_count() as f64;
        1.0 - (1.0 - self.opacity.clamp(0.0, 1.0)).powf(1.0 / n)
    }

    /// The band's width across the path in units of the diameter `D`.
    pub fn spread(&self) -> f64 {
        let mut s = 0.7 + 0.5 * self.length;
        if self.shape.flat() {
            s *= 1.15;
        }
        match self.shape.profile() {
            Profile::Fan => s * 1.25,
            Profile::Angle => s * 0.8,
            _ => s,
        }
    }

    /// How much later along the path a bristle touches down (and lifts off)
    /// per unit of its offset right of the path: the footprint's slant.
    pub fn shear(&self) -> f64 {
        match self.shape.profile() {
            Profile::Point | Profile::Curve => 0.55,
            Profile::Blunt => 0.6,
            Profile::Angle => 0.6,
            Profile::Fan => 0.3,
        }
    }

    /// Largest overshoot past the path's end, in units of `D`.
    pub fn overshoot(&self) -> f64 {
        0.3 + 0.47 * self.length
    }
}

/// One bristle's per-stroke draws (band units across, `D` units along).
#[derive(Debug, Clone, Copy)]
struct Strand {
    /// Centre across the band, 0 (near edge) .. 1 (far edge).
    u: f64,
    /// Width, as a fraction of the band.
    w: f64,
    /// Where it starts after the path's start (negative: before it).
    start: f64,
    /// How far past the path's end it runs.
    end: f64,
    /// Lag length (corner drag).
    lag: f64,
    /// Pressure below which it lifts off the paper.
    lift: f64,
    /// Lateral wander, fraction of the band, and its wavelength (`D` units).
    wob: f64,
    wob_len: f64,
    phase: f64,
    /// End taper length and the caps' slant, as multiples of the width.
    tail: f64,
    cut: f64,
}

fn strands(brush: &BristleBrush, seed: u32, subpath: usize) -> Vec<Strand> {
    let n = brush.strand_count();
    let profile = brush.shape.profile();
    let thick = 0.7 + 0.6 * brush.thickness;
    let len = brush.length;
    let soft = 1.15 - brush.stiffness;
    (0..n)
        .map(|i| {
            let k = ((subpath as u64) << 8) | i as u64;
            let d = |salt: u64| draw(seed, salt, k);
            let (r1, r2, rw) = (d(SALT_POS), d(SALT_POS2), d(SALT_WIDTH));
            // Triangular around the middle: more bristles in the belly.
            let tri = (r1 + r2) / 2.0;
            // (width, centre, end taper and slant as multiples of the width)
            let (w, u, tail, cut) = match profile {
                Profile::Point => {
                    let w = (0.22 + 0.45 * rw) * thick;
                    (w, 0.5 + (tri - 0.5) * 0.75, 1.5 + 1.5 * d(SALT_TAIL), 0.7)
                }
                Profile::Blunt => {
                    let w = (0.35 + 0.35 * rw) * thick;
                    (w, 0.5 + (r1 - 0.5) * 0.5, 0.3 + 0.2 * d(SALT_TAIL), 0.35)
                }
                Profile::Curve => {
                    let w = (0.3 + 0.4 * rw) * thick;
                    (w, 0.5 + (tri - 0.5) * 0.9, 0.8 + 0.5 * d(SALT_TAIL), 0.5)
                }
                Profile::Angle => {
                    // Anchored at the near edge: a staircase of widths.
                    let w = (0.12 + 0.88 * rw) * thick.min(1.0);
                    (w, w / 2.0 + 0.04 * (r1 - 0.5), 0.5 + 0.4 * d(SALT_TAIL), 0.9)
                }
                Profile::Fan => {
                    let w = (0.2 + 0.3 * rw) * thick;
                    (w, r1, 0.4 + 0.3 * d(SALT_TAIL), 0.25)
                }
            };
            let w = w.clamp(0.04, 0.95);
            let u = u.clamp(w / 2.0, 1.0 - w / 2.0);
            // How far from the middle (0) to the edge (1).
            let edge = ((u - 0.5).abs() * 2.0).min(1.0);
            let reach = match profile {
                // Pointed / curved tips: the outer bristles are shorter.
                Profile::Point => 1.0 - 0.6 * edge,
                Profile::Curve => 1.0 - 0.35 * edge,
                Profile::Fan => 1.0 - 0.25 * edge,
                // Angled: longer towards the far edge.
                Profile::Angle => 0.55 + 0.2 * u,
                Profile::Blunt => 1.0,
            };
            Strand {
                u,
                w,
                start: len * (-0.2 + 0.6 * d(SALT_START)),
                end: brush.overshoot() * reach * (0.5 + 0.5 * d(SALT_END)),
                lag: len * soft * (0.35 + 0.65 * d(SALT_LAG)),
                lift: 0.5 * edge * (0.3 + 0.7 * d(SALT_LIFT)),
                wob: 0.006 * (1.6 - brush.stiffness) * d(SALT_WOBBLE),
                wob_len: 8.0 + 12.0 * d(SALT_PHASE),
                phase: std::f64::consts::TAU * d(SALT_PHASE ^ 0x5A),
                tail,
                cut,
            }
        })
        .collect()
}

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The band's pose at one point: `(spread factor, lean, pressure)`; `lean` is
/// where the band's centre sits across the path as a fraction of the spread
/// (right of travel positive).
fn pose(brush: &BristleBrush, ink: &InkAt, right: V2) -> (f64, f64, Option<f64>) {
    let mut spread = 1.0;
    // The mouse's fixed lean: the band from −0.15 S to +0.85 S.
    let mut lean = 0.35;
    if let Some(p) = ink.pressure {
        spread *= 0.45 + 0.55 * p;
    }
    if let (Some(tilt), Some(bearing)) = (ink.tilt, ink.bearing) {
        let k = tilt.to_radians().sin().clamp(0.0, 1.0);
        // Bearing is counter-clockwise on screen from +x (y-down).
        let (s, c) = bearing.to_radians().sin_cos();
        let side = v2(c, -s).dot(right);
        lean = 0.5 * (side * k * 1.4).clamp(-1.0, 1.0);
        spread *= 1.0 + 0.4 * k;
    }
    if brush.shape.flat() {
        if let Some(r) = ink.rotation {
            spread *= 0.3 + 0.7 * r.to_radians().cos().abs();
        }
    }
    (spread, lean, ink.pressure)
}

/// One strand's edges, sample by sample, with the feed distance of each.
struct Ribbon {
    feed: Vec<f64>,
    left: Vec<V2>,
    right: Vec<V2>,
}

/// Remove samples that lie within `tol` of the chord between their kept
/// neighbours (Douglas–Peucker on the index range), keeping `must` indices.
fn simplify(points: &[V2], tol: f64) -> Vec<usize> {
    let n = points.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (pa, pb) = (points[a], points[b]);
        let d = pb - pa;
        let l = d.len();
        let mut worst = (0.0, a);
        for (i, p) in points.iter().enumerate().take(b).skip(a + 1) {
            let dist = if l > 1e-12 {
                (d.cross(*p - pa) / l).abs()
            } else {
                p.dist(pa)
            };
            if dist > worst.0 {
                worst = (dist, i);
            }
        }
        if worst.0 > tol {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    (0..n).filter(|i| keep[*i]).collect()
}

/// A closed smooth outline through `pts` (chord-length Catmull–Rom handles);
/// a vertex turning more than ~40° stays a corner (straight handles), so a
/// long simplified run next to a sharp turn never bulges.
fn closed_curve(pts: &[V2]) -> Option<Subpath> {
    let m = pts.len();
    if m < 3 {
        return None;
    }
    const COS_CORNER: f64 = 0.766;
    // (direction into vertex i's outgoing segment, into its incoming one)
    let handles = |i: usize| -> (V2, V2) {
        let prev = pts[(i + m - 1) % m];
        let here = pts[i];
        let next = pts[(i + 1) % m];
        let (d_in, d_out) = ((here - prev).unit(), (next - here).unit());
        match (d_in, d_out) {
            (Some(a), Some(b)) if a.dot(b) >= COS_CORNER => {
                let t = (next - prev).unit().unwrap_or(b);
                (t, t)
            }
            (a, b) => (b.unwrap_or_default(), a.unwrap_or_default()),
        }
    };
    let mut segs = Vec::with_capacity(m);
    for i in 0..m {
        let (a, b) = (pts[i], pts[(i + 1) % m]);
        let l = a.dist(b) / 3.0;
        let out_a = handles(i).0;
        let in_b = handles((i + 1) % m).1;
        segs.push(Cubic([a, a + out_a * l, b - in_b * l, b]));
    }
    Some(Subpath { segs, closed: true })
}

/// The feed intervals a window list keeps on one subpath (spine units);
/// `None` = all of it. A window reaching a path end keeps the strands' run
/// past that end.
fn feed_intervals(
    windows: Option<&[ArcWindow]>,
    si: usize,
    to_spine: f64,
    lp: f64,
    closed: bool,
) -> Option<Vec<(f64, f64)>> {
    let ws = windows?;
    const EPS: f64 = 1e-6;
    let mut out = Vec::new();
    for w in ws.iter().filter(|w| w.subpath == si) {
        let (mut a, mut b) = (w.start * to_spine, w.end * to_spine);
        if !(a.is_finite() && b.is_finite()) || b <= a {
            continue;
        }
        if closed {
            if a <= EPS && b >= lp - EPS {
                return None;
            }
            for k in [-1.0, 0.0, 1.0] {
                out.push((a + k * lp, b + k * lp));
            }
        } else {
            if a <= EPS {
                a = f64::NEG_INFINITY;
            }
            if b >= lp - EPS {
                b = f64::INFINITY;
            }
            out.push((a, b));
        }
    }
    Some(out)
}

/// Place a bristle brush along a stroke. Pure and total.
pub fn bristle_brush_geometry(
    stroke: &BrushStroke,
    brush: &BristleBrush,
    stroke_opacity: f64,
    windows: Option<&[ArcWindow]>,
) -> Placement {
    let mut out = Placement {
        bounds: Bounds::empty(),
        ..Placement::default()
    };
    let weight = if stroke.stroke_width.is_finite() {
        stroke.stroke_width.max(0.0)
    } else {
        0.0
    };
    // Sizing: under `scaleWithWeight` the diameter is size (mm) × weight
    // (Illustrator); otherwise the band (`spread × D`) is as wide as the
    // stroke and `size` doesn't enter.
    let dia = if brush.scale_with_weight {
        brush.size * MM * weight
    } else {
        weight / brush.spread()
    };
    if dia <= 1e-9 || !dia.is_finite() {
        return out;
    }
    let total_len: f64 = stroke.subpaths.iter().map(subpath_length).sum();
    if !total_len.is_finite() || total_len <= 1e-9 {
        return out;
    }
    let colour = stroke
        .stroke_color
        .and_then(Value::as_str)
        .unwrap_or("#000000")
        .to_string();
    let opacity = brush.strand_opacity() * stroke_opacity.clamp(0.0, 1.0);
    let mut paint = Map::new();
    paint.insert(
        "fills".into(),
        json!([{"id": "fill-0", "paint": {"type": "solid", "color": colour}}]),
    );
    paint.insert("opacity".into(), json!(opacity));
    let spread = brush.spread() * dia;
    let edge_shift = if brush.shape.profile() == Profile::Angle {
        0.27
    } else {
        0.0
    };
    let n = brush.strand_count();
    let shear = brush.shear();
    // Sample step: a third of the diameter, coarser when the budget needs it.
    let per_strand_extra = (brush.overshoot() + 1.2 * brush.length + 0.5) * dia;
    let feed_total = (total_len + per_strand_extra * stroke.subpaths.len() as f64) * n as f64;
    let step = (dia * 0.3)
        .max(0.2)
        .max(feed_total / MAX_FEED_SAMPLES as f64);
    let tol = (dia * 0.05).max(0.05);
    let floor_samples = min_strand_samples() * n * stroke.subpaths.len();
    let uses_ink = stroke.ink.is_some();
    let mut segments = 0usize;
    let mut cum = 0.0;
    for (si, sp) in stroke.subpaths.iter().enumerate() {
        let orig_len = subpath_length(sp);
        let t_start = cum / total_len;
        cum += orig_len;
        let t_span = orig_len / total_len;
        let Some(spine) = Spine::new(sp, 0.05) else {
            continue;
        };
        if spine.truncated {
            out.too_complex = Some("the stroke path needs more than 200 000 samples".into());
            return out;
        }
        let lp = spine.length;
        let measured = stroke
            .measured
            .get(si)
            .copied()
            .filter(|m| m.is_finite() && *m > 1e-9)
            .unwrap_or(lp);
        let to_m = measured / lp;
        let intervals = feed_intervals(windows, si, lp / measured, lp, spine.closed);
        if intervals.as_ref().is_some_and(|v| v.is_empty()) {
            continue;
        }
        let at_s = |s: f64| {
            if spine.closed {
                spine.eval(s.rem_euclid(lp))
            } else {
                spine.eval(s)
            }
        };
        let t_of = |s: f64| t_start + (s / lp).clamp(0.0, 1.0) * t_span;
        // Corners (unrounded kinks) with their miter vectors (a right-hand
        // offset of 1 on both legs), repeated past the seam of a closed path.
        let corners: Vec<(f64, V2)> = {
            let mut v = Vec::new();
            let reps: &[f64] = if spine.closed {
                &[-1.0, 0.0, 1.0, 2.0]
            } else {
                &[0.0]
            };
            for &k in spine.kinks.iter().take(10_000) {
                let (a, b) = (spine.eval(k - 1e-6).right(), spine.eval(k + 1e-6).right());
                let d = 1.0 + a.dot(b);
                let m = if d > 1e-3 { (a + b) * (1.0 / d) } else { a };
                let m = if m.len() > 4.0 { m * (4.0 / m.len()) } else { m };
                for r in reps {
                    v.push((k + r * lp, m));
                }
            }
            v
        };
        for (i, st) in strands(brush, stroke.seed, si).iter().enumerate() {
            // The footprint is sheared: bristles further right of travel
            // touch down and lift off later (Illustrator's slanted ends).
            let lateral = spread * (0.35 - 0.5 + edge_shift + st.u);
            let s0 = st.start * dia + shear * lateral;
            let lag = (st.lag * dia).max(1e-6);
            let tip = lp + st.end * dia + 0.5 * shear * lateral;
            // The bristle tips trail the ferrule: a point `trail` behind it
            // along the (lagging) heading.
            let trail = lag * TRAIL;
            let hw = 0.5 * st.w * spread;
            // Slanted caps: the left edge touches down and lifts off first.
            let cut = st.cut * hw;
            let (f0, f1) = (s0 + trail - cut, tip + trail + lag * POS_LAG + cut);
            // At least MIN_STRAND_SAMPLES along a strand (unless the budget
            // coarsened the step, or the floor itself would break it): a
            // stroke short next to its diameter would otherwise get a handful
            // of samples, and its caps and pressure changes turn into
            // polygonal chevrons.
            let min_count = if step > dia * 0.3 + 1e-9 || floor_samples > MAX_FEED_SAMPLES / 2 {
                2
            } else {
                min_strand_samples()
            };
            let count = (((f1 - f0) / step).ceil() as usize).max(min_count);
            let h = (f1 - f0) / count as f64;
            let head = (hw * 0.25).max(h * 0.5);
            let tail = (st.tail * hw).max(h * 0.5);
            // The grid plus the path's corners, where the offset takes the
            // miter (so a sharp outer corner stays a V, an inner one hooks).
            let mut feed: Vec<(f64, Option<V2>)> =
                (0..=count).map(|k| (f0 + h * k as f64, None)).collect();
            for (ks, miter) in &corners {
                if *ks > f0 && *ks < f1 {
                    feed.push((*ks, Some(*miter)));
                }
            }
            feed.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut rib = Ribbon {
                feed: Vec::with_capacity(feed.len()),
                left: Vec::with_capacity(feed.len()),
                right: Vec::with_capacity(feed.len()),
            };
            let mut prev_q: Option<V2> = None;
            let mut prev_s = f0;
            let mut heading = at_s(f0).tan;
            let mut prev_dir = heading;
            for (s, miter) in feed {
                let ds = (s - prev_s).max(0.0);
                prev_s = s;
                let a_dir = 1.0 - (-ds / lag).exp();
                let a_pos = 1.0 - (-ds / (lag * POS_LAG)).exp();
                let sp_pt = at_s(s);
                // The brush's heading turns after the path (bristles drag).
                heading = (heading + (sp_pt.tan - heading) * a_dir)
                    .unit()
                    .unwrap_or(sp_pt.tan);
                let right = miter.unwrap_or_else(|| sp_pt.right());
                let ink = if uses_ink {
                    ink_at(stroke.ink, t_of(s))
                } else {
                    InkAt::default()
                };
                let (fs, lean, pressure) = pose(brush, &ink, sp_pt.right());
                let band = spread * fs;
                let wobble = st.wob
                    * (std::f64::consts::TAU * s / (st.wob_len * dia) + st.phase).sin();
                let off = band * (lean - 0.5 + edge_shift + st.u + wobble);
                let raw = sp_pt.pos - heading * trail + right * off;
                let q = match prev_q {
                    None => raw,
                    Some(pq) => pq + (raw - pq) * a_pos,
                };
                let dir = match prev_q {
                    Some(pq) => (q - pq).unit().unwrap_or(prev_dir),
                    None => heading,
                };
                prev_dir = dir;
                prev_q = Some(q);
                // Width: round head, tapered tail (each edge on its own
                // slanted run), lifted off by low pressure.
                let v = s - trail;
                let cap = |from: f64, to: f64| {
                    let x = ((v - from) / head).clamp(0.0, 1.0);
                    let head_f = (1.0 - (1.0 - x) * (1.0 - x)).sqrt();
                    let tail_f = ((to - v) / tail).clamp(0.0, 1.0).powf(0.8);
                    head_f * tail_f
                };
                let lift_f = pressure.map_or(1.0, |p| smoothstep(st.lift - 0.1, st.lift + 0.1, p));
                let breathe = 1.0
                    + 0.05
                        * (std::f64::consts::TAU * s / (st.wob_len * dia * 0.7) + 2.0 * st.phase)
                            .sin();
                let half = hw * fs * lift_f * breathe;
                let nrm = v2(-dir.y, dir.x);
                rib.feed.push(v);
                rib.left.push(q - nrm * (half * cap(s0 - cut, tip - cut)));
                rib.right.push(q + nrm * (half * cap(s0 + cut, tip + cut)));
            }
            // The pieces the windows keep (feed-distance runs), else the whole.
            let runs: Vec<(usize, usize)> = match &intervals {
                None => vec![(0, rib.feed.len() - 1)],
                Some(iv) => {
                    let mut runs = Vec::new();
                    let mut open: Option<usize> = None;
                    for (k, s) in rib.feed.iter().enumerate() {
                        let inside = iv.iter().any(|(a, b)| *s >= *a && *s <= *b);
                        match (inside, open) {
                            (true, None) => open = Some(k),
                            (false, Some(o)) => {
                                runs.push((o, k - 1));
                                open = None;
                            }
                            _ => {}
                        }
                    }
                    if let Some(o) = open {
                        runs.push((o, rib.feed.len() - 1));
                    }
                    runs
                }
            };
            let multi = runs.len() > 1 || intervals.is_some();
            for (piece, (a, b)) in runs.into_iter().enumerate() {
                if b < a + 1 {
                    continue;
                }
                let left = &rib.left[a..=b];
                let right = &rib.right[a..=b];
                // Never simplify a thin strand away: the tolerance stays a
                // fraction of its widest half-width.
                let widest = left
                    .iter()
                    .zip(right)
                    .map(|(l, r)| l.dist(*r))
                    .fold(0.0, f64::max);
                if widest < 1e-6 {
                    continue;
                }
                let tol = tol.min(widest * 0.15);
                let mut ring: Vec<V2> = simplify(left, tol).into_iter().map(|k| left[k]).collect();
                let rk = simplify(right, tol);
                ring.extend(rk.into_iter().rev().map(|k| right[k]));
                // Drop coincident neighbours (the zero-width ends).
                ring.dedup_by(|x, y| x.dist(*y) < 1e-9);
                if ring.len() > 2 && ring[0].dist(ring[ring.len() - 1]) < 1e-9 {
                    ring.pop();
                }
                if ring.iter().any(|p| !p.is_finite()) {
                    continue;
                }
                let Some(sub) = closed_curve(&ring) else {
                    continue;
                };
                segments += sub.segs.len();
                if segments > MAX_OUTPUT_SEGMENTS {
                    out.too_complex = Some(format!(
                        "the bristle strokes need more than {MAX_OUTPUT_SEGMENTS} segments"
                    ));
                    return out;
                }
                let (fa, fb) = (rib.feed[a], rib.feed[b]);
                let mid = ((fa + fb) / 2.0).clamp(0.0, lp);
                let at = PlacedAt {
                    subpath: si,
                    s: mid * to_m,
                    s_span: (fa.max(s0) * to_m, fb.min(tip) * to_m),
                };
                let id = if multi {
                    format!("bristle#{si}.{i}.{piece}")
                } else {
                    format!("bristle#{si}.{i}")
                };
                out.emit(id, "bristle", &paint, vec![sub], 1.0, None, at);
            }
        }
    }
    out
}

/// Place a bristle brush on a path node (`windows`: trim / dash, clipping
/// the strands).
pub fn bristle_brush_for_node(
    node: &Map,
    brush: &BristleBrush,
    windows: Option<&[ArcWindow]>,
) -> Placement {
    let ns = node_stroke(node);
    let stroke = BrushStroke {
        subpaths: ns.subpaths,
        stroke_width: ns.stroke_width,
        stroke_color: ns.stroke_color.as_ref(),
        profile: ns.profile.as_ref(),
        ink: ns.ink.as_ref(),
        seed: ns.seed,
        measured: ns.measured,
    };
    bristle_brush_geometry(&stroke, brush, ns.stroke_opacity, windows)
}

/// JSON in / JSON out, same envelope as [`super::art::art_brush_json`] with a
/// `BristleBrushDefinition` as `brush` (`windows` clip the strands;
/// `tolerance` is ignored).
pub fn bristle_brush_json(request_json: &str) -> String {
    match parse_request(request_json) {
        Ok(req) => bristle_request(req),
        Err(e) => e,
    }
}

pub(crate) fn bristle_request(req: Request) -> String {
    let Some(brush) = parse_bristle_brush(&req.brush) else {
        return error("invalid_brush", "brush must be a bristle brush definition");
    };
    placement_json(&with_json_cap(|| {
        bristle_brush_for_node(&req.node, &brush, req.windows.as_deref())
    }))
}
