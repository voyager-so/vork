//! The blend itself: from two or more objects (and an optional spine) to the steps between them.
//!
//! Objects blend in order, each with the next (a chain, Illustrator's "blend any two or more objects"). For one pair
//! the steps are:
//!
//! - **Where**: along the spine. Every object sits at the arc length of the spine point nearest its centre; a step at
//!   parameter `t` between objects at `s_a` and `s_b` is centred on the spine at `lerp(s_a, s_b, t)`. An object that
//!   is off the spine carries its offset from it, and the steps interpolate the offsets, so they meet the objects
//!   exactly wherever the objects are. Without a replaced spine the spine is the polyline through the centres, which
//!   makes the steps a straight interpolation between the objects.
//! - **What shape**: the two outlines relative to their centres, corresponded ([`super::geometry::correspond`]) and
//!   lerped. With **Align to Path** orientation each outline is first expressed in the spine's frame at its own
//!   position (rotated by minus the spine's direction there), and each step is rotated by the spine's direction at
//!   its position: steps turn with the spine and still meet both objects exactly. **Align to Page** keeps them
//!   unrotated.
//! - **How many** (Illustrator's Spacing): **Specified Steps** is `steps` per pair; **Specified Distance** places
//!   steps every `distance` along the spine (centre to centre) short of the next object; **Smooth Color** takes as
//!   many steps as the largest colour-channel difference between the two objects (0-255, alpha included, at most
//!   254), so neighbouring steps differ by at most one level. Objects of one colour instead take as many steps as the
//!   gap between their bounding boxes holds objects of their mean size: on the axis the centres are further apart,
//!   `round((|Δcentre| - meanExtent) / meanExtent)`, at least 1 (Illustrator 30.8, measured; its docs: "based on
//!   the longest distance between the bounding box edges"). This uses the objects' own boxes, spine or no spine.
//! - **Easing** (the Blend panel's ease-in / ease-out, Illustrator 30.7): step parameters are evenly spaced, then
//!   mapped through `cubic-bezier(easeIn, 0, 1 - easeOut, 1)`. Both 0 is linear; ease-in bunches steps at the first
//!   object, ease-out at the second. Shape, position and paint all follow the eased parameter.
//! - **Paint**: fills, strokes and opacity interpolate at the step's parameter ([`super::paint_lerp`]).
//!
//! Stacking: steps are returned back to front, the first object's end first, so steps nearer later objects paint
//! over earlier ones; `reverseStack` (Reverse Front to Back) flips the order.

use super::geometry::{correspond, Pt, Shape};
use super::paint_lerp::{lerp_fills, lerp_strokes, stack_colors};
use super::spine::Spine;
use crate::paint::{Fill, Stroke};
use crate::types::JsonMap;
use crate::value::{json, Value};

/// Specified Steps / Specified Distance never make more than this many steps between two objects.
pub const MAX_STEPS_PER_PAIR: usize = 1000;
/// Smooth Color never makes more than this many steps between two objects (Illustrator 30.8 stops at 254, also
/// for a full 0-255 channel ramp).
pub const MAX_SMOOTH_STEPS: usize = 254;
/// A whole blend never draws more than this many steps (a guard for long chains at fine spacing).
pub const MAX_TOTAL_STEPS: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Spacing {
    Smooth,
    Steps(usize),
    Distance(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Page,
    Path,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendOptions {
    pub spacing: Spacing,
    pub orientation: Orientation,
    /// 0..1: `cubic-bezier(ease_in, 0, 1 - ease_out, 1)`.
    pub ease_in: f64,
    pub ease_out: f64,
    pub reverse_stack: bool,
}

impl Default for BlendOptions {
    fn default() -> Self {
        BlendOptions {
            spacing: Spacing::Smooth,
            orientation: Orientation::Page,
            ease_in: 0.0,
            ease_out: 0.0,
            reverse_stack: false,
        }
    }
}

impl BlendOptions {
    /// From a blend node's (or a `blend_steps` request's) fields: `spacing`, `steps`, `distance`, `orientation`,
    /// `easeIn`, `easeOut`, `reverseStack`. Missing or invalid values take the defaults.
    pub fn from_json(m: &JsonMap) -> BlendOptions {
        let num = |k: &str| m.get(k).and_then(Value::as_f64).filter(|v| v.is_finite());
        let unit = |k: &str| num(k).unwrap_or(0.0).clamp(0.0, 1.0);
        let spacing = match m.get("spacing").and_then(Value::as_str) {
            Some("steps") => Spacing::Steps(
                num("steps")
                    .unwrap_or(4.0)
                    .floor()
                    .clamp(1.0, MAX_STEPS_PER_PAIR as f64) as usize,
            ),
            Some("distance") => Spacing::Distance(num("distance").unwrap_or(20.0).max(0.01)),
            _ => Spacing::Smooth,
        };
        BlendOptions {
            spacing,
            orientation: if m.get("orientation").and_then(Value::as_str) == Some("path") {
                Orientation::Path
            } else {
                Orientation::Page
            },
            ease_in: unit("easeIn"),
            ease_out: unit("easeOut"),
            reverse_stack: m
                .get("reverseStack")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        }
    }
}

/// One blend object: its outline in canvas space and its paint.
#[derive(Debug, Clone)]
pub struct BlendSource {
    pub shape: Shape,
    pub fills: Vec<Fill>,
    pub strokes: Vec<Stroke>,
    pub opacity: f64,
    pub fill_rule: Option<String>,
}

impl BlendSource {
    /// From an outline and a node-shaped map carrying `fills`, `strokes`, `opacity` and `fillRule`.
    pub fn new(shape: Shape, style: &JsonMap) -> BlendSource {
        BlendSource {
            shape,
            fills: crate::paint::parse_fills(style.get("fills")),
            strokes: crate::paint::parse_strokes(style.get("strokes")),
            opacity: style
                .get("opacity")
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite())
                .unwrap_or(1.0),
            fill_rule: style
                .get("fillRule")
                .and_then(Value::as_str)
                .map(str::to_string),
        }
    }
}

/// One step.
#[derive(Debug, Clone)]
pub struct BlendStep {
    /// The pair it lies between: the index (in the sources given) of the object it follows.
    pub pair: usize,
    /// Its (eased) parameter between them, 0..1 exclusive.
    pub t: f64,
    pub shape: Shape,
    pub fills: Vec<Fill>,
    pub strokes: Vec<Stroke>,
    pub opacity: f64,
    pub fill_rule: Option<String>,
}

impl BlendStep {
    /// The step as a synthetic PathNode (points in canvas space, shifted by `-origin`).
    pub fn to_path_node(&self, origin: Pt) -> JsonMap {
        let mut m = JsonMap::new();
        m.insert("type".into(), json!("path"));
        m.insert(
            "points".into(),
            Value::Array(self.shape.translated(-origin.0, -origin.1).to_points()),
        );
        m.insert("closed".into(), json!(self.shape.is_closed()));
        m.insert(
            "fillRule".into(),
            json!(self.fill_rule.clone().unwrap_or_else(|| "nonzero".into())),
        );
        if (self.opacity - 1.0).abs() > 1e-12 {
            m.insert("opacity".into(), json!(self.opacity.clamp(0.0, 1.0)));
        }
        if !self.fills.is_empty() {
            crate::paint::set_fills(&mut m, self.fills.clone());
        }
        if !self.strokes.is_empty() {
            crate::paint::set_strokes(&mut m, self.strokes.clone());
        }
        m
    }
}

/// `cubic-bezier(x1, 0, x2, 1)` at `u` (x solved by bisection; monotone for x1, x2 in 0..1).
pub fn ease(u: f64, ease_in: f64, ease_out: f64) -> f64 {
    if ease_in <= 0.0 && ease_out <= 0.0 {
        return u;
    }
    let (x1, x2) = (ease_in, 1.0 - ease_out);
    let bez = |p1: f64, p2: f64, s: f64| {
        let ms = 1.0 - s;
        3.0 * ms * ms * s * p1 + 3.0 * ms * s * s * p2 + s * s * s
    };
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if bez(x1, x2, mid) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bez(0.0, 1.0, (lo + hi) / 2.0)
}

/// Smooth Color's step count for a pair.
pub fn smooth_steps(a: &BlendSource, b: &BlendSource) -> usize {
    let (ca, cb) = (
        stack_colors(&a.fills, &a.strokes),
        stack_colors(&b.fills, &b.strokes),
    );
    let mut delta: f64 = 0.0;
    let n = ca.len().max(cb.len());
    for i in 0..n {
        let (Some(x), Some(y)) = (
            ca.get(i.min(ca.len().saturating_sub(1))),
            cb.get(i.min(cb.len().saturating_sub(1))),
        ) else {
            continue;
        };
        delta = delta
            .max((x.r - y.r).abs())
            .max((x.g - y.g).abs())
            .max((x.b - y.b).abs())
            .max((x.a - y.a).abs() * 255.0);
    }
    delta = delta.max((a.opacity - b.opacity).abs() * 255.0);
    let n = if delta >= 1.0 {
        delta.ceil()
    } else {
        same_colour_steps(a, b)
    };
    n.clamp(1.0, MAX_SMOOTH_STEPS as f64) as usize
}

/// Smooth Color for objects of one colour: the gap between the bounding boxes in units of their mean extent, on
/// the axis their centres are further apart, rounded half up.
fn same_colour_steps(a: &BlendSource, b: &BlendSource) -> f64 {
    let (Some(ba), Some(bb)) = (a.shape.bounds(), b.shape.bounds()) else {
        return 1.0;
    };
    let (ca, cb) = (ba.center(), bb.center());
    let (dx, dy) = ((cb.0 - ca.0).abs(), (cb.1 - ca.1).abs());
    let (delta, extent) = if dx >= dy {
        (dx, (ba.width() + bb.width()) / 2.0)
    } else {
        (dy, (ba.height() + bb.height()) / 2.0)
    };
    if extent <= 1e-9 {
        return 1.0;
    }
    // A hair of tolerance so exact halves (gap 45 at size 30) round up despite float error.
    ((delta - extent) / extent + 1e-9 + 0.5).floor()
}

/// The pair's evenly spaced (not yet eased) step parameters.
fn step_params(spacing: Spacing, a: &BlendSource, b: &BlendSource, length: f64) -> Vec<f64> {
    let evenly = |n: usize| {
        (1..=n)
            .map(|k| k as f64 / (n as f64 + 1.0))
            .collect::<Vec<_>>()
    };
    match spacing {
        Spacing::Steps(n) => evenly(n.min(MAX_STEPS_PER_PAIR)),
        Spacing::Smooth => evenly(smooth_steps(a, b)),
        Spacing::Distance(d) => {
            if length <= 0.0 || d <= 0.0 {
                return Vec::new();
            }
            let n = ((length / d - 1e-9).ceil() as usize)
                .saturating_sub(1)
                .min(MAX_STEPS_PER_PAIR);
            (1..=n).map(|k| k as f64 * d / length).collect()
        }
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// The steps of a blend of `sources` (in chain order, at least two) along `spine` (a replaced spine; `None` for the
/// straight one through the centres), back to front. Sources with an empty outline are left out of the chain.
pub fn blend_steps(
    sources: &[BlendSource],
    spine: Option<&Shape>,
    opts: &BlendOptions,
) -> Vec<BlendStep> {
    let (index, sources): (Vec<usize>, Vec<&BlendSource>) = sources
        .iter()
        .enumerate()
        .filter(|(_, s)| s.shape.center().is_some())
        .unzip();
    if sources.len() < 2 {
        return Vec::new();
    }
    let centers: Vec<Pt> = sources.iter().map(|s| s.shape.center().unwrap()).collect();

    // The spine and each object's arc length along it.
    let (spine, arc): (Option<Spine>, Vec<f64>) = match spine.and_then(Spine::from_shape) {
        Some(sp) => {
            let arc = centers.iter().map(|&c| sp.project(c)).collect();
            (Some(sp), arc)
        }
        None => {
            let mut arc = vec![0.0];
            for w in centers.windows(2) {
                arc.push(arc.last().unwrap() + (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1));
            }
            (Spine::from_polyline(&centers), arc)
        }
    };
    let offsets: Vec<Pt> = centers
        .iter()
        .zip(&arc)
        .map(|(&c, &s)| match &spine {
            Some(sp) => {
                let p = sp.point_at(s);
                (c.0 - p.0, c.1 - p.1)
            }
            None => (0.0, 0.0),
        })
        .collect();

    let path_oriented = opts.orientation == Orientation::Path && spine.is_some();
    let mut out: Vec<BlendStep> = Vec::new();
    for k in 0..sources.len() - 1 {
        let (a, b) = (sources[k], sources[k + 1]);
        let (sa, sb) = (arc[k], arc[k + 1]);
        let forward = sb >= sa;
        // Direction of travel along the spine at arc length s.
        let heading = |s: f64, after: bool| -> f64 {
            match (&spine, path_oriented) {
                (Some(sp), true) => {
                    sp.tangent_at(s, after) + if forward { 0.0 } else { std::f64::consts::PI }
                }
                _ => 0.0,
            }
        };
        let (ha, hb) = (heading(sa, forward), heading(sb, !forward));
        let local = |src: &BlendSource, c: Pt, h: f64| src.shape.translated(-c.0, -c.1).rotated(-h);
        let (shape_a, shape_b) =
            correspond(&local(a, centers[k], ha), &local(b, centers[k + 1], hb));

        let length = if spine.is_some() {
            (sb - sa).abs()
        } else {
            (centers[k + 1].0 - centers[k].0).hypot(centers[k + 1].1 - centers[k].1)
        };
        for u in step_params(opts.spacing, a, b, length) {
            if out.len() >= MAX_TOTAL_STEPS {
                break;
            }
            let t = ease(u, opts.ease_in, opts.ease_out);
            let s = lerp(sa, sb, t);
            let pos = match &spine {
                Some(sp) => {
                    let p = sp.point_at(s);
                    (
                        p.0 + lerp(offsets[k].0, offsets[k + 1].0, t),
                        p.1 + lerp(offsets[k].1, offsets[k + 1].1, t),
                    )
                }
                None => (
                    lerp(centers[k].0, centers[k + 1].0, t),
                    lerp(centers[k].1, centers[k + 1].1, t),
                ),
            };
            let shape = shape_a
                .lerp(&shape_b, t)
                .rotated(heading(s, forward))
                .translated(pos.0, pos.1);
            out.push(BlendStep {
                pair: index[k],
                t,
                shape,
                fills: lerp_fills(&a.fills, &b.fills, t),
                strokes: lerp_strokes(&a.strokes, &b.strokes, t),
                opacity: lerp(a.opacity, b.opacity, t),
                fill_rule: if t < 0.5 {
                    a.fill_rule.clone()
                } else {
                    b.fill_rule.clone()
                },
            });
        }
    }
    if opts.reverse_stack {
        out.reverse();
    }
    out
}
