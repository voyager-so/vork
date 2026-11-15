//! Scatter brush (brushes.md §2.1, §4.4, §4.6; plan P18): copies of the
//! artwork placed along the stroke at arc-length intervals, each copy an
//! affine transform of the artwork (no warp).
//!
//! # Placement
//!
//! The artwork is centred on its visual bounds (strokes included, so a
//! hairline artwork has a real size). Every copy `i`
//! draws its own values from the stroke seed (`draw(seed, salt, i)`, one salt
//! per parameter) or reads the pen channels at its position:
//!
//! - **size** scales the copy by `size × strokeWidth` — Illustrator's rule
//!   (lead decision, P18 review): the artwork's own size at weight 1 and
//!   size 1, the stroke weight a multiplier. `overrides.scale` replaces
//!   `size` per stroke; with
//!   `overrides.widthFrom = 'profile'` the stroke's width profile (mean of
//!   both sides) multiplies it.
//! - **spacing** is the distance between consecutive copy centres as a
//!   multiple of the artwork's LARGER visual dimension times the stroke
//!   weight — Illustrator's rule, measured against its stock brushes (Arrow
//!   1.01 steps its 31.74 height; Dots steps 0.2 × its 72.48 width whatever
//!   each copy's random size; Cross in Circle at a fixed 150 % size still
//!   steps its 32.58 width, so its copies overlap; weight 2 doubles the
//!   step). Size, its variation and the width profile scale the copies, not
//!   the step. Floored at 5 % of the dimension.
//! - **scatter** offsets the copy across the path by `scatter ×` the same
//!   larger dimension × weight (not the copy's size), positive to the LEFT of
//!   the direction of travel (the side the model's counter-clockwise angles
//!   turn towards; Illustrator agrees).
//! - **rotation** turns the copy counter-clockwise on screen, in degrees,
//!   relative to the path's tangent (`rotationRelativeTo: 'path'`: the art's
//!   x axis runs along the path, its top on the path's left) or to the page.
//!
//! Copies are placed along Illustrator's flattening of the path
//! ([`Spine::chords`]: each cubic cut into equal-parameter chords, walked by
//! chord length), and a copy relative to the path takes its chord's direction,
//! so neighbours on one chord share a rotation, as in Illustrator.
//!
//! An open path's first copy is centred on the path's start (as in
//! Illustrator) and copies continue while their centre is on the path. A closed
//! path's first copy is centred on the seam and there is a copy at every step
//! before the seam, however close the last lands to the first (Illustrator:
//! ⌈length / step⌉ copies). Compound paths place copies on each subpath; draws keep
//! counting across subpaths. A placement needing more than `MAX_COPIES`
//! copies or `MAX_OUTPUT_SEGMENTS` segments fails with `too_complex`.

use crate::value::{Map, Value};

use super::artwork::{collect_artwork, default_geometry, Artwork, GeometryFn};
use super::colorize::{self, parse_colorize, Colorize};
use super::common::{
    error, node_stroke, parse_request, placement_json, subpath_length, with_json_cap, BrushStroke,
    PlacedAt, Placement, Request, MAX_OUTPUT_SEGMENTS,
};
use super::dyn_value::{draw, ink_at, parse_dyn, Dyn, InkAt};
use super::geom::Bounds;
use super::spine::Spine;
use crate::path_geom::vec2::{v2, V2};
use crate::transform;
use crate::types::Affine;

/// Most copies per stroke.
pub const MAX_COPIES: usize = 10_000;
const SALT_SIZE: u64 = 0x51;
const SALT_SPACING: u64 = 0x52;
const SALT_SCATTER: u64 = 0x53;
const SALT_ROTATION: u64 = 0x54;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationRelativeTo {
    Page,
    Path,
}

/// `ScatterBrushDefinition` (nodes.ts), normalised like `normalizeBrushDefinition`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterBrush {
    pub id: String,
    pub name: String,
    pub artwork: Vec<Value>,
    pub size: Dyn,
    pub spacing: Dyn,
    pub scatter: Dyn,
    pub rotation: Dyn,
    pub relative_to: RotationRelativeTo,
    pub colorize: Colorize,
    /// `scaleWithWeight`: Illustrator's sizing — the stroke weight multiplies
    /// the brush's designed size. Without it (the default, as in Figma) the
    /// stroke width is the width of the mark: the design is scaled to it.
    pub scale_with_weight: bool,
}

pub fn parse_scatter_brush(raw: &Value) -> Option<ScatterBrush> {
    let o = raw.as_object()?;
    let id = o
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    if o.get("type").and_then(Value::as_str) != Some("scatter") {
        return None;
    }
    let artwork = super::art::artwork_nodes(o.get("artwork"))?;
    Some(ScatterBrush {
        id: id.to_string(),
        name: o
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .to_string(),
        artwork,
        size: parse_dyn(o.get("size"), 1.0, 0.0, f64::INFINITY),
        spacing: parse_dyn(o.get("spacing"), 1.0, 0.0, f64::INFINITY),
        scatter: parse_dyn(o.get("scatter"), 0.0, f64::NEG_INFINITY, f64::INFINITY),
        rotation: parse_dyn(o.get("rotation"), 0.0, f64::NEG_INFINITY, f64::INFINITY),
        relative_to: match o.get("rotationRelativeTo").and_then(Value::as_str) {
            Some("page") => RotationRelativeTo::Page,
            _ => RotationRelativeTo::Path,
        },
        colorize: parse_colorize(o.get("colorize")),
        scale_with_weight: super::common::scale_with_weight(o),
    })
}

/// `BrushApplication.overrides` as they apply to a scatter brush.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScatterOverrides {
    /// Replaces `size` (the override's key is `scale`).
    pub size: Option<Dyn>,
    pub spacing: Option<Dyn>,
    pub scatter: Option<Dyn>,
    pub rotation: Option<Dyn>,
    pub relative_to: Option<RotationRelativeTo>,
    pub colorize: Option<Colorize>,
    pub width_from_profile: bool,
}

pub fn parse_scatter_overrides(raw: Option<&Map>) -> ScatterOverrides {
    let Some(o) = raw else {
        return ScatterOverrides::default();
    };
    ScatterOverrides {
        size: o
            .get("scale")
            .map(|v| parse_dyn(Some(v), 1.0, 0.0, f64::INFINITY)),
        spacing: o
            .get("spacing")
            .map(|v| parse_dyn(Some(v), 1.0, 0.0, f64::INFINITY)),
        scatter: o
            .get("scatter")
            .map(|v| parse_dyn(Some(v), 0.0, f64::NEG_INFINITY, f64::INFINITY)),
        rotation: o
            .get("rotation")
            .map(|v| parse_dyn(Some(v), 0.0, f64::NEG_INFINITY, f64::INFINITY)),
        relative_to: match o.get("rotationRelativeTo").and_then(Value::as_str) {
            Some("page") => Some(RotationRelativeTo::Page),
            Some("path") => Some(RotationRelativeTo::Path),
            _ => None,
        },
        colorize: o.get("colorize").map(|v| parse_colorize(Some(v))),
        width_from_profile: o.get("widthFrom").and_then(Value::as_str) == Some("profile"),
    }
}

/// Counter-clockwise (on a y-down screen) rotation of `v` by `deg`.
fn rotate_ccw(v: V2, deg: f64) -> V2 {
    let (s, c) = deg.to_radians().sin_cos();
    v2(v.x * c + v.y * s, -v.x * s + v.y * c)
}

/// Place a scatter brush along a stroke. Pure and total.
pub fn scatter_brush_geometry(
    stroke: &BrushStroke,
    brush: &ScatterBrush,
    overrides: &ScatterOverrides,
    artwork: &Artwork,
) -> Placement {
    let mut out = Placement {
        skipped: artwork.skipped.clone(),
        bounds: Bounds::empty(),
        ..Placement::default()
    };
    let b = artwork.visual_bounds();
    if b.is_empty() {
        return out;
    }
    let (w_art, h_art) = (b.width(), b.height());
    if !w_art.is_finite() || !h_art.is_finite() || w_art.max(h_art) <= 1e-9 {
        return out;
    }
    // Illustrator measures spacing and scatter in the artwork's larger
    // dimension (see the module docs); the copy's extent along the path
    // (for `PlacedAt`) still uses its own width and height.
    let dim = w_art.max(h_art);
    let h_ref = h_art;
    let len_ref = w_art.max(1e-6 * h_art);
    let art = artwork.transformed(transform::translate(
        -(b.min.x + b.max.x) / 2.0,
        -(b.min.y + b.max.y) / 2.0,
    ));
    let segs_per_copy = art.segment_count().max(1);
    let size = overrides.size.unwrap_or(brush.size);
    let spacing_dyn = overrides.spacing.unwrap_or(brush.spacing);
    let scatter_dyn = overrides.scatter.unwrap_or(brush.scatter);
    let rotation_dyn = overrides.rotation.unwrap_or(brush.rotation);
    let relative_to = overrides.relative_to.unwrap_or(brush.relative_to);
    let colorize = overrides.colorize.as_ref().unwrap_or(&brush.colorize);
    let ctx = colorize::resolve(colorize, stroke.stroke_color, &artwork.weighted_colors());
    let profile = if overrides.width_from_profile {
        stroke.profile
    } else {
        None
    };
    let stroke_width = if stroke.stroke_width.is_finite() {
        stroke.stroke_width.max(0.0)
    } else {
        0.0
    };
    // Sizing: a copy's larger side is `size` × the stroke width (spacing and
    // scatter in the same unit), unless the weight multiplies the artwork.
    let base_k = stroke_width * super::common::weight_unit(brush.scale_with_weight, dim);
    let total_len: f64 = stroke.subpaths.iter().map(subpath_length).sum();
    if !total_len.is_finite() || total_len <= 1e-9 {
        return out;
    }
    let mut copy: u64 = 0;
    let mut cum = 0.0;
    for (si, sp) in stroke.subpaths.iter().enumerate() {
        let orig_len = subpath_length(sp);
        let t_start = cum / total_len;
        cum += orig_len;
        let t_span = orig_len / total_len;
        let Some(spine) = Spine::chords(sp) else {
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
        let t_of = |s: f64| t_start + (s / lp).clamp(0.0, 1.0) * t_span;
        let uses_ink = size.uses_ink()
            || spacing_dyn.uses_ink()
            || scatter_dyn.uses_ink()
            || rotation_dyn.uses_ink();
        let ink_for = |s: f64| {
            if uses_ink {
                ink_at(stroke.ink, t_of(s))
            } else {
                InkAt::default()
            }
        };
        let k_for = |s: f64, ink: &InkAt, i: u64| {
            let mut k = base_k * size.eval(ink, draw(stroke.seed, SALT_SIZE, i)).max(0.0);
            if let Some(p) = profile {
                let (l, r) = p.width_at(t_of(s), None);
                k *= (l + r) / 2.0;
            }
            if k.is_finite() {
                k
            } else {
                0.0
            }
        };
        // Open or closed, the first copy is centred on the subpath's start.
        let mut s = 0.0;
        let mut placed_here = 0usize;
        loop {
            let ink = ink_for(s);
            let k = k_for(s, &ink, copy);
            let spacing = spacing_dyn
                .eval(&ink, draw(stroke.seed, SALT_SPACING, copy))
                .max(0.0);
            let step = (dim * base_k * spacing).max(dim * base_k * 0.05).max(0.1);
            // A closed subpath gets a copy at every step before the seam, however close
            // the last lands to the first (Illustrator: copies = ⌈length / step⌉, checked
            // on 21 circles).
            if spine.closed && placed_here > 0 && s >= lp - 1e-6 * lp.max(1.0) {
                break;
            }
            if !spine.closed && s > lp + 1e-9 {
                break;
            }
            if copy as usize >= MAX_COPIES
                || (copy as usize + 1) * segs_per_copy > MAX_OUTPUT_SEGMENTS
            {
                out.too_complex = Some(format!(
                    "scatter brush placement needs more than {MAX_COPIES} copies or {MAX_OUTPUT_SEGMENTS} segments (raise spacing or size)"
                ));
                return out;
            }
            let q = spine.eval(s);
            let offset = scatter_dyn
                .eval(&ink, draw(stroke.seed, SALT_SCATTER, copy))
                * dim
                * base_k;
            let rot = rotation_dyn
                .eval(&ink, draw(stroke.seed, SALT_ROTATION, copy));
            let base = match relative_to {
                RotationRelativeTo::Path => q.tan,
                RotationRelativeTo::Page => v2(1.0, 0.0),
            };
            let e1 = rotate_ccw(base, if rot.is_finite() { rot } else { 0.0 });
            let e2 = v2(-e1.y, e1.x);
            let left = v2(q.tan.y, -q.tan.x);
            let center = q.pos + left * if offset.is_finite() { offset } else { 0.0 };
            if k > 1e-12 {
                let m = Affine {
                    a: e1.x * k,
                    b: e1.y * k,
                    c: e2.x * k,
                    d: e2.y * k,
                    e: center.x,
                    f: center.y,
                };
                let placed = art.transformed(m);
                // Anchor and extent along the path, in the outliner's measure:
                // the rotated copy's box projected on the tangent (the scatter
                // offset is across the path, so it doesn't move the extent).
                let to_m = measured / lp;
                let half = (e1.dot(q.tan).abs() * len_ref + e2.dot(q.tan).abs() * h_ref) * k / 2.0;
                let at = PlacedAt {
                    subpath: si,
                    s: s * to_m,
                    s_span: ((s - half) * to_m, (s + half) * to_m),
                };
                for shape in placed.shapes {
                    out.emit(
                        format!("{}#{}.{}", shape.source_id, si, placed_here),
                        &shape.source_id,
                        &shape.paint,
                        shape.subpaths,
                        shape.transform_scale * k,
                        ctx.as_ref(),
                        at,
                    );
                }
            }
            copy += 1;
            placed_here += 1;
            s += step;
        }
    }
    out
}

/// Place a scatter brush on a path node.
pub fn scatter_brush_for_node(
    node: &Map,
    brush: &ScatterBrush,
    geometry: GeometryFn,
) -> Placement {
    let ns = node_stroke(node);
    let artwork = collect_artwork(&brush.artwork, geometry);
    let stroke = BrushStroke {
        subpaths: ns.subpaths,
        stroke_width: ns.stroke_width,
        stroke_color: ns.stroke_color.as_ref(),
        profile: ns.profile.as_ref(),
        ink: ns.ink.as_ref(),
        seed: ns.seed,
        measured: ns.measured,
    };
    let overrides = parse_scatter_overrides(ns.overrides.as_ref());
    scatter_brush_geometry(&stroke, brush, &overrides, &artwork)
}

/// JSON in / JSON out, same envelope as [`super::art::art_brush_json`] with a
/// `ScatterBrushDefinition` as `brush` (`tolerance` is ignored: copies are
/// exact affine images).
pub fn scatter_brush_json(request_json: &str) -> String {
    scatter_brush_json_with(request_json, &default_geometry)
}

pub fn scatter_brush_json_with(request_json: &str, geometry: GeometryFn) -> String {
    match parse_request(request_json) {
        Ok(req) => scatter_request(req, geometry),
        Err(e) => e,
    }
}

pub(crate) fn scatter_request(req: Request, geometry: GeometryFn) -> String {
    let Some(brush) = parse_scatter_brush(&req.brush) else {
        return error(
            "invalid_brush",
            "brush must be a scatter brush definition with artwork.nodes",
        );
    };
    placement_json(&with_json_cap(|| {
        scatter_brush_for_node(&req.node, &brush, geometry)
    }))
}
