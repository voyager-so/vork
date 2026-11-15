//! gradient.rs — gradient model + byte-compatible instance packing (M3b).
//!
//! Ports, line for line:
//! - `WebGPUGradientSystem.{validateGradientParams, parseGradientStops,
//!   createLinearGradient, createRadialGradient, createConicGradient}`
//!   (`renderers/webgpu/rendering/systems/WebGPUGradientSystem.ts:345-580`),
//! - `packGradientStops` (`renderers/webgpu/utils/WebGPUGradientUtils.ts:13-41`):
//!   `vec4(offset, r, g, floor(b·255) + a·0.99999)` per stop, unused slots get
//!   offset 10000 with the LAST stop's color (white when no stops),
//! - `buildFillGradientPaint` / `buildStrokeGradientPaint`
//!   (`renderers/webgpu/rendering/systems/shapes/gradientPaintBuilders.ts`),
//! - the `PathGradientData` WGSL struct layout (extracted
//!   `wgsl/path-gradient.wgsl`; 388 floats / 1552 bytes, the last 272 the
//!   freeform gradient's points, colours, slopes and reaches, E8.18).
//!
//! COPY THE BUGS (quirk ledger Q-18/Q-19 + new entries this milestone):
//! - the gradient a node draws is its top fill's paint (`moda_scene::paint`);
//!   this module builds the paint it is handed,
//! - conic gradient strokes are not drawn yet (`strokeGradientMeta.x` ∈
//!   {0,1,2}); a conic, image, video or shader stroke paints nothing,

//! - stops are clamped to [0,1] and sorted, keeping their positions: two stops
//!   at one position are a hard stop (the shaders step there, as CSS and the
//!   SVG/PDF exports do); ≤ 8 stops parsed,
//! - stop colors parse via `colorToNormalizedRgba` (no 'transparent' keyword,
//!   no lowercase retry — unlike the fill/stroke solid-color path), parse
//!   failure → opaque white.

use bytemuck::{Pod, Zeroable};
use moda_scene::paint::{
    self as scene_paint, ConicGradientPaint, FreeformGradientPaint, GradientStop as PaintStop,
    LinearGradientPaint, Paint, PaintPoint, RadialGradientPaint, Stroke, StrokeGradientMode,
    MAX_FREEFORM_POINTS,
};
use moda_scene::types::RawNode;
use moda_scene::value::Value;

use crate::color::{color_to_normalized_rgba, parse_color, Rgba};

/// `MAX_GRADIENT_STOPS` (`WebGPUGradientSystem.ts:59`; manifest pinned).
pub const MAX_GRADIENT_STOPS: usize = 8;
/// `COLOR_ENCODING_CONSTANTS` (`WebGPUConstants.ts`): blue integer multiplier
/// and the alpha fractional-encoding limit.
pub const BLUE_CHANNEL_MULTIPLIER: f64 = 255.0;
pub const ALPHA_ENCODING_LIMIT: f64 = 0.99999;

// ---------------------------------------------------------------------------
// GPU instance ABI — `PathGradientData` (wgsl/path-gradient.wgsl)
// ---------------------------------------------------------------------------

/// The 1552-byte `PathGradientData` storage-buffer instance. Field order and
/// offsets mirror the WGSL struct exactly (linear layout, no implicit padding:
/// every vec2 lands on an 8-byte and every vec4/array on a 16-byte boundary).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GradientInstance {
    /// floats 0..16 — column-major model matrix.
    pub model_matrix: [f32; 16],
    /// floats 16..20 — gradient fill: white with a = node opacity; solid
    /// fill: real color (a × opacity); stroke-only overlay: (0,0,0,0).
    pub color: [f32; 4],
    /// floats 20..24 — [vertexOffset, vertexCount, depth, fillType]
    /// (fillType: 0 solid, 1 linear, 2 radial, 3 conic, 4 diamond,
    /// 5 freeform). NOTE: depth in .z —
    /// different from path-solid/pattern (.w).
    pub path_info: [f32; 4],
    /// floats 24..28 — gradient stroke: white with a = node opacity; solid
    /// stroke: real color (a × opacity).
    pub stroke_color: [f32; 4],
    /// floats 28..30 — linear start point OR radial/conic center (% 0-100).
    pub gradient_start: [f32; 2],
    /// floats 30..32 — linear end point OR radial radiusX/radiusY (%).
    pub gradient_end: [f32; 2],
    /// floats 32..36 — the box gradient percentages span, [minX, minY, w, h]
    /// in node-local space: the path's geometric bounds, not the stroke's
    /// ([`gradient_box`]).
    pub bounds: [f32; 4],
    /// float 36 (u32 bits) — number of packed fill stops.
    pub num_stops: u32,
    /// float 37 — `node.strokeWidth || 1`.
    pub stroke_width: f32,
    /// float 38 — radial rotation OR conic start angle, radians.
    pub gradient_rotation: f32,
    /// The fill's interpolation space + 8 × dither (`GradientOptions::shader_flags`).
    pub gradient_flags: f32,
    /// floats 40..72 — packed fill stops.
    pub gradient_stops: [[f32; 4]; MAX_GRADIENT_STOPS],
    /// floats 72..76.
    pub page_clip_bounds: [f32; 4],
    /// floats 76..78.
    pub stroke_gradient_start: [f32; 2],
    /// floats 78..80.
    pub stroke_gradient_end: [f32; 2],
    /// floats 80..84 — x = paint type (0 solid, 1 linear, 2 radial),
    /// y = numStops, z = rotation (radians), w = how the gradient lies on
    /// the stroke (0 within, 1 along, 2 across; E8.18).
    pub stroke_gradient_meta: [f32; 4],
    /// floats 84..116 — packed stroke stops.
    pub stroke_gradient_stops: [[f32; 4]; MAX_GRADIENT_STOPS],
    /// floats 116..180 — freeform fill (fillType 5): per point
    /// `[x%, y%, spread%, link]`, `link` 0 = a lone point, 1 = joined to the
    /// next point (a line segment), 2 = a line's last point ([`pack_freeform`]).
    pub freeform_points: [[f32; 4]; MAX_FREEFORM_POINTS],
    /// floats 180..244 — the freeform points' straight RGBA.
    pub freeform_colors: [[f32; 4]; MAX_FREEFORM_POINTS],
    /// floats 244..372 — each freeform point's colour slope per box px,
    /// RGBA along x at `2i`, along y at `2i + 1` ([`freeform_slopes`]).
    pub freeform_slopes: [[f32; 4]; 2 * MAX_FREEFORM_POINTS],
    /// floats 372..388 — each freeform point's slope reach (box px), point
    /// `i` at `[i / 4][i % 4]`.
    pub freeform_reach: [[f32; 4]; MAX_FREEFORM_POINTS / 4],
}

const _: () = assert!(core::mem::size_of::<GradientInstance>() == 1552);

// ---------------------------------------------------------------------------
// Gradient model (WebGPUGradientSystem port)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradientStop {
    pub offset: f64,
    pub color: Rgba,
}

/// How far outside its box a gradient point may sit, and the largest radial radius, in percent of the box (a
/// safety bound against overflow, not a design limit: one gradient spread over several layers places each layer's
/// ends and radii far outside it, `gradientAcross.ts`). The WGSL (`path-gradient`, `shape-gradient`,
/// `mask-coverage`) and the SVG / PDF exports (`svgPaint.ts`) use the same bound.
pub const MAX_POINT_PERCENTAGE: f64 = 100_000.0;
const MAX_RADIAL_RADIUS: f64 = 100_000.0;

fn clamp_point(p: (f64, f64)) -> (f64, f64) {
    (
        p.0.clamp(-MAX_POINT_PERCENTAGE, MAX_POINT_PERCENTAGE),
        p.1.clamp(-MAX_POINT_PERCENTAGE, MAX_POINT_PERCENTAGE),
    )
}

/// rotation normalization: `((r % 360) + 360) % 360` (`:399-407`).
fn normalize_rotation_degrees(r: f64) -> f64 {
    let mut normalized = r % 360.0;
    if normalized < 0.0 {
        normalized += 360.0;
    }
    normalized
}

/// `parseGradientStops` (`WebGPUGradientSystem.ts:418-481`). `color_stops` is
/// the raw `[offset, color, offset, color, …]` JSON array.
pub fn parse_gradient_stops(color_stops: Option<&Vec<Value>>) -> Vec<GradientStop> {
    match color_stops {
        None => parse_stop_pairs(std::iter::empty()),
        Some(a) => parse_stop_pairs(a.chunks(2).map(|c| {
            (
                c[0].as_f64().unwrap_or(0.0),
                c.get(1).and_then(Value::as_str),
            )
        })),
    }
}

/// [`parse_gradient_stops`] for a paint's stops.
pub fn parse_paint_stops(stops: &[PaintStop]) -> Vec<GradientStop> {
    parse_stop_pairs(stops.iter().map(|s| (s.offset, s.color.as_str())))
}

pub(crate) fn parse_repeating_stops(stops: &[PaintStop], repeat: Option<bool>) -> Vec<GradientStop> {
    if repeat != Some(true) {
        return parse_paint_stops(stops);
    }
    let mut parsed: Vec<_> = stops.iter().take(MAX_GRADIENT_STOPS)
        .map(|source| {
            let mut parsed = parse_paint_stops(std::slice::from_ref(source)).remove(0);
            parsed.offset = if source.offset.is_finite() { source.offset } else { 0.0 };
            parsed
        }).collect();
    parsed.sort_by(|first, second| first.offset.total_cmp(&second.offset));
    parsed
}

/// `(offset, color)` pairs → validated stops. A non-number offset arrives as
/// 0 (JS would carry NaN; unreachable on valid scenes), a non-string colour
/// (variable refs) as white.
fn parse_stop_pairs<'a>(pairs: impl Iterator<Item = (f64, Option<&'a str>)>) -> Vec<GradientStop> {
    let white = Rgba {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    let mut stops: Vec<GradientStop> = Vec::new();

    for (offset, color_value) in pairs.take(MAX_GRADIENT_STOPS) {
        let validated_offset = offset.max(0.0).min(1.0);
        let color = match color_value {
            Some(s) => color_to_normalized_rgba(s).unwrap_or(white),
            None => white, // variable refs / non-strings → white (TODO in TS)
        };
        let clamped = Rgba {
            r: color.r.clamp(0.0, 1.0),
            g: color.g.clamp(0.0, 1.0),
            b: color.b.clamp(0.0, 1.0),
            a: color.a.clamp(0.0, 1.0),
        };
        stops.push(GradientStop {
            offset: validated_offset,
            color: clamped,
        });
    }

    if stops.is_empty() {
        stops.push(GradientStop {
            offset: 0.0,
            color: white,
        });
    }

    // Sort by offset (stable: stops at one position keep their order, a hard stop).
    stops.sort_by(|a, b| a.offset.partial_cmp(&b.offset).unwrap());
    stops
}

/// `packGradientStops` (`WebGPUGradientUtils.ts:13-41`).
pub fn pack_gradient_stops(stops: &[GradientStop]) -> [[f32; 4]; MAX_GRADIENT_STOPS] {
    let last_color = stops.last().map(|s| s.color).unwrap_or(Rgba {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    });
    let last_blue_alpha = ((last_color.b * BLUE_CHANNEL_MULTIPLIER).floor()
        + last_color.a * ALPHA_ENCODING_LIMIT) as f32;

    let mut packed = [[0.0f32; 4]; MAX_GRADIENT_STOPS];
    for slot in packed.iter_mut() {
        *slot = [
            10000.0,
            last_color.r as f32,
            last_color.g as f32,
            last_blue_alpha,
        ];
    }
    for (i, stop) in stops.iter().take(MAX_GRADIENT_STOPS).enumerate() {
        packed[i] = [
            stop.offset as f32,
            stop.color.r as f32,
            stop.color.g as f32,
            ((stop.color.b * BLUE_CHANNEL_MULTIPLIER).floor() + stop.color.a * ALPHA_ENCODING_LIMIT)
                as f32,
        ];
    }
    packed
}

pub const EMPTY_GRADIENT_STOPS: [[f32; 4]; MAX_GRADIENT_STOPS] = [[0.0; 4]; MAX_GRADIENT_STOPS];
pub const EMPTY_FREEFORM: [[f32; 4]; MAX_FREEFORM_POINTS] = [[0.0; 4]; MAX_FREEFORM_POINTS];

// ---------------------------------------------------------------------------
// Node paint views (E3.1 P3: every read goes through `moda_scene::paint`)
// ---------------------------------------------------------------------------
//
// The paint the body draws is the node's top visible fill (`paint::top_fill`):
// its `type` is the route. No fill, or a colour that doesn't parse, draws
// nothing (there is no default fill or stroke colour).
// A paint stack draws each layer as its own clone (`layers`, E3.1 P6), so
// this is the layer being drawn.

/// The node's top fill paint (`None`: no fill).
pub fn fill_paint(node: &RawNode) -> Option<Paint> {
    scene_paint::top_fill(node).map(|f| f.paint)
}

/// The node's top stroke (`None`: no stroke paint).
pub fn top_stroke(node: &RawNode) -> Option<Stroke> {
    scene_paint::top_stroke(node)
}

fn is_gradient(paint: &Paint) -> bool {
    matches!(
        paint,
        Paint::Linear(_)
            | Paint::Radial(_)
            | Paint::Conic(_)
            | Paint::Diamond(_)
            | Paint::Freeform(_)
    )
}

/// A gradient's stop (or freeform point) colours, parsed as stops are:
/// unparseable → white; a freeform gradient's first [`MAX_FREEFORM_POINTS`].
fn gradient_colors(paint: &Paint) -> Vec<Rgba> {
    match paint {
        Paint::Freeform(f) => f
            .points
            .iter()
            .take(MAX_FREEFORM_POINTS)
            .map(|p| point_color(&p.color))
            .collect(),
        _ => parse_paint_stops(scene_paint::paint_stops(paint).unwrap_or(&[]))
            .iter()
            .map(|s| s.color)
            .collect(),
    }
}

/// `hasGradientFill`: the top fill is a gradient.
pub fn has_gradient_fill(node: &RawNode) -> bool {
    fill_paint(node).as_ref().is_some_and(is_gradient)
}

/// Fully transparent: what no paint (or an unparseable colour) draws.
pub const NO_PAINT: Rgba = Rgba {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 0.0,
};

/// The colour a solid fill paints; transparent for no fill or an unparseable
/// colour.
pub fn solid_fill_color(paint: Option<&Paint>) -> Rgba {
    parse_color(paint.and_then(scene_paint::solid_color)).unwrap_or(NO_PAINT)
}

/// Whether the fill can contribute alpha. This is intentionally independent
/// of [`super::FillRoute`]: the gradient pipeline can be selected solely for a
/// gradient STROKE, while its fill remains the node's ordinary solid color.
/// Image, video and shader fills always paint (their source alpha is not
/// knowable from node fields).
pub fn fill_paints(node: &RawNode) -> bool {
    match fill_paint(node) {
        Some(ref p) if is_gradient(p) => gradient_colors(p).iter().any(|c| c.a > 0.0),
        Some(Paint::Image(_) | Paint::Video(_) | Paint::Shader(_)) => true,
        p => solid_fill_color(p.as_ref()).a > 0.0,
    }
}

/// `getStrokeGradientType`: a gradient stroke is linear or radial ([Q6]:
/// conic and incomplete gradients read as the solid `stroke`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeGradientType {
    Linear,
    Radial,
}

fn stroke_gradient_type_of(stroke: &Stroke) -> Option<StrokeGradientType> {
    match stroke.paint {
        Paint::Linear(_) => Some(StrokeGradientType::Linear),
        Paint::Radial(_) | Paint::Diamond(_) => Some(StrokeGradientType::Radial),
        _ => None,
    }
}

pub fn stroke_gradient_type(node: &RawNode) -> Option<StrokeGradientType> {
    moda_scene::paint::with_top_stroke(node, |s| s.and_then(stroke_gradient_type_of))
}

/// `hasVisibleGradientStroke` (`WebGPUPathRenderer.ts:154-156`).
pub fn has_visible_gradient_stroke(node: &RawNode) -> bool {
    moda_scene::paint::with_top_stroke(node, |s| s.is_some_and(|s| stroke_gradient_type_of(s).is_some() && s.width > 0.0))
}

/// `hasStrokePaint`: the node has a visible stroke (any paint; a transparent
/// colour still counts, as hit testing sees the band).
pub fn has_stroke_paint(node: &RawNode) -> bool {
    moda_scene::paint::with_top_stroke(node, |s| s.is_some())
}

/// `getPaintedStrokeWidth`: the stroke width a body tessellates with, the top
/// stroke's width, `0.0` without a stroke.
pub fn painted_stroke_width(node: &RawNode) -> f64 {
    moda_scene::paint::with_top_stroke(node, |s| s.map_or(0.0, |s| s.width.max(0.0)))
}

fn stops_have_transparency(stops: &[PaintStop]) -> bool {
    stops.iter().any(|s| {
        s.color
            .as_str()
            .and_then(color_to_normalized_rgba)
            .is_some_and(|c| c.a < 1.0)
    })
}

/// `gradientHasTransparency`: the top fill is a gradient with a translucent
/// stop (or freeform point).
pub fn gradient_has_transparency(node: &RawNode) -> bool {
    match fill_paint(node) {
        Some(Paint::Freeform(f)) => {
            f.points.is_empty()
                || gradient_colors(&Paint::Freeform(f))
                    .iter()
                    .any(|c| c.a < 1.0)
        }
        Some(p) => scene_paint::paint_stops(&p).is_some_and(stops_have_transparency),
        None => false,
    }
}

/// `strokeGradientHasTransparency` (`utils/gradientHelpers.ts:120-137`).
pub fn stroke_gradient_has_transparency(node: &RawNode) -> bool {
    moda_scene::paint::with_top_stroke(node, |s| {
        s.is_some_and(|s| {
            stroke_gradient_type_of(s).is_some()
                && scene_paint::paint_stops(&s.paint).is_some_and(stops_have_transparency)
        })
    })
}

// ---------------------------------------------------------------------------
// Paint builders (gradientPaintBuilders.ts ports)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FillGradientPaint {
    pub fill_color: Rgba,
    /// 0 solid · 1 linear · 2 radial · 3 conic · 4 diamond · 5 freeform.
    pub fill_type: f32,
    pub gradient_start: [f32; 2],
    pub gradient_end: [f32; 2],
    pub gradient_rotation: f32,
    pub num_stops: u32,
    pub gradient_stops: [[f32; 4]; MAX_GRADIENT_STOPS],
    /// Freeform fills (fill type 5): [`pack_freeform`]'s points and colours;
    /// `num_stops` counts them.
    pub freeform_points: [[f32; 4]; MAX_FREEFORM_POINTS],
    pub freeform_colors: [[f32; 4]; MAX_FREEFORM_POINTS],
    /// `GradientOptions::shader_flags` of a stop gradient (0 otherwise).
    pub gradient_flags: f32,
}

#[derive(Debug, Clone)]
pub struct StrokeGradientPaint {
    pub stroke_color: Rgba,
    pub stroke_gradient_start: [f32; 2],
    pub stroke_gradient_end: [f32; 2],
    pub stroke_gradient_meta: [f32; 4],
    pub stroke_gradient_stops: [[f32; 4]; MAX_GRADIENT_STOPS],
}

fn pt(p: &PaintPoint) -> (f64, f64) {
    (p.x, p.y)
}

/// `createLinearGradient` (`WebGPUGradientSystem.ts:484-516`).
fn create_linear(l: &LinearGradientPaint) -> ((f64, f64), (f64, f64), Vec<GradientStop>) {
    let valid_start = clamp_point(pt(&l.start));
    let mut valid_end = clamp_point(pt(&l.end));
    let dx = valid_end.0 - valid_start.0;
    let dy = valid_end.1 - valid_start.1;
    if (dx * dx + dy * dy).sqrt() < 1.0 {
        valid_end.0 = valid_start.0 + 50.0;
    }
    (valid_start, valid_end, parse_repeating_stops(&l.stops, l.options.repeat))
}

/// `createRadialGradient` (`WebGPUGradientSystem.ts:521-553`).
fn create_radial(r: &RadialGradientPaint) -> ((f64, f64), (f64, f64), f64, Vec<GradientStop>) {
    let valid_center = clamp_point(pt(&r.center));
    let mut final_rx = r.radius_x.max(0.0).min(MAX_RADIAL_RADIUS);
    let mut final_ry = r.radius_y.max(0.0).min(MAX_RADIAL_RADIUS);
    let final_rotation = normalize_rotation_degrees(r.rotation).to_radians();
    if final_rx == 0.0 {
        final_rx = 50.0;
    }
    if final_ry == 0.0 {
        final_ry = 50.0;
    }
    (
        valid_center,
        (final_rx, final_ry),
        final_rotation,
        parse_repeating_stops(&r.stops, r.options.repeat),
    )
}

/// `createConicGradient` (`WebGPUGradientSystem.ts:560-580`).
fn create_conic(c: &ConicGradientPaint) -> ((f64, f64), f64, Vec<GradientStop>) {
    let valid_center = clamp_point(pt(&c.center));
    let angle_radians = normalize_rotation_degrees(c.angle).to_radians();
    (valid_center, angle_radians, parse_paint_stops(&c.stops))
}

/// A freeform point's colour, parsed as a stop's (unparseable or a variable
/// reference → white), clamped to 0..1.
fn point_color(color: &Value) -> Rgba {
    let c = color
        .as_str()
        .and_then(color_to_normalized_rgba)
        .unwrap_or(Rgba {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        });
    Rgba {
        r: c.r.clamp(0.0, 1.0),
        g: c.g.clamp(0.0, 1.0),
        b: c.b.clamp(0.0, 1.0),
        a: c.a.clamp(0.0, 1.0),
    }
}

/// A freeform gradient packed for `calculateFreeformColor`
/// (path-gradient.wgsl): `(count, points, colours)`. The first
/// [`MAX_FREEFORM_POINTS`] points count (as `placeFreeform` in
/// `core/scene/freeformGradient.ts`): lone points first, then each line of
/// two or more points (by first appearance), its points in array order,
/// linked to the next (`link` 1) up to the last (`link` 2). A line of one
/// point is a lone point. Positions stay in the 0-100 box space and spread
/// in % (the shader scales both by the gradient box).
pub fn pack_freeform(
    f: &FreeformGradientPaint,
) -> (
    u32,
    [[f32; 4]; MAX_FREEFORM_POINTS],
    [[f32; 4]; MAX_FREEFORM_POINTS],
) {
    let finite = |v: f64, or: f64| if v.is_finite() { v } else { or };
    let mut lone: Vec<&scene_paint::FreeformGradientPoint> = Vec::new();
    let mut lines: Vec<(f64, Vec<&scene_paint::FreeformGradientPoint>)> = Vec::new();
    for p in f.points.iter().take(MAX_FREEFORM_POINTS) {
        match p.line.filter(|l| l.is_finite()) {
            Some(id) => match lines.iter_mut().find(|(l, _)| *l == id) {
                Some((_, pts)) => pts.push(p),
                None => lines.push((id, vec![p])),
            },
            None => lone.push(p),
        }
    }
    for (_, pts) in &lines {
        if pts.len() == 1 {
            lone.push(pts[0]);
        }
    }
    let mut ordered: Vec<(&scene_paint::FreeformGradientPoint, f32)> =
        lone.into_iter().map(|p| (p, 0.0)).collect();
    for (_, pts) in lines.iter().filter(|(_, pts)| pts.len() > 1) {
        for (i, p) in pts.iter().enumerate() {
            ordered.push((p, if i + 1 < pts.len() { 1.0 } else { 2.0 }));
        }
    }
    let mut points = [[0.0f32; 4]; MAX_FREEFORM_POINTS];
    let mut colors = [[0.0f32; 4]; MAX_FREEFORM_POINTS];
    for (i, (p, link)) in ordered.iter().enumerate() {
        let spread = p.spread.map_or(0.0, |s| finite(s, 0.0).clamp(0.0, 100.0));
        points[i] = [
            finite(p.x, 50.0) as f32,
            finite(p.y, 50.0) as f32,
            spread as f32,
            *link,
        ];
        let c = point_color(&p.color);
        colors[i] = [c.r as f32, c.g as f32, c.b as f32, c.a as f32];
    }
    (ordered.len() as u32, points, colors)
}

/// Each packed freeform point's colour slope (box px) and reach, as
/// `withSlopes` in `core/scene/freeformGradient.ts`: the least-squares fit
/// (weights `1/d⁴`, ridge `1e-3` of the trace) of the colour change to every
/// other packed point; the reach is twice the nearest other point's distance.
/// Line points (`link` > 0) keep no slope. `size` is the gradient box's size.
pub fn freeform_slopes(
    count: u32,
    points: &[[f32; 4]; MAX_FREEFORM_POINTS],
    colors: &[[f32; 4]; MAX_FREEFORM_POINTS],
    size: [f32; 2],
) -> (
    [[f32; 4]; 2 * MAX_FREEFORM_POINTS],
    [[f32; 4]; MAX_FREEFORM_POINTS / 4],
) {
    let n = (count as usize).min(MAX_FREEFORM_POINTS);
    let at = |i: usize| {
        (
            points[i][0] as f64 * 0.01 * size[0] as f64,
            points[i][1] as f64 * 0.01 * size[1] as f64,
        )
    };
    let mut slopes = [[0.0f32; 4]; 2 * MAX_FREEFORM_POINTS];
    let mut reach = [[0.0f32; 4]; MAX_FREEFORM_POINTS / 4];
    for i in 0..n {
        if points[i][3] > 0.5 {
            continue;
        }
        let (px, py) = at(i);
        let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
        let mut near = f64::INFINITY;
        let mut bx = [0.0f64; 4];
        let mut by = [0.0f64; 4];
        for j in 0..n {
            let (qx, qy) = at(j);
            let (dx, dy) = (qx - px, qy - py);
            let d2 = dx * dx + dy * dy;
            if j == i || d2 <= 1e-12 {
                continue;
            }
            near = near.min(d2.sqrt());
            let w = 1.0 / (d2 * d2);
            a += w * dx * dx;
            b += w * dx * dy;
            c += w * dy * dy;
            for k in 0..4 {
                let dc = colors[j][k] as f64 - colors[i][k] as f64;
                bx[k] += w * dx * dc;
                by[k] += w * dy * dc;
            }
        }
        if !near.is_finite() {
            continue;
        }
        let ridge = 1e-3 * (a + c);
        let (aa, cc) = (a + ridge, c + ridge);
        let det = aa * cc - b * b;
        if !(det > 0.0) {
            continue;
        }
        for k in 0..4 {
            slopes[2 * i][k] = ((cc * bx[k] - b * by[k]) / det) as f32;
            slopes[2 * i + 1][k] = ((aa * by[k] - b * bx[k]) / det) as f32;
        }
        reach[i / 4][i % 4] = (2.0 * near) as f32;
    }
    (slopes, reach)
}

/// `buildFillGradientPaint` (`gradientPaintBuilders.ts:72-140`) for one fill
/// paint. `transparent_solid_fill` mirrors `options.transparentSolidFill` (the
/// pattern/shader gradient-stroke overlay case).
pub fn fill_gradient_paint(
    paint: Option<&Paint>,
    transparent_solid_fill: bool,
) -> FillGradientPaint {
    let white = Rgba {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    match paint {
        Some(Paint::Linear(l)) => {
            let (start, end, stops) = create_linear(l);
            return FillGradientPaint {
                fill_color: white,
                fill_type: 1.0,
                gradient_start: [start.0 as f32, start.1 as f32],
                gradient_end: [end.0 as f32, end.1 as f32],
                gradient_rotation: 0.0,
                num_stops: stops.len() as u32,
                gradient_stops: pack_gradient_stops(&stops),
                freeform_points: EMPTY_FREEFORM,
                freeform_colors: EMPTY_FREEFORM,
                gradient_flags: paint.map_or(0.0, |p| p.gradient_options().shader_flags()),
            };
        }
        Some(Paint::Conic(c)) => {
            let (center, angle, stops) = create_conic(c);
            return FillGradientPaint {
                fill_color: white,
                fill_type: 3.0,
                gradient_start: [center.0 as f32, center.1 as f32],
                gradient_end: [0.0, 0.0],
                gradient_rotation: angle as f32,
                num_stops: stops.len() as u32,
                gradient_stops: pack_gradient_stops(&stops),
                freeform_points: EMPTY_FREEFORM,
                freeform_colors: EMPTY_FREEFORM,
                gradient_flags: paint.map_or(0.0, |p| p.gradient_options().shader_flags()),
            };
        }
        // Diamond is placed like the radial and ramps by the L1 distance
        // (fill type 4 in the gradient WGSL; E3.1 P6).
        Some(p @ (Paint::Radial(r) | Paint::Diamond(r))) => {
            let (center, radii, rotation, stops) = create_radial(r);
            return FillGradientPaint {
                fill_color: white,
                fill_type: if matches!(p, Paint::Diamond(_)) { 4.0 } else { 2.0 },
                gradient_start: [center.0 as f32, center.1 as f32],
                gradient_end: [radii.0 as f32, radii.1 as f32],
                gradient_rotation: rotation as f32,
                num_stops: stops.len() as u32,
                gradient_stops: pack_gradient_stops(&stops),
                freeform_points: EMPTY_FREEFORM,
                freeform_colors: EMPTY_FREEFORM,
                gradient_flags: paint.map_or(0.0, |p| p.gradient_options().shader_flags()),
            };
        }
        Some(Paint::Freeform(f)) => {
            let (count, freeform_points, freeform_colors) = pack_freeform(f);
            return FillGradientPaint {
                fill_color: white,
                fill_type: 5.0,
                gradient_start: [0.0, 0.0],
                gradient_end: [0.0, 0.0],
                gradient_rotation: 0.0,
                num_stops: count,
                gradient_stops: EMPTY_GRADIENT_STOPS,
                freeform_points,
                freeform_colors,
                gradient_flags: 0.0,
            };
        }
        _ => {}
    }

    FillGradientPaint {
        fill_color: if transparent_solid_fill {
            Rgba {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            }
        } else {
            solid_fill_color(paint)
        },
        fill_type: 0.0,
        gradient_start: [0.0, 0.0],
        gradient_end: [0.0, 0.0],
        gradient_rotation: 0.0,
        num_stops: 0,
        gradient_stops: EMPTY_GRADIENT_STOPS,
        freeform_points: EMPTY_FREEFORM,
        freeform_colors: EMPTY_FREEFORM,
        gradient_flags: 0.0,
    }
}

/// [`fill_gradient_paint`] for the node's top fill.
pub fn build_fill_gradient_paint(
    node: &RawNode,
    transparent_solid_fill: bool,
) -> FillGradientPaint {
    fill_gradient_paint(fill_paint(node).as_ref(), transparent_solid_fill)
}

/// `buildStrokeGradientPaint` (`gradientPaintBuilders.ts:148-196`) for one
/// stroke (`None`: no stroke, transparent).
pub fn stroke_gradient_paint(stroke: Option<&Stroke>) -> StrokeGradientPaint {
    let white = Rgba {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    // E8.18: meta.w = how the gradient lies on the stroke (0 within, 1 along, 2 across).
    // Plus 4 × the paint's interpolation / dither flags (`GradientOptions::shader_flags`), decoded in the WGSL.
    let mode = match stroke.map(Stroke::gradient_mode) {
        Some(StrokeGradientMode::Along) => 1.0,
        Some(StrokeGradientMode::Across) => 2.0,
        _ => 0.0,
    } + 4.0 * stroke.map_or(0.0, |s| s.paint.gradient_options().shader_flags());
    match stroke.map(|s| &s.paint) {
        Some(Paint::Linear(l)) => {
            let (start, end, stops) = create_linear(l);
            StrokeGradientPaint {
                stroke_color: white,
                stroke_gradient_start: [start.0 as f32, start.1 as f32],
                stroke_gradient_end: [end.0 as f32, end.1 as f32],
                stroke_gradient_meta: [1.0, stops.len() as f32, 0.0, mode],
                stroke_gradient_stops: pack_gradient_stops(&stops),
            }
        }
        Some(p @ (Paint::Radial(r) | Paint::Diamond(r))) => {
            let (center, radii, rotation, stops) = create_radial(r);
            let kind = if matches!(p, Paint::Diamond(_)) {
                4.0
            } else {
                2.0
            };
            StrokeGradientPaint {
                stroke_color: white,
                stroke_gradient_start: [center.0 as f32, center.1 as f32],
                stroke_gradient_end: [radii.0 as f32, radii.1 as f32],
                stroke_gradient_meta: [kind, stops.len() as f32, rotation as f32, mode],
                stroke_gradient_stops: pack_gradient_stops(&stops),
            }
        }
        paint => StrokeGradientPaint {
            stroke_color: solid_stroke_color(paint),
            stroke_gradient_start: [0.0, 0.0],
            stroke_gradient_end: [0.0, 0.0],
            stroke_gradient_meta: [0.0, 0.0, 0.0, 0.0],
            stroke_gradient_stops: EMPTY_GRADIENT_STOPS,
        },
    }
}

/// The colour a solid stroke paints; transparent for no stroke, a paint the
/// stroke lanes don't draw yet (image, video, shader, conic), or an
/// unparseable colour.
pub fn solid_stroke_color(paint: Option<&Paint>) -> Rgba {
    parse_color(paint.and_then(scene_paint::solid_color)).unwrap_or(NO_PAINT)
}

/// [`stroke_gradient_paint`] for the node's top stroke.
pub fn build_stroke_gradient_paint(node: &RawNode) -> StrokeGradientPaint {
    stroke_gradient_paint(top_stroke(node).as_ref())
}

/// `strokeWidth || 1`: the instance stroke width (the top stroke's).

pub fn instance_stroke_width(node: &RawNode) -> f32 {
    let w = scene_paint::node_stroke_style(node).width.unwrap_or(0.0);
    (if w == 0.0 { 1.0 } else { w }) as f32
}

/// An axis of a path's bounds narrower than this (node-local units) takes the
/// stroke's width across it instead ([`gradient_box`]).
const GRADIENT_BOX_MIN_EXTENT: f64 = 1e-6;

/// The box a node's fill and stroke gradients span, node-local
/// `[x, y, w, h]`: the path's geometric bounds (`path_bounds`, from
/// `tess::path_bounds`), never the stroke — Illustrator's convention, and the
/// box the gradient handles and the SVG / PDF exports (`paintBox`) use. A
/// straight horizontal or vertical path has no extent across itself; that axis
/// spans the stroke's width, centred on the path. `fallback` (the tessellated
/// bounds) when the path has no bounds.
pub fn gradient_box(
    path_bounds: Option<[f64; 4]>,
    stroke_width: f64,
    fallback: [f32; 4],
) -> [f32; 4] {
    let Some([x0, y0, x1, y1]) = path_bounds else {
        return fallback;
    };
    let half = (stroke_width / 2.0).max(0.0);
    let axis = |lo: f64, hi: f64| {
        if hi - lo > GRADIENT_BOX_MIN_EXTENT {
            (lo, hi - lo)
        } else {
            ((lo + hi) / 2.0 - half, 2.0 * half)
        }
    };
    let (x, w) = axis(x0, x1);
    let (y, h) = axis(y0, y1);
    [x as f32, y as f32, w as f32, h as f32]
}

/// Pack the full `PathGradientData` instance for one node
/// (`WebGPUPathRenderer.renderGradientPathInstances`,
/// `WebGPUPathRenderer.ts:1670-1805`). `vertex_offset`/`vertex_count` are
/// executor-time values; pass 0 at plan time and fix up at upload.
#[allow(clippy::too_many_arguments)]
pub fn build_gradient_instance(
    node: &RawNode,
    model_matrix: [f32; 16],
    depth: f32,
    bounds: [f32; 4],
    page_clip_bounds: [f32; 4],
    stroke_only_overlay: bool,
) -> GradientInstance {
    let opacity = node.f64("opacity").unwrap_or(1.0) as f32;
    let fill = build_fill_gradient_paint(node, stroke_only_overlay);
    let stroke = build_stroke_gradient_paint(node);
    let stroke_width = instance_stroke_width(node);
    let (freeform_slopes, freeform_reach) = if fill.fill_type == 5.0 {
        freeform_slopes(
            fill.num_stops,
            &fill.freeform_points,
            &fill.freeform_colors,
            [bounds[2], bounds[3]],
        )
    } else {
        (
            [[0.0; 4]; 2 * MAX_FREEFORM_POINTS],
            [[0.0; 4]; MAX_FREEFORM_POINTS / 4],
        )
    };

    GradientInstance {
        model_matrix,
        color: [
            fill.fill_color.r as f32,
            fill.fill_color.g as f32,
            fill.fill_color.b as f32,
            fill.fill_color.a as f32 * opacity,
        ],
        path_info: [0.0, 0.0, depth, fill.fill_type],
        stroke_color: [
            stroke.stroke_color.r as f32,
            stroke.stroke_color.g as f32,
            stroke.stroke_color.b as f32,
            stroke.stroke_color.a as f32 * opacity,
        ],
        gradient_start: fill.gradient_start,
        gradient_end: fill.gradient_end,
        bounds,
        num_stops: fill.num_stops,
        stroke_width,
        gradient_rotation: fill.gradient_rotation,
        gradient_flags: fill.gradient_flags,
        gradient_stops: fill.gradient_stops,
        page_clip_bounds,
        stroke_gradient_start: stroke.stroke_gradient_start,
        stroke_gradient_end: stroke.stroke_gradient_end,
        stroke_gradient_meta: stroke.stroke_gradient_meta,
        stroke_gradient_stops: stroke.stroke_gradient_stops,
        freeform_points: fill.freeform_points,
        freeform_colors: fill.freeform_colors,
        freeform_slopes,
        freeform_reach,
    }
}

/// Gradient params for the SDF gradient shape instance (`shape-gradient.wgsl`),
/// mirroring `enqueueGradientShape` (`WebGPUShapeSystem.ts:2219-2254`) for a
/// linear or radial fill (the only two SDF-eligible gradients; the classifier
/// decided). Returns `(gradient_start, gradient_end, gradient_rotation,
/// num_stops, gradient_stops)`.
pub fn shape_sdf_gradient_params(
    paint: &Paint,
) -> ([f32; 2], [f32; 2], f32, u32, [[f32; 4]; MAX_GRADIENT_STOPS]) {
    let g = fill_gradient_paint(Some(paint), false);
    (
        g.gradient_start,
        g.gradient_end,
        g.gradient_rotation,
        g.num_stops,
        g.gradient_stops,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn raw(v: moda_scene::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    fn stroked(stroke: moda_scene::value::Value) -> RawNode {
        raw(json!({"id": "n", "type": "rectangle", "strokes": [stroke]}))
    }

    #[test]
    fn painted_stroke_width_gates_on_paint_route_and_width() {
        // no stroke → 0.0 (there is no default black stroke)
        assert_eq!(
            painted_stroke_width(&raw(json!({"id": "n", "type": "rectangle"}))),
            0.0
        );
        // a stroke without a width is not a stroke layer
        assert_eq!(
            painted_stroke_width(&stroked(
                json!({"id": "stroke-0", "paint": {"type": "solid", "color": "#ff0000"}})
            )),
            0.0
        );
        // a hidden stroke paints nothing
        assert_eq!(
            painted_stroke_width(&stroked(json!({"id": "stroke-0", "visible": false,
                "paint": {"type": "solid", "color": "#ff0000"}, "width": 2.0}))),
            0.0
        );
        // solid stroke → width; transparent solid is still a route
        assert_eq!(
            painted_stroke_width(&stroked(
                json!({"id": "stroke-0", "paint": {"type": "solid", "color": "#ff0000"}, "width": 2.0})
            )),
            2.0
        );
        assert_eq!(
            painted_stroke_width(&stroked(
                json!({"id": "stroke-0", "paint": {"type": "solid", "color": "rgba(0,0,0,0)"}, "width": 2.0})
            )),
            2.0
        );
        // gradient strokes → width
        assert_eq!(
            painted_stroke_width(&stroked(json!({"id": "stroke-0", "width": 2.0,
                "paint": {"type": "linear", "start": {"x": 0.0, "y": 0.0}, "end": {"x": 1.0, "y": 0.0},
                          "stops": [{"offset": 0.0, "color": "#ff0000"}, {"offset": 1.0, "color": "#0000ff"}]}}))),
            2.0
        );
        assert_eq!(
            painted_stroke_width(&stroked(json!({"id": "stroke-0", "width": 2.0,
                "paint": {"type": "radial", "center": {"x": 50.0, "y": 50.0}, "radiusX": 50.0, "radiusY": 50.0,
                          "rotation": 0.0, "stops": []}}))),
            2.0
        );
        // width gate still applies with a paint
        assert_eq!(
            painted_stroke_width(&stroked(
                json!({"id": "stroke-0", "paint": {"type": "solid", "color": "#ff0000"}, "width": 0.0})
            )),
            0.0
        );
    }

    #[test]
    fn pack_stops_byte_encoding() {
        // floor(b·255) + a·0.99999 — #3a86ff at alpha 0.6:
        // b = 255/255 = 1 → floor(255) = 255; a·0.99999 = 0.599994.
        let stops = vec![
            GradientStop {
                offset: 0.0,
                color: Rgba {
                    r: 1.0,
                    g: 0.0,
                    b: 110.0 / 255.0,
                    a: 1.0,
                },
            },
            GradientStop {
                offset: 1.0,
                color: Rgba {
                    r: 58.0 / 255.0,
                    g: 134.0 / 255.0,
                    b: 1.0,
                    a: 0.6,
                },
            },
        ];
        let packed = pack_gradient_stops(&stops);
        assert_eq!(packed[0][0], 0.0);
        assert_eq!(packed[0][3], (110.0f64 + 0.99999) as f32);
        assert_eq!(packed[1][3], (255.0f64 + 0.6 * 0.99999) as f32);
        // Sentinel slots: offset 10000, last stop's color.
        for slot in &packed[2..] {
            assert_eq!(slot[0], 10000.0);
            assert_eq!(slot[1], (58.0 / 255.0f64) as f32);
            assert_eq!(slot[3], (255.0f64 + 0.6 * 0.99999) as f32);
        }
    }

    #[test]
    fn stop_parse_clamp_sort_keeps_positions() {
        // A hard stop (two stops at 0.5) and a near-hard one (0.5 / 0.501) keep their positions: the shaders step
        // between stops closer than 0.001 and interpolate the rest, as CSS draws them.
        let stops_json = json!([0, "#ef476f", 0.5, "#ef476f", 0.5, "#118ab2", 1, "#118ab2"]);
        let stops = parse_gradient_stops(Some(stops_json.as_array().unwrap()));
        assert_eq!(stops.iter().map(|s| s.offset).collect::<Vec<_>>(), vec![0.0, 0.5, 0.5, 1.0]);
        assert!(stops[1].color.r > stops[1].color.b && stops[2].color.b > stops[2].color.r, "equal stops keep their order");
        let stops_json = json!([0, "#ef476f", 0.501, "#118ab2", 0.5, "#ef476f", 1.5, "#118ab2"]);
        let stops = parse_gradient_stops(Some(stops_json.as_array().unwrap()));
        assert_eq!(stops.iter().map(|s| s.offset).collect::<Vec<_>>(), vec![0.0, 0.5, 0.501, 1.0]);

        // Unparsable color → white; missing stops → single white stop.
        let stops = parse_gradient_stops(None);
        assert_eq!(stops.len(), 1);
        assert_eq!(stops[0].color.r, 1.0);
    }

    #[test]
    fn fill_dispatch_copies_the_bugs() {
        // A radial paint with zero radii and no stops: the radii default to
        // 50/50 and the stops to a single white stop. (Was: `fillPriority`
        // 'radial-gradient' with only linear fields present; the stored paint
        // type now routes, so that quirk is gone.)
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "radial", "center": {"x": 50, "y": 50},
                       "radiusX": 0, "radiusY": 0, "rotation": 0, "stops": []}}]
        }));
        assert!(has_gradient_fill(&node));
        let paint = build_fill_gradient_paint(&node, false);
        assert_eq!(paint.fill_type, 2.0);
        assert_eq!(paint.gradient_start, [50.0, 50.0]);
        assert_eq!(paint.gradient_end, [50.0, 50.0]);
        assert_eq!(paint.num_stops, 1);

        // Degenerate linear gradient (start == end) extends +50 in x.
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 50, "y": 50}, "end": {"x": 50, "y": 50}, "stops": [{"offset": 0, "color": "#ffbe0b"}, {"offset": 1, "color": "#fb5607"}]}}]
        }));
        let paint = build_fill_gradient_paint(&node, false);
        assert_eq!(paint.fill_type, 1.0);
        assert_eq!(paint.gradient_end, [100.0, 50.0]);

        // Points outside the box are kept (gl-beyond-100: start x −50), within ±100,000 %.
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": -50, "y": 50}, "end": {"x": 250, "y": 50}, "stops": [{"offset": 0, "color": "#ef476f"}, {"offset": 1, "color": "#118ab2"}]}}]
        }));
        let paint = build_fill_gradient_paint(&node, false);
        assert_eq!(paint.gradient_start, [-50.0, 50.0]);
        assert_eq!(paint.gradient_end, [250.0, 50.0]);
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": -800, "y": 900}, "end": {"x": 3e6, "y": 50}, "stops": []}}]
        }));
        let paint = build_fill_gradient_paint(&node, false);
        assert_eq!(paint.gradient_start, [-800.0, 900.0]);
        assert_eq!(paint.gradient_end, [100_000.0, 50.0]);
    }

    #[test]
    fn conic_angle_and_rotation_normalization() {
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "conic", "center": {"x": 30, "y": 70}, "angle": -90, "stops": [{"offset": 0, "color": "#000000"}, {"offset": 1, "color": "#ffffff"}]}}]
        }));
        let paint = build_fill_gradient_paint(&node, false);
        assert_eq!(paint.fill_type, 3.0);
        // −90° normalizes to 270° then radians.
        assert!((paint.gradient_rotation - (270.0f64.to_radians() as f32)).abs() < 1e-6);
    }

    #[test]
    fn stroke_gradient_meta_and_conic_gap() {
        // A conic stroke is NOT a gradient stroke (Q-18): not drawn yet, so
        // it paints nothing (transparent), and a solid stroke paints its colour.
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "strokes": [{"id": "stroke-0", "paint": {"type": "conic", "center": {"x": 50, "y": 50}, "angle": 0,
                         "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}]}, "width": 4}]
        }));
        assert_eq!(stroke_gradient_type(&node), None);
        let paint = build_stroke_gradient_paint(&node);
        assert_eq!(paint.stroke_gradient_meta, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(paint.stroke_color.a, 0.0);
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#102030"}, "width": 4}]
        }));
        assert_eq!(stroke_gradient_type(&node), None);
        let paint = build_stroke_gradient_paint(&node);
        assert_eq!(paint.stroke_gradient_meta, [0.0, 0.0, 0.0, 0.0]);
        assert!((paint.stroke_color.r - 16.0 / 255.0).abs() < 1e-9);

        // Linear stroke gradient: meta [1, numStops, 0, 0], white color.
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "strokes": [{"id": "stroke-0", "paint": {"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0}, "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 0.5, "color": "rgba(0, 255, 0, 0.5)"}, {"offset": 1, "color": "#0000ff"}]}, "width": 4}]
        }));
        assert_eq!(
            stroke_gradient_type(&node),
            Some(StrokeGradientType::Linear)
        );
        assert!(stroke_gradient_has_transparency(&node));
        let paint = build_stroke_gradient_paint(&node);
        assert_eq!(paint.stroke_gradient_meta, [1.0, 3.0, 0.0, 0.0]);
        assert_eq!(paint.stroke_color.r, 1.0);

        // E8.18: meta.w carries how the gradient lies on the stroke.
        for (mode, w) in [("within", 0.0), ("along", 1.0), ("across", 2.0)] {
            let node = raw(json!({
                "id": "n", "type": "rectangle",
                "strokes": [{"id": "stroke-0", "gradientMode": mode, "width": 4, "paint": {"type": "radial",
                    "center": {"x": 50, "y": 50}, "radiusX": 50, "radiusY": 50, "rotation": 0,
                    "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}]}}]
            }));
            assert_eq!(
                build_stroke_gradient_paint(&node).stroke_gradient_meta[3],
                w,
                "{mode}"
            );
        }
    }

    #[test]
    fn transparency_predicates() {
        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "radial", "center": {"x": 50, "y": 50}, "radiusX": 70, "radiusY": 70, "rotation": 0, "stops": [{"offset": 0, "color": "#ef476f"}, {"offset": 1, "color": "rgba(239, 71, 111, 0)"}]}}]
        }));
        assert!(gradient_has_transparency(&node));

        let node = raw(json!({
            "id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 50}, "end": {"x": 100, "y": 50}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}]
        }));
        assert!(!gradient_has_transparency(&node));
    }

    #[test]
    fn freeform_packs_lone_points_then_lines() {
        let paint: Paint = moda_scene::value::from_value(json!({"type": "freeform", "points": [
            {"x": 10, "y": 20, "color": "#ff0000", "line": 7},
            {"x": 30, "y": 40, "color": "#00ff00"},
            {"x": 50, "y": 60, "color": "#0000ff", "line": 7, "spread": 150},
            {"x": 70, "y": 80, "color": "not a colour", "line": 9},
            {"x": 90, "y": 10, "color": "rgba(0, 0, 0, 0.5)", "line": 7}
        ]}))
        .unwrap();
        let Paint::Freeform(f) = &paint else {
            unreachable!()
        };
        let (count, points, colors) = pack_freeform(f);
        assert_eq!(count, 5);
        // Lone point, then the one-point line 9 (a lone point), then line 7 in array order.
        assert_eq!(points[0], [30.0, 40.0, 0.0, 0.0]);
        assert_eq!(points[1], [70.0, 80.0, 0.0, 0.0]);
        assert_eq!(
            colors[1],
            [1.0, 1.0, 1.0, 1.0],
            "an unparseable colour is white, as a stop"
        );
        assert_eq!(points[2], [10.0, 20.0, 0.0, 1.0]);
        assert_eq!(points[3], [50.0, 60.0, 100.0, 1.0], "spread clamps to 100");
        assert_eq!(points[4], [90.0, 10.0, 0.0, 2.0]);
        assert_eq!(colors[4], [0.0, 0.0, 0.0, 0.5]);

        let node =
            raw(json!({"id": "n", "type": "rectangle", "fills": [{"id": "f", "paint": paint}]}));
        let fill = build_fill_gradient_paint(&node, false);
        assert_eq!(fill.fill_type, 5.0);
        assert_eq!(fill.num_stops, 5);
        assert!(has_gradient_fill(&node) && fill_paints(&node));
        assert!(gradient_has_transparency(&node), "a translucent point");

        // At most MAX_FREEFORM_POINTS points draw.
        let many: Vec<_> = (0..20)
            .map(|i| json!({"x": i, "y": 0, "color": "#000"}))
            .collect();
        let Paint::Freeform(f) =
            moda_scene::value::from_value(json!({"type": "freeform", "points": many})).unwrap()
        else {
            unreachable!()
        };
        assert_eq!(pack_freeform(&f).0 as usize, MAX_FREEFORM_POINTS);
    }

    #[test]
    fn a_freeform_stroke_paints_nothing() {
        let node = raw(
            json!({"id": "n", "type": "rectangle", "strokes": [{"id": "s", "width": 4,
            "paint": {"type": "freeform", "points": [{"x": 0, "y": 0, "color": "#ff0000"}]}}]}),
        );
        assert_eq!(stroke_gradient_type(&node), None);
        assert_eq!(build_stroke_gradient_paint(&node).stroke_color.a, 0.0);
    }

    #[test]
    fn instance_layout_matches_wgsl_offsets() {
        use core::mem::offset_of;
        // Float offsets from the PathGradientData WGSL struct.
        assert_eq!(offset_of!(GradientInstance, color), 16 * 4);
        assert_eq!(offset_of!(GradientInstance, path_info), 20 * 4);
        assert_eq!(offset_of!(GradientInstance, stroke_color), 24 * 4);
        assert_eq!(offset_of!(GradientInstance, gradient_start), 28 * 4);
        assert_eq!(offset_of!(GradientInstance, gradient_end), 30 * 4);
        assert_eq!(offset_of!(GradientInstance, bounds), 32 * 4);
        assert_eq!(offset_of!(GradientInstance, num_stops), 36 * 4);
        assert_eq!(offset_of!(GradientInstance, stroke_width), 37 * 4);
        assert_eq!(offset_of!(GradientInstance, gradient_rotation), 38 * 4);
        assert_eq!(offset_of!(GradientInstance, gradient_stops), 40 * 4);
        assert_eq!(offset_of!(GradientInstance, page_clip_bounds), 72 * 4);
        assert_eq!(offset_of!(GradientInstance, stroke_gradient_start), 76 * 4);
        assert_eq!(offset_of!(GradientInstance, stroke_gradient_end), 78 * 4);
        assert_eq!(offset_of!(GradientInstance, stroke_gradient_meta), 80 * 4);
        assert_eq!(offset_of!(GradientInstance, stroke_gradient_stops), 84 * 4);
        assert_eq!(offset_of!(GradientInstance, freeform_points), 116 * 4);
        assert_eq!(offset_of!(GradientInstance, freeform_colors), 180 * 4);
        assert_eq!(offset_of!(GradientInstance, freeform_slopes), 244 * 4);
        assert_eq!(offset_of!(GradientInstance, freeform_reach), 372 * 4);
        assert_eq!(core::mem::size_of::<GradientInstance>(), 1552);
    }
}
