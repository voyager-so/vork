//! Plan side of the capture shadows: drop and inner shadows for every shape
//! the SDF shadow lane (`effects::shadow`, rect / ellipse silhouettes only)
//! can't draw — paths (vector networks, booleans, outlined text), stars,
//! polygons, lines, partial ellipses, distorted nodes — and rich text's inner
//! shadows (its drop shadows are the text lane's own glyph passes).
//!
//! Figma draws a layer's shadows from the layer's shape: a drop shadow is the
//! shape moved by the offset and blurred (σ = blur / 2), knocked out under
//! the layer unless "show behind"; an inner shadow is the shape's outside
//! moved and blurred, clipped to the shape. Spread applies to rectangles,
//! ellipses and frames only (Figma's `DropShadowEffect.spread`; the SDF lane,
//! `effects::shadow`), so it plays no part here: a path, star, polygon, line,
//! text or fill-less frame casts its plain silhouette whatever the spread, as
//! in Figma, and a shadow that only spreads draws nothing.
//!
//! The shape is the node's SILHOUETTE ([`silhouette_node`]): every visible
//! fill and its widest stroke, painted opaque black, at opacity 1. A fill's
//! own opacity doesn't weaken the shadow (Figma: fill opacity keeps the shadow,
//! layer opacity fades it) — the node's effective opacity multiplies each
//! shadow's alpha instead, like the SDF lane.
//!
//! One capture group per shadow kind (`LayerBlurGroup` with
//! `LayerFilters::shadow`), rendered in the layer-filter pre-pass; its result
//! holds the shadows alone, and the main pass composites it at the body's z
//! (drop, drawn before the body in its bucket) or z + `INNER_SHADOW_Z_OFFSET`
//! (inner, over the body), clipped to the node's effective clip.
//!
//! Glows (Illustrator's Outer and Inner Glow, `glow` / `inner-glow`) are the
//! same passes with no offset and blur = the glow's radius, on every shape
//! and on text: an outer glow shown behind, an inner glow clipped to the
//! shape, from its outside (edge) or the shape itself (center). Each glow
//! mode is its own group, composited with that mode (`CaptureKind::Glow` /
//! `InnerGlow`, `seams::CaptureBlend`).
//!
//! The capture region is the silhouette's drawn bounds grown by the shadows'
//! reach, cut to the part of the target the shadows can reach (the viewport
//! grown by the same reach), so it stays bounded at any zoom and the capture
//! is at full target resolution: the knockout and the inner-shadow clip stay
//! as crisp as the body.

use moda_scene::effect::{DropShadow, InnerShadow};
use moda_scene::types::RawNode;
use moda_scene::Scene;

use crate::backdrop::plan::{PlanCtx, Rect};
use crate::seams::{CaptureBlend, LayerBlurGroup, LayerFilters, ShadowFilter, ShadowPass};

/// wgpu texture dimension ceiling (as the layer blur).
const MAX_TEXTURE_DIM: f64 = 8192.0;
/// Capture area budget (as the layer blur): beyond it the capture scale drops.
const MAX_CAPTURE_AREA_PX: f64 = 4096.0 * 4096.0;
/// Empty border kept around the capture, in capture px: samples past the
/// capture clamp to its edge, which must read "no shape".
const EDGE_PX: f64 = 2.0;
/// How close to a whole target pixel a capture edge must be to count as on it.
const SNAP_PX: f64 = 1e-3;
/// How far off the target's pixel grid a full-scale capture sits (see its use).
const CAPTURE_NUDGE_PX: f64 = 1.0 / 64.0;

/// Node types whose shadows the capture lane may draw.
fn is_capture_type(node_type: &str) -> bool {
    matches!(
        node_type,
        "rectangle"
            | "container"
            | "ellipse"
            | "star"
            | "polygon"
            | "path"
            | "line"
            | "richtext"
            | "group"
    )
}

/// Node types that contribute a silhouette to a subtree capture (a group's
/// or an unfilled frame's shadows, [`is_subtree_capture_node`]).
pub(crate) fn is_silhouette_type(node_type: &str) -> bool {
    is_capture_type(node_type) && node_type != "group"
}

/// Whether the node's shadows (and layer filters) come from its whole
/// subtree, as in Figma: a group, or a frame without a visible fill that has
/// children. Its capture is the composited silhouette of every visible
/// descendant (plus the frame's own stroke) instead of its own shape. A frame
/// WITH a fill keeps its rectangle shadow (the SDF lane).
pub fn is_subtree_capture_node(node: &RawNode) -> bool {
    let has_children = node.array("children").is_some_and(|c| !c.is_empty());
    match node.node_type() {
        "group" => true,
        "container" => {
            has_children
                && !moda_scene::paint::node_fills_shared(node)
                    .iter()
                    .any(|f| moda_scene::paint::shown(f.visible, f.opacity))
        }
        _ => false,
    }
}

/// The opacity `id` paints with inside `root`'s subtree: its own opacity
/// times every parent node's between it and `root` (neither `root`'s nor
/// above it — those fade the root's shadow as a whole). `None` when `root`
/// is not an ancestor of `id` (or `id` is `root`).
pub(crate) fn opacity_within(scene: &Scene, id: &str, root: &str) -> Option<f64> {
    let own = scene
        .effective_node(id)
        .and_then(|n| n.f64("opacity"))
        .unwrap_or(1.0);
    let parent_of = |c: &str| {
        scene
            .parent_map()
            .get(c)
            .or_else(|| scene.transient_parent_of().get(c))
            .cloned()
    };
    let mut product = own;
    let mut cursor = parent_of(id);
    // Depth-bounded: corrupt parent data must not hang the planner.
    for _ in 0..256 {
        let pid = cursor?;
        if pid == root {
            return Some(product);
        }
        if let Some(p) = scene.effective_node(&pid) {
            if moda_scene::scene::is_parent_node(p) {
                product *= p.f64("opacity").unwrap_or(1.0);
            }
        }
        cursor = parent_of(&pid);
    }
    None
}

/// Whether `root` is an ancestor of `id` (through groups, frames and
/// transient hosts).
pub(crate) fn is_within(scene: &Scene, id: &str, root: &str) -> bool {
    opacity_within(scene, id, root).is_some()
}

/// The drop shadows the capture lane draws for `node`, back to front: those of
/// a shape without an SDF silhouette. Rich text draws its own (text lane),
/// except text in a complex script, which the degraded raster tier draws
/// without effects: its shadows come from its silhouette here.
/// The spread is ignored, as Figma does on these shapes, so a shadow that
/// only spreads draws nothing here.
pub fn capture_drop_shadows(node: &RawNode) -> Vec<DropShadow> {
    let t = node.node_type();
    if !is_capture_type(t)
        || (t == "richtext" && !richtext_draws_degraded(node))
        || crate::effects::shadow::has_sdf_silhouette(node)
    {
        return Vec::new();
    }

    moda_scene::effect::drop_shadows(node)
        .into_iter()
        .filter(|s| s.blur > 0.0 || !s.offset.is_zero())
        .collect()
}

/// Whether rich text draws through the degraded raster tier, which paints its
/// glyphs' fill only: text with a letter of a complex script (Arabic, Hebrew,
/// Indic, Thai, …; `moda_text::complex_script`), as the vector lane declines it.
pub(crate) fn richtext_draws_degraded(node: &RawNode) -> bool {
    node.str("htmlContent")
        .is_some_and(moda_text::complex_script::text_needs_complex_shaping)
}

/// The inner shadows the capture lane draws for `node`, back to front: those
/// of a shape without an SDF silhouette, and rich text's. A shadow that only
/// spreads draws nothing here (spread is rectangles', ellipses' and frames').
pub fn capture_inner_shadows(node: &RawNode) -> Vec<InnerShadow> {
    if !is_capture_type(node.node_type()) || crate::effects::shadow::has_sdf_silhouette(node) {
        return Vec::new();
    }
    moda_scene::effect::inner_shadows(node)
        .into_iter()
        .filter(|s| s.blur > 0.0 || !s.offset.is_zero())
        .collect()
}

/// One glow of a shape or a text, as the capture lane draws it: Illustrator's
/// Outer Glow (`glow`) or Inner Glow (`inner-glow`).
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureGlow {
    pub color: moda_scene::value::Value,
    /// Halo radius (the Blur), node-local px: σ = radius / 2.
    pub radius: f64,
    /// `opacity` (0-1) × `intensity` (≥ 0, may exceed 1): the halo's weight.
    pub weight: f64,
    /// `mode`: how the halo composites (absent = screen, as in Illustrator).
    pub blend: CaptureBlend,
    /// Inner glows only: `source: 'center'` (glow from the middle outward)
    /// rather than the default `edge` (from the edges inward).
    pub center: bool,
}

/// Every visible entry of `effect_type` (`glow` / `inner-glow`) of a capture
/// shape or text that has a radius and a weight, back to front.
fn glows_of(node: &RawNode, effect_type: &str) -> Vec<CaptureGlow> {
    if !is_capture_type(node.node_type()) {
        return Vec::new();
    }
    moda_scene::effect::visible_effects_json(node, effect_type)
        .filter_map(|o| {
            let f = |k: &str| o.get(k).and_then(moda_scene::value::Value::as_f64);
            let radius = f("radiusPx").filter(|r| r.is_finite() && *r > 0.0)?;
            let weight = f("opacity").unwrap_or(1.0).clamp(0.0, 1.0)
                * f("intensity").unwrap_or(1.0).max(0.0);
            (weight.is_finite() && weight > 0.0).then(|| CaptureGlow {
                color: o
                    .get("color")
                    .filter(|c| c.is_string())
                    .cloned()
                    .unwrap_or_else(|| moda_scene::value::Value::from("#000000")),
                radius,
                weight,
                blend: CaptureBlend::from_glow_mode(
                    o.get("mode").and_then(moda_scene::value::Value::as_str),
                ),
                center: o.get("source").and_then(moda_scene::value::Value::as_str) == Some("center"),
            })
        })
        .collect()
}

/// The outer glows the capture lane draws for `node` (every shape, rich text
/// included), back to front.
pub fn capture_glows(node: &RawNode) -> Vec<CaptureGlow> {
    glows_of(node, "glow")
}

/// The inner glows the capture lane draws for `node`, back to front.
pub fn capture_inner_glows(node: &RawNode) -> Vec<CaptureGlow> {
    glows_of(node, "inner-glow")
}

/// Whether the node draws any capture shadow or glow.
pub fn has_capture_shadow(node: &RawNode) -> bool {
    !capture_drop_shadows(node).is_empty()
        || !capture_inner_shadows(node).is_empty()
        || !capture_glows(node).is_empty()
        || !capture_inner_glows(node).is_empty()
}

/// Which of a node's capture effects one group draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    /// Drop shadows: at the body's z, before it.
    Drop,
    /// The outer glows of one mode: the silhouette blurred in the glow
    /// colour, not knocked out, composited with the mode over what lies below;
    /// at the body's z, before it and after the drop shadows.
    Glow(CaptureBlend),
    /// Inner shadows: at z + `INNER_SHADOW_Z_OFFSET`, over the body.
    Inner,
    /// The inner glows of one mode: the silhouette's outside (edge) or the
    /// silhouette itself (center) blurred, clipped to the silhouette and
    /// composited with the mode; over the body, after the inner shadows.
    InnerGlow(CaptureBlend),
}

impl CaptureKind {
    /// The groups `node` plans, in plan order: drop shadows, the outer glows
    /// (one group per mode, in the order the modes first appear), inner
    /// shadows, the inner glows (likewise). A kind with nothing to draw plans
    /// no group ([`plan_shadow_group`] declines it).
    pub fn for_node(node: &RawNode) -> Vec<CaptureKind> {
        let modes = |glows: Vec<CaptureGlow>| {
            let mut out: Vec<CaptureBlend> = Vec::new();
            for g in glows {
                if !out.contains(&g.blend) {
                    out.push(g.blend);
                }
            }
            out
        };
        let mut kinds = vec![CaptureKind::Drop];
        kinds.extend(
            modes(capture_glows(node))
                .into_iter()
                .map(CaptureKind::Glow),
        );
        kinds.push(CaptureKind::Inner);
        kinds.extend(
            modes(capture_inner_glows(node))
                .into_iter()
                .map(CaptureKind::InnerGlow),
        );
        kinds
    }

    /// Drawn over the body, clipped to the silhouette.
    pub fn is_inner(self) -> bool {
        matches!(self, CaptureKind::Inner | CaptureKind::InnerGlow(_))
    }

    /// The composite's node-id suffix: `glow` / `innerglow` for the default
    /// (screen) mode, `glow-<mode>` / `innerglow-<mode>` for the others.
    pub fn suffix(self) -> String {
        let glow = |base: &str, blend: CaptureBlend| {
            if blend == CaptureBlend::Screen {
                base.to_string()
            } else {
                format!("{base}-{}", blend.glow_mode())
            }
        };
        match self {
            CaptureKind::Drop => "dropshadow".into(),
            CaptureKind::Glow(b) => glow("glow", b),
            CaptureKind::Inner => "innershadow".into(),
            CaptureKind::InnerGlow(b) => glow("innerglow", b),
        }
    }
}

/// Whether the node's visible fills are all images: its pixels' alpha is its
/// shape (a cut-out PNG casts the cut-out's shadow and glow, not its box's),
/// as Figma shadows a layer by what it draws. No visible fill: false.
pub fn fills_are_images(node: &RawNode) -> bool {
    use moda_scene::paint as sp;
    let fills = sp::node_fills_shared(node);
    let mut shown = fills.iter().filter(|f| sp::shown(f.visible, f.opacity)).peekable();
    shown.peek().is_some() && shown.all(|f| matches!(f.paint, sp::Paint::Image(_)))
}

/// The node's silhouette: the node with its visible fills folded to one opaque
/// black fill (image fills keep their pixels, at full opacity, for their
/// alpha: the shadow passes read only the capture's alpha), its widest visible
/// stroke painted opaque black, opacity 1, normal blend, and no effects but its
/// live geometry ones (which reshape the outline). Rich text also paints its
/// glyphs black (`textFills`).
pub fn silhouette_node(node: &RawNode) -> RawNode {
    use moda_scene::paint as sp;
    let black = || moda_scene::value::Value::from("#000000");
    let mut m = node.fields().clone();
    m.insert("opacity".into(), moda_scene::value::Value::from(1.0));
    m.remove("blendMode");
    let geometry: Vec<moda_scene::value::Value> = node
        .array("effects")
        .map(|effects| {
            effects
                .iter()
                .filter(|e| {
                    e.get("type")
                        .and_then(moda_scene::value::Value::as_str)
                        .is_some_and(|t| moda_scene::effect::GEOMETRY_EFFECT_TYPES.contains(&t))
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    if geometry.is_empty() {
        m.remove("effects");
    } else {
        m.insert("effects".into(), moda_scene::value::Value::Array(geometry));
    }
    let has_fill = sp::node_fills_shared(node)
        .iter()
        .any(|f| sp::shown(f.visible, f.opacity));
    sp::set_fills(
        &mut m,
        if fills_are_images(node) {
            sp::node_fills(node)
                .into_iter()
                .filter(|f| sp::shown(f.visible, f.opacity))
                .map(|mut f| {
                    f.opacity = None;
                    f.blend_mode = None;
                    f.effects = None;
                    f
                })
                .collect()
        } else if has_fill {
            vec![sp::Fill::solid("fill-0", black())]
        } else {
            Vec::new()
        },
    );
    let stroke = crate::paint::layers::widest_stroke(node).map(|mut s| {
        s.paint = sp::Paint::Solid(sp::SolidPaint { color: black() });
        s.opacity = None;
        s.visible = None;
        s.blend_mode = None;
        s
    });
    sp::set_strokes(&mut m, stroke.into_iter().collect());
    if node.node_type() == "richtext" {
        m.insert(
            "textFills".into(),
            moda_scene::value::json!([{"id": "text-fill-0", "paint": {"type": "solid", "color": "#000000"}}]),
        );
    }
    node.with_fields(m)
}

/// The shape a noise effect covers (`seams::LayerBlurGroup::coverage`): the
/// node's silhouette with its fill area always filled, at the node's opacity,
/// or `None` when the node's own paint already covers its shape (a visible,
/// opaque solid fill) or it has no fill area (a line, text). Figma grains a
/// shape with a transparent or missing fill too.
pub fn noise_coverage_node(node: &RawNode) -> Option<RawNode> {
    use moda_scene::paint as sp;
    if matches!(node.node_type(), "line" | "richtext") {
        return None;
    }
    let covered = sp::node_fills_shared(node).iter().any(|f| {
        f.visible != Some(false)
            && f.opacity.unwrap_or(1.0) >= 1.0
            && f.blend_mode.as_deref().is_none_or(|b| b == "normal")
            && matches!(&f.paint, sp::Paint::Solid(s)
                if crate::color::parse_color(s.color.as_str()).is_some_and(|c| c.a >= 1.0))
    });
    if covered {
        return None;
    }
    let sil = silhouette_node(node);
    let mut m = sil.fields().clone();
    m.insert(
        "opacity".into(),
        moda_scene::value::Value::from(node.f64("opacity").unwrap_or(1.0)),
    );
    sp::set_fills(
        &mut m,
        vec![sp::Fill::solid("fill-0", moda_scene::value::Value::from("#000000"))],
    );
    Some(sil.with_fields(m))
}

/// Local → canvas scale of the node's blur: the shorter axis of its canvas
/// affine (the SDF lane blurs in node-local units, so a scaled node's shadow
/// blurs with it; the offset is canvas-space).
fn node_blur_scale(scene: &Scene, node: &RawNode) -> f64 {
    scene
        .node_canvas_affine(moda_scene::scene::paint_transform_id(node))
        .map(|a| a.a.hypot(a.b).min(a.c.hypot(a.d)))
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(1.0)
}

/// The canvas rect the target shows (the projection is axis-aligned).
fn target_canvas_rect(ctx: &PlanCtx) -> Option<Rect> {
    let (x0, y0) = ctx.project_point(0.0, 0.0)?;
    let (x1, y1) = ctx.project_point(1.0, 1.0)?;
    let (sx, sy) = (x1 - x0, y1 - y0);
    if !(sx.abs() > 1e-12 && sy.abs() > 1e-12) {
        return None;
    }
    let ax = (0.0 - x0) / sx;
    let bx = (ctx.width_px as f64 - x0) / sx;
    let ay = (0.0 - y0) / sy;
    let by = (ctx.height_px as f64 - y0) / sy;
    let r = Rect {
        x: ax.min(bx),
        y: ay.min(by),
        w: (bx - ax).abs(),
        h: (by - ay).abs(),
    };
    r.finite().then_some(r)
}

/// One shadow in canvas units, before the capture scale.
struct CanvasShadow {
    color: [f32; 4],
    offset: (f64, f64),
    /// 3σ in canvas units.
    reach: f64,
    show_behind: bool,
    /// Inner glow from the centre (`ShadowPass::center`).
    center: bool,
}

/// The capture group of one kind of the node's shadows, or `None` when it has
/// none, or none of it can reach the target. `items` are the silhouette's
/// draw items (canvas-absolute, unclipped, unmasked); `bounds` their canvas
/// AABB; `node_opacity` the node's effective opacity.
pub(crate) fn plan_shadow_group(
    scene: &Scene,
    node: &RawNode,
    ctx: &PlanCtx,
    kind: CaptureKind,
    items: Vec<crate::seams::DrawItem>,
    bounds: Rect,
    node_opacity: f64,
) -> Option<LayerBlurGroup> {
    let blur_scale = node_blur_scale(scene, node);
    let shadow = |color: &moda_scene::value::Value, opacity: f64, blur: f64, ox: f64, oy: f64, behind| {
        let rgba = crate::effects::shadow::shadow_color_rgba(color);
        CanvasShadow {
            color: [
                rgba[0] as f32,
                rgba[1] as f32,
                rgba[2] as f32,
                (rgba[3] * opacity * node_opacity).clamp(0.0, 1.0) as f32,
            ],
            offset: (ox, oy),
            reach: 1.5 * blur.max(0.0) * blur_scale,
            show_behind: behind,
            center: false,
        }
    };
    let inner = kind.is_inner();
    // A glow is a shadow with no offset and blur = radius (σ = radius / 2),
    // whose weight may exceed 1 (intensity): the halo is min(1, weight ×
    // blurred source), so a stronger glow stays bright further out. The
    // colour's own alpha scales it.
    let glow = |g: &CaptureGlow| {
        let mut s = shadow(&g.color, 1.0, g.radius, 0.0, 0.0, true);
        let a = crate::effects::shadow::shadow_color_rgba(&g.color)[3].clamp(0.0, 1.0);
        s.color[3] = (g.weight * a * node_opacity.clamp(0.0, 1.0)) as f32;
        s.center = g.center;
        s
    };
    let shadows: Vec<CanvasShadow> = match kind {
        CaptureKind::Inner => capture_inner_shadows(node)
            .iter()
            .map(|s| shadow(&s.color, s.opacity(), s.blur, s.offset.x, s.offset.y, false))
            .collect(),
        // An outer glow shows behind the node (not knocked out).
        CaptureKind::Glow(blend) => capture_glows(node)
            .iter()
            .filter(|g| g.blend == blend)
            .map(glow)
            .collect(),
        // Edge: the outside blurred into the shape (an offset-free inner
        // shadow); center: the shape itself blurred, clipped to it.
        CaptureKind::InnerGlow(blend) => capture_inner_glows(node)
            .iter()
            .filter(|g| g.blend == blend)
            .map(glow)
            .collect(),
        CaptureKind::Drop => capture_drop_shadows(node)
            .iter()
            .map(|s| {
                shadow(
                    &s.color,
                    s.opacity(),
                    s.blur,
                    s.offset.x,
                    s.offset.y,
                    s.shows_behind_node(),
                )
            })
            .collect(),
    };
    let shadows: Vec<CanvasShadow> = shadows.into_iter().filter(|s| s.color[3] > 0.0).collect();
    if shadows.is_empty() || !bounds.finite() || !(bounds.w >= 0.0 && bounds.h >= 0.0) {
        return None;
    }
    // How far a shadow pixel reads from its source (offset + 3σ).
    let reach = shadows
        .iter()
        .map(|s| s.offset.0.abs().max(s.offset.1.abs()) + s.reach)
        .fold(0.0, f64::max);
    if !reach.is_finite() {
        return None;
    }
    let canvas_scale = ctx.canvas_target_scale().max(1e-4);
    // Where the shadows paint: a drop shadow around the shape by its reach, an
    // inner shadow inside the shape. One target px of antialiasing either way.
    let aa = 1.0 / canvas_scale;
    let paint_pad = if inner { aa } else { reach + aa };
    let mut region = Rect {
        x: bounds.x - paint_pad,
        y: bounds.y - paint_pad,
        w: bounds.w + 2.0 * paint_pad,
        h: bounds.h + 2.0 * paint_pad,
    };
    // Only what the target shows matters, and a shown pixel reads at most
    // `reach` away from itself.
    if let Some(view) = target_canvas_rect(ctx) {
        let pad = reach + aa;
        let reachable = Rect {
            x: view.x - pad,
            y: view.y - pad,
            w: view.w + 2.0 * pad,
            h: view.h + 2.0 * pad,
        };
        region = region.intersect(&reachable)?;
    }
    // Full target resolution, reduced only to fit the texture limits.
    let mut zoom = canvas_scale;
    let edge = |zoom: f64| EDGE_PX / zoom;
    let dims = |zoom: f64| {
        (
            (region.w + 2.0 * edge(zoom)) * zoom,
            (region.h + 2.0 * edge(zoom)) * zoom,
        )
    };
    let (w, h) = dims(zoom);
    if w.max(h) > MAX_TEXTURE_DIM {
        zoom *= MAX_TEXTURE_DIM / w.max(h);
    }
    let (w, h) = dims(zoom);
    if w * h > MAX_CAPTURE_AREA_PX {
        zoom *= (MAX_CAPTURE_AREA_PX / (w * h)).sqrt();
    }
    let e = edge(zoom);
    region = Rect {
        x: region.x - e,
        y: region.y - e,
        w: region.w + 2.0 * e,
        h: region.h + 2.0 * e,
    };
    // A full-scale capture lands its texels on target pixels (the composite
    // then reads one texel per pixel).
    if zoom == canvas_scale {
        if let Some((px, py)) = ctx.project_point(region.x, region.y) {
            let off = |v: f64| {
                if (v - v.round()).abs() < SNAP_PX {
                    v - v.round()
                } else {
                    v - v.floor()
                }
            };
            // Less a 1/64 px nudge: texel centres sit a hair off the pixel
            // centres (the composite's bilinear read moves 1.6% of a
            // neighbour in), so a centre that would land EXACTLY on a glyph's
            // horizontal extremum doesn't — there the vector text lane's
            // winding ray grazes the curve and leaks a hairline of coverage,
            // which the inner shadow would paint.
            let nudge = CAPTURE_NUDGE_PX / canvas_scale;
            let (dx, dy) = (
                off(px) / canvas_scale + nudge,
                off(py) / canvas_scale + nudge,
            );
            region = Rect {
                x: region.x - dx,
                y: region.y - dy,
                w: region.w + dx,
                h: region.h + dy,
            };
        }
    }
    let whole = |v: f64| {
        if (v - v.round()).abs() < SNAP_PX {
            v.round()
        } else {
            v.ceil()
        }
    };
    // A wide blur runs at 1/2^n resolution (`exec::shadow_capture`): whole
    // 2^n-texel blocks keep each halving an exact 2×2 average, so pad the
    // capture (right and bottom, with more empty border) to a multiple.
    let block = {
        let mut r = shadows.iter().map(|s| s.reach * zoom).fold(0.0, f64::max);
        let mut b = 1.0;
        while r > crate::layer_blur::exec::PROGRESSIVE_BLUR_MAX_TAPS as f64 {
            r /= 2.0;
            b *= 2.0;
        }
        b
    };
    let whole_blocks = |v: f64| {
        let v = whole(v);
        let padded = (v / block).ceil() * block;
        if padded <= MAX_TEXTURE_DIM {
            padded
        } else {
            v
        }
    };
    let width_px = whole_blocks(region.w * zoom).clamp(1.0, MAX_TEXTURE_DIM) as u32;
    let height_px = whole_blocks(region.h * zoom).clamp(1.0, MAX_TEXTURE_DIM) as u32;
    region.w = width_px as f64 / zoom;
    region.h = height_px as f64 / zoom;
    if width_px < 2 || height_px < 2 {
        return None;
    }
    let passes = shadows
        .iter()
        .map(|s| ShadowPass {
            color: s.color,
            offset_px: [(s.offset.0 * zoom) as f32, (s.offset.1 * zoom) as f32],
            radius_px: (s.reach * zoom) as f32,
            show_behind: s.show_behind,
            center: s.center,
        })
        .collect();
    Some(LayerBlurGroup {
        items,
        region_canvas: [region.x, region.y, region.w, region.h],
        extent_canvas: [region.x, region.y, region.w, region.h],
        width_px,
        height_px,
        // The offset and blur, and the halved blur's blocks (each read whole).
        reach_canvas: reach + aa + block / zoom,
        filters: LayerFilters {
            shadow: Some(ShadowFilter {
                inner,
                blend: match kind {
                    CaptureKind::Glow(b) | CaptureKind::InnerGlow(b) => b,
                    _ => CaptureBlend::Normal,
                },
                shadows: passes,
            }),
            ..Default::default()
        },
        cache_key: None,
        identity_key: None,
        coverage: Vec::new(),
        masks: Vec::new(),
    })
}

/// The canvas AABB of drawn items (tessellated bodies' geometry bounds, text
/// glyph and op vertices), through each item's model matrix. `None` when
/// nothing measurable draws.
pub(crate) fn items_canvas_bounds(items: &[crate::seams::DrawItem]) -> Option<Rect> {
    use crate::seams::DrawItem;
    let mut min = (f64::INFINITY, f64::INFINITY);
    let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut add = |m: &[f32; 16], x: f32, y: f32| {
        let (x, y) = (x as f64, y as f64);
        let cx = m[0] as f64 * x + m[4] as f64 * y + m[12] as f64;
        let cy = m[1] as f64 * x + m[5] as f64 * y + m[13] as f64;
        if cx.is_finite() && cy.is_finite() {
            min = (min.0.min(cx), min.1.min(cy));
            max = (max.0.max(cx), max.1.max(cy));
        }
    };
    for item in items {
        match item {
            DrawItem::Body { instance, .. } => {
                let m = &instance.model_matrix;
                let [x, y, w, h] = instance.bounds;
                for (px, py) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)] {
                    add(m, px, py);
                }
            }
            DrawItem::Text(batch) => {
                let m = &batch.model_matrix;
                for op in &batch.ops {
                    for v in op.verts.chunks_exact(4) {
                        add(m, v[0], v[1]);
                    }
                }
                // The degraded tier (complex scripts): one raster quad, placed as the executor draws it.
                if let Some(dq) = &batch.degraded {
                    let [x0, y0] = dq.offset;
                    let (x1, y1) = (
                        x0 + dq.raster.logical_w as f32,
                        y0 + dq.raster.logical_h as f32,
                    );
                    for (px, py) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                        add(m, px, py);
                    }
                }
                // The vector lane: tessellated triangles, and analytic glyph
                // boxes placed as `kineticGlyphLocal` places them.
                for op in batch.vector.iter().flat_map(|v| &v.ops) {
                    for v in op.tris.chunks_exact(2) {
                        add(m, v[0], v[1]);
                    }
                    for r in &op.records {
                        let [x0, y0, x1, y1] = r.em_box;
                        let s = r.font_size_px;
                        let center = (
                            r.pen_x + (x0 + x1) * 0.5 * s,
                            r.baseline_y + (y0 + y1) * 0.5 * s,
                        );
                        let [a, c, b, d] = r.glyph_lin;
                        for (ex, ey) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                            let rel = (
                                r.pen_x + ex * s - center.0,
                                r.baseline_y + ey * s - center.1,
                            );
                            add(
                                m,
                                center.0 + a * rel.0 + c * rel.1 + r.glyph_off[0],
                                center.1 + b * rel.0 + d * rel.1 + r.glyph_off[1],
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
    (min.0 <= max.0 && min.1 <= max.1).then(|| Rect {
        x: min.0,
        y: min.1,
        w: max.0 - min.0,
        h: max.1 - min.1,
    })
}

/// The canvas AABB every one of `items` paints (a subtree layer capture's
/// content): tessellated bodies and text as [`items_canvas_bounds`], SDF
/// shapes by their box grown by the stroke, SDF shadows by their silhouette
/// grown by the blur and moved by the offset (a conservative reach), nested
/// captures by their region. `None` when nothing measurable draws.
pub(crate) fn all_items_canvas_bounds(items: &[crate::seams::DrawItem]) -> Option<Rect> {
    use crate::seams::DrawItem;
    let mut acc = items_canvas_bounds(items);
    let mut add = |r: Rect| {
        if !r.finite() {
            return;
        }
        acc = Some(match acc {
            None => r,
            Some(a) => {
                let (x0, y0) = (a.x.min(r.x), a.y.min(r.y));
                let (x1, y1) = ((a.x + a.w).max(r.x + r.w), (a.y + a.h).max(r.y + r.h));
                Rect {
                    x: x0,
                    y: y0,
                    w: x1 - x0,
                    h: y1 - y0,
                }
            }
        });
    };
    // The canvas AABB of a local box through a column-major model matrix.
    let boxed = |m: &[f32; 16], x0: f64, y0: f64, x1: f64, y1: f64| {
        let mut min = (f64::INFINITY, f64::INFINITY);
        let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
            let cx = m[0] as f64 * x + m[4] as f64 * y + m[12] as f64;
            let cy = m[1] as f64 * x + m[5] as f64 * y + m[13] as f64;
            min = (min.0.min(cx), min.1.min(cy));
            max = (max.0.max(cx), max.1.max(cy));
        }
        Rect {
            x: min.0,
            y: min.1,
            w: max.0 - min.0,
            h: max.1 - min.1,
        }
    };
    for item in items {
        match item {
            DrawItem::SolidShapeSdf { instance, .. } => {
                let [w, h, ..] = instance.dimensions;
                let sw = instance.stroke_params[0].max(0.0) as f64;
                add(boxed(
                    &instance.model_matrix,
                    -sw,
                    -sw,
                    w as f64 + sw,
                    h as f64 + sw,
                ));
            }
            DrawItem::GradientShapeSdf { instance, .. } => {
                let [w, h, ..] = instance.dimensions;
                let sw = instance.stroke_params[0].max(0.0) as f64;
                add(boxed(
                    &instance.model_matrix,
                    -sw,
                    -sw,
                    w as f64 + sw,
                    h as f64 + sw,
                ));
            }
            DrawItem::Shadow {
                instance, inner, ..
            } => {
                // A rect's quad spans [0, w], an ellipse's is centred: cover both.
                let [w, h, blur, _] = instance.shape_data;
                let (w, h) = (w as f64, h as f64);
                let pad = if *inner {
                    0.0
                } else {
                    2.0 * blur.max(0.0) as f64
                };
                let r = boxed(
                    &instance.model_matrix,
                    -0.5 * w - pad,
                    -0.5 * h - pad,
                    w + pad,
                    h + pad,
                );
                let (ox, oy) = if *inner {
                    (0.0, 0.0)
                } else {
                    (
                        instance.shadow_offset[0] as f64,
                        instance.shadow_offset[1] as f64,
                    )
                };
                add(Rect {
                    x: r.x + ox,
                    y: r.y + oy,
                    ..r
                });
            }
            DrawItem::LayerBlurComposite { group, .. } => {
                // The picture's extent, not the stored region: a capture cut to what can be read still
                // occupies its whole bounds for whatever encloses it.
                let [x, y, w, h] = group.extent_canvas;
                add(Rect { x, y, w, h });
            }
            _ => {}
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(v: moda_scene::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    #[test]
    fn capture_lane_takes_every_shape_without_an_sdf_silhouette() {
        let fx = moda_scene::value::json!([
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 8},
            {"id": "i", "type": "inner-shadow", "color": "#000", "blur": 8}
        ]);
        for t in ["star", "polygon", "path", "line"] {
            let n = raw(moda_scene::value::json!({"id": "a", "type": t, "effects": fx}));
            assert_eq!(capture_drop_shadows(&n).len(), 1, "{t} drop");
            assert_eq!(capture_inner_shadows(&n).len(), 1, "{t} inner");
        }
        // The SDF lane keeps rectangles and full ellipses.
        let rect = raw(
            moda_scene::value::json!({"id": "a", "type": "rectangle", "width": 10, "height": 10, "effects": fx}),
        );
        assert!(!has_capture_shadow(&rect));
        // A wedge has no silhouette: the capture lane draws it.
        let wedge = raw(
            moda_scene::value::json!({"id": "a", "type": "ellipse", "width": 10, "height": 10,
            "sweepAngle": 180, "effects": fx}),
        );
        assert!(has_capture_shadow(&wedge));
        // Rich text: inner shadows only (its drop shadows are the text lane's).
        let text = raw(moda_scene::value::json!({"id": "a", "type": "richtext", "effects": fx}));
        assert!(capture_drop_shadows(&text).is_empty());
        assert_eq!(capture_inner_shadows(&text).len(), 1);
        // Groups cast their subtree's shadows; so does a frame without a fill
        // that has children. A filled frame keeps the SDF lane.
        let group =
            raw(moda_scene::value::json!({"id": "a", "type": "group", "children": [], "effects": fx}));
        assert!(has_capture_shadow(&group) && is_subtree_capture_node(&group));
        let frame = |fills: moda_scene::value::Value| {
            raw(
                moda_scene::value::json!({"id": "a", "type": "container", "width": 10, "height": 10,
                "children": ["c"], "fills": fills, "effects": fx}),
            )
        };
        assert!(is_subtree_capture_node(&frame(moda_scene::value::json!([]))));
        assert!(has_capture_shadow(&frame(moda_scene::value::json!([]))));
        let filled =
            frame(moda_scene::value::json!([{"id": "f", "paint": {"type": "solid", "color": "#fff"}}]));
        assert!(!is_subtree_capture_node(&filled) && !has_capture_shadow(&filled));
    }

    #[test]
    fn an_image_filled_shape_shadows_and_glows_from_its_pixels() {
        // A cut-out PNG (`<img>`, a rectangle with an image fill) casts the
        // cut-out's shadow and glow, not its box's: the capture lane draws it,
        // and its silhouette keeps the image (its alpha) at full opacity.
        let image = moda_scene::value::json!([{"id": "f", "opacity": 0.5, "blendMode": "multiply",
            "paint": {"type": "image", "src": "cutout.png"}}]);
        let fx = moda_scene::value::json!([
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 8},
            {"id": "g", "type": "glow", "color": "#ffd36b", "radiusPx": 20}
        ]);
        for t in ["rectangle", "ellipse"] {
            let n = raw(moda_scene::value::json!({"id": "a", "type": t, "width": 10, "height": 10,
                "fills": image, "effects": fx}));
            assert!(fills_are_images(&n), "{t}");
            assert!(!crate::effects::shadow::has_sdf_silhouette(&n), "{t}");
            assert_eq!(capture_drop_shadows(&n).len(), 1, "{t} drop");
            assert_eq!(capture_glows(&n).len(), 1, "{t} glow");
            let fills = moda_scene::paint::node_fills(&silhouette_node(&n));
            assert_eq!(fills.len(), 1);
            assert!(matches!(fills[0].paint, moda_scene::paint::Paint::Image(_)));
            assert_eq!(fills[0].opacity, None);
            assert_eq!(fills[0].blend_mode, None);
        }
        // A solid fill under the image fills the box: the box is the shape.
        let mixed = raw(moda_scene::value::json!({"id": "a", "type": "rectangle", "width": 10, "height": 10,
            "fills": [{"id": "s", "paint": {"type": "solid", "color": "#fff"}},
                      {"id": "f", "paint": {"type": "image", "src": "cutout.png"}}],
            "effects": fx}));
        assert!(!fills_are_images(&mixed));
        assert!(crate::effects::shadow::has_sdf_silhouette(&mixed));
    }

    #[test]
    fn spread_only_drop_shadow_draws_nothing_on_a_path() {
        let n = raw(moda_scene::value::json!({"id": "a", "type": "path", "effects": [
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 0, "spread": 4}]}));
        assert!(capture_drop_shadows(&n).is_empty());
        // With a blur it draws, spread ignored (the capture has no spread input).
        let n = raw(moda_scene::value::json!({"id": "a", "type": "path", "effects": [
            {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 6, "spread": -4}]}));
        assert_eq!(capture_drop_shadows(&n).len(), 1);
    }

    #[test]
    fn spread_only_inner_shadow_draws_nothing_on_a_path() {
        let n = raw(moda_scene::value::json!({"id": "a", "type": "path", "effects": [
            {"id": "i", "type": "inner-shadow", "color": "#000", "blur": 0, "spread": 4}]}));
        assert!(capture_inner_shadows(&n).is_empty());
    }

    #[test]
    fn silhouette_is_opaque_black_fill_and_widest_stroke() {
        let n = raw(moda_scene::value::json!({
            "id": "a", "type": "star", "opacity": 0.4, "blendMode": "multiply",
            "fills": [
                {"id": "f0", "paint": {"type": "solid", "color": "#ff000080"}, "opacity": 0.5},
                {"id": "f1", "paint": {"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 0},
                    "stops": [{"offset": 0, "color": "#00000000"}, {"offset": 1, "color": "#fff"}]}}
            ],
            "strokes": [
                {"id": "s0", "paint": {"type": "solid", "color": "#0f0"}, "width": 2},
                {"id": "s1", "paint": {"type": "solid", "color": "#00f"}, "width": 6, "align": "outside", "opacity": 0.3}
            ],
            "effects": [
                {"id": "d", "type": "drop-shadow", "color": "#000", "blur": 8},
                {"id": "r", "type": "roughen", "size": 3, "detail": 5}
            ]
        }));
        let sil = silhouette_node(&n);
        assert_eq!(sil.f64("opacity"), Some(1.0));
        assert!(sil.json_field("blendMode").is_none());
        let fills = moda_scene::paint::node_fills(&sil);
        assert_eq!(fills.len(), 1);
        assert_eq!(
            moda_scene::paint::solid_color(&fills[0].paint),
            Some("#000000")
        );
        let strokes = moda_scene::paint::node_strokes(&sil);
        assert_eq!(strokes.len(), 1);
        assert_eq!(strokes[0].width, 6.0);
        assert_eq!(strokes[0].align.as_deref(), Some("outside"));
        assert_eq!(strokes[0].opacity, None);
        // Only the geometry effect survives.
        let effects = sil.array("effects").unwrap();
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0]["type"], "roughen");
    }
}
