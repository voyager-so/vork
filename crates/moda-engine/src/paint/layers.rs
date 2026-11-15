//! Paint stacks (E3.1 P6): how a node's `fills` / `strokes` turn into draws.
//!
//! Every visible layer draws, in Illustrator's Appearance order: the fills bottom to top, then the strokes
//! bottom to top (a stroke is always above every fill: the two lists never interleave). Each layer draws with
//! its own opacity and blend mode. The rest of the engine reads a node's paint through its TOP visible fill and
//! stroke (`moda_scene::paint::top_fill` / `top_stroke`), so a layer draws as a CLONE of the node that carries
//! just that layer: every paint kind (solid, the gradients, image, video, shader) and every stroke kind (dash,
//! inside/outside alignment, width profiles, brushes, arrowheads) then draws through its existing path.
//!
//! Before splitting, the stack is folded where one draw gives the same pixels as several:
//! - fills under an opaque, normal-blend, full-opacity fill are covered (every fill has the node's geometry),
//!   so they are dropped;
//! - adjacent solid, normal-blend fills merge into one solid colour (source-over of two constants is a
//!   constant).
//!
//! What is left is either one fill and one stroke, each at full opacity with normal blend, which draws as
//! today's single body ([`LayerPlan::Single`]) unless the node is translucent or blends (not normal) and the
//! stroke lies over the fill (see [`plan_at`]), or a stack of layer clones ([`LayerPlan::Stack`]), which
//! `frame.rs` draws at fractional z inside the node's slot, with blend splits for blend modes and for the node's own
//! opacity / blend over the stack (a transparency group, as Illustrator composites an object's appearance).

use moda_scene::appearance as ap;
use moda_scene::paint::{self as sp, Fill, Paint, SolidPaint, Stroke};
use moda_scene::types::RawNode;
use moda_scene::value::{Map, Value};

use crate::color::{color_to_normalized_rgba, parse_color, Rgba};
use crate::seams::BlendMode;

/// Node types whose fills / strokes the body lanes draw (`line` has strokes only).
const LAYERED_TYPES: &[&str] = &[
    "rectangle",
    "container",
    "ellipse",
    "path",
    "star",
    "polygon",
    "line",
];

/// One layer of a stack: a clone of the node carrying only this layer.
#[derive(Debug, Clone)]
pub struct PaintLayer {
    /// The node with `fills` = [this fill] (no strokes) or `strokes` = [this stroke] (no fills), a synthetic id
    /// (`<id>::paint::<k>`, so the per-node tessellation caches key each layer apart) and `_transformSourceId`
    /// pointing at the node. Shadows are stripped (the node draws them once). `opacity` / `blendMode` are the
    /// node's: the caller sets them per its grouping.
    pub node: RawNode,
    /// The layer's own opacity (0..1).
    pub opacity: f64,
    /// The layer's own blend mode.
    pub blend: BlendMode,
    pub is_stroke: bool,
    /// Effects scoped to this layer (Illustrator's Appearance, E8.2): its visible entry effects in list order
    /// (`moda_scene::appearance::entry_effects`). Empty for every stack outside the appearance lane.
    pub effects: Vec<Value>,
}

#[derive(Debug, Clone)]
pub enum LayerPlan {
    /// One body. `None`: the node as stored (what it drew before P6: its top fill and top stroke, which the
    /// fold left alone). `Some`: the node with its stack folded to at most one fill and one stroke.
    Single(Option<RawNode>),
    /// Two or more layers, or layers with their own opacity / blend mode, bottom to top.
    Stack(Vec<PaintLayer>),
}

use moda_scene::paint::shown as shows;

fn layer_opacity(opacity: Option<f64>) -> f64 {
    opacity.unwrap_or(1.0).clamp(0.0, 1.0)
}

fn layer_blend(mode: Option<&str>) -> BlendMode {
    BlendMode::from_node_value(mode)
}

/// A solid paint's colour (a resolved CSS string the path renderer parses).
fn solid_rgba(paint: &Paint) -> Option<Rgba> {
    match paint {
        Paint::Solid(s) => parse_color(s.color.as_str()),
        _ => None,
    }
}

/// Whether a fill paints every pixel of its geometry opaquely: an opaque solid, or a gradient whose stops are
/// all opaque (a gradient pads its end stops, so it has no gaps).
fn paint_is_opaque(paint: &Paint) -> bool {
    match paint {
        Paint::Solid(_) => solid_rgba(paint).is_some_and(|c| c.a >= 1.0),
        Paint::Linear(_) | Paint::Radial(_) | Paint::Conic(_) => sp::paint_stops(paint)
            .is_some_and(|stops| {
                !stops.is_empty()
                    && stops.iter().all(|s| {
                        s.color
                            .as_str()
                            .and_then(color_to_normalized_rgba)
                            .is_some_and(|c| c.a >= 1.0)
                    })
            }),
        Paint::Freeform(f) => {
            !f.points.is_empty()
                && f.points.iter().all(|p| {
                    p.color
                        .as_str()
                        .and_then(color_to_normalized_rgba)
                        .is_some_and(|c| c.a >= 1.0)
                })
        }
        // Diamond has no renderer of its own yet; images, videos and shaders may be transparent.
        _ => false,
    }
}

fn is_plain(opacity: Option<f64>, blend: Option<&str>) -> bool {
    layer_opacity(opacity) >= 1.0 && layer_blend(blend) == BlendMode::Normal
}

/// A fill that draws nothing: a solid whose colour is fully transparent, with normal blending (source-over of
/// alpha 0 is the backdrop, exactly). A layer opacity of 0 is already hidden (`shown`).
fn paints_nothing(fill: &Fill) -> bool {
    layer_blend(fill.blend_mode.as_deref()) == BlendMode::Normal
        && solid_rgba(&fill.paint).is_some_and(|c| c.a <= 0.0)
}

/// A stroke that tessellates as one plain band: no brush, width profile, taper, wave, dynamics, dash or
/// arrowhead (each of those draws pieces that may overlap one another).
fn plain_stroke_body(s: &Stroke) -> bool {
    let arrow = |a: &Option<String>| a.as_deref().is_some_and(|v| v != "none");
    s.brush.as_ref().is_none_or(Value::is_null)
        && s.profile.as_ref().is_none_or(Value::is_null)
        && s.taper.as_ref().is_none_or(Value::is_null)
        && s.wave.as_ref().is_none_or(Value::is_null)
        && s.dynamic.as_ref().is_none_or(Value::is_null)
        && s.dash.as_ref().is_none_or(|d| d.iter().all(|v| *v <= 0.0))
        && !arrow(&s.start_arrow)
        && !arrow(&s.end_arrow)
}

/// A fill that hides every fill under it.
fn covers(fill: &Fill) -> bool {
    is_plain(fill.opacity, fill.blend_mode.as_deref()) && paint_is_opaque(&fill.paint)
}

/// A fill that can merge with an adjacent one: solid, normal blend, a parseable colour.
fn mergeable(fill: &Fill) -> Option<Rgba> {
    if layer_blend(fill.blend_mode.as_deref()) != BlendMode::Normal {
        return None;
    }
    let mut c = solid_rgba(&fill.paint)?;
    c.a *= layer_opacity(fill.opacity);
    Some(c)
}

/// `top` source-over `bottom` (straight colours in, straight out).
fn over(top: Rgba, bottom: Rgba) -> Rgba {
    let a = top.a + bottom.a * (1.0 - top.a);
    if a <= 0.0 {
        return Rgba {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
    }
    let ch = |t: f64, b: f64| (t * top.a + b * bottom.a * (1.0 - top.a)) / a;
    Rgba {
        r: ch(top.r, bottom.r),
        g: ch(top.g, bottom.g),
        b: ch(top.b, bottom.b),
        a,
    }
}

/// A colour as the path renderer's parser reads it (integer rgb, as `rgba()` requires).
fn css(c: Rgba) -> String {
    let b = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let a = (c.a.clamp(0.0, 1.0) * 1_000_000.0).round() / 1_000_000.0;
    format!("rgba({},{},{},{})", b(c.r), b(c.g), b(c.b), a)
}

/// The visible fills after the fold (see the module docs), bottom to top. `changed` reports a fold.
fn fold_fills(fills: Vec<Fill>, changed: &mut bool) -> Vec<Fill> {
    let mut fills = fills;
    if let Some(i) = fills.iter().rposition(covers) {
        if i > 0 {
            fills.drain(..i);
            *changed = true;
        }
    }
    let mut out: Vec<(Fill, Option<Rgba>)> = Vec::with_capacity(fills.len());
    for f in fills {
        let rgba = mergeable(&f);
        if let (Some(top), Some((last, Some(bottom)))) = (rgba, out.last_mut()) {
            let merged = over(top, *bottom);
            *bottom = merged;
            // The merged layer keeps the upper layer's id; its opacity is folded into the colour.
            last.id = f.id.clone();
            last.opacity = None;
            last.blend_mode = None;
            last.paint = Paint::Solid(SolidPaint {
                color: Value::String(css(merged)),
            });
            *changed = true;
            continue;
        }
        out.push((f, rgba));
    }
    out.into_iter().map(|(f, _)| f).collect()
}

/// The node's layer plan (see [`LayerPlan`]) at the node's own `opacity` (the frame walk has already multiplied
/// its ancestors' opacity into it).
pub fn plan(node: &RawNode) -> LayerPlan {
    plan_at(node, node.f64("opacity").unwrap_or(1.0))
}

/// Whether a stroke's band lies over the fill: a centred or inside stroke with a width. An outside stroke meets
/// the fill only along the antialiased edge.
fn overlaps_fill(stroke: &Stroke) -> bool {
    stroke.width > 0.0 && stroke.align.as_deref() != Some("outside")
}

/// Whether the node's opacity or blend mode applies to its appearance as one group: it is translucent, or it
/// blends with the backdrop (not normal).
fn appearance_is_grouped(node: &RawNode, opacity: f64) -> bool {
    opacity < 1.0 || BlendMode::from_node_value(node.str("blendMode")) != BlendMode::Normal
}

/// [`plan`] for a node drawn at `opacity` (its effective opacity).
///
/// When the node is translucent or has a (non-normal) blend mode, and has one fill with one stroke over it, the
/// two draw as a two-layer stack, not one body. Illustrator fades and blends an object's appearance as ONE group
/// (isolated under a blend mode): the stroke's inner half hides the fill under it, then the result fades / blends
/// with the backdrop. A single body fades or blends the fill and the stroke separately, so the fill shows through
/// (`stack-fill-stroke-object-50`: 46/255 on the stroke's inner half), or the stroke blends with the fill.
pub fn plan_at(node: &RawNode, opacity: f64) -> LayerPlan {
    if !LAYERED_TYPES.contains(&node.node_type()) {
        return LayerPlan::Single(None);
    }
    // Illustrator's Appearance (E8.2): an interleaved order or an entry effect draws every visible layer as
    // its own clone, in the stored order, with its own effects (`appearance_stack`).
    if node.json_field(ap::APPEARANCE_ORDER_KEY).is_some() || has_entry_effects(node) {
        if let Some(plan) = appearance_stack(node) {
            return plan;
        }
    }
    let mut changed = false;
    let all_fills = sp::node_fills(node);
    // On a translucent or blending node whose strokes are plain, a fill that paints nothing (a fully transparent
    // solid with normal blending: source-over of alpha 0 leaves the backdrop as it is) is dropped like a hidden
    // one, so the stroke over it draws as one body instead of a transparency group (a split pair per node, two
    // render passes and a resolve on every frame). The node then draws exactly as the same node without the fill.
    // Kept on a node with effects (its shadows and filters read the node as stored), on an opaque node (one body
    // already), and under a stroke drawn as several overlapping pieces (brushes, width profiles, arrowheads,
    // dashes), which the group composites as one.
    let drop_clear_fills = appearance_is_grouped(node, opacity)
        && node.array("effects").is_none_or(|e| e.is_empty())
        && sp::node_strokes_shared(node).iter().all(plain_stroke_body);
    let fills: Vec<Fill> = if node.node_type() == "line" {
        Vec::new()
    } else {
        all_fills
            .iter()
            .filter(|f| shows(f.visible, f.opacity) && !(drop_clear_fills && paints_nothing(f)))
            .cloned()
            .collect()
    };
    if fills.len() != all_fills.iter().filter(|f| shows(f.visible, f.opacity)).count() {
        changed = true;
    }
    let fill_count = fills.len();
    // A renderable source-mask shader effect replaces the fill (`paint::classify_fill`): it paints once, in the
    // top fill's slot.
    let effect_fill = crate::paint::classify_fill_is_effect(node);
    let fills: Vec<Fill> = if effect_fill {
        changed |= fill_count > 1;
        fills.into_iter().last().into_iter().collect()
    } else {
        fold_fills(fills, &mut changed)
    };
    let strokes: Vec<Stroke> = sp::node_strokes_shared(node)
        .iter()
        .filter(|s| shows(s.visible, s.opacity))
        .cloned()
        .collect();

    // A vector network whose regions carry their own fills (Figma's region
    // `fills`) draws each region group as its own fill layers.
    if let Some(groups) = region_fill_groups(node, &fills) {
        return region_stack(node, groups, strokes);
    }

    let plain_fill = |f: &Fill| is_plain(f.opacity, f.blend_mode.as_deref());
    let plain_stroke = |s: &Stroke| is_plain(s.opacity, s.blend_mode.as_deref());
    // E8.5: a vector pattern layer draws in its own lane (`frame::build_pattern_layer_items`), so a node with one
    // always draws as a stack.
    let has_pattern = crate::pattern_paint::enabled()
        && (fills.iter().any(|f| matches!(f.paint, Paint::Pattern(_)))
            || strokes.iter().any(|s| matches!(s.paint, Paint::Pattern(_))));
    let single = !has_pattern
        && fills.len() <= 1
        && strokes.len() <= 1
        && fills.iter().all(plain_fill)
        && strokes.iter().all(plain_stroke)
        && !(appearance_is_grouped(node, opacity)
            && !fills.is_empty()
            && strokes.iter().any(overlaps_fill))
        // E6.2: a per-side stroke draws as its own layer (a filled ring, `side_stroke_ring_node` in frame.rs).
        && !(matches!(node.node_type(), "rectangle" | "container") && strokes.iter().any(|s| s.sides.is_some()))
        // E6.1: an open path's arrowed stroke draws as its own layer (its ends cut back under the arrowheads,
        // which the fill must not follow).
        && !(node.node_type() == "path"
            && !node.bool("closed").unwrap_or(false)
            && node.json_field("vectorNetwork").is_none_or(Value::is_null)
            && strokes.iter().any(crate::tess::path_arrows::stroke_has_arrows));
    if single {
        // Unchanged when the fold kept exactly what the node draws as stored (its top fill and top stroke).
        let as_stored = !changed
            && fills.first() == sp::top_fill(node).as_ref()
            && sp::with_top_stroke(node, |top| strokes.first() == top);
        if as_stored {
            return LayerPlan::Single(None);
        }
        let mut m = node.fields().clone();
        sp::set_fills(&mut m, fills);
        sp::set_strokes(&mut m, strokes);
        return LayerPlan::Single(Some(node.with_fields(m)));
    }

    let mut layers = Vec::with_capacity(fills.len() + strokes.len());
    let top_video = top_video_layer(&fills);
    for (i, f) in fills.into_iter().enumerate() {
        let (opacity, blend) = (
            layer_opacity(f.opacity),
            layer_blend(f.blend_mode.as_deref()),
        );
        let bare = Fill {
            visible: None,
            opacity: None,
            blend_mode: None,
            overprint: None,
            ..f
        };
        let mut m = layer_base(node, layers.len(), effect_fill && i == 0);
        if top_video == Some(i) {
            let stream = node.str(VIDEO_STREAM_KEY).unwrap_or(node.id()).to_string();
            m.insert(VIDEO_STREAM_KEY.into(), Value::String(stream));
        }
        sp::set_fills(&mut m, vec![bare]);
        layers.push(PaintLayer {
            node: node.with_fields(m),
            opacity,
            blend,
            is_stroke: false,
            effects: Vec::new(),
        });
    }
    for s in strokes {
        let (opacity, blend) = (
            layer_opacity(s.opacity),
            layer_blend(s.blend_mode.as_deref()),
        );
        let bare = Stroke {
            visible: None,
            opacity: None,
            blend_mode: None,
            overprint: None,
            ..s
        };
        let mut m = layer_base(node, layers.len(), false);
        sp::set_strokes(&mut m, vec![bare]);
        layers.push(PaintLayer {
            node: node.with_fields(m),
            opacity,
            blend,
            is_stroke: true,
            effects: Vec::new(),
        });
    }
    LayerPlan::Stack(layers)
}

/// Whether any stored fill or stroke carries an `effects` list (a cheap pre-check before parsing the stack).
pub fn has_entry_effects(node: &RawNode) -> bool {
    let any = |k: &str| {
        node.array(k)
            .is_some_and(|a| a.iter().any(|l| l.get("effects").is_some_and(|e| !e.is_null())))
    };
    any("fills") || any("strokes")
}

/// The appearance lane (E8.2): `Some` when the node's stack is custom (`appearance::has_custom_stack`: an
/// order other than fills-then-strokes, or an effect on a visible layer). Every visible layer then draws as
/// its own clone, bottom to top in the repaired `appearanceOrder`, unfolded (a covered or mergeable fill may
/// carry effects of its own), each carrying its entry effects for `frame.rs` to apply. A vector network with
/// region fills keeps its region stack (`None`).
fn appearance_stack(node: &RawNode) -> Option<LayerPlan> {
    let fills = if node.node_type() == "line" {
        Vec::new()
    } else {
        sp::node_fills(node)
    };
    let strokes = sp::node_strokes(node);
    if !ap::has_custom_stack(node, &fills, &strokes) {
        return None;
    }
    let visible_fills: Vec<Fill> = fills
        .iter()
        .filter(|f| shows(f.visible, f.opacity))
        .cloned()
        .collect();
    if region_fill_groups(node, &visible_fills).is_some() {
        return None;
    }
    let effect_fill = crate::paint::classify_fill_is_effect(node);
    let top_fill = fills.iter().rposition(|f| shows(f.visible, f.opacity));
    let top_video = fills
        .iter()
        .rposition(|f| shows(f.visible, f.opacity) && matches!(f.paint, Paint::Video(_)));
    let mut layers = Vec::new();
    for entry in ap::ordered_entries(node, &fills, &strokes, false) {
        match entry {
            ap::Entry::Fill(i) => {
                let f = &fills[i];
                if !shows(f.visible, f.opacity) {
                    continue;
                }
                // A source-mask shader effect replaces the fill: it paints once, in the top fill's slot.
                if effect_fill && Some(i) != top_fill {
                    continue;
                }
                let effects = ap::fill_effects(f);
                let bare = Fill {
                    visible: None,
                    opacity: None,
                    blend_mode: None,
                    overprint: None,
                    effects: None,
                    ..f.clone()
                };
                let mut m = layer_base(node, layers.len(), effect_fill && Some(i) == top_fill);
                if top_video == Some(i) {
                    let stream = node.str(VIDEO_STREAM_KEY).unwrap_or(node.id()).to_string();
                    m.insert(VIDEO_STREAM_KEY.into(), Value::String(stream));
                }
                sp::set_fills(&mut m, vec![bare]);
                layers.push(PaintLayer {
                    node: node.with_fields(m),
                    opacity: layer_opacity(f.opacity),
                    blend: layer_blend(f.blend_mode.as_deref()),
                    is_stroke: false,
                    effects,
                });
            }
            ap::Entry::Stroke(i) => {
                let s = &strokes[i];
                if !shows(s.visible, s.opacity) {
                    continue;
                }
                let effects = ap::stroke_effects(s);
                let bare = Stroke {
                    visible: None,
                    opacity: None,
                    blend_mode: None,
                    overprint: None,
                    effects: None,
                    ..s.clone()
                };
                let mut m = layer_base(node, layers.len(), false);
                sp::set_strokes(&mut m, vec![bare]);
                layers.push(PaintLayer {
                    node: node.with_fields(m),
                    opacity: layer_opacity(s.opacity),
                    blend: layer_blend(s.blend_mode.as_deref()),
                    is_stroke: true,
                    effects,
                });
            }
            ap::Entry::Contents => {}
        }
    }
    Some(LayerPlan::Stack(layers))
}

/// The fill groups of a network path whose regions override the node's fills: the regions without an override
/// with the node's (visible, folded) `fills`, then each overriding region with its own. `None` when no region
/// overrides (the node draws as any path).
fn region_fill_groups(node: &RawNode, node_fills: &[Fill]) -> Option<Vec<(Vec<usize>, Vec<Fill>)>> {
    if node.node_type() != "path" {
        return None;
    }
    let regions = node.json_field("vectorNetwork")?.get("regions")?.as_array()?;
    fn own(r: &Value) -> Option<&Value> {
        r.get("fills").filter(|f| f.is_array())
    }
    if !regions.iter().any(|r| own(r).is_some()) {
        return None;
    }
    let mut base = Vec::new();
    let mut groups = Vec::new();
    for (i, r) in regions.iter().enumerate() {
        match own(r) {
            Some(f) => {
                let fills: Vec<Fill> = sp::parse_fills(Some(f))
                    .into_iter()
                    .filter(|f| shows(f.visible, f.opacity))
                    .collect();
                let mut changed = false;
                groups.push((vec![i], fold_fills(fills, &mut changed)));
            }
            None => base.push(i),
        }
    }
    let mut out = Vec::with_capacity(groups.len() + 1);
    if !base.is_empty() {
        out.push((base, node_fills.to_vec()));
    }
    out.extend(groups);
    Some(out)
}

/// A clone's network keeps only `keep` of its regions (their own `fills` stripped: the layer's fill paints them).
fn keep_regions(m: &mut Map, keep: &[usize]) {
    let Some(Value::Object(net)) = m.get_mut("vectorNetwork") else {
        return;
    };
    let Some(Value::Array(regions)) = net.get("regions") else {
        return;
    };
    let kept: Vec<Value> = keep
        .iter()
        .filter_map(|&i| regions.get(i).cloned())
        .map(|mut r| {
            if let Value::Object(o) = &mut r {
                o.remove("fills");
            }
            r
        })
        .collect();
    net.insert("regions".into(), Value::Array(kept));
}

/// The layers of a network path with region fills: each group's fills over its regions, bottom to top in region
/// order, then the strokes (over the whole network).
fn region_stack(
    node: &RawNode,
    groups: Vec<(Vec<usize>, Vec<Fill>)>,
    strokes: Vec<Stroke>,
) -> LayerPlan {
    let mut layers = Vec::new();
    for (regions, fills) in groups {
        for f in fills {
            let (opacity, blend) = (
                layer_opacity(f.opacity),
                layer_blend(f.blend_mode.as_deref()),
            );
            let bare = Fill {
                visible: None,
                opacity: None,
                blend_mode: None,
                overprint: None,
                ..f
            };
            let mut m = layer_base(node, layers.len(), false);
            keep_regions(&mut m, &regions);
            sp::set_fills(&mut m, vec![bare]);
            layers.push(PaintLayer {
                node: node.with_fields(m),
                opacity,
                blend,
                is_stroke: false,
                effects: Vec::new(),
            });
        }
    }
    for s in strokes {
        let (opacity, blend) = (
            layer_opacity(s.opacity),
            layer_blend(s.blend_mode.as_deref()),
        );
        let bare = Stroke {
            visible: None,
            opacity: None,
            blend_mode: None,
            overprint: None,
            ..s
        };
        let mut m = layer_base(node, layers.len(), false);
        sp::set_strokes(&mut m, vec![bare]);
        layers.push(PaintLayer {
            node: node.with_fields(m),
            opacity,
            blend,
            is_stroke: true,
            effects: Vec::new(),
        });
    }
    LayerPlan::Stack(layers)
}

/// Whether a stack draws as a transparency group: the node's opacity or blend mode applies to its layers
/// composited together (Illustrator's object opacity / blending over the object's whole appearance), which
/// per-layer draws can't express once two layers overlap. `opacity` is the node's effective opacity (its own
/// times its ancestors').
pub fn needs_group(opacity: f64, blend: BlendMode, layers: &[PaintLayer]) -> bool {
    match layers {
        [] => false,
        [only] => blend != BlendMode::Normal && only.blend != BlendMode::Normal,
        _ => opacity < 1.0 || blend != BlendMode::Normal,
    }
}

/// The blend a layer draws with outside a group: its own, else the node's (a one-layer stack).
pub fn effective_blend(node_blend: BlendMode, layer: &PaintLayer) -> BlendMode {
    if layer.blend != BlendMode::Normal {
        layer.blend
    } else {
        node_blend
    }
}

/// Whether the node's paint stack needs blend splits (a group, or a layer blend that reads the backdrop) on a
/// target that may be `transparent`: its page then plans them (`backdrop::blend::page_has_split_blend_over`).
pub fn stack_needs_split(node: &RawNode, ancestor_opacity: f64, transparent: bool) -> bool {
    let opacity = node.f64("opacity").unwrap_or(1.0) * ancestor_opacity;
    let LayerPlan::Stack(layers) = plan_at(node, opacity) else {
        return false;
    };
    let blend = BlendMode::from_node_value(node.str("blendMode"));
    needs_group(opacity, blend, &layers)
        || layers.iter().any(|l| {
            effective_blend(blend, l)
                .split_id_over(transparent)
                .is_some()
        })
}

/// A rich-text node's glyph-fill override (`textFills`): the text lane draws ONE paint per glyph, so a stack
/// folds to one when it can (solids merge, a covering layer hides those under it, a solid's layer opacity
/// folds into its colour); otherwise the top visible layer draws (the manifest reports `paint:text-stack`).
pub fn text_fill_paint(node: &RawNode) -> Option<Paint> {
    if node.node_type() != "richtext" {
        return None;
    }
    let fills: Vec<Fill> = sp::parse_fills(node.json_field("textFills"))
        .into_iter()
        .filter(|f| shows(f.visible, f.opacity))
        .collect();
    let mut changed = false;
    let mut folded = fold_fills(fills, &mut changed);
    let top = folded.pop()?;
    // A lone translucent solid: its opacity is its colour's alpha.
    match (&top.paint, mergeable(&top)) {
        (Paint::Solid(_), Some(c)) if top.opacity.is_some() => Some(Paint::Solid(SolidPaint {
            color: Value::String(css(c)),
        })),
        _ => Some(top.paint),
    }
}

/// The node field naming whose live video a layer clone draws: set on the clone of the node's top visible
/// video layer (the host streams one video per node, under `video:<nodeId>`); other video layers draw their
/// posters.
pub const VIDEO_STREAM_KEY: &str = "_videoStreamId";

/// The index (in `fills`) of the top visible video layer.
fn top_video_layer(fills: &[Fill]) -> Option<usize> {
    fills
        .iter()
        .rposition(|f| matches!(f.paint, Paint::Video(_)))
}

/// The synthetic id of layer `k` of `node`.
pub fn layer_id(node_id: &str, k: usize) -> String {
    format!("{node_id}::paint::{k}")
}

/// The scene node a draw item id belongs to: the owner of a layer clone (`layer_id`), else the id itself.
pub fn layer_owner_id(item_id: &str) -> &str {
    match item_id.rsplit_once("::paint::") {
        Some((owner, k)) if !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()) => owner,
        _ => item_id,
    }
}

/// A layer clone without paints: the node's own fields, a synthetic id, the transform redirect, no drop or
/// inner shadow effects (the node draws its shadows once) and no shader effect unless `keep_effect`.
fn layer_base(node: &RawNode, k: usize, keep_effect: bool) -> Map {
    let mut m = node.fields().clone();
    m.remove("fills");
    m.remove("strokes");
    if let Some(Value::Array(effects)) = m.get_mut("effects") {
        effects.retain(|e| match e.get("type").and_then(Value::as_str) {
            Some("drop-shadow" | "inner-shadow") => false,
            Some("shader") => keep_effect,
            _ => true,
        });
        if effects.is_empty() {
            m.remove("effects");
        }
    }
    m.remove(VIDEO_STREAM_KEY);
    m.remove(ap::APPEARANCE_ORDER_KEY);
    let source = moda_scene::scene::paint_transform_id(node).to_string();
    m.insert("_transformSourceId".into(), Value::String(source));
    m.insert("id".into(), Value::String(layer_id(node.id(), k)));
    m
}

/// The widest visible stroke (the one whose band reaches farthest): what the node's drop shadow and blend
/// regions pad by. `None` without a visible stroke.
pub fn widest_stroke(node: &RawNode) -> Option<Stroke> {
    sp::node_strokes(node)
        .into_iter()
        .filter(|s| shows(s.visible, s.opacity))
        .max_by(|a, b| a.width.total_cmp(&b.width))
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn raw(v: Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    fn solid(id: &str, color: &str) -> Value {
        json!({"id": id, "paint": {"type": "solid", "color": color}})
    }

    fn stack(p: &LayerPlan) -> &[PaintLayer] {
        match p {
            LayerPlan::Stack(l) => l,
            LayerPlan::Single(_) => panic!("expected a stack, got {p:?}"),
        }
    }

    #[test]
    fn network_region_fills_draw_as_their_own_layers() {
        let n = raw(json!({
            "id": "v", "type": "path", "points": [],
            "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ff0000"}}],
            "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000000"}, "width": 2.0}],
            "vectorNetwork": {
                "vertices": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 0, "y": 10}, {"x": 20, "y": 0}],
                "segments": [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 2, "end": 3}, {"start": 3, "end": 0},
                             {"start": 1, "end": 4}, {"start": 4, "end": 2}],
                "regions": [
                    {"windingRule": "NONZERO", "loops": [[0, 1, 2, 3]]},
                    {"windingRule": "NONZERO", "loops": [[4, 5, 1]],
                     "fills": [{"id": "g", "paint": {"type": "solid", "color": "#00ff00"}}]}
                ]
            }
        }));
        let planned = plan(&n);
        let layers = stack(&planned);
        assert_eq!(layers.len(), 3);
        let regions = |l: &PaintLayer| {
            l.node.json_field("vectorNetwork").unwrap()["regions"]
                .as_array()
                .unwrap()
                .len()
        };
        assert_eq!(regions(&layers[0]), 1);
        assert_eq!(regions(&layers[1]), 1);
        assert!(layers[1].node.json_field("vectorNetwork").unwrap()["regions"][0]
            .get("fills")
            .is_none());
        let color = |l: &PaintLayer| {
            sp::top_fill(&l.node)
                .map(|f| format!("{:?}", f.paint))
                .unwrap_or_default()
        };
        assert!(color(&layers[0]).contains("ff0000"));
        assert!(color(&layers[1]).contains("00ff00"));
        assert!(layers[2].is_stroke);
    }

    #[test]
    fn layer_owner_id_maps_layer_clones_back_to_their_node() {
        assert_eq!(layer_owner_id(&layer_id("card_3", 0)), "card_3");
        assert_eq!(layer_owner_id(&layer_id("a::paint::b", 12)), "a::paint::b");
        assert_eq!(layer_owner_id("plain"), "plain");
        assert_eq!(layer_owner_id("n::paint::"), "n::paint::");
        assert_eq!(layer_owner_id("n::paint::x1"), "n::paint::x1");
    }

    #[test]
    fn single_layer_nodes_draw_as_stored() {
        let n = raw(
            json!({"id": "n", "type": "rectangle", "fills": [solid("f", "#ff0000")],
            "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000"}, "width": 2}]}),
        );
        assert!(matches!(plan(&n), LayerPlan::Single(None)));
        // A hidden lower layer changes nothing either.
        let n = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "a", "visible": false, "paint": {"type": "solid", "color": "#00f"}}, solid("f", "#ff0000")]}));
        assert!(matches!(plan(&n), LayerPlan::Single(None)));
        // Text and groups are not layered here.
        let n = raw(
            json!({"id": "n", "type": "richtext", "fills": [solid("a", "#fff"), solid("b", "#000")]}),
        );
        assert!(matches!(plan(&n), LayerPlan::Single(None)));
    }

    #[test]
    fn an_opaque_fill_covers_the_fills_under_it() {
        let lin = json!({"id": "g", "paint": {"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0},
            "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}]}});
        let n = raw(json!({"id": "n", "type": "rectangle", "fills": [solid("a", "#00ff00"), lin]}));
        let LayerPlan::Single(Some(folded)) = plan(&n) else {
            panic!("the gradient covers the solid")
        };
        assert_eq!(sp::node_fills(&folded).len(), 1);
        assert_eq!(folded.id(), "n", "a folded single body keeps the node's id");
    }

    #[test]
    fn adjacent_solids_merge_into_one_colour() {
        // 50% red under 50% blue: a = 0.75, colour = (0.5·blue + 0.25·red) / 0.75.
        let n = raw(json!({"id": "n", "type": "ellipse", "fills": [
            {"id": "a", "opacity": 0.5, "paint": {"type": "solid", "color": "#ff0000"}},
            {"id": "b", "paint": {"type": "solid", "color": "rgba(0,0,255,0.5)"}}]}));
        let LayerPlan::Single(Some(folded)) = plan(&n) else {
            panic!("two solids fold into one")
        };
        let fills = sp::node_fills(&folded);
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].id, "b");
        assert_eq!(
            sp::solid_color(&fills[0].paint),
            Some("rgba(85,0,170,0.75)")
        );
    }

    #[test]
    fn stacks_run_fills_then_strokes_bottom_to_top() {
        let img = json!({"id": "i", "opacity": 0.5, "paint": {"type": "image", "src": "data:x"}});
        let n = raw(
            json!({"id": "n", "type": "rectangle", "opacity": 0.8,
            "effects": [{"id": "drop-shadow-0", "type": "drop-shadow", "color": "#000", "blur": 4},
                {"id": "layer-blur-0", "type": "layer-blur", "radius": 2}],
            "fills": [solid("a", "#00ff00"), img],
            "strokes": [
                {"id": "s1", "paint": {"type": "solid", "color": "#000"}, "width": 8},
                {"id": "s2", "blendMode": "multiply", "paint": {"type": "solid", "color": "#f00"}, "width": 2}]}),
        );
        let p = plan(&n);
        let l = stack(&p);
        assert_eq!(l.len(), 4);
        assert_eq!(
            l.iter().map(|l| l.is_stroke).collect::<Vec<_>>(),
            [false, false, true, true]
        );
        assert_eq!(l[1].opacity, 0.5);
        assert_eq!(l[3].blend, BlendMode::Multiply);
        for (k, layer) in l.iter().enumerate() {
            assert_eq!(layer.node.id(), format!("n::paint::{k}"));
            assert_eq!(layer.node.str("_transformSourceId"), Some("n"));
            assert!(
                moda_scene::effect::drop_shadow(&layer.node).is_none(),
                "shadows draw once, from the node"
            );
            assert!(moda_scene::effect::layer_blur(&layer.node).is_some(), "other effects stay");
            let (f, s) = (sp::node_fills(&layer.node), sp::node_strokes(&layer.node));
            assert_eq!(f.len() + s.len(), 1);
            assert!(f.iter().all(|f| f.opacity.is_none() && f.visible.is_none()));
            assert!(s.iter().all(|s| s.blend_mode.is_none()));
        }
        assert_eq!(sp::node_strokes(&l[2].node)[0].width, 8.0);
    }

    #[test]
    fn a_translucent_or_blended_single_layer_is_a_stack() {
        // One fill at 50%: its opacity must apply, so it draws as a one-layer stack.
        let n = raw(json!({"id": "n", "type": "rectangle", "fills": [
            {"id": "i", "opacity": 0.5, "paint": {"type": "image", "src": "data:x"}}]}));
        assert_eq!(stack(&plan(&n)).len(), 1);
        let n = raw(json!({"id": "n", "type": "line",
            "strokes": [{"id": "s", "blendMode": "screen", "paint": {"type": "solid", "color": "#f00"}, "width": 2}]}));
        assert_eq!(stack(&plan(&n))[0].blend, BlendMode::Screen);
        // Opacity 0 hides a layer (the TS `layerShows`).
        let n = raw(json!({"id": "n", "type": "rectangle", "fills": [
            {"id": "a", "opacity": 0, "paint": {"type": "solid", "color": "#f00"}}, solid("b", "#0f0")]}));
        assert!(matches!(plan(&n), LayerPlan::Single(None)));
    }
}
