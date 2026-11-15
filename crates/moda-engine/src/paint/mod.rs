//! paint/ — gradients, patterns, filters, perspective. OWNED BY M3b.
//!
//! Submodules:
//! - `gradient` — gradient model port + byte-compatible `PathGradientData`
//!   packing (8 stops, `floor(b·255)+a·0.99999` encoding, offset-10000
//!   sentinels, bounds-include-stroke, conic CSS parity, gradient strokes;
//!   conic STROKES unsupported — Q-18, copy the gap),
//! - `pattern` — `PathPatternData` packing (repeat encode, HSV/colorChange/
//!   overlay chain data, inverse-homography perspective, placeholder
//!   constants Q-22),
//! - `homography` — `core/scene/projectiveTransform.ts` port,
//! - `image_source` — minimal data-URI texture sourcing (SEAM: replaced by
//!   moda-assets at M3c/M4 integration),
//! - `draw` — GPU-side helpers (pattern texture upload + bind-group cache).
//!
//! The extracted WGSL (`wgsl/path-gradient.wgsl`, `wgsl/path-pattern.wgsl`)
//! does ALL the paint math — this module only packs data and routes draws
//! (DECISION §1.1). Blend/depth state comes from the manifest pipeline
//! descriptors, never hand-coded.
//!
//! ## Paint routing (`WebGPUPathRenderer.renderUnifiedPathBatch`,
//! `WebGPUPathRenderer.ts:1076-1133` — order is normative, copy the bugs)
//!
//! The fill is the node's top visible layer (`moda_scene::paint::node_fills`):
//! its paint `type` is the route. Every visible layer draws (E3.1 P6): a
//! stack draws one clone of the node per layer (`layers`), so each clone's
//! top layer is the layer it draws.

//!
//! 1. custom shader fill (a renderable `shader` effect, or a shader paint) →
//!    shader pipeline (M3f),
//! 2. a gradient paint → gradient pipeline,
//! 3. an image or video paint → pattern pipeline; a visible gradient
//!    stroke adds a second, stroke-only gradient overlay draw,
//! 4. visible gradient stroke on a solid/no fill → gradient pipeline
//!    (fillType 0 keeps the fill solid),
//! 5. otherwise → solid pipeline.

pub mod draw;
pub mod gradient;
pub mod homography;
pub mod image_source;
pub mod layers;
pub mod pattern;
pub mod shape_sdf;

pub use gradient::{
    build_gradient_instance, fill_paints, gradient_box, gradient_has_transparency,
    has_gradient_fill, has_stroke_paint, has_visible_gradient_stroke, painted_stroke_width,
    stroke_gradient_has_transparency, GradientInstance,
};
pub use image_source::{
    resolve_pattern_source, resolve_video_source, DecodedImage, ExternalTextureRef, PatternTexture,
    PatternTextureState,
};
pub use pattern::{
    build_pattern_instance, node_distort, NodeDistort, PatternInstance, PatternInstanceContext,
    PerspectivePattern,
};
pub use shape_sdf::{
    build_gradient_shape_instance, build_solid_shape_instance, classify_gradient_shape,
    classify_solid_shape_body, stroke_preserves_shadow_cover, ShapeGradientSdfInstance,
    ShapeSolidSdfInstance, ShapeType, StrokeFields,
};

use moda_scene::paint::Paint;
use moda_scene::types::RawNode;

/// A pattern body's full draw payload: packed instance + resolved texture +
/// sampler selection. Carried by `seams::BodyPaint::Pattern` (GPU-free, so
/// the `CoveragePlan` IR stays consumable by a CPU backend).
#[derive(Debug, Clone, PartialEq)]
pub struct PatternBody {
    pub instance: PatternInstance,
    pub texture: PatternTexture,
    /// `fillPatternRepeat === 'no-repeat'` → clamp sampler, else repeat
    /// (`WebGPUPathRenderer.createTextureBindGroup`, `:2276-2293`).
    pub clamp_sampler: bool,
}

impl PartialEq for PatternTexture {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.state == other.state
    }
}

impl PartialEq for GradientInstance {
    fn eq(&self, other: &Self) -> bool {
        bytemuck::bytes_of(self) == bytemuck::bytes_of(other)
    }
}

impl PartialEq for PatternInstance {
    fn eq(&self, other: &Self) -> bool {
        bytemuck::bytes_of(self) == bytemuck::bytes_of(other)
    }
}

/// How a body's FILL routes (step 1 is reported separately since M3f owns it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillRoute {
    /// Custom shader fill — M3f pipeline.
    Shader,
    /// Gradient pipeline owns the whole body (fill gradient, or solid fill +
    /// gradient stroke).
    Gradient,
    /// Pattern pipeline owns the fill (gradient-stroke overlay drawn extra).
    Pattern,
    Solid,
}

/// Whether a renderable source-mask shader EFFECT paints the node's fill (it beats every fill layer, see
/// [`classify_fill`]).
pub fn classify_fill_is_effect(node: &RawNode) -> bool {
    moda_scene::shader_effects::renderable_source_mask_effect(node).is_some()
}

/// Production routing order — see module docs.
///
/// Routes the node's fill stack (`moda_scene::paint::node_fills`) from the top
/// layer down: the first visible layer's paint type names the pipeline (a
/// paint stack routes each layer's clone, `layers`).
pub fn classify_fill(node: &RawNode) -> FillRoute {
    // M4e: the custom-shader route gate is `getRenderableSourceMaskShaderEffect
    // || a shader fill` (WebGPUPathRenderer.ts:
    // 1076, 605-608) — a renderable source-mask EFFECT beats every fill

    // (including pattern); a stale self-asset effect falls through to the
    // underlying fill.
    if moda_scene::shader_effects::renderable_source_mask_effect(node).is_some() {
        return FillRoute::Shader;
    }
    for fill in moda_scene::paint::node_fills_shared(node).iter().rev() {
        // The layers the layer plan draws (`shown`: switched on and not at 0% opacity), so the route is the top
        // DRAWN layer's: a 0% image over a solid used to route Pattern and draw the white placeholder.
        if !moda_scene::paint::shown(fill.visible, fill.opacity) {
            continue;
        }
        match fill.paint {
            Paint::Shader(_) => return FillRoute::Shader,
            Paint::Linear(_)
            | Paint::Radial(_)
            | Paint::Conic(_)
            | Paint::Diamond(_)
            | Paint::Freeform(_) => return FillRoute::Gradient,
            // A video fill renders through the pattern texture path (its
            // poster, or the live `video:<nodeId>` stream), exactly like an
            // image; a missing poster/src resolves to the same white
            // placeholder.
            Paint::Image(_) | Paint::Video(_) => return FillRoute::Pattern,
            // A gradient mesh is baked to an image and drawn by the image lane (`crate::mesh_paint`).
            Paint::Mesh(_) => return FillRoute::Pattern,
            // A vector pattern (E8.5) draws in its own lane (`pattern_paint`), never through a body pipeline:
            // the layer under it routes.
            Paint::Pattern(_) => continue,
            Paint::Solid(_) => break,
        }
    }
    if has_visible_gradient_stroke(node) {
        return FillRoute::Gradient;
    }
    FillRoute::Solid
}

#[cfg(test)]
mod tests {
    use super::*;
    use moda_scene::value::json;

    fn raw(v: moda_scene::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    const LINEAR: &str = r##"{"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0},
        "stops": [{"offset": 0, "color": "#fff"}, {"offset": 1, "color": "#000"}]}"##;

    fn linear() -> moda_scene::value::Value {
        moda_scene::value::from_str(LINEAR).unwrap()
    }

    #[test]
    fn routing_order_copies_the_bugs() {
        // The top visible fill's paint type routes ([Q1]: gradient fields no
        // longer beat the chosen image). An image over a gradient → Pattern.
        let node = raw(json!({"id": "n", "type": "rectangle", "fills": [
            {"id": "fill-0", "paint": linear()},
            {"id": "fill-1", "paint": {"type": "image", "src": "data:x"}}]}));
        assert_eq!(classify_fill(&node), FillRoute::Pattern);
        // …and with the image hidden, the gradient under it routes.
        let node = raw(json!({"id": "n", "type": "rectangle", "fills": [
            {"id": "fill-0", "paint": linear()},
            {"id": "fill-1", "visible": false, "paint": {"type": "image", "src": "data:x"}}]}));
        assert_eq!(classify_fill(&node), FillRoute::Gradient);
        // …and so does a 0% image (it doesn't draw): the solid under it routes, as the layer plan draws it.
        let node = raw(json!({"id": "n", "type": "rectangle", "fills": [
            {"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}},
            {"id": "fill-1", "opacity": 0.0, "paint": {"type": "image", "src": "data:x"}}]}));
        assert_eq!(classify_fill(&node), FillRoute::Solid);

        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "shader", "shaderId": "clouds"}}]}));
        assert_eq!(classify_fill(&node), FillRoute::Shader);

        // Solid fill + visible gradient stroke → gradient pipeline.
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],
            "strokes": [{"id": "stroke-0", "paint": linear(), "width": 4}]}));
        assert_eq!(classify_fill(&node), FillRoute::Gradient);

        // Zero-width gradient stroke falls through to solid.
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],
            "strokes": [{"id": "stroke-0", "paint": linear(), "width": 0}]}));
        assert_eq!(classify_fill(&node), FillRoute::Solid);

        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:x", "placement": {}}}]}));
        assert_eq!(classify_fill(&node), FillRoute::Pattern);
    }

    #[test]
    fn video_fill_routes_to_pattern() {
        // A `'video'` fill renders its poster through the pattern
        // texture path — it routes to Pattern just like an image pattern.
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "video", "posterSrc": "data:x", "playback": {}, "placement": {}}}]}));
        assert_eq!(classify_fill(&node), FillRoute::Pattern);

        // Mirrors pattern: a video fill with no poster still routes to Pattern
        // (resolves to the white placeholder downstream).
        let node = raw(json!({"id": "n", "type": "rectangle", "fills": [{"id": "fill-0", "paint": {"type": "video", "playback": {}, "placement": {}}}]}));
        assert_eq!(classify_fill(&node), FillRoute::Pattern);

        // A gradient layer above the video routes to Gradient (the top layer
        // draws; it used to be any stale gradient field, [Q1]).
        let node = raw(json!({"id": "n", "type": "rectangle", "fills": [
            {"id": "fill-0", "paint": {"type": "video", "posterSrc": "data:x"}},
            {"id": "fill-1", "paint": linear()}]}));
        assert_eq!(classify_fill(&node), FillRoute::Gradient);
    }
}
