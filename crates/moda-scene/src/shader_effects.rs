//! M4e — source-mask shader-effect predicates
//! (`core/shaders/shaderEffects.ts` clones over `RawNode`).
//!
//! Shared by the asset collector (moda-assets), the engine's paint routing
//! (moda-engine `paint::classify_fill` / `build_body_item`), and the manifest
//! walker (`assess.rs`) so all three agree with the production renderer's
//! routing exactly.

use crate::types::RawNode;
use crate::value::Value;

pub const SOURCE_MASK_SHADER_EFFECT_MODE: &str = "source-mask-material";

/// The top fill's image source, when that fill is an image with a source.
fn image_fill_src(node: &RawNode) -> Option<&str> {
    let paint = crate::paint::top_fill_paint_json(node)?;
    if paint.get("type").and_then(Value::as_str) != Some("image") {
        return None;
    }
    paint
        .get("src")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// `canApplyShaderEffectToImageFill` (`shaderEffects.ts`): the top fill is an
/// image with a source.
fn can_apply_to_image_fill(node: &RawNode) -> bool {
    image_fill_src(node).is_some()
}

/// The node's `shader` entries of `effects` (visible or not), in list order.
pub fn shader_effect_entries(node: &RawNode) -> impl Iterator<Item = &Value> {
    crate::effect::node_effects_json(node)
        .iter()
        .filter(|e| e.get("type").and_then(Value::as_str) == Some("shader"))
}

/// `getRenderableSourceMaskShaderEffect` (`shaderEffects.ts:59-66`): the
/// FIRST `type: 'shader', mode: 'source-mask-material'` effect, unless it is
/// hidden (`visible: false`) or a stale `self-asset` effect whose pattern-fill
/// source has gone away.
///
/// CONTRACT: must return a reference INTO the node's own `effects`
/// array (never a clone) — `assess.rs` identity-compares entries against this
/// return with `std::ptr::eq` to split renderable vs inert; a cloned return
/// would silently classify the renderable entry as inert.
pub fn renderable_source_mask_effect(node: &RawNode) -> Option<&Value> {
    let effect = shader_effect_entries(node)
        .find(|e| e.get("mode").and_then(Value::as_str) == Some(SOURCE_MASK_SHADER_EFFECT_MODE))?;
    // Hidden with the Effects eye toggle (`visible: false`, design-editor E1.9):
    // the node renders its underlying fill, as for a stale effect.
    if !crate::effect::is_visible_json(effect) {
        return None;
    }
    let source_type = effect
        .get("source")
        .and_then(|s| s.get("type"))
        .and_then(Value::as_str);
    if source_type == Some("self-asset") && !can_apply_to_image_fill(node) {
        return None;
    }
    Some(effect)
}

/// `getShaderEffectImageSource` (`WebGPUPathRenderer.ts:115-136`): the image
/// SRC for a renderable effect — `self-asset` → the node's image fill
/// source, `asset` → `source.assetId`; `node-mask` /
/// `rendered-node` (and absent sources) resolve to None (the manager's
/// fallback texture, `ready: false`).
pub fn effect_image_src<'a>(node: &'a RawNode, effect: &'a Value) -> Option<&'a str> {
    let source = effect.get("source")?;
    match source.get("type").and_then(Value::as_str) {
        Some("self-asset") => image_fill_src(node),
        Some("asset") => source
            .get("assetId")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty()),
        _ => None,
    }
}

/// The effective custom-shader routing id + raw params: a renderable
/// source-mask effect's `shaderId` and params, else the top fill's shader
/// paint (`shaderId`, `params`).
pub fn effective_shader_fill<'a>(
    node: &'a RawNode,
) -> Option<(&'a str, Option<&'a Value>, Option<&'a Value>)> {
    if let Some(effect) = renderable_source_mask_effect(node) {
        // `effect.shaderId || fillShaderId` — JS falsy: empty string falls
        // through to the fill route.
        if let Some(id) = effect
            .get("shaderId")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return Some((id, effect.get("params"), Some(effect)));
        }
    }
    let paint = crate::paint::top_fill_paint_json(node)?;
    if paint.get("type").and_then(Value::as_str) != Some("shader") {
        return None;
    }
    let id = paint
        .get("shaderId")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    Some((id, paint.get("params"), None))
}

/// Overlay live image-fill placement onto effect params for shaders whose
/// descriptor declares `placement: source-fill`. The pattern fill remains the
/// single source of truth after crop, pan, rotate, or flip edits; stale copies
/// persisted in an older effect are deliberately ignored.
pub fn with_source_fill_placement(node: &RawNode, params: &Value) -> Value {
    let mut out = params.as_object().cloned().unwrap_or_default();
    // The image or video fill's placement (the top fill).
    let placement = crate::paint::top_fill_paint_json(node)
        .filter(|p| matches!(p.get("type").and_then(Value::as_str), Some("image" | "video")))
        .and_then(|p| p.get("placement"));
    let at = |key: &str| placement.and_then(|p| p.get(key));
    let width = positive_number(at("naturalSize").and_then(|value| value.get("width")))
    .or_else(|| positive_number(node.json_field("width")))
    .unwrap_or(1.0);
    let height = positive_number(at("naturalSize").and_then(|value| value.get("height")),
    )
    .or_else(|| positive_number(node.json_field("height")))
    .unwrap_or(1.0);
    let fallback_scale_x = positive_number(node.json_field("width"))
        .map(|node_width| node_width / width)
        .unwrap_or(1.0);
    let fallback_scale_y = positive_number(node.json_field("height"))
        .map(|node_height| node_height / height)
        .unwrap_or(1.0);
    // The image's scale mode resolved against the node's box (the TS twin
    // `getImagePlacementParams`); without an image fill the source stretches.
    let resolved = placement
        .and_then(|p| crate::value::from_value::<crate::paint::ImagePlacement>(p.clone()).ok())
        .map(|p| {
            p.resolve(
                [
                    0.0,
                    0.0,
                    finite_number(node.json_field("width")).unwrap_or(0.0),
                    finite_number(node.json_field("height")).unwrap_or(0.0),
                ],
                None,
            )
        });
    let (scale_x, scale_y) = resolved
        .map(|r| r.scale)
        .unwrap_or((fallback_scale_x, fallback_scale_y));
    let (offset_x, offset_y) = resolved.map(|r| r.offset).unwrap_or((0.0, 0.0));
    let rotation = resolved.map(|r| r.rotation).unwrap_or(0.0).to_radians();

    for (key, value) in [
        ("scale", (scale_x + scale_y) / 2.0),
        ("scaleX", scale_x),
        ("scaleY", scale_y),
        ("rotation", rotation),
        ("offsetX", offset_x),
        ("offsetY", offset_y),
        ("sourceWidth", width),
        ("sourceHeight", height),
    ] {
        out.insert(key.into(), crate::value::json!(value));
    }
    Value::Object(out)
}

fn finite_number(value: Option<&Value>) -> Option<f64> {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
}

fn positive_number(value: Option<&Value>) -> Option<f64> {
    finite_number(value).filter(|value| *value > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn raw(v: crate::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    #[test]
    fn hidden_effects_are_skipped() {
        // design-editor E1.9: a hidden `visible: false` effect is not rendered (the
        // node shows its fill); later source-mask entries stay unread, as before.
        let hidden = json!({"id": "fx-a", "type": "shader", "mode": "source-mask-material",
            "shaderId": "a", "source": {"type": "asset", "assetId": "img"}, "params": {}, "visible": false});
        let node = raw(json!({"id": "n", "type": "rectangle", "effects": [hidden.clone()]}));
        assert!(renderable_source_mask_effect(&node).is_none());
        assert!(effective_shader_fill(&node).is_none());

        let node = raw(
            json!({"id": "n", "type": "rectangle", "effects": [hidden,
            {"id": "fx-b", "type": "shader", "mode": "source-mask-material",
             "shaderId": "b", "source": {"type": "asset", "assetId": "img"}, "params": {}}]}),
        );
        assert!(renderable_source_mask_effect(&node).is_none());
    }

    #[test]
    fn self_asset_requires_resolvable_pattern_fill() {
        // Stale self-asset effect: no pattern fill → NOT renderable.
        let node = raw(json!({"id": "n", "type": "rectangle",
            "effects": [{"id": "fx", "type": "shader", "mode": "source-mask-material",
                "shaderId": "liquid-metal-image", "source": {"type": "self-asset"}, "params": {}}]}));
        assert!(renderable_source_mask_effect(&node).is_none());
        assert!(effective_shader_fill(&node).is_none());

        // Resolvable: pattern fill present.
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:image/svg+xml;base64,AAAA"}}],
            "effects": [{"id": "fx", "type": "shader", "mode": "source-mask-material",
                "shaderId": "liquid-metal-image", "source": {"type": "self-asset"},
                "params": {"speed": 2}}]}));
        let effect = renderable_source_mask_effect(&node).expect("renderable");
        assert_eq!(
            effect_image_src(&node, effect),
            Some("data:image/svg+xml;base64,AAAA")
        );
        let (id, params, eff) = effective_shader_fill(&node).unwrap();
        assert_eq!(id, "liquid-metal-image");
        assert_eq!(params.unwrap().get("speed"), Some(&json!(2)));
        assert!(eff.is_some());
    }

    #[test]
    fn non_source_mask_effects_do_not_route() {
        // post-process mode / non-shader effects are ignored by this path.
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}],
            "effects": [{"id": "fx", "type": "shader", "mode": "post-process",
                "shaderId": "x", "source": {"type": "self-asset"}, "params": {}}]}));
        assert!(renderable_source_mask_effect(&node).is_none());
        assert!(effective_shader_fill(&node).is_none());
    }

    #[test]
    fn fill_shader_id_route_has_no_effect_and_no_image() {
        let node = raw(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "shader", "shaderId": "dither-image",
                "params": {"image": {"kind": "image", "src": "https://x/y.png"}}}}]}));
        let (id, params, eff) = effective_shader_fill(&node).unwrap();
        assert_eq!(id, "dither-image");
        assert!(eff.is_none(), "shader-paint route carries NO effect (and thus no image source — WebGPUPathRenderer.ts:2050)");
        assert!(params.unwrap().get("image").is_some());
    }

    #[test]
    fn asset_source_uses_asset_id() {
        let node = raw(json!({"id": "n", "type": "rectangle",
            "effects": [{"id": "fx", "type": "shader", "mode": "source-mask-material",
                "shaderId": "heatmap-image",
                "source": {"type": "asset", "assetId": "https://cdn/x.svg"}, "params": {}}]}));
        let effect = renderable_source_mask_effect(&node)
            .expect("asset source renderable without pattern fill");
        assert_eq!(effect_image_src(&node, effect), Some("https://cdn/x.svg"));
    }

    #[test]
    fn source_fill_placement_overrides_stale_effect_params_and_preserves_flips() {
        let node = raw(json!({
            "id": "n",
            "type": "rectangle",
            "width": 200,
            "height": 100,
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://x/a.png",
                "placement": {
                    "fit": "crop",
                    "naturalSize": {"width": 400, "height": 200},
                    "scale": {"x": -0.75, "y": 1.25},
                    "offset": {"x": 12, "y": -8},
                    "rotation": 90
                }}}]
        }));
        let params = json!({
            "scaleX": 9,
            "scaleY": 9,
            "rotation": 0,
            "offsetX": 0,
            "offsetY": 0,
            "colors": ["#000000", "#ffffff"]
        });

        let live = with_source_fill_placement(&node, &params);

        assert_eq!(live["scaleX"], json!(-0.75));
        assert_eq!(live["scaleY"], json!(1.25));
        assert_eq!(live["offsetX"], json!(12.0));
        assert_eq!(live["offsetY"], json!(-8.0));
        assert_eq!(live["sourceWidth"], json!(400.0));
        assert_eq!(live["sourceHeight"], json!(200.0));
        assert!((live["rotation"].as_f64().unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(live["colors"], params["colors"]);
    }
}
