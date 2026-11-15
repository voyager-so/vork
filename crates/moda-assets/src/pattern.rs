//! Pattern (image-fill) instance packing — port of the pattern branch of
//! `WebGPUPathRenderer.preparePathInstances` (`renderers/webgpu/rendering/
//! systems/shapes/WebGPUPathRenderer.ts:1232-1505`) against the extracted
//! `path-pattern.wgsl` `PathPatternData` storage struct (416 bytes).
//!
//! Placeholder semantics (the export bakes these on timeout — DECISION §1.3,
//! render-semantics §6.7):
//! - loading/timed-out → 1x1 grey (0.9³) bound, FORCE-TILED (repeat mode 0),
//!   adjustments/color-change/overlay/perspective disabled;
//! - failed → identical treatment with the 1x1 rose placeholder.
//!
//! Perspective (`_webgpuPerspectivePattern`) packing belongs to M3b (PLAN
//! M3b); this packer emits the disabled identity rows.

use bytemuck::{Pod, Zeroable};
use moda_scene::types::RawNode;
use moda_scene::value::Value;

use crate::constants::{BLEND_MODE_DISABLED, DEFAULT_OVERLAY_OPACITY, DEFAULT_PATTERN_BOUNDS};
use crate::AssetStore;

/// Presentation state of an image fill (`WebGPUTextureManager.ts:27` /
/// `getImageFillState`, lines 445-459).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFillState {
    Loading,
    Loaded,
    Failed,
}

/// The `PathPatternData` WGSL storage struct, byte-for-byte
/// (`wgsl/path-pattern.wgsl`; manifest module `path-pattern`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct PatternInstance {
    pub model_matrix: [f32; 16],
    /// White tint; only alpha (node opacity) matters
    /// (`WebGPUPathRenderer.ts:1433`).
    pub fill_color: [f32; 4],
    pub stroke_color: [f32; 4],
    /// [vertexOffset, vertexCount, strokeWidth || 1, depth]
    pub path_info: [f32; 4],
    /// [repeatMode, scaleX, scaleY, rotationRad]
    pub pattern_data: [f32; 4],
    pub texture_dims: [f32; 2],
    pub pattern_offset: [f32; 2],
    /// [0, 0, width, height] — shape bounds (UV math does not read it).
    pub bounds: [f32; 4],
    pub page_clip_bounds: [f32; 4],
    /// Image adjustments [exposure, contrast, saturation, temperature].
    pub adjust_a: [f32; 4],
    pub overlay_color: [f32; 4],
    /// [opacity, blendMode (BLEND_MODE_DISABLED = off)]
    pub overlay_settings: [f32; 2],
    /// Image adjustments [tint, highlights].
    pub adjust_b: [f32; 2],
    pub color_change_source: [[f32; 4]; 4],
    pub color_change_target: [[f32; 4]; 4],
    /// [count, tolerance, smoothness]
    pub color_change_params: [f32; 3],
    /// Image adjustment: shadows.
    pub adjust_shadows: f32,
    /// xyz = inverse homography rows, row0.w = enabled flag.
    pub perspective_row0: [f32; 4],
    pub perspective_row1: [f32; 4],
    pub perspective_row2: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<PatternInstance>() == 416);

/// Everything the engine must carry to draw one pattern body.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternPaint {
    pub instance: PatternInstance,
    /// Texture lookup key (None → 1x1 white fallback).
    pub src: Option<String>,
    pub state: ImageFillState,
    /// `fillPatternRepeat === 'no-repeat'` → clamp sampler
    /// (`WebGPUPathRenderer.ts:2277, 2293`).
    pub clamp_sampler: bool,
}

/// `encodePatternRepeatMode` (`WebGPUPathRenderer.ts:194-209`).
pub fn encode_repeat_mode(repeat: Option<&str>, force_repeat: bool) -> f32 {
    if force_repeat {
        return 0.0;
    }
    match repeat {
        Some("repeat-x") => 0.25,
        Some("repeat-y") => 0.5,
        Some("no-repeat") => 0.75,
        _ => 0.0,
    }
}

/// `getBlendModeValue` (`renderers/webgpu/constants/blendModes.ts:49-62`).
pub fn blend_mode_value(mode: Option<&str>) -> f32 {
    match mode {
        Some("multiply") => 1.0,
        Some("screen") => 2.0,
        Some("overlay") => 3.0,
        _ => 0.0,
    }
}

/// Geometry/transform context computed by the engine (it owns tessellation,
/// depth, transforms and color parsing — moda-assets stays engine-free).
#[derive(Debug, Clone, Copy)]
pub struct PatternGeometry {
    pub model_matrix: [f32; 16],
    pub depth: f32,
    pub vertex_offset: f32,
    pub vertex_count: f32,
    /// Tessellated bounds extents (fill ∪ stroke), node-local.
    pub bounds_size: Option<(f32, f32)>,
    pub page_clip_bounds: [f32; 4],
    /// Parsed stroke rgba × node opacity; the caller passes transparent
    /// black for visible gradient strokes (`WebGPUPathRenderer.ts:1268-1272`).
    pub stroke_color: [f32; 4],
}

/// JS `x || fallback` on a number (0/NaN → fallback).
fn js_or(v: Option<f64>, fallback: f64) -> f64 {
    match v {
        Some(x) if x != 0.0 && x.is_finite() => x,
        _ => fallback,
    }
}

/// The node's top image fill (`moda_scene::paint::top_fill`), or an empty
/// image (the white placeholder) when the top fill is anything else.
fn image_fill(node: &RawNode) -> moda_scene::paint::ImagePaint {
    match moda_scene::paint::top_fill(node).map(|f| f.paint) {
        Some(moda_scene::paint::Paint::Image(p)) => p,
        _ => moda_scene::paint::ImagePaint {
            src: None,
            placement: Default::default(),
            adjust: None,
        },
    }
}

/// Build the pattern paint for a node whose top fill is an image.
/// `parse_color` is the production color parser (straight rgba 0-1), supplied
/// by the engine so the quirky parsing lives in exactly one place.
pub fn build_pattern_paint(
    node: &RawNode,
    store: &AssetStore,
    geom: &PatternGeometry,
    parse_color: &dyn Fn(&str) -> Option<[f32; 4]>,
) -> PatternPaint {
    let image = image_fill(node);
    let placement = &image.placement;
    let adjust = image.adjust.clone().unwrap_or_default();
    let src = image.src.clone().filter(|s| !s.is_empty());

    // `getImageFillState` is 'loaded' for a srcless pattern too
    // (`WebGPUPathRenderer.ts:1242-1246`).
    let state = match &src {
        Some(s) => store.fill_state(s),
        None => ImageFillState::Loaded,
    };
    let is_placeholder = state != ImageFillState::Loaded;

    let opacity = node.f64("opacity").unwrap_or(1.0) as f32;

    // Texture dims (1x1 unless a real texture is cached) + SVG-upscale
    // compensation (`WebGPUPathRenderer.ts:1258-1264, 1338-1345`).
    let mut texture_dims = [1.0f32, 1.0];
    let mut scale_factor = (1.0, 1.0);
    let mut natural = None;
    if let Some(s) = &src {
        if let Some(sf) = store.scale_factor(s) {
            scale_factor = sf;
        }
        if state == ImageFillState::Loaded {
            if let Some((w, h)) = store.dimensions(s) {
                texture_dims = [w as f32, h as f32];
                natural = Some((w as f64 / scale_factor.0, h as f64 / scale_factor.1));
            }
        }
    }
    // The scale mode resolved against the box (Fill / Fit re-fit live).
    let image_box = moda_scene::paint::node_image_box(node).unwrap_or_else(|| {
        let (bw, bh) = geom
            .bounds_size
            .unwrap_or((DEFAULT_PATTERN_BOUNDS[0], DEFAULT_PATTERN_BOUNDS[1]));
        [0.0, 0.0, bw as f64, bh as f64]
    });
    let resolved = placement.resolve(image_box, natural);
    let pattern_scale_x = js_or(Some(resolved.scale.0), 1.0) / scale_factor.0;
    let pattern_scale_y = js_or(Some(resolved.scale.1), 1.0) / scale_factor.1;

    // Image adjustments (E7.18; moda-engine `paint::pattern`). A placeholder
    // draws unadjusted.
    let adj = |v: Option<f64>| {
        if is_placeholder {
            0.0
        } else {
            v.filter(|v| v.is_finite()).unwrap_or(0.0).clamp(-1.0, 1.0) as f32
        }
    };
    let adjust_a = [
        adj(adjust.exposure),
        adj(adjust.contrast),
        adj(adjust.saturation),
        adj(adjust.temperature),
    ];
    let adjust_b = [adj(adjust.tint), adj(adjust.highlights)];
    let adjust_shadows = adj(adjust.shadows);

    // Overlay (`WebGPUPathRenderer.ts:1372-1382, 1450-1455`).
    let overlay = adjust.overlay.clone().unwrap_or_default();
    let mut overlay_color = [1.0f32, 0.0, 0.0, 1.0];
    if let Some(c) = overlay
        .color
        .as_ref()
        .and_then(Value::as_str)
        .and_then(parse_color)
    {
        overlay_color = c;
    }
    let overlay_opacity = overlay
        .opacity
        .map(|v| v as f32)
        .unwrap_or(DEFAULT_OVERLAY_OPACITY);
    let overlay_blend = if !is_placeholder && overlay.enabled == Some(true) {
        blend_mode_value(overlay.blend_mode.as_deref())
    } else {
        BLEND_MODE_DISABLED
    };

    // Color change (`WebGPUPathRenderer.ts:1384-1420`). The source/target
    // slots are parsed from whatever exists in the arrays even when the
    // feature is disabled (colorCount 0) — copy that.
    let recolor = adjust.recolor.clone().unwrap_or_default();
    let extracted = recolor.from.clone().unwrap_or_default();
    let replacements = recolor.to.clone().unwrap_or_default();
    let cc_enabled = !is_placeholder && recolor.enabled == Some(true);
    let color_count = if cc_enabled {
        extracted.len().min(replacements.len()).min(4)
    } else {
        0
    };
    let mut cc_src = [[0.0f32, 0.0, 0.0, 1.0]; 4];
    let mut cc_dst = [[0.0f32, 0.0, 0.0, 1.0]; 4];
    for i in 0..4 {
        if let Some(c) = extracted.get(i).and_then(|s| parse_color(s)) {
            cc_src[i] = c;
        }
        if let Some(c) = replacements.get(i).and_then(|s| parse_color(s)) {
            cc_dst[i] = c;
        }
    }
    let tolerance = recolor.tolerance.unwrap_or(0.3).clamp(0.0, 1.0) as f32;
    let smoothness = recolor.smoothness.unwrap_or(0.6).clamp(0.0, 1.0) as f32;

    let (bw, bh) = geom
        .bounds_size
        .unwrap_or((DEFAULT_PATTERN_BOUNDS[0], DEFAULT_PATTERN_BOUNDS[1]));

    let stroke_width = js_or(moda_scene::paint::node_stroke_style(node).width, 1.0) as f32;
    let rotation_rad = resolved.rotation.to_radians() as f32;

    let instance = PatternInstance {
        model_matrix: geom.model_matrix,
        fill_color: [1.0, 1.0, 1.0, opacity],
        stroke_color: geom.stroke_color,
        path_info: [
            geom.vertex_offset,
            geom.vertex_count,
            stroke_width,
            geom.depth,
        ],
        pattern_data: [
            encode_repeat_mode(Some(resolved.repeat), is_placeholder),
            pattern_scale_x as f32,
            pattern_scale_y as f32,
            rotation_rad,
        ],
        texture_dims,
        pattern_offset: [resolved.offset.0 as f32, resolved.offset.1 as f32],
        bounds: [0.0, 0.0, bw, bh],
        page_clip_bounds: geom.page_clip_bounds,
        adjust_a,
        overlay_color,
        overlay_settings: [overlay_opacity, overlay_blend],
        adjust_b,
        color_change_source: cc_src,
        color_change_target: cc_dst,
        color_change_params: [color_count as f32, tolerance, smoothness],
        adjust_shadows,
        // Disabled identity rows (placeholders always disable perspective;
        // real perspective packing is M3b — `WebGPUPathRenderer.ts:1422-1429`).
        perspective_row0: [1.0, 0.0, 0.0, 0.0],
        perspective_row1: [0.0, 1.0, 0.0, 0.0],
        perspective_row2: [0.0, 0.0, 1.0, 0.0],
    };

    PatternPaint {
        instance,
        src,
        state,
        clamp_sampler: resolved.repeat == "no-repeat",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::PreparedImage;
    use crate::{AssetEntry, AssetStore};
    use std::sync::Arc;

    fn node(json: moda_scene::value::Value) -> RawNode {
        RawNode::from_value(json).unwrap()
    }

    fn geom() -> PatternGeometry {
        PatternGeometry {
            model_matrix: [1.0; 16],
            depth: 0.5,
            vertex_offset: 0.0,
            vertex_count: 6.0,
            bounds_size: Some((300.0, 220.0)),
            page_clip_bounds: [0.0, 0.0, 1280.0, 800.0],
            stroke_color: [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn parse(_: &str) -> Option<[f32; 4]> {
        Some([0.25, 0.5, 0.75, 1.0])
    }

    fn store_with(src: &str, entry: AssetEntry) -> AssetStore {
        let mut s = AssetStore::default();
        s.insert(src.to_string(), entry);
        s
    }

    fn prepared(w: u32, h: u32, natural_w: u32, natural_h: u32) -> AssetEntry {
        AssetEntry::Loaded(Arc::new(PreparedImage {
            width: w,
            height: h,
            natural_width: natural_w,
            natural_height: natural_h,
            scale_factor: (w as f64 / natural_w as f64, h as f64 / natural_h as f64),
            rgba_premul: Arc::new(vec![0; (w * h * 4) as usize]),
            is_svg: false,
            icc_applied: false,
            coverage_permille: 1000,
            straight: false,
            blank_cause: "",
        }))
    }

    #[test]
    fn loaded_pattern_packs_dims_and_compensated_scale() {
        let store = store_with("data:x", prepared(512, 342, 96, 64));
        let n = node(moda_scene::value::json!({
            "id": "n1", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:x", "placement": {"fit": "crop"}}}],
            "opacity": 0.8,
        }));
        let p = build_pattern_paint(&n, &store, &geom(), &parse);
        assert_eq!(p.state, ImageFillState::Loaded);
        assert!(p.clamp_sampler);
        assert_eq!(p.instance.texture_dims, [512.0, 342.0]);
        // patternScale 1 / scaleFactor (512/96, 342/64)
        let exp_x = 1.0 / (512.0 / 96.0);
        let exp_y = 1.0 / (342.0 / 64.0);
        assert!((p.instance.pattern_data[1] - exp_x as f32).abs() < 1e-7);
        assert!((p.instance.pattern_data[2] - exp_y as f32).abs() < 1e-7);
        // no-repeat encodes 0.75
        assert_eq!(p.instance.pattern_data[0], 0.75);
        assert_eq!(p.instance.fill_color, [1.0, 1.0, 1.0, 0.8]);
        // Adjustments off, overlay disabled, colorChange count 0, perspective off
        assert_eq!(p.instance.adjust_a, [0.0; 4]);
        assert_eq!(p.instance.overlay_settings[1], BLEND_MODE_DISABLED);
        assert_eq!(p.instance.color_change_params[0], 0.0);
        assert_eq!(p.instance.perspective_row0[3], 0.0);
    }

    #[test]
    fn placeholder_forces_tiling_and_disables_filters() {
        let store = AssetStore::default();
        let n = node(moda_scene::value::json!({
            "id": "n1", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://x/img.png", "placement": {"repeat": "no-repeat"}, "adjust": {"exposure": 0.5, "shadows": 0.2, "overlay": {"enabled": true, "blendMode": "multiply"}, "recolor": {"enabled": true, "from": ["#ff0000"], "to": ["#00ff00"]}}}}],
        }));
        let p = build_pattern_paint(&n, &store, &geom(), &parse);
        assert_eq!(p.state, ImageFillState::Loading); // timeout = still loading
                                                      // force-tiled (repeat 0) regardless of no-repeat
        assert_eq!(p.instance.pattern_data[0], 0.0);
        // texture dims stay 1x1
        assert_eq!(p.instance.texture_dims, [1.0, 1.0]);
        // Adjustments suppressed
        assert_eq!(p.instance.adjust_a, [0.0; 4]);
        assert_eq!(p.instance.adjust_shadows, 0.0);
        // overlay disabled
        assert_eq!(p.instance.overlay_settings[1], BLEND_MODE_DISABLED);
        // color change suppressed (but slots still parsed — copy the loop)
        assert_eq!(p.instance.color_change_params[0], 0.0);
        assert_eq!(p.instance.color_change_source[0], [0.25, 0.5, 0.75, 1.0]);
        // sampler still follows the node's repeat mode
        assert!(p.clamp_sampler);
    }

    #[test]
    fn failed_state_maps_to_failed() {
        let store = store_with("https://x/dead.png", AssetEntry::Failed);
        let n = node(moda_scene::value::json!({
            "id": "n1", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "https://x/dead.png", "placement": {}}}],
        }));
        let p = build_pattern_paint(&n, &store, &geom(), &parse);
        assert_eq!(p.state, ImageFillState::Failed);
        assert_eq!(p.instance.pattern_data[0], 0.0); // force-tiled
    }

    #[test]
    fn zero_scale_falls_back_to_one_js_or() {
        let store = AssetStore::default();
        let n = node(moda_scene::value::json!({
            "id": "n1", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "placement": {"fit": "tile", "scale": {"x": 0, "y": 2}}}}],
        }));
        let p = build_pattern_paint(&n, &store, &geom(), &parse);
        // srcless pattern: state 'loaded', white fallback
        assert_eq!(p.state, ImageFillState::Loaded);
        assert!(p.src.is_none());
        assert_eq!(p.instance.pattern_data[1], 1.0); // 0 || 1
        assert_eq!(p.instance.pattern_data[2], 2.0);
    }

    #[test]
    fn hsv_and_overlay_pack_when_loaded() {
        let store = store_with("data:x", prepared(16, 16, 16, 16));
        let n = node(moda_scene::value::json!({
            "id": "n1", "type": "rectangle",
            "fills": [{"id": "fill-0", "paint": {"type": "image", "src": "data:x", "placement": {}, "adjust": {"saturation": 1.5, "tint": -0.25, "overlay": {"enabled": true, "opacity": 0.4, "blendMode": "screen"}, "recolor": {"enabled": true, "from": ["#a", "#b", "#c", "#d", "#e"], "to": ["#a", "#b"]}}}}],
        }));
        let p = build_pattern_paint(&n, &store, &geom(), &parse);
        // Adjustments clamp to −1..1.
        assert_eq!(p.instance.adjust_a, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.instance.adjust_b, [-0.25, 0.0]);
        assert_eq!(p.instance.overlay_settings, [0.4, 2.0]);
        // colorCount = min(4, min(5, 2)) = 2
        assert_eq!(p.instance.color_change_params[0], 2.0);
        assert_eq!(p.instance.color_change_params[1], 0.3);
        assert_eq!(p.instance.color_change_params[2], 0.6);
    }

    #[test]
    fn instance_is_416_bytes() {
        assert_eq!(core::mem::size_of::<PatternInstance>(), 416);
        assert_eq!(core::mem::offset_of!(PatternInstance, texture_dims), 128);
        assert_eq!(core::mem::offset_of!(PatternInstance, bounds), 144);
        assert_eq!(
            core::mem::offset_of!(PatternInstance, color_change_params),
            352
        );
        assert_eq!(
            core::mem::offset_of!(PatternInstance, perspective_row0),
            368
        );
    }
}
