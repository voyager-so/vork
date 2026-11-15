//! effects/shadow.rs — drop + inner (inset) shadow instance packing. M3d.
//!
//! The shadows are the node's `drop-shadow` / `inner-shadow` entries of
//! `effects` (`moda_scene::effect`). One quad per entry that draws
//! (`effect::drop_shadows` / `effect::inner_shadows`), in list order (back to
//! front: `effects[0]` is emitted first, so it draws furthest behind).
//!
//! Faithful port of:
//! - `renderers/webgpu/utils/WebGPUShadowUtils.ts` (`hasShadow`,
//!   `extractShadowParameters`, `hasInnerShadow` → `utils/shadowHelpers.ts
//!   hasInnerShadowEffect/hasInnerShadowConfig`, `extractInnerShadowParameters`)
//! - `renderers/webgpu/rendering/systems/shapes/WebGPUPathShadowRenderer.ts`
//!   (`prepareShadowInstanceData`, lines 92-171)
//! - `renderers/webgpu/rendering/systems/shapes/WebGPUPathRenderer.ts`
//!   (`getPathDrawMeta` rectMetadata gate :586-600, `buildInnerShadowStub`
//!   + `INNER_SHADOW_Z_OFFSET` :157-184, :727-744)
//!
//! This SDF lane covers rect/ellipse-derived silhouettes only
//! (`convertEllipseToPathNode` only attaches metadata for full sweeps,
//! `shapeToPathNodeUtils.ts:262-268`; distorted rects have none). Every other
//! shape — paths (vector networks, booleans, outlined text), stars, polygons,
//! lines, partial-sweep / inner-radius ellipses, distorted nodes, and rich
//! text's inner shadows — draws its shadows from a coverage capture of its
//! silhouette instead (`layer_blur::shadow`), as Figma does for every layer.
//!
//! COPY THE BUGS (quirk ledger Q-05 / Q-15):
//! - The falloff and knockout live in `path-shadow.wgsl` — nothing to port
//!   on the CPU side beyond instance packing. The falloff is no longer
//!   production's: it is a Gaussian of σ = blur / 2 (CSS / Figma), see
//!   docs/qa/effects-parity.md. `inner_params.y` also carries a DROP
//!   shadow's spread: the silhouette grows (shrinks, negative) by it before
//!   the blur, a rounded corner's radius with it and a square corner stays
//!   square (CSS / Figma), all in the shader. Only this lane applies spread
//!   (rectangles, ellipses, filled frames), as Figma does.
//! - `cornerRadii` are the RAW node values via `normalizeCornerRadii`
//!   (array entries `|| 0`, scalar splat) — NO min(w,h)/2 clamp and NO CSS
//!   pairwise scaling (those belong to other code paths).
//! - Packed alpha = the parsed shadow color's own alpha × the effect's
//!   `opacity × node.opacity`. (Production discarded the colour's alpha,
//!   `WebGPUPathShadowRenderer.ts:162`, so `shadow="0 8px 24px
//!   rgba(0,0,0,0.25)"` drew a solid black shadow; not copied.)
//! - Color parse is `colorToNormalizedRgba` (hex / integer rgb()/rgba()
//!   ONLY) falling back to DEFAULT_STROKE_COLOR black — so the common
//!   `color: "black"` literally renders via the parse-failure default.

use moda_scene::effect::{DropShadow, InnerShadow};
use moda_scene::types::RawNode;
use moda_scene::Scene;
use moda_scene::value::Value;

use crate::color;
use crate::seams::ShadowInstance;

/// `PATH_RENDERER_DEFAULTS.DEFAULT_STROKE_COLOR` (`WebGPUConstants.ts:150`).
const DEFAULT_SHADOW_COLOR: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// Rect/ellipse silhouette metadata — the native equivalent of
/// `RectanglePathMetadata` from `getPathRectangleMetadata`.
#[derive(Debug, Clone)]
struct Silhouette {
    width: f64,
    height: f64,
    /// [tl, tr, br, bl]; ellipse sentinel [rx, ry, 0, -1].
    corner_radii: [f64; 4],
    is_ellipse: bool,
}

/// `WebGPUShapeUtils.normalizeCornerRadii` (`WebGPUShapeUtils.ts:` —
/// array → `[a||0, b||0, c||0, d||0]`, scalar → splat, absent → zeros).
fn normalize_corner_radii(corner_radius: Option<&Value>) -> [f64; 4] {
    match corner_radius {
        Some(Value::Array(arr)) => {
            let g = |i: usize| arr.get(i).and_then(Value::as_f64).unwrap_or(0.0);
            [g(0), g(1), g(2), g(3)]
        }
        Some(Value::Number(n)) => {
            let r = n.as_f64().unwrap_or(0.0);
            [r, r, r, r]
        }
        _ => [0.0; 4],
    }
}

/// Which nodes carry a rect/ellipse shadow silhouette in the export path.
/// rectangle/container → rect metadata (`convertRectangleToPathNode` +
/// `roundedRectSurfaceToRenderRectangle`); ellipse → sentinel metadata only
/// for full sweep with no inner radius. A valid `distort` warp disables
/// shadows.
fn silhouette_of(node: &RawNode) -> Option<Silhouette> {
    // Distort-warped nodes skip the SDF shadow: the silhouette
    // ABI is an axis-aligned rounded-rect/ellipse and cannot represent the
    // warped outline, so drawing it would put an unwarped shadow behind
    // warped geometry. The flat-2D "sticker" shadow of the warped silhouette
    // is a follow-up (needs a tessellated-geometry shadow path, which the
    // engine does not have — stars/polygons already render with no shadow,
    // Q-05/Q-15).
    if crate::paint::node_distort(node).is_some() {
        return None;
    }
    // One fill's or stroke's own shadows (E8.2) are captures of that layer's silhouette, never the box.
    if node.json_field(moda_scene::appearance::ENTRY_PROBE_KEY).is_some() {
        return None;
    }
    // A frame without a fill casts its content's shadow, from a capture of
    // its subtree (`layer_blur::shadow::is_subtree_capture_node`), as Figma.
    if crate::layer_blur::shadow::is_subtree_capture_node(node) {
        return None;
    }
    // An image fill's shape is its pixels' alpha (a cut-out PNG), not its box:
    // the capture lane shadows what it draws (`layer_blur::shadow::fills_are_images`).
    if crate::layer_blur::shadow::fills_are_images(node) {
        return None;
    }
    let w = node.f64("width").unwrap_or(0.0);
    let h = node.f64("height").unwrap_or(0.0);
    match node.node_type() {
        // `cornerSmoothing` is not in the ABI: a smoothed rect casts the plain
        // rounded rect of the same radii. The squircle shares that shape's 45°
        // point and differs by at most a few percent of R near the tangents,
        // which the blur hides (a geometry-based shadow is the Q-05 follow-up).
        // Styled corners (`cornerStyle` inverted / chamfer) have no rect silhouette: the capture lane shadows the
        // drawn outline (`layer_blur::shadow`).
        "rectangle" if moda_scene::rect_corners::is_styled_rect(node.json_field("cornerRadius"), node.json_field("cornerStyle")) => None,
        "rectangle" | "container" => Some(Silhouette {
            width: w,
            height: h,
            corner_radii: normalize_corner_radii(node.json_field("cornerRadius")),
            is_ellipse: false,
        }),
        "ellipse" => {
            let sweep = node.f64("sweepAngle");
            let inner = node.f64("innerRadiusRatio").unwrap_or(0.0);
            let full_sweep = sweep.is_none() || sweep.is_some_and(|s| s.abs() >= 360.0);
            if full_sweep && inner <= 0.0 {
                Some(Silhouette {
                    width: w,
                    height: h,
                    corner_radii: [w / 2.0, h / 2.0, 0.0, -1.0],
                    is_ellipse: true,
                })
            } else {
                None // partial arc / ring: no metadata, no shadow.
            }
        }
        _ => None,
    }
}

/// Whether the SDF shadow lane draws this node's shadows (a rect / ellipse
/// silhouette). Every other shape's shadows come from its coverage capture
/// (`layer_blur::shadow`).
pub fn has_sdf_silhouette(node: &RawNode) -> bool {
    silhouette_of(node).is_some()
}

/// A shadow's colour as the SDF lane parses it (see [`shadow_rgba`]), so the
/// capture lane tints identically.
pub(crate) fn shadow_color_rgba(color: &Value) -> [f64; 4] {
    shadow_rgba(color_str(color))
}

/// The drop shadows the node's quads draw (`WebGPUShadowUtils.hasShadow`),
/// back to front: every `drop-shadow` entry with a colour and a blur or an
/// offset (`effect::drop_shadows`). Groups draw none.
fn drop_shadows_of(node: &RawNode) -> Vec<DropShadow> {
    if node.node_type() == "group" {
        return Vec::new();
    }
    moda_scene::effect::drop_shadows(node)
}

/// The inner shadows the node's quads draw (`hasInnerShadowEffect`,
/// `utils/shadowHelpers.ts`), back to front: every `inner-shadow` entry with
/// `opacity !== 0`, a colour, and a blur, a spread or an offset
/// (`effect::inner_shadows`).
fn inner_shadows_of(node: &RawNode) -> Vec<InnerShadow> {
    moda_scene::effect::inner_shadows(node)
}

/// A shadow's colour as a CSS string (an unresolved variable reference is
/// not one: it takes the parse-failure default, like any unparseable colour).
fn color_str(color: &Value) -> Option<&str> {
    color.as_str()
}

/// `colorToNormalizedRgba(color) || DEFAULT_STROKE_COLOR` — note this is the
/// bare helper, NOT `parseColor` (no 'transparent' special case, no
/// lowercase retry). `color::parse_color` adds those two behaviors, so undo
/// them here: 'transparent' must FAIL to black like production.
fn shadow_rgba(color: Option<&str>) -> [f64; 4] {
    let Some(c) = color else {
        return DEFAULT_SHADOW_COLOR;
    };
    if c.trim().eq_ignore_ascii_case("transparent") {
        return DEFAULT_SHADOW_COLOR; // colorToNormalizedRgba(null) path
    }
    match color::parse_color(Some(c)) {
        Some(rgba) => [rgba.r, rgba.g, rgba.b, rgba.a],
        None => DEFAULT_SHADOW_COLOR,
    }
}

/// `affineToMatrix4x4` (`core/scene/affineTransform.ts:123-125`).
fn affine_to_matrix(a: &moda_scene::types::Affine) -> [f32; 16] {
    [
        a.a as f32, a.b as f32, 0.0, 0.0, //
        a.c as f32, a.d as f32, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        a.e as f32, a.f as f32, 0.0, 1.0,
    ]
}

/// Post-multiply a local translation: `M × T(dx, dy)`
/// (`WebGPUMath.multiply4x4(modelMatrix, affineToMatrix4x4(translateAffine))`;
/// f32 operands, f64 products — only the translation column changes).
fn translate_local(m: &mut [f32; 16], dx: f64, dy: f64) {
    let col =
        |r: usize| -> f32 { (m[r] as f64 * dx + m[4 + r] as f64 * dy + m[12 + r] as f64) as f32 };
    let (t0, t1, t2, t3) = (col(0), col(1), col(2), col(3));
    m[12] = t0;
    m[13] = t1;
    m[14] = t2;
    m[15] = t3;
}

/// Model matrix (canvas affine incl. page offset + group chain —
/// `computeModelMatrixWithPreview` without selection preview) + node opacity.
fn base_instance(scene: &Scene, node: &RawNode) -> Option<([f32; 16], f64)> {
    // `WebGPUPathShadowRenderer.ts:98`: `node._transformSourceId || node.id` —
    // synthetic proxies shadow at their SOURCE node's transform.
    let affine = scene.node_canvas_affine(moda_scene::scene::paint_transform_id(node))?;
    Some((
        affine_to_matrix(&affine),
        node.f64("opacity").unwrap_or(1.0),
    ))
}

/// Build the drop-shadow instances of a rect/ellipse-derived node, one per
/// drop shadow that draws, back to front (empty when the node has none or no
/// eligible silhouette). `z` is the body's integer render z; `page_clip` is
/// the PLAIN page clip. `opaque_sdf_covers_shadow` (the opaque body occludes
/// its own silhouette, so the knockout is skipped) holds for every shadow
/// alike; `showBehindNode` is per shadow.
pub fn drop_shadow_instances(
    scene: &Scene,
    node: &RawNode,
    z: f64,
    page_clip: [f32; 4],
    opaque_sdf_covers_shadow: bool,
) -> Vec<Box<ShadowInstance>> {
    let Some(sil) = silhouette_of(node) else {
        return Vec::new();
    };
    drop_shadows_of(node)
        .iter()
        .filter_map(|shadow| {
            drop_shadow_instance(scene, node, &sil, shadow, z, page_clip, opaque_sdf_covers_shadow)
        })
        .collect()
}

/// One drop shadow's instance.
fn drop_shadow_instance(
    scene: &Scene,
    node: &RawNode,
    sil: &Silhouette,
    shadow: &DropShadow,
    z: f64,
    page_clip: [f32; 4],
    opaque_sdf_covers_shadow: bool,
) -> Option<Box<ShadowInstance>> {
    let depth = crate::depth::linear_depth(z) as f32;
    let (mut model, node_opacity) = base_instance(scene, node)?;

    // extractShadowParameters (WebGPUShadowUtils.ts:96-111).
    let blur = shadow.blur;
    let (ox, oy) = (shadow.offset.x, shadow.offset.y);
    let opacity = shadow.opacity();
    let spread = shadow.spread();
    let show_behind = shadow.shows_behind_node();
    let rgba = shadow_rgba(color_str(&shadow.color));

    // Drop shadows expand the silhouette to the stroke's OUTER edge
    // (prepareShadowInstanceData :128-158) — of the stroke that is PAINTED.
    // A `strokeWidth` with no stroke paint route emits no band, so
    // it must not grow the shadow either, or a halo protrudes around the
    // fill on every tessellated body (pattern/shader fills, masked, dashed).
    let stroke_width = crate::paint::painted_stroke_width(node);
    let half_stroke = stroke_width * 0.5;
    let shadow_w = sil.width + stroke_width;
    let shadow_h = sil.height + stroke_width;
    let mut radii = sil.corner_radii;
    if sil.is_ellipse {
        radii[0] += half_stroke;
        radii[1] += half_stroke;
        // Center-local quad mapping: translate local origin to the center.
        translate_local(&mut model, sil.width / 2.0, sil.height / 2.0);
    } else if stroke_width > 0.0 {
        translate_local(&mut model, -half_stroke, -half_stroke);
    }

    Some(Box::new(ShadowInstance {
        model_matrix: model,
        shadow_color: [
            rgba[0] as f32,
            rgba[1] as f32,
            rgba[2] as f32,
            (rgba[3] * opacity * node_opacity) as f32,
        ],
        shape_data: [shadow_w as f32, shadow_h as f32, blur as f32, depth],
        corner_radii: radii.map(|r| r as f32),
        shadow_offset: [
            ox as f32,
            oy as f32,
            if show_behind { 1.0 } else { 0.0 },
            if opaque_sdf_covers_shadow { 1.0 } else { 0.0 },
        ],
        page_clip_bounds: page_clip,
        // Drop shadow (x = 0); y = the spread, applied in the shader around
        // the stroke's outer edge (the silhouette above).
        inner_params: [0.0, spread as f32, 0.0, 0.0],
    }))
}

/// How many inner-shadow quads the node gets (0 without an eligible
/// silhouette). Any makes the body transparent-phase (`getPathDrawMeta`,
/// `WebGPUPathRenderer.ts:649-660`); more than one keeps the body off the
/// SDF lane, which bakes exactly one inner shadow into its instance.
pub fn inner_shadow_quad_count(node: &RawNode) -> usize {
    if silhouette_of(node).is_none() {
        return 0;
    }
    inner_shadows_of(node).len()
}

/// Whether the node will get an inner-shadow quad (used by the transparency
/// classification: a body with an inner shadow is ALWAYS transparent-phase —
/// `getPathDrawMeta`, `WebGPUPathRenderer.ts:649-660`).
pub fn has_inner_shadow_quad(node: &RawNode) -> bool {
    inner_shadow_quad_count(node) > 0
}

/// Whether the node will get a drop-shadow quad. Drop shadows do NOT make
/// the body transparent by themselves (render-semantics §1.5).
pub fn has_drop_shadow_quad(node: &RawNode) -> bool {
    !drop_shadow_quads(node).is_empty()
}

/// The drop shadows the node's quads draw, back to front (the layer-blur
/// planner pads its region by the widest reach).
pub fn drop_shadow_quads(node: &RawNode) -> Vec<DropShadow> {
    if silhouette_of(node).is_none() {
        return Vec::new();
    }
    drop_shadows_of(node)
}

/// Build the inner (inset) shadow instances, one per inner shadow that draws,
/// back to front, at `z = body z + INNER_SHADOW_Z_OFFSET` (caller passes the
/// bumped z). Inner shadows ignore the stroke (silhouette = fill geometry)
/// and apply their offset in-SDF on the opposite side (CSS inset convention)
/// — both handled by the extracted WGSL via innerParams.x = 1.
pub fn inner_shadow_instances(
    scene: &Scene,
    node: &RawNode,
    z: f64,
    page_clip: [f32; 4],
) -> Vec<Box<ShadowInstance>> {
    let Some(sil) = silhouette_of(node) else {
        return Vec::new();
    };
    inner_shadows_of(node)
        .iter()
        .filter_map(|shadow| inner_shadow_instance(scene, node, &sil, shadow, z, page_clip))
        .collect()
}

/// One inner shadow's instance.
fn inner_shadow_instance(
    scene: &Scene,
    node: &RawNode,
    sil: &Silhouette,
    shadow: &InnerShadow,
    z: f64,
    page_clip: [f32; 4],
) -> Option<Box<ShadowInstance>> {
    let depth = crate::depth::linear_depth(z) as f32;
    let (mut model, node_opacity) = base_instance(scene, node)?;

    // extractInnerShadowParameters. `blur` and `offset` are always stored;
    // a missing `opacity` is 1 and a missing `spread` 0 (effect.ts).
    let blur = shadow.blur;
    let (ox, oy) = (shadow.offset.x, shadow.offset.y);
    let opacity = shadow.opacity();
    let spread = shadow.spread();
    let rgba = shadow_rgba(color_str(&shadow.color));

    // silhouetteStroke = 0 for inner shadows; ellipse still center-maps.
    if sil.is_ellipse {
        translate_local(&mut model, sil.width / 2.0, sil.height / 2.0);
    }

    Some(Box::new(ShadowInstance {
        model_matrix: model,
        shadow_color: [
            rgba[0] as f32,
            rgba[1] as f32,
            rgba[2] as f32,
            (rgba[3] * opacity * node_opacity) as f32,
        ],
        shape_data: [sil.width as f32, sil.height as f32, blur as f32, depth],
        corner_radii: sil.corner_radii.map(|r| r as f32),
        // Offset applied in-SDF (vertex shader skips the world shift when
        // innerParams.x > 0.5); show-behind slot stays 0.
        shadow_offset: [ox as f32, oy as f32, 0.0, 0.0],
        page_clip_bounds: page_clip,
        inner_params: [1.0, spread as f32, 0.0, 0.0],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(json: moda_scene::value::Value) -> RawNode {
        RawNode::from_value(json).unwrap()
    }

    #[test]
    fn corner_radii_are_raw_and_unclamped() {
        // 100% radii on a 240x160 rect would clamp to 80 on the SDF shape
        // path — the shadow silhouette keeps the RAW values (copy the bug).
        let r = normalize_corner_radii(Some(&moda_scene::value::json!([200, 200, 0, 0])));
        assert_eq!(r, [200.0, 200.0, 0.0, 0.0]);
        // Scalar splat; missing entries -> 0 (NOT the tl/tr shorthand of the
        // tessellation path).
        let r2 = normalize_corner_radii(Some(&moda_scene::value::json!([10, 20])));
        assert_eq!(r2, [10.0, 20.0, 0.0, 0.0]);
    }

    #[test]
    fn shadow_color_alpha_is_kept_and_named_colors_fail_to_black() {
        // "black" is not parseable by colorToNormalizedRgba → default black.
        assert_eq!(shadow_rgba(Some("black")), [0.0, 0.0, 0.0, 1.0]);
        // rgba / #rrggbbaa alpha parses and is kept (the packer multiplies
        // it into the effect's opacity × node.opacity).
        assert_eq!(shadow_rgba(Some("rgba(0, 0, 0, 0.8)")), [0.0, 0.0, 0.0, 0.8]);
        assert_eq!(shadow_rgba(Some("#00000080"))[3], 128.0 / 255.0);
        // 'transparent' must fail to black (parseColor would return
        // transparent — the shadow path uses the bare helper).
        assert_eq!(shadow_rgba(Some("transparent")), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn drop_shadow_silhouette_grows_only_for_a_painted_stroke() {
        // No stroke → no band, so the shadow must hug the fill (it used to
        // be a strokeWidth with no stroke paint; that is no stroke layer now).
        let scene = Scene::load(&moda_scene::value::json!({
            "nodes": [
                {"id":"unpainted","type":"rectangle","x":10.0,"y":20.0,"width":100.0,"height":50.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],"effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000000","blur":12.0}]},
                {"id":"solid","type":"rectangle","x":10.0,"y":20.0,"width":100.0,"height":50.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000000"}, "width": 16}],"effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000000","blur":12.0}]},
                {"id":"gradient","type":"rectangle","x":10.0,"y":20.0,"width":100.0,"height":50.0,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],
                 "strokes": [{"id": "stroke-0", "width": 16, "paint": {"type": "linear", "start": {"x": 0, "y": 0},
                     "end": {"x": 100, "y": 0}, "stops": [{"offset": 0, "color": "#ff0000"}, {"offset": 1, "color": "#0000ff"}]}}],
                 "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000000","blur":12.0}]}
            ],
            "pages": [{"id":"p","width":800.0,"height":600.0,"nodeIds":["unpainted","solid","gradient"]}]
        })).unwrap();
        let shape = |id: &str| {
            let n = scene.effective_node(id).unwrap();
            drop_shadow_instances(&scene, n, 1.0, [0.0; 4], false)[0].shape_data
        };
        assert_eq!(&shape("unpainted")[..2], &[100.0, 50.0]);
        assert_eq!(&shape("solid")[..2], &[116.0, 66.0]);
        // A gradient stroke is a painted stroke too.
        assert_eq!(&shape("gradient")[..2], &[116.0, 66.0]);
    }

    #[test]
    fn eligibility_copies_the_bug() {
        let star = node(
            moda_scene::value::json!({"id":"s","type":"star","width":100.0,"height":100.0,
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"black","blur":20.0}]}),
        );
        assert!(silhouette_of(&star).is_none(), "Q-05: stars get no shadow");
        let wedge = node(
            moda_scene::value::json!({"id":"w","type":"ellipse","width":100.0,"height":100.0,
            "sweepAngle":180.0,"effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"black","blur":20.0}]}),
        );
        assert!(
            silhouette_of(&wedge).is_none(),
            "partial sweep: no metadata"
        );
        let ell = node(moda_scene::value::json!({"id":"e","type":"ellipse","width":100.0,"height":60.0}));
        let sil = silhouette_of(&ell).unwrap();
        assert_eq!(sil.corner_radii, [50.0, 30.0, 0.0, -1.0]);
    }

    #[test]
    fn distort_disables_the_sdf_silhouette() {
        // A valid distort warps the geometry; the axis-aligned SDF
        // silhouette cannot represent it, so the shadow is skipped entirely.
        let warped = node(
            moda_scene::value::json!({"id":"r","type":"rectangle","width":100.0,"height":60.0,
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"black","blur":20.0}],
            "distort":{"type":"matrix","matrix":[1,0,0,0,1,0,0.3,0.1,1]}}),
        );
        assert!(silhouette_of(&warped).is_none());
        // An INVALID distort renders unwarped everywhere, so the silhouette
        // (and its shadow) stays — exactly the pre-existing behavior.
        let invalid = node(
            moda_scene::value::json!({"id":"r","type":"rectangle","width":100.0,"height":60.0,
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"black","blur":20.0}],
            "distort":{"type":"matrix","matrix":[1,0,0,2,0,0,0,0,1]}}),
        );
        assert!(silhouette_of(&invalid).is_some());
    }

    #[test]
    fn has_shadow_requires_blur_or_nonzero_offset() {
        let shadow = |blur: f64, x: f64| {
            node(moda_scene::value::json!({"id":"a","type":"rectangle","width":10.0,"height":10.0,
                "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":blur,"offset":{"x":x,"y":x}}]}))
        };
        assert!(drop_shadows_of(&shadow(0.0, 0.0)).is_empty());
        assert_eq!(drop_shadows_of(&shadow(0.0, 14.0)).len(), 1);
        let group = node(moda_scene::value::json!({"id":"g","type":"group",
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":4.0,"offset":{"x":0,"y":0}}]}));
        assert!(drop_shadows_of(&group).is_empty(), "groups draw no drop shadow");
    }

    #[test]
    fn inner_shadow_defaults_and_opacity_zero_gate() {
        let inner = |extra: moda_scene::value::Value| {
            let mut e = moda_scene::value::json!({"id":"inner-shadow-0","type":"inner-shadow","color":"#000","blur":10.0,"offset":{"x":0,"y":0}});
            e.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            node(moda_scene::value::json!({"id":"a","type":"rectangle","width":10.0,"height":10.0,"effects":[e]}))
        };
        assert!(inner_shadows_of(&inner(moda_scene::value::json!({"opacity": 0.0}))).is_empty(), "explicit opacity 0 disables");
        let dflt = inner_shadows_of(&inner(moda_scene::value::json!({})));
        assert_eq!(dflt.len(), 1, "absent opacity passes (≠ 0)");
        assert_eq!((dflt[0].opacity(), dflt[0].spread()), (1.0, 0.0));
        assert!(inner_shadows_of(&inner(moda_scene::value::json!({"visible": false}))).is_empty(), "hidden");
    }

    #[test]
    fn every_drawing_entry_is_a_quad_in_list_order() {
        // One quad per drop shadow that draws, back to front; a non-drawing
        // entry (no blur, no offset) gets none.
        let n = node(moda_scene::value::json!({"id":"a","type":"rectangle","width":10.0,"height":10.0,"effects":[
            {"id":"a","type":"drop-shadow","color":"#f00","blur":0.0,"offset":{"x":0,"y":0}},
            {"id":"b","type":"drop-shadow","color":"#0f0","blur":3.0,"offset":{"x":0,"y":0}},
            {"id":"c","type":"drop-shadow","color":"#00f","blur":5.0,"offset":{"x":0,"y":0}}
        ]}));
        let colors: Vec<_> = drop_shadow_quads(&n).into_iter().map(|s| s.color).collect();
        assert_eq!(colors, [moda_scene::value::json!("#0f0"), moda_scene::value::json!("#00f")]);
    }

    #[test]
    fn a_drop_shadow_packs_its_spread_and_a_spread_alone_draws() {
        let rect = |spread: moda_scene::value::Value| {
            moda_scene::value::json!({"id":"a","type":"rectangle","x":0,"y":0,"width":200.0,"height":120.0,
                "cornerRadius":16,"effects":[{"id":"d","type":"drop-shadow","color":"#000",
                "blur":0.0,"offset":{"x":0,"y":0},"spread":spread}]})
        };
        // A spread alone draws (a hard ring, as in Figma); absent or 0 does not.
        assert_eq!(drop_shadows_of(&node(rect(moda_scene::value::json!(4.0)))).len(), 1);
        assert_eq!(drop_shadows_of(&node(rect(moda_scene::value::json!(-4.0)))).len(), 1);
        assert!(drop_shadows_of(&node(rect(moda_scene::value::json!(0.0)))).is_empty());

        let json = moda_scene::value::json!({
            "nodes": [rect(moda_scene::value::json!(-4.0))],
            "pages": [{"id": "p1", "name": "p", "width": 400, "height": 300, "nodeIds": ["a"]}],
            "config": {}, "variables": [], "canvasRootNodeIds": [], "layoutMode": "vertical",
            "category": null, "animations": null,
        });
        let scene = Scene::load(moda_scene::extract_canvas_data(&json)).unwrap();
        let n = scene.effective_node("a").unwrap();
        let q = drop_shadow_instances(&scene, n, 1.0, [0.0; 4], false);
        assert_eq!(q.len(), 1);
        // Drop shadow (x = 0) with its spread in y; the raw radii ride along
        // (the shader grows / shrinks them with the spread).
        assert_eq!(q[0].inner_params, [0.0, -4.0, 0.0, 0.0]);
        assert_eq!(q[0].corner_radii, [16.0; 4]);
        assert_eq!(&q[0].shape_data[..2], &[200.0, 120.0]);
    }
}
