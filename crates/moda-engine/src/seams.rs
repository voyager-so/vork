//! seams.rs — the cross-track interface contract of moda-engine.
//!
//! ============================================================================
//! FROZEN AT M1 EXIT (DECISION.md tripwire #5; PLAN.md §0 rule 4).
//!
//! Changes after M1 require a recorded interface-change note and sign-off from
//! all in-flight track owners (M2 tess, M3a text, M3b paint, M3c assets,
//! M3d effects, M3e backdrop, M3f shader runtime). Read SEAMS.md (same dir)
//! for the prose contract; this file is the compilable version of it.
//! ============================================================================
//!
//! Normative sources are cited per type. "Production" = the TS WebGPU renderer
//! under `renderers/webgpu/` at repo root; manifest = the M0a extraction
//! manifest (`moda_shaders::manifest()`), which is the ONLY authority for
//! blend/depth state, packing offsets and pinned constants — never hand-code
//! those (DECISION §1.3, S10).

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Pinned indices / fractional z bumps (asserted against the manifest in tests)
// ---------------------------------------------------------------------------

/// Mask bind group index for path-solid / path-gradient pipelines
/// (`manifest.pinned_constants.mask_bind_group_index.default`).
pub const MASK_BIND_GROUP_INDEX: u32 = 2;
/// Mask bind group index for pattern / image-backed procedural pipelines
/// (`manifest.pinned_constants.mask_bind_group_index.image_backed`).
pub const MASK_BIND_GROUP_INDEX_IMAGE_BACKED: u32 = 3;

/// Fractional z bumps (render-semantics.md §1.4). The render z-counter is
/// integer per leaf; these offsets place secondary items inside a bucket gap.
/// shape/connector text: `WebGPUShapeSystem.renderShapeText(z + 0.5)`.
pub const SHAPE_TEXT_Z_OFFSET: f64 = 0.5;
/// arrowhead paths (`WebGPUShapeSystem.ts:942`).
pub const ARROWHEAD_Z_OFFSET: f64 = 0.25;
/// inner-shadow quads (`WebGPUPathRenderer.ts INNER_SHADOW_Z_OFFSET`).
pub const INNER_SHADOW_Z_OFFSET: f64 = 0.5;

// ---------------------------------------------------------------------------
// GPU instance ABIs (bytemuck-mirrored, size-asserted)
// ---------------------------------------------------------------------------

/// The 656-byte `CustomShaderData` instance ABI used by EVERY procedural-fill
/// pipeline and the vector-text-fill variants (manifest `instance_abi`;
/// production packing `defineProceduralShader.ts:237-292`).
///
/// Field offsets (floats) are extractor-emitted data
/// (`manifest.instance_abi.offsets_floats`) and asserted in tests — if the
/// frontend ever grows this struct the extractor diff + these asserts trip.
///
/// M3b/M3f fill `shader_params`/`color_data` via the per-shader packing tables
/// (`manifest.modules.<id>.packing`); M1 only uses the shared header fields.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PaintInstance {
    /// offset 0 floats — column-major model matrix (`affineToMatrix4x4`,
    /// `core/scene/affineTransform.ts:123-125`).
    pub model_matrix: [f32; 16],
    /// offset 16 floats — fill rgba, straight (non-premultiplied); alpha
    /// already multiplied by node opacity (`WebGPUPathRenderer.ts:1480`).
    pub fill_color: [f32; 4],
    /// offset 20 floats — stroke rgba, straight; alpha × node opacity.
    pub stroke_color: [f32; 4],
    /// offset 24 floats — [vertexOffset, vertexCount, strokeWidth, depth]
    /// (`WebGPUPathRenderer.ts:1482`); depth from `depth::linear_depth`.
    pub path_info: [f32; 4],
    /// offset 28 floats — tessellated geometry bounds [x, y, w, h] in node-
    /// local space, INCLUDING stroke (`WebGPUPathRenderer.ts:1707-1713`).
    pub bounds: [f32; 4],
    /// offset 32 floats — page clip rect [x, y, w, h] in canvas space;
    /// `INFINITE_CLIP` for floating nodes (`renderers/webgpu/utils/
    /// transientOrigin.ts:4-9` — sentinel 9_999_999).
    pub page_clip_bounds: [f32; 4],
    /// offset 36 floats — per-shader params, packed per manifest tables.
    pub shader_params: [f32; 64],
    /// offset 100 floats — per-shader colors, packed per manifest tables.
    pub color_data: [f32; 64],
}

// M4e FIX (SEAMS.md change log): 164 floats / 656 bytes — the stride the
// production storage buffers are sized AND written with
// (`BUFFER_SIZES.CUSTOM_SHADER`, `WebGPUStorageBufferManager.ts:1307-1311`)
// and the WGSL `CustomShaderData` struct stride. The freeze-time 672 came
// from `ShaderBufferLayout.INSTANCE_SIZE` (a dead prod constant); the extra
// `_pad` sheared every instance >= 1 in multi-instance shader buffers.
const _: () = assert!(core::mem::size_of::<PaintInstance>() == 656);

/// The 128-byte `PathData` instance of the path-solid pipeline (WGSL struct in
/// `wgsl/path-solid.wgsl`; packing `WebGPUPathRenderer.ts:1471-1485`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SolidPathInstance {
    pub model_matrix: [f32; 16],
    pub fill_color: [f32; 4],
    pub stroke_color: [f32; 4],
    /// [vertexOffset, vertexCount, strokeWidth(|| 1), depth]
    pub path_info: [f32; 4],
    pub page_clip_bounds: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<SolidPathInstance>() == 128);

/// The 112-byte `PageInstance` of the page-fill/page-stroke pipelines
/// (WGSL struct in `wgsl/page-fill.wgsl`; packing
/// `WebGPUPageSystem.ts:425-447, 460-479`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PageInstance {
    /// fill: [w,0.., 0,h.., .., x,y,0,1]; stroke: identity.
    pub model_matrix: [f32; 16],
    /// rgb from the page fill/stroke, ALPHA ALWAYS 1.0 (copy the bug:
    /// `WebGPUPageSystem.ts:427-430` ignores fill alpha).
    pub color: [f32; 4],
    /// [pageX, pageY, pageWidth, pageHeight] canvas space.
    pub dimensions: [f32; 4],
    /// [strokeWidth, 0, 0, 0] — exports always pass strokeWidth 0.
    pub stroke_data: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<PageInstance>() == 112);

/// The 160-byte `RectangleShadowData` instance of the path-shadow pipeline
/// (WGSL struct in `wgsl/path-shadow.wgsl`; packing
/// `WebGPUPathShadowRenderer.prepareShadowInstanceData`,
/// `WebGPUPathShadowRenderer.ts:92-171`). M3d interface-change note (see
/// SEAMS.md change log): `DrawItem::Shadow` carries THIS struct, not the
/// 672-byte `PaintInstance` — the production shadow pipeline's storage stride
/// is 160 bytes and M1's `PaintInstance` placeholder was never consumed.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct ShadowInstance {
    /// Canvas-affine model matrix; ellipses pre-translated by (w/2, h/2)
    /// local (center-mapped quad), rect drop shadows by (−paintedStroke/2, ·)
    /// where paintedStroke = `paint::painted_stroke_width` (0 with
    /// no stroke paint route; an alpha-0 stroke string is still a route).
    pub model_matrix: [f32; 16],
    /// rgb from the effect's color (parse-fail → black); a = effect opacity ×
    /// node.opacity (the parsed color's own alpha is DISCARDED — copy the
    /// bug, `WebGPUPathShadowRenderer.ts:162`).
    pub shadow_color: [f32; 4],
    /// [silhouetteWidth, silhouetteHeight, blurRadius, depth]. Drop shadows
    /// expand by the FULL PAINTED stroke width (`painted_stroke_width`);
    /// inner shadows ignore the stroke.
    pub shape_data: [f32; 4],
    /// [tl, tr, br, bl] — RAW node radii (`WebGPUShapeUtils.
    /// normalizeCornerRadii`: NO clamp, missing entries → 0); ellipse
    /// sentinel: [radiusX(+paintedStroke/2 for drop), radiusY(+paintedStroke/2), 0, −1].
    pub corner_radii: [f32; 4],
    /// [x, y, showBehindNode, opaqueSdfCoversShadow] — world-space offset (drop)
    /// / in-SDF inset offset (inner, CSS inset convention).
    pub shadow_offset: [f32; 4],
    /// PLAIN page clip (NOT the mask-intersected effective clip —
    /// `prepareShadowInstanceData` uses `calculatePageClipBounds`).
    pub page_clip_bounds: [f32; 4],
    /// [isInner, spread, 0, 0].
    pub inner_params: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<ShadowInstance>() == 160);

/// `INFINITE_CLIP` clone (`renderers/webgpu/utils/transientOrigin.ts:4-9`).
pub const INFINITE_CLIP_SENTINEL: f32 = 9_999_999.0;
pub const INFINITE_CLIP: [f32; 4] = [
    -INFINITE_CLIP_SENTINEL,
    -INFINITE_CLIP_SENTINEL,
    INFINITE_CLIP_SENTINEL * 2.0,
    INFINITE_CLIP_SENTINEL * 2.0,
];

// ---------------------------------------------------------------------------
// Geometry seam (filled by M2's tess port; placeholder in tess/ until then)
// ---------------------------------------------------------------------------

/// Tessellated fill+stroke geometry for one path node — the output contract of
/// `tess::tessellate_combined` (production `WebGPUPathTessellator.
/// tessellateCombined`). Vertex stream layout (PLAN M1 / render-semantics
/// §3.2): one combined position stream `[x0,y0, x1,y1, ...]` (node-local
/// units) with a parallel per-vertex type stream (0.0 = fill, 1.0 = stroke).
/// Fill vertices come first, then stroke vertices.
#[derive(Clone, Default, PartialEq)]
pub struct TessellatedGeometry {
    pub positions: Vec<f32>,
    pub vertex_types: Vec<f32>,
    pub vertex_count: u32,
    /// Node-local bounds of fill ∪ stroke geometry (includes the stroke,
    /// `WebGPUPathRenderer.ts:1707-1713`). Gradients span the path's own
    /// bounds instead (`paint::gradient_box`).
    pub bounds: [f32; 4],
    /// `Some(reason)` when the fill was refused fail-closed (a non-finite
    /// coordinate, or lyon erroring/aborting at the vertex cap) and the stream
    /// carries stroke vertices only. The plan builder reports it through
    /// `PlanWarnings::dropped` (a body fill that would have painted) or
    /// `PlanWarnings::skipped` (a body fill whose resolved paint is invisible)
    /// or via `PlanMasks` (a clip mask, which then covers nothing) — the engine
    /// installs no `log` sink, so this field is the only way a dropped fill
    /// reaches anyone. Every lane draws the rest of the page and reports the
    /// node (`moda-render`: stderr + the resource report's `fills_dropped`).
    pub fill_dropped: Option<String>,
    /// `Some(reason)` when the stroke was refused: its stream would be over
    /// `tess::tessellator::MAX_STROKE_STREAM_VERTICES`. The stream carries the
    /// fill only; the plan builder reports it through `PlanWarnings::dropped`.
    pub stroke_dropped: Option<String>,
    /// Stroke gradient coordinates (E8.18, `tess::stroke_param`): two floats
    /// per vertex, `(along, across)`, for a stroke whose gradient lies along
    /// or across it; fill vertices carry zeros. Empty when not needed (the
    /// frame uploads zeros). Only the path-gradient pipeline reads them.
    pub stroke_uv: Vec<f32>,
}

/// The derived format, with `stroke_dropped` only when set: plans are compared by their `Debug` text
/// (`tests/existing_documents_unchanged.rs`), and a document whose stroke draws must plan as it did before the field.
impl std::fmt::Debug for TessellatedGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("TessellatedGeometry");
        d.field("positions", &self.positions)
            .field("vertex_types", &self.vertex_types)
            .field("vertex_count", &self.vertex_count)
            .field("bounds", &self.bounds)
            .field("fill_dropped", &self.fill_dropped);
        if self.stroke_dropped.is_some() {
            d.field("stroke_dropped", &self.stroke_dropped);
        }
        d.field("stroke_uv", &self.stroke_uv).finish()
    }
}

// ---------------------------------------------------------------------------
// Mask seam (M3d fills; consumed by every pipeline bind)
// ---------------------------------------------------------------------------

/// A rendered R8 coverage mask and how to bind/sample it
/// (render-semantics §4.6; `WebGPUPathRenderer.bindMaskGroup`).
///
/// Binding contract: group index `MASK_BIND_GROUP_INDEX` (or
/// `_IMAGE_BACKED` for pattern/image-backed pipelines) with entries
/// 0 = linear sampler, 1 = r8unorm texture view, 2 = uniform
/// `MaskUniforms { maskActive: u32, maskOrigin: vec2f, maskInvSize: vec2f }`
/// (size 24 B, allocate 32). When no mask: bind the shared inactive group
/// (maskActive = 0, 1×1 white texture). Fragment contract:
/// `uv = (worldPos − origin) × invSize`; outside [0,1] → discard;
/// `alpha ×= mask.r` (LINEAR sample). Masked content is ALWAYS
/// transparent-phase (`WebGPUPathRenderer.ts:634-645`).
#[derive(Debug, Clone, PartialEq)]
pub struct MaskRef {
    /// Key into the frame's mask-texture table (built in a pre-pass) —
    /// index into `CoveragePlan::masks`.
    pub mask_id: u64,
    /// Canvas-space origin of the mask texture.
    pub origin: [f32; 2],
    /// 1 / size in canvas units.
    pub inv_size: [f32; 2],
}

/// M3d interface-change note (SEAMS.md change log): the GPU-free description
/// of one mask owner's r8unorm coverage texture, carried on
/// `CoveragePlan::masks` so the wgpu executor (and a future CPU backend) can
/// build coverage in a pre-pass (`WebGPUMaskRenderer.getOrCreateMask` /
/// `renderCoverage`). The texture spans the mask node's canvas AABB; its
/// resolution is chosen by the executor from the frame's canvas→pixel scale
/// (`effects::mask::coverage_placement`).
#[derive(Clone, PartialEq)]
pub struct MaskPlan {
    /// The mask owner (group with `clipMaskNodeId` / container with
    /// `clipContent`).
    pub owner_node_id: String,
    /// Canvas-space AABB origin of the mask node.
    pub origin: [f64; 2],
    /// Canvas-space AABB size (canvas units).
    pub size: [f64; 2],
    /// Mask node's canvas-affine model matrix (column-major).
    pub model_matrix: [f32; 16],
    /// Node-local coverage vertices (tessellated fill+stroke positions, or
    /// the full-quad fallback) — `arrayStride` 8, float32x2. `draws` index
    /// into it.
    pub positions: Vec<f32>,
    /// The mask node's `maskType` (E7.18): alpha, vector or luminance. A
    /// clipping container is always `Vector`.
    pub mask_type: MaskType,
    /// Coverage draws, in order, composited source-over into the r8 texture:
    /// one per fill and stroke layer of an alpha / luminance mask (fills
    /// bottom to top, then strokes), one white draw of the whole geometry for
    /// a vector mask.
    pub draws: Vec<MaskDraw>,
    /// 0 = geometry, 1 = text (M4d), 2 = image, 3 = rendered layer: the source texture bound
    /// for [`MaskPaint::Source`] draws.
    pub mode: u32,
    /// Image or rendered-layer source (modes 2 and 3).
    pub image: Option<MaskImage>,
    /// Text source (mode 1 only) — M4d interface-change note: the
    /// Degraded-tier raster standing in for the browser's Canvas2D
    /// standalone texture (`WebGPUMaskRenderer.renderCoverage` text mode;
    /// `getStandaloneTextureForMask`). Coverage = sampled `.a` at
    /// world-UV (worldPos − text_origin) / raster logical size.
    pub text: Option<MaskText>,
    /// Nearest ancestor mask (index into `CoveragePlan::masks`) composed in
    /// during coverage rendering — always a smaller index (built first).
    pub ancestor: Option<usize>,
    pub invert: bool,
    pub feather: f64,
}

impl std::fmt::Debug for MaskPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("MaskPlan");
        debug.field("owner_node_id", &self.owner_node_id)
            .field("origin", &self.origin).field("size", &self.size)
            .field("model_matrix", &self.model_matrix).field("positions", &self.positions)
            .field("mask_type", &self.mask_type).field("draws", &self.draws)
            .field("mode", &self.mode).field("image", &self.image)
            .field("text", &self.text).field("ancestor", &self.ancestor);
        if self.invert || self.feather > 0.0 {
            debug.field("invert", &self.invert).field("feather", &self.feather);
        }
        debug.finish()
    }
}

/// Figma's mask types (`maskType` on the mask node; absent = alpha).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaskType {
    /// The layer's rendered alpha: paint alpha × layer opacity × node opacity.
    #[default]
    Alpha,
    /// The layer's outline: its fill and stroke geometry at full coverage.
    Vector,
    /// The rendered colour's luminance (0.2125 R + 0.7154 G + 0.0721 B on the
    /// sRGB-encoded values, CSS Masking's coefficients) times its alpha.
    Luminance,
}

impl MaskType {
    pub fn from_node_value(v: Option<&str>) -> MaskType {
        match v {
            Some("vector") => MaskType::Vector,
            Some("luminance") => MaskType::Luminance,
            _ => MaskType::Alpha,
        }
    }
}

/// One coverage draw: `count` vertices of `MaskPlan::positions` from
/// `first`, painted with `paint`.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskDraw {
    pub first: u32,
    pub count: u32,
    pub paint: MaskPaint,
}

/// What a coverage draw paints. The coverage shader turns the straight RGBA
/// it resolves into coverage by the mask type (alpha: `a`; luminance:
/// `luma(rgb) × a`) and composites it source-over.
#[derive(Debug, Clone, PartialEq)]
pub enum MaskPaint {
    /// A constant straight colour; `a` carries layer × node opacity. A vector
    /// mask draws opaque white.
    Solid([f32; 4]),
    /// A gradient (the path-gradient WGSL's model: `fill_type` 1 linear,
    /// 2 radial, 3 conic, 4 diamond; points in the 0-100 box space of
    /// `bounds`, node-local `[x, y, w, h]`), alpha × `opacity`.
    Gradient {
        fill_type: f32,
        start: [f32; 2],
        end: [f32; 2],
        rotation: f32,
        num_stops: u32,
        stops: [[f32; 4]; 8],
        bounds: [f32; 4],
        opacity: f32,
    },
    /// The mask's source texture (the image of mode 2, the text raster of
    /// mode 1), alpha × `opacity`.
    Source { opacity: f32 },
}

/// Decoded image-alpha source + pattern placement for a mode-2 mask
/// (`WebGPUMaskRenderer.computeImagePlacement`).
#[derive(Clone, PartialEq)]
pub struct MaskImage {
    /// Straight (non-premultiplied) RGBA8 — only `.a` is sampled.
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// fillPatternOffset (shape-local px).
    pub offset: [f32; 2],
    /// Source texture dims used for UV normalization.
    pub dims: [f32; 2],
    /// fillPatternScale ÷ texture scale factor (1 for raster sources).
    pub scale: [f32; 2],
    /// fillPatternRotation in radians.
    pub rotation_rad: f32,
    /// A paint layer's surface (`paint-surface:` src), or a graph-pixel unit kept on the GPU (a registered
    /// `external_texture::CaptureTexture`): `rgba` is empty and the mask samples the GPU texture (its pixels live
    /// outside the engine; `MaskAtlas::build*` take the views).
    pub surface: Option<String>,
    pub rendered: Option<std::sync::Arc<RenderedMask>>,
}

#[derive(Debug, Clone)]
pub struct RenderedMask {
    pub document: std::sync::Arc<moda_scene::types::SceneDocument>,
    pub node_id: String,
    pub shader_time_s: f64,
    pub source_fingerprint: u64,
}

impl PartialEq for RenderedMask {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id && self.source_fingerprint == other.source_fingerprint
    }
}

/// As derived, except that `surface` is printed only when set: a plan's Debug form is what
/// `tests/existing_documents_unchanged.rs` digests, and a mask with no surface prints as it did before the field.
impl std::fmt::Debug for MaskImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("MaskImage");
        d.field("rgba", &self.rgba)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("offset", &self.offset)
            .field("dims", &self.dims)
            .field("scale", &self.scale)
            .field("rotation_rad", &self.rotation_rad);
        if let Some(s) = &self.surface {
            d.field("surface", s);
        }
        if let Some(rendered) = &self.rendered {
            d.field("rendered", rendered);
        }
        d.finish()
    }
}

/// Text-alpha source for a mode-1 mask (M4d). The raster is the Degraded
/// tier's qualityScale-2 RGBA of the mask-source richtext node; its
/// `logical_w/h` are the texture's CSS-px size (`textMaskInfo.logicalSize`).
/// NOTE the production quirk: the coverage quad is `(0,0,logicalSize)` at
/// the node affine with `textOrigin` = node canvas position — richtext
/// `verticalAlign` does NOT move the mask silhouette (top-left anchored),
/// unlike the display quad's alignment offsets (`transforms.ts:96-101`).
#[derive(Debug, Clone, PartialEq)]
pub struct MaskText {
    pub raster: std::sync::Arc<moda_text::degraded::DegradedRaster>,
    /// `originCanvas` — the mask node's canvas position (top-left).
    pub origin_canvas: [f64; 2],
}

// ---------------------------------------------------------------------------
// Text seam (M3a produces, engine text_draw.rs consumes)
// ---------------------------------------------------------------------------

/// M3a Gate 2 (interface-change note in SEAMS.md change log): the ordered
/// draw-op stream types are DEFINED in `moda_text::ops` (the producer crate)
/// and re-exported here so the seam stays importable from `seams::`.
/// `TextOp` = node-local `[x,y,u,v]` vertices + u32 triangle indices + the
/// per-draw uniforms (color / opacity / blur / stroke); pipelines are the
/// extracted `text-decoration` / `text-emoji` modules (manifest descriptors).
pub use moda_text::ops::{TextGradientUniforms, TextOp, TextOpPipeline};

/// vork: the batch is moda-text-draw's, where the text GPU side is ported (without the `mask` and `fill_shader`
/// fields, which belong to text masks and text shader fills that are not ported). See PROVENANCE.md.
pub use moda_text_draw::seams::TextQuadBatch;

/// Text shader fill (M4 "richtext shader fills"): everything
/// the executor needs to pack ONE 656-byte `CustomShaderData` instance for
/// the `<id>.vector-text-fill` pipeline. The engine owns the remaining header
/// fields at prepare time: `model` = batch `model_matrix`, `pathInfo` =
/// `[0, 0, 0, linear_depth(z)]`, `pageClipBounds` = batch
/// `page_clip_bounds` (`buildTextShaderFillCommonData`,
/// `renderers/webgpu/text/msdf/textShaderFillData.ts`).
#[derive(Debug, Clone, PartialEq)]
pub struct TextShaderFill {
    /// Packed shaderParams/colorData tail + resolved module id
    /// (`packInstanceData` over the manifest packing tables).
    pub packed: moda_shaders::runtime::PackedFill,
    /// `[1, 1, 1, nodeOpacity]` — the shader supplies RGB; node opacity is
    /// applied once by the fragment via `fillColor.a`.
    pub fill_color: [f32; 4],
    /// Node stroke (alpha × opacity) or transparent — ABI parity only, the
    /// fill fragment never reads it (stroke is a separate solid pass).
    pub stroke_color: [f32; 4],
    /// `[0, 0, max(1, boxW), max(1, boxH)]` — text-box-local coordinate
    /// field, zero-size-box NaN guard included.
    pub bounds: [f32; 4],
}

/// A progressive background blur's ramp, in TARGET px: the radius runs from
/// `start_radius_px` at `start_px` to the op's `radius_px` at `end_px`
/// (`progressive-blur.wgsl`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurRamp {
    pub start_radius_px: f32,
    pub start_px: [f32; 2],
    pub end_px: [f32; 2],
}

/// vork: moda-text-draw's (see PROVENANCE.md).
pub use moda_text_draw::seams::DegradedQuad;

// ---------------------------------------------------------------------------
// Backdrop seam (M3e fills; the pass-split hook lives in frame.rs)
// ---------------------------------------------------------------------------

/// One backdrop effect operation (render-semantics §4.7; params already
/// converted to TARGET PIXELS with the production clamps: blur ≤ 200,
/// pixelate ≤ 400, dither cell ≤ 256 — `WebGPURenderOrchestrator.ts:1317-1400`).
#[derive(Debug, Clone, PartialEq)]
pub enum BackdropOp {
    /// A background blur: uniform at `radius_px`, or progressive when `ramp`
    /// is set (`radius_px` is then the END radius).
    Blur {
        radius_px: f32,
        frost: f32,
        ramp: Option<BlurRamp>,
    },
    Pixelate {
        cell_px: f32,
    },
    Dither {
        cell_px: f32,
        /// 0 = threshold, 1 = bayer-4x4, 2 = floyd-steinberg, 3 = atkinson
        /// (`ditherPatternToRendererMode`, `WebGPURenderOrchestrator.ts:73-81`).
        /// Diffusion patterns (>= 2) degrade to Bayer at execute time when the
        /// grid exceeds `MAX_DIFFUSION_DITHER_CELLS` (`resolveDitherRenderPattern`).
        pattern: u32,
        /// Clamped [-100, 100] (`WebGPURenderOrchestrator.ts:1357`).
        brightness: f32,
        /// Clamped [0, 200] (`WebGPURenderOrchestrator.ts:1358`).
        contrast: f32,
    },
    Glass {
        blur_px: f32,
        depth_px: f32,
        refraction_px: f32,
        dispersion_px: f32,
        rim_width_px: f32,
        light_angle_rad: f32,
        light_splay: f32,
        light_intensity: f32,
        /// 0 = barrel, 1 = lens, 2 = swirl.
        refraction_style: u32,
    },
    /// A backdrop filter kind (live-nodes stage 7b: grain, vignette, glaze,
    /// displace, grade; params in `moda_scene::backdrop_kinds`): one
    /// `backdrop-filter` pass over the copied region, then the ordinary
    /// composite. Several backdrops on one node are several splits in a row,
    /// each copying what the one before composited.
    Filter(Box<BackdropFilter>),
    /// An adjustment kind (adjustment-layers.md: levels, curves, …, channel-mixer): one `backdrop-adjust` pass over
    /// the copied region with moda-color's tables, then the ordinary composite.
    Adjust(Box<BackdropAdjust>),
    /// Blend split (`backdrop::blend`): save `region_px` of the target (the
    /// backdrop) and clear it, so the node's body draws alone into it.
    BlendBegin,
    /// Paint-stack group (E3.1 P6): save `region_px` of the target WITHOUT
    /// clearing it, so the node's layers draw onto the backdrop (a
    /// non-isolated group, as Illustrator composites an object's appearance:
    /// a layer's blend mode reaches the art behind the object). Closed by a
    /// `StackEnd` with [`STACK_FADE_MODE`].
    GroupBegin,
    /// Blend split: composite the node's isolated paint over the saved
    /// backdrop with the W3C formula for `mode` (`BlendMode::split_id`).
    BlendEnd {
        mode: u32,
    },
    /// Paint-stack split (E3.1 P6). After a `BlendBegin`: as `BlendEnd`, for an isolated group of a node's
    /// layers (or one blended layer): the isolated paint S is scaled by `opacity` before the W3C formula for
    /// `mode` ([`BlendMode::exact_split_id`]: 0 is source-over). After a `GroupBegin` (`mode` =
    /// [`STACK_FADE_MODE`]): the region becomes `B + opacity·(C − B)` (premultiplied), B the saved backdrop
    /// and C the layers drawn onto it: a non-isolated group at the node's opacity (PDF 1.7 §11.4.8 with a
    /// constant group opacity).
    StackEnd {
        mode: u32,
        opacity: f32,
    },
}

/// A [`BackdropOp::Filter`]'s kernel inputs, resolved to TARGET PIXELS by the planner
/// (`backdrop::plan`): the `backdrop-filter.wgsl` uniforms from `header` on (the executor fills the node rect
/// and local frame from the split).
#[derive(Debug, Clone, PartialEq)]
pub struct BackdropFilter {
    /// 0 grain, 1 vignette, 2 glaze, 3 displace, 4 grade (the WGSL's `header.x`).
    pub kind: u32,
    /// `header`, `p0` .. `p4`, `colors[8]`, `stops[2]`: 64 floats (`p2` is the clip box the planner fills,
    /// `p3` / `p4` the rounded clip the executor fills).
    pub uniforms: [f32; BACKDROP_FILTER_UNIFORM_FLOATS],
    /// How far past the node the kernel samples, in target px (displace's largest offset; 0 for the rest).
    pub reach_px: f32,
}

/// A [`BackdropOp::Adjust`]'s kernel inputs: the `backdrop-adjust.wgsl` uniforms and its 256 × 2 RGBA8 lookup
/// texture (row 0 the per-channel tables, row 1 the gradient map's colours), built by the planner with moda-color.
#[derive(Debug, Clone, PartialEq)]
pub struct BackdropAdjust {
    pub uniforms: [f32; BACKDROP_ADJUST_UNIFORM_FLOATS],
    pub lut: std::sync::Arc<Vec<u8>>,
}

/// Floats in [`BackdropAdjust::uniforms`] (8 × vec4).
pub const BACKDROP_ADJUST_UNIFORM_FLOATS: usize = 32;

/// Floats in [`BackdropFilter::uniforms`].
pub const BACKDROP_FILTER_UNIFORM_FLOATS: usize = 64;

impl BackdropOp {
    /// A blend / paint-stack split (`backdrop::blend`), as opposed to a backdrop effect. These run on a
    /// multisample target too: they copy the resolved target and composite back into the multisample one.
    pub fn is_blend_split(&self) -> bool {
        matches!(
            self,
            BackdropOp::BlendBegin
                | BackdropOp::GroupBegin
                | BackdropOp::BlendEnd { .. }
                | BackdropOp::StackEnd { .. }
        )
    }
}

/// The `StackEnd` mode closing a `GroupBegin`: fade the layers drawn onto the backdrop by the group opacity.
pub const STACK_FADE_MODE: u32 = 16;

/// Pass-split contract (the M1 hook; M3e implements `run`):
///
/// The frame walk emits draw items in z order. When it reaches a node with an
/// active backdrop effect it emits `DrawItem::BackdropSplit` BEFORE the node's
/// own body. The executor then:
///   1. ends the current render pass (storeOp: store; MSAA resolved),
///   2. `copy_texture_to_texture` of `region_px` from the RESOLVED target,
///   3. runs the effect chain (refract → blur/pixelate/dither → composite
///      back, sampling only already-drawn content — painter's algorithm),
///   4. begins a new pass with loadOp: Load on color AND depth, and continues
///      the remaining draw items.
///
/// Pages containing any active backdrop effect render with sample_count = 1
/// and opaque white clear; `transparent` and `exclude` are unsupported there
/// (`WebGPUExportCoordinator.ts:786-795`, `WebGPURenderer.ts:2548-2552`).
#[derive(Debug, Clone, PartialEq)]
pub struct BackdropSplit {
    pub node_id: String,
    pub op: BackdropOp,
    /// Quantized sampling region in target pixels [x, y, w, h]
    /// (visible rect ∩ clip, expanded per-effect, floor/ceil quantized —
    /// `compositeShapeBackdropBlur`, BackdropBlurRenderer:520-585).
    pub region_px: [u32; 4],
    /// Coverage source for the composite alpha: analytic rect/ellipse SDF or
    /// the node's tessellated geometry (star/polygon/path).
    pub coverage: CoverageSource,
    /// M3e (interface-change note, see SEAMS.md change log): the production
    /// composite/refraction shaders need the rotated AABB, the visible rect
    /// (scissor), and the local frame — all in TARGET PIXELS
    /// (`runRefractionPass`/`runCompositePass` uniforms, BackdropBlurRenderer
    /// :1751-2118). Full rotated node AABB [x, y, w, h] in target px
    /// (pre-clip; may extend off-target).
    pub node_rect_px: [f32; 4],
    /// nodeRect ∩ effective clip ∩ target — the composite scissor rect
    /// (`visibleNodeRect`).
    pub visible_rect_px: [f32; 4],
    /// Absolute rotation in radians (canvas-space Y-down, `atan2(b, a)` of
    /// the canvas affine — `WebGPURenderOrchestrator.ts:1253-1262`).
    pub rotation_rad: f32,
    /// +1, or −1 for mirrored orthogonal frames (det < 0).
    pub local_frame_y_sign: f32,
    /// Local (unrotated) half-extents of the shape in target px — distinct
    /// from `node_rect_px.zw / 2` (the rotated-AABB half-size).
    pub local_half_size_px: [f32; 2],
    /// A clipping container's rounded box the composite is clipped to, in
    /// target px: ([x, y, w, h], radii [tl, tr, br, bl]). Set when the visible
    /// part of the node reaches a rounded corner of a frame above it (its
    /// plain box already narrows `visible_rect_px`); `None` otherwise.
    pub clip_rrect_px: Option<([f32; 4], [f32; 4])>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoverageSource {
    /// Analytic rounded-rect SDF. M3e note: `radii` are per-corner
    /// [tl, tr, br, bl] in TARGET PIXELS, already `× effectiveScale` and
    /// clamped to the local half-extent (`WebGPURenderOrchestrator.ts:
    /// 1410-1419`). `size` = 2 × `local_half_size_px` (kept for CPU-backend
    /// consumers).
    Rect { size: [f32; 2], radii: [f32; 4] },
    /// Analytic ellipse SDF (`size` = 2 × `local_half_size_px`, target px).
    Ellipse { size: [f32; 2] },
    /// Tessellated coverage mask. M3e note: `positions` are fill triangles in
    /// TARGET-PIXEL space (the production `buildCoverageVerticesTargetPx`
    /// projects local → canvas → target px before rasterizing the mask;
    /// node-local was M1's guess). `vertex_types`/`bounds` unused here.
    Geometry(TessellatedGeometry),
}

// ---------------------------------------------------------------------------
// The draw-item stream (engine-internal IR; tracks append new variants ONLY
// via an interface-change note after M1)
// ---------------------------------------------------------------------------

/// Paint for a body draw. M1 implements `Solid`; M3b/M3f fill the rest. The
/// blend/depth state for each variant comes from the manifest pipeline
/// descriptors — by module id — never hand-coded.
///
/// M3b interface-change note (SEAMS.md change log): `Gradient` and `Pattern`
/// now CARRY their packed instance payloads (`PathGradientData` 1552 B /
/// `PathPatternData` 432 B mirrors of the extracted WGSL structs, plus the
/// decoded pattern texture). Payloads are GPU-free, so `CoveragePlan` remains
/// consumable by a CPU backend. `instance.path_info[0..2]`
/// (vertexOffset/vertexCount) are executor-time values fixed up at upload.
#[derive(Debug, Clone, PartialEq)]
pub enum BodyPaint {
    /// path-solid module; instance = `SolidPathInstance`.
    Solid,
    /// path-gradient module (M3b): packed `PathGradientData` (fill gradient,
    /// gradient stroke, or stroke-only overlay over a PATTERN fill — pattern
    /// overlays draw in the regular gradient slot, before custom shaders).
    Gradient(Box<crate::paint::GradientInstance>),
    /// path-gradient module (Q-116): a SHADER-fill body's stroke-only
    /// gradient overlay. Identical packing to `Gradient`, but drawn in its
    /// own slot AFTER custom shaders within the bucket
    /// (`postShaderGradientStrokePaths`, `WebGPUPathRenderer.ts:1016,
    /// 1190-1194`) — the shader body would otherwise paint over the inner
    /// half of the centered stroke band (~half-width strokes).
    PostShaderGradientStroke(Box<crate::paint::GradientInstance>),
    /// path-pattern module (M3b): packed `PathPatternData` + decoded texture
    /// + sampler choice (texture bind group rides group 2; mask group 3).
    Pattern(Box<crate::paint::PatternBody>),
    /// procedural fill (M3f); instance = `PaintInstance`. M4e
    /// interface-change note (SEAMS.md change log): the variant now carries
    /// `ShaderFillBody` — module id + the OPTIONAL preprocessed shader-image
    /// texture for image-backed shaders (texture bind group rides group 2,
    /// mask moves to group 3 — `manifest.packing.mask_bind_group_index`).
    /// Payload is GPU-free: the `CoveragePlan` CPU-backend invariant holds.
    Shader(Box<ShaderFillBody>),
}

/// M4e: a routed procedural fill.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderFillBody {
    /// Alias-resolved module id (keys WGSL module + pipeline descriptors).
    pub module_id: String,
    /// Preprocessed shader image (image-backed shaders only; `None` for
    /// self-contained fills). Fallback entries are a 1×1 transparent texture
    /// (`ShaderImageResourceManager.createFallbackEntry`).
    pub image: Option<ShaderImageTexture>,
    /// Shader playback: the `shader_params` float offsets carrying
    /// the renderer clock (`moda_shaders::runtime::clock_param_offsets`), so the
    /// live camera present path can re-apply a new clock to this CACHED fill's
    /// instance without rebuilding the plan. `Some([])` = no clock (static
    /// fill); `Some([..])` = patch these slots with `shader_time_s as f32` at
    /// present; `None` = a non-direct-copy clock use that must NOT be re-threaded
    /// (the plan then keeps the clock in its cache key — see
    /// `residency::plan_shader_time_rethreadable`).
    pub clock_offsets: Option<Vec<u16>>,
}

/// M4e: CPU-side shader-image texture (straight-alpha sRGB RGBA8 — the
/// production upload is `copyExternalImageToTexture` with `premultipliedAlpha`
/// unset). Keyed for GPU texture/bind-group dedup by the production resource
/// key (`createImageResourceKey`) or `fallback:<processing-key>`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderImageTexture {
    pub key: String,
    pub width: u32,
    pub height: u32,
    pub rgba_straight: std::sync::Arc<Vec<u8>>,
}

/// One item in the per-page draw stream, in render-z order.
///
/// RENDER-Z RULE (binding for every track): z is a SEPARATE counter from the
/// display list's `PaintItem.z_index`. It increments by exactly 1 per
/// *rendered leaf* in display-list order — groups consume NO z
/// (`renderNode.ts:32-34`), skipped mask-source nodes consume NO z
/// (`WebGPUShapeSystem.shouldSkipRender` early-returns before the increment),
/// containers consume 1 even when they draw nothing, richtext consumes 1 even
/// when text is skipped, blend nodes consume z per expanded intermediate.
/// Fractional bumps: shape/connector text +0.5, arrowheads +0.25,
/// inner-shadow quads +0.5. Depth = `depth::linear_depth(z)`.
#[derive(Debug, Clone)]
pub enum DrawItem {
    /// The page-fill quad (depth hardcoded 0.99999 in WGSL; drawn before all
    /// nodes; skipped for transparent exports and non-solid page fills).
    PageFill { instance: PageInstance },
    /// A tessellated body (fill + optional stroke vertices).
    Body {
        node_id: String,
        z: f64,
        phase: Phase,
        paint: BodyPaint,
        /// `Arc`-shared so cloning a `DrawItem::Body` is a pointer-bump, not a
        /// deep `Vec<f32>` copy. This matters on the CAMERA lane: the engine's
        /// PreparedFrame holds a 2nd strong `Arc<CoveragePlan>` (for `ptr_eq`
        /// frame reuse), so the incremental patcher's `Arc::make_mut` must clone
        /// the whole plan each edit — with `Arc` geometry that clone is O(n)
        /// cheap bumps + O(dirty) real rebuilds instead of an O(n) deep copy.
        /// The tess cache (`PlanCaches::tess`) stores the SAME `Arc`, so a
        /// cache hit hands the plan a shared body with no copy either.
        geometry: std::sync::Arc<TessellatedGeometry>,
        instance: Box<PaintInstance>,
        blend_mode: BlendMode,
        mask: Option<MaskRef>,
    },
    /// Drop/inner shadow quad (M3d): drawn in the transparent phase,
    /// shadow-first inside its EXACT z bucket (production buckets transparent
    /// items by the raw — fractional — zIndex value, not floor:
    /// `groupByZIndexWithOrdering` keys the Map by `item.zIndex`). Drop
    /// shadows share the body's integer z; inner shadows sit at
    /// z + `INNER_SHADOW_Z_OFFSET` in their own bucket ABOVE the body.
    Shadow {
        node_id: String,
        z: f64,
        /// 9-tap multiSample falloff params, SDF silhouette etc. — packed by
        /// M3d per `WebGPUPathShadowRenderer.prepareShadowInstanceData`.
        instance: Box<ShadowInstance>,
        inner: bool,
    },
    /// Text quads (M3a). `Arc` so a camera-plan clone shares them, and the prepared frame can tell an unchanged
    /// batch by pointer (a mutation goes through `Arc::make_mut`, which gives a changed batch a new one).
    Text(std::sync::Arc<TextQuadBatch>),
    /// Backdrop pass split (M3e) — see `BackdropSplit`.
    BackdropSplit(Box<BackdropSplit>),
    /// Analytic SDF solid rect/ellipse body. Replaces the tessellated
    /// `Body` for SDF-eligible solid shapes; drawn as a unit quad via the
    /// `shape-solid` module (one instanced draw — the opaque phase batches a
    /// segment's instances into a single `draw`, the transparent phase draws
    /// per z-bucket), so it stays crisp at any zoom. Slots into the SAME
    /// z-buckets as the tessellated body it replaces (opaque depth-write bucket
    /// when opaque; otherwise the transparent bucket, between the shadows and
    /// the tessellated paths). Carries no geometry and no mask (an active mask
    /// disqualifies it).
    SolidShapeSdf {
        node_id: String,
        z: f64,
        phase: Phase,
        opaque_sdf_covers_shadow: bool,
        instance: Box<crate::paint::shape_sdf::ShapeSolidSdfInstance>,
    },
    /// Analytic SDF gradient (linear/radial) rect/ellipse body via the
    /// `shape-gradient` module. Same bucketing as `SolidShapeSdf`.
    GradientShapeSdf {
        node_id: String,
        z: f64,
        phase: Phase,
        opaque_sdf_covers_shadow: bool,
        instance: Box<crate::paint::shape_sdf::ShapeGradientSdfInstance>,
    },
    /// Layer blur (SEAMS.md change log; additive, native-only — no TS
    /// counterpart): the node's own draw items are diverted into `group` and
    /// replaced in the main stream by this single composite. A PRE-pass (before
    /// the main scene pass — the mask-atlas slot, NOT a backdrop-style mid-pass
    /// split) renders the group's items into a padded offscreen texture
    /// (transparent clear, sample_count 1), runs the shared separable-Gaussian
    /// blur over it, and the main pass draws the blurred texture as one
    /// premultiplied quad here, in the transparent phase at the body's z. The
    /// item consumes NO extra render-z (it takes the diverted body's z), so
    /// plans without a layer blur are byte-identical to before the feature.
    LayerBlurComposite {
        node_id: String,
        /// The diverted body's z (shadow/inner/text offsets ride inside the
        /// group, so one bucket slot represents the whole node).
        z: f64,
        /// GPU-free capture description (CPU-backend seam invariant holds: a
        /// CPU backend can rasterize the group's items into a padded buffer,
        /// convolve, and composite). `Arc` so camera-lane plan clones stay
        /// pointer-bumps like `Body::geometry`.
        group: std::sync::Arc<LayerBlurGroup>,
        /// The composite quad's clip (canvas space, fragment test): the
        /// node's effective clip, the page clip narrowed by the boxes of the
        /// clipping frames above it, as its body's.
        page_clip_bounds: [f32; 4],
        /// A rounded clip on top of `page_clip_bounds` (canvas space,
        /// fragment test): the innermost rounded clipping frame above the
        /// node ([`ClipRRect`]), or `None`.
        clip_rrect: Option<ClipRRect>,
        /// The clip mask of a clip-mask group above the node (its nearest
        /// mask, whose coverage composes the ones above it): the capture is
        /// taken unmasked and the composite quad samples this mask (group 2,
        /// the path shaders' contract), so the filtered result is soft inside
        /// the mask and draws nothing outside it. `None` outside a clip-mask
        /// group (a clipping frame clips by `page_clip_bounds` / `clip_rrect`).
        mask: Option<MaskRef>,
    },
}

/// An axis-aligned rounded rectangle in canvas space: a clipping frame's box
/// and its corner radii (top-left, top-right, bottom-right, bottom-left),
/// already clamped to fit the box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipRRect {
    pub rect: [f32; 4],
    pub radii: [f32; 4],
}

/// One layer-blurred node's isolation capture (see
/// `DrawItem::LayerBlurComposite`). Region/scale math lives in
/// `layer_blur::plan`; execution in `layer_blur::exec` + `frame.rs`.
#[derive(Clone)]
pub struct LayerBlurGroup {
    /// The node's own draw items in their normal relative order (shadow →
    /// body → inner shadow → shape text) — a group's or a frame's: its whole
    /// subtree's —, canvas-absolute space, page clip kept as planned. Nested
    /// `LayerBlurComposite`s are shadow captures only, one level deep (the
    /// mini-plan builder asserts it).
    pub items: Vec<DrawItem>,
    /// Padded capture rect in canvas-absolute units [x, y, w, h]: the node's
    /// canvas AABB expanded by the blur bleed (≈ radius) + stroke + shadow
    /// reach.
    pub region_canvas: [f64; 4],
    /// The canvas rect the capture's picture occupies: `region_canvas`, or for a capture cut to what can be read
    /// (`layer_blur::plan::readable_window`) the region before the cut. What encloses the capture measures the
    /// picture by this (`layer_blur::shadow::all_items_canvas_bounds`): a cut is how much is stored, and it
    /// follows the target, where the picture's bounds and anything placed by them do not.
    pub extent_canvas: [f64; 4],
    /// Offscreen capture size in physical px = region_canvas × capture scale.
    /// The capture scale is the canvas→target scale REDUCED for large uniform
    /// radii so the per-pass sigma stays within the shared 31-tap kernel's
    /// exact range (σ ≤ 5); the composite's bilinear upsample supplies the
    /// rest — the Skia-style downsampled-Gaussian scheme (layer-blur-spec.md
    /// §6.3).
    pub width_px: u32,
    pub height_px: u32,
    /// How far (canvas units) a texel of the filtered result reads the capture: its filters' reach. A result shown
    /// only over part of the target needs the capture only that far around it (`frame.rs` capture scissors).
    pub reach_canvas: f64,
    /// What runs over the capture (`layer_blur::exec::LayerBlurRenderer::filter_capture`).
    pub filters: LayerFilters,
    /// Content key of the capture's result (`residency::capture_key`): equal
    /// keys draw identical pixels, so the retained engine reuses the filtered
    /// result across frames (`layer_blur::cache`) instead of re-rendering it.
    /// `None` = render every time (exports, shader-clock content).
    pub cache_key: Option<u64>,
    /// The capture's content without its geometry (`residency::capture_key`):
    /// what it draws, not at which zoom step or window. The live lane composites
    /// an earlier step's result with this key while the new one waits for the
    /// frame's capture budget (`Engine::set_capture_frame_budget`).
    pub identity_key: Option<u64>,
    /// The node's shape silhouette (at its opacity), captured like `items`
    /// into a second texture that clips the noise: grain covers the shape
    /// even where its paint is transparent or absent, as in Figma. Empty = the
    /// noise clips to the capture's own alpha (no noise, or the paint covers
    /// the shape).
    pub coverage: Vec<DrawItem>,
    /// The capture's own mask table (`CoveragePlan::masks` of its mini plan):
    /// a clipping frame's clip when the frame itself is the filtered root,
    /// whose children are clipped BEFORE the filter (`frame.rs`
    /// `divert_subtree_filter`). `items` mask ids index this table, not the
    /// page plan's. Empty for every other capture (their items carry no mask).
    pub masks: Vec<MaskPlan>,
}

/// Debug leaves out derived execution metadata: `cache_key` (retained lanes
/// set it, one-shot builds do not) and `reach_canvas` (a conservative capture
/// scissor margin derived from the filters and geometry already printed).
/// Plans that draw the same thing print — and fingerprint
/// (`residency::plan_items_fingerprint`) — the same.
impl std::fmt::Debug for LayerBlurGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerBlurGroup")
            .field("items", &self.items)
            .field("region_canvas", &self.region_canvas)
            .field("width_px", &self.width_px)
            .field("height_px", &self.height_px)
            .field("filters", &self.filters)
            .field("coverage", &self.coverage)
            .field("masks", &self.masks)
            .finish_non_exhaustive()
    }
}

/// The layer filters over one capture, in CAPTURE pixels, run in a fixed
/// order: filter shadows, texture, blur, every noise in list order, then
/// feather (the TS `effect.ts` doc). At least one is set, but for a motion
/// capture.
#[derive(Clone, PartialEq, Default)]
pub struct LayerFilters {
    /// The node's canvas AABB origin in capture px: the grain lattice anchor.
    pub origin_px: [f32; 2],
    /// Filter shadows (`moda_scene::effect::FilterShadow`, CSS `filter:
    /// drop-shadow()`), first, back to front: each is the capture's own alpha
    /// moved, blurred and tinted, and the capture is drawn over them. Every
    /// pass is shown behind (`ShadowPass::show_behind`).
    pub drop_shadows: Vec<ShadowPass>,
    /// The filter shadows are the only filter and the layer's items hold no
    /// capture of their own: the result is the shadows ALONE, composited
    /// before the layer's own items, which draw directly as they would
    /// unfiltered. The layer's picture then never passes through the capture,
    /// whose size limits soften only the shadows. With another filter (a blur
    /// over the picture and its shadows), or around a descendant's capture
    /// (which a copy would hold twice), the capture is the picture, as for
    /// every other filter.
    pub shadows_alone: bool,
    pub texture: Option<TextureFilter>,
    pub blur: Option<BlurFilter>,
    pub noise: Vec<NoiseFilter>,
    /// Illustrator's Feather, last: the edges fade inside the outline.
    pub feather: Option<FeatherFilter>,
    /// A shadow capture (drop or inner shadows of a shape the SDF shadow lane
    /// can't draw): the capture is the node's silhouette and the result is
    /// its shadows ALONE, not the capture. Set only on shadow groups, which
    /// carry no other filter.
    pub shadow: Option<ShadowFilter>,
    /// How a (non-shadow) filtered node's composite blends over the target: its own blend mode when a
    /// fixed-function state writes it exactly there (`backdrop::blend::filter_composite_fixed_blend`),
    /// instead of a blend split around the composite. Source-over otherwise.
    pub composite_blend: CaptureBlend,
    /// A motion capture (`crate::motion_capture`): no filter runs, the capture is the result. Only the live
    /// camera lane's moving frames plan one; a still frame, an export or a snapshot never does.
    pub motion: bool,
    /// Agent shaders (`moda_scene::effect::agent_shader_effects`), last, in list order.
    pub shaders: Vec<AgentShaderFilter>,
}

/// One agent shader layer effect, compiled for its capture (docs/design/effects-code-first.md §4.3): the moda-wgsl
/// effect module and its header slots (host values in capture px, then the parameters).
#[derive(Clone, PartialEq)]
pub struct AgentShaderFilter {
    /// The code key (`b3:…`): names it in errors and warnings.
    pub code: String,
    /// The node it's on.
    pub node_id: String,
    pub wgsl: std::sync::Arc<str>,
    pub slots: Vec<[f32; 4]>,
    /// Output px of the capture the 1080p time budget is scaled by.
    pub area_px: f64,
}

/// The module prints as its length: plans print (and fingerprint) the code key and the slots, which hold the clock.
impl std::fmt::Debug for AgentShaderFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentShaderFilter")
            .field("code", &self.code)
            .field("node_id", &self.node_id)
            .field("wgsl_len", &self.wgsl.len())
            .field("slots", &self.slots)
            .finish()
    }
}

/// The derived Debug's output, with `motion` and `drop_shadows` printed only when set: plans print (and
/// fingerprint, and pin in goldens) as they did before those existed.
impl std::fmt::Debug for LayerFilters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("LayerFilters");
        d.field("origin_px", &self.origin_px)
            .field("texture", &self.texture)
            .field("blur", &self.blur)
            .field("noise", &self.noise)
            .field("feather", &self.feather)
            .field("shadow", &self.shadow)
            .field("composite_blend", &self.composite_blend);
        if self.motion {
            d.field("motion", &self.motion);
        }
        if !self.drop_shadows.is_empty() {
            d.field("drop_shadows", &self.drop_shadows);
        }
        if self.shadows_alone {
            d.field("shadows_alone", &self.shadows_alone);
        }
        if !self.shaders.is_empty() {
            d.field("shaders", &self.shaders);
        }
        d.finish()
    }
}

/// The drop or inner shadows of one node, drawn from its silhouette capture
/// (`layer-filter-shadow`): each shadow is the capture's coverage (drop) or
/// its outside (inner) moved by the offset and blurred, then knocked out by
/// (drop) or clipped to (inner) the coverage. The composite of a drop-shadow
/// group draws at the body's z, before the body; an inner-shadow group's at
/// z + `INNER_SHADOW_Z_OFFSET`, over it.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowFilter {
    /// Clipped to the coverage, composited over the body: inner shadows and
    /// inner glows.
    pub inner: bool,
    /// How the result composites over the target: source-over for shadows;
    /// a glow's mode (`layer_blur::shadow::CaptureKind::Glow` /
    /// `InnerGlow`), Illustrator's Outer / Inner Glow "Mode".
    pub blend: CaptureBlend,
    /// Back to front (effects order).
    pub shadows: Vec<ShadowPass>,
}

/// The blend a capture-shadow group's composite draws with: the modes a
/// fixed-function blend state writes exactly over the target (premultiplied
/// source S over backdrop D). Every other blend mode would need the backdrop
/// split (`backdrop::blend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CaptureBlend {
    /// Source-over: S + D (1 - αS). Shadows, and a glow in Normal mode.
    #[default]
    Normal,
    /// S D + D (1 - αS): exact over an opaque backdrop (over a transparent
    /// one it drops the S (1 - αD) term).
    Multiply,
    /// S + D (1 - S): exact over any backdrop. A glow's default (Illustrator's).
    Screen,
    /// Linear dodge (CSS `plus-lighter`): S + D, clamped. Emitted light.
    Add,
}

impl CaptureBlend {
    /// A glow's `mode` value: `normal`, `multiply`, `screen` (also the
    /// default and any other value) or `linear-dodge` (`plus-lighter`).
    pub fn from_glow_mode(mode: Option<&str>) -> CaptureBlend {
        match mode {
            Some("normal") => CaptureBlend::Normal,
            Some("multiply") => CaptureBlend::Multiply,
            Some("linear-dodge" | "plus-lighter") => CaptureBlend::Add,
            _ => CaptureBlend::Screen,
        }
    }

    /// The mode's `mode` value.
    pub fn glow_mode(self) -> &'static str {
        match self {
            CaptureBlend::Normal => "normal",
            CaptureBlend::Multiply => "multiply",
            CaptureBlend::Screen => "screen",
            CaptureBlend::Add => "linear-dodge",
        }
    }
}

/// One shadow of a [`ShadowFilter`], in capture px.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowPass {
    /// Straight RGB in the frame's encoding; alpha = the shadow's opacity ×
    /// the node's (effective) opacity.
    pub color: [f32; 4],
    pub offset_px: [f32; 2],
    /// Gaussian reach 3σ (σ = blur / 2), the `progressive-blur` radius.
    pub radius_px: f32,
    /// Drop shadows only: not knocked out under the node.
    pub show_behind: bool,
    /// Inner glows from the centre (Illustrator's Center): the source is the
    /// coverage itself, not its outside, so the glow is brightest in the
    /// middle and fades toward the edge.
    pub center: bool,
}

/// A layer blur in capture px: `radius_px` is the 3σ reach (a node's blur
/// radius R has σ = R / 2, so reach = 1.5 R; `BLUR_REACH_PER_RADIUS`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlurFilter {
    /// The shared separable kernel (`backdrop-blur`).
    Uniform { radius_px: f32 },
    /// The radius ramps from `start_radius_px` at `start_px` to
    /// `end_radius_px` at `end_px` (`progressive-blur`).
    Progressive {
        start_radius_px: f32,
        end_radius_px: f32,
        start_px: [f32; 2],
        end_px: [f32; 2],
    },
}

/// Illustrator's Feather in capture px (`layer-filter-shadow` `fs_feather`):
/// the capture's hard alpha `min(1, 127·a)` blurred at `radius_px` reach (3σ,
/// σ = radius / 2), every pixel scaled by `clamp(2·blur − 1, 0, 1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatherFilter {
    pub radius_px: f32,
}

/// Figma's texture in capture px (`layer-filter-texture`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureFilter {
    pub size_px: f32,
    pub radius_px: f32,
    pub clip_to_shape: bool,
}

/// One Figma noise in capture px (`layer-filter-noise`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFilter {
    /// 0 mono, 1 duo, 2 multi.
    pub mode: u32,
    /// Straight RGB in the frame's encoding.
    pub color: [f32; 3],
    /// Duo's off-cell colour.
    pub secondary: Option<[f32; 3]>,
    pub size_px: f32,
    pub density: f32,
    pub opacity: f32,
    /// The noise's position among the node's noises: two noises draw different cells.
    pub seed: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Depth-write ON, drawn first in submission order (compare: less).
    Opaque,
    /// Depth-write OFF, grouped by integer z bucket ascending; within a
    /// bucket: shadows → SDF solid bodies → SDF gradient bodies → paths →
    /// text (`WebGPUTransparentGroupingUtils.groupByZIndexWithOrdering` +
    /// `WebGPUPathRenderer.renderTransparentItemsWithText`).
    Transparent,
}

/// A node's blend mode (`nodes.ts` `BLEND_MODES`).
///
/// `normal`, `multiply` and `screen` are fixed-function blend states (manifest
/// `blend_modes`; `WebGPUBlendModeRenderer.ts:30-101`): `screen` (one,
/// one-minus-src) is exact, and `multiply` is exact over an opaque backdrop with
/// the engine's corrected dst factor (see `frame.rs` `pipeline`). Every other
/// mode reads the backdrop per pixel, which no blend state can express: those
/// bodies draw inside an exact blend split (`backdrop::blend`) with the `Normal`
/// pipeline. [`BlendMode::fixed_function`] is only the fallback for a node that
/// can't take the split (region renders): overlay keeps its manifest state
/// (`src-alpha` over), the modes added later draw plain source-over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
    /// Figma's LINEAR_BURN (CSS `plus-darker`): B = max(0, Cb + Cs − 1).
    LinearBurn,
    /// Figma's LINEAR_DODGE (CSS `plus-lighter`): B = min(1, Cb + Cs).
    LinearDodge,
    /// Overprint preview (the editor's View ▸ Overprint Preview, `core/rendering/overprintPreview.ts`): not a
    /// document blend mode (not in [`BlendMode::ALL`]); the live canvas feed marks an overprinting layer with it.
    /// B composites Cs over Cb as process inks: each through the naive CMYK conversion, every ink Cs leaves at 0 %
    /// taken from Cb (PDF overprint mode 1), back to RGB.
    Overprint,
}

impl BlendMode {
    /// Every mode, in `BLEND_MODES` order.
    pub const ALL: [BlendMode; 18] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::ColorDodge,
        BlendMode::ColorBurn,
        BlendMode::HardLight,
        BlendMode::SoftLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
        BlendMode::LinearBurn,
        BlendMode::LinearDodge,
    ];

    /// The node value (`blendMode`): the CSS `mix-blend-mode` keyword, or for the two Figma modes CSS names
    /// only as `plus-darker` / `plus-lighter`, `linear-burn` / `linear-dodge`.
    pub fn node_value(self) -> &'static str {
        match self {
            BlendMode::Normal => "normal",
            BlendMode::Multiply => "multiply",
            BlendMode::Screen => "screen",
            BlendMode::Overlay => "overlay",
            BlendMode::Darken => "darken",
            BlendMode::Lighten => "lighten",
            BlendMode::ColorDodge => "color-dodge",
            BlendMode::ColorBurn => "color-burn",
            BlendMode::HardLight => "hard-light",
            BlendMode::SoftLight => "soft-light",
            BlendMode::Difference => "difference",
            BlendMode::Exclusion => "exclusion",
            BlendMode::Hue => "hue",
            BlendMode::Saturation => "saturation",
            BlendMode::Color => "color",
            BlendMode::Luminosity => "luminosity",
            BlendMode::LinearBurn => "linear-burn",
            BlendMode::LinearDodge => "linear-dodge",
            BlendMode::Overprint => "overprint",
        }
    }

    /// The blend split's shader mode id (`backdrop::blend`), or `None` for the
    /// modes drawn with a fixed-function state. 3 = overlay matches the
    /// `BlendMode` enum in `renderers/webgpu/constants/blendModes.ts`.
    pub fn split_id(self) -> Option<u32> {
        match self {
            BlendMode::Normal | BlendMode::Multiply | BlendMode::Screen => None,
            BlendMode::Overlay => Some(3),
            BlendMode::Darken => Some(4),
            BlendMode::Lighten => Some(5),
            BlendMode::ColorDodge => Some(6),
            BlendMode::ColorBurn => Some(7),
            BlendMode::HardLight => Some(8),
            BlendMode::SoftLight => Some(9),
            BlendMode::Difference => Some(10),
            BlendMode::Exclusion => Some(11),
            BlendMode::Hue => Some(12),
            BlendMode::Saturation => Some(13),
            BlendMode::Color => Some(14),
            BlendMode::Luminosity => Some(15),
            // 16 is `STACK_FADE_MODE`.
            BlendMode::LinearBurn => Some(17),
            BlendMode::LinearDodge => Some(18),
            BlendMode::Overprint => Some(19),
        }
    }

    /// [`BlendMode::split_id`], for a target that may be `transparent` (a transparent export): multiply is
    /// only exact as a fixed-function state over an opaque backdrop (`cs·cb + cb·(1 − αs)` lacks the
    /// W3C term `cs·(1 − αb)`, so over nothing it paints black), so there it takes the split too (id 1).
    /// Screen's state (`cs + cb − cs·cb`) is exact at any backdrop alpha.
    pub fn split_id_over(self, transparent: bool) -> Option<u32> {
        match self {
            BlendMode::Multiply if transparent => Some(1),
            _ => self.split_id(),
        }
    }

    /// The blend split shader's id for EVERY mode, for a split that must be exact whatever the backdrop (a
    /// paint-stack group, whose backdrop is the node's own lower layers over a cleared region): 0 = normal
    /// (source-over), 1 = multiply, 2 = screen, then [`BlendMode::split_id`].
    pub fn exact_split_id(self) -> u32 {
        match self {
            BlendMode::Normal => 0,
            BlendMode::Multiply => 1,
            BlendMode::Screen => 2,
            other => other.split_id().expect("every other mode has a split id"),
        }
    }

    /// The fixed-function state a body draws with outside a blend split: the
    /// manifest's four draw as themselves, the rest as plain source-over.
    pub fn fixed_function(self) -> BlendMode {
        match self {
            BlendMode::Normal | BlendMode::Multiply | BlendMode::Screen | BlendMode::Overlay => {
                self
            }
            _ => BlendMode::Normal,
        }
    }

    /// Manifest key for `manifest().blend_modes` (of [`BlendMode::fixed_function`]).
    pub fn manifest_key(self) -> &'static str {
        self.fixed_function().node_value()
    }

    /// `core/rendering/displayList.ts` semantics: an unknown string (or
    /// 'normal'/missing) is Normal.
    pub fn from_node_value(v: Option<&str>) -> BlendMode {
        if v == Some(BlendMode::Overprint.node_value()) {
            return BlendMode::Overprint;
        }
        v.and_then(|v| BlendMode::ALL.into_iter().find(|m| m.node_value() == v))
            .unwrap_or(BlendMode::Normal)
    }
}

// ---------------------------------------------------------------------------
// CoveragePlan — the renderer-agnostic IR (Proposal B §3.9 contingency seam)
// ---------------------------------------------------------------------------

/// The B-contingency seam (DECISION §1.7): a GPU-free description of one
/// page's paint, sufficient for a CPU coverage rasterizer (tiny-skia /
/// vello_cpu) to be slotted behind the same ingest/text/harness without
/// re-architecting. The wgpu executor consumes `Vec<DrawItem>` built FROM
/// this plan's items; a phase-2 CPU backend would consume the plan directly.
///
/// Invariants the CPU backend may rely on:
///   - items sorted by (phase==Opaque first as a depth optimization is NOT
///     semantic — painting items in `z` order with the documented intra-bucket
///     ordering produces identical output, because depth is a pure function
///     of z),
///   - all blending is premultiplied source-over IN GAMMA (sRGB) SPACE except
///     the fixed-function approximations named by `BlendMode`,
///   - clip = rect intersect (page_clip) then optional mask multiply.
#[derive(Debug, Clone)]
pub struct CoveragePlan {
    pub page_id: String,
    pub page_index: usize,
    /// Target size in physical px (ceil(page size × scale)).
    pub width_px: u32,
    pub height_px: u32,
    /// Logical page size in canvas units (the ortho projection uses THIS;
    /// the physical pixel scale comes solely from the target size —
    /// render-semantics §2).
    pub page_size: [f64; 2],
    /// Export scale factor (pixelRatio).
    pub scale: f64,
    /// Clear color: white opaque or transparent (0,0,0,0).
    pub transparent: bool,
    /// MSAA sample count for the GPU executor (1 on backdrop pages).
    pub sample_count: u32,
    /// Canvas-space camera origin (= page position; zoom 1).
    pub camera: [f64; 2],
    pub items: Vec<DrawItem>,
    /// M3d interface-change note (SEAMS.md change log; additive): per-owner
    /// mask coverage plans, indexed by `MaskRef::mask_id`. Ancestor masks
    /// always precede descendants. Empty when the page has no mask owners.
    pub masks: Vec<MaskPlan>,
    /// Live-viewport: when `Some`, the wgpu executor
    /// projects the whole plan with this editor camera (N visible pages under
    /// ONE view-projection matrix into a screen-sized target) instead of the
    /// page-local export VP. `width_px/height_px` are then the device-pixel
    /// viewport target (`viewport × dpr`), and `items`/`masks` are in
    /// canvas-absolute space (page positions baked into every model matrix).
    /// `None` keeps the byte-stable page-local export/editor-snapshot path.
    pub camera_frame: Option<CameraFrameView>,
    /// Live-viewport: clear color (the editor canvas
    /// backdrop shown in the gaps around/between pages in the camera frame).
    /// `None` keeps the export clear (opaque white, or transparent).
    pub clear_rgba: Option<[f32; 4]>,
    /// Tiled export: the raster band the tile projects through and the tile's
    /// place in it (`RegionViewport`). `None` for every other plan.
    pub region_viewport: Option<RegionViewport>,
}

/// Tiled export: how a tile projects — through a `raster_w x raster_h`
/// window of the page's output raster (the tile's band), under a viewport
/// shifted so the tile at `(shift_x, shift_y)` inside the band lands on the
/// target.
///
/// A tile could project through its own translated view-projection
/// (`camera_frame`), and it did: that maps canvas space to the same pixels,
/// but with f32 coefficients that round differently from the neighbor's. On
/// tessellated edges the difference, though tiny, moves a vertex across a
/// rasterizer sub-pixel step now and then, flipping one MSAA sample. A flip is
/// one 4x-sample step under source-over, but a backdrop-reading blend
/// (overlay, hard-light, the burns, hue, …) amplifies it by its slope, and
/// stacked blends compound it. Tiles
/// of one band share its matrix; only an integer viewport shift separates
/// their rasterizations, so no seam opens between them. A band covering the
/// whole page projects with the page's own matrix, which makes its tiles
/// integer shifts of the single-target render too.
///
/// Bands are bounded ([`region_band`]) because the viewport overhangs the
/// target. Measured on Apple M4 Pro (Metal): primitives are clipped only
/// beyond a 1.5x guard band around the viewport, and varyings interpolate as
/// if any vertex left of / above -4096 px in window space sat AT -4096 —
/// coverage stays exact, but gradients, SDF edges and UVs of a primitive
/// reaching past it shade wrong. So a tile's shift plus a quarter of its
/// band's extent must stay within 4096 px.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionViewport {
    /// The band's projection; `None` = the page's own view-projection (the
    /// band is the whole page).
    pub band: Option<CameraFrameView>,
    pub shift_x: u32,
    pub shift_y: u32,
    pub raster_w: u32,
    pub raster_h: u32,
}

/// See [`RegionViewport`]: the window-space reach below 0 that interpolates
/// correctly, and the guard band's overhang as a fraction (1/4) of the
/// viewport extent.
const REGION_WINDOW_REACH_PX: u32 = 4096;

/// On one axis of a tile grid with `step` px tiles over a `full` px raster:
/// the band `(origin, len)` holding the tile at `origin`. Consecutive tiles
/// are grouped `k` to a band, the most for which every tile's shift plus a
/// quarter of the band stays within the window reach: `(k-1)*step + k*step/4
/// <= 4096`.
pub fn region_band(origin: u32, step: u32, full: u32) -> (u32, u32) {
    let step = step.max(1);
    let k = ((REGION_WINDOW_REACH_PX + step) * 4 / (step * 5)).max(1);
    let len = k * step;
    let band = origin / len * len;
    (band, len.min(full - band))
}

/// Live-viewport: the editor camera that projects a multi-page
/// `CoveragePlan` into the visible viewport. Mirrors
/// `createViewProjectionMatrix(cameraX, cameraY, zoom, viewportW, viewportH)`
/// (`WebGPUCoordinateUtils.ts`): a canvas-absolute point `(X, Y)` lands at
/// device pixel `((X − cameraX) × zoom × dpr, (Y − cameraY) × zoom × dpr)`,
/// where `dpr = width_px / viewport_w`. `viewport_w/h` are LOGICAL (CSS) px;
/// the device-pixel scale comes solely from the `width_px/height_px` target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraFrameView {
    pub camera_x: f64,
    pub camera_y: f64,
    pub zoom: f64,
    pub viewport_w: f64,
    pub viewport_h: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_bands_group_tiles_within_the_window_reach() {
        // (step, full, tiles per band)
        for (step, full, k) in [
            (256u32, 9600u32, 13u32),
            (384, 9600, 9),
            (1024, 20000, 4),
            (4096, 16384, 1),
        ] {
            for origin in (0..full).step_by(step as usize) {
                let (band, len) = region_band(origin, step, full);
                assert_eq!(band % (k * step), 0, "step {step} origin {origin}");
                assert!(band <= origin && origin + step.min(full - origin) <= band + len);
                assert!(band + len <= full);
                // shift + a quarter of the band stays within the reach.
                assert!(
                    (origin - band) + len / 4 <= REGION_WINDOW_REACH_PX,
                    "step {step} origin {origin}"
                );
            }
        }
        // A page that fits one band is the whole page.
        assert_eq!(region_band(768, 256, 1000), (0, 1000));
    }

    #[test]
    fn instance_abis_match_manifest() {
        let abi = &moda_shaders::manifest().instance_abi;
        assert_eq!(
            abi.size_bytes as usize,
            core::mem::size_of::<PaintInstance>()
        );
        let off = &abi.offsets_floats;
        assert_eq!(off["modelMatrix"], 0);
        assert_eq!(off["fillColor"], 16);
        assert_eq!(off["strokeColor"], 20);
        assert_eq!(off["pathInfo"], 24);
        assert_eq!(off["bounds"], 28);
        assert_eq!(off["pageClipBounds"], 32);
        assert_eq!(off["shaderParams"], 36);
        assert_eq!(off["colorData"], 100);
        assert_eq!(abi.shader_params_capacity_floats, 64);
        assert_eq!(abi.color_data_capacity_floats, 64);
        // Rust-side field offsets agree with the manifest float offsets.
        assert_eq!(core::mem::offset_of!(PaintInstance, fill_color), 16 * 4);
        assert_eq!(core::mem::offset_of!(PaintInstance, stroke_color), 20 * 4);
        assert_eq!(core::mem::offset_of!(PaintInstance, path_info), 24 * 4);
        assert_eq!(core::mem::offset_of!(PaintInstance, bounds), 28 * 4);
        assert_eq!(
            core::mem::offset_of!(PaintInstance, page_clip_bounds),
            32 * 4
        );
        assert_eq!(core::mem::offset_of!(PaintInstance, shader_params), 36 * 4);
        assert_eq!(core::mem::offset_of!(PaintInstance, color_data), 100 * 4);
    }

    #[test]
    fn mask_bind_group_indices_match_manifest() {
        let c = &moda_shaders::manifest().pinned_constants;
        assert_eq!(c.mask_bind_group_index.default, MASK_BIND_GROUP_INDEX);
        assert_eq!(
            c.mask_bind_group_index.image_backed,
            MASK_BIND_GROUP_INDEX_IMAGE_BACKED
        );
    }

    #[test]
    fn blend_modes_exist_in_manifest_as_data() {
        let m = moda_shaders::manifest();
        for mode in BlendMode::ALL {
            assert!(m.blend_modes.contains_key(mode.manifest_key()));
            assert_eq!(BlendMode::from_node_value(Some(mode.node_value())), mode);
            // Only the exact fixed-function states skip the split.
            assert_eq!(
                mode.split_id().is_none(),
                matches!(
                    mode,
                    BlendMode::Normal | BlendMode::Multiply | BlendMode::Screen
                )
            );
        }
        assert_eq!(BlendMode::from_node_value(Some("bogus")), BlendMode::Normal);
        assert_eq!(BlendMode::from_node_value(None), BlendMode::Normal);
        // The S10/DECISION catch: overlay is NOT normal.
        assert_eq!(m.blend_modes["overlay"].color.src_factor, "src-alpha");
    }
}
