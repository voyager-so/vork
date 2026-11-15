//! frame.rs — per-page frame orchestration: display list → CoveragePlan →
//! two-phase GPU execution → readback (render-semantics.md §1, §10).
//!
//! Pass structure (raster export path, `WebGPUExportCoordinator.ts:592-776`):
//!   clear (white opaque or transparent) → page-fill quad @ depth 0.99999 →
//!   opaque tessellated bodies (depth-write, submission order) →
//!   transparent bodies grouped by ascending integer z bucket
//!   (shadows → bodies → paths → text inside each bucket) → resolve MSAA →
//!   readback. Backdrop ops split the pass (seams::BackdropSplit; hook wired,
//!   M3e implements the effect chain).
//!
//! All pipeline state (blend/depth/vertex layouts) comes from the extraction
//! manifest — never hand-coded (DECISION §1.3).


use anyhow::{anyhow, bail, Context, Result};
use moda_scene::types::PaintKind;
use moda_scene::Scene;
use moda_scene::value::Value;

use crate::color::{color_to_normalized_rgb, parse_color, Rgba};
use crate::depth;
use crate::device::GpuContext;
use crate::paint;
use crate::seams::{
    BlendMode, BodyPaint, CoveragePlan, DrawItem, PageInstance, PaintInstance, Phase,
    SolidPathInstance, TessellatedGeometry, INFINITE_CLIP,
};
use crate::shape_to_path;
use crate::tess::{self, moda_engine_path_types as pt};

// ---------------------------------------------------------------------------
// View-projection (port of WebGPUCoordinateUtils.ts:32-82 + WebGPUMath.ts)
// ---------------------------------------------------------------------------

/// `createViewProjectionMatrix`: VP = Ortho(0, w, h, 0, −1, 1) · S(zoom) ·
/// T(−cameraX, −cameraY). Column-major, f32 storage (JS Float32Array), f64
/// arithmetic — mirrored here.
pub fn view_projection(
    camera_x: f64,
    camera_y: f64,
    zoom: f64,
    viewport_w: f64,
    viewport_h: f64,
) -> [f32; 16] {
    let translation = translation_matrix(-camera_x, -camera_y, 0.0);
    let scale = scale_matrix(zoom, zoom, 1.0);
    let view = multiply4x4(&scale, &translation);
    // orthographicInto(0, w, h, 0, -1, 1): width = w, height = top - bottom = -h
    let (left, right, bottom, top, near, far) = (0.0f64, viewport_w, viewport_h, 0.0, -1.0, 1.0);
    let width = right - left;
    let height = top - bottom;
    let depth_range = far - near;
    let mut proj = [0.0f32; 16];
    proj[0] = (2.0 / width) as f32;
    proj[5] = (2.0 / height) as f32;
    proj[10] = (-2.0 / depth_range) as f32;
    proj[12] = (-(right + left) / width) as f32;
    proj[13] = (-(top + bottom) / height) as f32;
    proj[14] = (-(far + near) / depth_range) as f32;
    proj[15] = 1.0;
    multiply4x4(&proj, &view)
}

fn translation_matrix(x: f64, y: f64, z: f64) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[12] = x as f32;
    m[13] = y as f32;
    m[14] = z as f32;
    m[15] = 1.0;
    m
}

fn scale_matrix(x: f64, y: f64, z: f64) -> [f32; 16] {
    let mut m = [0.0f32; 16];
    m[0] = x as f32;
    m[5] = y as f32;
    m[10] = z as f32;
    m[15] = 1.0;
    m
}

/// `WebGPUMath.multiply4x4Into` (column-major; f32 operands, f64 products).
fn multiply4x4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0f64;
            for k in 0..4 {
                sum += a[k * 4 + j] as f64 * b[i * 4 + k] as f64;
            }
            out[i * 4 + j] = sum as f32;
        }
    }
    out
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

// ---------------------------------------------------------------------------
// Plan building (display list → CoveragePlan)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Export scale factor (pixelRatio).
    pub scale: f64,
    pub transparent: bool,
    /// Keep the authored page background while clearing outside it to alpha 0.
    /// Main Edit compositions use this; ordinary transparent exports omit it.
    pub preserve_page_fill: bool,

    /// 4 (export) or 1 (screenshot config / backdrop pages).
    pub sample_count: u32,
    /// Deterministic shader clock (M3f): substituted for descriptor clock
    /// params at pack time (production export context pins 0 —
    /// `WebGPUPathRenderer.ts:367`; goldens captured at 0).
    pub shader_time_s: f64,
    /// Node ids excluded from the render (`--exclude`).
    pub exclude: Vec<String>,
    /// Node ids the render is ISOLATED to (`--isolate`) — the positive
    /// counterpart to [`Self::exclude`], and the native half of the browser
    /// `isolateNodeIds` capture option (`renderers/webgpu/exportExclusion.ts`).
    /// A paint item survives only when its node, or one of its group /
    /// transient-host ancestors, is listed here
    /// (`moda_scene::Scene::is_node_isolated`).
    ///
    /// EMPTY (the default) means NO isolation — every node renders, and the
    /// filter is short-circuited before any ancestry walk, so an unset
    /// `isolate` produces a byte-identical plan to before the option existed.
    ///
    /// Per-element alpha plates pair this with `transparent: true`: isolation
    /// filters node items, never the page-background fill.
    pub isolate: Vec<String>,
    /// M4d: `--best-effort-text` (PLAN §2). It no longer decides whether text draws: every node the vector lane
    /// declines (complex scripts — Arabic, Hebrew, Devanagari, Thai… — RTL paragraphs, coverage gaps, emoji) draws
    /// through the Degraded tier either way (`text_draw::build_richtext_item`; pinned by
    /// `complex_script_renders_via_cosmic_on_strict_lane`). It is part of the text cache key, and an animated
    /// export's missing-font abort is off with it (`moda-cli` `abort_if_font_missing`).
    pub best_effort_text: bool,
    /// Whether richtext (text-as-mask) coverage is built: `None` (default)
    /// and `Some(true)` build it on every page; `Some(false)` drops text masks
    /// (their children render unmasked). Exports built it on page index 0
    /// only (a copied browser quirk, Q-95), so page 2 of a PDF differed from
    /// the editor, which clips on every page.
    pub text_masks_available: Option<bool>,
    /// The working colour of the engine the plan is for (`Engine::new_with_color`), when it blends in another
    /// encoding than sRGB. The plan is that engine's: a luminance mask that is not one layer of one
    /// solid colour is then planned as the mask's rendered picture, which the engine reduces once
    /// (`effects::PlanMasks::in_working_color`).
    pub working: Option<moda_shaders::WorkingColor>,
}

/// Stable prefix of the plan warning raised when `--isolate` meets a page with
/// active backdrop content (see the emit site in `build_coverage_plan_inner`).
/// A constant because the CLI matches on it to turn the warning into a typed
/// exit-2 decline — a string both sides spell by hand would drift.
pub const ISOLATE_BACKDROP_WARNING: &str = "isolate-unsupported-backdrop";

/// Stable prefixes of the region-render (tiled export) refusals for pages
/// whose pixels sample the TARGET — a tile of such a page would seam at tile
/// edges instead of matching the single-target render. Constants so callers
/// (the tiled driver, the backend's typed-decline mapping) match on them
/// rather than hand-spelling the string.
pub const REGION_BACKDROP_ERROR: &str = "region-unsupported-backdrop";
pub const REGION_LAYER_BLUR_ERROR: &str = "region-unsupported-layer-blur";

/// A sub-rect of a page's OUTPUT raster, in output pixels at the render's
/// `opts.scale` — the tile-grid coordinate space. Must lie inside the page's
/// full output size (`ceil(page_dim x scale)` per axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRegionPx {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// The raster band (containing this region) the tile projects through
    /// (`seams::region_band`); tiles sharing a band cannot seam.
    pub band_x: u32,
    pub band_y: u32,
    pub band_w: u32,
    pub band_h: u32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            scale: 1.0,
            transparent: false,
            preserve_page_fill: false,
            sample_count: 4,
            shader_time_s: 0.0,
            exclude: Vec::new(),
            isolate: Vec::new(),
            best_effort_text: false,
            text_masks_available: None,
            working: None,
        }
    }
}

/// What a plan build could not draw as authored, split by what the host owes
/// the user for it.
///
/// `skipped` is the TOLERATED bucket: draws declined or degraded by design —
/// unsupported node kinds, degraded-text tiers, an unsupported transparent
/// backdrop forced opaque, a node whose body build errored. Every export lane
/// prints these and ships the page; the feature manifest / assess gates own
/// declining the scene classes that must not ship.
///
/// `dropped` is the REFUSED bucket: geometry the engine REFUSED and
/// substituted with nothing — a VISIBLE fill emptied because a coordinate was
/// non-finite or the fill ran past `MAX_FILL_VERTICES` (or its cell/work
/// bounds, `tess::fill_cells`), a clip mask given ZERO coverage for the same
/// reasons (`effects::mask`). The same refusal on a fill whose resolved paint
/// is invisible (an alpha-0 solid fill, a line body or open arrowhead — Q-51)
/// paints nothing either way and files under `skipped` with the same message.
/// Nothing in the manifest anticipates these (the outline is a plain path), so
/// the plan is the first and only place they surface. Every lane draws the rest
/// of the page and names the node: `moda-render` warns and lists it under
/// `fills_dropped` in the resource report (it used to exit 3 and lose the whole
/// page to one over-complex path); live viewports (wasm, mobile) surface the
/// line as a warning — never blank, never panic.
#[derive(Debug, Clone, Default)]
pub struct PlanWarnings {
    pub skipped: Vec<String>,
    pub dropped: Vec<String>,
}

impl PlanWarnings {
    pub fn is_empty(&self) -> bool {
        self.skipped.is_empty() && self.dropped.is_empty()
    }

    /// Both buckets, for hosts that surface every plan warning as one list
    /// (the wasm editor's `warnings` array, mobile logcat).
    pub fn host_lines(&self) -> impl Iterator<Item = &String> {
        self.skipped.iter().chain(self.dropped.iter())
    }

    /// Refuse plans that changed authored coverage. A legitimately empty draw
    /// (for example a dash gap) needs no refusal.
    pub fn ensure_drawn(&self) -> Result<()> {
        if let Some(line) = self.dropped.iter().chain(
            self.skipped.iter().filter(|line| !line.ends_with("): empty geometry"))
        ).next() {
            bail!("Moda's engine did not draw part of this frame: {line}");
        }
        Ok(())
    }

    pub fn absorb(&mut self, other: PlanWarnings) {
        self.skipped.extend(other.skipped);
        self.dropped.extend(other.dropped);
    }
}

/// rust-edit-incremental-z4: where one primary display-list node's DrawItems live
/// in a built `CoveragePlan`. Recorded during a full plan build so the
/// incremental patcher (`residency::build_plan_retained`) can rebuild ONLY the
/// edited nodes' items and splice them in place. `primary` is the node's
/// display-list `PaintItem` clone — its STRUCTURAL fields (node_id, kind,
/// is_primary, fallback_reasons) drive the rebuild, while the geometry/transform
/// is re-read fresh from the (edited) scene, so a stale `primary` from before a
/// property edit is still correct as long as the edit was non-structural (the
/// caller's `structural_gen` watermark guarantees this).
#[derive(Debug, Clone)]
pub struct NodeSpan {
    pub node_id: String,
    /// Start index of this node's items in `CoveragePlan::items`.
    pub start: usize,
    /// Number of items this node contributes (contiguous from `start`).
    pub len: usize,
    /// The integer render-z the node was built at (== its DL primary position).
    pub z: u64,
    /// Integer z the node consumed (1 for a leaf, N for a blend, 0 if skipped).
    pub z_consumed: u64,
    pub primary: moda_scene::types::PaintItem,
    pub placement: Option<TranslationPlacement>,
}

/// How one plain path's span follows its node's placement without being prepared again. Certified by the
/// ordinary local-path emission route, not inferred from a coincidentally equal matrix in a synthetic draw.
///
/// The span's items depend on the node's canvas affine through their model matrices and through the
/// tessellation band its scale selects, and on its opacity through the solid colours' alpha and the phase.
/// A new placement in the same band, with the node on the same side of opaque, rewrites those values; anything
/// else is prepared again by the planner.
#[derive(Debug, Clone)]
pub struct TranslationPlacement {
    source: String,
    /// The view band the plan was built in and the band the node's scale raised it to.
    view_band: i32,
    band: i32,
    /// The node opacity the span holds.
    opacity_bits: u64,
    /// The solid fill and stroke alphas before opacity, when every item is a solid body whose alphas are
    /// exactly those times the opacity. `None`: the span follows moves but not opacity.
    alphas: Option<[f64; 2]>,
}

fn local_path_placement(scene: &Scene, node: &moda_scene::types::RawNode) -> Option<TranslationPlacement> {
    let canonical = scene.node(node.id())?;
    let local_fields = |n: &moda_scene::types::RawNode| n.node_type() == "path"
        && n.fields().keys().all(|key| matches!(key.as_str(),
            "id" | "type" | "name" | "x" | "y" | "width" | "height" | "rotation"
            | "scale" | "scaleX" | "scaleY" | "opacity" | "points" | "closed"
            | "fillRule" | "fills" | "strokes" | "visible" | "hidden" | "zIndex"))
        && tess::outline::brush_id(n).is_none()
        && !crate::pattern_paint::has_pattern_layer(n)
        && !paint::layers::has_entry_effects(n);
    if !local_fields(canonical) || !local_fields(node)
        || scene.transient_by_id().contains_key(node.id())
        || moda_scene::scene::paint_transform_id(node) != node.id() { return None; }
    let a = scene.node_canvas_affine(node.id())?;
    if ![a.a, a.b, a.c, a.d].into_iter().all(f64::is_finite) { return None; }
    let view_band = tess::outline::view_band();
    let fill = paint::gradient::solid_fill_color(paint::gradient::fill_paint(node).as_ref());
    let top_stroke = paint::gradient::top_stroke(node);
    let stroke = paint::gradient::solid_stroke_color(top_stroke.as_ref().map(|s| &s.paint));
    Some(TranslationPlacement {
        source: node.id().to_string(),
        view_band,
        band: tess::outline::node_band(view_band, tess::outline::affine_max_scale(&a)),
        opacity_bits: node.f64("opacity").unwrap_or(1.0).to_bits(),
        alphas: Some([fill.a, stroke.a]),
    })
}

fn translation_body(item: &DrawItem) -> bool {
    matches!(item, DrawItem::Body { mask: None, paint: BodyPaint::Solid
        | BodyPaint::Gradient(_) | BodyPaint::PostShaderGradientStroke(_) | BodyPaint::Pattern(_), .. })
}

/// The alphas a solid body holds for `alphas` under `opacity`, as `build_body_item_inner` computes them.
fn placed_alphas(alphas: [f64; 2], opacity: f64) -> [f32; 2] {
    [(alphas[0] * opacity) as f32, (alphas[1] * opacity) as f32]
}

/// Check a fresh span against its recipe: every item is a movable body at the node's matrix. The opacity
/// part of the recipe is kept only when every item is a solid body holding exactly the recipe's alphas.
fn certify_placement(scene: &Scene, mut recipe: TranslationPlacement, items: &[DrawItem]) -> Option<TranslationPlacement> {
    let a = scene.node_canvas_affine(&recipe.source)?;
    let matrix = affine_to_matrix(&a);
    let moves = !items.is_empty() && items.iter().all(|item| {
        let DrawItem::Body { instance, paint, .. } = item else { return false; };
        translation_body(item) && instance.model_matrix == matrix && match paint {
            BodyPaint::Gradient(g) | BodyPaint::PostShaderGradientStroke(g) => g.model_matrix == matrix,
            BodyPaint::Pattern(p) => p.instance.model_matrix == matrix,
            BodyPaint::Solid => true,
            _ => false,
        }
    });
    if !moves { return None; }
    let opacity = f64::from_bits(recipe.opacity_bits);
    recipe.alphas = recipe.alphas.filter(|alphas| {
        let [fill, stroke] = placed_alphas(*alphas, opacity);
        opacity > 0.0 && items.iter().all(|item| matches!(item, DrawItem::Body { paint: BodyPaint::Solid, instance, .. }
            if instance.fill_color[3].to_bits() == fill.to_bits() && instance.stroke_color[3].to_bits() == stroke.to_bits()))
    });
    Some(recipe)
}

/// A recipe for a span the planner prepared again in place: the node's band, opacity and alphas as they are
/// now, certified against the new items. The span was unclipped when its recipe was first certified; it is
/// kept only while its items still carry the page clip and no mask.
pub(crate) fn recertify_placement(scene: &Scene, recipe: TranslationPlacement, items: &[DrawItem], page_clip: [f32; 4]) -> Option<TranslationPlacement> {
    let node = scene.effective_node(&recipe.source)?;
    if !items.iter().all(|item| matches!(item, DrawItem::Body { instance, mask: None, .. } if instance.page_clip_bounds == page_clip)) {
        return None;
    }
    let mut fresh = local_path_placement(scene, node)?;
    // The view band is the plan's: a patch runs outside the build that set it.
    let a = scene.node_canvas_affine(&recipe.source)?;
    fresh.view_band = recipe.view_band;
    fresh.band = tess::outline::node_band(recipe.view_band, tess::outline::affine_max_scale(&a));
    certify_placement(scene, fresh, items)
}

/// A span's values at its node's current placement, computed without touching the plan.
pub(crate) struct Placed {
    matrix: [f32; 16],
    /// New solid alphas and the opacity they came from, when the opacity changed.
    alphas: Option<([f32; 2], u64)>,
}

/// Whether `recipe`'s span can follow the node's current placement, and with what values. `None`: the planner
/// prepares the node again (another band, an opacity that crosses opaque or reaches zero, a paint that folds
/// opacity in some other way, or a node that is gone).
pub(crate) fn placed_span(scene: &Scene, recipe: &TranslationPlacement, items: &[DrawItem]) -> Option<Placed> {
    let a = scene.node_canvas_affine(&recipe.source)?;
    let opacity = scene.effective_node(&recipe.source)?.f64("opacity").unwrap_or(1.0);
    placed_at(&a, opacity, recipe, items)
}

/// [`placed_span`] for a recipe whose source node is the scene row `row`: the matrix is read from the scene's
/// world column (`Scene::row_canvas_affine`), the same numbers the walk by name composes, and no name is looked
/// up. `None` also when the column does not answer for that row.
pub(crate) fn placed_span_row(scene: &Scene, row: u32, recipe: &TranslationPlacement, items: &[DrawItem]) -> Option<Placed> {
    let a = scene.row_canvas_affine(row)?;
    placed_at(&a, scene.row(row)?.f64("opacity").unwrap_or(1.0), recipe, items)
}

/// The values of a span whose node sits at canvas matrix `a` with `opacity`, when its recipe admits them.
fn placed_at(a: &moda_scene::types::Affine, opacity: f64, recipe: &TranslationPlacement, items: &[DrawItem]) -> Option<Placed> {
    if ![a.a, a.b, a.c, a.d, a.e, a.f].into_iter().all(f64::is_finite) || !items.iter().all(translation_body)
        || tess::outline::node_band(recipe.view_band, tess::outline::affine_max_scale(a)) != recipe.band { return None; }
    let alphas = if opacity.to_bits() == recipe.opacity_bits { None } else {
        let old = f64::from_bits(recipe.opacity_bits);
        // Both drawn, and on the same side of opaque: the phase and the display list stand.
        if !(opacity > 0.0 && old > 0.0 && (opacity < 1.0) == (old < 1.0)) { return None; }
        Some((placed_alphas(recipe.alphas?, opacity), opacity.to_bits()))
    };
    Some(Placed { matrix: affine_to_matrix(a), alphas })
}

/// Write [`placed_span`]'s values. Recomputed from the scene's f64 placement; never a delta on rounded f32 data.
pub(crate) fn place_span(recipe: &mut TranslationPlacement, placed: &Placed, items: &mut [DrawItem]) {
    for item in items {
        let DrawItem::Body { instance, paint, .. } = item else { unreachable!("placed_span admits bodies") };
        instance.model_matrix = placed.matrix;
        match paint {
            BodyPaint::Gradient(g) | BodyPaint::PostShaderGradientStroke(g) => g.model_matrix = placed.matrix,
            BodyPaint::Pattern(p) => p.instance.model_matrix = placed.matrix,
            BodyPaint::Solid => {},
            _ => unreachable!("placed_span admits these paints"),
        }
        if let Some(([fill, stroke], _)) = placed.alphas {
            instance.fill_color[3] = fill;
            instance.stroke_color[3] = stroke;
        }
    }
    if let Some((_, bits)) = placed.alphas { recipe.opacity_bits = bits; }
}

/// Move a span to its node's current placement when its recipe allows it.
pub(crate) fn translate_span(scene: &Scene, recipe: &mut TranslationPlacement, items: &mut [DrawItem]) -> bool {
    match placed_span(scene, recipe, items) {
        Some(placed) => { place_span(recipe, &placed, items); true }
        None => false,
    }
}

/// rust-edit-incremental-z4: page-level constants + per-node spans captured by a
/// full plan build, so the incremental patcher can re-derive a single node's
/// items without rebuilding the display list or re-deriving page geometry.
#[derive(Debug, Clone, Default)]
pub struct PlanBuildAux {
    pub spans: Vec<NodeSpan>,
    pub page_clip: [f32; 4],
    pub transparent: bool,
    pub has_backdrop: bool,
    /// Like `has_backdrop`, a layer-blur page's plan is
    /// camera-dependent (capture regions in target px) and its diverted spans
    /// aren't patchable in isolation — excluded from incremental seeding.
    pub has_layer_blur: bool,
    /// Viewport culling: the display-list nodes the build skipped as out of the cull
    /// window (range roots; a skipped parent's descendants are not listed). The
    /// incremental patcher must not treat these as "no items".
    pub culled_roots: Vec<String>,
}

/// Scale RFC stage C: one rebuildable piece of a camera-plan segment — a
/// primary display-list item with the non-primary items after it, a subtree
/// capture's whole range (its captures swallow its descendants' items), or a
/// culled range. Units tile the segment's display list and items in order, so
/// the retained camera plan (`camera_slot`) re-plans an edited node by
/// re-running [`append_node_draw_items`] over its unit's range at the unit's
/// render-z, and splices the result in.
#[derive(Debug, Clone, Default)]
pub struct PlanUnit {
    /// Display-list range `[dl_start, dl_end)` of the segment.
    pub dl_start: usize,
    pub dl_end: usize,
    /// Items, relative to the segment's first node item (after its page fill).
    pub item_start: usize,
    pub item_len: usize,
    /// Render-z at the unit's start and the z it consumed.
    pub z: u64,
    pub z_consumed: u64,
    /// Masks in the segment's table before the unit was planned.
    pub masks_before: usize,
    /// Skipped by the cull window (no items; still consumed its z).
    pub culled: bool,
    /// A range inside it (a subtree capture's) was skipped by the cull window.
    pub part_culled: bool,
    /// The warnings its planning raised.
    pub skipped: Vec<String>,
    pub dropped: Vec<String>,
    /// `camera_slot` flags: the unit's items depend on the exact camera (a backdrop
    /// or blend split) or on the capture view (a layer capture).
    pub flags: u8,
}

/// Records [`PlanUnit`]s while [`append_node_draw_items`] plans a display list.
#[derive(Debug, Default)]
pub(crate) struct UnitRecorder {
    pub units: Vec<PlanUnit>,
    /// The open unit, with the warning counts at its start.
    open: Option<(PlanUnit, usize, usize)>,
    /// `items.len()` when the recording started (unit item offsets are relative).
    base: usize,
}

impl UnitRecorder {
    fn start(&mut self, items_len: usize) {
        self.base = items_len;
    }

    fn open_at(
        &mut self,
        dl_index: usize,
        items_len: usize,
        z: u64,
        warnings: &PlanWarnings,
        masks_len: usize,
    ) {
        self.close(dl_index, items_len, z, warnings);
        self.open = Some((
            PlanUnit {
                dl_start: dl_index,
                item_start: items_len - self.base,
                z,
                masks_before: masks_len,
                ..PlanUnit::default()
            },
            warnings.skipped.len(),
            warnings.dropped.len(),
        ));
    }

    fn mark_culled(&mut self, whole: bool) {
        if let Some((u, _, _)) = self.open.as_mut() {
            if whole {
                u.culled = true;
            } else {
                u.part_culled = true;
            }
        }
    }

    fn close(&mut self, dl_end: usize, items_len: usize, z: u64, warnings: &PlanWarnings) {
        if let Some((mut u, s0, d0)) = self.open.take() {
            u.dl_end = dl_end;
            u.item_len = items_len - self.base - u.item_start;
            u.z_consumed = z - u.z;
            u.skipped = warnings.skipped[s0..].to_vec();
            u.dropped = warnings.dropped[d0..].to_vec();
            self.units.push(u);
        }
    }
}

/// Which part of a combined camera plan a [`CameraSegment`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Page(usize),
    CanvasRoot,
}

/// Scale RFC stage C: one page's (or the freeform canvas root's) share of a
/// combined camera plan, as its build planned it — everything
/// [`rebuild_segment_range`] needs to re-plan a range of it in place.
#[derive(Debug, Clone)]
pub struct CameraSegment {
    pub kind: SegmentKind,
    /// The display list the segment was planned from.
    pub dl: std::sync::Arc<moda_scene::DisplayList>,
    pub page_clip: [f32; 4],
    /// The `transparent` its node items were planned with.
    pub node_transparent: bool,
    /// Present when the segment splits (backdrop effects, blend splits):
    /// whether rect / ellipse bodies keep the SDF path under the split context.
    pub backdrop: Option<bool>,
    /// The segment plans layer captures.
    pub layer_blur: bool,
    /// Text masks are available to its mask table.
    pub text_masks: bool,
    /// Where its items start in the combined plan, how many lead the units
    /// (the page fill), and how many it has in all.
    pub item_start: usize,
    pub prefix: usize,
    pub item_len: usize,
    /// Where its masks start in the combined table, and how many.
    pub mask_start: usize,
    pub mask_len: usize,
    pub z_start: u64,
    pub z_end: u64,
    pub units: Vec<PlanUnit>,
    /// Warnings raised outside its units: before them (page-level) and by its
    /// mask table after them.
    pub head_warnings: PlanWarnings,
    pub tail_warnings: PlanWarnings,
}

/// What a camera plan build recorded for the retained camera plan: its
/// segments in plan order.
#[derive(Debug, Clone, Default)]
pub struct CameraPlanAux {
    pub segments: Vec<CameraSegment>,
}

/// A range of a segment re-planned by [`rebuild_segment_range`].
pub(crate) struct RangeBuild {
    pub items: Vec<DrawItem>,
    /// The range's own (scratch) mask table, in first-use order, and the
    /// warnings building it raised.
    pub masks: Vec<crate::seams::MaskPlan>,
    pub mask_warnings: PlanWarnings,
    pub end_z: u64,
    /// Units relative to the range (display-list indices from 0).
    pub units: Vec<PlanUnit>,
}

/// Scale RFC stage C: re-plan `dl_items` (a unit-aligned range of `seg`'s display
/// list, or new items for it) from render-z `z_start`, exactly as the segment's
/// full build plans them — the same node loop under the same page clip, split
/// and capture contexts (projected for `cam` and `capture_view`) and mask
/// configuration — so a spliced range is byte-identical to a from-scratch plan.
/// `cull` culls the range to that window as a full build would — exactly, when
/// the range is whole top-level subtrees (cull decisions never cross one); `None`
/// draws everything the range holds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rebuild_segment_range(
    scene: &Scene,
    seg: &CameraSegment,
    dl_items: &[moda_scene::types::PaintItem],
    opts: &RenderOptions,
    cam: &CameraParams,
    capture_view: Option<CaptureView>,
    caches: Option<CacheCtx<'_>>,
    z_start: u64,
    cull: Option<&crate::cull::Aabb>,
) -> RangeBuild {
    let cam_backdrop = cam.backdrop_projection();
    let backdrop_ctx = seg.backdrop.map(|keep_sdf| {
        crate::backdrop::PlanCtx::new_camera(
            scene,
            cam_backdrop.width_px,
            cam_backdrop.height_px,
            seg.page_clip,
            cam_backdrop.vp,
        )
        .over_transparent(opts.transparent)
        .keeping_shape_sdf(keep_sdf)
    });
    let layer_blur_ctx = seg.layer_blur.then(|| {
        let p = capture_view.map(|v| v.projection()).unwrap_or(cam_backdrop);
        crate::backdrop::PlanCtx::new_camera(scene, p.width_px, p.height_px, seg.page_clip, p.vp)
            .at_shader_time(opts.shader_time_s)
    });
    let mut plan_masks = crate::effects::PlanMasks::with_text_masks(seg.text_masks)
        .with_vector_text_masks(true)
        .in_working_color(opts.working.is_some())
        .at_shader_time(opts.shader_time_s);
    // A range's display list is not the segment's: keyed apart so the cull-index
    // cache (per display list) never serves the segment's for it.
    let dl = moda_scene::DisplayList {
        page_id: RANGE_DL_KEY.to_string(),
        page_index: seg.dl.page_index,
        items: dl_items.to_vec(),
    };
    let mut items = Vec::new();
    let mut warnings = PlanWarnings::default();
    let mut recorder = UnitRecorder::default();
    let mut cull_out = CullOut::default();
    let end_z = append_node_draw_items(
        scene,
        &dl,
        opts,
        seg.node_transparent,
        seg.page_clip,
        backdrop_ctx.as_ref(),
        layer_blur_ctx.as_ref(),
        &mut plan_masks,
        caches,
        &mut items,
        &mut warnings,
        z_start,
        None,
        cull.map(|w| (w, &mut cull_out)),
        Some(&mut recorder),
    );
    let mut mask_warnings = PlanWarnings::default();
    let masks = plan_masks.into_masks(&mut mask_warnings);
    RangeBuild {
        items,
        masks,
        mask_warnings,
        end_z,
        units: recorder.units,
    }
}

/// The display-list key of a range [`rebuild_segment_range`] re-plans: never
/// cached (`append_node_draw_items`' cull index).
const RANGE_DL_KEY: &str = "\u{0}range";

/// What a culled plan build left out (`crate::cull`).
#[derive(Debug, Clone, Default)]
pub struct CullOut {
    /// Culled range roots, in display-list order.
    pub roots: Vec<String>,
    /// Primary display-list items skipped (roots and their descendants).
    pub items: usize,
    /// Primary display-list items considered.
    pub considered: usize,
    /// Nodes whose items were copied from an earlier culled build (`PlanCaches::node_items`).
    pub copied_nodes: usize,
}

/// A cull window plus where to report what it left out.
pub type CullCtx<'a> = (&'a crate::cull::Aabb, &'a mut CullOut);

pub fn build_coverage_plan(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
) -> Result<(CoveragePlan, PlanWarnings)> {
    build_coverage_plan_cached(scene, page_index, opts, None)
}

/// Task #95 Stage-1: plan building with optional per-node residency caches
/// (R3 tessellation / R4 text / R6 expansion). `None` (the CLI/export
/// default) is byte-identical Stage-0 behavior — the caches are pure
/// memoization of deterministic functions, keyed by deny-list content hashes
/// (residency.rs), so the cached path is byte-identical too; the fuzzer
/// mutation-replay gate (PLAN §3.4.1) enforces that.
pub fn build_coverage_plan_cached(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: Option<(
        &mut crate::residency::PlanCaches,
        &crate::residency::HashMap<String, crate::residency::NodeHashes>,
    )>,
) -> Result<(CoveragePlan, PlanWarnings)> {
    build_coverage_plan_cached_with_dl(scene, page_index, opts, caches, None)
}

/// Reuse the display list built for a frame fingerprint without collecting an
/// incremental editing index that the offscreen renderer does not consume.
pub fn build_coverage_plan_cached_with_dl(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: Option<CacheCtx<'_>>,
    prebuilt_dl: Option<moda_scene::DisplayList>,
) -> Result<(CoveragePlan, PlanWarnings)> {
    // Per-page export/snapshot path: z starts at 0 (each page is its own target).
    let (plan, warnings, _end_z) = build_coverage_plan_inner(
        scene, page_index, opts, caches, None, None, 0, None, prebuilt_dl, None, None, None,
    )?;
    Ok((plan, warnings))
}

/// Tiled export: the coverage plan for ONE tile (`region`, in output pixels)
/// of a page's raster. Identical draw items to [`build_coverage_plan`] — only
/// the target size and the view-projection differ — so a stitched grid of
/// these reproduces the single-target render (the tiled-seam golden pins it).
/// Errors on pages with active backdrop / layer-blur content
/// ([`REGION_BACKDROP_ERROR`] / [`REGION_LAYER_BLUR_ERROR`]): those sample the
/// target and cannot tile without seams.
pub fn build_coverage_plan_region(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    region: PageRegionPx,
) -> Result<(CoveragePlan, PlanWarnings)> {
    build_coverage_plan_region_cached(scene, page_index, opts, region, None)
}

/// [`build_coverage_plan_region`] through plan caches (an export job's, `crate::export_job`).
pub fn build_coverage_plan_region_cached(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    region: PageRegionPx,
    caches: Option<(
        &mut crate::residency::PlanCaches,
        &crate::residency::HashMap<String, crate::residency::NodeHashes>,
    )>,
) -> Result<(CoveragePlan, PlanWarnings)> {
    let (plan, warnings, _end_z) = build_coverage_plan_inner(
        scene,
        page_index,
        opts,
        caches,
        None,
        Some(region),
        0,
        None,
        None,
        None,
        None,
        None,
    )?;
    Ok((plan, warnings))
}

/// The part of a transparent page render that can hold drawn pixels: the box of what `opts` keeps (its
/// `exclude` / `isolate`), from the viewport cull's visual bounds ([`crate::cull`], which never leave out a
/// painted pixel), grown by two pixels and cut to the page raster, as a region in the page's own band (its
/// projection, so the region's pixels are the full render's under an integer viewport shift). `None` when the
/// render is opaque or keeps the page fill, or what it keeps can't be bounded (a brushed stroke, a blend, a node
/// drawn through another's transform).
pub fn drawn_region_px(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
) -> Option<PageRegionPx> {
    if !opts.transparent || opts.preserve_page_fill {
        return None;
    }
    let page = scene.doc().pages.get(page_index)?;
    let (page_x, page_y) = scene.page_position(page_index);
    let (page_w, page_h) = (page.width(), page.height());
    let full_w = crate::export_scale::px_ceil(page_w * opts.scale) as u32;
    let full_h = crate::export_scale::px_ceil(page_h * opts.scale) as u32;
    if full_w == 0 || full_h == 0 {
        return None;
    }
    let dl = moda_scene::build_display_list(scene, page.id());
    let index = crate::cull::CullIndex::build(scene, &dl, |_| 0, |_| None);
    let isolate: Option<std::collections::HashSet<&str>> =
        (!opts.isolate.is_empty()).then(|| opts.isolate.iter().map(String::as_str).collect());
    let b = index.drawn_bounds(&dl, |item| {
        !moda_scene::types::is_page_bg_id(&item.node_id)
            && !is_excluded(scene, &item.node_id, &opts.exclude)
            && isolate
                .as_ref()
                .is_none_or(|iso| scene.is_node_isolated(&item.node_id, iso))
    })?;
    let whole = |x, y, w, h| PageRegionPx {
        x,
        y,
        w,
        h,
        band_x: 0,
        band_y: 0,
        band_w: full_w,
        band_h: full_h,
    };
    if crate::cull::is_empty(&b) {
        return Some(whole(0, 0, 1, 1));
    }
    // Canvas to raster as the plan maps it: the full target's `full_px / page` ratio.
    let (kx, ky) = (f64::from(full_w) / page_w, f64::from(full_h) / page_h);
    let span = |lo: f64, hi: f64, origin: f64, k: f64, full: u32| {
        let a = (((lo - origin) * k).floor() - 2.0).clamp(0.0, f64::from(full - 1)) as u32;
        let z = (((hi - origin) * k).ceil() + 2.0).clamp(f64::from(a + 1), f64::from(full)) as u32;
        (a, z - a)
    };
    let (x, w) = span(b[0], b[2], page_x, kx, full_w);
    let (y, h) = span(b[1], b[3], page_y, ky, full_h);
    Some(whole(x, y, w, h))
}

/// rust-edit-incremental-z4: like [`build_coverage_plan_cached`] but also returns
/// the per-node spans + page constants needed to seed the incremental patcher's
/// plan index. Used by `residency::build_plan_retained` on the (full-rebuild)
/// fallback path; byte-identical plan output to `build_coverage_plan_cached`.
pub fn build_coverage_plan_indexed(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: Option<CacheCtx<'_>>,
) -> Result<(CoveragePlan, PlanWarnings, PlanBuildAux)> {
    build_coverage_plan_indexed_with_dl(scene, page_index, opts, caches, None)
}

/// renderer-perf-more-z6: like [`build_coverage_plan_indexed`] but reuses an
/// already-built display list (the cold/structural-rebuild path builds it once
/// for the fingerprint and hands it here, instead of building it a second time).
/// `prebuilt_dl == None` is identical to `build_coverage_plan_indexed`. The DL is
/// a pure fn of (scene, page), so the plan output is byte-identical either way.
pub fn build_coverage_plan_indexed_with_dl(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: Option<CacheCtx<'_>>,
    prebuilt_dl: Option<moda_scene::DisplayList>,
) -> Result<(CoveragePlan, PlanWarnings, PlanBuildAux)> {
    let mut aux = PlanBuildAux::default();
    let (plan, warnings, _end_z) = build_coverage_plan_inner(
        scene,
        page_index,
        opts,
        caches,
        None,
        None,
        0,
        Some(&mut aux),
        prebuilt_dl,
        None,
        None,
        None,
    )?;
    Ok((plan, warnings, aux))
}

/// rust-edit-incremental-z4: rebuild the DrawItems for ONE primary display-list
/// node at a fixed render-z `z_in`, for the incremental plan patcher. Drives the
/// EXACT `append_node_draw_items` machinery over a single-item display list, so
/// an incrementally-rebuilt node is byte-identical to a from-scratch rebuild of
/// that node (single source of truth — no parallel code path to drift).
///
/// `primary` is the node's display-list `PaintItem` (its structural fields select
/// the build route; geometry/transform/paint are re-read fresh from `scene`).
/// `plan_masks` should be a SCRATCH table: an eligible (unmasked) node never
/// touches it, and the caller verifies the rebuilt items carry no mask ref before
/// splicing. Returns the items and the integer z consumed (the caller checks it
/// equals the cached span's `z_consumed`, else falls back).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rebuild_node_items(
    scene: &Scene,
    primary: &moda_scene::types::PaintItem,
    opts: &RenderOptions,
    transparent: bool,
    page_clip: [f32; 4],
    backdrop_ctx: Option<&crate::backdrop::PlanCtx>,
    plan_masks: &mut crate::effects::PlanMasks,
    caches: Option<CacheCtx<'_>>,
    z_in: u64,
) -> (Vec<DrawItem>, u64, PlanWarnings) {
    let dl = moda_scene::DisplayList {
        page_id: String::new(),
        page_index: 0,
        items: vec![primary.clone()],
    };
    let mut items: Vec<DrawItem> = Vec::new();
    let mut warnings = PlanWarnings::default();
    let end_z = append_node_draw_items(
        scene,
        &dl,
        opts,
        transparent,
        page_clip,
        backdrop_ctx,
        // Layer-blur pages are excluded from incremental seeding
        // (`aux.has_layer_blur`), so the patcher never rebuilds a diverted
        // node — no ctx here.
        None,
        plan_masks,
        caches,
        &mut items,
        &mut warnings,
        z_in,
        None,
        None,
        None,
    );
    (items, end_z - z_in, warnings)
}

/// Live-viewport: the editor camera + target sizing for
/// `build_camera_plan`. `camera_x/y` and `zoom` are the editor viewport state;
/// `viewport_w/h` are LOGICAL (CSS) px and `dpr` the device-pixel ratio, so the
/// combined target is `ceil(viewport × dpr)` and a canvas-absolute point lands
/// at `((X − camera_x) × zoom × dpr, …)`.
#[derive(Debug, Clone, Copy)]
pub struct CameraParams {
    pub camera_x: f64,
    pub camera_y: f64,
    pub zoom: f64,
    pub viewport_w: f64,
    pub viewport_h: f64,
    pub dpr: f64,
    /// Editor canvas backdrop (the gaps around/between pages), straight rgb 0..1.
    pub clear_rgb: [f32; 3],
}

impl CameraParams {
    /// Residency: the present-time override derived from these camera
    /// params. The combined camera plan's draw items are canvas-absolute, so a
    /// pan/zoom changes ONLY the view-projection, the (viewport × dpr) target
    /// size, and the clear color — never the draw lists. `render_camera_frame`
    /// caches the combined plan and feeds THIS override per frame so a static
    /// pan/zoom reuses the cached plan with no rebuild (PLAN §3 #21 / risk R13).
    ///
    /// `shader_time_s` is the LIVE renderer clock, re-threaded onto the
    /// cached plan's procedural-fill instances at present so an animating shader
    /// hits the camera plan cache (no per-frame rebuild) instead of re-keying on
    /// the clock.
    pub fn present_override(&self, shader_time_s: f64) -> CameraPresentOverride {
        CameraPresentOverride {
            view: crate::seams::CameraFrameView {
                camera_x: self.camera_x,
                camera_y: self.camera_y,
                zoom: self.zoom,
                viewport_w: self.viewport_w,
                viewport_h: self.viewport_h,
            },
            width_px: crate::export_scale::px_ceil(self.viewport_w * self.dpr) as u32,
            height_px: crate::export_scale::px_ceil(self.viewport_h * self.dpr) as u32,
            clear_rgba: [self.clear_rgb[0], self.clear_rgb[1], self.clear_rgb[2], 1.0],
            shader_time_s,
        }
    }
}

/// Residency: the camera-variant fields of a combined camera plan,
/// supplied to `render_plan_to_texture_with_camera` so a CACHED plan can be
/// re-presented at a new camera (pan/zoom) without rebuilding draw lists. All
/// three fields are pure functions of the editor camera + viewport; the draw
/// items (canvas-absolute) are untouched.
#[derive(Debug, Clone, Copy)]
pub struct CameraPresentOverride {
    pub view: crate::seams::CameraFrameView,
    pub width_px: u32,
    pub height_px: u32,
    pub clear_rgba: [f32; 4],
    /// The live renderer clock for this present. The cached plan's
    /// procedural-fill instances have a (possibly stale) baked clock; the
    /// present path re-threads THIS value into their `shader_params` clock slots
    /// (`ShaderFillBody::clock_offsets`) so an animating shader advances without
    /// rebuilding the plan. Export / per-page paths pass no override, so their
    /// baked clock (pinned 0 for export) is untouched.
    pub shader_time_s: f64,
}

/// The camera projection threaded into per-page plan building so
/// a backdrop page's `BackdropSplit` geometry is computed in CAMERA target pixels
/// (viewport × dpr, under the editor view-projection) rather than the per-page
/// export target. This is what lets backdrop pages render crisp INSIDE the
/// combined camera frame (no per-page snapshot fallback): the split's sampling
/// region / node rect / coverage all land at the correct on-screen position and
/// zoom. `None` (the export / per-page render path) keeps the per-page projection.
#[derive(Debug, Clone, Copy)]
pub struct CameraBackdropProjection {
    pub vp: [f32; 16],
    pub width_px: u32,
    pub height_px: u32,
}

impl CameraParams {
    /// The camera projection for backdrop split planning (see
    /// `CameraBackdropProjection`): the editor view-projection + the (viewport ×
    /// dpr) target the camera frame renders into.
    pub fn backdrop_projection(&self) -> CameraBackdropProjection {
        CameraBackdropProjection {
            vp: view_projection(
                self.camera_x,
                self.camera_y,
                self.zoom,
                self.viewport_w,
                self.viewport_h,
            ),
            width_px: crate::export_scale::px_ceil(self.viewport_w * self.dpr) as u32,
            height_px: crate::export_scale::px_ceil(self.viewport_h * self.dpr) as u32,
        }
    }
}

/// Live-viewport: build ONE `CoveragePlan` painting the N given
/// visible pages under the editor camera into a screen-sized target — the
/// editor twin of the per-page export `build_coverage_plan`. Each page's items
/// keep their canvas-absolute model matrices (page positions baked in), so a
/// single camera view-projection places them all; the target is `viewport ×
/// dpr`, so crispness is the screen's, independent of zoom (the fix: the old
/// snapshot path rendered each page to a `clamp(zoom × dpr, …, 3)`-capped
/// bitmap and CSS-upscaled it).
///
/// Backdrop pages are now composited crisp INSIDE the combined
/// camera frame — a backdrop page's `BackdropSplit` is projected into the camera
/// target (see `CameraBackdropProjection`) and the whole frame renders at
/// sample_count 1 (what backdrop requires). The `Ok(None)` snapshot-fallback
/// return is retained only for defensive/degenerate cases (it is no longer used
/// for backdrop). NOTE the combined plan is then camera-DEPENDENT (the split is
/// in target pixels), so the residency camera plan cache must NOT cache it — see
/// `build_camera_plan_retained`.
pub fn build_camera_plan(
    scene: &Scene,
    visible_page_indices: &[usize],
    cam: &CameraParams,
    opts: &RenderOptions,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    // Degraded text sized to the screen, as on the retained lane
    // (`residency::build_camera_plan_retained`): this plan is built for one camera.
    let _degraded = crate::text_draw::enter_degraded_camera(Some(
        crate::text_draw::degraded_camera_bucket(cam.zoom * cam.dpr),
    ));
    build_camera_plan_cached(scene, visible_page_indices, cam, opts, None, None)
}

/// Residency: `build_camera_plan` threaded with the R3/R4 per-node
/// plan caches (tessellation / text layout). `None` is byte-identical Stage-0;
/// `Some((caches, hashes))` reuses unchanged nodes' tessellation + text batches
/// across frames, so a pan/zoom (or playback resync) never re-tessellates or
/// re-lays-out static nodes. The combine + mask-rebase is identical either way.
pub fn build_camera_plan_cached(
    scene: &Scene,
    visible_page_indices: &[usize],
    cam: &CameraParams,
    opts: &RenderOptions,
    caches: Option<CacheCtx<'_>>,
    // rust-edit-incremental-z4: when Some AND the frame is exactly ONE in-range page
    // with NO freeform canvas-root contribution, filled with that page's
    // `PlanBuildAux` — the combined plan's items then EQUAL that page's (appended at
    // offset 0, mask_base 0), so the camera incremental patcher can seed its index
    // from it. Left `None` in every other shape (multi-page / canvas-root), where a
    // single page's spans wouldn't describe the combined plan.
    single_page_aux: Option<&mut Option<PlanBuildAux>>,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    build_camera_plan_culled(
        scene,
        visible_page_indices,
        cam,
        opts,
        caches,
        single_page_aux,
        None,
    )
}

/// Target pixels the cull window of [`build_camera_box_plan`] reaches past the camera's view on every side: the
/// target is the view rounded up to whole pixels, and an edge's antialiasing reaches a little past its geometry.
const BOX_CULL_PAD_PX: f64 = 2.0;

/// An export's camera plan over exactly the camera's target: [`build_camera_plan_cached`] culled to the view
/// (`crate::cull`), so nodes whose paint cannot reach the target are not planned (a credits roll lays out the rows on
/// screen, not all of them). Culling changes no pixel inside the window; culled nodes still advance the render-z
/// counter, so every drawn item is the one the unculled plan has.
pub fn build_camera_box_plan(
    scene: &Scene,
    cam: &CameraParams,
    opts: &RenderOptions,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    camera_box_plan(scene, cam, opts, None)
}

/// [`build_camera_box_plan`] through `caches`, which the caller keeps across the frames of one export: a text (or
/// brushed outline) whose content, everything but its transform, is unchanged since an earlier frame reuses that
/// frame's layout, as `render_page_cached` does for pages. The caches are content-keyed: with no resident hash map,
/// a node the plan looks up is hashed in this frame's scene (`residency::node_hashes_in`), and plain bodies keep the
/// command-keyed tessellation cache they have without caches. So the plan is the one [`build_camera_box_plan`]
/// builds. Outside the live lane the generation memos stay off: nothing is reused by generation.
pub fn build_camera_box_plan_cached(
    scene: &Scene,
    cam: &CameraParams,
    opts: &RenderOptions,
    caches: &mut crate::residency::PlanCaches,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    build_camera_box_plan_cached_with_dl(scene, cam, opts, caches, None)
}

/// [`build_camera_box_plan_cached`] with the canvas root's display list, when the caller built it already for this
/// frame (`canvas_dl` must be `moda_scene::build_display_list_for_canvas(scene)`): the plan reads it instead of
/// building it again. It is held only for this call.
pub fn build_camera_box_plan_cached_with_dl(
    scene: &Scene,
    cam: &CameraParams,
    opts: &RenderOptions,
    caches: &mut crate::residency::PlanCaches,
    canvas_dl: Option<std::sync::Arc<moda_scene::DisplayList>>,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    debug_assert!(canvas_dl.as_deref().is_none_or(|dl| *dl == moda_scene::build_display_list_for_canvas(scene)));
    let hashes = crate::residency::HashMap::default();
    caches.begin_frame();
    caches.set_frame_canvas_dl(scene, canvas_dl);
    let built = camera_box_plan(scene, cam, opts, Some((caches, &hashes)));
    caches.set_frame_canvas_dl(scene, None);
    built
}

fn camera_box_plan(
    scene: &Scene,
    cam: &CameraParams,
    opts: &RenderOptions,
    caches: Option<CacheCtx<'_>>,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    let units_per_px = 1.0 / (cam.zoom * cam.dpr);
    let pad = BOX_CULL_PAD_PX * units_per_px;
    let window = [
        cam.camera_x - pad,
        cam.camera_y - pad,
        cam.camera_x + cam.viewport_w / cam.zoom + pad,
        cam.camera_y + cam.viewport_h / cam.zoom + pad,
    ];
    if !window.iter().all(|v| v.is_finite()) {
        return build_camera_plan_cached(scene, &[], cam, opts, caches, None);
    }
    let mut out = CullOut::default();
    build_camera_plan_culled(scene, &[], cam, opts, caches, None, Some((&window, &mut out)))
}

/// [`build_camera_plan_cached`] with viewport culling: nodes whose visual bounds
/// miss `cull`'s window (canvas units) are left out (`crate::cull`), each still
/// advancing the render-z counter, so everything inside the window renders exactly
/// as the unculled plan draws it. `None` = no culling.
#[allow(clippy::too_many_arguments)]
pub fn build_camera_plan_culled(
    scene: &Scene,
    visible_page_indices: &[usize],
    cam: &CameraParams,
    opts: &RenderOptions,
    caches: Option<CacheCtx<'_>>,
    single_page_aux: Option<&mut Option<PlanBuildAux>>,
    cull: Option<CullCtx<'_>>,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    build_camera_plan_culled_with(
        scene,
        visible_page_indices,
        cam,
        opts,
        caches,
        single_page_aux,
        cull,
        None,
        None,
    )
}

/// The live camera lane's projection for layer captures (layer filters and
/// capture shadows): `scale` target px per canvas unit — the camera's scale
/// rounded UP to a zoom bucket ([`capture_scale_bucket`]) — over `window`
/// (canvas `[min_x, min_y, max_x, max_y]`, a margin around the view). Captures
/// planned under it depend on the camera only through the bucket and the
/// window, so a pan inside the window (or a zoom inside the bucket) keeps the
/// plan and every capture's content key: the engine reuses their results.
/// The composite resamples a capture made at the bucket scale onto the screen
/// (bilinear, at most a quarter-octave apart; exact on a bucket).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureView {
    pub scale: f64,
    pub window: [f64; 4],
}

/// Zoom buckets per octave for layer captures ([`CaptureView`]).
pub const CAPTURE_BUCKETS_PER_OCTAVE: f64 = 4.0;

/// `scale` rounded up to its capture zoom bucket (a power of 2^(1/4)), so a
/// capture is never coarser than the screen.
pub fn capture_scale_bucket(scale: f64) -> f64 {
    if !(scale.is_finite() && scale > 0.0) {
        return 1.0;
    }
    let steps = (scale.log2() * CAPTURE_BUCKETS_PER_OCTAVE - 1e-9).ceil();
    (steps / CAPTURE_BUCKETS_PER_OCTAVE).exp2()
}

impl CaptureView {
    /// The capture target this view stands for: the window snapped out to the
    /// bucket's texel grid, at the bucket scale.
    pub fn projection(&self) -> CameraBackdropProjection {
        let s = self.scale;
        let [x0, y0, x1, y1] = self.window;
        let (x0, y0) = ((x0 * s).floor() / s, (y0 * s).floor() / s);
        let w_px = (((x1 - x0) * s).ceil().max(1.0)) as u32;
        let h_px = (((y1 - y0) * s).ceil().max(1.0)) as u32;
        CameraBackdropProjection {
            vp: view_projection(x0, y0, 1.0, w_px as f64 / s, h_px as f64 / s),
            width_px: w_px,
            height_px: h_px,
        }
    }
}

/// [`build_camera_plan_culled`] with the live lane's capture projection
/// ([`CaptureView`]; `None` = captures follow the exact camera).
#[allow(clippy::too_many_arguments)]
pub fn build_camera_plan_culled_with(
    scene: &Scene,
    visible_page_indices: &[usize],
    cam: &CameraParams,
    opts: &RenderOptions,
    mut caches: Option<CacheCtx<'_>>,
    mut single_page_aux: Option<&mut Option<PlanBuildAux>>,
    mut cull: Option<CullCtx<'_>>,
    capture_view: Option<CaptureView>,
    // Scale RFC stage C: when Some, filled with every segment's units and
    // planning contexts, for the retained camera plan (`camera_slot`).
    mut plan_aux: Option<&mut CameraPlanAux>,
) -> Result<Option<(CoveragePlan, PlanWarnings)>> {
    let capture_projection = capture_view.map(|v| v.projection());
    if let Some(a) = plan_aux.as_deref_mut() {
        *a = CameraPlanAux::default();
    }
    // rust-edit-incremental-z4: only the single-in-range-page shape can seed the
    // camera incremental index (see `single_page_aux`).
    let in_range_pages = visible_page_indices
        .iter()
        .filter(|&&i| i < scene.doc().pages.len())
        .count();
    let want_aux = single_page_aux.is_some() && in_range_pages == 1;
    // Brush plan P8: outlines refit at the camera's zoom band (the retained
    // path sets this already, memoised; a direct call scans once).
    // Curves flatten at the camera's band too (F4.1).
    let _outline_scale = crate::tess::outline::camera_key_band().is_none().then(|| {
        let has_outlines = scene_flag(&mut caches, flag::SCENE, flag::HAS_OUTLINES, || {
            crate::tess::outline::scene_has_outlines(scene)
        });
        let has_curves = scene_flag(&mut caches, flag::SCENE, flag::HAS_CURVES, || {
            crate::tess::outline::scene_has_curves(scene)
        });
        crate::tess::outline::enter_camera(crate::tess::outline::camera_scale_for(
            has_outlines,
            has_curves,
            cam.zoom * cam.dpr,
        ))
    });
    let mut captured_aux: Option<PlanBuildAux> = None;
    let mut canvas_root_appended = false;
    let width_px = crate::export_scale::px_ceil(cam.viewport_w * cam.dpr) as u32;
    let height_px = crate::export_scale::px_ceil(cam.viewport_h * cam.dpr) as u32;
    if width_px == 0 || height_px == 0 {
        bail!("camera frame has zero pixel area ({width_px}x{height_px})");
    }

    let mut items: Vec<DrawItem> = Vec::new();
    let mut masks: Vec<crate::seams::MaskPlan> = Vec::new();
    let mut warnings = PlanWarnings::default();
    let mut any_backdrop = false;
    // A backdrop EFFECT (blur, glass, …), as opposed to a blend / paint-stack split:
    // only these force the combined frame single-sample.
    let mut any_backdrop_effect = false;
    // Layer-blur pages make the combined plan camera-dependent too
    // (capture regions in target px) — tracked for the aux seed gate below;
    // the residency camera cache bypasses on its own twin predicate.
    let mut any_layer_blur = false;

    // Project backdrop splits into the CAMERA target (not the
    // per-page export target) so backdrop pages composite crisp inside the
    // combined frame instead of falling back to the per-page snapshot path. This
    // matches the WebGPU LIVE editor (the relevant oracle for the editor): the
    // backdrop samples the rendered viewport. NOTE when `render_page_clipping` is
    // OFF, a backdrop node overhanging its page edge therefore samples the
    // viewport beyond the page (adjacent pages / the canvas gap) — same as WebGPU
    // live, but NOT the per-page snapshot path, which clamps to the page raster.
    // The per-page export plan (`build_coverage_plan`, camera_backdrop = None) is
    // unchanged, so export-golden parity is unaffected.
    let cam_backdrop = cam.backdrop_projection();

    // Freeform pages can overlap, so their render-z bands must be cumulative in
    // document order. Standard layouts never overlap page interiors;
    // keep their bands page-local so every page has the whole render-z range
    // (`depth::MAX_RENDER_Z`; a page or canvas root that draws an item past it is
    // refused where it is planned).
    // `max_page_z` still tracks the highest page-local/cumulative end so freeform
    // canvas-root nodes can start ABOVE every page.
    let cumulative_page_z = scene.is_freeform();
    let mut max_page_z: u64 = 0;
    for &page_index in visible_page_indices {
        let page_id = match scene.doc().pages.get(page_index) {
            Some(p) => p.id().to_string(),
            None => continue, // a stale index from the host cull — skip it
        };
        // A backdrop page splits the pass and forces
        // sample_count 1; render it crisp inside the camera frame with
        // camera-projected splits. The whole combined frame then renders at
        // sample_count 1 (the host already passes 1 for backdrop scenes via the
        // editor MSAA policy; we force it here too as a safety since the executor
        // bails if a split rides a multisample target).
        // Blend / paint-stack splits (no backdrop effect) keep MSAA (`BlendExec`).
        let page_effect = scene_flag(&mut caches, &page_id, flag::ACTIVE_BACKDROP, || {
            crate::backdrop::page_has_active_backdrop(scene, &page_id)
        });
        if page_effect
            || scene_flag(&mut caches, &page_id, flag::SPLIT_BLEND, || {
                crate::backdrop::page_has_split_blend(scene, &page_id)
            })
        {
            any_backdrop = true;
        }
        if page_effect {
            any_backdrop_effect = true;
            if opts.sample_count > 1 {
                crate::backdrop::plan::note_backdrop_aliases_frame(
                    scene,
                    &scene.flattened_z_order_for_page(&page_id),
                    opts.sample_count,
                );
            }
        }
        if scene_flag(&mut caches, &page_id, flag::LAYER_BLUR, || {
            crate::layer_blur::page_has_active_layer_filter(scene, &page_id)
        }) {
            any_layer_blur = true;
        }
        // Reborrow the caches for this page: the `Some` arm threads R3/R4 reuse
        // (tessellation + text layout) into every visible page; `None` is Stage-0.
        let page_caches = caches.as_mut().map(|(c, h)| (&mut **c, *h));
        // Freeform pages may OVERLAP, and page z-order is document
        // order (later pages paint on top). Thread the render-z cursor across
        // those pages so every item of page N+1 — including its `__page_bg_`
        // background node — sits nearer than all of page N's items. Standard
        // layouts restart at 0 because their page interiors cannot overlap.
        // Known fallback divergence: that orchestrator draws page BACKGROUNDS in
        // a separate pass at one fixed depth (`WebGPUPageSystem`, z = 0.99999),
        // so in the WebGPU fallback a later page's backdrop does not occlude an
        // earlier page's content inside an overlap region the way it does here.
        // The first visible page still starts at 0, so the single-in-range-page
        // camera incremental seed below stays valid.
        // rust-edit-incremental-z4: capture this page's spans/constants ONLY in the
        // single-in-range-page case, so the camera incremental index can be seeded.
        let mut page_aux = PlanBuildAux::default();
        let aux_arg = if want_aux { Some(&mut page_aux) } else { None };
        let page_z_start = if cumulative_page_z { max_page_z } else { 0 };
        let mut segment: Option<CameraSegment> = None;
        let (page_plan, page_warnings, page_end_z) = build_coverage_plan_inner(
            scene,
            page_index,
            opts,
            page_caches,
            Some(cam_backdrop),
            None,
            page_z_start,
            aux_arg,
            None,
            cull.as_mut().map(|(w, o)| (&**w, &mut **o)),
            capture_projection,
            plan_aux.is_some().then_some(&mut segment),
        )?;
        if let (Some(a), Some(mut seg)) = (plan_aux.as_deref_mut(), segment) {
            seg.item_start = items.len();
            seg.mask_start = masks.len();
            a.segments.push(seg);
        }
        if want_aux {
            captured_aux = Some(page_aux);
        }
        max_page_z = max_page_z.max(page_end_z);
        warnings.absorb(page_warnings);

        // Rebase this page's mask table onto the combined one: every appended
        // mask's ancestor index and every item's `mask_id` shift by the count
        // already present (mask ids are plan-local indices into `masks`).
        let mask_base = masks.len() as u64;
        for mut m in page_plan.masks {
            if let Some(a) = m.ancestor {
                m.ancestor = Some(a + mask_base as usize);
            }
            masks.push(m);
        }
        for mut item in page_plan.items {
            offset_item_mask(&mut item, mask_base);
            items.push(item);
        }
    }

    // Render freeform canvas-root (off-page) nodes into the combined
    // camera frame. They share the camera's world projection, so their
    // canvas-absolute transforms land correctly; appended AFTER all page items
    // so floating nodes paint on top — matching the WebGPU orchestrator, which
    // paints the freeform pass after every page and only in freeform layout
    // (`WebGPURenderOrchestrator.ts:923-977`). Gating on the layout mode keeps
    // exact parity: WebGPU never paints floating nodes outside freeform.
    if scene.is_freeform() && !scene.doc().canvas_root_node_ids.is_empty() {
        // A floating glass/blur node splits the pass like a page backdrop node:
        // force single-sample and project the split into the camera target so it
        // composites crisp in-frame (parity with the WebGPU freeform pass). The
        // residency camera cache also bypasses on this (build_camera_plan_retained).
        let canvas_effect = scene_flag(
            &mut caches,
            flag::CANVAS_ROOT,
            flag::ACTIVE_BACKDROP,
            || crate::backdrop::canvas_root_has_active_backdrop(scene),
        );
        let canvas_backdrop = canvas_effect
            || scene_flag(
                &mut caches,
                flag::CANVAS_ROOT,
                if opts.transparent { flag::SPLIT_BLEND_OVER_TRANSPARENT } else { flag::SPLIT_BLEND },
                || crate::backdrop::canvas_root_has_split_blend_over(scene, opts.transparent),
            );
        if canvas_backdrop {
            any_backdrop = true;
        }
        if canvas_effect {
            any_backdrop_effect = true;
            if opts.sample_count > 1 {
                crate::backdrop::plan::note_backdrop_aliases_frame(
                    scene,
                    &scene.flattened_z_order_for_canvas(),
                    opts.sample_count,
                );
            }
        }
        let backdrop_ctx = canvas_backdrop.then(|| {
            crate::backdrop::PlanCtx::new_camera(
                scene,
                cam_backdrop.width_px,
                cam_backdrop.height_px,
                // Off-page nodes are never page-clipped — the blur samples the
                // viewport, matching WebGPU's unclipped freeform pass.
                INFINITE_CLIP,
                cam_backdrop.vp,
            )
            // A transparent export (a Design frame or layer, always): multiply
            // blends through the split there (`split_id_over`), as on a page. The
            // split was planned for it (`canvas_root_has_split_blend_over`) but
            // the context said opaque, so multiply took its fixed-function state
            // and painted black over nothing.
            .over_transparent(opts.transparent)
        });
        // Floating layer-blurred nodes capture in camera-target px,
        // like the backdrop splits above.
        let canvas_layer_blur =
            scene_flag(&mut caches, flag::CANVAS_ROOT, flag::LAYER_BLUR, || {
                crate::layer_blur::canvas_root_has_active_layer_filter(scene)
            });
        if canvas_layer_blur {
            any_layer_blur = true;
        }
        let layer_blur_ctx = canvas_layer_blur.then(|| {
            let p = capture_projection.unwrap_or(cam_backdrop);
            crate::backdrop::PlanCtx::new_camera(
                scene,
                p.width_px,
                p.height_px,
                INFINITE_CLIP,
                p.vp,
            )
            .at_shader_time(opts.shader_time_s)
        });
        let canvas_caches = caches.as_mut().map(|(c, h)| (&mut **c, *h));
        let culled_before_root = cull.as_ref().map_or(0, |(_, o)| o.roots.len());
        // Seat canvas-root z above every page so floating nodes win the depth
        // test over any page they overlap (not merely paint after it).
        let mut segment: Option<CameraSegment> = None;
        let (root_items, root_masks, root_warnings) = build_canvas_root_items(
            scene,
            opts,
            max_page_z,
            backdrop_ctx.as_ref(),
            layer_blur_ctx.as_ref(),
            canvas_caches,
            cull.as_mut().map(|(w, o)| (&**w, &mut **o)),
            plan_aux.is_some().then_some(&mut segment),
        )?;
        if let (Some(a), Some(mut seg)) = (plan_aux.as_deref_mut(), segment) {
            seg.item_start = items.len();
            seg.mask_start = masks.len();
            // The canvas root's contexts, as `build_canvas_root_items` got them.
            seg.backdrop = canvas_backdrop.then_some(false);
            seg.layer_blur = canvas_layer_blur;
            a.segments.push(seg);
        }
        let root_culled = cull.as_ref().map_or(0, |(_, o)| o.roots.len()) > culled_before_root;
        warnings.absorb(root_warnings);
        // rust-edit-incremental-z4: a canvas-root contribution means the combined
        // plan is NOT just the single page's items — disqualify the index seed.
        // Culled floating nodes count too: the page's spans don't describe them.
        canvas_root_appended = !root_items.is_empty() || root_culled;
        let mask_base = masks.len() as u64;
        for mut m in root_masks {
            if let Some(a) = m.ancestor {
                m.ancestor = Some(a + mask_base as usize);
            }
            masks.push(m);
        }
        for mut item in root_items {
            offset_item_mask(&mut item, mask_base);
            items.push(item);
        }
    }

    // rust-edit-incremental-z4: hand back the single-page aux ONLY when the combined
    // plan really is that one page (no canvas-root, and — defensively — no backdrop
    // or layer blur, since those plans are camera-dependent and never cached/patched).
    if let Some(out) = single_page_aux.as_deref_mut() {
        *out = if want_aux && !canvas_root_appended && !any_backdrop && !any_layer_blur {
            captured_aux.take()
        } else {
            None
        };
    }

    // Backdrop splits require a single-sample target (the copy
    // source must be the resolved single-sample texture — the executor bails
    // otherwise). The host passes sample_count 1 for backdrop scenes already;
    // this is belt-and-suspenders against a detection mismatch. Trade-off: a
    // frame that co-renders a backdrop page WITH non-backdrop pages drops MSAA on
    // all of them that frame (one combined pass has one sample count) — crispness
    // is preserved (vector geometry at viewport×dpr); only edge AA is reduced.
    // Blend and paint-stack splits run on the multisample target (`BlendExec`).
    let sample_count = if any_backdrop_effect {
        1
    } else {
        opts.sample_count
    };

    let plan = CoveragePlan {
        page_id: "__camera__".to_string(),
        page_index: 0,
        width_px,
        height_px,
        page_size: [cam.viewport_w, cam.viewport_h],
        scale: cam.dpr,
        // Transparent camera readback (freeform floating-shape export).
        // Every LIVE camera lane (`render_camera_frame`/`_resident`) and the
        // eyedropper pass `opts.transparent == false`, so this stays byte-identical
        // for them (and the cache keys already fold `transparent`, so a transparent
        // export can't reuse — or poison — an opaque live plan). Only the offscreen
        // region-export readback (`render_camera_frame_to_bytes`, transparent = true)
        // makes this true; the clear's alpha then goes to 0 so the PNG carries true
        // transparency around non-rectangular floating shapes.
        transparent: opts.transparent,
        sample_count,
        camera: [cam.camera_x, cam.camera_y],
        items,
        masks,
        camera_frame: Some(crate::seams::CameraFrameView {
            camera_x: cam.camera_x,
            camera_y: cam.camera_y,
            zoom: cam.zoom,
            viewport_w: cam.viewport_w,
            viewport_h: cam.viewport_h,
        }),
        // A transparent export clears to a fully-zero backdrop (0,0,0,0) so
        // anti-aliased edges don't pick up a background-colored halo through the
        // un-premultiply — matching the page path (`clear_rgba: None` → [0,0,0,0]
        // when transparent) and WebGPU. Opaque keeps the caller's backdrop rgb at
        // alpha 1 (every live camera lane + the eyedropper), so those plans stay
        // byte-identical. NOTE the offscreen readback lane
        // (`render_camera_frame_to_bytes`) RE-forces this via the camera present
        // override (which `present_override` otherwise hardcodes opaque for the live
        // swapchain), so the executor's override-wins clear is what actually reaches
        // that pass; this plan-level value is kept for the plan/CPU consumers + the
        // byte-identity tests, and both are keyed off `opts.transparent` so they
        // can't drift out of lock-step.
        clear_rgba: Some(if opts.transparent {
            [0.0, 0.0, 0.0, 0.0]
        } else {
            [cam.clear_rgb[0], cam.clear_rgb[1], cam.clear_rgb[2], 1.0]
        }),
        region_viewport: None,
    };
    Ok(Some((plan, warnings)))
}

/// Clear the masks of capture items clipped by `mask` (the composite's): a
/// capture renders unmasked, with no mask table, and its composite applies it.
fn unmask_captured(items: &mut [DrawItem], mask: Option<&crate::seams::MaskRef>) {
    let Some(mask) = mask else { return };
    for item in items {
        match item {
            DrawItem::Body { mask: m, .. } | DrawItem::LayerBlurComposite { mask: m, .. }
                if m.as_ref().is_some_and(|m| m.mask_id == mask.mask_id) =>
            {
                *m = None
            }
            // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
            _ => {}
        }
    }
}

/// The clip mask a layer capture's composite takes: the node's nearest mask
/// when a clip-mask group is above it, else `None` (a clipping frame alone
/// clips the composite by its box and rounded corners).
fn capture_composite_mask(
    scene: &Scene,
    plan_masks: &mut crate::effects::PlanMasks,
    node_id: &str,
) -> Option<crate::seams::MaskRef> {
    if crate::backdrop::plan::has_ancestor_clip_mask_group(scene, node_id) {
        plan_masks.mask_ref_for_node(scene, node_id)
    } else {
        None
    }
}

/// Shift a draw item's mask reference (if any) onto the combined mask table.
/// `Body`, `Text` and `LayerBlurComposite` carry a `MaskRef`; shadows clip by
/// page bounds only. The items inside a capture group keep theirs unshifted:
/// a capture renders with no mask table (its composite is masked instead).
/// `BackdropSplit` now CAN reach a camera plan (backdrop pages
/// composite in-frame), but it carries NO index into `CoveragePlan::masks` (its
/// coverage is a target-pixel SDF / tessellated geometry, self-contained), so it
/// needs no rebasing here — the catch-all arm is correct for it.
fn offset_item_mask(item: &mut DrawItem, mask_base: u64) {
    match item {
        DrawItem::Body { mask: Some(m), .. } => m.mask_id += mask_base,
        DrawItem::LayerBlurComposite { mask: Some(m), .. } => m.mask_id += mask_base,
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        _ => {}
    }
}

/// R3/R4 cache context threaded through the plan builders: the caches plus
/// the resident per-node hash maps (synthetic nodes hash on the fly).
pub(crate) type CacheCtx<'a> = (
    &'a mut crate::residency::PlanCaches,
    &'a crate::residency::HashMap<String, crate::residency::NodeHashes>,
);

/// Keys of the page-level scene predicates memoised per resident generation on the
/// camera lane (`PlanCaches::memo_flag`); each walks a page's whole z-order.
mod flag {
    pub const ACTIVE_BACKDROP: u8 = 0;
    pub const SPLIT_BLEND: u8 = 1;
    pub const SPLIT_BLEND_OVER_OPAQUE: u8 = 2;
    pub const SPLIT_BLEND_OVER_TRANSPARENT: u8 = 3;
    pub const LAYER_BLUR: u8 = 4;
    pub const STACKS_ONLY_OPAQUE: u8 = 5;
    pub const STACKS_ONLY_TRANSPARENT: u8 = 6;
    pub const HAS_OUTLINES: u8 = 7;
    pub const HAS_CURVES: u8 = 8;
    /// The key the canvas root's predicates are stored under.
    pub const CANVAS_ROOT: &str = "\u{0}canvas-root";
    pub const SCENE: &str = "\u{0}scene";
}

fn scene_flag(
    caches: &mut Option<CacheCtx<'_>>,
    key: &str,
    which: u8,
    f: impl FnOnce() -> bool,
) -> bool {
    match caches {
        Some((c, _)) => c.memo_flag(key, which, f),
        None => f(),
    }
}

fn build_coverage_plan_inner(
    scene: &Scene,
    page_index: usize,
    opts: &RenderOptions,
    mut caches: Option<CacheCtx<'_>>,
    // When set (the combined camera frame), backdrop splits are
    // planned in CAMERA target pixels via this projection instead of the per-page
    // export target — so backdrop pages render crisp inside the camera frame.
    camera_backdrop: Option<CameraBackdropProjection>,
    // Tiled export: when set, the target is this sub-rect of the page's full
    // output raster instead of the whole page. The draw items are IDENTICAL to
    // the full-page plan (same display list, same z, same masks); only the
    // view-projection (via `camera_frame`) and the target size change — a tile
    // is a projection change, not a re-layout. Pages whose content samples the
    // TARGET (active backdrop / layer blur) bail: their pixels depend on
    // neighbors outside the tile, so tiling them would seam.
    region: Option<PageRegionPx>,
    // The render-z counter's starting value. 0 for the per-page export
    // path; the combined camera plan threads it so freeform canvas-root nodes can
    // start ABOVE every page's z (floating-on-top depth, not just paint order).
    // Returned (3rd tuple element) so the caller can continue the cursor.
    z_start: u64,
    // rust-edit-incremental-z4: when Some, fill the per-node spans + page-level
    // constants so the incremental patcher can later rebuild single nodes.
    aux: Option<&mut PlanBuildAux>,
    // renderer-perf-more-z6: an already-built display list for this page, so the
    // cold/structural-rebuild path can build the DL ONCE and share it with the
    // fingerprint instead of building it twice (it's a pure fn of scene+page, so
    // a shared DL is byte-identical). `None` → build it here (every other caller).
    prebuilt_dl: Option<moda_scene::DisplayList>,
    // Viewport culling (the live camera lane and `build_camera_box_plan`): skip nodes
    // whose visual bounds miss this window (`crate::cull`). `None` = draw all.
    mut cull: Option<CullCtx<'_>>,
    // The live camera lane's capture projection (`CaptureView`): layer captures
    // are planned at its zoom bucket and cut to its window instead of the exact
    // camera, so the plan (and every capture's content key) holds while the
    // camera pans inside the window. `None` = the camera / page projection.
    capture_projection: Option<CameraBackdropProjection>,
    // Scale RFC stage C: when Some, filled with the page's `CameraSegment` (its
    // units and planning contexts) for the retained camera plan.
    seg_out: Option<&mut Option<CameraSegment>>,
) -> Result<(CoveragePlan, PlanWarnings, u64)> {
    let page = scene.doc().pages.get(page_index).ok_or_else(|| {
        anyhow!(
            "page index {page_index} out of range ({} pages)",
            scene.doc().pages.len()
        )
    })?;
    let page_id = page.id().to_string();
    let (page_x, page_y) = scene.page_position(page_index);
    let page_w = page.width();
    let page_h = page.height();
    let full_width_px = crate::export_scale::px_ceil(page_w * opts.scale) as u32;
    let full_height_px = crate::export_scale::px_ceil(page_h * opts.scale) as u32;
    if full_width_px == 0 || full_height_px == 0 {
        bail!("page '{page_id}' has zero pixel area");
    }
    let (width_px, height_px) = match region {
        None => (full_width_px, full_height_px),
        Some(r) => {
            if r.w == 0 || r.h == 0 {
                bail!(
                    "page '{page_id}': region has zero pixel area ({}x{})",
                    r.w,
                    r.h
                );
            }
            // Saturating so a hostile x/w pair can't wrap; the full-page dims
            // are the authoritative bound the tile grid was computed from.
            if r.x.saturating_add(r.w) > full_width_px || r.y.saturating_add(r.h) > full_height_px {
                bail!(
                    "page '{page_id}': region {}+{}x{}+{} exceeds the {}x{} output raster",
                    r.x,
                    r.w,
                    r.y,
                    r.h,
                    full_width_px,
                    full_height_px
                );
            }
            if r.band_x > r.x
                || r.band_y > r.y
                || r.band_x.saturating_add(r.band_w) < r.x + r.w
                || r.band_y.saturating_add(r.band_h) < r.y + r.h
                || r.band_x.saturating_add(r.band_w) > full_width_px
                || r.band_y.saturating_add(r.band_h) > full_height_px
            {
                bail!("page '{page_id}': region band does not contain the region or exceeds the raster");
            }
            (r.w, r.h)
        }
    };

    let mut items: Vec<DrawItem> = Vec::new();
    let mut warnings = PlanWarnings::default();

    // M3e hook (SEAMS §7 / quirk Q-02): pages with any ACTIVE backdrop effect
    // render at sample_count 1 with an opaque white clear; `--transparent`
    // and `--exclude` are unsupported on that path
    // (`WebGPUExportCoordinator.ts:786-795`, `WebGPURenderer.ts:2548-2583`).
    let page_has_backdrop = scene_flag(&mut caches, &page_id, flag::ACTIVE_BACKDROP, || {
        crate::backdrop::page_has_active_backdrop(scene, &page_id)
    });
    if region.is_some() && page_has_backdrop {
        // A BackdropSplit samples ALREADY-DRAWN target pixels, so a tile's edge
        // pixels would blur against a target boundary the full-page render does
        // not have — a visible seam, not a rendering. Typed refusal instead.
        bail!("{REGION_BACKDROP_ERROR}: page '{page_id}' has active backdrop content");
    }
    // An isolated render (a node export, a plate) keeps its transparency: it
    // draws only its subtree, so a backdrop inside it samples that subtree
    // over the transparent clear (premultiplied, so the blur stays exact).
    let transparent = opts.transparent && (!page_has_backdrop || !opts.isolate.is_empty());
    // Edit retains an authored opaque page fill: the backdrop's opaque clear
    // is fully covered and does not degrade that composition's alpha. Ordinary
    // transparent exports and empty-fill compositions still cannot promise it.
    let preserves_opaque_page =
        opts.preserve_page_fill && !moda_scene::page_render::page_render_fill(page).is_empty();
    if opts.transparent && !transparent && !preserves_opaque_page {
        warnings.skipped.push(format!(
            "page '{page_id}': --transparent unsupported on backdrop pages (forced opaque white)"
        ));
    }
    // Isolation half: the backdrop path samples ALREADY-DRAWN content, and an
    // isolated render draws only the isolated subtrees. A backdrop inside an
    // isolated clipping frame samples the frame's own content (a frame export,
    // Figma's semantics); isolating the blurred node itself or a non-frame
    // around it would ship a plate whose blur ignores what sits behind it on
    // the page (`backdrop::isolated_backdrop_conflict`). Report that rather than
    // emit a silently-wrong plate; `--isolate` callers (the plates lane) treat
    // this warning as the typed refusal.
    if page_has_backdrop && !opts.isolate.is_empty() && {
        let isolate: std::collections::HashSet<&str> =
            opts.isolate.iter().map(String::as_str).collect();
        crate::backdrop::isolated_backdrop_conflict(
            scene,
            &scene.flattened_z_order_for_page(&page_id),
            &isolate,
        )
    } {
        warnings.skipped.push(format!(
            "{ISOLATE_BACKDROP_WARNING}: page '{page_id}' has active backdrop content"
        ));
    }
    // Exact overlay (`backdrop::blend`) and paint-stack groups split the pass
    // too, but keep transparency, isolation and MSAA: their splits copy the
    // resolved target and composite back into the multisample one
    // (`BlendExec`). Only backdrop effects force the single-sample target. A
    // region (tile) render splits too: a blend or group composite reads only
    // the pixel it writes, so a tile's splits (projected into the tile target,
    // `backdrop_ctx` below) give the full render's pixels, seam-free.
    // Over a transparent target multiply splits too (`BlendMode::split_id_over`).
    let transparent_target = transparent && !opts.preserve_page_fill;
    let blend_key = if transparent_target {
        flag::SPLIT_BLEND_OVER_TRANSPARENT
    } else {
        flag::SPLIT_BLEND_OVER_OPAQUE
    };
    let page_has_blend_split = scene_flag(&mut caches, &page_id, blend_key, || {
        crate::backdrop::page_has_split_blend_over(scene, &page_id, transparent_target)
    });
    let stacks_key = if transparent_target {
        flag::STACKS_ONLY_TRANSPARENT
    } else {
        flag::STACKS_ONLY_OPAQUE
    };
    // Splits from paint stacks only (E3.1 P6) keep rect / ellipse bodies on the SDF
    // path (read by `backdrop_ctx` below, which exists only when the page splits).
    let splits_are_stacks_only = page_has_blend_split
        && !page_has_backdrop
        && scene_flag(&mut caches, &page_id, stacks_key, || {
            crate::backdrop::page_splits_are_stacks_only(scene, &page_id, transparent_target)
        });
    let page_splits = page_has_backdrop || page_has_blend_split;
    let sample_count = if page_has_backdrop {
        // Say so (#1680): the export asked for MSAA and the page ships without it.
        if opts.sample_count > 1 {
            crate::backdrop::plan::note_backdrop_aliases_frame(
                scene,
                &scene.flattened_z_order_for_page(&page_id),
                opts.sample_count,
            );
        }
        1
    } else {
        opts.sample_count
    };

    // Page-fill quad (`createPageBackgroundData` + `WebGPUPageSystem`): the
    // export/screenshot paths bake `pageData.fill` from `getPageRenderData()`
    // (`WebGPUExportCoordinator.ts:267-331,605-650`), which is NOT the page's
    // stored `fill` for non-color priorities: shader pages get the
    // "representative" palette average, gradient/pattern pages get white
    // (`core/scene/scene.pages.ts:258-265` — ported as
    // `moda_scene::page_render::page_render_fill`). The page system skips
    // only an EMPTY fill string and otherwise draws
    // `colorToNormalizedRgb(fill) || white`, alpha forced 1.0
    // (`WebGPUPageSystem.ts:421-431`).
    if !transparent || opts.preserve_page_fill {
        let fill = moda_scene::page_render::page_render_fill(page);
        if !fill.is_empty() {
            let (r, g, b) = color_to_normalized_rgb(&fill).unwrap_or((1.0, 1.0, 1.0));
            items.push(DrawItem::PageFill {
                instance: PageInstance {
                    model_matrix: [
                        page_w as f32,
                        0.0,
                        0.0,
                        0.0, //
                        0.0,
                        page_h as f32,
                        0.0,
                        0.0, //
                        0.0,
                        0.0,
                        1.0,
                        0.0, //
                        page_x as f32,
                        page_y as f32,
                        0.0,
                        1.0,
                    ],
                    color: [r as f32, g as f32, b as f32, 1.0],
                    dimensions: [page_x as f32, page_y as f32, page_w as f32, page_h as f32],
                    stroke_data: [0.0; 4],
                },
            });
        }
    }

    let page_clip: [f32; 4] = if scene.doc().config.render_page_clipping {
        [page_x as f32, page_y as f32, page_w as f32, page_h as f32]
    } else {
        INFINITE_CLIP
    };

    // M3e hook: target-pixel projection context for backdrop param mapping.
    // Inside the combined camera frame, project to the camera
    // target (viewport × dpr under the editor VP) so the split lands crisp at the
    // on-screen position; the per-page export target is used otherwise.
    // Tiled export: the tile's blend splits (and the executor's fallback
    // projection when the device can't take the tile's shifted viewport,
    // `RegionViewport`) project through `camera_frame` — the
    // page-local VP translated to the tile origin over the tile's logical span
    // (zoom 1). The per-axis canvas→pixel ratio must be the FULL target's
    // `full_px / page_logical` (which `ceil` stretches a hair above
    // `opts.scale` on non-integral products), not `opts.scale` itself, so a
    // tile places canvas content at exactly the full-page pixel position minus
    // the tile offset.
    let region_view = region.map(|r| {
        let kx = full_width_px as f64 / page_w;
        let ky = full_height_px as f64 / page_h;
        crate::seams::CameraFrameView {
            camera_x: page_x + r.x as f64 / kx,
            camera_y: page_y + r.y as f64 / ky,
            zoom: 1.0,
            viewport_w: r.w as f64 / kx,
            viewport_h: r.h as f64 / ky,
        }
    });
    let backdrop_ctx = page_splits.then(|| {
        match (camera_backdrop, region_view) {
            (Some(cam), _) => crate::backdrop::PlanCtx::new_camera(
                scene,
                cam.width_px,
                cam.height_px,
                page_clip,
                cam.vp,
            ),
            // A tile's blend splits project into the tile target.
            (None, Some(v)) => crate::backdrop::PlanCtx::new_camera(
                scene,
                width_px,
                height_px,
                page_clip,
                view_projection(v.camera_x, v.camera_y, 1.0, v.viewport_w, v.viewport_h),
            )
            .over_transparent(transparent_target),
            (None, None) => crate::backdrop::PlanCtx::new(
                scene, width_px, height_px, page_x, page_y, page_w, page_h, page_clip,
            )
            .over_transparent(transparent_target),
        }
        // Splits from paint stacks only (E3.1 P6) keep rect / ellipse bodies on the SDF path.
        .keeping_shape_sdf(splits_are_stacks_only)
    });

    // Layer-blur capture regions/scales are target-px values under
    // the same projection, so build a twin ctx when the page carries an active
    // layer blur. Deliberately DOES NOT feed the sample_count/transparent
    // forcing above — layer blur reads nothing from the target, so pages keep
    // MSAA and transparent exports (layer-blur-spec.md §7.2).
    let page_has_layer_blur = scene_flag(&mut caches, &page_id, flag::LAYER_BLUR, || {
        crate::layer_blur::page_has_active_layer_filter(scene, &page_id)
    });
    if region.is_some()
        && page_has_layer_blur
        && crate::layer_blur::plan::page_has_layer_filter_effect(scene, &page_id)
    {
        // Layer-blur capture regions are target-px reads of rendered content —
        // the same tile-edge seam hazard as backdrop (see above). Capture
        // shadows alone tile: each tile captures what its shadows reach.
        bail!("{REGION_LAYER_BLUR_ERROR}: page '{page_id}' has active layer-blur content");
    }
    let layer_blur_ctx = page_has_layer_blur.then(|| match (camera_backdrop, region_view) {
        _ if capture_projection.is_some() => {
            let p = capture_projection.expect("checked");
            crate::backdrop::PlanCtx::new_camera(scene, p.width_px, p.height_px, page_clip, p.vp)
        }
        (Some(cam), _) => crate::backdrop::PlanCtx::new_camera(
            scene,
            cam.width_px,
            cam.height_px,
            page_clip,
            cam.vp,
        ),
        // A tile's capture shadows project into the tile target.
        (None, Some(v)) => crate::backdrop::PlanCtx::new_camera(
            scene,
            width_px,
            height_px,
            page_clip,
            view_projection(v.camera_x, v.camera_y, 1.0, v.viewport_w, v.viewport_h),
        ),
        (None, None) => crate::backdrop::PlanCtx::new(
            scene, width_px, height_px, page_x, page_y, page_w, page_h, page_clip,
        ),
    })
    .map(|c| c.at_shader_time(opts.shader_time_s));

    // The live camera lane keeps each page's display list for the resident
    // generation (`PlanCaches::dl_get`): a culled plan is rebuilt whenever the
    // camera leaves its cull window, and the display list (bounds of every node)
    // is most of an unchanged scene's rebuild.
    let cached_dl = caches.as_mut().and_then(|(c, _)| c.dl_get(&page_id));
    let dl: std::sync::Arc<moda_scene::DisplayList> = match cached_dl {
        Some(dl) => dl,
        None => {
            let dl = std::sync::Arc::new(
                prebuilt_dl.unwrap_or_else(|| moda_scene::build_display_list(scene, &page_id)),
            );
            if let Some((c, _)) = caches.as_mut() {
                c.dl_put(&page_id, dl.clone());
            }
            dl
        }
    };

    // M3d hook (SEAMS change log): lazy per-owner mask coverage planning
    // (r8unorm, rendered at the frame's pixel scale by the executor), text
    // masks included on every page (`opts.text_masks_available`).
    let mut plan_masks =
        crate::effects::PlanMasks::with_text_masks(opts.text_masks_available.unwrap_or(true))
            .with_vector_text_masks(true)
            .in_working_color(opts.working.is_some())
            .at_shader_time(opts.shader_time_s);

    // rust-edit-incremental-z4: capture per-node spans when an index is wanted.
    let mut spans_buf: Option<Vec<NodeSpan>> = aux.as_ref().map(|_| Vec::new());
    let culled_before = cull.as_ref().map_or(0, |(_, o)| o.roots.len());
    let mut recorder = seg_out.as_ref().map(|_| UnitRecorder::default());
    let head_warnings = seg_out.as_ref().map(|_| warnings.clone());
    let prefix = items.len();
    let end_z = append_node_draw_items(
        scene,
        &dl,
        opts,
        transparent && !opts.preserve_page_fill,
        page_clip,
        backdrop_ctx.as_ref(),
        layer_blur_ctx.as_ref(),
        &mut plan_masks,
        caches,
        &mut items,
        &mut warnings,
        z_start,
        spans_buf.as_mut(),
        cull.as_mut().map(|(w, o)| (&**w, &mut **o)),
        recorder.as_mut(),
    );
    depth::ensure_render_z(drawn_z_end(&items))?;
    let mut tail_warnings = PlanWarnings::default();
    let masks = plan_masks.into_masks(&mut tail_warnings);
    if let (Some(out), Some(recorder)) = (seg_out, recorder) {
        *out = Some(CameraSegment {
            kind: SegmentKind::Page(page_index),
            dl: dl.clone(),
            page_clip,
            node_transparent: transparent && !opts.preserve_page_fill,
            backdrop: page_splits.then_some(splits_are_stacks_only),
            layer_blur: page_has_layer_blur,
            text_masks: opts.text_masks_available.unwrap_or(true),
            item_start: 0,
            prefix,
            item_len: items.len(),
            mask_start: 0,
            mask_len: masks.len(),
            z_start,
            z_end: end_z,
            units: recorder.units,
            head_warnings: head_warnings.unwrap_or_default(),
            tail_warnings: tail_warnings.clone(),
        });
    }
    warnings.absorb(tail_warnings);
    if let Some(aux) = aux {
        aux.spans = spans_buf.unwrap_or_default();
        aux.page_clip = page_clip;
        aux.transparent = transparent;
        aux.has_backdrop = page_splits;
        aux.has_layer_blur = page_has_layer_blur;
        aux.culled_roots = cull
            .as_ref()
            .map(|(_, o)| o.roots[culled_before..].to_vec())
            .unwrap_or_default();
    }

    Ok((
        CoveragePlan {
            page_id,
            page_index,
            width_px,
            height_px,
            page_size: [page_w, page_h],
            scale: opts.scale,
            transparent,
            sample_count,
            camera: [page_x, page_y],
            items,
            masks,
            // Page-local export/editor-snapshot path: no camera override, no
            // backdrop clear color (the page-fill quad covers the target).
            // Tiled export: the region projects through `camera_frame`
            // (`region_view` above).
            camera_frame: region_view,
            clear_rgba: None,
            // Tiled export: the executor draws with the band's VP (the page's
            // own when the band is the whole page) under a viewport shifted to
            // the tile (`RegionViewport`), when the device allows that
            // viewport; `camera_frame` is the fallback projection.
            region_viewport: region.map(|r| {
                let kx = full_width_px as f64 / page_w;
                let ky = full_height_px as f64 / page_h;
                let whole_page = (r.band_x, r.band_y, r.band_w, r.band_h)
                    == (0, 0, full_width_px, full_height_px);
                crate::seams::RegionViewport {
                    band: (!whole_page).then(|| crate::seams::CameraFrameView {
                        camera_x: page_x + r.band_x as f64 / kx,
                        camera_y: page_y + r.band_y as f64 / ky,
                        zoom: 1.0,
                        viewport_w: r.band_w as f64 / kx,
                        viewport_h: r.band_h as f64 / ky,
                    }),
                    shift_x: r.x - r.band_x,
                    shift_y: r.y - r.band_y,
                    raster_w: r.band_w,
                    raster_h: r.band_h,
                }
            }),
        },
        warnings,
        end_z,
    ))
}

/// Whether `--exclude` drops the node `id`: it is listed, or it is a transient child (a generated host's chart
/// bars, axis labels, …) of a listed node. Only transient links are walked: excluding a group or frame still
/// leaves its real children to their own entries, as callers that exclude a group but raster its children expect.
fn is_excluded(scene: &Scene, id: &str, exclude: &[String]) -> bool {
    if exclude.is_empty() {
        return false;
    }
    let mut current = id;
    for _ in 0..64 {
        if exclude.iter().any(|e| e == current) {
            return true;
        }
        match scene.transient_parent_of().get(current) {
            Some(parent) => current = parent.as_str(),
            None => return false,
        }
    }
    false
}

/// The skip-filters of [`append_node_draw_items`]: a primary display-list item that
/// draws nothing AND consumes no render-z. Every other primary item consumes z
/// (1, or a blend's expansion count) whether or not it draws — which is what lets
/// the viewport cull (`crate::cull`) skip it while keeping every later item's depth.
fn item_skipped_without_z(
    scene: &Scene,
    item: &moda_scene::types::PaintItem,
    opts: &RenderOptions,
    isolate: Option<&std::collections::HashSet<&str>>,
    transparent: bool,
) -> bool {
    if is_excluded(scene, &item.node_id, &opts.exclude) {
        return true;
    }
    // Positive filter (`--isolate`): keep only nodes inside one of the
    // isolated subtrees. Placed with the other skip-filters, ABOVE the span
    // bookkeeping's fall-through, so a filtered node emits no items and
    // consumes no z — the same shape `--exclude` has.
    //
    // The page background is EXEMPT. It is not a node paint item: it is the
    // synthetic `__page_bg_<pageId>` quad, it appears in no isolate set, and
    // it has no parent chain to walk — so without this exemption an
    // isolated render would drop the page fill and land the element on the
    // bare clear color. That is invisible on the plates lane (always
    // `--transparent`, where the branch below drops the background anyway)
    // but wrong for every OPAQUE isolated render, which is exactly what
    // `batch --isolate` is. The page fill stays governed solely by
    // `transparent`, matching the browser (`isExportNodeIsolated` is applied
    // to node paint items only) and the three docstrings that promise it.
    if let Some(isolate) = isolate {
        if !moda_scene::types::is_page_bg_id(&item.node_id)
            && !scene.is_node_isolated(&item.node_id, isolate)
        {
            return true;
        }
    }
    if transparent && moda_scene::types::is_page_bg_id(&item.node_id) {
        return true; // skipPageBackgrounds (displayListAdapter.ts:68)
    }
    // Group boundaries render nothing and consume NO z (renderNode.ts:32).
    if item.kind == PaintKind::GroupBoundary {
        return true;
    }
    // Active clip-mask source nodes are skipped without consuming z
    // (WebGPUShapeSystem.shouldSkipRender early-return).
    if item.fallback_reasons.contains(&"text-mask-source") {
        return true;
    }
    scene.effective_node(&item.node_id).is_none()
}

/// Whether a primary item is a group that paints its own appearance (E8.2, `Scene::group_appearance`): its
/// fills and strokes on its merged outline, the ones under its contents at the group's item and the ones over
/// them after its last descendant. Such a group consumes [`GROUP_APPEARANCE_Z`] render-z (one slot each side);
/// every other group consumes none.
fn paints_group_appearance(
    scene: &Scene,
    item: &moda_scene::types::PaintItem,
    opts: &RenderOptions,
    isolate: Option<&std::collections::HashSet<&str>>,
) -> bool {
    item.kind == PaintKind::GroupBoundary
        && scene.group_appearance(&item.node_id).is_some()
        && !is_excluded(scene, &item.node_id, &opts.exclude)
        && isolate.is_none_or(|iso| {
            scene.is_node_isolated(&item.node_id, iso)
                || ["under", "over"]
                    .iter()
                    .any(|side| iso.contains(group_appearance_id(&item.node_id, side).as_str()))
        })
}

/// The id one side of a group's appearance draws under (`<group>::appearance::under|over`): its draw items' owner,
/// and what `--isolate` names to render that side alone, or `--exclude` to leave it out (the PDF export's raster
/// steps). Excluding the group's own id leaves out both sides.
pub fn group_appearance_id(group_id: &str, side: &str) -> String {
    format!("{group_id}::appearance::{side}")
}

/// Whether one side of a painting group draws under `--isolate`: the group is isolated, or the side is named.
fn group_side_isolated(
    scene: &Scene,
    group_id: &str,
    side: &str,
    isolate: Option<&std::collections::HashSet<&str>>,
) -> bool {
    isolate.is_none_or(|iso| {
        scene.is_node_isolated(group_id, iso) || iso.contains(group_appearance_id(group_id, side).as_str())
    })
}

/// The render-z a group painting its appearance consumes (see [`paints_group_appearance`]).
const GROUP_APPEARANCE_Z: u64 = 2;

/// One side of a group's appearance (E8.2): the group's fills and strokes in `part` (bottom to top) painted on
/// its merged outline at render-z `z`, as a shape's stack draws (their own opacity, blend and effects), at the
/// group's effective opacity. The group's own blend, shadows and filters apply to the whole subtree (its
/// capture), this included.
#[allow(clippy::too_many_arguments)]
fn group_appearance_items(
    scene: &Scene,
    group_id: &str,
    part: &[moda_scene::appearance::Entry],
    side: &str,
    z: u64,
    page_clip: [f32; 4],
    opts: &RenderOptions,
    backdrop_ctx: Option<&crate::backdrop::PlanCtx>,
    layer_blur_ctx: Option<&crate::backdrop::PlanCtx>,
    plan_masks: &mut crate::effects::PlanMasks,
    mut caches: Option<CacheCtx<'_>>,
    warnings: &mut PlanWarnings,
    faded: &dyn Fn(&str) -> bool,
) -> Vec<DrawItem> {
    use moda_scene::appearance as ap;
    let (Some(record), Some(group)) = (scene.group_appearance(group_id), scene.effective_node(group_id))
    else {
        return Vec::new();
    };
    if part.is_empty() || record.points().is_none_or(|p| p.is_empty()) {
        return Vec::new();
    }
    let (fills, strokes) = (ap::group_fills(group), ap::group_strokes(group));
    let ids = ap::entry_ids(part, &fills, &strokes);
    let mut m = record.fields().clone();
    m.insert("id".into(), Value::from(group_appearance_id(group_id, side)));
    moda_scene::paint::set_fills(
        &mut m,
        fills.into_iter().filter(|f| ids.contains(&f.id)).collect(),
    );
    moda_scene::paint::set_strokes(
        &mut m,
        strokes.into_iter().filter(|s| ids.contains(&s.id)).collect(),
    );
    m.insert(
        ap::APPEARANCE_ORDER_KEY.into(),
        Value::Array(ids.into_iter().map(Value::from).collect()),
    );
    // Inside a fade split (its own, or an ancestor's) that opacity is the split's.
    let own = if faded(group_id) { 1.0 } else { group.f64("opacity").unwrap_or(1.0) };
    let opacity = own * scene.ancestor_opacity_product_except(group_id, faded);
    m.insert("opacity".into(), Value::from(opacity));
    // Cached under its content hash (no scene hash): the zoom band its curves flatten at is part of it.
    m.insert("_appearanceBand".into(), Value::from(tess::outline::current_band()));
    let node = record.with_fields(m);
    let eff_clip =
        crate::effects::effective_page_clip(scene, group_id, page_clip).unwrap_or(page_clip);
    let mask = plan_masks.mask_ref_for_node(scene, group_id);
    let hashes = caches.as_ref().map(|(_, h)| *h);
    let fx = EntryFx::for_node(scene, group_id, layer_blur_ctx, opts, hashes, page_clip, plan_masks);
    let built = match paint::layers::plan(&node) {
        paint::layers::LayerPlan::Single(folded) => build_body_item(
            scene,
            folded.as_ref().unwrap_or(&node),
            z as f64,
            eff_clip,
            mask,
            true,
            opts.shader_time_s,
            None,
            None,
            caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
            warnings,
        ),
        paint::layers::LayerPlan::Stack(layers) => Ok(build_stack_items(
            scene,
            &node,
            &layers,
            z as f64,
            crate::tess::placed::PLACED_Z_SPAN,
            backdrop_ctx,
            true,
            warnings,
            Some(&fx),
            |layer, lz, _, w| {
                build_body_item(
                    scene,
                    layer,
                    lz,
                    eff_clip,
                    mask.clone(),
                    true,
                    opts.shader_time_s,
                    None,
                    None,
                    caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                    w,
                )
            },
        )),
    };
    match built {
        Ok(items) => items,
        Err(e) => {
            warnings
                .skipped
                .push(format!("{group_id} (group appearance): {e}"));
            Vec::new()
        }
    }
}

/// Shared node-loop that turns a display list's paint items into
/// engine `DrawItem`s. Extracted from `build_coverage_plan_inner` so the page
/// path and the freeform canvas-root path build identical geometry. `page_clip`
/// is the effective clip for these nodes (a page rect for page content,
/// `INFINITE_CLIP` for unclipped off-page nodes); `transparent` skips
/// page-background items; `backdrop_ctx` (page-scoped) plans backdrop splits.
#[allow(clippy::too_many_arguments)]
fn append_node_draw_items(
    scene: &Scene,
    dl: &moda_scene::DisplayList,
    opts: &RenderOptions,
    transparent: bool,
    page_clip: [f32; 4],
    backdrop_ctx: Option<&crate::backdrop::PlanCtx>,
    // Layer-blur projection ctx (page-scoped, built only when the
    // page/canvas-root carries an active layer blur). When an eligible node's
    // items land, they are diverted into a `LayerBlurGroup` and replaced by one
    // `LayerBlurComposite` at the body's z — see the loop tail.
    layer_blur_ctx: Option<&crate::backdrop::PlanCtx>,
    plan_masks: &mut crate::effects::PlanMasks,
    mut caches: Option<CacheCtx<'_>>,
    items: &mut Vec<DrawItem>,
    warnings: &mut PlanWarnings,
    z_start: u64,
    // rust-edit-incremental-z4: when Some, record one NodeSpan per item-producing
    // primary node (its item range + z) so the incremental patcher can later
    // rebuild that node in isolation. None = the canvas-root / non-indexed path.
    mut spans: Option<&mut Vec<NodeSpan>>,
    // Viewport culling: skip display-list ranges whose visual bounds miss the
    // window, advancing z by what they would have consumed (`crate::cull`).
    cull: Option<CullCtx<'_>>,
    // Scale RFC stage C: when Some, record the display list's rebuildable units
    // (`PlanUnit`) for the retained camera plan.
    mut units: Option<&mut UnitRecorder>,
) -> u64 {
    // Brush plan P8: outlined strokes refit at the render scale (a camera
    // build's zoom band, set by the camera builder, takes precedence).
    let _outline_scale = tess::outline::enter_plan(opts.scale);
    if let Some(r) = units.as_deref_mut() {
        r.start(items.len());
    }
    // RENDER-Z RULE (seams.rs): a separate counter from PaintItem.z_index.
    // Starts at `z_start` so a combined plan can keep z monotonic across
    // appended display lists (canvas-root nodes above all pages).
    let mut z: u64 = z_start;

    // `--isolate` (per-element alpha plates): built ONCE per display list, and
    // `None` whenever no isolation was requested — the overwhelmingly common
    // case pays neither the set build nor a per-item ancestry walk, so the
    // default plan is byte-identical to the pre-`isolate` plan.
    let isolate: Option<std::collections::HashSet<&str>> = if opts.isolate.is_empty() {
        None
    } else {
        Some(opts.isolate.iter().map(String::as_str).collect())
    };

    let (cull, mut cull_out) = match cull {
        Some((w, o)) => (Some(w), Some(o)),
        None => (None, None),
    };
    let hashes = caches.as_ref().map(|(_, h)| *h);
    let cull_plan = cull.map(|window| {
        let out = cull_out.as_deref_mut().expect("cull out rides the window");
        let build = || {
            crate::cull::CullIndex::build(
                scene,
                dl,
                |item| {
                    u64::from(!item_skipped_without_z(
                        scene,
                        item,
                        opts,
                        isolate.as_ref(),
                        transparent,
                    )) + if paints_group_appearance(scene, item, opts, isolate.as_ref()) {
                        GROUP_APPEARANCE_Z
                    } else {
                        0
                    }
                },
                |item| {
                    let node = scene.effective_node(&item.node_id)?;
                    let geom = hashes?.get(&item.node_id)?.geom;
                    brush_bounds_key(scene, node, geom)
                },
            )
        };
        // The subtree index depends on the scene and on which items consume z
        // (exclude / isolate / transparent), not on the window: kept per generation.
        let index = match caches.as_mut().filter(|_| dl.page_id != RANGE_DL_KEY) {
            Some((c, _)) => {
                let mut h = rustc_hash::FxHasher::default();
                use std::hash::Hasher;
                h.write_u8(transparent as u8);
                for e in &opts.exclude {
                    h.write_usize(e.len());
                    h.write(e.as_bytes());
                }
                h.write_u8(0xff);
                for e in &opts.isolate {
                    h.write_usize(e.len());
                    h.write(e.as_bytes());
                }
                c.cull_index(&dl.page_id, h.finish(), build)
            }
            None => std::sync::Arc::new(build()),
        };
        let learned = |id: &str, key: u64| caches.as_ref().and_then(|(c, _)| c.brush_bounds_get(id, key));
        let plan = if index.len() == dl.items.len() {
            index.decide(dl, window, &learned)
        } else {
            build().decide(dl, window, &learned)
        };
        if let Some((c, _)) = caches.as_mut() {
            c.brush_kept_note(&plan.brush_seen);
        }
        out.considered += dl.items.iter().filter(|i| i.is_primary_node_item).count();
        out.items += plan.culled_items;
        out.roots.extend(plan.culled_roots.iter().cloned());
        plan
    });
    // Items before this index belong to a culled range.
    let mut culled_until = 0usize;
    // The resident node hashes (retained lanes): layer captures get a content
    // key, so the engine reuses their results across frames.
    let hashes = caches.as_ref().map(|(_, h)| *h);
    // Subtree captures (a group's or an unfilled frame's shadows and layer
    // filters) whose display-list range is still being drawn, innermost last.
    let mut subtrees: Vec<SubtreeCapture> = Vec::new();

    for (index, item) in dl.items.iter().enumerate() {
        while subtrees.last().is_some_and(|c| c.dl_end <= index) {
            let done = subtrees.pop().expect("checked");
            if let Some(over) = done.appearance_over.as_deref() {
                items.extend(group_appearance_items(
                    scene,
                    done.node.id(),
                    over,
                    "over",
                    z,
                    done.page_clip,
                    opts,
                    backdrop_ctx,
                    layer_blur_ctx,
                    plan_masks,
                    caches.as_mut().map(|(c, h)| (&mut **c, *h)),
                    warnings,
                    &|id: &str| {
                        (done.fade.is_some() && id == done.node.id())
                            || subtrees.iter().any(|c| c.fade.is_some() && c.node.id() == id)
                    },
                ));
                z += 1;
            }
            finish_subtree_capture(
                scene,
                layer_blur_ctx,
                backdrop_ctx,
                done,
                hashes,
                &plan_masks.masks,
                opts,
                items,
                z,
                warnings,
            );
        }
        if index < culled_until || !item.is_primary_node_item {
            continue; // shapeText/connectorText ride their node's z + 0.5
        }
        // A primary item outside every open subtree capture starts a unit.
        if subtrees.is_empty() {
            if let Some(r) = units.as_deref_mut() {
                r.open_at(index, items.len(), z, warnings, plan_masks.masks.len());
            }
        }
        if let Some((end, z_cost)) = cull_plan.as_ref().and_then(|p| p.skip_at(index)) {
            z += z_cost;
            culled_until = end;
            if let Some(r) = units.as_deref_mut() {
                r.mark_culled(subtrees.is_empty());
            }
            continue;
        }
        let group_appearance = paints_group_appearance(scene, item, opts, isolate.as_ref());
        if layer_blur_ctx.is_some() || backdrop_ctx.is_some() || group_appearance {
            if let Some(capture) = begin_subtree_capture(
                scene,
                dl,
                index,
                opts,
                isolate.as_ref(),
                transparent,
                layer_blur_ctx,
                backdrop_ctx,
                page_clip,
                caches.as_mut().map(|(c, h)| (&mut **c, *h)),
                items,
                z,
                &|id: &str| subtrees.iter().any(|c| c.fade.is_some() && c.node.id() == id),
            ) {
                let mut capture = capture;
                capture.mask = capture_composite_mask(scene, plan_masks, capture.node.id());
                if capture.mask.is_some() {
                    // The drop / glow composites it already emitted.
                    for it in &mut items[capture.items_start..] {
                        if let DrawItem::LayerBlurComposite { mask, .. } = it {
                            *mask = capture.mask.clone();
                        }
                    }
                }
                subtrees.push(capture);
            }
        }
        // E8.2: a group's appearance under its contents, in its own z slot before them (its capture, opened
        // above, draws the rest after the last descendant).
        if group_appearance {
            let under = subtrees
                .last_mut()
                .filter(|c| c.node.id() == item.node_id)
                .and_then(|c| c.appearance_under.take());
            items.extend(group_appearance_items(
                scene,
                &item.node_id,
                under.as_deref().unwrap_or(&[]),
                "under",
                z,
                page_clip,
                opts,
                backdrop_ctx,
                layer_blur_ctx,
                plan_masks,
                caches.as_mut().map(|(c, h)| (&mut **c, *h)),
                warnings,
                &|id: &str| subtrees.iter().any(|c| c.fade.is_some() && c.node.id() == id),
            ));
            z += 1;
        }
        // rust-edit-incremental-z4: span bookkeeping (only nodes that fall
        // through to the end of the match — i.e. emit items — are recorded; the
        // skip-filter / empty-container / null-clip `continue`s below produce no
        // items and intentionally leave no span).
        let span_start = items.len();
        let span_z = z;
        if item_skipped_without_z(scene, item, opts, isolate.as_ref(), transparent) {
            continue;
        }
        let _in_capture = tess::outline::enter_capture(!subtrees.is_empty());

        let Some(base_node) = scene.effective_node(&item.node_id) else {
            continue;
        };
        // Viewport culling: a camera plan built for a new cull window copies the
        // items of nodes an earlier window's build of this scene generation already
        // made (`PlanCaches::node_items`), and keeps the ones it makes. Not for a
        // node whose items a split or capture context projects into the camera
        // target or the capture view (a background blur, a blend split, a layer
        // capture, anything inside a subtree capture): those follow the camera.
        let reuse = cull_plan.is_some()
            && subtrees.is_empty()
            && backdrop_ctx.is_none_or(|ctx| {
                !crate::backdrop::plan::has_active_background_blur(base_node)
                    && !crate::backdrop::blend::node_splits(
                        scene,
                        &item.node_id,
                        ctx.transparent_target,
                    )
            })
            && (layer_blur_ctx.is_none() || !crate::layer_blur::plan::needs_layer_capture_in(scene, base_node));
        let (skipped_before, dropped_before) = (warnings.skipped.len(), warnings.dropped.len());
        let degraded_limited_before = crate::text_draw::degraded_zoom_limited_count();
        let pending_before = caches.as_ref().map_or(0, |(c, _)| c.outline_pending_len());
        if reuse {
            if let Some(cached) = caches
                .as_ref()
                .and_then(|(c, _)| c.node_items_get(&item.node_id, z))
            {
                items.extend(cached.items.iter().cloned());
                warnings.skipped.extend(cached.skipped.iter().cloned());
                let z_consumed = cached.z_consumed;
                z += z_consumed;
                if let Some(o) = cull_out.as_deref_mut() {
                    o.copied_nodes += 1;
                }
                if let Some(s) = spans.as_deref_mut() {
                    s.push(NodeSpan {
                        node_id: item.node_id.clone(),
                        start: span_start,
                        len: items.len() - span_start,
                        z: span_z,
                        z_consumed,
                        primary: item.clone(),
                        placement: None,
                    });
                }
                continue;
            }
        }
        // Parity with the WebGPU `displayListAdapter.ts`
        // resolvePaintItemNode path: fold ancestor group/container opacity into
        // the node's own opacity so it flows into the body fill/stroke AND its
        // shadows/text below — exactly as the baked node does on the WebGPU path.
        // Leaf-only multiply-down (no group-compositing layer); the clone happens
        // only when an ancestor actually dims this node.
        // A group composited as one unit (`SubtreeCapture::fade`) fades its layers in its split, not each layer:
        // its opacity leaves their product, and its own paint (a frame's fill) draws at full opacity inside.
        let faded = |id: &str| subtrees.iter().any(|c| c.fade.is_some() && c.node.id() == id);
        let ancestor_opacity = scene.ancestor_opacity_product_except(&item.node_id, faded);
        let own_faded = faded(&item.node_id);
        let dimmed_node;
        let node: &moda_scene::types::RawNode = if ancestor_opacity < 1.0 || own_faded {
            let mut m = base_node.fields().clone();
            let own = if own_faded { 1.0 } else { base_node.f64("opacity").unwrap_or(1.0) };
            m.insert("opacity".into(), Value::from(own * ancestor_opacity));
            dimmed_node = base_node.with_fields(m);
            &dimmed_node
        } else {
            base_node
        };
        // A section (design-editor E7.10) draws its background as an unclipped, unrotated frame body: its fill
        // and stroke stacks through the container lane. Its name tab is the editor's overlay, never rendered.
        let section_body;
        let node: &moda_scene::types::RawNode = if node.node_type() == "section" {
            section_body = section_as_container_body(node);
            &section_body
        } else {
            node
        };
        let node_type = node.node_type().to_string();
        let this_z = z;

        // M3e hook: an active backdrop effect splits the pass BEFORE the
        // node's own body (painter's algorithm — the effect samples only
        // already-drawn content). Stacked backdrops split once each, in list
        // order, each sampling what the one before composited. Splits consume
        // no z.
        if let Some(ctx) = backdrop_ctx {
            for split in crate::backdrop::compute_backdrop_splits(scene, node, ctx, &mut warnings.dropped) {
                items.push(DrawItem::BackdropSplit(Box::new(split)));
            }
        }
        // Everything from here to the loop tail is the node's OWN
        // paint (a coexisting BackdropSplit stays in the main stream above —
        // it must sample the target, not the isolation capture).
        let layer_blur_divert_from = items.len();
        // E3.1 P6: a paint stack plans its own blend splits (`build_stack_items`).
        let mut stacked = false;
        let mut placement = None;

        match node_type.as_str() {
            // M2: star/polygon route through the same body path (inscribed
            // closed nonzero paths, shapeToPathNodeUtils.ts). No drop
            // shadows for them — `silhouette_of` already guards the shadow
            // hooks to rect/ellipse-derived nodes (copy the bug).
            "rectangle" | "container" | "ellipse" | "path" | "star" | "polygon" => {
                // Container: only draws when it has a visible fill or a
                // visible stroke with a width, whatever their paint types
                // (a gradient-only stroke draws too); consumes z either way.
                // A stroke-only container's body has no fill, so it paints
                // only the stroke.
                //
                // E3.1 P6: every visible fill and stroke draws (`paint::layers`): a
                // folded stack is one body as before (`container_body_override`), a
                // real stack one body per layer inside the node's slot.
                let (container_body_override, stack) = match paint::layers::plan(node) {
                    paint::layers::LayerPlan::Single(folded) => (folded, None),
                    paint::layers::LayerPlan::Stack(layers) => (None, Some(layers)),
                };
                stacked = stack.is_some();
                if node_type == "container" && stack.is_none() {
                    let (has_fill, has_stroke) = moda_scene::paint::container_body_gate(
                        container_body_override.as_ref().unwrap_or(node),
                    );
                    if !has_fill && !has_stroke {
                        z += 1;
                        continue;
                    }
                }

                // A clipped body tessellates (an active mask declines the SDF fast path) and keeps doing so when
                // its clips change nothing for it (`clip_mask_ref_for_node`), so it lands on the same MSAA samples
                // as before, just not cut a second time at its frame's edge.
                let clipped = plan_masks.mask_ref_for_node(scene, &item.node_id).is_some();
                let mask = if clipped { plan_masks.clip_mask_ref_for_node(scene, &item.node_id) } else { None };
                // Q-113: empty intersection → production's computeEffectivePageClip
                // returns null and the SHAPE path sets no override — the body keeps
                // the PLAIN page clip and the R8 mask alone clips it
                // (`WebGPUShapeSystem.ts:720-721` `if (clip) …`).
                let eff_clip = crate::effects::effective_page_clip(scene, &item.node_id, page_clip)
                    .unwrap_or(page_clip);
                // M3b/M3f: returns 1 body (solid/gradient/pattern/shader) or
                // 2 (pattern|shader + stroke-only gradient overlay); empty =
                // empty geometry.
                // A node ON a blur-segmented (backdrop) page declines
                // the SDF fast path and tessellates, mirroring WebGPU's
                // `disableOpaqueFastPath`.
                let mut body_is_sdf = false;
                let disable_shape_sdf = clipped || backdrop_ctx.is_some_and(|c| !c.keep_shape_sdf);
                // Live geometry effects (E8.1): a Transform effect with copies
                // resolved to one path per copy (`live_effects::split_copies`);
                // each copy draws its own fill and stroke, in turn, as
                // Illustrator does, so a later copy covers an earlier one's
                // stroke. Copies share the node id, so they skip the per-node
                // tessellation cache (one entry per node would thrash).
                let copy_parts = moda_scene::live_effects::split_copies(node);
                // E8.2: a fill or stroke with effects of its own draws them through the plan's captures.
                let entry_fx = paint::layers::has_entry_effects(node).then(|| {
                    EntryFx::for_node(
                        scene,
                        &item.node_id,
                        layer_blur_ctx,
                        opts,
                        hashes,
                        page_clip,
                        plan_masks,
                    )
                });
                let body_result = match (&copy_parts, &stack) {
                    (Some(parts), _) => {
                        // Copy k draws in its own share of the node's z slot, in
                        // painter's order (as a paint stack's layers do): a
                        // stroke's vertices sit nearer in depth, so ordering by
                        // depth would keep copy 1's stroke over copy 2's fill.
                        let cspan = crate::tess::placed::PLACED_Z_SPAN / parts.len() as f64;
                        let mut all = Vec::new();
                        let mut failed = None;
                        for (k, part) in parts.iter().enumerate() {
                            let cz = this_z as f64 + k as f64 * cspan;
                            let r = match paint::layers::plan(part) {
                                paint::layers::LayerPlan::Stack(layers) => Ok(build_stack_items(
                                    scene,
                                    part,
                                    &layers,
                                    cz,
                                    cspan,
                                    backdrop_ctx,
                                    true,
                                    warnings,
                                    entry_fx.as_ref(),
                                    |layer, lz, slice, w| {
                                        if crate::pattern_paint::layer_pattern(layer).is_some() {
                                            return build_pattern_layer_items(
                                                scene,
                                                layer,
                                                &item.node_id,
                                                lz,
                                                slice,
                                                eff_clip,
                                                plan_masks,
                                                opts.shader_time_s,
                                                None,
                                                None,
                                                w,
                                                |white, w| {
                                                    build_body_item(
                                                        scene, white, lz, eff_clip, None, true, 0.0, None, None, None, w,
                                                    )
                                                },
                                            );
                                        }
                                        build_body_item(
                                            scene,
                                            layer,
                                            lz,
                                            eff_clip,
                                            mask.clone(),
                                            disable_shape_sdf,
                                            opts.shader_time_s,
                                            None,
                                            None,
                                            None,
                                            w,
                                        )
                                    },
                                )),
                                paint::layers::LayerPlan::Single(folded) => {
                                    tess::placed::with_z_span(cspan, || {
                                        build_body_item(
                                            scene,
                                            folded.as_ref().unwrap_or(part),
                                            cz,
                                            eff_clip,
                                            mask.clone(),
                                            disable_shape_sdf,
                                            opts.shader_time_s,
                                            None,
                                            None,
                                            None,
                                            warnings,
                                        )
                                    })
                                    .map(|mut items| {
                                        for it in &mut items {
                                            if let DrawItem::Body { phase, .. }
                                            | DrawItem::SolidShapeSdf { phase, .. }
                                            | DrawItem::GradientShapeSdf { phase, .. } = it
                                            {
                                                *phase = Phase::Transparent;
                                            }
                                        }
                                        items
                                    })
                                }
                            };
                            match r {
                                Ok(items) => all.extend(items),
                                Err(e) => {
                                    failed = Some(e);
                                    break;
                                }
                            }
                        }
                        match failed {
                            Some(e) => Err(e),
                            None => Ok(all),
                        }
                    }
                    (None, Some(layers)) => Ok(build_stack_items(
                        scene,
                        node,
                        layers,
                        this_z as f64,
                        crate::tess::placed::PLACED_Z_SPAN,
                        backdrop_ctx,
                        true,
                        warnings,
                        entry_fx.as_ref(),
                        |layer, lz, slice, w| {
                            if crate::pattern_paint::layer_pattern(layer).is_some() {
                                return build_pattern_layer_items(
                                    scene,
                                    layer,
                                    &item.node_id,
                                    lz,
                                    slice,
                                    eff_clip,
                                    plan_masks,
                                    opts.shader_time_s,
                                    None,
                                    caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                                    w,
                                    |white, w| {
                                        build_body_item(scene, white, lz, eff_clip, None, true, 0.0, None, None, None, w)
                                    },
                                );
                            }
                            build_body_item(
                                scene,
                                layer,
                                lz,
                                eff_clip,
                                mask.clone(),
                                disable_shape_sdf,
                                opts.shader_time_s,
                                None,
                                None,
                                caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                                w,
                            )
                        },
                    )),
                    (None, None) => {
                        let body_node = container_body_override.as_ref().unwrap_or(node);
                        if !clipped && eff_clip == page_clip && subtrees.is_empty()
                            && !caches.as_ref().is_some_and(|(c, _)| c.outline_progressive()) {
                            placement = local_path_placement(scene, body_node);
                        }
                        build_body_item(
                        scene,
                        body_node,
                        this_z as f64,
                        eff_clip,
                        mask,
                        disable_shape_sdf,
                        opts.shader_time_s,
                        None,
                        None,
                        caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                        warnings,
                    )
                    },
                };
                // A stack's shadow pads by its widest stroke; its opaque SDF fill
                // covers the shadow only when no stroke band reaches past it.
                let shadow_node = stack.as_ref().map(|_| {
                    let mut m = node.fields().clone();
                    moda_scene::paint::set_strokes(
                        &mut m,
                        paint::layers::widest_stroke(node).into_iter().collect(),
                    );
                    node.with_fields(m)
                });
                let stack_has_stroke = stack
                    .as_ref()
                    .is_some_and(|l| l.iter().any(|l| l.is_stroke));
                let opaque_sdf_covers_shadow = body_result
                    .as_ref()
                    .map(|bodies| {
                        bodies.iter().any(|d| {
                            matches!(
                                d,
                                DrawItem::SolidShapeSdf {
                                    opaque_sdf_covers_shadow: true,
                                    ..
                                } | DrawItem::GradientShapeSdf {
                                    opaque_sdf_covers_shadow: true,
                                    ..
                                }
                            )
                        })
                    })
                    .unwrap_or(false)
                    && !stack_has_stroke;
                // M3d hooks (SEAMS change log). Drop shadow at the body's z
                // (shadow-first in the same bucket); inner-shadow quad at
                // z + 0.5 in its OWN bucket above the body
                // (`WebGPUPathRenderer.ts:727-744`). Shadows take the body's
                // effective clip (the page clip intersected with the clipping
                // ancestors' boxes): a child's shadow stops at its frame's
                // clip, as in Figma (production used the plain page clip, so a
                // shadow leaked out of a clipping board).
                // One quad per drop shadow, back to front: same-bucket shadows
                // draw in plan order, so `effects[0]` lands furthest behind.
                for sh in crate::effects::drop_shadow_instances(
                    scene,
                    shadow_node.as_ref().unwrap_or(node),
                    this_z as f64,
                    eff_clip,
                    opaque_sdf_covers_shadow,
                ) {
                    items.push(DrawItem::Shadow {
                        node_id: item.node_id.clone(),
                        z: this_z as f64,
                        instance: sh,
                        inner: false,
                    });
                }
                match body_result {
                    Ok(bodies) if !bodies.is_empty() => {
                        // A stack's layer clones carry no inner shadow: the node's
                        // quad draws it once, over every layer.
                        body_is_sdf = stack.is_none()
                            && bodies.iter().any(|d| {
                                matches!(
                                    d,
                                    DrawItem::SolidShapeSdf { .. }
                                        | DrawItem::GradientShapeSdf { .. }
                                )
                            });
                        items.extend(bodies);
                    }
                    // A brush stroke left out of this live frame (never placed yet) draws on a later one.
                    Ok(_)
                        if caches
                            .as_ref()
                            .is_some_and(|(c, _)| c.placement_deferred_last(&item.node_id)) => {}
                    Ok(_) => warnings
                        .skipped
                        .push(format!("{} ({node_type}): empty geometry", item.node_id)),
                    Err(e) => warnings.skipped.push(format!("{}: {e}", item.node_id)),
                }
                // An SDF body composites its inner shadow IN-FRAGMENT
                // (baked into the instance), so the separate inner-shadow quad is
                // suppressed — emitting it too would double the inset shadow.
                // The SDF lane only takes a node with at most one inner shadow
                // (`classify_*_shape` decline more), so a node with several
                // tessellates and gets one quad per inner shadow, back to front.
                let inner_z = this_z as f64 + crate::seams::INNER_SHADOW_Z_OFFSET;
                if !body_is_sdf {
                    for sh in
                        crate::effects::inner_shadow_instances(scene, node, inner_z, eff_clip)
                    {
                        items.push(DrawItem::Shadow {
                            node_id: format!("{}:innershadow", item.node_id),
                            z: inner_z,
                            instance: sh,
                            inner: true,
                        });
                    }
                }
                // M3a (SEAMS change log): shape text frame at z + 0.5
                // (`renderShapeText`; containers are NOT text-in-shape —
                // TEXT_IN_SHAPE_TYPES). Synthetic node takes the PLAIN page
                // clip (`_pageClipBoundsOverride` = getPageClipFromScene).
                if node_type != "container" {
                    match crate::text_draw::build_shape_text_item(
                        scene,
                        node,
                        this_z as f64 + crate::seams::SHAPE_TEXT_Z_OFFSET,
                        page_clip,
                        opts.best_effort_text,
                        opts.scale,
                        caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                    ) {
                        Ok(crate::text_draw::TextBuildOutcome::Item(text_item)) => {
                            items.push(*text_item)
                        }
                        Ok(crate::text_draw::TextBuildOutcome::Degraded { reason }) => {
                            warnings.skipped.push(format!(
                                "{} (shape-text): degraded-text: {reason}",
                                item.node_id
                            ))
                        }
                        Ok(crate::text_draw::TextBuildOutcome::Empty) => {}
                        Err(e) => warnings
                            .skipped
                            .push(format!("{} (shape-text): {e}", item.node_id)),
                    }
                }
                z += 1;
            }
            "line" => {
                // M2: body (stroked path, dash-capable) at z, arrowheads at
                // z + ARROWHEAD_Z_OFFSET (WebGPUShapeSystem.renderLine
                // :873-983). M4b: connector labels + arc-length text-gap
                // carving — the label batch rides z + SHAPE_TEXT_Z_OFFSET
                // with the PLAIN page clip (renderConnectorText computes its
                // own page clip = getPageClipFromScene), and the gap is
                // carved into the body via `line_parts_with_gap`.
                let mask = plan_masks.mask_ref_for_node(scene, &item.node_id);
                // Q-113: null effective clip → plain page clip, like the shape
                // path (`renderLine` sets `_pageClipBoundsOverride` only
                // `if (clipLine)`, WebGPUShapeSystem.ts:885/:932).
                let eff_clip = crate::effects::effective_page_clip(scene, &item.node_id, page_clip)
                    .unwrap_or(page_clip);
                let label = match crate::text_draw::plan_connector_text(
                    scene,
                    node,
                    this_z as f64 + crate::seams::SHAPE_TEXT_Z_OFFSET,
                    page_clip,
                    opts.scale,
                    caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                ) {
                    Ok(plan) => plan,
                    Err(e) => {
                        warnings
                            .skipped
                            .push(format!("{} (connector-text): {e}", item.node_id));
                        crate::text_draw::ConnectorTextPlan::default()
                    }
                };
                // E3.1 P6: every visible stroke layer draws, each with its own
                // arrowheads, in its slice of the line's slot.
                let line_entry_fx = paint::layers::has_entry_effects(node).then(|| {
                    EntryFx::for_node(
                        scene,
                        &item.node_id,
                        layer_blur_ctx,
                        opts,
                        hashes,
                        page_clip,
                        plan_masks,
                    )
                });
                let line_result = match paint::layers::plan(node) {
                    paint::layers::LayerPlan::Stack(layers) => {
                        stacked = true;
                        Ok(build_stack_items(
                            scene,
                            node,
                            &layers,
                            this_z as f64,
                            crate::tess::placed::PLACED_Z_SPAN,
                            backdrop_ctx,
                            false,
                            warnings,
                            line_entry_fx.as_ref(),
                            |layer, lz, slice, w| {
                                if crate::pattern_paint::layer_pattern(layer).is_some() {
                                    return build_pattern_layer_items(
                                        scene,
                                        layer,
                                        &item.node_id,
                                        lz,
                                        slice,
                                        eff_clip,
                                        plan_masks,
                                        opts.shader_time_s,
                                        None,
                                        caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                                        w,
                                        |white, w| {
                                            build_line_items(scene, white, lz, slice * 0.5, eff_clip, None, label.gap, None, w)
                                        },
                                    );
                                }
                                build_line_items(
                                    scene,
                                    layer,
                                    lz,
                                    slice * 0.5,
                                    eff_clip,
                                    mask.clone(),
                                    label.gap,
                                    caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                                    w,
                                )
                            },
                        ))
                    }
                    paint::layers::LayerPlan::Single(folded) => build_line_items(
                        scene,
                        folded.as_ref().unwrap_or(node),
                        this_z as f64,
                        crate::seams::ARROWHEAD_Z_OFFSET,
                        eff_clip,
                        mask,
                        label.gap,
                        caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                        warnings,
                    ),
                };
                match line_result {
                    Ok(line_items) if !line_items.is_empty() => items.extend(line_items),
                    Ok(_) => warnings.skipped.push(format!(
                        "{} (line): no renderable geometry (invalid points or gradient stroke)",
                        item.node_id
                    )),
                    Err(e) => warnings.skipped.push(format!("{}: {e}", item.node_id)),
                }
                match label.outcome {
                    Some(crate::text_draw::TextBuildOutcome::Item(text_item)) => {
                        items.push(*text_item)
                    }
                    Some(crate::text_draw::TextBuildOutcome::Degraded { reason }) => {
                        warnings.skipped.push(format!(
                            "{} (connector-text): degraded-text: {reason}",
                            item.node_id
                        ))
                    }
                    Some(crate::text_draw::TextBuildOutcome::Empty) | None => {}
                }
                z += 1;
            }
            "blend" => {
                // E8.12: a blend node draws the steps between its objects
                // (`moda_scene::blend`), one z per step.
                let mask = plan_masks.mask_ref_for_node(scene, &item.node_id);
                z += build_blend_items(
                    scene,
                    node,
                    this_z,
                    page_clip,
                    items,
                    warnings,
                    opts.shader_time_s,
                    mask,
                    caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                );
            }
            "richtext" => {
                // M3a (SEAMS change log): MSDF text batch at the node's z;
                // fallback-class nodes (browser Canvas2D path) draw nothing
                // and are reported (manifest-Degraded until M4's cosmic-text
                // tier). Consumes z even when not drawn
                // (WebGPUShapeSystem.renderRichText). Text takes the
                // EFFECTIVE clip and never binds the R8 mask (the MSDF
                // pipelines carry no mask group — copy the bug). A mask the
                // effective clip can't hold (a path, an ellipse, …) clips it
                // through a mask capture instead (#2148, `needs_mask_capture`).
                //
                // Q-113: a NULL effective clip (disjoint ancestor mask AABBs)
                // draws NOTHING in production — vector
                // `WebGPUTextSystem.ts:663-665` and MSDF `:883-886` both
                // early-return on a null computeEffectivePageClip. z is still
                // consumed (the early return happens inside renderText).
                //
                // Table cell text overflow extends that with a
                // FLAG-GATED second narrowing: a node carrying a truthy
                // `clipTextToBounds` (`expand_table_node` sets it on each cell's
                // richtext content node) additionally intersects with its OWN
                // canvas AABB, so cell text cannot bleed into neighbouring cells.
                // Without the flag `clip_to_node_box` returns the clip verbatim,
                // which is what keeps every pre-flag doc's plan byte-identical.
                //
                // An empty intersection takes the SAME path as the null effective
                // clip above — consume z, draw nothing — because it goes through
                // the same `intersect_rect`, and both TS arms already early-return
                // on a null clip. The mirrored TS narrowing is
                // `applyNodeTextBoundsClip` (`MaskContextUtils.ts:137`), wired into
                // the vector arm `WebGPUTextSystem.ts:740-747` and the MSDF arm
                // `:980-985` in the same PR.
                //
                // This arm is the SINGLE call site of `build_richtext_item` and is
                // shared by the page lane and the camera lane, and the incremental
                // patcher re-enters it via `rebuild_node_items` — so the narrowing
                // covers all three. On a text-cache HIT `patch_cached_batch`
                // re-patches `page_clip_bounds` from this value, so a cached
                // layout still picks up a moved/resized cell's narrower clip.
                //
                // RESIDUAL: the box is an AABB, so the clip is LOOSE under
                // rotation — same accepted approximation as container
                // `clipContent` (see `clip_to_node_box`).
                let Some(eff_clip) =
                    crate::effects::effective_page_clip(scene, &item.node_id, page_clip)
                        .and_then(|c| crate::effects::clip_to_node_box(scene, node, c))
                else {
                    z += 1;
                    continue;
                };
                match crate::text_draw::build_richtext_item(
                    scene,
                    node,
                    this_z as f64,
                    eff_clip,
                    opts.best_effort_text,
                    opts.shader_time_s,
                    opts.scale,
                    caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                ) {
                    Ok(crate::text_draw::TextBuildOutcome::Item(text_item)) => {
                        items.push(*text_item)
                    }
                    Ok(crate::text_draw::TextBuildOutcome::Degraded { reason }) => {
                        warnings.skipped.push(format!(
                            "{} (richtext): degraded-text: {reason}",
                            item.node_id
                        ))
                    }
                    Ok(crate::text_draw::TextBuildOutcome::Empty) => {}
                    Err(e) => warnings.skipped.push(format!("{}: {e}", item.node_id)),
                }
                z += 1;
            }
            other => {
                warnings
                    .skipped
                    .push(format!("{} ({other}): unhandled node type", item.node_id));
                z += 1;
            }
        }
        // Capture shadows and glows (`layer_blur::shadow`): the drop / inner
        // shadows of a shape the SDF shadow lane can't draw, and every shape's
        // glows, from a capture of its silhouette. The drop-shadow and glow
        // composites go before the node's own items (same z bucket, drawn
        // before the body, the glow over the drop shadows), the inner one after
        // them at z + INNER_SHADOW_Z_OFFSET. None consumes z. Each is clipped
        // like the body by the clipping frames above it: their boxes (the
        // effective clip) and the innermost rounded one's corners.
        if let Some(lb_ctx) = layer_blur_ctx {
            if items.len() > layer_blur_divert_from
                && crate::layer_blur::shadow::has_capture_shadow(node)
                && !crate::layer_blur::shadow::is_subtree_capture_node(node)
            {
                let clip = crate::effects::effective_page_clip(scene, &item.node_id, page_clip)
                    .unwrap_or(page_clip);
                let clip_rrect = crate::effects::clip_rounded_rect(scene, &item.node_id);
                let mask = capture_composite_mask(scene, plan_masks, &item.node_id);
                let sil = crate::layer_blur::shadow::silhouette_node(node);
                let capture = build_shadow_capture_items(
                    scene,
                    &sil,
                    this_z as f64,
                    INFINITE_CLIP,
                    opts,
                    caches.as_mut().map(|(c, h)| (&mut **c, *h)),
                );
                if let Some(bounds) = crate::layer_blur::shadow::items_canvas_bounds(&capture) {
                    use crate::layer_blur::shadow::CaptureKind;
                    let opacity = node.f64("opacity").unwrap_or(1.0);
                    let mut before = layer_blur_divert_from;
                    for kind in CaptureKind::for_node(node) {
                        let Some(mut group) = crate::layer_blur::shadow::plan_shadow_group(
                            scene,
                            node,
                            lb_ctx,
                            kind,
                            capture.clone(),
                            bounds,
                            opacity,
                        ) else {
                            continue;
                        };
                        let inner = kind.is_inner();
                        (group.cache_key, group.identity_key) = crate::residency::capture_key(
                            scene,
                            hashes,
                            [item.node_id.as_str()],
                            &group,
                            page_clip,
                            opts,
                        );
                        let composite = DrawItem::LayerBlurComposite {
                            node_id: format!("{}:{}", item.node_id, kind.suffix()),
                            z: if inner {
                                this_z as f64 + crate::seams::INNER_SHADOW_Z_OFFSET
                            } else {
                                this_z as f64
                            },
                            group: std::sync::Arc::new(group),
                            page_clip_bounds: clip,
                            clip_rrect,
                            mask: mask.clone(),
                        };
                        if inner {
                            items.push(composite);
                        } else {
                            items.insert(before, composite);
                            before += 1;
                        }
                    }
                }
            }
        }
        // Divert an eligible layer-blurred node's items into a
        // capture group and replace them with ONE composite at the body's z.
        // Runs after the arm completed, so `z` already advanced — the diversion
        // never disturbs the render-z counter, and plans without a layer blur
        // are byte-identical to before the feature (the params fn declines).
        // A blended node's composite takes a blend split (`composite_blend_split`):
        // without a split context (or inside a paint stack, whose own splits
        // carry the mode) the node stays unfiltered and blends as a body.
        // A multiply or screen filtered leaf composites with its fixed-function state instead
        // (`filter_composite_fixed_blend`; the page's split predicates agree, so a page with no other split
        // has no split context).
        let fixed_capture_blend = crate::backdrop::blend::filter_composite_fixed_blend(
            scene,
            &item.node_id,
            backdrop_ctx.map_or(transparent, |c| c.transparent_target),
        )
        .filter(|_| !stacked);
        let composite_blend = crate::backdrop::blend::composite_blend_split(node)
            .filter(|_| !crate::backdrop::blend::is_blend_subtree_node(node))
            .filter(|_| fixed_capture_blend.is_none());
        let blend_split_ctx = backdrop_ctx.filter(|_| !stacked);
        // Whether a split already blends this node's own items as one layer.
        let mut blended_as_unit = false;
        if let Some(lb_ctx) = layer_blur_ctx.filter(|_| {
            composite_blend.is_none()
                || blend_split_ctx.is_some()
                || crate::backdrop::blend::is_blend_unit_leaf(node)
        }) {
            // Text captures what its glyphs draw, which may reach past its box; a filter shadow is cast by what
            // the node draws, whatever its box.
            let drawn = ((node.node_type() == "richtext" || !moda_scene::effect::filter_shadows(node).is_empty())
                && items.len() > layer_blur_divert_from)
                .then(|| {
                    crate::layer_blur::shadow::all_items_canvas_bounds(
                        &items[layer_blur_divert_from..],
                    )
                })
                .flatten();
            let mut empty_mask_capture = false;
            let params =
                crate::layer_blur::plan::compute_layer_blur_params_over(scene, node, lb_ctx, drawn, &mut warnings.skipped)
                    .or_else(|| {
                        // Text has no mask binding. Capture its clipped drawing
                        // and let the composite sample the mask. Enclosing subtree
                        // captures remap that composite's mask like any other child.
                        let [cx, cy, cw, ch] =
                            crate::effects::effective_page_clip(scene, &item.node_id, page_clip)?;
                        let clip = crate::backdrop::plan::Rect {
                            x: cx as f64,
                            y: cy as f64,
                            w: cw as f64,
                            h: ch as f64,
                        };
                        let drawn = crate::layer_blur::shadow::all_items_canvas_bounds(
                            &items[layer_blur_divert_from..],
                        )?;
                        if !drawn.finite() || !clip.finite() {
                            return None;
                        }
                        let Some(bounds) = drawn.intersect(&clip) else {
                            // A scrolling glyph can lie wholly outside its clip.
                            // This is empty coverage, not a failed mask capture.
                            empty_mask_capture = crate::layer_blur::plan::needs_mask_capture(scene, node);
                            return None;
                        };
                        crate::layer_blur::plan::compute_mask_capture_params(scene, node, lb_ctx, bounds, &mut warnings.skipped)
                    });
            if empty_mask_capture {
                items.truncate(layer_blur_divert_from);
            }
            // Noise grains the node's shape, painted or not
            // (`LayerBlurGroup::coverage`): a node with no paint still diverts.
            let coverage = params
                .as_ref()
                .filter(|p| !p.filters.noise.is_empty())
                .and_then(|_| crate::layer_blur::shadow::noise_coverage_node(node))
                .map(|sil| {
                    build_shadow_capture_items(scene, &sil, this_z as f64, INFINITE_CLIP, opts, None)
                })
                .unwrap_or_default();
            if items.len() > layer_blur_divert_from || !coverage.is_empty() {
                // The node's own blend mode is its composite's; what else its items hold must be exact in the
                // capture (`capture_cannot_hold`, as for a filtered subtree), or the node stays unfiltered.
                let own_blend_lifted = composite_blend.is_some() || fixed_capture_blend.is_some();
                let params = params.filter(|_| {
                    let declined = capture_cannot_hold(&items[layer_blur_divert_from..], own_blend_lifted);
                    if declined {
                        warnings.skipped.push(capture_refusal(&item.node_id));
                    }
                    !declined
                });
                if let Some(params) = params {
                    // Filter shadows alone (`LayerFilters::shadows_alone`): the capture is a copy of the node's
                    // items, which stay and draw directly over the shadows' composite. A blended node's
                    // composite carries its blend, so its picture goes through the capture as with any filter;
                    // so does a picture that holds a capture already (`holds_capture`).
                    let shadows_alone = params.filters.shadows_alone
                        && composite_blend.is_none()
                        && fixed_capture_blend.is_none()
                        && !holds_capture(&items[layer_blur_divert_from..]);
                    let mut diverted: Vec<DrawItem> = if shadows_alone {
                        items[layer_blur_divert_from..].to_vec()
                    } else {
                        items.drain(layer_blur_divert_from..).collect()
                    };
                    let composite_mask = capture_composite_mask(scene, plan_masks, &item.node_id);
                    unmask_captured(&mut diverted, composite_mask.as_ref());
                    // The node's own paint-stack group (a translucent fill and stroke) composites inside the
                    // capture as it does on the page.
                    crate::backdrop::blend::retarget_isolation_splits(
                        &mut diverted,
                        params.region_canvas,
                        params.width_px,
                        params.height_px,
                    );
                    if own_blend_lifted {
                        // The capture holds the node's own paint (source-over);
                        // the split below (or the composite's blend state) blends the filtered result.
                        for it in &mut diverted {
                            if let DrawItem::Body { blend_mode, .. } = it {
                                *blend_mode = BlendMode::Normal;
                            }
                        }
                    }
                    let mut filters = params.filters;
                    filters.composite_blend = fixed_capture_blend.unwrap_or_default();
                    filters.shadows_alone = shadows_alone;
                    let mut group = crate::seams::LayerBlurGroup {
                        items: diverted,
                        region_canvas: params.region_canvas,
                        extent_canvas: params.extent_canvas,
        reach_canvas: params.reach_canvas,
                        width_px: params.width_px,
                        height_px: params.height_px,
                        filters,
                        cache_key: None,
                        identity_key: None,
                        coverage,
                        masks: Vec::new(),
                    };
                    (group.cache_key, group.identity_key) = crate::residency::capture_key(
                        scene,
                        hashes,
                        [item.node_id.as_str()],
                        &group,
                        page_clip,
                        opts,
                    );
                    // A capture holds the node's picture, not the part of it on the page (see
                    // `divert_subtree_filter`); after the content key.
                    widen_captured_clips(scene, &mut group.items, &item.node_id, INFINITE_CLIP);
                    let [rx, ry, rw, rh] = group.region_canvas;
                    let splits = composite_blend.zip(blend_split_ctx).and_then(|(mode, ctx)| {
                        crate::backdrop::blend::compute_region_blend_splits(
                            node,
                            ctx,
                            Some(crate::backdrop::plan::Rect {
                                x: rx,
                                y: ry,
                                w: rw,
                                h: rh,
                            }),
                            mode,
                        )
                    });
                    blended_as_unit = splits.is_some();
                    let blend_end = splits.map(|(begin, end)| {
                        items.push(DrawItem::BackdropSplit(Box::new(begin)));
                        DrawItem::BackdropSplit(Box::new(end))
                    });
                    let composite = DrawItem::LayerBlurComposite {
                        node_id: item.node_id.clone(),
                        z: this_z as f64,
                        group: std::sync::Arc::new(group),
                        // Clipped like the body by the clipping frames
                        // above (`compute_layer_blur_params` declines any
                        // other clip that would cut the capture).
                        page_clip_bounds: crate::effects::effective_page_clip(
                            scene,
                            &item.node_id,
                            page_clip,
                        )
                        .unwrap_or(page_clip),
                        clip_rrect: crate::effects::clip_rounded_rect(scene, &item.node_id),
                        mask: composite_mask,
                    };
                    if shadows_alone {
                        // Before the node's own items, in its z bucket, as a capture shadow's composite.
                        items.insert(layer_blur_divert_from, composite);
                    } else {
                        items.push(composite);
                    }
                    items.extend(blend_end);
                }
            }
        }
        // A text under a mask it can't honour that no capture took (no capture context, or the capture declined):
        // it draws clipped by the mask's box only.
        if items[layer_blur_divert_from..].iter().any(|i| matches!(i, DrawItem::Text(_)))
            && crate::layer_blur::plan::needs_mask_capture(scene, node)
        {
            warnings.skipped.push(format!(
                "{} (richtext): clip mask: clipped by the mask's bounding box only",
                item.node_id
            ));
        }
        // A text's or a line's blend mode (`is_blend_unit_leaf`): its glyphs,
        // effects, arrowheads and label draw plainly into a split that blends
        // them over the backdrop as one layer (a multiply text multiplies onto
        // what's behind it). Its bodies drop their own mode: the split carries it.
        if let Some((mode, ctx)) = composite_blend
            .filter(|_| !blended_as_unit && crate::backdrop::blend::is_blend_unit_leaf(node))
            .zip(blend_split_ctx)
        {
            if items.len() > layer_blur_divert_from {
                let bounds = crate::layer_blur::shadow::all_items_canvas_bounds(
                    &items[layer_blur_divert_from..],
                );
                if let Some((begin, end)) =
                    crate::backdrop::blend::compute_region_blend_splits(node, ctx, bounds, mode)
                {
                    for it in &mut items[layer_blur_divert_from..] {
                        if let DrawItem::Body { blend_mode, .. } = it {
                            *blend_mode = BlendMode::Normal;
                        }
                    }
                    items.push(DrawItem::BackdropSplit(Box::new(end)));
                    items.insert(
                        layer_blur_divert_from,
                        DrawItem::BackdropSplit(Box::new(begin)),
                    );
                }
            }
        }
        // Exact backdrop-reading blends (overlay, darken, …): wrap the node's
        // blended bodies in a BlendBegin/BlendEnd split pair so they draw alone
        // (plain source-over) into a cleared region, then composite over the
        // saved backdrop (`backdrop::blend`). Without a split context, or if
        // nothing lands on the target, the bodies keep the fixed-function
        // fallback (`BlendMode::fixed_function`). A blended subtree's root
        // draws plainly (its subtree's split carries the mode), and inside one
        // the backdrop starts cleared, so multiply splits too.
        let blend_root = subtrees
            .last()
            .is_some_and(|c| c.blend.is_some() && c.node.id() == item.node_id);
        let in_blend_subtree = subtrees.iter().any(|c| c.blend.is_some());
        if let Some(ctx) = backdrop_ctx
            .filter(|_| !stacked && !blend_root)
            .map(|c| c.over_transparent(c.transparent_target || in_blend_subtree))
        {
            let ctx = &ctx;
            let is_split_body = |i: &DrawItem| matches!(i, DrawItem::Body { blend_mode, .. } if blend_mode.split_id_over(ctx.transparent_target).is_some());
            let own = &items[layer_blur_divert_from..];
            if let (Some(first), Some(last)) = (
                own.iter().position(is_split_body),
                own.iter().rposition(is_split_body),
            ) {
                if let Some((begin, end)) = crate::backdrop::compute_blend_splits(scene, node, ctx)
                {
                    let (first, last) = (
                        layer_blur_divert_from + first,
                        layer_blur_divert_from + last,
                    );
                    for it in &mut items[first..=last] {
                        if let DrawItem::Body { blend_mode, .. } = it {
                            *blend_mode = BlendMode::Normal;
                        }
                    }
                    items.insert(last + 1, DrawItem::BackdropSplit(Box::new(end)));
                    items.insert(first, DrawItem::BackdropSplit(Box::new(begin)));
                }
            }
        }
        // rust-edit-incremental-z4: record this node's item span. Reached only by
        // arms that fall through the match (item-producing nodes); the
        // `continue`-ing skip/empty/null-clip paths above never get here, so the
        // span set holds exactly the nodes the patcher can rebuild in isolation.
        if let Some(s) = spans.as_deref_mut() {
            s.push(NodeSpan {
                node_id: item.node_id.clone(),
                start: span_start,
                len: items.len() - span_start,
                z: span_z,
                z_consumed: z - span_z,
                primary: item.clone(),
                placement: placement.and_then(|recipe| certify_placement(scene, recipe, &items[span_start..])),
            });
        }
        // A node that refused geometry (a `dropped` line) is rebuilt every time, and
        // so is one whose Degraded text was sized to this zoom (coarse or skipped).
        if reuse
            && warnings.dropped.len() == dropped_before
            && crate::text_draw::degraded_zoom_limited_count() == degraded_limited_before
        {
            if let Some((c, _)) = caches.as_mut() {
                c.node_items_put(
                    &item.node_id,
                    span_z,
                    z - span_z,
                    &items[span_start..],
                    &warnings.skipped[skipped_before..],
                    pending_before,
                );
            }
        }
    }
    while let Some(done) = subtrees.pop() {
        if let Some(over) = done.appearance_over.as_deref() {
            items.extend(group_appearance_items(
                scene,
                done.node.id(),
                over,
                "over",
                z,
                done.page_clip,
                opts,
                backdrop_ctx,
                layer_blur_ctx,
                plan_masks,
                caches.as_mut().map(|(c, h)| (&mut **c, *h)),
                warnings,
                &|id: &str| {
                    (done.fade.is_some() && id == done.node.id())
                        || subtrees.iter().any(|c| c.fade.is_some() && c.node.id() == id)
                },
            ));
            z += 1;
        }
        finish_subtree_capture(
            scene,
            layer_blur_ctx,
            backdrop_ctx,
            done,
            hashes,
            &plan_masks.masks,
            opts,
            items,
            z,
            warnings,
        );
    }
    if let Some(r) = units {
        r.close(dl.items.len(), items.len(), z, warnings);
    }
    z
}

/// The render-z slots `items` reach: one past the highest slot any of them, or any item inside a capture they
/// composite, sits in (0 when none carries a z). This is what depth has to order (`depth::ensure_render_z`):
/// slots a culled range consumed draw nothing and reach no depth.
pub(crate) fn drawn_z_end(items: &[DrawItem]) -> u64 {
    items
        .iter()
        .map(|item| match item {
            DrawItem::Body { z, .. }
            | DrawItem::Shadow { z, .. }
            | DrawItem::SolidShapeSdf { z, .. }
            | DrawItem::GradientShapeSdf { z, .. } => z.max(0.0) as u64 + 1,
            DrawItem::Text(batch) => batch.z.max(0.0) as u64 + 1,
            DrawItem::LayerBlurComposite { z, group, .. } => (z.max(0.0) as u64 + 1)
                .max(drawn_z_end(&group.items))
                .max(drawn_z_end(&group.coverage)),
            DrawItem::BackdropSplit(_) | DrawItem::PageFill { .. } => 0,
        })
        .max()
        .unwrap_or(0)
}

/// A subtree capture in progress: the shadows of a group or of a frame
/// without a fill that has children, and the layer filters of a group or of
/// any frame with children
/// (`layer_blur::shadow::is_subtree_capture_node`), drawn from its whole
/// subtree as in Figma. Opened at the root's display-list item (the drop
/// shadow composite goes out first, behind every descendant), closed once
/// the loop passes the subtree's range (the inner shadow over them, then the
/// layer filters divert everything the range drew).
struct SubtreeCapture {
    node: moda_scene::types::RawNode,
    /// End (exclusive) of the subtree's display-list range.
    dl_end: usize,
    /// Where the subtree's draw items start in the plan.
    items_start: usize,
    /// The render-z of the subtree's first item.
    z_start: u64,
    /// The inner shadow and inner glow groups (with their composites'
    /// node-id suffixes), composited over the last descendant.
    inner: Vec<(String, crate::seams::LayerBlurGroup)>,
    /// The root's effective clip (its composites clip like its descendants).
    clip: [f32; 4],
    /// The innermost rounded clipping frame above the root
    /// (`effects::clip_rounded_rect`), on top of `clip`.
    clip_rrect: Option<crate::seams::ClipRRect>,
    /// The root has an active layer filter.
    filtered: bool,
    /// The subtree's node ids (the root first) for the captures' content keys;
    /// empty outside the retained lanes.
    node_ids: Vec<String>,
    /// The page clip the subtree's items were planned under.
    page_clip: [f32; 4],
    /// The root's blend mode as a split id (`backdrop::blend::composite_blend_split`):
    /// everything the subtree drew (its shadows and filtered composite
    /// included) blends as one layer, as in Figma.
    blend: Option<u32>,
    /// The root's clip mask when a clip-mask group is above it
    /// (`capture_composite_mask`): its composites take it. Set by the caller
    /// once the capture opens (it holds the plan's mask table).
    mask: Option<crate::seams::MaskRef>,
    /// A group painting its appearance (E8.2): the entries it draws under its contents (taken by the caller
    /// at the root's item) and over them (drawn when the range closes, before the capture finishes).
    appearance_under: Option<Vec<moda_scene::appearance::Entry>>,
    appearance_over: Option<Vec<moda_scene::appearance::Entry>>,
    /// A motion unit's fixed-function composite blend (`crate::motion_capture`, a screen or multiply group):
    /// `blend` is then `None`, and the capture composites with this instead of inside a split.
    fixed_blend: Option<crate::seams::CaptureBlend>,
    /// The root's opacity when it composites as one unit (`moda_scene::scene::is_opacity_group`): everything the
    /// subtree drew goes into one split, faded by it, and its layers (the root's own paint included) draw at full
    /// opacity inside.
    fade: Option<f64>,
}

/// Open a subtree capture at display-list item `index` when its node is a
/// subtree-capture root with a capture shadow or a layer filter: emits the
/// drop-shadow composite at render-z `z` (before any descendant), and plans
/// the inner shadow for [`finish_subtree_capture`]. Also opened for a blended
/// group or frame with children (`blend_ctx`, the plan's split context), whose
/// range [`finish_subtree_capture`] wraps in a blend split.
#[allow(clippy::too_many_arguments)]
fn begin_subtree_capture(
    scene: &Scene,
    dl: &moda_scene::DisplayList,
    index: usize,
    opts: &RenderOptions,
    isolate: Option<&std::collections::HashSet<&str>>,
    transparent: bool,
    lb_ctx: Option<&crate::backdrop::PlanCtx>,
    blend_ctx: Option<&crate::backdrop::PlanCtx>,
    page_clip: [f32; 4],
    mut caches: Option<CacheCtx<'_>>,
    items: &mut Vec<DrawItem>,
    z: u64,
    faded: &dyn Fn(&str) -> bool,
) -> Option<SubtreeCapture> {
    let hashes = caches.as_ref().map(|(_, h)| *h);
    use crate::layer_blur::shadow as sh;
    let root_id = dl.items[index].node_id.as_str();
    let root = scene.effective_node(root_id)?;
    let blend = blend_ctx
        .and(crate::backdrop::blend::composite_blend_split(root))
        .filter(|_| crate::backdrop::blend::is_blend_subtree_node(root));
    // Under `--isolate` naming only a side of its appearance, the group draws that side alone: no shadows,
    // filters or blend of its own.
    let root_drawn = !opts.exclude.iter().any(|e| e == root_id)
        && isolate.is_none_or(|iso| scene.is_node_isolated(root_id, iso));
    let blend = blend.filter(|_| root_drawn);
    // A moving frame's motion unit (`crate::motion_capture`) in screen or multiply: its capture composites with
    // that mode's fixed-function blend, and needs no split.
    let fixed_blend = (root_drawn && crate::motion_capture::is_unit(root_id))
        .then(|| {
            crate::backdrop::blend::filter_composite_fixed_blend(
                scene,
                root_id,
                blend_ctx.map_or(transparent, |c| c.transparent_target),
            )
        })
        .flatten();
    let blend = blend.filter(|_| fixed_blend.is_none());
    // A group (or a frame with children) at an opacity between 0 and 1 composites as one unit: one split, faded.
    let fade = (root_drawn && blend_ctx.is_some() && fixed_blend.is_none() && moda_scene::scene::is_opacity_group(root))
        .then(|| root.f64("opacity").unwrap_or(1.0));
    // Shadows come from the subtree for a group or an UNFILLED frame (a filled
    // frame keeps its box shadow); layer filters filter the whole flattened
    // subtree of any group or frame with children, as in Figma.
    let shadowed = root_drawn && lb_ctx.is_some() && sh::is_subtree_capture_node(root) && sh::has_capture_shadow(root);
    let filtered = root_drawn
        && lb_ctx.is_some()
        && crate::backdrop::blend::is_blend_subtree_node(root)
        && crate::layer_blur::has_active_layer_filter(root);
    let lb_ctx = lb_ctx.filter(|_| shadowed || filtered);
    let appearance = paints_group_appearance(scene, &dl.items[index], opts, isolate).then(|| {
        let (under, over) = moda_scene::appearance::group_split(root);
        let keep = |side: &str, entries: Vec<moda_scene::appearance::Entry>| {
            let excluded = opts.exclude.iter().any(|e| *e == group_appearance_id(root_id, side));
            if !excluded && group_side_isolated(scene, root_id, side, isolate) {
                entries
            } else {
                Vec::new()
            }
        };
        (keep("under", under), keep("over", over))
    });
    if !shadowed && !filtered && blend.is_none() && appearance.is_none() && fade.is_none() {
        return None;
    }
    // Descendants follow their root contiguously.
    let mut dl_end = index + 1;
    while dl_end < dl.items.len() && sh::is_within(scene, &dl.items[dl_end].node_id, root_id) {
        dl_end += 1;
    }
    let clip = crate::effects::effective_page_clip(scene, root_id, page_clip).unwrap_or(page_clip);
    let clip_rrect = crate::effects::clip_rounded_rect(scene, root_id);
    let node_ids: Vec<String> = if hashes.is_some() {
        dl.items[index..dl_end]
            .iter()
            .filter(|i| i.is_primary_node_item)
            .map(|i| i.node_id.clone())
            .collect()
    } else {
        Vec::new()
    };
    let key = |group: &crate::seams::LayerBlurGroup| {
        crate::residency::capture_key(
            scene,
            hashes,
            node_ids.iter().map(String::as_str),
            group,
            page_clip,
            opts,
        )
    };
    // The drop composite belongs to the subtree's items: a layer filter on the
    // root blurs its shadow with its content.
    let items_start = items.len();
    let mut inner = Vec::new();
    if let Some(lb_ctx) = lb_ctx.filter(|_| shadowed) {
        let capture = subtree_silhouette_items(
            scene,
            dl,
            index,
            dl_end,
            opts,
            isolate,
            transparent,
            caches.as_mut().map(|(c, h)| (&mut **c, *h)),
        );
        if let Some(bounds) = sh::items_canvas_bounds(&capture) {
            // Faded as one unit, the root's opacity is its split's; so is an open fade split's above it.
            let own = if fade.is_some() { 1.0 } else { root.f64("opacity").unwrap_or(1.0) };
            let opacity = own * scene.ancestor_opacity_product_except(root_id, faded);
            // Drop shadows, then glows, behind every descendant; the inner
            // shadows and inner glows wait for the range's end.
            for kind in sh::CaptureKind::for_node(root) {
                let Some(mut group) = sh::plan_shadow_group(
                    scene,
                    root,
                    lb_ctx,
                    kind,
                    capture.clone(),
                    bounds,
                    opacity,
                ) else {
                    continue;
                };
                (group.cache_key, group.identity_key) = key(&group);
                if kind.is_inner() {
                    inner.push((kind.suffix(), group));
                    continue;
                }
                items.push(DrawItem::LayerBlurComposite {
                    node_id: format!("{root_id}:{}", kind.suffix()),
                    z: z as f64,
                    group: std::sync::Arc::new(group),
                    page_clip_bounds: clip,
                    clip_rrect,
                    // Set once the capture opens (`SubtreeCapture::mask`).
                    mask: None,
                });
            }
        }
    }
    Some(SubtreeCapture {
        node: root.clone(),
        dl_end,
        items_start,
        z_start: z,
        inner,
        clip,
        clip_rrect,
        filtered,
        node_ids,
        page_clip,
        blend,
        mask: None,
        appearance_under: appearance.as_ref().map(|(under, _)| under.clone()),
        appearance_over: appearance.map(|(_, over)| over),
        fixed_blend,
        fade,
    })
}

/// Close a subtree capture once the loop has drawn its range (`z` = the
/// render-z after its last item): the inner shadow composites over the last
/// descendant (at its z + INNER_SHADOW_Z_OFFSET), then a layer filter diverts
/// every item the subtree drew (its shadows included) into one capture,
/// composited at the subtree's first z. Filtered descendants remain composites
/// inside the capture, so their filters run before the parent's filter. Items
/// that cannot be isolated (a backdrop split or an unrelated mask) produce a
/// diagnostic rather than silently dropping the filter.
/// Last, a blended root's range — the filtered composite, or every item it
/// drew — goes between a `BlendBegin` / `BlendEnd` pair: it draws alone into
/// a cleared region and composites over the backdrop with the root's mode.
#[allow(clippy::too_many_arguments)]
fn finish_subtree_capture(
    scene: &Scene,
    lb_ctx: Option<&crate::backdrop::PlanCtx>,
    blend_ctx: Option<&crate::backdrop::PlanCtx>,
    mut capture: SubtreeCapture,
    hashes: Option<&crate::residency::HashMap<String, crate::residency::NodeHashes>>,
    masks: &[crate::seams::MaskPlan],
    opts: &RenderOptions,
    items: &mut Vec<DrawItem>,
    z: u64,
    warnings: &mut PlanWarnings,
) {
    let root_id = capture.node.id().to_string();
    mask_subtree_shadows(scene, lb_ctx, &mut capture, masks, items, warnings);
    for (suffix, group) in std::mem::take(&mut capture.inner) {
        if z > capture.z_start {
            items.push(DrawItem::LayerBlurComposite {
                node_id: format!("{root_id}:{suffix}"),
                z: (z - 1) as f64 + crate::seams::INNER_SHADOW_Z_OFFSET,
                group: std::sync::Arc::new(group),
                page_clip_bounds: capture.clip,
                clip_rrect: capture.clip_rrect,
                mask: capture.mask.clone(),
            });
        }
    }
    let blend = capture.blend.zip(blend_ctx);
    if blend.is_some() || capture.fixed_blend.is_some() {
        // The root's own paint (a frame's fill) is one more layer of the
        // flattened subtree.
        for it in &mut items[capture.items_start..] {
            if let DrawItem::Body {
                node_id,
                blend_mode,
                ..
            } = it
            {
                if *node_id == root_id {
                    *blend_mode = BlendMode::Normal;
                }
            }
        }
    }
    // A blended root filters only when its blend can wrap the composite.
    let blend_ok = blend.is_some()
        || capture.fixed_blend.is_some()
        || crate::backdrop::blend::composite_blend_split(&capture.node).is_none();
    if let Some(lb_ctx) = lb_ctx.filter(|_| blend_ok) {
        divert_subtree_filter(scene, lb_ctx, &capture, hashes, masks, opts, items, warnings);
    }
    if items.len() <= capture.items_start {
        return;
    }
    let fade = capture.fade.zip(blend_ctx);
    let splits = match (blend, fade) {
        // Faded as one unit (and blended with its mode, when it has one): the isolated subtree is scaled by the
        // opacity, then composited.
        (b, Some((opacity, ctx))) => {
            let bounds = crate::layer_blur::shadow::all_items_canvas_bounds(&items[capture.items_start..]);
            let mode = b.map_or(BlendMode::Normal.exact_split_id(), |(mode, _)| mode);
            crate::backdrop::blend::compute_region_fade_splits(&capture.node, ctx, bounds, mode, opacity)
        }
        (Some((mode, ctx)), None) => {
            let bounds = crate::layer_blur::shadow::all_items_canvas_bounds(&items[capture.items_start..]);
            crate::backdrop::blend::compute_region_blend_splits(&capture.node, ctx, bounds, mode)
        }
        (None, None) => None,
    };
    if let Some((begin, end)) = splits {
        items.push(DrawItem::BackdropSplit(Box::new(end)));
        items.insert(capture.items_start, DrawItem::BackdropSplit(Box::new(begin)));
    }
}

/// The masks owned inside a subtree (`inner`: the root's own clip, clip groups and clipping frames below it), with
/// the ancestors they compose that are inside it too, as a capture's own mask table in page-plan order (an ancestor
/// keeps a smaller index). What clips the root from above is the composite's business, so a chain stops at the
/// root. Returns the page-plan ids the table holds (an item's id maps to its position) and the table.
fn subtree_mask_table(
    masks: &[crate::seams::MaskPlan],
    inner: Vec<u64>,
    inside: impl Fn(&str) -> bool,
) -> (Vec<u64>, Vec<crate::seams::MaskPlan>) {
    let owner = |id: u64| masks.get(id as usize).map(|m| m.owner_node_id.as_str());
    let mut table: Vec<u64> = inner;
    let mut k = 0;
    while k < table.len() {
        if let Some(a) = masks.get(table[k] as usize).and_then(|m| m.ancestor) {
            let a = a as u64;
            if !table.contains(&a) && owner(a).is_some_and(&inside) {
                table.push(a);
            }
        }
        k += 1;
    }
    table.sort_unstable();
    let index_of = |id: u64| table.iter().position(|t| *t == id);
    let plans = table
        .iter()
        .map(|id| {
            let mut plan = masks[*id as usize].clone();
            plan.ancestor = plan.ancestor.and_then(|a| index_of(a as u64));
            plan
        })
        .collect();
    (table, plans)
}

/// A subtree's shadows (and glows) are captures of its descendants' silhouettes, planned as it opens, before its
/// items are masked. Once it has drawn, each silhouette takes the clip mask its body drew with when that mask is
/// owned inside the subtree (the root's own clip, or a clip group or clipping frame below it): what a mask cuts
/// away casts no shadow (an SVG `mask` hole, a clip group's cut-off part). The captures render with their own
/// table of those masks, as a filtered subtree's does ([`divert_subtree_filter`]); a mask above the root clips the
/// composite instead.
fn mask_subtree_shadows(
    scene: &Scene,
    lb_ctx: Option<&crate::backdrop::PlanCtx>,
    capture: &mut SubtreeCapture,
    masks: &[crate::seams::MaskPlan],
    items: &mut [DrawItem],
    warnings: &mut PlanWarnings,
) {
    let root_id = capture.node.id().to_string();
    let inside = |o: &str| o == root_id || scene.transform_parent_chain(o).iter().any(|p| p.as_str() == root_id);
    let start = capture.items_start.min(items.len());
    // Captured descendants use local mask IDs. Owners uniquely identify the
    // canonical page plans, whose ancestor chains have not been truncated at
    // a child's filter boundary. Recover those IDs before building our table.
    let canonical: std::collections::HashMap<&str, u64> = masks.iter().enumerate()
        .map(|(id, plan)| (plan.owner_node_id.as_str(), id as u64)).collect();
    // Only the root's own capture shadows are masked here: without one there is nothing to collect for.
    let prefix = format!("{root_id}:");
    let has_shadow_group = !capture.inner.is_empty()
        || items[start..].iter().any(|item| matches!(item,
            DrawItem::LayerBlurComposite { node_id, group, .. } if node_id.starts_with(&prefix) && group.filters.shadow.is_some()));
    if !has_shadow_group {
        return;
    }
    let mut by_node: std::collections::HashMap<String, crate::seams::MaskRef> = Default::default();
    let mut inner: Vec<u64> = Vec::new();
    fn collect(
        items: &[DrawItem],
        masks: &[crate::seams::MaskPlan],
        canonical: &std::collections::HashMap<&str, u64>,
        inside: &impl Fn(&str) -> bool,
        inherited: Option<&crate::seams::MaskRef>,
        by_node: &mut std::collections::HashMap<String, crate::seams::MaskRef>,
        inner: &mut Vec<u64>,
    ) {
        let resolve = |mask: &crate::seams::MaskRef| {
            let plan = masks.get(mask.mask_id as usize)?;
            if !inside(&plan.owner_node_id) { return None; }
            let &id = canonical.get(plan.owner_node_id.as_str())?;
            Some(crate::seams::MaskRef { mask_id: id, ..mask.clone() })
        };
        for item in items {
            match item {
                DrawItem::Body { node_id, mask, .. } => {
                    // A deeper mask's canonical ancestor chain includes the
                    // enclosing composite coverage. Otherwise that composite
                    // is the only remaining record of the body's clipping.
                    let own = mask.as_ref().and_then(&resolve);
                    let Some(mask) = own.as_ref().or(inherited) else { continue };
                    if !inner.contains(&mask.mask_id) { inner.push(mask.mask_id); }
                    by_node.entry(node_id.clone()).or_insert_with(|| mask.clone());
                }
                DrawItem::LayerBlurComposite { group, mask, .. } => {
                    // Resolve the composite in the enclosing table BEFORE
                    // entering its contents' local mask namespace.
                    let own = mask.as_ref().and_then(&resolve);
                    collect(&group.items, &group.masks, canonical, inside,
                        own.as_ref().or(inherited), by_node, inner);
                }
                DrawItem::Text(batch) => {
                    // Glyph coverage is intrinsic, but an enclosing composite
                    // may additionally clip it with an authored mask.
                    if let Some(mask) = inherited {
                        if !inner.contains(&mask.mask_id) { inner.push(mask.mask_id); }
                        by_node.entry(batch.node_id.clone()).or_insert_with(|| mask.clone());
                    }
                }
                _ => {}
            }
        }
    }
    collect(&items[start..], masks, &canonical, &inside, None, &mut by_node, &mut inner);
    if by_node.is_empty() {
        return;
    }
    let (table, plans) = subtree_mask_table(masks, inner, inside);
    let mut mask_group = |group: &mut crate::seams::LayerBlurGroup| {
        let mut masked = false;
        for it in &mut group.items {
            let node_id = match it {
                DrawItem::Body { node_id, .. } => node_id.as_str(),
                DrawItem::Text(batch) => batch.node_id.as_str(),
                _ => continue,
            };
            // Cached silhouettes have their own tessellation ID; coverage belongs
            // to the original body, just as it does on the uncached path.
            let Some(m) = by_node.get(silhouette_owner_id(node_id)) else { continue };
            let Some(i) = table.iter().position(|t| *t == m.mask_id) else { continue };
            let mask = Some(crate::seams::MaskRef { mask_id: i as u64, ..m.clone() });
            match it {
                DrawItem::Body { mask: slot, phase, .. } => {
                    *slot = mask;
                    *phase = Phase::Transparent;
                }
                DrawItem::Text(batch) => {
                    // Text has no mask binding. Reuse the same identity capture
                    // as visible masked text, then clip its composite before the
                    // ancestor's shadow filter sees its coverage.
                    let Some((ctx, node)) = lb_ctx.zip(scene.effective_node(&batch.node_id)) else { continue };
                    let Some(bounds) = crate::layer_blur::shadow::all_items_canvas_bounds(std::slice::from_ref(it)) else { continue };
                    let Some(params) = crate::layer_blur::plan::compute_mask_capture_params(
                        scene, node, ctx, bounds, &mut warnings.skipped,
                    ) else { continue };
                    let DrawItem::Text(batch) = it else { unreachable!() };
                    *it = DrawItem::LayerBlurComposite {
                        node_id: format!("{}:shadow-mask", batch.node_id),
                        z: batch.z,
                        page_clip_bounds: batch.page_clip_bounds,
                        clip_rrect: None,
                        mask,
                        group: std::sync::Arc::new(crate::seams::LayerBlurGroup {
                            items: vec![DrawItem::Text(batch.clone())],
                            region_canvas: params.region_canvas,
                        extent_canvas: params.extent_canvas,
                            reach_canvas: params.reach_canvas,
                            width_px: params.width_px,
                            height_px: params.height_px,
                            filters: params.filters,
                            cache_key: None,
                            identity_key: None,
                            coverage: Vec::new(),
                            masks: Vec::new(),
                        }),
                    };
                }
                _ => unreachable!(),
            }
            masked = true;
        }
        if masked {
            group.masks = plans.clone();
        }
    };
    for item in &mut items[start..] {
        if let DrawItem::LayerBlurComposite { node_id, group, .. } = item {
            if node_id.starts_with(&prefix) && group.filters.shadow.is_some() && group.masks.is_empty() {
                mask_group(std::sync::Arc::make_mut(group));
            }
        }
    }
    for (_, group) in &mut capture.inner {
        if group.masks.is_empty() {
            mask_group(group);
        }
    }
}

/// Whether a layer's items include a capture's composite (a descendant's layer filter, or a capture shadow).
/// Such a layer's filter shadow is not cast from a COPY of its items (`LayerFilters::shadows_alone`): the copy
/// would hold the capture a second time, and a chain of n such layers would hold the innermost 2^(n-1) times
/// for everything that walks a plan's captures (content keys, memory accounting, mask lookups). Its items move
/// into the capture instead, as for any other filter, so every capture is held once.
fn holds_capture(items: &[DrawItem]) -> bool {
    items.iter().any(|item| matches!(item, DrawItem::LayerBlurComposite { .. }))
}

/// Whether a layer capture cannot hold these items: the one admission rule of both capture forms (a leaf's own
/// items, a filtered subtree's). A capture's target starts transparent, holds only the capture's items, and
/// has its own pixels; an operation runs in it when it is exact there:
///
/// - a fade pair is (`is_fade_split`): it scales what is drawn between its halves, and its region is re-aimed
///   at the capture (`retarget_isolation_splits`);
/// - a body's fixed-function blend state is when it is exact over a transparent backdrop
///   (`BlendMode::split_id_over`: normal and screen; multiply's lacks the term for an empty backdrop);
/// - a nested capture's composite is by the same test of its blend, a filtered node's or a shadow or glow
///   group's (`CaptureBlend::Multiply` is not), and
///   what it holds is judged as these items are, so the answer does not depend on where a nested filter's
///   boundary falls;
/// - a blend split is not: its region and formula were chosen for the page target (until blend pairs are
///   planned for the target they land in);
/// - a backdrop effect and a page fill are not: they read or are the page, which a capture does not hold.
///
/// `own_blend_lifted`: the layer's own blend mode is its composite's (a split around it, or its blend state),
/// and its bodies draw source-over inside the capture, so their modes are not held.
fn capture_cannot_hold(items: &[DrawItem], own_blend_lifted: bool) -> bool {
    items.iter().any(|item| match item {
        DrawItem::Body { blend_mode, .. } => !own_blend_lifted && blend_mode.split_id_over(true).is_some(),
        DrawItem::BackdropSplit(split) => !crate::backdrop::blend::is_fade_split(split),
        DrawItem::PageFill { .. } => true,
        DrawItem::LayerBlurComposite { group, .. } => {
            // A filtered node's composite blends with `composite_blend`; a shadow or glow group's with its own.
            let multiply = crate::seams::CaptureBlend::Multiply;
            group.filters.composite_blend == multiply
                || group.filters.shadow.as_ref().is_some_and(|shadow| shadow.blend == multiply)
                || capture_cannot_hold(&group.items, false)
        }
        _ => false,
    })
}

/// What a declined capture says ([`capture_cannot_hold`]).
fn capture_refusal(id: &str) -> String {
    format!("{id}: subtree filter cannot isolate backdrop or blend operations; remove the descendant backdrop or blend")
}

/// The layer-filter half of [`finish_subtree_capture`].
///
/// The capture renders with its own mask table, so every mask an item inside
/// the subtree carries must be one of: the root's own clip mask when a
/// clip-mask group is above it (the composite takes it), a clipping frame's
/// ABOVE the root (the composite is clipped by its box and rounded corners:
/// the layer is filtered, then clipped, as in Figma), or a mask owned inside
/// the subtree — a clipping root frame's own, or a clip group or clipping
/// frame below the root — whose content is clipped before the filter (the
/// capture's table holds those, with their ancestors inside the subtree). Any
/// other mask refuses the filter with a diagnostic.
fn divert_subtree_filter(
    scene: &Scene,
    lb_ctx: &crate::backdrop::PlanCtx,
    capture: &SubtreeCapture,
    hashes: Option<&crate::residency::HashMap<String, crate::residency::NodeHashes>>,
    masks: &[crate::seams::MaskPlan],
    opts: &RenderOptions,
    items: &mut Vec<DrawItem>,
    warnings: &mut PlanWarnings,
) {
    let root_id = capture.node.id().to_string();
    if !capture.filtered || items.len() <= capture.items_start {
        return;
    }
    let own = &items[capture.items_start..];
    let Some(bounds) = crate::layer_blur::shadow::all_items_canvas_bounds(own) else {
        return;
    };
    let Some(params) = crate::layer_blur::plan::compute_subtree_layer_blur_params(
        scene,
        &capture.node,
        lb_ctx,
        bounds,
        &mut warnings.skipped,
    ) else {
        return;
    };
    let root_mask = capture.mask.as_ref().map(|m| m.mask_id);
    let owner = |id: u64| masks.get(id as usize).map(|m| m.owner_node_id.as_str());
    // Clipping frames above the root (only without a clip-mask group above
    // it: then `root_mask` is the one mask its items carry).
    let above: std::collections::HashSet<String> = if root_mask.is_none() {
        scene.transform_parent_chain(&root_id).into_iter().collect()
    } else {
        Default::default()
    };
    // A mask owned inside the subtree: the root's own clip, or a clip group / clipping frame below it.
    let inside = |o: &str| o == root_id || scene.transform_parent_chain(o).iter().any(|p| p.as_str() == root_id);
    let mut strip: Vec<u64> = Vec::new();
    let mut strips_above = false;
    let mut inner: Vec<u64> = Vec::new();
    // What a capture cannot hold declines the filter (`capture_cannot_hold`). A descendant's fade (a group or a
    // translucent fill-and-stroke composited as one unit) composites inside the capture as it does on the page:
    // its pair is re-aimed at the capture below (`retarget_isolation_splits`).
    if capture_cannot_hold(own, false) {
        warnings.skipped.push(capture_refusal(&root_id));
        return;
    }
    for item in own {
        let Some(id) = crate::camera_slot::item_mask_id(item) else {
            continue;
        };
        if Some(id) == root_mask || inner.contains(&id) || strip.contains(&id) {
            continue;
        }
        match owner(id) {
            Some(o) if inside(o) => inner.push(id),
            Some(o) if above.contains(o) => {
                strips_above = true;
                strip.push(id);
            }
            _ => {
                warnings.skipped.push(format!("{root_id}: subtree filter cannot capture an unrelated mask; move the mask inside the filtered subtree"));
                return;
            }
        }
    }
    let region = crate::backdrop::plan::Rect {
        x: params.region_canvas[0],
        y: params.region_canvas[1],
        w: params.region_canvas[2],
        h: params.region_canvas[3],
    };
    if strips_above
        && !crate::layer_blur::plan::clipping_containers_clip_the_composite(scene, &root_id, &region)
    {
        warnings.skipped.push(format!("{root_id}: subtree filter crosses an ancestor clip; remove the ancestor clip"));
        return;
    }
    // Filter shadows alone (`LayerFilters::shadows_alone`): the capture is a copy of the subtree's items, which
    // stay and draw directly over the shadows' composite. A blended root's composite carries its blend, so its
    // picture goes through the capture as with any filter.
    // So does a picture that holds a capture already (`holds_capture`).
    let shadows_alone = params.filters.shadows_alone
        && capture.blend.is_none()
        && capture.fixed_blend.is_none()
        && !holds_capture(&items[capture.items_start..]);
    let mut diverted: Vec<DrawItem> = if shadows_alone {
        items[capture.items_start..].to_vec()
    } else {
        items.drain(capture.items_start..).collect()
    };
    unmask_captured(&mut diverted, capture.mask.as_ref());
    for id in strip {
        let mask = crate::seams::MaskRef {
            mask_id: id,
            origin: [0.0; 2],
            inv_size: [0.0; 2],
        };
        unmask_captured(&mut diverted, Some(&mask));
    }
    // The masks inside the subtree form the capture's own mask table.
    let (table, group_masks) = subtree_mask_table(masks, inner, inside);
    let index_of = |id: u64| table.iter().position(|t| *t == id);
    for it in &mut diverted {
        if let Some(i) = crate::camera_slot::item_mask_id(it).and_then(index_of) {
            crate::camera_slot::set_item_mask_id(it, i as u64);
        }
    }
    crate::backdrop::blend::retarget_isolation_splits(&mut diverted, params.region_canvas, params.width_px, params.height_px);
    let mut group = crate::seams::LayerBlurGroup {
        items: diverted,
        region_canvas: params.region_canvas,
                        extent_canvas: params.extent_canvas,
        reach_canvas: params.reach_canvas,
        width_px: params.width_px,
        height_px: params.height_px,
        filters: crate::seams::LayerFilters {
            composite_blend: capture.fixed_blend.unwrap_or(params.filters.composite_blend),
            shadows_alone,
            ..params.filters
        },
        cache_key: None,
        identity_key: None,
        coverage: Vec::new(),
        masks: group_masks,
    };
    (group.cache_key, group.identity_key) = crate::residency::capture_key(
        scene,
        hashes,
        capture.node_ids.iter().map(String::as_str),
        &group,
        capture.page_clip,
        opts,
    );
    // A capture holds its root's picture, not the part of it on the page: a blur spreads what lies past the page
    // edge back onto the page, and a source off the page casts a shadow that lands on it. So what a capture
    // draws is cut by the clips inside its root and by the capture's own texture, never by the page or by a
    // clipping frame above the root: those cut the filtered composite, where it is drawn. The rule is the same
    // for every capture, so it holds through nested ones: a descendant's capture was built by it, and here its
    // composite takes its clip inside this root. Nothing here depends on the filters above, so a capture's
    // content key needs nothing more. After the content key: re-clipping a text batch makes a new batch, and
    // an unkeyed batch is named by its allocation.
    widen_captured_clips(scene, &mut group.items, &root_id, INFINITE_CLIP);
    let composite = DrawItem::LayerBlurComposite {
        node_id: root_id,
        z: capture.z_start as f64,
        group: std::sync::Arc::new(group),
        page_clip_bounds: capture.clip,
        clip_rrect: capture.clip_rrect,
        mask: capture.mask.clone(),
    };
    if shadows_alone {
        // Behind every item the subtree drew, at its first z, as a subtree's capture shadows.
        items.insert(capture.items_start, composite);
    } else {
        items.push(composite);
    }
}

/// Clip a subtree capture's items as they clip inside the root (the clipping
/// frames between each and the root, the root's own included), not by the
/// clipping frames above it: those clip the filtered composite instead
/// (`divert_subtree_filter`). The root's own items take the plain page clip.
/// What an item clips by itself stays: a text batch cut to its own box (`clip_to_node_box`; a source clip, so
/// not a filtered text's composite, whose filter reaches past the box), and a nested capture's composite cut
/// by the rounded corners of a clipping frame inside the root.
fn widen_captured_clips(scene: &Scene, items: &mut [DrawItem], root_id: &str, page_clip: [f32; 4]) {
    // Sub-items carry `<node id>:<suffix>` ids.
    let owner = |node_id: &str| -> Option<String> {
        let id = if scene.effective_node(node_id).is_some() {
            node_id
        } else {
            node_id.split(':').next().unwrap_or(node_id)
        };
        scene.effective_node(id).map(|_| id.to_string())
    };
    let clip_of = |node_id: &str| -> Option<[f32; 4]> {
        let id = owner(node_id)?;
        if id == root_id {
            return Some(page_clip);
        }
        crate::effects::effective_page_clip_until(scene, &id, page_clip, Some(root_id))
    };
    let own_box = |node_id: &str, clip: [f32; 4]| -> Option<[f32; 4]> {
        let id = owner(node_id)?;
        crate::effects::clip_to_node_box(scene, scene.effective_node(&id)?, clip)
    };
    let rrect_of = |node_id: &str| -> Option<crate::seams::ClipRRect> {
        let id = owner(node_id)?;
        (id != root_id)
            .then(|| crate::effects::clip_rounded_rect_until(scene, &id, Some(root_id)))
            .flatten()
    };
    for item in items {
        match item {
            DrawItem::Body {
                node_id, instance, ..
            } => {
                if let Some(c) = clip_of(node_id) {
                    instance.page_clip_bounds = c;
                }
            }
            DrawItem::SolidShapeSdf {
                node_id, instance, ..
            } => {
                if let Some(c) = clip_of(node_id) {
                    instance.page_clip_bounds = c;
                }
            }
            DrawItem::GradientShapeSdf {
                node_id, instance, ..
            } => {
                if let Some(c) = clip_of(node_id) {
                    instance.page_clip_bounds = c;
                }
            }
            DrawItem::Text(batch) => {
                if let Some(c) = clip_of(&batch.node_id).and_then(|c| own_box(&batch.node_id, c)) {
                    if batch.page_clip_bounds != c {
                        std::sync::Arc::make_mut(batch).page_clip_bounds = c;
                    }
                }
            }
            DrawItem::LayerBlurComposite {
                node_id,
                page_clip_bounds,
                clip_rrect,
                ..
            } => {
                if let Some(c) = clip_of(node_id) {
                    *page_clip_bounds = c;
                    *clip_rrect = rrect_of(node_id);
                }
            }
            _ => {}
        }
    }
}

/// The silhouette draw items of a subtree (display-list items
/// `index..dl_end`, rooted at `index`) for its shadow capture: every drawn
/// descendant's silhouette (`layer_blur::shadow::silhouette_node`) at the
/// opacity it paints with inside the root, clipped by the clipping frames
/// between it and the root (the root's own clip included), in paint order.
/// The root's own silhouette (an unfilled frame's stroke) comes first.
fn subtree_silhouette_items(
    scene: &Scene,
    dl: &moda_scene::DisplayList,
    index: usize,
    dl_end: usize,
    opts: &RenderOptions,
    isolate: Option<&std::collections::HashSet<&str>>,
    transparent: bool,
    mut caches: Option<CacheCtx<'_>>,
) -> Vec<DrawItem> {
    use crate::layer_blur::shadow as sh;
    let root_id = dl.items[index].node_id.as_str();
    let mut out = Vec::new();
    // A group's own appearance (E8.2) is part of what it shadows.
    if let Some(record) = scene
        .group_appearance(root_id)
        .filter(|_| paints_group_appearance(scene, &dl.items[index], opts, isolate))
    {
        let mut m = record.fields().clone();
        m.remove(moda_scene::appearance::APPEARANCE_ORDER_KEY);
        let sil = sh::silhouette_node(&record.with_fields(m));
        out.extend(build_shadow_capture_items(
            scene,
            &sil,
            0.0,
            INFINITE_CLIP,
            opts,
            caches.as_mut().map(|(c, h)| (&mut **c, *h)),
        ));
    }
    for (k, item) in dl.items[index..dl_end].iter().enumerate() {
        if !item.is_primary_node_item
            || item_skipped_without_z(scene, item, opts, isolate, transparent)
        {
            continue;
        }
        let Some(node) = scene.effective_node(&item.node_id) else {
            continue;
        };
        if !sh::is_silhouette_type(node.node_type()) {
            continue;
        }
        let (opacity, clip) = if k == 0 {
            (1.0, INFINITE_CLIP)
        } else {
            let Some(opacity) = sh::opacity_within(scene, &item.node_id, root_id) else {
                continue;
            };
            let Some(clip) = crate::effects::effective_page_clip_until(
                scene,
                &item.node_id,
                INFINITE_CLIP,
                Some(root_id),
            ) else {
                continue; // clipped away entirely
            };
            (opacity, clip)
        };
        if !(opacity > 0.0) {
            continue;
        }
        let mut sil = sh::silhouette_node(node);
        if opacity < 1.0 {
            sil.set("opacity", Value::from(opacity));
        }
        out.extend(build_shadow_capture_items(
            scene,
            &sil,
            k as f64,
            clip,
            opts,
            caches.as_mut().map(|(c, h)| (&mut **c, *h)),
        ));
    }
    out
}

/// What a paint stack needs to draw its layers' own drawing effects (Illustrator's Appearance, E8.2): the plan's
/// capture context and the owner's clips, as the node's own captures take them.
pub(crate) struct EntryFx<'a, 'c> {
    /// The plan's layer-capture context (`None`: the page plans no captures, so entry drawing effects can't draw).
    pub lb_ctx: Option<&'a crate::backdrop::PlanCtx<'c>>,
    pub opts: &'a RenderOptions,
    pub hashes: Option<&'a crate::residency::HashMap<String, crate::residency::NodeHashes>>,
    pub page_clip: [f32; 4],
    /// The owner's effective clip, rounded clip and capture mask (`capture_composite_mask`).
    pub clip: [f32; 4],
    pub clip_rrect: Option<crate::seams::ClipRRect>,
    pub mask: Option<crate::seams::MaskRef>,
}

impl<'a, 'c> EntryFx<'a, 'c> {
    /// The context for the node `id` (a leaf, or a group painting its appearance).
    #[allow(clippy::too_many_arguments)]
    fn for_node(
        scene: &Scene,
        id: &str,
        lb_ctx: Option<&'a crate::backdrop::PlanCtx<'c>>,
        opts: &'a RenderOptions,
        hashes: Option<&'a crate::residency::HashMap<String, crate::residency::NodeHashes>>,
        page_clip: [f32; 4],
        plan_masks: &mut crate::effects::PlanMasks,
    ) -> EntryFx<'a, 'c> {
        EntryFx {
            lb_ctx,
            opts,
            hashes,
            page_clip,
            clip: crate::effects::effective_page_clip(scene, id, page_clip).unwrap_or(page_clip),
            clip_rrect: crate::effects::clip_rounded_rect(scene, id),
            mask: capture_composite_mask(scene, plan_masks, id),
        }
    }
}

/// A layer clone reshaped by its entry geometry effects (E8.2): the node's outline (already through the node's
/// own geometry effects) then the entry's, in list order (`live_effects::resolve_node_in`). A node type the
/// effects can't reshape draws unchanged.
fn entry_geometry_node(
    scene: &Scene,
    layer: &moda_scene::types::RawNode,
    geometry: &[Value],
) -> moda_scene::types::RawNode {
    let mut m = layer.fields().clone();
    let mut effects: Vec<Value> = moda_scene::effect::node_effects_json(layer)
        .iter()
        .filter(|e| {
            !e.get("type")
                .and_then(Value::as_str)
                .is_some_and(|t| moda_scene::effect::GEOMETRY_EFFECT_TYPES.contains(&t))
        })
        .cloned()
        .collect();
    effects.extend(geometry.iter().cloned());
    m.insert("effects".into(), Value::Array(effects));
    let with = layer.with_fields(m);
    let mut out = moda_scene::live_effects::resolve_node_in(&with, &scene.doc())
        .unwrap_or_else(|| layer.clone());
    // The clone's tessellation is cached under its content hash (it has no scene hash): fold in the zoom band its
    // curves flatten at, so a build at another band never serves this one's vertices.
    out.set("_appearanceBand", Value::from(tess::outline::current_band()));
    out
}

/// One layer's drawing effects (E8.2): its drop shadows and outer glows (captures of the layer's own
/// silhouette) before its `items`, its inner shadows and inner glows after them, then its layer filters (blur,
/// noise, texture, feather) over all of it as one capture composited at the layer's z. The captures are planned
/// on a PROBE: the layer at its owner's id (so the owner's transform and clips apply) carrying only these
/// effects. Without a capture context the layer draws plain. Under an owner that filters itself, the layer's
/// filters are left out (captures nest one shadow level deep) and the plan warns.
#[allow(clippy::too_many_arguments)]
fn entry_drawing_items(
    scene: &Scene,
    fx: Option<&EntryFx<'_, '_>>,
    layer: &moda_scene::types::RawNode,
    drawing: &[Value],
    k: usize,
    lz: f64,
    slice: f64,
    opacity: f64,
    items: Vec<DrawItem>,
    warnings: &mut PlanWarnings,
) -> Vec<DrawItem> {
    use crate::layer_blur::shadow as sh;
    let owner = moda_scene::scene::paint_transform_id(layer).to_string();
    let Some((fx, lb)) = fx.and_then(|f| f.lb_ctx.map(|lb| (f, lb))) else {
        warnings
            .skipped
            .push(format!("{owner} (paint layer {k}): appearance effects need a capture context"));
        return items;
    };
    let layer_id = layer.id().to_string();
    let mut m = layer.fields().clone();
    m.insert("id".into(), Value::from(owner.as_str()));
    m.insert("effects".into(), Value::Array(drawing.to_vec()));
    m.insert("opacity".into(), Value::from(1.0));
    m.remove("blendMode");
    m.insert(
        moda_scene::appearance::ENTRY_PROBE_KEY.into(),
        Value::Bool(true),
    );
    let probe = layer.with_fields(m);
    let mut out = Vec::with_capacity(items.len() + 2);
    let mut after = Vec::new();
    if sh::has_capture_shadow(&probe) {
        let sil = sh::silhouette_node(&probe);
        let capture = build_shadow_capture_items(scene, &sil, lz, INFINITE_CLIP, fx.opts, None);
        if let Some(bounds) = sh::items_canvas_bounds(&capture) {
            for kind in sh::CaptureKind::for_node(&probe) {
                let Some(mut group) =
                    sh::plan_shadow_group(scene, &probe, lb, kind, capture.clone(), bounds, opacity)
                else {
                    continue;
                };
                (group.cache_key, group.identity_key) = crate::residency::capture_key(
                    scene,
                    fx.hashes,
                    [owner.as_str(), layer_id.as_str()],
                    &group,
                    fx.page_clip,
                    fx.opts,
                );
                let inner = kind.is_inner();
                let composite = DrawItem::LayerBlurComposite {
                    node_id: format!("{layer_id}:{}", kind.suffix()),
                    z: if inner { lz + slice * 0.5 } else { lz },
                    group: std::sync::Arc::new(group),
                    page_clip_bounds: fx.clip,
                    clip_rrect: fx.clip_rrect,
                    mask: fx.mask.clone(),
                };
                if inner {
                    after.push(composite);
                } else {
                    out.push(composite);
                }
            }
        }
    }
    out.extend(items);
    out.extend(after);
    if !crate::layer_blur::has_active_layer_filter(&probe) {
        return out;
    }
    if scene
        .effective_node(&owner)
        .is_some_and(crate::layer_blur::has_active_layer_filter)
    {
        warnings.skipped.push(format!(
            "{owner} (paint layer {k}): appearance: a layer's blur, noise, texture or feather is left out under the layer's own"
        ));
        return out;
    }
    let Some(params) = crate::layer_blur::compute_layer_blur_params(scene, &probe, lb, &mut warnings.skipped) else {
        return out;
    };
    let mut diverted = out;
    unmask_captured(&mut diverted, fx.mask.as_ref());
    // The capture holds the layer's paint source-over; the stack's split blends the filtered result.
    for it in &mut diverted {
        if let DrawItem::Body { blend_mode, .. } = it {
            *blend_mode = BlendMode::Normal;
        }
    }
    let mut group = crate::seams::LayerBlurGroup {
        items: diverted,
        region_canvas: params.region_canvas,
                        extent_canvas: params.extent_canvas,
        reach_canvas: params.reach_canvas,
        width_px: params.width_px,
        height_px: params.height_px,
        filters: params.filters,
        cache_key: None,
        identity_key: None,
        coverage: Vec::new(),
        masks: Vec::new(),
    };
    (group.cache_key, group.identity_key) = crate::residency::capture_key(
        scene,
        fx.hashes,
        [owner.as_str(), layer_id.as_str()],
        &group,
        fx.page_clip,
        fx.opts,
    );
    vec![DrawItem::LayerBlurComposite {
        node_id: layer_id,
        z: lz,
        group: std::sync::Arc::new(group),
        page_clip_bounds: fx.clip,
        clip_rrect: fx.clip_rrect,
        mask: fx.mask.clone(),
    }]
}

/// E3.1 P6: a paint stack's draw items (`paint::layers`), bottom to top: layer `k` draws through `build` at
/// `z + k·slice` (its brush artwork squeezed into its slice), with its own opacity and blend mode.
///
/// - When the node's opacity or blend mode must apply to the layers together (`layers::needs_group`), they
///   draw inside a transparency group over the node's region, as Illustrator composites an object's
///   appearance before the object's opacity and blending. With normal blending the group is NON-isolated
///   (`GroupBegin` / `StackEnd` fade): the layers draw onto the backdrop, so a layer's blend mode reaches the
///   art behind the object as in Illustrator, and the result fades toward the backdrop by the node's opacity.
///   With a blend mode it is isolated (`BlendBegin` / `StackEnd`): the layers composite alone, then blend
///   onto the backdrop at the node's opacity.
/// - A layer's blend mode that reads the backdrop draws inside its own pair (inside a group, every mode but
///   normal does: its backdrop is the node's lower layers over a cleared region). Outside a group, multiply and
///   screen keep their exact fixed-function states over an opaque target (`fixed_function_ok`; lines draw
///   every body with normal blending, so they split every mode).
/// - Without a split context (a region render), the layers draw one by one at the node's opacity times their
///   own, with their fixed-function blend: the pre-split approximation.
#[allow(clippy::too_many_arguments)]
fn build_stack_items(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    layers: &[paint::layers::PaintLayer],
    z: f64,
    // The z range the layers share (`PLACED_Z_SPAN` for a node's own slot; a
    // Transform copy's share of it for live-effect copies).
    span: f64,
    backdrop_ctx: Option<&crate::backdrop::PlanCtx>,
    fixed_function_ok: bool,
    warnings: &mut PlanWarnings,
    // E8.2: what a layer's own drawing effects need (its captures); `None` draws them without.
    entry_fx: Option<&EntryFx<'_, '_>>,
    mut build: impl FnMut(
        &moda_scene::types::RawNode,
        f64,
        f64,
        &mut PlanWarnings,
    ) -> Result<Vec<DrawItem>>,
) -> Vec<DrawItem> {
    use crate::backdrop::{compute_group_splits, compute_stack_splits};
    let slice = span / layers.len().max(1) as f64;
    let node_opacity = node.f64("opacity").unwrap_or(1.0);
    let node_blend = BlendMode::from_node_value(node.str("blendMode"));
    let group = paint::layers::needs_group(node_opacity, node_blend, layers)
        .then(|| {
            backdrop_ctx.and_then(|ctx| {
                if node_blend == BlendMode::Normal {
                    compute_group_splits(scene, node, ctx, node_opacity)
                } else {
                    compute_stack_splits(scene, node, ctx, node_blend, node_opacity)
                }
            })
        })
        .flatten();
    let mut out = Vec::new();
    if let Some((begin, _)) = &group {
        out.push(DrawItem::BackdropSplit(Box::new(begin.clone())));
    }
    for (k, layer) in layers.iter().enumerate() {
        let (opacity, blend) = if group.is_some() {
            (layer.opacity, layer.blend)
        } else {
            (
                node_opacity * layer.opacity,
                paint::layers::effective_blend(node_blend, layer),
            )
        };
        // E8.2: the layer's own effects. Geometry reshapes its outline; drawing effects capture its pixels, so a
        // blend mode on a filtered layer must blend the filtered result (a split).
        let (entry_geometry, entry_drawing) =
            moda_scene::appearance::split_entry_effects(&layer.effects);
        let entry_filtered = entry_drawing.iter().any(|e| {
            matches!(
                e.get("type").and_then(Value::as_str),
                Some("layer-blur" | "noise" | "texture" | "feather")
            )
        });
        let split_mode = if group.is_some() || !fixed_function_ok || entry_filtered {
            blend != BlendMode::Normal
        } else {
            backdrop_ctx.is_some_and(|ctx| blend.split_id_over(ctx.transparent_target).is_some())
        };
        let splits = split_mode
            .then(|| {
                backdrop_ctx.and_then(|ctx| compute_stack_splits(scene, node, ctx, blend, 1.0))
            })
            .flatten();
        let mut m = layer.node.fields().clone();
        m.insert("opacity".into(), Value::from(opacity));
        if splits.is_some() || blend == BlendMode::Normal {
            m.remove("blendMode");
        } else {
            m.insert("blendMode".into(), Value::from(blend.node_value()));
        }
        let layer_node = layer.node.with_fields(m);
        let layer_node = if entry_geometry.is_empty() {
            layer_node
        } else {
            entry_geometry_node(scene, &layer_node, &entry_geometry)
        };
        let lz = z + k as f64 * slice;
        // A Transform effect with copies on the layer: each copy in its share of the layer's slice.
        let copies = (!entry_geometry.is_empty())
            .then(|| moda_scene::live_effects::split_copies(&layer_node))
            .flatten();
        let built = match copies {
            Some(parts) => {
                let cspan = slice / parts.len().max(1) as f64;
                let mut all = Vec::new();
                let mut failed = None;
                for (j, part) in parts.iter().enumerate() {
                    let cz = lz + j as f64 * cspan;
                    match tess::placed::with_z_span(cspan, || build(part, cz, cspan, warnings)) {
                        Ok(items) => all.extend(items),
                        Err(e) => {
                            failed = Some(e);
                            break;
                        }
                    }
                }
                match failed {
                    Some(e) => Err(e),
                    None => Ok(all),
                }
            }
            None => tess::placed::with_z_span(slice, || build(&layer_node, lz, slice, warnings)),
        };
        let built = built
            .map(|mut items| {
                // Layers order by their fractional z buckets (painter's order), never
                // by depth: a stroke's vertices sit STROKE_DEPTH_OFFSET (≈ 0.6 z)
                // nearer, so an opaque stroke layer would depth-reject the filled
                // arrowheads and brush artwork of the layers above it.
                for it in &mut items {
                    if let DrawItem::Body { phase, .. }
                    | DrawItem::SolidShapeSdf { phase, .. }
                    | DrawItem::GradientShapeSdf { phase, .. } = it
                    {
                        *phase = Phase::Transparent;
                    }
                }
                items
            })
            .map(|items| {
                if entry_drawing.is_empty() || items.is_empty() {
                    return items;
                }
                entry_drawing_items(
                    scene,
                    entry_fx,
                    &layer_node,
                    &entry_drawing,
                    k,
                    lz,
                    slice,
                    opacity,
                    items,
                    warnings,
                )
            });
        match built {
            Ok(items) if items.is_empty() => {}
            Ok(items) => match &splits {
                Some((begin, end)) => {
                    out.push(DrawItem::BackdropSplit(Box::new(begin.clone())));
                    out.extend(items);
                    out.push(DrawItem::BackdropSplit(Box::new(end.clone())));
                }
                None => out.extend(items),
            },
            Err(e) => warnings
                .skipped
                .push(format!("{} (paint layer {k}): {e}", node.id())),
        }
    }
    if let Some((_, end)) = group {
        out.push(DrawItem::BackdropSplit(Box::new(end)));
    }
    out
}

/// Build the draw items + masks for the freeform canvas-root
/// (off-page) nodes, in canvas/world space. They belong to no page, so they
/// take `INFINITE_CLIP` (never page-clipped — matching WebGPU's unclipped
/// freeform pass) and carry no page fill. `z_start` is the render-z above every
/// page's z so floating nodes win the depth test over pages they overlap, not
/// just paint order. `backdrop_ctx` (camera-projected, set only when a
/// canvas-root node has an active background blur) lets off-page glass/blur
/// nodes split the pass like page backdrop nodes do. Empty when the canvas has
/// no visible floating nodes. Refused when it draws an item at a render-z past
/// what depth orders (`depth::ensure_render_z`).
fn build_canvas_root_items(
    scene: &Scene,
    opts: &RenderOptions,
    z_start: u64,
    backdrop_ctx: Option<&crate::backdrop::PlanCtx>,
    layer_blur_ctx: Option<&crate::backdrop::PlanCtx>,
    mut caches: Option<CacheCtx<'_>>,
    cull: Option<CullCtx<'_>>,
    // Scale RFC stage C: filled with the canvas root's `CameraSegment` when Some
    // (its contexts' flags are the caller's to set).
    seg_out: Option<&mut Option<CameraSegment>>,
) -> Result<(Vec<DrawItem>, Vec<crate::seams::MaskPlan>, PlanWarnings)> {
    let dl = match caches.as_mut().and_then(|(c, _)| {
        c.frame_canvas_dl(scene)
            .or_else(|| c.dl_get(crate::residency::CANVAS_ROOT_DL_KEY))
    }) {
        Some(dl) => dl,
        None => {
            let dl = std::sync::Arc::new(moda_scene::build_display_list_for_canvas(scene));
            if let Some((c, _)) = caches.as_mut() {
                c.dl_put(crate::residency::CANVAS_ROOT_DL_KEY, dl.clone());
            }
            dl
        }
    };
    let mut items: Vec<DrawItem> = Vec::new();
    let mut warnings = PlanWarnings::default();
    let text_masks = opts.text_masks_available.unwrap_or(true);
    let segment =
        |dl: std::sync::Arc<moda_scene::DisplayList>, units, item_len, mask_len, z_end, tail| {
            CameraSegment {
                kind: SegmentKind::CanvasRoot,
                dl,
                page_clip: INFINITE_CLIP,
                node_transparent: opts.transparent,
                backdrop: None,
                layer_blur: false,
                text_masks,
                item_start: 0,
                prefix: 0,
                item_len,
                mask_start: 0,
                mask_len,
                z_start,
                z_end,
                units,
                head_warnings: PlanWarnings::default(),
                tail_warnings: tail,
            }
        };
    if dl.items.is_empty() {
        if let Some(out) = seg_out {
            *out = Some(segment(
                dl,
                Vec::new(),
                0,
                0,
                z_start,
                PlanWarnings::default(),
            ));
        }
        return Ok((items, Vec::new(), warnings));
    }
    let mut plan_masks =
        crate::effects::PlanMasks::with_text_masks(opts.text_masks_available.unwrap_or(true))
            .with_vector_text_masks(true)
            .in_working_color(opts.working.is_some())
            .at_shader_time(opts.shader_time_s);
    let mut recorder = seg_out.as_ref().map(|_| UnitRecorder::default());
    let end_z = append_node_draw_items(
        scene,
        &dl,
        opts,
        opts.transparent,
        INFINITE_CLIP,
        backdrop_ctx,
        layer_blur_ctx,
        &mut plan_masks,
        caches,
        &mut items,
        &mut warnings,
        z_start,
        None,
        cull,
        recorder.as_mut(),
    );
    depth::ensure_render_z(drawn_z_end(&items))?;
    let mut tail = PlanWarnings::default();
    let masks = plan_masks.into_masks(&mut tail);
    if let (Some(out), Some(recorder)) = (seg_out, recorder) {
        *out = Some(segment(
            dl,
            recorder.units,
            items.len(),
            masks.len(),
            end_z,
            tail.clone(),
        ));
    }
    warnings.absorb(tail);
    Ok((items, masks, warnings))
}

/// Build the body draw item(s) for a rect/container/ellipse/path/star/polygon
/// node. M3b/M3f: routes the paint per `paint::classify_fill` (production
/// else-if order) and returns 1 body (solid / gradient / pattern / shader)
/// or 2 (pattern|shader + stroke-only gradient overlay at the same z).
/// Empty vec = empty geometry.
/// M4e: resolve a shader-image binding for an image-backed fill —
/// (pack metadata, texture payload). `ShaderImageResourceManager.get` /
/// `.getFallback` analog over the prefetched store: a missing source, a
/// store-less run, a never-prefetched key, or a failed/timed-out load all bind
/// the manager's 1×1 transparent fallback texture. Metadata distinguishes
/// pending from terminal failure so the shader can paint the same visible
/// grey/rose placeholders as the underlying pattern-fill lane.
fn resolve_shader_image_binding(
    src: Option<&str>,
    processing: &moda_shaders::runtime::image_processing::Processing,
) -> (
    Option<moda_shaders::runtime::ShaderImageMeta>,
    Option<crate::seams::ShaderImageTexture>,
) {
    use moda_shaders::runtime as rt;
    let fallback = |failed: bool| {
        (
            Some(if failed {
                rt::ShaderImageMeta::failed()
            } else {
                rt::ShaderImageMeta::fallback()
            }),
            Some(crate::seams::ShaderImageTexture {
                key: format!(
                    "fallback:{}",
                    rt::image_processing::processing_key(processing)
                ),
                width: 1,
                height: 1,
                rgba_straight: std::sync::Arc::new(vec![0, 0, 0, 0]),
            }),
        )
    };
    let Some(src) = src else {
        return fallback(false);
    };
    let key = rt::image_resource_key(src, processing);
    match moda_assets::engine_seam::resolve_shader_image(&key) {
        Some(r) if r.ready => (
            Some(rt::ShaderImageMeta::loaded(
                r.width,
                r.height,
                r.content_rect,
            )),
            Some(crate::seams::ShaderImageTexture {
                key,
                width: r.width,
                height: r.height,
                rgba_straight: r.rgba_straight,
            }),
        ),
        Some(_) => fallback(true),
        None => fallback(false),
    }
}

/// M3d: `page_clip` is the EFFECTIVE clip (plain page clip intersected with
/// ancestor rectangle-mask/container AABBs — `computeEffectivePageClip`);
/// `mask` is the nearest ancestor R8 mask context, if any.
#[allow(clippy::too_many_arguments)]
/// Try the analytic SDF fast path for a solid/gradient rect/ellipse
/// body. Returns `Some(items)` when the node is SDF-eligible (the caller returns
/// it and skips tessellation), or `None` to fall through to tessellation. The
/// `disable_shape_sdf` (blur-segmented) disqualifier is the caller's; the
/// active-mask one is handled here.
fn try_shape_sdf_body(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    node_type: &str,
    // f64: a paint-stack layer draws at fractional z inside its node's slot.
    z: f64,
    page_clip: [f32; 4],
    mask: Option<&crate::seams::MaskRef>,
    canvas_affine_override: Option<moda_scene::types::Affine>,
) -> Result<Option<Vec<DrawItem>>> {
    // Active clip-mask context disqualifies (WebGPU `getActiveMaskContextForNode`).
    if mask.is_some() {
        return Ok(None);
    }
    let Some(shape_type) = paint::ShapeType::from_node_type(node_type) else {
        return Ok(None);
    };

    enum Routed {
        Solid(paint::shape_sdf::SolidShapeClass),
        Gradient(paint::shape_sdf::GradientShapeClass),
    }
    // Solid takes precedence (its fillPriority gate already excludes the
    // gradient priorities), then gradient; neither ⇒ tessellate.
    let routed = if let Some(c) = paint::classify_solid_shape_body(node, shape_type) {
        Routed::Solid(c)
    } else if let Some(c) = paint::classify_gradient_shape(node, shape_type) {
        Routed::Gradient(c)
    } else {
        return Ok(None);
    };

    // Canvas-space model matrix: the same chain the tessellated body uses
    // (`createNodeModelMatrix` transform-id redirect for synthetic proxies).
    let affine = match canvas_affine_override {
        Some(a) => a,
        None => {
            let tid = moda_scene::scene::paint_transform_id(node);
            scene
                .node_canvas_affine(tid)
                .ok_or_else(|| anyhow!("no canvas affine for node {}", node.id()))?
        }
    };
    let model_matrix = affine_to_matrix(&affine);
    let depth_val = depth::linear_depth(z) as f32;

    let item = match routed {
        Routed::Solid(c) => {
            // Opaque phase passes `opacity`; transparent passes `bodyAlpha`
            // (`tryEnqueueSolidShapeSdf`, WebGPUShapeSystem.ts:2114-2136).
            // `c.is_opaque` already folds in a translucent stroke.
            let alpha = if c.is_opaque { c.opacity } else { c.body_alpha };
            let instance = paint::build_solid_shape_instance(
                node,
                shape_type,
                c.fill_rgb,
                alpha,
                c.stroke,
                model_matrix,
                depth_val,
                page_clip,
            );
            DrawItem::SolidShapeSdf {
                node_id: node.id().to_string(),
                z,
                phase: if c.is_opaque {
                    Phase::Opaque
                } else {
                    Phase::Transparent
                },
                // covers_shadow tracks whether the opaque body occludes its own
                // drop shadow (so the shadow knockout can be skipped). The FILL
                // must be opaque, AND the stroke must let the body reach the
                // silhouette-expanded shadow band (opaque center/outside, or no
                // stroke) — see `stroke_preserves_shadow_cover`. Inside or
                // translucent center/outside strokes keep the knockout.
                opaque_sdf_covers_shadow: c.body_alpha >= 0.999
                    && paint::stroke_preserves_shadow_cover(c.stroke),
                instance: Box::new(instance),
            }
        }
        Routed::Gradient(c) => {
            // Opaque iff fully-opaque body, no drop shadow, and (no stroke OR an
            // opaque stroke) (`enqueueGradientShape`, WebGPUShapeSystem.ts:2308-
            // 2313, + SDF stroke). `is_transparent` is the FILL's
            // transparency only.
            let stroke_translucent = c.stroke.is_some_and(|s| s.color[3] < 1.0);
            let is_opaque = !c.is_transparent && !c.has_shadow && !stroke_translucent;
            let instance = paint::build_gradient_shape_instance(
                node,
                shape_type,
                c.fill_type,
                c.stroke,
                model_matrix,
                depth_val,
                page_clip,
            );
            DrawItem::GradientShapeSdf {
                node_id: node.id().to_string(),
                z,
                phase: if is_opaque {
                    Phase::Opaque
                } else {
                    Phase::Transparent
                },
                // Fill must be opaque AND any stroke must not translucently
                // extend outside the fill (same reasoning as the solid arm).
                opaque_sdf_covers_shadow: !c.is_transparent
                    && paint::stroke_preserves_shadow_cover(c.stroke),
                instance: Box::new(instance),
            }
        }
    };
    Ok(Some(vec![item]))
}

/// E6.2: a rectangle's or frame's stroke with per-side weights (`Stroke::sides`) as a path node: the ring it paints
/// (`shape_to_path::side_stroke_ring_cmds_smoothed`: the smoothed outline under `cornerSmoothing`), filled nonzero with the stroke's paint, gradients over the node's box.
/// `None` unless the node is a stroke layer (no visible fill: the paint plan draws a per-side stroke as its own layer)
/// whose top stroke has sides. The ring carries the node's id, so the tessellation cache keys it as the node.
fn side_stroke_ring_node(node: &moda_scene::types::RawNode) -> Option<moda_scene::types::RawNode> {
    use moda_scene::paint as sp;
    if !matches!(node.node_type(), "rectangle" | "container") {
        return None;
    }
    let stroke = sp::top_stroke(node)?;
    let sides = stroke.sides?;
    if sp::node_fills_shared(node)
        .iter()
        .any(|f| sp::shown(f.visible, f.opacity))
    {
        return None;
    }
    let (w, h) = (node.f64("width").unwrap_or(0.0), node.f64("height").unwrap_or(0.0));
    let cmds = shape_to_path::side_stroke_ring_cmds_smoothed(
        w,
        h,
        node.json_field("cornerRadius"),
        node.json_field("cornerSmoothing"),
        sides.weights(),
        stroke.align.as_deref(),
    );
    let mut m = moda_scene::value::Map::new();
    for key in [
        "id", "x", "y", "rotation", "scale", "scaleX", "scaleY", "opacity", "blendMode", "visible",
        "_transformSourceId",
    ] {
        if let Some(v) = node.json_field(key) {
            m.insert(key.into(), v.clone());
        }
    }
    m.insert("type".into(), Value::String("path".into()));
    m.insert(
        "points".into(),
        tess::line_to_path::path_cmds_to_json_points(&cmds),
    );
    m.insert("closed".into(), Value::Bool(true));
    m.insert("_paintBox".into(), moda_scene::value::json!([0.0, 0.0, w, h]));
    sp::set_fill_paint(&mut m, stroke.paint.clone());
    Some(moda_scene::types::RawNode::new(m))
}

/// E6.1: an open path's arrowed stroke as synthetic path nodes (`tess::path_arrows`): the stroke on the path
/// cut back under its arrowheads, then the filled arrowheads (the stroke's paint as their fill) and the open
/// carets (stroked like the stroke). `None` when the path is closed, has a visible fill (the paint plan draws
/// an arrowed stroke as its own layer, so a node reaching here with a fill is not a stroke layer), or its top
/// stroke has no arrowhead. Gradients span the uncut path's box on every part (`_paintBox`), as a line's run on
/// into its arrowheads.
struct PathArrowParts {
    body: moda_scene::types::RawNode,
    heads: Vec<moda_scene::types::RawNode>,
}

fn path_arrow_parts(node: &moda_scene::types::RawNode) -> Option<PathArrowParts> {
    use moda_scene::paint as sp;
    use moda_scene::types::RawNode;
    // A vector network draws its own arrow caps (`tess::network`); a stroke's start / end arrows don't apply.
    if node.bool("closed").unwrap_or(false)
        || node.json_field("vectorNetwork").is_some_and(|v| !v.is_null())
    {
        return None;
    }
    let stroke = sp::top_stroke(node)?;
    if !tess::path_arrows::stroke_has_arrows(&stroke)
        || sp::node_fills_shared(node)
            .iter()
            .any(|f| sp::shown(f.visible, f.opacity))
    {
        return None;
    }
    let cmds = tess::path_cmds_for_node(node);
    let arrows = tess::path_arrows::path_arrows(&cmds, false, &stroke)?;
    let bx = paint::gradient_box(tess::path_bounds(&cmds), stroke.width, [0.0; 4]);
    let paint_box = Value::from(bx.iter().map(|v| *v as f64).collect::<Vec<f64>>());
    let plain = {
        let mut s = stroke.clone();
        s.start_arrow = None;
        s.end_arrow = None;
        s.arrow_scale = None;
        s
    };
    let points = |cmds: &[pt::PathCmd]| tess::line_to_path::path_cmds_to_json_points(cmds);

    let mut body = node.fields().clone();
    body.insert("points".into(), points(&arrows.body));
    body.remove("cornerRadius");
    body.insert("_paintBox".into(), paint_box.clone());
    sp::set_strokes(&mut body, vec![plain.clone()]);

    // An arrowhead: the node's transform, none of its effects, trim or text.
    let head_base = |suffix: &str, cmds: &[pt::PathCmd], closed: bool| {
        let mut m = moda_scene::value::Map::new();
        for key in [
            "type",
            "x",
            "y",
            "rotation",
            "scale",
            "scaleX",
            "scaleY",
            "opacity",
            "blendMode",
            "visible",
        ] {
            if let Some(v) = node.json_field(key) {
                m.insert(key.into(), v.clone());
            }
        }
        m.insert(
            "id".into(),
            Value::String(format!("{}::{suffix}", node.id())),
        );
        m.insert(
            "_transformSourceId".into(),
            Value::String(moda_scene::scene::paint_transform_id(node).to_string()),
        );
        m.insert("points".into(), points(cmds));
        m.insert("closed".into(), Value::Bool(closed));
        m.insert("_paintBox".into(), paint_box.clone());
        m
    };
    let mut heads = Vec::new();
    if !arrows.filled.is_empty() {
        let mut m = head_base("arrow-fill", &arrows.filled, true);
        sp::set_fill_paint(&mut m, stroke.paint.clone());
        heads.push(RawNode::new(m));
    }
    if !arrows.open.is_empty() {
        let mut m = head_base("arrow-open", &arrows.open, false);
        let mut caret = plain;
        caret.align = None;
        caret.dash = None;
        caret.dash_offset = None;
        caret.profile = None;
        caret.taper = None;
        caret.wave = None;
        caret.dynamic = None;
        sp::set_strokes(&mut m, vec![caret]);
        heads.push(RawNode::new(m));
    }
    Some(PathArrowParts {
        body: RawNode::new(body),
        heads,
    })
}

/// Read trim-path props off a node, returning `None` when trim is identity (full
/// path) so the caller stays on the byte-identical no-trim path. Shared by the
/// shape (`build_body_item`) and line (`build_line_items`) lanes.
fn trim_params_from_node(node: &moda_scene::types::RawNode) -> Option<tess::TrimParams> {
    tess::outline::trim_params_from_node(node)
}

/// E8.5: a vector pattern layer (`pattern_paint`): the pattern's tile copies that reach into the layer's region,
/// each tile shape drawn at a fractional z inside the layer's slice `[z, z + slice)` and clipped by a coverage
/// mask of the region. `clip_items` draws the layer painted opaque white (the same lane that draws the layer
/// otherwise), which is where the region's triangles come from. A missing or unusable pattern, an empty
/// region or a layer past `MAX_PATTERN_DRAWS` draws nothing, with a note.
#[allow(clippy::too_many_arguments)]
fn build_pattern_layer_items(
    scene: &Scene,
    layer: &moda_scene::types::RawNode,
    node_id: &str,
    z: f64,
    slice: f64,
    page_clip: [f32; 4],
    plan_masks: &mut crate::effects::PlanMasks,
    shader_time_s: f64,
    canvas_affine_override: Option<moda_scene::types::Affine>,
    mut caches: Option<CacheCtx<'_>>,
    warnings: &mut PlanWarnings,
    clip_items: impl FnOnce(&moda_scene::types::RawNode, &mut PlanWarnings) -> Result<Vec<DrawItem>>,
) -> Result<Vec<DrawItem>> {
    use crate::pattern_paint as pp;
    let Some((paint, is_stroke)) = pp::layer_pattern(layer) else {
        return Ok(Vec::new());
    };
    pp::note_layer_planned();
    let note = |warnings: &mut PlanWarnings, msg: String| {
        warnings.skipped.push(format!("{node_id} (pattern): {msg}"));
    };
    let id = paint.pattern_id.as_str();
    let Some(def_value) = scene.doc().pattern(id) else {
        note(warnings, format!("pattern '{id}' is not in the document; the layer draws nothing"));
        return Ok(Vec::new());
    };
    let Some(def) = moda_scene::pattern::parse_pattern(def_value) else {
        note(warnings, format!("pattern '{id}' has no usable tile size; the layer draws nothing"));
        return Ok(Vec::new());
    };
    let affine = match canvas_affine_override {
        Some(a) => a,
        None => scene
            .node_canvas_affine(moda_scene::scene::paint_transform_id(layer))
            .ok_or_else(|| anyhow!("no canvas affine for node {node_id}"))?,
    };
    // The region: the layer's own draw, painted white, as triangles.
    let mut scratch = PlanWarnings::default();
    let region_items = clip_items(&pp::white_layer(layer), &mut scratch)?;
    warnings.dropped.extend(scratch.dropped);
    let positions = pp::clip_positions(&region_items, is_stroke);
    let Some(local) = pp::positions_bounds(&positions) else {
        return Ok(Vec::new());
    };
    let Some(mask) = plan_masks.pattern_clip(
        scene,
        &format!("{}#pattern", layer.id()),
        node_id,
        positions,
        affine,
    ) else {
        return Ok(Vec::new());
    };
    let p = paint.affine();
    let Some(inverse) = pp::invert(p) else {
        note(warnings, format!("pattern '{id}' has a zero scale; the layer draws nothing"));
        return Ok(Vec::new());
    };
    let region = pp::map_bounds(local, inverse);
    let hash = scene.doc().pattern_hash(id).unwrap_or(0);
    let tile = pp::tile_art(&def, hash);
    for s in &tile.skipped {
        note(warnings, format!("pattern '{id}': artwork {s} is not drawn"));
    }
    let per_copy = tile.art.shapes.len();
    if per_copy == 0 {
        return Ok(Vec::new());
    }
    let cap = (crate::pattern_paint::MAX_PATTERN_DRAWS / per_copy).max(1);
    let copies = match def.tile.copies(region, tile.bounds, cap) {
        Ok(c) => c,
        Err(moda_scene::pattern::CopiesError::NoPitch) => {
            note(warnings, format!("pattern '{id}': the spacing leaves no room between tiles; the layer draws nothing"));
            return Ok(Vec::new());
        }
        Err(moda_scene::pattern::CopiesError::TooMany(n)) => {
            note(
                warnings,
                format!(
                    "pattern '{id}' needs at least {n} tile copies here, more than the {cap} one layer draws; \
                     the layer draws nothing (scale the pattern up or use fewer shapes in its tile)"
                ),
            );
            return Ok(Vec::new());
        }
    };
    let total = (copies.len() * per_copy).max(1) as f64;
    let mut out = Vec::with_capacity(copies.len() * per_copy);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let band = tess::outline::current_band();
    for (ci, &(i, j)) in copies.iter().enumerate() {
        let (ox, oy) = def.tile.origin(i, j);
        // Tile space → node-local for this copy, applied to the geometry in f64 (see `pattern_paint`).
        let m = moda_scene::transform::compose(p, moda_scene::transform::translate(ox, oy));
        let mut shapes = tile.copy(m, &format!("{}#tile:{i}:{j}:", layer.id()));
        for (k, shape) in shapes.iter_mut().enumerate() {
            // A tile shape is cached under its content hash (it has no scene hash): fold in the zoom band its curves
            // flatten at, or the first band's vertices (from the zoom the pattern was made or edited at) were
            // served at every zoom and its curves showed facets zoomed in.
            shape.set("_appearanceBand", Value::from(band));
            let painted = tess::placed::with_node_paint(shape, layer);
            let order = (ci * per_copy + k) as f64;
            let mut scratch = PlanWarnings::default();
            let built = build_body_item(
                scene,
                &painted,
                z + slice * (0.05 + 0.9 * order / total),
                page_clip,
                Some(mask.clone()),
                true,
                shader_time_s,
                Some(affine),
                None,
                caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                &mut scratch,
            );
            for w in scratch.skipped {
                if seen.insert(w.clone()) {
                    warnings.skipped.push(w);
                }
            }
            for w in scratch.dropped {
                if seen.insert(w.clone()) {
                    warnings.dropped.push(w);
                }
            }
            match built {
                Ok(items) => out.extend(items),
                Err(e) => {
                    let msg = format!("{node_id} (pattern): tile shape {k}: {e}");
                    if seen.insert(msg.clone()) {
                        warnings.skipped.push(msg);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// A path whose stroke is an art / scatter brush: its body without the stroke,
/// then each placed artwork shape (a synthetic path in the node's local
/// space) at a fractional z in the node's slot. Placements are cached per node
/// + zoom band (art) + brush, and refined like outlines in live frames.
#[allow(clippy::too_many_arguments)]
/// The learned-bounds key of a placed brush stroke (`PlanCaches::brush_bounds`, read by `crate::cull`): its
/// geometry, its brush and its canvas transform. `None` when its box isn't learned: not a placed brush, a trim or
/// dash (what draws moves along the stroke), or more than one brushed stroke in its stack.
pub(crate) fn brush_bounds_key(scene: &Scene, node: &moda_scene::types::RawNode, geom: u64) -> Option<u64> {
    if paint::painted_stroke_width(node) <= 0.0 {
        return None;
    }
    tess::placed::placed_brush(scene, node)?;
    let brushed = node.array("strokes").map_or(0, |a| {
        a.iter()
            .filter(|s| s.get("brush").is_some_and(|b| !b.is_null()))
            .count()
    });
    let style = moda_scene::paint::node_stroke_style(node);
    if brushed != 1
        || style.dash.as_deref().is_some_and(|d| !d.is_empty())
        || trim_params_from_node(node).is_some()
    {
        return None;
    }
    let a = scene.node_canvas_affine(moda_scene::scene::paint_transform_id(node))?;
    let brush = tess::outline::brush_id(node)
        .as_deref()
        .and_then(|id| scene.doc().brush_hash(id));
    Some(brush_bounds_key_of(geom, brush, &a))
}

/// [`brush_bounds_key`] from its parts, for a caller that has them (`build_placed_brush_body`).
fn brush_bounds_key_of(geom: u64, brush: Option<u64>, a: &moda_scene::types::Affine) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    (geom, brush, 0xB0B5_u16).hash(&mut h);
    for v in [a.a, a.b, a.c, a.d, a.e, a.f] {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

/// The canvas box of a placed brush stroke's drawn bodies (every triangle, through its model matrix), grown by the
/// widest stroke reach among its shapes (a joint of another band's flattening can reach a little further). `None`
/// when something other than a tessellated body draws.
fn drawn_bodies_bounds(items: &[DrawItem], shapes: &[moda_scene::types::RawNode]) -> Option<crate::cull::Aabb> {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for item in items {
        let DrawItem::Body {
            geometry, instance, ..
        } = item
        else {
            return None;
        };
        let m = &instance.model_matrix;
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for p in geometry.positions.chunks_exact(2) {
            let (x, y) = (f64::from(p[0]), f64::from(p[1]));
            lo = [lo[0].min(x), lo[1].min(y)];
            hi = [hi[0].max(x), hi[1].max(y)];
        }
        if lo[0] > hi[0] {
            continue;
        }
        for (x, y) in [(lo[0], lo[1]), (hi[0], lo[1]), (lo[0], hi[1]), (hi[0], hi[1])] {
            let cx = f64::from(m[0]) * x + f64::from(m[4]) * y + f64::from(m[12]);
            let cy = f64::from(m[1]) * x + f64::from(m[5]) * y + f64::from(m[13]);
            if !(cx.is_finite() && cy.is_finite()) {
                return None;
            }
            b = [b[0].min(cx), b[1].min(cy), b[2].max(cx), b[3].max(cy)];
        }
    }
    if b[0] > b[2] {
        return None;
    }
    // The widest stroke among the shapes, at its miter reach, in canvas units (through the widest model scale).
    let reach = shapes
        .iter()
        .map(|s| {
            let w = paint::painted_stroke_width(s);
            if w > 0.0 {
                w * s.f64("miterLimit").filter(|m| m.is_finite()).unwrap_or(10.0).max(4.0)
            } else {
                0.0
            }
        })
        .fold(0.0, f64::max);
    let scale = items
        .iter()
        .filter_map(|i| match i {
            DrawItem::Body { instance, .. } => {
                let m = &instance.model_matrix;
                Some((f64::from(m[0]).abs() + f64::from(m[4]).abs()).max(f64::from(m[1]).abs() + f64::from(m[5]).abs()))
            }
            _ => None,
        })
        .fold(0.0, f64::max);
    let pad = reach * scale;
    Some([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad])
}

fn build_placed_brush_body(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    kind: tess::placed::PlacedKind,
    def: &Value,
    z: f64,
    page_clip: [f32; 4],
    mask: Option<crate::seams::MaskRef>,
    disable_shape_sdf: bool,
    shader_time_s: f64,
    canvas_affine_override: Option<moda_scene::types::Affine>,
    mut caches: Option<CacheCtx<'_>>,
    warnings: &mut PlanWarnings,
) -> Result<Vec<DrawItem>> {
    let band = if tess::placed::band_dependent(kind) {
        tess::outline::current_band()
    } else {
        0
    };
    let tolerance = tess::outline::tolerance_for_band(band);
    let geom = caches
        .as_ref()
        .and_then(|(_, h)| h.get(node.id()).map(|h| h.geom))
        .unwrap_or_else(|| crate::residency::node_hashes_in(scene, node).geom);
    // Keyed on geometry, the brush and the paint colourisation reads — never
    // on opacity / blend (applied per draw), so paint-only edits don't re-place.
    let brush_hash = tess::outline::brush_id(node)
        .as_deref()
        .and_then(|id| scene.doc().brush_hash(id));
    // Trim / dash as arc-length windows (the outliner's measure, as for
    // outlines). Art clips its artwork to them in the placer (a reveal; an
    // animated trim re-places per frame); scatter / pattern place the whole
    // stroke once and keep the copies / tiles inside the windows per frame.
    let style = moda_scene::paint::node_stroke_style(node);
    let dash = style.dash.clone();
    let trim = trim_params_from_node(node);
    let has_dash = dash.as_deref().is_some_and(|d| !d.is_empty());
    // Measured once per build, only when trim / dash apply.
    let spans = if trim.is_some() || has_dash {
        moda_scene::stroke_outline::subpath_spans(node)
    } else {
        Vec::new()
    };
    let (windows, dash_capped) = tess::outline::arc_windows_on(
        &spans,
        node,
        trim,
        dash.as_deref().filter(|d| !d.is_empty()),
        style.dash_offset.unwrap_or(0.0),
        paint::painted_stroke_width(node),
    );
    let in_placer = tess::placed::windows_in_placer(kind);
    // The windows (with their taper extents) key an art placement, but not
    // its `base`: under an animated trim, a live frame past its budget draws
    // the previous trim's reveal and refines it (stale-while-refine), like a
    // band change.
    let windows_hash = if in_placer {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        if let Some(ws) = &windows {
            for w in ws {
                (w.subpath, w.start.to_bits(), w.end.to_bits()).hash(&mut h);
                w.taper_extent
                    .map(|(a, b)| (a.to_bits(), b.to_bits()))
                    .hash(&mut h);
            }
        } else {
            0xFFu8.hash(&mut h);
        }
        h.finish()
    } else {
        0
    };
    let base = geom
        ^ tess::outline::outline_cache_salt(i32::MIN, brush_hash)
        ^ tess::placed::placement_paint_hash(node).rotate_left(29);
    let key = base
        ^ tess::outline::outline_cache_salt(band, None).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ windows_hash.rotate_left(41);
    let cached = caches.as_mut().map(|(c, _)| {
        // Where the node is on the canvas, for viewport-first refinement.
        let bbox = canvas_affine_override
            .or_else(|| scene.node_canvas_affine(moda_scene::scene::paint_transform_id(node)))
            .map(|a| {
                let cmds = tess::path_cmds_for_node(node);
                let w = paint::painted_stroke_width(node);
                let scale = (a.a.hypot(a.b)).max(a.c.hypot(a.d));
                tess::outline::refine_bbox(tess::outline::canvas_bbox(tess::outline::cmds_bounds(&cmds), &a, 4.0 * w * scale))
            })
            .unwrap_or([
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ]);
        let worker_placeable = tess::placed::worker_placeable(kind, windows.is_some());
        c.placed_get(node.id(), key, base, bbox, worker_placeable)
    });
    let placed = match cached {
        Some(crate::residency::PlacedLookup::Hit(p)) => p,
        // Left out of this live frame (never placed, past the placement budget): it draws on a later one.
        Some(crate::residency::PlacedLookup::Defer) => return Ok(Vec::new()),
        Some(crate::residency::PlacedLookup::Request) => {
            if let Some((c, _)) = caches.as_mut() {
                c.placement_request(node, kind, def, brush_hash, key, base);
            }
            return Ok(Vec::new());
        }
        // Another band's placement draws while a host worker places this band's.
        Some(crate::residency::PlacedLookup::Refine(p)) => {
            if let Some((c, _)) = caches.as_mut() {
                c.placement_request(node, kind, def, brush_hash, key, base);
            }
            p
        }
        Some(crate::residency::PlacedLookup::Place) | None => {
            let p = std::sync::Arc::new(tess::placed::place(
                node,
                kind,
                def,
                tolerance,
                if in_placer { windows.as_deref() } else { None },
            ));
            if let Some((c, _)) = caches.as_mut() {
                c.placed_put(node.id(), key, base, p.clone());
            }
            p
        }
    };
    let note = |warnings: &mut PlanWarnings, msg: &str| {
        warnings
            .skipped
            .push(format!("{} (brush): {msg}", node.id()));
    };
    for n in &placed.notes {
        note(warnings, n);
    }
    if dash_capped {
        note(
            warnings,
            &format!("dash pattern cut at {} dashes", tess::outline::MAX_WINDOWS),
        );
    }
    // Scatter / pattern under trim / dash: the copies / tiles whose anchor lies
    // in a window (shown or hidden whole).
    let keep: Option<Vec<bool>> = match (&windows, in_placer) {
        (Some(ws), false) => Some(
            placed
                .at
                .iter()
                .map(|at| tess::placed::in_windows(at, ws, &spans))
                .collect(),
        ),
        _ => None,
    };
    let affine = match canvas_affine_override {
        Some(a) => a,
        None => {
            let tid = moda_scene::scene::paint_transform_id(node);
            scene
                .node_canvas_affine(tid)
                .ok_or_else(|| anyhow!("no canvas affine for node {}", node.id()))?
        }
    };
    // Placed shapes first, into a scratch: a shape whose fill the tessellator
    // refuses must not take the page down — the node falls back to its plain
    // stroke instead (with a note).
    let mut shape_items: Vec<DrawItem> = Vec::new();
    let mut refused: Option<String> = None;
    // A bristle stroke's strands share one paint, as do a scatter brush's copies of one-colour artwork: one body of
    // all their triangles (`strand_run_body`) when that draws the same.
    let strands = if !placed.too_complex
        && keep.is_none()
        && STRAND_RUNS.with(std::cell::Cell::get)
    {
        strand_run_body(
            scene,
            node,
            &placed,
            z,
            page_clip,
            mask.clone(),
            shader_time_s,
            affine,
            // A host worker can re-tessellate these strands at a new band when it can place the stroke as this
            // thread did (`tess::placed::strands_job`: a bristle stroke without trim / dash).
            (kind == tess::placed::PlacedKind::Bristle && windows.is_none()).then_some((def, brush_hash)),
            caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
        )
    } else {
        None
    };
    if let Some(body) = strands {
        shape_items.extend(body);
    } else if !placed.too_complex {
        let n = placed.nodes.len();
        for (k, shape) in placed.nodes.iter().enumerate() {
            if keep
                .as_ref()
                .is_some_and(|keep| !keep.get(k).copied().unwrap_or(true))
            {
                continue;
            }
            let painted = tess::placed::with_node_paint(shape, node);
            let mut scratch = PlanWarnings::default();
            let built = build_body_item(
                scene,
                &painted,
                tess::placed::placed_z(z, k, n),
                page_clip,
                mask.clone(),
                true,
                shader_time_s,
                Some(affine),
                placed.geom_hashes.get(k).copied(),
                caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                &mut scratch,
            );
            if let Some(why) = scratch.dropped.first() {
                refused = Some(why.clone());
                break;
            }
            warnings.skipped.extend(scratch.skipped);
            match built {
                Ok(v) => shape_items.extend(v),
                Err(e) => note(warnings, &format!("artwork {k}: {e}")),
            }
        }
    }
    if let Some(why) = &refused {
        note(
            warnings,
            &format!("placed artwork too complex to fill ({why}); drew the plain stroke"),
        );
        shape_items.clear();
    }
    // A refused placement draws the plain stroke (the node without its brush).
    let plain = placed.too_complex || refused.is_some();
    let body = if plain {
        tess::placed::without_brush(node)
    } else {
        tess::placed::without_stroke(node)
    };
    // The stroke-less body draws the node's fill only: none when that paints nothing (a brushed stroke's usual
    // `fills: []` would still tessellate its open path's implicit fill, Q-51, and draw it at alpha 0 — discarded).
    let mut items = if plain || paint::fill_paints(&body) {
        build_body_item(
            scene,
            &body,
            z,
            page_clip,
            mask,
            disable_shape_sdf,
            shader_time_s,
            canvas_affine_override,
            // The two body variants (stroke-less / plain stroke) key apart.
            Some(if plain {
                geom ^ 0x5157_4F4B_0E5A_17A1
            } else {
                geom
            }),
            caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
            warnings,
        )
        .unwrap_or_default()
    } else {
        Vec::new()
    };
    // Placed shapes order by their fractional z buckets (painter's order), never by depth: n shapes share
    // 0.4 of the node's slot, and depth keeps apart only what is a twentieth of a slot from it (`depth`).
    for it in &mut shape_items {
        if let DrawItem::Body { phase, .. }
        | DrawItem::SolidShapeSdf { phase, .. }
        | DrawItem::GradientShapeSdf { phase, .. } = it
        {
            *phase = Phase::Transparent;
        }
    }
    items.extend(shape_items);
    // Learn the stroke's drawn box for the viewport cull (`crate::cull`): a live camera build of the stroke in
    // place (not a blend intermediate) with the whole placement drawn.
    if let Some((c, _)) = caches.as_mut() {
        // `brush_bounds_key`'s conditions, from what this build already knows (no trim, no dash: `windows`).
        let brushed = || {
            node.array("strokes").map_or(0, |a| {
                a.iter()
                    .filter(|s| s.get("brush").is_some_and(|b| !b.is_null()))
                    .count()
            })
        };
        if !plain && canvas_affine_override.is_none() && windows.is_none() && brushed() == 1 {
            let key = brush_bounds_key_of(geom, brush_hash, &affine);
            {
                if c.brush_bounds_get(node.id(), key).is_none() {
                    if let Some(b) = drawn_bodies_bounds(&items, &placed.nodes) {
                        c.brush_bounds_put(node.id(), key, b, band);
                    }
                }
            }
        }
    }
    Ok(items)
}

thread_local! {
    static STRAND_RUNS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Build a bristle stroke's strands as one body (`strand_run_body`; default on). Off only to compare against: the
/// pixels are the same.
pub fn set_strand_runs(on: bool) {
    STRAND_RUNS.with(|c| c.set(on));
}

/// The fields a placed shape carries for itself (its geometry and where it sits along the stroke), not its paint.
const PLACED_SHAPE_OWN: [&str; 7] = ["id", "sourceId", "points", "closed", "subpath", "s", "sSpan"];

/// A placement's shapes as ONE transparent solid body: every shape's fill triangles, in order, drawn with the first
/// shape's paint instance at the first shape's depth. A bristle stroke is dozens of strands of one paint; one body
/// each made a painting of 15k strokes 220k plan items (each paint parsed, cloned and packed on every build).
///
/// The same pixels as one body per shape, when (checked here, else `None` and the caller builds them one by one):
/// every shape is a stroke-less path with the first one's paint (its fields but its own geometry), and the first
/// one draws as a single transparent solid body. Then every shape's instance is the same but for its depth; the
/// transparent phase writes no depth, and nothing but this stroke's own items has a z inside its slot, so drawing
/// them at the first depth tests the same; their triangles rasterize and blend in the same order.
///
/// Each shape's triangles are what `build_body_item` tessellates for it (the same call on the same commands, at the
/// same band). The concatenation is cached as the tess entry `<node>::strands`, keyed on every shape's geometry
/// hash and the band.
///
/// `worker`: the brush definition and its hash when a host worker can place the stroke as this thread did (a bristle
/// stroke without trim / dash); with the host's workers on, a band change's refinement of on-screen strands is then
/// requested from them (`PlanCaches::strands_request`) and the previous band's strands draw until it arrives.
#[allow(clippy::too_many_arguments)]
fn strand_run_body(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    placed: &tess::placed::Placed,
    z: f64,
    page_clip: [f32; 4],
    mask: Option<crate::seams::MaskRef>,
    shader_time_s: f64,
    affine: moda_scene::types::Affine,
    worker: Option<(&Value, Option<u64>)>,
    mut caches: Option<CacheCtx<'_>>,
) -> Option<Vec<DrawItem>> {
    let n = placed.nodes.len();
    if n < 2 || placed.geom_hashes.len() != n {
        return None;
    }
    let first = &placed.nodes[0];
    fn paint_of(shape: &moda_scene::types::RawNode) -> Vec<(&moda_scene::value::Key, &Value)> {
        shape
            .fields()
            .iter()
            .filter(|(k, _)| !PLACED_SHAPE_OWN.contains(&k.as_str()))
            .collect()
    }
    let paint = paint_of(first);
    // `type` and the stroke fields are paint here: equal to the first shape's, so checked on it alone.
    if first.node_type() != "path"
        || paint::painted_stroke_width(first) != 0.0
        || !*placed
            .one_paint
            .get_or_init(|| placed.nodes[1..].iter().all(|s| paint_of(s) == paint))
    {
        return None;
    }
    // The first shape's paint, instance and blend stand for all. They don't depend on its geometry (a solid
    // instance is the matrix, the colours, the stroke width, the clip and the depth; the SDF lane is off for placed
    // shapes), so they come from the first shape's paint on a triangle: building the first shape itself tessellated
    // it again on every build of every stroke, cache hit or not.
    let stub = {
        let mut m = first.fields().clone();
        m.insert(
            "points".into(),
            moda_scene::value::json!([{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 1, "y": 0}, {"type": "line", "x": 0, "y": 1}]),
        );
        m.insert("closed".into(), Value::Bool(true));
        moda_scene::types::RawNode::new(m)
    };
    let painted = tess::placed::with_node_paint(&stub, node);
    let mut scratch = PlanWarnings::default();
    let z0 = tess::placed::placed_z(z, 0, n);
    let built = build_body_item(
        scene,
        &painted,
        z0,
        page_clip,
        mask,
        true,
        shader_time_s,
        Some(affine),
        None,
        None,
        &mut scratch,
    )
    .ok()?;
    if !scratch.dropped.is_empty() || !scratch.skipped.is_empty() {
        return None;
    }
    let (node_id, phase, body_paint, instance, blend_mode, mask) = match built.as_slice() {
        [DrawItem::Body { node_id, phase: Phase::Transparent, paint: paint @ BodyPaint::Solid, instance, blend_mode, mask, .. }] => {
            (node_id.clone(), Phase::Transparent, paint.clone(), instance.clone(), *blend_mode, mask.clone())
        }
        _ => return None,
    };
    // The strands' curves flatten per zoom band: keyed on it (`base`: the same strands at any band). After a band
    // change they refine like outlines: a live frame reuses a finer band's strands, tessellates what its refine
    // budget allows on and near the screen, and draws the rest from their previous band, queued until refined.
    // A host worker's placement brings them along at the band it was requested at.
    let band = tess::outline::current_band();
    let base = tess::placed::strand_run_base(placed);
    let key = tess::placed::strand_run_key(placed, band);
    let id = format!("{}::strands", node.id());
    let geometry = match caches.as_mut() {
        None => std::sync::Arc::new(tess::placed::strand_run_geometry(placed)?),
        Some((c, _)) => match c.tess_get(&id, key) {
            Some(g) => g,
            None => {
                let progressive = c.outline_progressive();
                let finer = if progressive {
                    c.outline_finer_get(&id, base, band)
                } else {
                    None
                };
                match finer {
                    Some(g) => g,
                    None => {
                        // The strands at another band, and where they are (canvas: their triangles' box), for
                        // viewport-first refinement.
                        let stale = if progressive {
                            c.outline_stale_get(&id, base).and_then(|g| {
                                let b = tess::outline::refine_bbox(tess::outline::canvas_bbox(g.bounds, &affine, 0.0));
                                let far = tess::outline::camera_view().is_some_and(|v| {
                                    tess::outline::visible_area(b, tess::outline::near_view(v)) <= 0.0
                                });
                                // With the host's workers on, on-screen strands refine there (the main thread only
                                // draws what arrives), as outlines do. A stroke a worker is re-placing (art at a
                                // new band) gets its strands with that placement: these are the old placement's.
                                let to_worker = !far
                                    && c.outline_worker()
                                    && !c.outline_no_worker_has(&id)
                                    && (c.placement_in_flight(node.id())
                                        || worker.is_some_and(|(def, brush_hash)| {
                                            c.strands_request(&id, node, def, brush_hash, key, base, band);
                                            true
                                        }));
                                (far || to_worker || !c.outline_budget_take()).then_some((g, b))
                            })
                        } else {
                            None
                        };
                        match stale {
                            Some((g, b)) => {
                                c.outline_pending_push(node.id(), b);
                                g
                            }
                            None => {
                                let g = std::sync::Arc::new(tess::placed::strand_run_geometry(placed)?);
                                c.outline_put(&id, key, base, Some(band), g.clone());
                                g
                            }
                        }
                    }
                }
            }
        },
    };
    if geometry.vertex_count == 0 {
        return Some(Vec::new());
    }
    Some(vec![DrawItem::Body { node_id, z: z0, phase, paint: body_paint, geometry, instance, blend_mode, mask }])
}

#[allow(clippy::too_many_arguments)]
/// The draw items of a node's silhouette (`layer_blur::shadow::silhouette_node`)
/// for its shadow capture: tessellated (no SDF, no mask, unclipped — the
/// composite clips), uncached (the silhouette shares the node's id, so the
/// per-node caches would mix the two up). Shapes and paths draw through the
/// body lane (live Transform copies each drawn), lines through the line lane,
/// rich text through the text lane.
fn build_shadow_capture_items(
    scene: &Scene,
    sil: &moda_scene::types::RawNode,
    z: f64,
    clip: [f32; 4],
    opts: &RenderOptions,
    mut caches: Option<CacheCtx<'_>>,
) -> Vec<DrawItem> {
    let mut warnings = PlanWarnings::default();
    match sil.node_type() {
        "richtext" => match crate::text_draw::build_richtext_item(
            scene,
            sil,
            z,
            clip,
            opts.best_effort_text,
            opts.shader_time_s,
            opts.scale,
            None,
        ) {
            Ok(crate::text_draw::TextBuildOutcome::Item(text)) => vec![*text],
            _ => Vec::new(),
        },
        "line" => build_line_items(
            scene,
            sil,
            z,
            crate::seams::ARROWHEAD_Z_OFFSET,
            clip,
            None,
            None,
            None,
            &mut warnings,
        )
        .unwrap_or_default(),
        _ => {
            let parts = moda_scene::live_effects::split_copies(sil);
            let single = parts.is_none();
            let parts = parts.unwrap_or_else(|| vec![sil.clone()]);
            parts
                .iter()
                .flat_map(|part| {
                    let mut folded = match paint::layers::plan(part) {
                        paint::layers::LayerPlan::Single(Some(folded)) => folded,
                        _ => part.clone(),
                    };
                    // A single silhouette tessellates through the per-node
                    // cache under its own id (drawn through the node's
                    // transform), so a plan rebuild — an edit elsewhere on the
                    // page — reuses it. Transform copies share one id and skip
                    // the cache, as the node's own copies do.
                    let cached = caches.as_mut().filter(|_| single).map(|(c, h)| {
                        let source = moda_scene::scene::paint_transform_id(&folded).to_string();
                        let id = format!("{}{SILHOUETTE_SUFFIX}", folded.id());
                        folded.set("id", Value::from(id));
                        folded.set("_transformSourceId", Value::from(source));
                        (&mut **c, *h)
                    });
                    let geom_hash = cached
                        .is_some()
                        .then(|| crate::residency::node_hashes(&folded).geom);
                    build_body_item(
                        scene,
                        &folded,
                        z,
                        clip,
                        None,
                        true,
                        opts.shader_time_s,
                        None,
                        geom_hash,
                        cached,
                        &mut warnings,
                    )
                    .unwrap_or_default()
                })
                .collect()
        }
    }
}

/// The suffix of a shadow / glow silhouette's per-node cache id (`<node id>` + this).
const SILHOUETTE_SUFFIX: &str = "\u{1}silhouette";

/// The node a silhouette's cache id (`build_shadow_capture_items`) belongs to; any other id as it is. A silhouette
/// drawn from another zoom band's flattening queues under its own id, and refines when its node re-plans.
pub(crate) fn silhouette_owner_id(item_id: &str) -> &str {
    item_id.strip_suffix(SILHOUETTE_SUFFIX).unwrap_or(item_id)
}

/// [`build_body_item_inner`] with the node's dash alignment set for every dash
/// walk under it ([`tess::with_dash_alignment`]).
#[allow(clippy::too_many_arguments)]
fn build_body_item(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    z: f64,
    page_clip: [f32; 4],
    mask: Option<crate::seams::MaskRef>,
    disable_shape_sdf: bool,
    shader_time_s: f64,
    canvas_affine_override: Option<moda_scene::types::Affine>,
    geom_hash: Option<u64>,
    caches: Option<CacheCtx<'_>>,
    warnings: &mut PlanWarnings,
) -> Result<Vec<DrawItem>> {
    let aligned = moda_scene::paint::node_stroke_style(node).dash_aligned();
    // Curves flatten for the size the node reaches the screen at: the view's zoom band raised by the node's own
    // canvas scale (its scale × its parents', animated values included; the larger axis of a non-uniform one), so a
    // small shape scaled up is not drawn from its source-size facets. The tess caches key on the resulting band.
    let node_scale = canvas_affine_override
        .or_else(|| scene.node_canvas_affine(moda_scene::scene::paint_transform_id(node)))
        .map_or(1.0, |a| tess::outline::affine_max_scale(&a));
    let _node_scale = tess::outline::enter_node_scale(node_scale);
    tess::with_dash_alignment(aligned, || {
        build_body_item_inner(
            scene,
            node,
            z,
            page_clip,
            mask,
            disable_shape_sdf,
            shader_time_s,
            canvas_affine_override,
            geom_hash,
            caches,
            warnings,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn build_body_item_inner(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    // f64: brush artwork draws at fractional z inside its stroke node's slot.
    z: f64,
    page_clip: [f32; 4],
    mask: Option<crate::seams::MaskRef>,
    // The engine's `disableOpaqueFastPath` equivalent — true on a
    // blur-segmented (backdrop) page, where the analytic SDF fast path declines
    // and the body tessellates, matching WebGPU. This is the ONLY context the
    // ticket maps `disableOpaqueFastPath` to ("not in the engine's multi-pass/
    // blur-segmented context"). We deliberately do NOT mirror WebGPU's OTHER,
    // export-wide `disableOpaqueFastPath` (`WebGPURenderer` export passes /
    // `renderConfig.ts`): that one tessellates ALL shapes on export to dodge
    // A2C phantom lines at shape boundaries after the MSAA resolve on
    // SwiftShader-class export backends. Keeping the fast path ON for export is
    // the whole point of this ticket: the WASM engine IS the export/headless
    // lane and its exports must be crisp (not faceted) at arbitrary zoom. The
    // native opaque SDF body matches the WebGPU EDITOR SDF path's AA (alpha-to-
    // coverage under MSAA, `shape_sdf_pipeline`) rather than that path; on the
    // real-GPU export backends the native engine targets, single-quad SDF
    // bodies carry no internal edges and so do not exhibit the SwiftShader
    // phantom-line artifact. Consequence (expected, not a regression): native
    // exports are crisper than — and so not pixel-identical to — WebGPU exports
    // for these shapes.
    disable_shape_sdf: bool,
    shader_time_s: f64,
    // M4b: synthetic blend intermediates are not in the scene graph — the
    // production renderer translates them by `_pageOrigin` (+ node.x/y;
    // `createNodeModelMatrix` transient arm, `WebGPUPathRenderer.ts:2198-2221`).
    // `Some(affine)` supplies that transform; `None` = scene lookup.
    canvas_affine_override: Option<moda_scene::types::Affine>,
    // The node's geometry hash when the caller has it and the resident maps
    // don't (placed brush artwork); `None` = look it up / hash the node.
    geom_hash: Option<u64>,
    // Task #95 R3: per-node tessellation cache + resident hash maps.
    caches: Option<CacheCtx<'_>>,
    // A fail-closed fill drop lands in `warnings.dropped` (no log sink).
    warnings: &mut PlanWarnings,
) -> Result<Vec<DrawItem>> {
    let node_type = node.node_type();

    // E8.5: pattern layers draw only in the paint-stack lanes (`build_pattern_layer_items`). Any other lane
    // that reaches here with one (a blend step, a boolean operand, …) draws the node's other layers.
    if crate::pattern_paint::has_pattern_layer(node) {
        warnings.skipped.push(format!(
            "{} (pattern): a pattern paint is not drawn in this context; the node's other paints draw",
            node.id()
        ));
        return build_body_item(
            scene,
            &crate::pattern_paint::without_pattern_layers(node),
            z,
            page_clip,
            mask,
            disable_shape_sdf,
            shader_time_s,
            canvas_affine_override,
            None,
            caches,
            warnings,
        );
    }

    // E6.2: a rectangle / frame stroke with per-side weights paints as a filled ring.
    if let Some(ring) = side_stroke_ring_node(node) {
        return build_body_item(
            scene,
            &ring,
            z,
            page_clip,
            mask,
            true,
            shader_time_s,
            canvas_affine_override,
            None,
            caches,
            warnings,
        );
    }

    // E6.1: an open path whose stroke has arrowheads draws its stroke cut back
    // under them, then the arrowheads in the stroke's paint.
    if node_type == "path" {
        if let Some(parts) = path_arrow_parts(node) {
            let mut out = build_body_item(
                scene,
                &parts.body,
                z,
                page_clip,
                mask.clone(),
                disable_shape_sdf,
                shader_time_s,
                canvas_affine_override,
                None,
                caches,
                warnings,
            )?;
            let arrow_z = tess::placed::placed_z(z, 1, 2);
            for head in parts.heads {
                out.extend(build_body_item(
                    scene,
                    &head,
                    arrow_z,
                    page_clip,
                    mask.clone(),
                    disable_shape_sdf,
                    shader_time_s,
                    canvas_affine_override,
                    None,
                    None,
                    warnings,
                )?);
            }
            return Ok(out);
        }
    }

    // Brush plan P8 follow-up: an art / scatter brush replaces the stroke
    // with placed artwork, drawn like a group in this node's slot.
    if paint::painted_stroke_width(node) > 0.0 {
        if let Some((kind, def)) = tess::placed::placed_brush(scene, node) {
            return build_placed_brush_body(
                scene,
                node,
                kind,
                def,
                z,
                page_clip,
                mask,
                disable_shape_sdf,
                shader_time_s,
                canvas_affine_override,
                caches,
                warnings,
            );
        }
    }

    // Analytic SDF fast path. SDF-eligible solid/gradient rect &
    // ellipse bodies render as a zoom-invariant unit quad through the
    // shape-solid / shape-gradient modules instead of being tessellated (crisp
    // at arbitrary zoom). An active mask (`mask.is_some()`) or the blur-segmented
    // context (`disable_shape_sdf`) decline the fast path; anything else
    // ineligible falls through to the tessellation path below, byte-unchanged.
    if !disable_shape_sdf {
        if let Some(items) = try_shape_sdf_body(
            scene,
            node,
            node_type,
            z,
            page_clip,
            mask.as_ref(),
            canvas_affine_override,
        )? {
            return Ok(items);
        }
    }

    // M3b: paint routing (WebGPUPathRenderer.ts:1076-1133 order; see
    // paint::classify_fill docs for the copied quirks).
    let route = paint::classify_fill(node);
    // M3f: custom shader fills — pack the 672-byte tail off the manifest
    // packing tables (deterministic `--shader-time-s` clock). M4e: the
    // effective shader id/params follow production precedence (renderable
    // source-mask effect beats fillShaderId — `WebGPUPathRenderer.ts:
    // 2034-2043`), and image-backed shaders resolve their preprocessed
    // texture + metadata from the prefetched shader-image store. Unknown ids
    // still error into a skip warning (manifest `unknown`).
    let shader_fill: Option<(
        moda_shaders::runtime::PackedFill,
        Option<crate::seams::ShaderImageTexture>,
        Option<Vec<u16>>,
    )> = if route == paint::FillRoute::Shader {
        let Some((shader_id, raw_params, effect)) =
            moda_scene::shader_effects::effective_shader_fill(node)
        else {
            // Routed to customShaderPaths with a null shaderId — the
            // production draw loop skips the node (`if (shaderId)`).
            bail!("shader fill: source-mask effect without a shaderId draws nothing");
        };
        if crate::agent_fill::is_code_ref(shader_id) {
            // An agent shader: a code key in the document's code table (crate::agent_fill).
            let empty = Value::Object(moda_scene::value::Map::new());
            let packed =
                crate::agent_fill::pack(scene, node, shader_id, raw_params.unwrap_or(&empty), shader_time_s)?;
            Some((packed, None, crate::agent_fill::clock_offsets()))
        } else {
            let empty = Value::Object(moda_scene::value::Map::new());
            let effect_params = raw_params.unwrap_or(&empty);
            let live_source_fill_params = (effect.is_some()
                && moda_shaders::runtime::shader_uses_source_fill_placement(shader_id))
            .then(|| moda_scene::shader_effects::with_source_fill_placement(node, effect_params));
            let params = live_source_fill_params.as_ref().unwrap_or(effect_params);
            // moda-shaders reads serde_json values: the one conversion of this fill's parameters.
            let params = &params.to_json();
            // Image source comes ONLY from a renderable effect (the
            // fillShaderId-only route always binds the fallback texture —
            // `WebGPUPathRenderer.ts:2050`, quirk Q-68).
            let processing = moda_shaders::runtime::shader_image_processing(shader_id, params);
            let (image_meta, image_tex) = match &processing {
                Some(proc) => {
                    let src =
                        effect.and_then(|e| moda_scene::shader_effects::effect_image_src(node, e));
                    resolve_shader_image_binding(src, proc)
                }
                None => (None, None),
            };
            // The `shader_params` clock slots, so the live camera present
            // path can re-thread a new clock onto this CACHED fill's instance with
            // no plan rebuild. `None` = a clock use we cannot safely re-thread,
            // which keeps the plan clock-keyed (`plan_shader_time_rethreadable`).
            let clock_offsets = moda_shaders::runtime::clock_param_offsets(shader_id, params);
            match moda_shaders::runtime::pack_fill_with_image(
                shader_id,
                params,
                shader_time_s,
                image_meta,
            ) {
                Ok(packed) => Some((packed, image_tex, clock_offsets)),
                Err(e) => bail!("shader fill: {e}"),
            }
        }
    } else {
        None
    };
    let fill_paint = paint::gradient::fill_paint(node);
    // No stroke paint route → no stroke band (`getPaintedStrokeWidth`
    // at every `tessellateCombined` call in `WebGPUPathRenderer.ts`). Without
    // this, width alone emitted a band painted with the opaque-black `stroke`
    // fallback below — an outline hit-testing and bounds never accounted for.
    let stroke_width = paint::painted_stroke_width(node);

    // The canonical `distort` warp (stored 3×3 unit-square matrix)
    // pre-warps a shape's GEOMETRY on the CPU and warps pattern fills via the
    // inverse homography (`getNodeDistort`, core/scene/distortGeometry.ts).
    // `node_distort` gates on the capable leaf types (rectangle/ellipse/star/
    // polygon; richtext warps in the vector text lane — but a shape's
    // textContent LABEL does not warp yet, the documented gap in
    // `build_shape_text_item`) and validates the matrix — a node WITHOUT a
    // valid distort takes exactly the pre-existing paths.
    let distort = paint::node_distort(node);

    // Geometry (export mode: rect/ellipse → cubic paths, shapeToPathNodeUtils).
    let style = moda_scene::paint::node_stroke_style(node);
    let (cmds, closed, fill_rule, line_cap, line_join, stroke_align) = match node_type {
        "rectangle" | "container" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            let cmds = shape_to_path::rect_outline_cmds(w, h, node.json_field("cornerRadius"), node.json_field("cornerSmoothing"), node.json_field("cornerStyle"));
            (
                cmds,
                true,
                pt::FillRule::NonZero,
                pt::line_cap_from_str(style.cap.as_deref()),
                pt::line_join_from_str(style.join.as_deref()),
                pt::stroke_align_from_str(style.align.as_deref()),
            )
        }
        "ellipse" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            (
                shape_to_path::ellipse_to_path_cmds(
                    w,
                    h,
                    node.f64("startAngle"),
                    node.f64("sweepAngle"),
                    node.f64("innerRadiusRatio"),
                ),
                true,
                pt::FillRule::NonZero,
                pt::line_cap_from_str(style.cap.as_deref()),
                pt::line_join_from_str(style.join.as_deref()),
                pt::stroke_align_from_str(style.align.as_deref()),
            )
        }
        "path" => (
            tess::path_cmds_for_node(node),
            node.bool("closed").unwrap_or(false),
            pt::fill_rule_from_str(node.str("fillRule")),
            pt::line_cap_from_str(style.cap.as_deref()),
            pt::line_join_from_str(style.join.as_deref()),
            pt::stroke_align_from_str(style.align.as_deref()),
        ),
        // M2: stars/polygons are always converted to closed nonzero paths
        // (`convertStarToPathNode` / `convertPolygonToPathNode`).
        "star" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            // `star.cornerRadius || 0` (JS falsy: 0/NaN/missing → 0).
            let cr = node
                .f64("cornerRadius")
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            (
                tess::shape_paths::star_to_path_cmds(
                    w,
                    h,
                    node.f64("numPoints").unwrap_or(f64::NAN),
                    node.f64("innerRadiusRatio"),
                    cr,
                ),
                true,
                pt::FillRule::NonZero,
                pt::line_cap_from_str(style.cap.as_deref()),
                pt::line_join_from_str(style.join.as_deref()),
                pt::stroke_align_from_str(style.align.as_deref()),
            )
        }
        "polygon" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            let cr = node
                .f64("cornerRadius")
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            (
                tess::shape_paths::polygon_to_path_cmds(
                    w,
                    h,
                    node.f64("sides").unwrap_or(f64::NAN),
                    cr,
                ),
                true,
                pt::FillRule::NonZero,
                pt::line_cap_from_str(style.cap.as_deref()),
                pt::line_join_from_str(style.join.as_deref()),
                pt::stroke_align_from_str(style.align.as_deref()),
            )
        }
        other => bail!("build_body_item: unsupported node type '{other}'"),
    };

    // Distort pre-warp: flatten the converted outline (rounded
    // corners included) and project every point through the node's local-px
    // homography, so the tessellator and the stroke walk see already-warped
    // geometry. Stroke width stays uniform along the warped outline — the
    // accepted MVP fallback (outline-then-warp is a follow-up). If the warp
    // declines (belt-and-braces — validation precludes it; only a degenerate
    // outline with < 3 flattened points per contour reaches it), drop the
    // warp metadata too so pattern UVs are never inverse-mapped over unwarped
    // geometry. The other distort consumers (SDF/shadow gates, mask/backdrop
    // AABBs, the text lane) re-derive their gate from `node_distort` alone
    // and would still treat the node as warped — accepted: the only reachable
    // case is a sub-3-point outline whose body is at most a stroke sliver.
    let (cmds, distort) = match distort {
        Some(d) => match tess::shape_paths::warp_path_cmds(&cmds, &d.local) {
            Some(warped) => (warped, Some(d)),
            None => (cmds, None),
        },
        None => (cmds, None),
    };
    // Pattern-UV warp metadata: the distort pattern-warp view. (The legacy
    // corner-pin field never survives the load-time migration, so this is the
    // only warp source.)
    let perspective = distort.as_ref().map(paint::NodeDistort::as_pattern_warp);

    // M2: dash routes through the arc-length walk in the tessellator
    // (scaled by strokeWidth/2, odd patterns doubled; each sub-path dashes
    // from its own start, joins survive at vertices, and a closed sub-path
    // closes at its seam). Non-array dash values are skipped.
    let dash = style.dash.clone();
    // Dash offset (marching ants): phase into the dash pattern. Inert without dash.
    let dash_offset = style.dash_offset.unwrap_or(0.0);

    // Trim paths (stroke-only): a visible [start,end] arc-length window plus an
    // offset, applied before dash. `None` when absent or identity so the no-trim
    // render stays byte-identical.
    let trim = trim_params_from_node(node);

    // R3 (task #95): per-node tessellation cache, keyed by the deny-list
    // GeomKey (residency.rs — transform/paint props excluded, strokeWidth/
    // dash/caps/joins/fillRule/strokeAlign/perspective/fillPriority IN).
    // ONE entry per node, evict-and-replace on key change. Synthetic nodes
    // (blend intermediates) hash their own value on the fly — that key folds
    // every tessellation input they carry.
    //
    // Brush plan P8: a path whose stroke has a width profile / taper / wave /
    // wobble / calligraphic brush draws its stroke as a filled OUTLINE
    // (`tess::outline`), keyed additionally on the zoom band and the brush
    // definition. Trim and dash become arc-length windows on the outline.
    let outlined = node_type == "path"
        && distort.is_none()
        && stroke_width > 0.0
        && tess::outline::scene_needs_outline(scene, node);
    let outline_brush = if outlined {
        tess::outline::brush_id(node)
            .as_deref()
            .and_then(|id| scene.doc().brush(id))
    } else {
        None
    };
    let outline_brush_hash = if outlined {
        tess::outline::brush_id(node)
            .as_deref()
            .and_then(|id| scene.doc().brush_hash(id))
    } else {
        None
    };
    let outline_band = tess::outline::current_band();
    let mut outline_note: Option<String> = None;
    // A cached outline re-reports the note it was built with.
    let mut hit_note: Option<String> = None;
    // A vector network (E7.9) draws natively: fills per region, strokes per
    // segment with caps and joins at its vertices (`tess::network`). A brush
    // or width profile outlines the derived runs above instead.
    let network = if node_type == "path" && !outlined {
        tess::network::node_network(node)
    } else {
        None
    };
    let mut tess = || {
        if let Some(net) = &network {
            let dash = dash.as_deref().filter(|d| !d.is_empty());
            // Aligned strokes are undashed and untrimmed closed loops, as on a path.
            let align = if dash.is_some() || trim.is_some() {
                pt::StrokeAlign::Center
            } else {
                stroke_align
            };
            let plan = tess::network::plan_stroke(net, line_cap, line_join, align);
            let arrow_scale = moda_scene::paint::top_stroke(node).and_then(|s| s.arrow_scale);
            return tess::with_miter_limit(node.miter_limit(), || {
                tess::tessellate_network(
                    &tess::network::region_fills(net),
                    &plan,
                    &tess::NetworkStrokeStyle {
                        width: stroke_width,
                        dash,
                        dash_offset,
                        line_cap,
                        line_join,
                        miter_limit: node.miter_limit(),
                        trim,
                        arrow_size: tess::path_arrows::path_arrow_size(stroke_width, arrow_scale),
                    },
                )
            });
        }
        if outlined {
            let (g, note) = tess::outline::tessellate_outlined(
                node,
                outline_brush,
                &tess::outline::OutlineBody {
                    cmds: &cmds,
                    closed,
                    fill_rule,
                    stroke_width,
                    trim,
                    dash: dash.as_deref().filter(|d| !d.is_empty()),
                    dash_offset,
                    line_cap,
                    line_join,
                    stroke_align,
                    cut: None,
                },
                outline_band,
            );
            outline_note = note;
            return g;
        }
        tess::with_miter_limit(node.miter_limit(), || {
            tess::tessellate_combined_with_dash_trim(
                &cmds,
                stroke_width,
                closed,
                trim,
                dash.as_deref().filter(|d| !d.is_empty()),
                dash_offset,
                line_cap,
                line_join,
                fill_rule,
                stroke_align,
            )
        })
    };
    // `Arc`-shared geometry: a tess-cache hit hands back a pointer-bump clone,
    // and the same `Arc` lands in `DrawItem::Body.geometry` so the plan never
    // deep-copies the vertex stream (the camera lane's `make_mut` clone bumps it).
    // The node's geometry hash when the caller already has it (the resident scene's map), and a repeat copy's source
    // (`moda_scene::repeat`): a copy has its source's geometry, so a plain body keys the tessellation cache by the
    // source's id and geometry hash, and the source is tessellated once however many copies draw.
    // An export job (`crate::export_job`) tessellates plain bodies through the command-keyed cache as an uncached
    // render does: it is process-wide, so it already shares their geometry across the job's pages and layers.
    let job_plain = !outlined
        && network.is_none()
        && caches.as_ref().is_some_and(|(c, _)| c.job_sharing());
    let resident = caches
        .as_ref()
        .filter(|_| !job_plain)
        .and_then(|(_, hashes)| geom_hash.or_else(|| hashes.get(node.id()).map(|h| h.geom)));
    let shared = caches.as_ref().filter(|_| !job_plain).and_then(|(_, hashes)| {
        if outlined || geom_hash.is_some() {
            return None;
        }
        node.str(moda_scene::repeat::REPEAT_SOURCE_KEY)
            .and_then(|src| hashes.get(src).map(|h| (src, h.geom)))
    });
    // A plain body with neither (an export's frames, which have no resident map) tessellates through the
    // command-keyed cache below, as without caches: hashing the whole node every frame costs more than that cache's
    // lookup. Outlines and networks have no such cache, so they hash the node here.
    let keyed = resident.is_some() || shared.is_some() || outlined || network.is_some();
    let geometry: std::sync::Arc<TessellatedGeometry> = if let Some((cache, _)) = caches.filter(|_| keyed) {
        let key = resident.unwrap_or_else(|| crate::residency::node_hashes_in(scene, node).geom);
        let (tess_id, key) = shared.unwrap_or((node.id(), key));
        // Live frames: the node's estimated canvas box, and whether it is on or
        // near screen (refinement goes there first). Outlines and banded
        // uniform strokes share this.
        let est_bbox = if cache.outline_progressive() {
            let affine = canvas_affine_override
                .or_else(|| scene.node_canvas_affine(moda_scene::scene::paint_transform_id(node)));
            affine.map(|a| {
                let pad = 4.0 * stroke_width
                    + style
                        .dynamic
                        .as_ref()
                        .and_then(|d| d.get("wiggle"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        .abs();
                let scale = (a.a.hypot(a.b)).max(a.c.hypot(a.d));
                tess::outline::refine_bbox(tess::outline::canvas_bbox(tess::outline::cmds_bounds(&cmds), &a, pad * scale))
            })
        } else {
            None
        };
        let near = est_bbox
            .zip(tess::outline::camera_view())
            .map(|(b, v)| tess::outline::visible_area(b, tess::outline::near_view(v)) > 0.0);
        let pending_bbox = est_bbox.unwrap_or([
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        ]);
        if outlined {
            // Brush plan P8: keyed on the zoom band and the brush definition.
            // After a zoom-band change, a live frame past its refine budget
            // draws the node's outline from another band and the host renders
            // again until all are refined. A node whose geometry (or brush)
            // changed has no such entry and is always computed this frame.
            let base = key ^ tess::outline::outline_cache_salt(i32::MIN, outline_brush_hash);
            let key = key
                ^ tess::outline::outline_cache_salt(outline_band, outline_brush_hash)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15);
            // Live camera frames refine what is on screen first: an outline
            // off-screen (beyond half a viewport) is drawn from another band,
            // or — never computed yet — as a uniform placeholder nobody sees,
            // and queued; on-screen ones spend the frame's time budget.
            let has_base = cache.outline_has_base(node.id(), base);
            let finer = if cache.outline_progressive() {
                cache.outline_finer_get(node.id(), base, outline_band)
            } else {
                None
            };
            match (cache.tess_get(node.id(), key), finer) {
                (Some(g), _) => {
                    hit_note = cache.outline_note(node.id(), key);
                    g
                }
                // Live frames: an outline fitted at a finer band is exact here.
                (None, Some(g)) => g,
                (None, None) => {
                    let far = near == Some(false);
                    // Exports and captures (no live budget) always compute. So
                    // does a stroke never outlined for this geometry / brush
                    // (new or edited) on or near screen: an edited stroke is
                    // never drawn without its stroke, even mid-drag.
                    let now = !far && cache.outline_budget_take();
                    // A big multi-select edit spreads over frames: past the
                    // budget, an edited stroke draws its pre-edit outline for
                    // at most OUTLINE_MAX_PREV_FRAMES frames (queued), then is
                    // computed over budget — never nothing, never starved by
                    // strokes earlier in document order. A stroke with no
                    // outline at all is computed now.
                    let prev = (!has_base && !far && !now)
                        .then(|| cache.outline_prev_get(node.id()))
                        .flatten();
                    if now || (!has_base && !far && prev.is_none()) {
                        let g = std::sync::Arc::new(tess());
                        cache.outline_put(node.id(), key, base, Some(outline_band), g.clone());
                        cache.outline_note_put(node.id(), key, outline_note.clone());
                        g
                    } else {
                        // Deferred: the same geometry at another band (a
                        // band refinement — requested from the host's
                        // workers when on / near screen), or far off-screen:
                        // draw the previous band's outline, else a uniform
                        // stand-in; queue the node for refinement.
                        // Band refinements and edited strokes drawing their
                        // pre-edit outline go to the workers when on.
                        if (has_base || prev.is_some()) && !far && cache.outline_worker() {
                            cache.outline_request(node, key, base, outline_band, outline_brush);
                        }
                        let g = match cache.outline_stale_get(node.id(), base).or(prev) {
                            Some(g) => g,
                            None => {
                                let g =
                                    std::sync::Arc::new(tess::tessellate_combined_with_dash_trim(
                                        &cmds,
                                        stroke_width,
                                        closed,
                                        trim,
                                        dash.as_deref().filter(|d| !d.is_empty()),
                                        dash_offset,
                                        line_cap,
                                        line_join,
                                        fill_rule,
                                        stroke_align,
                                    ));
                                // Its key never matches a band; `base` lets it be served stale.
                                cache.outline_put(node.id(), base, base, None, g.clone());
                                g
                            }
                        };
                        // No canvas box (no affine): queue it as if on screen.
                        cache.outline_pending_push(node.id(), pending_bbox);
                        g
                    }
                }
            }
        } else {
            // Curves (strokes and fills) flatten per zoom band
            // (`flatness_tolerance`, F4.1): key on it off 1×, so band 0 keeps
            // today's keys. Panning, and zooming within a band, hit this
            // entry. After a band change these refine like outlines (#96's
            // rules): a flattening from a finer band is reused on live frames;
            // otherwise on / near-screen strokes spend the frame's budget and
            // the rest draw their previous band's geometry and are queued
            // (with their canvas box) until refined. A new or edited stroke is
            // always computed. Workers are never asked: a uniform stroke is
            // not an outline.
            let band = outline_band;
            // The fill is flattened per band too, so any curve keys on it.
            // So do round caps and joins (`tessellator::round_arc_segments`).
            let round_ends = stroke_width > 0.0
                && (line_cap == pt::LineCap::Round || line_join == pt::LineJoin::Round);
            let curved = round_ends
                || cmds
                    .iter()
                    .any(|c| !matches!(c, pt::PathCmd::Move { .. } | pt::PathCmd::Line { .. }));
            if curved {
                let base = key ^ 0x5EED_BA5E_C0DE_F1A7;
                let key = if band != 0 {
                    key ^ tess::outline::outline_cache_salt(band, None)
                        .wrapping_mul(0xD6E8_FEB8_6659_FD93)
                } else {
                    key
                };
                let has_base = cache.outline_has_base(tess_id, base);
                let finer = if cache.outline_progressive() {
                    cache.outline_finer_get(tess_id, base, band)
                } else {
                    None
                };
                match (cache.tess_get(tess_id, key), finer) {
                    (Some(g), _) => g,
                    (None, Some(g)) => g,
                    (None, None) => {
                        let far = near == Some(false);
                        let stale = if !has_base || (!far && cache.outline_budget_take()) {
                            None
                        } else {
                            cache.outline_stale_get(tess_id, base)
                        };
                        match stale {
                            Some(g) => {
                                cache.outline_pending_push(node.id(), pending_bbox);
                                g
                            }
                            None => {
                                // A plain body is tessellated through the command-keyed cache: nodes with the same
                                // geometry, stroke and band share one tessellation and one `Arc`, whatever their ids.
                                let g = if network.is_none() {
                                    tess::with_miter_limit(node.miter_limit(), || {
                                        tess::tessellate_combined_cached(
                                            &cmds, stroke_width, closed, trim, dash.as_deref().filter(|dash| !dash.is_empty()),
                                            dash_offset, line_cap, line_join, fill_rule, stroke_align,
                                        )
                                    })
                                } else {
                                    std::sync::Arc::new(tess())
                                };
                                cache.outline_put(tess_id, key, base, Some(band), g.clone());
                                g
                            }
                        }
                    }
                }
            } else {
                // Zoomed out (band < 0), a straight-edged fill's outline is simplified to the band's tolerance
                // (`tessellator::fill_path_at`), so it keys on the band there too: a flattening cached at another
                // band dropped (or kept) other vertices — found by the slot fuzz's blend steps, whose outlines hold
                // near-collinear points. Two bands stay cached (as curves do), so a page plan at 1× and a camera
                // plan zoomed out don't evict each other's.
                let base = key ^ 0x51D3_5EED_F1A7_0B1D;
                let key = if band < 0 {
                    key ^ tess::outline::outline_cache_salt(band, None).wrapping_mul(0xA24B_AED4_963E_E407)
                } else {
                    key
                };
                match cache.tess_get(tess_id, key) {
                    Some(g) => g,
                    None => {
                        // A plain body is tessellated through the command-keyed cache: nodes with the same
                        // geometry, stroke and band share one tessellation and one `Arc`, whatever their ids.
                        let g = if network.is_none() {
                            tess::with_miter_limit(node.miter_limit(), || {
                                tess::tessellate_combined_cached(
                                    &cmds, stroke_width, closed, trim, dash.as_deref().filter(|dash| !dash.is_empty()),
                                    dash_offset, line_cap, line_join, fill_rule, stroke_align,
                                )
                            })
                        } else {
                            std::sync::Arc::new(tess())
                        };
                        cache.outline_put(tess_id, key, base, Some(band.min(0)), g.clone());
                        g
                    }
                }
            }
        }
    } else if !outlined && network.is_none() {
        tess::with_miter_limit(node.miter_limit(), || {
            tess::tessellate_combined_cached(
                &cmds, stroke_width, closed, trim, dash.as_deref().filter(|dash| !dash.is_empty()),
                dash_offset, line_cap, line_join, fill_rule, stroke_align,
            )
        })
    } else {
        std::sync::Arc::new(tess())
    };
    if let Some(note) = outline_note.or(hit_note) {
        warnings
            .skipped
            .push(format!("{} (path): {note}", node.id()));
    }
    // A dropped fill is a fail-closed refusal when the fill would have PAINTED:
    // the stroke (if any) still draws, so the item is not otherwise empty and
    // nothing else would flag the page. The reason rides the cached geometry,
    // so a cache hit reports it again on every build. Visibility follows the
    // resolved fill, not the overall pipeline route: the gradient pipeline may
    // be selected solely for a gradient STROKE while the fill remains an
    // alpha-0 solid; an actual gradient is invisible when every resolved stop
    // is alpha 0. Pattern/shader fills are conservatively visible because their
    // source alpha is not knowable from node fields. An invisible fill's empty
    // stream degrades only the bounds it feeds (Q-51), so that
    // refusal is a tolerated `skipped` line — a stroke-only path over the
    // vertex cap must not block an export.
    if let Some(why) = &geometry.fill_dropped {
        let fill_visible = node.f64("opacity").unwrap_or(1.0) > 0.0
            && (matches!(route, paint::FillRoute::Pattern | paint::FillRoute::Shader)
                || paint::fill_paints(node));
        let line = format!("{} ({node_type}): fill dropped: {why}", node.id());
        if fill_visible {
            warnings.dropped.push(line);
        } else {
            warnings.skipped.push(line);
        }
    }
    // A refused stroke (over `MAX_STROKE_STREAM_VERTICES`) would have painted: the stroke was asked for, and the
    // reason rides the cached geometry like the fill's.
    if let Some(why) = &geometry.stroke_dropped {
        warnings
            .dropped
            .push(format!("{} ({node_type}): stroke dropped: {why}", node.id()));
    }
    if geometry.vertex_count == 0 {
        return Ok(Vec::new());
    }
    // E8.18: a gradient stroke laid along or across the path reads each stroke
    // vertex's (along, across) coordinates, from the outline's centreline.
    let geometry = tess::stroke_param::with_stroke_uv(geometry, node, !outlined, || {
        tess::stroke_param::centerline(&cmds, closed, tess::stroke_param::CENTERLINE_TOLERANCE)
    });

    // Colors + opacity (WebGPUPathRenderer.ts:1250-1285, 1480-1481).
    let opacity = node.f64("opacity").unwrap_or(1.0);
    let top_stroke = paint::gradient::top_stroke(node);
    let stroke_paint = top_stroke.as_ref().map(|s| &s.paint);
    let fill = paint::gradient::solid_fill_color(fill_paint.as_ref());
    let stroke = paint::gradient::solid_stroke_color(stroke_paint);

    // Transparency classification (getPathDrawMeta, WebGPUPathRenderer.ts:577-666).
    // M3b terms: pattern fills classify against the neutral
    // IMAGE_LOADING_FILL (a = 1), not node.fill; patterns are ALWAYS
    // transparent (Q-22); a translucent gradient stop (fill OR stroke set)
    // forces transparency. NOTE `hasPattern` here is fillPriority only —
    // independent of the gradient-fields routing quirk above.
    let blend_mode = BlendMode::from_node_value(node.str("blendMode"));
    // A 'video' fill routes through the same pattern texture path
    // (`classify_fill`) and shares the neutral IMAGE_LOADING_FILL placeholder
    // while its poster loads, so it classifies for transparency exactly like a
    // pattern — otherwise the textured poster body lands in the opaque phase and
    // can depth-reject the overlay / z-fight.
    let has_pattern_priority = matches!(
        fill_paint,
        Some(moda_scene::paint::Paint::Image(_) | moda_scene::paint::Paint::Video(_))
    );
    let fill_transparent = if has_pattern_priority {
        false // IMAGE_LOADING_FILL has alpha 1
    } else {
        parse_color(fill_paint.as_ref().and_then(moda_scene::paint::solid_color))
            .map(|c| c.a < 1.0)
            .unwrap_or(false)
    };
    let stroke_transparent = stroke_width > 0.0
        && parse_color(stroke_paint.and_then(moda_scene::paint::solid_color))
            .map(|c| c.a < 1.0)
            .unwrap_or(false);
    let node_transparent = node
        .json_field("opacity")
        .and_then(Value::as_f64)
        .map(|o| o < 1.0)
        .unwrap_or(false);
    let gradient_stop_transparent =
        paint::has_gradient_fill(node) && paint::gradient_has_transparency(node);
    let stroke_gradient_transparent =
        stroke_width > 0.0 && paint::stroke_gradient_has_transparency(node);
    // M3d: the inset-shadow quad draws over the body in the transparent
    // phase, so the body must be transparent too; masked (R8-sampled)
    // content is ALWAYS transparent. Drop shadows do NOT make
    // the body transparent by themselves (render-semantics §1.5).
    let has_inner_shadow = crate::effects::has_inner_shadow_quad(node);
    // M3f: a custom shader fill + visible gradient stroke renders as two
    // separate draws; both stay transparent so the overlay is not
    // depth-rejected against the shader fill (`getPathDrawMeta`, :654-657).
    let shader_with_gradient_stroke =
        shader_fill.is_some() && paint::has_visible_gradient_stroke(node);
    let is_transparent = fill_transparent
        || stroke_transparent
        || node_transparent
        || gradient_stop_transparent
        || stroke_gradient_transparent
        || has_pattern_priority
        || shader_with_gradient_stroke
        || has_inner_shadow
        || mask.is_some()
        || blend_mode != BlendMode::Normal;

    // Model matrix: canvas-space affine incl. page offset + group chain
    // (computeModelMatrixWithPreview without selection preview — exports
    // skip the overlay merge). Synthetic blend intermediates use the
    // transient-origin translate instead (M4b).
    let affine = match canvas_affine_override {
        Some(a) => a,
        // `createNodeModelMatrix`'s transform-id redirect
        // (`WebGPUPathRenderer.ts:2202-2208`): synthetic proxies
        // (generated-graphic expansion, Q-37-adjacent transients) inherit the
        // SOURCE node's chain via `_transformSourceId`; their own x/y are
        // ignored on this path. Without it, raster-asset generated proxies
        // nested in containers render at host-local page coords (attribution
        // sweep: 3d0f70ef p0 map, d31a991e p3 latex).
        None => {
            let tid = moda_scene::scene::paint_transform_id(node);
            scene
                .node_canvas_affine(tid)
                .ok_or_else(|| anyhow!("no canvas affine for node {}", node.id()))?
        }
    };
    let model_matrix = affine_to_matrix(&affine);

    let depth_val = depth::linear_depth(z) as f32;
    let mut instance = PaintInstance::zeroed_box();
    instance.model_matrix = model_matrix;
    instance.fill_color = [
        fill.r as f32,
        fill.g as f32,
        fill.b as f32,
        (fill.a * opacity) as f32,
    ];
    instance.stroke_color = [
        stroke.r as f32,
        stroke.g as f32,
        stroke.b as f32,
        (stroke.a * opacity) as f32,
    ];
    let sw = if stroke_width == 0.0 {
        1.0
    } else {
        stroke_width
    };
    instance.path_info = [0.0, geometry.vertex_count as f32, sw as f32, depth_val];
    instance.bounds = geometry.bounds;
    instance.page_clip_bounds = page_clip;

    // M3f: shader-fill instance overrides (`renderCustomShaderPaths`,
    // `WebGPUPathRenderer.ts:2017-2090`): fillColor = [1,1,1, nodeOpacity]
    // (the shader epilogue applies fillColor.a); a visible gradient stroke
    // zeroes strokeColor (the overlay draw paints it); shaderParams/colorData
    // come from the manifest-driven packer.
    if let Some((packed, _, _)) = &shader_fill {
        instance.fill_color = [1.0, 1.0, 1.0, opacity as f32];
        if paint::has_visible_gradient_stroke(node) {
            instance.stroke_color = [0.0, 0.0, 0.0, 0.0];
        }
        instance.shader_params = packed.shader_params;
        instance.color_data = packed.color_data;
    }

    let phase = if is_transparent {
        Phase::Transparent
    } else {
        Phase::Opaque
    };

    // M3b: build the routed paint payload. M3f: a shader fill with a visible
    // gradient stroke ALSO emits the stroke-only gradient overlay (drawn
    // after the shader body — `postShaderGradientStrokePaths`,
    // `WebGPUPathRenderer.ts:1079-1112`).
    let needs_gradient_stroke_overlay = (route == paint::FillRoute::Pattern
        || route == paint::FillRoute::Shader)
        && paint::has_visible_gradient_stroke(node);
    // The box gradients span: the path's geometric bounds, not the tessellated
    // fill ∪ stroke (`paint::gradient_box`).
    // An arrowed path's parts (`path_arrow_parts`) carry the uncut path's box.
    let gradient_box =
        (route == paint::FillRoute::Gradient || needs_gradient_stroke_overlay).then(|| {
            node.array("_paintBox")
                .filter(|b| b.len() == 4)
                .map(|b| std::array::from_fn(|i| b[i].as_f64().unwrap_or(0.0) as f32))
                .unwrap_or_else(|| {
                    paint::gradient_box(tess::path_bounds(&cmds), stroke_width, geometry.bounds)
                })
        });
    let body_paint = match route {
        paint::FillRoute::Gradient => {
            // Intended semantic, per the donor spike: gradient
            // fills on a distorted shape re-fit to the WARPED AABB
            // (`gradient_box` measures the post-warp `cmds`) WITHOUT warping
            // the gradient field itself — an axis-aligned ramp over warped
            // geometry, matching how shape gradients bind to the path's
            // bounds everywhere else. Only pattern/video fills inverse-map through
            // the homography; vector-text gradients warp with the glyphs.
            BodyPaint::Gradient(Box::new(paint::build_gradient_instance(
                node,
                model_matrix,
                depth_val,
                gradient_box.unwrap_or(geometry.bounds),
                page_clip,
                false,
            )))
        }
        paint::FillRoute::Pattern => {
            // A `'video'` fill resolves its texture through
            // the poster-vs-streaming rule: prefer the LIVE frame installed under
            // the per-node key `video:<nodeId>` (P1 — VideoPlaybackController),
            // else the poster URL (`fillVideoPosterSrc`). An image `'pattern'`
            // fill sources from `fillPatternImageSrc`. Everything downstream
            // (instance packing, fit, treatments) reads the shared `fillPattern*`
            // fields and is identical for both.
            // A gradient mesh (`crate::mesh_paint`): its bake, drawn as an image placed over the mesh's bounds
            // (the node's mesh layer swapped for that image). The mesh's % coordinates are of the paint box, as a
            // gradient's.
            let mesh_image = match &fill_paint {
                Some(moda_scene::paint::Paint::Mesh(m)) => {
                    let paint_box = node
                        .array("_paintBox")
                        .filter(|b| b.len() == 4)
                        .map(|b| std::array::from_fn(|i| b[i].as_f64().unwrap_or(0.0)))
                        .unwrap_or_else(|| {
                            paint::gradient_box(tess::path_bounds(&cmds), stroke_width, geometry.bounds)
                                .map(f64::from)
                        });
                    Some(match crate::mesh_paint::baked(m, paint_box) {
                        Some((texture, rect)) => (crate::mesh_paint::as_image_node(node, &texture, rect), texture),
                        // An invalid mesh draws nothing.
                        None => (node.clone(), crate::mesh_paint::transparent()),
                    })
                }
                _ => None,
            };
            let node = mesh_image.as_ref().map_or(node, |(n, _)| n);
            let texture = match &fill_paint {
                Some(moda_scene::paint::Paint::Mesh(_)) => mesh_image.as_ref().map(|(_, t)| t.clone()).unwrap(),
                Some(moda_scene::paint::Paint::Video(v)) => {
                    // A paint stack's top video layer streams its node's video
                    // (`_videoStreamId`); lower video layers draw their posters.
                    let stream_key = format!(
                        "video:{}",
                        node.str(paint::layers::VIDEO_STREAM_KEY)
                            .unwrap_or(node.id())
                    );
                    paint::resolve_video_source(&stream_key, v.poster_src.as_deref())
                }
                Some(moda_scene::paint::Paint::Image(i)) => {
                    paint::resolve_pattern_source(i.src.as_deref())
                }
                _ => paint::resolve_pattern_source(None),
            };
            // The box the image's scale mode resolves against: the node's
            // width × height, or a path's geometric bounds.
            let image_box = moda_scene::paint::node_image_box(node).unwrap_or_else(|| {
                tess::path_bounds(&cmds)
                    .map(|[x0, y0, x1, y1]| [x0, y0, x1 - x0, y1 - y0])
                    .unwrap_or([
                        geometry.bounds[0] as f64,
                        geometry.bounds[1] as f64,
                        geometry.bounds[2] as f64,
                        geometry.bounds[3] as f64,
                    ])
            });
            let inst = paint::build_pattern_instance(&paint::PatternInstanceContext {
                node,
                model_matrix,
                depth: depth_val,
                page_clip_bounds: page_clip,
                tess_bounds_size: Some((geometry.bounds[2], geometry.bounds[3])),
                texture_state: texture.state,
                texture_dims: (texture.image.width, texture.image.height),
                perspective: perspective.as_ref(),
                gradient_stroke_overlay: needs_gradient_stroke_overlay,
                image_box,
                // M3c: SVG/small-raster upscale compensation.
                texture_scale_factor: texture.scale_factor,
            });
            BodyPaint::Pattern(Box::new(paint::PatternBody {
                instance: inst,
                clamp_sampler: paint::pattern::image_fill_parts(node).0.mode() != "tile",
                texture,
            }))
        }
        paint::FillRoute::Solid => BodyPaint::Solid,
        paint::FillRoute::Shader => {
            let (packed, image, clock_offsets) = shader_fill
                .as_ref()
                .expect("Shader route packs above (or bails)");
            BodyPaint::Shader(Box::new(crate::seams::ShaderFillBody {
                module_id: packed.module_id.clone(),
                image: image.clone(),
                clock_offsets: clock_offsets.clone(),
            }))
        }
    };

    // Pattern + visible gradient stroke: a SECOND stroke-only gradient
    // overlay draw at the same z, after the pattern within the bucket
    // (`renderUnifiedPathBatch` pushes `{...pathInfo, strokeOnlyOverlay}`
    // into gradientPaths — gradients draw after non-gradients).
    let overlay_geometry = needs_gradient_stroke_overlay.then(|| geometry.clone());

    let mut out = Vec::with_capacity(2);
    out.push(DrawItem::Body {
        node_id: node.id().to_string(),
        z,
        phase,
        paint: body_paint,
        geometry,
        instance,
        blend_mode,
        mask: mask.clone(),
    });
    if let Some(overlay_geometry) = overlay_geometry {
        let g = paint::build_gradient_instance(
            node,
            model_matrix,
            depth_val,
            gradient_box.unwrap_or(overlay_geometry.bounds),
            page_clip,
            true, // strokeOnlyOverlay → transparentSolidFill
        );
        let mut overlay_instance = PaintInstance::zeroed_box();
        overlay_instance.model_matrix = model_matrix;
        overlay_instance.path_info = [
            0.0,
            overlay_geometry.vertex_count as f32,
            sw as f32,
            depth_val,
        ];
        overlay_instance.bounds = overlay_geometry.bounds;
        overlay_instance.page_clip_bounds = page_clip;
        // Q-116: SHADER-fill overlays draw in the post-shader slot
        // (`postShaderGradientStrokePaths` — after `renderCustomShaderPaths`,
        // `WebGPUPathRenderer.ts:1190-1194`); PATTERN-fill overlays stay in
        // the regular gradient slot (pushed into `gradientPaths`, :1121).
        let overlay_paint = if route == paint::FillRoute::Shader {
            BodyPaint::PostShaderGradientStroke(Box::new(g))
        } else {
            BodyPaint::Gradient(Box::new(g))
        };
        out.push(DrawItem::Body {
            node_id: format!("{}:gradient-stroke-overlay", node.id()),
            z,
            phase,
            paint: overlay_paint,
            geometry: overlay_geometry,
            instance: overlay_instance,
            blend_mode,
            mask,
        });
    }
    Ok(out)
}

/// M2: line node → body + arrowhead DrawItems
/// (`WebGPUShapeSystem.renderLine` split; geometry from
/// `tess::line_to_path`). Body draws at `z`; arrowheads at
/// `z + ARROWHEAD_Z_OFFSET` (fractional — consumes no z slot). Gradient
/// strokes on lines are M3b (skip like other non-solid paint).
#[allow(clippy::too_many_arguments)]
fn build_line_items(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    // f64: a stroke layer of a paint stack draws at fractional z inside the line's slot.
    z: f64,
    // Arrowheads' z above the body (`ARROWHEAD_Z_OFFSET`; a stroke layer's own slice in a stack).
    arrow_dz: f64,
    page_clip: [f32; 4],
    mask: Option<crate::seams::MaskRef>,
    // M4b: connector-text gap (arc-length fractions) carved into the body.
    text_gap: Option<tess::line_to_path::TextGap>,
    // Task #95 R3: per-node line tessellation cache (one Vec per node, in
    // part order; key = GeomKey ⊕ gap bits ⊕ font generation, since the
    // label gap rides font metrics).
    caches: Option<CacheCtx<'_>>,
    // A fail-closed fill drop on a part lands in `warnings.dropped` (a filled
    // arrowhead) or `warnings.skipped` (the alpha-0 body / open arrowhead).
    warnings: &mut PlanWarnings,
) -> Result<Vec<DrawItem>> {
    use tess::line_to_path::{line_parts_with_gap, LinePartKind, LineSpec};

    // M3b: a visible gradient stroke routes the line BODY (and open
    // arrowheads — stroked geometry) through the gradient pipeline with a
    // transparent solid fill, mirroring the synthetic path node production
    // builds (`WebGPULineToPath.ts:487-505`: fill 'transparent' + the stroke
    // gradient fields copied; `renderUnifiedPathBatch` then routes it via
    // needsGradientStrokeOverlay). FILLED arrowheads with a gradient stroke
    // map the gradient onto the FILL (`getStrokeAsFillPaint`): every part is
    // painted with the stroke's paint, mapped over the line body's bounds so
    // the gradient runs on into the arrowheads.
    let line_gradient_stroke = paint::has_visible_gradient_stroke(node);
    // The line draws its top stroke: without one it draws nothing, arrowheads
    // included. A stack of strokes draws one clone per stroke (E3.1 P6,
    // `build_stack_items`), so this is the layer being drawn.
    let Some(line_stroke) = paint::gradient::top_stroke(node) else {
        return Ok(Vec::new());
    };
    // A filled arrowhead under a gradient stroke: the stroke's gradient as the
    // arrowhead's fill.
    let arrow_fill_node = line_gradient_stroke.then(|| {
        let mut m = node.fields().clone();
        m.insert("type".into(), Value::String("path".into()));
        moda_scene::paint::set_fill_paint(&mut m, line_stroke.paint.clone());
        moda_scene::paint::clear_strokes(&mut m);
        node.with_fields(m)
    });

    // Brush plan §4.9: a line with a width profile / taper / wave draws its BODY as a filled
    // outline (like an outlined path), built on the undashed body with trim and dash as
    // arc-length windows. Arrowheads stay ordinary parts.
    let outlined_body = tess::outline::scene_needs_outline(scene, node);
    let mut spec = LineSpec::from_json(|k| node.json_field(k));
    if outlined_body {
        tess::outline::outlined_line_spec(node, &mut spec);
    }
    let parts = line_parts_with_gap(&spec, text_gap);

    // Trim paths (stroke-only) AND dash offset (marching ants) for the line BODY.
    // The line→path conversion PRE-APPLIES dash into `part.cmds`, so neither trim
    // (would measure ink length not path length) nor a phased dash offset can be
    // done on the dashed cmds. When either is active we re-derive the parts from a
    // dash-suppressed spec (undashed body cmds) and run trim→dash(+offset)→tessellate
    // together via `tessellate_combined_with_dash_trim`. Arrowheads are separate
    // parts here (arrow-bearing lines keep a real stroke width with the heads as
    // their own parts), so they are dash/trim-independent and stay at the
    // endpoints — a trimmed line keeps its arrowheads. See the roadmap doc.
    let trim = trim_params_from_node(node);
    let dash_offset = moda_scene::paint::node_stroke_style(node)
        .dash_offset
        .unwrap_or(0.0);
    // dashOffset is inert without a dash pattern, so only the undashed rebuild for
    // trim (always) or a phased dash offset when the line actually has a dash.
    let line_has_dash = spec.dash.as_ref().map(|d| !d.is_empty()).unwrap_or(false);
    let outline_band = tess::outline::current_band();
    let needs_undashed_body =
        outlined_body || trim.is_some() || (dash_offset != 0.0 && line_has_dash);
    let mut nodash_spec = spec.clone();
    nodash_spec.dash = None;
    // An outlined body is outlined WHOLE, with its label gap cut out afterwards as an
    // arc-length window (as trim and dash are), so the profile, taper and wave run once over
    // the whole line rather than again after the label.
    let nodash_parts: Option<Vec<tess::line_to_path::LinePart>> = needs_undashed_body
        .then(|| line_parts_with_gap(&nodash_spec, if outlined_body { None } else { text_gap }));
    let outline_gap = match (&nodash_parts, text_gap.filter(|g| g.active())) {
        (Some(full), Some(gap)) if outlined_body => {
            let gapped = line_parts_with_gap(&nodash_spec, Some(gap));
            full.first()
                .zip(gapped.first())
                .and_then(|(f, g)| tess::outline::line_gap_interval(&f.cmds, &g.cmds))
        }
        _ => None,
    };

    // R3 (task #95): line-node tessellation residency. The key folds the
    // node GeomKey (points/waypoints/arrows/dash/strokeWidth/... — deny-list
    // law), the carved text gap, and the text-host generation (the gap is
    // measured with font metrics).
    let mut line_cache: Option<(&mut crate::residency::PlanCaches, u64)> = None;
    let mut cached_geoms: Option<Vec<crate::seams::TessellatedGeometry>> = None;
    if let Some((cache, hashes)) = caches {
        use std::hash::{Hash, Hasher};
        let geom = hashes
            .get(node.id())
            .map(|h| h.geom)
            .unwrap_or_else(|| crate::residency::node_hashes_in(scene, node).geom);
        let mut hasher = rustc_hash::FxHasher::default();
        geom.hash(&mut hasher);
        if let Some(g) = &text_gap {
            g.start_t.to_bits().hash(&mut hasher);
            g.end_t.to_bits().hash(&mut hasher);
        }
        crate::text_draw::text_host_generation().hash(&mut hasher);
        // The outline is built per zoom band, and curved parts (curved routes,
        // rounded elbows, circle arrowheads) and round caps and joins flatten per
        // zoom band (F4.1, `tessellator::round_arc_segments`): key on the band for
        // them, so a part flattened at one band is never drawn at another.
        let curved_parts = parts.iter().any(|p| {
            (p.stroke_width > 0.0
                && (p.line_cap.as_deref() == Some("round") || p.line_join.as_deref() == Some("round")))
                || p.cmds
                    .iter()
                    .any(|c| !matches!(c, pt::PathCmd::Move { .. } | pt::PathCmd::Line { .. }))
        });
        if outlined_body || curved_parts {
            outline_band.hash(&mut hasher);
        }
        let key = hasher.finish();
        cached_geoms = cache
            .line_tess_get(node.id(), key)
            .filter(|v| v.len() == parts.len());
        line_cache = Some((cache, key));
    }
    let mut built_geoms: Vec<crate::seams::TessellatedGeometry> = Vec::new();

    let opacity = node.f64("opacity").unwrap_or(1.0);
    // The colour of the solid parts: the stroke's colour for a solid stroke;
    // opaque white under a gradient (the gradient pipeline paints it); nothing
    // for a paint the line lane doesn't draw yet (image, video, shader, conic).
    let stroke = if line_gradient_stroke {
        Rgba {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    } else {
        paint::gradient::solid_stroke_color(Some(&line_stroke.paint))
    };
    let mut body_bounds: Option<[f32; 4]> = None;
    let node_transparent = node
        .json_field("opacity")
        .and_then(Value::as_f64)
        .map(|o| o < 1.0)
        .unwrap_or(false);

    let affine = scene
        .node_canvas_affine(moda_scene::scene::paint_transform_id(node))
        .ok_or_else(|| anyhow!("no canvas affine for node {}", node.id()))?;
    let model_matrix = affine_to_matrix(&affine);

    // E8.18: a gradient laid along / across the stroke measures every part
    // (arrowheads included) against the undashed, ungapped body centreline.
    let body_line = tess::stroke_param::needs_stroke_uv(node).then(|| {
        line_parts_with_gap(&nodash_spec, None)
            .into_iter()
            .find(|p| p.kind == LinePartKind::Body)
            .map(|p| {
                tess::stroke_param::centerline(
                    &p.cmds,
                    p.closed,
                    tess::stroke_param::CENTERLINE_TOLERANCE,
                )
            })
            .unwrap_or_default()
    });

    let mut out: Vec<DrawItem> = Vec::new();
    for (part_idx, part) in parts.into_iter().enumerate() {
        // Dash is pre-applied by the line→path conversion (M-L segments);
        // re-dashing here would double-dash (WebGPULineToPath.ts:506-511).
        let geometry = match &cached_geoms {
            Some(geoms) => geoms[part_idx].clone(),
            None => {
                // Trimmed BODY: tessellate the UNDASHED body cmds with trim→dash.
                // All other parts (arrowheads) and the untrimmed case are
                // unchanged — their `part.cmds` already carry any dash.
                let nodash_body = nodash_parts.as_ref().and_then(|np| np.get(part_idx));
                let g = match nodash_body {
                    Some(nbody)
                        if part.kind == LinePartKind::Body
                            && outlined_body
                            && part.stroke_width > 0.0 =>
                    {
                        let body =
                            tess::outline::line_body_as_path(node, &nbody.cmds, part.stroke_width);
                        let (g, note) = tess::outline::tessellate_outlined(
                            &body,
                            None,
                            &tess::outline::OutlineBody {
                                cmds: &nbody.cmds,
                                closed: false,
                                fill_rule: pt::FillRule::NonZero,
                                stroke_width: part.stroke_width,
                                trim,
                                dash: spec.dash.as_deref().filter(|d| !d.is_empty()),
                                dash_offset,
                                line_cap: pt::line_cap_from_str(part.line_cap.as_deref()),
                                line_join: pt::line_join_from_str(part.line_join.as_deref()),
                                stroke_align: pt::stroke_align_from_str(None),
                                cut: outline_gap,
                            },
                            outline_band,
                        );
                        if let Some(note) = note {
                            warnings
                                .skipped
                                .push(format!("{} (line body outline): {note}", node.id()));
                        }
                        g
                    }
                    Some(nbody) if part.kind == LinePartKind::Body => {
                        // Trim (may be None when only dash offset is active) then
                        // dash+offset, on the UNDASHED body.
                        tess::tessellate_combined_with_dash_trim(
                            &nbody.cmds,
                            part.stroke_width,
                            nbody.closed,
                            trim,
                            spec.dash.as_deref().filter(|d| !d.is_empty()),
                            dash_offset,
                            pt::line_cap_from_str(part.line_cap.as_deref()),
                            pt::line_join_from_str(part.line_join.as_deref()),
                            pt::FillRule::NonZero,
                            pt::stroke_align_from_str(None),
                        )
                    }
                    _ => tess::tessellate_combined(
                        &part.cmds,
                        part.stroke_width,
                        part.closed,
                        pt::line_cap_from_str(part.line_cap.as_deref()),
                        pt::line_join_from_str(part.line_join.as_deref()),
                        pt::FillRule::NonZero,
                        pt::stroke_align_from_str(None),
                    ),
                };
                if line_cache.is_some() {
                    built_geoms.push(g.clone());
                }
                g
            }
        };
        let geometry = match body_line.as_ref().filter(|_| part.stroke_width > 0.0) {
            Some(line) if geometry.stroke_uv.is_empty() => {
                tess::stroke_param::stroke_uv(&geometry, line, !outlined_body).unwrap_or(geometry)
            }
            _ => geometry,
        };
        // Paint per part (convertLine*ToPathNode property mapping):
        //   Body / ArrowOpen → fill 'transparent', stroke = line stroke.
        //   ArrowFill        → fill = line stroke color, no stroke.
        let (fill, part_stroke, z_f, id_suffix) = match part.kind {
            LinePartKind::Body => (
                Rgba {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                },
                stroke,
                z,
                "",
            ),
            LinePartKind::ArrowFill => (
                // The stroke's paint as the arrowhead's fill.
                stroke,
                Rgba {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                z + arrow_dz,
                "-arrow-fill",
            ),
            LinePartKind::ArrowOpen => (
                Rgba {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.0,
                },
                stroke,
                z + arrow_dz,
                "-arrow-open",
            ),
        };

        // A refused fill fails a server export closed only when it would have
        // painted. The body and open arrowheads carry an alpha-0 fill by design
        // (Q-51 — it must still be tessellated for the gradient bounds and
        // Q-78 intermediates), so a self-crossing connector over the vertex cap
        // is a tolerated `skipped` line; a filled arrowhead paints the resolved
        // stroke colour and stays a `dropped` refusal.
        if let Some(why) = &geometry.fill_dropped {
            let fill_visible = fill.a > 0.0 && opacity > 0.0;
            let line = format!("{} (line part {part_idx}): fill dropped: {why}", node.id());
            if fill_visible {
                warnings.dropped.push(line);
            } else {
                warnings.skipped.push(line);
            }
        }
        if let Some(why) = &geometry.stroke_dropped {
            warnings.dropped.push(format!(
                "{} (line part {part_idx}): stroke dropped: {why}",
                node.id()
            ));
        }
        if geometry.vertex_count == 0 {
            continue;
        }

        let fill_transparent = fill.a < 1.0;
        let stroke_transparent = part.stroke_width > 0.0 && part_stroke.a < 1.0;
        let stroke_gradient_transparent = part.stroke_width > 0.0
            && line_gradient_stroke
            && paint::stroke_gradient_has_transparency(node);
        let is_transparent = fill_transparent
            || stroke_transparent
            || stroke_gradient_transparent
            || node_transparent
            || mask.is_some();

        let depth_val = depth::linear_depth(z_f) as f32;
        let mut instance = PaintInstance::zeroed_box();
        instance.model_matrix = model_matrix;
        instance.fill_color = [
            fill.r as f32,
            fill.g as f32,
            fill.b as f32,
            (fill.a * opacity) as f32,
        ];
        instance.stroke_color = [
            part_stroke.r as f32,
            part_stroke.g as f32,
            part_stroke.b as f32,
            (part_stroke.a * opacity) as f32,
        ];
        let sw = if part.stroke_width == 0.0 {
            1.0
        } else {
            part.stroke_width
        };
        instance.path_info = [0.0, geometry.vertex_count as f32, sw as f32, depth_val];
        instance.bounds = geometry.bounds;
        instance.page_clip_bounds = page_clip;
        if part.kind == LinePartKind::Body {
            body_bounds = Some(geometry.bounds);
        }
        // The gradient maps over the body's bounds for every part.
        let gradient_bounds = body_bounds.unwrap_or(geometry.bounds);

        // M3b: stroked parts of a gradient-stroke line draw via the gradient
        // pipeline (stroke gradient from the node's stroke; fill stays
        // transparent via stroke_only_overlay — production fill 'transparent');
        // filled arrowheads fill with the stroke's gradient.
        let paint_kind = match (&arrow_fill_node, part.kind) {
            (Some(_), LinePartKind::Body | LinePartKind::ArrowOpen) => {
                BodyPaint::Gradient(Box::new(paint::build_gradient_instance(
                    node,
                    model_matrix,
                    depth_val,
                    gradient_bounds,
                    page_clip,
                    true, // transparentSolidFill — the body has no fill of its own
                )))
            }
            (Some(fill_node), LinePartKind::ArrowFill) => {
                BodyPaint::Gradient(Box::new(paint::build_gradient_instance(
                    fill_node,
                    model_matrix,
                    depth_val,
                    gradient_bounds,
                    page_clip,
                    false,
                )))
            }
            _ => BodyPaint::Solid,
        };

        out.push(DrawItem::Body {
            node_id: format!("{}{}", node.id(), id_suffix),
            z: z_f,
            phase: if is_transparent {
                Phase::Transparent
            } else {
                Phase::Opaque
            },
            paint: paint_kind,
            // Line parts cache as plain `Vec<TessellatedGeometry>` (lines are not
            // a hot path); wrap per-part for the `Arc`-shared plan field.
            geometry: std::sync::Arc::new(geometry),
            instance,
            blend_mode: BlendMode::Normal,
            mask: mask.clone(),
        });
    }

    if let (Some((cache, key)), None) = (line_cache, &cached_geoms) {
        cache.line_tess_put(node.id(), key, &built_geoms);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// M4b — blend-node expansion drawing (`WebGPUShapeSystem.renderBlend`)
// ---------------------------------------------------------------------------

/// `getPageClipFromScene(scene, pageIndex)` for the blend SOURCES' page —
/// production assigns it as `_pageClipBoundsOverride` on every intermediate.
fn blend_page_clip(scene: &Scene, page_index: usize) -> [f32; 4] {
    if !scene.doc().config.render_page_clipping {
        return INFINITE_CLIP;
    }
    let Some(page) = scene.doc().pages.get(page_index) else {
        return INFINITE_CLIP;
    };
    let (px, py) = scene.page_position(page_index);
    [
        px as f32,
        py as f32,
        page.width() as f32,
        page.height() as f32,
    ]
}

/// `PathCmd` → blend path point (same variants, different crates).
fn path_cmds_to_blend_points(cmds: &[pt::PathCmd]) -> Vec<moda_scene::blend::BlendPathPoint> {
    use moda_scene::blend::BlendPathPoint as BP;
    cmds.iter()
        .map(|c| match c {
            pt::PathCmd::Move { x, y } => BP::Move { x: *x, y: *y },
            pt::PathCmd::Line { x, y } => BP::Line { x: *x, y: *y },
            pt::PathCmd::Curve { x, y, cp1, cp2 } => BP::Curve {
                x: *x,
                y: *y,
                cp1: *cp1,
                cp2: *cp2,
            },
            pt::PathCmd::Quadratic { x, y, cp } => BP::Quadratic {
                x: *x,
                y: *y,
                cp: *cp,
            },
            pt::PathCmd::Arc {
                x,
                y,
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
            } => BP::Arc {
                x: *x,
                y: *y,
                rx: *rx,
                ry: *ry,
                x_axis_rotation: *x_axis_rotation,
                large_arc: *large_arc,
                sweep: *sweep,
            },
        })
        .collect()
}

/// `shapeNodeToAbsolutePathPoints` (`core/utils/shapeToPath.ts`) —
/// the converter injected into the blend expansion. Local geometry comes
/// from the engine's existing shape→path converters (export-mode cubics);
/// `applyAbsolute` composes the node's absolute TRS + page offset like the
/// production helper. NOTE: production uses `rectangleToPathPoints` /
/// `ellipseDimensionsToPathPoints` (command-path generators); the engine
/// reuses its `shape_to_path` cubics instead — same silhouettes, different
/// flattening density. Only the `path` arm is corpus-covered (`blend-nodes/rounded-paths`);
/// if a real canvas surfaces a divergence in the others the generators get ported then.
pub(crate) fn blend_shape_to_abs_points(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
) -> Option<Vec<moda_scene::blend::BlendPathPoint>> {
    let apply_absolute = |pts: Vec<moda_scene::blend::BlendPathPoint>| -> Option<Vec<moda_scene::blend::BlendPathPoint>> {
        // The node's full canvas affine (page offset included): rotation, shear and nested scale.
        let affine = scene.node_canvas_affine(node.id())?;
        Some(pts.into_iter().map(|p| p.apply_affine(affine)).collect())
    };

    match node.node_type() {
        // The TS `path` arm short-circuits to `computePathRenderAbsolutePoints` — the
        // DRAWN (rounded) outline under the node's FULL canvas affine, NOT the
        // TRS + page-offset seam `apply_absolute` implements for the other arms. It calls
        // the same helper the path-mode lane uses so the two cannot drift.
        "path" => moda_scene::blend::path_absolute_points(scene, node),
        "line" => {
            // convertLineBodyToPathNode({...ln, dash: undefined}) — body
            // centerline only, dashes stripped (the 'line' arm of
            // `shapeNodeToNodeLocalPathPoints` in core/utils/shapeToPath.ts,
            // whose default 'body-only' mode strips the dash).
            let mut spec = tess::line_to_path::LineSpec::from_json(|k| node.json_field(k));
            spec.dash = None;
            let part = tess::line_to_path::convert_line_body(&spec);
            apply_absolute(path_cmds_to_blend_points(&part.cmds))
        }
        "rectangle" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            let cmds = shape_to_path::rect_outline_cmds(w, h, node.json_field("cornerRadius"), node.json_field("cornerSmoothing"), node.json_field("cornerStyle"));
            apply_absolute(path_cmds_to_blend_points(&cmds))
        }
        "ellipse" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            let cmds = shape_to_path::ellipse_to_path_cmds(
                w,
                h,
                node.f64("startAngle"),
                node.f64("sweepAngle"),
                node.f64("innerRadiusRatio"),
            );
            apply_absolute(path_cmds_to_blend_points(&cmds))
        }
        "polygon" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            let cr = node
                .f64("cornerRadius")
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            let cmds = tess::shape_paths::polygon_to_path_cmds(
                w,
                h,
                node.f64("sides").unwrap_or(f64::NAN),
                cr,
            );
            apply_absolute(path_cmds_to_blend_points(&cmds))
        }
        "star" => {
            let w = node.f64("width").unwrap_or(0.0);
            let h = node.f64("height").unwrap_or(0.0);
            let cr = node
                .f64("cornerRadius")
                .filter(|v| v.is_finite())
                .unwrap_or(0.0);
            let cmds = tess::shape_paths::star_to_path_cmds(
                w,
                h,
                node.f64("numPoints").unwrap_or(f64::NAN),
                node.f64("innerRadiusRatio"),
                cr,
            );
            apply_absolute(path_cmds_to_blend_points(&cmds))
        }
        _ => None,
    }
}

/// The blend arm of the plan builder: draws a blend node's steps (`moda_scene::blend`, back to front, each at its own
/// z) and returns the z slots it consumed (one per step; one when it draws nothing).
#[allow(clippy::too_many_arguments)]
fn build_blend_items(
    scene: &Scene,
    node: &moda_scene::types::RawNode,
    this_z: u64,
    page_clip: [f32; 4],
    items: &mut Vec<DrawItem>,
    warnings: &mut PlanWarnings,
    shader_time_s: f64,
    mask: Option<crate::seams::MaskRef>,
    mut caches: Option<CacheCtx<'_>>,
) -> u64 {
    use moda_scene::blend::{expand_blend_node, BlendExpansion};

    match expand_blend_node(scene, node, &|n| blend_shape_to_abs_points(scene, n)) {
        BlendExpansion::Skip(reason) => {
            warnings
                .skipped
                .push(format!("{} (blend): {reason}", node.id()));
            1
        }
        BlendExpansion::Paths { page_index, nodes } => {
            // Objects on a page draw their steps under that page's clip; objects on the canvas (canvas roots,
            // frame children in a page-less Design document) under the blend node's own clip, which a
            // clipping frame narrows. A disjoint clip draws nothing.
            let (clip, (px, py)) = match page_index {
                Some(pi) => (blend_page_clip(scene, pi), scene.page_position(pi)),
                None => match crate::effects::effective_page_clip(scene, node.id(), page_clip) {
                    Some(clip) => (clip, (0.0, 0.0)),
                    None => return (nodes.len() as u64).max(1),
                },
            };
            let origin = moda_scene::transform::translate(px, py);
            for (i, synth) in nodes.iter().enumerate() {
                match build_body_item(
                    scene,
                    synth,
                    (this_z + i as u64) as f64,
                    clip,
                    mask.clone(),
                    // Synthetic blend steps never take the SDF fast path.
                    true,
                    shader_time_s,
                    Some(origin),
                    None,
                    caches.as_mut().map(|(c, h)| (&mut **c, &**h)),
                    warnings,
                ) {
                    Ok(bodies) => items.extend(bodies),
                    Err(e) => warnings
                        .skipped
                        .push(format!("{} (blend step): {e}", synth.id())),
                }
            }
            (nodes.len() as u64).max(1)
        }
    }
}

trait ZeroedBox {
    fn zeroed_box() -> Box<PaintInstance>;
}

impl ZeroedBox for PaintInstance {
    fn zeroed_box() -> Box<PaintInstance> {
        Box::new(bytemuck::Zeroable::zeroed())
    }
}

// ---------------------------------------------------------------------------
// GPU execution
// ---------------------------------------------------------------------------

struct Layouts {
    frame_bgl: wgpu::BindGroupLayout,
    storage_bgl: wgpu::BindGroupLayout,
    mask_bgl: wgpu::BindGroupLayout,
    path_layout: wgpu::PipelineLayout,
    page_layout: wgpu::PipelineLayout,
    /// M3b: pattern texture BGL (manifest BGL 15: sampler + texture).
    pattern_texture_bgl: wgpu::BindGroupLayout,
    /// M3b: path-pattern pipeline layout (manifest layout 16:
    /// [frame, storage, texture, mask] — mask rides group 3).
    pattern_layout: wgpu::PipelineLayout,
    /// M4e: shader-image BGL (manifest BGL 42 "Custom Shader Image Resource
    /// BGL": texture@0, sampler@1 — NOTE the reversed order vs patterns).
    shader_image_bgl: wgpu::BindGroupLayout,
    /// M4e: image-backed procedural-fill layout (manifest layout 43:
    /// [frame, storage, image, mask] — mask rides group 3).
    image_shader_layout: wgpu::PipelineLayout,
}

pub struct Engine {
    pub gpu: GpuContext,
    /// The IMMUTABLE per-engine frame color format — the format of
    /// every frame color attachment (frame/MSAA targets, main-pass +
    /// text/SDF/mask-composite/backdrop/layer-blur pipelines). `Rgba8Unorm` by
    /// default (native/CLI/export/server and the surfaceless wasm export
    /// engine); the live DOM wasm engine constructs with the browser's
    /// preferred canvas format (`Bgra8Unorm` on most desktops) so a camera
    /// frame can MSAA-resolve/render DIRECTLY into the preferred-format
    /// swapchain — no offscreen render pass + fullscreen present blit + second
    /// submit (the hot-path cost this restores). The offscreen target
    /// itself is still allocated/cached by `frame_targets` on the direct path
    /// (unused as a destination); dropping that retained VRAM is a separate
    /// optimization.
    ///
    /// Immutable by design: pipeline/target caches are keyed WITHOUT a format
    /// dimension, which is only sound because a given engine can never hold
    /// two frame formats. Sampled SOURCE textures (decoded assets, emoji,
    /// atlases, patterns) stay RGBA regardless — sampling is channel-order
    /// neutral. Never an `-srgb` format: `Rgba8Unorm`/`Bgra8Unorm` preserve
    /// the gamma-space blending contract (render-semantics §7).
    frame_format: wgpu::TextureFormat,
    /// The encoding this engine blends in when it is not sRGB (`moda_shaders::working_color`): every module that
    /// puts an authored colour into the frame is compiled with its conversion. Immutable, like the frame format:
    /// the shader module and pipeline caches carry no encoding.
    working: Option<moda_shaders::WorkingColor>,
    layouts: Layouts,
    /// (module id, pipeline label index, sample_count, blend key, color target format) → pipeline.
    pipelines: std::cell::RefCell<std::sync::Arc<PipelineMap>>,
    shader_modules: std::cell::RefCell<std::collections::HashMap<String, wgpu::ShaderModule>>,
    inactive_mask: Option<wgpu::BindGroup>,
    /// M3a (SEAMS change log): text pipelines + atlas/emoji texture caches.
    text: crate::text_draw::TextDraw,
    /// Task #95 R2: retained-mode GPU frame residency. OFF by default — the
    /// CLI/export lane keeps the exact Stage-0 per-render allocations; the
    /// wasm editor lane (and the resident_smoke test) turns it on.
    retained: bool,
    /// False when the caller may defer reading an output until after the next render.
    reuse_frame_targets: bool,
    /// Pattern (image) textures persist across renders as retained mode keeps them (`retained_patterns`, the same
    /// keys and budget) though the engine is otherwise per-render: the float capture twin
    /// ([`Engine::float_capture`]), which draws a graph export's pixel captures, uploads a frame's images once an
    /// export, not once a capture.
    keep_patterns: bool,
    /// R2a/b/c: frame uniform + bind group, render targets keyed by
    /// (w, h, samples), and the grow-only per-class buffer pool
    /// (`WebGPUStorageBufferManager` semantics: capacity retained, contents
    /// immediate; bind groups re-created only when the buffer is replaced).
    retained_frame: std::cell::RefCell<RetainedFrame>,
    /// R2d: pattern textures persistent across renders; keys carry settle
    /// state + decoded-pixels identity so Loading→Loaded flips
    /// self-invalidate (the browser's bind-group-key trick, MAP §5.7).
    retained_patterns: std::cell::RefCell<Option<paint::draw::PatternTextureCache>>,
    retained_shader_images: std::cell::RefCell<Option<ShaderImageTextureCache>>,
    /// Retained mask coverage (pipeline, full-rect sharing, textures of unchanged
    /// masks) — see `effects::MaskCache`. Retained mode only.
    retained_masks: std::cell::RefCell<Option<crate::effects::MaskCache>>,
    /// False until the first `render_plan` readback completes. The cold first
    /// render compiles its pipelines inside the fence wait; on watchdog-
    /// interrupting drivers that can exhaust the steady-state poll budget and
    /// surface a spurious load failure (succeeds on a warm retry). The first
    /// readback therefore gets a larger timeout-retry budget; this flag flips
    /// it back to steady state afterward.
    first_readback_done: std::cell::Cell<bool>,
    /// The pixel graph's float capture engine (motion-model.md §10.1): the same device, frame format
    /// `Rgba16Float`, built on first use by [`Engine::float_capture`]. The same gamma-space blend math into a float
    /// target: more precision, the same look.
    float_twin: std::cell::OnceCell<Box<Engine>>,
    /// A non-retained engine's MSAA colour and depth attachments, kept by `(w, h, samples)` between renders: every
    /// render clears them first (retained mode reuses them the same way), so only the resolve target, which the
    /// caller keeps, needs to be new. Allocating them was most of a float capture's cost: a 1080p capture at MSAA ×4
    /// is 100 MB of fresh memory for the OS to zero. Bounded by [`SCRATCH_ATTACHMENT_BYTES`].
    scratch_attachments: std::cell::RefCell<Vec<ScratchAttachments>>,
    /// Cumulative debug counters for the prepared-frame path, flat-merged into
    /// the wasm host's `residency_stats()` JSON so an A/B run can confirm the
    /// fast path engaged (hits rising, upload/text bytes plateauing).
    prepared_stats: std::cell::Cell<PreparedFrameStats>,
    /// M2 prepared-frame packet: the reusable text + identity for the last
    /// resident camera frame. Present only while the fast path is engaged
    /// (prepared-frame + retained on, non-backdrop). A render whose plan
    /// identity, target params, and pool epoch all match skips the per-frame
    /// text prepare and geometry uploads; anything else rebuilds it.
    prepared_frame: std::cell::RefCell<Option<PreparedFrame>>,
    /// What a frame let go of that is slow to free (a replaced packet and the whole-document plan it pinned: ~8 ms
    /// at 20k layers), dropped by [`Engine::drop_deferred`] when the host is idle instead of inside the frame.
    deferred_drops: std::cell::RefCell<Vec<Box<dyn std::any::Any>>>,
    /// Draw background-blur composites inside the frame's next pass (default on; off for an A/B or a test,
    /// `set_backdrop_inline`).
    backdrop_inline: std::cell::Cell<bool>,
    /// The last camera frame's prepared text when no packet kept it (a plan with blend or backdrop splits is
    /// never packeted: its split geometry follows the camera, so each pan frame is another plan). The next such
    /// frame keeps every chunk whose batches are the same `Arc`s, as a packet rebuild does (`TextCarry`).
    retained_text: std::cell::RefCell<Option<TextCarry>>,
    /// M2: monotonic epoch of the retained geometry pool — bumped on every
    /// `make_buffer` upload/growth. A packet records the epoch it was built at;
    /// a mismatch means another render (page/export/miss) overwrote or grew a
    /// shared pool buffer since, so the reuse is suspect and the packet rebuilds.
    pool_epoch: std::cell::Cell<u64>,
    /// M2: set true only for a proven prepared-frame HIT, read by `make_buffer`
    /// to skip re-uploading byte-identical geometry into the (unchanged) retained
    /// pool. Reset at the top of every render so non-camera/miss paths upload
    /// normally.
    pf_skip_uploads: std::cell::Cell<bool>,
    /// The largest vertex-stream buffer a frame uploads (bytes); `None`: the device's buffer size limit.
    vertex_chunk_bytes: std::cell::Cell<Option<u64>>,
    /// How many buffers the last frame's (opaque, transparent) vertex streams took.
    vertex_chunks: std::cell::Cell<[usize; 2]>,
    /// The (opaque, transparent) vertex streams' residency ([`PhaseResidency`]); `None`: packed in plan order.
    vertex_residency: std::cell::RefCell<[Option<PhaseResidency>; 2]>,
    /// Vertex residency on (default; off for A/Bs and tests of the packed-afresh path).
    vertex_residency_on: std::cell::Cell<bool>,
    /// Vertex residency for retained offscreen frames too (a host that draws every frame of a clip).
    offscreen_residency: std::cell::Cell<bool>,
    /// Background blurs narrower than this (their blur reach on screen, device px) are drawn plainly on the next
    /// live camera present ([`Engine::set_motion_blur_min_px`]); 0 = never. Only while the canvas moves.
    motion_blur_min_px: std::cell::Cell<f64>,
    /// Whether consecutive transparent solid bodies that draw the same merge into one draw (`SolidRun`); on by
    /// default, off only to compare against (the pixels are the same).
    solid_run_merging: std::cell::Cell<bool>,
    /// Why the last prepared-frame patch fell back (`Engine::patch_prepared`).
    patch_fallback_reason: std::cell::Cell<&'static str>,
    /// Why the last live camera present could not use a prepared frame (diagnostics), or `""`.
    prepared_ineligible_reason: std::cell::Cell<&'static str>,
    /// Prepared-frame patching on (default) or off (an A/B; `set_prepared_patch`).
    prepared_patch: std::cell::Cell<bool>,
    /// Vector text under this many device px per em draws as word bars on the next live camera present
    /// ([`Engine::set_motion_text_greek_px`]); 0 = never. Only while the canvas moves.
    motion_text_greek_px: std::cell::Cell<f64>,
    /// Greeked copies of text batches, by the batch `Arc` they were made from (`None`: can't be greeked).
    greek_cache: std::cell::RefCell<
        std::collections::HashMap<
            usize,
            (
                std::sync::Weak<crate::seams::TextQuadBatch>,
                Option<std::sync::Arc<crate::seams::TextQuadBatch>>,
            ),
        >,
    >,
    /// Text layers the last live present drew as word bars.
    greeked_text: std::cell::Cell<u32>,
    /// Blurs the last live present drew plainly (the host renders an exact frame once motion stops).
    simplified_blurs: std::cell::Cell<u32>,
    /// Layer-capture results (layer filters, capture shadows) kept across
    /// frames under their content key — retained mode only
    /// (`layer_blur::cache`).
    capture_cache: std::cell::RefCell<crate::layer_blur::cache::CaptureCache>,
    /// New layer captures a live frame renders before earlier zoom steps'
    /// results stand in for the rest (`layer_blur::cache` stand-ins); `None` =
    /// render every capture the frame needs (exports, the default).
    capture_frame_budget: std::cell::Cell<Option<u32>>,
    /// The last frame composited a stand-in: the host should render another.
    captures_pending: std::cell::Cell<bool>,
    /// This live frame is mid-zoom: an earlier zoom step's result that covers what the target shows of a capture
    /// stands in for it (`set_capture_stand_ins_mid_zoom`).
    capture_stand_ins_mid_zoom: std::cell::Cell<bool>,
    /// Depth of the capture pre-pass recursion (a capture's own mini-plan).
    capture_depth: std::cell::Cell<u32>,
    /// The buffers and textures a render makes and drops, reused once the GPU is done with them, and the bind groups
    /// over them (`scratch`): a non-retained render's own, and every capture's.
    scratch: crate::scratch::ScratchPool,
    /// The part of the next render's target (target px) its output is read on, and that part in canvas units: set by
    /// a layer capture for its own render, taken when that render starts (`capture_clip`).
    capture_scissor: std::cell::Cell<Option<[u32; 4]>>,
    capture_window: std::cell::Cell<Option<[f64; 4]>>,
    /// The layer-capture pipelines, built once per main-pass sample count
    /// (they hold no per-frame state).
    layer_blur_renderers: std::cell::RefCell<
        std::collections::HashMap<u32, std::sync::Arc<crate::layer_blur::LayerBlurRenderer>>,
    >,
    /// The backdrop-effect and blend-split pipelines, built once per main-pass sample count (they hold no
    /// per-frame state but the blend splits' saved-backdrop stack, which every frame empties). A frame with a
    /// split used to build them all again: a dozen shader modules and pipelines on every frame of a page with a
    /// blend mode.
    backdrop_renderers: std::cell::RefCell<
        std::collections::HashMap<u32, std::sync::Arc<crate::backdrop::BackdropRenderer>>,
    >,
    /// Shade procedural fills once per pixel in a single-sample pre-pass on the live camera lane under
    /// MSAA ([`Engine::set_shader_prepass`]). Off by default.
    shader_prepass: bool,
    /// The pre-pass targets (target-sized colour slots + one depth), kept while the size holds.
    prepass_pool: std::cell::RefCell<Option<ShaderPrepassPool>>,
    /// Fills the last render shaded in the pre-pass ([`Engine::shader_prepass_fills`]).
    prepass_fills: std::cell::Cell<usize>,
}

/// Module id of the engine-owned composite that draws a pre-passed shader fill
/// (`wgsl/shader-fill-composite.wgsl`).
const SHADER_FILL_COMPOSITE: &str = "shader-fill-composite";
/// Colour targets one frame's shader pre-pass may use: fills whose on-screen boxes don't overlap share one;
/// a fill that finds none of these free draws directly (per sample) as before.
const SHADER_PREPASS_MAX_SLOTS: usize = 3;

/// Retained targets of the shader-fill pre-pass: target-sized single-sample colour textures in the frame
/// format and a depth texture, for one target size.
struct ShaderPrepassPool {
    width: u32,
    height: u32,
    slots: Vec<wgpu::Texture>,
    depth: wgpu::Texture,
    sampler: wgpu::Sampler,
}

/// One pre-passed shader fill: its module, its instance with fill alpha 1 and no page clip, the local quad
/// (two triangles over its bounds, padded when needed) and the pixel rect it may write.
struct ShaderPrepassJob {
    slot: usize,
    module: String,
    instance: PaintInstance,
    quad: [f32; 12],
    rect: [u32; 4],
}

fn quad_covers_target(o: (f64, f64), ax: (f64, f64), ay: (f64, f64), w: f64, h: f64) -> bool {
    let dx = (ax.0 - o.0, ax.1 - o.1);
    let dy = (ay.0 - o.0, ay.1 - o.1);
    let det = dx.0 * dy.1 - dx.1 * dy.0;
    if !det.is_finite() || det == 0.0 || w < 1.0 || h < 1.0 {
        return false;
    }
    // Cover the whole viewport, leaving a half-texel margin around every sample center.
    // A near-center boundary is not enough: top-left rasterization can exclude that texel.
    [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
        .iter()
        .all(|&(x, y)| {
            let p = (x - o.0, y - o.1);
            let u = (p.0 * dy.1 - p.1 * dy.0) / det;
            let v = (dx.0 * p.1 - dx.1 * p.0) / det;
            u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0
        })
}

/// The pixel rect and local quad a fill shades before compositing. Keep two pixels of padding for
/// partial coverage; an axis-aligned affine quad covering the whole viewport needs no expanded geometry.
/// `None` when it covers nothing on screen or its transform is degenerate.
fn shader_prepass_region(
    inst: &PaintInstance,
    vp: &[f32; 16],
    tw: f64,
    th: f64,
) -> Option<([u32; 4], [f32; 12])> {
    let b = inst.bounds.map(f64::from);
    if !(b[2] > 0.0 && b[3] > 0.0) {
        return None;
    }
    let m = inst.model_matrix.map(f64::from);
    let v = vp.map(f64::from);
    let world = |x: f64, y: f64| (m[0] * x + m[4] * y + m[12], m[1] * x + m[5] * y + m[13]);
    let pixel = |(wx, wy): (f64, f64)| {
        let cx = v[0] * wx + v[4] * wy + v[12];
        let cy = v[1] * wx + v[5] * wy + v[13];
        ((cx + 1.0) * 0.5 * tw, (1.0 - cy) * 0.5 * th)
    };
    let local = |x: f64, y: f64| pixel(world(x, y));
    let (x0, y0, x1, y1) = (b[0], b[1], b[0] + b[2], b[1] + b[3]);
    let o = local(x0, y0);
    let ax = local(x1, y0);
    let ay = local(x0, y1);
    let sx = (ax.0 - o.0).hypot(ax.1 - o.1) / b[2];
    let sy = (ay.0 - o.0).hypot(ay.1 - o.1) / b[3];
    let s = sx.min(sy);
    if !(s.is_finite() && s > 1e-9) {
        return None;
    }
    // The composite textureLoads this target's texels. If the original quad already covers the whole
    // viewport, padding cannot supply another required texel. Keep the original interpolation
    // geometry there: expanding it can change float interpolation in a spatial shader. Else retain
    // the padding needed by partially covered MSAA edge pixels whose centers are outside the body.
    let axis_aligned_affine = m[3] == 0.0 && m[7] == 0.0 && m[15] == 1.0
        && v[3] == 0.0 && v[7] == 0.0 && v[11] == 0.0 && v[15] == 1.0
        && v[8] == 0.0 && v[9] == 0.0
        && m[1] == 0.0 && m[4] == 0.0 && v[1] == 0.0 && v[4] == 0.0;
    let pad = if axis_aligned_affine && quad_covers_target(o, ax, ay, tw, th) { 0.0 } else { 2.0 / s };
    let (qx0, qy0, qx1, qy1) = (x0 - pad, y0 - pad, x1 + pad, y1 + pad);
    let mut r = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for (x, y) in [local(qx0, qy0), local(qx1, qy0), local(qx0, qy1), local(qx1, qy1)] {
        r = [r[0].min(x), r[1].min(y), r[2].max(x), r[3].max(y)];
    }
    // The page clip is a canvas-space box; the view-projection is axis-aligned, so its corners bound it.
    let c = inst.page_clip_bounds.map(f64::from);
    let (c0, c1) = (pixel((c[0], c[1])), pixel((c[0] + c[2], c[1] + c[3])));
    let clip = [
        c0.0.min(c1.0) - 2.0,
        c0.1.min(c1.1) - 2.0,
        c0.0.max(c1.0) + 2.0,
        c0.1.max(c1.1) + 2.0,
    ];
    let rect = [
        r[0].max(clip[0]).max(0.0).floor(),
        r[1].max(clip[1]).max(0.0).floor(),
        r[2].min(clip[2]).min(tw).ceil(),
        r[3].min(clip[3]).min(th).ceil(),
    ];
    if !rect.iter().all(|v| v.is_finite()) || rect[2] <= rect[0] || rect[3] <= rect[1] {
        return None;
    }
    let q = [qx0, qy0, qx1, qy0, qx0, qy1, qx1, qy0, qx1, qy1, qx0, qy1].map(|v| v as f32);
    Some((rect.map(|v| v as u32), q))
}

/// M2 prepared-frame packet ("prepare once per plan, replay per camera frame").
/// Holds the reusable text GPU resources plus the identity/target keys that
/// prove a later frame is the SAME draw, only the camera differing. The geometry
/// buffers themselves live in the retained pool (reused in place, guarded by
/// `pool_epoch`); only the camera/frame uniform is rewritten per frame.
struct PreparedFrame {
    /// The plan this packet was built for.
    key: PlanKey,
    /// A plan shared as an `Arc`, pinned: its address is the key, and holding it keeps that address from
    /// naming another plan (a freed-then-reused allocation can never alias while the packet stands). It is
    /// also what a later plan of that lane is compared with. `None` for an owned plan.
    pinned: Option<std::sync::Arc<CoveragePlan>>,
    /// Target params the packet was built for — a change (viewport resize, DPR,
    /// MSAA) is NOT a camera change and must rebuild (the buffers/pipelines or
    /// attachment sizes differ). Camera pose is deliberately excluded.
    width: u32,
    height: u32,
    sample_count: u32,
    /// Whether the pass cleared opaque: with MSAA it decides which phase an
    /// opaque SDF body's instance was uploaded into (see `sdf_opaque_lane`),
    /// so a clear-opacity change must re-encode, not replay those buffers.
    clear_is_opaque: bool,
    /// Live views pack masks into an atlas; offscreen frames keep exact coverage.
    live_masks: bool,
    /// Retained-pool epoch at build time (see `Engine::pool_epoch`).
    pool_epoch: u64,
    /// The live shader clock the packet's uploaded instance buffers
    /// hold. A packet hit reuses those buffers in place (skips the upload), but a
    /// re-threaded procedural-fill clock lives in per-instance `shader_params`
    /// (NOT the frame uniform), so a clock change on a clock-bearing plan must
    /// MISS and re-encode — otherwise the animated shader would freeze at this
    /// clock. Camera-only frames keep the same clock and still hit. Irrelevant
    /// (never compared) when `has_clock_fill` is false.
    shader_time_s: f64,
    /// Whether any procedural-fill body re-threads the clock
    /// (`clock_offsets` non-empty). Only such plans gate the hit on `shader_time_s`
    /// — a static/clock-free plan ignores the clock so M2 still hits during
    /// "playback" of nothing.
    has_clock_fill: bool,
    /// Reused text arenas + bind groups (camera-independent; the camera lives in
    /// the shared frame uniform the text bind groups reference, rewritten per
    /// frame).
    text: std::sync::Arc<TextChunks>,
    /// The text batches `text` was prepared from, in draw order, and the text host generation then. A frame of
    /// another plan reuses each chunk whose batches are the same `Arc`s (an edit re-plans a few layers: a batch that
    /// changed went through `Arc::make_mut`, or was built again, and is a new one). Holding the `Arc`s keeps the
    /// pointers from being reused.
    text_batches: Vec<std::sync::Arc<crate::seams::TextQuadBatch>>,
    text_host_generation: u64,
    /// Which text batches the packet draws greeked (`Engine::greek_selection`; 0: none).
    text_greek_sig: u64,
    /// How many background blurs the packet's frame drew plainly while the canvas moved
    /// (`Engine::set_motion_blur_min_px`): a hit draws the same splits.
    simplified_blurs: u32,
    /// The motion threshold those were drawn plainly under: a recorded pass (which leaves them out) replays only
    /// under the same one.
    motion_blur_px: f64,
    /// The frame's main pass, recorded once as a render bundle when it can be (one segment, no viewport, no layer
    /// captures or shader pre-pass, every mask texture retained): a hit replays it (`execute_bundles`) instead of
    /// collecting and recording every draw again. It references this packet's buffers and bind groups, which a hit
    /// leaves unchanged; only the camera uniform is rewritten.
    bundle: Option<wgpu::RenderBundle>,
    /// The layer captures (shadows, layer filters) the packet drew: which of the plan's composites were on screen,
    /// and their results. A frame of the same plan with the same composites on screen draws these results again
    /// without looking them up or rendering them (every one is content-keyed, so the same key is the same pixels),
    /// and its recorded pass (which binds them) replays. `None`: the plan has no captures.
    captures: Option<PacketCaptures>,
    /// The instance records the packet's storage buffers hold, and where each of the plan's items has its own.
    records: PacketRecords,
}

/// Which plan a prepared packet is of: the plan's owner (unique in the process, from 1) and the owner's count of
/// the plan's states. An owner writes its plan in place and moves the revision; nothing else names a state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanKey {
    pub owner: u64,
    pub revision: u64,
}

/// What its owner did to a plan between revision `from` and the one being drawn.
pub struct PlanChange<'a> {
    pub from: u64,
    pub items: ChangedItems<'a>,
}

/// The items of a plan that differ from an earlier state of it.
pub enum ChangedItems<'a> {
    /// Items written in place, by index. `placed`: bodies whose values changed and whose draw did not (their
    /// kind, phase, pipeline, geometry, z, blend and mask stand: a placement recipe's promise). `replaced`:
    /// items prepared again, each with the item it took the place of.
    Written { placed: &'a [u32], replaced: &'a [(u32, DrawItem)] },
    /// Another plan took the earlier one's place: this is the earlier one, and any item may differ.
    Whole(&'a CoveragePlan),
}

/// The plan a frame draws, as its packet knows it. A plan shared as an `Arc` is named by its address, under
/// owner 0, and the packet pins it (so the address names nothing else while the packet stands); the packet
/// then also holds the earlier state a later plan is compared with. An owned plan is named by its owner's key
/// and brings its own account of what changed.
#[derive(Clone, Copy)]
struct PacketPlan<'a> {
    key: PlanKey,
    pinned: Option<&'a std::sync::Arc<CoveragePlan>>,
    change: Option<&'a PlanChange<'a>>,
}

impl<'a> PacketPlan<'a> {
    fn pinned(plan: &'a std::sync::Arc<CoveragePlan>) -> Self {
        let key = PlanKey { owner: 0, revision: std::sync::Arc::as_ptr(plan) as usize as u64 };
        Self { key, pinned: Some(plan), change: None }
    }
}

/// One storage buffer of a packet as the CPU holds it: the records its collect uploaded, kept so a patch compares
/// a changed item's record with the one the buffer holds and writes it where it stands. Record `i` is the
/// buffer's bytes from `i * size_of::<T>()`.
struct Column<T> {
    records: Vec<T>,
    /// Records a patch asked for and has not committed, by index (`Column::put`).
    pending: Vec<(u32, T)>,
}

impl<T> Default for Column<T> {
    fn default() -> Self {
        Self { records: Vec::new(), pending: Vec::new() }
    }
}

impl<T: bytemuck::Pod> Column<T> {
    fn of(records: Vec<T>) -> Self {
        Self { records, pending: Vec::new() }
    }

    /// Ask for `record` at `index`. `Ok(true)`: it differs from the one held and is pending until
    /// [`Column::commit`]; `Ok(false)`: the buffer already holds it. `Err`: the column has no such record.
    fn put(&mut self, index: u32, record: &T) -> std::result::Result<bool, &'static str> {
        let held = self.records.get(index as usize).ok_or("record")?;
        if bytemuck::bytes_of(held) == bytemuck::bytes_of(record) {
            return Ok(false);
        }
        self.pending.push((index, *record));
        Ok(true)
    }
}

/// A [`Column`] without its record type, for what is done to every column of a packet alike.
trait PendingRecords {
    /// The bytes the column's buffer holds.
    fn bytes(&self) -> u64;
    fn is_pending(&self) -> bool;
    /// Forget what was asked for.
    fn discard(&mut self);
    /// Take the pending records into the column and hand `upload` the column from its first changed record to
    /// its last, as (byte offset in the buffer, bytes): one write for a column, however its changed records lie.
    fn commit(&mut self, upload: &mut dyn FnMut(u64, &[u8]));
}

impl<T: bytemuck::Pod> PendingRecords for Column<T> {
    fn bytes(&self) -> u64 {
        std::mem::size_of_val(self.records.as_slice()) as u64
    }
    fn is_pending(&self) -> bool {
        !self.pending.is_empty()
    }
    fn discard(&mut self) {
        self.pending.clear();
    }
    fn commit(&mut self, upload: &mut dyn FnMut(u64, &[u8])) {
        let mut span: Option<(usize, usize)> = None;
        for (index, record) in self.pending.drain(..) {
            let index = index as usize;
            self.records[index] = record;
            span = Some(span.map_or((index, index + 1), |(from, to)| (from.min(index), to.max(index + 1))));
        }
        if let Some((from, to)) = span {
            let size = std::mem::size_of::<T>() as u64;
            upload(from as u64 * size, bytemuck::cast_slice(&self.records[from..to]));
        }
    }
}

/// What a packet's state is compared with to reach a new one (`Engine::patch_prepared`).
#[derive(Clone, Copy)]
enum PatchBase<'a> {
    /// Its owner's account of the items written in place (`ChangedItems::Written`).
    Written { placed: &'a [u32], replaced: &'a [(u32, DrawItem)] },
    /// The plan the packet was of: every item is compared.
    Whole(&'a CoveragePlan),
}

/// Where the collect put one plan item's record: its index in its column, and (a body) its first vertex in its
/// phase's vertex streams. Which column is the item's own fact (its kind, phase and paint family), and a patch
/// keeps it.
#[derive(Clone, Copy)]
struct RecordAt {
    index: u32,
    first_vertex: u32,
}

impl RecordAt {
    /// An item with no record of its own (text, a capture's composite, a background effect).
    const NONE: Self = Self { index: u32::MAX, first_vertex: 0 };
}

/// A packet's records: one column per storage buffer (by phase, 0 opaque and 1 transparent, where the buffer is
/// per phase; the SDF columns by lane, `sdf_opaque_lane`), and each plan item's place in its column. Made by the
/// collect that built the packet; a patch keeps every item's place.
#[derive(Default)]
struct PacketRecords {
    at: Vec<RecordAt>,
    page: Column<PageInstance>,
    solid: [Column<SolidPathInstance>; 2],
    gradient: [Column<paint::GradientInstance>; 2],
    pattern: [Column<paint::PatternInstance>; 2],
    shader: [Column<PaintInstance>; 2],
    shadow: Column<crate::seams::ShadowInstance>,
    solid_sdf: [Column<paint::shape_sdf::ShapeSolidSdfInstance>; 2],
    gradient_sdf: [Column<paint::shape_sdf::ShapeGradientSdfInstance>; 2],
}

impl PacketRecords {
    /// Ask for a body's packed record at `index` of its family's column in phase `k` (`Column::put`).
    fn put_body(&mut self, k: usize, index: u32, packed: &PackedBody) -> std::result::Result<bool, &'static str> {
        match packed {
            PackedBody::Solid(record) => self.solid[k].put(index, record),
            PackedBody::Gradient(record, _) => self.gradient[k].put(index, record),
            PackedBody::Pattern(record, _) => self.pattern[k].put(index, record),
            PackedBody::Shader(record, _, _) => self.shader[k].put(index, record),
        }
    }

    /// Every column with the pooled buffer it mirrors.
    fn columns(&mut self) -> [(&'static str, &mut dyn PendingRecords); 14] {
        let [solid_opaque, solid_trans] = &mut self.solid;
        let [gradient_opaque, gradient_trans] = &mut self.gradient;
        let [pattern_opaque, pattern_trans] = &mut self.pattern;
        let [shader_opaque, shader_trans] = &mut self.shader;
        let [solid_sdf_opaque, solid_sdf_trans] = &mut self.solid_sdf;
        let [gradient_sdf_opaque, gradient_sdf_trans] = &mut self.gradient_sdf;
        [
            (PAGE_STORAGE, &mut self.page),
            (SOLID_STORAGE[0], solid_opaque),
            (SOLID_STORAGE[1], solid_trans),
            (GRADIENT_STORAGE[0], gradient_opaque),
            (GRADIENT_STORAGE[1], gradient_trans),
            (PATTERN_STORAGE[0], pattern_opaque),
            (PATTERN_STORAGE[1], pattern_trans),
            (SHADER_STORAGE[0], shader_opaque),
            (SHADER_STORAGE[1], shader_trans),
            (SHADOW_STORAGE, &mut self.shadow),
            (SOLID_SDF_STORAGE[0], solid_sdf_opaque),
            (SOLID_SDF_STORAGE[1], solid_sdf_trans),
            (GRADIENT_SDF_STORAGE[0], gradient_sdf_opaque),
            (GRADIENT_SDF_STORAGE[1], gradient_sdf_trans),
        ]
    }
}

/// The pooled storage buffers of a frame's instance records: per phase or SDF lane, opaque then transparent.
const PAGE_STORAGE: &str = "Page Storage Buffer";
const SOLID_STORAGE: [&str; 2] = ["Path Storage (Opaque)", "Path Storage (Transparent)"];
const GRADIENT_STORAGE: [&str; 2] = ["Path Gradient Storage (Opaque)", "Path Gradient Storage (Transparent)"];
const PATTERN_STORAGE: [&str; 2] = ["Path Pattern Storage (Opaque)", "Path Pattern Storage (Transparent)"];
const SHADER_STORAGE: [&str; 2] = ["Custom Shader Storage (Opaque)", "Custom Shader Storage (Transparent)"];
const SHADOW_STORAGE: &str = "Shadow Storage (Transparent)";
const SOLID_SDF_STORAGE: [&str; 2] = ["Shape SDF Solid Storage (Opaque)", "Shape SDF Solid Storage (Transparent)"];
const GRADIENT_SDF_STORAGE: [&str; 2] =
    ["Shape SDF Gradient Storage (Opaque)", "Shape SDF Gradient Storage (Transparent)"];

/// Text batches per prepared chunk. An edit that changes a few text layers prepares their chunks again and keeps the
/// rest (at 20k layers, all ~5,700 text batches took 19 ms to prepare).
const TEXT_CHUNK: usize = 256;

/// A frame's prepared text, in chunks of [`TEXT_CHUNK`] batches (batch `i` is chunk `i / TEXT_CHUNK`'s
/// `i % TEXT_CHUNK`).
#[derive(Default)]
struct TextChunks {
    chunks: Vec<std::sync::Arc<crate::text_draw::PreparedText>>,
}

/// A prepared frame's layer captures (`PreparedFrame::captures`).
struct PacketCaptures {
    on_screen: Vec<bool>,
    results: std::collections::HashMap<String, wgpu::TextureView>,
}

/// Vertex residency: which body geometries a phase's pooled vertex streams hold, and where. A live camera frame
/// whose plan changed packs only the geometries its streams don't hold yet, appended after the ones they do (an
/// edit that adds or removes a body, a painting filling in, a recolour and its undo), instead of packing and
/// uploading every vertex again; draws address each geometry at its resident offset. Geometries no longer drawn
/// leave holes, and the streams are packed afresh once they are half holes, when they would outgrow their
/// buffers, or when anything else wrote those buffers.
#[derive(Default)]
struct PhaseResidency {
    /// (Geometry Arc address, owner words or UV lane) → resident range and weak identity. A solid's vertices
    /// name its body's record slot, which is stable for the body's life, so inserting or removing another body
    /// leaves them as they are. A geometry several solid bodies share is held once under INSTANCE_OWNER and the
    /// draw names each body's slot.
    at: std::collections::HashMap<(usize, Option<u32>), (u32, u32, std::sync::Weak<TessellatedGeometry>)>,
    /// Vertices the streams hold (live and holes).
    total: u32,
    /// The (positions, types, stroke UVs) pooled buffers' write counters after this residency's last write.
    writes: [u64; 3],
    /// Record slots of the solid bodies in this phase: draw id → the slots of the solid bodies that carry it,
    /// in plan order across the whole phase (an id can recur apart: transform copies, paint layers, a
    /// generated part named like an authored node). The solid storage buffer is indexed by slot.
    slots: crate::residency::HashMap<String, NodeSlots>,
    /// Counts [`PhaseResidency::assign_slots`] calls: which nodes the latest one drew.
    assigned: u64,
    /// Slots no body holds, and one past the highest slot ever given out.
    free: Vec<u32>,
    slot_count: u32,
    /// The geometries the last collect drew through INSTANCE_OWNER (shared by several solid bodies).
    shared: std::collections::HashSet<usize, rustc_hash::FxBuildHasher>,
}

/// One node's record slots in a phase, and how many of them the latest slot assignment drew.
#[derive(Default)]
struct NodeSlots {
    slots: Vec<u32>,
    drawn: usize,
    assigned: u64,
}

/// Per phase (0 opaque, 1 transparent), how many solid bodies of a plan draw each geometry (by Arc address).
/// A resident phase holds a geometry several solid bodies draw once, under INSTANCE_OWNER.
fn solid_geometry_uses(items: &[DrawItem]) -> [crate::residency::HashMap<usize, u32>; 2] {
    let mut uses: [crate::residency::HashMap<usize, u32>; 2] = Default::default();
    for item in items {
        if let DrawItem::Body { phase, geometry, paint: BodyPaint::Solid, .. } = item {
            *uses[usize::from(*phase == Phase::Transparent)]
                .entry(std::sync::Arc::as_ptr(geometry) as usize).or_insert(0) += 1;
        }
    }
    uses
}

impl PhaseResidency {
    /// Give every solid body of phase `k` in `items` its record slot, written to `item_slots` by item: the
    /// one the same occurrence of its draw id held, else a free one. Occurrences are counted in one pass. A body no longer drawn gives its slot back for a later collect.
    /// The same plan again changes nothing.
    fn assign_slots(&mut self, items: &[DrawItem], k: usize, item_slots: &mut [u32]) {
        self.assigned += 1;
        let assigned = self.assigned;
        for (at, item) in items.iter().enumerate() {
            let DrawItem::Body { phase, paint: BodyPaint::Solid, node_id, .. } = item else { continue };
            if usize::from(*phase == Phase::Transparent) != k { continue; }
            // One lookup for a node that already has slots (every node, on most frames).
            let node = match self.slots.get_mut(node_id.as_str()) {
                Some(node) => node,
                None => self.slots.entry(node_id.clone()).or_default(),
            };
            if node.assigned != assigned {
                node.assigned = assigned;
                node.drawn = 0;
            }
            let ordinal = node.drawn;
            while node.slots.len() <= ordinal {
                let slot = match self.free.pop() {
                    Some(slot) => slot,
                    None => {
                        self.slot_count += 1;
                        self.slot_count - 1
                    }
                };
                node.slots.push(slot);
            }
            item_slots[at] = node.slots[ordinal];
            node.drawn += 1;
        }
        let free = &mut self.free;
        self.slots.retain(|_, node| {
            let keep = if node.assigned == assigned { node.drawn } else { 0 };
            free.extend(node.slots.drain(keep..));
            keep > 0
        });
    }
    /// The resident offset of `g`, if it is held.
    fn offset_of(&self, g: &std::sync::Arc<TessellatedGeometry>, owner: Option<u32>) -> Option<u32> {
        let (o, n, w) = self.at.get(&(std::sync::Arc::as_ptr(g) as usize, owner))?;
        (*n == g.vertex_count && std::ptr::eq(w.as_ptr(), std::sync::Arc::as_ptr(g)) && w.strong_count() > 0)
            .then_some(*o)
    }
}

/// A phase's vertex streams as uploaded (`Engine::upload_vertex_streams`): chunk `k` holds the vertices from
/// `bases[k]` up to the next base, in its (positions, types, stroke UVs) buffers.
struct VertexStreams {
    bases: Vec<u32>,
    bufs: Vec<[wgpu::Buffer; 3]>,
}

impl VertexStreams {
    /// The chunk holding vertex `offset`.
    fn chunk(&self, offset: u32) -> usize {
        self.bases.partition_point(|&b| b <= offset).saturating_sub(1)
    }
}

/// The owner word of a resident geometry shared by several solid bodies: the shader reads the draw's
/// instance index instead. Every other solid vertex carries its body's record index (a stable slot when
/// resident, the packing ordinal otherwise), so adjacent bodies can merge into one draw.
const INSTANCE_OWNER: u32 = u32::MAX;

/// Solids use the otherwise-unused stroke-UV stream as integer occurrence ownership.
/// Other paint families keep their original UV bytes. Never perform floating-point arithmetic on owner words.
fn append_body_aux(out: &mut Vec<f32>, geometry: &TessellatedGeometry, owner: Option<u32>) {
    if let Some(owner) = owner {
        for _ in 0..geometry.vertex_types.len() { out.extend_from_slice(&[f32::from_bits(owner), 0.]); }
    } else if geometry.stroke_uv.len() == 2 * geometry.vertex_types.len() {
        out.extend_from_slice(&geometry.stroke_uv);
    } else {
        out.resize(out.len() + 2 * geometry.vertex_types.len(), 0.);
    }
}

/// The pooled-buffer label of chunk `k` (≥ 1) of the vertex stream `label` (leaked once per label and chunk: a
/// handful at most).
fn vertex_chunk_label(label: &'static str, k: usize) -> &'static str {
    thread_local! {
        static LABELS: std::cell::RefCell<std::collections::HashMap<(&'static str, usize), &'static str>> =
            Default::default();
    }
    LABELS.with(|m| {
        *m.borrow_mut()
            .entry((label, k))
            .or_insert_with(|| Box::leak(format!("{label} #{k}").into_boxed_str()))
    })
}

/// The pooled buffers of each phase's vertex streams: (positions, types, stroke UVs), opaque then transparent.
const VERTEX_STREAM_LABELS: [[&str; 3]; 2] = [
    [
        "Path Vertex Buffer (Opaque)",
        "Path Vertex Type Buffer (Opaque)",
        "Path Stroke UV Buffer (Opaque)",
    ],
    [
        "Path Vertex Buffer (Transparent)",
        "Path Vertex Type Buffer (Transparent)",
        "Path Stroke UV Buffer (Transparent)",
    ],
];

trait TakeAll {
    fn take_all(&mut self) -> [Option<PhaseResidency>; 2];
}

impl TakeAll for [Option<PhaseResidency>; 2] {
    fn take_all(&mut self) -> [Option<PhaseResidency>; 2] {
        [self[0].take(), self[1].take()]
    }
}

/// One body's packed instance record (`Engine::pack_body`), by pipeline family.
enum PackedBody {
    Solid(SolidPathInstance),
    /// `true`: the Q-116 post-shader gradient-stroke slot.
    Gradient(paint::GradientInstance, bool),
    Pattern(paint::PatternInstance, (String, bool)),
    Shader(PaintInstance, String, Option<(String, bool)>),
}

/// A replaced packet's prepared text, kept while the next frame's plan is packed: each chunk whose batches the new
/// plan draws unchanged (the same `Arc`s at the same positions), at the same sample count and text host generation,
/// is reused instead of prepared again.
struct TextCarry {
    text: std::sync::Arc<TextChunks>,
    batches: Vec<std::sync::Arc<crate::seams::TextQuadBatch>>,
    host_generation: u64,
    sample_count: u32,
}

impl TextCarry {
    /// The carried chunk `index`, when `batches` (that chunk of the new plan) are exactly its batches.
    fn chunk(
        &self,
        index: usize,
        batches: &[&std::sync::Arc<crate::seams::TextQuadBatch>],
        sample_count: u32,
    ) -> Option<std::sync::Arc<crate::text_draw::PreparedText>> {
        let start = index * TEXT_CHUNK;
        let old = self.batches.get(start..(start + TEXT_CHUNK).min(self.batches.len()))?;
        (self.sample_count == sample_count
            && self.host_generation == crate::text_draw::text_host_generation()
            && old.len() == batches.len()
            && old.iter().zip(batches).all(|(a, b)| std::sync::Arc::ptr_eq(a, b)))
        .then(|| self.text.chunks.get(index).cloned())
        .flatten()
    }
}

/// Cumulative counters for the prepared-frame ("prepare once, replay per
/// camera frame") path. Snapshotted via [`Engine::prepared_frame_stats`] and
/// flat-merged into the wasm `residency_stats()` JSON as `pf_*` keys.
#[derive(Clone, Copy, Default)]
pub struct PreparedFrameStats {
    /// Resident camera frames that replayed a valid packet (fast path taken).
    pub hits: u64,
    /// Frames eligible for the fast path whose packet was missing/invalid, so
    /// they fell back to a full encode (and rebuilt the packet).
    pub misses: u64,
    /// Packets (re)built from a full encode.
    pub rebuilds: u64,
    /// Prepared text chunks a packet rebuild kept from the one it replaced (`TextCarry`).
    pub text_reuses: u64,
    /// Hits that replayed the packet's recorded pass (a render bundle) instead of recording it.
    pub bundle_replays: u64,
    /// Text chunks a patch prepared again (a changed text layer's chunk).
    pub patched_text_chunks: u64,
    /// Cumulative geometry-upload bytes written via `make_buffer` (retained or
    /// not), counted on every buffer upload regardless of the flag. It climbs
    /// steadily with the flag off and plateaus once the fast path starts
    /// skipping the per-frame geometry uploads.
    pub upload_bytes: u64,
    /// Cumulative text-arena upload bytes (`TextDraw::prepare`). Stays 0 until
    /// the text path is rerouted through the counted pool (M2); then it
    /// plateaus on the fast path the same way `upload_bytes` does.
    pub text_bytes: u64,
    /// Cumulative transparent-phase SDF body draws and the instances they
    /// covered: runs of consecutive SDF-only z buckets batch into one
    /// instanced draw, so `sdf_draws` < `sdf_instances` whenever batching
    /// happened (#106 / #127).
    pub sdf_draws: u64,
    pub sdf_instances: u64,
    /// Plan changes applied to the packet in place (`Engine::patch_prepared`): only the changed items' bytes
    /// were written, and the packet (its buffers, text and recorded pass) kept.
    pub patches: u64,
    /// Items a patch rewrote.
    pub patched_items: u64,
    /// Bytes the patches wrote.
    pub patch_bytes: u64,
    /// Plan changes a patch couldn't apply (the frame rebuilt the packet); `patch_fallback_reason` says why.
    pub patch_fallbacks: u64,
    /// Transparent solid bodies drawn as part of the previous one's draw (`SolidRun`): draw calls saved.
    pub merged_bodies: u64,
    /// Rebuilds whose vertex streams appended only the geometries they didn't hold ([`PhaseResidency`]), per phase.
    pub resident_appends: u64,
    /// Vertex streams packed afresh under a residency (the first frame, too many holes, no room, another writer).
    pub resident_repacks: u64,
}

/// R2 retained frame resources (one per Engine; wasm editor lane only).
#[derive(Default)]
struct RetainedFrame {
    tick: u64,
    frame_uniform: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    /// (w, h, samples) → target set; tiny LRU (page size + capture scales).
    targets: Vec<TargetEntry>,
    bufs: std::collections::HashMap<&'static str, PooledBuf>,
}

struct ScratchAttachments {
    key: (u32, u32, u32),
    msaa: Option<wgpu::Texture>,
    depth: wgpu::Texture,
    bytes: u64,
    tick: u64,
}

/// What a non-retained engine keeps of its attachments ([`Engine::scratch_attachments`]): a few 1080p MSAA sets.
const SCRATCH_ATTACHMENT_BYTES: u64 = 640 << 20;
const SCRATCH_ATTACHMENT_SLOTS: usize = 3;

struct TargetEntry {
    key: (u32, u32, u32),
    target: wgpu::Texture,
    msaa: Option<wgpu::Texture>,
    depth: wgpu::Texture,
    tick: u64,
}

struct PooledBuf {
    buf: wgpu::Buffer,
    capacity: u64,
    /// Content writes (and handle replacements) so far: a vertex residency is valid only while its streams'
    /// buffers hold what it last wrote ([`PhaseResidency::writes`]).
    writes: u64,
    /// Storage bind group for this buffer (None until requested; cleared on
    /// every buffer replacement so it can never alias a dead buffer).
    bg: Option<wgpu::BindGroup>,
}

impl RetainedFrame {
    fn destroy_gpu_resources(&mut self) {
        if let Some((buffer, _)) = self.frame_uniform.take() {
            crate::scratch::destroy(&buffer);
        }
        for target in self.targets.drain(..) {
            target.destroy();
        }
        for (_, pooled) in self.bufs.drain() {
            pooled.destroy();
        }
    }
}

impl Drop for RetainedFrame {
    fn drop(&mut self) {
        self.destroy_gpu_resources();
    }
}

impl TargetEntry {
    fn destroy(self) {
        crate::scratch::destroy(&self.target);
        if let Some(msaa) = self.msaa {
            crate::scratch::destroy(&msaa);
        }
        crate::scratch::destroy(&self.depth);
    }
}

impl PooledBuf {
    fn destroy(self) {
        crate::scratch::destroy(&self.buf);
    }
}

/// Keep at most this many target sizes resident (the outermost renders' sizes; captures never use these slots).
const RETAINED_TARGET_SLOTS: usize = 3;
/// R2d GPU texture budgets (PLAN §4).
const PATTERN_TEXTURE_BUDGET_BYTES: usize = 256 * 1024 * 1024;
const SHADER_IMAGE_BUDGET_BYTES: usize = 128 * 1024 * 1024;

impl Engine {
    /// Engine with the default `Rgba8Unorm` frame format — the native / CLI /
    /// export / server lanes and the surfaceless wasm export engine, whose
    /// readback and golden contracts are RGBA end to end.
    pub fn new(gpu: GpuContext) -> Result<Engine> {
        Self::new_with_frame_format(gpu, wgpu::TextureFormat::Rgba8Unorm)
    }

    /// Engine with an explicit immutable frame color format (see
    /// [`Engine::frame_format`]). Only the two guaranteed WebGPU canvas-context
    /// 8-bit formats are legal; the live DOM wasm engine passes the browser's
    /// preferred canvas format so camera frames present directly into the
    /// swapchain. Everything readback-shaped (`render_plan`,
    /// `render_to_bytes`) still returns canonical RGBA bytes — the readback
    /// canonicalizes BGRA texels before un-premultiplication.
    pub fn new_with_frame_format(
        gpu: GpuContext,
        frame_format: wgpu::TextureFormat,
    ) -> Result<Engine> {
        Self::new_with_color(gpu, frame_format, None)
    }

    /// [`Engine::new_with_frame_format`] for an engine that blends in `working` instead of sRGB: authored colours
    /// and image texels stay sRGB code values and are converted where they meet the frame
    /// (`moda_shaders::working_color`). `None` is sRGB.
    pub fn new_with_color(
        gpu: GpuContext,
        frame_format: wgpu::TextureFormat,
        working: Option<moda_shaders::WorkingColor>,
    ) -> Result<Engine> {
        // Agent shaders' time budget is larger on a software adapter (crate::pixel::agent_budget_per_1080p).
        crate::pixel::set_software_adapter(gpu.adapter_info.device_type == wgpu::DeviceType::Cpu);
        if !is_supported_frame_format(frame_format) {
            bail!(
                "unsupported engine frame format {frame_format:?} \
                 (rgba8unorm|bgra8unorm, or rgba16float for float captures; never -srgb — render-semantics §7)"
            );
        }
        let device = &gpu.device;
        // Bind group layouts mirror manifest BGLs 1/2/3 + 313/314.
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Frame Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let storage_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Path Storage Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mask_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Path Mask Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let path_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Path Storage Pipeline Layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&storage_bgl), Some(&mask_bgl)],
            ..Default::default()
        });
        let page_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Page Pipeline Layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&storage_bgl)],
            ..Default::default()
        });
        // M3b: pattern pipeline layout (manifest pipeline layout 16).
        let pattern_texture_bgl = paint::draw::pattern_texture_bgl(device);
        let pattern_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Path Pattern Pipeline Layout"),
            bind_group_layouts: &[
                Some(&frame_bgl),
                Some(&storage_bgl),
                Some(&pattern_texture_bgl),
                Some(&mask_bgl),
            ],
            ..Default::default()
        });

        // M4e: image-backed procedural fills (manifest BGL 42 + layout 43).
        let shader_image_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Custom Shader Image Resource BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let image_shader_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Custom Shader Pipeline Layout (image-backed)"),
            bind_group_layouts: &[
                Some(&frame_bgl),
                Some(&storage_bgl),
                Some(&shader_image_bgl),
                Some(&mask_bgl),
            ],
            ..Default::default()
        });

        // vork: moda-text-draw's TextDraw takes its own GpuContext (device and queue); see PROVENANCE.md.
        let text = crate::text_draw::TextDraw::new(
            &moda_text_draw::GpuContext { device: gpu.device.clone(), queue: gpu.queue.clone() },
            frame_format,
            working,
        )?;
        let mut engine = Engine {
            gpu,
            frame_format,
            working,
            layouts: Layouts {
                frame_bgl,
                storage_bgl,
                mask_bgl,
                path_layout,
                page_layout,
                pattern_texture_bgl,
                pattern_layout,
                shader_image_bgl,
                image_shader_layout,
            },
            pipelines: Default::default(),
            shader_modules: Default::default(),
            inactive_mask: None,
            text,
            retained: false,
            reuse_frame_targets: true,
            keep_patterns: false,
            retained_frame: Default::default(),
            retained_patterns: Default::default(),
            retained_shader_images: Default::default(),
            retained_masks: Default::default(),
            first_readback_done: std::cell::Cell::new(false),
            float_twin: std::cell::OnceCell::new(),
            scratch_attachments: Default::default(),
            prepared_stats: std::cell::Cell::new(PreparedFrameStats::default()),
            prepared_frame: std::cell::RefCell::new(None),
            deferred_drops: Default::default(),
            backdrop_inline: std::cell::Cell::new(true),
            retained_text: Default::default(),
            pool_epoch: std::cell::Cell::new(0),
            pf_skip_uploads: std::cell::Cell::new(false),
            vertex_chunk_bytes: std::cell::Cell::new(None),
            vertex_chunks: std::cell::Cell::new([1, 1]),
            vertex_residency: std::cell::RefCell::new([None, None]),
            vertex_residency_on: std::cell::Cell::new(true),
            offscreen_residency: Default::default(),
            motion_blur_min_px: std::cell::Cell::new(0.0),
            solid_run_merging: std::cell::Cell::new(true),
            patch_fallback_reason: std::cell::Cell::new(""),
            prepared_ineligible_reason: std::cell::Cell::new(""),
            prepared_patch: std::cell::Cell::new(true),
            motion_text_greek_px: std::cell::Cell::new(0.0),
            greek_cache: Default::default(),
            greeked_text: std::cell::Cell::new(0),
            simplified_blurs: std::cell::Cell::new(0),
            capture_cache: Default::default(),
            capture_frame_budget: std::cell::Cell::new(None),
            captures_pending: std::cell::Cell::new(false),
            capture_stand_ins_mid_zoom: std::cell::Cell::new(false),
            capture_depth: std::cell::Cell::new(0),
            scratch: Default::default(),
            capture_scissor: Default::default(),
            capture_window: Default::default(),
            layer_blur_renderers: Default::default(),
            backdrop_renderers: Default::default(),
            shader_prepass: false,
            prepass_pool: Default::default(),
            prepass_fills: std::cell::Cell::new(0),
        };
        engine.inactive_mask = Some(engine.create_inactive_mask_group()?);
        // T9: the node-local text arena cache is retained GPU residency, so it must
        // start in sync with the engine's retained state (Default OFF). Otherwise
        // CLI/export/bench/mobile/Python lanes that never call `set_retained(true)`
        // would cache + retain text buffers, violating the "retained Default OFF —
        // CLI/export untouched" contract. `set_retained` keeps them in sync after.
        engine.text.set_vec_arena_cache_enabled(engine.retained);
        Ok(engine)
    }

    /// vork: the device and queue as moda-text-draw's GpuContext, which its TextDraw takes; see PROVENANCE.md.
    fn text_gpu(&self) -> moda_text_draw::GpuContext {
        moda_text_draw::GpuContext { device: self.gpu.device.clone(), queue: self.gpu.queue.clone() }
    }

    /// The immutable per-engine frame color format (see the field
    /// doc). Callers use this to gate direct present: a swapchain view is a
    /// legal direct render/MSAA-resolve target only when its format equals
    /// this.
    /// The layer-capture pipelines for a main pass of `sample_count` samples,
    /// built on first use.
    /// How many backdrop / blend-split renderers this engine has built (one per main-pass sample count used).
    pub fn backdrop_renderers_built(&self) -> usize {
        self.backdrop_renderers.borrow().len()
    }

    /// The backdrop / blend-split renderer for a main pass of `sample_count` (built on first use).
    fn backdrop_renderer(&self, sample_count: u32) -> Result<std::sync::Arc<crate::backdrop::BackdropRenderer>> {
        if let Some(r) = self.backdrop_renderers.borrow().get(&sample_count) {
            return Ok(r.clone());
        }
        let r = std::sync::Arc::new(crate::backdrop::BackdropRenderer::new(
            &self.gpu.device,
            self.frame_format,
            sample_count,
        )?);
        self.backdrop_renderers
            .borrow_mut()
            .insert(sample_count, r.clone());
        Ok(r)
    }

    fn layer_blur_renderer(
        &self,
        sample_count: u32,
    ) -> Result<std::sync::Arc<crate::layer_blur::LayerBlurRenderer>> {
        if let Some(r) = self.layer_blur_renderers.borrow().get(&sample_count) {
            return Ok(r.clone());
        }
        let r = std::sync::Arc::new(crate::layer_blur::LayerBlurRenderer::new(
            &self.gpu.device,
            &self.layouts.frame_bgl,
            &self.layouts.mask_bgl,
            sample_count,
            self.frame_format,
            self.working.as_ref(),
        )?);
        self.layer_blur_renderers
            .borrow_mut()
            .insert(sample_count, r.clone());
        Ok(r)
    }

    /// Simplify small background blurs on live camera presents while the canvas moves: a blur whose reach on
    /// screen is at most `px` device pixels draws its layer plainly (no copy / blur / composite of the backdrop,
    /// which splits the frame's render pass). The host sets it for frames during a pan, zoom or drag and clears it
    /// when motion stops, then renders the still frame exactly. Only a present into a view at a live camera
    /// (`render_plan_to_view` with a camera override) reads it: exports, captures and readbacks never do. 0 = off.
    /// Merge consecutive transparent solid bodies that draw the same into one draw (default on; see `SolidRun`).
    pub fn set_solid_run_merging(&self, on: bool) {
        self.solid_run_merging.set(on);
    }

    pub fn set_motion_blur_min_px(&self, px: f64) {
        self.motion_blur_min_px.set(if px.is_finite() && px > 0.0 { px } else { 0.0 });
    }

    /// Simplify small text on live camera presents while the canvas moves: a vector text layer whose type is under
    /// `px` device pixels per em draws each word as a translucent bar (`text_draw::greek_batch`) instead of its
    /// glyphs. The host sets it for frames during a pan, zoom or drag and clears it when motion stops, then renders
    /// the still frame exactly. As [`Self::set_motion_blur_min_px`], only a present at a live camera reads it:
    /// exports, captures, readbacks and motion-cache tiles never do. 0 = off.
    pub fn set_motion_text_greek_px(&self, px: f64) {
        self.motion_text_greek_px.set(if px.is_finite() && px > 0.0 { px } else { 0.0 });
    }

    /// Text layers the last live present drew as word bars (see [`Self::set_motion_text_greek_px`]).
    pub fn greeked_text(&self) -> u32 {
        self.greeked_text.get()
    }

    /// The greeked copy of `batch`, cached by its `Arc` (so a greeked frame's text chunks and packet are reused
    /// while the batch is), or `None` when it can't be greeked.
    fn greek_of(
        &self,
        batch: &std::sync::Arc<crate::seams::TextQuadBatch>,
    ) -> Option<std::sync::Arc<crate::seams::TextQuadBatch>> {
        let key = std::sync::Arc::as_ptr(batch) as usize;
        let mut cache = self.greek_cache.borrow_mut();
        if let Some((weak, greek)) = cache.get(&key) {
            if weak.upgrade().is_some_and(|b| std::sync::Arc::ptr_eq(&b, batch)) {
                return greek.clone();
            }
        }
        if cache.len() > 4096 {
            cache.retain(|_, (weak, _)| weak.strong_count() > 0);
        }
        let greek = crate::text_draw::greek_batch(batch).map(std::sync::Arc::new);
        cache.insert(key, (std::sync::Arc::downgrade(batch), greek.clone()));
        greek
    }

    /// Which of the plan's text batches (in plan order) a live motion present draws greeked, and a signature of
    /// that choice (0: none). `device_px_per_canvas_px` is the camera's zoom × device pixel ratio.
    fn greek_selection(&self, plan: &CoveragePlan, px: f64, device_px_per_canvas_px: f64) -> (u64, Vec<bool>) {
        if px <= 0.0 || !device_px_per_canvas_px.is_finite() || device_px_per_canvas_px <= 0.0 {
            return (0, Vec::new());
        }
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        let mut any = false;
        let mut picks = Vec::new();
        for item in &plan.items {
            if let DrawItem::Text(b) = item {
                let pick = crate::text_draw::greek_em_px(b, device_px_per_canvas_px).is_some_and(|em| em < px);
                if pick {
                    any = true;
                    picks.len().hash(&mut h);
                }
                picks.push(pick);
            }
        }
        if !any {
            return (0, Vec::new());
        }
        (h.finish() | 1, picks)
    }

    /// Blurs the last live present drew plainly (see [`Self::set_motion_blur_min_px`]).
    pub fn simplified_blurs(&self) -> u32 {
        self.simplified_blurs.get()
    }

    /// Layer-capture cache counters: (hits, misses, evictions, entries, bytes).
    pub fn capture_cache_stats(&self) -> (u64, u64, u64, usize, u64) {
        let c = self.capture_cache.borrow();
        let (entries, bytes) = c.resident();
        (c.hits, c.misses, c.evictions, entries, bytes)
    }

    /// The byte budget of the retained layer-capture results.
    pub fn set_capture_cache_budget(&self, bytes: u64) {
        self.capture_cache.borrow_mut().set_budget(bytes);
    }

    /// Smooth zoom-step crossings of layer captures (scale RFC stage C): a
    /// retained frame renders at most `n` captures whose earlier zoom step's
    /// result is still cached, and composites that result (over its own region)
    /// for the rest; the next frames render the rest. Captures with no earlier
    /// result always render. `None` (the default) renders every capture.
    pub fn set_capture_frame_budget(&self, n: Option<u32>) {
        self.capture_frame_budget.set(n);
    }

    /// The live lane's frames while the zoom changes (`PlanCaches::zooming`): a layer capture whose earlier zoom
    /// step's result covers all of it the target shows composites that result (scaled) instead of capturing again,
    /// whatever the budget, and the frame reports `captures_pending`, so the frame after the zoom holds captures
    /// exactly. Without it every visible capture shadow or layer blur is captured again at each zoom step: over the
    /// capture window and the shadows' reach, up to 16 Mpx at MSAA 4 per shadowed layer at 3200 %.
    pub fn set_capture_stand_ins_mid_zoom(&self, on: bool) {
        self.capture_stand_ins_mid_zoom.set(on);
    }

    /// Whether the last frame drew a layer capture from an earlier zoom step's
    /// result: render another frame to replace it.
    pub fn captures_pending(&self) -> bool {
        self.captures_pending.get()
    }

    /// Composites drawn from an earlier zoom step's result, cumulative.
    pub fn capture_stand_in_hits(&self) -> u64 {
        self.capture_cache.borrow().stand_in_hits
    }

    pub fn frame_format(&self) -> wgpu::TextureFormat {
        self.frame_format
    }

    /// The engine that draws the pixel graph's float captures (`@frame`, `@layer.render`, `@layer.backdrop`;
    /// motion-model.md §10.1): this engine when its frame format is already `Rgba16Float`, otherwise a twin on the
    /// same device, built on first use. The twin draws with the same pipelines' math (gamma-space premultiplied
    /// source-over) into `Rgba16Float` targets, so a capture matches the 8-bit render within quantization.
    pub fn float_capture(&self) -> Result<&Engine> {
        if self.frame_format == wgpu::TextureFormat::Rgba16Float {
            return Ok(self);
        }
        if let Some(twin) = self.float_twin.get() {
            return Ok(twin);
        }
        let mut twin = Engine::new_with_color(self.gpu.clone(), wgpu::TextureFormat::Rgba16Float, self.working)?;
        twin.keep_patterns = true;
        Ok(self.float_twin.get_or_init(|| Box::new(twin)))
    }

    /// Render `plan` into a fresh `Rgba16Float` texture through [`Engine::float_capture`]: premultiplied, the
    /// vector renderer's gamma-encoded (`srgb`-tagged) values. Returned without readback; the pixel graph samples
    /// it on the GPU. (Non-retained engines allocate a new target per render, so the texture stays valid.)
    pub fn render_plan_rgba16f(&self, plan: &CoveragePlan) -> Result<wgpu::Texture> {
        let engine = self.float_capture()?;
        anyhow::ensure!(!engine.retained, "float captures need a non-retained engine (targets are reused)");
        engine.render_plan_to_texture(plan)
    }

    /// [`Engine::render_plan_rgba16f`] read back: `width × height × 4` half floats, premultiplied, as stored.
    pub fn render_plan_rgba16f_readback(&self, plan: &CoveragePlan) -> Result<Vec<half::f16>> {
        let target = self.render_plan_rgba16f(plan)?;
        let gpu = &self.float_capture()?.gpu;
        crate::readback::poll_until_complete_with_budget(&gpu.device, crate::readback::READBACK_POLL_TIMEOUT_RETRIES_FIRST)
            .context("device poll after submit")?;
        crate::readback::read_texture_rgba16f(&gpu.device, &gpu.queue, &target, plan.width_px, plan.height_px)
    }

    /// The adapter's MSAA ceiling for THIS engine's frame format (color ∩
    /// depth) — the single clamp every frame attachment and pipeline
    /// `MultisampleState.count` derives from. Format-aware: a BGRA
    /// frame engine consults the BGRA intersection.
    fn max_frame_sample_count(&self) -> u32 {
        self.gpu.max_sample_count_for(self.frame_format)
    }

    /// Task #95 R2: enable retained GPU frame residency (wasm editor lane +
    /// the resident_smoke test). Default OFF — CLI/export is untouched.
    pub fn set_retained(&mut self, on: bool) {
        self.retained = on;
        // T9: the node-local text arena cache is retained GPU residency, so tie it
        // to retained mode. The one-way disables (#149 memory guard, `?wasmRetained=0`)
        // come through here as `set_retained(false)` — `set_enabled(false)` clears it
        // AND stops it caching, so it can't immediately repopulate Stage-0 frames
        // (the `clear_retained_gpu` below only flushes once). Re-enabled if retained
        // is ever turned back on.
        self.text.set_vec_arena_cache_enabled(on);
        if !on {
            self.clear_retained_gpu();
        }
    }

    /// Keep other retained GPU caches while giving each offscreen render its own resolve texture.
    /// Configure before rendering when output consumers may be submitted after a subsequent render.
    pub fn set_reuse_frame_targets(&mut self, on: bool) {
        self.reuse_frame_targets = on;
    }

    /// Shade procedural fills once per pixel on the live camera lane under MSAA: each fill renders into a
    /// single-sample, target-sized texture before the main pass, and the main pass draws the node's geometry
    /// with the `shader-fill-composite` module reading its pixel back (docs/qa/shader-perf.md). The in-app
    /// WebGPU backend otherwise runs the fill's fragment shader per sample (4× under the editor's MSAA).
    /// Output matches the direct draw: the pre-pass shades the same pixel centres. Off by default (exports
    /// and the CLI draw fills directly); moda-wasm turns it on.
    pub fn set_shader_prepass(&mut self, on: bool) {
        self.shader_prepass = on;
        if !on {
            self.prepass_pool.borrow_mut().take();
        }
    }

    /// How many shader fills the last render shaded in the pre-pass (0 when it drew them directly).
    pub fn shader_prepass_fills(&self) -> usize {
        self.prepass_fills.get()
    }

    /// The pre-pass targets for a `width × height` frame with at least `slots` colour slots.
    fn shader_prepass_targets(
        &self,
        width: u32,
        height: u32,
        slots: usize,
    ) -> (Vec<wgpu::TextureView>, wgpu::TextureView, wgpu::Sampler) {
        let device = &self.gpu.device;
        let mut pool = self.prepass_pool.borrow_mut();
        if pool.as_ref().is_some_and(|p| p.width != width || p.height != height) {
            if let Some(old) = pool.take() {
                old.slots.iter().for_each(|slot| crate::scratch::destroy(slot));
                crate::scratch::destroy(&old.depth);
            }
        }
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let p = pool.get_or_insert_with(|| ShaderPrepassPool {
            width,
            height,
            slots: Vec::new(),
            depth: device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shader pre-pass depth"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("shader pre-pass sampler"),
                ..Default::default()
            }),
        });
        while p.slots.len() < slots {
            p.slots.push(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shader pre-pass colour"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.frame_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }));
        }
        (
            p.slots
                .iter()
                .take(slots)
                .map(|t| t.create_view(&Default::default()))
                .collect(),
            p.depth.create_view(&Default::default()),
            p.sampler.clone(),
        )
    }

    /// Retained mask-cache counters: (served from the cache, rendered, full-rect
    /// masks seen, resident texture bytes, entries). `None` before the first
    /// retained frame with masks.
    pub fn mask_cache_stats(&self) -> Option<(u64, u64, u64, u64, usize)> {
        self.retained_masks.borrow().as_ref().map(|c| {
            let (bytes, entries) = c.resident();
            (c.hits, c.renders, c.full_rect, bytes, entries)
        })
    }

    /// Shared mask atlas passes live frames have rendered (`effects::mask::set_mask_atlas_enabled`).
    pub fn mask_atlas_passes(&self) -> u64 {
        self.retained_masks.borrow().as_ref().map_or(0, |c| c.atlas_passes)
    }

    /// Retained GPU memory by owner, in bytes (benches; see `crate::gpu_profile`). Frame targets count
    /// colour + MSAA + depth at 4 B per sample.
    pub fn gpu_memory_json(&self) -> moda_scene::value::Value {
        let frame = self.retained_frame.borrow();
        let pool: u64 = frame.bufs.values().map(|b| b.capacity).sum();
        let targets: u64 = frame
            .targets
            .iter()
            .map(|t| {
                let (w, h, s) = t.key;
                let px = u64::from(w) * u64::from(h);
                px * 4
                    + if t.msaa.is_some() {
                        px * 4 * u64::from(s)
                    } else {
                        0
                    }
                    + px * 4 * u64::from(s.max(1))
            })
            .sum();
        let (pattern_count, pattern_bytes) = self
            .retained_patterns
            .borrow()
            .as_ref()
            .map_or((0, 0), |c| c.resident());
        let shader_image_bytes = self
            .retained_shader_images
            .borrow()
            .as_ref()
            .map_or(0, |c| c.total_bytes);
        let emoji = self.text.emoji_texture_bytes();
        let arena = self.text.vec_arena_stats();
        let (mask_cache_bytes, mask_cache_entries) = self
            .retained_masks
            .borrow()
            .as_ref()
            .map_or((0, 0), |c| c.resident());
        moda_scene::value::json!({
            "geometryPoolBytes": pool,
            "geometryPoolBuffers": frame.bufs.len(),
            "frameTargetBytes": targets,
            "patternTextures": pattern_count,
            "patternTextureBytes": pattern_bytes,
            "shaderImageBytes": shader_image_bytes,
            "emojiBytes": emoji,
            "textArenaBytes": arena.bytes,
            "maskCacheBytes": mask_cache_bytes,
            "maskCacheEntries": mask_cache_entries,
            "captureCacheBytes": self.capture_cache.borrow().resident().1,
            "captureCacheEntries": self.capture_cache.borrow().resident().0,
        })
    }

    /// Snapshot of the prepared-frame counters; the wasm host flat-merges these
    /// into `residency_stats()` as `pf_*` keys.
    pub fn prepared_frame_stats(&self) -> PreparedFrameStats {
        self.prepared_stats.get()
    }

    /// T9 cache gate: enable/disable the content-addressed node-local text arena
    /// cache (`TextDraw::prepare`). Tracks retained mode — `set_retained` drives it
    /// (ON in the wasm editor lane, OFF in CLI/export and any non-retained lane), so
    /// the cache only retains GPU buffers where retained residency is active.
    /// Disabling forces the legacy create-and-upload-every-frame text geometry path.
    pub fn set_vec_arena_cache_enabled(&mut self, on: bool) {
        self.text.set_vec_arena_cache_enabled(on);
    }

    /// T9: snapshot of the vector-text arena cache counters; the wasm host
    /// flat-merges these into `residency_stats()` as `vec_arena_*` keys.
    pub fn vec_arena_stats(&self) -> crate::vec_arena_cache::VecArenaStats {
        self.text.vec_arena_stats()
    }

    /// Accumulate geometry-upload bytes for the `pf_upload_bytes` counter (text
    /// arenas count into `pf_text_bytes` separately). Counted at the actual
    /// `queue.write_buffer`, flag-independent, so an A/B run sees the steady
    /// climb with the flag off plateau once the fast path skips the uploads.
    fn add_pf_upload_bytes(&self, n: u64) {
        let mut s = self.prepared_stats.get();
        s.upload_bytes = s.upload_bytes.saturating_add(n);
        self.prepared_stats.set(s);
    }

    /// Accumulate text-arena upload bytes for the `pf_text_bytes` counter
    /// (counted only when `TextDraw::prepare` actually runs — i.e. on a miss).
    fn add_pf_text_bytes(&self, n: u64) {
        let mut s = self.prepared_stats.get();
        s.text_bytes = s.text_bytes.saturating_add(n);
        self.prepared_stats.set(s);
    }

    /// M2: bump the retained-pool epoch (a pool buffer's contents or handle just
    /// changed), so any packet built against the prior epoch invalidates.
    fn bump_pool_epoch(&self) {
        self.pool_epoch.set(self.pool_epoch.get().wrapping_add(1));
    }

    /// Drop all retained GPU resources (memory-guard failure path; also
    /// called when retained mode is switched off).
    pub fn clear_retained_gpu(&mut self) {
        {
            let mut frame = self.retained_frame.borrow_mut();
            frame.destroy_gpu_resources();
            *frame = RetainedFrame::default();
        }
        self.retained_patterns.borrow_mut().take();
        self.retained_shader_images.borrow_mut().take();
        self.retained_masks.borrow_mut().take();
        if let Some(pool) = self.prepass_pool.borrow_mut().take() {
            pool.slots.iter().for_each(wgpu::Texture::destroy);
            crate::scratch::destroy(&pool.depth);
        }
        // M2: the packet's reused text bind groups reference the (now-destroyed)
        // retained frame uniform, and its geometry reuse assumes the (now-reset)
        // pool — drop it and bump the epoch so nothing survives the reset.
        self.prepared_frame.borrow_mut().take();
        self.deferred_drops.borrow_mut().clear();
        self.retained_text.borrow_mut().take();
        self.bump_pool_epoch();
        // Cached layer-capture results are GPU textures too, and so is what captures took from the frame's
        // generation (targets, filter passes, buffers, the bind groups over them).
        self.capture_cache.borrow_mut().clear();
        self.scratch.clear();
        // T9: the node-local text arena cache holds Arc<wgpu::Buffer> GPU
        // residency (up to VEC_ARENA_BUDGET_BYTES) that the rest of this teardown
        // would otherwise leave alive — drop it so the memory-guard ALARM /
        // retained-off path actually returns to a minimal GPU footprint. It
        // repopulates lazily on the next prepare.
        self.text.clear_vec_arena_cache();
    }

    /// R2c: grow-only pooled buffer per class label (retained mode); falls
    /// back to a fresh allocation when retained mode is off. Stale bytes
    /// beyond the written range are never read — draws use explicit vertex
    /// ranges and storage indices stay below the written instance count.
    /// Upload one phase's vertex streams (positions, vertex types, stroke UVs: 2, 1 and 2 floats a vertex), split
    /// into chunks at draw boundaries so no buffer passes the device's buffer size limit (a document of dense paths
    /// drew a 935 MB stream, and the frame failed on the 256 MB limit). Each chunk's buffers are pooled under their
    /// own labels; the first chunk keeps the stream's label, so an unsplit frame is exactly what it was.
    /// A phase's vertex streams under its residency (`appending`: `Some(true)` = `streams` hold only what the
    /// residency didn't, written after what it holds; `Some(false)` = packed afresh; `None`: no residency). A
    /// fresh pack that splits across buffers drops the residency (appending needs one buffer a stream).
    fn upload_resident_streams(
        &self,
        k: usize,
        residency: &mut Option<PhaseResidency>,
        appending: Option<bool>,
        streams: [&[f32]; 3],
        draws: impl Iterator<Item = (u32, u32)>,
    ) -> VertexStreams {
        let labels = VERTEX_STREAM_LABELS[k];
        if appending == Some(true) {
            if let Some(r) = residency.as_mut() {
                let added = (streams[0].len() / 2) as u32;
                let per_vertex = [8u64, 4, 8];
                let mut g = self.retained_frame.borrow_mut();
                let mut bufs: Vec<wgpu::Buffer> = Vec::with_capacity(3);
                for s in 0..3 {
                    let e = g.bufs.get_mut(labels[s]).expect("checked resident");
                    if !streams[s].is_empty() {
                        let data: &[u8] = bytemuck::cast_slice(streams[s]);
                        self.add_pf_upload_bytes(data.len() as u64);
                        crate::gpu_profile::count_buffer_write(data.len());
                        self.gpu.queue.write_buffer(&e.buf, u64::from(r.total) * per_vertex[s], data);
                        e.writes += 1;
                    }
                    r.writes[s] = e.writes;
                    bufs.push(e.buf.clone());
                }
                if added > 0 {
                    self.bump_pool_epoch();
                }
                r.total += added;
                let mut st = self.prepared_stats.get();
                st.resident_appends = st.resident_appends.wrapping_add(1);
                self.prepared_stats.set(st);
                let [a, b, c]: [wgpu::Buffer; 3] = bufs.try_into().expect("three streams");
                return VertexStreams {
                    bases: vec![0],
                    bufs: vec![[a, b, c]],
                };
            }
        }
        // Packed afresh under a residency: room to append a quarter more before the next fresh pack.
        if residency.is_some() && self.retained {
            let per_vertex = [8u64, 4, 8];
            let n = (streams[0].len() / 2) as u64;
            for s in 0..3 {
                self.reserve_pooled(labels[s], (n + n / 4) * per_vertex[s], wgpu::BufferUsages::VERTEX);
            }
        }
        let out = self.upload_vertex_streams(labels, streams, draws);
        if let Some(r) = residency.as_mut() {
            let mut st = self.prepared_stats.get();
            st.resident_repacks = st.resident_repacks.wrapping_add(1);
            self.prepared_stats.set(st);
            if out.bases.len() == 1 {
                r.total = (streams[0].len() / 2) as u32;
                let g = self.retained_frame.borrow();
                for s in 0..3 {
                    r.writes[s] = g.bufs.get(labels[s]).map_or(u64::MAX, |e| e.writes);
                }
            } else {
                *residency = None;
            }
        }
        out
    }

    /// Grow pooled buffer `label` to at least `size` bytes ahead of a write (its contents are not kept).
    fn reserve_pooled(&self, label: &'static str, size: u64, usage: wgpu::BufferUsages) {
        let size = size.max(4).div_ceil(4) * 4;
        let limit = self
            .vertex_chunk_bytes
            .get()
            .unwrap_or_else(|| self.gpu.device.limits().max_buffer_size);
        if size > limit {
            return;
        }
        let mut g = self.retained_frame.borrow_mut();
        let entry = g.bufs.get_mut(label);
        if entry.as_ref().is_some_and(|e| e.capacity >= size) {
            return;
        }
        let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        match entry {
            Some(e) => {
                crate::scratch::destroy(&std::mem::replace(&mut e.buf, buf));
                e.capacity = size;
                e.bg = None;
                e.writes += 1;
            }
            None => {
                g.bufs.insert(label, PooledBuf { buf, capacity: size, writes: 1, bg: None });
            }
        }
        drop(g);
        self.bump_pool_epoch();
    }

    /// Keep vertex residency for retained offscreen frames as live presents do (off by default: a one-off
    /// export draws each plan once). With it, a frame that cannot be patched uploads only the geometry the
    /// streams do not hold, and a solid body keeps its record slot from frame to frame.
    pub fn set_offscreen_vertex_residency(&self, on: bool) {
        if self.offscreen_residency.replace(on) != on {
            self.bump_pool_epoch();
            *self.vertex_residency.borrow_mut() = [None, None];
        }
    }

    /// Vertex residency on or off (on by default); off drops what it holds.
    pub fn set_vertex_residency(&self, on: bool) {
        if self.vertex_residency_on.replace(on) != on {
            // Resident solids read the draw's instance; packed solids carry their owner per vertex. A packet
            // built in the other mode holds the other encoding: it must not be hit, patched or replayed.
            self.bump_pool_epoch();
        }
        if !on {
            *self.vertex_residency.borrow_mut() = [None, None];
        }
    }

    fn upload_vertex_streams(
        &self,
        labels: [&'static str; 3],
        streams: [&[f32]; 3],
        draws: impl Iterator<Item = (u32, u32)>,
    ) -> VertexStreams {
        let limit = self
            .vertex_chunk_bytes
            .get()
            .unwrap_or_else(|| self.gpu.device.limits().max_buffer_size)
            .max(8);
        // The widest stream holds 8 bytes a vertex.
        let max_verts = (limit / 8).min(u64::from(u32::MAX)) as u32;
        let total = (streams[0].len() / 2) as u32;
        let mut bases = vec![0u32];
        for (offset, count) in draws {
            let base = *bases.last().expect("a chunk");
            if offset > base && offset + count - base > max_verts {
                bases.push(offset);
            }
        }
        let per_vertex = [2usize, 1, 2];
        let bufs = (0..bases.len())
            .map(|k| {
                let (lo, hi) = (bases[k] as usize, bases.get(k + 1).map_or(total as usize, |&b| b as usize));
                std::array::from_fn(|s| {
                    let label = if k == 0 { labels[s] } else { vertex_chunk_label(labels[s], k) };
                    let data = &streams[s][lo * per_vertex[s]..(hi * per_vertex[s]).min(streams[s].len())];
                    self.make_buffer(label, bytemuck::cast_slice(data), wgpu::BufferUsages::VERTEX)
                })
            })
            .collect();
        VertexStreams { bases, bufs }
    }

    /// Cap each vertex-stream buffer at `bytes` (tests force a split with a small cap); `None`: the device's limit.
    pub fn set_vertex_chunk_bytes(&self, bytes: Option<u64>) {
        self.vertex_chunk_bytes.set(bytes);
    }

    /// How many buffers the last frame's (opaque, transparent) vertex streams took.
    pub fn last_vertex_chunks(&self) -> [usize; 2] {
        self.vertex_chunks.get()
    }

    /// Whether this render draws from the engine's resident frame resources (the labelled buffer pool and its bind
    /// groups, the frame uniform, vertex residency): a retained engine's outermost render. A capture's render takes
    /// its buffers from the frame's generation (`scratch`) as a non-retained render does, each for that one use,
    /// so it never writes under the outer frame's draws, whenever its commands reach the queue.
    fn resident(&self) -> bool {
        self.retained && self.capture_depth.get() == 0
    }

    fn make_buffer(
        &self,
        label: &'static str,
        data: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        let size = (data.len().max(4) as u64).div_ceil(4) * 4;
        if !self.resident() {
            if data.is_empty() {
                return self.scratch.empty(device, label, usage | wgpu::BufferUsages::COPY_DST);
            }
            let buf = self.scratch.buffer(device, &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.add_pf_upload_bytes(data.len() as u64);
            crate::gpu_profile::count_buffer_write(data.len());
            crate::scratch::write_buffer(device, queue, &buf, 0, data);
            return buf;
        }
        let mut g = self.retained_frame.borrow_mut();
        // M2: whether the pooled buffer already existed BEFORE this call — only an
        // existing, non-grown buffer holds the prior frame's (byte-identical, on a
        // hit) data, so only then is skipping the upload safe.
        let existed = g.bufs.contains_key(label);
        let entry = g.bufs.entry(label).or_insert_with(|| PooledBuf {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity: size,
            writes: 0,
            bg: None,
        });
        let mut grew = false;
        if entry.capacity < size {
            // Grow 1.5× (WebGPUStorageBufferManager's resize policy class).
            let cap = size.max(entry.capacity + entry.capacity / 2).div_ceil(4) * 4;
            let old = std::mem::replace(
                &mut entry.buf,
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: cap,
                    usage: usage | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            crate::scratch::destroy(&old);
            entry.capacity = cap;
            entry.bg = None;
            grew = true;
        }
        // M2 prepared-frame HIT: the freshly-collected `data` is byte-identical to
        // what this pooled buffer already holds (the plan/identity check proved
        // the draw is unchanged), so reuse the buffer in place and skip the
        // upload. Guarded so a missing/grown buffer (which would otherwise hold
        // stale/empty bytes) always falls through to a real write.
        let skip_upload = self.pf_skip_uploads.get() && existed && !grew && entry.capacity >= size;
        // Any buffer-handle replacement (grow) or real content write mutates the
        // shared pool, so advance the epoch — a packet built against the old epoch
        // must rebuild. A grow with empty data still replaces the handle, so it
        // must bump even though nothing is written below.
        if grew || (!data.is_empty() && !skip_upload) {
            self.bump_pool_epoch();
            entry.writes += 1;
        }
        if !data.is_empty() && !skip_upload {
            self.add_pf_upload_bytes(data.len() as u64);
            crate::gpu_profile::count_buffer_write(data.len());
            queue.write_buffer(&entry.buf, 0, data);
        }
        entry.buf.clone()
    }

    /// R2c: storage bind group for a pooled buffer — re-created only when the
    /// buffer was replaced (pool growth clears `bg`). A render that is not resident
    /// (`Engine::resident`): the group of its generation buffer, kept with it
    /// (`ScratchPool::bind`). `buf_label` is the pool key the buffer was made under.
    fn make_storage_bg(
        &self,
        label: &'static str,
        buf_label: &'static str,
        buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let mk = || {
            self.gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.layouts.storage_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
        };
        if !self.resident() {
            return self.scratch.bind(
                &self.gpu.device,
                label,
                &self.layouts.storage_bgl,
                &[crate::scratch::Bound::Buffer(buf.clone())],
            );
        }
        let mut g = self.retained_frame.borrow_mut();
        if let Some(entry) = g.bufs.get_mut(buf_label) {
            if entry.bg.is_none() {
                entry.bg = Some(mk());
            }
            return entry.bg.clone().expect("bg just ensured");
        }
        mk()
    }

    /// R2b: frame uniform buffer + bind group created once, written per frame.
    fn frame_uniform_and_bg(&self, vp: &[f32; 16]) -> (wgpu::Buffer, wgpu::BindGroup) {
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        let desc = wgpu::BufferDescriptor {
            label: Some("frame uniforms"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        // The resident uniform is the engine's own for as long as it is retained (a prepared frame's text and
        // recorded pass bind it on later frames). Any other render takes one for this use, with the group that
        // names it.
        let (buf, bg) = if self.resident() {
            let mut g = self.retained_frame.borrow_mut();
            let (buf, bg) = g.frame_uniform.get_or_insert_with(|| {
                let buf = device.create_buffer(&desc);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("frame bg"),
                    layout: &self.layouts.frame_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                (buf, bg)
            });
            (buf.clone(), bg.clone())
        } else {
            let buf = self.scratch.buffer(device, &desc);
            let bg = self.scratch.bind(
                device,
                "frame bg",
                &self.layouts.frame_bgl,
                &[crate::scratch::Bound::Buffer(buf.clone())],
            );
            (buf, bg)
        };
        crate::gpu_profile::count_buffer_write(64);
        crate::scratch::write_buffer(device, queue, &buf, 0, bytemuck::cast_slice(vp));
        (buf, bg)
    }

    /// R2a: target + MSAA + depth textures cached by (w, h, samples).
    fn frame_targets(
        &self,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> (wgpu::Texture, Option<wgpu::Texture>, wgpu::Texture) {
        let device = &self.gpu.device;
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let mk = || {
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("export target"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // The engine's immutable frame format: rgba8unorm
                // by default, the browser-preferred format on the live wasm
                // lane; never -srgb (render-semantics §7).
                format: self.frame_format,
                // COPY_SRC: export/PDF/screenshot lanes read the texels back.
                // TEXTURE_BINDING: the wasm editor's blit lanes sample this
                // single-sample resolved target in a fullscreen-blit pass into
                // the canvas swapchain — the per-page direct-present path
                // (`present_page`) and the backdrop-split fallback in
                // `present_plan` (non-backdrop frames now render directly into the
                // swapchain and never sample this). The blit avoids a
                // `copy_texture_to_texture` into the swapchain, which needs the
                // canvas surface to support COPY_DST (Android/Dawn often can't,
                // blanking the canvas). Adding the bind usage does not change
                // any rendered output and is free for lanes that never sample.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let msaa = (sample_count > 1).then(|| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("export msaa"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.frame_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
            });
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("export depth"),
                size: extent,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            (target, msaa, depth)
        };
        // A capture (a rendered mask, a layer-blur group) is sampled by the render that asked for it, after other
        // captures and that render's own target are taken: from the size-keyed retained slots it could share a
        // texture with another capture of its size, or be evicted and destroyed (more sizes than slots) before the
        // frame that reads it is submitted. Captures take their own scratch targets, as in a non-retained render.
        if !self.retained || !self.reuse_frame_targets || self.capture_depth.get() > 0 {
            return self.scratch_frame_targets(width, height, sample_count, extent);
        }
        let mut g = self.retained_frame.borrow_mut();
        g.tick += 1;
        let tick = g.tick;
        let key = (width, height, sample_count);
        if let Some(e) = g.targets.iter_mut().find(|e| e.key == key) {
            e.tick = tick;
            return (e.target.clone(), e.msaa.clone(), e.depth.clone());
        }
        let (target, msaa, depth) = mk();
        g.targets.push(TargetEntry {
            key,
            target: target.clone(),
            msaa: msaa.clone(),
            depth: depth.clone(),
            tick,
        });
        if g.targets.len() > RETAINED_TARGET_SLOTS {
            g.targets.sort_by_key(|e| std::cmp::Reverse(e.tick));
            let evicted = g.targets.split_off(RETAINED_TARGET_SLOTS);
            for target in evicted {
                target.destroy();
            }
        }
        (target, msaa, depth)
    }

    /// A non-retained engine's targets, and any capture's: a new resolve target, and MSAA colour and depth
    /// attachments kept from an earlier render of the same size ([`Engine::scratch_attachments`]).
    fn scratch_frame_targets(
        &self,
        width: u32,
        height: u32,
        sample_count: u32,
        extent: wgpu::Extent3d,
    ) -> (wgpu::Texture, Option<wgpu::Texture>, wgpu::Texture) {
        let device = &self.gpu.device;
        let desc = wgpu::TextureDescriptor {
            label: Some("export target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.frame_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        // A layer capture's target lives as long as its render's generation (its filters read it there); the
        // outermost render's is the caller's.
        let target = if self.capture_depth.get() > 0 {
            self.scratch.texture(device, &desc)
        } else {
            device.create_texture(&desc)
        };
        let key = (width, height, sample_count);
        let mut kept = self.scratch_attachments.borrow_mut();
        let tick = kept.iter().map(|e| e.tick).max().unwrap_or(0) + 1;
        if let Some(e) = kept.iter_mut().find(|e| e.key == key) {
            e.tick = tick;
            return (target, e.msaa.clone(), e.depth.clone());
        }
        let msaa = (sample_count > 1).then(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("export msaa"),
                size: extent,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.frame_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export depth"),
            size: extent,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let texels = u64::from(width) * u64::from(height) * u64::from(sample_count);
        let bytes = texels * (u64::from(self.frame_format.block_copy_size(None).unwrap_or(4)) + 4);
        if bytes <= SCRATCH_ATTACHMENT_BYTES {
            kept.push(ScratchAttachments {
                key,
                msaa: msaa.clone(),
                depth: depth.clone(),
                bytes,
                tick,
            });
            // Least recently used out first, until the rest fit.
            while kept.len() > SCRATCH_ATTACHMENT_SLOTS
                || kept.iter().map(|e| e.bytes).sum::<u64>() > SCRATCH_ATTACHMENT_BYTES
            {
                let oldest = kept.iter().enumerate().min_by_key(|(_, e)| e.tick).map(|(i, _)| i).expect("non-empty");
                kept.swap_remove(oldest);
            }
        }
        (target, msaa, depth)
    }

    /// Retained-mode draw key (R2d): delegates to [`retained_pattern_draw_key`].
    /// Non-retained video renders also keep pattern storage across frames.
    /// What a body's draw binds besides its record: a procedural fill's module, and the texture bind-group key
    /// (texture key, clamp sampler) of a pattern or of an image-backed procedural fill (M4e: bound at group 2
    /// with the clamp+linear sampler). Two bodies of one paint family draw through the same pipeline when they
    /// bind the same.
    fn body_binding<'a>(&self, body_paint: &'a BodyPaint) -> (Option<&'a str>, Option<(String, bool)>) {
        match body_paint {
            BodyPaint::Pattern(p) => (None, Some((self.pattern_draw_key(p), p.clamp_sampler))),
            BodyPaint::Shader(body) => (
                Some(body.module_id.as_str()),
                body.image.as_ref().map(|t| (self.shader_image_draw_key(t), true)),
            ),
            _ => (None, None),
        }
    }

    /// One body's instance record as the frame's collect packs it into its pipeline family's stream, with
    /// `vertex_offset` (its first vertex in the phase's vertex stream) filled in and the live shader clock threaded
    /// in. The prepared-frame patch (`patch_prepared`) packs a changed body with this too, so its bytes are the
    /// ones a rebuild would upload.
    fn pack_body(
        &self,
        body_paint: &BodyPaint,
        instance: &PaintInstance,
        geometry: &TessellatedGeometry,
        vertex_offset: u32,
        present_shader_time: Option<f64>,
    ) -> PackedBody {
        match body_paint {
            BodyPaint::Gradient(g) | BodyPaint::PostShaderGradientStroke(g) => {
                let mut gi = **g;
                // PathGradientData.pathInfo = [vertexOffset,
                // vertexCount, depth, fillType].
                gi.path_info[0] = vertex_offset as f32;
                gi.path_info[1] = geometry.vertex_count as f32;
                PackedBody::Gradient(gi, !matches!(body_paint, BodyPaint::Gradient(_)))
            }
            BodyPaint::Pattern(p) => {
                let mut pi = p.instance;
                // PathPatternData.pathInfo = [vertexOffset,
                // vertexCount, strokeWidth, depth].
                pi.path_info[0] = vertex_offset as f32;
                pi.path_info[1] = geometry.vertex_count as f32;
                PackedBody::Pattern(pi, self.body_binding(body_paint).1.expect("a pattern binds its texture"))
            }
            BodyPaint::Shader(body) => {
                // M3f: full 672-byte CustomShaderData; pathInfo =
                // [vertexOffset, vertexCount, strokeWidth, depth]
                // (`WebGPUPathRenderer.ts:2062`).
                let mut si = *instance;
                si.path_info[0] = vertex_offset as f32;
                si.path_info[1] = geometry.vertex_count as f32;
                // Re-thread the LIVE clock onto this cached
                // fill's `shader_params` clock slots so an animating
                // shader advances on a camera-plan cache HIT (no
                // rebuild). The clock is a direct copy in every pack,
                // so this reproduces a fresh build byte-for-byte. No
                // override (export / per-page) leaves the baked clock.
                if let (Some(t), Some(offsets)) = (present_shader_time, body.clock_offsets.as_ref()) {
                    // Offsets come from `clock_param_offsets`'s
                    // `0..SHADER_PARAM_FLOAT_CAPACITY` scan, so they
                    // are always in bounds; `get_mut` degrades to a
                    // no-op (not a panic) if the capacities ever drift.
                    for &off in offsets {
                        if let Some(slot) = si.shader_params.get_mut(off as usize) {
                            *slot = t as f32;
                        }
                    }
                }
                PackedBody::Shader(si, body.module_id.clone(), self.body_binding(body_paint).1)
            }
            _ => PackedBody::Solid(SolidPathInstance {
                model_matrix: instance.model_matrix,
                fill_color: instance.fill_color,
                stroke_color: instance.stroke_color,
                path_info: [
                    vertex_offset as f32,
                    geometry.vertex_count as f32,
                    instance.path_info[2],
                    instance.path_info[3],
                ],
                page_clip_bounds: instance.page_clip_bounds,
            }),
        }
    }

    fn pattern_draw_key(&self, p: &paint::PatternBody) -> String {
        retained_pattern_draw_key(&p.texture)
    }

    fn shader_image_draw_key(&self, t: &crate::seams::ShaderImageTexture) -> String {
        if self.retained {
            format!("{}|{:p}", t.key, std::sync::Arc::as_ptr(&t.rgba_straight))
        } else {
            t.key.clone()
        }
    }

    /// Shared "no mask" bind group: maskActive = 0, 1×1 white r-channel
    /// texture (`WebGPUPathRenderer.bindMaskGroup` inactive branch).
    fn create_inactive_mask_group(&self) -> Result<wgpu::BindGroup> {
        let device = &self.gpu.device;
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("inactive mask"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // MaskUniforms { maskActive: u32, maskOrigin: vec2f, maskInvSize: vec2f } — 24 B, pad 32.
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inactive mask uniforms"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        crate::scratch::write_buffer(device, &self.gpu.queue, &uniforms, 0, &[0u8; 32]);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("inactive mask group"),
                layout: &self.layouts.mask_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniforms.as_entire_binding(),
                    },
                ],
            }))
    }

    fn shader_module(&self, id: &str) -> Result<()> {
        let mut modules = self.shader_modules.borrow_mut();
        if !modules.contains_key(id) {
            // An agent fill's module (crate::agent_fill), or an extracted one.
            let source: std::borrow::Cow<'static, str> = if crate::agent_fill::is_agent_module(id) {
                crate::agent_fill::module_source(id)
                    .ok_or_else(|| anyhow!("no agent shader module '{id}' (its fill wasn't planned)"))?
                    .to_string()
                    .into()
            } else {
                moda_shaders::module_source(id)
                    .ok_or_else(|| anyhow!("no extracted WGSL module '{id}'"))?
                    .source
                    .into()
            };
            let module = self
                .gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(id),
                    source: wgpu::ShaderSource::Wgsl(moda_shaders::working_color::specialize(self.working.as_ref(), source)),
                });
            modules.insert(id.to_string(), module);
        }
        Ok(())
    }

    /// Compile the hot body/shape pipelines at BOTH
    /// editor sample counts (MSAA 1 and 4) up front, so toggling a backdrop-blur
    /// node — which flips the editor MSAA policy between 4 (normal) and 1 (a
    /// resolved single-sample target backdrop requires) — does not stall on a
    /// lazy `create_render_pipeline`. The census + rationale are in
    /// `docs/productionization.md` (the ~12-hot-pipeline list); this prewarms the
    /// non-text body pipelines that are reachable without a live scene
    /// (`page-fill`/`path-solid` are 100% of renders; shadow/gradient/pattern are
    /// the next tier). Best-effort: a prewarm miss just defers that one compile to
    /// first organic use, so errors are swallowed (never fatal to startup). Text /
    /// mask / procedural pipelines stay lazy (they need the text subsystem or a
    /// concrete module id, and the persisted shader cache covers them).
    pub fn prewarm_pipelines(&self) {
        for &requested in &[1u32, 4u32] {
            let sc = requested.min(self.max_frame_sample_count()).max(1);
            let _ = self.pipeline(
                "page-fill",
                0,
                sc,
                BlendMode::Normal,
                &self.layouts.page_layout,
            );
            let _ = self.pipeline(
                "path-solid",
                0,
                sc,
                BlendMode::Normal,
                &self.layouts.path_layout,
            );
            let _ = self.pipeline(
                "path-solid",
                1,
                sc,
                BlendMode::Normal,
                &self.layouts.path_layout,
            );
            let _ = self.pipeline(
                "path-shadow",
                1,
                sc,
                BlendMode::Normal,
                &self.layouts.path_layout,
            );
            for desc in [0usize, 1] {
                let _ = self.pipeline(
                    "path-gradient",
                    desc,
                    sc,
                    BlendMode::Normal,
                    &self.layouts.path_layout,
                );
                let _ = self.pipeline(
                    "path-pattern",
                    desc,
                    sc,
                    BlendMode::Normal,
                    &self.layouts.pattern_layout,
                );
            }
        }
    }

    /// Build (and cache) a render pipeline for `module_id`, using pipeline
    /// descriptor index `desc_index` from the manifest, with the given sample
    /// count and optional blend-mode override (manifest `blend_modes` data).
    fn pipeline(
        &self,
        module_id: &str,
        desc_index: usize,
        sample_count: u32,
        blend: BlendMode,
        layout: &wgpu::PipelineLayout,
    ) -> Result<()> {
        // Split-only modes (darken, …) outside a blend split fall back to a
        // fixed-function state; the cache key (`manifest_key`) is that state's.
        let blend = blend.fixed_function();
        let key = (
            module_id.to_string(),
            desc_index,
            sample_count,
            blend.manifest_key(),
            self.frame_format,
        );
        if self.pipelines.borrow().contains_key(&key) {
            return Ok(());
        }
        // M5b (docs/productionization.md): env-gated pipeline-compile stats.
        // Every cache miss here = one real create_render_pipeline — the data
        // the ~12-hot-pipeline prewarm list is derived from.
        if std::env::var_os("MODA_PIPELINE_STATS").is_some() {
            eprintln!(
                "moda-pipeline-stat: {module_id} desc={desc_index} msaa={sample_count} blend={}",
                blend.manifest_key()
            );
        }
        self.shader_module(module_id)?;
        let manifest = moda_shaders::manifest();
        // An agent fill uses its template fill's descriptors (crate::agent_fill::MANIFEST_TEMPLATE).
        let manifest_id = if crate::agent_fill::is_agent_module(module_id) {
            crate::agent_fill::manifest_id(module_id)
        } else {
            module_id.to_string()
        };
        let meta = manifest
            .modules
            .get(&manifest_id)
            .ok_or_else(|| anyhow!("module '{module_id}' missing from manifest"))?;
        let desc = meta.pipelines.get(desc_index).ok_or_else(|| {
            anyhow!("module '{module_id}' has no pipeline descriptor {desc_index}")
        })?;

        let modules = self.shader_modules.borrow();
        let module = &modules[module_id];

        let empty: Vec<moda_shaders::VertexBufferDesc> = Vec::new();
        let buffer_descs = desc
            .vertex
            .as_ref()
            .and_then(|v| v.buffers.as_ref())
            .unwrap_or(&empty);
        let attr_storage: Vec<Vec<wgpu::VertexAttribute>> = buffer_descs
            .iter()
            .map(|b| {
                b.attributes
                    .iter()
                    .map(|a| wgpu::VertexAttribute {
                        format: vertex_format(&a.format),
                        offset: a.offset,
                        shader_location: a.shader_location,
                    })
                    .collect()
            })
            .collect();
        let buffers: Vec<wgpu::VertexBufferLayout> = buffer_descs
            .iter()
            .zip(attr_storage.iter())
            .map(|(b, attrs)| wgpu::VertexBufferLayout {
                array_stride: b.array_stride,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: attrs,
            })
            .collect();

        let fragment = desc
            .fragment
            .as_ref()
            .ok_or_else(|| anyhow!("render pipeline lacks fragment state"))?;
        let blend_state = if blend == BlendMode::Normal {
            fragment.targets[0]
                .as_ref()
                .and_then(|t| t.blend.as_ref())
                .map(to_wgpu_blend)
        } else {
            let mut state = to_wgpu_blend(&manifest.blend_modes[blend.manifest_key()]);
            if blend == BlendMode::Multiply {
                // The manifest's multiply (dst, zero) keeps only cs·cb, so a
                // translucent or antialiased edge darkens instead of fading
                // out. W3C premultiplied multiply over an opaque backdrop is
                // cs·cb + cb·(1 − αs): the (dst, one-minus-src-alpha) state.
                state.color.dst_factor = wgpu::BlendFactor::OneMinusSrcAlpha;
            }
            Some(state)
        };
        let targets: Vec<Option<wgpu::ColorTargetState>> = fragment
            .targets
            .iter()
            .map(|t| {
                t.as_ref().map(|t| wgpu::ColorTargetState {
                    // Main-pass color targets render into the
                    // engine's frame attachment, whose format is per-engine.
                    format: frame_color_target_format(&t.format, self.frame_format),
                    blend: blend_state,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();

        // Blend-mode pipelines render in the transparent phase: depth-write
        // OFF (`WebGPUPathPipelineManager.ts:834-929`); otherwise the
        // manifest depth state applies verbatim.
        let depth_stencil = desc
            .depth_stencil
            .as_ref()
            .map(|d| wgpu::DepthStencilState {
                format: texture_format(&d.format),
                depth_write_enabled: Some(d.depth_write_enabled && blend == BlendMode::Normal),
                depth_compare: Some(compare_function(&d.depth_compare)),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            });

        let pipeline = moda_text_draw::device::create_render_pipeline(&self.gpu.device, &wgpu::RenderPipelineDescriptor {
                label: Some(&format!(
                    "{module_id}[{desc_index}] s{sample_count} {}",
                    blend.manifest_key()
                )),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(
                        desc.vertex
                            .as_ref()
                            .map(|v| v.entry_point.as_str())
                            .unwrap_or("vertexMain"),
                    ),
                    compilation_options: Default::default(),
                    buffers: &buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(&fragment.entry_point),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview_mask: None,
                // The pipeline host chooses the cache (moda_text_draw::device).
                cache: None,
            });
        drop(modules);
        std::sync::Arc::make_mut(&mut *self.pipelines.borrow_mut()).insert(key, pipeline);
        Ok(())
    }

    /// Build (and cache) a shape-SDF pipeline (`shape-solid` /
    /// `shape-gradient`). These two modules carry NO manifest pipeline
    /// descriptor — the WGSL is extracted but the native engine owns the
    /// pipeline state, hand-built here to mirror the WebGPU shape renderers: a
    /// unit quad with NO vertex buffers (the WGSL `vertexMain` derives the quad
    /// from `@builtin(vertex_index)`) over the 2-group `[frame, storage]` page
    /// layout, depth-write ON for the opaque variant (`desc_index == 0`) / OFF
    /// for the transparent variant (`desc_index == 1`).
    ///
    /// The opaque variant exists only under MSAA over an opaque clear and uses
    /// **alpha-to-coverage** (`fragmentMainA2C`, no blend), so the analytic
    /// edge coverage becomes a per-sample mask and depth is written only on
    /// covered samples — matching `WebGPUOpaqueShapeRenderer.shouldUseA2C`.
    /// Elsewhere opaque SDF bodies draw in the transparent phase (#106). The
    /// transparent variant uses `fragmentMainPremul` with premultiplied
    /// `(one, one-minus-src-alpha)` blend. Cached under
    /// `shape_variant(...)`, so `with_shape_pipeline` retrieves the matching one.
    fn shape_sdf_pipeline(
        &self,
        module_id: &str,
        desc_index: usize,
        sample_count: u32,
        clear_is_opaque: bool,
    ) -> Result<()> {
        let key = (
            module_id.to_string(),
            desc_index,
            sample_count,
            shape_variant(desc_index, sample_count, clear_is_opaque),
            self.frame_format,
        );
        if self.pipelines.borrow().contains_key(&key) {
            return Ok(());
        }
        self.shader_module(module_id)?;
        let modules = self.shader_modules.borrow();
        let module = &modules[module_id];
        let opaque = desc_index == 0;
        let use_a2c = opaque && sample_count > 1 && clear_is_opaque;
        // #106: the opaque depth-write lane is used only with A2C; a premul
        // depth-write variant would knock out what is behind its fringe.
        debug_assert!(!opaque || use_a2c, "opaque SDF pipeline without A2C");
        // Write mask is keyed on `use_a2c` ALONE — not `clear_is_opaque` directly —
        // so it is a pure function of `shape_variant`, this pipeline's cache-key
        // tag. Keying it on `clear_is_opaque` would let a cached "shape-premul"
        // pipeline built for one clear be reused for the other with the wrong mask
        // (the `Engine` pipeline cache outlives a single render, and pages/exports
        // with different background opacity share it), silently dropping alpha on a
        // transparent export. (pipeline-cache aliasing fix.)
        //
        // - A2C: the fragment's alpha IS the per-sample coverage mask; writing it
        //   would resolve a feathered edge into an opaque export, so mask it off
        //   (RGB). A2C only runs on an opaque clear, where the masked alpha stays at
        //   the clear's 1.0 — matching WebGPU's `writeMask = RGB`.
        // - Premul: write real alpha (RGBA). On a transparent clear this writes the
        //   body alpha; on an opaque clear the `(one, 1-src_a)` blend saturates alpha
        //   back to 1.0 regardless, so RGBA is pixel-identical to WebGPU's
        //   `writeMask = RGB` there while staying clear-independent.
        let write_mask = if use_a2c {
            wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN | wgpu::ColorWrites::BLUE
        } else {
            wgpu::ColorWrites::ALL
        };
        // Premul `(one, one-minus-src-alpha)` for the non-A2C variants (the
        // shader outputs premultiplied rgba). A2C disables blend (the coverage
        // mask does the compositing) and outputs unpremultiplied rgba.
        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let pipeline = moda_text_draw::device::create_render_pipeline(&self.gpu.device, &wgpu::RenderPipelineDescriptor {
                label: Some(&format!(
                    "{module_id}[{desc_index}] s{sample_count} {}",
                    if use_a2c { "a2c" } else { "premul" }
                )),
                layout: Some(&self.layouts.page_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vertexMain"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(if use_a2c {
                        "fragmentMainA2C"
                    } else {
                        "fragmentMainPremul"
                    }),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        // The engine's frame format — SDF bodies
                        // draw straight into the main-pass frame attachment.
                        format: self.frame_format,
                        blend: if use_a2c { None } else { Some(blend) },
                        write_mask,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(opaque),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    alpha_to_coverage_enabled: use_a2c,
                    ..Default::default()
                },
                multiview_mask: None,
                // The pipeline host chooses the cache (moda_text_draw::device).
                cache: None,
            });
        drop(modules);
        std::sync::Arc::make_mut(&mut *self.pipelines.borrow_mut()).insert(key, pipeline);
        Ok(())
    }

    /// Submit a coverage plan and start readback. Oversized readbacks retain the
    /// synchronous banded fallback; ordinary frames do not wait for GPU completion.
    pub fn render_plan_start(&self, plan: &CoveragePlan, pool: &crate::readback::ReadbackPool) -> Result<crate::readback::PendingRgba> {
        let _encode = crate::prof::span("gpu.encode_submit");
        let target = self.render_plan_to_texture(plan)?;
        let retries = if self.first_readback_done.get() {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES
        } else {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES_FIRST
        };
        crate::readback::read_texture_rgba_start(&self.gpu.device, &self.gpu.queue, &target, pool, plan.transparent, retries)
    }

    /// Finish a submitted readback, including format canonicalization and straight alpha.
    pub fn render_plan_finish(&self, pending: crate::readback::PendingRgba) -> Result<Vec<u8>> {
        let rgba = pending.finish(&self.gpu.device)?;
        self.first_readback_done.set(true);
        Ok(rgba)
    }

    /// Execute a coverage plan and read back tightly-packed RGBA bytes (the
    /// export path: native CLI / batch / PDF lanes). ALWAYS canonical RGBA
    /// regardless of the engine's frame format — a BGRA-frame engine's readback
    /// canonicalizes the texels before the un-premultiply below.
    pub fn render_plan(&self, plan: &CoveragePlan) -> Result<Vec<u8>> {
        let encode = crate::prof::span("gpu.encode_submit");
        let target = self.render_plan_to_texture(plan)?;
        drop(encode);
        let wait = crate::prof::span("gpu.wait");
        // Tolerate the GPU watchdog interrupting a slow first-render fence wait
        // with `PollError::Timeout` (NOT device loss — see
        // `device::is_device_lost_error`); re-poll a bounded number of times so
        // a legitimately slow compile settles instead of failing the render.
        // The first render after device creation compiles every pipeline it
        // touches inside this wait, so it gets the larger first-render budget;
        // subsequent renders drop back to steady state.
        let poll_retries = if self.first_readback_done.get() {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES
        } else {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES_FIRST
        };
        crate::readback::poll_until_complete_with_budget(&self.gpu.device, poll_retries)
            .context("device poll after submit")?;
        drop(wait);
        let _readback = crate::prof::span("gpu.readback");
        let mut rgba = crate::readback::read_texture_rgba_with_budget(
            &self.gpu.device,
            &self.gpu.queue,
            &target,
            plan.width_px,
            plan.height_px,
            poll_retries,
        )?;
        // The cold compile is paid; every later readback uses steady state.
        self.first_readback_done.set(true);
        if plan.transparent {
            crate::readback::unpremultiply_in_place(&mut rgba);
        }
        Ok(rgba)
    }

    /// [`Engine::render_plan`] with the GPU RGBA→NV12 conversion at the
    /// readback seam: same render, same submit; the finished texture is
    /// converted by `converter`'s compute pass and 1.5 bytes/px are read back
    /// instead of 4 (BT.709, limited range — see `nv12.rs` for the math and
    /// its measured relationship to the ffmpeg lane it replaces). Opaque
    /// plans only: NV12 carries no alpha, and the conversion white-composites
    /// exactly like the mp4 filter chain would.
    pub fn render_plan_nv12(
        &self,
        plan: &CoveragePlan,
        converter: &crate::nv12::Nv12Converter,
    ) -> Result<Vec<u8>> {
        if plan.transparent {
            anyhow::bail!("nv12 render: transparent plans have no NV12 representation");
        }
        let target = self.render_plan_to_texture(plan)?;
        let poll_retries = if self.first_readback_done.get() {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES
        } else {
            crate::readback::READBACK_POLL_TIMEOUT_RETRIES_FIRST
        };
        // No CPU-side poll between the render submit and the compute pass:
        // wgpu orders the compute pass's texture read after the render pass's
        // write on the queue itself, so the only synchronous wait this frame
        // needs is `convert`'s map poll — which carries the same retry budget
        // and therefore the same slow-first-frame tolerance the RGBA lane's
        // pre-readback poll provides.
        let nv12 = converter.convert(
            &self.gpu.device,
            &self.gpu.queue,
            &target,
            plan.width_px,
            plan.height_px,
            poll_retries,
        )?;
        self.first_readback_done.set(true);
        Ok(nv12)
    }

    /// Execute a coverage plan into the frame-format offscreen target texture
    /// and return it WITHOUT readback. Used by export/readback (PDF/screenshot) and
    /// by the remaining present BLIT lanes — the per-page direct-present path
    /// (`present_page`) and `present_plan`'s backdrop-split fallback — which
    /// sample this target into the canvas swapchain. The common `present_plan`
    /// path no longer uses this: it renders directly into the swapchain via
    /// `render_plan_to_view`. Identical GPU work to the readback path (same
    /// submit, same target); buffer-map readback is async-only on the webgpu
    /// backend, and presentation never needs it.
    pub fn render_plan_to_texture(&self, plan: &CoveragePlan) -> Result<wgpu::Texture> {
        // Offscreen (present_view: None) — `direct_present_ok` is moot; pass true.
        self.render_plan_to_texture_impl(plan, None, None, None, None, true)
            .map(|(t, _)| t)
    }

    /// Offscreen rendering with the caller's retained plan identity. Mask coverage
    /// and resolve-target lifetime are the same as ordinary offscreen rendering.
    pub fn render_retained_plan_to_texture(&self, plan: &std::sync::Arc<CoveragePlan>) -> Result<wgpu::Texture> {
        self.render_plan_to_texture_impl(plan, None, None, None, Some(PacketPlan::pinned(plan)), true)
            .map(|(texture, _)| texture)
    }

    /// Offscreen rendering of a plan its owner holds and writes in place. `key` names the plan's state; `change`
    /// says what the owner did to it since an earlier state. When the packet is of that earlier state it is
    /// brought to this one by writing the changed items' records (`patch_prepared`); a packet of this very state
    /// is replayed; anything else collects the plan afresh. Mask coverage and resolve-target lifetime are the
    /// same as ordinary offscreen rendering.
    pub fn render_owned_plan_to_texture(
        &self,
        plan: &CoveragePlan,
        key: PlanKey,
        change: Option<&PlanChange<'_>>,
    ) -> Result<wgpu::Texture> {
        let packet = PacketPlan { key, pinned: None, change };
        self.render_plan_to_texture_impl(plan, None, None, None, Some(packet), true)
            .map(|(texture, _)| texture)
    }

    /// Residency: render a CACHED combined camera plan at a fresh
    /// camera (pan/zoom) by overriding only its camera-variant fields
    /// (view-projection, target size, clear). The draw items are canvas-absolute
    /// and reused untouched, so a static pan/zoom never rebuilds draw lists
    /// (PLAN §3 #21 / risk R13). Identical GPU work to `render_plan_to_texture`.
    pub fn render_plan_to_texture_with_camera(
        &self,
        plan: &CoveragePlan,
        cam: &CameraPresentOverride,
    ) -> Result<wgpu::Texture> {
        // Offscreen (present_view: None) — `direct_present_ok` is moot; pass true.
        self.render_plan_to_texture_impl(plan, Some(cam), None, None, None, true)
            .map(|(t, _)| t)
    }

    /// Present fast path: render the (optionally camera-overridden) plan into the
    /// canvas swapchain view.
    ///
    /// When the plan has NO backdrop splits, the scene's MSAA pass resolves
    /// DIRECTLY into `present_view` (sample_count==1 renders into it) and this
    /// returns `Ok(None)` — the caller just presents, skipping the fullscreen-
    /// blit pass. (The offscreen target is still allocated via `frame_targets`
    /// (cached in retained mode) but goes unused as a render destination here.)
    ///
    /// Backdrop-split plans present direct too WHEN `present_copy_src`
    /// is supplied (the swapchain texture, advertised COPY_SRC — see
    /// `GpuContext::surface_copy_src`). `execute_split` then copies only the small
    /// backdrop region back OUT of the swapchain into a scratch, runs the effect,
    /// and composites the result directly into `present_view` — mirroring the
    /// WebGPU renderer (`WebGPUBackdropBlurRenderer`) and dropping the full-frame
    /// offscreen + blit. This also returns `Ok(None)`.
    ///
    /// Without `present_copy_src` (e.g. Dawn's Android Vulkan swapchains, which
    /// can't be a copy source), backdrop plans fall back to the offscreen path:
    /// `execute_split` reads the rendered scene back from the offscreen target
    /// mid-pass and this returns `Ok(Some(target))` for the caller to blit in.
    ///
    /// `direct_present_ok` says the `present_view`'s texture
    /// format matches the engine's frame format ([`Engine::frame_format`] —
    /// `Rgba8Unorm` by default, the browser-preferred format on the live wasm
    /// lane), so the scene MAY resolve/render directly into it. Pass `false`
    /// when the swapchain was configured with a DIFFERENT format: a direct MSAA
    /// resolve into a mismatched-format view is illegal (and letting the
    /// browser convert a non-preferred canvas format is the source of the
    /// Firefox-on-resize red↔blue swap), so this forces the offscreen path and
    /// returns `Ok(Some(target))` for the caller to blit in with the
    /// format-neutral present pipeline (channel-correct on every swapchain
    /// format).
    ///
    /// `prepared_plan` is the caller's cache-slot `Arc<CoveragePlan>` used as
    /// the prepared-frame packet identity (M2). Two lanes pass `Some`: the
    /// resident camera present (`finish_camera_frame`) and the isolated-target
    /// lane (`isolated::render_isolated_target`). The export/page-snapshot
    /// paths pass `None` and never engage the packet.
    pub fn render_plan_to_view(
        &self,
        plan: &CoveragePlan,
        cam: Option<&CameraPresentOverride>,
        present_view: &wgpu::TextureView,
        // The swapchain TEXTURE (must be the backing texture of
        // `present_view`) when the surface supports COPY_SRC, else `None`. Lets a
        // backdrop frame copy its region back out of the swapchain instead of
        // rendering to a full-frame offscreen.
        present_copy_src: Option<&wgpu::Texture>,
        prepared_plan: Option<&std::sync::Arc<CoveragePlan>>,
        // `true` when `present_view`'s format equals the
        // engine's frame format (present direct); `false` on a mismatch
        // (route offscreen + blit).
        direct_present_ok: bool,
    ) -> Result<Option<wgpu::Texture>> {
        let (target, presented_to_view) = self.render_plan_to_texture_impl(
            plan,
            cam,
            Some(present_view),
            present_copy_src,
            prepared_plan.map(PacketPlan::pinned),
            direct_present_ok,
        )?;
        Ok(if presented_to_view {
            None
        } else {
            Some(target)
        })
    }

    fn render_plan_to_texture_impl(
        &self,
        plan: &CoveragePlan,
        cam_override: Option<&CameraPresentOverride>,
        present_view: Option<&wgpu::TextureView>,
        present_copy_src: Option<&wgpu::Texture>,
        packet_plan: Option<PacketPlan<'_>>,
        direct_present_ok: bool,
    ) -> Result<(wgpu::Texture, bool)> {
        // The outermost render is one generation of `scratch`: what it and its captures take there (a capture's
        // target, buffers and filter passes; a non-retained render's own buffers) is handed back once the GPU has
        // done it. A non-retained render holds what it submits to its end; a retained engine's work goes to the
        // queue as it is recorded (`scratch`).
        let hold = !self.retained;
        self.scratch.enter(&self.gpu.device, hold);
        let out = self.render_plan_to_texture_body(plan, cam_override, present_view, present_copy_src, packet_plan, direct_present_ok);
        self.scratch.leave(&self.gpu.queue, hold);
        out
    }

    fn render_plan_to_texture_body(
        &self,
        plan: &CoveragePlan,
        cam_override: Option<&CameraPresentOverride>,
        present_view: Option<&wgpu::TextureView>,
        // COPY_SRC-capable swapchain texture backing `present_view`,
        // enabling the backdrop direct-present path. `None` keeps backdrop on the
        // offscreen + blit fallback. Ignored when `present_view` is `None`.
        present_copy_src: Option<&wgpu::Texture>,
        // M2: the plan's identity for the prepared-frame packet (`PacketPlan`): the caller's cache-slot plan
        // Arc (camera present or isolated-target lane), or an owned plan's key. `None` ⇒ never engage the packet.
        packet_plan: Option<PacketPlan<'_>>,
        // `false` forces the offscreen + blit path even for
        // a no-backdrop plan, because `present_view`'s format differs from the
        // engine's frame format (a mismatched direct resolve is illegal). Moot
        // when `present_view` is `None` (already offscreen).
        direct_present_ok: bool,
    ) -> Result<(wgpu::Texture, bool)> {
        let _profile = crate::prof::span("canvas.host.prepare_record");
        // M2: default to "upload normally"; a proven hit flips this true below
        // and `make_buffer` then reuses the unchanged pool buffers in place.
        self.pf_skip_uploads.set(false);
        // A layer capture's render: only `scissor` of its target is read (`capture_clip`); `window` is that part in
        // canvas units, where this render's own captures must be exact.
        let scissor = self.capture_scissor.take();
        let window = self.capture_window.take();
        let mut rendered_mask_views = crate::effects::SurfaceViews::new();
        for mask in &plan.masks {
            let Some(image) = &mask.image else { continue };
            let Some(rendered) = &image.rendered else { continue };
            let capture_scene = Scene::from_document((*rendered.document).clone())?;
            let scale = cam_override.map_or_else(
                || plan.camera_frame.as_ref().map_or(plan.scale, |camera| camera.zoom * plan.width_px as f64 / camera.viewport_w),
                |camera| camera.view.zoom * camera.width_px as f64 / camera.view.viewport_w,
            );
            let scale = crate::effects::mask::coverage_scale(scale);
            let limit = self.gpu.device.limits().max_texture_dimension_2d as f64;
            let texel_limit = (crate::effects::mask::MAX_MASK_TEXELS as f64 / (mask.size[0] * mask.size[1])).sqrt();
            let scale = scale.min(limit / mask.size[0]).min(limit / mask.size[1]).min(texel_limit);
            let camera = CameraParams {
                camera_x: mask.origin[0], camera_y: mask.origin[1], zoom: 1.0,
                viewport_w: mask.size[0], viewport_h: mask.size[1], dpr: scale,
                clear_rgb: [0.0; 3],
            };
            let options = RenderOptions {
                transparent: true, isolate: vec![rendered.node_id.clone()], scale,
                text_masks_available: Some(true),
                shader_time_s: cam_override.map_or(rendered.shader_time_s, |camera| camera.shader_time_s),
                working: self.working,
                ..RenderOptions::default()
            };
            let pages: Vec<_> = (0..capture_scene.doc().pages.len()).collect();
            let Some((mut capture_plan, warnings)) = build_camera_plan(&capture_scene, &pages, &camera, &options)? else {
                bail!("could not build rendered mask {}", rendered.node_id);
            };
            warnings.ensure_drawn().with_context(|| format!("rendered mask {}", rendered.node_id))?;
            capture_plan.clear_rgba = Some([0.0; 4]);
            let clip = [mask.origin[0] as f32, mask.origin[1] as f32, (mask.origin[0] + mask.size[0]) as f32, (mask.origin[1] + mask.size[1]) as f32];
            widen_captured_clips(&capture_scene, &mut capture_plan.items, &rendered.node_id, clip);
            self.capture_depth.set(self.capture_depth.get() + 1);
            // The mask's render has submitted everything that uses its buffers and textures by the time it returns.
            let scope = self.scratch.mark();
            let result = self.render_plan_to_texture(&capture_plan);
            // All but the mask's own target, which this render's mask pass reads.
            self.scratch.release(scope, result.as_ref().ok());
            self.capture_depth.set(self.capture_depth.get() - 1);
            let texture = result?;
            rendered_mask_views.insert(image.surface.clone().expect("rendered mask surface"), texture.create_view(&Default::default()));
        }
        let device = &self.gpu.device;
        let queue = &self.gpu.queue;
        // Paint surfaces: regenerate the mips under the tiles uploaded since the last render, once for all of
        // them, before anything this render samples (bundle replays included). A per-render cache creates its
        // textures (and their mips) fresh. A nested render (a capture) finds the cache borrowed: the outer render
        // already flushed.
        if let Ok(mut cache) = self.retained_patterns.try_borrow_mut() {
            if let Some(cache) = cache.as_mut() {
                cache.flush_surface_mips(device, queue);
            }
        }
        // The (viewport × dpr) target size is camera-variant — it
        // comes from the override on the cached-plan pan/zoom fast path, else
        // from the plan. Draw geometry is canvas-absolute and identical either
        // way; only the projection + target + clear differ.
        let eff_width_px = cam_override.map(|c| c.width_px).unwrap_or(plan.width_px);
        let eff_height_px = cam_override.map(|c| c.height_px).unwrap_or(plan.height_px);
        // The live renderer clock re-threaded onto cached procedural-
        // fill instances below (`Some` only on the camera present path; the
        // export / per-page path keeps the plan's baked clock). Non-finite is
        // mapped to 0.0 to match the packer's `resolve_shader_time_seconds`, so a
        // re-threaded HIT stays byte-identical to a fresh build at any clock.
        let present_shader_time = cam_override.map(|c| {
            if c.shader_time_s.is_finite() {
                c.shader_time_s
            } else {
                0.0
            }
        });
        // Device-resilience clamp: the plan is GPU-agnostic and asks for the
        // configured MSAA count (default 4), but some adapters (Adreno /
        // Android-Vulkan tiers) do NOT advertise 4× for every multisampled
        // attachment format in use (notably Depth24Plus). Creating the depth
        // texture / pipelines at an unsupported count fails the whole pass.
        // Clamp to what THIS adapter supports for all those formats (computed
        // once at GpuContext creation; desktop/sim stay at 4; format-aware for
        // the engine's frame format). This single local feeds the
        // MSAA color + depth textures (`frame_targets`) AND every pipeline
        // `MultisampleState.count`, so they always agree.
        let sample_count = plan.sample_count.min(self.max_frame_sample_count()).max(1);

        // The effective clear color for the pass (consumed by the first-segment
        // clear in the encode section below) — computed ONCE here so the A2C
        // decision derives from the SAME value (no duplicated formula to drift).
        // The override's clear wins on the cached-plan pan/zoom path.
        let bg = if plan.transparent { 0.0 } else { 1.0 };
        let clear_color = cam_override
            .map(|c| c.clear_rgba)
            .or(plan.clear_rgba)
            .unwrap_or([bg, bg, bg, bg]);
        // The clear is an authored sRGB colour like any paint (opaque, or all zero).
        let clear_color = self.working.map_or(clear_color, |working| working.rgba(clear_color));
        // Alpha-to-coverage for the opaque analytic SDF body is valid
        // ONLY over an opaque clear (the `writeAlpha` guard). The SDF quad
        // covers every MSAA sample of an edge pixel, so without A2C the analytic
        // edge coverage would feather via premul blend + full-pixel depth-write
        // (occlusion fringe) — A2C instead turns the coverage into a per-sample
        // mask, matching WebGPU `shouldUseA2C = sampleCount>1 && !writeAlpha`.
        // Over a transparent (alpha-0) clear A2C double-attenuates the edge
        // samples, so those renders fall back to premul (matching WebGPU).
        let clear_is_opaque = clear_color[3] >= 1.0;
        // A changed plan whose packet can be patched in place (`patch_prepared`): only its changed items' bytes
        // are written and the packet adopts the plan, so the frame below is a hit (a bundle replay).
        let can_patch_target = present_view.is_some()
            || (self.capture_depth.get() == 0 && scissor.is_none() && window.is_none());
        if self.retained && self.prepared_patch.get() && can_patch_target {
            if let Some(new) = packet_plan {
                // The plan the packet would adopt: the shared plan itself, or the owned plan being drawn.
                let new_plan: &CoveragePlan = new.pinned.map_or(plan, |pinned| &**pinned);
                // Items written in place bring no clock-driven fill the packet does not know (a plan that gains
                // one is not kept by its owner); any other plan is asked.
                let written = matches!(new.change, Some(PlanChange { items: ChangedItems::Written { .. }, .. }));
                let asked = (!written).then(|| new_plan.items.iter().any(|item| match item {
                    DrawItem::Body { paint: BodyPaint::Shader(body), .. } =>
                        body.clock_offsets.as_ref().is_some_and(|offsets| !offsets.is_empty()),
                    _ => false,
                }));
                let candidate = {
                    let pf = self.prepared_frame.borrow();
                    pf.as_ref().is_some_and(|p| {
                        let new_has_clock_fill = asked.unwrap_or(p.has_clock_fill);
                        p.key != new.key
                            // Offscreen procedural clocks remain on the full
                            // preparation path; live clocks use the override.
                            && p.has_clock_fill == new_has_clock_fill
                            && (present_view.is_some() || !new_has_clock_fill)
                            && p.live_masks == present_view.is_some()
                            && p.width == eff_width_px
                            && p.height == eff_height_px
                            && p.sample_count == sample_count
                            && p.clear_is_opaque == clear_is_opaque
                            && p.pool_epoch == self.pool_epoch.get()
                            && (!p.has_clock_fill || p.shader_time_s == present_shader_time.unwrap_or(0.0))
                    })
                };
                if candidate {
                    match self.patch_prepared(new, new_plan, sample_count, clear_is_opaque, present_shader_time) {
                        Ok(()) => self.patch_fallback_reason.set(""),
                        Err(why) => {
                            self.patch_fallback_reason.set(why);
                            let mut st = self.prepared_stats.get();
                            st.patch_fallbacks = st.patch_fallbacks.wrapping_add(1);
                            self.prepared_stats.set(st);
                        }
                    }
                }
            }
        }
        // A packet for another plan can't serve this one, and presenting this plan
        // replaces it: let it (and the old plan it pins) go before this frame packs its
        // buffers, not after. After an open's first frame the old plan is the whole
        // document's; holding it through the next frame's packing stacked two frames.
        // Its prepared text is kept aside (`text_carry`): the new plan may draw the same text batches.
        let mut text_carry: Option<TextCarry> = None;
        if let Some(new) = packet_plan {
            let mut pf = self.prepared_frame.borrow_mut();
            if pf.as_ref().is_some_and(|p| p.key != new.key) {
                if let Some(p) = pf.take() {
                    text_carry = Some(TextCarry {
                        text: p.text,
                        batches: p.text_batches,
                        host_generation: p.text_host_generation,
                        sample_count: p.sample_count,
                    });
                    if let Some(pinned) = p.pinned {
                        self.defer_drop(Box::new(pinned));
                    }
                }
            }
        }

        // Layer-capture PRE-pass (layer filters and capture shadows): for each
        // capture, render its mini-plan through this very function (filtered
        // descendants recursively render before their parent), then filter the
        // capture target into a per-capture result texture. All
        // of this is recorded and submitted BEFORE any of the main pass's GPU work
        // below, each step in its own command buffer, in that order (at once, or
        // together when the render ends: `scratch`): a source runs before whatever reads it. No pass
        // split, no MSAA constraint — the composite is just another draw in the
        // main pass.
        //
        // RE-ENTRANCY INVARIANT: because of this recursion, any per-frame
        // engine state this method touches must tolerate a mid-call recursive
        // invocation, and because a holding render's command buffers wait to be
        // submitted together, nothing a capture's render writes may be written
        // again before the frame ends. That holds by construction: the recursive call passes
        // `packet_plan: None` and is not resident (`Engine::resident`), so its
        // buffers, uniform and targets are taken from the frame's generation
        // for that one use (they are handed back for the next capture only
        // when nothing is held: `ScratchPool::release`), and it leaves the
        // engine's labelled pool, pool
        // epoch and vertex residency to the outermost render;
        // `pf_skip_uploads` / `prepared_frame` borrows are not held across the
        // call; the MSAA and depth attachments captures of one size share are
        // cleared by each pass that draws into them.
        let mut layer_blur_results: std::collections::HashMap<String, wgpu::TextureView> =
            std::collections::HashMap::new();
        // Composites drawn from another zoom step's result: the region it covers.
        let mut layer_blur_regions: std::collections::HashMap<String, [f64; 4]> =
            std::collections::HashMap::new();
        let outermost = self.capture_depth.get() == 0;
        // Whether the plan has captures; which are on screen this frame; whether this frame drew the prepared
        // frame's captures again (`PacketCaptures`); whether a packet may keep them (every capture content-keyed,
        // no zoom-step stand-in, the live retained lane).
        let mut has_composites = false;
        let mut capture_on_screen: Vec<bool> = Vec::new();
        let mut captures_reused = false;
        let mut captures_packetable = false;
        let layer_blur_renderer = {
            let composites: Vec<(&String, &std::sync::Arc<crate::seams::LayerBlurGroup>)> = plan
                .items
                .iter()
                .filter_map(|item| match item {
                    DrawItem::LayerBlurComposite { node_id, group, .. } => Some((node_id, group)),
                    _ => None,
                })
                .collect();
            if composites.is_empty() {
                None
            } else {
                let renderer = self.layer_blur_renderer(sample_count)?;
                // The frame's projection (the main pass's VP below, tiles
                // aside): a capture whose composite lands wholly off screen is
                // not rendered — its draw is skipped like a failed one's, and
                // a cached plan served for a later camera renders it when it
                // comes into view. A camera plan holds every capture of its
                // window, so this keeps a rebuild frame to what it shows.
                let view_vp = plan.region_viewport.is_none().then(|| match cam_override {
                    Some(c) => view_projection(
                        c.view.camera_x,
                        c.view.camera_y,
                        c.view.zoom,
                        c.view.viewport_w,
                        c.view.viewport_h,
                    ),
                    None => match &plan.camera_frame {
                        Some(cf) => view_projection(
                            cf.camera_x,
                            cf.camera_y,
                            cf.zoom,
                            cf.viewport_w,
                            cf.viewport_h,
                        ),
                        None => view_projection(
                            plan.camera[0],
                            plan.camera[1],
                            1.0,
                            plan.page_size[0],
                            plan.page_size[1],
                        ),
                    },
                });
                let on_screen = |group: &crate::seams::LayerBlurGroup| {
                    let Some(m) = view_vp else { return true };
                    let [x, y, w, h] = group.region_canvas;
                    let ndc = |px: f64, py: f64| {
                        (
                            m[0] as f64 * px + m[4] as f64 * py + m[12] as f64,
                            m[1] as f64 * px + m[5] as f64 * py + m[13] as f64,
                        )
                    };
                    let (a, b) = (ndc(x, y), ndc(x + w, y + h));
                    let (x0, x1) = (a.0.min(b.0), a.0.max(b.0));
                    let (y0, y1) = (a.1.min(b.1), a.1.max(b.1));
                    // Not finite ⇒ can't tell ⇒ render it.
                    !(x1 < -1.0 || x0 > 1.0 || y1 < -1.0 || y0 > 1.0)
                };
                // Where this render's output is read, in canvas units: the whole target for the outermost render (an
                // axis-aligned view), a layer capture's `window` for its own.
                let level_window: Option<[f64; 4]> = if outermost {
                    view_vp.and_then(|m| {
                        let (a, e, b, f) = (m[0] as f64, m[12] as f64, m[5] as f64, m[13] as f64);
                        if m[1] != 0.0 || m[4] != 0.0 || a == 0.0 || b == 0.0 {
                            return None;
                        }
                        let (x0, x1) = ((-1.0 - e) / a, (1.0 - e) / a);
                        let (y0, y1) = ((-1.0 - f) / b, (1.0 - f) / b);
                        Some([x0.min(x1), y0.min(y1), (x1 - x0).abs(), (y1 - y0).abs()])
                    })
                } else {
                    window
                };
                // Zoom-step smoothing (`layer_blur::cache` stand-ins): the live
                // lane's budget of new captures this frame.
                let budget = self
                    .capture_frame_budget
                    .get()
                    .filter(|_| self.retained && outermost);
                let mut rendered: u32 = 0;
                let mut stood_in = false;
                has_composites = true;
                capture_on_screen = composites.iter().map(|(_, g)| on_screen(g)).collect();
                // The prepared frame of this very plan drew these composites: its results are this frame's.
                let packet_results = packet_plan
                    .filter(|_| self.retained && outermost && present_view.is_some())
                    .and_then(|pp| {
                        let pf = self.prepared_frame.borrow();
                        pf.as_ref()
                            .filter(|p| p.key == pp.key)
                            .and_then(|p| p.captures.as_ref())
                            .filter(|c| c.on_screen == capture_on_screen)
                            .map(|c| c.results.clone())
                    });
                if let Some(results) = packet_results {
                    layer_blur_results = results;
                    captures_reused = true;
                }
                for (k, (node_id, group)) in composites.iter().copied().enumerate() {
                    if captures_reused || !capture_on_screen[k] {
                        continue;
                    }
                    // A result the retained engine kept from an earlier frame
                    // (same content key ⇒ same pixels): no capture, no filter.
                    let key = group.cache_key.filter(|_| self.retained);
                    if let Some(view) = key.and_then(|k| self.capture_cache.borrow_mut().get(k)) {
                        layer_blur_results.insert(node_id.clone(), view);
                        continue;
                    }
                    // Mid-zoom: the same capture at another zoom step stands in when it covers what the target shows of
                    // this one (a zoom out past its region captures anew).
                    if self.capture_stand_ins_mid_zoom.get() && self.retained && outermost {
                        let [x, y, w, h] = group.region_canvas;
                        let needed = match level_window {
                            Some([wx, wy, ww, wh]) => {
                                let (x0, y0) = (x.max(wx), y.max(wy));
                                let (x1, y1) = ((x + w).min(wx + ww), (y + h).min(wy + wh));
                                [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)]
                            }
                            None => group.region_canvas,
                        };
                        let prior = group
                            .identity_key
                            .and_then(|i| self.capture_cache.borrow_mut().stand_in_covering(i, node_id, needed));
                        if let Some((view, region)) = prior {
                            layer_blur_results.insert(node_id.clone(), view);
                            layer_blur_regions.insert(node_id.clone(), region);
                            stood_in = true;
                            continue;
                        }
                    }
                    // Over the budget: the same capture at another zoom step
                    // stands in, over its own region, until a later frame
                    // renders this one.
                    if budget.is_some_and(|b| rendered >= b) {
                        let prior = group
                            .identity_key
                            .and_then(|i| self.capture_cache.borrow_mut().stand_in(i, node_id));
                        if let Some((view, region)) = prior {
                            layer_blur_results.insert(node_id.clone(), view);
                            layer_blur_regions.insert(node_id.clone(), region);
                            stood_in = true;
                            continue;
                        }
                    }
                    rendered += 1;
                    // Per-node degradation: a failed capture/blur skips this
                    // node's composite (the bind-group builder below sees no
                    // result and the draw is skipped) — the node doesn't paint
                    // this frame, but one bad node never kills the whole
                    // render and never wrong-paints.
                    // The capture's renders and filters submit everything that uses their buffers and textures before
                    // the next capture starts: it takes the same ones (`ScratchPool::release`).
                    let scope = self.scratch.mark();
                    let mut mini_plan = crate::layer_blur::exec::build_capture_plan(node_id, group);
                    // A supersampled single-sample capture (a 3D unit) takes its shadows' coverage at one sample too.
                    if plan.sample_count == 1 && crate::layer_blur::exec::supersampled_capture() {
                        mini_plan.sample_count = 1;
                    }
                    // Only the part of the capture the target shows (with its filters' reach) is drawn and filtered.
                    // A capture kept for later frames (`key`) is drawn whole, and so is a blurred one: its result is
                    // read past the window this assumes (#2275: a top-level uniform blur differed with any margin
                    // under 30× its reach), while noise and shadow captures are read only inside it.
                    let clip = level_window
                        .filter(|_| key.is_none() && group.filters.blur.is_none())
                        .and_then(|w| capture_clip(group, w));
                    self.capture_depth.set(self.capture_depth.get() + 1);
                    // The noise's shape coverage first, copied out of the
                    // shared target the capture renders into next.
                    let coverage = if group.coverage.is_empty() {
                        Ok(None)
                    } else {
                        let shape = crate::seams::LayerBlurGroup {
                            items: group.coverage.clone(),
                            coverage: Vec::new(),
                            ..(**group).clone()
                        };
                        let plan = crate::layer_blur::exec::build_capture_plan(node_id, &shape);
                        self.capture_scissor.set(clip.map(|c| c.0));
                        self.capture_window.set(clip.map(|c| c.1));
                        self.render_plan_to_texture_impl(&plan, None, None, None, None, true)
                            .map(|(tex, _)| {
                                Some(renderer.copy_capture(device, queue, &self.scratch, &tex, group))
                            })
                    };
                    let capture = coverage.and_then(|coverage| {
                        self.capture_scissor.set(clip.map(|c| c.0));
                        self.capture_window.set(clip.map(|c| c.1));
                        self.render_plan_to_texture_impl(&mini_plan, None, None, None, None, true)
                            .and_then(|(capture_tex, _)| {
                                renderer.filter_capture(
                                    device,
                                    queue,
                                    &self.scratch,
                                    clip.map(|c| c.0),
                                    &capture_tex,
                                    group,
                                    coverage.as_ref(),
                                )
                            })
                    });
                    self.capture_depth.set(self.capture_depth.get() - 1);
                    // All but the capture's result, which this render's composite reads.
                    self.scratch.release(scope, capture.as_ref().ok().map(|result| result.texture()));
                    match capture {
                        Ok(result) => {
                            if let Some(k) = key {
                                // Kept for later frames: the result leaves the frame's generation.
                                self.scratch.keep(result.texture());
                                self.capture_cache.borrow_mut().put_with(
                                    k,
                                    result.clone(),
                                    group.width_px,
                                    group.height_px,
                                    group.region_canvas,
                                    group.identity_key.map(|i| (i, node_id.as_str())),
                                );
                            }
                            layer_blur_results.insert(node_id.clone(), result);
                        }
                        Err(e) => {
                            // An agent shader effect that fails an export (over its time budget, or refused by the
                            // device) fails the render: never a silently missing layer (a preview draws magenta).
                            if !crate::pixel::is_preview()
                                && e.downcast_ref::<crate::pixel::PixelError>().is_some_and(|p| matches!(p, crate::pixel::PixelError::Shader { .. }))
                            {
                                return Err(e);
                            }
                            eprintln!(
                                "layer-blur capture failed for node '{node_id}' \
                                 ({}x{}): {e:#} — node skipped this frame",
                                group.width_px, group.height_px
                            );
                        }
                    }
                }
                if outermost {
                    self.captures_pending.set(stood_in);
                }
                captures_packetable = self.retained
                    && outermost
                    && !stood_in
                    && plan.items.iter().all(|item| match item {
                        DrawItem::LayerBlurComposite { group, .. } => group.cache_key.is_some(),
                        _ => true,
                    });
                Some(renderer)
            }
        };

        // Pipelines: page-fill (desc 0), path-solid opaque (desc 0) /
        // transparent (desc 1), plus any blend-mode variants in the plan.
        self.pipeline(
            "page-fill",
            0,
            sample_count,
            BlendMode::Normal,
            &self.layouts.page_layout,
        )?;
        self.pipeline(
            "path-solid",
            0,
            sample_count,
            BlendMode::Normal,
            &self.layouts.path_layout,
        )?;
        self.pipeline(
            "path-solid",
            1,
            sample_count,
            BlendMode::Normal,
            &self.layouts.path_layout,
        )?;
        let mut has_shadows = false;
        let mut has_gradients = false;
        let mut has_patterns = false;
        // Analytic SDF body modules present in the plan.
        let mut has_solid_sdf = false;
        let mut has_gradient_sdf = false;
        // M3f: procedural-fill module ids in the plan (pipeline pair per id).
        let mut shader_modules_in_plan: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        for item in &plan.items {
            match item {
                DrawItem::Body {
                    blend_mode, paint, ..
                } => {
                    // M3b: per-paint pipeline modules; blend variants ride the
                    // manifest blend_modes data (getBlendModePipeline).
                    let (module, layout): (&str, &wgpu::PipelineLayout) = match paint {
                        BodyPaint::Gradient(_) | BodyPaint::PostShaderGradientStroke(_) => {
                            has_gradients = true;
                            ("path-gradient", &self.layouts.path_layout)
                        }
                        BodyPaint::Pattern(_) => {
                            has_patterns = true;
                            ("path-pattern", &self.layouts.pattern_layout)
                        }
                        BodyPaint::Shader(body) => {
                            shader_modules_in_plan.insert(body.module_id.clone());
                            // M4e: image-backed modules use layout 43
                            // ([frame, storage, image, mask]).
                            let layout = if module_is_image_backed(&body.module_id) {
                                &self.layouts.image_shader_layout
                            } else {
                                &self.layouts.path_layout
                            };
                            (body.module_id.as_str(), layout)
                        }
                        _ => ("path-solid", &self.layouts.path_layout),
                    };
                    if *blend_mode != BlendMode::Normal {
                        self.pipeline(module, 1, sample_count, *blend_mode, layout)?;
                    }
                }
                DrawItem::Shadow { .. } => has_shadows = true,
                DrawItem::SolidShapeSdf { .. } => has_solid_sdf = true,
                DrawItem::GradientShapeSdf { .. } => has_gradient_sdf = true,
                _ => {}
            }
        }
        // M3f: shader pipelines (desc 0 opaque / desc 1 transparent) from the
        // extracted manifest descriptors; bind groups [frame, storage, mask]
        // (non-image fills — mask at group 2, `manifest.pinned_constants`).
        // M4e: image-backed modules take [frame, storage, image, mask].
        for module_id in &shader_modules_in_plan {
            let layout = if module_is_image_backed(module_id) {
                &self.layouts.image_shader_layout
            } else {
                &self.layouts.path_layout
            };
            for desc in [0, 1] {
                self.pipeline(module_id, desc, sample_count, BlendMode::Normal, layout)?;
            }
        }
        // M3b: gradient/pattern pipelines (desc 0 = opaque depth-write,
        // desc 1 = transparent) from the extracted manifest descriptors.
        if has_gradients {
            for desc in [0, 1] {
                self.pipeline(
                    "path-gradient",
                    desc,
                    sample_count,
                    BlendMode::Normal,
                    &self.layouts.path_layout,
                )?;
            }
        }
        if has_patterns {
            for desc in [0, 1] {
                self.pipeline(
                    "path-pattern",
                    desc,
                    sample_count,
                    BlendMode::Normal,
                    &self.layouts.pattern_layout,
                )?;
            }
        }
        // M3d: shadows always render via the TRANSPARENT shadow pipeline
        // (desc 1, depth-write OFF) — `renderTransparentItemsWithText` binds
        // `getPipeline('shadow', false)` for every shadow batch.
        if has_shadows {
            self.pipeline(
                "path-shadow",
                1,
                sample_count,
                BlendMode::Normal,
                &self.layouts.path_layout,
            )?;
        }
        // Analytic SDF body pipelines (desc 0 = opaque depth-write,
        // alpha-to-coverage; desc 1 = transparent premul). These two modules
        // have no manifest descriptor, so they use the dedicated hand-built
        // pipeline (see `shape_sdf_pipeline`) over the 2-group [frame,
        // storage] page layout. desc 0 exists only under MSAA over an opaque
        // clear: every other opaque SDF body draws in the transparent phase
        // (#106, see `sdf_opaque_lane`).
        let sdf_descs: &[usize] = if sample_count > 1 && clear_is_opaque {
            &[0, 1]
        } else {
            &[1]
        };
        if has_solid_sdf {
            for &desc in sdf_descs {
                self.shape_sdf_pipeline("shape-solid", desc, sample_count, clear_is_opaque)?;
            }
        }
        if has_gradient_sdf {
            for &desc in sdf_descs {
                self.shape_sdf_pipeline("shape-gradient", desc, sample_count, clear_is_opaque)?;
            }
        }

        // Frame uniform (VP matrix).
        // Page-local export path: the LOGICAL page size projection (zoom 1,
        // camera at page origin); the physical pixel scale comes solely from
        // the ceil()ed target size (`WebGPUExportCoordinator.ts:614-647`).
        // Live-viewport: when a `camera_frame` is set the plan holds
        // N pages in canvas-absolute space, so project with the editor camera
        // (zoom + cameraX/Y) over the logical viewport; the device-pixel scale
        // still comes solely from the (viewport × dpr) target size.
        // The override's view wins on the cached-plan pan/zoom fast
        // path (the cached plan's baked `camera_frame` is the stale build-frame
        // camera); else fall back to the plan's own camera_frame / page-local VP.
        // Tiled export: the band's VP under a viewport shifted to the tile
        // (`RegionViewport`: tiles of one band rasterize identically up to an
        // integer shift). A viewport may extend past the target, within the
        // device's limits (width <= max texture dimension, origin >= -2x it);
        // past them the tile keeps its translated VP (`camera_frame`).
        let region_viewport = plan.region_viewport.filter(|r| {
            let max = device.limits().max_texture_dimension_2d;
            cam_override.is_none()
                && r.raster_w <= max
                && r.raster_h <= max
                && r.shift_x <= max.saturating_mul(2)
                && r.shift_y <= max.saturating_mul(2)
        });
        let vp = match cam_override {
            _ if region_viewport.is_some() => match region_viewport.and_then(|r| r.band) {
                Some(b) => view_projection(b.camera_x, b.camera_y, b.zoom, b.viewport_w, b.viewport_h),
                None => view_projection(
                    plan.camera[0],
                    plan.camera[1],
                    1.0,
                    plan.page_size[0],
                    plan.page_size[1],
                ),
            },
            Some(c) => view_projection(
                c.view.camera_x,
                c.view.camera_y,
                c.view.zoom,
                c.view.viewport_w,
                c.view.viewport_h,
            ),
            None => match &plan.camera_frame {
                Some(cf) => view_projection(
                    cf.camera_x,
                    cf.camera_y,
                    cf.zoom,
                    cf.viewport_w,
                    cf.viewport_h,
                ),
                None => view_projection(
                    plan.camera[0],
                    plan.camera[1],
                    1.0,
                    plan.page_size[0],
                    plan.page_size[1],
                ),
            },
        };
        let (frame_uniform, frame_bg) = self.frame_uniform_and_bg(&vp);
        // A hit whose packet holds its pass as a render bundle: the camera uniform (just written) is all that changed,
        // so replay the bundle into this frame's targets and skip collecting, packing and recording every draw.
        // While the canvas moves, small vector text on a live present draws as word bars (see
        // `Engine::set_motion_text_greek_px`).
        let greek_px = match (self.capture_depth.get() == 0, present_view.is_some(), cam_override) {
            (true, true, Some(o)) if o.view.viewport_w > 0.0 => {
                let px = self.motion_text_greek_px.get();
                (px, o.view.zoom * f64::from(o.width_px) / o.view.viewport_w)
            }
            _ => (0.0, 0.0),
        };
        let (greek_sig, greek_picks) = self.greek_selection(plan, greek_px.0, greek_px.1);
        let greek_arcs: Vec<Option<std::sync::Arc<crate::seams::TextQuadBatch>>> = if greek_sig == 0 {
            Vec::new()
        } else {
            let mut k = 0;
            plan.items
                .iter()
                .filter_map(|item| match item {
                    DrawItem::Text(b) => {
                        let pick = greek_picks[k];
                        k += 1;
                        Some(if pick { self.greek_of(b) } else { None })
                    }
                    _ => None,
                })
                .collect()
        };
        self.greeked_text.set(greek_arcs.iter().filter(|g| g.is_some()).count() as u32);
        if let Some(bundle) = self.replayable_bundle(
            packet_plan,
            present_view.is_none() || direct_present_ok,
            present_view.is_some(),
            eff_width_px,
            eff_height_px,
            sample_count,
            clear_is_opaque,
            present_shader_time.unwrap_or(0.0),
            greek_sig,
            !has_composites || captures_reused,
        ) {
            let (target, msaa, depth_tex) = self.frame_targets(eff_width_px, eff_height_px, sample_count);
            let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let pv = present_view.unwrap_or(&target_view);
            let msaa_view = msaa
                .as_ref()
                .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let (color_view, resolve_view) = match msaa_view.as_ref() {
                Some(mv) => (mv, Some(pv)),
                None => (pv, None),
            };
            let _replay_profile = crate::prof::span("canvas.host.replay_record_submit");
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame bundle replay"),
            });
            {
                let ts = crate::gpu_profile::pass_timestamps();
                let mut pass = crate::gpu_profile::CountedPass::new(encoder.begin_render_pass(
                    &wgpu::RenderPassDescriptor {
                        label: Some("export pass (bundle)"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target: resolve_view,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: clear_color[0] as f64,
                                    g: clear_color[1] as f64,
                                    b: clear_color[2] as f64,
                                    a: clear_color[3] as f64,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: crate::gpu_profile::render_writes(&ts),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    },
                ));
                pass.execute_bundles(std::iter::once(&bundle));
            }
            crate::scratch::submit(queue, [encoder.finish()]);
            if self.capture_depth.get() == 0 {
                self.simplified_blurs.set(0);
            }
            self.prepass_fills.set(0);
            let _ = (&frame_uniform, &frame_bg);
            let mut s = self.prepared_stats.get();
            s.hits = s.hits.wrapping_add(1);
            s.bundle_replays = s.bundle_replays.wrapping_add(1);
            self.prepared_stats.set(s);
            if let Some(carry) = text_carry.take() {
                self.defer_drop(Box::new(carry));
            }
            return Ok((target, present_view.is_some()));
        }

        // --- Collect draw data ---------------------------------------------
        let packing_profile = crate::prof::span("canvas.host.pack");
        let mut page_instances: Vec<PageInstance> = Vec::new();
        /// M3b: which pipeline family a body draw uses. Intra-bucket slot
        /// order is solids → patterns → gradients (`renderUnifiedPathBatch`:
        /// non-gradient batch first — solids then patterns — then gradients).
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum PaintDrawKind {
            Solid,
            Gradient,
            Pattern,
            /// M3f: procedural fill — pipeline keyed by `PathDraw::shader`.
            Shader,
            /// Q-116: shader-fill gradient-stroke overlay — path-gradient
            /// pipeline, but in its own slot AFTER custom shaders
            /// (`postShaderGradientStrokePaths`).
            PostShaderGradientStroke,
        }
        #[derive(Clone)]
        struct PathDraw {
            /// M3b: pipeline family; `instance_index` indexes that family's
            /// own storage buffer.
            kind: PaintDrawKind,
            /// M3f: procedural-fill module id (kind == Shader).
            shader: Option<String>,
            instance_index: u32,
            vertex_offset: u32,
            vertex_count: u32,
            /// M3d: EXACT z as f64 bits (production buckets transparent items
            /// by the raw fractional zIndex — inner shadows at z + 0.5 live
            /// in their own bucket ABOVE the body). Monotonic for z >= 0.
            z_bucket: u64,
            blend: BlendMode,
            /// M3d: nearest ancestor mask (index into plan.masks) bound at
            /// MASK_BIND_GROUP_INDEX for this draw.
            mask: Option<u64>,
            /// M3b: pattern texture bind-group key (texture key, clamp?).
            tex: Option<(String, bool)>,
            /// M3e: pass segment (count of BackdropSplits seen before this
            /// draw). Segment N draws between splits N-1 and N.
            segment: usize,
            /// A solid whose vertices are shared with other bodies: the draw's instance names its record.
            instanced: bool,
        }
        let mut opaque_vertices: Vec<f32> = Vec::new();
        let mut opaque_types: Vec<f32> = Vec::new();
        // E8.18: per-vertex (along, across) stroke coordinates, parallel to the
        // type streams (zeros where a body carries none); path-gradient reads them.
        let mut opaque_uvs: Vec<f32> = Vec::new();
        let mut trans_uvs: Vec<f32> = Vec::new();
        let mut opaque_instances: Vec<SolidPathInstance> = Vec::new();
        let mut opaque_draws: Vec<PathDraw> = Vec::new();
        let mut trans_vertices: Vec<f32> = Vec::new();
        let mut trans_types: Vec<f32> = Vec::new();
        let mut trans_instances: Vec<SolidPathInstance> = Vec::new();
        let mut trans_draws: Vec<PathDraw> = Vec::new();
        // M3b: per-phase gradient/pattern instance streams (their own storage
        // buffers; production uses separate gradient/pattern buffers too).
        let mut opaque_grad_instances: Vec<paint::GradientInstance> = Vec::new();
        let mut trans_grad_instances: Vec<paint::GradientInstance> = Vec::new();
        let mut opaque_pat_instances: Vec<paint::PatternInstance> = Vec::new();
        let mut trans_pat_instances: Vec<paint::PatternInstance> = Vec::new();
        // M3f: per-phase 672-byte CustomShaderData streams (production:
        // `getOpaque/TransparentCustomShaderBuffer` — one shared buffer for
        // ALL procedural fills; the module id only selects the pipeline).
        let mut opaque_shader_instances: Vec<PaintInstance> = Vec::new();
        let mut trans_shader_instances: Vec<PaintInstance> = Vec::new();
        // M3d: shadow instances in plan order + (segment, z-bits) keys for
        // the bucket loop (shadow-first inside each exact-z bucket).
        let mut shadow_instances: Vec<crate::seams::ShadowInstance> = Vec::new();
        let mut shadow_draws: Vec<(usize, u64, u32)> = Vec::new();
        // Analytic SDF body instance streams (own storage buffers,
        // mirroring WebGPU's separate rectangle/gradient-shape buffers). Opaque
        // draws record (segment, instance_index); transparent draws record
        // (segment, z-bits, instance_index) and bucket like paths.
        //
        // An opaque-phase SDF body keeps the depth-write opaque lane ONLY when
        // its pipeline is the alpha-to-coverage variant (MSAA over an opaque
        // clear), where the analytic edge coverage becomes a per-sample mask
        // and depth is written only on the covered samples. Every other case
        // (a transparent clear, or sample_count 1) would draw the premul
        // variant with depth-write ON: the quad's feathered edge band writes
        // depth on EVERY sample of each fringe pixel, so content BEHIND the
        // body that is drawn later (the opaque SDF batches and the tessellated
        // opaque bodies are not ordered back-to-front) fails the depth test
        // there — a cut-out halo along the silhouette, and a fully transparent
        // pixel where the fringe coverage is ~0 (just above the discard
        // threshold). Those bodies draw in the transparent phase instead (their
        // exact-z bucket, depth-write off, painter's order), which composites
        // the fringe over what is behind it. The instance is identical in both
        // phases: an opaque solid body's `opacity` equals its `body_alpha`.
        let sdf_opaque_lane = sample_count > 1 && clear_is_opaque;
        let mut opaque_solid_sdf: Vec<paint::shape_sdf::ShapeSolidSdfInstance> = Vec::new();
        let mut trans_solid_sdf: Vec<paint::shape_sdf::ShapeSolidSdfInstance> = Vec::new();
        let mut opaque_grad_sdf: Vec<paint::shape_sdf::ShapeGradientSdfInstance> = Vec::new();
        let mut trans_grad_sdf: Vec<paint::shape_sdf::ShapeGradientSdfInstance> = Vec::new();
        let mut opaque_solid_sdf_draws: Vec<(usize, u32)> = Vec::new();
        let mut opaque_grad_sdf_draws: Vec<(usize, u32)> = Vec::new();
        let mut trans_solid_sdf_draws: Vec<(usize, u64, u32)> = Vec::new();
        let mut trans_grad_sdf_draws: Vec<(usize, u64, u32)> = Vec::new();
        // M3a: text batches in plan order + (segment, z-bits) keys — drawn
        // LAST in their exact-z bucket (shadows → paths → text,
        // `renderTransparentItemsWithText`).
        let mut text_batches: Vec<&crate::seams::TextQuadBatch> = Vec::new();
        let mut text_arcs: Vec<&std::sync::Arc<crate::seams::TextQuadBatch>> = Vec::new();
        let mut text_draws: Vec<(usize, u64, usize)> = Vec::new();
        // M3e: backdrop splits in plan order; segment k draws precede split k.
        let mut splits: Vec<&crate::seams::BackdropSplit> = Vec::new();
        // While the canvas moves, a small background blur on a live present draws its layer plainly: its split
        // is left out, so the draws around it share one pass (see `Engine::set_motion_blur_min_px`).
        let motion_blur_min_px = if outermost && present_view.is_some() && cam_override.is_some() {
            self.motion_blur_min_px.get()
        } else {
            0.0
        };
        let mut simplified_blurs = 0u32;
        // Layer-blur composites in plan order + (segment, z-bits)
        // keys — each draws alone in its exact-z bucket (z is per-node).
        // Tuple: (node_id, group, page_clip_bounds, z, clip_rrect, mask id).
        #[allow(clippy::type_complexity)]
        let mut layer_blur_items: Vec<(
            &String,
            &std::sync::Arc<crate::seams::LayerBlurGroup>,
            [f32; 4],
            f64,
            Option<crate::seams::ClipRRect>,
            Option<u64>,
        )> = Vec::new();
        let mut layer_blur_bucket_draws: Vec<(usize, u64, usize)> = Vec::new();
        // Size the body vertex streams up front: at 100k nodes they are hundreds of MB,
        // and grown by doubling they held up to three times their final size at once
        // (the old and new blocks of each realloc) — the biggest transient of a large
        // document's first frame, in wasm linear memory that never shrinks.
        {
            let (mut opaque, mut trans) = ((0usize, 0usize), (0usize, 0usize));
            for item in &plan.items {
                if let DrawItem::Body {
                    phase, geometry, ..
                } = item
                {
                    let n = match phase {
                        Phase::Opaque => &mut opaque,
                        Phase::Transparent => &mut trans,
                    };
                    n.0 += geometry.positions.len();
                    n.1 += geometry.vertex_types.len();
                }
            }
            opaque_vertices.reserve_exact(opaque.0);
            opaque_types.reserve_exact(opaque.1);
            opaque_uvs.reserve_exact(2 * opaque.1);
            trans_vertices.reserve_exact(trans.0);
            trans_types.reserve_exact(trans.1);
            trans_uvs.reserve_exact(2 * trans.1);
        }
        // Vertex residency (live camera presents only): per phase, `Some(appending)` — `true` to append what the
        // streams don't hold, `false` to pack afresh (and record where everything went).
        // The engine's residency is the outermost render's: a capture's render packs its own streams and leaves it be.
        let mut residency = if outermost { self.vertex_residency.borrow_mut().take_all() } else { [None, None] };
        let resident_lane = self.retained
            && self.vertex_residency_on.get()
            && outermost
            && packet_plan.is_some()
            && (present_view.is_some() || self.offscreen_residency.get());
        // Per phase, the geometries more than one solid body draws: held once, drawn instanced.
        let mut shared_solids: [std::collections::HashSet<usize, rustc_hash::FxBuildHasher>; 2] = Default::default();
        // Each solid body's record slot in a resident phase, by item. Slots are given before deciding whether
        // the streams can be kept, so a body that returns to the slot it left finds its vertices still there.
        let mut item_slots: Vec<u32> = Vec::new();
        if resident_lane {
            let uses = solid_geometry_uses(&plan.items);
            for k in 0..2 {
                shared_solids[k] = uses[k].iter().filter(|(_, count)| **count > 1).map(|(geometry, _)| *geometry).collect();
            }
            item_slots = vec![u32::MAX; plan.items.len()];
            for k in 0..2 {
                if let Some(r) = residency[k].as_mut() { r.assign_slots(&plan.items, k, &mut item_slots); }
            }
        }
        let appending: [Option<bool>; 2] = std::array::from_fn(|k| {
            if !resident_lane {
                residency[k] = None;
                return None;
            }
            let labels = VERTEX_STREAM_LABELS[k];
            let g = self.retained_frame.borrow();
            let usable = residency[k].as_ref().and_then(|r| {
                let bufs: Option<Vec<&PooledBuf>> = labels.iter().map(|l| g.bufs.get(l)).collect();
                let bufs = bufs?;
                (0..3).all(|s| bufs[s].writes == r.writes[s]).then_some((r, bufs))
            });
            let fits = usable.is_some_and(|(r, bufs)| {
                // What this frame draws that the streams don't hold, and what they hold that it draws.
                let (mut add, mut live) = (0u64, 0u64);
                let mut seen = std::collections::HashSet::new();
                for (at, item) in plan.items.iter().enumerate() {
                    if let DrawItem::Body { phase, geometry, paint, .. } = item {
                        if (*phase == Phase::Opaque) != (k == 0) { continue; }
                        let address = std::sync::Arc::as_ptr(geometry) as usize;
                        // The owner words this body's vertices will carry.
                        let owner = match paint {
                            BodyPaint::Solid if shared_solids[k].contains(&address) => Some(INSTANCE_OWNER),
                            BodyPaint::Solid => Some(item_slots[at]),
                            _ => None,
                        };
                        if !seen.insert((address, owner)) { continue; }
                        match r.offset_of(geometry, owner) {
                            Some(_) => live += u64::from(geometry.vertex_count),
                            None => add += u64::from(geometry.vertex_count),
                        }
                    }
                }
                let total = u64::from(r.total) + add;
                let limit = self
                    .vertex_chunk_bytes
                    .get()
                    .unwrap_or_else(|| self.gpu.device.limits().max_buffer_size);
                total * 8 <= bufs[0].capacity
                    && total * 4 <= bufs[1].capacity
                    && total * 8 <= bufs[2].capacity
                    && total * 8 <= limit
                    && total <= u64::from(u32::MAX)
                    && (live + add) * 2 >= total
            });
            if fits {
                Some(true)
            } else {
                residency[k] = Some(PhaseResidency::default());
                Some(false)
            }
        });

        // A phase packed afresh starts its slots again; either way, remember which geometries are shared.
        if resident_lane {
            for k in 0..2 {
                let Some(r) = residency[k].as_mut() else { continue };
                if appending[k] == Some(false) { r.assign_slots(&plan.items, k, &mut item_slots); }
                r.shared = shared_solids[k].clone();
            }
        }
        // Where each item's record goes (`RecordAt`), for the packet this frame may build.
        let mut record_at: Vec<RecordAt> =
            if packet_plan.is_some() { vec![RecordAt::NONE; plan.items.len()] } else { Vec::new() };
        let mut note = |item_at: usize, index: u32, first_vertex: u32| {
            if let Some(at) = record_at.get_mut(item_at) {
                *at = RecordAt { index, first_vertex };
            }
        };
        for (item_at, item) in plan.items.iter().enumerate() {
            match item {
                DrawItem::PageFill { instance } => {
                    note(item_at, page_instances.len() as u32, 0);
                    page_instances.push(*instance);
                }
                DrawItem::Body {
                    phase,
                    geometry,
                    instance,
                    blend_mode,
                    z,
                    mask,
                    paint: body_paint,
                    ..
                } => {
                    let (
                        verts,
                        types,
                        instances,
                        grad_instances,
                        pat_instances,
                        shader_instances,
                        draws,
                    ) = match phase {
                        Phase::Opaque => (
                            &mut opaque_vertices,
                            &mut opaque_types,
                            &mut opaque_instances,
                            &mut opaque_grad_instances,
                            &mut opaque_pat_instances,
                            &mut opaque_shader_instances,
                            &mut opaque_draws,
                        ),
                        Phase::Transparent => (
                            &mut trans_vertices,
                            &mut trans_types,
                            &mut trans_instances,
                            &mut trans_grad_instances,
                            &mut trans_pat_instances,
                            &mut trans_shader_instances,
                            &mut trans_draws,
                        ),
                    };
                    let k = usize::from(*phase == Phase::Transparent);
                    // A resident solid's record sits at its slot; a packed one's at its place in the stream.
                    let solid_slot = (resident_lane && matches!(body_paint, BodyPaint::Solid)).then(|| item_slots[item_at]);
                    let instanced = solid_slot.is_some()
                        && shared_solids[k].contains(&(std::sync::Arc::as_ptr(geometry) as usize));
                    let owner = matches!(body_paint, BodyPaint::Solid).then(|| match solid_slot {
                        Some(_) if instanced => INSTANCE_OWNER,
                        Some(slot) => slot,
                        None => instances.len() as u32,
                    });
                    let held = match (&residency[k], appending[k]) {
                        (Some(r), Some(_)) => r.offset_of(geometry, owner),
                        _ => None,
                    };
                    let vertex_offset = match held {
                        Some(o) => o,
                        None => {
                            let base = residency[k].as_ref().map_or(0, |r| r.total);
                            let o = base + (verts.len() / 2) as u32;
                            verts.extend_from_slice(&geometry.positions);
                            types.extend_from_slice(&geometry.vertex_types);
                            let uvs = match phase {
                                Phase::Opaque => &mut opaque_uvs,
                                Phase::Transparent => &mut trans_uvs,
                            };
                            append_body_aux(uvs, geometry, owner);
                            if let Some(r) = residency[k].as_mut() {
                                r.at.entry((std::sync::Arc::as_ptr(geometry) as usize, owner)).or_insert((
                                    o,
                                    geometry.vertex_count,
                                    std::sync::Arc::downgrade(geometry),
                                ));
                            }
                            o
                        }
                    };
                    // M3b: pack into the routed pipeline family's instance
                    // stream, fixing up vertexOffset/vertexCount at upload (`Engine::pack_body`, shared with
                    // the prepared-frame patch).
                    let (kind, shader, instance_index, tex) =
                        match self.pack_body(body_paint, instance, geometry, vertex_offset, present_shader_time) {
                            PackedBody::Gradient(gi, post) => {
                                let idx = grad_instances.len() as u32;
                                grad_instances.push(gi);
                                let kind = if post {
                                    // Q-116: post-shader overlay slot.
                                    PaintDrawKind::PostShaderGradientStroke
                                } else {
                                    PaintDrawKind::Gradient
                                };
                                (kind, None, idx, None)
                            }
                            PackedBody::Pattern(pi, tex) => {
                                let idx = pat_instances.len() as u32;
                                pat_instances.push(pi);
                                (PaintDrawKind::Pattern, None, idx, Some(tex))
                            }
                            PackedBody::Shader(si, module, tex) => {
                                let idx = shader_instances.len() as u32;
                                shader_instances.push(si);
                                (PaintDrawKind::Shader, Some(module), idx, tex)
                            }
                            PackedBody::Solid(solid) => {
                                let idx = match solid_slot {
                                    Some(slot) => {
                                        if instances.len() <= slot as usize {
                                            instances.resize(slot as usize + 1, bytemuck::Zeroable::zeroed());
                                        }
                                        instances[slot as usize] = solid;
                                        slot
                                    }
                                    None => {
                                        instances.push(solid);
                                        instances.len() as u32 - 1
                                    }
                                };
                                (PaintDrawKind::Solid, None, idx, None)
                            }
                        };
                    note(item_at, instance_index, vertex_offset);
                    draws.push(PathDraw {
                        kind,
                        shader,
                        instance_index,
                        vertex_offset,
                        vertex_count: geometry.vertex_count,
                        z_bucket: z.to_bits(),
                        blend: *blend_mode,
                        mask: mask.as_ref().map(|m| m.mask_id),
                        tex,
                        segment: splits.len(),
                        instanced,
                    });
                }
                DrawItem::Shadow { z, instance, .. } => {
                    // M3d: shadow quads — transparent phase, shadow-first in
                    // their exact-z bucket (drop = body z; inner = z + 0.5).
                    let idx = shadow_instances.len() as u32;
                    note(item_at, idx, 0);
                    shadow_instances.push(**instance);
                    shadow_draws.push((splits.len(), z.to_bits(), idx));
                }
                DrawItem::Text(batch) => {
                    // M3a: text draws in the transparent phase at the batch's
                    // exact z, after paths within the bucket.
                    let idx = text_batches.len();
                    // A greeked motion frame draws the batch's word-bar copy (same z, model and clip).
                    let batch = greek_arcs.get(idx).and_then(Option::as_ref).unwrap_or(batch);
                    text_batches.push(batch);
                    text_arcs.push(batch);
                    text_draws.push((splits.len(), batch.z.to_bits(), idx));
                }
                DrawItem::SolidShapeSdf {
                    phase, instance, z, ..
                } => match phase {
                    Phase::Opaque if sdf_opaque_lane => {
                        let idx = opaque_solid_sdf.len() as u32;
                        note(item_at, idx, 0);
                        opaque_solid_sdf.push(**instance);
                        opaque_solid_sdf_draws.push((splits.len(), idx));
                    }
                    Phase::Opaque | Phase::Transparent => {
                        let idx = trans_solid_sdf.len() as u32;
                        note(item_at, idx, 0);
                        trans_solid_sdf.push(**instance);
                        trans_solid_sdf_draws.push((splits.len(), z.to_bits(), idx));
                    }
                },
                DrawItem::GradientShapeSdf {
                    phase, instance, z, ..
                } => match phase {
                    Phase::Opaque if sdf_opaque_lane => {
                        let idx = opaque_grad_sdf.len() as u32;
                        note(item_at, idx, 0);
                        opaque_grad_sdf.push(**instance);
                        opaque_grad_sdf_draws.push((splits.len(), idx));
                    }
                    Phase::Opaque | Phase::Transparent => {
                        let idx = trans_grad_sdf.len() as u32;
                        note(item_at, idx, 0);
                        trans_grad_sdf.push(**instance);
                        trans_grad_sdf_draws.push((splits.len(), z.to_bits(), idx));
                    }
                },
                DrawItem::BackdropSplit(split) => {
                    if motion_blur_min_px > 0.0 && split_blur_reach_px(split).is_some_and(|r| r <= motion_blur_min_px) {
                        simplified_blurs += 1;
                        continue;
                    }
                    // An isolation pair that lands nowhere on this target (`split_pair_in`): both halves
                    // carry the same empty region, and neither ends a pass.
                    if split.op.is_blend_split() && (split.region_px[2] == 0 || split.region_px[3] == 0) {
                        continue;
                    }
                    // M3e pass-split (SEAMS §7): ends the current pass,
                    // runs the effect chain, resumes with LoadOp::Load.
                    splits.push(split.as_ref());
                }
                DrawItem::LayerBlurComposite {
                    node_id,
                    z,
                    group,
                    page_clip_bounds,
                    clip_rrect,
                    mask,
                } => {
                    // One blurred-capture quad in the transparent
                    // phase at the diverted node's z (pre-pass rendered the
                    // capture above).
                    let idx = layer_blur_items.len();
                    layer_blur_items.push((
                        node_id,
                        group,
                        *page_clip_bounds,
                        *z,
                        *clip_rrect,
                        mask.as_ref().map(|m| m.mask_id),
                    ));
                    layer_blur_bucket_draws.push((splits.len(), z.to_bits(), idx));
                }
            }
        }

        // Shader-fill pre-pass (`set_shader_prepass`): under MSAA on the live camera lane, each procedural
        // fill (not image-backed) is shaded once per pixel into a target-sized single-sample slot before the
        // main pass, and its main-pass draw becomes the `shader-fill-composite` module reading that slot at
        // group 2 (the image-backed fills' layout; mask at group 3). Fills whose pixel rects don't overlap
        // share a slot; one that finds no free slot draws directly as before.
        let mut prepass_jobs: Vec<ShaderPrepassJob> = Vec::new();
        let mut prepass_slot_rects: Vec<Vec<[u32; 4]>> = Vec::new();
        if self.shader_prepass
            && cam_override.is_some()
            && sample_count > 1
            && region_viewport.is_none()
        {
            let (tw, th) = (eff_width_px as f64, eff_height_px as f64);
            let overlap = |a: &[u32; 4], b: &[u32; 4]| {
                a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
            };
            // (draws, their instances, the composite's descriptor: 0 opaque, 1 transparent)
            for (draws, instances, desc) in [
                (&mut opaque_draws, &opaque_shader_instances, 0usize),
                (&mut trans_draws, &trans_shader_instances, 1usize),
            ] {
                for d in draws.iter_mut() {
                    if d.kind != PaintDrawKind::Shader || d.tex.is_some() {
                        continue;
                    }
                    let Some(module) = d.shader.clone() else {
                        continue;
                    };
                    let inst = &instances[d.instance_index as usize];
                    let Some((rect, quad)) = shader_prepass_region(inst, &vp, tw, th) else {
                        continue;
                    };
                    let slot = match prepass_slot_rects
                        .iter()
                        .position(|rs| rs.iter().all(|r| !overlap(r, &rect)))
                    {
                        Some(slot) => slot,
                        None if prepass_slot_rects.len() < SHADER_PREPASS_MAX_SLOTS => {
                            prepass_slot_rects.push(Vec::new());
                            prepass_slot_rects.len() - 1
                        }
                        None => continue,
                    };
                    prepass_slot_rects[slot].push(rect);
                    let mut instance = *inst;
                    instance.fill_color[3] = 1.0;
                    instance.page_clip_bounds = crate::seams::INFINITE_CLIP;
                    self.pipeline(&module, 1, 1, BlendMode::Normal, &self.layouts.path_layout)?;
                    self.pipeline(
                        SHADER_FILL_COMPOSITE,
                        desc,
                        sample_count,
                        d.blend,
                        &self.layouts.image_shader_layout,
                    )?;
                    prepass_jobs.push(ShaderPrepassJob {
                        slot,
                        module,
                        instance,
                        quad,
                        rect,
                    });
                    d.shader = Some(SHADER_FILL_COMPOSITE.to_string());
                    d.tex = Some((format!("prepass:{slot}"), true));
                }
            }
        }

        // M3e invariant: backdrop-effect pages plan at sample_count 1 (the
        // effect chain composites into the resolved single-sample target —
        // Q-02). Blend and paint-stack splits run at any sample count: they
        // copy the resolved target and composite back into the multisample one.
        if sample_count != 1 && splits.iter().any(|s| !s.op.is_blend_split()) {
            bail!("backdrop splits require sample_count 1 (plan builder forces this)");
        }
        if outermost {
            self.simplified_blurs.set(simplified_blurs);
        }
        let backdrop_renderer = if splits.is_empty() {
            None
        } else {
            Some(self.backdrop_renderer(sample_count)?)
        };

        // Transparent draws sorted by ascending EXACT z bucket (f64 bits —
        // production keys `groupByZIndexWithOrdering`'s Map by the raw
        // fractional zIndex); within a bucket: shadows → SDF solid → SDF
        // gradient → paths → text (shadowFirst; the SDF path inserts the SDF
        // bodies before tessellated paths, mirroring WebGPU's transparent
        // intra-bucket order). Relative order inside a slot is plan order
        // (stable). M3e: bucketed per pass segment.
        // One bucket per (segment, z): what each kind draws there, in plan order. Built by sorting each kind's
        // (key, index) list (stable, so plan order holds within a key) and walking the keys, instead of a map
        // insert per draw (a 20k-layer frame has ~20k buckets: the tree's inserts and frees were ~2 ms a frame).
        struct Bucket<'a> {
            /// Indices into shadow_instances (drawn first — shadowFirst).
            shadows: &'a [u32],
            /// Indices into layer_blur_items — the diverted node's
            /// whole paint (its own shadows/body/text ride inside the capture),
            /// so the bucket at that z holds ONLY this quad; slot order within
            /// the bucket is therefore moot, kept next to shadows for
            /// determinism.
            layer_blur: &'a [usize],
            /// firstInstance offsets into trans_solid_sdf (drawn after
            /// shadows, before paths — `renderTransparentItemsWithText`).
            solid_sdf: &'a [u32],
            /// firstInstance offsets into trans_grad_sdf (after the
            /// SDF solids, before paths).
            gradient_sdf: &'a [u32],
            /// Indices into trans_draws.
            paths: &'a [usize],
            /// M3a: indices into text_batches (drawn last in the bucket).
            texts: &'a [usize],
        }
        type BucketKey = (usize, u64);
        fn by_key<T: Copy>(mut v: Vec<(BucketKey, T)>) -> (Vec<BucketKey>, Vec<T>) {
            v.sort_by_key(|e| e.0);
            v.into_iter().unzip()
        }
        let (path_keys, path_idx) = by_key(
            trans_draws
                .iter()
                .enumerate()
                .map(|(i, d)| ((d.segment, d.z_bucket), i))
                .collect(),
        );
        let (shadow_keys, shadow_idx) = by_key(shadow_draws.iter().map(|&(s, z, i)| ((s, z), i)).collect());
        let (solid_keys, solid_idx) =
            by_key(trans_solid_sdf_draws.iter().map(|&(s, z, i)| ((s, z), i)).collect());
        let (grad_keys, grad_idx) =
            by_key(trans_grad_sdf_draws.iter().map(|&(s, z, i)| ((s, z), i)).collect());
        let (text_keys, text_idx) = by_key(text_draws.iter().map(|&(s, z, i)| ((s, z), i)).collect());
        let (blur_keys, blur_idx) =
            by_key(layer_blur_bucket_draws.iter().map(|&(s, z, i)| ((s, z), i)).collect());
        let mut all_keys: Vec<BucketKey> = Vec::with_capacity(
            path_keys.len()
                + shadow_keys.len()
                + solid_keys.len()
                + grad_keys.len()
                + text_keys.len()
                + blur_keys.len(),
        );
        for keys in [&path_keys, &shadow_keys, &solid_keys, &grad_keys, &text_keys, &blur_keys] {
            all_keys.extend_from_slice(keys);
        }
        // Six sorted runs: the stable sort merges them (near-linear).
        all_keys.sort();
        all_keys.dedup();
        // The run of `keys` (sorted) equal to `key` starting at `*at`, as a range; advances `*at`.
        fn take_run(keys: &[BucketKey], key: BucketKey, at: &mut usize) -> std::ops::Range<usize> {
            let start = *at;
            while *at < keys.len() && keys[*at] == key {
                *at += 1;
            }
            start..*at
        }
        let mut cursors = [0usize; 6];
        let bucket_order: Vec<(BucketKey, Bucket<'_>)> = all_keys
            .iter()
            .map(|&key| {
                let bucket = Bucket {
                    paths: &path_idx[take_run(&path_keys, key, &mut cursors[0])],
                    shadows: &shadow_idx[take_run(&shadow_keys, key, &mut cursors[1])],
                    solid_sdf: &solid_idx[take_run(&solid_keys, key, &mut cursors[2])],
                    gradient_sdf: &grad_idx[take_run(&grad_keys, key, &mut cursors[3])],
                    texts: &text_idx[take_run(&text_keys, key, &mut cursors[4])],
                    layer_blur: &blur_idx[take_run(&blur_keys, key, &mut cursors[5])],
                };
                (key, bucket)
            })
            .collect();
        // The buckets of pass segment `segment`, in z order.
        let segment_buckets = |segment: usize| -> &[(BucketKey, Bucket<'_>)] {
            let lo = bucket_order.partition_point(|(k, _)| k.0 < segment);
            let hi = bucket_order.partition_point(|(k, _)| k.0 <= segment);
            &bucket_order[lo..hi]
        };
        // M2 prepared-frame: decide reuse BEFORE the text prepare + geometry
        // uploads. Eligible only for a camera present (packet_plan + present_view)
        // of a NON-backdrop plan (`splits.is_empty()`) with retained mode on
        // (the packet's geometry/text buffers live in the retained pool). A HIT
        // requires the SAME plan Arc (identity), the SAME target params (a
        // viewport resize / DPR / MSAA change is NOT a camera change and must
        // rebuild), and an unchanged pool epoch (no other render overwrote/grew
        // a shared geometry buffer since the packet was built).
        //
        // The reused text packet's bind groups also reference the text engine's
        // atlas/emoji textures + pipelines (owned by `self.text`, NOT the geometry
        // pool), but the plan-Arc identity check transitively covers them: an atlas
        // grows only when NEW glyphs/fonts appear, which is a scene/font-host change
        // → a different camera fingerprint → a different cached plan Arc → ptr_eq
        // fails → miss → fresh `prepare`. So a stable plan Arc guarantees a stable
        // atlas; both `packet_plan: Some` callers (finish_camera_frame across a
        // static pan, `isolated::render_isolated_target` across a static resident
        // re-render) hand the same cache-slot Arc while the content is unchanged.
        // The packet is a SINGLE engine-global slot: a frame that runs both
        // lanes evicts it both ways (correct, just re-prepares) — see the
        // isolated module doc.
        // Plans with layer captures are eligible when every capture is content-keyed and none stood in for
        // another zoom step: the packet keeps the results it drew (`PacketCaptures`), so an evicted and re-rendered
        // capture can't change under it, and the same key is the same pixels.
        // Background effects (splits) run their chains between the segments every frame, reading the frame drawn
        // so far, so a hit still records the passes (no recorded pass); it skips the uploads, which the splits
        // don't read.
        let prepared_eligible = self.retained
            && packet_plan.is_some()
            && (layer_blur_items.is_empty() || (present_view.is_some() && captures_packetable));
        if self.retained && packet_plan.is_some() && outermost {
            self.prepared_ineligible_reason
                .set(if prepared_eligible { "" } else { "captures" });
        }
        let present_clock = present_shader_time.unwrap_or(0.0);
        let is_hit = prepared_eligible && {
            let pf = self.prepared_frame.borrow();
            match pf.as_ref() {
                Some(p) => {
                    p.key == packet_plan.expect("eligible ⇒ Some").key
                        && p.live_masks == present_view.is_some()
                        && p.width == eff_width_px
                        && p.height == eff_height_px
                        && p.sample_count == sample_count
                        && p.clear_is_opaque == clear_is_opaque
                        && p.pool_epoch == self.pool_epoch.get()
                        && p.text_greek_sig == greek_sig
                        && p.simplified_blurs == self.simplified_blurs.get()
                        // A clock-bearing plan whose re-threaded clock
                        // advanced must re-encode (the new `shader_params` ride the
                        // skipped instance upload, not the frame uniform), or the
                        // shader freezes. A clock-free plan ignores this.
                        && (!p.has_clock_fill || p.shader_time_s == present_clock)
                }
                None => false,
            }
        };
        // On a hit, reuse the unchanged pool buffers in place (skip the geometry
        // write_buffers) and the stored text arena (skip TextDraw::prepare). Only
        // the camera/frame uniform was rewritten (above).
        //
        // CORRECTNESS INVARIANT: a hit reuses text bind groups that reference the
        // *persistent* retained frame-uniform buffer, so they pick up the camera
        // rewritten this frame. This holds only because (a) `prepared_eligible`
        // requires `self.retained`, and (b) the retained frame uniform is dropped
        // ONLY by `clear_retained_gpu`, which also drops the packet. If a future
        // change recreates the retained frame uniform without dropping the packet,
        // the camera would freeze on hits — `prepared_frame_hit_matches_full_encode`
        // guards this by asserting a hit at a panned camera matches a fresh encode.
        self.pf_skip_uploads.set(is_hit);
        // M3a: pre-build (or, on a hit, reuse) all text GPU resources (pipelines,
        // atlas/emoji textures, the uniform/vertex/index arenas) before the
        // encoder so the pass only takes immutable borrows.
        drop(packing_profile);
        let upload_profile = crate::prof::span("canvas.host.prepare_upload");
        let text_prepared = if is_hit {
            self.prepared_frame
                .borrow()
                .as_ref()
                .expect("hit ⇒ packet present")
                .text
                .clone()
        } else {
            // Prepare in chunks; a chunk the replaced packet prepared from the same batches (an edit that left its
            // text layers alone) is kept.
            // The camera lane (a retained present with a plan identity) carries text whether or not the frame is
            // packeted: from the replaced packet, else from the last unpacketed frame.
            let camera_lane = self.retained && present_view.is_some() && packet_plan.is_some();
            let carry = if camera_lane {
                text_carry.take().or_else(|| self.retained_text.borrow_mut().take())
            } else {
                None
            };
            let mut chunks = Vec::with_capacity(text_batches.len().div_ceil(TEXT_CHUNK));
            let mut reused = 0u64;
            for (i, batches) in text_batches.chunks(TEXT_CHUNK).enumerate() {
                let arcs = &text_arcs[i * TEXT_CHUNK..i * TEXT_CHUNK + batches.len()];
                if let Some(chunk) = carry.as_ref().and_then(|c| c.chunk(i, arcs, sample_count)) {
                    chunks.push(chunk);
                    reused += 1;
                    continue;
                }
                let prepared = self
                    .text
                    .prepare(&self.text_gpu(), &frame_uniform, batches, sample_count)?;
                self.add_pf_text_bytes(prepared.uploaded_bytes());
                chunks.push(std::sync::Arc::new(prepared));
            }
            if reused > 0 {
                let mut s = self.prepared_stats.get();
                s.text_reuses = s.text_reuses.wrapping_add(reused);
                self.prepared_stats.set(s);
            }
            // Arc so a HIT reuses the prepared text in O(1) (refcount bump) rather
            // than deep-cloning its draw metadata Vecs/Strings every fast-path frame.
            let prepared = std::sync::Arc::new(TextChunks { chunks });
            // A camera frame no packet will keep: keep its text for the next one.
            if camera_lane && !prepared_eligible {
                *self.retained_text.borrow_mut() = Some(TextCarry {
                    text: prepared.clone(),
                    batches: text_arcs.iter().map(|b| (*b).clone()).collect(),
                    host_generation: crate::text_draw::text_host_generation(),
                    sample_count,
                });
            }
            if let Some(c) = carry {
                self.defer_drop(Box::new(c));
            }
            prepared
        };

        // Per-composite uniform + bind group (fresh each render —
        // the capture textures are per-frame). A missing result (a mini render
        // that failed after the plan was built) skips the draw: the node
        // simply doesn't paint this frame, never wrong-paints.
        let layer_blur_bind_groups: Vec<Option<wgpu::BindGroup>> = layer_blur_items
            .iter()
            .map(|(node_id, group, clip, z, rrect, _)| {
                let renderer = layer_blur_renderer.as_ref()?;
                let view = layer_blur_results.get(node_id.as_str())?;
                // A capture inner shadow (at body z + 0.5) must also draw over
                // the body's opaque STROKE, which sits `stroke_depth_offset`
                // (5/8 of a z) nearer than its fill: it takes the depth of
                // body z + 0.75 — nearer than that stroke, still behind the
                // next node (z + 1). Its bucket stays z + 0.5.
                let depth_z = match &group.filters.shadow {
                    Some(s) if s.inner => *z + 0.25,
                    _ => *z,
                };
                let depth = crate::depth::linear_depth(depth_z) as f32;
                let mut data = crate::layer_blur::LayerBlurRenderer::composite_uniform_data(
                    group, *clip, *rrect, depth, 1.0,
                );
                // A stand-in covers the region its own zoom step captured.
                if let Some(r) = layer_blur_regions.get(node_id.as_str()) {
                    for (d, v) in data.iter_mut().zip(r) {
                        *d = *v as f32;
                    }
                }
                Some(renderer.composite_bind_group(device, queue, &self.scratch, prepared_eligible, &data, view))
            })
            .collect();

        // --- Upload buffers -------------------------------------------------
        // R2c: per-class buffers ride the grow-only pool in retained mode.
        let page_buf = self.make_buffer(
            PAGE_STORAGE,
            bytemuck::cast_slice(&page_instances),
            wgpu::BufferUsages::STORAGE,
        );
        // The vertex streams, each split across buffers at draw boundaries when it would pass the buffer size
        // limit (`VertexStreams`).
        let opaque_streams = self.upload_resident_streams(
            0,
            &mut residency[0],
            appending[0],
            [&opaque_vertices, &opaque_types, &opaque_uvs],
            opaque_draws.iter().map(|d| (d.vertex_offset, d.vertex_count)),
        );
        let opaque_ib = self.make_buffer(
            SOLID_STORAGE[0],
            bytemuck::cast_slice(&opaque_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let trans_streams = self.upload_resident_streams(
            1,
            &mut residency[1],
            appending[1],
            [&trans_vertices, &trans_types, &trans_uvs],
            trans_draws.iter().map(|d| (d.vertex_offset, d.vertex_count)),
        );
        if outermost {
            *self.vertex_residency.borrow_mut() = residency;
            self.vertex_chunks
                .set([opaque_streams.bases.len(), trans_streams.bases.len()]);
        }
        let trans_ib = self.make_buffer(
            SOLID_STORAGE[1],
            bytemuck::cast_slice(&trans_instances),
            wgpu::BufferUsages::STORAGE,
        );
        // Uploaded: free the vertex streams now, not at the end of the frame, so they
        // are not still held while the text below packs its own arenas.
        drop((
            opaque_vertices,
            opaque_types,
            opaque_uvs,
            trans_vertices,
            trans_types,
            trans_uvs,
        ));

        let page_bg = self.make_storage_bg("page storage bg", PAGE_STORAGE, &page_buf);
        let opaque_bg =
            self.make_storage_bg("opaque storage bg", SOLID_STORAGE[0], &opaque_ib);
        let trans_bg = self.make_storage_bg(
            "transparent storage bg",
            SOLID_STORAGE[1],
            &trans_ib,
        );
        let mask_bg = self.inactive_mask.as_ref().expect("inactive mask group");

        // M3b: gradient/pattern storage buffers + texture bind groups.
        let opaque_grad_ib = self.make_buffer(
            GRADIENT_STORAGE[0],
            bytemuck::cast_slice(&opaque_grad_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let trans_grad_ib = self.make_buffer(
            GRADIENT_STORAGE[1],
            bytemuck::cast_slice(&trans_grad_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let opaque_pat_ib = self.make_buffer(
            PATTERN_STORAGE[0],
            bytemuck::cast_slice(&opaque_pat_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let trans_pat_ib = self.make_buffer(
            PATTERN_STORAGE[1],
            bytemuck::cast_slice(&trans_pat_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let opaque_grad_bg = self.make_storage_bg(
            "opaque gradient storage bg",
            GRADIENT_STORAGE[0],
            &opaque_grad_ib,
        );
        let trans_grad_bg = self.make_storage_bg(
            "transparent gradient storage bg",
            GRADIENT_STORAGE[1],
            &trans_grad_ib,
        );
        let opaque_pat_bg = self.make_storage_bg(
            "opaque pattern storage bg",
            PATTERN_STORAGE[0],
            &opaque_pat_ib,
        );
        let trans_pat_bg = self.make_storage_bg(
            "transparent pattern storage bg",
            PATTERN_STORAGE[1],
            &trans_pat_ib,
        );
        // M3f: shared CustomShaderData storage per phase (672-byte stride).
        let opaque_shader_ib = self.make_buffer(
            SHADER_STORAGE[0],
            bytemuck::cast_slice(&opaque_shader_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let trans_shader_ib = self.make_buffer(
            SHADER_STORAGE[1],
            bytemuck::cast_slice(&trans_shader_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let opaque_shader_bg = self.make_storage_bg(
            "opaque custom shader storage bg",
            SHADER_STORAGE[0],
            &opaque_shader_ib,
        );
        let trans_shader_bg = self.make_storage_bg(
            "transparent custom shader storage bg",
            SHADER_STORAGE[1],
            &trans_shader_ib,
        );
        // Pre-upload all pattern textures + (key, sampler) bind groups so the
        // render pass only takes immutable borrows. Retained mode (R2d): the
        // cache persists on the Engine, keyed with settle-state + pixels
        // identity (`pattern_draw_key`), trimmed to the MB budget.
        let mut local_patterns: Option<paint::draw::PatternTextureCache> = None;
        let mut patterns_guard;
        // Paint layers used as masks: their surface textures, for the mask pre-pass below.
        let mut mask_surfaces = rendered_mask_views;
        let pattern_textures: &paint::draw::PatternTextureCache = {
            let streaming = plan.items.iter().any(|item| matches!(item,
                DrawItem::Body { paint: BodyPaint::Pattern(pattern), .. }
                    if pattern.texture.image.external.is_none() && pattern.texture.key.starts_with("video:")
            ));
            // `keep_patterns`: a render inside another (a capture) finds the cache borrowed and uploads its own (the
            // keys are the same either way).
            let kept = if self.retained {
                Some(self.retained_patterns.borrow_mut())
            } else if self.keep_patterns || streaming {
                self.retained_patterns.try_borrow_mut().ok()
            } else {
                None
            };
            let cache: &mut paint::draw::PatternTextureCache = match kept {
                Some(guard) => {
                    patterns_guard = guard;
                    patterns_guard.get_or_insert_with(|| paint::draw::PatternTextureCache::new(device, self.working))
                }
                None => local_patterns.get_or_insert_with(|| paint::draw::PatternTextureCache::new(device, self.working)),
            };
            cache.begin_frame();
            for item in &plan.items {
                if let DrawItem::Body {
                    paint: BodyPaint::Pattern(p),
                    ..
                } = item
                {
                    // A host's decoded texels (`DecodedImage::straight`) are uploaded at their natural size: one
                    // this device cannot hold refuses the frame here, at the device that draws it. A
                    // browser-prepared image was sized by the asset store's policy.
                    let image = &p.texture.image;
                    let limit = device.limits().max_texture_dimension_2d;
                    if image.straight && image.width.max(image.height) > limit {
                        bail!(
                            "image {}: {}x{} exceeds this device's {limit} px texture limit; resize the source",
                            moda_assets::natural_source(&p.texture.key).unwrap_or(&p.texture.key),
                            image.width,
                            image.height
                        );
                    }
                    cache.bind_group(
                        device,
                        queue,
                        &self.layouts.pattern_texture_bgl,
                        &self.pattern_draw_key(p),
                        &p.texture.image,
                        p.clamp_sampler,
                    );
                }
            }
            // A paint layer used as a mask: the same texture its content draw uses (created and filled here if the
            // layer only masks), so its pixels reach the mask without entering the engine's heap.
            for m in &plan.masks {
                let Some(src) = m.image.as_ref().and_then(|i| i.surface.as_deref()) else { continue };
                if mask_surfaces.contains_key(src) {
                    continue;
                }
                // A graph-pixel unit kept on the GPU (external_texture::CaptureTexture): its texture as it is.
                if let Some(texture) = crate::external_texture::capture_texture(src) {
                    mask_surfaces.insert(src.to_string(), texture.create_view(&Default::default()));
                    continue;
                }
                let Some(texture) = crate::surface::pattern_texture(src) else { continue };
                let key = retained_pattern_draw_key(&texture);
                cache.bind_group(device, queue, &self.layouts.pattern_texture_bgl, &key, &texture.image, true);
                if let Some(view) = cache.texture_view(&key) {
                    mask_surfaces.insert(src.to_string(), view);
                }
            }
            if self.retained || self.keep_patterns || streaming {
                cache.trim(PATTERN_TEXTURE_BUDGET_BYTES);
            }
            cache
        };
        // M4e: pre-upload shader-image textures (straight-alpha rgba8unorm,
        // clamp-to-edge linear sampler — `ShaderImageResourceManager`).
        let mut local_shader_images: Option<ShaderImageTextureCache> = None;
        let mut shader_images_guard;
        let shader_image_textures: &ShaderImageTextureCache = {
            let cache: &mut ShaderImageTextureCache = if self.retained {
                shader_images_guard = self.retained_shader_images.borrow_mut();
                shader_images_guard.get_or_insert_with(|| ShaderImageTextureCache::new(device))
            } else {
                local_shader_images.get_or_insert_with(|| ShaderImageTextureCache::new(device))
            };
            cache.begin_frame();
            for item in &plan.items {
                if let DrawItem::Body {
                    paint: BodyPaint::Shader(body),
                    ..
                } = item
                {
                    if let Some(tex) = &body.image {
                        cache.bind_group(
                            device,
                            queue,
                            &self.layouts.shader_image_bgl,
                            &self.shader_image_draw_key(tex),
                            tex,
                        );
                    }
                }
            }
            if self.retained {
                cache.trim(SHADER_IMAGE_BUDGET_BYTES);
            }
            cache
        };

        // M3d: shadow storage buffer (160-byte RectangleShadowData stride;
        // `updateTransparentShadows` equivalent).
        let shadow_buf = self.make_buffer(
            SHADOW_STORAGE,
            bytemuck::cast_slice(&shadow_instances),
            wgpu::BufferUsages::STORAGE,
        );
        let shadow_bg = self.make_storage_bg(
            "shadow storage bg",
            SHADOW_STORAGE,
            &shadow_buf,
        );

        // Analytic SDF body storage buffers + bind groups (group 1 of
        // the 2-group page layout; the shapes are indexed by `firstInstance`).
        let opaque_solid_sdf_buf = self.make_buffer(
            SOLID_SDF_STORAGE[0],
            bytemuck::cast_slice(&opaque_solid_sdf),
            wgpu::BufferUsages::STORAGE,
        );
        let trans_solid_sdf_buf = self.make_buffer(
            SOLID_SDF_STORAGE[1],
            bytemuck::cast_slice(&trans_solid_sdf),
            wgpu::BufferUsages::STORAGE,
        );
        let opaque_grad_sdf_buf = self.make_buffer(
            GRADIENT_SDF_STORAGE[0],
            bytemuck::cast_slice(&opaque_grad_sdf),
            wgpu::BufferUsages::STORAGE,
        );
        let trans_grad_sdf_buf = self.make_buffer(
            GRADIENT_SDF_STORAGE[1],
            bytemuck::cast_slice(&trans_grad_sdf),
            wgpu::BufferUsages::STORAGE,
        );
        let opaque_solid_sdf_bg = self.make_storage_bg(
            "opaque shape sdf solid storage bg",
            SOLID_SDF_STORAGE[0],
            &opaque_solid_sdf_buf,
        );
        let trans_solid_sdf_bg = self.make_storage_bg(
            "transparent shape sdf solid storage bg",
            SOLID_SDF_STORAGE[1],
            &trans_solid_sdf_buf,
        );
        let opaque_grad_sdf_bg = self.make_storage_bg(
            "opaque shape sdf gradient storage bg",
            GRADIENT_SDF_STORAGE[0],
            &opaque_grad_sdf_buf,
        );
        let trans_grad_sdf_bg = self.make_storage_bg(
            "transparent shape sdf gradient storage bg",
            GRADIENT_SDF_STORAGE[1],
            &trans_grad_sdf_buf,
        );

        // --- Targets (R2a: cached by size in retained mode) -------------------
        let (target, msaa, depth_tex) =
            self.frame_targets(eff_width_px, eff_height_px, sample_count);
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let msaa_view = msaa
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Present fast path: when a present target (the canvas swapchain view) is
        // supplied AND the plan has no backdrop splits, the MSAA pass resolves
        // DIRECTLY into it (or renders into it at sample_count==1), skipping the
        // fullscreen-blit pass. (`target` above is still allocated/cached by
        // frame_targets but goes unused as a render destination on this path.)
        //
        // A backdrop-split plan ALSO presents direct when the swapchain
        // is a copy source (`present_copy_src`): `execute_split` copies only its
        // small region back OUT of the swapchain into a scratch, runs the effect,
        // and composites the result straight into the swapchain view — no
        // offscreen, no blit (the WebGPU renderer's approach). Without a
        // copy-source swapchain it can't read the region back, so it keeps the
        // offscreen fallback: render into `target`, copy the region out of it,
        // composite into `target_view`, and the caller blits the whole frame in.
        // Export/readback always passes `present_view: None` (offscreen).
        //
        // INVARIANT this path relies on: a plan with backdrop-EFFECT splits is
        // single-sample (`sample_count == 1`), enforced by the `bail!` above (and
        // forced at plan-build), so those segments render STRAIGHT into the
        // swapchain and `execute_split` reads/composites that same swapchain.
        // Blend / paint-stack splits may ride MSAA: each segment's pass resolves
        // into the swapchain (the copy source), and `BlendExec` composites into
        // the multisample view with the swapchain as its resolve target, so the
        // next segment (LoadOp::Load on the MSAA view) keeps the composite.
        // A swapchain whose format differs from the engine's
        // frame format cannot be a direct MSAA-resolve / backdrop-composite
        // target — force the offscreen + blit path (the present pipeline is
        // format-neutral, so the blit is channel-correct on any swapchain
        // format). The live wasm engine now constructs WITH the preferred
        // swapchain format, so its common camera frame presents direct.
        let can_present_to_view = present_view.is_some() && direct_present_ok;
        let backdrop_direct =
            can_present_to_view && present_copy_src.is_some() && !splits.is_empty();
        let presented_to_view = can_present_to_view && (splits.is_empty() || backdrop_direct);
        let effective_present_view = if presented_to_view {
            present_view
        } else {
            None
        };
        let (color_view, resolve_view): (&wgpu::TextureView, Option<&wgpu::TextureView>) =
            match (effective_present_view, msaa_view.as_ref()) {
                (Some(pv), Some(mv)) => (mv, Some(pv)),
                (Some(pv), None) => (pv, None),
                (None, Some(mv)) => (mv, Some(&target_view)),
                (None, None) => (&target_view, None),
            };

        // --- Encode ----------------------------------------------------------
        // Clear: white opaque or transparent (createExportRenderPassDescriptor).
        // M3e: the draw stream is executed in pass SEGMENTS split at backdrop
        // ops (SEAMS §7). Segment 0 clears; later segments resume with
        // LoadOp::Load on color AND depth, after the effect chain composites
        // into the (sample_count-1) target between segments.
        // Live-viewport: the camera frame clears to the editor canvas
        // backdrop (the gray shown around/between pages); the per-page fills
        // paint over it. The export path keeps white-opaque / transparent.
        // `clear_color` is computed once up front (next to `clear_is_opaque`).
        drop(upload_profile);
        let _record_profile = crate::prof::span("canvas.host.record_submit");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Page Export Commands"),
        });

        // M3d: mask coverage pre-pass — r8unorm textures rendered into the
        // frame's encoder BEFORE the main pass (production mask pre-pass;
        // `WebGPURenderOrchestrator.ts:543-632`), plus per-mask consumption
        // bind groups at MASK_BIND_GROUP_INDEX.
        // Retained mode keeps mask coverage across frames (canvas-space textures do
        // not change with a pan; zoom re-renders at power-of-two steps); exports
        // render every mask fresh. Coverage renders at the frame's canvas→pixel
        // scale (the VP's x/y scale over the raster the viewport maps onto), so a
        // mask edge is anti-aliased at output resolution.
        let mask_px_scale = {
            let (rw, rh) = match region_viewport {
                Some(r) => (r.raster_w, r.raster_h),
                None => (eff_width_px, eff_height_px),
            };
            let sx = (vp[0] as f64).abs() * rw as f64 * 0.5;
            let sy = (vp[5] as f64).abs() * rh as f64 * 0.5;
            sx.max(sy)
        };
        let mask_atlas = if self.retained && !plan.masks.is_empty() {
            let mut guard = self.retained_masks.borrow_mut();
            if guard.is_none() {
                *guard = Some(crate::effects::MaskCache::new(device, queue, self.working.as_ref())?);
            }
            let cache = guard.as_mut().expect("mask cache just ensured");
            match packet_plan.and_then(|packet| packet.pinned) {
                // The camera lane: the plan's identity lets a pan reuse the last frame's atlas.
                Some(identity) if std::ptr::eq(std::sync::Arc::as_ptr(identity), plan) => {
                    crate::effects::MaskAtlas::build_cached_for(
                        device,
                        queue,
                        &mut encoder,
                        &self.layouts.mask_bgl,
                        identity,
                        mask_px_scale,
                        cache,
                        // A live canvas present packs its small masks into shared atlas passes; a readback of the
                        // same plan (a thumbnail) renders each exactly.
                        present_view.is_some(),
                        &mask_surfaces,
                    )?
                }
                _ => crate::effects::MaskAtlas::build_cached(
                    device,
                    queue,
                    &mut encoder,
                    &self.layouts.mask_bgl,
                    &plan.masks,
                    mask_px_scale,
                    cache,
                    false,
                    &mask_surfaces,
                )?,
            }
        } else {
            crate::effects::MaskAtlas::build(
                device,
                queue,
                &mut encoder,
                &self.layouts.mask_bgl,
                &plan.masks,
                mask_px_scale,
                &mask_surfaces,
                self.working.as_ref(),
            )?
        };

        // The pipelines the passes below bind, the engine's map shared: borrowed from here they outlive the pass, which
        // a render bundle recording needs (`CountedPass::Bundle`).
        let pipeline_snapshot = PipeSnap {
            map: self.pipelines.borrow().clone(),
            format: self.frame_format,
        };
        self.prepass_fills.set(prepass_jobs.len());
        // The shader-fill pre-pass (planned above): one pass per slot, each fill a padded quad over its
        // bounds, scissored to its pixel rect, at fill alpha 1 with no clip — the composite applies them.
        let mut prepass_bgs: Vec<wgpu::BindGroup> = Vec::new();
        if !prepass_jobs.is_empty() {
            use wgpu::util::DeviceExt;
            let (slot_views, depth_view, sampler) =
                self.shader_prepass_targets(eff_width_px, eff_height_px, prepass_slot_rects.len());
            let instances: Vec<PaintInstance> = prepass_jobs.iter().map(|j| j.instance).collect();
            let positions: Vec<f32> = prepass_jobs.iter().flat_map(|j| j.quad).collect();
            let types = vec![0.0f32; prepass_jobs.len() * 6];
            let inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shader Pre-pass Storage"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shader Pre-pass Quads"),
                contents: bytemuck::cast_slice(&positions),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let type_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shader Pre-pass Vertex Types"),
                contents: bytemuck::cast_slice(&types),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let inst_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shader pre-pass storage bg"),
                layout: &self.layouts.storage_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inst_buf.as_entire_binding(),
                }],
            });
            for (slot, view) in slot_views.iter().enumerate() {
                let ts = crate::gpu_profile::pass_timestamps();
                let mut pass = crate::gpu_profile::CountedPass::new(encoder.begin_render_pass(
                    &wgpu::RenderPassDescriptor {
                        label: Some("Shader Fill Pre-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Discard,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: crate::gpu_profile::render_writes(&ts),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    },
                ));
                pass.set_bind_group(0, &frame_bg, &[]);
                pass.set_bind_group(1, &inst_bg, &[]);
                pass.set_bind_group(2, mask_bg, &[]);
                pass.set_vertex_buffer(0, pos_buf.slice(..));
                pass.set_vertex_buffer(1, type_buf.slice(..));
                for (i, job) in prepass_jobs.iter().enumerate().filter(|(_, j)| j.slot == slot) {
                    pass.set_pipeline(snap_pipe(&pipeline_snapshot, (&job.module, 1, 1, BlendMode::Normal)));
                    let [x0, y0, x1, y1] = job.rect;
                    pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
                    let (v, n) = (i as u32 * 6, i as u32);
                    pass.draw(v..v + 6, n..n + 1);
                }
            }
            prepass_bgs = slot_views
                .iter()
                .map(|view| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("shader pre-pass texture bg"),
                        layout: &self.layouts.shader_image_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sampler),
                            },
                        ],
                    })
                })
                .collect();
        }
        // A shader draw's group-2 texture: a pre-pass slot, or its shader image.
        let shader_tex_bg = |key: &str| -> &wgpu::BindGroup {
            match key
                .strip_prefix("prepass:")
                .and_then(|i| i.parse::<usize>().ok())
            {
                Some(i) => &prepass_bgs[i],
                None => shader_image_textures
                    .get(key)
                    .expect("shader image pre-uploaded"),
            }
        };

        // The renderer is shared by every render at this sample count: only the outermost frame forgets what an
        // earlier (failed) frame left. A nested render (a layer capture) pairs its own splits over the outer
        // frame's saved backdrops.
        if let Some(b) = backdrop_renderer.as_ref().filter(|_| outermost) {
            b.begin_frame();
        }
        // An eligible live miss records its pass immediately. Offscreen frames first prove packet reuse
        // through the same plan or a compatible patch, then record the pass; later hits replay it.
        // A hit whose packet has no recorded pass (a patch dropped it: changed text or masks) records it again.
        // So does one whose captures changed (a composite came on or went off screen): its pass binds others.
        let hit_needs_bundle = is_hit
            && (self.prepared_frame.borrow().as_ref().is_some_and(|p| p.bundle.is_none())
                || (has_composites && !captures_reused));
        let record_bundle = prepared_eligible
            // Record offscreen commands only after packet reuse (including a compatible patch);
            // a miss draws directly.
            && (present_view.is_some() || is_hit)
            && (!is_hit || hit_needs_bundle)
            && splits.is_empty()
            && region_viewport.is_none()
            && (layer_blur_items.is_empty() || captures_packetable)
            && prepass_jobs.is_empty()
            && mask_atlas.is_retained()
            && text_arcs.is_empty()
            && (present_view.is_none() || presented_to_view);
        let mut frame_bundle: Option<wgpu::RenderBundle> = None;
        // An effect split's composite draws as the first draw of the next segment's pass instead of in a pass of
        // its own (`BackdropRenderer::draw_composite_in_frame`): same order, same pixels, one full-target pass
        // per background blur instead of two (a page of blurred cards is mostly those passes' loads and stores).
        let inline_composites = self.backdrop_inline.get() && region_viewport.is_none() && sample_count == 1;
        let mut pending_composite: Option<crate::backdrop::exec::PreparedComposite> = None;
        // A tessellated body's draw: binds the vertex-stream chunk holding its vertices when another is bound, and
        // draws them at their offset in that chunk.
        macro_rules! draw_body {
            ($pass:expr, $streams:expr, $bound:expr, $d:expr) => {
                draw_body!($pass, $streams, $bound, $d, $d.instance_index + 1)
            };
            ($pass:expr, $streams:expr, $bound:expr, $d:expr, $instance_end:expr) => {{
                let c = $streams.chunk($d.vertex_offset);
                if $bound != Some(c) {
                    let [v, t, u] = &$streams.bufs[c];
                    $pass.set_vertex_buffer(0, v.slice(..));
                    $pass.set_vertex_buffer(1, t.slice(..));
                    $pass.set_vertex_buffer(2, u.slice(..));
                    $bound = Some(c);
                }
                let o = $d.vertex_offset - $streams.bases[c];
                $pass.draw(o..o + $d.vertex_count, $d.instance_index..$instance_end);
            }};
        }
        // Consecutive transparent solid bodies draw as one draw when that is the same pixels (`SolidRun`): a
        // bristle stroke is dozens of strands of one paint, and a painting hundreds of thousands of them.
        // Keep the existing conservative opaque-depth interval boundary for transparent runs.
        // Each vertex now retains its own depth, matrix, paint and clip through its owner attribute.
        let depth_zs: Vec<f64> = {
            let mut v: Vec<f64> = plan
                .items
                .iter()
                .filter_map(|item| match item {
                    DrawItem::Body { phase: Phase::Opaque, z, .. }
                    | DrawItem::SolidShapeSdf { phase: Phase::Opaque, z, .. }
                    | DrawItem::GradientShapeSdf { phase: Phase::Opaque, z, .. }
                    | DrawItem::LayerBlurComposite { z, .. } => Some(*z),
                    _ => None,
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v
        };
        let mut merged_bodies = 0u64;
        for segment in 0..=splits.len() {
            let first = segment == 0;
            // Depth is consumed only WITHIN this frame: later segments resume
            // with LoadOp::Load, but nothing reads the depth texture after the
            // final segment (sole consumer is this attachment; next frame's
            // segment 0 clears). Discarding on the last segment skips a
            // full-target depth write-back (~4 B/px of bandwidth per frame).
            let last = segment == splits.len();
            let color_attachment = wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: resolve_view,
                ops: wgpu::Operations {
                    load: if first {
                        wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear_color[0] as f64,
                            g: clear_color[1] as f64,
                            b: clear_color[2] as f64,
                            a: clear_color[3] as f64,
                        })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            };
            let ts = crate::gpu_profile::pass_timestamps();
            let color_attachments = [Some(color_attachment)];
            let pass_desc = wgpu::RenderPassDescriptor {
                    label: Some("export pass"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: if first {
                                wgpu::LoadOp::Clear(1.0)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: if last {
                                wgpu::StoreOp::Discard
                            } else {
                                wgpu::StoreOp::Store
                            },
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: crate::gpu_profile::render_writes(&ts),
                    occlusion_query_set: None,
                    multiview_mask: None,
            };
            let mut pass = if record_bundle {
                crate::gpu_profile::CountedPass::bundle(device.create_render_bundle_encoder(
                    &wgpu::RenderBundleEncoderDescriptor {
                        label: Some("frame bundle"),
                        color_formats: &[Some(self.frame_format())],
                        depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                            format: depth_tex.format(),
                            depth_read_only: false,
                            stencil_read_only: true,
                        }),
                        sample_count,
                        multiview: None,
                    },
                ))
            } else {
                crate::gpu_profile::CountedPass::new(encoder.begin_render_pass(&pass_desc))
            };
            if let Some([x, y, w, h]) = scissor {
                pass.set_scissor_rect(x, y, w, h);
            }
            if let Some(r) = region_viewport {
                pass.set_viewport(
                    -(r.shift_x as f32),
                    -(r.shift_y as f32),
                    r.raster_w as f32,
                    r.raster_h as f32,
                    0.0,
                    1.0,
                );
            }
            // The previous split's composite, first.
            if let (Some(c), Some(renderer)) = (pending_composite.as_ref(), backdrop_renderer.as_ref()) {
                renderer.draw_composite_in_frame(&mut pass, c, (eff_width_px, eff_height_px));
            }

            // 1. Page-fill quads at depth 0.99999 (first segment only).
            if first && !page_instances.is_empty() {
                pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("page-fill", 0, sample_count, BlendMode::Normal)));
                pass.set_bind_group(0, &frame_bg, &[]);
                pass.set_bind_group(1, &page_bg, &[]);
                pass.draw(0..6, 0..page_instances.len() as u32);
            }

            // 2. Opaque tessellated bodies (depth-write ON, submission order).
            // M3b batch order mirrors `renderUnifiedPathBatch`: non-gradient
            // (solids, then patterns) first, then gradients — depth-write
            // makes the order correctness-neutral within the opaque phase.
            if opaque_draws.iter().any(|d| d.segment == segment) {
                let mut opaque_chunk: Option<usize> = None;
                if opaque_draws
                    .iter()
                    .any(|d| d.segment == segment && d.kind == PaintDrawKind::Solid)
                {
                    pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("path-solid", 0, sample_count, BlendMode::Normal)));
                    pass.set_bind_group(0, &frame_bg, &[]);
                    pass.set_bind_group(1, &opaque_bg, &[]);
                    pass.set_bind_group(2, mask_bg, &[]);
                    let mut draws = opaque_draws.iter()
                        .filter(|d| d.segment == segment && d.kind == PaintDrawKind::Solid).peekable();
                    while let Some(first) = draws.next() {
                        let mut draw = first.clone();
                        let chunk = opaque_streams.chunk(draw.vertex_offset);
                        let mut instance_end = draw.instance_index + 1;
                        while let Some(next) = draws.peek() {
                            if !self.solid_run_merging.get() || draw.vertex_count == 0 || next.vertex_count == 0
                                || next.blend != draw.blend || next.mask != draw.mask
                                || opaque_streams.chunk(next.vertex_offset + next.vertex_count - 1) != chunk { break; }
                            if draw.instanced || next.instanced {
                                // Shared resident geometry batches by occurrence, without duplicating vertices:
                                // the same range again, for the next record.
                                if !(draw.instanced && next.instanced)
                                    || next.vertex_offset != draw.vertex_offset || next.vertex_count != draw.vertex_count
                                    || next.instance_index != instance_end { break; }
                                instance_end += 1;
                            } else {
                                if next.vertex_offset != draw.vertex_offset + draw.vertex_count { break; }
                                draw.vertex_count += next.vertex_count;
                            }
                            merged_bodies += 1;
                            draws.next();
                        }
                        draw_body!(pass, opaque_streams, opaque_chunk, draw, instance_end);
                    }
                }
                // M3b: opaque patterns are unreachable in practice (Q-22
                // forces patterns transparent) but handled for completeness.
                if opaque_draws
                    .iter()
                    .any(|d| d.segment == segment && d.kind == PaintDrawKind::Pattern)
                {
                    pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("path-pattern", 0, sample_count, BlendMode::Normal)));
                    pass.set_bind_group(0, &frame_bg, &[]);
                    pass.set_bind_group(1, &opaque_pat_bg, &[]);
                    for d in opaque_draws
                        .iter()
                        .filter(|d| d.segment == segment && d.kind == PaintDrawKind::Pattern)
                    {
                        let (key, clamp) = d.tex.as_ref().expect("pattern draw has texture");
                        let tex_bg = pattern_textures
                            .get(key, *clamp)
                            .expect("pattern texture pre-uploaded");
                        pass.set_bind_group(2, tex_bg, &[]);
                        pass.set_bind_group(3, mask_bg, &[]);
                        draw_body!(pass, opaque_streams, opaque_chunk, d);
                    }
                }
                // M3b: opaque gradients (all stops opaque + opaque node.fill).
                // Q-116: post-shader overlays are unreachable here (a shader
                // fill + visible gradient stroke forces the transparent
                // phase) but included so nothing is ever silently dropped;
                // depth-write makes opaque order correctness-neutral anyway.
                let is_opaque_gradient = |d: &&PathDraw| {
                    d.segment == segment
                        && matches!(
                            d.kind,
                            PaintDrawKind::Gradient | PaintDrawKind::PostShaderGradientStroke
                        )
                };
                if opaque_draws.iter().any(|d| is_opaque_gradient(&d)) {
                    pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("path-gradient", 0, sample_count, BlendMode::Normal)));
                    pass.set_bind_group(0, &frame_bg, &[]);
                    pass.set_bind_group(1, &opaque_grad_bg, &[]);
                    pass.set_bind_group(2, mask_bg, &[]);
                    for d in opaque_draws.iter().filter(is_opaque_gradient) {
                        draw_body!(pass, opaque_streams, opaque_chunk, d);
                    }
                }
                // M3f: opaque procedural fills — pipeline switches per module
                // id; one shared CustomShaderData buffer (masked content is
                // always transparent, so opaque shader draws bind the
                // inactive mask group). Depth-write makes opaque batch order
                // correctness-neutral.
                if opaque_draws
                    .iter()
                    .any(|d| d.segment == segment && d.kind == PaintDrawKind::Shader)
                {
                    let mut current_module: Option<&str> = None;
                    let mut current_image: Option<&str> = None;
                    for d in opaque_draws
                        .iter()
                        .filter(|d| d.segment == segment && d.kind == PaintDrawKind::Shader)
                    {
                        let module = d.shader.as_deref().expect("shader draw has module id");
                        if current_module != Some(module) {
                            pass.set_pipeline(snap_pipe(&pipeline_snapshot, (module, 0, sample_count, BlendMode::Normal)));
                            pass.set_bind_group(0, &frame_bg, &[]);
                            pass.set_bind_group(1, &opaque_shader_bg, &[]);
                            // M4e: image-backed modules carry the image at
                            // group 2 (bound below) and the mask at group 3;
                            // the other 32 fills keep mask at group 2. Opaque
                            // shader draws are never masked (masked content
                            // is always transparent), so the inactive group
                            // is correct in both slots.
                            if d.tex.is_none() {
                                pass.set_bind_group(2, mask_bg, &[]);
                            } else {
                                pass.set_bind_group(3, mask_bg, &[]);
                            }
                            current_module = Some(module);
                            current_image = None;
                        }
                        if let Some((key, _)) = &d.tex {
                            if current_image != Some(key.as_str()) {
                                pass.set_bind_group(2, shader_tex_bg(key), &[]);
                                current_image = Some(key.as_str());
                            }
                        }
                        draw_body!(pass, opaque_streams, opaque_chunk, d);
                    }
                }
            }

            // Opaque analytic SDF bodies (depth-write ON; only under
            // MSAA over an opaque clear, A2C — see `sdf_opaque_lane`). Drawn as
            // one INSTANCED draw per segment: a segment's instances were pushed
            // into the storage buffer in plan order, and splits partition that
            // push order into contiguous index runs, so a segment's indices are
            // contiguous [min, max] with no gaps — `draw(6, max-min+1, 0, min)`
            // covers them in one call (firstInstance = min). Order vs the
            // tessellated opaque bodies is correctness-neutral under depth-write.
            if let Some((min, max)) = segment_index_range(&opaque_solid_sdf_draws, segment) {
                pass.set_pipeline(snap_shape_pipe(&pipeline_snapshot, "shape-solid", 0, sample_count, clear_is_opaque));
                pass.set_bind_group(0, &frame_bg, &[]);
                pass.set_bind_group(1, &opaque_solid_sdf_bg, &[]);
                pass.draw(0..6, min..max + 1);
            }
            if let Some((min, max)) = segment_index_range(&opaque_grad_sdf_draws, segment) {
                pass.set_pipeline(snap_shape_pipe(&pipeline_snapshot, "shape-gradient", 0, sample_count, clear_is_opaque));
                pass.set_bind_group(0, &frame_bg, &[]);
                pass.set_bind_group(1, &opaque_grad_sdf_bg, &[]);
                pass.draw(0..6, min..max + 1);
            }

            // 3. Transparent buckets ascending (EXACT fractional z); inside
            // each bucket: shadows first (shadowFirst), then paths
            // (`renderTransparentItemsWithText`). Depth-write OFF. M3d binds
            // each draw's nearest-ancestor mask at group 2 (inactive group
            // when unmasked).
            if !segment_buckets(segment).is_empty() {
                let mut trans_chunk: Option<usize> = None;
                // (pipeline kind, blend, mask) state tracking across draws.
                #[derive(PartialEq, Clone, Copy)]
                enum Bound {
                    Shadow,
                    Path(BlendMode),
                    /// M3b: path-gradient / path-pattern pipelines.
                    Gradient(BlendMode),
                    Pattern(BlendMode),
                    /// M3f: procedural fill; module id tracked separately.
                    Shader(BlendMode),
                    /// Analytic SDF bodies (2-group page layout, no mask).
                    SolidSdf,
                    GradientSdf,
                }
                let mut bound: Option<Bound> = None;
                let mut bound_mask: Option<Option<u64>> = None;
                // M3f: currently-bound shader module (kind == Shader only).
                let mut bound_shader: Option<String> = None;
                // Transparent-phase SDF bodies batch ACROSS z: consecutive
                // buckets holding only SDF bodies of one kind, whose instances
                // are contiguous in the storage buffer (pushed in plan order),
                // draw as one instanced draw. Instances rasterize and blend in
                // instance order, so painter's order is kept. The run is
                // flushed before any other draw. (#106: opaque SDF bodies over
                // a transparent / non-MSAA clear draw here, one per z.)
                let mut sdf_run: Option<(bool, u32, u32)> = None;
                // Transparent solid bodies merge into one draw (`SolidRun`) while each continues the run: its
                // vertices follow the run's in the same chunk and its instance is the next one. Each vertex
                // selects its own matrix, colours, clip and depth; the run binds the same blend and mask,
                // nothing is drawn between them, and no opaque depth lies between the run's first depth and its
                // own. Its triangles then rasterize and blend in the same order under the same state, and pass the
                // same depth tests, as its own draw would: the same pixels.
                struct SolidRun {
                    chunk: usize,
                    start: u32,
                    end: u32,
                    head: u32,
                    last: u32,
                    z0: f64,
                    blend: BlendMode,
                    mask: Option<u64>,
                    /// The run repeats one shared range for consecutive records, instead of joining ranges.
                    instanced: bool,
                    /// Bodies in the run.
                    bodies: u64,
                }
                let mut solid_run: Option<SolidRun> = None;
                macro_rules! flush_run {
                    () => {
                        if let Some(r) = solid_run.take() {
                            if trans_chunk != Some(r.chunk) {
                                let [v, t, u] = &trans_streams.bufs[r.chunk];
                                pass.set_vertex_buffer(0, v.slice(..));
                                pass.set_vertex_buffer(1, t.slice(..));
                                pass.set_vertex_buffer(2, u.slice(..));
                                trans_chunk = Some(r.chunk);
                            }
                            let o = r.start - trans_streams.bases[r.chunk];
                            pass.draw(o..o + (r.end - r.start), r.head..if r.instanced { r.last + 1 } else { r.head + 1 });
                            merged_bodies += r.bodies - 1;
                        }
                    };
                }
                macro_rules! flush_sdf {
                    () => {
                        if let Some((grad, start, end)) = sdf_run.take() {
                            flush_run!();
                            let want = if grad {
                                Bound::GradientSdf
                            } else {
                                Bound::SolidSdf
                            };
                            if bound != Some(want) {
                                pass.set_pipeline(snap_shape_pipe(&pipeline_snapshot, if grad {
                                        "shape-gradient"
                                    } else {
                                        "shape-solid"
                                    },
                                    1,
                                    sample_count,
                                    clear_is_opaque));
                                pass.set_bind_group(0, &frame_bg, &[]);
                                pass.set_bind_group(
                                    1,
                                    if grad {
                                        &trans_grad_sdf_bg
                                    } else {
                                        &trans_solid_sdf_bg
                                    },
                                    &[],
                                );
                                // The SDF pipelines replace the path pipeline
                                // and bind groups: the next path draw rebinds.
                                bound = Some(want);
                                bound_mask = None;
                                bound_shader = None;
                            }
                            pass.draw(0..6, start..end);
                            let mut s = self.prepared_stats.get();
                            s.sdf_draws += 1;
                            s.sdf_instances += u64::from(end - start);
                            self.prepared_stats.set(s);
                        }
                    };
                }
                macro_rules! push_sdf {
                    ($grad:expr, $idx:expr) => {
                        let idx: u32 = $idx;
                        match sdf_run {
                            Some((g, _, ref mut end)) if g == $grad && *end == idx => {
                                *end = idx + 1
                            }
                            _ => {
                                flush_sdf!();
                                sdf_run = Some(($grad, idx, idx + 1));
                            }
                        }
                    };
                }
                for (_, bucket) in segment_buckets(segment) {
                    if !bucket.shadows.is_empty() || !bucket.layer_blur.is_empty() {
                        flush_sdf!();
                        flush_run!();
                    }
                    // Shadows at this z (transparent shadow pipeline:
                    // depth-write OFF; quad = 6 vertices, firstInstance =
                    // packed offset — `renderPass.draw(6, 1, 0, offset)`).
                    if !bucket.shadows.is_empty() {
                        if bound != Some(Bound::Shadow) {
                            pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("path-shadow", 1, sample_count, BlendMode::Normal)));
                            pass.set_bind_group(0, &frame_bg, &[]);
                            pass.set_bind_group(1, &shadow_bg, &[]);
                            // Shadows always bind the inactive mask group
                            // (`bindInactiveMaskGroup`).
                            pass.set_bind_group(2, mask_bg, &[]);
                            bound = Some(Bound::Shadow);
                            bound_mask = Some(None);
                        }
                        for idx in bucket.shadows {
                            pass.draw(0..6, *idx..*idx + 1);
                        }
                    }
                    // Layer-blur composite quads — the diverted
                    // node's entire paint. Uses its own 2-group [frame,
                    // composite] pipeline (no vertex buffers, like the SDF
                    // quads), so it invalidates the tracked path bindings.
                    if !bucket.layer_blur.is_empty() {
                        if let Some(renderer) = layer_blur_renderer.as_ref() {
                            let mut drew_composite = false;
                            // Glow composites blend with their mode; the
                            // rest draw source-over. Same layout, so only the
                            // pipeline switches.
                            let mut bound_blend: Option<crate::seams::CaptureBlend> = None;
                            for idx in bucket.layer_blur {
                                if let Some(bg) = layer_blur_bind_groups[*idx].as_ref() {
                                    let filters = &layer_blur_items[*idx].1.filters;
                                    let blend = filters
                                        .shadow
                                        .as_ref()
                                        .map_or(filters.composite_blend, |s| s.blend);
                                    if bound_blend != Some(blend) {
                                        pass.set_pipeline(renderer.blend_composite_pipeline(blend));
                                        bound_blend = Some(blend);
                                    }
                                    if !drew_composite {
                                        pass.set_bind_group(0, &frame_bg, &[]);
                                        drew_composite = true;
                                    }
                                    pass.set_bind_group(1, bg, &[]);
                                    // The node's clip mask (a clip-mask group
                                    // above it), else the inactive group.
                                    let mask = layer_blur_items[*idx]
                                        .5
                                        .and_then(|id| mask_atlas.bind_group(id))
                                        .unwrap_or(mask_bg);
                                    pass.set_bind_group(2, mask, &[]);
                                    pass.draw(0..6, 0..1);
                                }
                            }
                            if drew_composite {
                                bound = None;
                                bound_mask = None;
                                bound_shader = None;
                            }
                        }
                    }
                    // Analytic SDF bodies at this z — after the
                    // shadows, before the tessellated paths (the WebGPU
                    // intra-bucket order is shadows → SDF solid → SDF gradient →
                    // paths → text). The SDF pipelines use the 2-group page
                    // layout (no mask group), so they invalidate the path
                    // pipeline + bind-group state tracked by `bound`.
                    for idx in bucket.solid_sdf {
                        push_sdf!(false, *idx);
                    }
                    for idx in bucket.gradient_sdf {
                        push_sdf!(true, *idx);
                    }
                    if !bucket.paths.is_empty() || !bucket.texts.is_empty() {
                        flush_sdf!();
                    }
                    // M3b intra-bucket slot order (`renderUnifiedPathBatch`):
                    // non-gradient (solids, then patterns) → gradients →
                    // custom shaders → post-shader gradient-stroke overlays.
                    // Pattern-fill overlays are Gradient draws, so they
                    // composite over their pattern body; SHADER-fill overlays
                    // draw LAST so the shader body cannot paint over the
                    // inner half of the centered stroke band
                    // (`postShaderGradientStrokePaths`,
                    // `WebGPUPathRenderer.ts:1190-1194`; Q-116).
                    for slot_kind in [
                        PaintDrawKind::Solid,
                        PaintDrawKind::Pattern,
                        PaintDrawKind::Gradient,
                        PaintDrawKind::Shader,
                        PaintDrawKind::PostShaderGradientStroke,
                    ] {
                        for i in bucket.paths {
                            let d = &trans_draws[*i];
                            if d.kind != slot_kind {
                                continue;
                            }
                            let z = f64::from_bits(d.z_bucket);
                            if let Some(r) = solid_run.as_mut() {
                                let extends = d.kind == PaintDrawKind::Solid
                                    && d.blend == r.blend
                                    && d.mask == r.mask
                                    && d.vertex_count > 0
                                    && d.instanced == r.instanced
                                    && if r.instanced {
                                        d.vertex_offset == r.start && d.vertex_count == r.end - r.start
                                    } else { d.vertex_offset == r.end }
                                    && trans_streams.chunk(d.vertex_offset) == r.chunk
                                    && trans_streams.chunk(d.vertex_offset + d.vertex_count - 1) == r.chunk
                                    // Joined ranges carry their records per vertex, in any slot order; a
                                    // repeated shared range needs the next record.
                                    && (!r.instanced || d.instance_index == r.last + 1)
                                    && depth_zs.partition_point(|x| *x <= r.z0) == depth_zs.partition_point(|x| *x <= z);
                                if extends {
                                    if !r.instanced { r.end += d.vertex_count; }
                                    r.last = d.instance_index;
                                    r.bodies += 1;
                                    continue;
                                }
                            }
                            flush_run!();
                            // Pattern binds its texture at group 2 and the
                            // mask at group 3 (MASK_BIND_GROUP_INDEX_
                            // IMAGE_BACKED); every other kind takes the
                            // shared group-2 mask logic below.
                            let mut mask_at_group_3 = false;
                            match d.kind {
                                PaintDrawKind::Shader => {
                                    // M3f: procedural fill — per-module
                                    // pipeline, shared CustomShaderData
                                    // buffer.
                                    let module =
                                        d.shader.as_deref().expect("shader draw has module id");
                                    if bound != Some(Bound::Shader(d.blend))
                                        || bound_shader.as_deref() != Some(module)
                                    {
                                        pass.set_pipeline(snap_pipe(&pipeline_snapshot, (module, 1, sample_count, d.blend)));
                                        pass.set_bind_group(0, &frame_bg, &[]);
                                        pass.set_bind_group(1, &trans_shader_bg, &[]);
                                        bound = Some(Bound::Shader(d.blend));
                                        bound_shader = Some(module.to_string());
                                        bound_mask = None; // force mask rebind
                                    }
                                    // M4e: image-backed fills bind the shader
                                    // image at group 2; mask moves to group 3
                                    // (`bindMaskGroup(node, pipeline, 3)` —
                                    // WebGPUPathRenderer.ts:2184).
                                    if let Some((key, _)) = &d.tex {
                                        pass.set_bind_group(2, shader_tex_bg(key), &[]);
                                        bound_mask = None; // group 2 holds texture
                                        mask_at_group_3 = true;
                                    }
                                }
                                // M3b: gradient draws — mask at group 2 like
                                // solids; gradient storage buffer at group 1.
                                // Q-116: post-shader overlays bind the SAME
                                // path-gradient pipeline/buffers — only the
                                // slot position differs.
                                PaintDrawKind::Gradient
                                | PaintDrawKind::PostShaderGradientStroke => {
                                    if bound != Some(Bound::Gradient(d.blend)) {
                                        pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("path-gradient", 1, sample_count, d.blend)));
                                        pass.set_bind_group(0, &frame_bg, &[]);
                                        pass.set_bind_group(1, &trans_grad_bg, &[]);
                                        bound = Some(Bound::Gradient(d.blend));
                                        bound_mask = None;
                                    }
                                }
                                // M3b: pattern draws — texture bind group +
                                // image-backed mask index.
                                PaintDrawKind::Pattern => {
                                    if bound != Some(Bound::Pattern(d.blend)) {
                                        pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("path-pattern", 1, sample_count, d.blend)));
                                        pass.set_bind_group(0, &frame_bg, &[]);
                                        pass.set_bind_group(1, &trans_pat_bg, &[]);
                                        bound = Some(Bound::Pattern(d.blend));
                                    }
                                    let (key, clamp) =
                                        d.tex.as_ref().expect("pattern draw has texture");
                                    let tex_bg = pattern_textures
                                        .get(key, *clamp)
                                        .expect("pattern texture pre-uploaded");
                                    pass.set_bind_group(2, tex_bg, &[]);
                                    bound_mask = None; // group 2 holds texture
                                    mask_at_group_3 = true;
                                }
                                PaintDrawKind::Solid => {
                                    if bound != Some(Bound::Path(d.blend)) {
                                        pass.set_pipeline(snap_pipe(&pipeline_snapshot, ("path-solid", 1, sample_count, d.blend)));
                                        pass.set_bind_group(0, &frame_bg, &[]);
                                        pass.set_bind_group(1, &trans_bg, &[]);
                                        bound = Some(Bound::Path(d.blend));
                                        bound_mask = None; // force mask rebind
                                    }
                                }
                            }
                            if mask_at_group_3 {
                                let bg = d
                                    .mask
                                    .and_then(|id| mask_atlas.bind_group(id))
                                    .unwrap_or(mask_bg);
                                pass.set_bind_group(3, bg, &[]);
                            } else if bound_mask != Some(d.mask) {
                                let bg = d
                                    .mask
                                    .and_then(|id| mask_atlas.bind_group(id))
                                    .unwrap_or(mask_bg);
                                pass.set_bind_group(2, bg, &[]);
                                bound_mask = Some(d.mask);
                            }
                            if d.kind == PaintDrawKind::Solid && d.vertex_count > 0 && self.solid_run_merging.get() {
                                // Held as a run: the next solid body may join it (drawn at the next flush).
                                solid_run = Some(SolidRun {
                                    chunk: trans_streams.chunk(d.vertex_offset),
                                    start: d.vertex_offset,
                                    end: d.vertex_offset + d.vertex_count,
                                    head: d.instance_index,
                                    last: d.instance_index,
                                    z0: z,
                                    blend: d.blend,
                                    mask: d.mask,
                                    instanced: d.instanced,
                                    bodies: 1,
                                });
                            } else {
                                draw_body!(pass, trans_streams, trans_chunk, d);
                            }
                        }
                    }
                    // M3a: text LAST in the bucket (shadows → paths → text,
                    // `renderTransparentItemsWithText`). Text swaps the
                    // pipeline, group 0/1 binds and vertex/index buffers —
                    // restore the path state for the next bucket.
                    if !bucket.texts.is_empty() {
                        flush_run!();
                        for t in bucket.texts {
                            // vork: moda-text-draw's draw_batch records into a render pass; a bundle is recorded only
                            // on the retained camera lane, which vork does not use. See PROVENANCE.md.
                            match &mut pass {
                                crate::gpu_profile::CountedPass::Pass(p) => {
                                    self.text.draw_batch(p, &text_prepared.chunks[*t / TEXT_CHUNK], *t % TEXT_CHUNK)
                                }
                                crate::gpu_profile::CountedPass::Bundle(_) => {
                                    unreachable!("text in a recorded bundle: the retained camera lane is not ported")
                                }
                            }
                        }
                        bound = None;
                        bound_mask = None;
                        bound_shader = None;
                        trans_chunk = None;
                    }
                }
                flush_sdf!();
                flush_run!();
            }
            if let Some(bundle) = pass.finish_bundle("frame bundle") {
                let mut pass = crate::gpu_profile::CountedPass::new(encoder.begin_render_pass(&pass_desc));
                pass.execute_bundles(std::iter::once(&bundle));
                drop(pass);
                frame_bundle = Some(bundle);
            }

            // M3e: between segments, run the backdrop effect chain
            // (copy region → refract/blur/pixelate/dither → composite).
            // On the direct path the segments rendered into the
            // swapchain itself, so the effect copies its region out of (and
            // composites back into) the swapchain; otherwise it uses the
            // offscreen target. `execute_split` only ever touches the small
            // region either way — identical effect, just a different backing.
            if segment < splits.len() {
                let (split_source, split_dest): (&wgpu::Texture, &wgpu::TextureView) =
                    if backdrop_direct {
                        (
                            present_copy_src
                                .expect("backdrop_direct implies copy-source swapchain"),
                            present_view.expect("backdrop_direct implies present view"),
                        )
                    } else {
                        (&target, &target_view)
                    };
                let renderer = backdrop_renderer
                    .as_ref()
                    .expect("backdrop renderer exists when splits exist");
                pending_composite = None;
                if inline_composites {
                    pending_composite = renderer.execute_split_deferred(
                        device,
                        queue,
                        &mut encoder,
                        split_source,
                        (eff_width_px, eff_height_px),
                        splits[segment],
                    )?;
                }
                if pending_composite.is_none() {
                    renderer.execute_split(
                        device,
                        queue,
                        &mut encoder,
                        split_source,
                        split_dest,
                        msaa_view.as_ref(),
                        (eff_width_px, eff_height_px),
                        splits[segment],
                    )?;
                }
                // Hundreds of splits in one command buffer lose the Metal device
                // (800 overlay nodes, or 300 paint-stack groups with a blended
                // layer each): submit the work so far every SPLIT_FLUSH splits.
                if (segment + 1) % SPLIT_FLUSH == 0 {
                    let fresh = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Page Export Commands (after splits)"),
                    });
                    crate::scratch::submit(queue, [std::mem::replace(&mut encoder, fresh).finish()]);
                }
            }
        }
        crate::scratch::submit(queue, [encoder.finish()]);
        if merged_bodies > 0 {
            let mut s = self.prepared_stats.get();
            s.merged_bodies = s.merged_bodies.wrapping_add(merged_bodies);
            self.prepared_stats.set(s);
        }
        // M2 prepared-frame bookkeeping. A hit just replayed reused state; an
        // eligible miss (re)builds the packet from THIS frame's text + the
        // post-upload pool epoch so the next identical camera frame can reuse it.
        if is_hit {
            let mut s = self.prepared_stats.get();
            s.hits = s.hits.wrapping_add(1);
            self.prepared_stats.set(s);
            if let Some(p) = self.prepared_frame.borrow_mut().as_mut() {
                // A pass recorded now leaves out the blurs this frame drew plainly (the same count as the packet's).
                p.motion_blur_px = self.motion_blur_min_px.get();
                if has_composites && !captures_reused {
                    // Other captures than the packet's: keep this frame's, and its pass (or none).
                    p.captures = Some(PacketCaptures {
                        on_screen: capture_on_screen.clone(),
                        results: layer_blur_results.clone(),
                    });
                    p.bundle = frame_bundle.take();
                } else if let Some(bundle) = frame_bundle.take() {
                    p.bundle = Some(bundle);
                }
            }
        } else if prepared_eligible {
            // Record the clock these buffers were uploaded at + whether
            // any fill re-threads it, so a later clock-only frame re-encodes
            // instead of replaying a stale `shader_params` upload (frozen shader).
            let has_clock_fill = plan.items.iter().any(|item| match item {
                DrawItem::Body {
                    paint: BodyPaint::Shader(b),
                    ..
                } => b.clock_offsets.as_ref().is_some_and(|o| !o.is_empty()),
                _ => false,
            });
            *self.prepared_frame.borrow_mut() = Some(PreparedFrame {
                key: packet_plan.expect("eligible ⇒ Some").key,
                pinned: packet_plan.and_then(|packet| packet.pinned).cloned(),
                width: eff_width_px,
                height: eff_height_px,
                sample_count,
                clear_is_opaque,
                live_masks: present_view.is_some(),
                pool_epoch: self.pool_epoch.get(),
                shader_time_s: present_clock,
                has_clock_fill,
                text: text_prepared.clone(),
                text_batches: text_arcs.iter().map(|b| (*b).clone()).collect(),
                text_host_generation: crate::text_draw::text_host_generation(),
                text_greek_sig: greek_sig,
                simplified_blurs: self.simplified_blurs.get(),
                motion_blur_px: self.motion_blur_min_px.get(),
                bundle: frame_bundle,
                captures: has_composites.then(|| PacketCaptures {
                    on_screen: capture_on_screen.clone(),
                    results: layer_blur_results.clone(),
                }),
                // What the buffers above were uploaded from.
                records: PacketRecords {
                    at: record_at,
                    page: Column::of(page_instances),
                    solid: [Column::of(opaque_instances), Column::of(trans_instances)],
                    gradient: [Column::of(opaque_grad_instances), Column::of(trans_grad_instances)],
                    pattern: [Column::of(opaque_pat_instances), Column::of(trans_pat_instances)],
                    shader: [Column::of(opaque_shader_instances), Column::of(trans_shader_instances)],
                    shadow: Column::of(shadow_instances),
                    solid_sdf: [Column::of(opaque_solid_sdf), Column::of(trans_solid_sdf)],
                    gradient_sdf: [Column::of(opaque_grad_sdf), Column::of(trans_grad_sdf)],
                },
            });
            let mut s = self.prepared_stats.get();
            s.misses = s.misses.wrapping_add(1);
            s.rebuilds = s.rebuilds.wrapping_add(1);
            self.prepared_stats.set(s);
        }
        if let Some(carry) = text_carry.take() {
            self.defer_drop(Box::new(carry));
        }
        Ok((target, presented_to_view))
    }

    /// The packet's recorded pass, when this frame is a hit on it (the same plan, target, sample count, clear
    /// opacity, pool epoch and — for a clock-driven plan — clock) and it holds one.
    #[allow(clippy::too_many_arguments)]
    /// Bring the prepared packet to the plan `new` names, in place, and adopt it as the packet's plan. The
    /// packet must be of the state the change starts from: for an owned plan the revision its owner's account
    /// (`PlanChange`) names, for a shared plan the one the packet pins. Every changed item must keep its draw
    /// (the same kind, phase, pipeline, paint family and texture, z, blend and mask; a body its vertex count).
    ///
    /// The work follows the account. Items the owner placed are packed and written, with nothing to compare
    /// them with but the record held; items it prepared again are checked against the item they replace; only
    /// a plan that took another's place whole (and a shared plan, which brings no account) has every item
    /// compared.
    ///
    /// A changed item's record is written where the packet's collect put it (`PacketRecords`): it is compared
    /// with the record the buffer holds, and each column goes up once, from its first changed record to its
    /// last. A body with new geometry also writes its vertices at the offset the collect gave them. The
    /// packet's buffers, text and recorded pass stay valid (no buffer is replaced and nothing moves), so the
    /// frame is then a hit. The bytes written are the ones a rebuild would upload (`pack_body` is the collect's
    /// own packer), so a patched frame renders exactly as a rebuilt one.
    ///
    /// A changed text layer re-prepares only its chunk of the packet's text (as a rebuild would), and a changed
    /// text layer or mask table drops the packet's recorded pass (it binds the old text and mask groups): the next
    /// frame encodes with the packet's buffers and records the pass again.
    ///
    /// Returns `Err(reason)`, having changed nothing, when it can't: a packet of another state, another item
    /// count or order, a split, a draw that moves between streams, greeked text, a text host that changed, or
    /// a pooled buffer that isn't there.
    fn patch_prepared(
        &self,
        new: PacketPlan<'_>,
        new_plan: &CoveragePlan,
        sample_count: u32,
        clear_is_opaque: bool,
        present_shader_time: Option<f64>,
    ) -> std::result::Result<(), &'static str> {
        // The records leave the packet while they are patched: nothing below holds the packet across a call.
        let (key, pinned, mut records) = {
            let mut pf = self.prepared_frame.borrow_mut();
            let p = pf.as_mut().ok_or("no packet")?;
            (p.key, p.pinned.clone(), std::mem::take(&mut p.records))
        };
        // What the packet's state is compared with: the owner's account from that state, or the pinned plan.
        let base = match (new.change, &pinned) {
            (Some(change), _) if key == (PlanKey { owner: new.key.owner, revision: change.from }) => Some(match change.items {
                ChangedItems::Written { placed, replaced } => PatchBase::Written { placed, replaced },
                ChangedItems::Whole(old) => PatchBase::Whole(old),
            }),
            (None, Some(old)) if new.pinned.is_some() => Some(PatchBase::Whole(&**old)),
            _ => None,
        };
        let patched = match base {
            Some(base) => {
                self.patch_records(base, new, new_plan, &mut records, sample_count, clear_is_opaque, present_shader_time)
            }
            None => Err("plan base"),
        };
        if patched.is_err() {
            for (_, column) in records.columns() {
                column.discard();
            }
        }
        if let Some(p) = self.prepared_frame.borrow_mut().as_mut() {
            p.records = records;
        }
        patched
    }

    /// [`Engine::patch_prepared`] over the packet's records. On `Err` the columns may hold pending records, which
    /// the caller discards; nothing else has changed.
    #[allow(clippy::too_many_arguments)]
    fn patch_records(
        &self,
        base: PatchBase<'_>,
        new: PacketPlan<'_>,
        new_plan: &CoveragePlan,
        records: &mut PacketRecords,
        sample_count: u32,
        clear_is_opaque: bool,
        present_shader_time: Option<f64>,
    ) -> std::result::Result<(), &'static str> {
        if records.at.len() != new_plan.items.len()
            || matches!(base, PatchBase::Whole(old) if old.items.len() != new_plan.items.len())
        {
            return Err("item count");
        }
        if self.vertex_chunks.get() != [1, 1] {
            return Err("split vertex stream");
        }
        // The mask table is rebuilt by every encode (`MaskAtlas`); only the recorded pass holds the old one. Items
        // written in place keep the plan's mask table (a placement touches no mask, and the planner prepares a
        // node again only into the masks the plan has).
        let masks_changed = match base {
            PatchBase::Whole(old) => !std::ptr::eq(old.masks.as_ptr(), new_plan.masks.as_ptr()) && old.masks != new_plan.masks,
            PatchBase::Written { .. } => false,
        };
        let mut text_changed = false;
        let sdf_opaque_lane = sample_count > 1 && clear_is_opaque;
        // The packet's vertex residency (a slot-owned body's new geometry takes its old one's place in it), and
        // the moves to record once the patch is known to apply.
        let residency_guard = self.vertex_residency.borrow();
        let residency = &*residency_guard;
        let mut resident_moves: Vec<(usize, std::sync::Arc<TessellatedGeometry>, std::sync::Arc<TessellatedGeometry>, u32, Option<u32>)> =
            Vec::new();
        let mut new_uses: Option<[crate::residency::HashMap<usize, u32>; 2]> = None;
        // The vertices of bodies whose geometry changed: (pooled buffer, byte offset, bytes).
        let mut writes: Vec<(&'static str, u64, Vec<u8>)> = Vec::new();
        let mut items = 0u64;
        let ph = |p: &Phase| match p {
            Phase::Opaque => 0usize,
            Phase::Transparent => 1usize,
        };
        let bytes = |x: &[u8]| x.to_vec();
        // One item of the plan, at the place the collect gave its record, with the item it replaces; `None` for
        // a placed body, whose draw stands by its owner's promise.
        let mut visit = |at: usize, old: Option<&DrawItem>| -> std::result::Result<(), &'static str> {
            let (Some(n), Some(&rec)) = (new_plan.items.get(at), records.at.get(at)) else {
                return Err("item count");
            };
            let Some(o) = old else {
                let DrawItem::Body { phase, geometry, instance, paint, .. } = n else {
                    return Err("placed item kind");
                };
                let packed = self.pack_body(paint, instance, geometry, rec.first_vertex, present_shader_time);
                let changed = records.put_body(ph(phase), rec.index, &packed)?;
                // As below: a procedural fill drawn through the shader pre-pass rebuilds the packet when it changes.
                if self.shader_prepass && matches!(paint, BodyPaint::Shader(_)) && changed {
                    return Err("shader prepass");
                }
                if changed {
                    items += 1;
                }
                return Ok(());
            };
            match (o, n) {
                (DrawItem::PageFill { .. }, DrawItem::PageFill { instance: b }) => {
                    if records.page.put(rec.index, b)? {
                        items += 1;
                    }
                }
                (
                    DrawItem::Body { phase: pa, geometry: ga, blend_mode: ba, z: za, mask: ma, paint: qa, node_id: na, .. },
                    DrawItem::Body { phase: pb, geometry: gb, instance: ib, blend_mode: bb, z: zb, mask: mb, paint: qb, node_id: nb, .. },
                ) => {
                    if pa != pb || za.to_bits() != zb.to_bits() || ba != bb || ma.as_ref().map(|m| m.mask_id) != mb.as_ref().map(|m| m.mask_id) {
                        return Err("body draw");
                    }
                    if std::mem::discriminant(qa) != std::mem::discriminant(qb) {
                        return Err("body paint family");
                    }
                    // The same family draws through the same pipeline when it binds the same module and texture.
                    if self.body_binding(qa) != self.body_binding(qb) {
                        return Err("body pipeline");
                    }
                    let same_geometry = std::sync::Arc::ptr_eq(ga, gb);
                    if !same_geometry
                        && (ga.vertex_count != gb.vertex_count
                            || ga.positions.len() != gb.positions.len()
                            || ga.vertex_types.len() != gb.vertex_types.len()
                            || (ga.stroke_uv.len() == 2 * ga.vertex_types.len())
                                != (gb.stroke_uv.len() == 2 * gb.vertex_types.len()))
                    {
                        return Err("body vertex count");
                    }
                    let k = ph(pa);
                    // A resident solid's record sits at a slot, and slots belong to draw ids. The new plan's
                    // body here must carry the same id, or the next frame would address another body's
                    // record: a reorder or a replaced node goes through collection, which gives slots by id.
                    if matches!(qa, BodyPaint::Solid) && residency[k].is_some() && na != nb {
                        return Err("solid body identity");
                    }
                    // The owner words the body's vertices carry: a resident solid's are its slot's (or the
                    // draw's, for a shared geometry); a packed one's are its place in the stream. Either is
                    // its record's index.
                    let owner = matches!(qa, BodyPaint::Solid).then(|| match residency[k].as_ref() {
                        Some(r) if r.shared.contains(&(std::sync::Arc::as_ptr(ga) as usize)) => INSTANCE_OWNER,
                        _ => rec.index,
                    });
                    let vo = rec.first_vertex;
                    // A resident range that names one body's slot is that body's alone: its new geometry (the
                    // same vertex count) takes its place. A range other bodies may share is never overwritten;
                    // that change goes through ordinary collection.
                    if !same_geometry {
                        if let Some(r) = residency[k].as_ref() {
                            if !matches!(owner, Some(slot) if slot != INSTANCE_OWNER) {
                                return Err("resident geometry changed");
                            }
                            if r.offset_of(ga, owner) != Some(vo) {
                                return Err("vertex residency");
                            }
                            // The collect holds a geometry several solid bodies draw once, for instanced draws. A
                            // body taking a geometry another body draws changes that layout, and a patch uploads
                            // no vertices: collection lays the streams out again. Counted once, on the first
                            // geometry change, so placement frames never pay for it.
                            let uses = new_uses.get_or_insert_with(|| solid_geometry_uses(&new_plan.items));
                            if uses[k].get(&(std::sync::Arc::as_ptr(gb) as usize)).copied().unwrap_or(0) > 1 {
                                return Err("resident geometry becomes shared");
                            }
                            resident_moves.push((k, ga.clone(), gb.clone(), vo, owner));
                        }
                    }
                    let mut changed =
                        records.put_body(k, rec.index, &self.pack_body(qb, ib, gb, vo, present_shader_time))?;
                    // A procedural fill may be drawn through the shader pre-pass. Its jobs (slots, rects, the
                    // composite draw) are derived from the packet's instances by every frame's collect, a hit's
                    // included, so an unchanged fill carries over as on a hit; a changed one rebuilds the packet.
                    if self.shader_prepass && matches!(qb, BodyPaint::Shader(_)) && (!same_geometry || changed) {
                        return Err("shader prepass");
                    }
                    if !same_geometry {
                        if ga.positions != gb.positions {
                            writes.push((VERTEX_STREAM_LABELS[k][0], u64::from(vo) * 8, bytes(bytemuck::cast_slice(&gb.positions))));
                            changed = true;
                        }
                        if ga.vertex_types != gb.vertex_types {
                            writes.push((VERTEX_STREAM_LABELS[k][1], u64::from(vo) * 4, bytes(bytemuck::cast_slice(&gb.vertex_types))));
                            changed = true;
                        }
                        let uv = |g: &TessellatedGeometry| -> Vec<f32> {
                            let mut out = Vec::new();
                            append_body_aux(&mut out, g, owner);
                            out
                        };
                        let (ua, ub) = (uv(ga), uv(gb));
                        if bytemuck::cast_slice::<f32, u8>(&ua) != bytemuck::cast_slice::<f32, u8>(&ub) {
                            writes.push((VERTEX_STREAM_LABELS[k][2], u64::from(vo) * 8, bytes(bytemuck::cast_slice(&ub))));
                            changed = true;
                        }
                    }
                    if changed {
                        items += 1;
                    }
                }
                (DrawItem::Shadow { z: za, .. }, DrawItem::Shadow { z: zb, instance: b, .. }) => {
                    if za.to_bits() != zb.to_bits() {
                        return Err("shadow z");
                    }
                    if records.shadow.put(rec.index, &**b)? {
                        items += 1;
                    }
                }
                (DrawItem::Text(a), DrawItem::Text(b)) => {
                    if !std::sync::Arc::ptr_eq(a, b) {
                        text_changed = true;
                    }
                }
                (
                    DrawItem::SolidShapeSdf { phase: pa, z: za, opaque_sdf_covers_shadow: ca, .. },
                    DrawItem::SolidShapeSdf { phase: pb, instance: b, z: zb, opaque_sdf_covers_shadow: cb, .. },
                ) => {
                    if pa != pb || za.to_bits() != zb.to_bits() || ca != cb {
                        return Err("sdf draw");
                    }
                    let lane = if *pa == Phase::Opaque && sdf_opaque_lane { 0 } else { 1 };
                    if records.solid_sdf[lane].put(rec.index, &**b)? {
                        items += 1;
                    }
                }
                (
                    DrawItem::GradientShapeSdf { phase: pa, z: za, opaque_sdf_covers_shadow: ca, .. },
                    DrawItem::GradientShapeSdf { phase: pb, instance: b, z: zb, opaque_sdf_covers_shadow: cb, .. },
                ) => {
                    if pa != pb || za.to_bits() != zb.to_bits() || ca != cb {
                        return Err("sdf draw");
                    }
                    let lane = if *pa == Phase::Opaque && sdf_opaque_lane { 0 } else { 1 };
                    if records.gradient_sdf[lane].put(rec.index, &**b)? {
                        items += 1;
                    }
                }
                (
                    DrawItem::LayerBlurComposite { node_id: na, z: za, group: ga, page_clip_bounds: ca, clip_rrect: ra, mask: ma },
                    DrawItem::LayerBlurComposite { node_id: nb, z: zb, group: gb, page_clip_bounds: cb, clip_rrect: rb, mask: mb },
                ) => {
                    // The packet keeps its capture results by node: a composite patches when it draws the same
                    // capture (content key) through the same quad, clip and depth.
                    let glow = |g: &crate::seams::LayerBlurGroup| {
                        (g.filters.shadow.as_ref().map(|s| s.blend), g.filters.composite_blend)
                    };
                    let same_capture = std::sync::Arc::ptr_eq(ga, gb)
                        || (ga.cache_key.is_some()
                            && ga.cache_key == gb.cache_key
                            && ga.region_canvas == gb.region_canvas
                            && (ga.width_px, ga.height_px) == (gb.width_px, gb.height_px)
                            && glow(ga) == glow(gb));
                    if na != nb || za.to_bits() != zb.to_bits() || ca != cb || ra != rb || ma != mb || !same_capture {
                        return Err("capture");
                    }
                }
                // A background effect runs from its item every frame (its region, coverage and op are read at
                // execution, and upload nothing into the packet's buffers): the same layer's effect patches, at
                // whatever camera it was projected for. A pan over background blurs re-projects every split,
                // which rebuilt (and re-uploaded) the whole packet each frame.
                (DrawItem::BackdropSplit(a), DrawItem::BackdropSplit(b)) if a.node_id == b.node_id => {}
                (DrawItem::BackdropSplit(_), _) | (DrawItem::LayerBlurComposite { .. }, _) => return Err("split"),
                _ => return Err("item kind"),
            }
            Ok(())
        };
        match base {
            PatchBase::Written { placed, replaced } => {
                for &at in placed {
                    visit(at as usize, None)?;
                }
                for (at, old) in replaced {
                    visit(*at as usize, Some(old))?;
                }
            }
            PatchBase::Whole(old) => {
                for (at, old) in old.items.iter().enumerate() {
                    visit(at, Some(old))?;
                }
            }
        }
        // Every write lands inside its pooled buffer (the packet's collect sized each to its stream), and every
        // column with a changed record still has the buffer it was uploaded to.
        {
            let g = self.retained_frame.borrow();
            for (label, offset, data) in &writes {
                match g.bufs.get(label) {
                    Some(e) if offset + data.len() as u64 <= e.capacity => {}
                    _ => return Err("pooled buffer"),
                }
            }
            for (label, column) in records.columns() {
                if column.is_pending() && !g.bufs.get(&label).is_some_and(|e| column.bytes() <= e.capacity) {
                    return Err("pooled buffer");
                }
            }
        }
        // Changed text: the new plan's batches, each chunk kept when its batches are the packet's (the rebuild's
        // `TextCarry` rule), else prepared again against the retained frame uniform the packet's text binds.
        let text = if text_changed {
            let pf = self.prepared_frame.borrow();
            let p = pf.as_ref().ok_or("no packet")?;
            if p.text_greek_sig != 0 {
                return Err("greeked text");
            }
            if p.text_host_generation != crate::text_draw::text_host_generation() || p.sample_count != sample_count {
                return Err("text host");
            }
            let uniform = self
                .retained_frame
                .borrow()
                .frame_uniform
                .as_ref()
                .map(|(b, _)| b.clone())
                .ok_or("text uniform")?;
            let arcs: Vec<std::sync::Arc<crate::seams::TextQuadBatch>> = new_plan
                .items
                .iter()
                .filter_map(|i| match i {
                    DrawItem::Text(b) => Some(b.clone()),
                    _ => None,
                })
                .collect();
            let mut chunks = Vec::with_capacity(arcs.len().div_ceil(TEXT_CHUNK));
            let mut prepared_chunks = 0u64;
            for (i, chunk) in arcs.chunks(TEXT_CHUNK).enumerate() {
                let start = i * TEXT_CHUNK;
                let kept = p
                    .text_batches
                    .get(start..start + chunk.len())
                    .filter(|old| old.iter().zip(chunk).all(|(a, b)| std::sync::Arc::ptr_eq(a, b)))
                    .and_then(|_| p.text.chunks.get(i).cloned());
                match kept {
                    Some(c) => chunks.push(c),
                    None => {
                        let batches: Vec<&crate::seams::TextQuadBatch> = chunk.iter().map(|b| &**b).collect();
                        let prepared = self
                            .text
                            .prepare(&self.text_gpu(), &uniform, &batches, sample_count)
                            .map_err(|_| "text prepare")?;
                        self.add_pf_text_bytes(prepared.uploaded_bytes());
                        chunks.push(std::sync::Arc::new(prepared));
                        prepared_chunks += 1;
                    }
                }
            }
            let mut st = self.prepared_stats.get();
            st.patched_text_chunks = st.patched_text_chunks.wrapping_add(prepared_chunks);
            self.prepared_stats.set(st);
            Some((std::sync::Arc::new(TextChunks { chunks }), arcs))
        } else {
            None
        };
        drop(residency_guard);
        {
            let mut res = self.vertex_residency.borrow_mut();
            for (k, old, new_g, vo, owner) in resident_moves {
                if let Some(r) = res[k].as_mut() {
                    r.at.remove(&(std::sync::Arc::as_ptr(&old) as usize, owner));
                    r.at.insert(
                        (std::sync::Arc::as_ptr(&new_g) as usize, owner),
                        (vo, new_g.vertex_count, std::sync::Arc::downgrade(&new_g)),
                    );
                }
            }
        }
        let mut written = 0u64;
        {
            let g = self.retained_frame.borrow();
            for (label, offset, data) in &writes {
                let e = &g.bufs[label];
                crate::gpu_profile::count_buffer_write(data.len());
                self.gpu.queue.write_buffer(&e.buf, *offset, data);
                written += data.len() as u64;
            }
            for (label, column) in records.columns() {
                let Some(e) = g.bufs.get(&label).filter(|_| column.is_pending()) else { continue };
                column.commit(&mut |offset, data| {
                    crate::gpu_profile::count_buffer_write(data.len());
                    self.gpu.queue.write_buffer(&e.buf, offset, data);
                    written += data.len() as u64;
                });
            }
        }
        if let Some(p) = self.prepared_frame.borrow_mut().as_mut() {
            p.key = new.key;
            if let Some(replaced) = std::mem::replace(&mut p.pinned, new.pinned.cloned()) {
                self.defer_drop(Box::new(replaced));
            }
            if let Some((chunks, batches)) = text {
                let old_text = std::mem::replace(&mut p.text, chunks);
                let old_batches = std::mem::replace(&mut p.text_batches, batches);
                self.defer_drop(Box::new((old_text, old_batches)));
            }
            if text_changed || masks_changed {
                if let Some(bundle) = p.bundle.take() {
                    self.defer_drop(Box::new(bundle));
                }
            }
        }
        let mut st = self.prepared_stats.get();
        st.patches = st.patches.wrapping_add(1);
        st.patched_items = st.patched_items.wrapping_add(items);
        st.patch_bytes = st.patch_bytes.wrapping_add(written);
        st.upload_bytes = st.upload_bytes.saturating_add(written);
        self.prepared_stats.set(st);
        Ok(())
    }

    /// Why the last live camera present could not use a prepared frame (diagnostics), or `""`.
    pub fn prepared_ineligible_reason(&self) -> &'static str {
        self.prepared_ineligible_reason.get()
    }

    /// Why the last prepared-frame patch fell back (diagnostics), or `""`.
    pub fn patch_fallback_reason(&self) -> &'static str {
        self.patch_fallback_reason.get()
    }

    fn replayable_bundle(
        &self,
        packet_plan: Option<PacketPlan<'_>>,
        target_compatible: bool,
        live_masks: bool,
        width: u32,
        height: u32,
        sample_count: u32,
        clear_is_opaque: bool,
        clock: f64,
        greek_sig: u64,
        captures_same: bool,
    ) -> Option<wgpu::RenderBundle> {
        let plan = packet_plan.filter(|_| self.retained && target_compatible && captures_same)?;
        let pf = self.prepared_frame.borrow();
        let p = pf.as_ref()?;
        (p.key == plan.key
            && p.live_masks == live_masks
            && p.width == width
            && p.height == height
            && p.sample_count == sample_count
            && p.clear_is_opaque == clear_is_opaque
            && p.pool_epoch == self.pool_epoch.get()
            && p.text_greek_sig == greek_sig
            && (p.simplified_blurs == 0 || p.motion_blur_px == self.motion_blur_min_px.get())
            && (!p.has_clock_fill || p.shader_time_s == clock))
            .then(|| p.bundle.clone())
            .flatten()
    }

    /// Keep `x` until [`Engine::drop_deferred`] (at most a few frames' worth: past that, the oldest goes now).
    fn defer_drop(&self, x: Box<dyn std::any::Any>) {
        let mut d = self.deferred_drops.borrow_mut();
        if d.len() >= 8 {
            d.remove(0);
        }
        d.push(x);
    }

    /// Regenerate the paint-surface mips under the tiles uploaded since the last render now (every render does this
    /// first; benches time it alone). Returns the CPU time spent encoding and submitting, in ms.
    pub fn flush_paint_surface_mips(&self) -> f64 {
        let before = crate::surface::stats().mip_regen_ms;
        if let Some(cache) = self.retained_patterns.borrow_mut().as_mut() {
            cache.flush_surface_mips(&self.gpu.device, &self.gpu.queue);
        }
        crate::surface::stats().mip_regen_ms - before
    }

    /// Patch the prepared frame in place for small plan changes (default on; off for an A/B or a test: every plan
    /// change rebuilds the packet, as before).
    pub fn set_prepared_patch(&self, on: bool) {
        self.prepared_patch.set(on);
    }

    /// Draw background-blur composites inside the frame's next pass (default on; off for an A/B or a test).
    pub fn set_backdrop_inline(&self, on: bool) {
        self.backdrop_inline.set(on);
    }

    /// Free what frames deferred (`defer_drop`): the host calls it when idle.
    pub fn drop_deferred(&self) {
        let items = std::mem::take(&mut *self.deferred_drops.borrow_mut());
        drop(items);
    }
}

/// The engine's render pipelines by (module, descriptor index, sample count, blend / variant key, color target
/// format). The format is in the key so a pipeline built for one color target is never bound to another: the frame
/// format (8-bit) and the pixel graph's `Rgba16Float` capture targets (motion-model.md §10.1) must never share one.
type PipelineMap =
    std::collections::HashMap<(String, usize, u32, &'static str, wgpu::TextureFormat), wgpu::RenderPipeline>;

/// The engine's pipelines as one render sees them (the shared map: a pipeline built later copies it), with the color
/// target format that render draws into.
struct PipeSnap {
    map: std::sync::Arc<PipelineMap>,
    format: wgpu::TextureFormat,
}

/// A pipeline from a snapshot of the engine's map, by the key `Engine::with_pipeline` takes.
fn snap_pipe<'m>(m: &'m PipeSnap, key: (&str, usize, u32, BlendMode)) -> &'m wgpu::RenderPipeline {
    &m.map[&(key.0.to_string(), key.1, key.2, key.3.manifest_key(), m.format)]
}

/// A shape-SDF pipeline from a snapshot of the engine's map, by the key `Engine::with_shape_pipeline` takes.
fn snap_shape_pipe<'m>(
    m: &'m PipeSnap,
    module_id: &str,
    desc_index: usize,
    sample_count: u32,
    clear_is_opaque: bool,
) -> &'m wgpu::RenderPipeline {
    &m.map[&(
        module_id.to_string(),
        desc_index,
        sample_count,
        shape_variant(desc_index, sample_count, clear_is_opaque),
        m.format,
    )]
}

/// The cache-key variant tag for a shape-SDF pipeline. Separates the
/// alpha-to-coverage opaque variant (`desc 0`, MSAA, opaque clear) from the
/// premultiplied-blend variant used in every other case, so both can coexist in
/// the shared pipeline cache for one (module, desc, sample_count). Since #106
/// desc 0 is built only in the A2C case, so desc 0 never maps to
/// `"shape-premul"` in practice (`desc_index` is in the key either way).
/// Splits per command buffer (see the segment loop in `render_plan_to_texture_impl`).
const SPLIT_FLUSH: usize = 64;

fn shape_variant(desc_index: usize, sample_count: u32, clear_is_opaque: bool) -> &'static str {
    if desc_index == 0 && sample_count > 1 && clear_is_opaque {
        "shape-a2c"
    } else {
        "shape-premul"
    }
}

/// The contiguous `[min, max]` storage-instance range of the opaque
/// SDF draws belonging to `segment`, or `None` when the segment has none.
/// Opaque SDF instances are pushed into their storage buffer in plan order and
/// the pass splits partition that order into gap-free runs, so a segment's
/// indices form one contiguous range — drawable as a single instanced
/// `draw(6, max-min+1, 0, min)`. The `debug_assert` pins that invariant.
fn segment_index_range(draws: &[(usize, u32)], segment: usize) -> Option<(u32, u32)> {
    let mut range: Option<(u32, u32)> = None;
    let mut count: u32 = 0;
    for &(s, idx) in draws {
        if s != segment {
            continue;
        }
        count += 1;
        range = Some(match range {
            None => (idx, idx),
            Some((lo, hi)) => (lo.min(idx), hi.max(idx)),
        });
    }
    debug_assert!(
        match range {
            Some((lo, hi)) => hi - lo + 1 == count,
            None => true,
        },
        "opaque SDF segment {segment} indices not contiguous: {range:?} count {count}"
    );
    range
}

fn to_wgpu_blend(b: &moda_shaders::BlendState) -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: blend_factor(&b.color.src_factor),
            dst_factor: blend_factor(&b.color.dst_factor),
            operation: blend_operation(&b.color.operation),
        },
        alpha: wgpu::BlendComponent {
            src_factor: blend_factor(&b.alpha.src_factor),
            dst_factor: blend_factor(&b.alpha.dst_factor),
            operation: blend_operation(&b.alpha.operation),
        },
    }
}

fn blend_factor(s: &str) -> wgpu::BlendFactor {
    match s {
        "zero" => wgpu::BlendFactor::Zero,
        "one" => wgpu::BlendFactor::One,
        "src" => wgpu::BlendFactor::Src,
        "one-minus-src" => wgpu::BlendFactor::OneMinusSrc,
        "src-alpha" => wgpu::BlendFactor::SrcAlpha,
        "one-minus-src-alpha" => wgpu::BlendFactor::OneMinusSrcAlpha,
        "dst" => wgpu::BlendFactor::Dst,
        "one-minus-dst" => wgpu::BlendFactor::OneMinusDst,
        "dst-alpha" => wgpu::BlendFactor::DstAlpha,
        "one-minus-dst-alpha" => wgpu::BlendFactor::OneMinusDstAlpha,
        other => panic!("unmapped blend factor '{other}'"),
    }
}

fn blend_operation(s: &str) -> wgpu::BlendOperation {
    match s {
        "add" => wgpu::BlendOperation::Add,
        "subtract" => wgpu::BlendOperation::Subtract,
        "reverse-subtract" => wgpu::BlendOperation::ReverseSubtract,
        "min" => wgpu::BlendOperation::Min,
        "max" => wgpu::BlendOperation::Max,
        other => panic!("unmapped blend operation '{other}'"),
    }
}

fn texture_format(s: &str) -> wgpu::TextureFormat {
    match s {
        // The offscreen/export engine is rgba8unorm (never -srgb); the
        // manifest records rgba8unorm for export pipelines already. Frame
        // COLOR targets go through `frame_color_target_format` instead,
        // Which folds onto the per-engine frame format.
        "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
        "bgra8unorm" => wgpu::TextureFormat::Rgba8Unorm,
        "r8unorm" => wgpu::TextureFormat::R8Unorm,
        "depth24plus" => wgpu::TextureFormat::Depth24Plus,
        other => panic!("unmapped texture format '{other}'"),
    }
}

/// Map a manifest COLOR-TARGET format string onto the engine's
/// immutable frame format. The manifest was extracted in a browser whose
/// preferred canvas format colored these entries (`bgra8unorm` on desktop,
/// `rgba8unorm` elsewhere) — both spellings name the SAME attachment, the
/// engine's single frame color target, so both resolve to `frame_format`
/// (`Rgba8Unorm` by default, the browser-preferred format on the live wasm
/// lane). Non-frame color formats (e.g. the mask pre-pass's `r8unorm`) keep
/// their literal meaning via [`texture_format`].
fn frame_color_target_format(s: &str, frame_format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match s {
        "rgba8unorm" | "bgra8unorm" => frame_format,
        other => texture_format(other),
    }
}

/// Map the browser's `navigator.gpu.getPreferredCanvasFormat()` string (threaded
/// from JS at `init_engine`) to the on-screen swapchain format. Only
/// the two guaranteed canvas-context 8-bit formats matter; anything else (`None`,
/// a future `rgba16float`, or garbage) falls back to `Rgba8Unorm`, which is always
/// a legal canvas format.
///
/// Deliberately SEPARATE from [`texture_format`], which coalesces
/// `"bgra8unorm" -> Rgba8Unorm` for its offscreen/export pipelines. The on-screen
/// SURFACE must honor `bgra8unorm` faithfully: configuring a canvas with a
/// NON-preferred format makes the browser insert a per-present rgba->bgra
/// conversion that Firefox mis-applies after a resize, swapping red<->blue.
/// Routing the surface format through `texture_format` would silently re-hardcode
/// rgba and reintroduce that bug.
pub fn present_format_from_preferred(preferred: Option<&str>) -> wgpu::TextureFormat {
    // Only bgra8unorm diverges from the default; rgba8unorm / None / unknown all
    // resolve to Rgba8Unorm.
    match preferred {
        Some("bgra8unorm") => wgpu::TextureFormat::Bgra8Unorm,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Is `format` a legal engine frame color format? The two guaranteed WebGPU
/// canvas-context 8-bit formats, plus `Rgba16Float` for the pixel graph's float
/// captures (motion-model.md §10.1; the same gamma-space blend math, stored with
/// more precision; never presented) — never `-srgb` (the
/// gamma-space blending contract, render-semantics §7). Pure, so the allowlist
/// is pinned in the native CI test lane; [`Engine::new_with_frame_format`] is
/// the enforcing seam.
fn is_supported_frame_format(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba16Float
    )
}

fn vertex_format(s: &str) -> wgpu::VertexFormat {
    match s {
        "float32" => wgpu::VertexFormat::Float32,
        "float32x2" => wgpu::VertexFormat::Float32x2,
        "float32x3" => wgpu::VertexFormat::Float32x3,
        "float32x4" => wgpu::VertexFormat::Float32x4,
        "uint32" => wgpu::VertexFormat::Uint32,
        other => panic!("unmapped vertex format '{other}'"),
    }
}

fn compare_function(s: &str) -> wgpu::CompareFunction {
    match s {
        "less" => wgpu::CompareFunction::Less,
        "less-equal" => wgpu::CompareFunction::LessEqual,
        "always" => wgpu::CompareFunction::Always,
        other => panic!("unmapped compare function '{other}'"),
    }
}

/// The retained-mode draw key for a resolved pattern texture (R2d): settle
/// state + decoded-pixel identity baked into the key make async settles
/// self-invalidating with no message (MAP §5.7). The heap lane keys on the
/// decoded-pixel `Arc` pointer; a browser-decoded external raster has no such
/// `Arc` (its bytes never touch the wasm heap), so it keys on the per-install
/// `upload_id` — stable across plan rebuilds, bumped when the source is
/// re-installed so the retained cache self-invalidates.
/// Heap video frames retain storage by node, state and dimensions; the pattern
/// cache validates the pixel allocation separately and writes changed frames.
fn retained_pattern_draw_key(texture: &paint::PatternTexture) -> String {
    if texture.image.external.is_none() && texture.key.starts_with("video:") {
        return format!("{}|{:?}|{}x{}", texture.key, texture.state, texture.image.width, texture.image.height);
    }
    match &texture.image.external {
        Some(ext) => format!("{}|{:?}|ext{}", texture.key, texture.state, ext.upload_id),
        None => format!(
            "{}|{:?}|{:p}",
            texture.key,
            texture.state,
            std::sync::Arc::as_ptr(&texture.image.rgba_premultiplied)
        ),
    }
}

/// A section's background as the container body it draws like: the same fills, strokes, box and opacity, with
/// no rotation, scale, corner radius or clip (sections have none). `children` is dropped: the z-order already
/// visits them (or skips them under Hide contents).
fn section_as_container_body(node: &moda_scene::types::RawNode) -> moda_scene::types::RawNode {
    let mut m = moda_scene::value::Map::new();
    for key in ["id", "x", "y", "width", "height", "opacity", "fills", "strokes", "_transformSourceId"] {
        if let Some(v) = node.json_field(key) {
            m.insert(key.into(), v.clone());
        }
    }
    m.insert("type".into(), Value::from("container"));
    m.insert("children".into(), Value::Array(Vec::new()));
    m.insert("clipContent".into(), Value::from(false));
    moda_scene::types::RawNode::new(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_recipes_keep_geometry_and_match_cold_nested_models() {
        use moda_scene::value::json;
        for (solid, paint) in [(true, json!({"type":"solid","color":"#e43d26"})),
            (false, json!({"type":"linear","start":{"x":0,"y":0},"end":{"x":100,"y":100},
                "stops":[{"offset":0,"color":"#ff0000"},{"offset":1,"color":"#0000ff"}]}))] {
            let mut scene = Scene::load(&json!({
                "nodes":[
                    {"id":"g","type":"group","rotation":23,"scaleX":1.25,"scaleY":0.8,"children":["p"]},
                    {"id":"p","type":"path","x":0.125,"y":-0.375,"closed":true,"opacity":0.6,
                     "points":[{"type":"move","x":0,"y":0},{"type":"line","x":41,"y":3},
                        {"type":"line","x":17,"y":37}],
                     "fills":[{"id":"fill","paint":paint}]}],
                "pages":[{"id":"page","width":128,"height":128,"nodeIds":["g"]}],"config":{}
            })).unwrap();
            let opts = RenderOptions::default();
            let (mut plan, warnings, aux) = build_coverage_plan_indexed_with_dl(&scene,0,&opts,None,None).unwrap();
            assert!(warnings.is_empty());
            let span = aux.spans.iter().find(|s| s.node_id == "p").unwrap();
            let mut recipe = span.placement.clone().expect("ordinary local path has a placement recipe");
            let geometry = match &plan.items[span.start] { DrawItem::Body {geometry,..} => geometry.clone(), _ => panic!() };
            // Moves, turns, scales within the band and (for solids) fades follow the recipe.
            for (id, field, value, follows) in [
                ("g","x",0.1,true), ("g","x",16_777_217.,true), ("g","x",-12.875,true), ("g","x",0.,true),
                ("g","rotation",-71.5,true), ("p","rotation",211.,true), ("p","scaleX",-1.1,true), ("p","scaleY",0.3,true),
                ("p","y",9.25,true), ("p","opacity",0.25,solid), ("p","opacity",0.99,solid), ("p","x",3.,true),
            ] {
                let mut node = scene.node(id).unwrap().clone(); node.set(field, json!(value));
                scene.apply_node_edits(vec![node], &[], &[], None).unwrap();
                let before = format!("{:?}",plan.items);
                let followed = translate_span(&scene, &mut recipe, &mut plan.items[span.start..span.start+span.len]);
                assert_eq!(followed, follows, "{id}.{field}={value}");
                if !followed {
                    assert_eq!(before, format!("{:?}",plan.items), "a declined placement changes nothing");
                    // Put the opacity back so later steps compare against a plan the recipe still describes.
                    let mut node = scene.node(id).unwrap().clone(); node.set(field, json!(0.6));
                    scene.apply_node_edits(vec![node], &[], &[], None).unwrap();
                    continue;
                }
                let (fresh, warnings) = build_coverage_plan(&scene,0,&opts).unwrap();
                assert!(warnings.is_empty());
                assert_eq!(format!("{:?}",plan.items),format!("{:?}",fresh.items), "{id}.{field}={value}");
                assert!(matches!(&plan.items[span.start], DrawItem::Body {geometry: kept,..} if std::sync::Arc::ptr_eq(kept,&geometry)));
            }
            // Another band, opaque, or hidden: the planner prepares the node again.
            for (id, field, value) in [("g","scaleX",2.), ("p","opacity",1.), ("p","opacity",0.)] {
                let restore = scene.node(id).unwrap().clone();
                let mut node = restore.clone(); node.set(field, json!(value));
                scene.apply_node_edits(vec![node], &[], &[], None).unwrap();
                let before = format!("{:?}",plan.items);
                assert!(!translate_span(&scene,&mut recipe,&mut plan.items[span.start..span.start+span.len]), "{id}.{field}={value}");
                assert_eq!(before,format!("{:?}",plan.items));
                scene.apply_node_edits(vec![restore], &[], &[], None).unwrap();
            }
        }
    }

    #[test]
    fn translation_recipe_rejects_unproven_content_and_empty_spans() {
        use moda_scene::value::json;
        let scene = Scene::load(&json!({"nodes":[{"id":"p","type":"path",
            "points":[{"type":"move","x":0,"y":0},{"type":"line","x":10,"y":10}]}],
            "pages":[{"id":"page","width":64,"height":64,"nodeIds":["p"]}],"config":{}})).unwrap();
        let node = scene.node("p").unwrap();
        let recipe = local_path_placement(&scene, node).unwrap();
        assert!(certify_placement(&scene, recipe.clone(), &[]).is_none());
        for (field, value) in [("effects", json!([{"type":"blur","radius":4}])),
            ("clipContent", json!(true)), ("unknownGeometryDependency", json!(1))] {
            let mut changed = node.clone();
            changed.set(field, value);
            assert!(local_path_placement(&scene, &changed).is_none(), "{field}");
        }
    }

    #[test]
    fn body_aux_preserves_integer_owners_and_uv_residency() {
        let geometry = std::sync::Arc::new(TessellatedGeometry {
            positions: vec![0., 0., 1., 0., 0., 1.], vertex_types: vec![0., 1., 1.], vertex_count: 3,
            bounds: [0., 0., 1., 1.], fill_dropped: None, stroke_dropped: None,
            stroke_uv: vec![0., 0., 0.25, 0.5, 1., 1.],
        });
        for owner in [0, 17, 0x8000_0000, 0x7fc0_0001, u32::MAX] {
            let mut aux = Vec::new();
            append_body_aux(&mut aux, &geometry, Some(owner));
            assert!(aux.chunks_exact(2).all(|word| word[0].to_bits() == owner && word[1].to_bits() == 0));
        }
        let mut uv = Vec::new();
        append_body_aux(&mut uv, &geometry, None);
        assert_eq!(uv, geometry.stroke_uv);
        let mut residency = PhaseResidency { total: 9, ..Default::default() };
        for (owner, offset) in [(None, 0), (Some(0), 3), (Some(1), 6)] {
            residency.at.insert((std::sync::Arc::as_ptr(&geometry) as usize, owner),
                (offset, 3, std::sync::Arc::downgrade(&geometry)));
        }
        assert_eq!(residency.offset_of(&geometry, None), Some(0));
        assert_eq!(residency.offset_of(&geometry, Some(0)), Some(3));
        assert_eq!(residency.offset_of(&geometry, Some(1)), Some(6));
        assert_eq!(residency.offset_of(&geometry, Some(2)), None);
    }

    #[test]
    fn a_column_uploads_once_from_its_first_changed_record_to_its_last() {
        let mut column = Column::of((0..8u32).map(|i| [i, 0]).collect::<Vec<[u32; 2]>>());
        // A record the buffer already holds asks for nothing.
        assert_eq!(column.put(3, &[3, 0]), Ok(false));
        assert!(!column.is_pending());
        // Asked out of order, with unchanged records between them.
        assert_eq!(column.put(5, &[5, 9]), Ok(true));
        assert_eq!(column.put(2, &[2, 9]), Ok(true));
        assert_eq!(column.put(1, &[1, 9]), Ok(true));
        assert_eq!(column.put(8, &[8, 9]), Err("record"));
        // Nothing is taken before the commit.
        assert_eq!(column.records[2], [2, 0]);
        let mut uploads: Vec<(u64, Vec<u8>)> = Vec::new();
        column.commit(&mut |offset, data| uploads.push((offset, data.to_vec())));
        let bytes = |records: &[[u32; 2]]| bytemuck::cast_slice::<[u32; 2], u8>(records).to_vec();
        assert_eq!(uploads, vec![(8, bytes(&[[1, 9], [2, 9], [3, 0], [4, 0], [5, 9]]))]);
        assert!(!column.is_pending());
        assert_eq!(column.records, vec![[0, 0], [1, 9], [2, 9], [3, 0], [4, 0], [5, 9], [6, 0], [7, 0]]);
        assert_eq!(column.bytes(), 64);
        // A discarded record is never taken, and a column with nothing pending uploads nothing.
        assert_eq!(column.put(0, &[7, 7]), Ok(true));
        column.discard();
        column.commit(&mut |_, _| panic!("nothing is pending"));
        assert_eq!(column.records[0], [0, 0]);
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn solid_owners_match_unmerged_through_residency_and_patches() {
        use std::sync::Arc;
        let nodes: Vec<_> = (0..8).map(|i| moda_scene::value::json!({
            "id": format!("p{i}"), "type": "path", "closed": true,
            "x": 8 + i * 9, "y": 8 + (i % 3) * 9, "width": 40, "height": 35, "rotation": i * 7,
            "points": [{"type":"move","x":0,"y":0},{"type":"line","x":40,"y":0},
                       {"type":"line","x":30,"y":35},{"type":"line","x":0,"y":20}],
            "fills": [{"id":"fill","paint":{"type":"solid","color": if i % 2 == 0 {"#e73856"} else {"#35cace"}},
                "opacity": if i < 4 {1.0} else {0.6}}]
        })).collect();
        let ids: Vec<_> = (0..8).map(|i| format!("p{i}")).collect();
        let scene = moda_scene::Scene::load(&moda_scene::value::json!({"nodes":nodes,
            "pages":[{"id":"page","width":128,"height":96,"nodeIds":ids}]})).unwrap();
        let (mut base, _) = build_coverage_plan(&scene, 0, &present_test_options(4)).unwrap();
        let shared = base.items.iter().find_map(|item| match item {
            DrawItem::Body { geometry, .. } => Some(geometry.clone()), _ => None,
        }).expect("tessellated paths");
        for item in &mut base.items {
            if let DrawItem::Body { geometry, .. } = item { *geometry = shared.clone(); }
        }
        let mut plans = vec![Arc::new(base.clone()), Arc::new(base.clone())];
        for item in &mut base.items {
            if let DrawItem::Body { instance, .. } = item {
                instance.model_matrix[12] += 3.;
                instance.fill_color[0] *= 0.5;
            }
        }
        plans.push(Arc::new(base.clone()));
        let replacement = Arc::new({ let mut g = (*shared).clone(); g.positions[0] += 2.; g });
        if let Some(DrawItem::Body { geometry, .. }) = base.items.iter_mut().find(|i| matches!(i, DrawItem::Body { .. })) {
            *geometry = replacement;
        }
        plans.push(Arc::new(base.clone()));
        let first = base.items.iter().position(|i| matches!(i, DrawItem::Body { .. })).unwrap();
        base.items.insert(first, base.items[first].clone());
        plans.push(Arc::new(base.clone()));
        base.items.remove(first + 1);
        plans.push(Arc::new(base));
        let mut engine = Engine::new(gpu("solid_owners_match_unmerged_through_residency_and_patches")).unwrap();
        engine.set_solid_run_merging(false);
        let expected: Vec<_> = plans.iter().map(|p| engine.render_plan(p).unwrap()).collect();
        // The comparison means something only if the edits show: a moved and recoloured frame, a reshaped body.
        assert_ne!(expected[1], expected[2], "the paint and placement edit is visible");
        assert_ne!(expected[2], expected[3], "the geometry edit is visible");
        engine.set_solid_run_merging(true);
        engine.set_retained(true);
        for offscreen in [false, true] {
            let before = engine.prepared_frame_stats().patched_items;
            for chunks in [None, Some(u64::from(shared.vertex_count) * 8)] {
                engine.set_vertex_chunk_bytes(chunks);
                for (index, plan) in plans.iter().enumerate() {
                    let texture = if offscreen {
                        engine.render_retained_plan_to_texture(plan).unwrap()
                    } else {
                        let texture = test_texture(&engine, "solid owner test", plan.width_px, plan.height_px,
                            wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
                        let view = texture.create_view(&Default::default());
                        assert!(engine.render_plan_to_view(plan, None, &view, None, Some(plan), true).unwrap().is_none());
                        texture
                    };
                    let actual = crate::readback::read_texture_rgba(&engine.gpu.device, &engine.gpu.queue,
                        &texture, plan.width_px, plan.height_px).unwrap();
                    assert_eq!(actual, expected[index], "frame {index}, chunks {chunks:?}, offscreen {offscreen}");
                }
            }
            assert!(engine.prepared_frame_stats().patched_items > before, "offscreen {offscreen} must patch changed items");
        }
        assert!(engine.prepared_frame_stats().merged_bodies > 0);
    }

    /// Record slots belong to nodes and vertex ranges to geometries. Sequences that change which body owns
    /// what must match fresh unmerged draws: a body taking another's geometry (unique to shared), bodies
    /// trading places before one moves, and a freed slot reused by a new body.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn solid_slots_follow_nodes_through_sharing_reorders_and_reuse() {
        use std::sync::Arc;
        let nodes: Vec<_> = (0..6).map(|i| moda_scene::value::json!({
            "id": format!("p{i}"), "type": "path", "closed": true,
            "x": 6 + i * 18, "y": 8 + (i % 3) * 14, "width": 40, "height": 35, "rotation": i * 5,
            "points": [{"type":"move","x":0,"y":0},{"type":"line","x":40,"y":0},
                       {"type":"line","x":30,"y":35},{"type":"line","x":0,"y":20}],
            "fills": [{"id":"fill","paint":{"type":"solid","color": (["#e73856", "#35cace", "#f2c530"][i % 3])},
                "opacity": if i < 2 {1.0} else {0.6}}]
        })).collect();
        let ids: Vec<_> = (0..6).map(|i| format!("p{i}")).collect();
        let scene = moda_scene::Scene::load(&moda_scene::value::json!({"nodes":nodes,
            "pages":[{"id":"page","width":160,"height":96,"nodeIds":ids}]})).unwrap();
        let (mut original, _) = build_coverage_plan(&scene, 0, &present_test_options(4)).unwrap();
        // Every body its own geometry (identical paths would share one tessellation), same vertex counts.
        let bodies: Vec<usize> = original.items.iter().enumerate()
            .filter_map(|(at, item)| matches!(item, DrawItem::Body { .. }).then_some(at)).collect();
        assert_eq!(bodies.len(), 6);
        for (n, &at) in bodies.iter().enumerate() {
            if let DrawItem::Body { geometry, .. } = &mut original.items[at] {
                *geometry = Arc::new({ let mut g = (**geometry).clone(); g.positions[0] += n as f32; g });
            }
        }
        let geometry_of = |plan: &CoveragePlan, at: usize| match &plan.items[at] {
            DrawItem::Body { geometry, .. } => geometry.clone(), _ => unreachable!(),
        };
        // The first body takes the second's geometry: both now draw one shared range.
        let mut share = original.clone();
        let taken = geometry_of(&original, bodies[1]);
        if let DrawItem::Body { geometry, .. } = &mut share.items[bodies[0]] { *geometry = taken; }
        // Two bodies that share a geometry trade places, each position keeping its depth and phase...
        let mut twins = share.clone();
        let (a, b) = (bodies[0], bodies[1]);
        let (mut first, mut second) = (twins.items[b].clone(), twins.items[a].clone());
        for (item, at) in [(&mut first, a), (&mut second, b)] {
            if let (DrawItem::Body { z, .. }, DrawItem::Body { z: kept, .. }) = (item, &twins.items[at]) { *z = *kept; }
        }
        let mut traded = twins.clone();
        traded.items[a] = first;
        traded.items[b] = second;
        // ...and then the one now first moves.
        let mut moved = traded.clone();
        if let DrawItem::Body { instance, .. } = &mut moved.items[a] { instance.model_matrix[12] += 30.; }
        // A translucent body leaves; later a new one arrives after the last, in the freed slot.
        let mut deleted = original.clone();
        deleted.items.remove(bodies[4]);
        let mut reused = deleted.clone();
        let last = bodies[5] - 1;
        let mut arrival = reused.items[last].clone();
        if let DrawItem::Body { node_id, geometry, instance, .. } = &mut arrival {
            *node_id = "arrival".into();
            *geometry = Arc::new({ let mut g = (**geometry).clone(); g.positions[1] += 1.; g });
            instance.model_matrix[12] -= 40.;
            instance.model_matrix[13] += 20.;
        }
        reused.items.insert(last + 1, arrival);
        // One draw id on two bodies that are not neighbours (as transform copies and paint layers emit), with
        // different records; then the later one moves.
        let mut recurring = original.clone();
        let earlier = match &original.items[bodies[2]] { DrawItem::Body { node_id, .. } => node_id.clone(), _ => unreachable!() };
        if let DrawItem::Body { node_id, .. } = &mut recurring.items[bodies[4]] { *node_id = earlier; }
        let mut recurring_moved = recurring.clone();
        if let DrawItem::Body { instance, .. } = &mut recurring_moved.items[bodies[4]] { instance.model_matrix[13] += 25.; }
        let plans: Vec<(&str, Arc<CoveragePlan>)> = [
            ("recurring id, same picture", &recurring), ("recurring moved", &recurring_moved),
            ("original", &original), ("share", &share), ("share again", &share), ("original", &original),
            ("twins", &twins), ("traded", &traded), ("moved", &moved), ("moved again", &moved),
            ("original", &original), ("deleted", &deleted), ("reused", &reused), ("reused again", &reused),
            ("original", &original),
        ].into_iter().map(|(label, plan)| (label, Arc::new(plan.clone()))).collect();
        let fresh = Engine::new(gpu("solid_slots_follow_nodes_through_sharing_reorders_and_reuse")).unwrap();
        fresh.set_solid_run_merging(false);
        let expected: Vec<_> = plans.iter().map(|(_, plan)| fresh.render_plan(plan).unwrap()).collect();
        // The comparison means something only if every step shows.
        for (index, (label, _)) in plans.iter().enumerate().skip(1) {
            if index == 2 { assert_eq!(expected[index], expected[0], "back to the original picture"); continue; }
            // Trading places changes which body each position holds, not the picture.
            assert_eq!(expected[index] == expected[index - 1], label.ends_with("again") || label.ends_with("same picture") || *label == "traded", "{label} shows");
        }
        assert!(matches!(&original.items[bodies[1]], DrawItem::Body { phase: Phase::Opaque, .. })
            && matches!(&original.items[bodies[2]], DrawItem::Body { phase: Phase::Transparent, .. }));
        for offscreen in [false, true] {
            for chunks in [None, Some(u64::from(geometry_of(&original, bodies[0]).vertex_count) * 8)] {
                let mut engine = Engine::new(gpu("solid_slots_follow_nodes_through_sharing_reorders_and_reuse")).unwrap();
                engine.set_retained(true);
                engine.set_offscreen_vertex_residency(true);
                engine.set_vertex_chunk_bytes(chunks);
                for (index, (label, plan)) in plans.iter().enumerate() {
                    let merged = engine.prepared_frame_stats().merged_bodies;
                    let texture = if offscreen {
                        engine.render_retained_plan_to_texture(plan).unwrap()
                    } else {
                        let texture = test_texture(&engine, "solid slot test", plan.width_px, plan.height_px,
                            wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
                        let view = texture.create_view(&Default::default());
                        assert!(engine.render_plan_to_view(plan, None, &view, None, Some(plan), true).unwrap().is_none());
                        texture
                    };
                    let actual = crate::readback::read_texture_rgba(&engine.gpu.device, &engine.gpu.queue,
                        &texture, plan.width_px, plan.height_px).unwrap();
                    assert_eq!(actual, expected[index], "frame {index} ({label}), chunks {chunks:?}, offscreen {offscreen}");
                    // The arrival takes the freed slot, lower than its neighbour's; its range, appended after
                    // that neighbour's, still joins it: at least the opaque pair, the first two translucent
                    // bodies, and the last one with the arrival.
                    if *label == "reused" && chunks.is_none() {
                        assert!(engine.prepared_frame_stats().merged_bodies >= merged + 3,
                            "a body in a recycled slot merges with its neighbour, offscreen {offscreen}");
                    }
                }
            }
        }
    }


    #[test]
    #[ignore = "requires a real GPU; retained insertion/repeated-geometry measurement"]
    fn retained_solid_insertion_measurement() {
        use std::sync::Arc;
        let points: Vec<_> = (0..128).map(|i| {
            let angle = i as f64 * std::f64::consts::TAU / 128.;
            moda_scene::value::json!({"type": if i == 0 {"move"} else {"line"},
                "x": 4. + 4. * angle.cos(), "y": 4. + 4. * angle.sin()})
        }).collect();
        let scene = Scene::load(&moda_scene::value::json!({"nodes":[{"id":"p","type":"path",
            "closed":true,"points":points,"fills":[{"paint":{"type":"solid","color":"#35cace"}}]}],
            "pages":[{"id":"page","width":512,"height":512,"nodeIds":["p"]}]})).unwrap();
        let (template, _) = build_coverage_plan(&scene, 0, &present_test_options(4)).unwrap();
        let body = template.items.iter().find(|i| matches!(i, DrawItem::Body {..})).unwrap().clone();
        for repeated in [false, true] {
            let mut plan = template.clone();
            plan.items.clear();
            for i in 0..4096 {
                let mut item = body.clone();
                if let DrawItem::Body { geometry, instance, .. } = &mut item {
                    if !repeated { *geometry = Arc::new((**geometry).clone()); }
                    instance.model_matrix[12] = (i % 64) as f32 * 8.;
                    instance.model_matrix[13] = (i / 64) as f32 * 8.;
                }
                plan.items.push(item);
            }
            let mut engine = Engine::new(gpu("retained_solid_insertion_measurement")).unwrap();
            engine.set_retained(true);
            let target = test_texture(&engine, "residency measurement",512,512,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
            let view = target.create_view(&Default::default());
            let draw = |plan: &Arc<CoveragePlan>| {
                engine.render_plan_to_view(plan,None,&view,None,Some(plan),true).unwrap();
                crate::readback::poll_until_complete_with_budget(&engine.gpu.device,
                    crate::readback::READBACK_POLL_TIMEOUT_RETRIES_FIRST).unwrap();
            };
            let plan = Arc::new(plan);
            let cold = std::time::Instant::now(); draw(&plan); let cold = cold.elapsed().as_secs_f64();
            draw(&plan);
            let before = engine.prepared_frame_stats();
            let start = std::time::Instant::now();
            for _ in 0..20 { draw(&plan); }
            let held = start.elapsed().as_secs_f64();
            let mut inserted = (*plan).clone();
            inserted.items.insert(0,body.clone());
            let inserted = Arc::new(inserted);
            let start = std::time::Instant::now();
            for _ in 0..10 { draw(&inserted); draw(&plan); }
            let edits = start.elapsed().as_secs_f64();
            // The inserted frame itself, not only the plan drawn after it.
            draw(&inserted);
            let actual = crate::readback::read_texture_rgba(&engine.gpu.device,&engine.gpu.queue,&target,512,512).unwrap();
            assert_eq!(actual, Engine::new(engine.gpu.clone()).unwrap().render_plan(&inserted).unwrap());
            // Leaving and re-entering residency repacks: neither mode may reuse the other's vertex owners.
            engine.set_vertex_residency(false);
            draw(&plan);
            let packed = crate::readback::read_texture_rgba(&engine.gpu.device,&engine.gpu.queue,&target,512,512).unwrap();
            engine.set_vertex_residency(true);
            draw(&plan);
            let after = engine.prepared_frame_stats();
            let resident: u32 = engine.vertex_residency.borrow().iter().flatten().map(|r| r.total).sum();
            let actual = crate::readback::read_texture_rgba(&engine.gpu.device,&engine.gpu.queue,&target,512,512).unwrap();
            let expected = Engine::new(engine.gpu.clone()).unwrap().render_plan(&plan).unwrap();
            assert_eq!(actual,expected);
            assert_eq!(packed,expected);
            if let Ok(dir) = std::env::var("VORK_RESIDENCY_EVIDENCE") {
                std::fs::write(format!("{dir}/repeated-{repeated}.rgba"), &actual).unwrap();
            }
            eprintln!("RESIDENCY repeated={repeated} cold_s={cold:.6} held_20_s={held:.6} insertion_20_s={edits:.6} upload_bytes={} resident_vertices={resident} appends={} repacks={}",
                after.upload_bytes-before.upload_bytes, after.resident_appends-before.resident_appends,
                after.resident_repacks-before.resident_repacks);
        }
    }

    #[test]
    fn prepass_coverage_keeps_fractional_edges_and_handles_mirrored_quads() {
        assert!(quad_covers_target((0.0, 0.0), (640.0, 0.0), (0.0, 480.0), 640.0, 480.0));
        assert!(quad_covers_target((640.0, 0.0), (0.0, 0.0), (640.0, 480.0), 640.0, 480.0));
        assert!(quad_covers_target((320.0, -400.0), (1000.0, 240.0), (-360.0, 240.0), 640.0, 480.0));
        assert!(!quad_covers_target((4.25, 4.25), (636.25, 4.25), (4.25, 476.25), 640.0, 480.0));
        assert!(quad_covers_target((-0.000001, -0.000001), (640.000001, -0.000001), (-0.000001, 480.000001), 640.0, 480.0));
        assert!(!quad_covers_target((0.000001, 0.000001), (640.000001, 0.000001), (0.000001, 480.000001), 640.0, 480.0));
        assert!(!quad_covers_target((0.5, 0.5), (639.5, 0.5), (0.5, 479.5), 640.0, 480.0));
        assert!(!quad_covers_target((0.0, 0.0), (0.0, 0.0), (0.0, 480.0), 640.0, 480.0));
    }

    #[test]
    fn retained_video_storage_key_ignores_pixels_but_not_dimensions_or_state() {
        use crate::paint::{DecodedImage, PatternTexture, PatternTextureState};
        use std::sync::Arc;

        let mut texture = PatternTexture {
            state: PatternTextureState::Loaded,
            key: "video:clip".into(),
            image: DecodedImage {
                width: 4,
                height: 4,
                rgba_premultiplied: Arc::new(vec![255; 64]),
                straight: false,
                external: None,
            },
            scale_factor: (1.0, 1.0),
        };
        let key = retained_pattern_draw_key(&texture);
        texture.image.rgba_premultiplied = Arc::new(vec![0; 64]);
        assert_eq!(key, retained_pattern_draw_key(&texture));
        texture.image.width = 2;
        assert_ne!(key, retained_pattern_draw_key(&texture));
        texture.image.width = 4;
        texture.state = PatternTextureState::Loading;
        assert_ne!(key, retained_pattern_draw_key(&texture));
    }

    #[test]
    fn retained_pattern_draw_key_external_keys_on_upload_id() {
        use crate::paint::{DecodedImage, ExternalTextureRef, PatternTexture, PatternTextureState};
        use std::sync::Arc;

        let make = |external: Option<ExternalTextureRef>, rgba: Arc<Vec<u8>>| PatternTexture {
            state: PatternTextureState::Loaded,
            key: "src".into(),
            image: DecodedImage {
                width: 4,
                height: 4,
                rgba_premultiplied: rgba,
                straight: false,
                external,
            },
            scale_factor: (1.0, 1.0),
        };

        // External raster: keyed on `upload_id` (decoded pixels never hit the
        // heap, so there is no `Arc` to key on). A re-install bumps `upload_id`,
        // which MUST change the key so the retained cache self-invalidates.
        let ext1 = retained_pattern_draw_key(&make(
            Some(ExternalTextureRef {
                key: "src".into(),
                upload_id: 1,
            }),
            Arc::new(Vec::new()),
        ));
        let ext2 = retained_pattern_draw_key(&make(
            Some(ExternalTextureRef {
                key: "src".into(),
                upload_id: 2,
            }),
            Arc::new(Vec::new()),
        ));
        assert_eq!(ext1, "src|Loaded|ext1");
        assert_ne!(ext1, ext2);

        // Heap raster: keyed on the decoded-pixel `Arc` pointer, and visibly
        // distinct from the external shape (no `ext` marker).
        let heap = retained_pattern_draw_key(&make(None, Arc::new(vec![0u8; 16])));
        assert!(heap.starts_with("src|Loaded|0x"), "heap key: {heap}");
        assert!(!heap.contains("ext"));
    }

    // -----------------------------------------------------------------------
    // `clipTextToBounds` through the richtext arm
    // -----------------------------------------------------------------------

    /// One flagged/unflagged richtext node on an 800x600 page. Fonts come from
    /// bundled fixture dir so the arm actually emits a `DrawItem::Text`.
    fn clip_bounds_scene(flagged: bool) -> moda_scene::Scene {
        let flag = if flagged {
            r#","clipTextToBounds":true"#
        } else {
            ""
        };
        let json = format!(
            r##"{{"nodes":[
                    {{"id":"t","type":"richtext","x":40,"y":30,"width":120,"height":60,
                      "rotation":0,"scale":1,
                      "htmlContent":"<p style=\"font-size: 28px;\">overflowing cell text</p>"{flag}}}
                 ],
                 "pages":[{{"id":"p1","name":"P1","width":800,"height":600,"nodeIds":["t"]}}],
                 "config":{{}},"variables":[],"canvasRootNodeIds":[],
                 "layoutMode":"vertical","category":null,"animations":null}}"##
        );
        let value: moda_scene::value::Value = moda_scene::value::from_str(&json).unwrap();
        moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap()
    }

    fn text_batch_clip(scene: &moda_scene::Scene) -> [f32; 4] {
        let opts = RenderOptions::default();
        let (plan, _w) = build_coverage_plan(scene, 0, &opts).expect("plan builds");
        plan.items
            .iter()
            .find_map(|d| match d {
                DrawItem::Text(b) => Some(b.page_clip_bounds),
                _ => None,
            })
            .expect("a text draw item (fonts must be available)")
    }

    /// The richtext arm narrows the effective clip to the node's own
    /// canvas box ONLY under a truthy `clipTextToBounds`. The unflagged case is
    /// the byte-exactness contract at the plan level — it must still carry the
    /// plain page clip, exactly as before the field existed.
    #[test]
    fn richtext_clip_text_to_bounds_narrows_only_when_flagged() {
        assert_eq!(
            text_batch_clip(&clip_bounds_scene(false)),
            [0.0, 0.0, 800.0, 600.0],
            "an unflagged richtext node keeps the plain page clip — text is \
             deliberately NOT clipped to its own rect"
        );
        assert_eq!(
            text_batch_clip(&clip_bounds_scene(true)),
            [40.0, 30.0, 120.0, 60.0],
            "a flagged richtext node clips to its own canvas box so a table \
             cell's text cannot bleed into its neighbours"
        );
    }

    /// On a text-cache HIT the batch is served from the cache and only
    /// its volatile fields are re-patched (`patch_cached_batch`) — so the clip
    /// MUST come from the caller's (narrowed) value, not from whatever clip was
    /// in force when the layout was first shaped. Without this, a flagged cell
    /// that moved or resized would keep a stale clip for as long as its shaped
    /// text stayed cached, and the incremental page/camera lanes both serve
    /// through exactly this path.
    #[test]
    fn richtext_cache_hit_repatches_the_narrowed_clip() {
        let scene = clip_bounds_scene(true);
        let node = scene.effective_node("t").expect("richtext node");
        let hashes = crate::residency::node_hashes_for_scene(&scene);
        let mut caches = crate::residency::PlanCaches::default();

        let clip_of = |caches: &mut crate::residency::PlanCaches, clip: [f32; 4]| -> [f32; 4] {
            let out = crate::text_draw::build_richtext_item(
                &scene,
                node,
                0.0,
                clip,
                false,
                0.0,
                1.0,
                Some((caches, &hashes)),
            )
            .expect("builds");
            let crate::text_draw::TextBuildOutcome::Item(item) = out else {
                panic!("expected a text item (fonts must be available)");
            };
            let DrawItem::Text(b) = *item else {
                panic!("expected Text");
            };
            b.page_clip_bounds
        };

        // Cold: shapes and caches the batch under the first clip.
        let first = [40.0, 30.0, 120.0, 60.0];
        assert_eq!(clip_of(&mut caches, first), first);
        assert_eq!(caches.text_hits, 0, "the first build must be a cache MISS");

        // Warm: same node/key, narrower clip (the cell moved/shrank). The served
        // batch must carry the NEW clip.
        let second = [44.0, 34.0, 100.0, 40.0];
        assert_eq!(clip_of(&mut caches, second), second);
        assert_eq!(
            caches.text_hits, 1,
            "the second build must be a cache HIT — otherwise this test proves \
             nothing about the re-patch path"
        );
    }

    #[test]
    fn cached_subtree_shadows_preserve_descendant_masks() {
        // A masked paint rectangle also represents gradient/image text and
        // painted strokes. Its shadow must use the same coverage as its body,
        // both before and after silhouettes enter the tessellation cache.
        for mask_type in ["alpha", "luminance"] {
            let scene = Scene::load(&moda_scene::value::json!({
                "nodes": [
                    {"id":"shadow", "type":"group", "children":["clipped"],
                     "effects":[{"id":"shadow", "type":"drop-shadow",
                         "color":"#000000", "blur":8, "offset":{"x":10, "y":10}}]},
                    {"id":"clipped", "type":"group", "children":["mask", "paint"],
                     "clipMaskNodeId":"mask"},
                    {"id":"mask", "type":"ellipse", "x":20, "y":20,
                     "width":40, "height":40, "maskType":mask_type,
                     "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ffffff"}}]},
                    {"id":"paint", "type":"path", "x":0, "y":0, "closed":true,
                     "points":[{"type":"move", "x":0, "y":0}, {"type":"line", "x":80, "y":0},
                         {"type":"line", "x":80, "y":80}, {"type":"line", "x":0, "y":80}],
                     "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ff0000"}}]}
                ],
                "pages":[{"id":"page", "width":128, "height":128, "nodeIds":["shadow"]}],
                "config":{}, "canvasRootNodeIds":[], "layoutMode":"vertical"
            })).unwrap();
            let hashes = crate::residency::node_hashes_for_scene(&scene);
            let mut caches = crate::residency::PlanCaches::default();
            for cached in [false, true, true] {
                let (plan, warnings) = build_coverage_plan_cached(
                    &scene, 0, &RenderOptions::default(),
                    cached.then_some((&mut caches, &hashes)),
                ).unwrap();
                assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
                let group = plan.items.iter().find_map(|item| match item {
                    DrawItem::LayerBlurComposite { group, .. } if group.filters.shadow.is_some() => Some(group),
                    _ => None,
                }).expect("subtree shadow must be captured");
                let masks: Vec<_> = group.items.iter().filter_map(|item| match item {
                    DrawItem::Body { node_id, mask, .. } if silhouette_owner_id(node_id) == "paint" =>
                        Some(mask.as_ref().expect("paint silhouette must retain its coverage mask")),
                    _ => None,
                }).collect();
                assert!(!masks.is_empty(), "paint must contribute to the shadow");
                for mask in masks {
                    assert_eq!(group.masks[mask.mask_id as usize].owner_node_id, "clipped");
                }
            }
        }
    }

    #[test]
    fn nested_subtree_shadows_preserve_local_mask_tables_and_ancestors() {
        for mask_type in ["alpha", "luminance"] {
            for depth in 1..=3 {
                let mut nodes = vec![
                    moda_scene::value::json!({"id":"shadow", "type":"group", "children":["outer"],
                        "effects":[{"id":"shadow", "type":"drop-shadow", "color":"#000000",
                            "blur":0, "offset":{"x":100, "y":0}}]}),
                    moda_scene::value::json!({"id":"outer", "type":"group", "children":["outer-mask", "a0", "b0"],
                        "clipMaskNodeId":"outer-mask"}),
                    moda_scene::value::json!({"id":"outer-mask", "type":"rectangle", "x":0, "y":0,
                        "width":70, "height":80, "maskType":"alpha",
                        "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ffffff"}}]}),
                ];
                for prefix in ["a", "b"] {
                    for level in 0..depth {
                        let child = if level + 1 == depth { format!("{prefix}-clip") }
                            else { format!("{prefix}{}", level + 1) };
                        nodes.push(moda_scene::value::json!({"id":format!("{prefix}{level}"), "type":"group",
                            "children":[child], "effects":[{"id":"blur", "type":"layer-blur", "radius":2}]}));
                    }
                    nodes.extend([
                        moda_scene::value::json!({"id":format!("{prefix}-clip"), "type":"group",
                            "children":[format!("{prefix}-mask"), format!("{prefix}-paint")],
                            "clipMaskNodeId":format!("{prefix}-mask")}),
                        moda_scene::value::json!({"id":format!("{prefix}-mask"), "type":"ellipse",
                            "x":20, "y":20, "width":40, "height":40, "maskType":mask_type,
                            "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ffffff"}}]}),
                        moda_scene::value::json!({"id":format!("{prefix}-paint"), "type":"rectangle",
                            "x":0, "y":0, "width":80, "height":80,
                            "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ff0000"}}]}),
                    ]);
                }
                let scene = Scene::load(&moda_scene::value::json!({"nodes":nodes,
                    "pages":[{"id":"page", "width":256, "height":128, "nodeIds":["shadow"]}],
                    "config":{}, "canvasRootNodeIds":[], "layoutMode":"vertical"})).unwrap();
                let hashes = crate::residency::node_hashes_for_scene(&scene);
                let mut caches = crate::residency::PlanCaches::default();
                for cached in [false, true, true] {
                    let (plan, warnings) = build_coverage_plan_cached(&scene, 0, &RenderOptions::default(),
                        cached.then_some((&mut caches, &hashes))).unwrap();
                    assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
                    let group = plan.items.iter().find_map(|item| match item {
                        DrawItem::LayerBlurComposite { group, .. } if group.filters.shadow.is_some() => Some(group),
                        _ => None,
                    }).expect("parent shadow");
                    for prefix in ["a", "b"] {
                        let body = group.items.iter().find_map(|item| match item {
                            DrawItem::Body { node_id, mask, phase, .. }
                                if silhouette_owner_id(node_id) == format!("{prefix}-paint") => Some((mask, phase)),
                            _ => None,
                        }).expect("paint silhouette");
                        let mask = body.0.as_ref().expect("nested silhouette must retain its mask");
                        assert_eq!(*body.1, Phase::Transparent);
                        let local = &group.masks[mask.mask_id as usize];
                        assert_eq!(local.owner_node_id, format!("{prefix}-clip"));
                        assert_eq!(group.masks[local.ancestor.expect("restore ancestor clip")].owner_node_id, "outer");
                    }
                }
            }
        }
    }

    #[test]
    fn masked_text_outside_and_crossing_clip_bounds() {
        // A scrolling feed keeps text nodes alive while their glyphs cross
        // the clip boundary. Empty intersections are valid; visible glyphs
        // must still use the exact mask.
        for y in [-100.0, 19.75, 30.0, 200.0] {
            let scene = Scene::load(&moda_scene::value::json!({"nodes":[
                {"id":"feed", "type":"group", "children":["mask", "scroll"], "clipMaskNodeId":"mask"},
                {"id":"mask", "type":"ellipse", "x":0, "y":40, "width":120, "height":80},
                {"id":"scroll", "type":"group", "children":["text"], "y":y},
                {"id":"text", "type":"richtext", "width":100, "height":40,
                    "htmlContent":"<p style=\"font-size: 20px;\">Stargazer</p>"}],
                "pages":[{"id":"page", "width":160, "height":160, "nodeIds":["feed"]}],
                "config":{}, "canvasRootNodeIds":[], "layoutMode":"vertical"})).unwrap();
            let (plan, warnings) = build_coverage_plan_cached(&scene, 0, &RenderOptions::default(), None).unwrap();
            assert!(warnings.skipped.is_empty(), "y={y}: {:?}", warnings.skipped);
            assert!(!plan.items.iter().any(|item| matches!(item, DrawItem::Text(_))),
                "masked text must be culled or captured, never drawn without its mask");
            let captured = plan.items.iter().any(|item| matches!(item,
                DrawItem::LayerBlurComposite { group, mask: Some(_), .. }
                if group.items.iter().any(|item| matches!(item, DrawItem::Text(_)))));
            assert_eq!(captured, y > -100.0 && y < 200.0,
                "visible glyphs need exact mask coverage; outside glyphs need no capture (y={y})");
        }
    }

    #[test]
    fn subtree_shadows_preserve_composite_masks_on_text_batches() {
        for parent_filtered in [false, true] {
        for filtered in [false, true] {
            let mut text = moda_scene::value::json!({"id":"text", "type":"richtext", "width":80, "height":80,
                "htmlContent":"<p style=\"font-size: 80px;\">H</p>"});
            if filtered { text["effects"] = moda_scene::value::json!([{"id":"blur", "type":"layer-blur", "radius":2}]); }
            let scene = Scene::load(&moda_scene::value::json!({"nodes":[
                {"id":"shadow", "type":"group", "children":["clipped"],
                    "effects":[{"id":"shadow", "type":"drop-shadow", "color":"#000000",
                        "blur":0, "offset":{"x":100, "y":0}}]},
                {"id":"clipped", "type":"group", "children":["mask", "text"], "clipMaskNodeId":"mask",
                    "effects":if parent_filtered { moda_scene::value::json!([{"id":"blur", "type":"layer-blur", "radius":2}]) }
                        else { moda_scene::value::json!([]) }},
                {"id":"mask", "type":"ellipse", "x":20, "y":20, "width":40, "height":40,
                    "maskType":"alpha", "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ffffff"}}]},
                text], "pages":[{"id":"page", "width":256, "height":128, "nodeIds":["shadow"]}],
                "config":{}, "canvasRootNodeIds":[], "layoutMode":"vertical"})).unwrap();
            let hashes = crate::residency::node_hashes_for_scene(&scene);
            let mut caches = crate::residency::PlanCaches::default();
            for cached in [false, true, true] {
                let (plan, warnings) = build_coverage_plan_cached(&scene, 0, &RenderOptions::default(),
                    cached.then_some((&mut caches, &hashes))).unwrap();
                assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
                let shadow = plan.items.iter().find_map(|item| match item {
                    DrawItem::LayerBlurComposite { group, .. } if group.filters.shadow.is_some() => Some(group),
                    _ => None,
                }).expect("parent shadow");
                let (group, mask) = shadow.items.iter().find_map(|item| match item {
                    DrawItem::LayerBlurComposite { group, mask, .. }
                        if group.items.iter().any(|item| matches!(item, DrawItem::Text(batch) if batch.node_id == "text")) => Some((group, mask)),
                    _ => None,
                }).expect("text shadow must clip glyph coverage through an identity capture");
                assert_eq!(shadow.masks[mask.as_ref().unwrap().mask_id as usize].owner_node_id, "clipped");
                assert!(group.masks.is_empty());
                assert!(group.filters.blur.is_none() && group.filters.shadow.is_none(), "do not reapply text effects");
            }
        }
        }
    }

    #[test]
    fn subtree_shadows_preserve_masks_at_every_capture_and_paint_depth() {
        // Exercise every subset of four nested masks, with paint before, between,
        // and after capture boundaries. Leaf captures move masks off bodies;
        // group captures reindex them. Siblings must never inherit each other's masks.
        for mask_type in ["alpha", "luminance"] {
            for placements in 0..16 {
                for leaf_filter in [false, true] {
                    let mut nodes = vec![
                        moda_scene::value::json!({"id":"above", "type":"group",
                            "children":["above-mask", "shadow"], "clipMaskNodeId":"above-mask"}),
                        moda_scene::value::json!({"id":"above-mask", "type":"rectangle", "width":90, "height":90,
                            "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ffffff"}}]}),
                        moda_scene::value::json!({"id":"shadow", "type":"group", "children":["level0", "sibling"],
                            "effects":[{"id":"shadow", "type":"drop-shadow", "color":"#000000",
                                "blur":0, "offset":{"x":100, "y":0}}]}),
                        moda_scene::value::json!({"id":"sibling", "type":"rectangle", "width":80, "height":80,
                            "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ff0000"}}]}),
                    ];
                    for level in 0..4 {
                        let mut children = vec![format!("paint{level}")];
                        if level < 3 { children.push(format!("level{}", level + 1)); }
                        let mut group = moda_scene::value::json!({"id":format!("level{level}"), "type":"group",
                            "effects":[{"id":"blur", "type":"layer-blur", "radius":2}]});
                        if placements & (1 << level) != 0 {
                            children.insert(0, format!("mask{level}"));
                            group["clipMaskNodeId"] = moda_scene::value::json!(format!("mask{level}"));
                            nodes.push(moda_scene::value::json!({"id":format!("mask{level}"), "type":"ellipse",
                                "x":level * 2, "y":level * 2, "width":70, "height":70, "maskType":mask_type,
                                "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ffffff"}}]}));
                        }
                        group["children"] = moda_scene::value::json!(children);
                        nodes.push(group);
                        let mut paint = moda_scene::value::json!({"id":format!("paint{level}"), "type":"rectangle",
                            "width":80, "height":80,
                            "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ff0000"}}]});
                        if leaf_filter {
                            paint["effects"] = moda_scene::value::json!([{"id":"blur", "type":"layer-blur", "radius":2}]);
                        }
                        nodes.push(paint);
                    }
                    let scene = Scene::load(&moda_scene::value::json!({"nodes":nodes,
                        "pages":[{"id":"page", "width":256, "height":128, "nodeIds":["above"]}],
                        "config":{}, "canvasRootNodeIds":[], "layoutMode":"vertical"})).unwrap();
                    let hashes = crate::residency::node_hashes_for_scene(&scene);
                    let mut caches = crate::residency::PlanCaches::default();
                    for cached in [false, true, true] {
                        let (plan, warnings) = build_coverage_plan_cached(&scene, 0, &RenderOptions::default(),
                            cached.then_some((&mut caches, &hashes))).unwrap();
                        assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
                        let group = plan.items.iter().find_map(|item| match item {
                            DrawItem::LayerBlurComposite { group, .. } if group.filters.shadow.is_some() => Some(group),
                            _ => None,
                        }).expect("parent shadow");
                        for level in 0..=4 {
                            let id = if level == 4 { "sibling".to_string() } else { format!("paint{level}") };
                            let expected: Vec<_> = (0..=level.min(3)).rev()
                                .filter(|i| level != 4 && placements & (1 << i) != 0)
                                .map(|i| format!("level{i}")).collect();
                            let bodies: Vec<_> = group.items.iter().filter_map(|item| match item {
                                DrawItem::Body { node_id, mask, .. } if silhouette_owner_id(node_id) == id => Some(mask),
                                _ => None,
                            }).collect();
                            assert!(!bodies.is_empty(), "missing {id}");
                            for mask in bodies {
                                let mut actual = Vec::new();
                                let mut next = mask.as_ref().map(|m| m.mask_id as usize);
                                while let Some(index) = next {
                                    let plan = &group.masks[index];
                                    actual.push(plan.owner_node_id.clone());
                                    next = plan.ancestor;
                                }
                                assert_eq!(actual, expected,
                                    "{mask_type}, placements {placements}, leaf {leaf_filter}, cached {cached}, {id}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn vp_maps_page_rect_to_ndc() {
        // Page at (0, 0), 1280×800: world (0,0) → NDC (−1, +1); world
        // (1280,800) → (+1, −1) (Y flipped).
        let vp = view_projection(0.0, 0.0, 1.0, 1280.0, 800.0);
        let apply = |m: &[f32; 16], x: f32, y: f32| -> (f32, f32) {
            (m[0] * x + m[4] * y + m[12], m[1] * x + m[5] * y + m[13])
        };
        let (x0, y0) = apply(&vp, 0.0, 0.0);
        assert!((x0 + 1.0).abs() < 1e-6 && (y0 - 1.0).abs() < 1e-6);
        let (x1, y1) = apply(&vp, 1280.0, 800.0);
        assert!((x1 - 1.0).abs() < 1e-6 && (y1 + 1.0).abs() < 1e-6);
    }

    /// Shadow knockout: a fully-opaque SDF body flags its drop shadow
    /// (`shadow_offset.w == 1`) so the shader skips the coverage knockout — the
    /// opaque body occludes its own shadow, so the knockout would only carve an
    /// AA seam. A translucent body (low fill alpha / `opacity < 1`, or a gradient
    /// with a transparent stop) keeps the flag clear so knockout still applies.
    /// CPU plan-only (no GPU adapter needed).
    #[test]
    fn opaque_sdf_body_flags_drop_shadow_to_skip_knockout() {
        fn shadow_w_flag(node_json: &str) -> f32 {
            let json = format!(
                r#"{{"nodes":[{node_json}],
                    "pages":[{{"id":"p1","name":"P1","width":400,"height":300,"nodeIds":["n"]}}],
                    "config":{{}},"variables":[],"canvasRootNodeIds":[],
                    "layoutMode":"vertical","category":null,"animations":null}}"#
            );
            let value: moda_scene::value::Value = moda_scene::value::from_str(&json).unwrap();
            let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
            let opts = RenderOptions::default();
            let (plan, _warnings) = build_coverage_plan(&scene, 0, &opts).expect("plan builds");
            let shadow = plan
                .items
                .iter()
                .find_map(|d| match d {
                    DrawItem::Shadow {
                        instance,
                        inner: false,
                        ..
                    } => Some(instance),
                    _ => None,
                })
                .expect("a drop-shadow draw item");
            shadow.shadow_offset[3]
        }

        // Opaque solid rect + drop shadow → flag set (body covers the shadow).
        let opaque_solid = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],"effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(opaque_solid), 1.0);

        // Translucent solid (opacity < 1) + drop shadow → flag clear (knockout on).
        let translucent_solid = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],"opacity":0.5,"effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(translucent_solid), 0.0);

        // Opaque linear-gradient rect + drop shadow → flag set.
        let opaque_gradient = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 40}, "end": {"x": 120, "y": 40}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}],
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(opaque_gradient), 1.0);

        // Gradient with a transparent stop + drop shadow → flag clear.
        let transparent_gradient = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 40}, "end": {"x": 120, "y": 40}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "rgba(58,134,255,0)"}]}}],
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(transparent_gradient), 0.0);

        // Stroked SDF bodies (curved so the stroke routes to the ring):
        // opaque fill + opaque CENTER stroke + drop shadow → flag set (the opaque
        // body occludes its shadow; knockout is skipped to avoid an AA seam).
        let opaque_center = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"cornerRadius":18,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#1d2733"}, "width": 8, "align": "center"}],
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(opaque_center), 1.0);

        // Opaque fill + TRANSLUCENT OUTSIDE stroke + drop shadow → flag CLEAR: the
        // translucent band sits outside the fill, so the shadow there must still be
        // carved (it must not show through the stroke — CSS box-shadow semantics).
        let translucent_outside = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"cornerRadius":18,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "rgba(29,39,51,0.5)"}, "width": 8, "align": "outside"}],
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(translucent_outside), 0.0);

        // Opaque fill + INSIDE stroke + drop shadow → flag CLEAR regardless of stroke
        // opacity: the drop-shadow silhouette is expanded by half the stroke width on
        // every side, but an inside stroke paints nothing beyond the fill edge, so the
        // outer half-stroke band is uncovered. Skipping the knockout there would leak
        // the shadow as a halo, so the knockout must stay on (matching the tessellated
        // path, which never sets this flag).
        let opaque_inside = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"cornerRadius":18,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#1d2733"}, "width": 8, "align": "inside"}],
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(opaque_inside), 0.0);
        let translucent_inside = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"cornerRadius":18,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "rgba(29,39,51,0.5)"}, "width": 8, "align": "inside"}],
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(translucent_inside), 0.0);

        // Opaque fill + ALPHA-0 (invisible) stroke + drop shadow → flag CLEAR. The
        // stroke paints no ring, but `drop_shadow_instance` still expands the
        // silhouette by the PAINTED stroke width — an alpha-0 stroke string is a
        // paint route (`painted_stroke_width`) — so the knockout must stay on or
        // the expanded shadow leaks as a halo around the fill (MB-1 regression guard).
        let alpha_zero_stroke = r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"cornerRadius":18,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "rgba(0,0,0,0)"}, "width": 8}],
            "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000","blur":12}]}"##;
        assert_eq!(shadow_w_flag(alpha_zero_stroke), 0.0);
    }

    /// Every drawing drop / inner shadow gets its own quad, in `effects` order
    /// (back to front); a hidden or non-drawing entry gets none. A body with
    /// two inner shadows leaves the SDF lane (it bakes one) and tessellates,
    /// so both inset shadows draw as quads over it. CPU plan-only.
    #[test]
    fn every_drop_and_inner_shadow_draws_in_list_order() {
        fn plan_items(node_json: &str) -> Vec<DrawItem> {
            let json = format!(
                r#"{{"nodes":[{node_json}],
                    "pages":[{{"id":"p1","name":"P1","width":400,"height":300,"nodeIds":["n"]}}],
                    "config":{{}},"variables":[],"canvasRootNodeIds":[],
                    "layoutMode":"vertical","category":null,"animations":null}}"#
            );
            let value: moda_scene::value::Value = moda_scene::value::from_str(&json).unwrap();
            let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
            let opts = RenderOptions::default();
            build_coverage_plan(&scene, 0, &opts).expect("plan builds").0.items
        }
        fn shadows(items: &[DrawItem], want_inner: bool) -> Vec<(f64, [f32; 4], [f32; 4])> {
            items
                .iter()
                .filter_map(|d| match d {
                    DrawItem::Shadow {
                        z, instance, inner, ..
                    } if *inner == want_inner => {
                        Some((*z, instance.shadow_color, instance.shadow_offset))
                    }
                    _ => None,
                })
                .collect()
        }

        let drops = plan_items(
            r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
            "effects":[
              {"id":"drop-shadow-0","type":"drop-shadow","color":"#ff0000","blur":0,"offset":{"x":-8,"y":-8},"opacity":0.5},
              {"id":"drop-shadow-1","type":"drop-shadow","color":"#00ff00","blur":4,"offset":{"x":0,"y":0},"visible":false},
              {"id":"drop-shadow-2","type":"drop-shadow","color":"#0000ff","blur":0,"offset":{"x":8,"y":8},"showBehindNode":true}]}"##,
        );
        let d = shadows(&drops, false);
        assert_eq!(d.len(), 2, "the hidden entry draws nothing");
        // Both at the body's z: same bucket, drawn in plan order before the body.
        assert_eq!(d[0].0, d[1].0);
        assert_eq!(d[0].1, [1.0, 0.0, 0.0, 0.5], "effects[0] first (furthest behind)");
        assert_eq!(d[1].1, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(&d[0].2[..3], &[-8.0, -8.0, 0.0]);
        assert_eq!(&d[1].2[..3], &[8.0, 8.0, 1.0], "showBehindNode is per shadow");
        // The opaque SDF body covers both shadows (knockout skipped for each).
        assert_eq!((d[0].2[3], d[1].2[3]), (1.0, 1.0));

        // The node's own SDF body (the page background is an SDF item too).
        let is_node_sdf =
            |i: &DrawItem| matches!(i, DrawItem::SolidShapeSdf { node_id, .. } if node_id == "n");
        let one_inner = plan_items(
            r##"{"id":"n","type":"ellipse","x":40,"y":40,"width":120,"height":120,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
            "effects":[{"id":"inner-shadow-0","type":"inner-shadow","color":"#ff0000","blur":6,"offset":{"x":4,"y":4}}]}"##,
        );
        assert!(
            one_inner.iter().any(is_node_sdf),
            "one inner shadow stays baked into the SDF body"
        );
        assert!(shadows(&one_inner, true).is_empty());

        let two_inner = plan_items(
            r##"{"id":"n","type":"ellipse","x":40,"y":40,"width":120,"height":120,
            "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
            "effects":[
              {"id":"inner-shadow-0","type":"inner-shadow","color":"#ff0000","blur":6,"offset":{"x":4,"y":4}},
              {"id":"inner-shadow-1","type":"inner-shadow","color":"#0000ff","blur":6,"offset":{"x":-4,"y":-4},"opacity":0.5}]}"##,
        );
        assert!(
            !two_inner.iter().any(is_node_sdf),
            "two inner shadows leave the SDF lane"
        );
        let i = shadows(&two_inner, true);
        assert_eq!(i.len(), 2);
        assert_eq!(i[0].1, [1.0, 0.0, 0.0, 1.0], "effects[0] first");
        assert_eq!(i[1].1, [0.0, 0.0, 1.0, 0.5]);
        assert_eq!(&i[0].2[..2], &[4.0, 4.0]);
        assert_eq!(&i[1].2[..2], &[-4.0, -4.0]);
    }

    /// End-to-end plan routing for the SDF stroke ring — a solid stroke
    /// on a CURVED body emits a `SolidShapeSdf`/`GradientShapeSdf` with the stroke
    /// fields packed and the correct phase; a sharp-rect stroke still tessellates.
    #[test]
    fn sdf_stroke_ring_routing_phase_and_fields() {
        struct SdfStroke {
            gradient: bool,
            opaque: bool,
            stroke_color: [f32; 4],
            stroke_params: [f32; 4],
        }
        // The body node's SDF draw item, or None if it routed to tessellation.
        fn sdf_stroke(node_json: &str) -> Option<SdfStroke> {
            let json = format!(
                r#"{{"nodes":[{node_json}],
                    "pages":[{{"id":"p1","name":"P1","width":400,"height":300,"nodeIds":["n"]}}],
                    "config":{{}},"variables":[],"canvasRootNodeIds":[],
                    "layoutMode":"vertical","category":null,"animations":null}}"#
            );
            let value: moda_scene::value::Value = moda_scene::value::from_str(&json).unwrap();
            let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
            let (plan, _w) =
                build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
            // Match the body node "n" specifically — the page background is also a
            // (strokeless) SDF body and is drawn first.
            plan.items.iter().find_map(|d| match d {
                DrawItem::SolidShapeSdf {
                    node_id,
                    phase,
                    instance,
                    ..
                } if node_id == "n" => Some(SdfStroke {
                    gradient: false,
                    opaque: matches!(phase, Phase::Opaque),
                    stroke_color: instance.stroke_color,
                    stroke_params: instance.stroke_params,
                }),
                DrawItem::GradientShapeSdf {
                    node_id,
                    phase,
                    instance,
                    ..
                } if node_id == "n" => Some(SdfStroke {
                    gradient: true,
                    opaque: matches!(phase, Phase::Opaque),
                    stroke_color: instance.stroke_color,
                    stroke_params: instance.stroke_params,
                }),
                _ => None,
            })
        }

        // Opaque solid stroke on a rounded rect → SolidShapeSdf, opaque, packed.
        let rr = sdf_stroke(
            r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"cornerRadius":16,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#ff0000"}, "width": 6, "align": "center"}]}"##,
        )
        .expect("rounded rect stroke routes via SDF");
        assert!(!rr.gradient);
        assert!(rr.opaque);
        assert_eq!(rr.stroke_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(rr.stroke_params, [6.0, 0.0, 0.0, 0.0]); // center

        // Translucent solid stroke on an ellipse → transparent phase, outside align.
        let el = sdf_stroke(
            r##"{"id":"n","type":"ellipse","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "rgba(0,0,0,0.5)"}, "width": 4, "align": "outside"}]}"##,
        )
        .expect("ellipse stroke routes via SDF");
        assert!(!el.gradient);
        assert!(!el.opaque);
        assert!((el.stroke_color[3] - 0.5).abs() < 1e-6);
        assert_eq!(el.stroke_params[1], 2.0); // outside

        // Solid stroke on a gradient-fill rounded rect → GradientShapeSdf, inside.
        let gr = sdf_stroke(
            r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"cornerRadius":20,"fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 40}, "end": {"x": 120, "y": 40}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}],
            "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#222222"}, "width": 5, "align": "inside"}]}"##,
        )
        .expect("gradient rect stroke routes via SDF");
        assert!(gr.gradient);
        assert_eq!(gr.stroke_params, [5.0, 1.0, 0.0, 0.0]); // inside

        // Sharp rect + stroke → tessellates (no SDF item).
        assert!(
            sdf_stroke(
                r##"{"id":"n","type":"rectangle","x":40,"y":40,"width":120,"height":80,
            "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#3a86ff"}}],"strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 4}]}"##
            )
            .is_none(),
            "sharp rect + stroke must tessellate, not route to SDF"
        );
    }

    /// Task #95 R2c: pooled buffers retain capacity (reuse on shrink, grow on
    /// overflow).
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn buffer_pool_grow_reuse() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let mut engine = Engine::new(gpu).expect("engine init");
        engine.set_retained(true);
        let b1 = engine.make_buffer("test pool buf", &[1u8; 16], wgpu::BufferUsages::STORAGE);
        assert_eq!(b1.size(), 16);
        // Smaller write reuses the retained capacity (no shrink).
        let b2 = engine.make_buffer("test pool buf", &[2u8; 8], wgpu::BufferUsages::STORAGE);
        assert_eq!(b2.size(), 16, "capacity must be retained on smaller writes");
        // Larger write grows (>= requested).
        let b3 = engine.make_buffer("test pool buf", &[3u8; 64], wgpu::BufferUsages::STORAGE);
        assert!(b3.size() >= 64, "pool must grow to fit");
        // Non-retained path: exact-size allocations every call.
        engine.set_retained(false);
        let c1 = engine.make_buffer("test pool buf", &[1u8; 8], wgpu::BufferUsages::STORAGE);
        assert_eq!(c1.size(), 8);
    }

    /// Task #95 R2a: targets cached by (w, h, samples), bounded LRU.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn target_cache_by_size() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let mut engine = Engine::new(gpu).expect("engine init");
        engine.set_retained(true);
        let _ = engine.frame_targets(640, 480, 4);
        let _ = engine.frame_targets(640, 480, 4);
        assert_eq!(engine.retained_frame.borrow().targets.len(), 1);
        let _ = engine.frame_targets(320, 240, 4);
        let _ = engine.frame_targets(640, 480, 1);
        let _ = engine.frame_targets(100, 100, 4);
        assert!(engine.retained_frame.borrow().targets.len() <= RETAINED_TARGET_SLOTS);
    }

    /// Captures (rendered masks, layer-blur groups) are sampled after the render's other captures and its own target
    /// are taken, so a retained engine gives each its own target: two of one size never share a texture, and more
    /// sizes than retained slots evict (destroy) nothing a frame still reads. Six mask sizes in one frame drew the
    /// whole canvas black ("Destroyed texture used in a submit").
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn captures_never_share_or_evict_retained_targets() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let mut engine = Engine::new(gpu).expect("engine init");
        engine.set_retained(true);
        let (page, _, _) = engine.frame_targets(640, 480, 4);
        engine.capture_depth.set(1);
        let (a, _, _) = engine.frame_targets(64, 64, 1);
        let (b, _, _) = engine.frame_targets(64, 64, 1);
        for side in 1..=(RETAINED_TARGET_SLOTS as u32 + 3) {
            let _ = engine.frame_targets(32 * side, 16 * side, 1);
        }
        engine.capture_depth.set(0);
        assert!(a != b, "two captures of one size share a target");
        let g = engine.retained_frame.borrow();
        assert_eq!(g.targets.len(), 1, "captures took retained slots");
        assert!(g.targets[0].target == page, "a capture evicted the page's target");
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn scratch_attachment_lru_preserves_hot_exact_sizes() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let engine = Engine::new(gpu).expect("engine init");
        let (target, msaa, depth) = engine.frame_targets(64, 48, 4);
        let _ = engine.frame_targets(16, 16, 1);
        let _ = engine.frame_targets(32, 32, 1);
        let (next_target, next_msaa, next_depth) = engine.frame_targets(64, 48, 4);
        assert_ne!(target, next_target);
        assert_eq!(msaa, next_msaa);
        assert_eq!(depth, next_depth);
        let _ = engine.frame_targets(64, 64, 1);
        let kept = engine.scratch_attachments.borrow();
        assert_eq!(kept.len(), SCRATCH_ATTACHMENT_SLOTS);
        assert!(kept.iter().any(|entry| entry.key == (64, 48, 4)));
        assert!(!kept.iter().any(|entry| entry.key == (16, 16, 1)));
        drop(kept);
        for width in 65..125 {
            let (target, msaa, depth) = engine.frame_targets(width, 48, 1);
            assert_eq!((target.width(), target.height(), target.sample_count()), (width, 48, 1));
            assert_eq!((depth.width(), depth.height(), depth.sample_count()), (width, 48, 1));
            assert!(msaa.is_none());
            let kept = engine.scratch_attachments.borrow();
            assert!(kept.len() <= SCRATCH_ATTACHMENT_SLOTS);
            assert!(kept.iter().map(|entry| entry.bytes).sum::<u64>() <= SCRATCH_ATTACHMENT_BYTES);
        }
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn scratch_attachment_keys_keep_sample_counts_distinct() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let engine = Engine::new(gpu).expect("engine init");
        let (_, single_msaa, single_depth) = engine.frame_targets(32, 32, 1);
        let (_, multi_msaa, multi_depth) = engine.frame_targets(32, 32, 4);
        assert!(single_msaa.is_none());
        assert_eq!(multi_msaa.as_ref().unwrap().sample_count(), 4);
        assert_eq!(single_depth.sample_count(), 1);
        assert_eq!(multi_depth.sample_count(), 4);
        assert_ne!(single_depth, multi_depth);
        assert_eq!(engine.scratch_attachments.borrow().len(), 2);
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn scratch_attachment_byte_budget_evicts_below_slot_limit() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let engine = Engine::new(gpu).expect("engine init");
        let _ = engine.frame_targets(16, 16, 1);
        engine.scratch_attachments.borrow_mut()[0].bytes = SCRATCH_ATTACHMENT_BYTES;
        let _ = engine.frame_targets(32, 32, 1);
        let kept = engine.scratch_attachments.borrow();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].key, (32, 32, 1));
        assert!(kept.iter().map(|entry| entry.bytes).sum::<u64>() <= SCRATCH_ATTACHMENT_BYTES);
    }

    fn test_camera(
        position: [f64; 2],
        zoom: f64,
        viewport: [f64; 2],
        dpr: f64,
        clear_rgb: [f32; 3],
    ) -> CameraParams {
        CameraParams {
            camera_x: position[0],
            camera_y: position[1],
            zoom,
            viewport_w: viewport[0],
            viewport_h: viewport[1],
            dpr,
            clear_rgb,
        }
    }

    fn simple_present_scene() -> moda_scene::Scene {
        let json = r##"{
            "nodes": [
                {"id":"r1","type":"rectangle","x":40,"y":40,"width":120,"height":80,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}],"nodeIds":["r1"]}
            ],
            "config":{}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap()
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn retained_offscreen_packets_match_fresh_and_keep_target_lifetimes() {
        let gpu = gpu("retained_offscreen_packets_match_fresh_and_keep_target_lifetimes");
        let mut engine = Engine::new(gpu).expect("engine init");
        engine.set_retained(true);
        let scene = moda_scene::Scene::load(&moda_scene::value::json!({
            "nodes": [
                {"id":"g", "type":"group", "x":23, "y":19, "children":["m", "r"], "clipMaskNodeId":"m"},
                {"id":"m", "type":"ellipse", "x":8, "y":5, "width":90, "height":70, "maskType":"vector"},
                {"id":"r", "type":"rectangle", "x":0, "y":0, "width":160, "height":120,
                 "fills":[{"id":"fill", "paint":{"type":"solid", "color":"#ff0000"}}]}
            ],
            "pages":[{"id":"p", "width":320, "height":240, "nodeIds":["g"],
                "fills":[{"id":"bg", "paint":{"type":"solid", "color":"#0000ff"}}]}],
            "config":{}, "canvasRootNodeIds":[], "layoutMode":"vertical"
        })).unwrap();
        for samples in [1, 4] {
            for reuse_targets in [true, false] {
                engine.set_reuse_frame_targets(reuse_targets);
                // Every assertion names the case, and the packet and counter checks print the packet's
                // counters: the test needs a GPU, so a failure is read from a log.
                let case = format!("samples {samples}, reuse_targets {reuse_targets}");
                let counters = |engine: &Engine| {
                    let s = engine.prepared_stats.get();
                    format!("hits {} misses {} rebuilds {} patches {} patch_fallbacks {} ({}) bundle_replays {}, packet bundle {:?}",
                        s.hits, s.misses, s.rebuilds, s.patches, s.patch_fallbacks, engine.patch_fallback_reason.get(), s.bundle_replays,
                        engine.prepared_frame.borrow().as_ref().map(|p| p.bundle.is_some()))
                };
                let plan = std::sync::Arc::new(build_coverage_plan(&scene, 0, &present_test_options(samples)).unwrap().0);
                assert!(!plan.masks.is_empty(), "samples {samples}, reuse_targets {reuse_targets}: the fixture's clip mask is not in the plan");
                let read = |texture: &wgpu::Texture| crate::readback::read_texture_rgba(
                    &engine.gpu.device, &engine.gpu.queue, texture, plan.width_px, plan.height_px).unwrap();
                let expected = read(&engine.render_plan_to_texture(&plan).unwrap());
                let first = engine.render_retained_plan_to_texture(&plan).unwrap();
                assert!(read(&first) == expected, "{case}: the first retained frame differs from a fresh one");
                assert!(engine.prepared_frame.borrow().as_ref().unwrap().bundle.is_none(), "{case}: a first offscreen frame recorded its pass: {}", counters(&engine));
                let before = engine.prepared_stats.get().bundle_replays;
                let second = engine.render_retained_plan_to_texture(&plan).unwrap();
                assert!(read(&second) == expected, "{case}: the second retained frame differs from a fresh one");
                assert!(engine.prepared_frame.borrow().as_ref().unwrap().bundle.is_some(), "{case}: the second frame (a hit) did not record its pass: {}", counters(&engine));
                assert_eq!(engine.prepared_stats.get().bundle_replays, before, "{case}: the replay count changed on the second frame, before any replay was expected: {}", counters(&engine));
                assert!(read(&engine.render_retained_plan_to_texture(&plan).unwrap()) == expected, "{case}: the third retained frame (a replay) differs from a fresh one");
                assert!(engine.prepared_stats.get().bundle_replays > before, "{case}: the third frame did not replay the recorded pass: {}", counters(&engine));
                assert!(!engine.prepared_frame.borrow().as_ref().unwrap().live_masks, "{case}: an offscreen packet has live masks");

                // The same plan identity through a live view must not reuse exact-mask packets.
                let target = test_texture(&engine, "packet mode", plan.width_px, plan.height_px,
                    wgpu::TextureFormat::Rgba8Unorm,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
                let view = target.create_view(&wgpu::TextureViewDescriptor::default());
                engine.render_plan_to_view(&plan, None, &view, None, Some(&plan), true).unwrap();
                assert!(engine.prepared_frame.borrow().as_ref().unwrap().live_masks, "{case}: a live view's packet has no live masks");
                assert!(read(&engine.render_retained_plan_to_texture(&plan).unwrap()) == expected, "{case}: offscreen after a live view differs from a fresh frame");
                assert!(!engine.prepared_frame.borrow().as_ref().unwrap().live_masks, "{case}: the offscreen frame after a live view kept live masks");

                // A new Arc with compatible contents patches/adopts the packet. The first
                // patch records the offscreen bundle; subsequent patches keep and replay it.
                for i in 0..3 {
                    let changed = std::sync::Arc::new((*plan).clone());
                    let before = engine.prepared_stats.get();
                    assert!(read(&engine.render_retained_plan_to_texture(&changed).unwrap()) == expected, "{case}, patched plan {i}: differs from a fresh frame");
                    let after = engine.prepared_stats.get();
                    let at = format!("{case}, patched plan {i}: {}", counters(&engine));
                    let pf = engine.prepared_frame.borrow();
                    let packet = pf.as_ref().unwrap();
                    assert!(packet.pinned.as_ref().is_some_and(|pinned| std::sync::Arc::ptr_eq(pinned, &changed)), "{at}: the packet did not adopt the plan");
                    assert!(packet.bundle.is_some(), "{at}: the packet has no recorded pass");
                    assert_eq!(after.patches, before.patches + 1, "{at}: patches");
                    assert_eq!(after.patch_bytes, before.patch_bytes, "{at}: an equal plan wrote bytes");
                    assert_eq!(after.hits, before.hits + 1, "{at}: hits");
                    assert_eq!(after.rebuilds, before.rebuilds, "{at}: rebuilds");
                    assert_eq!(after.bundle_replays, before.bundle_replays + u64::from(i > 0), "{at}: bundle replays");
                }

                // Without patching, distinct plan identities really are misses. They must
                // rebuild and draw directly rather than record a bundle with no proven reuse.
                engine.set_prepared_patch(false);
                for i in 0..3 {
                    let changed = std::sync::Arc::new((*plan).clone());
                    let before = engine.prepared_stats.get();
                    let was = counters(&engine);
                    assert!(read(&engine.render_retained_plan_to_texture(&changed).unwrap()) == expected, "{case}, unpatched plan {i}: differs from a fresh frame");
                    let after = engine.prepared_stats.get();
                    let at = format!("{case}, unpatched plan {i}: before [{was}], after [{}]", counters(&engine));
                    assert!(engine.prepared_frame.borrow().as_ref().unwrap().bundle.is_none(), "{at}: a miss recorded a pass");
                    assert_eq!(after.patches, before.patches, "{at}: patches");
                    assert_eq!(after.hits, before.hits, "{at}: hits");
                    assert_eq!(after.rebuilds, before.rebuilds + 1, "{at}: rebuilds");
                    assert_eq!(after.bundle_replays, before.bundle_replays, "{at}: bundle replays");
                }
                engine.set_prepared_patch(true);

                if !reuse_targets {
                    // Deferred consumers of embedded pictures retain their own resolve textures.
                    let mut other = (*plan).clone();
                    other.clear_rgba = Some([0., 1., 0., 1.]);
                    engine.render_retained_plan_to_texture(&std::sync::Arc::new(other)).unwrap();
                    assert!(read(&first) == expected, "{case}: the first frame's texture changed under a later render");
                }
            }
        }
    }

    fn render_prepared_test_frame(
        engine: &Engine,
        plan: &CoveragePlan,
        ov: &CameraPresentOverride,
        pp: Option<&std::sync::Arc<CoveragePlan>>,
        label: &str,
        w: u32,
        h: u32,
    ) -> Vec<u8> {
        let tex = test_texture(
            engine,
            label,
            w,
            h,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let blit = engine
            .render_plan_to_view(plan, Some(ov), &view, None, pp, true)
            .expect("render_plan_to_view succeeds");
        assert!(blit.is_none(), "non-backdrop ⇒ direct present (Ok(None))");
        crate::readback::read_texture_rgba(&engine.gpu.device, &engine.gpu.queue, &tex, w, h)
            .expect("readback")
    }

    fn gpu(test_name: &str) -> crate::device::GpuContext {
        crate::device::GpuContext::new(crate::device::BackendChoice::Auto)
            .unwrap_or_else(|error| panic!("{test_name}: GPU adapter required: {error:#}"))
    }

    fn present_test_options(sample_count: u32) -> RenderOptions {
        RenderOptions {
            preserve_page_fill: false,
            scale: 1.0,
            transparent: false,
            sample_count,
            shader_time_s: 0.0,
            exclude: Vec::new(),
            isolate: Vec::new(),
            best_effort_text: true,
            text_masks_available: Some(true),
            working: None,
        }
    }

    fn test_texture(
        engine: &Engine,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        engine.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// Direct-present contract: for a plan with NO backdrop splits,
    /// `render_plan_to_view` renders the scene (MSAA-resolving) DIRECTLY into the
    /// supplied swapchain-like view and returns `Ok(None)` — the caller presents
    /// with no fullscreen blit. This pins both halves of
    /// the gate's common case: (1) the no-backdrop branch is selected, and (2)
    /// the MSAA resolve into an external RENDER_ATTACHMENT view is wgpu-valid (a
    /// validation error here is exactly what re-blanks the canvas). Backdrop-split
    /// plans take the `Ok(Some(target))` fallback; that arm needs a backdrop
    /// fixture and is not covered here.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_plan_to_view_presents_directly_without_backdrop() {
        let gpu = gpu("render_plan_to_view_presents_directly_without_backdrop");
        let engine = Engine::new(gpu).expect("engine init");

        // One page, one rectangle — no fonts/assets, no backdrop effects.
        let scene = simple_present_scene();
        let opts = present_test_options(4);
        let (plan, _warnings) = build_coverage_plan(&scene, 0, &opts).expect("plan builds");

        // A swapchain-like target: RENDER_ATTACHMENT-only, single-sample,
        // matching this engine's (default Rgba8Unorm) frame format — the MSAA
        // pass must resolve straight into it.
        let view = test_texture(
            &engine,
            "test present view",
            plan.width_px,
            plan.height_px,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

        let blit_src = engine
            .render_plan_to_view(&plan, None, &view, None, None, true)
            .expect("render_plan_to_view succeeds resolving into an external view");
        assert!(
            blit_src.is_none(),
            "a no-backdrop plan must render DIRECTLY into the swapchain view (Ok(None)); \
             Ok(Some(_)) means it wrongly fell back to the offscreen+blit path"
        );
    }

    /// Firefox resize red↔blue swap: when the swapchain format
    /// differs from the engine's frame format, a direct MSAA resolve into it
    /// would be illegal — so `direct_present_ok: false` MUST force the
    /// offscreen path EVEN for a no-backdrop plan, returning `Ok(Some(target))`
    /// for the caller to blit in (channel-correct via the format-neutral
    /// present pipeline). Here: a default (Rgba8Unorm-frame) engine against a
    /// bgra swapchain. The normal live engine now always matches its
    /// preferred swapchain, but this generic mismatch path remains required for
    /// actual capability/effect fallbacks. The fallback contract this pins is unchanged: the
    /// returned offscreen target carries the ENGINE's frame format (Rgba8Unorm
    /// here) so the blit is a straight sample.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_plan_to_view_uses_offscreen_when_direct_present_disabled() {
        let gpu = gpu("render_plan_to_view_uses_offscreen_when_direct_present_disabled");
        let engine = Engine::new(gpu).expect("engine init");

        // Same simple one-rectangle, no-backdrop scene as the direct-present test.
        let scene = simple_present_scene();
        let opts = present_test_options(4);
        let (plan, _warnings) = build_coverage_plan(&scene, 0, &opts).expect("plan builds");

        // A desktop-preferred BGRA swapchain-like target — the engine must NOT
        // resolve its Rgba8Unorm MSAA pass straight into this.
        let view = test_texture(
            &engine,
            "test bgra present view",
            plan.width_px,
            plan.height_px,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

        let blit_src = engine
            .render_plan_to_view(&plan, None, &view, None, None, false)
            .expect("render_plan_to_view succeeds with direct present disabled");
        let target = blit_src.expect(
            "with direct_present_ok=false a no-backdrop plan must fall back to the offscreen \
             target (Ok(Some(_))) so the caller can blit into the non-rgba swapchain",
        );
        assert_eq!(
            target.format(),
            wgpu::TextureFormat::Rgba8Unorm,
            "the offscreen blit source must stay Rgba8Unorm so the format-neutral present \
             blit is a straight, channel-correct sample into the bgra swapchain"
        );
    }

    /// The on-screen swapchain format must map DIRECTLY from the
    /// browser's preferred canvas format — NEVER coalesced through
    /// `texture_format` (which folds `bgra8unorm -> Rgba8Unorm` for offscreen
    /// pipelines). Coalescing would silently re-hardcode rgba and reintroduce the
    /// Firefox-on-resize red<->blue swap. Pure — runs in the native CI test lane
    /// (no GPU adapter needed), unlike the surrounding present tests.
    #[test]
    fn present_format_from_preferred_maps_bgra_faithfully() {
        assert_eq!(
            present_format_from_preferred(Some("bgra8unorm")),
            wgpu::TextureFormat::Bgra8Unorm,
            "desktop-preferred bgra8unorm must configure a bgra swapchain, not rgba"
        );
        assert_eq!(
            present_format_from_preferred(Some("rgba8unorm")),
            wgpu::TextureFormat::Rgba8Unorm
        );
        // Absent / unsupported / garbage → the Rgba8Unorm fallback (always legal).
        assert_eq!(
            present_format_from_preferred(None),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            present_format_from_preferred(Some("rgba16float")),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            present_format_from_preferred(Some("not-a-format")),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }

    /// The frame-format allowlist is EXACTLY the two guaranteed
    /// WebGPU canvas-context 8-bit formats — in particular the `-srgb`
    /// variants (which would break the gamma-space blending contract) and
    /// non-frame formats are out. Pure — pins the constructor's rejection in
    /// the native CI lane, where the GPU-gated constructor test skips.
    #[test]
    fn frame_format_allowlist_is_the_canvas_formats_and_float_captures() {
        assert!(is_supported_frame_format(wgpu::TextureFormat::Rgba8Unorm));
        assert!(is_supported_frame_format(wgpu::TextureFormat::Bgra8Unorm));
        assert!(is_supported_frame_format(wgpu::TextureFormat::Rgba16Float));
        for illegal in [
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::R8Unorm,
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureFormat::Depth24Plus,
        ] {
            assert!(
                !is_supported_frame_format(illegal),
                "{illegal:?} must not be a legal frame format"
            );
        }
    }

    /// Manifest COLOR-TARGET spellings fold onto the engine's frame
    /// format — both `rgba8unorm` and `bgra8unorm` name the single frame
    /// attachment (the manifest was extracted in a browser whose preferred
    /// format colored the entry) — while non-frame color formats (the mask
    /// pre-pass's `r8unorm`) keep their literal meaning. Pure — native CI
    /// lane; a regression here otherwise only surfaces at GPU-gated pipeline
    /// creation.
    #[test]
    fn frame_color_target_format_folds_frame_spellings_only() {
        for frame in [
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureFormat::Bgra8Unorm,
        ] {
            assert_eq!(frame_color_target_format("rgba8unorm", frame), frame);
            assert_eq!(frame_color_target_format("bgra8unorm", frame), frame);
            assert_eq!(
                frame_color_target_format("r8unorm", frame),
                wgpu::TextureFormat::R8Unorm
            );
        }
    }

    /// Only the two guaranteed WebGPU canvas-context 8-bit formats
    /// are legal engine frame formats — anything else (notably an `-srgb`
    /// variant, which would break the gamma-space blending contract) is
    /// rejected at construction rather than surfacing later as a pipeline
    /// validation error.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn engine_rejects_unsupported_frame_formats() {
        let gpu = gpu("engine_rejects_unsupported_frame_formats");
        match Engine::new_with_frame_format(gpu, wgpu::TextureFormat::Rgba8UnormSrgb) {
            Ok(_) => panic!("srgb frame format must be rejected"),
            Err(err) => assert!(
                err.to_string().contains("unsupported engine frame format"),
                "unexpected error: {err:#}"
            ),
        }
    }

    /// Goal state: an engine CONSTRUCTED with the desktop-preferred
    /// `Bgra8Unorm` frame format renders a no-backdrop plan DIRECTLY into a
    /// BGRA swapchain-like view (`Ok(None)` — no offscreen render, no
    /// fullscreen blit, no second submit), and the readback canonicalizes the
    /// physically-BGRA texels so the scene's channels come out correct. Both
    /// halves are load-bearing: a frame pipeline whose target format was
    /// missed in the BGRA threading fails validation loudly here, and a red
    /// rect that reads back blue means the readback canonicalization (or a
    /// pipeline swizzle) is wrong.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn bgra_engine_presents_direct_into_bgra_view_with_correct_channels() {
        let gpu = gpu("bgra_engine_presents_direct_into_bgra_view_with_correct_channels");
        let engine = Engine::new_with_frame_format(gpu, wgpu::TextureFormat::Bgra8Unorm)
            .expect("bgra engine init");
        assert_eq!(engine.frame_format(), wgpu::TextureFormat::Bgra8Unorm);

        // Red rect on a GREEN page. Deliberately R↔B-ASYMMETRIC: a red-on-BLUE
        // fixture is invariant under a whole-frame channel swap (the rect and
        // page trade colors but "some red exists" + "some blue exists" both
        // still hold), which made this assertion vacuous against the exact bug
        // it names — proven by mutation (disabling the readback swizzle still
        // passed it). With green as the second color, a swap turns the red
        // rect blue: `has_red` fails AND `has_blue` fires as an explicit
        // detector.
        let json = r##"{
            "nodes": [
                {"id":"r1","type":"rectangle","x":40,"y":40,"width":120,"height":80,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}],"nodeIds":["r1"]}
            ],
            "config":{}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let opts = present_test_options(4);
        let (plan, _warnings) = build_coverage_plan(&scene, 0, &opts).expect("plan builds");

        // A desktop-preferred BGRA swapchain-like target; COPY_SRC so the test
        // can read the presented pixels back.
        let tex = test_texture(
            &engine,
            "test bgra present view (bgra engine)",
            plan.width_px,
            plan.height_px,
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        // Formats match (engine bgra == view bgra) ⇒ the caller passes
        // `direct_present_ok: true` and the MSAA pass must resolve straight in.
        let blit_src = engine
            .render_plan_to_view(&plan, None, &view, None, None, true)
            .expect("bgra engine renders directly into a bgra view");
        assert!(
            blit_src.is_none(),
            "a bgra-frame engine + bgra swapchain must present DIRECT (Ok(None)); \
             Ok(Some(_)) re-adds the offscreen + blit the format match exists to remove"
        );

        // Canonicalized readback: red is red, green is green — and NOTHING is
        // blue (blue pixels can only come from an un-canonicalized swap of the
        // red rect).
        let rgba = crate::readback::read_texture_rgba(
            &engine.gpu.device,
            &engine.gpu.queue,
            &tex,
            plan.width_px,
            plan.height_px,
        )
        .expect("readback of the directly-presented bgra view");
        let has_red = rgba
            .chunks_exact(4)
            .any(|p| p[0] > 200 && p[1] < 50 && p[2] < 50);
        let has_green = rgba
            .chunks_exact(4)
            .any(|p| p[1] > 200 && p[0] < 50 && p[2] < 50);
        let has_blue = rgba
            .chunks_exact(4)
            .any(|p| p[2] > 200 && p[1] < 50 && p[0] < 50);
        assert!(
            has_red && has_green && !has_blue,
            "canonicalized BGRA readback lost the scene's channels \
             (red rect present: {has_red}, green page present: {has_green}, \
             swapped-blue present: {has_blue}) — a red↔blue swap here means \
             the readback canonicalization or a pipeline target format is wrong"
        );
    }

    /// Acceptance: RGBA- and BGRA-frame engines produce PIXEL-IDENTICAL
    /// canonical RGBA bytes for representative content on the same adapter —
    /// the frame format changes texel STORAGE order, never rendered content.
    /// Three fixtures:
    ///   1. an opaque scene (MSAA 4 where the adapter's RGBA and BGRA MSAA
    ///      ceilings agree; single-sample where they differ, since unequal
    ///      ceilings make the engines LEGITIMATELY diverge at edges): page
    ///      fill, opaque + semi-transparent solid bodies, a linear gradient, a
    ///      drop shadow, a rotated ellipse, a richtext node, and a clip-masked
    ///      group (analytic SDF bodies + manifest gradient pipeline + shadow
    ///      pipeline + text pipelines + the mask pre-pass/consumption path —
    ///      text renders best-effort, so where the test env lacks fonts the
    ///      node degrades identically on both engines);
    ///   2. the same scene as a TRANSPARENT export — canonicalization under
    ///      un-premultiplied output (the swizzle is alpha-independent, so this
    ///      guards the combination; the canonicalize-first CALL order is a
    ///      documentation contract, not numerically observable);
    ///   3. a sample-count-1 backdrop-blur + layer-blur scene over a hard
    ///      green/blue edge (the copy/blur/composite chains whose scratch
    ///      textures and pipelines are format-parameterized).
    /// Engines are created sequentially (drop A before B) — GPU device init
    /// deadlocks under parallelism, and plans are engine-agnostic so they can
    /// be built once and rendered on each.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn rgba_and_bgra_engines_render_identical_canonical_bytes() {
        let scene_main = r##"{
            "nodes": [
                {"id":"solid","type":"rectangle","x":20,"y":20,"width":120,"height":80,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff3300"}}],
                 "effects":[{"id":"drop-shadow-0","type":"drop-shadow","color":"#000000","blur":12}]},
                {"id":"trans","type":"rectangle","x":90,"y":60,"width":140,"height":90,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "rgba(0,128,255,0.45)"}}]},
                {"id":"grad","type":"rectangle","x":40,"y":140,"width":200,"height":70,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "linear", "start": {"x": 0, "y": 0}, "end": {"x": 200, "y": 0}, "stops": [{"offset": 0, "color": "#ff006e"}, {"offset": 1, "color": "#3a86ff"}]}}]},
                {"id":"ell","type":"ellipse","x":200,"y":30,"width":90,"height":60,
                 "rotation":25,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00cc44"}}]},
                {"id":"txt","type":"richtext","x":18,"y":8,"width":200,"height":40,
                 "rotation":0,"scale":1,
                 "htmlContent":"<p style=\"font-size: 28px;\">Hi</p>"},
                {"id":"grp","type":"group","x":240,"y":150,"children":["mk","mc"],
                 "clipMaskNodeId":"mk"},
                {"id":"mk","type":"rectangle","x":0,"y":0,"width":70,"height":70,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}],"cornerRadius":12},
                {"id":"mc","type":"ellipse","x":-12,"y":-12,"width":95,"height":95,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffff00"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f5f0e8"}}],
                 "nodeIds":["solid","trans","grad","ell","txt","grp"]}
            ],
            "config":{}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let scene_effects = r##"{
            "nodes": [
                {"id":"bg","type":"rectangle","x":0,"y":0,"width":100,"height":240,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]},
                {"id":"bd","type":"rectangle","x":40,"y":30,"width":120,"height":80,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "rgba(255,0,0,0.4)"}}],
                 "effects": [{"id": "backdrop-0", "type": "background-blur","radius":10.0}]},
                {"id":"lb","type":"rectangle","x":60,"y":130,"width":120,"height":80,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffaa00"}}],
                 "effects": [{"id": "layer-blur-0", "type": "layer-blur","radius":8}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}],
                 "nodeIds":["bg","bd","lb"]}
            ],
            "config":{}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let mk_opts = |transparent: bool, sample_count: u32| RenderOptions {
            scale: 1.0,
            transparent,
            preserve_page_fill: false,
            sample_count,
            shader_time_s: 0.0,
            exclude: Vec::new(),
            isolate: Vec::new(),
            best_effort_text: true,
            text_masks_available: Some(true),
            working: None,
        };
        let plan_for = |json: &str, opts: &RenderOptions| {
            let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
            let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
            let (plan, _warnings) = build_coverage_plan(&scene, 0, opts).expect("plan builds");
            plan
        };

        // Probe the adapter's per-format MSAA ceilings FIRST (reusing this
        // GpuContext for the RGBA engine below). When they differ — the very
        // hardware class `max_sample_count_bgra` exists for — the two engines
        // would LEGITIMATELY diverge at MSAA 4 (one edge-antialiased, one not),
        // so the main fixtures drop to sample_count 1 there: this test proves
        // format threading + canonicalization, not cross-format MSAA parity.
        let gpu_rgba = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let msaa = if gpu_rgba.max_sample_count == gpu_rgba.max_sample_count_bgra {
            4
        } else {
            eprintln!(
                "rgba_and_bgra_engines_render_identical_canonical_bytes: per-format MSAA \
                 ceilings differ (rgba {} vs bgra {}) — main fixtures run single-sample",
                gpu_rgba.max_sample_count, gpu_rgba.max_sample_count_bgra
            );
            1
        };
        // Plans are engine-agnostic — build once, render on each engine.
        let plans = [
            ("opaque", plan_for(scene_main, &mk_opts(false, msaa))),
            ("transparent", plan_for(scene_main, &mk_opts(true, msaa))),
            // Backdrop pages plan at sample_count 1 (the engine bails otherwise).
            (
                "backdrop+layer-blur",
                plan_for(scene_effects, &mk_opts(false, 1)),
            ),
        ];

        let render_all = |gpu: GpuContext, frame_format: wgpu::TextureFormat| -> Vec<Vec<u8>> {
            let engine = Engine::new_with_frame_format(gpu, frame_format).expect("engine init");
            plans
                .iter()
                .map(|(label, plan)| {
                    engine
                        .render_plan(plan)
                        .unwrap_or_else(|e| panic!("{frame_format:?} '{label}' render: {e:#}"))
                })
                .collect()
        };

        // Sequential engine lifetimes (see doc comment): the RGBA engine (and
        // its device) drops at the end of its `render_all` before the BGRA
        // engine's device is created.
        let rgba_adapter = gpu_rgba.adapter_info.clone();
        let rgba_frames = render_all(gpu_rgba, wgpu::TextureFormat::Rgba8Unorm);
        let gpu_bgra = crate::device::GpuContext::new(crate::device::BackendChoice::Auto)
            .expect("adapter vanished mid-test");
        // The "effectively bit-identical, tolerance 2" contract below assumes
        // BOTH renders ran on the SAME adapter. `Auto` re-runs selection, so a
        // multi-GPU box could legitimately hand back a different device with
        // different rounding — skip the strict comparison there rather than
        // flake.
        if gpu_bgra.adapter_info.name != rgba_adapter.name
            || gpu_bgra.adapter_info.backend != rgba_adapter.backend
        {
            eprintln!(
                "rgba_and_bgra_engines_render_identical_canonical_bytes: SKIP — adapter \
                 changed between engines ({} vs {})",
                rgba_adapter.name, gpu_bgra.adapter_info.name
            );
            return;
        }
        let bgra_frames = render_all(gpu_bgra, wgpu::TextureFormat::Bgra8Unorm);

        for (((label, _), rgba), bgra) in plans.iter().zip(&rgba_frames).zip(&bgra_frames) {
            // Non-flat guard: a blank/flat frame would make equality vacuous.
            let first = &rgba[0..4];
            assert!(
                rgba.chunks_exact(4).any(|p| p != first),
                "'{label}' rendered flat — the equivalence check would be vacuous"
            );
            assert_eq!(rgba.len(), bgra.len(), "'{label}' byte lengths differ");
            let max_diff = rgba
                .iter()
                .zip(bgra)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            // Same adapter + identical ops ⇒ effectively bit-identical; allow a
            // hair of tolerance against driver rounding rather than a brittle ==.
            assert!(
                max_diff <= 2,
                "'{label}': BGRA-frame render diverged from RGBA-frame render \
                 (max per-channel diff {max_diff}) — a format thread or the readback \
                 canonicalization is wrong"
            );
        }
    }

    /// Backdrop fallback contract: WITHOUT a copy-source swapchain
    /// (`present_copy_src: None` — e.g. Dawn's Android Vulkan swapchains), a plan
    /// with an active backdrop effect CANNOT render direct: `execute_split` reads
    /// the rendered scene back out of the offscreen target mid-pass to blur it,
    /// and the swapchain can't be sampled/copied. So `render_plan_to_view` must
    /// render into the offscreen target and return `Ok(Some(target))` for the
    /// caller to blit, and that target must stay sampleable (TEXTURE_BINDING). A
    /// regression in the gate (wrongly taking the direct path with no copy source)
    /// would make the backdrop vanish — this pins against it. Backdrop pages plan
    /// at sample_count 1.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_plan_to_view_falls_back_to_offscreen_for_backdrop() {
        let gpu = gpu("render_plan_to_view_falls_back_to_offscreen_for_backdrop");
        let engine = Engine::new(gpu).expect("engine init");

        // A node with an active UNIFORM background blur emits a backdrop split.
        let json = r##"{
            "nodes": [
                {"id":"u","type":"rectangle","x":40,"y":40,"width":120,"height":80,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],
                 "effects": [{"id": "backdrop-0", "type": "background-blur","radius":10.0}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}],"nodeIds":["u"]}
            ],
            "config":{}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        // Backdrop pages plan at sample_count 1 (the engine bails otherwise).
        let opts = present_test_options(1);
        let (plan, _warnings) =
            build_coverage_plan(&scene, 0, &opts).expect("backdrop plan builds");
        let view = test_texture(
            &engine,
            "test present view (backdrop)",
            plan.width_px,
            plan.height_px,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());

        let target = engine
            .render_plan_to_view(&plan, None, &view, None, None, true)
            .expect("render_plan_to_view (backdrop) succeeds")
            .expect(
                "a backdrop-split plan with no copy-source swapchain must fall back to the \
                 offscreen target (Ok(Some)); Ok(None) means the gate wrongly took the direct \
                 path and the backdrop would vanish",
            );
        assert!(
            target
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING),
            "the offscreen fallback target must stay sampleable for the present blit"
        );
    }

    /// Backdrop direct-present contract: WITH a copy-source swapchain
    /// (`present_copy_src: Some(tex)` — the browser case, where the canvas
    /// advertises COPY_SRC), a backdrop-effect plan presents DIRECTLY into the
    /// supplied view and returns `Ok(None)` — no offscreen target, no blit. The
    /// effect chain copies only its small region back out of the swapchain and
    /// composites into it (the WebGPU renderer's approach).
    ///
    /// Two assertions, both load-bearing:
    /// 1. The direct path returns `Ok(None)` (a regression that kept backdrop on
    ///    the offscreen fallback even with a copy-source swapchain would silently
    ///    re-add the per-frame full-viewport blit), and the fallback path
    ///    (`present_copy_src: None`) returns `Ok(Some)`.
    /// 2. The two paths produce PIXEL-EQUIVALENT frames. Both run identical GPU
    ///    work (clear → segment → copy region → blur → composite → segment); the
    ///    ONLY difference is whether `execute_split` reads/composites the
    ///    swapchain or the offscreen target. So a direct-path region mis-wiring
    ///    (wrong copy source or composite destination) diverges from the
    ///    known-good offscreen result. The fixture is built to make that
    ///    detectable: a SEMI-TRANSPARENT backdrop node (so the composited effect
    ///    shows through instead of being hidden by an opaque body) over a
    ///    NON-UNIFORM backdrop (a green/blue edge the blur smears, so a "copied
    ///    the wrong/blank texture" bug produces visibly different pixels).
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_plan_to_view_presents_backdrop_direct_with_copy_src() {
        let gpu = gpu("render_plan_to_view_presents_backdrop_direct_with_copy_src");
        let engine = Engine::new(gpu).expect("engine init");

        // Backdrop fixture sized to expose the region copy/composite (see doc):
        // an opaque green rect over the left of a blue page makes a green/blue
        // edge; a SEMI-TRANSPARENT red node with a uniform blur straddles it, so
        // the composited (blurred) backdrop is visible through the node body.
        let json = r##"{
            "nodes": [
                {"id":"bg","type":"rectangle","x":0,"y":0,"width":100,"height":240,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]},
                {"id":"u","type":"rectangle","x":40,"y":40,"width":120,"height":80,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "rgba(255,0,0,0.4)"}}],
                 "effects": [{"id": "backdrop-0", "type": "background-blur","radius":10.0}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}],"nodeIds":["bg","u"]}
            ],
            "config":{}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        // Backdrop pages plan at sample_count 1 (the engine bails otherwise).
        let opts = present_test_options(1);
        let (plan, _warnings) =
            build_coverage_plan(&scene, 0, &opts).expect("backdrop plan builds");

        let mk_tex = |label: &str| {
            test_texture(
                &engine,
                label,
                plan.width_px,
                plan.height_px,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            )
        };

        // (1a) Direct path: copy-source swapchain supplied → Ok(None).
        let direct_tex = mk_tex("test present view (backdrop direct, copy-src)");
        let direct_view = direct_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let direct = engine
            .render_plan_to_view(&plan, None, &direct_view, Some(&direct_tex), None, true)
            .expect("render_plan_to_view (backdrop direct) succeeds");
        assert!(
            direct.is_none(),
            "a backdrop plan WITH a copy-source swapchain must present DIRECTLY (Ok(None)); \
             Ok(Some(_)) means it wrongly fell back to the offscreen + blit path"
        );
        let rgba_direct = crate::readback::read_texture_rgba(
            &engine.gpu.device,
            &engine.gpu.queue,
            &direct_tex,
            plan.width_px,
            plan.height_px,
        )
        .expect("readback of the directly-presented backdrop view");

        // (1b) Fallback path: no copy-source swapchain → Ok(Some(offscreen)). Its
        // returned target already holds the composited result (the caller would
        // blit it); read THAT back as the known-good reference.
        let fb_view = mk_tex("test present view (backdrop fallback)")
            .create_view(&wgpu::TextureViewDescriptor::default());
        let fallback_target = engine
            .render_plan_to_view(&plan, None, &fb_view, None, None, true)
            .expect("render_plan_to_view (backdrop fallback) succeeds")
            .expect("no copy-source swapchain ⇒ offscreen fallback returns Ok(Some)");
        let rgba_fallback = crate::readback::read_texture_rgba(
            &engine.gpu.device,
            &engine.gpu.queue,
            &fallback_target,
            plan.width_px,
            plan.height_px,
        )
        .expect("readback of the offscreen fallback target");

        // Sanity: the reference frame is non-trivial (not a single flat color) —
        // otherwise the equivalence below could pass on two blank frames.
        let first = &rgba_fallback[0..4];
        let has_variation = rgba_fallback.chunks_exact(4).any(|p| p != first);
        assert!(
            has_variation,
            "fixture produced a flat frame; the equivalence check would be vacuous"
        );

        // (2) The direct path and the offscreen fallback run identical GPU work —
        // their frames must match. A region copy/composite mis-wiring in the
        // direct path (e.g. reading the wrong texture) would diverge here. Same
        // adapter + identical ops ⇒ effectively bit-identical; allow a hair of
        // tolerance against driver rounding rather than risk a brittle ==.
        assert_eq!(rgba_direct.len(), rgba_fallback.len(), "frame sizes differ");
        let max_diff = rgba_direct
            .iter()
            .zip(&rgba_fallback)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 2,
            "backdrop direct-present frame diverged from the offscreen fallback \
             (max per-channel diff {max_diff}); the direct region copy/composite is mis-wired"
        );
    }

    /// Pixel-level guard for the direct path: it must not merely return `Ok(None)`
    /// without erroring — it must actually WRITE the rendered scene into the
    /// supplied view (the historical blank-canvas failure mode is "valid call,
    /// blank output"). Renders a red rectangle on a blue page directly into a
    /// readback-capable (RENDER_ATTACHMENT | COPY_SRC) view, reads it back, and
    /// asserts both the blue page fill and the red rectangle are present
    /// (orientation-independent — checks the pixels exist anywhere, not at fixed
    /// coords).
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn render_plan_to_view_writes_scene_pixels() {
        let gpu = gpu("render_plan_to_view_writes_scene_pixels");
        let engine = Engine::new(gpu).expect("engine init");

        let scene = simple_present_scene();
        let opts = present_test_options(4);
        let (plan, _warnings) = build_coverage_plan(&scene, 0, &opts).expect("plan builds");

        // Readback-capable swapchain-shaped view: RENDER_ATTACHMENT (resolve
        // target) + COPY_SRC (read the result back).
        let tex = test_texture(
            &engine,
            "test present view (readback)",
            plan.width_px,
            plan.height_px,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        let blit_src = engine
            .render_plan_to_view(&plan, None, &view, None, None, true)
            .expect("render_plan_to_view succeeds");
        assert!(
            blit_src.is_none(),
            "no-backdrop plan must present directly (Ok(None))"
        );

        let rgba = crate::readback::read_texture_rgba(
            &engine.gpu.device,
            &engine.gpu.queue,
            &tex,
            plan.width_px,
            plan.height_px,
        )
        .expect("readback of the directly-presented view");

        let has_blue = rgba
            .chunks_exact(4)
            .any(|p| p[2] > 200 && p[0] < 60 && p[1] < 60);
        let has_red = rgba
            .chunks_exact(4)
            .any(|p| p[0] > 200 && p[1] < 60 && p[2] < 60);
        assert!(
            has_blue,
            "direct path did not write the blue page fill into the supplied view (blank-present regression)"
        );
        assert!(
            has_red,
            "direct path did not write the red rectangle into the supplied view"
        );
    }

    /// A plan with a background blur follows the camera: each pan frame is another plan. When the packet is
    /// rebuilt for it (the rebuild path; an equal plan would be patched: tests/prepared_captures.rs), its text is
    /// kept from the last frame when the batches are the same `Arc`s, and the frame draws exactly what a full
    /// encode does.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn split_plans_keep_their_prepared_text_across_frames() {
        let gpu = gpu("split_plans_keep_their_prepared_text_across_frames");
        let mut engine = Engine::new(gpu).expect("engine init");
        engine.set_retained(true);
        engine.set_prepared_patch(false);
        let json = r##"{
            "nodes": [
                {"id":"t1","type":"richtext","x":18,"y":8,"width":200,"height":40,"rotation":0,"scale":1,
                 "htmlContent":"<p style=\"font-size: 28px;\">Hi</p>"},
                {"id":"g1","type":"rectangle","x":40,"y":30,"width":160,"height":100,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}, "opacity": 0.5}],
                 "effects": [{"id":"bg","type":"background-blur","radius":8}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}],
                 "nodeIds":["t1","g1"]}
            ],
            "config":{"renderPageClipping":true}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let cam = test_camera([0.0, 0.0], 1.0, [320.0, 240.0], 1.0, [0.0, 0.0, 1.0]);
        let (plan, warnings) = build_camera_plan(&scene, &[0], &cam, &editor_camera_opts())
            .expect("plan builds")
            .expect("Some");
        assert!(plan.items.iter().any(|i| matches!(i, DrawItem::BackdropSplit(_))), "the plan splits");
        // A plan without the text is a planning warning, not a packet defect: the usual one is the unit tests'
        // font fixtures missing where the test binary looks for them (a binary run on another machine).
        assert!(plan.items.iter().any(|i| matches!(i, DrawItem::Text(_))), "and draws text; planning said {:?}", warnings.skipped);
        let ov = cam.present_override(0.0);
        let (w, h) = (ov.width_px, ov.height_px);
        // A backdrop frame presents direct into a copy-source surface (as the editor's swapchain is).
        let render = |engine: &Engine, plan: &std::sync::Arc<CoveragePlan>, pp: bool| {
            let tex = test_texture(
                engine,
                "split text",
                w,
                h,
                engine.frame_format(),
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let blit = engine
                .render_plan_to_view(plan, Some(&ov), &view, Some(&tex), pp.then_some(plan), true)
                .expect("render_plan_to_view succeeds");
            assert!(blit.is_none(), "a copy-source surface presents direct");
            crate::readback::read_texture_rgba(&engine.gpu.device, &engine.gpu.queue, &tex, w, h).expect("readback")
        };
        let a = std::sync::Arc::new(plan);
        let full = render(&engine, &a, false);
        render(&engine, &a, true);
        let before = engine.prepared_frame_stats();
        // The next pan frame: another plan (the split re-planned), the same text batches.
        let b = std::sync::Arc::new((*a).clone());
        let px = render(&engine, &b, true);
        let after = engine.prepared_frame_stats();
        assert_eq!(after.text_reuses, before.text_reuses + 1, "the text is kept");
        assert_eq!(after.text_bytes, before.text_bytes, "and not uploaded again");
        assert!(px == full, "and draws what a full encode does");
    }

    /// A pan (the same plan `Arc`, the same coverage scale) binds the mask atlas the last frame built without keying
    /// its masks again, and draws what a full encode does; another plan (an edit) keys them again.
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn pans_reuse_the_mask_atlas_of_the_same_plan() {
        let gpu = gpu("pans_reuse_the_mask_atlas_of_the_same_plan");
        let mut engine = Engine::new(gpu).expect("engine init");
        engine.set_retained(true);
        // The rebuild path (a patch would adopt the equal plan in place: tests/prepared_patch.rs).
        engine.set_prepared_patch(false);
        let json = r##"{
            "nodes": [
                {"id":"g1","type":"group","x":12,"y":20,"children":["mk1","mc1"],"clipMaskNodeId":"mk1"},
                {"id":"mk1","type":"ellipse","x":0,"y":0,"width":90,"height":70,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]},
                {"id":"mc1","type":"rectangle","x":-12,"y":-12,"width":110,"height":110,
                 "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffff00"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}],
                 "nodeIds":["g1"]}
            ],
            "config":{"renderPageClipping":true}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let cam = test_camera([0.0, 0.0], 1.0, [320.0, 240.0], 1.0, [0.0, 0.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &editor_camera_opts())
            .expect("plan builds")
            .expect("Some");
        assert!(!plan.masks.is_empty(), "the plan has a mask");
        let a = std::sync::Arc::new(plan);
        let pan = test_camera([30.0, 12.0], 1.0, [320.0, 240.0], 1.0, [0.0, 0.0, 1.0]).present_override(0.0);
        let (w, h) = (pan.width_px, pan.height_px);
        let full = render_prepared_test_frame(&engine, &a, &pan, None, "mask memo", w, h);
        render_prepared_test_frame(&engine, &a, &cam.present_override(0.0), Some(&a), "mask memo", w, h);
        let before = engine.mask_cache_stats().expect("masks cached");
        let px = render_prepared_test_frame(&engine, &a, &pan, Some(&a), "mask memo", w, h);
        let after = engine.mask_cache_stats().expect("masks cached");
        assert_eq!(after.0, before.0, "the pan keyed no mask (the atlas was reused)");
        assert!(px == full, "and draws what a full encode does");
        let b = std::sync::Arc::new((*a).clone());
        render_prepared_test_frame(&engine, &b, &pan, Some(&b), "mask memo", w, h);
        assert!(engine.mask_cache_stats().unwrap().0 > after.0, "another plan keys its masks again");
    }



    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn prepared_frame_reencodes_on_shader_clock_change() {
        let gpu = gpu("prepared_frame_reencodes_on_shader_clock_change");
        let mut engine = Engine::new(gpu).expect("engine init");
        engine.set_retained(true);

        // A full-viewport animated procedural shader (clouds packs the clock into
        // shader_params[0]) — the animated-background case Lever 1 keeps on a
        // stable cached plan Arc.
        let json = r##"{
            "nodes": [
                {"id":"s","type":"rectangle","x":0,"y":0,"width":320,"height":240,
                 "rotation":0,"scaleX":1,"scaleY":1,
                 "fills": [{"id": "fill-0", "paint": {"type": "shader", "shaderId": "clouds"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}],
                 "nodeIds":["s"]}
            ],
            "config":{"renderPageClipping":true}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        let scene = moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([0.0, 0.0], 1.0, [320.0, 240.0], 1.0, [0.0, 0.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .expect("plan builds")
            .expect("non-backdrop ⇒ Some");
        let plan = std::sync::Arc::new(plan);

        // SAME camera, two distinct clocks — the pure Lever-1 hit case (only the
        // clock advances).
        let ov0 = cam.present_override(0.0);
        let ov1 = cam.present_override(6.0);
        let (w, h) = (ov0.width_px, ov0.height_px);

        let render = |engine: &Engine,
                      ov: &CameraPresentOverride,
                      pp: Option<&std::sync::Arc<CoveragePlan>>| {
            render_prepared_test_frame(engine, &plan, ov, pp, "shader-clock test view", w, h)
        };

        // Full-encode references at both clocks (ineligible ⇒ never touch the packet).
        let baseline_t0 = render(&engine, &ov0, None);
        let baseline_t1 = render(&engine, &ov1, None);
        // Sanity: clouds actually animates, so the two clocks differ on screen —
        // otherwise a freeze would be invisible and this test vacuous.
        assert_ne!(
            baseline_t0, baseline_t1,
            "clouds must animate between t=0 and t=6"
        );

        // Build the packet at t0 (miss), then render at t1 with the SAME plan Arc.
        let _ = render(&engine, &ov0, Some(&plan));
        let misses_after_build = engine.prepared_frame_stats().misses;
        let packet_t1 = render(&engine, &ov1, Some(&plan));

        assert_eq!(
            engine.prepared_frame_stats().misses,
            misses_after_build + 1,
            "a clock change on a clock-bearing plan must MISS the packet (re-encode)"
        );
        assert_eq!(
            packet_t1, baseline_t1,
            "the re-threaded clock must reach the GPU — packet frame == fresh encode at t1"
        );
        assert_ne!(
            packet_t1, baseline_t0,
            "the shader must NOT be frozen at the packet's build clock"
        );
        // Clock binding can change without changing the shader module or its
        // packed value at t=0. The new packet must carry the new clock policy.
        let mut unclocked = (*plan).clone();
        for item in &mut unclocked.items {
            if let DrawItem::Body { paint: BodyPaint::Shader(body), .. } = item {
                body.clock_offsets = None;
            }
        }
        let unclocked = std::sync::Arc::new(unclocked);
        for offscreen in [false, true] {
            engine.prepared_frame.borrow_mut().take();
            for current in [&unclocked, &plan] {
                if offscreen {
                    engine.render_retained_plan_to_texture(current).unwrap();
                } else {
                    render_prepared_test_frame(&engine, current, &ov0, Some(current), "clock transition", w, h);
                }
                assert_eq!(engine.prepared_frame.borrow().as_ref().unwrap().has_clock_fill,
                    std::sync::Arc::ptr_eq(current, &plan), "offscreen {offscreen}");
            }
        }
    }

    #[test]
    fn vp_camera_translates() {
        let vp = view_projection(100.0, 900.0, 1.0, 1280.0, 800.0);
        let apply = |m: &[f32; 16], x: f32, y: f32| -> (f32, f32) {
            (m[0] * x + m[4] * y + m[12], m[1] * x + m[5] * y + m[13])
        };
        let (x, y) = apply(&vp, 100.0, 900.0);
        assert!((x + 1.0).abs() < 1e-5 && (y - 1.0).abs() < 1e-5);
    }

    fn editor_camera_opts() -> RenderOptions {
        RenderOptions {
            sample_count: 4,
            best_effort_text: true,
            text_masks_available: Some(true),
            ..Default::default()
        }
    }

    // solid/gradient rect & ellipse bodies route to the analytic SDF
    // fast path, so plan-structure assertions accept a tessellated `Body` or an
    // analytic `SolidShapeSdf` / `GradientShapeSdf` interchangeably.
    fn body_node_id(item: &DrawItem) -> Option<&str> {
        match item {
            DrawItem::Body { node_id, .. }
            | DrawItem::SolidShapeSdf { node_id, .. }
            | DrawItem::GradientShapeSdf { node_id, .. } => Some(node_id.as_str()),
            _ => None,
        }
    }
    fn body_is(item: &DrawItem, id: &str) -> bool {
        body_node_id(item) == Some(id)
    }
    fn body_z(item: &DrawItem) -> Option<f64> {
        match item {
            DrawItem::Body { z, .. }
            | DrawItem::SolidShapeSdf { z, .. }
            | DrawItem::GradientShapeSdf { z, .. } => Some(*z),
            _ => None,
        }
    }
    fn body_model_matrix(item: &DrawItem) -> Option<[f32; 16]> {
        match item {
            DrawItem::Body { instance, .. } => Some(instance.model_matrix),
            DrawItem::SolidShapeSdf { instance, .. } => Some(instance.model_matrix),
            DrawItem::GradientShapeSdf { instance, .. } => Some(instance.model_matrix),
            _ => None,
        }
    }
    fn body_fill_alpha(item: &DrawItem) -> Option<f32> {
        match item {
            DrawItem::Body { instance, .. } => Some(instance.fill_color[3]),
            DrawItem::SolidShapeSdf { instance, .. } => Some(instance.color[3]),
            DrawItem::GradientShapeSdf { instance, .. } => Some(instance.color[3]),
            _ => None,
        }
    }

    fn camera_fixture_scene() -> moda_scene::Scene {
        let json = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/stress-500.json"
        ))
        .expect("residency fixture exists");
        let value: moda_scene::value::Value = moda_scene::value::from_str(&json).expect("fixture parses");
        moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap()
    }

    /// Live-viewport: `build_camera_plan` combines the N visible pages
    /// into ONE screen-sized plan with the editor-camera projection — the
    /// crispness fix (target = viewport × dpr, NOT page-at-zoom). CPU-only (no
    /// GPU): pure plan-building.
    #[test]
    fn camera_plan_combines_pages_and_sets_projection() {
        let scene = camera_fixture_scene();
        let n_pages = scene.doc().pages.len();
        assert!(n_pages >= 2, "fixture must have >= 2 pages");
        let opts = editor_camera_opts();
        let cam = test_camera([100.0, 50.0], 2.0, [800.0, 600.0], 2.0, [0.1, 0.2, 0.3]);
        let indices: Vec<usize> = (0..n_pages).collect();
        let (plan, _w) = build_camera_plan(&scene, &indices, &cam, &opts)
            .expect("camera plan builds")
            .expect("no backdrop page -> Some");

        // Target is the viewport × dpr (screen-sized), not the page-at-zoom.
        assert_eq!(plan.width_px, 1600);
        assert_eq!(plan.height_px, 1200);
        // The editor camera override + backdrop clear ride the plan.
        assert_eq!(
            plan.camera_frame,
            Some(crate::seams::CameraFrameView {
                camera_x: 100.0,
                camera_y: 50.0,
                zoom: 2.0,
                viewport_w: 800.0,
                viewport_h: 600.0,
            })
        );
        assert_eq!(plan.clear_rgba, Some([0.1, 0.2, 0.3, 1.0]));

        // Every page's fill is folded into the single combined plan.
        let page_fills = plan
            .items
            .iter()
            .filter(|i| matches!(i, DrawItem::PageFill { .. }))
            .count();
        assert_eq!(page_fills, n_pages, "one page-fill per combined page");

        // A host viewport cull renders only the requested pages.
        let (one, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .expect("single-page camera plan builds")
            .expect("Some");
        let one_fills = one
            .items
            .iter()
            .filter(|i| matches!(i, DrawItem::PageFill { .. }))
            .count();
        assert_eq!(one_fills, 1, "culled plan holds only the visible page");

        // Combined mask ids stay in range of the combined table (rebasing is
        // sound — every referenced mask exists).
        let mask_count = plan.masks.len() as u64;
        for item in &plan.items {
            if let DrawItem::Body { mask: Some(m), .. } = item {
                assert!(m.mask_id < mask_count, "item mask id within combined table");
            }

        }
        for m in &plan.masks {
            if let Some(a) = m.ancestor {
                assert!(
                    (a as u64) < mask_count,
                    "mask ancestor within combined table"
                );
            }
        }
    }

    /// A backdrop page composites INSIDE the combined camera
    /// frame (no `Ok(None)` snapshot fallback) — the plan is `Some`, renders at
    /// sample_count 1, and carries a `BackdropSplit` projected in CAMERA target
    /// pixels, so the split scales with zoom (a per-page-target split would be
    /// zoom-invariant). CPU-only (plan building, no GPU).
    #[test]
    fn camera_plan_renders_backdrop_in_frame() {
        let doc = moda_scene::value::json!({
            "nodes": [
                { "id": "bg", "type": "rectangle", "x": 0, "y": 0, "width": 400, "height": 300, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#cccccc"}}] },
                { "id": "r", "type": "rectangle", "x": 100, "y": 80, "width": 200, "height": 150,
                  "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ["bg", "r"] }],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        assert!(crate::backdrop::page_has_active_backdrop(&scene, "p1"));
        let opts = editor_camera_opts(); // sample_count 4
        let cam = |zoom: f64| test_camera([0.0, 0.0], zoom, [800.0, 600.0], 2.0, [0.1, 0.1, 0.1]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam(1.0), &opts)
            .expect("camera plan builds")
            .expect("backdrop renders in-frame (Some), not a snapshot fallback");
        // Backdrop forces single-sample even though opts asked for 4 (the split's
        // copy source must be the resolved single-sample target).
        assert_eq!(
            plan.sample_count, 1,
            "backdrop camera frame is sample_count 1"
        );
        let splits = plan
            .items
            .iter()
            .filter(|i| matches!(i, DrawItem::BackdropSplit(_)))
            .count();
        assert_eq!(
            splits, 1,
            "the backdrop node emits one split in the camera plan"
        );

        // The split is in CAMERA target pixels: at 2x zoom the projected node rect
        // is ~twice as large as at 1x (a per-page-target split would not change).
        let node_rect_at = |zoom: f64| -> [f32; 4] {
            let (p, _w) = build_camera_plan(&scene, &[0], &cam(zoom), &opts)
                .unwrap()
                .unwrap();
            p.items
                .iter()
                .find_map(|i| match i {
                    DrawItem::BackdropSplit(s) => Some(s.node_rect_px),
                    _ => None,
                })
                .expect("split present")
        };
        let r1 = node_rect_at(1.0);
        let r2 = node_rect_at(2.0);
        assert!(
            (r2[2] - r1[2] * 2.0).abs() < 1.0,
            "node rect width scales with zoom (camera projection): {} vs {}",
            r2[2],
            r1[2]
        );
        assert!(
            (r2[3] - r1[3] * 2.0).abs() < 1.0,
            "node rect height scales with zoom (camera projection): {} vs {}",
            r2[3],
            r1[3]
        );
    }

    /// Exact reported repro: a 1080×1080 page with one on-page rect and
    /// one rect on the freeform pasteboard below it. The editor resolves the
    /// canvas to `freeform` and auto-tracks the off-page rect into
    /// `canvasRootNodeIds`; this is the shape the WASM feed carries. The off-page
    /// rect must be present in the camera plan (it was invisible).
    #[test]
    fn camera_plan_renders_reported_repro_off_page_rect() {
        let doc = moda_scene::value::json!({
            // The editor store resolves this canvas to freeform (pages carry
            // freeformX/Y) and auto-tracks the parentless node — the live feed.
            "layoutMode": "freeform",
            "nodes": [
                { "id": "rect-on-page", "type": "rectangle", "x": 236.5, "y": 376.8,
                  "width": 150.1, "height": 156.5, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}] },
                { "id": "rect-off-page", "type": "rectangle", "x": 452.5, "y": 1213.5,
                  "width": 279.8, "height": 246.8, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}] },
            ],
            "pages": [{ "id": "page-1", "name": "Page 1", "width": 1080, "height": 1080,
                        "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ["rect-on-page"], "freeformX": 0, "freeformY": 0 }],
            "canvasRootNodeIds": ["rect-off-page"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        // A viewport covering the page AND the off-page rect below it.
        let cam = test_camera([0.0, 0.0], 1.0, [1080.0, 1600.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .expect("camera plan builds")
            .expect("no backdrop -> Some");
        assert!(
            plan.items.iter().any(|i| body_is(i, "rect-off-page")),
            "the off-page freeform rect must render in the WASM camera frame (ENG-3472)"
        );
    }

    /// Freeform canvas-root (off-page) nodes render in the combined
    /// camera frame. They belong to no page, so the per-page loop never emitted
    /// them — the bug. The body must now be present (was invisible) and painted
    /// AFTER the page content, matching the WebGPU orchestrator's floating-on-top
    /// freeform pass. CPU-only (plan building, no GPU).
    #[test]
    fn camera_plan_renders_freeform_canvas_root_nodes() {
        let doc = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                { "id": "onpage", "type": "rectangle", "x": 10, "y": 10, "width": 50, "height": 50, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}] },
                { "id": "offpage", "type": "rectangle", "x": 900, "y": 40, "width": 80, "height": 80, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": ["onpage"], "freeformX": 0.0, "freeformY": 0.0 }],
            "canvasRootNodeIds": ["offpage"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([0.0, 0.0], 1.0, [1200.0, 600.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .expect("camera plan builds")
            .expect("no backdrop -> Some");

        let body_index =
            |id: &str| -> Option<usize> { plan.items.iter().position(|i| body_is(i, id)) };
        let onpage_i = body_index("onpage").expect("on-page node renders");
        let offpage_i =
            body_index("offpage").expect("off-page canvas-root node renders (ENG-3472)");
        assert!(
            offpage_i > onpage_i,
            "floating node paints after page content (on top), matching WebGPU"
        );
    }

    /// The freeform region-export readback (`render_camera_frame_to_bytes`,
    /// transparent = true) threads `opts.transparent` into the camera plan — the
    /// plan flag AND the clear's alpha. Live camera lanes pass `transparent = false`
    /// and must stay byte-identical (opaque clear alpha 1.0). CPU-only (plan build).
    #[test]
    fn camera_plan_threads_transparent_flag_and_clear_alpha() {
        let doc = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                { "id": "floata", "type": "ellipse", "x": 900, "y": 40, "width": 80, "height": 80, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": [], "freeformX": 0.0, "freeformY": 0.0 }],
            "canvasRootNodeIds": ["floata"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let cam = test_camera([900.0, 40.0], 1.0, [80.0, 80.0], 2.0, [0.5, 0.5, 0.5]);

        // Opaque (every live camera lane + the eyedropper) — unchanged.
        let opaque_opts = editor_camera_opts();
        assert!(!opaque_opts.transparent);
        let (opaque, _w) = build_camera_plan(&scene, &[0], &cam, &opaque_opts)
            .expect("camera plan builds")
            .expect("no backdrop -> Some");
        assert!(!opaque.transparent, "live/eyedropper plan stays opaque");
        assert_eq!(
            opaque.clear_rgba,
            Some([0.5, 0.5, 0.5, 1.0]),
            "opaque clear keeps alpha 1.0 (byte-identical to pre-ENG-3836)"
        );

        // Transparent export readback: plan carries the flag + an alpha-0 clear so
        // the PNG is transparent around the floating shape.
        let transparent_opts = RenderOptions {
            transparent: true,
            ..editor_camera_opts()
        };
        let (transparent, _w) = build_camera_plan(&scene, &[0], &cam, &transparent_opts)
            .expect("camera plan builds")
            .expect("no backdrop -> Some");
        assert!(transparent.transparent, "export plan carries transparent");
        assert_eq!(
            transparent.clear_rgba,
            Some([0.0, 0.0, 0.0, 0.0]),
            "transparent export clears to a fully-zero backdrop (no edge halo)"
        );
    }

    /// Coupling guard: `present_override` (the camera-variant fields the
    /// offscreen readback re-applies on top of the plan) hardcodes an OPAQUE clear
    /// alpha, because the LIVE swapchain is `CompositeAlphaMode::Opaque` and the
    /// executor prefers the override's clear over the plan's. So a transparent region
    /// export CANNOT rely on the plan's `clear_rgba` alone — `render_camera_frame_to_bytes`
    /// must force `cam_override.clear_rgba = [0,0,0,0]` itself. This test pins the
    /// opaque default so that coupling stays visible: if `present_override` ever
    /// started honoring transparency, the wasm-side override could be dropped.
    #[test]
    fn present_override_clear_is_opaque_regardless_of_backdrop() {
        let cam = test_camera([5.0, 6.0], 2.0, [80.0, 60.0], 2.0, [0.3, 0.4, 0.5]);
        let over = cam.present_override(0.0);
        assert_eq!(
            over.clear_rgba,
            [0.3, 0.4, 0.5, 1.0],
            "present_override always emits an opaque clear (alpha 1.0); the transparent \
             region-export readback overrides this to [0,0,0,0] itself"
        );
    }

    /// Parity: the freeform region export passes an EMPTY page-index list
    /// so ONLY the backdrop + canvas-root (floating) nodes render — matching the
    /// WebGPU coordinator's canvas-region pass (`scope:'canvas'`), which never draws
    /// page content. The floating node is present; the on-page node is absent.
    #[test]
    fn camera_plan_with_no_pages_renders_only_canvas_root() {
        let doc = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                { "id": "onpage", "type": "rectangle", "x": 10, "y": 10, "width": 50, "height": 50, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}] },
                { "id": "floata", "type": "rectangle", "x": 900, "y": 40, "width": 80, "height": 80, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": ["onpage"], "freeformX": 0.0, "freeformY": 0.0 }],
            "canvasRootNodeIds": ["floata"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([900.0, 40.0], 1.0, [80.0, 80.0], 1.0, [1.0, 1.0, 1.0]);
        // NO visible pages — the region export's page-index list.
        let (plan, _w) = build_camera_plan(&scene, &[], &cam, &opts)
            .expect("camera plan builds")
            .expect("no backdrop -> Some");
        assert!(
            plan.items.iter().any(|i| body_is(i, "floata")),
            "floating canvas-root node renders with no pages (scope:'canvas')"
        );
        assert!(
            !plan.items.iter().any(|i| body_is(i, "onpage")),
            "page content is excluded when no page indices are passed"
        );
    }

    /// Parity guard: outside freeform layout the WebGPU orchestrator
    /// never paints floating nodes (`WebGPURenderOrchestrator.ts:923-977`), so
    /// the camera plan must omit canvas-root nodes there too — even if
    /// `canvasRootNodeIds` is populated. Rendering them would be a NEW parity gap.
    #[test]
    fn camera_plan_omits_canvas_root_nodes_outside_freeform() {
        let doc = moda_scene::value::json!({
            "layoutMode": "vertical",
            "nodes": [
                { "id": "onpage", "type": "rectangle", "x": 10, "y": 10, "width": 50, "height": 50, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}] },
                { "id": "offpage", "type": "rectangle", "x": 900, "y": 40, "width": 80, "height": 80, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ["onpage"] }],
            "canvasRootNodeIds": ["offpage"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([0.0, 0.0], 1.0, [1200.0, 600.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .unwrap()
            .unwrap();
        assert!(
            plan.items.iter().any(|i| body_is(i, "onpage")),
            "on-page node still renders"
        );
        assert!(
            !plan.items.iter().any(|i| body_is(i, "offpage")),
            "canvas-root node omitted outside freeform (WebGPU parity)"
        );
    }

    /// A clip-masked off-page group exercises the canvas-root mask
    /// rebasing. The combined plan holds masks from BOTH a page and the canvas
    /// root, so the off-page child's mask id must be rebased above the page's —
    /// distinct owners ⇒ distinct ids. Without the `mask_base` shift both would
    /// collapse to local index 0 (the off-page child would clip to the PAGE's
    /// mask). Also covers a group at the canvas root (children recursion).
    #[test]
    fn camera_plan_rebases_masks_for_off_page_group() {
        let doc = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                // On-page clip-masked group.
                { "id": "pg", "type": "group", "x": 0, "y": 0, "children": ["pm", "pc"], "clipMaskNodeId": "pm" },
                { "id": "pm", "type": "rectangle", "x": 0, "y": 0, "width": 40, "height": 40, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}] },
                { "id": "pc", "type": "rectangle", "x": 5, "y": 5, "width": 50, "height": 50, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}] },
                // Off-page clip-masked group (canvas root).
                { "id": "og", "type": "group", "x": 900, "y": 0, "children": ["om", "oc"], "clipMaskNodeId": "om" },
                { "id": "om", "type": "rectangle", "x": 900, "y": 0, "width": 40, "height": 40, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000"}}] },
                { "id": "oc", "type": "rectangle", "x": 905, "y": 5, "width": 50, "height": 50, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": ["pg"], "freeformX": 0.0, "freeformY": 0.0 }],
            "canvasRootNodeIds": ["og"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([0.0, 0.0], 1.0, [1200.0, 600.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .unwrap()
            .unwrap();

        let mask_of = |id: &str| -> Option<u64> {
            plan.items.iter().find_map(|i| match i {
                DrawItem::Body {
                    node_id,
                    mask: Some(m),
                    ..
                } if node_id.as_str() == id => Some(m.mask_id),
                _ => None,
            })
        };
        let pc_mask = mask_of("pc").expect("on-page masked child carries a mask");
        let oc_mask = mask_of("oc").expect("off-page masked child renders and carries a mask");
        assert_ne!(
            pc_mask, oc_mask,
            "page and canvas-root masks are distinct (canvas-root mask_base rebasing applied)"
        );
        // Every mask reference stays within the combined table.
        let mask_count = plan.masks.len() as u64;
        assert!(
            oc_mask < mask_count,
            "off-page mask id within combined table"
        );
        assert!(pc_mask < mask_count, "page mask id within combined table");
    }

    /// Review: a floating node positioned OVER a page must win the
    /// depth test, not just be painted later — freeform lets floating nodes
    /// overlap pages (pages never overlap each other). Larger render-z = nearer
    /// (depth.rs), so the canvas-root z must exceed the page's. Without the
    /// `max_page_z` seat the floating node would restart at z 0 and be occluded
    /// by the page node it overlaps.
    #[test]
    fn camera_plan_floating_node_z_above_overlapping_page() {
        let doc = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                { "id": "pagerect", "type": "rectangle", "x": 100, "y": 100, "width": 200, "height": 200, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}] },
                { "id": "floating", "type": "rectangle", "x": 150, "y": 150, "width": 200, "height": 200, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": ["pagerect"], "freeformX": 0.0, "freeformY": 0.0 }],
            "canvasRootNodeIds": ["floating"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([0.0, 0.0], 1.0, [800.0, 600.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .unwrap()
            .unwrap();
        let z_of = |id: &str| -> Option<f64> {
            plan.items
                .iter()
                .find_map(|i| if body_is(i, id) { body_z(i) } else { None })
        };
        let page_z = z_of("pagerect").expect("page node renders");
        let floating_z = z_of("floating").expect("floating node renders");
        assert!(
            floating_z > page_z,
            "floating z ({floating_z}) must exceed overlapping page z ({page_z}) to win the depth test"
        );
    }

    /// Pages may overlap on the freeform board with document-order
    /// z-order — EVERY item of a later page (its `__page_bg_` background node
    /// included) must carry a strictly higher render-z than all of an earlier
    /// page's items, so the later page's whole stack wins the depth test in the
    /// overlap region. Matches the WebGPU orchestrator's global-across-pages z
    /// counter. CPU-only (plan building, no GPU).
    #[test]
    fn camera_plan_bands_z_across_overlapping_pages() {
        let doc = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                { "id": "r1", "type": "rectangle", "x": 100, "y": 100, "width": 200, "height": 200, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}] },
                { "id": "r2", "type": "rectangle", "x": 50, "y": 50, "width": 100, "height": 100, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}] },
                { "id": "floating", "type": "rectangle", "x": 500, "y": 400, "width": 120, "height": 90, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [
                { "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                  "nodeIds": ["r1"], "freeformX": 0.0, "freeformY": 0.0 },
                { "id": "p2", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#eeeeee"}}],
                  "nodeIds": ["r2"], "freeformX": 400.0, "freeformY": 300.0 },
            ],
            "canvasRootNodeIds": ["floating"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([0.0, 0.0], 1.0, [1600.0, 1200.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0, 1], &cam, &opts)
            .unwrap()
            .unwrap();
        let z_of = |id: &str| -> f64 {
            plan.items
                .iter()
                .find_map(|i| if body_is(i, id) { body_z(i) } else { None })
                .unwrap_or_else(|| panic!("item '{id}' renders with a z"))
        };
        let p1_max = f64::max(z_of("__page_bg_p1"), z_of("r1"));
        // Page 2's background node — the first item of its stack — must already
        // sit above EVERYTHING page 1 emitted, so its backdrop covers page 1's
        // content in the overlap.
        assert!(
            z_of("__page_bg_p2") > p1_max,
            "page 2 background z ({}) must exceed all of page 1's z (max {})",
            z_of("__page_bg_p2"),
            p1_max
        );
        assert!(
            z_of("r2") > z_of("__page_bg_p2"),
            "page 2 content above its background"
        );
        // First visible page still starts its band at the bottom, preserving the
        // single-page incremental seed shape.
        assert!(z_of("__page_bg_p1") < z_of("r1"));
        // Canvas-root floating nodes seat above the CUMULATIVE band — over every
        // page's content, not merely the last page's own 0-based end.
        let p2_max = f64::max(z_of("__page_bg_p2"), z_of("r2"));
        assert!(
            z_of("floating") > p2_max,
            "floating z ({}) must exceed the cumulative page band (max {})",
            z_of("floating"),
            p2_max
        );
    }

    /// Standard layouts never overlap page interiors, so each page
    /// must retain the full page-local render-z range. Threading the cumulative
    /// freeform cursor through vertical/horizontal/grid documents needlessly
    /// shares the render-z range (`depth::MAX_RENDER_Z`) across every visible page and
    /// refuses otherwise valid dense documents.
    #[test]
    fn camera_plan_resets_z_between_standard_layout_pages() {
        let doc = moda_scene::value::json!({
            "layoutMode": "vertical",
            "nodes": [
                { "id": "p1r1", "type": "rectangle", "x": 10, "y": 10, "width": 100, "height": 100, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}] },
                { "id": "p1r2", "type": "rectangle", "x": 120, "y": 10, "width": 100, "height": 100, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}] },
                { "id": "p2r1", "type": "rectangle", "x": 10, "y": 10, "width": 100, "height": 100, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}] },
            ],
            "pages": [
                { "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}], "nodeIds": ["p1r1", "p1r2"] },
                { "id": "p2", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#eeeeee"}}], "nodeIds": ["p2r1"] },
            ],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([0.0, 0.0], 1.0, [800.0, 1200.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0, 1], &cam, &opts)
            .unwrap()
            .unwrap();
        let z_of = |id: &str| -> f64 {
            plan.items
                .iter()
                .find_map(|i| if body_is(i, id) { body_z(i) } else { None })
                .unwrap_or_else(|| panic!("item '{id}' renders with a z"))
        };

        assert_eq!(
            z_of("__page_bg_p2"),
            z_of("__page_bg_p1"),
            "each standard-layout page starts at the same page-local render-z"
        );
        assert!(
            z_of("__page_bg_p2") < z_of("p1r2"),
            "page 2 must restart below page 1's final item instead of consuming a cumulative band"
        );
    }

    /// Review: a freeform canvas-root node with an active background
    /// blur must split the pass like a page backdrop node — emit one
    /// `BackdropSplit` and force the combined frame to single-sample. Previously
    /// the canvas-root path passed no backdrop context, so off-page glass/blur
    /// nodes rendered their body without the effect (a parity gap vs WebGPU).
    #[test]
    fn camera_plan_floating_backdrop_node_emits_split_and_forces_single_sample() {
        let doc = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                { "id": "pagebg", "type": "rectangle", "x": 0, "y": 0, "width": 400, "height": 300, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#cccccc"}}] },
                { "id": "glass", "type": "rectangle", "x": 1200, "y": 80, "width": 200, "height": 150,
                  "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 24 }] },
            ],
            "pages": [{ "id": "p1", "width": 800, "height": 600, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                        "nodeIds": ["pagebg"], "freeformX": 0.0, "freeformY": 0.0 }],
            "canvasRootNodeIds": ["glass"],
            "config": {},
        });
        let scene = moda_scene::Scene::load(&doc).unwrap();
        assert!(
            crate::backdrop::canvas_root_has_active_backdrop(&scene),
            "the off-page glass node is detected as an active backdrop"
        );
        let opts = editor_camera_opts(); // sample_count 4
        let cam = test_camera([0.0, 0.0], 1.0, [1600.0, 600.0], 1.0, [1.0, 1.0, 1.0]);
        let (plan, _w) = build_camera_plan(&scene, &[0], &cam, &opts)
            .unwrap()
            .unwrap();
        let splits = plan
            .items
            .iter()
            .filter(|i| matches!(i, DrawItem::BackdropSplit(_)))
            .count();
        assert_eq!(splits, 1, "the off-page backdrop node emits one split");
        assert_eq!(
            plan.sample_count, 1,
            "a canvas-root backdrop forces single-sample, like a page backdrop"
        );
    }

    #[test]
    fn camera_plan_projects_backdrop_and_page_geometry_to_viewport_pixels() {
        let value = moda_scene::value::json!({
            "layoutMode": "freeform",
            "nodes": [
                {
                    "id": "bg",
                    "type": "rectangle",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 320.0,
                    "height": 240.0,
                    "rotation": 0.0,
                    "scaleX": 1.0,
                    "scaleY": 1.0,
                    "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]
                },
                {
                    "id": "glass",
                    "type": "rectangle",
                    "x": 260.0,
                    "y": 40.0,
                    "width": 120.0,
                    "height": 80.0,
                    "rotation": 0.0,
                    "scaleX": 1.0,
                    "scaleY": 1.0,
                    "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "rgba(255,255,255,0.2)"}}],
                    "effects": [{ "id": "backdrop-0", "type": "background-blur", "radius": 12.0 }]
                },
                {
                    "id": "p2box",
                    "type": "rectangle",
                    "x": 10.0,
                    "y": 20.0,
                    "width": 80.0,
                    "height": 60.0,
                    "rotation": 0.0,
                    "scaleX": 1.0,
                    "scaleY": 1.0,
                    "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]
                }
            ],
            "pages": [
                {
                    "id": "p1",
                    "name": "P1",
                    "freeformX": 100.0,
                    "freeformY": 50.0,
                    "width": 320.0,
                    "height": 240.0,
                    "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                    "nodeIds": ["bg", "glass"]
                },
                {
                    "id": "p2",
                    "name": "P2",
                    "freeformX": 300.0,
                    "freeformY": 50.0,
                    "width": 320.0,
                    "height": 240.0,
                    "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                    "nodeIds": ["p2box"]
                }
            ]
        });
        let scene = moda_scene::Scene::load(&value).unwrap();
        let opts = editor_camera_opts();
        let cam = test_camera([100.0, 0.0], 2.0, [800.0, 600.0], 2.0, [1.0, 1.0, 1.0]);

        let (plan, _warnings) = build_camera_plan(&scene, &[0, 1], &cam, &opts)
            .expect("camera plan builds")
            .expect("backdrop pages render in-frame");

        assert_eq!(plan.width_px, 1600);
        assert_eq!(plan.height_px, 1200);
        assert_eq!(
            plan.sample_count, 1,
            "backdrop camera frames render single-sample"
        );

        let p2_model = plan
            .items
            .iter()
            .find_map(|item| {
                if body_is(item, "p2box") {
                    body_model_matrix(item)
                } else {
                    None
                }
            })
            .expect("page 2 body is included in the combined camera plan");
        let vp = view_projection(
            cam.camera_x,
            cam.camera_y,
            cam.zoom,
            cam.viewport_w,
            cam.viewport_h,
        );
        let project_target_px = |x: f64, y: f64| -> (f64, f64) {
            let clip_x = vp[0] as f64 * x + vp[4] as f64 * y + vp[12] as f64;
            let clip_y = vp[1] as f64 * x + vp[5] as f64 * y + vp[13] as f64;
            let clip_w = vp[3] as f64 * x + vp[7] as f64 * y + vp[15] as f64;
            let ndc_x = clip_x / clip_w;
            let ndc_y = clip_y / clip_w;
            (
                (ndc_x * 0.5 + 0.5) * plan.width_px as f64,
                (1.0 - (ndc_y * 0.5 + 0.5)) * plan.height_px as f64,
            )
        };
        let assert_close = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() < 1e-3,
                "expected {expected}, got {actual}"
            );
        };
        let (p2_x, p2_y) = project_target_px(p2_model[12] as f64, p2_model[13] as f64);
        assert_close(p2_x, 840.0);
        assert_close(p2_y, 280.0);

        let split = plan
            .items
            .iter()
            .find_map(|item| match item {
                DrawItem::BackdropSplit(split) if split.node_id == "glass" => Some(split),
                _ => None,
            })
            .expect("active backdrop emits a split inside the camera plan");

        // Node canvas rect: page (100,50) + node (260,40), size 120x80.
        // Camera target: (viewport 800x600) * dpr 2, zoom 2, camera_x 100.
        // So target px rect is [(360-100)*4, 90*4, 120*4, 80*4].
        assert_close(split.node_rect_px[0] as f64, 1040.0);
        assert_close(split.node_rect_px[1] as f64, 360.0);
        assert_close(split.node_rect_px[2] as f64, 480.0);
        assert_close(split.node_rect_px[3] as f64, 320.0);

        // The node crosses page p1's right edge at canvas x=420. If the camera
        // backdrop context used page-local dimensions instead of the camera
        // target and scene-space page clip, this visible rect would not clip at
        // projected target x=(420-100)*4 = 1280.
        assert_close(split.visible_rect_px[0] as f64, 1040.0);
        assert_close(split.visible_rect_px[1] as f64, 360.0);
        assert_close(split.visible_rect_px[2] as f64, 240.0);
        assert_close(split.visible_rect_px[3] as f64, 320.0);

        // Uniform radius 12 (σ = 6, reach 18) at zoom*dpr 4 expands the
        // clipped visible rect by 72px, then the page clip clamps the right
        // edge at x=1280. The production projection matrix is f32, so
        // floor/ceil quantization keeps the same one-pixel conservative
        // expansion as the runtime path.
        assert_eq!(split.region_px, [967, 288, 312, 465]);
    }

    /// The camera VP maps a canvas point to the screen exactly like
    /// `createViewProjectionMatrix(cameraX, cameraY, zoom, viewportW,
    /// viewportH)` — `(X,Y) → device px ((X−cx)·zoom·dpr, …)` after the target's
    /// NDC→[0,width_px] rasterizer step. The camera origin lands at NDC (−1,+1),
    /// and a canvas span of `viewport_w / zoom` fills the viewport (NDC +1).
    #[test]
    fn camera_vp_maps_canvas_point_to_device_pixel() {
        let vp = view_projection(100.0, 50.0, 2.0, 800.0, 600.0);
        let apply = |m: &[f32; 16], x: f32, y: f32| -> (f32, f32) {
            (m[0] * x + m[4] * y + m[12], m[1] * x + m[5] * y + m[13])
        };
        let (x0, y0) = apply(&vp, 100.0, 50.0);
        assert!((x0 + 1.0).abs() < 1e-5 && (y0 - 1.0).abs() < 1e-5);
        // cx + viewport_w / zoom = 100 + 800/2 = 500 → right edge (NDC +1).
        let (x1, _) = apply(&vp, 500.0, 50.0);
        assert!((x1 - 1.0).abs() < 1e-5);
        // cy + viewport_h / zoom = 50 + 600/2 = 350 → bottom edge (NDC −1).
        let (_, y1) = apply(&vp, 100.0, 350.0);
        assert!((y1 + 1.0).abs() < 1e-5);
    }

    /// Ancestor group opacity multiplies into a child's body fill
    /// alpha (CPU plan, no GPU), and a container's OWN fill is not double-dimmed.
    #[test]
    fn a_groups_opacity_fades_it_as_one_and_a_childless_containers_own_fill_once() {
        use moda_scene::value::json;
        let value = json!({
            "nodes": [
                {"id":"g","type":"group","x":0,"y":0,"opacity":0.5,"children":["c"]},
                {"id":"c","type":"rectangle","x":10,"y":10,"width":40,"height":40,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id":"ct","type":"container","x":100,"y":100,"width":40,"height":40,
                 "rotation":0,"scaleX":1,"scaleY":1,"opacity":0.5,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P","width":320.0,"height":240.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                 "nodeIds":["g","ct"]}
            ]
        });
        let scene = moda_scene::Scene::load(&value).unwrap();
        let (plan, _warnings) =
            build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");

        let alpha_of = |target: &str| -> Option<f32> {
            plan.items.iter().find_map(|it| {
                if body_is(it, target) {
                    body_fill_alpha(it)
                } else {
                    None
                }
            })
        };

        // Child of the 0.5 group: the group composites as one unit, so the child draws at its own 1.0 inside a
        // split that fades the whole group by 0.5 (`moda_scene::scene::is_opacity_group`).
        let c_alpha = alpha_of("c").expect("rect c body present");
        assert!((c_alpha - 1.0).abs() < 1e-6, "child alpha {c_alpha} != 1.0 inside its group's fade");
        let fades: Vec<f32> = plan
            .items
            .iter()
            .filter_map(|it| match it {
                DrawItem::BackdropSplit(s) => match s.op {
                    crate::seams::BackdropOp::StackEnd { mode: 0, opacity } => Some(opacity),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(fades, vec![0.5], "one fade split at the group's opacity");

        // The container's OWN fill is dimmed only by its own opacity (it has no
        // translucent ancestor) — not double-dimmed.
        let ct_alpha = alpha_of("ct").expect("container ct body present");
        assert!(
            (ct_alpha - 0.5).abs() < 1e-6,
            "container own-fill alpha {ct_alpha} != 0.5 (double-dimmed?)"
        );
    }

    /// A filtered group captures a faded child group: the child's fade pair goes into the capture, re-aimed at
    /// the capture's own pixels, and nothing is refused.
    #[test]
    fn a_filtered_group_captures_a_faded_child_group_with_its_fade() {
        use moda_scene::value::json;
        let rect = |id: &str, x: f64| json!({"id":id,"type":"rectangle","x":x,"y":120,"width":40,"height":40,
            "rotation":0,"scaleX":1,"scaleY":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}]});
        let value = json!({
            "nodes": [
                {"id":"outer","type":"group","x":0,"y":0,"children":["faded"],
                 "effects":[{"id":"blur","type":"layer-blur","radius":4}]},
                {"id":"faded","type":"group","x":0,"y":0,"opacity":0.5,"children":["a","b"]},
                rect("a", 200.0), rect("b", 220.0)
            ],
            "pages": [
                {"id":"p1","name":"P","width":320.0,"height":240.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                 "nodeIds":["outer"]}
            ]
        });
        let scene = moda_scene::Scene::load(&value).unwrap();
        let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
        assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
        // The page itself has no split left: the fade is inside the capture.
        assert!(!plan.items.iter().any(|it| matches!(it, DrawItem::BackdropSplit(_))));
        let group = plan.items.iter().find_map(|it| match it {
            DrawItem::LayerBlurComposite { node_id, group, .. } if node_id == "outer" => Some(group),
            _ => None,
        }).expect("the filtered group's composite");
        let splits: Vec<&crate::seams::BackdropSplit> = group.items.iter().filter_map(|it| match it {
            DrawItem::BackdropSplit(s) => Some(&**s),
            _ => None,
        }).collect();
        assert_eq!(splits.len(), 2, "the fade's pair");
        assert!(matches!(splits[0].op, crate::seams::BackdropOp::BlendBegin));
        assert!(matches!(splits[1].op, crate::seams::BackdropOp::StackEnd { mode: 0, opacity } if opacity == 0.5));
        // Both halves cover the children in CAPTURE pixels: inside the capture, and not at the page's x of 200.
        let [rx, ry, rw, rh] = group.region_canvas;
        let (zx, zy) = (group.width_px as f64 / rw, group.height_px as f64 / rh);
        for split in &splits {
            let [x, y, w, h] = split.region_px;
            assert_eq!(split.region_px, splits[0].region_px);
            assert!(x + w <= group.width_px && y + h <= group.height_px, "{:?} in {}x{}", split.region_px, group.width_px, group.height_px);
            let (left, top) = ((200.0 - rx) * zx, (120.0 - ry) * zy);
            let (right, bottom) = ((260.0 - rx) * zx, (160.0 - ry) * zy);
            assert!((x as f64) <= left && (y as f64) <= top && (x + w) as f64 >= right && (y + h) as f64 >= bottom,
                "{:?} covers [{left}, {top}, {right}, {bottom}]", split.region_px);
            assert!((x as f64) >= left - 3.0 && (y as f64) >= top - 3.0, "{:?} is the children's box, not the page's", split.region_px);
        }
    }

    /// A capture holds its root's whole picture: what it draws is not cut by the page, for a blurred leaf, for a
    /// shadowed group, and for a filtered child inside a filtered group (the group's shadow is cast by the child's
    /// filtered picture, off the page or on it). The composite on the page is still cut by the page.
    #[test]
    fn a_capture_holds_its_whole_picture_through_nested_captures() {
        use moda_scene::value::json;
        // A child that lies wholly left of the page, blurred, inside a group whose shadow lands on the page.
        let value = json!({
            "nodes": [
                {"id":"outer","type":"group","x":0,"y":0,"children":["child"],
                 "effects":[{"id":"shadow","type":"filter-shadow","color":"#000000","blur":0,"offset":{"x":120,"y":0}}]},
                {"id":"child","type":"rectangle","x":-80,"y":40,"width":20,"height":20,"rotation":0,"scaleX":1,"scaleY":1,
                 "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}],
                 "effects":[{"id":"blur","type":"layer-blur","radius":4}]}
            ],
            "pages": [
                {"id":"p1","name":"P","width":200.0,"height":200.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                 "nodeIds":["outer"]}
            ]
        });
        let scene = moda_scene::Scene::load(&value).unwrap();
        let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
        assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
        fn clip_of(item: &DrawItem) -> Option<[f32; 4]> {
            match item {
                DrawItem::Body { instance, .. } => Some(instance.page_clip_bounds),
                DrawItem::SolidShapeSdf { instance, .. } => Some(instance.page_clip_bounds),
                DrawItem::GradientShapeSdf { instance, .. } => Some(instance.page_clip_bounds),
                DrawItem::Text(batch) => Some(batch.page_clip_bounds),
                DrawItem::LayerBlurComposite { page_clip_bounds, .. } => Some(*page_clip_bounds),
                _ => None,
            }
        }
        let capture = |items: &[DrawItem], id: &str| items.iter().find_map(|it| match it {
            DrawItem::LayerBlurComposite { node_id, group, .. } if node_id == id => Some((group.clone(), clip_of(it).unwrap())),
            _ => None,
        });
        let (outer, outer_clip) = capture(&plan.items, "outer").expect("the group's capture");
        // On the page, the composite is cut by the page.
        assert_ne!(outer_clip, INFINITE_CLIP);
        // Inside the group's capture, the child's composite is not; nor is what the child's capture draws.
        let (inner, inner_clip) = capture(&outer.items, "child").expect("the child's capture, inside the group's");
        assert_eq!(inner_clip, INFINITE_CLIP);
        let drawn: Vec<[f32; 4]> = inner.items.iter().filter_map(clip_of).collect();
        assert!(!drawn.is_empty() && drawn.iter().all(|clip| *clip == INFINITE_CLIP), "{drawn:?}");
        // The child's capture covers the child where it is, off the page.
        let [x, _, w, _] = inner.region_canvas;
        assert!(x <= -80.0 && x + w >= -60.0, "{:?}", inner.region_canvas);
    }

    /// What clips a nested capture's composite inside the outer root stays when the outer capture takes it: the
    /// rounded corners of a clipping frame between the two. A rounded frame above the outer root does not clip
    /// inside its capture; it clips the outer composite.
    #[test]
    fn a_nested_composite_keeps_the_rounded_clip_it_has_inside_the_root() {
        use moda_scene::value::json;
        let plan_of = |frame_inside: bool| {
            let child = json!({"id":"child","type":"rectangle","x":20,"y":20,"width":60,"height":60,"rotation":0,"scaleX":1,"scaleY":1,
                "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}],
                "effects":[{"id":"blur","type":"layer-blur","radius":2}]});
            let blur = json!([{"id":"blur","type":"layer-blur","radius":4}]);
            let frame = |children: &str| json!({"id":"frame","type":"container","x":10,"y":10,"width":160,"height":160,
                "rotation":0,"scaleX":1,"scaleY":1,"clipContent":true,"cornerRadius":30,"children":[children]});
            let group = |children: &str| json!({"id":"outer","type":"group","x":0,"y":0,"children":[children],"effects":blur.clone()});
            let (nodes, top) = if frame_inside {
                (json!([group("frame"), frame("child"), child]), "outer")
            } else {
                (json!([frame("outer"), group("child"), child]), "frame")
            };
            let value = json!({
                "nodes": nodes,
                "pages": [
                    {"id":"p1","name":"P","width":200.0,"height":200.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                     "nodeIds":[top]}
                ]
            });
            let scene = moda_scene::Scene::load(&value).unwrap();
            build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds")
        };
        let rrect_in = |items: &[DrawItem], id: &str| items.iter().find_map(|it| match it {
            DrawItem::LayerBlurComposite { node_id, group, clip_rrect, .. } if node_id == id => Some((group.clone(), *clip_rrect)),
            _ => None,
        });
        for frame_inside in [true, false] {
            let (plan, warnings) = plan_of(frame_inside);
            assert!(warnings.skipped.is_empty(), "frame inside {frame_inside}: {:?}", warnings.skipped);
            let (outer, _) = rrect_in(&plan.items, "outer").expect("the outer capture");
            let (_, inner_rrect) = rrect_in(&outer.items, "child").expect("the child's capture, inside the outer one");
            // The frame's corners cut the child's composite when the frame is inside the outer root; above
            // it they cut the outer composite on the page, not anything inside its capture.
            assert_eq!(inner_rrect.is_some(), frame_inside, "frame inside {frame_inside}");
        }
    }

    /// One rule says what a filtered group's capture holds: a child's blend that is exact over the capture's
    /// transparent start (none, screen) is captured; one that is not (multiply's fixed-function state, a blend
    /// split) declines the filter, with or without a fading sibling in the group.
    #[test]
    fn a_filtered_group_declines_a_blend_its_capture_cannot_hold() {
        use moda_scene::value::json;
        for (mode, held) in [("normal", true), ("screen", true), ("multiply", false), ("overlay", false)] {
            for faded_sibling in [false, true] {
                let blended = json!({"id":"blended","type":"rectangle","x":150,"y":120,"width":40,"height":40,
                    "rotation":0,"scaleX":1,"scaleY":1,"blendMode":mode,
                    "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}]});
                let plain = json!({"id":"plain","type":"rectangle","x":60,"y":120,"width":40,"height":40,
                    "rotation":0,"scaleX":1,"scaleY":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#0000ff"}}]});
                let value = json!({
                    "nodes": [
                        {"id":"outer","type":"group","x":0,"y":0,"children":["sibling","blended"],
                         "effects":[{"id":"blur","type":"layer-blur","radius":4}]},
                        {"id":"sibling","type":"group","x":0,"y":0,"opacity":if faded_sibling { 0.5 } else { 1.0 },"children":["plain"]},
                        plain, blended
                    ],
                    "pages": [
                        {"id":"p1","name":"P","width":320.0,"height":240.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                         "nodeIds":["outer"]}
                    ]
                });
                let scene = moda_scene::Scene::load(&value).unwrap();
                let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
                let captured = plan.items.iter().any(|it| matches!(it, DrawItem::LayerBlurComposite { node_id, .. } if node_id == "outer"));
                let refused = warnings.skipped.iter().any(|w| w.starts_with("outer: subtree filter cannot isolate"));
                assert_eq!((captured, refused), (held, !held), "{mode}, faded sibling {faded_sibling}: {:?}", warnings.skipped);
            }
        }
    }

    /// The same rule in the leaf path: a filtered leaf captures its own paint when every layer draws source-over,
    /// and declines when one of its paint layers blends; the leaf's own blend mode is its composite's, not
    /// something the capture holds.
    #[test]
    fn a_filtered_leaf_declines_a_paint_layer_blend_its_capture_cannot_hold() {
        use moda_scene::value::json;
        let leaf = |node_blend: &str, layer_blend: &str| {
            let value = json!({
                "nodes": [
                    {"id":"leaf","type":"rectangle","x":100,"y":80,"width":80,"height":60,"rotation":0,"scaleX":1,"scaleY":1,
                     "blendMode":node_blend,
                     "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}},
                              {"id":"fill-1","blendMode":layer_blend,"paint":{"type":"solid","color":"#00ff00"}}],
                     "effects":[{"id":"blur","type":"layer-blur","radius":4}]}
                ],
                "pages": [
                    {"id":"p1","name":"P","width":320.0,"height":240.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                     "nodeIds":["leaf"]}
                ]
            });
            let scene = moda_scene::Scene::load(&value).unwrap();
            let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
            let captured = plan.items.iter().any(|it| matches!(it, DrawItem::LayerBlurComposite { node_id, .. } if node_id == "leaf"));
            let refused = warnings.skipped.iter().any(|w| w.starts_with("leaf: subtree filter cannot isolate"));
            (captured, refused, warnings.skipped)
        };
        for (node_blend, layer_blend, held) in [
            ("normal", "normal", true),
            // The leaf's own mode blends its composite; its bodies draw source-over inside the capture.
            ("multiply", "normal", true),
            ("normal", "multiply", false),
            ("normal", "overlay", false),
        ] {
            let (captured, refused, skipped) = leaf(node_blend, layer_blend);
            assert_eq!((captured, refused), (held, !held), "node {node_blend}, layer {layer_blend}: {skipped:?}");
        }
    }

    /// The rule looks through nested captures: a capture that holds only exact operations is held by the
    /// capture around it, and one that holds or is composited with something inexact is not, at any depth.
    #[test]
    fn capture_admission_looks_through_nested_captures() {
        use moda_scene::value::json;
        let plan_of = |mode: &str| {
            let child = json!({"id":"child","type":"rectangle","x":150,"y":120,"width":40,"height":40,
                "rotation":0,"scaleX":1,"scaleY":1,"blendMode":mode,
                "fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}]});
            let value = json!({
                "nodes": [
                    {"id":"outer","type":"group","x":0,"y":0,"children":["inner"],
                     "effects":[{"id":"blur","type":"layer-blur","radius":4}]},
                    {"id":"inner","type":"group","x":0,"y":0,"children":["child"],
                     "effects":[{"id":"blur","type":"layer-blur","radius":2}]},
                    child
                ],
                "pages": [
                    {"id":"p1","name":"P","width":320.0,"height":240.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                     "nodeIds":["outer"]}
                ]
            });
            let scene = moda_scene::Scene::load(&value).unwrap();
            build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds")
        };
        let composite = |items: &[DrawItem], id: &str| items.iter().find(|it| matches!(it, DrawItem::LayerBlurComposite { node_id, .. } if node_id == id)).cloned();
        let refused = |warnings: &PlanWarnings, id: &str| warnings.skipped.iter().any(|w| w.starts_with(&format!("{id}: subtree filter cannot isolate")));

        // Held at both depths: the outer capture holds the inner one, which holds a screen body.
        let (plan, warnings) = plan_of("screen");
        assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
        let outer = composite(&plan.items, "outer").expect("the outer capture");
        let DrawItem::LayerBlurComposite { group: outer_group, .. } = &outer else { unreachable!() };
        assert!(composite(&outer_group.items, "inner").is_some(), "the inner capture is inside the outer one");
        assert!(!capture_cannot_hold(std::slice::from_ref(&outer), false));

        // Declined at both depths: neither capture can hold the multiply body, wherever the boundary falls.
        let (plan, warnings) = plan_of("multiply");
        assert!(refused(&warnings, "inner") && refused(&warnings, "outer"), "{:?}", warnings.skipped);
        assert!(composite(&plan.items, "outer").is_none() && composite(&plan.items, "inner").is_none());

        // The same answer for a capture that already exists: an inexact body two captures down, or an inner
        // capture composited with multiply, makes the outermost one unholdable.
        let nested = |change: &dyn Fn(&mut crate::seams::LayerBlurGroup)| {
            let mut outer = outer.clone();
            let DrawItem::LayerBlurComposite { group, .. } = &mut outer else { unreachable!() };
            let inner = std::sync::Arc::make_mut(group).items.iter_mut()
                .find_map(|it| match it {
                    DrawItem::LayerBlurComposite { node_id, group, .. } if node_id == "inner" => Some(std::sync::Arc::make_mut(group)),
                    _ => None,
                }).expect("the inner capture");
            change(inner);
            outer
        };
        let deep_body = nested(&|inner| {
            let body = inner.items.iter_mut().find_map(|it| match it {
                DrawItem::Body { blend_mode, .. } => Some(blend_mode),
                _ => None,
            }).expect("the child's body");
            *body = BlendMode::Multiply;
        });
        assert!(capture_cannot_hold(std::slice::from_ref(&deep_body), false));
        let multiplied = nested(&|inner| inner.filters.composite_blend = crate::seams::CaptureBlend::Multiply);
        assert!(capture_cannot_hold(std::slice::from_ref(&multiplied), false));
        let screened = nested(&|inner| inner.filters.composite_blend = crate::seams::CaptureBlend::Screen);
        assert!(!capture_cannot_hold(std::slice::from_ref(&screened), false));
    }

    /// A capture over the size limits is cut to what can be read from the page before it is scaled down: a
    /// filtered group whose content runs far off the page is captured at full scale around the page, by the
    /// filters' read reach. A capture under the limits is not cut, wherever its content lies.
    #[test]
    fn an_oversize_capture_is_cut_to_what_the_page_can_read() {
        use moda_scene::value::json;
        let capture_of = |far: f64, effects: moda_scene::value::Value| {
            let rect = |id: &str, y: f64| json!({"id":id,"type":"rectangle","x":80,"y":y,"width":40,"height":40,
                "rotation":0,"scaleX":1,"scaleY":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}]});
            let value = json!({
                "nodes": [
                    {"id":"roll","type":"group","x":0,"y":0,"children":["above","on","below"],"effects":effects},
                    rect("above", -far), rect("on", 80.0), rect("below", far)
                ],
                "pages": [
                    {"id":"p1","name":"P","width":200.0,"height":200.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                     "nodeIds":["roll"]}
                ]
            });
            let scene = moda_scene::Scene::load(&value).unwrap();
            let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
            assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
            plan.items.iter().find_map(|it| match it {
                DrawItem::LayerBlurComposite { node_id, group, .. } if node_id == "roll" => Some((group.region_canvas, group.extent_canvas, group.height_px)),
                _ => None,
            }).expect("the group's capture")
        };
        let blur = || json!([{"id":"blur","type":"layer-blur","radius":4}]);
        let shadow = || json!([{"id":"shadow","type":"filter-shadow","color":"#000000","blur":4,"offset":{"x":0,"y":30}}]);
        // An oversize filtered group that nothing can read (all of it far above the page) is not captured at
        // all: kept whole it would be reduced, and a reduced capture's texels spill.
        for effects in [blur(), shadow()] {
            let rect = |id: &str, y: f64| json!({"id":id,"type":"rectangle","x":80,"y":y,"width":40,"height":40,
                "rotation":0,"scaleX":1,"scaleY":1,"fills":[{"id":"fill-0","paint":{"type":"solid","color":"#ff0000"}}]});
            let value = json!({
                "nodes": [
                    {"id":"far","type":"group","x":0,"y":0,"children":["a","b"],"effects":effects},
                    rect("a", -30000.0), rect("b", -10000.0)
                ],
                "pages": [
                    {"id":"p1","name":"P","width":200.0,"height":200.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                     "nodeIds":["far"]}
                ]
            });
            let scene = moda_scene::Scene::load(&value).unwrap();
            let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
            assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
            assert!(!plan.items.iter().any(|it| matches!(it, DrawItem::LayerBlurComposite { .. })), "nothing reads it, so nothing is captured");
        }
        for (name, effects, reach) in [("blur", blur(), 12.0), ("shadow", shadow(), 42.0)] {
            // Under the limits: the whole group, as before.
            let (region, extent, height_px) = capture_of(1000.0, effects.clone());
            let [_, y, _, h] = region;
            assert!(y <= -1000.0 && y + h >= 1040.0, "{name}: an uncut capture holds the whole group: y {y}, h {h}");
            assert_eq!(extent, region, "{name}: an uncut capture's picture is its region");
            assert!((height_px as f64 - h).abs() < 1.5, "{name}: at full scale: {height_px} px for {h}");
            // Over them (18,040 tall): the page and what the filter reads past it, at full scale.
            let ([_, y, _, h], extent, height_px) = capture_of(9000.0, effects);
            // What encloses the capture still measures the whole picture.
            assert!(extent[1] <= -9000.0 && extent[1] + extent[3] >= 9040.0, "{name}: a cut capture's picture is still the whole group: {extent:?}");
            assert!(y <= -reach && y + h >= 200.0 + reach, "{name}: the cut keeps the filter's reach around the page: y {y}, h {h}");
            assert!(y >= -(reach + 8.0) && y + h <= 200.0 + reach + 8.0, "{name}: and no more: y {y}, h {h}");
            assert!((height_px as f64 - h).abs() < 1.5, "{name}: at full scale: {height_px} px for {h}");
        }
    }

    /// The retired `instancer` kind (live-nodes.md §3.3/§14.5) is now an unknown
    /// node type: dropped at hydration, so planning a doc that still carries one
    /// does not crash and draws exactly what the same doc without it draws.
    #[test]
    fn unknown_instancer_node_plans_like_it_is_absent() {
        let instancer = moda_scene::value::json!({
            "id":"inst1","type":"instancer","x":10.0,"y":20.0,"width":40.0,"height":40.0,
            "rotation":0,"scale":1,"count":12,"distribution":{"kind":"ring","radius":30.0}
        });
        let rect = moda_scene::value::json!({
            "id":"r1","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
            "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#f00"}}]
        });
        let plan_of = |nodes: moda_scene::value::Value, ids: moda_scene::value::Value| {
            let scene = moda_scene::Scene::load(&moda_scene::value::json!({
                "nodes": nodes,
                "pages": [{"id":"p1","name":"P","width":100.0,"height":100.0,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#fff"}}],
                           "nodeIds": ids}]
            })).unwrap();
            build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds").0
        };
        let dump = |p: &CoveragePlan| format!("{:?}", p.items);
        let bare = dump(&plan_of(moda_scene::value::json!([rect.clone()]), moda_scene::value::json!(["r1"])));
        for (nodes, ids) in [
            (moda_scene::value::json!([rect.clone(), instancer.clone()]), moda_scene::value::json!(["r1", "inst1"])),
            (moda_scene::value::json!([instancer, rect]), moda_scene::value::json!(["inst1", "r1"])),
        ] {
            assert_eq!(dump(&plan_of(nodes, ids)), bare);
        }
    }

    /// Three-node page (two loose rects + a group holding a third) used by the
    /// `--isolate` tests. Transparent renders drop the page background, so the
    /// painted set is exactly the node bodies.
    fn isolate_fixture_scene() -> moda_scene::Scene {
        let json = r##"{
            "nodes": [
                {"id":"r1","type":"rectangle","x":10,"y":10,"width":40,"height":40,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id":"g1","type":"group","x":0,"y":0,"scaleX":1,"scaleY":1,"children":["r2"]},
                {"id":"r2","type":"rectangle","x":80,"y":80,"width":40,"height":40,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#00ff00"}}]},
                {"id":"r3","type":"rectangle","x":160,"y":160,"width":40,"height":40,
                 "rotation":0,"scaleX":1,"scaleY":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#0000ff"}}]}
            ],
            "pages": [
                {"id":"p1","name":"P1","width":320,"height":240,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ffffff"}}],
                 "nodeIds":["r1","g1","r2","r3"]}
            ],
            "config":{}, "variables":[], "canvasRootNodeIds":[],
            "layoutMode":"vertical", "category":null, "animations":null
        }"##;
        let value: moda_scene::value::Value = moda_scene::value::from_str(json).unwrap();
        moda_scene::Scene::load(moda_scene::extract_canvas_data(&value)).unwrap()
    }

    fn isolate_opts(isolate: &[&str]) -> RenderOptions {
        RenderOptions {
            transparent: true,
            isolate: isolate.iter().map(|s| s.to_string()).collect(),
            ..RenderOptions::default()
        }
    }

    fn painted_bodies(plan: &crate::seams::CoveragePlan) -> Vec<String> {
        plan.items
            .iter()
            .filter_map(|it| body_node_id(it).map(str::to_string))
            .collect()
    }

    /// THE byte-exactness property for the `--isolate` threading: an UNSET
    /// isolate list must produce the identical plan it produced before the
    /// option existed. The filter short-circuits on an empty list, so the only
    /// way this can regress is someone making isolation unconditional. Compared
    /// on the full debug rendering of every draw item, not just the node ids —
    /// a geometry or z change would show up here too.
    #[test]
    fn isolate_unset_leaves_the_plan_untouched() {
        let scene = isolate_fixture_scene();
        // `RenderOptions::default()` is literally the pre-`isolate` shape.
        let (baseline, _) =
            build_coverage_plan(&scene, 0, &RenderOptions::default()).expect("plan builds");
        let (with_empty, _) =
            build_coverage_plan(&scene, 0, &isolate_empty_opts()).expect("plan builds");
        assert_eq!(
            format!("{:?}", baseline.items),
            format!("{:?}", with_empty.items),
            "an empty --isolate must not perturb a single draw item"
        );
        assert_eq!(baseline.width_px, with_empty.width_px);
        assert_eq!(baseline.height_px, with_empty.height_px);
    }

    fn isolate_empty_opts() -> RenderOptions {
        RenderOptions {
            isolate: Vec::new(),
            ..RenderOptions::default()
        }
    }

    /// Isolating a leaf keeps only that leaf; isolating a GROUP keeps the
    /// group's descendants. This is what makes a per-element alpha plate carry
    /// its element and nothing else.
    #[test]
    fn isolate_filters_the_plan_to_the_requested_subtrees() {
        let scene = isolate_fixture_scene();

        let (all, _) = build_coverage_plan(&scene, 0, &isolate_opts(&[])).expect("plan builds");
        let all_bodies = painted_bodies(&all);
        for id in ["r1", "r2", "r3"] {
            assert!(all_bodies.iter().any(|b| b == id), "{id} paints unisolated");
        }

        // One leaf.
        let (only_r1, _) = build_coverage_plan(&scene, 0, &isolate_opts(&["r1"])).expect("plan");
        assert_eq!(painted_bodies(&only_r1), vec!["r1".to_string()]);

        // A group keeps its child, and only its child.
        let (only_g1, _) = build_coverage_plan(&scene, 0, &isolate_opts(&["g1"])).expect("plan");
        assert_eq!(
            painted_bodies(&only_g1),
            vec!["r2".to_string()],
            "isolating a group must keep its descendant leaves (and nothing else)"
        );

        // A union of two elements.
        let (r1_r3, _) =
            build_coverage_plan(&scene, 0, &isolate_opts(&["r1", "r3"])).expect("plan");
        assert_eq!(
            painted_bodies(&r1_r3),
            vec!["r1".to_string(), "r3".to_string()],
            "an isolate union keeps both elements in paint order"
        );

        // An id that matches nothing paints nothing — an honest empty plate,
        // not a full frame.
        let (nothing, _) = build_coverage_plan(&scene, 0, &isolate_opts(&["nope"])).expect("plan");
        assert!(painted_bodies(&nothing).is_empty());
    }

    /// The page background is NOT a node paint item, so `--isolate` must not
    /// filter it: the page fill is governed by `transparent` alone. Invisible on
    /// the plates lane (always transparent, where the fill is dropped anyway)
    /// and load-bearing for `batch --isolate`, whose lane is opaque — an
    /// isolated screenshot must keep the page's colored fill, not land the
    /// element on the bare clear color.
    #[test]
    fn isolate_never_filters_the_page_background_on_an_opaque_render() {
        let scene = isolate_fixture_scene();
        let page_bg_of = |plan: &crate::seams::CoveragePlan| {
            plan.items
                .iter()
                .filter_map(|it| body_node_id(it))
                .any(|id| moda_scene::types::is_page_bg_id(id))
        };

        let opaque_unisolated = RenderOptions::default();
        let (baseline, _) = build_coverage_plan(&scene, 0, &opaque_unisolated).expect("plan");
        assert!(
            page_bg_of(&baseline),
            "fixture sanity: an opaque page paints its background"
        );

        let opaque_isolated = RenderOptions {
            isolate: vec!["r1".to_string()],
            ..RenderOptions::default()
        };
        let (isolated, _) = build_coverage_plan(&scene, 0, &opaque_isolated).expect("plan");
        assert!(
            page_bg_of(&isolated),
            "an OPAQUE isolated render must keep the page background — the fill is \
             gated on `transparent`, never on isolation"
        );
        // ...and the isolation itself still applies to the real nodes.
        let painted: Vec<String> = painted_bodies(&isolated)
            .into_iter()
            .filter(|id| !moda_scene::types::is_page_bg_id(id))
            .collect();
        assert_eq!(painted, vec!["r1".to_string()]);

        // Transparent stays transparent: the background is dropped by the
        // `transparent` branch, not by isolation.
        let (transparent, _) =
            build_coverage_plan(&scene, 0, &isolate_opts(&["r1"])).expect("plan");
        assert!(!page_bg_of(&transparent));
    }

    /// `--isolate` and `--exclude` are complements: isolating one of two
    /// elements must paint what excluding the other paints. Pins that the new
    /// filter sits in the same place in the loop as the old one (same z
    /// bookkeeping, same skip semantics) rather than merely producing the same
    /// node set.
    #[test]
    fn isolate_is_the_complement_of_exclude() {
        let scene = isolate_fixture_scene();
        let isolated = build_coverage_plan(&scene, 0, &isolate_opts(&["r1", "g1"]))
            .expect("plan")
            .0;
        let excluded = build_coverage_plan(
            &scene,
            0,
            &RenderOptions {
                transparent: true,
                exclude: vec!["r3".to_string()],
                ..RenderOptions::default()
            },
        )
        .expect("plan")
        .0;
        assert_eq!(painted_bodies(&isolated), painted_bodies(&excluded));
    }

    /// Excluding a generated host (a chart) excludes the transient children it expands into — the PPTX exporter
    /// excludes a native chart from its slide pictures by the chart's id — but excluding a group leaves its real
    /// children alone.
    #[test]
    fn exclude_reaches_transient_children_only() {
        let mut scene = isolate_fixture_scene();
        let host = vec!["chart".to_string()];
        assert!(!is_excluded(&scene, "bar-0", &host));
        scene.set_transient_parent("bar-0", "chart");
        scene.set_transient_parent("bar-0-label", "bar-0");
        assert!(is_excluded(&scene, "bar-0", &host));
        assert!(is_excluded(&scene, "bar-0-label", &host));
        assert!(!is_excluded(&scene, "r2", &["g1".to_string()]), "a group's real child is not excluded");
        assert!(is_excluded(&scene, "g1", &["g1".to_string()]));
        assert!(!is_excluded(&scene, "bar-0", &[]));
    }
}

// ---------------------------------------------------------------------------
// M4e — image-backed procedural fill support
// ---------------------------------------------------------------------------

/// Whether a (resolved) module id is one of the image-backed fills — i.e. its
/// extraction-manifest packing meta declares a `primary_image` (the only
/// authority; DECISION §1.3).
fn module_is_image_backed(module_id: &str) -> bool {
    moda_shaders::runtime::packing_tables()
        .get(module_id)
        .map(|m| m.primary_image.is_some())
        .unwrap_or(false)
}

/// M4e: GPU cache for preprocessed shader-image textures
/// (`ShaderImageResourceManager` analog): straight-alpha `rgba8unorm` upload
/// (the production copy is `copyExternalImageToTexture` with
/// `premultipliedAlpha` unset), clamp-to-edge linear sampler, one bind group
/// per resource key on manifest BGL 42 (texture@0, sampler@1).
struct ShaderImageTextureCache {
    sampler: wgpu::Sampler,
    bind_groups: std::collections::HashMap<String, wgpu::BindGroup>,
    textures: std::collections::HashMap<String, wgpu::Texture>,
    /// Task #95 R2d (retained mode): per-key (bytes, last-touch tick).
    usage: std::collections::HashMap<String, (usize, u64)>,
    total_bytes: usize,
    tick: u64,
}

impl ShaderImageTextureCache {
    fn new(device: &wgpu::Device) -> Self {
        ShaderImageTextureCache {
            // `ShaderImageResourceManager`'s sampler: clamp-to-edge, linear.
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Shader effect image sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            bind_groups: Default::default(),
            textures: Default::default(),
            usage: Default::default(),
            total_bytes: 0,
            tick: 0,
        }
    }

    fn get(&self, key: &str) -> Option<&wgpu::BindGroup> {
        self.bind_groups.get(key)
    }

    /// Task #95 R2d: LRU clock + budget trim (retained mode only).
    fn begin_frame(&mut self) {
        self.tick += 1;
    }

    fn trim(&mut self, budget: usize) {
        if self.total_bytes <= budget {
            return;
        }
        let mut keys: Vec<(String, u64)> = self
            .usage
            .iter()
            .map(|(k, (_, t))| (k.clone(), *t))
            .collect();
        keys.sort_by_key(|(_, t)| *t);
        for (key, tick) in keys {
            if self.total_bytes <= budget || tick == self.tick {
                break;
            }
            if let Some((bytes, _)) = self.usage.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(bytes);
            }
            self.bind_groups.remove(&key);
            if let Some(texture) = self.textures.remove(&key) {
                crate::scratch::destroy(&texture);
            }
        }
    }

    fn clear_gpu_resources(&mut self) {
        self.bind_groups.clear();
        for (_, texture) in self.textures.drain() {
            crate::scratch::destroy(&texture);
        }
        self.usage.clear();
        self.total_bytes = 0;
    }

    fn bind_group(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        cache_key: &str,
        tex: &crate::seams::ShaderImageTexture,
    ) {
        if let Some(u) = self.usage.get_mut(cache_key) {
            u.1 = self.tick;
        }
        if self.bind_groups.contains_key(cache_key) {
            return;
        }
        let bytes = tex.width.max(1) as usize * tex.height.max(1) as usize * 4;
        self.usage.insert(cache_key.to_string(), (bytes, self.tick));
        self.total_bytes += bytes;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("shader image {}", tex.key)),
            size: wgpu::Extent3d {
                width: tex.width.max(1),
                height: tex.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        crate::gpu_profile::count_texture_write(tex.rgba_straight.len());
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tex.rgba_straight,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * tex.width.max(1)),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: tex.width.max(1),
                height: tex.height.max(1),
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shader effect image bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.textures.insert(cache_key.to_string(), texture);
        self.bind_groups.insert(cache_key.to_string(), bg);
    }
}

impl Drop for ShaderImageTextureCache {
    fn drop(&mut self) {
        self.clear_gpu_resources();
    }
}

/// The reach of a plain background blur split in target px (its widest radius, progressive start included), or
/// `None` for any other split (glass, pixelate, dither, blend and group splits are never simplified).
fn split_blur_reach_px(split: &crate::seams::BackdropSplit) -> Option<f64> {
    match &split.op {
        crate::seams::BackdropOp::Blur { radius_px, ramp, .. } => Some(
            ramp.map_or(*radius_px, |r| radius_px.max(r.start_radius_px)) as f64,
        ),
        _ => None,
    }
}

/// The part of a layer capture to draw and filter when its result is read only over `window` (canvas units): the
/// window grown by the filters' reach, three times over (a blur's ladder of passes reaches past its σ) and by 8
/// texels, cut to the capture. Its texels there are drawn as the whole capture's are (same texture, same
/// projection), and every texel of the result a target pixel reads depends on texels there only. `None` when that is
/// most of the capture. Returns the part in target px of the capture and in canvas units.
fn capture_clip(group: &crate::seams::LayerBlurGroup, window: [f64; 4]) -> Option<([u32; 4], [f64; 4])> {
    let [rx, ry, rw, rh] = group.region_canvas;
    let (w, h) = (group.width_px, group.height_px);
    if !(rw > 0.0 && rh > 0.0) || w == 0 || h == 0 || !window.iter().all(|v| v.is_finite()) {
        return None;
    }
    let (sx, sy) = (f64::from(w) / rw, f64::from(h) / rh);
    let margin = 3.0 * group.reach_canvas.max(0.0) + 8.0 / sx.min(sy);
    if !margin.is_finite() {
        return None;
    }
    let x0 = (window[0] - margin).max(rx);
    let y0 = (window[1] - margin).max(ry);
    let x1 = (window[0] + window[2] + margin).min(rx + rw);
    let y1 = (window[1] + window[3] + margin).min(ry + rh);
    if !(x1 > x0 && y1 > y0) {
        return None;
    }
    let tx0 = (((x0 - rx) * sx).floor().max(0.0) as u32).min(w);
    let ty0 = (((y0 - ry) * sy).floor().max(0.0) as u32).min(h);
    let tx1 = (((x1 - rx) * sx).ceil().max(0.0) as u32).min(w);
    let ty1 = (((y1 - ry) * sy).ceil().max(0.0) as u32).min(h);
    if tx1 <= tx0 || ty1 <= ty0 {
        return None;
    }
    let part = u64::from(tx1 - tx0) * u64::from(ty1 - ty0);
    if part * 10 >= u64::from(w) * u64::from(h) * 9 {
        return None;
    }
    Some(([tx0, ty0, tx1 - tx0, ty1 - ty0], [x0, y0, x1 - x0, y1 - y0]))
}

#[cfg(test)]
mod capture_clip_tests {
    fn group(region: [f64; 4], w: u32, h: u32, reach: f64) -> crate::seams::LayerBlurGroup {
        crate::seams::LayerBlurGroup {
            items: Vec::new(),
            region_canvas: region,
            extent_canvas: region,
            width_px: w,
            height_px: h,
            reach_canvas: reach,
            filters: Default::default(),
            cache_key: None,
            identity_key: None,
            coverage: Vec::new(),
            masks: Vec::new(),
        }
    }

    #[test]
    fn a_capture_shown_in_part_is_cut_to_that_part_grown_by_its_reach() {
        // A 4000 × 3000 canvas capture at 0.5 texel per unit, shown over 1000 × 500 of it: the part shown, grown by
        // 3 × reach + 8 texels (16 canvas units), in texels.
        let g = group([0.0, 0.0, 4000.0, 3000.0], 2000, 1500, 10.0);
        let (px, canvas) = super::capture_clip(&g, [1000.0, 1000.0, 1000.0, 500.0]).expect("cut");
        assert_eq!(canvas, [954.0, 954.0, 1092.0, 592.0]);
        assert_eq!(px, [477, 477, 546, 296]);
        // At an edge it stays inside the capture.
        let (px, _) = super::capture_clip(&g, [-500.0, -500.0, 1000.0, 1000.0]).expect("cut");
        assert_eq!(&px[..2], &[0, 0]);
        // Shown nearly whole, or not at all: drawn whole.
        assert!(super::capture_clip(&g, [0.0, 0.0, 3900.0, 3000.0]).is_none());
        assert!(super::capture_clip(&g, [9000.0, 9000.0, 10.0, 10.0]).is_none());
    }
}

#[cfg(test)]
mod deferred_mask_tests {
    use super::*;

    #[test]
    fn every_plan_consumer_uses_the_same_refusal_policy() {
        assert!(PlanWarnings::default().ensure_drawn().is_ok());
        assert!(PlanWarnings { skipped: vec!["path (stroke): empty geometry".into()], dropped: vec![] }.ensure_drawn().is_ok());
        for warnings in [
            PlanWarnings { skipped: vec!["mask: layer blur sigma exceeds limit".into()], dropped: vec![] },
            PlanWarnings { skipped: vec![], dropped: vec!["path: fill dropped".into()] },
            PlanWarnings { skipped: vec![], dropped: vec!["path (fill): empty geometry".into()] },
        ] {
            assert!(warnings.ensure_drawn().is_err());
        }
    }

    #[test]
    #[ignore = "requires a real GPU"]
    fn audit_rendered_mask_does_not_discard_planning_refusals() {
        let scene = Scene::load(&moda_scene::value::json!({
            "nodes": [
                {"id":"g","type":"group","children":["mask","content"],"clipMaskNodeId":"mask"},
                {"id":"mask","type":"rectangle","width":100,"height":100,"maskType":"luminance",
                 "fills":[{"id":"f","paint":{"type":"solid","color":"#ffffff"}}],
                 "effects":[{"id":"blur","type":"layer-blur","radius":1000}]},
                {"id":"content","type":"rectangle","width":100,"height":100,
                 "fills":[{"id":"f","paint":{"type":"solid","color":"#ff0000"}}]}
            ],
            "pages":[{"id":"p","width":100,"height":100,"nodeIds":["g"]}]
        })).unwrap();
        let (plan, warnings) = build_coverage_plan(&scene, 0, &RenderOptions::default()).unwrap();
        assert!(warnings.is_empty(), "the refusal is deferred: {warnings:?}");
        assert!(plan.masks.iter().any(|m| m.image.as_ref().is_some_and(|i| i.rendered.is_some())));
        let engine = Engine::new(crate::GpuContext::new(crate::BackendChoice::Auto).expect("GPU adapter")).unwrap();
        let error = engine.render_plan(&plan).err().expect("a refused mask must fail the enclosing render");
        let error = format!("{error:#}");
        assert!(error.contains("mask") && error.contains("blur"), "{error}");
        let mut valid = scene.doc().clone();
        valid.nodes.iter_mut().find(|node| node.id() == "mask").unwrap()
            .fields_mut().insert("effects".into(), moda_scene::value::json!([{"id":"blur","type":"layer-blur","radius":4}]));
        let (plan, warnings) = build_coverage_plan(&Scene::from_document(valid).unwrap(), 0, &RenderOptions::default()).unwrap();
        warnings.ensure_drawn().unwrap();
        assert!(engine.render_plan(&plan).is_ok(), "a supported mask must still render");
    }
}

#[cfg(test)]
mod retained_teardown_tests {
    use super::*;

    thread_local! {
        static OWNER: std::cell::RefCell<Option<RetainedFrame>> = const { std::cell::RefCell::new(None) };
    }

    #[test]
    #[ignore = "requires a real GPU"]
    fn retained_buffers_and_textures_drop_after_scratch_tls() {
        // Keep device shutdown outside Windows' loader-locked TLS teardown. This
        // regression exercises retained resource cleanup, not last-device shutdown.
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto)
            .expect("GPU test requires a working adapter");
        eprintln!("TLS teardown adapter: {:?}", gpu.adapter_info);
        let worker_gpu = gpu.clone();
        std::thread::spawn(move || {
            let gpu = worker_gpu;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("TLS teardown buffer"), size: 16,
                usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false,
            });
            let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                }],
            });
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
            });
            let texture = |format| gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("TLS teardown texture"),
                size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
            });
            OWNER.with(|owner| *owner.borrow_mut() = Some(RetainedFrame {
                frame_uniform: Some((buffer, group)),
                targets: vec![TargetEntry {
                    key: (4, 4, 1), target: texture(wgpu::TextureFormat::Rgba8Unorm),
                    msaa: None, depth: texture(wgpu::TextureFormat::Depth24Plus), tick: 0,
                }],
                tick: 0,
                bufs: Default::default(),
            }));
            // Initialize scratch after the retained owner, as an embedded canvas does.
            struct InitScratch;
            impl crate::scratch::Destroy for InitScratch { fn destroy_now(&self) {} }
            crate::scratch::destroy(&InitScratch);
        }).join().unwrap();
        drop(gpu);
    }
}
