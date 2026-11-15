//! moda-engine — the moda-render engine core (PLAN.md M1).
//!
//! Scene JSON → (moda-scene hydration/display list) → `frame::CoveragePlan`
//! → headless wgpu execution → readback → encoded bytes.
//!
//! Module ownership after M1 (PLAN §0 rule 4):
//! - M1 (this milestone): lib.rs, seams.rs (FROZEN), device.rs, frame.rs,
//!   depth.rs, readback.rs, encode.rs, color.rs, shape_to_path.rs
//! - M2: tess/** (replaces the placeholder wholesale)
//! - M3a: text_draw.rs · M3b: paint/** · M3d: effects/** · M3e: backdrop/**
//!
//! See SEAMS.md for the frozen cross-track contract.

pub mod agent_fill;
pub mod backdrop;
pub mod boolean_operands;
pub mod brush_artwork;
pub(crate) mod camera_slot;
pub mod color;
pub mod convert;
pub mod cull;
/// vork: moda-text-draw's copy (the same file); see PROVENANCE.md.
pub use moda_text_draw::depth;
pub mod device;
pub mod effects;
pub mod encode;
pub mod export_job;
pub mod export_scale;
pub mod external_texture;
pub mod frame;
pub mod gpu_profile;
pub mod isolated;
pub mod layer_blur;
pub mod motion_capture;
pub mod motion_cache;
pub mod nv12;
pub mod page_cache;
pub mod paint;
pub mod pixel;
pub mod prof;
pub mod mesh_paint;
pub mod pattern_paint;
pub mod readback;
pub mod residency;
pub mod scratch;
pub mod seams;
pub mod shape_to_path;
pub mod solid;
pub mod surface;
pub mod tess;
pub mod text_area;
pub mod text_draw;
pub mod text_extract;
pub mod text_geometry;
pub mod text_metrics;
pub mod text_warp;
pub mod tiled;
/// vork: moda-text-draw's copy (the same file); see PROVENANCE.md.
pub use moda_text_draw::vec_arena_cache;

use anyhow::Result;

pub use boolean_operands::install_scene_seams;
pub use device::{
    clear_device_lost_flag, device_lost_flag, is_device_lost_error, BackendChoice, GpuContext,
};
pub use frame::{Engine, PageRegionPx, RenderOptions};
/// An engine's working colour ([`Engine::new_with_color`]).
pub use moda_shaders::working_color;
pub use tiled::{render_page_tiled_to, TiledPageRender, DEFAULT_TILE_PX};

/// One rendered page (tightly-packed RGBA + pixel dimensions).
pub struct RenderedPage {
    pub page_index: usize,
    pub page_id: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Tolerated declines/degrades (`frame::PlanWarnings::skipped`) — the CLI
    /// warns and ships the page.
    pub skipped: Vec<String>,
    /// Refusals (`frame::PlanWarnings::dropped`): a fill or clip mask the
    /// engine refused and drew as nothing. The rest of the page is drawn; every
    /// lane names these nodes (the CLI's resource report `fills_dropped`).
    pub dropped: Vec<String>,
}

/// Render one page of an already-hydrated scene.
pub fn render_page(
    engine: &Engine,
    scene: &moda_scene::Scene,
    page_index: usize,
    opts: &RenderOptions,
) -> Result<RenderedPage> {
    let (plan, warnings) = frame::build_coverage_plan(scene, page_index, opts)?;
    let rgba = engine.render_plan(&plan)?;
    Ok(RenderedPage {
        page_index,
        page_id: plan.page_id.clone(),
        width: plan.width_px,
        height: plan.height_px,
        rgba,
        skipped: warnings.skipped,
        dropped: warnings.dropped,
    })
}

/// Render a region of one page's raster ([`frame::build_coverage_plan_region`]): the same draw items, a target
/// the region's size. Errors where the region plan refuses (active backdrop or layer-blur content).
pub fn render_page_region(
    engine: &Engine,
    scene: &moda_scene::Scene,
    page_index: usize,
    opts: &RenderOptions,
    region: PageRegionPx,
) -> Result<RenderedPage> {
    let (plan, warnings) = frame::build_coverage_plan_region(scene, page_index, opts, region)?;
    let rgba = engine.render_plan(&plan)?;
    Ok(RenderedPage {
        page_index,
        page_id: plan.page_id.clone(),
        width: plan.width_px,
        height: plan.height_px,
        rgba,
        skipped: warnings.skipped,
        dropped: warnings.dropped,
    })
}

/// Fixed margin (canvas px) framing the canvas-root content bounds — the
/// native FRAMING CONTRACT for page-less freeform renders: the output is
/// `(content bounds + margin) × scale`. The browser's capture of these docs
/// fits the content into whatever viewport the consumer happened to mount
/// (`fitAllFloatingContent` → `fitToRect`), so its output size is
/// viewport-dependent; the native contract is content-derived and stable —
/// exact-pixel parity with historical browser captures is a non-goal.
/// Defined in moda-scene (next to `canvas_root_content_bounds`) so the assess
/// manifest's `canvas_root_frame_px` and this render frame share ONE value.
pub use moda_scene::CANVAS_ROOT_FRAME_MARGIN;

/// `len` grown to whole pixels at `scale`: the length in canvas units that is `ceil(len × scale)` px, so the camera
/// plan's `ceil(viewport × dpr)` gives that pixel count back. `n / scale × scale` can land an ulp over `n` (a
/// 240.9449-unit frame at 300 dpi: 1338 / (300/72) × (300/72) = 1338.0000000000002), and that ceil made the surface
/// a pixel wider than `export_scale::output_dims` says; the crop then read the extra pixel as a clamped scale and the
/// frame's PNG came out 1005 px wide, not 1004. The length is stepped down to the largest that still rounds to `n`.
fn snap_to_whole_px(len: f64, scale: f64) -> f64 {
    if scale > 0.0 {
        let n = export_scale::px_ceil(len * scale);
        let mut v = n / scale;
        while v > 0.0 && v * scale > n {
            v = v.next_down();
        }
        v
    } else {
        len
    }
}

/// The canvas-absolute frame a page-less freeform render targets: the union
/// bounds of the canvas-root (floating) nodes plus [`CANVAS_ROOT_FRAME_MARGIN`]
/// on every side. `None` when the scene has no canvas-root node with
/// computable bounds (nothing to frame).
pub fn canvas_root_frame(scene: &moda_scene::Scene) -> Option<moda_scene::types::Rect> {
    let b = moda_scene::canvas_root_content_bounds(scene)?;
    Some(moda_scene::types::Rect {
        x: b.x - CANVAS_ROOT_FRAME_MARGIN,
        y: b.y - CANVAS_ROOT_FRAME_MARGIN,
        width: b.width + CANVAS_ROOT_FRAME_MARGIN * 2.0,
        height: b.height + CANVAS_ROOT_FRAME_MARGIN * 2.0,
    })
}

/// Build the coverage plan for a page-less freeform doc's canvas-root
/// (floating) content — the camera lane pointed at
/// [`canvas_root_frame`] with zero visible pages, so ONLY the floating nodes
/// render (plus the clear). Shared by [`render_canvas_root`] and the CLI's
/// no-GPU dump lane so every consumer frames identically.
///
/// Browser parity: a page-less root-only doc renders in freeform
/// layout REGARDLESS of its stored `layoutMode`
/// (`resolveReadOnlySurfaceLayoutMode` derives the runtime layout from the
/// root-only topology), while the engine's canvas-root pass gates on
/// `is_freeform()` — so a non-freeform stored mode is projected to freeform
/// here before planning.
pub fn build_canvas_root_plan(
    scene: &moda_scene::Scene,
    opts: &RenderOptions,
) -> Result<(seams::CoveragePlan, frame::PlanWarnings)> {
    let fr = canvas_root_frame(scene).ok_or_else(|| {
        anyhow::anyhow!("scene has no canvas-root content to frame (no floating nodes with bounds)")
    })?;
    let projected;
    let scene = if scene.is_freeform() {
        scene
    } else {
        let mut doc = scene.doc().clone();
        doc.layout_mode = Some("freeform".to_string());
        projected = moda_scene::Scene::from_document(doc)?;
        &projected
    };
    // The camera maps canvas point (frame.x, frame.y) to target (0, 0) at
    // zoom 1; `dpr = opts.scale` sizes the target to `frame × scale` (the
    // per-page export analog: page px × scale). White clear matches the
    // opaque page-export backdrop; `opts.transparent` threads through the
    // camera plan exactly like the freeform region export.
    // The target is `frame × scale` rounded UP to whole pixels; the viewport is that target in canvas units, so one
    // canvas unit stays `scale` px. A viewport of the frame's own fractional size was stretched over the rounded-up
    // target: a frame 50.25 tall drew ~0.6 % taller than asked, its edges off the pixel grid (a translucent first
    // row on an opaque frame's export, a light line in its JPEG).
    let snap = |len: f64| snap_to_whole_px(len, opts.scale);
    // The content's top-left lands on a whole pixel: the margin is `CANVAS_ROOT_FRAME_MARGIN × scale` rounded to
    // whole px, the rounding `--crop-to-node` applies to the same offset. At a scale where the margin is fractional
    // (300 dpi is 4.1667 px per unit: 40 → 166.67 px) a frame's edges fell mid-pixel, and its export had a
    // translucent edge row or column however whole its own size.
    let content = moda_scene::canvas_root_content_bounds(scene)
        .ok_or_else(|| anyhow::anyhow!("scene has no canvas-root content to frame"))?;
    let align = |content_edge: f64, frame_edge: f64| {
        if opts.scale > 0.0 {
            content_edge - ((content_edge - frame_edge) * opts.scale).round() / opts.scale
        } else {
            frame_edge
        }
    };
    let cam = frame::CameraParams {
        camera_x: align(content.x, fr.x),
        camera_y: align(content.y, fr.y),
        zoom: 1.0,
        viewport_w: snap(fr.width),
        viewport_h: snap(fr.height),
        dpr: opts.scale,
        clear_rgb: [1.0, 1.0, 1.0],
    };
    // An export, not a camera frame: Degraded text (emoji, fallback scripts) renders at
    // full quality, as a page export's does. `build_camera_plan` sizes it to the screen
    // (#654), which drew a Design frame's emoji from a 1× raster instead of the 2× one.
    let _full_quality = text_draw::enter_degraded_camera(None);
    let (plan, warnings) = frame::build_camera_plan_cached(scene, &[], &cam, opts, None, None)?
        .ok_or_else(|| anyhow::anyhow!("canvas-root camera plan was degenerate (no plan)"))?;
    // Blank-render tripwire: the freeform gate or an id/visibility mismatch
    // silently yielding ZERO draw items must error (callers decline), never
    // export a blank frame — hidden-only floating content lands here (its
    // bounds still frame, but nothing plans). An isolated render (`opts.isolate`, the PSD exporter's one layer at a
    // time) is exempt: a layer that draws nothing (an empty or all-space text) is a transparent layer, not a failed
    // export.
    if plan.items.is_empty() && opts.isolate.is_empty() {
        anyhow::bail!("canvas-root camera plan is empty (nothing would render)");
    }
    Ok((plan, warnings))
}

/// Render a page-less freeform doc's canvas-root (floating) content — the
/// entry point the page-indexed lanes lack (`render_page` requires a page).
/// Output framing is the [`canvas_root_frame`] contract; the result poses as
/// the doc's single surface (`page_index` 0, `page_id` = the scene root id)
/// so page-shaped consumers (CLI `--out-dir` naming, the backend's
/// virtual-page-0 addressing) need no new plumbing.
///
/// The requested `opts.scale` is clamped against the DEVICE limits for the
/// frame's dimensions (the CLI's `cap_opts_for_dims` math) so a sprawling
/// freeform workspace renders at a fitting effective scale on EVERY lane
/// (CLI, moda-py, wasm) instead of failing texture allocation.
pub fn render_canvas_root(
    engine: &Engine,
    scene: &moda_scene::Scene,
    opts: &RenderOptions,
) -> Result<RenderedPage> {
    let fr = canvas_root_frame(scene).ok_or_else(|| {
        anyhow::anyhow!("scene has no canvas-root content to frame (no floating nodes with bounds)")
    })?;
    let limits = engine.gpu.device.limits();
    let eff_scale = export_scale::clamp_export_scale(
        opts.scale,
        fr.width,
        fr.height,
        limits.max_texture_dimension_2d,
        0,
        limits.max_buffer_size,
    );
    let (out_w, out_h) = export_scale::output_dims(fr.width, fr.height, eff_scale);
    let eff_sample = export_scale::capped_sample_count(opts.sample_count, out_w, out_h);
    let opts = RenderOptions {
        scale: eff_scale,
        sample_count: eff_sample,
        ..opts.clone()
    };
    let (plan, warnings) = build_canvas_root_plan(scene, &opts)?;
    let rgba = engine.render_plan(&plan)?;
    Ok(RenderedPage {
        page_index: 0,
        page_id: moda_scene::types::SCENE_ROOT_ID.to_string(),
        width: plan.width_px,
        height: plan.height_px,
        rgba,
        skipped: warnings.skipped,
        dropped: warnings.dropped,
    })
}

/// Render one page through the Stage-1 retained-mode plan caches (R3/R4/R5).
///
/// Byte-identical to [`render_page`] for any given `(scene, page_index, opts)`
/// — the caches are pure memoization of deterministic, content-hash-keyed
/// functions (see `residency.rs`), so a hit returns exactly the plan a fresh
/// build would. The caller owns the [`residency::PlanCaches`] across calls
/// (the mobile FFI holds one per `Session`) so the EXPENSIVE per-frame work —
/// tessellation, text shaping, full coverage-plan assembly — is reused across
/// `render_page_at_time` frames instead of rebuilt every frame. The caller
/// supplies the per-node hash map for `scene`
/// (`residency::node_hashes_for_scene`); it keys every cache layer.
pub fn render_page_cached(
    engine: &Engine,
    scene: &moda_scene::Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: &mut residency::PlanCaches,
    hashes: &residency::HashMap<String, residency::NodeHashes>,
) -> Result<RenderedPage> {
    let (plan, warnings) = resolve_page_plan_cached(scene, page_index, opts, caches, hashes)?;
    let rgba = engine.render_plan(&plan)?;
    Ok(RenderedPage {
        page_index,
        page_id: plan.page_id.clone(),
        width: plan.width_px,
        height: plan.height_px,
        rgba,
        skipped: warnings.skipped.clone(),
        dropped: warnings.dropped.clone(),
    })
}

/// The plan-cache resolution both [`render_page_cached`] and
/// [`render_page_cached_nv12`] ride — ONE body so the two readback flavors
/// can never drift on fingerprinting or cache admission (the hand-mirrored-
/// twins bug class this codebase keeps re-documenting).
fn resolve_page_plan_cached(
    scene: &moda_scene::Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: &mut residency::PlanCaches,
    hashes: &residency::HashMap<String, residency::NodeHashes>,
) -> Result<(
    std::sync::Arc<seams::CoveragePlan>,
    std::sync::Arc<frame::PlanWarnings>,
)> {
    caches.begin_frame();
    let page_id = scene
        .doc()
        .pages
        .get(page_index)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "page index {page_index} out of range ({} pages)",
                scene.doc().pages.len()
            )
        })?
        .id()
        .to_string();
    // R5: whole-plan reuse keyed by the page fingerprint (folds node hashes,
    // page geometry, scale + the plan-relevant options, and the asset/font
    // residency generations). A hit skips the entire coverage-plan rebuild.
    let fingerprint = residency::page_fingerprint(scene, page_index, opts, hashes);
    match caches.page_get(&page_id, fingerprint, opts.shader_time_s) {
        Some(hit) => Ok(hit),
        None => {
            let (plan, warnings) =
                frame::build_coverage_plan_cached(scene, page_index, opts, Some((caches, hashes)))?;
            let plan = std::sync::Arc::new(plan);
            let warnings = std::sync::Arc::new(warnings);
            caches.page_put(
                &page_id,
                fingerprint,
                opts.shader_time_s,
                plan.clone(),
                warnings.clone(),
            );
            Ok((plan, warnings))
        }
    }
}

/// CPU-only twin of [`render_page_cached`]: resolve the page's coverage plan
/// through the same caches and return its build warnings WITHOUT submitting
/// any GPU work. Skipped/dropped draws are produced at plan-build time, so a
/// caller that must refuse on them before creating output (the composited
/// Edit export preflight) can probe here first; the R5 entry the probe seeds
/// is the one a subsequent [`render_page_cached`] of the same frame reuses.
pub fn probe_page_plan_cached(
    scene: &moda_scene::Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: &mut residency::PlanCaches,
    hashes: &residency::HashMap<String, residency::NodeHashes>,
) -> Result<std::sync::Arc<frame::PlanWarnings>> {
    let (_plan, warnings) = resolve_page_plan_cached(scene, page_index, opts, caches, hashes)?;
    Ok(warnings)
}

/// One rendered page as tightly-packed NV12 planes (Y then interleaved UV) —
/// the GPU-converted twin of [`RenderedPage`] for the raw-frame stream's
/// `pixelFormat=nv12` lane. Carries no alpha by construction.
pub struct RenderedPageNv12 {
    pub page_index: usize,
    pub page_id: String,
    pub width: u32,
    pub height: u32,
    pub nv12: Vec<u8>,
    pub skipped: Vec<String>,
    /// See [`RenderedPage::dropped`].
    pub dropped: Vec<String>,
}

/// [`render_page_cached`] with the GPU RGBA→NV12 conversion at the readback
/// seam: identical plan-cache behavior and identical GPU raster; only the
/// readback differs (a compute pass converts the finished texture and 1.5
/// bytes/px come back instead of 4). Even dimensions only; the caller routes
/// odd-sized or transparent renders to the RGBA lane.
pub fn render_page_cached_nv12(
    engine: &Engine,
    scene: &moda_scene::Scene,
    page_index: usize,
    opts: &RenderOptions,
    caches: &mut residency::PlanCaches,
    hashes: &residency::HashMap<String, residency::NodeHashes>,
    converter: &nv12::Nv12Converter,
) -> Result<RenderedPageNv12> {
    let (plan, warnings) = resolve_page_plan_cached(scene, page_index, opts, caches, hashes)?;
    let nv12 = engine.render_plan_nv12(&plan, converter)?;
    Ok(RenderedPageNv12 {
        page_index,
        page_id: plan.page_id.clone(),
        width: plan.width_px,
        height: plan.height_px,
        nv12,
        skipped: warnings.skipped.clone(),
        dropped: warnings.dropped.clone(),
    })
}

/// `moda_scene::ShaderRegistry` derived from the GENERATED moda-shaders
/// extraction manifest (M0a) — the ground truth the render path itself packs
/// fills from — instead of the hand-maintained const default in
/// `moda_scene::assess`.
///
/// moda-scene must not depend on moda-shaders (dependency direction), so the
/// constructor lives here in the callers' orbit; every production assess site
/// (moda-cli, moda-py, moda-wasm) wires this into `AssessOptions` so a new
/// shader landing via extraction can never render live yet assess Unknown
/// server-side (the decline → browser-lane → 500 incident class). The
/// `default_registry_matches_extraction_manifest` tripwire below pins the
/// remaining hand-maintained default to these sets until it is deleted.
///
/// - `known_ids`: manifest modules of class `procedural-fill` (the 49
///   registry shaders).
/// - `aliases`: `shader_id_aliases` (LEGACY_SHADER_ID_ALIASES).
/// - `text_fill_ineligible_ids`: procedural − `vector_text_fill.eligible_shader_ids`
///   (image-backed shaders — `isTextShaderFillEligible` evaluated by the
///   extractor).
pub fn extracted_shader_registry() -> moda_scene::ShaderRegistry {
    use std::collections::HashSet;

    let m = moda_shaders::manifest();
    let known_ids: HashSet<String> = m
        .modules
        .iter()
        .filter(|(_, meta)| meta.class == "procedural-fill")
        .map(|(id, _)| id.clone())
        .collect();
    let eligible: HashSet<&str> = m
        .vector_text_fill
        .eligible_shader_ids
        .iter()
        .map(String::as_str)
        .collect();
    let text_fill_ineligible_ids = known_ids
        .iter()
        .filter(|id| !eligible.contains(id.as_str()))
        .cloned()
        .collect();
    moda_scene::ShaderRegistry {
        known_ids,
        aliases: m.shader_id_aliases.clone(),
        // Image-backed shaders were un-gated at M4e; `gated_ids` remains a
        // mechanism for future gates only, and the manifest encodes none.
        gated_ids: HashSet::new(),
        text_fill_ineligible_ids,
    }
}

#[cfg(test)]
mod canvas_root_tests {
    use super::*;

    fn pageless_scene(layout_mode: Option<&str>) -> moda_scene::Scene {
        let mut doc = moda_scene::value::json!({
            "nodes": [
                {"id":"float1","type":"rectangle","x":100.0,"y":80.0,"width":300.0,"height":200.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#e94040"}}]},
                {"id":"float2","type":"ellipse","x":450.0,"y":150.0,"width":180.0,"height":180.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#2255ee"}}]}
            ],
            "pages": [],
            "canvasRootNodeIds": ["float1","float2"]
        });
        if let Some(mode) = layout_mode {
            doc["layoutMode"] = moda_scene::value::json!(mode);
        }
        moda_scene::Scene::load(&doc).unwrap()
    }

    /// CPU-only: the page-less freeform plan frames content bounds + margin
    /// ((100,80)-(630,330) + 40 → 610×330 at scale 1) and carries BOTH
    /// floating nodes' draw items.
    #[test]
    fn canvas_root_plan_frames_content_bounds_plus_margin() {
        let scene = pageless_scene(Some("freeform"));
        let fr = canvas_root_frame(&scene).expect("frame");
        assert_eq!(
            (fr.x, fr.y, fr.width, fr.height),
            (60.0, 40.0, 610.0, 330.0)
        );
        let (plan, warnings) = build_canvas_root_plan(&scene, &RenderOptions::default()).unwrap();
        assert_eq!((plan.width_px, plan.height_px), (610, 330));
        assert!(warnings.skipped.is_empty(), "{:?}", warnings.skipped);
        assert!(
            plan.items.len() >= 2,
            "both floating nodes plan draw items (got {})",
            plan.items.len()
        );
        // Opaque export clear: white at alpha 1 (page-export parity).
        assert_eq!(plan.clear_rgba, Some([1.0, 1.0, 1.0, 1.0]));

        // `--scale 2` doubles the target (the per-page `page px × scale` analog).
        let opts2 = RenderOptions {
            scale: 2.0,
            ..RenderOptions::default()
        };
        let (plan2, _w) = build_canvas_root_plan(&scene, &opts2).unwrap();
        assert_eq!((plan2.width_px, plan2.height_px), (1220, 660));
        assert_eq!(plan2.camera, [60.0, 40.0], "a whole margin keeps the frame's own origin");
    }

    /// At 300 dpi (300/72 px per unit) the 40-unit margin is 166.67 px: the camera moves so the content's top-left
    /// is a whole pixel, the pixel `--crop-to-node` rounds the same offset to, and its edges are not half-covered.
    #[test]
    fn canvas_root_plan_puts_the_content_origin_on_a_whole_pixel() {
        let scene = pageless_scene(Some("freeform"));
        let fr = canvas_root_frame(&scene).expect("frame");
        for scale in [300.0 / 72.0, 150.0 / 72.0, 1.5, 0.3] {
            let opts = RenderOptions {
                scale,
                ..RenderOptions::default()
            };
            let (plan, _w) = build_canvas_root_plan(&scene, &opts).unwrap();
            for (content, camera, frame) in [(100.0, plan.camera[0], fr.x), (80.0, plan.camera[1], fr.y)] {
                let px = (content - camera) * scale;
                assert!((px - px.round()).abs() < 1e-9, "@{scale}: content edge at {px} px");
                assert_eq!(px.round(), ((content - frame) * scale).round(), "@{scale}: the crop's pixel");
            }
        }
    }

    /// The snapped viewport is the pixel count `output_dims` gives, at every scale: 1338 / (300/72) × (300/72) is an
    /// ulp over 1338, and its ceil made a 240.9449-unit frame's 300 dpi PNG 1005 px wide.
    #[test]
    fn snapped_viewport_is_the_rounded_up_pixel_count() {
        for len in [240.9449, 320.9449, 155.9055, 595.2756, 841.8898, 100.0, 50.25, 1.0, 0.3] {
            for scale in [300.0 / 72.0, 150.0 / 72.0, 600.0 / 72.0, 1.5, 0.3, 2.0, 4.1667] {
                let v = snap_to_whole_px(len, scale);
                assert_eq!(
                    export_scale::px_ceil(v * scale),
                    export_scale::px_ceil(len * scale),
                    "{len} @ {scale}"
                );
                assert!(
                    v >= len - 1e-9,
                    "{len} @ {scale}: snapped {v} is under the length"
                );
            }
        }
        assert_eq!(
            ((snap_to_whole_px(320.9449, 300.0 / 72.0)) * (300.0 / 72.0)).ceil(),
            1338.0
        );
    }

    /// Browser parity: a page-less root-only doc renders freeform
    /// REGARDLESS of its stored layoutMode — a legacy doc without
    /// `layoutMode: "freeform"` must still plan its floating nodes.
    #[test]
    fn canvas_root_plan_projects_freeform_for_non_freeform_stored_mode() {
        for mode in [None, Some("design"), Some("vertical")] {
            let scene = pageless_scene(mode);
            let (plan, _w) = build_canvas_root_plan(&scene, &RenderOptions::default()).unwrap();
            assert!(
                plan.items.len() >= 2,
                "stored mode {mode:?} must not blank the canvas-root plan"
            );
        }
    }

    /// `--transparent` threads into the camera plan: transparent
    /// flag + fully-zero clear so the readback un-premultiplies to true alpha.
    #[test]
    fn canvas_root_plan_threads_transparent() {
        let scene = pageless_scene(Some("freeform"));
        let opts = RenderOptions {
            transparent: true,
            ..RenderOptions::default()
        };
        let (plan, _w) = build_canvas_root_plan(&scene, &opts).unwrap();
        assert!(plan.transparent);
        assert_eq!(plan.clear_rgba, Some([0.0, 0.0, 0.0, 0.0]));
    }

    /// No canvas-root content → a typed error (the CLI/py lanes surface it),
    /// never a wrong/blank render.
    #[test]
    fn canvas_root_plan_errors_on_empty_scene() {
        let scene = moda_scene::Scene::load(
            &moda_scene::value::json!({"nodes": [], "pages": [], "canvasRootNodeIds": []}),
        ).unwrap();
        assert!(canvas_root_frame(&scene).is_none());
        let err = build_canvas_root_plan(&scene, &RenderOptions::default()).unwrap_err();
        assert!(err.to_string().contains("no canvas-root content"));
    }

    /// Blank-render tripwire (CPU-only): hidden-only floating content still
    /// has computable BOUNDS (so a frame exists) but plans zero draw items —
    /// that must be an ERROR the callers turn into a decline, never a blank
    /// white export.
    #[test]
    fn canvas_root_plan_errors_on_invisible_only_content() {
        let scene = moda_scene::Scene::load(&moda_scene::value::json!({
            "nodes": [
                {"id":"h","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],"hidden":true}
            ],
            "pages": [],
            "canvasRootNodeIds": ["h"]
        })).unwrap();
        let err = build_canvas_root_plan(&scene, &RenderOptions::default()).unwrap_err();
        assert!(
            err.to_string().contains("nothing would render")
                || err.to_string().contains("no canvas-root content"),
            "got: {err}"
        );
    }

    /// An isolated render of a layer that draws nothing (the PSD exporter's per-layer render of an empty text) plans
    /// zero items without an error: the layer is transparent, and the rest of the export goes on.
    #[test]
    fn canvas_root_isolated_render_of_a_layer_that_draws_nothing_is_empty_not_an_error() {
        let scene = moda_scene::Scene::load(&moda_scene::value::json!({
            "nodes": [
                {"id":"r","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]},
                {"id":"h","type":"rectangle","x":20.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],"hidden":true}
            ],
            "pages": [],
            "canvasRootNodeIds": ["r", "h"]
        })).unwrap();
        let opts = RenderOptions { isolate: vec!["h".to_string()], transparent: true, ..RenderOptions::default() };
        let (plan, _w) = build_canvas_root_plan(&scene, &opts).unwrap();
        assert!(plan.items.is_empty());
        let opts = RenderOptions { isolate: vec!["r".to_string()], transparent: true, ..RenderOptions::default() };
        assert!(!build_canvas_root_plan(&scene, &opts).unwrap().0.items.is_empty());
    }

    /// GPU golden for the canvas-root entry (skips without an adapter; run
    /// with --test-threads=1): a page-less doc with one red floating rect
    /// renders through `render_canvas_root` — bounds+margin framing
    /// (50×40 content + 40px margin → 130×120), red center, white margin
    /// corner, and scale riding the frame (2× → 260×240).
    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn canvas_root_entry_renders_floating_content() {
        let gpu = crate::device::GpuContext::new(crate::device::BackendChoice::Auto).expect("GPU test requires a working adapter");
        let engine = Engine::new(gpu).expect("engine init");

        // NO layoutMode on purpose: the content-derived freeform projection
        // (0 pages + valid canvas root ⇒ freeform) must kick in — a literal
        // `is_freeform` gate would silently render nothing here.
        let scene = moda_scene::Scene::load(&moda_scene::value::json!({
            "nodes": [
                {"id":"floatrect","type":"rectangle","x":100.0,"y":200.0,
                 "width":50.0,"height":40.0,"rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}]}
            ],
            "pages": [],
            "canvasRootNodeIds": ["floatrect"]
        })).unwrap();
        let opts = RenderOptions {
            scale: 1.0,
            sample_count: 1,
            ..RenderOptions::default()
        };
        let r = render_canvas_root(&engine, &scene, &opts).expect("canvas-root render");
        assert_eq!(r.page_id, moda_scene::types::SCENE_ROOT_ID);
        assert_eq!(
            (r.width, r.height),
            (130, 120),
            "content bounds + 40px margin framing"
        );
        assert!(r.skipped.is_empty(), "no skipped draws: {:?}", r.skipped);
        // Frame origin is (60,160); the rect's center (125,220) lands at the
        // output center (65,60).
        let center = (((r.height / 2) * r.width + r.width / 2) * 4) as usize;
        let px = &r.rgba[center..center + 4];
        assert!(
            px[0] > 200 && px[1] < 60 && px[2] < 60 && px[3] == 255,
            "center pixel must be the rect's red, got {px:?}"
        );
        // Margin corner: opaque white clear (page-export backdrop parity).
        let corner = &r.rgba[0..4];
        assert!(
            corner[0] > 240 && corner[1] > 240 && corner[2] > 240 && corner[3] == 255,
            "margin corner must be the white clear, got {corner:?}"
        );

        // Scale rides the frame: 2× doubles the output.
        let opts2 = RenderOptions {
            scale: 2.0,
            ..opts.clone()
        };
        let r2 = render_canvas_root(&engine, &scene, &opts2).expect("2x canvas-root render");
        assert_eq!((r2.width, r2.height), (260, 240));

        // Truly-empty canvas root: ERROR, not a blank frame.
        let empty = moda_scene::Scene::load(&moda_scene::value::json!({"nodes": [], "pages": []})).unwrap();
        assert!(render_canvas_root(&engine, &empty, &opts).is_err());

        // Invisible-only floating content (hidden node): ERROR, not blank.
        let hidden = moda_scene::Scene::load(&moda_scene::value::json!({
            "nodes": [
                {"id":"h","type":"rectangle","x":0.0,"y":0.0,"width":10.0,"height":10.0,
                 "rotation":0,"scale":1,"fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#ff0000"}}],"hidden":true}
            ],
            "pages": [],
            "canvasRootNodeIds": ["h"]
        })).unwrap();
        assert!(render_canvas_root(&engine, &hidden, &opts).is_err());
    }

    /// A blend whose objects are on the canvas — a canvas root, or a frame's
    /// children in a page-less Design document — draws its steps. Objects with
    /// no page used to be dropped from the chain, so the blend drew nothing.
    #[test]
    fn canvas_blends_draw_their_steps() {
        let solid = |c: &str| moda_scene::value::json!([{"id": "fill-0", "paint": {"type": "solid", "color": c}}]);
        let scene_with = |blends: bool| {
            let mut nodes = vec![
                moda_scene::value::json!({"id":"frame","type":"container","x":0.0,"y":0.0,"width":600.0,"height":300.0,
                 "rotation":0,"scale":1,"clipContent":true,"children":["r1","e1","bl1"],"fills": solid("#ffffff")}),
                moda_scene::value::json!({"id":"r1","type":"rectangle","x":20.0,"y":20.0,"width":80.0,"height":80.0,
                 "rotation":0,"scale":1,"parentId":"frame","fills": solid("#888888")}),
                moda_scene::value::json!({"id":"e1","type":"ellipse","x":450.0,"y":150.0,"width":100.0,"height":100.0,
                 "rotation":0,"scale":1,"parentId":"frame","fills": solid("#ff0000")}),
                moda_scene::value::json!({"id":"r2","type":"rectangle","x":20.0,"y":400.0,"width":80.0,"height":80.0,
                 "rotation":0,"scale":1,"fills": solid("#888888")}),
                moda_scene::value::json!({"id":"e2","type":"ellipse","x":450.0,"y":400.0,"width":100.0,"height":100.0,
                 "rotation":0,"scale":1,"fills": solid("#ff0000")}),
            ];
            if blends {
                nodes.push(moda_scene::value::json!({"id":"bl1","type":"blend","sourceIds":["r1","e1"],"parentId":"frame","spacing":"smooth"}));
                nodes.push(moda_scene::value::json!({"id":"bl2","type":"blend","sourceIds":["r2","e2"],"spacing":"smooth"}));
            }
            moda_scene::Scene::load(&moda_scene::value::json!({
                "nodes": nodes,
                "pages": [],
                "canvasRootNodeIds": if blends { moda_scene::value::json!(["frame","r2","bl2","e2"]) } else { moda_scene::value::json!(["frame","r2","e2"]) },
                "layoutMode": "freeform"
            })).unwrap()
        };
        let (plain, _) = build_canvas_root_plan(&scene_with(false), &RenderOptions::default()).unwrap();
        let (blended, warnings) = build_canvas_root_plan(&scene_with(true), &RenderOptions::default()).unwrap();
        assert!(
            !warnings.skipped.iter().any(|w| w.contains("blend")),
            "{:?}",
            warnings.skipped
        );
        // Smooth spacing between grey and red makes many steps for each blend.
        assert!(
            blended.items.len() >= plain.items.len() + 2 * 10,
            "blend steps missing: {} items with blends vs {} without",
            blended.items.len(),
            plain.items.len()
        );
    }
}

#[cfg(test)]
mod shader_registry_tests {
    /// Tripwire (CPU-only): the hand-maintained `ShaderRegistry::default()`
    /// const in moda-scene cannot drift from the generated extraction
    /// manifest while it still exists. A new shader landing via extraction
    /// without this pin would render live (WASM/editor) yet assess Unknown
    /// server-side — exactly the decline-to-browser incident class. Once the
    /// const is deleted (follow-up), this test goes with it.
    #[test]
    fn default_registry_matches_extraction_manifest() {
        let extracted = super::extracted_shader_registry();
        let hand = moda_scene::ShaderRegistry::default();
        assert_eq!(extracted.known_ids, hand.known_ids, "known_ids drifted");
        assert_eq!(extracted.aliases, hand.aliases, "aliases drifted");
        assert_eq!(extracted.gated_ids, hand.gated_ids, "gated_ids drifted");
        assert_eq!(
            extracted.text_fill_ineligible_ids, hand.text_fill_ineligible_ids,
            "text_fill_ineligible_ids drifted"
        );
    }
}

/// Initialize both text lanes from bundled fixtures before either global host is built.
#[cfg(test)]
pub(crate) fn test_fonts() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fonts");
        text_draw::set_fonts_dir(dir.clone());
        moda_text::degraded::set_catalog_fonts_dir(dir);
    });
}
