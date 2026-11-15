//! Isolated render targets — the visual render-source seam
//! (docs/media/edit-audio-video-plan.md §6.2, §12 V1).
//!
//! One composition/page renders to an alpha-capable offscreen GPU target
//! WITHOUT focusing the editor or capturing the full page; multiple targets
//! stay active in one output frame; lifetime is explicit. This module is the
//! STABLE engine API behind the public browser seam: the `moda-wasm` exports
//! (`render_isolated_target` / `release_isolated_target` /
//! `composite_isolated_targets_to_bytes` — capability-listed in its
//! `contract.json`) and the typed adapter in
//! `renderers/wasm/wasmIsolatedTargets.ts` are thin wrappers over it.
//!
//! ## Lane choice (settled here, per the ticket)
//!
//! Isolated targets ride the RETAINED page lane, not the one-shot export
//! lane: plans come from [`crate::residency::build_plan_retained`] (R3/R4/R5
//! caches + the incremental patcher), the target texture is resident across
//! frames (reallocated only on size change), and the render goes straight
//! into that texture via [`Engine::render_plan_to_view`] — no readback
//! anywhere on the path. Edit preview re-renders isolated targets every
//! frame, so the per-frame profile is "cache hit + GPU submit", the same
//! shape as the editor's resident camera lane.
//!
//! The engine's retained-mode offscreen pool (`frame_targets`) is keyed by
//! SIZE and shared across renders, so it cannot back isolated targets: two
//! same-sized compositions would alias one pooled texture and the second
//! render would clobber the first mid-frame. Every [`IsolatedTarget`]
//! therefore OWNS its texture. The pool is still EXERCISED on every render —
//! `render_plan_to_view` allocates the pooled single-sample offscreen (used
//! as a destination only on the backdrop-split fallback, then copied out
//! before the next render — safe under queue submission order) plus the MSAA
//! color + depth scratch. That pool holds only `RETAINED_TARGET_SLOTS` (3)
//! size keys, so a stack with MORE than 3 distinct target sizes evicts and
//! reallocates pooled scratch every frame (the hero-stack demo's 4-size
//! shape does; timings in ISOLATED-RENDER-TARGETS-PROTOTYPE.md include that
//! churn). Sizing per-lane scratch is an open concern.
//!
//! [`IsolatedTargets`] and [`IsolatedCompositor`] hold resources of ONE
//! engine's `wgpu` device: wgpu resources cannot cross devices, so on engine
//! replacement (device loss recovery) drop and recreate them alongside the
//! `Engine` — exactly how the wasm host treats every other engine-owned GPU
//! resource. The `ensure` format check guards a frame-format change, not a
//! device change. The public seam realizes that recovery by ownership: the
//! `WasmEngine` holds its `IsolatedTargets`/`IsolatedCompositor` as fields,
//! so every host reinit path drops them with the old handle and the host
//! re-renders targets against the fresh engine.
//!
//! Passing the plan `Arc` as the M2 prepared-frame packet identity makes
//! this lane the second engager of the engine-global SINGLE-SLOT packet
//! (the resident camera present is the first). The identity/epoch checks
//! keep every combination correct, but a frame that runs both a camera
//! present and an isolated render evicts the packet both ways — each lane
//! then re-prepares per frame. Also an open accounting item.
//!
//! ## Alpha convention: PREMULTIPLIED
//!
//! An isolated target holds premultiplied alpha in the engine's frame format
//! ([`Engine::frame_format`], `Rgba8Unorm` unless configured otherwise):
//! the plan renders with `transparent: true` (clear `[0,0,0,0]`, premul
//! blending — exactly the export-transparent path). Consumers compositing
//! the texture in-frame must blend with `(ONE, ONE_MINUS_SRC_ALPHA)`
//! ([`IsolatedCompositor`] does). Straight alpha exists only at the CPU
//! readback boundary (`readback::unpremultiply_in_place`) — that is the
//! export worker's concern, never this seam's. The pixel tests in
//! `tests/isolated_targets.rs` pin the convention.
//!
//! Known exception, inherited from the plan builder: a page with an ACTIVE
//! BACKDROP-BLUR node forces its plan opaque (white clear) and reports a
//! "--transparent unsupported on backdrop pages" skip warning — the backdrop
//! effect samples the scene behind the node, which does not exist over a
//! transparent clear.
//! Such a target still renders (through the offscreen + copy-in fallback
//! below) but holds an OPAQUE frame; callers see the warning on
//! [`IsolatedRenderInfo::warnings`]. Pinned by the backdrop test in
//! `tests/isolated_targets.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::frame::{Engine, PlanWarnings, RenderOptions};
use crate::residency::{build_plan_retained, PlanCaches, ResidentScene};

/// Usages every isolated target texture carries:
/// - `RENDER_ATTACHMENT`: the plan renders (or MSAA-resolves) directly into it.
/// - `TEXTURE_BINDING`: the same-frame compositor samples it.
/// - `COPY_DST`: the backdrop-split fallback renders to the pooled offscreen
///   and copies in.
/// - `COPY_SRC`: pixel tests (and a future export lane) read it back.
pub const ISOLATED_TARGET_USAGES: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::COPY_SRC);

/// One resident alpha target: an engine-frame-format texture owned by this
/// target (never the shared size-keyed pool — see the module doc).
pub struct IsolatedTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width_px: u32,
    height_px: u32,
    /// Bumped on every (re)allocation of the backing texture, so callers and
    /// tests can observe residency: an unchanged generation across frames
    /// proves the texture was reused, not recreated.
    generation: u64,
}

impl IsolatedTarget {
    /// The GPU-resident texture (premultiplied alpha, engine frame format).
    /// This handle IS the prototype seam: the V3 compositor consumes it in
    /// the same frame graph.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width_px, self.height_px)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// GPU bytes this target's color texture holds (4 bytes/px in every
    /// supported frame format) — the memory-accounting input for the
    /// prototype measurements.
    pub fn texture_bytes(&self) -> u64 {
        self.width_px as u64 * self.height_px as u64 * 4
    }
}

impl Drop for IsolatedTarget {
    /// Explicit-lifetime contract: the GPU texture dies WITH the target
    /// (release, size-change reallocation, or the pool dropping), not at some
    /// later refcount-driven point.
    fn drop(&mut self) {
        crate::scratch::destroy(&self.texture);
    }
}

/// The set of active isolated targets, keyed by caller-chosen id. Lifetime is
/// EXPLICIT: a target exists from its first render until [`release`]d
/// (or the pool is dropped); nothing evicts behind the caller's back.
///
/// [`release`]: IsolatedTargets::release
#[derive(Default)]
pub struct IsolatedTargets {
    targets: HashMap<String, IsolatedTarget>,
    /// Monotonic allocation counter feeding [`IsolatedTarget::generation`].
    allocations: u64,
}

impl IsolatedTargets {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, id: &str) -> Option<&IsolatedTarget> {
        self.targets.get(id)
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Total GPU bytes held by all active targets' color textures.
    pub fn texture_bytes(&self) -> u64 {
        self.targets.values().map(|t| t.texture_bytes()).sum()
    }

    /// Destroy one target's GPU texture now (via [`IsolatedTarget`]'s `Drop`)
    /// and forget it. Returns whether the id existed.
    pub fn release(&mut self, id: &str) -> bool {
        self.targets.remove(id).is_some()
    }

    /// Explicit owner-driven eviction; retained textures are never moved.
    pub fn retain(&mut self, mut keep: impl FnMut(&str) -> bool) {
        self.targets.retain(|id, _| keep(id));
    }

    /// Destroy every target's GPU texture now.
    pub fn release_all(&mut self) {
        self.targets.clear();
    }

    /// Populate an explicitly owned target without constructing a Scene node.
    /// Used by decoded media, contained pictures and Edit transition results.
    /// Budget the complete live set, including a prior allocation during resize.
    pub fn write_target(
        &mut self,
        engine: &Engine,
        id: &str,
        size: (u32, u32),
        byte_budget: u64,
        write: impl FnOnce(&IsolatedTarget) -> Result<()>,
    ) -> Result<&IsolatedTarget> {
        self.write_target_format(engine, id, size, engine.frame_format(), byte_budget, write)
    }

    /// An explicit format is needed for decoded-video textures, which also
    /// feed the engine's RGBA image cache without a CPU readback/swizzle.
    pub fn write_target_format(
        &mut self,
        engine: &Engine,
        id: &str,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        byte_budget: u64,
        write: impl FnOnce(&IsolatedTarget) -> Result<()>,
    ) -> Result<&IsolatedTarget> {
        let (width, height) = size;
        let max = engine.gpu.device.limits().max_texture_dimension_2d;
        if id.is_empty() || width == 0 || height == 0 || width > max || height > max {
            bail!("isolated target: invalid id or dimensions {width}x{height}");
        }
        if self.get_fitting(format, id, width, height).is_some() {
            if self.texture_bytes() > byte_budget {
                bail!("isolated target set exceeds its {byte_budget}-byte budget");
            }
            let target = self.get(id).expect("fitting target");
            write(target)?;
            return Ok(target);
        }
        let bytes = u64::from(width) * u64::from(height) * 4;
        if bytes.saturating_add(self.texture_bytes()) > byte_budget {
            bail!("isolated target allocation exceeds its {byte_budget}-byte budget");
        }
        let target = self.create_detached(&engine.gpu.device, format, id, width, height);
        write(&target)?;
        Ok(self.publish(id, target))
    }

    /// The existing target for `id`, if it already matches exactly
    /// `width_px × height_px` in `format` — the resident steady-state path.
    fn get_fitting(
        &self,
        format: wgpu::TextureFormat,
        id: &str,
        width_px: u32,
        height_px: u32,
    ) -> Option<&IsolatedTarget> {
        self.targets.get(id).filter(|t| {
            t.width_px == width_px && t.height_px == height_px && t.texture.format() == format
        })
    }

    /// Allocate a DETACHED target (not yet in the set). The caller renders
    /// into it and [`publish`]es it only on success — so a failed first
    /// render never exposes a phantom blank target, and a failed resize
    /// render never destroys the previous valid frame.
    ///
    /// [`publish`]: IsolatedTargets::publish
    fn create_detached(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        id: &str,
        width_px: u32,
        height_px: u32,
    ) -> IsolatedTarget {
        self.allocations += 1;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("isolated target {id}")),
            size: wgpu::Extent3d {
                width: width_px,
                height: height_px,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: ISOLATED_TARGET_USAGES,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        IsolatedTarget {
            texture,
            view,
            width_px,
            height_px,
            generation: self.allocations,
        }
    }

    /// Install a successfully-rendered detached target under `id`, displacing
    /// (and destroying, via Drop) any previous allocation for that id.
    fn publish(&mut self, id: &str, target: IsolatedTarget) -> &IsolatedTarget {
        self.targets.insert(id.to_string(), target);
        self.targets.get(id).expect("just published")
    }
}

/// What one isolated render produced (the caller samples the texture via
/// [`IsolatedTargets::get`]; this reports the plan-derived facts).
pub struct IsolatedRenderInfo {
    pub width_px: u32,
    pub height_px: u32,
    pub warnings: Arc<PlanWarnings>,
    /// The target texture's allocation generation after this render — stable
    /// across frames while the composition size is unchanged.
    pub target_generation: u64,
}

/// Render one page/composition of the resident scene into the isolated
/// target `target_id`, creating or resizing the target to the plan's output
/// size (`page px × opts.scale`). The editor is never focused and the full
/// page/viewport is never captured — the plan is scoped to `page_index`
/// exactly like an export plan, but the pixels stay GPU-resident.
///
/// `opts.transparent` is forced `true`: an isolated target is BY DEFINITION
/// an alpha target (premultiplied — module doc). Callers drive this once per
/// target per frame after a single `caches.begin_frame()`; plan reuse across
/// frames comes from the retained page caches, so an unchanged composition
/// re-renders without rebuilding tessellation, text layout, or the plan.
///
/// The engine should be in retained mode ([`Engine::set_retained`]) — the
/// resident lane's posture; non-retained still renders correctly but
/// allocates scratch per call.
///
/// `opts.scale` and `opts.sample_count` are clamped against THIS device's
/// limits for the page's dimensions (the `render_canvas_root` /
/// `cap_opts_for_dims` policy), so an oversized page or scale degrades to a
/// fitting effective scale instead of failing texture allocation or asking
/// for a multi-gigabyte MSAA scratch.
pub fn render_isolated_target(
    engine: &Engine,
    resident: &ResidentScene,
    caches: &mut PlanCaches,
    targets: &mut IsolatedTargets,
    target_id: &str,
    page_index: usize,
    opts: &RenderOptions,
) -> Result<IsolatedRenderInfo> {
    let scene = resident
        .scene()
        .ok_or_else(|| anyhow::anyhow!("no resident scene (update first)"))?;
    let page = scene.doc().pages.get(page_index).ok_or_else(|| {
        anyhow::anyhow!(
            "isolated target '{target_id}': page index {page_index} out of range ({} pages)",
            scene.doc().pages.len()
        )
    })?;
    let (page_w, page_h) = (page.width(), page.height());
    let limits = engine.gpu.device.limits();
    let eff_scale = crate::export_scale::clamp_export_scale(
        opts.scale,
        page_w,
        page_h,
        limits.max_texture_dimension_2d,
        0,
        limits.max_buffer_size,
    );
    let (out_w, out_h) = crate::export_scale::output_dims(page_w, page_h, eff_scale);
    let eff_sample = crate::export_scale::capped_sample_count(opts.sample_count, out_w, out_h);
    let opts = RenderOptions {
        transparent: true,
        scale: eff_scale,
        sample_count: eff_sample,
        ..opts.clone()
    };
    let (plan, warnings) =
        build_plan_retained(resident, caches, page_index, &opts).with_context(|| {
            format!("isolated target '{target_id}': plan build (page {page_index})")
        })?;
    if plan.width_px == 0 || plan.height_px == 0 {
        bail!(
            "isolated target '{target_id}': page {page_index} plans a zero-area target \
             ({}x{})",
            plan.width_px,
            plan.height_px
        );
    }
    // Render directly into the resident texture. The target is always the
    // engine's frame format, so a direct render/MSAA-resolve is legal
    // (`direct_present_ok: true`). `Some(offscreen)` is the backdrop-split
    // fallback: the frame rendered into the pooled offscreen instead — copy
    // it into the resident texture (same format, same size) so the caller's
    // handle is always the up-to-date pixels.
    //
    // The plan Arc doubles as the M2 prepared-frame packet identity: a
    // single-target steady state (same plan, same size, untouched pool
    // epoch) skips the per-frame text prepare + geometry uploads exactly
    // like the resident camera lane; interleaved multi-target frames miss
    // the packet's identity checks and upload normally. Byte-stability
    // across frames is pinned by `tests/isolated_targets.rs`.
    let render_into = |target: &IsolatedTarget| -> Result<()> {
        if let Some(offscreen) =
            engine.render_plan_to_view(&plan, None, &target.view, None, Some(&plan), true)?
        {
            let mut encoder =
                engine
                    .gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("isolated target copy-in"),
                    });
            encoder.copy_texture_to_texture(
                offscreen.as_image_copy(),
                target.texture.as_image_copy(),
                wgpu::Extent3d {
                    width: plan.width_px,
                    height: plan.height_px,
                    depth_or_array_layers: 1,
                },
            );
            engine.gpu.queue.submit([encoder.finish()]);
        }
        Ok(())
    };
    // Publish-on-success: the resident steady state renders into the existing
    // fitting texture; a first render or a size change renders into a
    // DETACHED allocation and installs it only after the render succeeds — a
    // failed render never exposes a phantom blank target and never destroys
    // the previous valid frame (the set is unchanged on error).
    let format = engine.frame_format();
    let fits = targets
        .get_fitting(format, target_id, plan.width_px, plan.height_px)
        .is_some();
    let target_generation = if fits {
        let existing = targets
            .get_fitting(format, target_id, plan.width_px, plan.height_px)
            .expect("just checked");
        render_into(existing)?;
        existing.generation
    } else {
        let fresh = targets.create_detached(
            &engine.gpu.device,
            format,
            target_id,
            plan.width_px,
            plan.height_px,
        );
        render_into(&fresh)?;
        targets.publish(target_id, fresh).generation
    };
    Ok(IsolatedRenderInfo {
        width_px: plan.width_px,
        height_px: plan.height_px,
        warnings,
        target_generation,
    })
}

/// WGSL for the prototype compositor: one textured quad per layer, placed by
/// an NDC rect uniform, premultiplied source scaled by a flat opacity.
const COMPOSITE_WGSL: &str = r#"
struct LayerUniform {
    // Destination rect in clip space: (x_left, y_top, x_right, y_bottom).
    rect: vec4<f32>,
    // x: flat layer opacity applied to the PREMULTIPLIED source (all four
    // channels scale together); yzw: padding.
    opacity: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: LayerUniform;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let c = corners[vi];
    var out: VsOut;
    out.pos = vec4<f32>(mix(u.rect.x, u.rect.z, c.x), mix(u.rect.y, u.rect.w, c.y), 0.0, 1.0);
    out.uv = c;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv) * u.opacity.x;
}
"#;

/// One host-authored compositor layer as it crosses the public seam:
/// The JSON-decoded, validated form of a [`CompositeLayer`]
/// before its target id is resolved against an [`IsolatedTargets`] set.
pub struct CompositeLayerSpec {
    /// Caller-chosen target id (must name an ACTIVE target at composite time).
    pub id: String,
    /// Destination rect in output pixels (x, y, w, h), y-down.
    pub dst_px: [f32; 4],
    /// 0..1, applied to the premultiplied source. Default 1.
    pub opacity: f32,
}

/// Parse the seam's layers payload `[{id, dstPx: [x, y, w, h], opacity?}]`.
/// Pure — lives here (not in the wasm crate) so the contract is natively
/// unit-tested; rejects malformed shapes and non-finite numbers with a
/// positional message instead of letting a NaN rect reach the GPU pass.
/// Out-of-range opacity clamps to 0..1 rather than erroring.
pub fn parse_composite_layers(layers_json: &str) -> Result<Vec<CompositeLayerSpec>, String> {
    let value: moda_scene::value::Value =
        moda_scene::value::from_str(layers_json).map_err(|e| format!("layers JSON: {e}"))?;
    let arr = value
        .as_array()
        .ok_or_else(|| "layers JSON must be an array".to_string())?;
    arr.iter()
        .enumerate()
        .map(|(i, layer)| {
            let obj = layer
                .as_object()
                .ok_or_else(|| format!("layer {i} must be an object"))?;
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("layer {i}: non-empty string 'id' required"))?;
            let dst = obj
                .get("dstPx")
                .and_then(|v| v.as_array())
                .filter(|a| a.len() == 4)
                .ok_or_else(|| format!("layer {i}: 'dstPx' must be [x, y, w, h]"))?;
            let mut dst_px = [0f32; 4];
            for (j, v) in dst.iter().enumerate() {
                // Check finiteness AFTER the f32 cast: a finite-in-f64 value
                // like 1e39 overflows to f32::INFINITY and would reach the
                // vertex shader as an inf/NaN clip rect.
                let n = v
                    .as_f64()
                    .map(|n| n as f32)
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| format!("layer {i}: 'dstPx[{j}]' must be a finite number"))?;
                dst_px[j] = n;
            }
            let opacity = match obj.get("opacity") {
                None => 1.0,
                Some(v) => v
                    .as_f64()
                    .filter(|n| n.is_finite())
                    .ok_or_else(|| format!("layer {i}: 'opacity' must be a finite number"))?
                    .clamp(0.0, 1.0) as f32,
            };
            Ok(CompositeLayerSpec {
                id: id.to_string(),
                dst_px,
                opacity,
            })
        })
        .collect()
}

/// Parse the seam's optional premultiplied clear color `[r, g, b, a]`
/// (components 0..1, clamped). `None`/blank clears transparent — the
/// compositor's own default posture.
pub fn parse_composite_clear(clear_json: Option<&str>) -> Result<[f64; 4], String> {
    let Some(raw) = clear_json.filter(|s| !s.trim().is_empty()) else {
        return Ok([0.0, 0.0, 0.0, 0.0]);
    };
    let value: moda_scene::value::Value =
        moda_scene::value::from_str(raw).map_err(|e| format!("clear JSON: {e}"))?;
    let arr = value
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or_else(|| "clear JSON must be [r, g, b, a]".to_string())?;
    let mut clear = [0f64; 4];
    for (i, v) in arr.iter().enumerate() {
        let n = v
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(|| format!("clear[{i}] must be a finite number"))?;
        clear[i] = n.clamp(0.0, 1.0);
    }
    Ok(clear)
}

/// One compositor input: an isolated target placed at `dst_px`
/// (x, y, width, height in output pixels, y-down) at `opacity` (0..1,
/// applied to the premultiplied source). Layers composite in slice order
/// (painter's algorithm) — the demo's "footage + overlays + captions" stack.
pub struct CompositeLayer<'a> {
    pub target: &'a IsolatedTarget,
    pub dst_px: [f32; 4],
    pub opacity: f32,
}

/// The prototype's same-frame consumer of isolated targets: draws N target
/// textures into one output frame with premultiplied-over blending. This is
/// deliberately the SIMPLEST correct compositor — the V3 shared compositor
/// (track z-order, transitions, caption overlay) replaces it; what it proves
/// is that the targets are consumable as plain `wgpu` textures in the same
/// frame graph with the documented alpha convention.
pub struct IsolatedCompositor {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl IsolatedCompositor {
    /// Pipeline targets the engine's frame format at sample_count 1.
    pub fn new(engine: &Engine) -> Self {
        let device = &engine.gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("isolated composite"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("isolated composite bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("isolated composite layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        // Premultiplied over — the module-doc alpha convention.
        let premul = wgpu::BlendState {
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
        let pipeline = moda_text_draw::device::create_render_pipeline(device, &wgpu::RenderPipelineDescriptor {
            label: Some("isolated composite pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: engine.frame_format(),
                    blend: Some(premul),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("isolated composite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        IsolatedCompositor {
            pipeline,
            bgl,
            sampler,
        }
    }

    /// Draw `layers` in order into `output_view` (`output_px` sized, engine
    /// frame format, sample_count 1). `clear` is the premultiplied clear
    /// color for the pass, or `None` to composite over the view's existing
    /// contents. Encodes and submits one render pass.
    pub fn composite(
        &self,
        engine: &Engine,
        layers: &[CompositeLayer<'_>],
        output_view: &wgpu::TextureView,
        output_px: (u32, u32),
        clear: Option<[f64; 4]>,
    ) -> Result<()> {
        let device = &engine.gpu.device;
        let (out_w, out_h) = output_px;
        if out_w == 0 || out_h == 0 {
            bail!("composite output has zero pixel area ({out_w}x{out_h})");
        }
        // No layers and nothing to clear: skip the empty encoder + pass.
        if layers.is_empty() && clear.is_none() {
            return Ok(());
        }
        // Per-layer uniform + bind group, created per call: prototype-simple.
        // (The seam commit caches these per target; noted in the measurements
        // doc. Six tiny buffers/frame did not register in the demo timings.)
        let bind_groups: Vec<wgpu::BindGroup> = layers
            .iter()
            .map(|layer| {
                let [x, y, w, h] = layer.dst_px;
                // Output px (y-down) → clip space (y-up): top edge y maps to
                // +ndc, bottom edge y+h to -ndc; uv rides the quad corners.
                let rect = [
                    (x / out_w as f32) * 2.0 - 1.0,
                    1.0 - (y / out_h as f32) * 2.0,
                    ((x + w) / out_w as f32) * 2.0 - 1.0,
                    1.0 - ((y + h) / out_h as f32) * 2.0,
                ];
                let uniform: [f32; 8] = [
                    rect[0],
                    rect[1],
                    rect[2],
                    rect[3],
                    layer.opacity,
                    0.0,
                    0.0,
                    0.0,
                ];
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("isolated composite layer uniform"),
                    size: std::mem::size_of_val(&uniform) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buf.slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::cast_slice(&uniform));
                buf.unmap();
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("isolated composite layer bg"),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&layer.target.view),
                        },
                    ],
                })
            })
            .collect();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("isolated composite"),
        });
        {
            let load = match clear {
                Some([r, g, b, a]) => wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                None => wgpu::LoadOp::Load,
            };
            let profile_ts = crate::gpu_profile::pass_timestamps();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("isolated composite pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: crate::gpu_profile::render_writes(&profile_ts),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            for bg in &bind_groups {
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..6, 0..1);
            }
        }
        engine.gpu.queue.submit([encoder.finish()]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composite_layers_parse_shape_defaults_and_order() {
        let layers = parse_composite_layers(
            r#"[
                {"id": "footage", "dstPx": [0, 0, 1920, 1080]},
                {"id": "lower-third", "dstPx": [0, 810, 1920, 270], "opacity": 0.5}
            ]"#,
        )
        .expect("well-formed layers parse");
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].id, "footage");
        assert_eq!(layers[0].dst_px, [0.0, 0.0, 1920.0, 1080.0]);
        assert_eq!(layers[0].opacity, 1.0, "omitted opacity defaults to 1");
        assert_eq!(layers[1].id, "lower-third");
        assert_eq!(layers[1].opacity, 0.5);
    }

    #[test]
    fn composite_layers_reject_malformed_input() {
        // Not an array / not objects / missing or malformed fields.
        assert!(parse_composite_layers("{}").is_err());
        assert!(parse_composite_layers("[42]").is_err());
        assert!(parse_composite_layers(r#"[{"dstPx": [0, 0, 1, 1]}]"#).is_err());
        assert!(parse_composite_layers(r#"[{"id": "", "dstPx": [0, 0, 1, 1]}]"#).is_err());
        assert!(parse_composite_layers(r#"[{"id": "a", "dstPx": [0, 0, 1]}]"#).is_err());
        // Non-finite numbers must not reach the GPU pass — including values
        // finite in f64 but overflowing the f32 the shader consumes.
        assert!(parse_composite_layers(r#"[{"id": "a", "dstPx": [0, 0, 1, null]}]"#).is_err());
        assert!(parse_composite_layers(r#"[{"id": "a", "dstPx": [0, 0, 1e39, 1]}]"#).is_err());
        assert!(
            parse_composite_layers(r#"[{"id": "a", "dstPx": [0, 0, 1, 1], "opacity": "x"}]"#)
                .is_err()
        );
        // Out-of-range opacity clamps rather than erroring (an authoring nicety).
        let clamped =
            parse_composite_layers(r#"[{"id": "a", "dstPx": [0, 0, 1, 1], "opacity": 7}]"#)
                .expect("clamps");
        assert_eq!(clamped[0].opacity, 1.0);
    }

    #[test]
    fn composite_clear_parses_defaults_and_clamps() {
        assert_eq!(parse_composite_clear(None).unwrap(), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            parse_composite_clear(Some("  ")).unwrap(),
            [0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            parse_composite_clear(Some("[1, 1, 1, 1]")).unwrap(),
            [1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            parse_composite_clear(Some("[2, -1, 0.5, 1]")).unwrap(),
            [1.0, 0.0, 0.5, 1.0],
            "components clamp to 0..1"
        );
        assert!(parse_composite_clear(Some("[1, 1, 1]")).is_err());
        assert!(parse_composite_clear(Some("nope")).is_err());
    }
}

/// Whole-pixel contain placement shared by Edit preview and native export.
/// Dimensions have already passed target allocation validation.
pub fn contained_rect(source: (u32, u32), output: (u32, u32)) -> [f32; 4] {
    let scale = (output.0 as f64 / source.0 as f64).min(output.1 as f64 / source.1 as f64);
    let w = (source.0 as f64 * scale).round().clamp(1., output.0 as f64);
    let h = (source.1 as f64 * scale).round().clamp(1., output.1 as f64);
    [
        ((output.0 as f64 - w) / 2.).floor() as f32,
        ((output.1 as f64 - h) / 2.).floor() as f32,
        w as f32,
        h as f32,
    ]
}
