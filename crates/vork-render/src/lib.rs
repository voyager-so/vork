//! Unstable, internal: all Rust APIs in this crate are free to change.
//!
//! Retained, headless wgpu renderer. FrameSlot owns frame-private resources;
//! InstalledScene retains ordered compute state. Submit several slots before collecting.
//! install.rs builds the InstalledScene (materials.rs, pipelines.rs); graph.rs
//! compiles the pass order; frame.rs owns FrameSlot and walks it; values.rs
//! prepares per-frame values;
//! passes/ encodes each GPU pass.
mod tool_report;
#[cfg(any(target_os = "linux", target_os = "windows"))]
mod vulkan_runtime;
mod warnings;
pub use warnings::RenderWarning;
mod polygon;
mod subdivision;
mod displacement;
use geometry::EvaluatedGeometry;
mod animation;
pub mod ask;
mod deformation;
mod attributes;
mod splats;
mod buffers;
mod vdb;
mod dense;
mod bundles;
mod camera;
pub use camera::{CameraRecord, Orthographic};
mod compressed_textures;
mod gpu_bc7;
#[allow(dead_code)]
mod object_abi;
mod compute;
mod compute_trace;
mod implicit_acceleration;
mod computed_geometry;
mod curves;
mod points;
mod point_publication;
pub mod constants;
pub mod failure;
#[doc(hidden)]
pub mod disk_cache;
mod ffmpeg_cache;
mod display_stats;
pub use display_stats::Exposure;
mod exr_codec;
pub use exr_codec::ExrCodec;
pub use png_output::PngDepth;
#[doc(hidden)]
pub mod fuzz;
mod gpu_memory;
mod residency;
mod frame;
mod command_finish;
mod denoise;
mod geometry;
mod geometry_chain;
mod geometry_sampling;
mod geometry_reference;
mod geometry_stage_source;
mod geometry_topology;
mod graph;
mod import;
/// The renderer's glTF reader, for callers that inspect an asset before rendering it.
pub use gltf;
pub fn open_gltf(path: &Path) -> Result<gltf::Gltf> { import::open(path) }
mod identity;
mod install;
mod std_warm;
#[doc(hidden)]
pub mod specialization;
mod materials;
#[allow(dead_code)]
mod binding_abi;
mod forward_variant;
mod surface_native;
mod native;
mod resolve_native;
mod slang_forward;
mod passes;
mod pipelines;
mod plan;
mod probes;
mod output_run;
mod audio;
mod audio_output;
mod png_output;
mod programs;
mod hair;
mod fibre_transport;
mod compositor;
mod display_program;
pub mod output_device;
#[cfg(test)]
mod display_program_tests;
mod render_images;
mod ring;
mod rt;
mod gpu_tlas;
mod sampling;
mod shadow_envelope;
mod shadow_containment;
mod shadows;
pub mod shot;
pub mod session;
mod texture_source;
#[doc(hidden)]
pub mod footage;
mod catcher;
mod timing;
mod mesh_exit;
mod transmission;
mod values;
mod view;
mod views;
pub mod measure;
use animation::{Asset, Deform, FrameDeform, Influence};
use anyhow::{Context, Result, bail, ensure};
use bytemuck::{Pod, Zeroable};
pub use frame::{
    AovPixels, FloatOutputs, FrameOutput, FrameReadback, FrameSlot, FrameTiming, NonfiniteCounts, ReadbackOutputs,
    ReadbackPrecision,
};
use glam::{Mat4, Vec3};
use install::EvalNode;
pub use install::{InstallOptions, InstalledScene};
pub use frame::{Reservations, FrameRedo, FrameRefused};
use materials::{Material, TextureInstaller};
use pipelines::CachedPipelines;
use plan::EffectPlan;
use std::{
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, mpsc},
    time::{Duration, Instant},
};
use vork_scene::{Node, NodeKind, Scene};

mod emitters;
mod emitter_weights;
mod environment;
mod ies;
mod environment_node;
mod environment_program;
mod camera_program;
mod camera_dispatch_native;
mod propagation;
mod aov;
mod medium;
mod lights;
mod light_temperature;
mod targets;
mod canvas;
mod canvas_draw;
mod canvas_world;

use lights::{LightPlan, Light};

const RESOLVE: &str = include_str!("../shaders/resolve.slang");
pub use constants::FRAMES_IN_FLIGHT;
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
struct Vertex {
    POSITION: [f32; 3],
    NORMAL: [f32; 3],
    /// TEXCOORD_0 and TEXCOORD_1; material taps transform them per texture.
    TEXCOORD_0: [f32; 2],
    TEXCOORD_1: [f32; 2],
    /// Vec4 alignment; lane zero marks split generated face normals in deformation sources.
    pad: [f32; 2],
    COLOR_0: [f32; 4],
    TANGENT: [f32; 4],
}
impl Vertex {
    fn texcoord(&self, set: usize) -> [f32; 2] {
        match set {
            0 => self.TEXCOORD_0,
            1 => self.TEXCOORD_1,
            _ => unreachable!("unsupported TEXCOORD set"),
        }
    }
}
/// Only these channels change during skinning/morphing.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
#[allow(non_snake_case)]
struct DeformedVertex {
    POSITION: [f32; 3],
    NORMAL: [f32; 3],
    TANGENT: [f32; 4],
}
impl DeformedVertex {
    const BYTES: u64 = std::mem::size_of::<Self>() as u64;
    const WORDS: usize = std::mem::size_of::<Self>() / std::mem::size_of::<u32>();
    const NORMAL_WORD: usize = std::mem::offset_of!(Self, NORMAL) / std::mem::size_of::<u32>();
    const TANGENT_WORD: usize = std::mem::offset_of!(Self, TANGENT) / std::mem::size_of::<u32>();
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Object {
    model: Mat4,
    previous_model: Mat4,
    next_model: Mat4,
    opacity: [f32; 4],
    material_index: [u32; 4],
    /// x: named groups; y: ray visibility; z/w reserved.
    metadata: [u32; 4],
}
impl Object {
    fn new(model: Mat4) -> Self {
        Self {
            model,
            previous_model: model,
            next_model: model,
            opacity: [1., 0., 0., 0.],
            material_index: [0; 4],
            metadata: [0, constants::RAY_VISIBILITY_ALL, 0, 0],
        }
    }
}
// GPU residency stores affine rows, two poses and the attributes shading reads.
// Normal matrices are derived at the vertex, never retained per instance.
include!(concat!(env!("OUT_DIR"), "/gpu_object.rs"));
impl Object {
    fn gpu(&self) -> GpuObject {
        let rows = |m: Mat4| {
            let rows = m.transpose().to_cols_array_2d();
            [rows[0], rows[1], rows[2]]
        };
        GpuObject {
            model: rows(self.model),
            previous_model: rows(self.previous_model),
            next_model: rows(self.next_model),
            opacity: [self.opacity[0], self.opacity[1]],
            material_index: self.material_index[0],
            reserved: self.metadata[2],
            metadata: [self.metadata[0], if collapsed(self.model) { 0 } else { self.metadata[1] }],
            padding: [self.material_index[1], self.metadata[3]],
        }
    }
}
/// Singular placements are absent, including rank collapse through a parent.
fn collapsed(model: Mat4) -> bool {
    model.determinant() == 0.
}
/// Hidden mesh-local frames still run deformation/displacement preparation.
/// Supply a finite frame there; their placement/mask prevents any contribution.
fn inverse_or_identity(model: Mat4) -> Mat4 {
    if collapsed(model) { return Mat4::IDENTITY; }
    let inverse = model.inverse();
    if inverse.is_finite() {inverse} else {Mat4::IDENTITY}
}
fn gpu_objects(objects: &[Object]) -> Vec<GpuObject> {
    if objects.is_empty() {
        vec![GpuObject::zeroed()]
    } else {
        objects.iter().map(Object::gpu).collect()
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniform {
    vp: Mat4,
    light_vp: Mat4,
    eye: [f32; 4],
    background: [f32; 3],
    light_count: f32,
    radiance: [f32; 4],
    params: [f32; 4],
    flags: [f32; 4],
    previous_vp: Mat4,
    next_vp: Mat4,
    inverse_vp: Mat4,
    view: Mat4,
    /// Radius, strength, exact-f32 packed Fast GI ray/step counts, resolution divisor.
    ao: [f32; 4],
    projection: [f32; 4],
    /// x: sampled thin-lens aperture radius in metres (0: pinhole or unsampled);
    /// y: focus distance; zw: one internal pixel in clip units for ray-hit footprints.
    lens: [f32; 4],
    aperture: [f32; 4],
    /// xy: current pass iteration and count; z: exposure count, including nominal
    /// control passes; w: accumulated denoising-guide flags.
    temporal: [f32; 4],
    environment: [f32; 4],
    environment_rotation: Mat4,
    environment_color: [f32; 4],
    environment_mean: [f32; 4],
    /// x: scene time in seconds of this sample (material programs' ctx.time).
    scene: [f32; 4],
    /// Shadowed-by selection: x group mask, y explicit wildcard.
    /// For surface lighting, z enables per-light receiver masks; w reserved.
    shadow_links: [u32; 4],
    /// x: sampled shadow bit plus sequence row offset; y: rays/sample, z: full camera budget, w: subsurface flags.
    integrator: [u32; 4],
    /// Installed sequence offsets: x camera budget, y shutter sample budget; z screen trace limit (0 complete).
    /// w: visibility samples each exposure sample integrates in decoupled colour (the shade
    /// then writes the ambient controls), 0 otherwise.
    sampling: [u32; 4],
    /// x: ray mask plus LIGHT_LINK_ALL, y: LIGHT_LINK_WORD selection, z: raster transmission boundary limit, w: environment-only refraction.
    environment_links: [u32; 4],
    /// x: medium input word base in the common material buffer; y: ray emitter count;
    /// z: environment program input word base; w: environment visible to volume rays.
    medium: [u32; 4],
    /// x: render seed bits (f32 bitcast); yz: final output extent in pixels;
    /// w: display-program material word base.
    display: [f32; 4],
    /// x: compositor/denoiser source; y: nominal output time in seconds (view programs' ctx.time); z reserved;
    /// w sampled-aperture CoC gain in internal pixels.
    post: [f32; 4],
    composite_projection: Mat4,
    /// Aperture radius metres, reserved, focus depth metres, projective certificate.
    composite_optics: [f32; 4],
    /// NDC screen scale (xy), sampled aperture flag, lens sample count.
    composite_sensor: [f32; 4],
    /// Exposure open/close and current step open/close, absolute scene seconds.
    composite_shutter: [f32; 4],
    /// Step index/count, convergence sample count, spatial iterations per step.
    composite_step: [f32; 4],
    /// The camera sample this exposure sample's projection carries for every pixel:
    /// xy clip-space pixel jitter, zw camera-space lens offset in metres. The path
    /// integrator removes it and draws its own per pixel (path_sampler.slang).
    camera_sample: [f32; 4],
    camera_program: [u32; 4],
    /// Exact-field ray directory (offset/count), raster object count, reserved.
    field_geometry: [u32; 4],
}
pub struct Renderer {
    pub(crate) instance: wgpu::Instance,
    pub(crate) adapter: wgpu::Adapter,
    denoiser: Mutex<Option<denoise::Engine>>,
    command_finish: OnceLock<command_finish::CommandFinish>,
    timestamp_marker: OnceLock<timing::Marker>,
    /// Numerical volume references can vary the internal integration grid.
    /// Equal high-bounce, nonadaptive transport for numerical cross-provider tests.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub reference_transport_bounces: Option<u32>,
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub reference_transparent_bounces: Option<u32>,
    /// Synchronous qualification-only timestamps, never a renderer setting.
    #[cfg(test)]
    pub(crate) test_gpu_segment_times: Option<Mutex<Vec<(usize,usize,f64,&'static str)>>>,
    /// Replaces a view's per-sample camera evaluation estimate (passes/decoupled.rs).
    #[cfg(test)]
    pub(crate) evaluation_estimate: Option<u64>,
    /// Test only: constrain evaluation storage without falsifying backend residency.
    #[cfg(test)]
    pub(crate) evaluation_memory: Option<u64>,
    pub(crate) max_texture_allocation: u64,
    /// The largest device-local memory heap (Vulkan) or the recommended
    /// working set (Metal); u64::MAX where the backend reports neither.
    pub(crate) device_memory: u64,
    /// Initial native headroom, bounded by device memory. Camera scheduling uses
    /// a stable quota; querying driver residency per view adds frame latency.
    pub(crate) camera_memory: u64,
    /// Usage sampled with camera_memory; downstream storage charges only growth.
    pub(crate) camera_memory_usage_at_creation: u64,
    /// Whether qualification lowered the native startup headroom.
    pub(crate) camera_memory_override_lowered: bool,
    pub(crate) errors: gpu_memory::Errors,
    pub(crate) residency: std::sync::Arc<residency::Ledger>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    release_preparation: bool,
    /// Release shader preparation: a path install also compiles its kernel at the
    /// other key cases the bundle covers (ProgramKernel::release_cases).
    release_path_cases: bool,
    /// Only the private no-op check renderer may omit native code generation.
    check_only: bool,
    /// What this device was opened to provide (`Renderer::providing`).
    needs: DeviceNeeds,
    pub startup: Duration,
    pub startup_stages_ms: [(&'static str, f64); 4],
    pub pipeline_cache_status: &'static str,
    pub pipeline_cache_input_bytes: usize,
    pub pipeline_cache_rejection: Option<String>,
    cache: Option<wgpu::PipelineCache>,
    cache_path: PathBuf,
    disk_cache: disk_cache::Store,
    /// Owned pipeline-creation handle shared with background variant workers. It also saves the
    /// driver's pipeline cache and known-key ledger to the disk store.
    pub(crate) compiler: pipelines::Compiler,
    pipelines: pipelines::Registry,
    program_pipelines:
        Mutex<std::collections::HashMap<blake3::Hash, std::sync::Arc<CachedPipelines>>>,
    brdf: OnceLock<wgpu::TextureView>,
    ltc: OnceLock<wgpu::TextureView>,
}
/// What work needs from the device beyond what every device vork opens has. Opening a device with a need costs
/// whether or not the work uses it (a ray-capable Vulkan device starts the driver's ray runtime), so a device is
/// opened for the scene it will draw (`rt::device_needs`), and a process that keeps its device replaces it with
/// one that provides more when a later scene needs more (`Renderer::providing`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DeviceNeeds {
    /// Hardware ray queries: acceleration structures and the techniques that trace them.
    pub(crate) ray_queries: bool,
}
impl DeviceNeeds {
    /// Everything vork can use: a renderer opened before its scenes are known.
    pub(crate) const ALL: Self = Self { ray_queries: true };
    /// Both needs together.
    pub(crate) fn with(self, other: Self) -> Self {
        Self { ray_queries: self.ray_queries || other.ray_queries }
    }
    /// Whether a device opened for `self` provides all of `needs`.
    fn covers(self, needs: Self) -> bool {
        self.with(needs) == self
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Library users that never call settle_pipelines still persist new pipelines.
        let _ = self.settle_pipelines();
        crate::timing::forget(&self.device);
        moda_text_draw::device::forget_pipeline_host(&self.device);
    }
}
/// The GPU a process renders on: `vork render --gpu N` / `vork setup --gpu N` (through `select_gpu`) or
/// `VORK_GPU=N`, the Nth hardware GPU in PCI bus order (nvidia-smi's index). Unset, wgpu's high-performance
/// choice. One process per GPU is how a multi-GPU machine splits a frame range; the cache is shared, since its
/// device key ignores where a card sits. `cpu` explicitly selects a software adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuSelection { Hardware(usize), Cpu }
impl std::str::FromStr for GpuSelection {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> Result<Self> {
        if value == "cpu" { return Ok(Self::Cpu); }
        value.parse().map(Self::Hardware).map_err(|_| anyhow::anyhow!("invalid GPU selection {value:?}; use a hardware GPU index (0, 1, ...) or cpu"))
    }
}
impl std::fmt::Display for GpuSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Hardware(index) => write!(f, "{index}"), Self::Cpu => f.write_str("cpu") }
    }
}
static GPU_INDEX: OnceLock<GpuSelection> = OnceLock::new();
pub fn select_gpu(index: GpuSelection) {
    let _ = GPU_INDEX.set(index);
}
pub fn selected_gpu() -> Result<Option<GpuSelection>> {
    gpu_index_with_override(std::env::var("VORK_GPU").ok().as_deref(), GPU_INDEX.get().copied())
}
fn gpu_index_with_override(env: Option<&str>, explicit: Option<GpuSelection>) -> Result<Option<GpuSelection>> {
    if let Some(selection) = explicit { return Ok(Some(selection)); }
    gpu_index(env)
}
fn gpu_index(env: Option<&str>) -> Result<Option<GpuSelection>> {
    match env.map(str::trim) {
        None | Some("") => Ok(None),
        Some(value) => value.parse().map(Some),
    }
}
/// Hardware adapters in PCI bus order (software adapters such as lavapipe excluded); a stable sort, so
/// adapters that report no bus address keep the driver's order.
fn gpu_order(adapters: &[wgpu::AdapterInfo]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..adapters.len()).filter(|&i| adapters[i].device_type != wgpu::DeviceType::Cpu).collect();
    order.sort_by(|&a, &b| adapters[a].device_pci_bus_id.cmp(&adapters[b].device_pci_bus_id));
    order
}

// Adapter limits can exceed wgpu-core's fixed-capacity descriptor storage (the
// no-op adapter intentionally advertises allocation-sized limits).
fn bounded_device_limits(mut limits: wgpu::Limits) -> wgpu::Limits {
    limits.max_bind_groups = limits.max_bind_groups.min(wgpu::hal::MAX_BIND_GROUPS as u32);
    limits.max_vertex_buffers = limits.max_vertex_buffers.min(wgpu::hal::MAX_VERTEX_BUFFERS as u32);
    limits.max_color_attachments = limits.max_color_attachments.min(wgpu::hal::MAX_COLOR_ATTACHMENTS as u32);
    limits
}

#[test]
fn adapter_limits_respect_fixed_descriptor_capacities() {
    let limits = bounded_device_limits(wgpu::Limits {
        max_bind_groups: u32::MAX,
        max_vertex_buffers: u32::MAX,
        max_color_attachments: u32::MAX,
        ..Default::default()
    });
    // max_bind_groups also bounds bind groups in each pipeline layout.
    assert_eq!(limits.max_bind_groups, wgpu::hal::MAX_BIND_GROUPS as u32);
    assert_eq!(limits.max_vertex_buffers, wgpu::hal::MAX_VERTEX_BUFFERS as u32);
    assert_eq!(limits.max_color_attachments, wgpu::hal::MAX_COLOR_ATTACHMENTS as u32);
    let defaults = wgpu::Limits::default();
    assert_eq!(bounded_device_limits(defaults.clone()), defaults);
}

#[test]
fn gpu_selection_follows_pci_bus_order_and_skips_software_adapters() {
    let card = |bus: &str, device_type| wgpu::AdapterInfo {
        name: bus.into(), vendor: 0x10de, device: 0x27b8, device_type, device_pci_bus_id: bus.into(),
        driver: String::new(), driver_info: String::new(), backend: wgpu::Backend::Vulkan,
        subgroup_min_size: 32, subgroup_max_size: 32, transient_saves_memory: false,
    };
    let adapters = [card("0000:80:03.0", wgpu::DeviceType::DiscreteGpu), card("", wgpu::DeviceType::Cpu),
        card("0000:00:07.0", wgpu::DeviceType::DiscreteGpu), card("0000:80:02.0", wgpu::DeviceType::DiscreteGpu)];
    assert_eq!(gpu_order(&adapters), [2, 3, 0]);
    assert_eq!(gpu_index(Some(" 1 ")).unwrap(), Some(GpuSelection::Hardware(1)));
    assert_eq!(gpu_index(None).unwrap(), None);
    assert_eq!(gpu_index(Some("")).unwrap(), None);
    assert_eq!(gpu_index(Some("cpu")).unwrap(), Some(GpuSelection::Cpu));
    assert_eq!(gpu_index_with_override(Some("cpu"), Some(GpuSelection::Hardware(2))).unwrap(), Some(GpuSelection::Hardware(2)));
    assert_eq!(gpu_index_with_override(Some("invalid"), Some(GpuSelection::Cpu)).unwrap(), Some(GpuSelection::Cpu));
    for value in ["first", "CPU", "-1", "1.5", "lavapipe"] { assert!(gpu_index(Some(value)).is_err()); }
}

/// The backends this build can run. Each build embeds its platform's shaders
/// only (native.rs): Metal source on Apple, SPIR-V everywhere else.
fn supported_backends(selection: Option<&str>) -> Result<wgpu::Backends> {
    let apple = cfg!(target_vendor = "apple");
    let Some(selection) = selection else {
        return Ok(if apple { wgpu::Backends::METAL } else { wgpu::Backends::METAL | wgpu::Backends::VULKAN });
    };
    anyhow::ensure!(selection.split(',').all(|token|
        matches!(token.trim().to_ascii_lowercase().as_str(), "vulkan" | "vk" | "metal" | "mtl")),
        "unsupported WGPU_BACKEND={selection:?}: vork supports only Vulkan and Metal; unset WGPU_BACKEND or select vulkan or metal");
    anyhow::ensure!(!apple || selection.split(',').all(|token|
        matches!(token.trim().to_ascii_lowercase().as_str(), "metal" | "mtl")),
        "unsupported WGPU_BACKEND={selection:?}: vork on macOS renders with Metal only, and this build carries no Vulkan shaders; unset WGPU_BACKEND or select metal");
    Ok(wgpu::Backends::from_comma_list(selection))
}

#[test]
fn backend_selection_refuses_unsupported_and_unknown_tokens() {
    for selection in ["dx12", "gl", "webgpu", "noop", "vulkan,dx12", "vulkan,typo", ""] {
        assert!(supported_backends(Some(selection)).unwrap_err().to_string().contains("only Vulkan and Metal"));
    }
    if cfg!(target_vendor = "apple") {
        // An Apple build carries Metal shaders only: Vulkan is refused by name.
        assert_eq!(supported_backends(None).unwrap(), wgpu::Backends::METAL);
        assert_eq!(supported_backends(Some("Metal")).unwrap(), wgpu::Backends::METAL);
        for selection in ["vulkan", "VK, Metal", "metal,vk"] {
            let error = supported_backends(Some(selection)).unwrap_err().to_string();
            assert!(error.contains("Metal only") && error.contains("select metal"), "{error}");
        }
    } else {
        assert_eq!(supported_backends(None).unwrap(), wgpu::Backends::VULKAN | wgpu::Backends::METAL);
        assert_eq!(supported_backends(Some("VK, Metal")).unwrap(), supported_backends(None).unwrap());
    }
}

impl Renderer {
    /// Wait for GPU work, reporting a lost or hung device as an error (gpu_memory::wait).
    pub(crate) fn wait(&self, submission: Option<wgpu::SubmissionIndex>) -> Result<()> {
        gpu_memory::wait(&self.device, &self.queue, &self.errors, submission)
    }
    /// Use BGRA storage when advertised, with RGBA as the portable fallback.
    pub fn output_format(&self) -> wgpu::TextureFormat {
        if self
            .device
            .features()
            .contains(wgpu::Features::BGRA8UNORM_STORAGE)
        {
            wgpu::TextureFormat::Bgra8Unorm
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        }
    }

    /// Populate the device's persistent cache for shipped built-in shaders.
    pub fn warm_shaders(&self) -> Result<()> {
        pipelines::warm_builtin(self)?;
        passes::decoupled::warm_shaders(self);
        self.warm_path_engines()?;
        std_warm::warm(self)?;
        // A release's shaders say nothing of the FFmpeg on the machine that prepared them.
        if !self.release_preparation { ffmpeg_cache::warm(); }
        self.finish_shader_warmup()
    }

    /// Warm the init starter at the default render profile and the prebuilt path
    /// engine, which is the same few pipelines for every scene. Other scenes and
    /// profiles compile on demand, or through exhaustive setup.
    pub fn warm_startup_shaders(&self) -> Result<()> {
        self.warm_path_engines()?;
        std_warm::warm_startup(self, pipelines::VariantCompile::Fallback)?;
        ffmpeg_cache::warm();
        self.finish_shader_warmup()
    }

    /// The prebuilt path engine's driver pipelines, so the first path render of
    /// any scene the engine draws finds them in this driver and GPU's cache.
    fn warm_path_engines(&self) -> Result<()> {
        vork_slang::progress::phase("Setup: warming the prebuilt path engine", 0);
        passes::path::warm_engines(self)
    }

    fn finish_shader_warmup(&self) -> Result<()> {
        vork_slang::progress::phase("Setup: finishing driver pipelines and saving cache", 0);
        self.settle_pipelines()?;
        if self.cache.is_some() {
            let saved = self.compiler.saved();
            anyhow::ensure!(saved.0 == self.compiler.created()
                && disk_cache::read_mapped(&self.cache_path).is_some_and(|data| Some(blake3::hash(&data)) == saved.1),
                "setup could not persist the complete driver cache; increase VORK_CACHE_BYTES or free disk space");
        }
        Ok(())
    }

    /// Without a persistent pipeline cache (a Metal GPU without binary archives),
    /// remember completed shader work in the same build/device-keyed store,
    /// without pretending to inspect Apple's shared shader cache.
    pub fn shader_cache_cold(&self) -> bool {
        matches!(self.pipeline_cache_status, "empty" | "rejected")
            || (self.pipeline_cache_status == "unsupported"
                && disk_cache::read(&self.disk_cache.path("shader-warm", 1, blake3::hash(b""))).is_none())
    }

    /// Report actual device/tool availability before warming shaders.
    pub fn setup_report(&self, out: &mut impl std::io::Write) -> Result<()> {
        self.setup_report_verbose(out, false)
    }
    /// Include captured subprocess diagnostics only when explicitly requested.
    pub fn setup_report_verbose(&self, out: &mut impl std::io::Write, verbose: bool) -> Result<()> {
        macro_rules! report {
            ($($arg:tt)*) => { writeln!(out, $($arg)*)? };
        }
        let driver = self.adapter_info.driver_info.trim();
        if driver.is_empty() {
            report!("GPU: {} ({:?}; driver version unavailable through the graphics API; this does not indicate a missing driver)", self.adapter_info.name, self.adapter_info.backend);
        } else {
            report!("GPU: {} ({:?}; driver {driver})", self.adapter_info.name, self.adapter_info.backend);
        }
        if self.adapter_info.device_type == wgpu::DeviceType::Cpu {
            report!("GPU: software rendering on the CPU; expect slow renders. Install your GPU vendor's Vulkan driver for hardware rendering.");
        }
        report!("Ray queries: {}", if self.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) { "available; the renderer can select scene queries" } else { "unavailable; profiles refuse effects that need scene queries" });
        let mut failures = Vec::new();
        match vork_slang::Compiler::discover().version() {
            Ok(version) => report!("Slang: {version} ready"),
            Err(error) => { report!("Slang: {error:#}"); failures.push("Slang"); }
        }
        match denoise::open_fast(self) {
            Ok(()) => report!("Denoiser: OIDN on the GPU ({})", self.adapter_info.name),
            Err(reason) => match denoise::check_runtime() {
                Ok(()) => report!("Denoiser: no fast GPU denoiser for {} ({reason}). std/draft and std/final turn denoising off and use more samples; --set denoise=true uses the slow CPU denoiser.", self.adapter_info.name),
                Err(error) => { report!("Denoiser: {error:#}. Reinstall the complete archive, or set VORK_OIDN to the OIDN 2.5.1 library."); report!("Rendering will continue without OIDN; std/draft and std/final use more samples."); }
            }
        }
        if cfg!(target_os = "linux") {
            report!("Video: native NVENC on NVIDIA, then FFmpeg NVENC after readback, then libx264, with a diagnostic at each step; other Linux adapters use libx264 at its quality target. MOV uses prores_ks.");
        }
        if cfg!(target_os = "windows") {
            report!("Video: FFmpeg NVENC after readback when usable; otherwise software libx264. MOV uses software prores_ks.");
        }
        if cfg!(target_os = "macos") {
            report!("Video: system VideoToolbox for MP4; MOV may need ffmpeg prores_ks as fallback.");
        }
        match std::process::Command::new("ffmpeg").args(["-hide_banner", "-encoders"]).output() {
            Ok(output) if output.status.success() => {
                let listing = String::from_utf8_lossy(&output.stdout);
                if cfg!(target_os = "windows") || (cfg!(target_os = "linux") && self.adapter_info.vendor == constants::NVIDIA_VENDOR_ID) {
                    if listing.contains("h264_nvenc") {
                        tool_report::encoder_probe(out, "FFmpeg h264_nvenc", vork_encode::probe_ffmpeg_nvenc(256, 256), verbose)?;
                    } else {
                        report!("FFmpeg h264_nvenc: not installed; readback MP4 uses libx264 if available");
                    }
                }
                // libx264 is an encoder, never the muxer: FFmpeg itself muxes native NVENC's stream (vork_encode::validate_video_output).
                let listed = listing.contains("libx264");
                report!("FFmpeg libx264: {}", if cfg!(target_os = "macos") {
                    if listed { "available (listed); optional, native VideoToolbox MP4 needs no FFmpeg" } else { "absent (optional; native VideoToolbox MP4 needs no FFmpeg)" }
                } else if cfg!(target_os = "windows") {
                    if listed { "software H.264 available (listed; initialization checked when opening output); required for MP4, as the fallback when NVENC is unusable" } else { "WARNING: libx264 missing; MP4 requires it as the software fallback. Install ffmpeg with libx264 (ffmpeg-release-essentials.zip from https://www.gyan.dev/ffmpeg/builds/) and add bin to PATH" }
                } else if self.adapter_info.vendor == constants::NVIDIA_VENDOR_ID {
                    if listed { "software H.264 fallback available (listed; initialization checked when opening output); used only when NVENC is unusable. Native NVENC needs FFmpeg to mux, not libx264" } else { "absent; MP4 has no software fallback if NVENC is unusable. Native NVENC needs FFmpeg to mux, not libx264" }
                } else if listed {
                    "software H.264 available (listed; initialization checked when opening output); it encodes this adapter's MP4"
                } else {
                    "WARNING: libx264 missing; this adapter's MP4 is encoded by libx264. Install an ffmpeg build with libx264"
                });
                report!("FFmpeg prores_ks: {}", if listing.contains("prores_ks") { "MOV available (listed; initialization checked when opening output)" } else { "prores_ks unavailable; install an ffmpeg build with prores_ks" });
            }
            _ if cfg!(target_os = "macos") => report!("FFmpeg: missing or unusable; native VideoToolbox MP4 is ready without FFmpeg; MOV fallback needs ffmpeg prores_ks (brew install ffmpeg). Encoder availability is checked when opening video output."),
            _ => report!("FFmpeg: missing or unusable; required for MP4/MOV, including Linux native NVENC muxing. Install ffmpeg (Ubuntu: sudo apt install ffmpeg; Windows: winget install Gyan.FFmpeg, or extract ffmpeg-release-essentials.zip from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH). PNG/EXR remain available."),
        }
        report!("Cache: {}", self.cache_path.parent().unwrap_or(&self.cache_path).display());
        match self.pipeline_cache_status {
            "unsupported" if self.adapter_info.device_type == wgpu::DeviceType::Cpu => report!("Pipeline cache: disabled on software adapters to avoid a lavapipe compiler crash. Portable shader artifacts and Mesa's own shader disk cache remain available."),
            "unsupported" if self.adapter_info.backend == wgpu::Backend::Metal => report!("Pipeline cache: this Metal GPU has no binary archives; Apple's shared shader cache may keep what setup compiles but evicts large pipelines. Compiled shader assets use the cache above."),
            "unsupported" => report!("Pipeline cache: driver does not expose persistent pipeline caching; compiled shader assets still use the cache above."),
            status => report!("Pipeline cache: {status} for this build and GPU driver"),
        }
        anyhow::ensure!(failures.is_empty(), "setup tools failed: {}; follow the fixes above", failures.join(", "));
        Ok(())
    }

    /// A renderer for scenes not yet known: its device provides everything vork can use.
    pub fn new() -> Result<Self> {
        Self::providing(None, DeviceNeeds::ALL)
    }
    /// Whether this renderer's device provides `needs`.
    pub(crate) fn provides(&self, needs: DeviceNeeds) -> bool {
        self.needs.covers(needs)
    }
    /// The renderer for work with `needs`: `held` when its device provides them, otherwise a new device that
    /// provides them and everything `held` did, so a process's device only ever gains. Nothing made on `held`
    /// can be used with the new device: the caller drops its installations first when `held` does not provide
    /// `needs`. The pipelines `held` compiled reach the new device through the disk pipeline cache, which
    /// `held` saves when it is dropped here, before the new device reads it.
    pub(crate) fn providing(held: Option<Self>, needs: DeviceNeeds) -> Result<Self> {
        let needs = match held {
            Some(held) if held.provides(needs) => return Ok(held),
            Some(held) => {
                let needs = held.needs.with(needs);
                drop(held);
                needs
            }
            None => needs,
        };
        pollster::block_on(Self::create(needs, false))
    }
    /// The device `providing` would replace this one with, opened while this one and what is installed on it
    /// stay: a session proves a candidate on it and only then lets this one go. This renderer's pipelines are
    /// saved first, so the new device reads them from the disk pipeline cache.
    pub(crate) fn beside(&self, needs: DeviceNeeds) -> Result<Self> {
        self.settle_pipelines()?;
        pollster::block_on(Self::create(self.needs.with(needs), false))
    }
    /// Compile release assets without a GPU or any native driver execution. The
    /// no-op device retains normal scene planning and Slang/cache paths. This
    /// entry point never renders and is separate from production device admission.
    #[doc(hidden)]
    pub fn prepare_release_shaders(scenes: &[PathBuf]) -> Result<()> {
        let mut renderer = pollster::block_on(Self::create(DeviceNeeds::ALL, true))?;
        // A path kernel's key also names facts that are not the scene's materials:
        // its light kinds and its camera proof. Another scene with the same
        // materials finds a prepared kernel only if those match too, so every path
        // kernel prepared here is prepared at each covered case of both.
        renderer.release_path_cases = true;
        renderer.warm_shaders()?;
        for path in scenes {
            // Use real installed material unions, bindings and program links,
            // rather than a Cartesian product of independent capability bits.
            let scene = load_scene_document(path)?;
            for profile in ["std/final", "std/path"] {
                let mut scene = scene.clone();
                scene.render.resolution = [64, 64];
                scene.render.settings = vork_scene::Profile::load(profile)?.resolve(&Default::default())?;
                // Stills can draw the setup superset; movies request the exact
                // material variant. Prepare both through their existing modes.
                for variants in [pipelines::VariantCompile::Setup, pipelines::VariantCompile::Inline] {
                    let installed = renderer.install_with_options(scene.clone(), path.parent().unwrap_or(Path::new(".")),
                        InstallOptions { variants, ..Default::default() })
                        .with_context(|| format!("prepare {} with {profile}", path.display()))?;
                    let _frame = installed.create_frame_slot(&renderer, Default::default())?;
                }
            }
        }
        renderer.finish_shader_warmup()
    }
    /// Run scene installation against the CPU-only no-op backend. No driver is
    /// opened and no frame is submitted; all install diagnostics are shared.
    pub fn check(path: &Path, profile: &vork_scene::Profile, shaders: bool) -> Result<()> {
        let warnings = warnings::Collector::default();
        let _scope = warnings.enter();
        let settings = profile.resolve(&Default::default()).map_err(|error| error.context(failure::FailureKind::Usage))?;
        let scene = load_with_settings(path, settings, &profile.name, None)?;
        let base = path.parent().unwrap_or(Path::new("."));
        if audio::has_audio(&scene.nodes) { audio::Mix::new(&scene, base).map_err(failure::scene)?; }
        let mut renderer = pollster::block_on(Self::create(DeviceNeeds::ALL, true)).map_err(failure::setup)?;
        renderer.check_only = !shaders;
        // A 2D canvas: its fonts load and its text lays out (Moda's text engine, CPU side).
        if scene.is_canvas() {
            return canvas::check(&scene, base, &renderer).with_context(|| format!("check scene {}", path.display())).map_err(failure::scene);
        }
        let installed = renderer.install_with_options(scene, base,
            InstallOptions { variants: pipelines::VariantCompile::Setup, ..Default::default() })
            .with_context(|| format!("install scene {}", path.display()))?;
        shot::check_timeline(&installed)?;
        Ok(())
    }

    async fn create(needs: DeviceNeeds, prepare_release: bool) -> Result<Self> {
        let start = Instant::now();
        // One process-wide instance: wgpu-core numbers devices per instance, so
        // separate instances would give every Renderer the same device id and
        // mix their timing receipts (one Renderer's drop forgot another's).
        static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
        static INSTANCE_INIT: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let backends = if prepare_release { wgpu::Backends::NOOP }
            else { supported_backends(std::env::var("WGPU_BACKEND").ok().as_deref())? };
        let make_instance = || -> Result<wgpu::Instance> {
            let mut flags = wgpu::InstanceFlags::default();
            if !cfg!(debug_assertions) {
                // Only engine-owned indexed draw commands are indirect: index ranges
                // come from installed meshes and instance counts are capacity-clamped
                // by compute/camera culling. Authored programs cannot write commands.
                // Avoid wgpu's eager validation pipelines and per-draw validation pass.
                flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
            }
            let descriptor = wgpu::InstanceDescriptor {
                backends,
                flags: if prepare_release { flags } else { flags.with_env() },
                backend_options: if prepare_release {
                    wgpu::BackendOptions { noop: wgpu::NoopBackendOptions { enable: true }, ..Default::default() }
                } else { Default::default() },
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            };
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            if !prepare_release { return vulkan_runtime::instance(descriptor); }
            Ok(wgpu::Instance::new(descriptor))
        };
        let preparation_instance;
        let instance = if prepare_release {
            preparation_instance = make_instance()?;
            &preparation_instance
        } else {
            if INSTANCE.get().is_none() {
                let _guard = INSTANCE_INIT.lock().map_err(|_| anyhow::anyhow!("GPU instance initialization lock poisoned"))?;
                if INSTANCE.get().is_none() {
                    // Failed initialization remains retryable in library callers.
                    let _ = INSTANCE.set(make_instance().map_err(|error| error.context(rt::adapter_help()))?);
                }
            }
            INSTANCE.get().unwrap()
        };
        let instance_ms = start.elapsed().as_secs_f64() * 1000.;
        let adapter_start = Instant::now();
        let selected = if prepare_release { None } else { selected_gpu()? };
        let adapter = if selected == Some(GpuSelection::Cpu) {
            instance.enumerate_adapters(backends).await.into_iter().find(|adapter| {
                adapter.get_info().device_type == wgpu::DeviceType::Cpu
                    && adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | rt::HIT_BUFFER_FEATURES)
            }).ok_or_else(|| anyhow::anyhow!("No CPU Vulkan adapter with ray queries and storage buffer arrays. {}", rt::adapter_help()))?
        } else if let Some(GpuSelection::Hardware(index)) = selected {
            let adapters = instance.enumerate_adapters(backends).await;
            let infos: Vec<_> = adapters.iter().map(wgpu::Adapter::get_info).collect();
            let order = gpu_order(&infos);
            let Some(&chosen) = order.get(index) else {
                anyhow::bail!("GPU {index} was requested (--gpu or VORK_GPU), but this machine has {} GPU(s); use 0 to {}",
                    order.len(), order.len().saturating_sub(1));
            };
            adapters.into_iter().nth(chosen).expect("index from the same enumeration")
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await.map_err(|error| {
                    #[cfg(any(target_os = "linux", target_os = "windows"))]
                    {
                        let diagnostic = vulkan_runtime::adapter_failure(instance, &error);
                        return anyhow::Error::new(error).context(diagnostic);
                    }
                    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
                    anyhow::Error::new(error).context("No usable GPU adapter exposed by the platform driver; check the graphics driver and device access (Metal requires Apple Silicon)")
                })?
        };
        let adapter_ms = adapter_start.elapsed().as_secs_f64() * 1000.;
        if !prepare_release { rt::require_adapter(&adapter.get_info(), adapter.features())?; }
        if !prepare_release && adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            eprintln!("GPU: software rendering on the CPU; expect slow renders.");
            eprintln!("First-run CPU shader compilation can take several minutes. vork's pipeline cache is disabled due to a lavapipe compiler crash; Mesa's own shader disk cache can still speed later runs.");
        }
        let device_start = Instant::now();
        // Vulkan exposes the actual single-allocation limit; it is independent
        // of buffer binding budgets and texture dimensions.
        #[allow(unused_mut)]
        let mut max_texture_allocation = u64::MAX;
        #[allow(unused_mut)]
        let mut device_memory = u64::MAX;
        #[cfg(target_os = "linux")]
        if let Some(hal) = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() } {
            let mut maintenance = ash::vk::PhysicalDeviceMaintenance3Properties::default();
            let mut properties = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut maintenance);
            unsafe { hal.shared_instance().raw_instance().get_physical_device_properties2(hal.raw_physical_device(), &mut properties); }
            max_texture_allocation = maintenance.max_memory_allocation_size;
            let memory = unsafe { hal.shared_instance().raw_instance().get_physical_device_memory_properties(hal.raw_physical_device()) };
            device_memory = memory.memory_heaps_as_slice().iter()
                .filter(|heap| heap.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL))
                .map(|heap| heap.size).max().unwrap_or(u64::MAX);
        }

        let mut info = adapter.get_info();
        if prepare_release {
            info.backend = if cfg!(target_os = "macos") { wgpu::Backend::Metal } else { wgpu::Backend::Vulkan };
            info.device_type = wgpu::DeviceType::DiscreteGpu;
            device_memory = 8 << 30;
        }
        // The scene material table indexes texture and sampler arrays per fragment.
        let material_table = wgpu::Features::TEXTURE_BINDING_ARRAY
            | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY;
        let features = adapter.features()
            & (wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                | material_table
                | wgpu::Features::INDIRECT_FIRST_INSTANCE
                | wgpu::Features::PIPELINE_CACHE
                | wgpu::Features::PASSTHROUGH_SHADERS
                | wgpu::Features::PRIMITIVE_INDEX
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                // Opening a ray-capable Vulkan device initializes the driver's
                // ray runtime even when no ray pipeline is ever created.
                | if needs.ray_queries {
                    wgpu::Features::EXPERIMENTAL_RAY_QUERY
                } else {
                    wgpu::Features::empty()
                }
                // Ray-hit surfaces read raster geometry through a buffer array.
                | crate::rt::HIT_BUFFER_FEATURES
                // Encoder/compositor surfaces may use BGRA storage whenever
                // the adapter advertises it, independently of its backend.
                | wgpu::Features::BGRA8UNORM_STORAGE);
        ensure!(
            features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
                && adapter
                    .get_texture_format_features(wgpu::TextureFormat::Rgba32Uint)
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4),
            "4x integer object ID target unsupported"
        );
        ensure!(
            features.contains(wgpu::Features::PASSTHROUGH_SHADERS),
            "the renderer's Slang shaders need passthrough shader modules; adapter {} lacks them",
            info.name
        );
        ensure!(
            features.contains(wgpu::Features::TIMESTAMP_QUERY),
            "GPU timestamp queries are required"
        );
        ensure!(
            features.contains(crate::rt::HIT_BUFFER_FEATURES),
            "camera and ray hits read geometry through buffer arrays; adapter {} lacks {:?}",
            info.name,
            crate::rt::HIT_BUFFER_FEATURES - features
        );
        ensure!(
            features.contains(material_table),
            "the material table needs {:?}; adapter {} lacks it",
            material_table - features,
            info.name
        );
        let mut limits = bounded_device_limits(adapter.limits());
        if prepare_release {
            // The no-op adapter checks shaders, not hardware admission. Keep a
            // finite ceiling, but admit production geometry (water needs 296 MiB
            // in one binding). Real rendering uses the selected adapter limits.
            limits.max_storage_buffer_binding_size = 1 << 30;
            limits.max_buffer_size = 1 << 30;
            limits.max_texture_dimension_2d = 16384;
            limits.min_uniform_buffer_offset_alignment = 256;
            limits.min_storage_buffer_offset_alignment = 256;
        }
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("vork"),
            required_features: features,
            required_limits: limits,
            // SAFETY: the only experimental feature requested is ray queries, used
            // through wgpu's validated BLAS/TLAS and WGSL ray-query API.
            experimental_features: if features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            ..Default::default()
        };
        #[cfg(target_os = "linux")]
        let native = if info.backend == wgpu::Backend::Vulkan
            && info.vendor == constants::NVIDIA_VENDOR_ID
        {
            // SAFETY: create from this adapter, retaining all wgpu features/limits;
            // enable only an extension advertised by the physical device.
            unsafe {
                let hal = adapter
                    .as_hal::<wgpu::hal::api::Vulkan>()
                    .context("Vulkan adapter unavailable")?;
                let extensions = hal
                    .shared_instance()
                    .raw_instance()
                    .enumerate_device_extension_properties(hal.raw_physical_device())?;
                let supported = extensions.iter().any(|ext| {
                    std::ffi::CStr::from_ptr(ext.extension_name.as_ptr())
                        == ash::khr::external_semaphore_fd::NAME
                });
                if supported {
                    let device = hal.open_with_callback(
                        features,
                        &descriptor.required_limits,
                        &descriptor.memory_hints,
                        Some(Box::new(|args| {
                            args.extensions.push(ash::khr::external_semaphore_fd::NAME)
                        })),
                    )?;
                    Some(
                        adapter.create_device_from_hal::<wgpu::hal::api::Vulkan>(
                            device,
                            &descriptor,
                        )?,
                    )
                } else {
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(target_os = "linux"))]
        let native: Option<(wgpu::Device, wgpu::Queue)> = None;
        let (device, queue) = if let Some(device) = native {
            device
        } else {
            adapter.request_device(&descriptor).await?
        };
        // Apple GPUs share system memory; Metal reports how much one device should use.
        #[cfg(target_os = "macos")]
        if let Some(hal) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
            use objc2_metal::MTLDevice;
            device_memory = hal.raw_device().recommendedMaxWorkingSetSize();
        }
        let errors = gpu_memory::Errors::install(&device);
        let residency = gpu_memory::ledger(&device);
        let device_ms = device_start.elapsed().as_secs_f64() * 1000.;
        let cache_start = Instant::now();
        // Build identity covers all shader/generator/state inputs; no per-blob
        // content input is needed. The store adds the device and entry format.
        let disk_cache = disk_cache::Store::new(&info);
        let cache_path = disk_cache.path("pipeline", 1, blake3::hash(b""));
        let mut pipeline_cache_status = "unsupported";
        let mut pipeline_cache_input_bytes = 0;
        let mut pipeline_cache_rejection = None;
        let mut saved_blob = None;
        // Software adapters (in practice lavapipe) compile uncached: built through a pipeline
        // cache object, decoupled_programs takes a form that hits a gallivm bug (a uniform storage
        // descriptor is read before the active-lane check) and segfaults. Drop once CI's Mesa fixes it.
        let cache = if !prepare_release && features.contains(wgpu::Features::PIPELINE_CACHE) && info.device_type != wgpu::DeviceType::Cpu {
            let data = disk_cache::read_pipeline(&cache_path);
            pipeline_cache_input_bytes = data.as_ref().map_or(0, |data| data.len());
            saved_blob = data.as_deref().map(blake3::hash);
            let cache_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            // SAFETY: only our own opaque wgpu cache data, keyed by shader, ABI and device.
            let candidate = unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("vork disk cache"),
                    data: data.as_deref(),
                    fallback: false,
                })
            };
            if let Some(error) = cache_scope.pop().await {
                pipeline_cache_status = "rejected";
                pipeline_cache_rejection = Some(error.to_string());
                // Preserve recovery while exposing why the previous cache was discarded.
                // SAFETY: no opaque data supplied to this empty cache.
                Some(unsafe {
                    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                        label: Some("vork recovered cache"),
                        data: None,
                        fallback: true,
                    })
                })
            } else {
                pipeline_cache_status = if data.is_some() { "accepted" } else { "empty" };
                Some(candidate)
            }
        } else {
            None
        };
        // Pipelines the accepted blob already holds; an empty or rejected blob holds none.
        let known_path = disk_cache.path("pipeline-keys", 1, blake3::hash(b""));
        // Metal GPUs without binary archives have no blob: Apple's shared shader cache
        // holds what earlier processes (`vork setup`) compiled, recorded by the
        // shader-warm marker (shader_cache_cold). It can evict large pipelines.
        let warm_system_cache = info.backend == wgpu::Backend::Metal
            && disk_cache::read(&disk_cache.path("shader-warm", 1, blake3::hash(b""))).is_some();
        let known = pipelines::Known::load(known_path, pipeline_cache_status == "accepted" || warm_system_cache);
        let compiler = pipelines::Compiler::new(&device, cache.as_ref(), known, pipelines::Blob {
            path: cache_path.clone(),
            marker: (pipeline_cache_status == "unsupported").then(|| disk_cache.path("shader-warm", 1, blake3::hash(b""))),
            loaded: saved_blob,
            save: !prepare_release,
        });
        let pipelines = pipelines::Registry::default();
        errors.check()?;
        // Qualification can lower the startup headroom on a larger device. It
        // cannot increase native memory or bypass allocation/binding limits.
        let camera_memory_override=match std::env::var("VORK_CAMERA_MEMORY_BYTES") {
            Ok(value)=>Some(value.parse::<u64>().context("VORK_CAMERA_MEMORY_BYTES must be an unsigned byte count")?),
            Err(std::env::VarError::NotPresent)=>None,
            Err(error)=>return Err(error.into()),
        };
        let (camera_memory, camera_memory_usage_at_creation, camera_memory_override_lowered) =
            residency::camera_budget(&device, device_memory, camera_memory_override);
        // Moda's 2D engine creates its pipelines on this device through the same compiler and cache.
        moda_text_draw::device::set_pipeline_host(&device, std::sync::Arc::new(compiler.clone()));
        Ok(Self {
            instance: instance.clone(),
            adapter,
            denoiser: Mutex::new(None),
            command_finish: OnceLock::new(),
            timestamp_marker: OnceLock::new(),
            #[cfg(any(test, feature = "test-support"))]
            reference_transport_bounces: None,
            #[cfg(any(test, feature = "test-support"))]
            reference_transparent_bounces: None,
            #[cfg(test)]
            test_gpu_segment_times: None,
            #[cfg(test)]
            evaluation_estimate: None,
            #[cfg(test)]
            evaluation_memory: None,
            errors,
            device,
            queue,
            adapter_info: info,
            release_preparation: prepare_release,
            release_path_cases: false,
            check_only: false,
            needs,
            max_texture_allocation,
            device_memory,
            camera_memory,
            camera_memory_usage_at_creation,
            camera_memory_override_lowered,
            residency,
            startup: start.elapsed(),
            startup_stages_ms: [
                ("instance", instance_ms),
                ("adapter", adapter_ms),
                ("device", device_ms),
                ("cache", cache_start.elapsed().as_secs_f64() * 1000.),
            ],
            pipeline_cache_status,
            pipeline_cache_input_bytes,
            pipeline_cache_rejection,
            cache,
            cache_path,
            disk_cache,
            compiler,
            program_pipelines: Mutex::new(Default::default()),
            pipelines,
            brdf: OnceLock::new(),
            ltc: OnceLock::new(),
        })
    }
    /// Pipelines for a scene shader variant with material programs, kept for
    /// the process lifetime per variant (the driver pipeline cache persists them).
    fn program_pipelines(
        &self,
        resolve:&str,
        raster_samples:u32,
        ray_shadows:bool,
        slang:slang_forward::SlangScene,
        geometry:&std::collections::HashSet<pipelines::GeometryKey>,
        kernels:&[pipelines::ProgramKernel],
    )->Result<std::sync::Arc<CachedPipelines>> {
        let scope=self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let candidate=std::sync::Arc::new(CachedPipelines::with_sources(self,resolve,raster_samples,ray_shadows,slang));
        if let Some(error)=pollster::block_on(scope.pop()) {
            // Slang accepted the program, so its own diagnostics are past: what the device's
            // validation then refuses is a device or renderer limit, after whatever it recorded.
            self.errors.check()?;
            return Err(anyhow::anyhow!("program shader: {error}").context(failure::FailureKind::Device));
        }
        // A check-only install never executes pipelines or reuses their registry.
        // In particular, it must not capture native compiler inputs or receipts.
        if self.check_only {return Ok(candidate);}
        // Requested kernels are scene inputs. Fallback selection and background
        // compilation are slot state and must not change the registry identity.
        let key=candidate.identity(resolve,geometry,kernels)?;
        let mut registries=self.program_pipelines.lock().unwrap();
        Ok(registries.entry(key).or_insert(candidate).clone())
    }
    /// The shared built-in pipelines for one raster coverage sample count.
    fn cached_pipelines(
        &self,
        raster_samples: u32,
    ) -> std::sync::Arc<CachedPipelines> {
        let cached = self.pipelines.scene[usize::from(raster_samples > 1)]
        .get_or_init(|| {
            std::sync::Arc::new(CachedPipelines::with_sources(
                self,
                RESOLVE,
                raster_samples,
                false,
                Default::default(),
            ))
        });
        std::sync::Arc::clone(cached)
    }
    /// Wall time per driver pipeline creation call; concurrent times overlap.
    pub fn pipeline_timings(&self) -> Vec<(String, f64)> {
        self.compiler
            .timings
            .lock()
            .unwrap()
            .iter()
            .map(|(name, ms)| (name.clone(), *ms))
            .collect()
    }
    fn create_compute_pipeline(
        &self,
        descriptor: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> wgpu::ComputePipeline {
        self.compiler.compute(descriptor)
    }
    fn create_render_pipeline(
        &self,
        descriptor: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> wgpu::RenderPipeline {
        self.compiler.render(descriptor)
    }
    pub fn pipeline_cache_supported(&self) -> bool {
        self.cache.is_some()
    }
    /// Save new pipelines in the background (`pipelines::Compiler::save_soon`): the caller never
    /// waits for the driver's serialization or for the filesystem.
    ///
    /// Delivering a render's output never waits for a save, and creating a pipeline never waits
    /// for one either: the save runs on another thread, and neither driver cache makes a creation
    /// wait for a serialization (Metal defers an addition that meets one). A render asks when a
    /// set of pipelines is complete: a 3D shot when preparation ends (a cold install's driver
    /// work, seconds of it, then survives a render that is killed), and every render when its last
    /// frame is drawn. A canvas creates its pipelines as its frames first use them, so it asks
    /// only at its end and its cache is serialized once; a serialization of Metal's archive takes
    /// 0.02 to 0.1 s. Nothing schedules a save around the next render: a kept-alive worker's save
    /// usually runs while it is idle, and may run beside its next render. A process about to exit
    /// saves whatever is finished,
    /// after its output is delivered (`finish_pipeline_save`): otherwise a one-shot would never
    /// leave its pipelines for the next process. Setup and library teardown, which
    /// must leave the whole cache on disk, call `settle_pipelines`.
    pub(crate) fn save_pipelines_soon(&self) {
        self.compiler.save_soon();
    }
    /// Leave on disk whatever pipelines this renderer has finished: wait for a save that is running,
    /// then save what is still new, such as a background variant that finished after the last
    /// `save_pipelines_soon`. Variants still compiling are not waited for. For a process that
    /// exits with its GPU owners alive, once every output is delivered and flushed.
    #[doc(hidden)]
    pub fn finish_pipeline_save(&self) {
        self.compiler.finish_save();
    }
    /// Wait for background feature variants and persist newly compiled pipelines.
    /// Library callers may explicitly warm all variants for a later process.
    /// Dropping the renderer also settles; one-shot CLI rendering exits with its
    /// GPU owners alive instead of waiting for unused variants.
    pub fn settle_pipelines(&self) -> Result<()> {
        self.compiler.settle()?;
        self.errors.check()?;
        self.compiler.save()
    }
    /// Install a fixed subtree and render settings; omitted options use the scene defaults.
    /// Paths in the scene resolve relative to `base`, the scene file's directory.
    pub fn install_with_options(
        &self,
        scene: Scene,
        base: &Path,
        options: InstallOptions,
    ) -> Result<InstalledScene> {
        let origins = scene.origins.clone();
        let program_files = scene.programs.iter().filter_map(|(name, program)|
            program.file.as_ref().map(|file| (name.clone(), file.clone()))).collect::<Vec<_>>();
        self.install_with_options_inner(scene, base, options)
            .map_err(|error| {
                let kind = failure::kind(&error);
                let text = format!("{error:#}").replace('\\', "/");
                // File-backed compiler diagnostics name their file, not the scene table key.
                let programs = program_files.iter().filter(|(_, file)| {
                    let file = file.replace('\\', "/");
                    [format!("{file}:"), format!("{file}("), format!("read {file}")]
                        .iter().any(|label| text.contains(label))
                }).map(|(name, _)| format!("program {name}")).collect::<Vec<_>>();
                let error = if programs.is_empty() { error } else { error.context(programs.join("; ")) };
                let error = vork_scene::v2::diagnostics::authored_error(&origins, error);
                // Installing reads, validates, compiles and lowers the scene, so what it
                // refuses is the scene's unless a device cause or category says otherwise:
                // the device's recorded errors outrank the installation's own result, and
                // the GPU work inside it categorises what the device hands back.
                // A rewritten diagnostic keeps the category its original had.
                failure::scene(match kind {
                    Some(kind) if kind != failure::FailureKind::Scene && failure::kind(&error).is_none() => error.context(kind),
                    _ => error,
                })
            })
    }
    fn install_with_options_inner(&self, scene: Scene, base: &Path, options: InstallOptions) -> Result<InstalledScene> {
        options.initial_time.numeric().context("initial scene time must be finite")?;
        self.errors.check()?;
        let start = Instant::now();
        let mut scene = scene;
        let mut options = options;
        if let Some(render) = options.render.take() {
            scene.render = render;
        }
        // Delivery drivers retain release admission, including test-support builds.
        #[cfg(any(test, feature = "test-support"))]
        let experimental_media = options.test_participating_media;
        #[cfg(not(any(test, feature = "test-support")))]
        let experimental_media = false;
        scene.validate()?;
        // A canvas in the world (a node with composition {space: "2d"}) is drawn by the 2D engine and seen
        // through its pictures; the world installs without its nodes.
        if scene.has_composition_parts() {
            for (image, root) in scene.canvas_pictures() {
                let resolution = scene.images[image].render.as_ref().expect("a picture renders").resolution;
                let picture = canvas_world::Picture::new(self, root, scene.composition_scene(root, resolution)?, base)
                    .with_context(|| format!("images.{image}: node {root:?}"))?;
                options.canvases.insert(image.to_string(), picture);
            }
            scene = scene.without_compositions()?.into_owned();
        }
        scene.delivery=!experimental_media;
        splats::prepare(self,&mut scene,base)?;
        if scene.delivery {
            let original=programs::mesh_stages(self,&scene,base)?;
            install::admit_stage_effects(&scene,base,&original)?;
            programs::normalize_launch_bindings(self,&mut scene,base,&original)?;
        }
        let (stages,transport_facts)=install::admit_delivery_effects(self,&scene,base,true)?;
        let automatic_probes=rt::automatic_indirect_requested(&scene);
        rt::resolve_scene_providers(self,&mut scene,base,transport_facts,&stages)?;
        rt::resolve_indirect(&mut scene,self.device.features())?;
        rt::resolve_reflections(&mut scene);
        rt::resolve_refraction(&mut scene, self.adapter_info.device_type, self.device.features());
        validate(&scene)?;
        // Approximate providers retain their existing visibility diagnostics.
        // Entirely unavailable effects were refused on resolved binding facts.
        lights::validate_visibility(&scene)?;
        // The media this open reads, once each: every clock plan compiled from here on and
        // the images installed share it, and it ends with the open.
        let opened = footage::Opened::default();
        animation::fold_material_inputs(&mut scene, base, &opened)?;
        if scene.propagation.is_some() && !scene.render.settings.integrator.is_path() {
            eprintln!("warning: authored curved propagation uses std/path transport for this scene and its render-image captures");
            rt::promote_to_path(&mut scene);
        }
        let camera_controls=camera_program::promote_in(self,&mut scene,base,&options.initial_time,&opened)?;
        if !scene.render.settings.integrator.is_path() && camera_program::scene_spectral(self,&scene,base)? {rt::promote_to_path(&mut scene);}
        ensure!(!options.raster_only || !scene.render.settings.integrator.is_path(),
            "interactive session requires raster transport; use a projective camera and remove path-only effects, or render this scene offline");
        if scene.render.settings.integrator.denoiser().is_some() {
            if let Err(error) = denoise::ensure_runtime(self) {
                crate::warnings::dropped("denoiser_unavailable", serde_json::json!({"denoise":true}), format!("warning: OIDN unavailable; rendering without denoising: {error:#}. Keep the bundled oidn runtime beside vork, or use --set denoise=false."));
                scene.render.settings.disable_denoiser();
            }
        }
        // Before any ray-query shader compiles, including the prewarm thread below.
        rt::require(self, &scene)?;
        // A scene in the working directory has an empty parent path.
        let base = if base.as_os_str().is_empty() {
            Path::new(".")
        } else {
            base
        };
        let mut effects = EffectPlan::for_scene(&scene);
        effects.implicit |= camera_controls;
        // Program variants own their validated compiler path; do not create an
        // unused built-in registry for those shots (including filtered roots).
        let background = scene.compositor().is_empty()
            && (options.gpu_color || options.readback.color || options.readback.color16 || options.readback.float.color)
            && pipelines::prewarm_builtin(&scene, effects)?;
        let samples = raster_samples(&scene.render);
        // Deferred variant workers wait until every foreground pipeline below exists.
        let _foreground = self.compiler.foreground_preparation();
        let installed:Result<InstalledScene> = std::thread::scope(|scope| {
            // Start base-registry initialization and required lazy effects while
            // install builds AgX, environment resources and the flattened scene.
            let compiler = background.then(|| {
                scope.spawn(|| {
                    self.cached_pipelines(samples)
                        .ensure_effects(self, effects)
                })
            });
            let installed = InstalledScene::new(
                self,
                scene,
                base,
                effects,
                background,
                automatic_probes,
                &options,
                &opened,
            );
            let wait = Instant::now();
            let compiled = compiler
                .map_or(Ok(Ok(())), |compiler| compiler.join())
                .map_err(|panic| failure::panicked("effect pipeline compiler", panic));
            let mut installed = installed?;
            compiled??;
            let _retained_shadows = installed.retain_shadows(self)?;
            installed.graph.views = views::ViewTable::new(&installed, self)?;
            if let Some(root) = installed.render.root.as_deref() {
                installed.main_subset = Some(render_images::subset(&installed, Some(root))?);
            }
            installed.pipeline_ms += wait.elapsed().as_secs_f64() * 1000.;
            installed.install_time = start.elapsed();
            Ok(installed)
        });
        self.errors.check()?;
        installed
    }
    /// Exercise retained participating-media transport in internal GPU fixtures.
    #[doc(hidden)]
    #[cfg(any(test, feature = "test-support"))]
    pub fn install_media_for_test(&self, scene: Scene, base: &Path) -> Result<InstalledScene> {
        self.install_with_options(scene, base, InstallOptions {
            test_participating_media: true, test_volume_bounces: 64, ..Default::default()
        })
    }
    /// Isolate direct medium connections for single-scattering analytic oracles.
    /// Production media retains complete-camera multiple scattering.
    #[doc(hidden)]
    #[cfg(any(test, feature = "test-support"))]
    pub fn install_single_scatter_media_for_test(&self, scene: Scene, base: &Path) -> Result<InstalledScene> {
        self.install_with_options(scene, base, InstallOptions {
            test_participating_media: true, test_volume_bounces: 0, ..Default::default()
        })
    }
    /// Decision D finite-carrier experiment; never enables delivery admission.
    #[doc(hidden)]
    #[cfg(any(test, feature = "test-support"))]
    pub fn install_convex_media_for_test(&self, scene: Scene, base: &Path) -> Result<InstalledScene> {
        self.install_with_options(scene, base, InstallOptions {
            test_participating_media: true, test_volume_bounces: 64,
            test_convex_media: true, test_convex_work_limit: Some(131072), ..Default::default()
        })
    }
    pub fn install(&self, scene: Scene, base: &Path) -> Result<InstalledScene> {
        self.install_with_options(scene, base, InstallOptions::default())
    }
}
/// Scene validation plus the checks for options this renderer does not implement yet.
pub fn validate(scene: &Scene) -> Result<()> {
    scene.validate()?;
    buffers::validate_attributes(scene)?;
    attributes::names(scene)?;
    // One environment, without light links (not implemented yet).
    environment_node::Settings::from_scene(scene)?;
    // std/openpbr inputs this renderer applies, and their images; then material tracks,
    // which write std/openpbr record parameters only.
    let animated = materials::animated_inputs(scene)?;
    for (name, material) in &scene.materials {
        if material.program == "std/openpbr" {
            let animated = animated.get(name).cloned().unwrap_or_default();
            materials::openpbr_values(scene, Some(name), &animated, |_| Ok(0))?;
        }
    }
    for target in scene.track_targets()?.into_iter().flatten() {
        if let vork_scene::TrackTarget::Material { material, input } = &target {
            if scene.materials[material].program == "std/openpbr" {
                materials::track_targets((material, input), &[], &[])?;
            }
        }
    }
    Ok(())
}
/// Load and validate a scene file; its directory is the install base.
fn load_scene_document(path: &Path) -> Result<Scene> {
    let bytes = if path == Path::new("-") {
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut bytes)?;
        bytes
    } else { std::fs::read(path)? };
    decode_scene_document(&bytes, path)
}
/// The ordinary expansion/parser path, also used for in-memory session documents.
fn decode_scene_document(bytes: &[u8], path: &Path) -> Result<Scene> {
    let base = path.parent().unwrap_or(Path::new("."));
    let expanded = || -> Result<serde_json::Value> {
        let mut document = vork_scene::v2::diagnostics::read_document(bytes)?;
        vork_std::expand_scene(&mut document, base)?;
        Ok(document)
    };
    // The expanded document is read into the scene and gone; a decode error makes it again to explain itself.
    expanded().and_then(|document| Scene::parse_value_with_source(document, bytes, Some(path), expanded)).map_err(failure::scene)
}
/// A repository scene with its authored render settings: `X.json` plus, when
/// present, the sibling profile `X.profile.json` (the repo's convention for its
/// own scenes, used by tests and harnesses; users pick profiles at render time).
#[doc(hidden)]
pub fn load_authored(path: &Path) -> Result<Scene> {
    load_authored_inner(path).map_err(failure::scene)
}
fn load_authored_inner(path: &Path) -> Result<Scene> {
    let mut scene = load_scene_document(path).with_context(|| format!("load scene {}", path.display()))?;
    let sibling = path.with_file_name(format!("{}.profile.json", path.file_stem().unwrap_or_default().to_string_lossy()));
    if sibling.exists() {
        let profile = vork_scene::Profile::load(&sibling.to_string_lossy())?;
        scene.apply_settings(profile.resolve(&Default::default())?, &profile.name)
            .with_context(|| format!("load scene {} with profile {}", path.display(), profile.name))?;
    }
    #[cfg(any(test, feature = "test-support"))]
    {
        let forced = sibling.with_extension("test-settings.json");
        if forced.exists() {
            let overrides = serde_json::from_slice(&std::fs::read(forced)?)?;
            scene.render.settings = vork_scene::test_settings(&scene.render.settings, &overrides)?;
        }
    }
    validate(&scene).with_context(|| format!("load scene {}", path.display()))?;
    Ok(scene)
}
/// Author-side decoding for renderer fixtures; internal test settings stay outside public scene JSON.
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn test_scene(mut document: serde_json::Value, base: &Path) -> Result<Scene> {
    let settings = document.get_mut("render").and_then(serde_json::Value::as_object_mut)
        .and_then(|render| render.remove("settings"));
    vork_std::expand_scene(&mut document, base)?;
    let mut scene = Scene::from_json(&serde_json::to_vec(&document)?)?;
    if let Some(settings) = settings { scene.render.settings = serde_json::from_value(settings)?; }
    Ok(scene)
}
/// Load a renderer test fixture with explicit internal provider controls.
/// Production profiles and CLI overrides never accept these controls.
#[cfg(any(test, feature = "test-support"))]
pub fn load_test_fixture(path: &Path, techniques: &str) -> Result<Scene> {
    (|| {
        let mut scene = load_scene_document(path)?;
        scene.apply_settings(serde_json::from_str(techniques)?, "test fixture")?;
        validate(&scene).with_context(|| format!("load test fixture {}", path.display()))?;
        Ok(scene)
    })().map_err(failure::scene)
}
/// Load content and resolved settings, apply the optional rewrite, then validate renderer support.
pub(crate) fn load_with_settings(path: &Path, settings: vork_scene::Settings, profile: &str, edit: Option<shot::SceneEdit>) -> Result<Scene> {
    let mut scene = load_scene_document(path).with_context(|| format!("load scene {}", path.display()))?;
    scene.render.settings = settings;
    validate_inputs(&scene, path.parent().unwrap_or(Path::new(".")))
        .with_context(|| format!("load scene {} with profile {profile}", path.display()))?;
    if let Some(edit) = edit {
        edit(&mut scene, path.parent().unwrap_or(Path::new(".")))
            .with_context(|| format!("edit scene {}", path.display())).map_err(failure::scene)?;
        validate_inputs(&scene, path.parent().unwrap_or(Path::new(".")))?;
    }
    Ok(scene)
}

/// Preserve the existing schema-first failure category while reporting all
/// typed scene and source problems, including after an author-side edit.
fn validate_inputs(scene: &Scene, base: &Path) -> Result<Option<std::collections::BTreeSet<PathBuf>>> {
    let mut errors = Vec::new();
    let mut files = None;
    for result in [validate(scene), validate_files(scene, base).map(|paths| files = paths)] {
        if let Err(error) = result { errors.push(scene.authored_error(error)); }
    }
    if errors.is_empty() {
        let result = (|| -> Result<()> {
            if !programs::needs_fog_admission(scene, base).map_err(|error| scene.authored_error(error))? {
                return Ok(());
            }
            let stages = programs::source_mesh_stages(&disk_cache::Store::host(), scene, base)
                .map_err(|error| scene.authored_error(error))?;
            install::admit_stage_effects(scene, base, &stages)
        })();
        if let Err(error) = result { errors.push(error); }
    }
    vork_scene::v2::diagnostics::joined(errors).map_err(failure::scene)?;
    Ok(files)
}

/// File-backed sources must be readable regular files, not directories that
/// File::open happens to accept on Unix.
fn source_file(path: impl AsRef<Path>) -> std::io::Result<std::fs::File> {
    let path = path.as_ref();
    if !std::fs::metadata(path)?.is_file() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
            format!("{} is not a regular file; restore a readable regular file or correct its path", path.display())));
    }
    std::fs::File::open(path)
}

/// Source admission uses the CPU validators before any native pipeline or upload.
/// Independent sources are all visited even when another source cannot be opened.
/// Return the same sources for installation identity, or None when their complete
/// dependency set is not known. This never adds an admission requirement.
fn validate_files(scene: &Scene, base: &Path) -> Result<Option<std::collections::BTreeSet<PathBuf>>> {
    let base = if base.as_os_str().is_empty() {Path::new(".")} else {base};
    let mut errors = Vec::new();
    let mut files = std::collections::BTreeSet::new();
    // A canvas with no world inside reads images through its paints, which `canvas::sources` lists, and through
    // its view and its image graph, whose image files the walk below lists: a rendered image is not read there.
    let paints_only = scene.is_canvas() && !canvas::has_worlds(&scene.nodes);
    let mut complete = scene.programs.is_empty() && scene.simulation.is_none()
        && (paints_only || scene.images.values().all(|image| image.render.is_none()));
    for file in scene.nodes.iter().flat_map(|node| scene.audio_sources(node)).collect::<std::collections::BTreeSet<_>>() {
        let path = base.join(&file);
        files.insert(path.clone());
        if let Err(error) = source_file(&path).map_err(anyhow::Error::from).and_then(|_| audio::load(&path).map(|_| ())) {
            errors.push(error.context(format!("audio {file}: restore the file or correct its path")));
        }
    }
    for (name, buffer) in &scene.buffers {
        if let Some(file) = &buffer.file { files.insert(base.join(file)); }
        if buffer.file.is_some() && buffer.grid.is_none() {
            if let Err(error) = buffers::AttributeData::file(buffer, base) {
                errors.push(error.context(format!("buffer {name}: restore the file or correct its offset/count")));
            }
        } else if let Some(file) = &buffer.file {
            if let Err(error) = source_file(base.join(file)) {
                errors.push(anyhow::Error::new(error).context(format!("grid buffer {name} ({file}): restore the grid file or correct its path")));
            }
        }
    }
    // Resolve the same node-owned layers installation compiles, including
    // ancestor-owned clips targeting descendant assets. Their glTF URI closure
    // participates in admission and installation identity just like asset files.
    let owned = scene.node_owned_layers()?;
    let mut paths: std::collections::BTreeSet<_> = scene.animation.iter()
        .filter_map(|layer| layer.clip.as_ref())
        .chain(owned.iter().filter_map(|owned| owned.layer.clip.as_ref()))
        .filter_map(|clip| clip.file.as_deref()).collect();
    // Every composition's local view reads sources: the scene's own and each node's.
    let mut views = vec![scene.view()];
    fn composition_views(nodes: &[vork_scene::v2::Node], views: &mut Vec<vork_scene::v2::NameOr<vork_scene::v2::Material>>) {
        for node in nodes {
            if let Some(composition) = &node.composition { views.push(composition.local_view()); }
            composition_views(&node.children, views);
        }
    }
    composition_views(&scene.nodes, &mut views);
    // Every composition's compositor reads sources too: the scene's own here, each node's in `sources`, and
    // each rendered image's in the loop below.
    let mut pending: Vec<&str> = scene.compositor().iter().map(|entry|entry.material().as_str()).collect();
    let mut images = std::collections::BTreeSet::new();
    fn sources<'a>(nodes: &'a [vork_scene::v2::Node], paths: &mut std::collections::BTreeSet<&'a str>, images: &mut std::collections::BTreeSet<&'a str>, materials: &mut Vec<&'a str>) {
        use vork_scene::v2::NodeKind;
        for node in nodes {
            if let Some(composition) = &node.composition {
                materials.extend(composition.compositor.iter().map(|entry| entry.material().as_str()));
            }
            match &node.kind {
                NodeKind::Asset {file,materials:bindings,..} => {paths.insert(file); materials.extend(bindings.values().map(String::as_str));}
                NodeKind::Mesh {materials:bindings,..} => materials.extend(bindings.iter().map(String::as_str)),
                NodeKind::Camera {material:Some(material),..} => materials.push(material),
                NodeKind::Light {image,material,medium,..} => {
                    images.extend(image.iter().map(String::as_str));
                    materials.extend(material.iter().chain(medium).map(String::as_str));
                }
                _ => {}
            }
            sources(&node.children, paths, images, materials);
        }
    }
    sources(&scene.nodes, &mut paths, &mut images, &mut pending);
    let mut inline_views = Vec::new();
    for view in &views {
        match view {
            vork_scene::v2::NameOr::Name(name) => pending.push(name),
            vork_scene::v2::NameOr::Inline(material) => inline_views.push(material),
        }
    }
    let view_materials = views.iter().filter_map(|view| match view {
        vork_scene::v2::NameOr::Name(name) => scene.materials.get(name),
        vork_scene::v2::NameOr::Inline(material) => Some(material),
    });
    for (input, value) in view_materials.flat_map(|material| &material.inputs) {
        let vork_scene::v2::Input::Texture(binding) = value else {continue};
        let Some(image) = scene.images.get(&binding.image) else {continue};
        let result = (|| -> Result<()> {
            let file = image.file.as_ref().context("display programs read file images; export the rendered image to a file and bind that file")?;
            let path = base.join(file);
            source_file(&path).with_context(||format!("restore display image {file} or correct its path"))?;
            files.insert(path);
            Ok(())
        })();
        if let Err(error) = result {errors.push(error.context(format!("composition.view: input {input}")));}
    }
    let mut programs: std::collections::BTreeSet<_> = scene.programs.iter()
        .filter(|(_, program)|program.is_compute()).map(|(name,_)|name.as_str()).collect();
    programs.extend(scene.propagation.iter().map(|p|p.program.as_str()));
    let mut materials = std::collections::BTreeSet::new();
    let mut visited_images = std::collections::BTreeSet::new();
    for material in &inline_views {
        programs.insert(material.program.as_str());
        images.extend(material.inputs.values().filter_map(|input|match input {
            vork_scene::v2::Input::Texture(binding) => Some(binding.image.as_str()), _ => None,
        }));
    }
    loop {
        while let Some(name) = pending.pop() {
            if !materials.insert(name) {continue;}
            let Some(material) = scene.materials.get(name) else {continue};
            programs.insert(material.program.as_str());
            images.extend(material.inputs.values().filter_map(|input|match input {
                vork_scene::v2::Input::Texture(binding) => Some(binding.image.as_str()), _ => None,
            }));
        }
        for &name in &images {
            if !visited_images.insert(name) {continue;}
            if let Some(image) = scene.images.get(name).filter(|image|image.render.is_some()) {
                pending.extend(image.compositor().iter().map(|entry|entry.material().as_str()));
            }
        }
        if pending.is_empty() {break;}
    }
    for name in programs {
        if let Some(file) = scene.programs.get(name).and_then(|program|program.file.as_ref()) {
            if let Err(error) = source_file(base.join(file)) {
                errors.push(anyhow::Error::new(error).context(format!("program {name} ({file}): restore the source file or correct its path")));
            }
        }
    }
    for material in materials.into_iter().filter_map(|name|scene.materials.get(name)).chain(inline_views.iter().copied()) {
        for library in vork_scene::libraries::LIBRARIES {
            for &vork_scene::libraries::ProgramData { program, input, image, .. } in library.images {
                if material.program != program || material.inputs.contains_key(input) {continue;}
                if let vork_scene::libraries::ProgramImage::File(relative) = image {
                    let result = (|| -> Result<()> {
                        let executable = std::env::current_exe()?.canonicalize()?;
                        let path = executable.parent().context("locate executable directory")?.join(relative);
                        source_file(&path).with_context(|| format!("library image {}", path.display()))?;
                        files.insert(path);
                        Ok(())
                    })();
                    if let Err(error) = result {
                        errors.push(error.context(format!("{program}.{input}: reinstall the complete vork archive with its libraries directory")));
                    }
                }
            }
        }
    }
    for name in images {
        let Some(image) = scene.images.get(name).filter(|image| image.file.is_some() || image.render.is_some()) else {continue};
        let source = texture_source::TextureSource::scene(name, image);
        let path = match source {
            texture_source::TextureSource::File(file) => base.join(file),
            texture_source::TextureSource::Media {file, ..} => {
                complete = false;
                base.join(file)
            }
            texture_source::TextureSource::Render(_) | texture_source::TextureSource::Canvas(_) => { complete = false; continue; }
        };
        let result = (|| -> Result<()> {
            if footage::is_sequence(&path) {
                files.extend(footage::sequence(&path)?);
            } else {
                files.insert(path.clone());
                match source_file(&path) {
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound
                        && !footage::is_media(&path)
                        && !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ies")) => {}
                    Err(error) => return Err(error.into()),
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            errors.push(error.context(format!("image {name} ({}): restore valid image/media frames or correct the path", path.display())));
        }
    }
    for file in paths {
        let path = base.join(file);
        files.insert(path.clone());
        let result=if splats::is_file(&path) {splats::inspect(&path).map(|_|Default::default())}
            else {import::validate_file(&path,path.parent().unwrap_or(base))};
        match result {
            Ok(dependencies) => files.extend(dependencies),
            Err(error) => errors.push(error.context(format!("asset {file}: restore a valid asset and its referenced files"))),
        }
    }
    // A 2D canvas reads its fonts and painted images itself (canvas::Document reports their errors); they are
    // part of what a session must see unchanged before it reuses the document.
    if scene.is_canvas() && !canvas::sources(scene, base, &mut files) { complete = false; }
    vork_scene::v2::diagnostics::joined(errors).context("scene input problems").map_err(failure::scene)?;
    Ok(complete.then_some(files))
}
pub fn load(path: &Path) -> Result<Scene> {
    let scene = load_scene_document(path).with_context(|| format!("load scene {}", path.display()))?;
    validate(&scene).with_context(|| format!("load scene {}", path.display())).map_err(failure::scene)?;
    Ok(scene)
}
/// Raster coverage samples: one for jitter AA, otherwise the msaa4 option.
fn raster_samples(render: &vork_scene::Render) -> u32 {
    if matches!(
        render.settings.integrator.aa(),
        vork_scene::Antialiasing::Jitter
    ) {
        1
    } else {
        constants::RASTER_MSAA_SAMPLES
    }
}
fn init(d: &wgpu::Device, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu_memory::buffer_init(d, label, contents, usage)
}

fn extent(w: u32, h: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    }
}
fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}
fn color_attachment<'a>(
    view: &'a wgpu::TextureView,
    clear: wgpu::Color,
) -> wgpu::RenderPassColorAttachment<'a> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(clear),
            store: wgpu::StoreOp::Store,
        },
    }
}
fn texture(
    d: &wgpu::Device,
    label: &str,
    w: u32,
    h: u32,
    samples: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    crate::gpu_memory::texture(&d, &wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(w, h),
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}
fn timestamps(q: &wgpu::QuerySet, index: u32) -> wgpu::RenderPassTimestampWrites<'_> {
    wgpu::RenderPassTimestampWrites {
        query_set: q,
        beginning_of_pass_write_index: Some(index),
        end_of_pass_write_index: Some(index + 1),
    }
}
fn depth_attachment(
    view: &wgpu::TextureView,
    read_only: bool,
    clear: bool,
) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: if read_only {
            None
        } else {
            Some(wgpu::Operations {
                load: if clear {
                    wgpu::LoadOp::Clear(1.)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            })
        },
        stencil_ops: None,
    }
}

#[cfg(all(test, vork_private_tests))]
include!("lib_device_startup_tests_private.rs");

#[cfg(test)]
mod tests {
    /// The loader behind render, check and session documents says the nesting limit in scene terms.
    #[test]
    fn the_scene_loader_states_the_nesting_limit() {
        // 63 nested nodes reach 127 levels; an empty children array on the innermost is the 128th.
        let mut node = serde_json::json!({"name":"n63","children":[]});
        for depth in (1..63).rev() { node = serde_json::json!({"name":format!("n{depth}"),"children":[node]}); }
        let bytes = serde_json::to_vec(&serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[node]})).unwrap();
        let error = format!("{:#}", super::decode_scene_document(&bytes, std::path::Path::new("deep.json")).err().unwrap());
        assert!(error.contains("nests arrays and objects more than 127 deep") && error.contains("flatten the hierarchy"), "{error}");
        // Any other reader error keeps its own words.
        let error = format!("{:#}", super::decode_scene_document(b"{\"version\":", std::path::Path::new("cut.json")).err().unwrap());
        assert!(!error.contains("nests arrays"), "{error}");
    }

    use super::*;
    #[test]
    fn file_and_session_document_admission_match_expanded_bytes() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-document-admission-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path = root.join("scene.json");
        let authored = br#"{"version":2,"nodes":[{"type":"camera","name":"camera",
            "material":{"program":"std/camera","inputs":{"lens":35}}}]}"#;
        std::fs::write(&path, authored)?;
        let result = (|| -> Result<()> {
            let mut expanded = serde_json::from_slice(authored)?;
            vork_std::expand_scene(&mut expanded, &root)?;
            let old = Scene::parse_json_with_source(&serde_json::to_vec(&expanded)?, authored, Some(&path))?;
            // File loading and session document loading share this decoder.
            for scene in [load_scene_document(&path)?, decode_scene_document(authored, &path)?] {
                assert_eq!(serde_json::to_vec(&scene)?, serde_json::to_vec(&old)?);
                assert_eq!(scene.origins, old.origins);
            }
            Ok(())
        })();
        std::fs::remove_dir_all(root)?;
        result
    }

    #[test]
    fn shader_cache_retries_failed_key_publication() -> Result<()> {
        let mut renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        // Use the no-op device to exercise normal publication without a GPU. CPU-only
        // preparation deliberately skips publication; the compiler below publishes its ledger.
        let root = std::env::temp_dir().join(format!("vork-cache-retry-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path = root.join("keys.entry");
        renderer.disk_cache = disk_cache::Store::for_test(root.clone());
        renderer.compiler = pipelines::Compiler::new(
            &renderer.device, None, pipelines::Known::load(path.clone(), false), pipelines::Blob::ledger_only(),
        );
        let key = blake3::hash(b"unsaved pipeline");
        renderer.compiler.remember(key);
        // A directory at the destination makes atomic rename fail on every OS.
        std::fs::create_dir(&path)?;
        assert!(renderer.settle_pipelines().is_err());
        std::fs::remove_dir(&path)?;
        renderer.settle_pipelines()?;
        assert_eq!(disk_cache::read(&path), Some(key.as_bytes().to_vec()));
        assert!(renderer.compiler.known_snapshot().is_none());
        drop(renderer);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    #[ignore = "requires a Vulkan device or Metal binary archives (a persistent pipeline cache)"]
    fn shader_cache_retries_failed_driver_publication() -> Result<()> {
        let mut renderer = Renderer::new()?;
        if renderer.adapter_info.device_type == wgpu::DeviceType::Cpu
            || !renderer.device.features().contains(wgpu::Features::PIPELINE_CACHE)
        {
            eprintln!("driver publication test skipped: {:?} has no supported persistent pipeline cache", renderer.adapter_info);
            return Ok(());
        }
        anyhow::ensure!(renderer.cache.is_some(), "persistent driver cache required");
        let root = std::env::temp_dir().join(format!("vork-driver-retry-{}", std::process::id()));
        let path = root.join("keys.entry");
        renderer.cache_path = root.join("driver.entry");
        renderer.disk_cache = disk_cache::Store::for_test(root.clone());
        renderer.compiler = pipelines::Compiler::new(
            &renderer.device, renderer.cache.as_ref(), pipelines::Known::load(path.clone(), false),
            pipelines::Blob { path: renderer.cache_path.clone(), marker: None, loaded: None, save: true },
        );
        let module = renderer.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cache retry"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let _pipeline = renderer.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cache retry"), layout: None, module: &module,
            entry_point: Some("main"), compilation_options: Default::default(),
            cache: renderer.cache.as_ref(),
        });
        let key = blake3::hash(b"driver retry pipeline");
        renderer.compiler.remember(key);
        std::fs::create_dir_all(&renderer.cache_path)?;
        assert!(renderer.settle_pipelines().is_err());
        assert_eq!(renderer.compiler.saved().0, 0);
        assert!(renderer.compiler.known_snapshot().is_some());
        std::fs::remove_dir(&renderer.cache_path)?;
        renderer.settle_pipelines()?;
        assert!(disk_cache::read_pipeline(&renderer.cache_path).is_some());
        assert_eq!(disk_cache::read(&path), Some(key.as_bytes().to_vec()));
        assert_eq!(renderer.compiler.saved().0, renderer.compiler.created());
        assert!(renderer.compiler.known_snapshot().is_none());
        drop(renderer);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    /// A Metal binary archive that wgpu's header accepts but Metal cannot open is
    /// rejected with a reason, never a failed render: the process compiles into
    /// a fresh archive and rewrites the entry, which the next process accepts.
    #[test]
    #[ignore = "requires a Metal GPU with binary archives"]
    fn metal_contract_corrupt_archive_recompiles_and_rewrites() -> Result<()> {
        const CHILD: &str = "VORK_CORRUPT_ARCHIVE_TEST";
        if std::env::var_os(CHILD).is_none() {
            let root = std::env::temp_dir().join(format!("vork-corrupt-archive-{}", std::process::id()));
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "tests::metal_contract_corrupt_archive_recompiles_and_rewrites", "--ignored", "--nocapture"])
                .env(CHILD, "1").env("VORK_CACHE_DIR", &root).status()?;
            let _ = std::fs::remove_dir_all(&root);
            anyhow::ensure!(status.success(), "corrupt archive child failed");
            return Ok(());
        }
        let compile = |renderer: &Renderer| -> Result<()> {
            let module = renderer.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("archive contract"),
                source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage, read_write> out: array<u32>;\n@compute @workgroup_size(1) fn main() { out[0] = 7u; }".into()),
            });
            let _pipeline = renderer.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("archive contract"), layout: None, module: &module,
                entry_point: Some("main"), compilation_options: Default::default(),
                cache: renderer.cache.as_ref(),
            });
            renderer.errors.check()?;
            renderer.settle_pipelines()
        };
        let first = metal_contract_renderer()?;
        anyhow::ensure!(first.cache.is_some(), "Metal binary archives required: {:?}", first.adapter_info);
        compile(&first)?;
        let saved = disk_cache::read_pipeline(&first.cache_path).expect("archive saved");
        let path = first.cache_path.clone();
        drop(first);
        // Keep wgpu's valid header (GPU and OS key); replace the archive body.
        let header = 64;
        let mut corrupt = saved[..header].to_vec();
        corrupt.extend(std::iter::repeat_n(0xA5, saved.len() - header));
        anyhow::ensure!(disk_cache::write_entry(&path, &corrupt)?, "corrupt entry not written");
        let second = metal_contract_renderer()?;
        assert_eq!(second.pipeline_cache_status, "rejected");
        let reason = second.pipeline_cache_rejection.clone().unwrap_or_default();
        assert!(reason.contains("binary archive"), "{reason}");
        assert!(second.cache.is_some(), "a rejected archive recovers with an empty one");
        compile(&second)?;
        drop(second);
        let third = metal_contract_renderer()?;
        assert_eq!(third.pipeline_cache_status, "accepted");
        Ok(())
    }
    #[test]
    fn shader_cache_budget_refusal_keeps_keys_dirty() -> Result<()> {
        const CHILD: &str = "VORK_KEY_BUDGET_TEST";
        if std::env::var_os(CHILD).is_none() {
            let root = std::env::temp_dir().join(format!("vork-key-budget-{}", std::process::id()));
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "tests::shader_cache_budget_refusal_keeps_keys_dirty"])
                .env(CHILD, "1").env("VORK_CACHE_BYTES", "0")
                .env("VORK_CACHE_DIR", &root).status()?;
            anyhow::ensure!(status.success(), "cache budget child failed");
            return Ok(());
        }
        let mut renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let root = PathBuf::from(std::env::var_os("VORK_CACHE_DIR").unwrap());
        let path = root.join("keys.entry");
        renderer.disk_cache = disk_cache::Store::for_test(root.clone());
        renderer.compiler = pipelines::Compiler::new(
            &renderer.device, None, pipelines::Known::load(path.clone(), false), pipelines::Blob::ledger_only(),
        );
        renderer.compiler.remember(blake3::hash(b"unsaved pipeline"));
        renderer.settle_pipelines()?;
        assert!(disk_cache::read(&path).is_none());
        assert!(renderer.compiler.known_snapshot().is_some());
        renderer.settle_pipelines()?;
        assert!(renderer.compiler.known_snapshot().is_some());
        drop(renderer);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn worker_types_are_shareable() {
        fn shared<T: Send + Sync>() {}
        shared::<Renderer>();
        shared::<InstalledScene>();
        shared::<FrameSlot>();
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn medium_ray_visibility_binds_multiple_lights() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version": 2,
            "render": {"resolution": [8,8],
                "settings": {"integrator":"raster","samples":1,"fast_gi":null,"shadow_method":"ray"}},
            "materials": {"fog": {"program":{"source":include_str!("../../vork-scene/tests/fixtures/medium.slang")}, "inputs": {"density": 0.1}}},
            "nodes": [
                {"type": "camera", "name": "camera", "translation": [0,0,3]},
                {"type": "light", "name": "world", "shape": "environment", "medium": "fog"},
                {"type": "light", "name": "first", "shape": "point"},
                {"type": "light", "name": "second", "shape": "point"}
            ]
        }))?;
        let mut installed = renderer.install(scene, Path::new("."))?;
        let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        installed.render_frame(&renderer, &mut frame, 0.)?;
        renderer.errors.check()?;
        Ok(())
    }
    /// VORK_TEST_SCENE can override the checked-in analytic fixture.
    #[test]
    #[ignore = "requires Metal/Vulkan and pinned Slang toolchain"]
    fn install_is_shared_and_worker_outputs_match() -> Result<()> {
        let path = std::env::var_os("VORK_TEST_SCENE").map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/analytic.json"));
        let scene = load(&path)?;
        let r = Renderer::new()?;
        let mut a = r.install(scene.clone(), path.parent().unwrap())?;
        let b = r.install(scene, path.parent().unwrap())?;
        assert!(
            std::sync::Arc::ptr_eq(&a.pipelines, &b.pipelines),
            "pipeline registry must be shared across scenes"
        );
        let mut warm = a.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        a.render_frame(&r, &mut warm, 0.)?;
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    scope.spawn(|| -> Result<FrameOutput> {
                        let mut slot = a.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
                        slot.profile = true;
                        slot.submit(&a, &r, 0.)?;
                        let first = slot.collect(&r)?;
                        slot.submit(&a, &r, 0.)?;
                        let second = slot.collect(&r)?;
                        crate::test_linear::assert_linear_eq((first).linear_rgba.as_ref().expect("linear readback"), (second).linear_rgba.as_ref().expect("linear readback"));
                        crate::test_linear::assert_depth_eq(first.depth_f32.as_ref().expect("float depth readback"), second.depth_f32.as_ref().expect("float depth readback"));
                        assert_eq!(first.ids, second.ids);
                        Ok(second)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Result<Vec<_>>>()
        })?;
        for output in &outputs[1..] {
            crate::test_linear::assert_linear_eq((outputs[0]).linear_rgba.as_ref().expect("linear readback"), (output).linear_rgba.as_ref().expect("linear readback"));
            crate::test_linear::assert_depth_eq(outputs[0].depth_f32.as_ref().expect("float depth"), output.depth_f32.as_ref().expect("float depth"));
            assert_eq!(outputs[0].ids, output.ids);
        }
        Ok(())
    }
}

#[cfg(test)]
mod gpu_object_tests {
    use super::*;
    #[test]
    fn affine_residency_preserves_all_poses_and_attributes() {
        assert_eq!(std::mem::size_of::<GpuObject>(), 176);
        let mut value = Object::new(Mat4::from_scale_rotation_translation(
            Vec3::new(-2., 3., 0.5),
            glam::Quat::from_rotation_y(0.7),
            Vec3::new(11., -3., 5.),
        ));
        value.previous_model = Mat4::from_translation(Vec3::new(3., 2., 1.));
        value.opacity = [0.25, 65535., 0., 0.];
        value.material_index[0] = 123;
        value.metadata = [17, 29, 0, 0];
        value.metadata[2] = 113;
        let gpu = value.gpu();
        let unpack = |rows: [[f32; 4]; 3]| {
            Mat4::from_cols_array_2d(&[rows[0], rows[1], rows[2], [0., 0., 0., 1.]]).transpose()
        };
        assert_eq!(unpack(gpu.model), value.model);
        assert_eq!(unpack(gpu.previous_model), value.previous_model);
        assert_eq!(unpack(gpu.next_model), value.next_model);
        assert_eq!(gpu.opacity, [0.25, 65535.]);
        assert_eq!(gpu.material_index, 123);
        assert_eq!(gpu.metadata, [17, 29]);
        assert_eq!(gpu.reserved, 113);
    }
}

/// Focused contracts must never turn missing native execution into a green skip.
#[cfg(test)]
fn metal_contract_renderer() -> Result<Renderer> {
    let renderer = Renderer::new()?;
    anyhow::ensure!(renderer.adapter_info.backend == wgpu::Backend::Metal,
        "Metal contracts require a real Metal GPU, got {:?}", renderer.adapter_info);
    eprintln!("Metal contract device: {:?}", renderer.adapter_info);
    Ok(renderer)
}

/// Ignored GPU contracts must execute when explicitly selected, or fail visibly.
#[cfg(test)]
fn test_renderer() -> Result<Renderer> {
    anyhow::ensure!(std::env::var("VORK_GPU_TESTS").as_deref() == Ok("1"),
        "GPU test requires VORK_GPU_TESTS=1, a supported GPU and pinned Slang; see vork help build");
    Renderer::new().context("GPU test requires a working renderer; adapter discovery and device creation must succeed")
}

#[cfg(test)]
mod gpu_test_harness_tests {
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn renderer_initialization() -> anyhow::Result<()> {
        super::test_renderer()?;
        Ok(())
    }

    #[test]
    fn explicitly_selected_gpu_test_fails_without_opt_in() -> anyhow::Result<()> {
        // Exercise libtest itself: a diagnostic printed by a passing test is
        // captured and cannot serve as an explicit skip or a required failure.
        let output = std::process::Command::new(std::env::current_exe()?)
            .args(["gpu_test_harness_tests::renderer_initialization", "--exact", "--ignored", "--nocapture"])
            .env_remove("VORK_GPU_TESTS")
            .output()?;
        assert!(!output.status.success(), "GPU test passed without executing its renderer");
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires VORK_GPU_TESTS=1"),
            "{}", String::from_utf8_lossy(&output.stderr));
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 failed;"),
            "{}", String::from_utf8_lossy(&output.stdout));
        Ok(())
    }
}

#[cfg(test)]
mod scene_error_context_tests {
    #[test]
    fn scene_edit_precedes_renderer_support_validation() -> anyhow::Result<()> {
        let path = std::env::temp_dir().join(format!("vork-edit-opacity-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&serde_json::json!({"version":2,
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]},
                "color":{"type":"vec4","data":[[1,0,0,1],[1,0,0,1],[1,0,0,1]]}},
            "geometry":{"g":{"attributes":{"POSITION":"p","COLOR_0":"color"}}},
            "materials":{"paint":{"inputs":{"geometry_opacity":{"attribute":"COLOR_0","channel":"r"}}}},
            "nodes":[{"type":"mesh","geometry":"g","materials":["paint"]},{"type":"camera"}]
        }))?)?;
        let profile = vork_scene::Profile::load(vork_scene::DEFAULT_PROFILE)?;
        let settings = vork_scene::test_settings(&profile.resolve(&Default::default())?, serde_json::json!({"refraction":"ray"}).as_object().unwrap())?;
        let load = |edit| super::load_with_settings(&path, settings.clone(), &profile.name, edit);
        let unedited = load(None);
        let edited = load(Some(|scene, _| {
            assert!(super::rt::uses_rays(scene), "the edit must receive resolved settings");
            assert!(scene.materials["paint"].inputs.contains_key("geometry_opacity"),
                "the edit must receive authored bindings before launch normalization");
            scene.materials.get_mut("paint").unwrap().inputs.remove("geometry_opacity");
            Ok(())
        }));
        let invalid_edit = load(Some(|scene, _| {
            scene.render.resolution = [0, 0];
            Ok(())
        }));
        std::fs::remove_file(&path)?;
        // Installation retains authored bindings; bound effect proofs select coverage.
        let unedited = unedited?;
        assert!(unedited.materials["paint"].inputs.contains_key("geometry_opacity"));
        assert!(!edited?.materials["paint"].inputs.contains_key("geometry_opacity"));
        assert!(format!("{:#}", invalid_edit.unwrap_err()).contains("positive dimensions"));
        Ok(())
    }
    #[test]
    fn resolved_profile_precedes_renderer_validation() -> anyhow::Result<()> {
        let path = std::env::temp_dir().join(format!("vork-profile-visibility-{}.json", std::process::id()));
        let profile = vork_scene::v2::write_authored(&path, &serde_json::json!({"version":2,
            "nodes":[{"type":"camera"},{"type":"node","name":"hidden","visible_transmission":false}],
            "render":{"settings":{"refraction":"ray"}}
        }))?.unwrap();
        let explicit = serde_json::json!({"refraction":"ray"}).as_object().unwrap().clone();
        let default = vork_scene::Profile::load(vork_scene::DEFAULT_PROFILE)?;
        let resolved = super::load_with_settings(&path, vork_scene::test_settings(&default.resolve(&Default::default())?, &explicit)?, &default.name, None);
        let authored = super::load_authored(&path);
        let deferred = super::load_with_settings(&path, vork_scene::test_settings(&default.resolve(&Default::default())?, serde_json::json!({"refraction":"environment"}).as_object().unwrap())?, &default.name, None);
        let repaired = super::load_with_settings(&path, vork_scene::test_settings(&default.resolve(&Default::default())?, serde_json::json!({"refraction":"environment"}).as_object().unwrap())?, &default.name, Some(|scene, _| {
            scene.nodes[1].visibility.transmission = true;
            Ok(())
        }));
        std::fs::remove_file(&path)?;
        std::fs::remove_file(&profile.name)?;
        assert!(super::rt::ray_refraction(&resolved?));
        authored?;
        // Loading preserves requests; installation checks capabilities after
        // release normalization. The resolved profile still reaches the edit.
        let deferred = deferred?;
        assert!(!deferred.nodes[1].visibility.transmission);
        assert!(!super::rt::ray_refraction(&deferred));
        assert!(!deferred.delivery);
        // Loading leaves installation-time capability checks deferred.
        assert!(super::lights::validate_visibility(&deferred).is_err());
        let repaired = repaired?;
        assert!(repaired.nodes[1].visibility.transmission);
        super::lights::validate_visibility(&repaired)?;
        Ok(())
    }
    #[test]
    fn scene_validation_retains_the_scene_filename() {
        let path = std::env::temp_dir().join(format!("vork-invalid-resolution-{}.json", std::process::id()));
        vork_scene::v2::write_authored(&path, &serde_json::json!({"version":2,"nodes":[{"type":"camera"},{"type":"node","visible_transmission":false}],
            "render":{"resolution":[0,1080],"settings":{"integrator":"raster","refraction":"environment"}}})).unwrap();
        let error = format!("{:#}", super::load_authored(&path).unwrap_err());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(path.with_file_name(format!("{}.profile.json", path.file_stem().unwrap().to_string_lossy()))).unwrap();
        assert!(error.contains(path.to_str().unwrap()), "{error}");
        assert!(error.contains("positive dimensions"), "{error}");
    }
}

#[cfg(all(test, vork_private_tests))]
include!("lib_path_dielectric_tests_private.rs");
#[cfg(test)]
mod implicit_cell_tests;
#[cfg(test)]
mod sample_clamp_tests;
#[cfg(test)]
mod implicit_scene_tests;
#[cfg(test)]
mod path_texture_tests;
#[cfg(all(test, vork_private_tests))]
mod trace_tests;
#[cfg(all(test, vork_private_tests))]
include!("lib_path_medium_tests_private.rs");

/// Authoring declarations from the renderer's current shader-program ABI.
pub fn program_help() -> String {
    let abi = programs::abi();
    abi.public_declarations()
}

#[cfg(all(test, vork_private_tests))]
include!("lib_subsurface_tests_private.rs");

#[cfg(test)]
mod texture_precision_tests;

#[cfg(test)]
mod hair_sampling;

#[cfg(test)]
mod sampling_regressions;

#[cfg(test)]
mod path_sampler;

mod camera_native;

#[cfg(test)]
mod visibility_regressions;

#[cfg(test)]
mod instance_attribute_tests;


#[cfg(test)]
mod light_connection_tests;

#[cfg(test)]
mod animinputs_tests;

/// Internal transport tooling links the same BSDF modules as production.
#[doc(hidden)]
pub fn transport_modules() -> &'static [(&'static str, &'static str)] { native::SOURCES }

/// Offline asset delivery, isolated from rendering.
pub mod export;
#[cfg(test)]
#[path = "../tests/support/linear.rs"]
mod test_linear;
#[cfg(test)]
#[path = "../tests/support/display.rs"]
mod test_display;

#[cfg(test)]
mod linear_equivalence_tests {
    use super::*;
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn linear_comparison_rejects_small_real_radiance_change() -> Result<()> {
        let r = crate::test_renderer()?;
        let render = |brightness: f32| -> Result<Vec<[f32; 4]>> {
            let scene = vork_scene::v2::from_test_value(serde_json::json!({
                "version":2,"render":{"resolution":[16,16],"background":[brightness,brightness,brightness],
                    "settings":{"integrator":"path","samples":4,"denoiser":null}},
                "nodes":[{"type":"camera"}]
            }))?;
            let mut installed = r.install(scene,Path::new("."))?;
            let mut slot = installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            Ok(installed.render_frame(&r,&mut slot,0.)?.linear_rgba.expect("linear readback"))
        };
        let base=render(0.18)?;
        let repeat=render(0.18)?;
        crate::test_linear::assert_linear_eq(&base,&repeat);
        // Inject a real scene radiance regression, not an encoded-byte difference.
        let changed=render(0.18*1.0001)?;
        assert!(std::panic::catch_unwind(||crate::test_linear::assert_linear_eq(&base,&changed)).is_err(),
            "0.01% real radiance change escaped the image comparator");
        Ok(())
    }
}
#[cfg(test)]
mod test_geometry;

#[cfg(test)]
mod metal_generated_tests;
#[cfg(all(test, vork_private_tests))]
#[path = "scene_code_budget_private.rs"]
mod scene_code_budget;

#[cfg(test)]
mod software_raster_tests;
