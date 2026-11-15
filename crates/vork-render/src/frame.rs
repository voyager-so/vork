//! FrameSlot: every writable per-frame GPU resource, the walk of the compiled
//! frame graph for one sample, the shutter sample loop, collection and timing.
use super::*;
use crate::view::ViewState;

/// Sticky per-view pixel flags share the existing packed output buffer.
pub(crate) fn invalid_pixel_bitmap_bytes(pixels: u64) -> u64 {
    pixels.div_ceil(u64::from(u32::BITS)) * 4
}

/// Bounded witnesses for diagnostics; the compact bitmap owns complete delivery marks.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct InvalidPixels {
    pub count: u64,
    pub coordinates: Vec<[u32; 2]>,
    #[serde(skip)]
    bitmap: Vec<u8>,
    #[serde(skip)]
    extent: [u32; 2],
}
const INVALID_PIXEL_WITNESSES: usize = 32;

fn invalid_pixel_coordinates(bitmap: &[u8], [width, height]: [u32; 2]) -> impl Iterator<Item = [u32; 2]> + '_ {
    let count = u64::from(width) * u64::from(height);
    bitmap.chunks_exact(4).enumerate().flat_map(|(word, bytes)| {
        let mut bits = u32::from_le_bytes(bytes.try_into().unwrap());
        std::iter::from_fn(move || {
            if bits == 0 { return None; }
            let pixel = word as u64 * u64::from(u32::BITS) + u64::from(bits.trailing_zeros());
            bits &= bits - 1;
            Some(pixel)
        })
    }).filter(move |&pixel| pixel < count)
        .map(move |pixel| [(pixel % u64::from(width)) as u32, (pixel / u64::from(width)) as u32])
}
impl InvalidPixels {
    fn merge(&mut self, bitmap: &[u8], extent: [u32; 2]) {
        self.extent = extent;
        self.bitmap.resize(bitmap.len(), 0);
        for (stored, new) in self.bitmap.iter_mut().zip(bitmap) { *stored |= new; }
        self.count = invalid_pixel_coordinates(&self.bitmap, extent).count() as u64;
        self.coordinates = invalid_pixel_coordinates(&self.bitmap, extent).take(INVALID_PIXEL_WITNESSES).collect();
    }
}

#[cfg(test)]
mod invalid_pixel_bitmap_tests {
    use super::*;

    #[test]
    fn coordinates_cross_word_and_image_boundaries_without_counting_padding() {
        let bitmap = [0x80000001u32.to_le_bytes(), u32::MAX.to_le_bytes()].concat();
        assert_eq!(invalid_pixel_bitmap_bytes(33), 8);
        assert_eq!(invalid_pixel_coordinates(&bitmap, [11, 3]).collect::<Vec<_>>(), vec![[0, 0], [9, 2], [10, 2]]);
        assert!(invalid_pixel_coordinates(&[0; 8], [11, 3]).next().is_none());
        let mut marked = InvalidPixels::default();
        marked.merge(&[255; 512], [64, 64]);
        assert_eq!(marked.count, 4096);
        assert_eq!(marked.coordinates.len(), INVALID_PIXEL_WITNESSES);
        assert_eq!(invalid_pixel_coordinates(&marked.bitmap, marked.extent).count(), 4096);
        let json = serde_json::to_string(&marked).unwrap();
        assert!(json.len() < 1024 && !json.contains("bitmap"));
        marked.merge(&[255; 512], [64, 64]);
        assert_eq!(marked.count, 4096, "repeated capture witnesses are a union");
    }
}

/// Storage an installation reserves that collection can outgrow (`FrameRedo`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reservations {
    /// Device storage for actual dielectric interiors beyond the inline list.
    pub dielectric_spill: u64,
    /// Measured visibility overflow demand; speculative capacity is device-budget limited.
    pub visibility_pool: u32,
    /// Submission-ordered shadow demand generation.
    /// Submission-ordered sparse probe support generation.
    pub probe_pages: u32,
    /// Decoupled transparent fragment records per merge batch; each view reserves at least two per pixel.
    pub transparent_fragments: u32,
    /// Retained camera material evaluations of the whole frame, shared by all
    /// material bins and split across shading bands (passes/decoupled.rs). Also
    /// covers deferred connection launch demand, in the same evaluation units.
    pub shading_evaluations: u64,
}
impl Default for Reservations {
    fn default() -> Self {
        Self { dielectric_spill: 0, probe_pages: 0, visibility_pool: 0, transparent_fragments: 0, shading_evaluations: 0 }
    }
}
impl Reservations {
    /// Each reservation's larger value.
    pub fn max(self, other: Self) -> Self {
        Self {
            dielectric_spill: self.dielectric_spill.max(other.dielectric_spill),
            visibility_pool: self.visibility_pool.max(other.visibility_pool),
            probe_pages: self.probe_pages.max(other.probe_pages),
            transparent_fragments: self.transparent_fragments.max(other.transparent_fragments),
            shading_evaluations: self.shading_evaluations.max(other.shading_evaluations),
        }
    }
}
/// This exposure must be redone with the reported storage demand. No output
/// from this exposure may be delivered. Independently accepted successors can
/// survive recovery in their output ring entries.
#[derive(Debug)]
pub struct FrameRedo(pub Reservations);
impl std::fmt::Display for FrameRedo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "redo frame with at least {:?}", self.0)
    }
}
impl std::error::Error for FrameRedo {}
/// The shader rejected this exposure's values (non-finite radiance, say). No
/// output from it may be delivered; later exposures do not depend on it, so a
/// frame sequence can skip it and report it.
#[derive(Debug)]
pub struct FrameRefused(pub String);
impl std::fmt::Display for FrameRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for FrameRefused {}
/// One frame's result from its publication receipts (`checked`) and its own collection.
/// Detailed renderer diagnostics are kept unless the producer supplied its own actionable
/// scene error before the downstream failure. A device failure is not such a diagnostic:
/// it outranks what a receipt read from the failed device says about the scene.
fn collected<T>(checked: Result<()>, result: Result<T>) -> Result<T> {
    let device = |error: &anyhow::Error| matches!(crate::failure::kind(error), Some(crate::failure::FailureKind::Device));
    match (checked, result) {
        (Err(_), Err(error)) if device(&error) => Err(error),
        (Err(error), _) if crate::failure::kind(&error).is_some() => Err(error),
        (checked, result) => result.and_then(|output| checked.map(|()| output)),
    }
}

// Incomplete storage can poison provisional colour. Replay that exposure before
// judging colour, but preserve immediate refusal for independent shader failures.
/// Each replay of an exhausted dielectric spill arena grows it this many times.
/// The arena starts small and a frame that needs it is replayed in full for
/// every step, so the step is larger than a doubling. For a frame whose need
/// does not change between replays the reservation ends at most this factor
/// above that need, after half the replays doubling takes, rounded up.
const SPILL_GROWTH: u64 = 4;
/// The size an exhausted dielectric spill arena of `bytes` grows to, or None
/// once it is as large as the device can bind. The growth is blind: nothing
/// measured in the exhausted frame sizes it.
fn spill_growth(bytes: u64, maximum: u64) -> Option<u64> {
    (bytes<maximum).then(||bytes.saturating_mul(SPILL_GROWTH).min(maximum))
}
fn reservation_retry_allowed(errors: Option<[u32; 2]>) -> bool {
    errors.is_none_or(|[code, _]| matches!(code,
        0 | vork_slang::registry::RADIANCE_NONFINITE | vork_slang::registry::DISPLAY_NONFINITE))
}

pub struct FrameSlot {
    pub(crate) dielectric_spill_reservation: u64,
    boundary_names: Vec<String>,
    pub(crate) nominal_frame: u32,
    pub(crate) nominal_request: vork_time::Request,
    pub(crate) sample_request: vork_time::Request,
    pub(crate) previous_request: vork_time::Request,
    pub(crate) exposure_cut: bool,
    propagation_motion: bool,
    warned_propagation_motion: bool,
    _retained_transients: std::sync::Arc<std::sync::Mutex<crate::graph::State>>,
    publication_error: std::sync::Arc<std::sync::OnceLock<String>>,
    publication_checks: Option<crate::graph::PublicationChecks>,
    solver_receipts: Vec<(crate::graph::PublicationDiagnostic, wgpu::Buffer)>,
    _retained_shadows: std::sync::Arc<crate::shadows::Persistent>,
    /// The frame's decoupled overflow pool and transparent fragment demands,
    /// nonzero once either was full (passes/decoupled.rs).
    slot_overflow: Option<wgpu::Buffer>,
    pub(crate) compute: Option<compute::Slot>,
    pub(crate) simulation_checkpoint: Option<compute::Checkpoint>,
    pub(crate) compute_commands: Option<wgpu::Buffer>,
    pub(crate) view: ViewState,
    /// Native-resolution view for control outputs under temporal upsampling,
    /// created on first use; it shares the main view's packed planes (same transient).
    controls: Option<ViewState>,
    /// The other spatial/temporal recording for an animated shutter. Both share
    /// the output texture; only the selected recording supplies this exposure.
    shutter_view: Option<ViewState>,
    capture_images: bool,
    seed_views: Vec<crate::render_images::CaptureView>,
    seed_queries: Option<wgpu::QuerySet>,
    pub(crate) image_views: Vec<crate::render_images::CaptureView>,
    pub(crate) evaluated_lights: Vec<Light>,
    /// Frame-private material records when tracks animate material inputs.
    pub(crate) materials: Option<materials::FrameMaterials>,
    pub(crate) changes: values::ChangeTracker,
    /// Consecutive colour samples share time-dependent scene work within one exposure.
    pub(crate) reuse_values: bool,
    /// Shutter sampled at nominal frame time, shared by all step intervals.
    pub(crate) exposure_shutter: Option<vork_scene::MotionBlur>,
    pub(crate) receiver_masks: u32,
    pub(crate) deforms: Vec<Option<EvaluatedGeometry>>,
    pub(crate) geometry_chain: crate::geometry_chain::Frame,
    pub(crate) geometry_bounds: Vec<Option<(Vec3,Vec3)>>,
    /// Frame-private TLAS and deformed BLAS of scenes with a ray technique.
    pub(crate) rays: Option<crate::rt::FrameRays>,
    pub(crate) camera_data: CameraRecord,
    pub(crate) objects: wgpu::Buffer,
    pub(crate) worlds: Vec<Mat4>,
    pub(crate) previous_worlds: Vec<Mat4>,
    pub(crate) next_worlds: Vec<Mat4>,
    pub(crate) next_animation_values: animation::Samples,
    pub(crate) next_nodes: Vec<Option<install::EvalNode>>,
    pub(crate) animation_values: animation::Samples,
    pub(crate) previous_animation_values: animation::Samples,
    pub(crate) offset_values: values::OffsetValues,
    pub(crate) sampled_nodes: Vec<Option<install::EvalNode>>,
    pub(crate) previous_nodes: Vec<Option<install::EvalNode>>,
    pub(crate) object_values: Vec<Object>,
    pub(crate) sample_upload: wgpu::Buffer,
    pub(crate) exposure_upload: Vec<(wgpu::Buffer, Vec<u8>)>,
    batch_uploads: bool,
    pub(crate) shadow_bind: wgpu::BindGroup,
    pub(crate) probe_generation: u32,
    probe_feedback: Option<(crate::probes::Feedback, wgpu::Buffer)>,
    probe_demand: Option<std::sync::Arc<std::sync::Mutex<crate::probes::Demand>>>,

    pub(crate) ring: crate::ring::Entry,
    pub(crate) spare_ring: std::collections::VecDeque<crate::ring::Entry>,
    pub(crate) queries: wgpu::QuerySet,
    sample_queries: Vec<wgpu::QuerySet>,
    pub(crate) query_resolve: wgpu::Buffer,
    pub(crate) query_readback: wgpu::Buffer,
    shader_error_readback: Option<wgpu::Buffer>,
    computed_bounds: computed_geometry::BoundsPublication,
    pub(crate) point_publication: point_publication::Frame,
    environment_diagnostics:Option<crate::environment::Validation>,
    nonfinite_readback: Option<wgpu::Buffer>,
    /// Display exposure report of the final colour sample.
    exposure_report: crate::display_stats::Stats,
    pub(crate) pending: bool,
    flight_ownership: crate::command_finish::FlightOwnership,
    pub(crate) completed: Option<CompletedFrame>,
    pub(crate) submission: Option<wgpu::SubmissionIndex>,
    /// Scene time of the submitted frame and the work it runs, named when the GPU fails it.
    submitted_time: f32,
    /// What drew the submitted frame's path transport (FrameTiming::path_kernel).
    path_kernel: Option<&'static str>,
    work: std::sync::Arc<str>,
    /// Collect optional named GPU timings and nonfinite diagnostics.
    pub profile: bool,
    /// Sample cost estimates rescaled from the last collected group envelopes.
    sample_gpu_ms: Vec<f64>,
    group_ends: Vec<bool>,
    /// CPU-bound flights submit ready validation prefixes; cold flights defer.
    pub(crate) stream_samples: bool,
    pub(crate) read_pixels: bool,
    pub(crate) cpu_ms: f64,
    denoise_ms: denoise::Timing,
    pub(crate) prep_ms: f64,
    pub(crate) finish_ms: f64,
    /// This slot's probe volume update bindings (its TLAS, hits and globals).
    probes: Option<crate::probes::Slot>,
    pub(crate) fibre: Option<crate::fibre_transport::Slot>,
    sample_spans: Vec<u64>,
    sample_segments: crate::command_finish::Segments,
}
/// An accepted queued exposure, held in its existing output entry while another
/// slot grows. It owns the mapped-pixel lease and the original completion fence.
pub(crate) struct CompletedFrame {
    nominal_frame: u32,
    nominal_request: vork_time::Request,
    time: f32,
    outputs: ReadbackOutputs,
    profile: bool,
    write_color: bool,
    submission: wgpu::SubmissionIndex,
    readback: FrameReadback,
}
impl std::ops::Deref for FrameSlot {
    type Target = ViewState;
    fn deref(&self) -> &ViewState {
        &self.view
    }
}
impl std::ops::DerefMut for FrameSlot {
    fn deref_mut(&mut self) -> &mut ViewState {
        &mut self.view
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct NonfiniteCounts {
    pub shade: u64,
    pub resolve: u64,
}
#[derive(Debug, Clone)]
pub struct FrameTiming {
    pub cpu_prepare_ms: f64,
    pub cpu_submit_ms: f64,
    pub cpu_finish_ms: f64,
    /// Host wall time submitting and waiting for OIDN inputs; includes preceding render work.
    pub denoise_wait_ms: f64,
    /// Host wall time unpacking and preparing CPU OIDN inputs; zero for GPU interop.
    pub denoise_pack_ms: f64,
    /// OIDN buffer transfers, filter commit and execution wall time (CPU or GPU backend).
    pub denoise_filter_ms: f64,
    /// Host wall time repacking and queuing CPU denoised output; zero for GPU interop.
    pub denoise_upload_ms: f64,
    pub gpu_ms: f64,
    /// Validated GPU counter intervals; invalid counters discard the frame.
    pub timing_source: &'static str,
    /// Bad RGBA evaluations summed across views and exposure samples, diagnostics only.
    pub nonfinite: Option<NonfiniteCounts>,
    /// Non-finite light contributions dropped, collected even without diagnostics.
    pub dropped_light_samples: u64,
    /// Camera path samples zeroed before accumulation for non-finite transport.
    pub dropped_path_samples: u64,
    /// Exact counts and bounded witnesses for the camera and named sampled-image views.
    pub invalid_pixels: std::collections::BTreeMap<String, InvalidPixels>,
    pub terminated_medium_paths: u64,
    /// Absolute luminance sum of evaluated path radiance discarded by fog caps.
    /// Includes catcher auxiliary paths; excludes the unknown untraced tail.
    pub discarded_medium_radiance: f64,
    /// First-to-last written GPU timestamp, including intervals between named spans.
    pub gpu_elapsed_ms: f64,
    /// Named GPU spans in milliseconds, summed over shutter samples.
    pub gpu_spans_ms: Vec<(&'static str, f64)>,
    pub sample_count: usize,
    /// This frame drew at least one superset fallback: a background variant of its
    /// scene had not landed when the exposure was submitted.
    pub variants_pending: bool,
    /// The kernel that drew this frame's path transport, snapshotted when the
    /// exposure was submitted: "prebuilt engine" or "scene kernel"; None without path transport.
    pub path_kernel: Option<&'static str>,
}
/// Storage precision is chosen once when a frame slot is installed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReadbackPrecision {
    #[default]
    Packed,
    Float,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct FloatOutputs {
    pub color: bool,
    pub depth: bool,
    pub normals: bool,
    pub motion: bool,
}
impl FloatOutputs {
    pub const NONE: Self = Self {
        color: false,
        depth: false,
        normals: false,
        motion: false,
    };
    pub const ALL: Self = Self {
        color: true,
        depth: true,
        normals: true,
        motion: true,
    };
    pub fn any(self) -> bool {
        self.color || self.depth || self.normals || self.motion
    }
}
/// Packed GPU planes to transfer for CPU outputs. Depth and IDs share a plane.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReadbackOutputs {
    pub color: bool,
    /// Display-referred RGBA16 colour, packed by the display stage into its staging planes.
    pub color16: bool,
    pub depth_ids: bool,
    pub normals: bool,
    pub motion: bool,
    pub float: FloatOutputs,
    pub aovs: bool,
}
impl ReadbackOutputs {
    pub const ALL: Self = Self {
        color: true,
        color16: false,
        depth_ids: true,
        normals: true,
        motion: true,
        float: FloatOutputs::NONE,
        aovs: true,
    };
    pub fn any(self) -> bool {
        self.color || self.color16 || self.depth_ids || self.normals || self.motion || self.aovs || self.float.any()
    }
    /// Every plane `other` requests is one of these.
    pub fn contains(self, other: Self) -> bool {
        let (a, b) = (self, other);
        (a.aovs || !b.aovs)
            && (a.color || !b.color)
            && (a.color16 || !b.color16)
            && (a.depth_ids || !b.depth_ids)
            && (a.normals || !b.normals)
            && (a.motion || !b.motion)
            && (a.float.color || !b.float.color)
            && (a.float.depth || !b.float.depth)
            && (a.float.normals || !b.float.normals)
            && (a.float.motion || !b.float.motion)
    }
}
#[derive(Clone)]
pub struct AovPixels {
    pub kind: vork_scene::v2::AovType,
    pub pixels: Vec<[f32;4]>,
}
#[derive(Clone)]
pub struct FrameOutput {
    /// Named scene-linear program AOVs, associated with surface coverage.
    pub aovs: std::collections::BTreeMap<String,AovPixels>,
    pub rgba: Vec<u8>,
    /// Display-referred RGBA, 16 bits per component; filled for `ReadbackOutputs::color16`.
    pub rgba16: Vec<[u16; 4]>,
    pub depth_u16: Vec<u16>,
    pub ids: Vec<u16>,
    pub normals: Vec<u8>,
    pub motion: Vec<[u16; 2]>,
    /// Premultiplied scene-linear radiance before exposure and display conversion.
    pub linear_rgba: Option<Vec<[f32; 4]>>,
    /// View-axis metres; Blender Z background sentinel 1e10.
    pub depth_f32: Option<Vec<f32>>,
    /// Signed normalized view-space normals; zero for background.
    pub normals_f32: Option<Vec<[f32; 3]>>,
    /// Current minus previous pixel position; +x right, +y down.
    pub motion_f32: Option<Vec<[f32; 2]>>,
    pub camera: serde_json::Value,
    pub exposure: Option<crate::Exposure>,
    pub timing: FrameTiming,
}
/// Frame metadata and a mapped pixel lease; cloning shares both without copies.
#[derive(Clone)]
pub struct FrameReadback {
    pub camera: CameraRecord,
    /// Display exposure report; None when the frame wrote no colour.
    pub exposure: Option<crate::Exposure>,
    pub timing: FrameTiming,
    readback: Option<std::sync::Arc<ReadbackLease>>,
    pub(crate) credit: Option<std::sync::Arc<crate::ring::Lease>>,
}
struct ReadbackLease {
    neutral_motion: bool,
    buffer: wgpu::Buffer,
    // Multiple output directories may share this readback. wgpu mapped ranges
    // cannot overlap, so serialize unpacking while sharing the same storage.
    mapping: std::sync::Mutex<()>,
    pixel_count: usize,
    outputs: ReadbackOutputs,
    aovs: Vec<vork_scene::v2::Aov>,
    aov_base: usize,
    /// First display staging plane; it holds RGBA16 after the display stage.
    display_base: Option<usize>,
    has_catchers: bool,
}
impl Drop for ReadbackLease {
    fn drop(&mut self) {
        self.buffer.unmap();
    }
}
impl FrameReadback {
    /// Unpack on the caller's thread; the CLI calls this only on PNG workers.
    pub fn unpack(self) -> Result<FrameOutput> {
        let mut output = FrameOutput {
            aovs: Default::default(),
            rgba: Vec::new(),
            rgba16: Vec::new(),
            depth_u16: Vec::new(),
            ids: Vec::new(),
            normals: Vec::new(),
            motion: Vec::new(),
            linear_rgba: None,
            depth_f32: None,
            normals_f32: None,
            motion_f32: None,
            camera: self.camera.into_value(),
            exposure: self.exposure,
            timing: self.timing,
        };
        if let Some(raw) = self.readback {
            let _mapping = raw.mapping.lock().unwrap();
            let mapped = raw.buffer.get_mapped_range(..);
            let count = raw.pixel_count;
            let rgba = if raw.outputs.color {
                mapped[..count * 4].to_vec()
            } else {
                Vec::new()
            };
            let mut depth = Vec::with_capacity(if raw.outputs.depth_ids { count } else { 0 });
            let mut ids = Vec::with_capacity(if raw.outputs.depth_ids { count } else { 0 });
            if raw.outputs.depth_ids {
                for b in mapped[count * 4..count * 8].chunks_exact(4) {
                    let v = u32::from_le_bytes(b.try_into().unwrap());
                    depth.push(v as u16);
                    ids.push((v >> 16) as u16);
                }
            }
            let normals = if raw.outputs.normals {
                mapped[count * 8..count * 12]
                    .chunks_exact(4)
                    .flat_map(|v| v[..3].iter().copied())
                    .collect()
            } else {
                Vec::new()
            };
            let motion = if raw.outputs.motion {
                mapped[count * 12..count * 16]
                    .chunks_exact(4)
                    .map(|b| {
                        let v = u32::from_le_bytes(b.try_into().unwrap());
                        [v as u16, (v >> 16) as u16]
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let value = |plane: usize, pixel: usize| {
                let offset = (plane * count + pixel) * 4;
                f32::from_le_bytes(mapped[offset..offset + 4].try_into().unwrap())
            };
            if raw.outputs.color16 {
                // pack2x16unorm: the first component in the low half of each word.
                let base = raw.display_base.context("16-bit colour readback requires a display stage")?;
                let word = |plane: usize, pixel: usize| {
                    let offset = (plane * count + pixel) * 4;
                    u32::from_le_bytes(mapped[offset..offset + 4].try_into().unwrap())
                };
                output.rgba16 = (0..count)
                    .map(|i| {
                        let (rg, ba) = (word(base, i), word(base + 1, i));
                        [rg as u16, (rg >> 16) as u16, ba as u16, (ba >> 16) as u16]
                    })
                    .collect();
            }
            if raw.outputs.aovs && !raw.has_catchers {
                output.aovs.insert("shadow_catcher".into(),AovPixels {kind:vork_scene::v2::AovType::Color,pixels:vec![[1.,1.,1.,1.];count]});
            }
            if raw.outputs.float.color {
                output.linear_rgba = Some(
                    (0..count)
                        .map(|i| {
                            std::array::from_fn(|c| value(constants::FLOAT_COLOR_PLANE + c, i))
                        })
                        .collect(),
                );
            }
            if raw.outputs.float.depth {
                output.depth_f32 = Some(
                    (0..count)
                        .map(|i| value(constants::FLOAT_DEPTH_PLANE, i))
                        .collect(),
                );
            }
            if raw.outputs.float.normals {
                output.normals_f32 = Some(
                    (0..count)
                        .map(|i| {
                            std::array::from_fn(|c| value(constants::FLOAT_NORMAL_PLANE + c, i))
                        })
                        .collect(),
                );
            }
            if raw.outputs.float.motion {
                output.motion_f32 = Some(
                    (0..count)
                        .map(|i| {
                            std::array::from_fn(|c| value(constants::FLOAT_MOTION_PLANE + c, i))
                        })
                        .collect(),
                );
            }
            if raw.outputs.aovs {
                for (i,aov) in raw.aovs.iter().enumerate() {
                    if aov.name=="shadow_catcher" && !raw.has_catchers {continue;}
                    output.aovs.insert(aov.name.clone(),AovPixels{kind:aov.kind,pixels:(0..count).map(|pixel|std::array::from_fn(|c|value(raw.aov_base+i*4+c,pixel))).collect()});
                }
            }
            drop(mapped);

            output.rgba = rgba;
            output.depth_u16 = depth;
            output.ids = ids;
            output.normals = normals;
            output.motion = motion;
            if raw.neutral_motion {
                output.motion.fill([32768,32768]);
                if let Some(motion)=&mut output.motion_f32 {motion.fill([0.,0.]);}
            }
            output.mark_invalid_pixels();
        }
        Ok(output)
    }
}
impl FrameOutput {
    // Delivery is after denoising, reconstruction, compositing and display.
    // The marker is opaque even when the failed sample had transparent coverage.
    fn mark_invalid_pixels(&mut self) {
        let Some(pixels) = self.timing.invalid_pixels.get("camera") else { return; };
        let rgba = vork_slang::registry::DIAGNOSTIC_MAGENTA_RGBA.to_le_bytes();
        let rgba16 = rgba.map(|c| u16::from(c) * 257);
        let linear = rgba.map(|c| f32::from(c) / 255.);
        for [x, y] in invalid_pixel_coordinates(&pixels.bitmap, pixels.extent) {
            let pixel = y as usize * pixels.extent[0] as usize + x as usize;
            if !self.rgba.is_empty() { self.rgba[pixel * 4..pixel * 4 + 4].copy_from_slice(&rgba); }
            if !self.rgba16.is_empty() { self.rgba16[pixel] = rgba16; }
            if let Some(output) = &mut self.linear_rgba { output[pixel] = linear; }
        }
    }
}
impl FrameSlot {
    fn views_mut(&mut self) -> impl Iterator<Item = &mut ViewState> {
        std::iter::once(&mut self.view).chain(self.image_views.iter_mut().chain(&mut self.seed_views).map(|capture| &mut capture.view))
    }
    fn views(&self) -> impl Iterator<Item = &ViewState> {
        std::iter::once(&self.view).chain(self.image_views.iter().chain(&self.seed_views).map(|capture| &capture.view))
    }

    pub(crate) fn new(
        s: &InstalledScene,
        r: &Renderer,
        precision: ReadbackPrecision,
    ) -> Result<Self> {
        let retained_transients = s.transients.retain();
        let retained_shadows = s.retain_shadows(r)?;
        let d = &r.device;
        // Storage bindings need one complete record even when there are no draws.
        let object_values = gpu_objects(&s.objects);
        let object_bytes = bytemuck::cast_slice(&object_values);
        let objects = init(
            d,
            "frame objects",
            object_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let geometry_chain = crate::geometry_chain::Frame::new(r, s)?;
        let deforms = geometry_chain.outputs();
        let point_publication=point_publication::Frame::new(r,s,&deforms)?;
        let point_buffer=point_publication.buffer.clone();
        let no_clusters = init(
            d,
            "unclustered config",
            &[0; 32],
            wgpu::BufferUsages::UNIFORM,
        );
        let empty_mask = init(d, "unclustered mask", &[0; 4], wgpu::BufferUsages::STORAGE);
        let make_global = |uniform: &wgpu::Buffer,
                           lights: &wgpu::Buffer,
                           config: &wgpu::Buffer,
                           mask: &wgpu::Buffer,
                           objects: &wgpu::Buffer| {
            crate::view::global(s, d, uniform, lights, config, mask, objects, &point_buffer, None)
        };
        let dummy_lights = init(
            d,
            "empty light",
            bytemuck::bytes_of(&Light::zeroed()),
            wgpu::BufferUsages::STORAGE,
        );

        let mut rays = crate::rt::FrameRays::new(s, r, &objects, &deforms)?;
        if let Some(rays) = rays.as_mut() { rays.resolve_hit_buffers(s, &deforms); }
        let shadow_bind = s.shadows().bind(s, d, rays.as_ref());
        r.errors.check()?;
        let image_targets: Vec<_> = s.graph.views.image_renders().map(|image| image.output.clone()).collect();
        let frame_materials = (s.compute.as_ref().is_some_and(|c|c.has_material_attributes()) || s.dense.is_some() || s.draws.iter().any(|d|d.computed) || s.displacement.is_some() || s.animation.has_material_bindings()
            || s.material_table.has_render_images() || s.checked_shader_capacity
            || (s.rays.is_some() && s.draws.iter().any(|draw|draw.evaluated())))
        .then(|| s.material_table.frame_with_images(r, &image_targets,s.reservations.dielectric_spill));
        let computed_bounds=computed_geometry::BoundsPublication::new(r,s,&deforms,frame_materials.as_ref());
        let compute_commands = s
            .compute
            .as_ref()
            .filter(|c| c.has_instances())
            .map(|_| compute::Compute::commands(r, &s.draws));
        let queries = d.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU frame passes"),
            ty: wgpu::QueryType::Timestamp,
            count: timing::count(),
        });
        let view = ViewState::new(
            s,
            r,
            &s.render,
            precision,
            &s.graph.views.entries[s.graph.views.camera],
            &queries,
            crate::view::SharedBindings {
                points: &point_buffer,
                rays: rays.as_ref(),
                compute: compute_commands.as_ref(),
                objects: &objects,
                deforms: &deforms,
                materials: &frame_materials,
                shadow_bind: &shadow_bind,
                output: None,
            },
            s.main_subset.as_ref(),
        )?;
        if let (Some(installed),Some(frame_rays))=(&s.rays,rays.as_mut()) {
            if let Some(plan)=&installed.emitter_weights {
                let hits=frame_rays.hits_bind(d);
                if let Some(slot)=&mut frame_rays.emitter_weights {slot.prepare(plan,s,d,&view.lighting,&objects,frame_materials.as_ref().map_or_else(||s.material_table.records(),|m|&m.records),hits);}
            }
        }
        let probes = match (&s.probes, rays.as_ref()) {
            (Some(volume), Some(frame_rays)) => Some(volume.slot(s, d, &view, &objects, frame_rays,
                frame_materials.as_ref().map_or_else(|| s.material_table.bind(), |m| &m.bind))?),
            _ => None,
        };
        // Shadow bundles see the complete scene, independent of the visible subtree.
        let recorder = crate::bundles::Recorder {
            shadow_bind: &shadow_bind,
            graph: &s.graph,
            transparent_instances: None,
            camera_instances: None,
            compute: compute_commands.as_ref(),
            included: None,
            included_objects: None,
            fibre_capture: None,
            visibility: None,
            fragments: None,
            fragments_geometry: None,
            material_bind: frame_materials
                .as_ref()
                .map_or_else(|| s.material_table.bind(), |m| &m.bind),
            s,
            deforms: &deforms,
            r,
        };
        let fibre=s.fibre.as_ref().map(|p|crate::fibre_transport::Slot::new(&p.storage(),&recorder,|u|make_global(u,&dummy_lights,&no_clusters,&empty_mask,&objects))).transpose()?;
        let compute = s
            .compute
            .as_ref()
            .filter(|c| c.has_instances())
            .map(|c| {
                c.slot(
                    r,
                    s,
                    &objects,
                    compute_commands.as_ref().unwrap().clone(),
                )
            })
            .transpose()?;
        let buffer = |label, size, usage| {
            crate::gpu_memory::buffer(&d, &wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let query_resolve = buffer(
            "timestamps resolve",
            timing::resolve_stride() * s.plan.samples,
            wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        );
        let query_readback = buffer(
            "timestamps readback",
            timing::resolve_stride() * s.plan.samples,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        // Dirty uploads are known only after sample evaluation. Reserve no scene-
        // sized regions here: each recorded sample retains exactly its packed bytes.
        let batch_uploads = Self::batch_plan(s, &view);
        let sample_upload = buffer(
            "sample values", 4,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let ring = crate::ring::Entry::new(
            d,
            &view.post.pixels,
            u64::from(view.width) * u64::from(view.height) * 4,
            view.float_output,
            s.readback,
            view.aovs.len(),
            view.post.display_base,
        );
        let solids=crate::passes::path::solids(&s.draws,&s.roots,&s.materials)?;
        let mut boundary_names=vec![String::new();solids.count as usize];
        for (row,&solid) in solids.owners.iter().enumerate() {
            if solid==u32::MAX {continue;}
            let root=&s.roots[row];let name=&s.nodes[root.node].node.name;
            let name=if name.is_empty() {format!("node {}",root.node)} else {name.clone()};
            boundary_names[solid as usize]=root.instance_index.map_or_else(||name.clone(),|instance|format!("{name}[{instance}]"));
        }
        let mut slot = Self {
            dielectric_spill_reservation: s.reservations.dielectric_spill,
            boundary_names,
            computed_bounds,
            point_publication,
            propagation_motion: s.scene.propagation.is_some(),
            warned_propagation_motion: false,
            _retained_shadows: retained_shadows,
            _retained_transients: retained_transients,
            publication_error: s.transients.recording_error.clone(),
            solver_receipts: Vec::new(),
            publication_checks: (s.has_media || s.compute.is_some() || s.displacement.is_some() || s.implicit_fields.iter().any(|f|!f.statuses.is_empty()) || !image_targets.is_empty()).then(||s.transients.publication_checks.clone()),
            simulation_checkpoint: None,
            slot_overflow: s.plan.decoupled.then(|| buffer(
                "visibility slot overflow readback", 8 * (1 + s.graph.views.images.len() * if s.plan.effects.image_feedback {2} else {1}) as u64,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )),
            view,
            controls: None,
            shutter_view: None,
            capture_images: false,
            seed_views: Vec::new(),
            seed_queries: s.plan.effects.image_feedback.then(|| d.create_query_set(&wgpu::QuerySetDescriptor {label: Some("seed capture timestamps"), ty: wgpu::QueryType::Timestamp, count: timing::count()})),
            image_views: Vec::new(),
            evaluated_lights: Vec::new(),
            materials: frame_materials,
            compute,
            compute_commands,
            changes: Default::default(),
            reuse_values: false,
            nominal_frame: 0,
            nominal_request: vork_time::Request::Numeric(0.),
            sample_request: vork_time::Request::Numeric(0.),
            previous_request: vork_time::Request::Numeric(0.),
            exposure_cut: false,
            exposure_shutter: s.render.motion_blur.clone(),
            receiver_masks: 0,
            sample_upload,
            exposure_upload: Vec::new(),
            batch_uploads,
            deforms,
            geometry_chain,
            geometry_bounds: vec![None;s.draws.len()],
            rays,
            camera_data: CameraRecord::default(),
            objects,
            worlds: s.nodes.iter().map(|n| n.world).collect(),
            previous_worlds: s.nodes.iter().map(|n| n.world).collect(),
            next_worlds: s.nodes.iter().map(|n| n.world).collect(),
            next_animation_values: s.animation.samples(),
            next_nodes: values::sample_storage(&s.nodes),
            animation_values: s.animation.samples(),
            previous_animation_values: s.animation.samples(),
            offset_values: values::offset_storage(s),
            sampled_nodes: values::sample_storage(&s.nodes),
            previous_nodes: values::sample_storage(&s.nodes),
            object_values: s.objects.clone(),
            shadow_bind,
            probe_generation: s.reservations.probe_pages,
            probe_feedback: s.probes.as_ref().and_then(|p|p.feedback()).map(|source|
                (source,buffer("probe request status",4,wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST))),
            probe_demand: s.probes.as_ref().map(|p|p.demand.clone()),
            ring,
            spare_ring: std::collections::VecDeque::with_capacity(constants::OUTPUT_RING_DEPTH - 1),
            sample_queries: vec![queries.clone()],
            queries,
            query_resolve,
            nonfinite_readback: None,
            exposure_report: crate::display_stats::Stats::new(d),
            shader_error_readback: s.checked_shader_capacity.then(|| buffer("shader diagnostics readback", crate::materials::SHADER_DIAGNOSTICS_BYTES, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)),
            environment_diagnostics:s.environment.diagnostics(r),
            query_readback,
            pending: false,
            flight_ownership: Default::default(),
            completed: None,
            submission: None,
            submitted_time: 0.,
            path_kernel: None,
            work: s.gpu_work().into(),
            profile: false,
            sample_gpu_ms: Vec::new(),
            group_ends: Vec::new(),
            stream_samples: false,
            read_pixels: true,
            cpu_ms: 0.,
            denoise_ms: denoise::Timing::default(),
            prep_ms: 0.,
            finish_ms: 0.,
            probes,
            fibre,
            sample_spans: Vec::new(),
            sample_segments: Default::default(),
        };
        for (index, image) in s.graph.views.image_renders().enumerate() {
            let desc = &s.graph.views.entries[s.graph.views.images.start + index];
            let view = crate::render_images::CaptureView::new(s, r, &mut slot, desc, image)?;
            slot.image_views.push(view);
            if s.plan.effects.image_feedback {
                let seed = crate::render_images::CaptureView::new(s, r, &mut slot, desc, image)?;
                slot.seed_views.push(seed);
            }
        }
        if slot.shader_error_readback.is_some() {
            let bytes = crate::materials::SHADER_DIAGNOSTICS_BYTES + slot.views()
                .map(|view| view.post.invalid_bitmap.end - view.post.invalid_bitmap.start).sum::<u64>();
            slot.shader_error_readback = Some(buffer("shader diagnostics readback", bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST));
        }
        Ok(slot)
    }
    pub(crate) fn advance_ring(&mut self) {
        self.completed = None;
        if let Some(next) = self.spare_ring.pop_front() {
            let previous = std::mem::replace(&mut self.ring, next);
            self.spare_ring.push_back(previous);
            if let Some((texture, bind)) = &self.ring.output {
                self.view.output = texture.clone();
                self.view.output_bind = bind.clone();
            }
        }
    }
    /// Set an acquired output surface before submitting this slot.
    pub fn set_output(
        &mut self,
        _s: &InstalledScene,
        r: &Renderer,
        output: wgpu::Texture,
    ) -> Result<()> {
        ensure!(
            !self.pending,
            "collect this frame slot before replacing its output"
        );
        ensure!(
            output.width() == self.view.width
                && output.height() == self.view.height
                && output.format() == r.output_format(),
            "output surface mismatch"
        );
        self.completed = None;
        // The exposure report reads the finished display image as a texture.
        ensure!(
            output.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING),
            "output surface needs TEXTURE_BINDING usage"
        );
        self.view.output_bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("encoder surface"),
            layout: &self.view.post.output_layout,
            entries: &[entry(
                0,
                wgpu::BindingResource::TextureView(&output.create_view(&Default::default())),
            )],
        });
        self.view.output = output;
        r.errors.check()?;
        Ok(())
    }
    /// Queues a frame. Receiver storage is sized once at its first capture;
    /// retained samples do not wait or map. Compute advances in queue order.
    pub fn submit(&mut self, s: &InstalledScene, r: &Renderer, time: f32) -> Result<()> {
        self.submit_outputs(s, r, time, ReadbackOutputs {
            float: if self.view.float_output { FloatOutputs::ALL } else { FloatOutputs::NONE },
            ..ReadbackOutputs::ALL
        })
    }
    /// Leaves colour on the GPU and collects only timestamp data. Encoder handoff
    /// can use `output`; collection returns empty pixel vectors in this mode.
    pub fn submit_gpu(&mut self, s: &InstalledScene, r: &Renderer, time: f32) -> Result<()> {
        self.submit_outputs(s, r, time, ReadbackOutputs::default())
    }
    /// Transfer only the packed planes required by the selected CPU outputs.
    /// An error after exposure recording starts invalidates the installation.
    /// Reinstall the scene before submitting through any of its frame slots;
    /// already submitted frames may still be collected.
    pub fn submit_outputs(
        &mut self,
        s: &InstalledScene,
        r: &Renderer,
        time: impl Into<vork_time::Request>,
        outputs: ReadbackOutputs,
    ) -> Result<()> {
        let request=time.into();
        let identity=request.exact()?.scale(&vork_time::exact::decimal(&s.render.fps.to_string())?)
            .add(&vork_time::exact::Germ::point(vork_time::exact::decimal("1/2")?)).floor();
        let index=u32::try_from(identity).context("frame index exceeds u32 identity capacity")?;
        self.submit_indexed_outputs(s,r,index,request,outputs)
    }
    pub(crate) fn submit_indexed_outputs(&mut self, s: &InstalledScene, r: &Renderer, index: u32, request: vork_time::Request, outputs: ReadbackOutputs) -> Result<()> {
        s.transients.check_recording()?;
        let time=request.numeric()?;
        self.nominal_request=request;
        self.nominal_frame = index;
        if self.propagation_motion && (outputs.motion || outputs.float.motion) && !self.warned_propagation_motion {
            crate::warnings::dropped("propagation_motion", serde_json::json!({"channel":"motion","propagation":"curved"}), format!("warning: ignored image motion under authored curved propagation; motion output is zero, sampled shutter blur is retained"));
            self.warned_propagation_motion=true;
        }
        self.gpu_check(r)?;
        ensure!(!self.pending, "collect this frame slot before resubmitting");
        if let Some(done) = &self.completed {
            if done.nominal_frame == index && done.nominal_request == self.nominal_request && done.time == time && done.outputs.contains(outputs) && outputs.contains(done.outputs)
                && done.profile == self.profile && done.write_color == self.write_color
            {
                // This ring entry already contains validated output. No scene work,
                // GPU submission, mapping or credit wait is needed a second time.
                self.submission = Some(done.submission.clone());
                self.submitted_time = time;
                self.view.readback_outputs = outputs;
                self.pending = true;
                return Ok(());
            }
        }
        self.completed = None;
        self.submit_exposure(s, r, time, outputs)
    }
    fn controls_view(&self, s: &InstalledScene, r: &Renderer) -> Result<ViewState> {
        self.camera_view(s, r, &s.render)
    }
    fn camera_view(&self, s: &InstalledScene, r: &Renderer, render: &vork_scene::Render) -> Result<ViewState> {
        ViewState::new(
            s,
            r,
            render,
            if self.view.float_output {
                ReadbackPrecision::Float
            } else {
                ReadbackPrecision::Packed
            },
            &s.graph.views.entries[s.graph.views.camera],
            &self.queries,
            crate::view::SharedBindings {
                points: &self.point_publication.buffer,
                compute: self.compute_commands.as_ref(),
                objects: &self.objects,
                deforms: &self.deforms,
                materials: &self.materials,
                shadow_bind: &self.shadow_bind,
                output: Some(&self.view.output),
                rays: self.rays.as_ref(),
            },
            s.main_subset.as_ref(),
        )
    }
    fn can_batch_samples(&self, s: &InstalledScene) -> bool {
        self.batch_uploads && Self::batch_plan(s, &self.view)
    }
    fn batch_plan(s: &InstalledScene, view: &ViewState) -> bool {
        s.dense.is_none() && s.displacement.is_none() && !s.draws.iter().any(|d|d.computed) && !s.environment.animated()
            // Fixed-clock replay streams bounded chunks, including on Vulkan.
            && !s.compute.as_ref().is_some_and(|compute| compute.has_simulation())

            && !view.recording().steps.iter().any(|step| step.needs_interop() || match step.pass {
                // This external-resource producer remains in plans without captures.
                crate::graph::Pass::Images => s.graph.views.image_renders().next().is_some(),
                crate::graph::Pass::Denoise => view.post.denoiser.is_some(),
                _ => false,
            })
    }
    fn submit_exposure(
        &mut self,
        s: &InstalledScene,
        r: &Renderer,
        time: f32,
        outputs: ReadbackOutputs,
    ) -> Result<()> {
        self.gpu_check(r)?;
        ensure!(!self.pending, "collect this frame slot before resubmitting");
        ensure!(time.is_finite() && time >= 0., "time must be nonnegative and finite");
        if let Some(rays) = &mut self.rays { rays.begin_exposure(); }
        // Resolve once at the nominal exposure time, before selecting recorded
        // spatial work. Sample-time render evaluation must not change this plan.
        s.animation.sample_into(time, &mut self.animation_values);
        s.animation.apply_render(&mut self.animation_values, &s.render);
        let exposure_render = self.animation_values.render.as_ref().unwrap().clone();
        ensure!(exposure_render.motion_blur.as_ref().is_none_or(|m| m.shutter.is_finite() && m.shutter >= 0.),
            "animated shutter must be nonnegative and finite");
        self.exposure_shutter = exposure_render.motion_blur.clone();
        let (exposure_samples, visibility_samples) = crate::plan::sampling_counts(&exposure_render, true);
        if self.view.visibility_samples != visibility_samples {
            if self.shutter_view.is_none() {
                self.shutter_view = Some(self.camera_view(s, r, &exposure_render)?);
            }
            let other = self.shutter_view.as_mut().unwrap();
            std::mem::swap(&mut other.write_color, &mut self.view.write_color);
            std::mem::swap(&mut self.view, other);
            ensure!(self.view.visibility_samples == visibility_samples, "shutter recording has wrong visibility budget");
            // These secondary producers bind the selected view's lighting globals.
            if let (Some(volume), Some(rays)) = (&s.probes, self.rays.as_ref()) {
                self.probes = Some(volume.slot(s, &r.device, &self.view, &self.objects, rays,
                    self.materials.as_ref().map_or_else(||s.material_table.bind(), |m| &m.bind))?);
            }
            if let (Some(installed), Some(rays)) = (&s.rays, self.rays.as_mut()) {
                if let Some(plan) = &installed.emitter_weights {
                    let hits = rays.hits_bind(&r.device);
                    if let Some(slot) = &mut rays.emitter_weights {
                        slot.prepare(plan, s, &r.device, &self.view.lighting, &self.objects,
                            self.materials.as_ref().map_or_else(||s.material_table.records(), |m| &m.records), hits);
                    }
                }
            }
        }
        self.simulation_checkpoint = None;
        ensure!(self.probe_generation==s.reservations.probe_pages,"frame slot has stale probe support; include every slot in redo_frame");
        ensure!(self.dielectric_spill_reservation==s.reservations.dielectric_spill,"frame slot has stale dielectric storage; include every slot in redo_frame");
        self.submitted_time = time;
        for view in self.views_mut() {
            view.refresh_variants(s, r)?;
        }
        // The encoders below choose each drawn pipeline by this same pending state.
        let specialized = !self.view.variants_pending();
        self.path_kernel = s.path.as_ref().map(|path| if path.draws_engine(specialized) { "prebuilt engine" } else { "scene kernel" });
        ensure!(
            !outputs.float.any() || self.view.float_output,
            "float channels require a Float precision frame slot"
        );
        ensure!(
            s.readback.contains(outputs),
            "readback {outputs:?} is outside the install's readback {:?}",
            s.readback
        );
        if let Some(validation)=&self.environment_diagnostics {validation.clear(&r.queue);}
        if self.shader_error_readback.is_some() {
            r.queue.write_buffer(&self.materials.as_ref().expect("installed material diagnostics").shader_errors, 0, bytemuck::cast_slice(&crate::materials::SHADER_DIAGNOSTICS_INIT));
        }
        self.ring.credit.wait();
        // These caches are useful even when this exposure reuses their values.
        drop(s.transients.retain());
        if let Some(probes) = &s.probes {drop(probes.storage());}
        if let Some(fibre) = &s.fibre {drop(fibre.storage());}
        // One submitter keeps shared frame/sample resources in queue order.
        let mut image_changes = s.transients.submit.lock().unwrap();
        s.transients.record(|| {
        // Missing pages make this layout provisional. Keep its bitmap until
        // a drained retry invalidates captures: later slots can reuse captures made
        // before collection discovers the miss, without issuing new shadow reads.
        // Views share this exposure-wide transient; queue order keeps a later
        // slot's reset after this exposure's final diagnostic copy.
        let mut counters=[0u32;vork_slang::registry::NONFINITE_WORDS];
        counters[0]=u32::from(self.profile);
        if self.profile {r.queue.write_buffer(&self.view.post.nonfinite,0,bytemuck::cast_slice(&counters));}
        if self.profile && self.nonfinite_readback.is_none() {
            self.nonfinite_readback=Some(crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {label:Some("nonfinite readback"),size:((vork_slang::registry::NONFINITE_WORDS-vork_slang::registry::NONFINITE_SHADE)*4) as u64,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false}));
        }
        self.view.readback_outputs = outputs;
        self.view.color_enabled = self.view.write_color || outputs.color || outputs.color16 || outputs.float.color || (outputs.aovs && !self.view.aovs.is_empty());
        self.capture_images = self.view.color_enabled || s.capture_controls;
        let read_pixels = outputs.any();
        // Keep tile history and all shutter samples in queue submission order.
        // Upload buffers and caster bundles remain private to this slot.
        let retained_shadows = s.shadows();
        let mut shadow_changes = retained_shadows
            .changes
            .lock()
            .expect("shadow submission poisoned");
        self.exposure_cut=if s.animation.has_camera_tracks() {
            let nominal=self.nominal_request.exact()?;
            let previous=self.nominal_request.previous(s.render.fps)?.exact()?;
            s.animation.clamp_camera_request(&nominal,&previous,s.camera)?!=previous
        }else{false};
        // A camera that did not exist at the previous frame has no earlier pose: its first
        // frame is a cut, like a camera switch (rule 1).
        if s.animation.has_clocks() && !self.exposure_cut {
            let previous=s.animation.clocks(&self.nominal_request.previous(s.render.fps)?.exact()?)?;
            let camera=s.animation.camera_at_request(&self.nominal_request,s.camera)?;
            self.exposure_cut=std::iter::once(camera).chain(s.graph.views.image_renders().map(|image|image.camera)).any(|camera|previous[camera].is_none());
        }
        let frame_motion_blur = self.exposure_shutter.clone();
        let shutter = frame_motion_blur.as_ref();
        // Geometric controls describe the nominal frame, not an exposure integral.
        // Static decoupled colour collects spatial visibility samples in one dispatch;
        // shutter steps retain scene-time strata in the exposure loop.
        let count = if self.view.color_enabled {
            exposure_samples as usize
        } else {
            1
        };
        let span = if self.exposure_cut {
            0.
        } else {
            shutter.map_or(0., |v| v.shutter / s.render.fps.as_f32())
        };
        // Temporal colour is jittered (and may be upsampled): controls come
        // from one unjittered native-resolution pass.
        let nominal_controls = crate::plan::nominal_control_pass(count as u32, false, shutter.is_some_and(|m| m.shutter > 0.))
            && (outputs.depth_ids
                || outputs.normals
                || outputs.motion
                || outputs.float.depth
                || outputs.float.normals
                || outputs.float.motion);
        self.view.nominal_sample = false;
        self.view.sample_count = count;
        self.denoise_ms = denoise::Timing::default();
        self.sample_spans.clear();
        let mut cpu = 0.;
        let mut prep = 0.;
        let mut finish = 0.;
        // Pure GPU plans validate recording chunks concurrently and submit
        // their ordered buffers together (or stream ready CPU prefixes).
        // Other plans start GPU work while later samples are prepared.
        self.exposure_upload.clear();
        // Native Metal validation can split each pass into command buffers.
        // Bound deferred exposures below the queue's unsubmitted-buffer budget;
        // larger exposures use the existing submission-ordered streaming path.
        let samples = count + usize::from(nominal_controls);
        ensure!(samples as u64 <= s.plan.samples,
            "frame records {samples} sample passes but install reserved upload and timestamp regions for {}", s.plan.samples);
        let group_ends = sample_group_ends(&self.sample_gpu_ms, samples);
        self.group_ends.clone_from(&group_ends);
        // CPU validation remains parallel inside a measured GPU submission.
        let validation_group_size = if r.adapter_info.backend == wgpu::Backend::Metal { samples }
            else { samples.div_ceil(constants::COMMAND_FINISH_THREADS).max(1) };
        // Metal's counter-buffer limit is device-wide. Use only the slot's
        // existing set: independent groups stream rather than allocating sets.
        let batch = count > 1 && self.can_batch_samples(s)
            && (r.adapter_info.backend != wgpu::Backend::Metal
                || (count <= constants::METAL_BATCH_SAMPLES
                    && self.view.visibility_samples <= vork_slang::registry::MERGE_BATCH
                    && group_ends[..samples - 1].iter().all(|end| !end)
                    && !self.view.recording().steps.iter().any(crate::graph::Step::separates)));
        // Diagnostic cuts must see already-published uniforms, like OIDN interop.
        #[cfg(test)]
        let batch=batch && r.test_gpu_segment_times.is_none();
        let mut finish_workers = batch.then(|| r.command_finish
            .get_or_init(crate::command_finish::CommandFinish::new).batch(crate::command_finish::batch_window(r)));
        let mut recording = None;
        // Vulkan groups own per-sample counters until the exposure resolves.
        // Metal's single group resolves each sample before reusing its set.
        let deferred_queries = r.adapter_info.backend != wgpu::Backend::Metal;
        if batch && deferred_queries {
            while self.sample_queries.len() < samples {
                self.sample_queries.push(r.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("GPU exposure sample"),
                    ty: wgpu::QueryType::Timestamp,
                    count: timing::count(),
                }));
            }
        }
        let offset = shutter.map_or(0., |s| s.position.offset());
        let exact = !s.footage.is_empty() || s.compute.is_some() || s.animation.has_camera_tracks() || s.environment.has_media() || s.animation.has_discrete_tracks() || s.animation.has_clocks();
        let nominal=exact.then(||self.nominal_request.exact()).transpose()?;
        let exact_span=exact.then(|| -> Result<_> {Ok(vork_time::exact::ieee(if self.exposure_cut {0.}else{shutter.map_or(0.,|s|s.shutter)})?
            / vork_time::exact::decimal(&s.render.fps.to_string())?)}).transpose()?;
        let request_at=|n:u64,d:u64| -> Result<vork_time::Request> {
            let Some(nominal)=&nominal else {
                return Ok(vork_time::Request::Numeric((time+(n as f32/d as f32-0.5+offset)*span).max(0.)));
            };
            let position=vork_time::exact::Rational::new(n.into(),d.into())-vork_time::exact::decimal("1/2")?+vork_time::exact::ieee(offset)?;
            let requested=nominal.add(&vork_time::exact::Germ::point(position*exact_span.as_ref().unwrap()));
            let zero=vork_time::exact::decimal("0")?;
            let requested=if requested.compare_point(&zero).is_lt() {vork_time::exact::Germ::point(zero)}else{requested};
            Ok(vork_time::Request::Exact(s.animation.clamp_camera_request(nominal,&requested,s.camera)?))
        };
        let sample_requests:Vec<_>=(0..count).map(|i|request_at(2*crate::sampling::shutter_index(i,count) as u64+1,2*count as u64))
            .chain(nominal_controls.then(||Ok(self.nominal_request.clone()))).collect::<Result<_>>()?;
        let sample_times:Vec<_>=sample_requests.iter().map(|request|request.numeric()).collect::<Result<_>>()?;
        if let Some(compute)=&s.compute {
            let mut cursor=compute.cursor.lock().expect("compute cursor poisoned");
            if self.compute.is_some() {
                compute.plan_placement_requests(&mut cursor,&sample_requests,self.exposure_cut)?;
            } else {
                compute.plan_requests(&mut cursor,sample_requests.iter().cloned())?;
            }
        }
        if self.compute.is_some() {
            let compute=s.compute.as_ref().expect("placement compute");
            let mut cursor=compute.cursor.lock().expect("compute cursor poisoned");
            compute.prepare_placement_history(r,&mut cursor,&sample_requests,self.exposure_cut,&mut self.simulation_checkpoint)?;
        }
        self.geometry_chain.prepare_history(r,s,&sample_requests,self.exposure_cut,&mut self.simulation_checkpoint)?;
        vork_slang::progress::sample_pass(0, count);
        for i in 0..count {
            if let Some(compute)=&s.compute {
                compute.consume_request(&mut compute.cursor.lock().expect("compute cursor poisoned"), sample_requests[i].clone())?;
            }
            self.view.sample_index = i;
            if batch && deferred_queries {
                self.queries = self.sample_queries[i].clone();
            }
            let sample_time = sample_times[i];
            self.sample_request=sample_requests[i].clone();
            let stratum=crate::sampling::shutter_index(i,count) as u64;
            // Record the scheduler's actual domain, including animated shutter
            // values and independently clamped endpoints at scene zero/cuts.
            self.view.composite_shutter = [
                request_at(0,1)?.numeric()?,
                request_at(1,1)?.numeric()?,
                request_at(stratum,count as u64)?.numeric()?,
                request_at(stratum+1,count as u64)?.numeric()?,
            ];
            self.view.composite_steps = if span > 0. { count as u32 } else { 1 };
            let encoder = self.encode_sample(
                s,
                r,
                sample_time,
                time,
                read_pixels && !nominal_controls && i + 1 == count,
                &mut shadow_changes,
                &mut image_changes,
                batch,
                i + 1 == count && !nominal_controls,
                recording.take(),
            )?;
            recording = self.finish_sample(r, encoder, finish_workers.as_mut(), group_ends[self.view.sample_index], validation_group_size);
            vork_slang::progress::sample_pass(i + 1, count);
            cpu += self.cpu_ms;
            prep += self.prep_ms;
            finish += self.finish_ms;
            // Streaming plans start GPU work while the next sample is prepared.
            // Batched command buffers preserve the same sample order at submission.
            self.pending = false;
        }
        // Upsampled colour has no native targets; swap in the native controls view,
        // whose packed planes share this view's transient and output ring readback.
        let native = nominal_controls && self.view.internal != [self.view.width, self.view.height];
        if native {
            if self.controls.is_none() {
                self.controls = Some(self.controls_view(s, r)?);
            }
            let controls = self.controls.as_mut().unwrap();
            std::mem::swap(&mut self.view, controls);
            self.view.readback_outputs = outputs;
            self.view.sample_count = count;
        }
        if nominal_controls {
            if let Some(compute)=&s.compute {
                compute.consume_request(&mut compute.cursor.lock().expect("compute cursor poisoned"),self.nominal_request.clone())?;
            }
            let color_enabled = self.view.color_enabled;
            self.view.nominal_sample = true;
            self.view.sample_index = count;
            if batch && deferred_queries {
                self.queries = self.sample_queries[count].clone();
            }
            self.view.color_enabled = false;
            self.sample_request=self.nominal_request.clone();
            let encoder = self.encode_sample(s, r, time, time, read_pixels, &mut shadow_changes, &mut image_changes, batch, true, recording.take())?;
            recording = self.finish_sample(r, encoder, finish_workers.as_mut(), group_ends[self.view.sample_index], validation_group_size);
            cpu += self.cpu_ms;
            prep += self.prep_ms;
            finish += self.finish_ms;
            self.view.nominal_sample = false;
            self.view.color_enabled = color_enabled;
        }
        if native {
            let controls = self.controls.as_mut().unwrap();
            std::mem::swap(&mut self.view, controls);
        }
        if batch {
            let start = Instant::now();
            debug_assert!(recording.is_none());
            for (buffer, bytes) in &self.exposure_upload {
                r.queue.write_buffer(buffer, 0, bytes);
            }
            let (submission, finish_ms) = crate::command_finish::Segments::publish(
                r, finish_workers.take().unwrap(), &self.sample_upload,
                &[], self.stream_samples);
            self.submission = Some(submission);
            finish += finish_ms;
            cpu += start.elapsed().as_secs_f64() * 1000.;
        }
        if let Some(checks) = &self.publication_checks {
            let mut buffers = std::mem::take(&mut self.solver_receipts);
            if let Some(buffer)=&self.shader_error_readback {buffers.push((crate::graph::PublicationDiagnostic::Shader,buffer.clone()));}
            if let Some(env)=&self.environment_diagnostics {buffers.push((crate::graph::PublicationDiagnostic::Code("environment"),env.readback.clone()));}
            if let Some(compute)=&self.compute {buffers.push((crate::graph::PublicationDiagnostic::Code("placement"),compute.error_readback.clone()));}
            checks.lock().unwrap().push((self.submission.clone().context("publication has no fence")?,buffers));
        }
        self.flight_ownership.submit();
        self.pending = true;
        self.read_pixels = read_pixels;
        self.cpu_ms = cpu;
        self.prep_ms = prep;
        self.finish_ms = finish;
        self.gpu_check(r)?;
        Ok(())
        })
    }
    fn encode_sample(
        &mut self,
        s: &InstalledScene,
        r: &Renderer,
        time: f32,
        frame_time: f32,
        read_pixels: bool,
        shadow_changes: &mut values::LightingTracker,
        image_changes: &mut values::ChangeTracker,
        batch: bool,
        last_sample: bool,
        recording: Option<wgpu::CommandEncoder>,
    ) -> Result<wgpu::CommandEncoder> {
        if let Some([start,end])=s.medium_certified_times {
            ensure!(time.is_finite() && time>=start && time<=end,
                "sampled medium time {time} is outside the certified shader interval [{start}, {end}]; extend the scene duration/shutter and reinstall");
        }
        let finish_before = self.denoise_ms.finish_ms;
        self.sample_segments.begin(!batch);
        self.gpu_check(r)?;
        ensure!(!self.pending, "collect this frame slot before resubmitting");
        ensure!(
            time.is_finite() && time >= 0.,
            "time must be nonnegative and finite"
        );
        self.read_pixels = read_pixels;
        let start = Instant::now();
        self.view.queries = self.queries.clone();
        // Rule 7: the environment's movies show its node's local time while its scene-owned tracks
        // and `time` stay in scene seconds. Rule 1: an absent environment light is not evaluated
        // (its visibility already darkens it).
        let media = if s.environment.animated() {s.environment.media_request(&s.animation,&self.sample_request)?} else {None};
        if let Some(media)=media {
            if shadow_changes.changed(values::LightingKey::EnvironmentInputs, &s.environment.sample_key(&self.sample_request,&media)?) {
                if let Err(error) = s.environment.update(r,&self.sample_request,&media,shadow_changes) {
                    shadow_changes.changed(values::LightingKey::EnvironmentInputs, &[]);
                    shadow_changes.changed(values::LightingKey::EnvironmentRadiance, &[]);
                    return Err(error);
                }
            }
        }
        if s.plan.effects.image_feedback && self.capture_images && self.view.sample_index == 0 {
            // Implicit ray hits, shadow opacity and displacement can sample images
            // outside the direct capture DAG. Give those inputs this frame's empty
            // initial publication rather than any speculative preceding frame.
            let mut initial = r.device.create_command_encoder(&Default::default());
            for image in s.graph.views.image_renders() { image.clear(&mut initial); }
            crate::command_finish::Segments::submit(&r.queue, initial);

        }
        let (mut upload_bytes, mut uploads, _view, uniform) =
            self.prepare_values(s, r, self.sample_request.clone(), frame_time, shadow_changes)?;
        let mut media_updates=Vec::new();
        for image in s.footage.values().filter(|image|image.per_sample) {
            let request=self.sample_request.exact()?;
            // Canvas absence publishes an empty picture: independent surfaces may still read it.
            let clock=if image.canvas.is_some() {
                image.canvas_instant(s,&self.animation_values,&request)?
            } else {
                let base=match s.animation.footage_clock(&self.animation_values,&image.name) {
                    crate::animation::FootageClock::Absent=>continue,
                    crate::animation::FootageClock::Node(clock)=>clock,
                    crate::animation::FootageClock::Scene=>image.time.as_ref().map(|t|vork_time::exact::Germ::point(t.exact())).unwrap_or_else(||request.clone()),
                };
                Some(s.animation.image_time_exact(&self.animation_values,&request,&image.name,base)?)
            };
            let key=image.sample_key(clock.as_ref(),s.render.fps)?;
            if image_changes.changed(values::ChangeKey::ImageClock(image.name.clone()),&key) {
                image_changes.image_revision+=1;
                media_updates.push((image,clock));
            }
        }
        // The installed volume follows shared submission-ordered lighting history.
        let probe_geometry=s.probes.as_ref().is_some_and(|volume| {
            let mut key=crate::probes::geometry_key(&self.object_values,volume.geometry_timed.then_some(time));
            if volume.grid.visibility_policy[0] == 0 {
                key.extend_from_slice(bytemuck::cast_slice(&[uniform.temporal[2].to_bits(),uniform.temporal[0].to_bits()]));
            }
            // Angular subsets differ even when replay clamps adjacent shutter
            // times to the same pose. Moments and lighting share this phase.
            if volume.grid.visibility_policy[0]>1 {key.extend_from_slice(&uniform.temporal[0].to_bits().to_le_bytes());}
            crate::probes::program_key(&mut key,&uniform,volume.dependencies,image_changes.image_revision+u64::from(s.plan.effects.image_feedback && self.view.sample_index==0));
            shadow_changes.changed(values::LightingKey::ProbeGeometry,&key)
        });
        let mut probes = s.probes.as_ref().is_some_and(|volume| {
            let mut key = crate::probes::key(&self.object_values, &self.evaluated_lights, &uniform, volume.timed.then_some(time));
            if volume.grid.visibility_policy[0] == 0 {
                key.extend_from_slice(bytemuck::cast_slice(&[uniform.temporal[2].to_bits(),uniform.temporal[0].to_bits()]));
            }
            if volume.grid.visibility_policy[0]>1 {key.extend_from_slice(&uniform.temporal[0].to_bits().to_le_bytes());}
            // Angular radiance can change without changing the mean. Consume
            // the retained producer revision, rather than its animation clock.
            key.extend_from_slice(&shadow_changes.revision(values::LightingKey::EnvironmentRadiance).to_le_bytes());
            crate::probes::program_key(&mut key, &uniform, volume.dependencies, image_changes.image_revision + u64::from(s.plan.effects.image_feedback && self.view.sample_index == 0));
            shadow_changes.changed(values::LightingKey::Probes, &key)
        });
        // CaptureView owns the complete dependency comparison. Equal numeric
        // poses can still select different exact footage/environment frames.
        let mut capture = self.capture_images && !self.image_views.is_empty();
        let seed = capture && s.plan.effects.image_feedback;
        if seed { probes = s.probes.is_some(); }
        let mut seed_views = std::mem::take(&mut self.seed_views);
        let mut image_views = std::mem::take(&mut self.image_views);
        let mut image_dirty = if capture { vec![false; image_views.len()] } else { Vec::new() };
        if capture {
            let lighting_revisions = [
                shadow_changes.revision(values::LightingKey::Probes),
                shadow_changes.revision(values::LightingKey::Fibre),
            ];
            let prepared = image_views.iter_mut().zip(s.graph.views.image_renders())
                .enumerate().try_for_each(|(index, (view, image))| -> Result<()> {
                    let id = s.graph.views.images.start + index;
                    let dependency_changed = s.graph.views.entries[id].reads.iter()
                        .any(|read| image_dirty[read.0 - s.graph.views.images.start]);
                    image_dirty[index] = view.prepare(s, self, image, uniform,
                        image_changes, id, dependency_changed || seed, lighting_revisions)?;
                    Ok(())
                });
            if let Err(error) = prepared {
                self.image_views = image_views;
                self.seed_views = seed_views;
                return Err(error);
            }
            capture = image_dirty.iter().any(|dirty| *dirty);
            if seed {
                let mut seed_changes = values::ChangeTracker::default();
                for (index, (view, image)) in seed_views.iter_mut().zip(s.graph.views.image_renders()).enumerate() {
                    if let Err(error) = view.prepare(s, self, image, uniform, &mut seed_changes, index, true, [0,0]) {
                        self.image_views = image_views;
                        self.seed_views = seed_views;
                        return Err(error);
                    }
                }
            }
            let base = upload_bytes.len() as u64;
            upload_bytes.append(&mut self.changes.bytes);
            uploads.extend(
                self.changes
                    .uploads
                    .drain(..)
                    .map(|(dst, offset, source, size)| (dst, offset, base + source, size)),
            );
        }
        self.prep_ms = start.elapsed().as_secs_f64() * 1000.;
        let mut encoder = recording.unwrap_or_else(|| self.sample_segments.encoder(r, Some("installed sample group")));
        let mut compute_cursor = s
            .compute
            .as_ref()
            .map(|c| c.cursor.lock().expect("compute cursor poisoned"));
        let sample = self.view.sample_index;
        if sample == 0 || self.group_ends[sample - 1] {
            timing::boundary(r, &mut encoder, &self.queries, timing::sample_query());
        }
        for (image,clock) in media_updates {
            if let Err(error)=image.encode(r,&mut encoder,clock.as_ref(),s.render.fps,self.shader_error_readback.as_ref().and(self.materials.as_ref()).map(|m|&m.shader_errors)) {
                image_changes.changed(values::ChangeKey::ImageClock(image.name.clone()),&[]);
                return Err(error);
            }
        }
        self.copy_values(r, &mut encoder, &upload_bytes, &uploads, batch);
        // Host uploads carry sampled camera/tint fields. Publish GPU-produced
        // environment moments and scale afterward, before every consumer.
        for view in std::iter::once(&*self.view).chain(image_views.iter().chain(&seed_views).map(|v|&*v.view)) {
            s.environment.publish_metadata(&mut encoder,&view.uniform);
            s.environment.publish_metadata(&mut encoder,&view.lighting.uniform);
        }
        if let Some(validation)=&self.environment_diagnostics {validation.encode(&mut encoder);}

        if sample == 0 {
            for view in std::iter::once(&*self.view).chain(image_views.iter().chain(&seed_views).map(|v|&*v.view)) {
                let bitmap = &view.post.invalid_bitmap;
                encoder.clear_buffer(&view.post.pixels, bitmap.start, Some(bitmap.end - bitmap.start));
            }
        }
        if sample==0 { if let Some(compute)=&self.compute {encoder.clear_buffer(&compute.errors,0,None);} }
        if !self.reuse_values {
        if let (Some(compute), Some(cursor)) = (&s.compute, compute_cursor.as_mut()) {
            if self.compute.is_some() {
                compute.consume_request(cursor,self.previous_request.clone())?;
                if !self.exposure_cut && !compute.restore_placement_history(&mut encoder,cursor,&self.previous_request)? {
                    if let Some(checkpoint) = compute.advance_with(&mut encoder, cursor, &self.previous_request, |encoder| {
                        self.sample_segments.cut(r, encoder);
                    })? {
                        if self.simulation_checkpoint.is_none() { self.simulation_checkpoint = Some(checkpoint); }
                    }
                    compute.snapshot_placement_previous(&mut encoder);
                }
            }
            if let Some(checkpoint) = compute.advance_with(&mut encoder, cursor, &self.sample_request, |encoder| {
                self.sample_segments.cut(r, encoder);
            })? {
                // At most one bounded host snapshot per exposure, including shutter strata.
                if self.simulation_checkpoint.is_none() { self.simulation_checkpoint = Some(checkpoint); }
            }
            if self.compute.is_some() {
                compute.retain_placement_history(&mut encoder,cursor,&self.sample_request)?;
                if self.exposure_cut {compute.snapshot_placement_previous(&mut encoder);}
            }
        }
        if let Some(cursor) = compute_cursor.as_ref() {
            self.changes.placement_revision = cursor.placement_revision;
            if let Some(compute) = s.compute.as_ref().filter(|c|c.has_material_attributes()) {
                compute.publish_material_attributes(&mut encoder, cursor, &mut self.changes,
                    &self.materials.as_ref().expect("computed material rows").records);
            }
        }
        if let Some(compute) = &self.compute {
            compute.encode(&mut encoder);
        }

        if let Some(dense)=&s.dense {
            dense.publish(r,&mut encoder,&mut self.sample_segments,&self.materials.as_ref().expect("dense frame table").records)?;
        }
        }
        if let Some(compute)=&s.compute {compute.publish_queries(r,&mut encoder,&self.materials.as_ref().expect("compute query diagnostics").shader_errors);}
        for field in &s.implicit_fields {
            for (message,buffer,offset) in &field.statuses {
                let receipt=r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("implicit solver status"),size:4,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
                encoder.copy_buffer_to_buffer(buffer,*offset,&receipt,0,4);
                self.solver_receipts.push((crate::graph::PublicationDiagnostic::SolverStatus(message.clone()),receipt));
            }
            let revision=compute_cursor.as_ref().and_then(|c|c.publications.get(&field.source)).copied().unwrap_or(0);
            let published=field.acceleration.publish(&mut encoder,image_changes,revision);
            if published || sample==0 {
                field.acceleration.publish_diagnostics(r,&mut encoder,&self.materials.as_ref().expect("installed material diagnostics").shader_errors);
            }
        }
        self.point_publication.encode(&mut encoder);
        self.computed_bounds.encode(&mut encoder);
        self.changes.bytes = upload_bytes;
        self.changes.uploads = uploads;

        // GPU material rows can publish a different generation at the same
        // clock. Shared lighting captures consume that publication too.
        let material_attributes_changed = self.changes.dirty(values::ChangeKey::MaterialAttributes);
        let probe_geometry = probe_geometry || material_attributes_changed;
        let bits = crate::graph::RunBits {
            color: self.view.color_enabled,
            final_sample: self.view.sample_index + 1 == self.view.sample_count,
            probes: probes || (material_attributes_changed && s.probes.is_some()),
            fibre:seed || shadow_changes.dirty(values::LightingKey::Fibre)
                || (material_attributes_changed && s.fibre.is_some()),
            images: capture,
            pixels: read_pixels,
        };
        use crate::graph::Pass;
        let recording = self.view.recording();
        let seed_views_len = seed_views.len();
        let mut spans = 0;
        for step in &recording.steps {
            if !bits.runs(step.run) {
                continue;
            }
            // Only an installed deformation stage writes this counter pair.
            // Computed-only geometry has no deformation pass; resolving its
            // unwritten timestamps would make Vulkan wait for unavailable queries.
            // Displacement is evaluated before recording this graph.
            if step.pass == Pass::Deform && (self.reuse_values || s.displacement.is_some()
                || self.geometry_chain.skin().is_empty()) {
                continue;
            }
            // Every render resolves its pass spans: the stage breakdown in each
            // render report needs them. Non-finite counting remains diagnostics.
            spans |= step.timestamp_mask(&self.view);
            if step.pass==Pass::Compositor && s.footage.values().any(|i|i.final_compositor) {
                let mut nominal=s.animation.samples();s.animation.sample_request_into(frame_time,&self.nominal_request,&mut nominal)?;
                for image in s.footage.values().filter(|i|i.final_compositor) {
                    let request=self.nominal_request.exact()?;
                    let clock=if image.canvas.is_some() {
                        image.canvas_instant(s,&nominal,&request)?
                    } else {
                        let base=image.time.as_ref().map(|t|vork_time::exact::Germ::point(t.exact())).unwrap_or_else(||request.clone());
                        Some(s.animation.image_time_exact(&nominal,&request,&image.name,base)?)
                    };
                    let key=image.sample_key(clock.as_ref(),s.render.fps)?;
                    if image_changes.changed(values::ChangeKey::ImageClock(image.name.clone()),&key) {
                        image_changes.image_revision+=1;
                        if let Err(error)=image.encode(r,&mut encoder,clock.as_ref(),s.render.fps,self.shader_error_readback.as_ref().and(self.materials.as_ref()).map(|m|&m.shader_errors)) {
                            image_changes.changed(values::ChangeKey::ImageClock(image.name.clone()),&[]);return Err(error);
                        }
                    }
                }
            }
            match step.pass {
                Pass::ImageInputs => {
                    r.timestamp_marker.get_or_init(|| timing::Marker::new(r))
                        .encode(&mut encoder, &self.queries, timing::query("image inputs"));
                    if let Some(volume) = &s.probes { volume.clear(&mut encoder); }
                    if let Some(fibre) = &s.fibre { fibre.clear(&mut encoder); }
                    for (i, (view, image)) in seed_views.iter_mut().zip(s.graph.views.image_renders()).enumerate() {
                        view.encode(&mut encoder, r, &s.environment, self.seed_queries.as_ref().expect("seed capture timestamps installed"), image, i + 1 == seed_views_len, &mut self.denoise_ms, &mut self.sample_segments, &mut self.changes)?;
                    }
                    r.timestamp_marker.get().expect("seed timing marker initialized")
                        .encode(&mut encoder, &self.queries, timing::query("image inputs") + 1);
                }
                Pass::Deform => self.geometry_chain.encode(&mut encoder, s, &self.queries),
                Pass::Rays => {
                    timing::boundary(r, &mut encoder, &self.queries, timing::query("acceleration structures"));
                    crate::rt::encode(&mut encoder, s, self);
                    timing::boundary(r, &mut encoder, &self.queries, timing::query("acceleration structures") + 1);
                },
                Pass::Fibre => {if let (Some(p),Some(slot))=(&s.fibre,&self.fibre) {slot.encode(&p.storage(),&mut encoder,&self.queries);}},
                Pass::Probes => {
                    if let (Some(volume), Some(slot)) = (&s.probes, &self.probes) {
                        volume.encode(&mut encoder, slot, &self.queries, probe_geometry, self.view.sample_count as u32);
                    }
                }
                Pass::Images => {
                    timing::boundary(r, &mut encoder, &self.queries, timing::query("images"));
                    let last = image_dirty.iter().rposition(|dirty| *dirty).expect("capture has dirty images");
                    for (i, (view, image)) in image_views
                        .iter_mut()
                        .zip(s.graph.views.image_renders())
                        .enumerate()
                    {
                        if image_dirty[i] {
                            view.encode(&mut encoder, r, &s.environment, &self.queries, image, i == last, &mut self.denoise_ms, &mut self.sample_segments, &mut self.changes)?;
                        }
                    }
                }
                Pass::Denoise => {
                    if let Some(stage) = &self.view.post.reconstruction {
                        stage.encode(&mut encoder,Some(&self.queries));
                        spans |= timing::mask("reconstruction");
                    }
                    if let Some(stage) = &self.view.post.denoiser {
                        stage.run(r, &mut encoder, self.view.sample_count, 1., &mut self.denoise_ms, &mut self.sample_segments)?;
                    }
                }
                Pass::Readback => self.ring.encode(&mut encoder, self.view.readback_outputs),
                _ => crate::graph::encode_view(&mut encoder, &self.view, step, r, &mut self.sample_segments)?,
            }
        }
        self.image_views = image_views;
        self.seed_views = seed_views;
        // Surface recording failures before native encoder interop needs a valid HAL encoder.
        self.gpu_check(r)?;
        if self.view.color_enabled
            && !self.view.nominal_sample
            && self.view.sample_index + 1 == self.view.sample_count
        {
            self.ring.encode_video(&mut encoder, &self.view.output);
            let pixels = u64::from(self.view.width) * u64::from(self.view.height);
            let display = if self.view.write_color {
                Some(crate::display_stats::Display::Texture(&self.view.output))
            } else if self.view.readback_outputs.color || self.view.readback_outputs.color16 {
                Some(crate::display_stats::Display::Packed(&self.view.post.pixels, pixels))
            } else {
                None // float colour only: no display image is made
            };
            if let Some(display) = display {
                // Like pass descriptors, the endpoints are always written and resolved.
                self.exposure_report.encode(r, &mut encoder, display, Some(&self.queries));
                spans |= timing::mask("exposure");
            }
        }
        if self.group_ends[sample] {
            timing::boundary(r, &mut encoder, &self.queries, timing::sample_query() + 1);
        }
        let deferred_queries = batch && r.adapter_info.backend != wgpu::Backend::Metal;
        if !deferred_queries {
            self.resolve_timestamps(&mut encoder, &self.queries, self.view.sample_index, spans);
        }
        if last_sample {if let Some(validation)=&self.environment_diagnostics {validation.copy(&mut encoder);}}
        // Shader diagnostics accumulate over the exposure; collection needs one final copy.
        if let Some(readback) = self.shader_error_readback.as_ref().filter(|_| last_sample) {
            encoder.copy_buffer_to_buffer(&self.materials.as_ref().expect("installed material diagnostics").shader_errors, 0, readback, 0, crate::materials::SHADER_DIAGNOSTICS_BYTES);
            let mut offset = crate::materials::SHADER_DIAGNOSTICS_BYTES;
            for view in self.views() {
                let bitmap = &view.post.invalid_bitmap;
                let bytes = bitmap.end - bitmap.start;
                encoder.copy_buffer_to_buffer(&view.post.pixels, bitmap.start, readback, offset, bytes);
                offset += bytes;
            }
        }
        self.sample_spans.push(spans);
        if self.profile && last_sample {
            encoder.copy_buffer_to_buffer(&self.view.post.nonfinite,(vork_slang::registry::NONFINITE_SHADE*4) as u64,self.nonfinite_readback.as_ref().unwrap(),0,((vork_slang::registry::NONFINITE_WORDS-vork_slang::registry::NONFINITE_SHADE)*4) as u64);
            encoder.clear_buffer(&self.view.post.nonfinite,0,None);
        }
        if last_sample {
            // Distinct sample query sets retain counters until all sample work is
            // recorded. Resolve at the exposure boundary, avoiding query waits
            // and readback copies between samples.
            if deferred_queries {
                for (sample, &mask) in self.sample_spans.iter().enumerate() {
                    self.resolve_timestamps(&mut encoder, &self.sample_queries[sample], sample, mask);
                }
            }
            encoder.copy_buffer_to_buffer(&self.query_resolve, 0, &self.query_readback, 0,
                timing::resolve_stride() * self.sample_spans.len() as u64);
            // Include every receiver/capture in this exposure, using the same
            // final submission and lifetime fence as its other readbacks.
            if let Some((source,feedback)) = &self.probe_feedback {
                encoder.copy_buffer_to_buffer(&source.source,source.status_offset,feedback,0,feedback.size());
            }

        }
        // The exposure owns one diagnostic result. Copy after its final colour
        // sample; nominal controls do not write transport diagnostics. Copying
        // after every accumulation sample would add needless storage barriers.
        if bits.final_sample {
            if let Some(compute)=&self.compute {encoder.copy_buffer_to_buffer(&compute.errors,0,&compute.error_readback,0,8);}
            if let Some(destination) = &self.slot_overflow {
                for (index, view) in self.views().enumerate() {
                    if let Some(view) = &view.decoupled {
                        let offset = 8 * index as u64;
                        encoder.copy_buffer_to_buffer(&view.overflow, 4, destination, offset, 4);
                        encoder.copy_buffer_to_buffer(&view.fragments.buffer, 4, destination, offset + 4, 4);
                    }
                }
            }
        }
        self.finish_ms = self.denoise_ms.finish_ms - finish_before;
        if capture { image_changes.image_revision += 1; }
        self.cpu_ms = start.elapsed().as_secs_f64() * 1000.;
        Ok(encoder)
    }
    fn finish_sample(
        &mut self,
        r: &Renderer,
        encoder: wgpu::CommandEncoder,
        batch: Option<&mut crate::command_finish::Batch<'_>>,
        submission_end: bool,
        validation_group_size: usize,
    ) -> Option<wgpu::CommandEncoder> {
        let start = Instant::now();
        let end = self.view.sample_index + 1;
        let streaming = batch.is_none();
        let mut encoder = encoder;
        if submission_end {
            if let Some(done) = vork_slang::progress::sample_counter() {
                encoder.on_submitted_work_done(move || { done.fetch_max(end, std::sync::atomic::Ordering::Relaxed); });
            }
        }
        let (recording, submission, finish_ms) = self.sample_segments.finish_sample(
            r, encoder, batch, submission_end,
            end != self.group_ends.len() && end % validation_group_size != 0);
        if streaming {
            self.submission = submission;
            self.finish_ms += finish_ms;
            self.cpu_ms += start.elapsed().as_secs_f64() * 1000.;
        }
        recording
    }

    fn resolve_timestamps(&self, encoder: &mut wgpu::CommandEncoder, queries: &wgpu::QuerySet, sample: usize, spans: u64) {
        for start in timing::spans(spans).map(timing::query) {
            encoder.resolve_query_set(queries, start..start + 2, &self.query_resolve,
                timing::resolve_offset(sample, start));
        }
        let begin = sample == 0 || self.group_ends[sample - 1];
        let end = self.group_ends[sample];
        let query = timing::sample_query();
        if begin && end {
            encoder.resolve_query_set(queries, query..query + 2, &self.query_resolve,
                timing::resolve_offset(sample, query));
        } else if begin {
            encoder.resolve_query_set(queries, query..query + 1, &self.query_resolve,
                timing::resolve_offset(sample, query));
        } else if end {
            encoder.resolve_query_set(queries, query + 1..query + 2, &self.query_resolve,
                timing::group_end_offset(sample));
        }
    }

    /// Drain a queued successor without throwing away accepted independent output.
    pub(crate) fn collect_for_redo(&mut self, r: &Renderer, independent: bool) -> Result<()> {
        let submission = self.submission.clone().context("submitted successor has no fence")?;
        let readback = self.collect_readback(r)?;
        if independent {
            self.completed = Some(CompletedFrame {
                nominal_frame: self.nominal_frame,
                nominal_request: self.nominal_request.clone(),
                time: self.submitted_time,
                outputs: self.view.readback_outputs,
                profile: self.profile,
                write_color: self.write_color,
                submission,
                readback,
            });
        }
        Ok(())
    }

    /// Waits for this slot’s submission, leaving later slots in flight.
    pub fn collect(&mut self, r: &Renderer) -> Result<FrameOutput> {
        self.collect_readback(r)?.unpack()
    }
    /// Transfer this entry to its consumers; resubmission waits for the last lease.
    /// Device errors, naming the most recent frame and the work frames run.
    fn gpu_check(&self, r: &Renderer) -> Result<()> {
        r.errors.check().with_context(|| self.failed_frame())
    }
    /// Whether this frame completed on the device, from its decoded evidence (`completion`:
    /// timestamps and exposure count) and what the device recorded. Some drivers report a reset
    /// only on the next submission, after the old fence/map succeeded: probe without draining
    /// later frame slots. Native queue submission already performs nonblocking maintenance and
    /// delivers its errors synchronously; another poll repeats that work.
    fn complete(&self, r: &Renderer, completion: Result<()>) -> Result<()> {
        crate::command_finish::Segments::flush(&r.queue);
        r.errors.frame_complete(completion).map_err(|error| error.context(self.failed_frame()))
    }
    /// A refusal the authored scene or one of its programs caused, as this frame's diagnostics
    /// report it. The diagnostics were read from the device, so the frame's completion is
    /// established first, exactly as for a frame that is delivered: a frame that did not
    /// complete is the device's failure, whatever its diagnostics say about the scene.
    fn authored(&self, r: &Renderer, completion: Result<()>, refusal: anyhow::Error) -> anyhow::Error {
        match self.complete(r, completion) {
            Err(error) => error,
            Ok(()) => crate::failure::scene(refusal),
        }
    }
    fn failed_frame(&self) -> String {
        format!("GPU failed at or before the frame at {}s; frames run {}; frame discarded", self.submitted_time, self.work)
    }
    pub fn collect_readback(&mut self, r: &Renderer) -> Result<FrameReadback> {
        // Validate every queued producer before returning any dependent image,
        // even when the caller collects successors out of submission order.
        // Receipts use the existing slot readbacks; no extra GPU pass is needed.
        let checked = self.check_publications(r);
        let result = self.collect_readback_inner(r);
        let result = collected(checked, result);
        // Captures and retained publications may already have been consumed by
        // queued successors. An asynchronous failure invalidates the install,
        // not just this slot. Reservation retries have their own drained replay.
        if let Err(error) = &result {
            if !error.is::<FrameRedo>() {
                self.publication_error.get_or_init(|| format!("{error:#}"));
            }
        }
        if result.is_ok() {
            if let Some(error) = self.publication_error.get() {
                anyhow::bail!("frame publication invalidated; reinstall the scene: {error}");
            }
        }
        result
    }
    fn check_publications(&self, r: &Renderer) -> Result<()> {
        let Some(checks) = &self.publication_checks else {return Ok(());};
        let mut checks = checks.lock().unwrap();
        // Register every map before waiting. Concurrent submitters can append
        // receipts out of queue order, so the last receipt is not a covering fence.
        let mut pending = Vec::new();
        let mut submissions = Vec::new();
        for (submission, buffers) in checks.drain(..) {
            submissions.push(submission);
            for (kind, buffer) in buffers {
                let (tx, rx) = mpsc::channel();
                buffer.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
                pending.push((kind, buffer, rx));
            }
        }
        wait_publication_submissions(submissions, |submission| r.wait(Some(submission)))?;
        // Decode and unmap every receipt before reporting an invalid publication.
        let mut failure = None;
        let mut spill = None;
        for (kind, buffer, rx) in pending {
            rx.recv()??;
            let mapped=buffer.get_mapped_range(..);
            let code=u32::from_le_bytes(mapped[..4].try_into().unwrap());
            drop(mapped);buffer.unmap();
            // A spill arena that can still grow is a reservation retry. The redo
            // collects every queued frame, invalidates captures and replays this
            // exposure and the successors that depend on it; independent
            // successors keep their completed readbacks. Nothing this exposure
            // published is consumed afterwards, so the install stays valid.
            // Whichever frame collects first asks for it.
            if matches!(kind,crate::graph::PublicationDiagnostic::Shader) && code==constants::PATH_BOUNDARY_OVERFLOW
                && let Some(bytes)=self.spill_growth(r) {spill=Some(bytes);continue;}
            if code!=0 {
                let solver=matches!(&kind,crate::graph::PublicationDiagnostic::SolverStatus(_));
                let reason=match kind {
                    crate::graph::PublicationDiagnostic::Shader=>format!("shader diagnostic {code}"),
                    crate::graph::PublicationDiagnostic::Code(kind)=>format!("{kind} diagnostic {code}"),
                    crate::graph::PublicationDiagnostic::ComputedGeometry=>computed_geometry::validation_message(code).to_owned(),
                    crate::graph::PublicationDiagnostic::PointDisplacement=>format!("point displacement diagnostic {code}"),
                    crate::graph::PublicationDiagnostic::SolverStatus(message)=>format!("{message} (reported value: {code})"),
                };
                let message=format!("frame publication invalidated by {reason}; reinstall the scene");
                self.publication_error.get_or_init(||message.clone());
                failure.get_or_insert_with(|| if solver {crate::failure::scene(anyhow::anyhow!(message))} else {anyhow::anyhow!(message)});
            }
        }
        // What the device recorded outranks what its receipts say about the scene.
        if let Some(failure) = failure {r.errors.check()?;return Err(failure);}
        spill.map_or(Ok(()), |dielectric_spill| Err(FrameRedo(Reservations {dielectric_spill,..Default::default()}).into()))
    }
    /// The arena size a dielectric spill overflow asks for next (`spill_growth`),
    /// from this slot's arena and the device's binding limit.
    fn spill_growth(&self, r: &Renderer) -> Option<u64> {
        let limits=r.device.limits();
        spill_growth(self.materials.as_ref().map_or(0,|m|m.shader_errors.size()),
            limits.max_storage_buffer_binding_size.min(limits.max_buffer_size)&!3)
    }
    fn collect_readback_inner(&mut self, r: &Renderer) -> Result<FrameReadback> {
        self.gpu_check(r)?;
        ensure!(self.pending, "frame slot has no submitted frame");
        if let Some(done) = self.completed.take() {
            self.submission = None;
            self.pending = false;
            return Ok(done.readback);
        }
        // Small diagnostic words, read and unmapped before any error is reported.
        let diagnostics = [
            self.nonfinite_readback.as_ref().filter(|_| self.profile),
            self.slot_overflow.as_ref(),
            self.shader_error_readback.as_ref(),
            self.environment_diagnostics.as_ref().map(|v|&v.readback),
            self.exposure_report.pending(),
            self.compute.as_ref().map(|c|&c.error_readback),
            self.probe_feedback.as_ref().map(|(_,buffer)|buffer),
        ];
        let (tx, rx) = mpsc::channel();
        for buffer in diagnostics.iter().flatten().copied()
            .chain(self.read_pixels.then_some(&self.ring.readback))
            .chain(std::iter::once(&self.query_readback))
            .chain(self.simulation_checkpoint.iter().flat_map(|c| c.buffers.iter()))
        {
            let tx = tx.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
        }
        drop(tx);
        // Recycle only this slot; later submissions remain in flight.
        r.wait(self.submission.take())
            .and_then(|_| rx.into_iter().try_for_each(|mapped| mapped.map_err(anyhow::Error::from)))
            .map_err(crate::failure::device)
            .with_context(|| self.failed_frame())?;
        self.flight_ownership.complete();
        let read = |buffer: &wgpu::Buffer| {
            let mapped = buffer.get_mapped_range(..);
            let pairs: Vec<[u32; 2]> = mapped.chunks(8).map(|pair| std::array::from_fn(|i|
                pair.get(4 * i..4 * i + 4).map_or(0, |word| u32::from_le_bytes(word.try_into().unwrap()))
            )).collect();
            drop(mapped);
            buffer.unmap();
            pairs
        };
        let nonfinite=self.nonfinite_readback.as_ref().filter(|_|self.profile).map(|buffer| {
            let pairs=read(buffer);
            let wide=|pair:[u32;2]|u64::from(pair[0])|(u64::from(pair[1])<<32);
            let counts=NonfiniteCounts {shade:wide(pairs[0]),resolve:wide(pairs[1])};
            if counts.shade!=0 || counts.resolve!=0 {
                eprintln!("warning: non-finite scene-linear output evaluations: shade={}, resolve={}",counts.shade,counts.resolve);
            }
            counts
        });
        let (exposure, exposure_validity) = match self.exposure_report.read(u64::from(self.view.width) * u64::from(self.view.height)) {
            Ok(exposure) => (exposure, Ok(())),
            Err(error) => (None, Err(error)),
        };
        let environment_invalid=self.environment_diagnostics.as_ref().is_some_and(|v|read(&v.readback)[0][0]!=0);
        let placement_errors=self.compute.as_ref().map(|c|read(&c.error_readback)).and_then(|p|p.first().copied()).map_or(0,|p|p[0]);
        let probe_pages=if let Some((source,status))=&self.probe_feedback {
            debug_assert_eq!(status.size(),4,"accepted exposures transfer only probe status");
            let missing=bytemuck::from_bytes::<u32>(&status.get_mapped_range(..))!=&0;
            status.unmap();
            if missing {
                // Only rejected exposures transfer the dictionary. The retained
                // source belongs to this generation; requests are append-only.
                // Queue order drains all already submitted writers before copying,
                // including successor/capture requests needed by the replay union.
                let buffer=crate::gpu_memory::buffer(&r.device,&wgpu::BufferDescriptor {
                    label:Some("rejected probe requests"),size:source.size,
                    usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false,
                });
                let mut encoder=r.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&source.source,source.offset,&buffer,0,source.size);
                let submission=crate::command_finish::Segments::submit(&r.queue,encoder);
                let (tx,rx)=mpsc::channel();
                buffer.map_async(wgpu::MapMode::Read,..,move |result| {let _=tx.send(result);});
                r.wait(Some(submission))?;rx.recv()??;
                let mapped=buffer.get_mapped_range(..);
                crate::probes::Volume::collect_demand(self.probe_demand.as_ref().unwrap(),&mapped);
                drop(mapped);buffer.unmap();
                self.probe_generation.checked_add(1).context("probe generation exhausted")?
            } else {0}
        } else {0};
        let slots = self.slot_overflow.as_ref().map(read);
        let shader_diagnostics = self.shader_error_readback.as_ref().map(read);
        let errors = shader_diagnostics.as_ref().and_then(|pairs| pairs.first().copied());
        let exhausted = shader_diagnostics.as_ref().and_then(|pairs| pairs.get(1)).map_or(0, |pair| pair[0]);
        let mut exhausted_fragments = 0;
        let mut demands = [0u32; 2];
        for ([pool, fragments], view) in slots.iter().flatten().copied().zip(self.views()) {
            demands[0] = demands[0].max(pool);
            demands[1] = demands[1].max(fragments);
            if view.decoupled.as_ref().is_some_and(|v| v.fragments.full) {
                exhausted_fragments = exhausted_fragments.max(fragments);
            }
        }
        // The frame's completion evidence, decoded before any diagnostic is read as the scene's:
        // its GPU timestamps and its exposure count. A reservation replay does not consult it.
        let mapped = self.query_readback.get_mapped_range(..);
        let mut gpu_spans_ms: Vec<_> = constants::GPU_SPANS
            .iter()
            .map(|&name| (name, 0.))
            .collect();
        let mut gpu_begin = u64::MAX;
        let mut gpu_end = 0;
        let pair = |sample, query| -> Result<(u64, u64, f64)> {
            let offset = timing::resolve_offset(sample, query) as usize;
            let begin = u64::from_le_bytes(mapped[offset..offset + 8].try_into().unwrap());
            let end = u64::from_le_bytes(mapped[offset + 8..offset + 16].try_into().unwrap());
            let ms = timing::elapsed_ms(begin, end, r.queue.get_timestamp_period())
                .with_context(|| format!("invalid GPU timestamp pair for query {query}, sample {sample}: begin={begin}, end={end}"))?;
            Ok((begin, end, ms))
        };
        let mut sample_gpu_ms = vec![0.; self.sample_spans.len()];
        let timings = (|| -> Result<()> {
            for (sample, &spans) in self.sample_spans.iter().enumerate() {
                for name in timing::spans(spans) {
                    let (_, _, ms) = pair(sample, timing::query(name))?;
                    gpu_spans_ms.iter_mut().find(|(span, _)| *span == name).unwrap().1 += ms;
                }
            }
            let mut first = 0;
            for (last, &ends) in self.group_ends.iter().enumerate() {
                if !ends { continue; }
                let begin_offset = timing::resolve_offset(first, timing::sample_query()) as usize;
                let end_offset = if first == last { begin_offset + 8 }
                    else { timing::group_end_offset(last) as usize };
                let begin = u64::from_le_bytes(mapped[begin_offset..begin_offset + 8].try_into().unwrap());
                let end = u64::from_le_bytes(mapped[end_offset..end_offset + 8].try_into().unwrap());
                let ms = timing::elapsed_ms(begin, end, r.queue.get_timestamp_period())
                    .with_context(|| format!("invalid GPU sample group {first}..={last}: begin={begin}, end={end}"))?;
                gpu_begin = gpu_begin.min(begin);
                gpu_end = gpu_end.max(end);
                let previous: f64 = self.sample_gpu_ms.get(first..=last).unwrap_or(&[]).iter().sum();
                for (index, cost) in sample_gpu_ms.iter_mut().enumerate().take(last + 1).skip(first) {
                    *cost = if previous.is_finite() && previous > 0. { ms * self.sample_gpu_ms[index] / previous }
                        else { ms / (last - first + 1) as f64 };
                }
                first = last + 1;
            }
            Ok(())
        })();
        drop(mapped);
        let completion = timings.and(exposure_validity);
        // On error, release the frame's large readbacks too.
        let fail = |slot: &mut Self| {
            if slot.read_pixels { slot.ring.readback.unmap(); }
            slot.query_readback.unmap();
            slot.flight_ownership.retire();
            slot.pending = false;
            if let Some(checkpoint)=slot.simulation_checkpoint.take() {for buffer in checkpoint.buffers {buffer.unmap();}}
        };
        if errors.is_some_and(|[code,_]|code==constants::PATH_BOUNDARY_OVERFLOW)
            && let Some(dielectric_spill)=self.spill_growth(r) {
            fail(self);
            return Err(FrameRedo(Reservations {dielectric_spill,..Default::default()}).into());
        }
        let dropped_path_samples = shader_diagnostics.as_ref().map_or(0, |words| {
            let pair=words[(vork_slang::registry::DROPPED_PATH_SAMPLES_WORD/2) as usize];
            u64::from(pair[0]) | (u64::from(pair[1])<<32)
        });
        let mut invalid_pixels = std::collections::BTreeMap::<String, InvalidPixels>::new();
        if let Some(words) = &shader_diagnostics {
            let bytes: &[u8] = bytemuck::cast_slice(words);
            let mut offset = crate::materials::SHADER_DIAGNOSTICS_BYTES as usize;
            for view in self.views() {
                let size = (view.post.invalid_bitmap.end - view.post.invalid_bitmap.start) as usize;
                let bitmap = &bytes[offset..offset + size];
                if bitmap.iter().any(|&byte| byte != 0) {
                    invalid_pixels.entry(view.diagnostic_name.clone()).or_default().merge(bitmap, [view.width, view.height]);
                }
                offset += size;
            }
        }
        if environment_invalid {
            fail(self);
            return Err(self.authored(r, completion, FrameRefused("environment radiance must be finite and nonnegative; fix the environment program; frame refused; see vork help upgrade".into()).into()));
        }
        if placement_errors!=0 {
            fail(self);
            return Err(self.authored(r, completion, anyhow::anyhow!("computed placement rejected: {}{}",
                if placement_errors&1!=0 {"live count exceeds installed capacity; "} else {""},
                if placement_errors&2!=0 {"nonfinite translation/scale or invalid unit rotation"} else {""})));
        }
        // The shader reports its largest band's evaluations; the frame estimate
        // gives every full band of the most finely banded view that many.
        let band_share=self.views().filter_map(|view|view.decoupled.as_ref()).map(|v|(u64::from(v.extent[1]),u64::from(v.band_rows)))
            .max_by(|a,b|(a.0*b.1).cmp(&(b.0*a.1))).unwrap_or((1,1));
        let shading_evaluations=shader_diagnostics.as_ref().and_then(|p|p.get(1)).map_or(0,|pair|(u64::from(pair[1])*band_share.0).div_ceil(band_share.1));
        if reservation_retry_allowed(errors) && (demands.iter().any(|&demand| demand != 0) || shading_evaluations!=0 || probe_pages!=0) {
            let [visibility_pool, transparent_fragments] = demands;
            if self.profile {
                eprintln!("frame reservation retry: visibility={visibility_pool}, fragments={transparent_fragments}, evaluations={shading_evaluations}");
            }
            fail(self);
            ensure!(exhausted_fragments == 0,
                crate::gpu_memory::device_limit("transparent fragments", format_args!("{exhausted_fragments} records per merge batch ({} bytes per record)",vork_slang::registry::FRAGMENT_RECORD_WORDS*4), format_args!("{} bytes",r.device.limits().max_storage_buffer_binding_size.min(r.device.limits().max_buffer_size)), "reduce resolution or transparent surfaces overlapping at a pixel"));
            return Err(FrameRedo(Reservations { probe_pages, visibility_pool, transparent_fragments, shading_evaluations, ..Default::default() }).into());
        }
        if exhausted != 0 {
            eprintln!("warning: solid transmission exhausted the internal continuation budget (object {}); simplify glass geometry", exhausted - 1);
        }
        if self.profile {
            let queries=shader_diagnostics.as_ref().and_then(|words|words.get(3)).map_or(0,|words|words[0]);
            if queries!=0 {eprintln!("trace rays: {queries}");}
        }
        if let Some([count, capacity]) = errors.filter(|[count, _]| *count != 0) {
            fail(self);
            if count == vork_slang::registry::CAMERA_COVERAGE_INVALID {anyhow::bail!("cached opaque slot coverage exceeds the camera sample budget; frame refused");}
            if count == constants::PATH_BOUNDARY_OVERFLOW {
                let mut ids=vec![capacity];
                if let Some(words)=&shader_diagnostics {
                    ids.extend(words.get(8).into_iter().flatten().copied());
                    if let Some(pair)=words.get(9) {ids.push(pair[0]);}
                }
                ids.sort_unstable();ids.dedup();
                let names=ids.iter().filter_map(|&id|self.boundary_names.get(id as usize)).cloned().collect::<Vec<_>>().join(", ");
                let bytes=self.materials.as_ref().map_or(0,|m|m.shader_errors.size());
                let limits=r.device.limits();
                anyhow::bail!(crate::gpu_memory::device_limit(format_args!("dielectric interior spill arena; active glass examples: {names}"), format_args!("more than the allocated {bytes} bytes"), format_args!("device memory={} bytes, max_storage_buffer_binding_size={} bytes, max_buffer_size={} bytes",r.device_memory,limits.max_storage_buffer_binding_size,limits.max_buffer_size), "reduce overlapping glass geometry or resolution"));
            }
            if count == vork_slang::registry::GEOMETRY_NONFINITE {return Err(self.authored(r, completion, anyhow::anyhow!("geometry prefix on object {capacity} produced nonfinite geometry; make displacement positions and shading frames finite")));}
            if count == vork_slang::registry::IMPLICIT_INVALID { return Err(self.authored(r, completion, anyhow::anyhow!("implicit grid has a nonfinite sample, constant-zero cell or singular zero gradient (object {capacity})"))); }
            if count == constants::PATH_BOUNDARY_AMBIGUOUS { return Err(self.authored(r, completion, anyhow::anyhow!("path solid boundary requires the same IOR and absorption descriptor in every surface lobe (object {capacity})"))); }
            if count == vork_slang::registry::PATH_MEDIUM_DISTANCE {
                return Err(self.authored(r, completion, anyhow::anyhow!("medium collision distance exceeds the renderer's finite coordinate range; use a finite fog carrier or rescale scene distances and coefficients; frame refused")));
            }
            if count == vork_slang::registry::PATH_MEDIUM_INVALID {
                return Err(self.authored(r, completion, anyhow::anyhow!("medium program {capacity} returned invalid physical coefficients: absorption, scattering and emission must be finite and nonnegative and anisotropy strictly between -1 and 1; frame refused")));
            }
            if count == vork_slang::registry::PATH_MEDIUM_WORK {
                if capacity>>24==255 {anyhow::bail!("stage 1 fog does not support custom light callbacks; use std/openpbr surfaces or render the fog externally; frame refused without an image");}
                anyhow::bail!("fog path exceeded the work limit; simplify the scene or render fog separately; frame refused");
            }
            if count == vork_slang::registry::PATH_MEDIUM_MAJORANT {
                return Err(FrameRefused(format!("certified medium extinction bound violated (observed {}): shader produced coefficients outside its compiler certificate; frame refused",f32::from_bits(capacity))).into());
            }
            if count == vork_slang::registry::ENVIRONMENT_INVALID {return Err(self.authored(r, completion, FrameRefused("environment radiance must be finite and nonnegative; fix the environment program; frame refused; see vork help upgrade".into()).into()));}
            if count == vork_slang::registry::RADIANCE_NONFINITE {return Err(FrameRefused("non-finite scene-linear radiance; frame refused".into()).into());}
            // The word says the displayed colour was non-finite, not where it became so: a view that
            // passes non-finite radiance through reports the same. Not known to be the scene's.
            if count == vork_slang::registry::DISPLAY_NONFINITE {return Err(FrameRefused(crate::display_program::NONFINITE.into()).into());}
            if count == vork_slang::registry::VIEW_DESTINATION {return Err(self.authored(r, completion, anyhow::anyhow!(crate::display_program::OTHER_DISPLAY)));}
            if count == vork_slang::registry::TRACE_EXHAUSTED {anyhow::bail!("trace work exhausted: ray budget or traversal candidate limit (code {capacity}); exposure discarded");}
            if count == vork_slang::registry::CAMERA_RAY_INVALID {return Err(self.authored(r, completion, anyhow::anyhow!("camera program returned a non-finite ray, zero direction or invalid t_min/t_max interval")));}
            if count == vork_slang::registry::PATH_PROPAGATION_ERROR {anyhow::bail!("authored propagation failed: invalid or non-progressing segment, or free-flight work reservation exhausted (code {capacity})");}
            if count == vork_slang::registry::PATH_MEDIUM_DIVERGENT {return Err(self.authored(r, completion, anyhow::anyhow!("unbounded medium emission has no proven finite integral: a constant zero-extinction channel diverges; varying emission without asymptotic closure is unsupported; use a finite mesh boundary")));}
            anyhow::bail!("unrecognized shader diagnostic code {count} (detail {capacity})");
        }
        if shader_diagnostics.as_ref().is_some_and(|words|
            words[(vork_slang::registry::TERMINATED_MEDIUM_PATHS_WORD/2) as usize] != [0,0]) {
            fail(self);
            anyhow::bail!("fog path exceeded the work budget; simplify the scene or render fog separately; frame refused; see vork help upgrade");
        }
        if let Err(error) = self.complete(r, completion) {
            fail(self);
            return Err(error);
        }
        self.query_readback.unmap();
        self.flight_ownership.retire();
        self.pending = false;
        self.sample_gpu_ms = sample_gpu_ms;
        let timing_source = "gpu_timestamp";
        let readback = if self.read_pixels {
            Some(std::sync::Arc::new(ReadbackLease {
                neutral_motion: self.propagation_motion,
                buffer: self.ring.readback.clone(),
                mapping: Default::default(),
                pixel_count: self.view.width as usize * self.view.height as usize,
                outputs: self.view.readback_outputs,
                has_catchers: self.view.has_catchers,
                aovs: self.view.aovs.clone(),
                aov_base: if self.view.float_output {constants::FLOAT_OUTPUT_PLANES as usize} else {constants::PACKED_OUTPUT_PLANES as usize},
                display_base: self.view.post.display_base.map(|b| b as usize),
            }))
        } else {
            None
        };
        self.gpu_check(r)?;
        // A rejected exposure must not persist partially evaluated compute state.
        if let Some(checkpoint) = self.simulation_checkpoint.take() { checkpoint.publish(); }
        Ok(FrameReadback {
            credit: self
                .read_pixels
                .then(|| std::sync::Arc::new(self.ring.credit.acquire())),
            readback,
            exposure,
            camera: self.camera_data.clone(),
            timing: FrameTiming {
                cpu_prepare_ms: self.prep_ms,
                cpu_submit_ms: self.cpu_ms,
                cpu_finish_ms: self.finish_ms,
                denoise_wait_ms: self.denoise_ms.wait_ms,
                denoise_pack_ms: self.denoise_ms.pack_ms,
                denoise_filter_ms: self.denoise_ms.filter_ms,
                denoise_upload_ms: self.denoise_ms.upload_ms,
                timing_source,
                gpu_ms: gpu_spans_ms.iter().map(|(_, ms)| ms).sum(),
                gpu_elapsed_ms: timing::elapsed_ms(gpu_begin, gpu_end, r.queue.get_timestamp_period()).unwrap_or(0.),
                gpu_spans_ms,
                nonfinite,
                terminated_medium_paths: shader_diagnostics.as_ref().map_or(0, |words| {
                    let pair=words[(vork_slang::registry::TERMINATED_MEDIUM_PATHS_WORD/2) as usize];
                    u64::from(pair[0]) | (u64::from(pair[1])<<32)
                }),
                discarded_medium_radiance: shader_diagnostics.as_ref().map_or(0., |words| {
                    let word=vork_slang::registry::DISCARDED_MEDIUM_RADIANCE_WORD as usize;
                    f64::from(f32::from_bits(words[word/2][word%2]))
                }),
                dropped_path_samples,
                invalid_pixels,
                dropped_light_samples: shader_diagnostics.as_ref().map_or(0, |words| {
                    let pair = words[(vork_slang::registry::DROPPED_LIGHT_SAMPLES_WORD / 2) as usize];
                    u64::from(pair[0]) | (u64::from(pair[1]) << 32)
                }),
                sample_count: self.sample_spans.len(),
                // Only a submission promotes variants, so this is what the frame drew.
                variants_pending: self.view.variants_pending(),
                path_kernel: self.path_kernel,
            },
        })
    }
}

/// Unknown samples stay alone; known adjacent costs share the GPU budget.
fn sample_group_ends(costs: &[f64], samples: usize) -> Vec<bool> {
    let mut ends = vec![false; samples];
    let mut total = 0.;
    for sample in 0..samples {
        let cost = costs.get(sample).copied().unwrap_or(0.);
        if !cost.is_finite() || cost <= 0. {
            if sample > 0 { ends[sample - 1] = true; }
            ends[sample] = true;
            total = 0.;
        } else {
            if total > 0. && total + cost > constants::SAMPLE_GROUP_GPU_MS {
                ends[sample - 1] = true;
                total = 0.;
            }
            total += cost;
        }
    }
    if let Some(last) = ends.last_mut() { *last = true; }
    ends
}

#[cfg(test)]
mod profiling_tests {
    use super::*;

    /// A frame with device evidence and a scene diagnostic is the device's; a scene
    /// diagnostic alone stays the scene's, ahead of an uncategorised downstream error.
    #[test]
    fn device_evidence_outranks_a_scene_diagnostic() {
        use crate::failure::{FailureKind, exit_code};
        let solver = || -> Result<()> { Err(crate::failure::scene(anyhow::anyhow!("frame publication invalidated by solver diverged"))) };
        let scene = || -> Result<u8> { Err(crate::failure::scene(anyhow::anyhow!("camera program returned a non-finite ray"))) };
        // A receipt the device would not map is device evidence too, whatever collection says.
        let receipt = || -> Result<()> { Err(wgpu::BufferAsyncError.into()) };
        assert_eq!(exit_code(&collected(receipt(), scene()).unwrap_err()), 4);
        assert_eq!(exit_code(&collected::<u8>(receipt(), Err(anyhow::anyhow!("downstream"))).unwrap_err()), 4);
        assert!(collected(receipt(), Ok(1)).unwrap_err().is::<wgpu::BufferAsyncError>());
        let lost = || -> Result<u8> { Err(anyhow::anyhow!("GPU device lost").context(FailureKind::Device)) };
        let unmapped = || -> Result<u8> { Err(anyhow::Error::from(wgpu::BufferAsyncError).context("frame 3")) };
        assert_eq!(exit_code(&collected(solver(), lost()).unwrap_err()), 4);
        assert_eq!(exit_code(&collected(solver(), unmapped()).unwrap_err()), 4);
        assert_eq!(exit_code(&collected(solver(), Ok(1)).unwrap_err()), 3);
        assert_eq!(exit_code(&collected::<u8>(solver(), Err(anyhow::anyhow!("downstream"))).unwrap_err()), 3);
        assert_eq!(exit_code(&collected(Ok(()), lost()).unwrap_err()), 4);
        assert_eq!(exit_code(&collected(Err(anyhow::anyhow!("receipt")), Ok(1)).unwrap_err()), 4);
        assert_eq!(collected(Ok(()), Ok(1)).unwrap(), 1);
    }
    #[test]
    fn reservation_replay_allows_only_provisional_colour_errors() {
        use vork_slang::registry::*;
        assert!(reservation_retry_allowed(None));
        for code in [0, RADIANCE_NONFINITE, DISPLAY_NONFINITE] {
            assert!(reservation_retry_allowed(Some([code, 17])));
        }
        for code in [PATH_MEDIUM_WORK, PATH_MEDIUM_MAJORANT, PATH_MEDIUM_INVALID,
            PATH_MEDIUM_DISTANCE, PATH_MEDIUM_DIVERGENT, PATH_BOUNDARY_OVERFLOW,
            PATH_BOUNDARY_AMBIGUOUS, TRACE_EXHAUSTED, CAMERA_RAY_INVALID,
            PATH_PROPAGATION_ERROR, CAMERA_COVERAGE_INVALID, IMPLICIT_INVALID,
            GEOMETRY_NONFINITE, ENVIRONMENT_INVALID, 1] {
            assert!(!reservation_retry_allowed(Some([code, 17])), "code {code}");
        }
    }


    #[test]
    fn gpu_budget_sizes_groups_from_sample_time() {
        assert_eq!(sample_group_ends(&[], 256), vec![true;256]);
        let unit = constants::SAMPLE_GROUP_GPU_MS / 5.;
        let uniform = sample_group_ends(&[unit;16],16);
        assert_eq!(uniform.iter().enumerate().filter_map(|(i,&end)|end.then_some(i)).collect::<Vec<_>>(),[4,9,14,15]);
        let mut costs = vec![unit / 10.;21];costs[0]=unit * 4.;
        let varied = sample_group_ends(&costs,21);
        assert_eq!(varied.iter().enumerate().filter_map(|(i,&end)|end.then_some(i)).collect::<Vec<_>>(),[10,20],
            "one expensive first sample must not size every later group");
        assert_eq!(sample_group_ends(&[constants::SAMPLE_GROUP_GPU_MS * 2.;3],3),[true;3]);
        assert_eq!(sample_group_ends(&[10.,0.,10.,10.],4),[true,true,false,true],
            "unknown samples cannot share a submission with measured work");
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn nonfinite_atomic_counts_carry_past_u32() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let code=r#"import output_diagnostics; import vork_constants;
            [[vk::binding(0,0)]] RWStructuredBuffer<uint> counters;
            [shader("compute")][numthreads(1,1,1)] void numerical_test() {
                check_nonfinite(counters,REG_NONFINITE_SHADE,float4(asfloat(0x7f800000u)));
                check_nonfinite(counters,REG_NONFINITE_RESOLVE,float4(asfloat(0x7fc00000u)));
            }"#;
        let pipeline=crate::medium::test_pipeline(&renderer,code,&Default::default(),&[
            crate::pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}),
        ],(1,1,1))?;
        let counters=init(&renderer.device,"carry counters",bytemuck::cast_slice(&[1u32,u32::MAX,7,u32::MAX,11]),wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let bind=renderer.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,counters.as_entire_binding())]});
        let readback=crate::gpu_memory::buffer(&renderer.device,&wgpu::BufferDescriptor {label:None,size:20,usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
        let mut encoder=renderer.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(1,1,1);}
        encoder.copy_buffer_to_buffer(&counters,0,&readback,0,20);
        let submission=renderer.queue.submit([encoder.finish()]);
        let (tx,rx)=mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|result|{let _=tx.send(result);});
        renderer.device.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;rx.recv()??;
        let data=readback.get_mapped_range(..);
        assert_eq!(bytemuck::cast_slice::<u8,u32>(&data),[1,0,8,0,12],"Inf and NaN counts carry into the high word");
        drop(data);readback.unmap();Ok(())
    }

    #[test]
    fn spill_growth_quadruples_up_to_the_device_limit() {
        let (mib, limit) = (1u64 << 20, 1u64 << 30);
        assert_eq!(spill_growth(mib, limit), Some(4 * mib));
        assert_eq!(spill_growth(4 * mib, limit), Some(16 * mib));
        // The last step stops at the limit, and then growth ends.
        assert_eq!(spill_growth(limit / 2, limit), Some(limit));
        assert_eq!(spill_growth(limit - 4, limit), Some(limit));
        assert_eq!(spill_growth(limit, limit), None);
        assert_eq!(spill_growth(limit + 4, limit), None);
    }

    /// A frame's shader diagnostics are also a publication receipt in scenes
    /// with media, compute, displacement or captures. A spill arena overflow
    /// there must ask for the redo that grows the arena; it used to invalidate
    /// the install before collection could. Every other code still invalidates.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn spill_overflow_receipts_redo_instead_of_invalidating() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version": 2, "render": {"resolution": [16, 16],
                "background": [0.2, 0.3, 0.4],
                "settings": {"integrator": "path", "samples": 1, "denoiser": null}},
            "nodes": [{"type": "camera"}]
        }))?;
        let installed = renderer.install(scene, Path::new("."))?;
        let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        let collect_with = |frame: &mut FrameSlot, code: u32| -> Result<anyhow::Error> {
            frame.submit(&installed, &renderer, 0.)?;
            let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shader diagnostic receipt"), size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            renderer.queue.write_buffer(&buffer, 0, &code.to_le_bytes());
            let submission = crate::command_finish::Segments::flush(&renderer.queue);
            let checks: crate::graph::PublicationChecks = Default::default();
            checks.lock().unwrap().push((submission, vec![(crate::graph::PublicationDiagnostic::Shader, buffer)]));
            frame.publication_checks = Some(checks);
            frame.collect(&renderer).err().context("a nonzero shader receipt was accepted")
        };
        let error = collect_with(&mut frame, constants::PATH_BOUNDARY_OVERFLOW)?;
        let redo = error.downcast_ref::<FrameRedo>().with_context(|| format!("overflow receipt must redo: {error:#}"))?;
        assert!(redo.0.dielectric_spill > 0);
        assert!(frame.publication_error.get().is_none(), "a reservation retry invalidated the install");
        let error = collect_with(&mut frame, vork_slang::registry::TRACE_EXHAUSTED)?;
        assert!(!error.is::<FrameRedo>(), "{error:#}");
        assert!(frame.publication_error.get().is_some());
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn collection_submits_only_the_post_map_loss_probe() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let scene = vork_scene::v2::from_test_value(serde_json::json!({
            "version": 2, "render": {"resolution": [16, 16],
                "background": [0.2, 0.3, 0.4],
                "settings": {"integrator": "path", "samples": 1, "denoiser": null}},
            "nodes": [{"type": "camera"}]
        }))?;
        let installed = renderer.install(scene, Path::new("."))?;
        let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        // Multiple producers and receipts exercise the publication path as well
        // as ordinary collection, with and without CPU pixel readback.
        for receipts in [0, 1, 4] {
            if receipts == 4 { frame.submit_gpu(&installed, &renderer, 0.)?; }
            else { frame.submit(&installed, &renderer, 0.)?; }
            let checks: crate::graph::PublicationChecks = Default::default();
            for _ in 0..receipts {
                let mut buffers = Vec::new();
                for _ in 0..3 {
                    let buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("publication count receipt"), size: 4,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    renderer.queue.write_buffer(&buffer, 0, &0u32.to_le_bytes());
                    buffers.push((crate::graph::PublicationDiagnostic::Code("test"), buffer));
                }
                let submission = crate::command_finish::Segments::flush(&renderer.queue);
                checks.lock().unwrap().push((submission, buffers));
            }
            frame.publication_checks = Some(checks);
            let before = crate::command_finish::submission_count();
            let heartbeats = crate::gpu_memory::wait_heartbeat_count();
            frame.collect(&renderer)?;
            // A wait that outlasts a loss-probe window submits one empty
            // heartbeat per window; their number follows host and GPU latency.
            // Everything else collection submits is the single post-map probe.
            let heartbeats = crate::gpu_memory::wait_heartbeat_count() - heartbeats;
            assert_eq!(crate::command_finish::submission_count() - before - heartbeats, 1,
                "collection must submit one loss probe regardless of receipt count ({heartbeats} wait heartbeats excluded)");
        }
        Ok(())
    }

    /// wgpu panicked ("Error in Device::poll: Parent device is lost") when a GPU
    /// timeout or fault lost the device mid-frame (Metal watchdog, NVIDIA Xid 109).
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn lost_device_fails_the_frame_with_its_work_instead_of_panicking() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let mut input: serde_json::Value = vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"), include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        input["programs"] = serde_json::json!({"grey":{"source":"struct Inputs {}; Closure surface(ShadingContext ctx,Inputs inputs) { Closure c={}; return c; }"}});
        input["materials"] = serde_json::json!({"grey":{"program":"grey"}});
        input["nodes"][1]["materials"] = serde_json::json!(["grey"]);
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let installed = renderer.install(vork_scene::v2::from_test_value(input)?, &base)?;
        let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        frame.submit(&installed, &renderer, 0.25)?;
        renderer.device.destroy();
        let error = format!("{:#}", frame.collect(&renderer).err().context("a lost device completed the frame")?);
        assert!(error.contains("GPU failed at or before the frame at 0.25s") && error.contains("passes ") && error.contains("scene programs grey")
            && error.contains("lost"), "{error}");
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn validation_failure_names_frame_without_shader_fault_hint() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let input = vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"), include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let installed = renderer.install(vork_scene::v2::from_test_value(input)?, &base)?;
        let frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        // An invalid host-side clear is rejected before any shader executes.
        let buffer = init(&renderer.device, "validation test source", &[0; 4], wgpu::BufferUsages::COPY_SRC);
        let mut encoder = renderer.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&buffer, 0, None); // requires COPY_DST
        let _ = encoder.finish();
        let error = format!("{:#}", frame.gpu_check(&renderer).err().context("invalid buffer usage was accepted")?);
        assert!(error.contains("validation error") && error.contains("passes "), "{error}");
        assert!(!error.contains("loops for seconds") && !error.contains("driver reset"), "{error}");
        Ok(())
    }

    /// A program that never finishes: NVIDIA times the dispatch out (Xid 109) and
    /// Metal's watchdog aborts it. vork hung for 20 minutes (L4) or panicked
    /// (Metal); now the frame fails with device loss and names the program.
    #[test]
    #[ignore = "needs a GPU whose driver times out hung dispatches (NVIDIA, Metal); llvmpipe has no watchdog"]
    fn endless_program_fails_the_frame_with_device_loss() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let mut input: serde_json::Value = vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"), include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        input["animation"] = serde_json::json!([]);
        input["programs"] = serde_json::json!({"spin":{"source":"struct Inputs { float loops = 4000000000.; }; Closure surface(ShadingContext ctx, Inputs inputs) { Closure c = {}; float acc = ctx.POSITION.x; for (uint i = 0; i < uint(inputs.loops); i++) { acc = frac(sin(acc * 12.9898 + float(i)) * 43758.5453); } c.base_color = float3(acc); return c; }"}});
        input["materials"] = serde_json::json!({"spin":{"program":"spin"}});
        input["nodes"][1]["materials"] = serde_json::json!(["spin"]);
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let installed = renderer.install(vork_scene::v2::from_test_value(input)?, &base)?;
        let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        let error = frame.submit(&installed, &renderer, 0.).and_then(|_| frame.collect(&renderer).map(|_| ()))
            .err().context("an endless program completed")?;
        let error = format!("{error:#}");
        assert!(error.contains("device lost") && error.contains("scene programs spin"), "{error}");
        Ok(())
    }

    /// Runs on lavapipe too: no watchdog/reset or physical GPU is required.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn missing_timestamps_discard_frame() -> Result<()> {
        for fault in [0, 1, 2, 3] {
            let renderer = crate::test_renderer()?;
            let mut input = vork_scene::v2::with_test_settings(
                include_str!("../tests/fixtures/analytic.json"),
                include_str!("../tests/presets/analytic-7fefd52a.json"))?;
            input["render"]["resolution"] = serde_json::json!([32,24]);
            input["render"]["settings"]["samples"] = 1.into();
            vork_scene::v2::replace_integrator(&mut input, serde_json::json!({"integrator":"path","denoiser":null}));
            // Self-contained fixture retained from fix/wintdr-readback so this
            // regression also runs as a standalone GPU test executable.
            let base = std::env::temp_dir().join(format!("vork-timestamp-readback-{}-{fault}", std::process::id()));
            std::fs::create_dir_all(&base)?;
            std::fs::write(base.join("geometry.bin"), include_bytes!("../tests/fixtures/geometry.bin"))?;
            let installed = renderer.install(vork_scene::v2::from_test_value(input)?, &base)?;
            let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
            frame.profile = true;
            frame.submit(&installed, &renderer, 0.)?;
            let valid = frame.collect(&renderer)?;
            assert!(valid.linear_rgba.as_ref().unwrap().iter().any(|pixel| pixel[3] > 0.));
            assert!(valid.exposure.is_some_and(|exposure| exposure.coverage > 0.));
            if let Ok(out) = std::env::var("VORK_TEST_FRAME_OUT") {
                std::fs::create_dir_all(&out)?;
                image::save_buffer(Path::new(&out).join(format!("valid-before-fault-{fault}.png")), &valid.rgba, 32, 24, image::ColorType::Rgba8)?;
            }
            frame.submit(&installed, &renderer, 0.)?;
            // Simulate successful map/fence but unwritten queries and zero image,
            // as observed after Windows TDR. No loss callback is injected.
            let mut encoder = renderer.device.create_command_encoder(&Default::default());
            let offset = timing::resolve_offset(0, timing::sample_query());
            if fault == 3 {
                encoder.clear_buffer(&frame.query_readback, 0, None);
            } else if fault < 2 {
                encoder.clear_buffer(&frame.query_readback, offset + if fault == 1 { 8 } else { 0 }, Some(if fault == 1 { 8 } else { 16 }));
            }
            encoder.clear_buffer(&frame.ring.readback, 0, None);
            if fault >= 2 {
                encoder.clear_buffer(frame.exposure_report.pending().unwrap(), 0, None);
            }
            frame.submission = Some(renderer.queue.submit([encoder.finish()]));
            let error = frame.collect(&renderer).err().context("corrupt frame was published")?;
            assert_eq!(crate::failure::exit_code(&error), 4);
            let message = format!("{error:#}");
            assert!(message.contains("GPU device lost") && message.contains("frame discarded"), "{message}");
            assert!(message.contains(if fault == 2 { "incomplete GPU exposure readback" } else { "invalid GPU" }), "{message}");
            assert!(!frame.pending);
            assert!(frame.publication_error.get().is_some());
            assert!(frame.submit(&installed, &renderer, 0.).is_err(), "suspect device accepted a successor");
            std::fs::remove_dir_all(base)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn actual_sample_uploads_preserve_animated_exposure() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let mut input = vork_scene::v2::with_test_settings(
            include_str!("../tests/fixtures/analytic.json"),
            include_str!("../tests/presets/analytic-7fefd52a.json"))?;
        input["render"]["resolution"] = serde_json::json!([32,24]);
        input["render"]["settings"]["samples"] = 8.into();
        input["render"]["motion_blur"] = serde_json::json!({"shutter":1});
        vork_scene::v2::replace_integrator(&mut input, serde_json::json!({"integrator":"path","denoiser":null}));
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let installed = renderer.install(vork_scene::v2::from_test_value(input)?, &base)?;
        let mut batched = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        assert!(batched.can_batch_samples(&installed));
        assert_eq!(batched.sample_upload.size(), 4, "install reserved speculative sample storage");
        batched.submit(&installed, &renderer, 0.5)?;
        let expected = batched.collect(&renderer)?.linear_rgba.unwrap();
        assert!(!batched.exposure_upload.is_empty());
        for (buffer, bytes) in &batched.exposure_upload {
            assert_eq!(buffer.size(), bytes.len() as u64,
                "staging must scale with actual dirty bytes, not destination capacity or sample budget");
        }
        let mut streamed = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
        streamed.batch_uploads = false;
        assert!(!streamed.can_batch_samples(&installed));
        streamed.submit(&installed, &renderer, 0.5)?;
        crate::test_linear::assert_linear_eq(&expected,&streamed.collect(&renderer)?.linear_rgba.unwrap());
        // Reusing the slots must discard the preceding exposure's sources without
        // aliasing any source still consumed by another sample.
        batched.submit(&installed, &renderer, 0.75)?;
        let expected = batched.collect(&renderer)?.linear_rgba.unwrap();
        streamed.submit(&installed, &renderer, 0.75)?;
        crate::test_linear::assert_linear_eq(&expected,&streamed.collect(&renderer)?.linear_rgba.unwrap());
        renderer.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    #[cfg(vork_private_tests)] // Uses the private multi-capture monitor scene.
    fn shutter_zero_crossings_match_fixed_exposures_with_captures() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut input: serde_json::Value = vork_scene::v2::from_test_str(include_str!(
            "../../../scenes/render-image-monitor.json"))?;
        input["render"]["resolution"] = serde_json::json!([32,32]);
        input["render"]["settings"] = serde_json::json!({"integrator":"raster","samples":4,
            "seed":11,"denoiser":null,"fast_gi":null,"reflections":"environment"});
        input["render"]["motion_blur"] = serde_json::json!({"shutter":0});
        input["render"]["fps"] = 1.into();
        input["render"]["duration"] = 3.into();
        for image in input["images"].as_object_mut().unwrap().values_mut() {
            image["render"]["resolution"] = serde_json::json!([16,16]);
        }
        let track = serde_json::json!({"tracks":[{"path":["render","motion_blur","shutter"],
            "keys":[[0,0],[1,1],[2,0]]}]});
        input["animation"].as_array_mut().unwrap().push(track);
        let installed = r.install(crate::test_scene(input.clone(), Path::new("."))?, Path::new("."))?;
        assert_eq!((installed.plan.exposure_samples,installed.plan.visibility_samples),(4,4));
        let mut slot = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        for (time,width) in [(0.,0.),(0.5,0.5),(1.,1.),(2.,0.),(1.,1.),(0.,0.)] {
            slot.submit(&installed,&r,time)?;
            let actual = slot.collect(&r)?;
            assert_eq!(slot.view.sample_count,if width>0. {4} else {1});
            assert_eq!(slot.view.visibility_samples,if width>0. {1} else {4});
            let mut fixed = input.clone();
            fixed["animation"].as_array_mut().unwrap().pop();
            fixed["render"]["motion_blur"] = if width>0. {serde_json::json!({"shutter":width})} else {serde_json::Value::Null};
            let reference = r.install(crate::test_scene(fixed, Path::new("."))?, Path::new("."))?;
            let mut reference_slot = reference.create_frame_slot(&r,crate::ReadbackPrecision::Float)?;
            reference_slot.submit(&reference,&r,time)?;
            crate::test_linear::assert_linear_eq((actual).linear_rgba.as_ref().expect("linear readback"), (reference_slot.collect(&r)?).linear_rgba.as_ref().expect("linear readback"));
        }
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn single_sample_motion_blur_reserves_its_nominal_control_pass() -> Result<()> {
        // One shutter sample with depth/id/normal/motion outputs records a nominal
        // control pass after it; install must reserve storage for both.
        let renderer = crate::test_renderer()?;
        for integrator in ["raster", "path"] {
            let input = vork_scene::v2::from_test_value(serde_json::json!({
                "version":2,
                "render":{"resolution":[16,16],"motion_blur":{},"settings":{"integrator":integrator,"denoiser":null,"samples":1}},
                "nodes":[{"type":"camera"}]
            }))?;
            let installed = renderer.install(input,Path::new("."))?;
            assert_eq!(installed.plan.samples, 2, "{integrator}: one shutter sample plus the nominal control pass");
            let mut frame = installed.create_frame_slot(&renderer,crate::ReadbackPrecision::Float)?;
            for time in [0., 0.5] {
                frame.submit(&installed,&renderer,time)?;
                frame.collect(&renderer)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn spatial_visibility_is_segmented_before_exposure_submission() -> Result<()> {
        let r = crate::test_renderer()?;
        let input = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,
            "render":{"resolution":[8,8],"settings":{"integrator":"raster","samples":1024}},
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,-3],[1,-1,-3],[0,1,-3]]}},
            "geometry":{"triangle":{"attributes":{"POSITION":"p"}}},
            "nodes":[{"type":"camera"},{"type":"mesh","geometry":"triangle"}]
        }))?;
        let installed = r.install(input, Path::new("."))?;
        assert_eq!(installed.plan.exposure_samples, 1);
        assert_eq!(installed.plan.visibility_samples, 1024);
        let mut frame = installed.create_frame_slot(&r, crate::ReadbackPrecision::Float)?;
        frame.submit(&installed, &r, 0.)?;
        assert!(frame.sample_segments.segmented(), "spatial samples must publish their merge groups before the entire exposure ends");
        frame.collect(&r)?;
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn sample_grouping_preserves_pixels_timing_and_deformation_queries() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let input = vork_scene::v2::from_test_value(serde_json::json!({
            "version":2,
            "render":{"resolution":[16,16],"settings":{"integrator":"raster","samples":64}},
            "buffers":{
                "p":{"type":"vec3","data":[[-1,-1,-3],[1,-1,-3],[0,1,-3]]},
                "delta":{"type":"vec3","data":[[0,0,0],[0,0,0],[0,0,0]]}},
            "geometry":{"triangle":{"attributes":{"POSITION":"p"},"morphs":{"pose":{"POSITION":"delta"}}}},
            "nodes":[{"type":"camera"},{"type":"mesh","geometry":"triangle","morph_weights":{"pose":0}}]
        }))?;
        let installed = renderer.install(input,Path::new("."))?;
        let mut frame = installed.create_frame_slot(&renderer,crate::ReadbackPrecision::Float)?;
        let mut previous:Option<Vec<[f32;4]>> = None;
        for (estimate, diagnostics) in [(0., false), (10., true), (0.01, false)] {
            frame.sample_gpu_ms = vec![estimate; installed.plan.samples as usize];
            frame.profile = diagnostics;
            frame.submit(&installed, &renderer, 0.)?;
            let pixels = frame.collect(&renderer)?;
            assert!(frame.sample_gpu_ms.iter().sum::<f64>() <= pixels.timing.gpu_elapsed_ms + 1e-6,
                "measured group envelopes must fit within the complete exposure");
            if let Some(previous) = &previous { crate::test_linear::assert_linear_eq(pixels.linear_rgba.as_ref().unwrap(),previous); }
            previous = pixels.linear_rgba;
            assert!(frame.sample_gpu_ms.iter().any(|&ms| ms > 0.), "scheduling feedback also runs without diagnostics");
            // Every render resolves its pass spans (the stage breakdown needs
            // them); only non-finite counting stays behind diagnostics.
            for (sample, &mask) in frame.sample_spans.iter().enumerate() {
                assert_eq!(mask & timing::mask("deform") != 0, sample == 0 || sample == 64,
                    "the deform span is resolved exactly where deformation is recorded, with or without diagnostics");
            }
            assert_eq!(pixels.timing.nonfinite.is_some(), diagnostics, "non-finite counters stay optional");
            if renderer.adapter_info.backend == wgpu::Backend::Metal {
                assert_eq!(frame.sample_queries.len(), 1);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn subsurface_queries_follow_installed_colour_passes() -> Result<()> {
        let renderer = crate::test_renderer()?;
        for enabled in [false, true] {
            let input = vork_scene::v2::from_test_value(serde_json::json!({
                "version": 2,
                "render": {"camera":"camera", "resolution":[16,16], "settings":{"integrator":"raster","samples":64}},
                "materials": {"skin":{"inputs":{"subsurface_weight":u8::from(enabled)}}},
                "buffers":{"p":{"type":"vec3","data":[[-1,-1,-3],[1,-1,-3],[0,1,-3]]}},
                "geometry":{"triangle":{"attributes":{"POSITION":"p"}}},
                "nodes":[{"type":"camera","name":"camera"},
                    {"type":"mesh","geometry":"triangle","materials":["skin"]}]
            }))?;
            let installed = renderer.install(input, std::path::Path::new("."))?;
            let mut frame = installed.create_frame_slot(&renderer, crate::ReadbackPrecision::Float)?;
            frame.profile = true;
            assert!(frame.can_batch_samples(&installed), "opaque exposure must batch despite the empty image step");
            frame.submit(&installed, &renderer, 0.)?;
            let output = frame.collect(&renderer)?;
            assert!(output.timing.gpu_elapsed_ms.is_finite() && output.timing.gpu_elapsed_ms > 0.);
            assert_eq!(frame.sample_spans.len(), frame.view.sample_count);
            for &mask in &frame.sample_spans {
                assert_eq!(timing::spans(mask).filter(|name| name.starts_with("subsurface_")).count(),
                    if enabled { 2 } else { 0 });
            }
            // Reusing a slot for controls alone switches from a deferred batch to
            // streaming counters; the view and frame passes must agree on its set.
            frame.view.write_color = false;
            frame.submit_outputs(&installed, &renderer, 0., ReadbackOutputs {
                depth_ids: true, ..Default::default()
            })?;
            frame.collect(&renderer)?;
            assert_eq!(frame.sample_spans.len(), 1);
            assert_eq!(timing::spans(frame.sample_spans[0]).filter(|name| name.starts_with("subsurface_")).count(),0);
        }
        Ok(())
    }    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn nonfinite_radiance_refuses_exposure_with_or_without_diagnostics() -> Result<()> {
        let renderer = crate::test_renderer()?;
        for (colour, poisoned) in [(".25", false), ("asfloat(0x7f800000u)", true)] {
            let source = format!("import vork; struct Inputs {{}}; Closure surface(ShadingContext ctx,Inputs inputs) {{ return unlit(float3(ctx.TEXCOORD_0.x>.5 ? {colour} : .25)); }}");
            let input: Scene = vork_scene::v2::from_test_value(serde_json::json!({
                "version":2,
                "render":{"resolution":[16,16],"settings":{"integrator":"raster","samples":2}},
                "buffers":{
                    "p":{"type":"vec3","data":[[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]]},
                    "uv":{"type":"vec2","data":[[0,0],[1,0],[1,1],[0,1]]},
                    "idx":{"type":"u32","data":[0,1,2,0,2,3]}},
                "geometry":{"tile":{"attributes":{"POSITION":"p","TEXCOORD_0":"uv"},"indices":"idx"}},
                "programs":{"poison":{"source":source}},
                "materials":{"poison":{"program":"poison"}},
                "nodes":[{"type":"camera","translation":[0,0,10],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2.0}}},
                    {"type":"mesh","geometry":"tile","materials":["poison"]}]
            }))?;
            let mut installed = renderer.install(input.clone(), Path::new("."))?;
            let mut frame = installed.create_frame_slot(&renderer, Default::default())?;
            let mut previous=None;
            for (case, diagnostics) in [false,true,true,false].into_iter().enumerate() {
                // Refusal invalidates every publication in the installation.
                // Keep valid exposures on one slot to test counter reset, but
                // give each invalid exposure its own installed scene.
                if poisoned && case > 0 {
                    installed = renderer.install(input.clone(), Path::new("."))?;
                    frame = installed.create_frame_slot(&renderer, Default::default())?;
                }
                frame.profile=diagnostics;
                frame.submit(&installed,&renderer,0.)?;
                let collected=frame.collect(&renderer);
                if poisoned {
                    let error=format!("{:#}",collected.err().context("invalid radiance was published")?);
                    assert!(error.contains("non-finite scene-linear radiance"),"{error}");
                    continue;
                }
                let counts=collected?.timing.nonfinite;
                assert_eq!(counts.is_some(),diagnostics);
                if let Some(counts)=counts {
                    assert_eq!(counts.shade>0,poisoned,"{counts:?}");
                    assert_eq!(counts.resolve>0,poisoned,"{counts:?}");
                    if let Some(previous)=previous {assert_eq!(counts,previous,"counts reset between exposures");}
                    previous=Some(counts);
                }
            }
        }
        Ok(())
    }

}

#[cfg(test)]
mod path_sample_drop_tests {
    use super::*;
    #[test]
    #[ignore="requires a GPU and pinned Slang; invokes isolated shader variants"]
    fn nonfinite_paths_mark_once_per_pixel_across_budgets_and_reset() -> Result<()> {
        let mode=std::env::var("VORK_TEST_POISON_PATH_SAMPLE").ok();
        if mode.is_none() {
            for mode in ["one", "all"] {
                let status=std::process::Command::new(std::env::current_exe()?)
                    .args(["--exact","frame::path_sample_drop_tests::nonfinite_paths_mark_once_per_pixel_across_budgets_and_reset","--ignored","--nocapture"])
                    .env("VORK_TEST_POISON_PATH_SAMPLE",mode).env("VORK_GPU_TESTS","1").status()?;
                ensure!(status.success(),"marker child mode={mode} failed");
            }
            return Ok(());
        }
        let r = crate::test_renderer()?;
        for size in [16u32,64] { for samples in [1u32,4] {
            let scene=vork_scene::v2::from_test_value(serde_json::json!({
                "version":2,"render":{"resolution":[size,size],"background":[0.18,0.18,0.18],
                    "settings":{"integrator":"path","samples":samples,"denoiser":null}},"nodes":[{"type":"camera"}]
            }))?;
            let mut installed=r.install(scene,Path::new("."))?;
            let mut slot=installed.create_frame_slot(&r,ReadbackPrecision::Float)?;
            for time in [0.,0.25] {
                let result=installed.render_frame(&r,&mut slot,time)?;
                let count=if mode.as_deref()==Some("all") {size*size} else {1};
                assert_eq!(result.timing.dropped_path_samples, if count==1 {1} else {u64::from(count*samples)});
                assert_eq!(result.timing.invalid_pixels["camera"].count,u64::from(count));
                let pixels=result.linear_rgba.context("missing linear image")?;
                assert_eq!(pixels[0],[1.,0.,1.,1.]);
                if count==1 {ensure!((pixels[1][0]-0.18).abs()<1e-5,"valid neighbour changed: {:?}",pixels[1]);}
                else {assert!(pixels.iter().all(|p|*p==[1.,0.,1.,1.]));}
                assert_eq!(&result.rgba[..4],&[255,0,255,255]);
            }
        }}
        Ok(())
    }

}

#[cfg(test)]
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn dielectric_arena_exhaustion_replays_before_publishing() -> Result<()> {
    let mut r = crate::test_renderer()?;
    r.reference_transport_bounces=Some(64);
    r.reference_transparent_bounces=Some(64);
    let mut input=vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"),include_str!("../tests/presets/analytic-7fefd52a.json"))?;
    input["buffers"].as_object_mut().unwrap().retain(|name,_|name.starts_with("shape_box_"));
    input["geometry"].as_object_mut().unwrap().retain(|name,_|name=="shape_box");
    input["animation"]=serde_json::json!([]);
    input["render"]=serde_json::json!({"resolution":[8,8],"background":[0.4,0.6,0.8],"settings":{"integrator":"path","samples":1,"denoiser":null}});
    input["materials"]=serde_json::json!({"glass":{"inputs":{"transmission_weight":1,"specular_ior":1,"specular_roughness":0}}});
    input["nodes"]=serde_json::json!([
        {"type":"camera","material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":0.001}}},
        {"type":"light","shape":"environment","color":[0.4,0.6,0.8],"energy":1}
    ]);
    for i in 0..33 {
        let size=2.+i as f32*0.04;
        input["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"mesh","name":format!("interior{i}"),"geometry":"shape_box","materials":["glass"],"scale":[size,size,size]}));
    }
    let mut scene=r.install(vork_scene::v2::from_test_value(input)?,std::path::Path::new("."))?;
    let mut frame=scene.create_frame_slot(&r,ReadbackPrecision::Float)?;
    let mut older_slot=scene.create_frame_slot(&r,ReadbackPrecision::Float)?;
    r.queue.write_buffer(&older_slot.materials.as_ref().unwrap().shader_errors,crate::materials::SHADER_DIAGNOSTICS_BYTES,&0u32.to_le_bytes());
    // Publish an empty arena directory without lying about device limits.
    // The first exposure must reject its incomplete interiors and grow storage.
    r.queue.write_buffer(&frame.materials.as_ref().unwrap().shader_errors,crate::materials::SHADER_DIAGNOSTICS_BYTES,&0u32.to_le_bytes());
    let result=scene.render_frame(&r,&mut frame,0.)?;
    assert!(scene.reservations.dielectric_spill>0,"exhausted storage was silently accepted");
    let later=scene.render_frame(&r,&mut older_slot,0.)?;
    for pixels in [result.linear_rgba.unwrap(),later.linear_rgba.unwrap()] {
        for p in pixels {for c in 0..3 {assert!((p[c]-[0.4,0.6,0.8][c]).abs()<0.004,"replayed interior transport: {p:?}");}}
    }
    Ok(())
}

// Receipt append order need not match queue order across frame slots.
fn wait_publication_submissions<T>(submissions: Vec<T>, mut wait: impl FnMut(T) -> Result<()>) -> Result<()> {
    for submission in submissions { wait(submission)?; }
    Ok(())
}

#[cfg(test)]
mod publication_wait_tests {
    use super::wait_publication_submissions;
    #[test]
    fn concurrent_submitters_publish_receipts_in_reverse_queue_order() {
        let receipts = std::sync::Mutex::new(Vec::new());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let receipts = &receipts;
            scope.spawn(move || {
                // Submission 1 pauses before publishing its receipt.
                rx.recv().unwrap();
                receipts.lock().unwrap().push(1);
            });
            scope.spawn(|| {
                receipts.lock().unwrap().push(2);
                tx.send(()).unwrap();
            });
        });
        let receipts = receipts.into_inner().unwrap();
        assert_eq!(receipts, [2, 1]);
        let mut completed = 0;
        wait_publication_submissions(receipts, |index| {
            completed = completed.max(index);
            Ok(())
        }).unwrap();
        assert_eq!(completed, 2, "all mapped receipts must be covered before receiving callbacks");
    }
}
