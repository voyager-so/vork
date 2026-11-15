//! Renderer implementation constants. Scene defaults live in vork_scene::v2 (constants.rs).
pub use vork_slang::registry::{FLAG_DOUBLE_SIDED,FLAG_VERTEX_ALPHA,FLAG_UV_SECOND,FLAG_UV_BASIS,DEFORM_GROUP,MORPH_FLOATS,CLUSTER_GROUP,REFLECTIONS_OFF,REFLECTIONS_RAY,LTC_SIZE,OUTPUT_GPU_COLOR,OUTPUT_COLOR_PLANE,OUTPUT_DEPTH_IDS,OUTPUT_NORMALS,OUTPUT_MOTION,CLUSTER_LIST_HEADER,LIGHT_SHADOW_OFFSET_WORDS,OUTPUT_FLOAT_COLOR,OUTPUT_FLOAT_DEPTH,OUTPUT_FLOAT_NORMALS,OUTPUT_FLOAT_MOTION,FLOAT_COLOR_PLANE,FLOAT_DEPTH_PLANE,FLOAT_NORMAL_PLANE,FLOAT_MOTION_PLANE,MATERIAL_ATTRIBUTE_FLAG,RAY_CAMERA,RAY_SHADOW,RAY_DIFFUSE,RAY_SPECULAR,RAY_TRANSMISSION,RAY_VOLUME,RAY_GI_OCCLUDER,RAY_SHADOW_SOLID,RAY_SLOT_WORDS,RAY_SHADOW_TABLE_WORD,RAY_OPAQUE_SOLID_OWNER,RAY_SHADOW_TABLE_HEADER_WORDS,SHADOW_BOUND_NODE_WORDS,RAY_HIT_RECORD_WORDS,RAY_HIT_BUFFERS,VISIBILITY_SLOTS,VISIBILITY_KEY_WORDS,VISIBILITY_REFLECTION_WORDS,REFLECTION_BSDF_VECTORS,VISIBILITY_DIFFUSION_WORDS,VISIBILITY_REFLECTION_GUIDE_WORDS,REFLECTION_CLASSES,REFLECTION_BATCH_SAMPLES,REFLECTION_MERGED_SLOTS,REFLECTION_DISPATCH_PIXELS,PATH_BOUNDARY_OVERFLOW,PATH_BOUNDARY_AMBIGUOUS,MATERIAL_TAP_WORDS,LIGHT_CONTRIBUTION_WORD,LIGHT_RAY_WORD,LIGHT_RAY_LINK_WORD,LIGHT_RAY_SHADOWED,LIGHT_RAY_SHADOWED_BY_ALL,SAMPLE_LATTICE_RATIO,LIGHT_LINK_WORD,LIGHT_LINK_ALL,MEDIUM_BOUND_WORDS,MEDIUM_BVH_WORDS,PATH_SOLID_WORDS,FLAG_VERTEX_COLOR,PROBE_VEC4S,PROBE_VISIBILITY_SIDE,PROBE_RADIANCE_SIDE,PROBE_RADIANCE_TEXELS,PROBE_RADIANCE_PAIRS,PROBE_RADIANCE_SH_PAIRS,PROBE_DISPATCH_WIDTH,PROBE_GROUP};
/// quality/performance setting: frame slots in flight.
pub const FRAMES_IN_FLIGHT: usize = 3;

/// scheduling: concurrent driver pipeline creations while the hardware encoder attaches.
/// Attach is the install critical path and shares the driver; a burst of creations
/// slowed it 1.5-2x (notes/shaders-migrate.md, spike-specialize diagnosis).
pub(crate) const CRITICAL_PIPELINE_THREADS: usize = 2;
/// Bound parallel command validation/replay without consuming every host core.
pub(crate) const COMMAND_FINISH_THREADS: usize = 4;
/// Bound Vulkan driver command storage before submission; one-ray media batches
/// can otherwise record millions of indirect dispatches into a single encoder.
pub(crate) const VULKAN_RECORDING_PASSES: u32 = 8;
/// Half a 500 ms target: leave 2x timing margin and stay well below the 2 s watchdog.
pub(crate) const SAMPLE_GROUP_GPU_MS: f64 = 250.;
/// Retire replay encoder host state at a bounded recording latency, independent of tick rate.
pub(crate) const SIMULATION_RECORD_MS: f64 = 16.;
/// Pixels per path dispatch: a quarter of measured 8K work (~250 ms on L4).
pub(crate) const PATH_DISPATCH_PIXELS: u64 = 3840 * 2160;
/// Bound retained Metal command buffers even when measured samples are cheap.
pub(crate) const METAL_BATCH_SAMPLES: usize = 64;
/// One bounded pixel/sample block per native compute pass and submission.
pub(crate) const REFLECTION_SEGMENT_DISPATCHES: usize = 1;
/// numerical safety epsilon: positive f64 radiance denominators and nonzero light moments.
pub(crate) const POSITIVE_EPSILON_F64: f64 = 1e-20;
/// physical/derived: Rec.709 linear RGB luminance coefficients for its primaries and D65 white.
pub(crate) const LUMINANCE_REC709: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// IEC 61966-2-1 sRGB decoding curve: cutoff, linear divisor, offset, scale, exponent.
pub(crate) const SRGB_DECODE_CURVE: [f32; 5] = [0.04045, 12.92, 0.055, 1.055, 2.4];
/// Texture-cache codec settings: high-quality UASTC, low search effort for first-use latency.
pub(crate) const TEXTURE_COMPRESSION_QUALITY: i32 = 100;
pub(crate) const TEXTURE_COMPRESSION_EFFORT: i32 = 1;
/// Amortize scoped-thread startup over at least 64K output texels; tiny mips stay serial.
pub(crate) const TEXTURE_MIP_TEXELS_PER_WORKER: usize = 64 * 1024;
/// BC7 and ASTC 4x4 both encode sixteen texels into sixteen bytes.
pub(crate) const TEXTURE_BLOCK_EDGE: u32 = 4;
pub(crate) const TEXTURE_BLOCK_BYTES: u32 = 16;
/// DirectXTex BC7 cooperative search ABI; checked against Slang reflection.
pub(crate) const TEXTURE_BC7_GROUP_SIZE: u32 = 64;
/// Install allocation bound: stream cold BC7 preparation through a two-slot in-flight
/// ring sharing one GPU mip/search workspace.
pub(crate) const TEXTURE_PREPARATION_IN_FLIGHT: usize = 2;
/// Keep SIMD alpha refinement parallel through small edge-heavy mip levels.
pub(crate) const TEXTURE_BC7_ALPHA_BLOCKS_PER_WORKER: usize = 256;
/// derived: maximum finite IEEE-754 binary16 value (2-2^-10)*2^15.
pub(crate) const HALF_FLOAT_MAX: f32 = 65504.;
/// math: first-order real spherical harmonic normalization sqrt(3/(4*pi)), existing precision.
pub(crate) const SH_FIRST_ORDER: f64 = 0.488602512;
/// quality/performance setting: existing specular environment map width before mip reduction.
pub(crate) const ENVIRONMENT_WIDTH: u32 = 256;
/// quality/performance setting: existing diffuse convolution resolution.
pub(crate) const DIFFUSE_SIZE: [u32; 2] = [32, 16];
/// quality/performance setting: existing split-sum BRDF lookup resolution.
pub(crate) use vork_slang::registry::{BRDF_SIZE, BRDF_HEIGHT};

/// allocation bound: amortize static-mesh resource tracking while bounding upload staging to 64 MiB per page.
pub(crate) const STATIC_GEOMETRY_PAGE_BYTES: u64 = 64 * 1024 * 1024;
/// quality/performance setting: screen tile width/height; limits binning cost at 1080p.
pub(crate) const CLUSTER_TILE_PIXELS: u32 = 64;
/// quality/performance setting: logarithmic slices cover near/far with bounded mask storage.
pub(crate) const CLUSTER_DEPTH_SLICES: u32 = 24;

/// Reject coincident orientation targets in squared metres.
pub(crate) const ORIENTATION_EPSILON: f32 = 1e-10;
/// Switch the up axis near parallel forward/up vectors to keep look-at stable.
pub(crate) const ORIENTATION_UP_PARALLEL: f32 = 0.999;

/// One full-width source reference per OpenPBR input and glTF occlusion, vec4-aligned.
pub(crate) const MATERIAL_INPUT_WORDS: usize = (vork_scene::v2::OPENPBR.len()+1).next_multiple_of(4);

pub use vork_slang::registry::MATERIAL_TAP_INDEX_MASK;

/// glTF KHR_materials_clearcoat: the coat's fixed index of refraction (glTF spec).
pub(crate) const GLTF_CLEARCOAT_IOR: f32 = 1.5;

pub(crate) const REFLECTIONS_ENVIRONMENT: f32 = 1.;

/// Named timestamp registry: each span owns a begin/end pair. Query storage
/// and lookup derive from this list; passes never assign numeric query slots.
pub(crate) const GPU_SPANS: &[&str] = &[
    "images",
    "image inputs",
    "fibre",
    "depth",
    "visibility",
    "shade",
    "material classify",
    "material compose",
    "refraction",
    "forward",
    "subsurface_diffuse",
    "subsurface_compose",
    "resolve",
    "deform",
    "acceleration structures",
    "ao",
    "gi demand",
    "gi pilots",
    "gi beauty",
    "gi reconstruction",
    "path camera",
    "path",
    "transparency",
    "clusters",
    "reflections",
    "controls",
    "compose",
    "post_source",
    "compositor",
    "probes",
    "exposure",
    "reconstruction",
];
/// Packed vertex UV slot sentinel: this channel uses the shared zero value.
pub(crate) const VERTEX_DEFAULT_UV: u8 = u8::MAX;
/// Output bound: zero is background; exact grayscale16 segmentation IDs.
pub(crate) const MAX_OBJECT_ID: usize = u16::MAX as usize;

/// Allocation/performance bound: importance proposal <= 2K per axis (32 MiB CDF), independent of HDR size.
pub(crate) const ENVIRONMENT_PROPOSAL_MAX: u32 = 2048;

/// Retained specular convolution mip count; includes the mirror level.
pub(crate) const ENVIRONMENT_LEVELS: u32 = 6;

/// PCI vendor ID used to enable NVIDIA Vulkan/CUDA encoder and footage interop.
pub(crate) const NVIDIA_VENDOR_ID: u32 = 0x10de;

// Canonical local lattice, independent of device memory pressure.
pub(crate) const PROBE_SPACING: f32 = 1.;
/// Initial angular cell size for retries under verified native memory pressure (radians).
pub(crate) const PROBE_ANGULAR_SPACING: f32 = 1. / 4.;

/// Fixed store ceiling (16 GiB), further limited by filesystem headroom.
pub(crate) const DISK_CACHE_BUDGET: u64 = 16 * 1024 * 1024 * 1024;
/// Leave half of reclaimable filesystem space available to other applications.
pub(crate) const DISK_CACHE_HEADROOM_DIVISOR: u64 = 2;

/// One GPU flight plus two consumer flights share one output ring; preserves CPU write overlap.
pub(crate) const OUTPUT_RING_DEPTH: usize = 3;

/// Optional raw EXR planes use the same output buffer after the four packed planes.
pub(crate) const PACKED_OUTPUT_PLANES: u32 = 4;
pub(crate) const FLOAT_OUTPUT_PLANES: u32 = 14;

/// One f32 RGBA sum per pixel, independent of output plane count.
pub(crate) const ACCUMULATION_PIXEL_BYTES: u64 = 16;
/// Packed layout plane indices, shared by transfer and unpack.
pub(crate) const PACKED_COLOR_PLANE: usize = 0;
pub(crate) const PACKED_DEPTH_IDS_PLANE: usize = 1;
pub(crate) const PACKED_NORMAL_PLANE: usize = 2;
pub(crate) const PACKED_MOTION_PLANE: usize = 3;

/// An independent low-discrepancy dimension for thin-lens sample azimuth.
pub(crate) const LENS_AZIMUTH_BASE: usize = 3;
/// Independent low-discrepancy dimension for thin-lens radius squared.
pub(crate) const LENS_RADIUS_BASE: usize = 5;

pub(crate) const RAY_VISIBILITY_ALL: u32 =
    RAY_CAMERA | RAY_SHADOW | RAY_DIFFUSE | RAY_SPECULAR | RAY_TRANSMISSION | RAY_VOLUME;

const _: () = {
    assert!(LIGHT_RAY_WORD < LIGHT_SHADOW_OFFSET_WORDS && LIGHT_RAY_LINK_WORD < LIGHT_SHADOW_OFFSET_WORDS);
    assert!(LIGHT_CONTRIBUTION_WORD < LIGHT_SHADOW_OFFSET_WORDS);
};
/// quality/performance setting: anisotropic filtering for linear program images
/// (spec §2 Image: linear is trilinear plus anisotropic).
pub(crate) const PROGRAM_IMAGE_ANISOTROPY: u16 = 16;

/// Hardware ray instances keep the 24-bit custom index of the object record.
pub(crate) const RAY_MAX_INSTANCE_INDEX: u32 = (1 << 24) - 1;
/// Allocation bound: static BLAS builds share one submission per this many
/// triangles, keeping build scratch memory small on unified-memory devices.
pub(crate) const RAY_BLAS_BATCH_TRIANGLES: u64 = 4 << 20;

/// The explicit msaa4 integrator option retains four raster coverage samples.
pub(crate) const RASTER_MSAA_SAMPLES: u32 = crate::forward_variant::RASTER_MSAA_SAMPLES;

/// physical unit: the photometric efficacy that converts glTF lux/candela to W/m² and W.
pub(crate) const LUMENS_PER_WATT: f32 = 683.;

/// Scene-program compute ABI workgroup width, shared with the Slang compiler.
pub(crate) const PROGRAM_WORKGROUP: u32 = 64;

/// Square workgroup for publishing linear render-image pixels to the material
/// table, and for every compositor pass.
pub(crate) const IMAGE_WORKGROUP_SIZE: u32 = 8;

/// Blender 5.2.1 Cycles defaults: reflective and refractive caustics after
/// diffuse scattering are on; adaptive noise is controlled by the path settings.
pub(crate) const PATH_CAUSTICS: u32 = 0b11;
/// Blender 5.2.1 LightProbeVolume defaults: capture distance (metres), normal
/// and view bias (smallest cell sides), facing bias and validity threshold.
pub(crate) const PROBE_CAPTURE_DISTANCE: f32 = 20.;
pub(crate) const PROBE_NORMAL_BIAS: f32 = 0.3;
pub(crate) const PROBE_VIEW_BIAS: f32 = 0.;
pub(crate) const PROBE_FACING_BIAS: f32 = 0.5;
pub(crate) const PROBE_VALIDITY_THRESHOLD: f32 = 0.4;

/// Counts warmed at build time; larger device-supported layouts compile through the shared cache.
pub(crate) const PRECOMPILED_ATTRIBUTE_SLOTS: usize = 8;

/// Compile budget: at most sixteen specialized (vertex layout, capability) keys per scene.
/// Larger sets share their conservative union; excess mandatory layouts use generic shading.
pub(crate) const SCENE_SHADING_VARIANTS: usize = 16;
/// Environment lighting quadrature: 2K lat-long resolves Blender's default solar disc before the existing HDR convolution. Larger bound images raise it.
pub(crate) const ENVIRONMENT_PROGRAM_SIZE: [u32; 2] = [2048, 1024];
/// GPU ABI: per-pixel adaptive path moment (mean, variance, two settled slots,
/// stopped flag, padding), matching PathMoment in path.slang.
pub(crate) const PATH_MOMENT_BYTES: u64 = 48;

/// IES angular-image sampling resolution: half-degree horizontal and vertical bins.
pub(crate) const IES_IMAGE_SIZE: [u32; 2] = [720, 360];

/// Cycles IES Texture: 4*pi divided by D65 luminous efficacy (177.83 lm/W).
pub(crate) const IES_CANDELA_TO_CYCLES_WATTS: f32 = 4. * std::f32::consts::PI / 177.83;

/// Keep a twelve-triangle boundary in one packet; larger meshes split spatially.
pub(crate) const MEDIUM_BVH_LEAF_TRIANGLES: usize = 12;

/// Small exit-tree packets amortize bounds traversal while bounding leaf work.
pub(crate) const EXIT_LEAF_TRIANGLES: usize = 4;

/// Blender camera object focus clamps the absolute view-axis depth above zero (metres).
pub(crate) const MIN_FOCUS_DISTANCE: f32 = 1e-5;

// Shared geometry can fit while its expanded draw work is effectively unbounded.
// This conservative admission ceiling is separate from runtime submission slicing.
pub(crate) const MAX_INSTALL_TRIANGLE_INSTANCES: u64 = u32::MAX as u64;

/// performance setting: retained camera evaluations per decoupled shading band, the same on
/// every device. Each band adds a classify/evaluate/compose round with its pass barriers, so
/// the target is the smallest band whose overhead stays in the noise; larger bands only cost
/// memory. Chosen from an L4 sweep, sizzle-city at 1080p/64 spp, warm GPU frame: 276.5 ms at
/// 67.1M and 33.5M records per band, 277.7 ms at 16.8M (shade +4%).
pub(crate) const EVALUATION_BAND_RECORDS: u64 = 1 << 25;
/// allocation bound: a band reserves its even share of the per-sample content estimate plus
/// 1/N, because rows hold uneven content and an overflow redoes the whole frame. It only spares
/// the first frame that redo; replay sizes later frames from reported demand. Chosen to cover
/// the largest band's measured excess over an even share (Classroom: 1.4-9.4% at 5-45 bands).
pub(crate) const EVALUATION_ESTIMATE_HEADROOM: u64 = 8;
/// Camera records share one quarter of stable startup headroom, leaving room
/// for retained images, scene resources, denoising and downstream passes.
pub(crate) const CAMERA_RECORD_MEMORY_SHARE: u64 = 4;
/// allocation bound: one view's band of evaluations takes at most 1/N of device memory, so a
/// small device splits the frame into more bands instead of exhausting its memory.
pub(crate) const EVALUATION_MEMORY_SHARE: u64 = 8;

/// Dense-field range reduction lanes; strided spans keep dispatches within backend limits.
pub(crate) const DENSE_REDUCTION_GROUP: u32 = 64;
/// Two adjacent displayed frames cover the usual shuffled shutter interval without reopening the decoder.
pub(crate) const FOOTAGE_DECODE_FRAMES: usize = 2;
// Fixed reference transport policy, following Cycles 5.2.1 defaults.
pub const PATH_MAX_BOUNCES: u32 = 12;
pub const PATH_DIFFUSE_BOUNCES: u32 = 4;
pub const PATH_GLOSSY_BOUNCES: u32 = 4;
pub const PATH_TRANSMISSION_BOUNCES: u32 = 12;
pub const PATH_TRANSPARENT_MAX_BOUNCES: u32 = 8;
pub const PATH_SAMPLE_CLAMP_DIRECT: f32 = 0.;
pub const PATH_SAMPLE_CLAMP_INDIRECT: f32 = 10.;
pub const PATH_LIGHT_SAMPLING_THRESHOLD: f32 = 0.01;
pub const PATH_USE_ADAPTIVE_SAMPLING: bool = true;
pub const PATH_ADAPTIVE_THRESHOLD: f32 = 0.01;
pub const PATH_ADAPTIVE_MIN_SAMPLES: u32 = 0;
pub const PATH_USE_ANIMATED_SEED: bool = false;

// Raster transport policy.
/// Resolution of the retained physical-fibre capture field.
pub const FIBRE_CAPTURE_RESOLUTION: u32 = 2048;
pub const RASTER_TRANSMISSION_BOUNCES: u32 = 12;

/// Maximum adaptive diffuse rays per visibility sample; independent pilots choose
/// this or one eighth of this budget before beauty sampling.
pub(crate) const GI_RADIUS_M: f32 = 0.5;
pub(crate) const GI_RAYS_PER_SAMPLE: u32 = 16;
pub(crate) const GI_RESOLUTION_DIVISOR: u32 = 2;

/// Bound mapped-at-creation uploads independently of a driver's large binding limit.
/// Named buffers already span pages; smaller allocations avoid giant mapped heaps.
pub(crate) const SCENE_BUFFER_PAGE_BYTES: usize = 256 * 1024 * 1024;

/// Optional in-flight cache captures and serialized publications, independent of disk eviction.
pub(crate) const CACHE_PERSISTENCE_BYTES: usize = 16 * 1024 * 1024;

/// Fixed SH JSON, image vector headers and reservation bookkeeping per optional HDR capture.
pub(crate) const CACHE_CAPTURE_METADATA_BYTES: usize = 4096;

/// Median window rejects an isolated scheduling interruption during replay.
pub(crate) const SIMULATION_GROUP_OBSERVATIONS: usize = 3;
/// Baseline windows between larger-group trials after a failed trial.
pub(crate) const SIMULATION_GROUP_RETRY: u32 = 8;

/// GiQuery in gi_ray.slang: six aligned float4 records per query.
pub(crate) const GI_QUERY_BYTES: u64 = 96;
