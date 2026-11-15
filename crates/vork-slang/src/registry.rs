//! Numeric CPU/shader ABI registry. Values are evaluated by Rust for both shader languages.
macro_rules! registry {
    ($( $(#[$meta:meta])* $name:ident : $rust:ty => $shader:ident, $wgsl:literal = $value:expr; )*) => {
        $( $(#[$meta])* pub const $name: $rust = $value;)*
        pub fn wgsl() -> String {
            let mut output = String::from("// Generated from vork-slang/src/registry.rs.\n");
            $(
                let value = if stringify!($shader) == "f32" {
                    format!("{:?}", $name as f32)
                } else { format!("{}u", $name as u32) };
                output.push_str(&format!("const {}:{}={};\n", $wgsl, stringify!($shader), value));
            )*
            output
        }
        pub fn slang() -> String {
            let mut output = String::from("// Generated from vork-slang/src/registry.rs.\n");
            $(
                let (ty, value) = if stringify!($shader) == "f32" {
                    ("float", format!("{:?}", $name as f32))
                } else { ("uint", format!("{}u", $name as u32)) };
                output.push_str(&format!("public static const {ty} {}={value};\n", $wgsl));
                if $wgsl != stringify!($name) {
                    output.push_str(&format!("public static const {ty} {}={};\n", stringify!($name), $wgsl));
                }
            )*
            output
        }
    };
}
registry! {
    // White-preserving XYZ basis for OIDN: inverse of vork_spectrum.slang's
    // calibrated spectral matching matrix, with rows normalized to unit white.
    // OIDN clamps negative HDR input; this basis preserves signed spectral RGB.
    DENOISE_TO_XYZ_00: f32 => f32, "DENOISE_TO_XYZ_00" = 0.4432179067;
    DENOISE_TO_XYZ_01: f32 => f32, "DENOISE_TO_XYZ_01" = 0.3639035846;
    DENOISE_TO_XYZ_02: f32 => f32, "DENOISE_TO_XYZ_02" = 0.1928785087;
    DENOISE_TO_XYZ_10: f32 => f32, "DENOISE_TO_XYZ_10" = 0.2210817506;
    DENOISE_TO_XYZ_11: f32 => f32, "DENOISE_TO_XYZ_11" = 0.7042597945;
    DENOISE_TO_XYZ_12: f32 => f32, "DENOISE_TO_XYZ_12" = 0.0746584549;
    DENOISE_TO_XYZ_20: f32 => f32, "DENOISE_TO_XYZ_20" = 0.0179135722;
    DENOISE_TO_XYZ_21: f32 => f32, "DENOISE_TO_XYZ_21" = 0.1047636634;
    DENOISE_TO_XYZ_22: f32 => f32, "DENOISE_TO_XYZ_22" = 0.8773227644;
    DENOISE_FROM_XYZ_00: f32 => f32, "DENOISE_FROM_XYZ_00" = 3.0152417703;
    DENOISE_FROM_XYZ_01: f32 => f32, "DENOISE_FROM_XYZ_01" = -1.4781299363;
    DENOISE_FROM_XYZ_02: f32 => f32, "DENOISE_FROM_XYZ_02" = -0.5371118340;
    DENOISE_FROM_XYZ_10: f32 => f32, "DENOISE_FROM_XYZ_10" = -0.9520724740;
    DENOISE_FROM_XYZ_11: f32 => f32, "DENOISE_FROM_XYZ_11" = 1.9048602038;
    DENOISE_FROM_XYZ_12: f32 => f32, "DENOISE_FROM_XYZ_12" = 0.0472122702;
    DENOISE_FROM_XYZ_20: f32 => f32, "DENOISE_FROM_XYZ_20" = 0.0521231760;
    DENOISE_FROM_XYZ_21: f32 => f32, "DENOISE_FROM_XYZ_21" = -0.1972837737;
    DENOISE_FROM_XYZ_22: f32 => f32, "DENOISE_FROM_XYZ_22" = 1.1451605977;
    /// Match the reflected response LUT grazing floor for transmitted guide albedo.
    DENOISE_GUIDE_COSINE_MIN: f32 => f32, "DENOISE_GUIDE_COSINE_MIN" = 0.0001;
    DENOISE_GROUP: u32 => u32, "DENOISE_GROUP" = 64;
    // Compiler effect-output slots for per-closure dispersion and wavelength
    // dependency proofs, independent of selected-lobe storage.
    // Surface emission alone defines sampled source support. Lit is authored response.
    EFFECT_SURFACE_WEIGHTED_IOR: u32 => u32, "REG_EFFECT_SURFACE_WEIGHTED_IOR" = 26u32;
    EFFECT_LIT_WEIGHTED_IOR: u32 => u32, "REG_EFFECT_LIT_WEIGHTED_IOR" = 27u32;
    EFFECT_SURFACE_EMISSION: u32 => u32, "REG_EFFECT_SURFACE_EMISSION" = 23u32;
    EFFECT_SURFACE_COAT_ANISOTROPY: u32 => u32, "REG_EFFECT_SURFACE_COAT_ANISOTROPY" = 24u32;
    EFFECT_LIT_COAT_ANISOTROPY: u32 => u32, "REG_EFFECT_LIT_COAT_ANISOTROPY" = 25u32;
    EFFECT_SURFACE_DISPERSION: u32 => u32, "REG_EFFECT_SURFACE_DISPERSION" = 19u32;
    EFFECT_LIT_DISPERSION: u32 => u32, "REG_EFFECT_LIT_DISPERSION" = 20u32;
    EFFECT_SURFACE_WAVELENGTH: u32 => u32, "REG_EFFECT_SURFACE_WAVELENGTH" = 21u32;
    EFFECT_LIT_WAVELENGTH: u32 => u32, "REG_EFFECT_LIT_WAVELENGTH" = 22u32;
    /// Frame.temporal.w: path transport owns the accumulated denoising guides.
    DENOISE_GUIDES_PATH: u32 => u32, "REG_DENOISE_GUIDES_PATH" = 2;
    INDIRECT_ID_MASK: u32 => u32, "REG_INDIRECT_ID_MASK" = 65535u32;
    /// Defensive mesh-source mixture: equal budget for the emission guide and
    /// radiance-independent source support. This controls variance, not energy.
    MESH_SOURCE_UNIFORM_MIX: f32 => f32, "REG_MESH_SOURCE_UNIFORM_MIX" = 0.5;
    EXR_BACKGROUND_DEPTH: f32 => f32, "REG_EXR_BACKGROUND_DEPTH" = 1e10;
    // Diagnostics word 0 enables checking; low/high pairs count bad output evaluations.
    NONFINITE_WORDS: usize => u32, "REG_NONFINITE_WORDS" = 5;
    NONFINITE_SHADE: usize => u32, "REG_NONFINITE_SHADE" = 1;
    NONFINITE_RESOLVE: usize => u32, "REG_NONFINITE_RESOLVE" = 3;
    // Exposure report over the 8-bit display image: low/high word pairs count
    // Alpha-weighted clipped, near-black and quantized luma sums, then coverage (0..255 per pixel).
    EXPOSURE_WORDS: usize => u32, "REG_EXPOSURE_WORDS" = 10;
    EXPOSURE_LUMA_SCALE: f32 => f32, "REG_EXPOSURE_LUMA_SCALE" = 65535.;
    // ARRI false-colour bands: white clipping 99-100% (here: any display channel),
    // black clipping 0-2.5% (display Rec.709 luma).
    EXPOSURE_CLIPPED: f32 => f32, "REG_EXPOSURE_CLIPPED" = 0.99;
    EXPOSURE_NEAR_BLACK: f32 => f32, "REG_EXPOSURE_NEAR_BLACK" = 0.025;
    // Forward error for seven floating-point affine operations.
    FP32_GAMMA7: f32 => f32, "REG_FP32_GAMMA7" = (7. * (f32::EPSILON * 0.5)) / (1. - 7. * (f32::EPSILON * 0.5));
    // Full filtered-shutter quadrature, shared by host sequence setup and shader.
    MOTION_TAPS: u32 => u32, "REG_MOTION_TAPS" = 16;
    // Four cumulative crossing counts packed in one RGBA transport target.
    FIBRE_LAYERS: usize => u32, "REG_FIBRE_LAYERS" = 4;
    // Exit bounds and leaf ranges occupy two uint4s; triangle attributes are separate.
    EXIT_NODE_VEC4S: usize => u32, "REG_EXIT_NODE_VEC4S" = 2;
    EXIT_TRIANGLE_VEC4S: usize => u32, "REG_EXIT_TRIANGLE_VEC4S" = 3;
    // Cell roots: bracket width <= 2^-32 cell segments, or float midpoint stalls.
    IMPLICIT_BLOCK: u32 => u32, "REG_IMPLICIT_BLOCK" = 8;
    IMPLICIT_ROOT_BISECTIONS: u32 => u32, "REG_IMPLICIT_ROOT_BISECTIONS" = 32;
    // Camera compaction: one bit per lane, encoded in two uint4 masks per block.
    CAMERA_CULL_GROUP: u32 => u32, "REG_CAMERA_CULL_GROUP" = 256;
    // Range header plus two mask vectors; draw range plus live-count metadata.
    CAMERA_CULL_BLOCK_VEC4S: usize => u32, "REG_CAMERA_CULL_BLOCK_VEC4S" = 3;
    CAMERA_CULL_DRAW_VEC4S: usize => u32, "REG_CAMERA_CULL_DRAW_VEC4S" = 2;
    // Conservative roundoff margin shared by CPU singleton and GPU interval tests.
    CAMERA_CULL_ERROR: f32 => f32, "REG_CAMERA_CULL_ERROR" = 0.00001;
    // One original-object index beyond the complete mesh attribute locations 0..16 (including the endpoint position).
    CAMERA_INSTANCE_LOCATION: u32 => u32, "REG_CAMERA_INSTANCE_LOCATION" = 17;
    // A decoupled visibility pass's parameter block, one constant per draw (passes/decoupled.rs).
    VISIBILITY_LENS_LOCATION: u32 => u32, "REG_VISIBILITY_LENS_LOCATION" = 19;
    VISIBILITY_PASS_LOCATION: u32 => u32, "REG_VISIBILITY_PASS_LOCATION" = 18;
    // The transparent fragment lists' header and one record (decoupled_slots.slang FragmentLayout).
    FRAGMENT_HEADER_WORDS: u32 => u32, "REG_FRAGMENT_HEADER_WORDS" = 5;
    // Full-roughness screen refraction footprint in pixels: the opaque image
    // pyramid's levels reach it (passes/decoupled.rs refraction_levels).
    TRANSMISSION_BLUR_DIAMETER: f32 => f32, "REG_TRANSMISSION_BLUR_DIAMETER" = 48.;
    FRAGMENT_RECORD_WORDS: u32 => u32, "REG_FRAGMENT_RECORD_WORDS" = 4;
    // Visibility passes per merge, a divisor of 32: a batch's samples share one
    // coverage mask word (passes/decoupled.rs).
    MERGE_BATCH: u32 => u32, "REG_MERGE_BATCH" = 8;
    // Match REG_DEFORM_GROUP: one thread per vertex and two pose rows per dispatch.
    DEFORM_GROUP: u32 => u32, "REG_DEFORM_GROUP" = 64;
    // Position, normal and tangent XYZ deltas per morph vertex; matches REG_MORPH_FLOATS.
    MORPH_FLOATS: usize => u32, "REG_MORPH_FLOATS" = 9;
    // GPU dispatch ABI: must match REG_CLUSTER_GROUP in constants.wgsl.
    CLUSTER_GROUP: u32 => u32, "REG_CLUSTER_GROUP" = 4;
    // Froxel scan dispatch ABI: adjacent columns and ordered depth blocks per group.
    // Froxel storage ABI: aligned float4 source and transmission plane elements.
    /// Compactified world lattice for per-light medium visibility.
    // GPU ABI: render reflection technique codes, matching constants.wgsl.
    REFLECTIONS_OFF: f32 => f32, "REG_REFLECTIONS_OFF" = 0.;
    REFLECTIONS_RAY: f32 => f32, "REG_REFLECTIONS_RAY" = 2.;
    // Resolution of the published Heitz GGX LTC fit and horizon-clipping tables.
    LTC_SIZE: u32 => f32, "REG_LTC_SIZE" = 64;
    // GPU-published view capacity feedback in the existing shader diagnostics binding.
    CAMERA_VISIBILITY_OVERFLOW_WORD: u32 => u32, "REG_CAMERA_VISIBILITY_OVERFLOW_WORD" = 4;
    CAMERA_FRAGMENT_OVERFLOW_WORD: u32 => u32, "REG_CAMERA_FRAGMENT_OVERFLOW_WORD" = 5;
    TRACE_RAYS_WORD: u32 => u32, "REG_TRACE_RAYS_WORD" = 6;
    // Offline exposure safety ceiling, independent of future real-time profile admission.
    TRACE_RAY_LIMIT: u32 => u32, "REG_TRACE_RAY_LIMIT" = 0x7fffffff;
    TRACE_TRAVERSAL_LIMIT: u32 => u32, "REG_TRACE_TRAVERSAL_LIMIT" = 65536;
    DROPPED_LIGHT_SAMPLES_WORD: u32 => u32, "REG_DROPPED_LIGHT_SAMPLES_WORD" = 8;
    DROPPED_PATH_SAMPLES_WORD: u32 => u32, "REG_DROPPED_PATH_SAMPLES_WORD" = 10;
    // RGBA8, low byte first: the visible marker for missing images and invalid samples.
    DIAGNOSTIC_MAGENTA_RGBA: u32 => u32, "REG_DIAGNOSTIC_MAGENTA_RGBA" = 0xffff00ff;
    TERMINATED_MEDIUM_PATHS_WORD: u32 => u32, "REG_TERMINATED_MEDIUM_PATHS_WORD" = 12;
    // Sum of absolute accumulated camera-path RGB radiance discarded by fog watchdogs.
    DISCARDED_MEDIUM_RADIANCE_WORD: u32 => u32, "REG_DISCARDED_MEDIUM_RADIANCE_WORD" = 14;
    // Linked dielectric interior: next, nine descriptor words, transaction tag, exit distance.
    PATH_BOUNDARY_NODE_WORDS: u32 => u32, "REG_PATH_BOUNDARY_NODE_WORDS" = 14;
    SHADER_DIAGNOSTICS_WORDS: u32 => u32, "REG_SHADER_DIAGNOSTICS_WORDS" = 20;
    // Requested output stores in the frame projection's reserved mask component.
    OUTPUT_GPU_COLOR: u32 => u32, "REG_OUTPUT_GPU_COLOR" = 1;
    OUTPUT_COLOR_PLANE: u32 => u32, "REG_OUTPUT_COLOR_PLANE" = 2;
    OUTPUT_DEPTH_IDS: u32 => u32, "REG_OUTPUT_DEPTH_IDS" = 4;
    OUTPUT_NORMALS: u32 => u32, "REG_OUTPUT_NORMALS" = 8;
    OUTPUT_MOTION: u32 => u32, "REG_OUTPUT_MOTION" = 16;
    // One count precedes each compact cluster list; paired with REG_CLUSTER_LIST_HEADER.
    CLUSTER_LIST_HEADER: u32 => u32, "REG_CLUSTER_LIST_HEADER" = 1;
    // Contribution and caster links, ray permissions, volume slot and alignment.
    LIGHT_SHADOW_OFFSET_WORDS: usize => u32, "REG_LIGHT_SHADOW_OFFSET_WORDS" = 8;
    // Packed volume visibility slot; remaining words preserve float4 alignment.
    LIGHT_VOLUME_SLOT_WORD: usize => u32, "REG_LIGHT_VOLUME_SLOT_WORD" = 4;
    // Enable unquantized pre-exposure colour and geometric control stores.
    OUTPUT_FLOAT_COLOR: u32 => u32, "REG_OUTPUT_FLOAT_COLOR" = 32;
    OUTPUT_FLOAT_DEPTH: u32 => u32, "REG_OUTPUT_FLOAT_DEPTH" = 64;
    OUTPUT_FLOAT_NORMALS: u32 => u32, "REG_OUTPUT_FLOAT_NORMALS" = 128;
    OUTPUT_FLOAT_MOTION: u32 => u32, "REG_OUTPUT_FLOAT_MOTION" = 256;
    // Producer requirements for declared compositor inputs; allocate no output planes.
    COMPOSITOR_IDS: u32 => u32, "REG_COMPOSITOR_IDS" = 512;
    COMPOSITOR_MOTION: u32 => u32, "REG_COMPOSITOR_MOTION" = 1024;
    // Display stage also stores its 16-bit RGBA packing in the display staging planes.
    OUTPUT_COLOR_PLANE16: u32 => u32, "REG_OUTPUT_COLOR_PLANE16" = 2048;
    FLOAT_COLOR_PLANE: usize => u32, "REG_FLOAT_COLOR_PLANE" = 4;
    FLOAT_DEPTH_PLANE: usize => u32, "REG_FLOAT_DEPTH_PLANE" = 8;
    FLOAT_NORMAL_PLANE: usize => u32, "REG_FLOAT_NORMAL_PLANE" = 9;
    FLOAT_MOTION_PLANE: usize => u32, "REG_FLOAT_MOTION_PLANE" = 12;
    // Object ray visibility, matching the six v2 node visibility fields.
    // Tap flags high bit marks a named vertex attribute, not an image; bits 8..15 index it.
    MATERIAL_ATTRIBUTE_FLAG: u32 => u32, "REG_MATERIAL_ATTRIBUTE_FLAG" = 1 << 31;
    // High object visibility bits stay outside the TLAS's eight-bit ray mask.
    OBJECT_HOLDOUT: u32 => u32, "REG_OBJECT_HOLDOUT" = 256;
    /// Camera carrier depth is synthetic; only the exact field root supplies depth.
    OBJECT_IMPLICIT: u32 => u32, "REG_OBJECT_IMPLICIT" = 1024;
    OBJECT_CATCHER: u32 => u32, "REG_OBJECT_CATCHER" = 512;
    RAY_CAMERA: u32 => u32, "REG_RAY_CAMERA" = 1;
    RAY_SHADOW: u32 => u32, "REG_RAY_SHADOW" = 2;
    RAY_DIFFUSE: u32 => u32, "REG_RAY_DIFFUSE" = 4;
    RAY_SPECULAR: u32 => u32, "REG_RAY_SPECULAR" = 8;
    RAY_TRANSMISSION: u32 => u32, "REG_RAY_TRANSMISSION" = 16;
    RAY_VOLUME: u32 => u32, "REG_RAY_VOLUME" = 32;
    // TLAS-only instance mask bit: diffuse-visible surfaces that are neither blended nor
    // transmissive, the occluders and bounce sources of ray GI (as the screen hemisphere sees
    // opaque depth only). Above the visibility bits.
    // TLAS-only mask for solid-shadow containment discovery.
    RAY_SHADOW_SOLID: u32 => u32, "REG_RAY_SHADOW_SOLID" = 64;
    RAY_GI_OCCLUDER: u32 => u32, "REG_RAY_GI_OCCLUDER" = 128;
    // Ray table words per TLAS instance: alpha corner start, material record and
    // hit record (REG_RAY_NO_ALPHA without one).
    RAY_SLOT_WORDS: usize => u32, "REG_RAY_SLOT_WORDS" = 4;
    // Fourth TLAS header word: optional absorbing-solid table, or NO_ALPHA.
    RAY_SHADOW_TABLE_WORD: usize => u32, "REG_RAY_SHADOW_TABLE_WORD" = 3;
    // A nontransmitting material group owned by a possible dielectric solid.
    // Real solid IDs fit the 24-bit instance budget; this is not a boundary ID.
    RAY_OPAQUE_SOLID_OWNER: u32 => u32, "REG_RAY_OPAQUE_SOLID_OWNER" = 0xfffffffe;
    // Unbounded count, solid bounds, instance identities, index buffer slot/node count; then unbounded IDs.
    RAY_SHADOW_TABLE_HEADER_WORDS: usize => u32, "REG_RAY_SHADOW_TABLE_HEADER_WORDS" = 5;
    /// Threaded bounds node: min/escape, max/solid, local-record range.
    SHADOW_BOUND_NODE_WORDS: usize => u32, "REG_SHADOW_BOUND_NODE_WORDS" = 10;
    // Words per ray-hit record (rt::hit_record): position, index and attribute
    // buffers with their bases and strides, and the layout's field offsets.
    RAY_HIT_RECORD_WORDS: usize => u32, "REG_RAY_HIT_RECORD_WORDS" = 11;
    // Hit buffer array size: geometry pages, per-draw streams and deformed
    // ranges a scene's ray hits read (partially bound).
    RAY_HIT_BUFFERS: u32 => u32, "REG_RAY_HIT_BUFFERS" = 4096;
    // Decoupled visibility slots per pixel (passes/decoupled.rs): distinct triangles
    // a pixel keeps densely; further keys go to its overflow list (decoupled_slots.slang).
    /// Centroid connections retain one coverage word per record and dispatch.
    CAMERA_INDIRECT_BLOCK: u32 => u32, "REG_CAMERA_INDIRECT_BLOCK" = 4096;
    /// Single-sample records use the same connection budget as full 32-bit words.
    CAMERA_INDIRECT_SAMPLE_BLOCK: u32 => u32, "REG_CAMERA_INDIRECT_SAMPLE_BLOCK" = CAMERA_INDIRECT_BLOCK * 32;
    /// Evaluation count/capacity, centroid count, connection count, sampled count;
    /// pad the shared queue header to two uint4s before its work records.
    CAMERA_INDIRECT_HEADER_WORDS: u32 => u32, "REG_CAMERA_INDIRECT_HEADER_WORDS" = 8;
    /// Existing camera indirect kernel workgroup width, also used by its argument writer.
    CAMERA_INDIRECT_GROUP: u32 => u32, "REG_CAMERA_INDIRECT_GROUP" = 64;
    VISIBILITY_SLOTS: u32 => u32, "REG_VISIBILITY_SLOTS" = 4;
    // Key words per visibility slot: draw + 1, object, primitive.
    VISIBILITY_KEY_WORDS: u32 => u32, "REG_VISIBILITY_KEY_WORDS" = 2;
    // Camera bin metadata: first result, sampled flag, count, centroid/lens sums and footprint.
    CAMERA_EVALUATION_METADATA_WORDS: u32 => u32, "REG_CAMERA_EVALUATION_METADATA_WORDS" = 10;
    // Opaque table work maps source visibility records to compact band-local results.
    CAMERA_IMPLICIT_SLOTS: u32 => u32, "REG_CAMERA_IMPLICIT_SLOTS" = 64;
    CAMERA_PROBE_WORK: u32 => u32, "REG_CAMERA_PROBE_WORK" = 128;
    CAMERA_PROBE_LIT: u32 => u32, "REG_CAMERA_PROBE_LIT" = 256;
    /// This dispatch reads single-sample requests from the queue's high end.
    CAMERA_INDIRECT_SAMPLED: u32 => u32, "REG_CAMERA_INDIRECT_SAMPLED" = 512;
    // Slot diffusion: FP32 light/view position/coverage, FP16 radius/albedo and RGB9E5 ambient.
    VISIBILITY_DIFFUSION_WORDS: u32 => u32, "REG_VISIBILITY_DIFFUSION_WORDS" = 12;
    // Slot reflection receiver: full-precision eye-relative plane, shading normal and raster payload.
    // Twelve float4s retain both GGX lobes, sheet response, composition weighting
    // and prepared conductor compensation. Reflection records pack the live
    // components into half2 words; receiver planes stay FP32.
    // Store each word as its own slot-indexed field binding, so full-resolution
    // records do not require their combined allocation in one storage binding.
    REFLECTION_BSDF_VECTORS: u32 => u32, "REG_REFLECTION_BSDF_VECTORS" = 12;
    // Evaluation storage uses spare FP32 components in the reflected/normal
    // headers and BSDF vectors 9/10 to retain the last logical vector losslessly.
    REFLECTION_EVALUATION_VECTORS: u32 => u32, "REG_REFLECTION_EVALUATION_VECTORS" = 11;
    // The final vector occupies the two spare components in vectors 9/10
    // and one final half2 word, preserving all 46 live half components.
    VISIBILITY_REFLECTION_WORDS: u32 => u32, "REG_VISIBILITY_REFLECTION_WORDS" = 29;
    // Per-pixel reflection reconstruction guide: world/depth, oct normal, coverage and packed-lobe roughness as floats.
    VISIBILITY_REFLECTION_GUIDE_WORDS: u32 => u32, "REG_VISIBILITY_REFLECTION_GUIDE_WORDS" = 7;
    // World-specular receiving is a Boolean; each class reconstructs independently.
    REFLECTION_CLASSES: u32 => u32, "REG_REFLECTION_CLASSES" = 2;
    /// Bound full hit-shader scratch and work independently of image size.
    REFLECTION_DISPATCH_PIXELS: u32 => u32, "REG_REFLECTION_DISPATCH_PIXELS" = 16384;
    /// One u32 visibility mask word per pixel invocation.
    REFLECTION_BATCH_SAMPLES: u32 => u32, "REG_REFLECTION_BATCH_SAMPLES" = 32;
    /// Diffuse GI retains all samples across bounded invocations, with or without media.
    GI_RAY_BATCH: u32 => u32, "REG_GI_RAY_BATCH" = 16;
    /// Bound the reconstruction footprint as a fraction of the view extent.
    GI_FILTER_MAX_STEP: u32 => u32, "REG_GI_FILTER_MAX_STEP" = 16;
    GI_FILTER_EXTENT_PER_STEP: u32 => u32, "REG_GI_FILTER_EXTENT_PER_STEP" = 64;
    MEDIUM_REFLECTION_RAY_BATCH: u32 => u32, "REG_MEDIUM_REFLECTION_RAY_BATCH" = 1;
    /// Retain boundary cursors only across a short primary-lighting depth batch.
    /// High bit of reflection dispatch pixel offset: merged slots exist even at one sample.
    REFLECTION_MERGED_SLOTS: u32 => u32, "REG_REFLECTION_MERGED_SLOTS" = 1 << 31;
    // Material-error buffer sentinels: larger than any supported Blend lobe count.
    PATH_BOUNDARY_OVERFLOW: u32 => u32, "REG_PATH_BOUNDARY_OVERFLOW" = u32::MAX;
    PATH_BOUNDARY_AMBIGUOUS: u32 => u32, "REG_PATH_BOUNDARY_AMBIGUOUS" = u32::MAX - 1;
    PATH_MEDIUM_MAJORANT: u32 => u32, "REG_PATH_MEDIUM_MAJORANT" = u32::MAX - 3;
    PATH_MEDIUM_WORK: u32 => u32, "REG_PATH_MEDIUM_WORK" = u32::MAX - 10;
    PATH_MEDIUM_INVALID: u32 => u32, "REG_PATH_MEDIUM_INVALID" = u32::MAX - 4;
    PATH_MEDIUM_DISTANCE: u32 => u32, "REG_PATH_MEDIUM_DISTANCE" = u32::MAX - 12;
    // InterlockedMax retains the highest diagnostic. Nonfinite output from an
    // incomplete reservation may be retried, so both output diagnostics must
    // rank below every independent failure that must refuse the attempt.
    /// Invalid scene-linear radiance refuses the exposure before publication.
    RADIANCE_NONFINITE: u32 => u32, "REG_RADIANCE_NONFINITE" = u32::MAX - 16;
    DISPLAY_NONFINITE: u32 => u32, "REG_DISPLAY_NONFINITE" = u32::MAX - 15;
    /// A view returned code values for a display other than the stage's destination: an independent failure.
    VIEW_DESTINATION: u32 => u32, "REG_VIEW_DESTINATION" = u32::MAX - 13;
    /// Invalid environment source radiance is independent of provisional output.
    ENVIRONMENT_INVALID: u32 => u32, "REG_ENVIRONMENT_INVALID" = u32::MAX - 14;
    /// Invalid authored segment or exhaustion of a free-flight work reservation.
    /// Invalid generated camera ray or visibility interval.
    TRACE_EXHAUSTED: u32 => u32, "REG_TRACE_EXHAUSTED" = u32::MAX - 11;
    CAMERA_RAY_INVALID: u32 => u32, "REG_CAMERA_RAY_INVALID" = u32::MAX - 8;
    PATH_PROPAGATION_ERROR: u32 => u32, "REG_PATH_PROPAGATION_ERROR" = u32::MAX - 7;
    /// Refuse unresolved invocations; never classify work exhaustion as absorption.
    PROPAGATION_WORK_LIMIT: u32 => u32, "REG_PROPAGATION_WORK_LIMIT" = 65536;
    PATH_MEDIUM_DIVERGENT: u32 => u32, "REG_PATH_MEDIUM_DIVERGENT" = u32::MAX - 2;
    /// Cached opaque slot counts must fit the camera visibility budget.
    CAMERA_COVERAGE_INVALID: u32 => u32, "REG_CAMERA_COVERAGE_INVALID" = u32::MAX - 6;
    // Low 28 bits address a tap; high bits carry channel/normal metadata.
    MATERIAL_TAP_INDEX_MASK: u32 => u32, "REG_MATERIAL_TAP_INDEX_MASK" = 0x0fff_ffff;
    /// Invalid implicit scalar publication.
    IMPLICIT_INVALID: u32 => u32, "REG_IMPLICIT_INVALID" = u32::MAX - 5;
    /// A stateless geometry prefix produced nonfinite positions or frames.
    GEOMETRY_NONFINITE: u32 => u32, "REG_GEOMETRY_NONFINITE" = u32::MAX - 9;
    // GPU ABI: words per texture tap in the material word buffer (table word, flags, a 2x2
    // UV matrix and offset). Matches REG_MATERIAL_TAP_WORDS.
    MATERIAL_TAP_WORDS: usize => u32, "REG_MATERIAL_TAP_WORDS" = 9;
    // Irradiance volume probe record (probes.rs): per colour channel the mean
    // radiance and twice the cosine-weighted direction moment (L1 SH as E/pi = a + b.n),
    // then (front-facing hit ratio, -, -, -).
    PROBE_VEC4S: usize => u32, "REG_PROBE_VEC4S" = 4;
    // Octahedral depth-moment texels per side and probe (Chebyshev visibility).
    PROBE_VISIBILITY_SIDE: usize => u32, "REG_PROBE_VISIBILITY_SIDE" = 8;
    // Directional-provider atlas: 8/4/2/1 levels, RGB + World visibility and
    // first/second distance moments (three float2 pairs per texel).
    PROBE_RADIANCE_SIDE: u32 => u32, "REG_PROBE_RADIANCE_SIDE" = 8;
    PROBE_RADIANCE_TEXELS: u32 => u32, "REG_PROBE_RADIANCE_TEXELS" = 85;
    PROBE_RADIANCE_PAIRS: u32 => u32, "REG_PROBE_RADIANCE_PAIRS" = 3;
    PROBE_RADIANCE_SH_PAIRS: u32 => u32, "REG_PROBE_RADIANCE_SH_PAIRS" = 6;
    // Probe update threads per workgroup, and workgroups per dispatch row.
    PROBE_GROUP: u32 => u32, "REG_PROBE_GROUP" = 64;
    PROBE_DISPATCH_WIDTH: u32 => u32, "REG_PROBE_DISPATCH_WIDTH" = 4096;
    // Existing light ABI padding carries the inherited ray-contribution mask.
    LIGHT_CONTRIBUTION_WORD: usize => u32, "REG_LIGHT_CONTRIBUTION_WORD" = 0;
    // Path consumers retain ray permissions independently of map faces. Keep the
    // ray words independent, with the complete scalar block aligned to float4.
    LIGHT_RAY_WORD: usize => u32, "REG_LIGHT_RAY_WORD" = 2;
    LIGHT_RAY_LINK_WORD: usize => u32, "REG_LIGHT_RAY_LINK_WORD" = 3;
    LIGHT_RAY_SHADOWED: u32 => u32, "REG_LIGHT_RAY_SHADOWED" = 1;
    LIGHT_RAY_SHADOWED_BY_ALL: u32 => u32, "REG_LIGHT_RAY_SHADOWED_BY_ALL" = 2;
    // Installed caster topology: no draw needs a candidate opacity test.
    LIGHT_RAY_OPAQUE: u32 => u32, "REG_LIGHT_RAY_OPAQUE" = 8;
    // Frame.integrator.w retains transport flags below its camera-budget ray sequence offset.
    // Compensated per-collision roulette bounds expected work in lossless SSS walks.
    SUBSURFACE_WALK_SURVIVAL: f32 => f32, "REG_SUBSURFACE_WALK_SURVIVAL" = 0.95;
    EMITTER_WEIGHT_GROUP: u32 => u32, "REG_EMITTER_WEIGHT_GROUP" = 64;
    INTEGRATOR_SUBSURFACE_THICKNESS: u32 => u32, "REG_INTEGRATOR_SUBSURFACE_THICKNESS" = 1;
    // Blocking uses deterministic central emitter quadrature, avoiding low-budget light noise.
    INTEGRATOR_PREVIS_CENTER: u32 => u32, "REG_INTEGRATOR_PREVIS_CENTER" = 2;
    INTEGRATOR_RAY_SEQUENCE_SHIFT: u32 => u32, "REG_INTEGRATOR_RAY_SEQUENCE_SHIFT" = 2;
    // Reciprocal golden ratio seeds a coprime rank-one lattice with one sample per row/column.
    SAMPLE_LATTICE_RATIO: f32 => f32, "REG_SAMPLE_LATTICE_RATIO" = 0.618_034;
    // Last scalar light padding word contains the illuminates group mask.
    LIGHT_LINK_WORD: usize => u32, "REG_LIGHT_LINK_WORD" = 1;
    // Separate wildcard bit: explicit all 32 groups must still exclude ungrouped objects.
    LIGHT_LINK_ALL: u32 => u32, "REG_LIGHT_LINK_ALL" = 64;
    // Packed mesh-medium descriptor: object, BVH root/end, program/input base.
    MEDIUM_BOUND_WORDS: usize => u32, "REG_MEDIUM_BOUND_WORDS" = 5;
    // Threaded boundary node: min.xyz/escape, max.xyz/leaf triangle count.
    MEDIUM_BVH_WORDS: usize => u32, "REG_MEDIUM_BVH_WORDS" = 8;
    // Per path solid: object-space bounds lo.xyz, hi.xyz, then the object row placing them.
    /// Linear inverse-CDF intervals for the fixed Cycles Blackman–Harris film.
    PATH_FILTER_INTERVALS: usize => u32, "REG_PATH_FILTER_INTERVALS" = 1024;
    PATH_SOLID_WORDS: usize => u32, "REG_PATH_SOLID_WORDS" = 7;
    // glTF COLOR_0 multiplies baseColor; authored OpenPBR does not consume vertex colour.
    // Null interface retained solely to bound participating coefficients.
    FLAG_MEDIUM_BOUNDARY: u32 => u32, "REG_FLAG_MEDIUM_BOUNDARY" = 32;
    // Imported unlit base colour becomes emission after texture and vertex decoding.
    // Stable descriptor layout for ordinary imports and setup; larger tables grow normally.
    MATERIAL_TEXTURE_CAPACITY: u32 => u32, "REG_MATERIAL_TEXTURE_CAPACITY" = 64;
    MATERIAL_SAMPLER_CAPACITY: u32 => u32, "REG_MATERIAL_SAMPLER_CAPACITY" = 16;
    // Imported textured materials retain covered-sample evaluation on the shared table path.
    FLAG_SAMPLE_MATERIAL: u32 => u32, "REG_FLAG_SAMPLE_MATERIAL" = 128;
    // A covered point may pass a straight path connection (a thin sheet, or a
    // solid whose relative IOR can be one): materials::PendingMaterial::null_connection.
    FLAG_NULL_CONNECTION: u32 => u32, "REG_FLAG_NULL_CONNECTION" = 256;
    FLAG_GLTF_UNLIT: u32 => u32, "REG_FLAG_GLTF_UNLIT" = 64;
    FLAG_VERTEX_COLOR: u32 => u32, "REG_FLAG_VERTEX_COLOR" = 8;
    // One fractional crossing per tap, plus the two endpoints.
    // math: pi, retaining the environment convolution literal.
    // MaterialRecord tag.z bits and texture-tap flags share this registry.
    FLAG_DOUBLE_SIDED: u32 => u32, "REG_FLAG_DOUBLE_SIDED" = 2;
    FLAG_VERTEX_ALPHA: u32 => u32, "REG_FLAG_VERTEX_ALPHA" = 4;
    FLAG_UV_SECOND: u32 => u32, "REG_FLAG_UV_SECOND" = 1;
    FLAG_UV_BASIS: u32 => u32, "REG_FLAG_UV_BASIS" = 2;
    // Device-owned BRDF atlas, shared by every scene and evaluator.
    BRDF_SIZE: u32 => u32, "REG_BRDF_SIZE" = 128;
    BRDF_ALBEDO_SIZE: u32 => u32, "REG_BRDF_ALBEDO_SIZE" = 64;
    BRDF_IOR_STEPS: u32 => u32, "REG_BRDF_IOR_STEPS" = 64;
    BRDF_HEIGHT: u32 => u32, "REG_BRDF_HEIGHT" = BRDF_SIZE + ((BRDF_IOR_STEPS + 2) / 2) * BRDF_ALBEDO_SIZE;
    // Quadrature budget for the same dielectric integral without a bound LUT.
    DIELECTRIC_ALBEDO_SAMPLES: u32 => u32, "REG_DIELECTRIC_ALBEDO_SAMPLES" = 512;
    PI: f32 => f32, "REG_PI" = 3.141592653589793;
    TAU: f32 => f32, "REG_TAU" = 2. * PI;
    // numerical safety epsilon: prevents singular zero-roughness GGX sampling.
    MIN_ALPHA: f32 => f32, "REG_GGX_ALPHA_MIN" = 0.0001;
    UINT_SCALE: f32 => f32, "REG_UINT_SCALE" = 0.000000059604644775390625;
    COSINE_MIN: f32 => f32, "REG_COSINE_MIN" = 1e-7;
    RNG_MIX_A: u32 => u32, "REG_RNG_MIX_A" = 0x7feb352d;
    RNG_MIX_B: u32 => u32, "REG_RNG_MIX_B" = 0x846ca68b;
    SAMPLE_HASH_MIX: u32 => u32, "REG_SAMPLE_HASH_MIX" = 0x45d9f3b;
    // Golden-ratio word step decorrelates streams and keeps dither independent of image width.
    RNG_PIXEL_MIX: u32 => u32, "REG_PIXEL_HASH_STREAM_STEP" = 0x9e3779b9;
    RNG_SAMPLE_MIX: u32 => u32, "REG_RNG_SAMPLE_MIX" = 0x85ebca6b;
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    #[test]
    fn independent_failures_outrank_retryable_output_diagnostics() {
        let independent = [TRACE_EXHAUSTED, PATH_MEDIUM_DISTANCE, PATH_MEDIUM_INVALID,
            PATH_MEDIUM_WORK, PATH_BOUNDARY_OVERFLOW, PATH_BOUNDARY_AMBIGUOUS,
            PATH_MEDIUM_MAJORANT, PATH_PROPAGATION_ERROR, CAMERA_RAY_INVALID,
            PATH_MEDIUM_DIVERGENT, CAMERA_COVERAGE_INVALID, IMPLICIT_INVALID,
            GEOMETRY_NONFINITE, ENVIRONMENT_INVALID, VIEW_DESTINATION];
        for code in independent {
            assert!(code > RADIANCE_NONFINITE && code > DISPLAY_NONFINITE,
                "independent diagnostic {code} must survive concurrent nonfinite output");
        }
    }
    #[test]
    fn frame_diagnostics_are_distinct() {
        let codes=[TRACE_EXHAUSTED,PATH_MEDIUM_DISTANCE,PATH_MEDIUM_INVALID,PATH_MEDIUM_WORK,PATH_BOUNDARY_OVERFLOW,PATH_BOUNDARY_AMBIGUOUS,PATH_MEDIUM_MAJORANT,DISPLAY_NONFINITE,RADIANCE_NONFINITE,PATH_PROPAGATION_ERROR,CAMERA_RAY_INVALID,PATH_MEDIUM_DIVERGENT,CAMERA_COVERAGE_INVALID,IMPLICIT_INVALID,GEOMETRY_NONFINITE, ENVIRONMENT_INVALID, VIEW_DESTINATION];
        assert_eq!(std::collections::BTreeSet::from(codes).len(),codes.len());
    }
}
