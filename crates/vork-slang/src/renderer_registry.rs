//! Renderer shader constants, evaluated in Rust for WGSL and Slang.
//! Shared CPU/shader ABI values come from the common numeric registry.
use crate::registry as common;
use std::fmt::Write;
trait Value {
    fn shader(&self, ty: &str, slang: bool) -> String;
}
impl Value for f32 {
    fn shader(&self, _: &str, _: bool) -> String {
        format!("{self:?}")
    }
}
impl Value for u32 {
    fn shader(&self, _: &str, _: bool) -> String {
        format!("{self}u")
    }
}
impl Value for i32 {
    fn shader(&self, _: &str, _: bool) -> String {
        self.to_string()
    }
}
impl<T: Value, const N: usize> Value for [T; N] {
    fn shader(&self, ty: &str, slang: bool) -> String {
        let vector = ty.starts_with("vec");
        let element = if vector {
            "f32"
        } else {
            ty.strip_prefix("array<")
                .unwrap()
                .strip_suffix('>')
                .unwrap()
                .rsplit_once(',')
                .unwrap()
                .0
                .trim()
        };
        let values = self
            .iter()
            .map(|v| v.shader(element, slang))
            .collect::<Vec<_>>()
            .join(",");
        if vector || !slang {
            format!("{}({values})", shader_type(ty, slang))
        } else {
            format!("{{{values}}}")
        }
    }
}
fn shader_type(ty: &str, slang: bool) -> String {
    if !slang {
        return ty.to_owned();
    }
    match ty {
        "f32" => "float".into(),
        "u32" => "uint".into(),
        "i32" => "int".into(),
        t if t.starts_with("vec") => format!("float{}", &t[3..4]),
        _ => panic!("unexpected shader type {ty}"),
    }
}
macro_rules! constants {
    ($($name:ident: $ty:ty = $value:expr => $shader:literal;)*) => {
        $(pub const $name: $ty = $value;)*
        pub fn generate(slang: bool) -> String {
            let shared = common::wgsl();
            let mut out = if slang { common::slang() } else { shared.clone() };
            // Runtime compiles specialize this from the declared AOV list. Shipped
            // generics have none. Slang SPIR-V cannot lower live zero-length arrays;
            // one unused element keeps types valid, while CHANNELS folds work away.
            if slang {out.push_str("#ifndef VORK_AOV_COUNT\n#define VORK_AOV_COUNT 0\n#endif\npublic static const uint VORK_AOV_CHANNELS = VORK_AOV_COUNT;\npublic static const uint VORK_AOV_SLOTS = VORK_AOV_COUNT > 0 ? VORK_AOV_COUNT : 1;\n");}
            $(
                if !shared.lines().any(|line| line.starts_with(concat!("const ", stringify!($name), ":"))) {
                let value = $name.shader($shader, slang);
                if slang {
                    let declaration = if let Some(array) = $shader.strip_prefix("array<") {
                        let (element, count) = array.strip_suffix('>').unwrap().rsplit_once(',').unwrap();
                        format!("{} {}[{}]", shader_type(element.trim(), true), stringify!($name), count.trim())
                    } else { format!("{} {}", shader_type($shader, true), stringify!($name)) };
                    writeln!(out, "public static const {declaration} = {value};").unwrap();
                } else { writeln!(out, "const {}:{}={value};", stringify!($name), $shader).unwrap(); }
                }
            )*
            if slang {
                out.push_str("public float vork_ggx_normal_inverse(int index) {switch(index) {");
                for (index,value) in REG_GGX_NORMAL_INVERSE.iter().enumerate() {
                    write!(out,"case {index}: return {value:?};").unwrap();
                }
                out.push_str("default:return 0.;}}\n");
            }
            out
        }
    };
}
constants! {
    // Existing path roulette policy, shared with raster continuation. These
    // bound survival probability, never optical-event depth.
    REG_PATH_ROULETTE_START: u32 = 3 => "u32";
    REG_PATH_ROULETTE_MIN: f32 = 0.05 => "f32";
    REG_PATH_ROULETTE_MAX: f32 = 0.95 => "f32";
    // Bound scalarized dielectric descriptors to 56 words; larger stacks stay indexed.
    REG_PATH_REGISTER_BOUNDARIES: u32 = 8 => "u32";
    // Visible detector interval; hero wavelength sampling and its reciprocal PDF agree.
    REG_SPECTRAL_MIN_NM: f32 = 380. => "f32";
    REG_SPECTRAL_SPAN_NM: f32 = 400. => "f32";
    // Spatial-source proposal: global uniform support, not a radiance clamp.
    REG_ANALYTIC_LIGHT_SAMPLES: u32 = 8u32 => "u32";
    REG_SAMPLING_STREAM_ANALYTIC_SOURCE: u32 = 26u32 => "u32";
    REG_LIGHT_TREE_UNIFORM_MIX: f32 = 0.25 => "f32";
    // Squared-distance guide regularization inside a punctual source.
    REG_LIGHT_TREE_MIN_DISTANCE_SQUARED: f32 = 1e-12 => "f32";
    // IEEE-754 binary32 spacing at one.
    REG_FP32_EPSILON: f32 = f32::EPSILON => "f32";
    // Largest finite IEEE-754 binary32 parameter, not an authored world-space cap.
    REG_FP32_MAX: f32 = f32::MAX => "f32";
    // Unit roundoff u=2^-24; gamma(7) bounds a four-term dot product without FMA.

    // math: pi, retaining the environment convolution literal.
    REG_PI: f32 = common::PI as f32 => "f32";
    // BC7 mode-5/6 principal-axis convergence and endpoint least-squares quality.
    REG_BC7_PCA_ITERATIONS: u32 = 8 => "u32";
    REG_BC7_ENDPOINT_REFINEMENTS: u32 = 2 => "u32";
    // BC7 conditioning matches the native encoder's regularized 8-bit covariance.
    REG_BC7_PCA_EPSILON: f32 = 0.000001 => "f32";
    // math: 2*pi, retaining the AO angle literal.
    // numerical safety epsilon: keeps positive denominators and logarithms finite.
    REG_POSITIVE_EPSILON: f32 = 1e-20 => "f32";
    // numerical safety epsilon: prevents singular zero-roughness GGX sampling.
    REG_GGX_ALPHA_MIN: f32 = common::MIN_ALPHA as f32 => "f32";
    // math: 2^-32 maps reversed 32-bit integers into the unit interval.
    REG_RADICAL_INVERSE_SCALE: f32 = 2.3283064365386963e-10 => "f32";
    // quality/performance setting: existing convolution sample count.
    REG_ENVIRONMENT_SAMPLES: u32 = 512u32 => "u32";
    // derived: 2^16-1, the largest unsigned 16-bit control value.
    REG_UNORM16_MAX: f32 = 65535. => "f32";
    // derived: 2^16-1, the largest unsigned 16-bit object ID.
    REG_UNORM16_MAX_U32: u32 = 65535u32 => "u32";
    // quality/performance setting: existing square image compute workgroup side.
    REG_COMPUTE_TILE: u32 = 8u32 => "u32";
    // quality/performance setting: existing deformation workgroup width.
    REG_DEFORM_GROUP: u32 = common::DEFORM_GROUP as u32 => "u32";
    // derived: packed Vertex ABI has 3 position + 3 normal + 4 UV + 2 padding + 4 colour + 4 tangent floats.
    REG_VERTEX_FLOATS: u32 = 20u32 => "u32";
    // derived: four columns of four floats in a joint matrix.
    REG_MATRIX_FLOATS: u32 = 16u32 => "u32";
    // derived: count/morph/joint/influence-set fields in the deformation header.
    REG_DEFORM_HEADER_FLOATS: u32 = 4u32 => "u32";
    // derived: three vec3 deltas (position, normal, tangent) per morph vertex.
    REG_MORPH_FLOATS: u32 = common::MORPH_FLOATS as u32 => "u32";
    // GPU dispatch ABI: matches CLUSTER_GROUP in constants.rs; one invocation per cluster.
    REG_CLUSTER_GROUP: u32 = common::CLUSTER_GROUP as u32 => "u32";
    // List address sentinel: off-frustum receivers visit the complete light batch.
    REG_CLUSTER_UNCULLED: u32 = u32::MAX => "u32";
    // numerical safety: round cluster bounds outward after inverse projection in f32.
    REG_CLUSTER_PADDING: f32 = 0.001 => "f32";
    // numerical safety: initial finite extrema for view-space cluster bounds.
    REG_CLUSTER_BOUND: f32 = 1e30 => "f32";
    // numerical safety: four f32 ULPs cover receiver-plane evaluation roundoff.
    REG_SHADOW_FP_ALLOWANCE: f32 = 0.000000477 => "f32";
    // Packing: two perceptually encoded 5-bit values fit exactly in a half-float integer.
    REG_REFLECTION_PACK_LEVELS: f32 = 31. => "f32";
    REG_REFLECTION_PACK_BASE: f32 = 32. => "f32";
    // GPU ABI: render technique codes stored in Frame.projection.z, matching constants.rs.
    REG_REFLECTIONS_OFF: f32 = common::REFLECTIONS_OFF as f32 => "f32";
    REG_REFLECTIONS_RAY: f32 = common::REFLECTIONS_RAY as f32 => "f32";
    // cos/sin(pi*(3-sqrt(5))): equal-area Fermat aperture quadrature.
    // Golden-ratio word step decorrelates streams and keeps dither independent of image width.
    REG_PIXEL_HASH_STREAM_STEP: u32 = common::RNG_PIXEL_MIX as u32 => "u32";
    // Published Heitz GGX fit resolution, with texel-centred linear interpolation.
    // Hammersley NDF quadrature for directional surface responses not represented by isotropic LTC/moment tables.
    REG_SURFACE_QUADRATURE_SAMPLES: u32 = 64 => "u32";
    // View-dependent substrate energy uses a bounded visible-normal integrand.
    REG_SURFACE_ALBEDO_SAMPLES: u32 = 256 => "u32";
    REG_LTC_SIZE: f32 = common::LTC_SIZE as f32 => "f32";
    // Four scalar components in a vector storage load.
    REG_VEC4_FLOATS: u32 = 4u32 => "u32";
    // Packed raster ABI and matrix columns expressed as vector storage strides.
    REG_VERTEX_VECTORS: u32 = REG_VERTEX_FLOATS / REG_VEC4_FLOATS => "u32";
    REG_MATRIX_VECTORS: u32 = REG_MATRIX_FLOATS / REG_VEC4_FLOATS => "u32";
    // Shared integer avalanche multiplier for spatial quadrature and display noise.
    // Distinct deterministic random dimensions for specular and diffuse screen estimators.
    REG_SAMPLING_STREAM_REFLECTION: u32 = 0u32 => "u32";
    REG_SAMPLING_STREAM_REFLECTION_EMITTER: u32 = 10u32 => "u32";
    // Shadow reserves base + light*STEP + {0,1,2}; singleton streams must
    // avoid all three dimensions, including those of nonzero light indices.
    // Independent rotation of the surface-response integration lattice per camera sample.
    REG_SAMPLING_STREAM_SUBSURFACE: u32 = 21u32 => "u32";
    REG_SAMPLING_STREAM_SURFACE_QUADRATURE: u32 = 20u32 => "u32";
    REG_SAMPLING_STREAM_RASTER_ALPHA: u32 = 14u32 => "u32";
    // Two independent VNDF coordinates for rough-sheet connection throughput.
    REG_SAMPLING_STREAM_SHADOW_SHEET_U: u32 = 30u32 => "u32";
    REG_SAMPLING_STREAM_SHADOW_SHEET_V: u32 = 31u32 => "u32";
    REG_SAMPLING_STREAM_SHADOW_ALPHA: u32 = 15u32 => "u32";
    // Independent camera-volume subcell quadrature; shares the accumulation lattice.
    REG_SAMPLING_STREAM_VOLUME: u32 = 6u32 => "u32";
    // Independent Bernoulli dimension for the world-visibility residual.
    REG_SAMPLING_STREAM_VOLUME_RESIDUAL: u32 = 24u32 => "u32";
    // Independent discrete analytic-source choice for secondary medium scattering.
    REG_SAMPLING_STREAM_MEDIUM_LIGHT: u32 = 25u32 => "u32";
    // Equal cached-only/direct-corrected strata bound residual amplification to two.
    REG_MEDIUM_RESIDUAL_STRATA: u32 = 2u32 => "u32";
    // Deformed output stores position/normal/tangent; UV/color remain in the shared stream.
    REG_DEFORM_VERTEX_FLOATS: u32 = 10u32 => "u32";
    // Each of Blender's at most sixteen horizon slices has two half-planes.
    // Plane-distance guide: 0.5% view depth covers half-res representative error;
    // a 1mm floor covers depth quantization near the lens without mixing silhouettes.
    REG_AO_FILTER_DEPTH_SCALE: f32 = 0.005 => "f32";
    REG_AO_FILTER_MIN_THICKNESS: f32 = 0.001 => "f32";
    // Finite-support variance-aware diffuse reconstruction, before receiver response.
    REG_GI_PILOT_RAYS: u32 = 4u32 => "u32";
    REG_GI_GUIDE_COSINE: f32 = 0.999 => "f32";
    REG_GI_GUIDE_MIX: f32 = 0.5 => "f32";
    // Independent pilots choose a beauty budget before beauty sampling starts.
    REG_GI_ADAPTIVE_RATIO: u32 = 8u32 => "u32";
    REG_GI_CONCENTRATION: f32 = 16. => "f32";
    REG_GI_FILTER_RADIUS: i32 = 2 => "i32";
    REG_BINOMIAL5_WEIGHTS: [f32; 3] = [6.,4.,1.] => "array<f32,3>";
    // Preserve normal discontinuities when reconstructing rotated horizon samples.
    REG_AO_BLUR_NORMAL_POWER: f32 = 8. => "f32";
    // Half-resolution crease representatives need a second same-surface neighbour.
    REG_AO_UPSAMPLE_RADIUS: i32 = 2 => "i32";
    // Numerical safety: dimensionless conditioning cutoff for receiver-plane gradients.
    REG_RECEIVER_GRADIENT_CONDITION_EPSILON: f32 = 0.00000001 => "f32";
    // Requested output stores; matches constants.rs, carried exactly in frame.projection.w.
    REG_OUTPUT_GPU_COLOR: u32 = common::OUTPUT_GPU_COLOR as u32 => "u32";
    REG_OUTPUT_COLOR_PLANE: u32 = common::OUTPUT_COLOR_PLANE as u32 => "u32";
    REG_OUTPUT_COLOR_PLANE16: u32 = common::OUTPUT_COLOR_PLANE16 as u32 => "u32";
    REG_OUTPUT_DEPTH_IDS: u32 = common::OUTPUT_DEPTH_IDS as u32 => "u32";
    REG_OUTPUT_NORMALS: u32 = common::OUTPUT_NORMALS as u32 => "u32";
    REG_OUTPUT_MOTION: u32 = common::OUTPUT_MOTION as u32 => "u32";
    // Resolved controls retain the 16-bit object ID; diffuse response is separate FP32 RGB.
    // Cluster list ABI: one count followed by installed light indices.
    REG_CLUSTER_LIST_HEADER: u32 = common::CLUSTER_LIST_HEADER as u32 => "u32";
    // Sun, point, spot, area: stable kind sweeps keep fragment paths coherent.
    REG_LIGHT_KIND_COUNT: u32 = 4u32 => "u32";
    // Contribution/link masks, ray permissions, volume slot and ABI padding.
    REG_LIGHT_SHADOW_OFFSET_WORDS: u32 = common::LIGHT_SHADOW_OFFSET_WORDS as u32 => "u32";
    // Optional raw float channels follow the four packed PNG/control planes.
    REG_OUTPUT_FLOAT_COLOR: u32 = common::OUTPUT_FLOAT_COLOR as u32 => "u32";
    REG_OUTPUT_FLOAT_DEPTH: u32 = common::OUTPUT_FLOAT_DEPTH as u32 => "u32";
    REG_OUTPUT_FLOAT_NORMALS: u32 = common::OUTPUT_FLOAT_NORMALS as u32 => "u32";
    REG_OUTPUT_FLOAT_MOTION: u32 = common::OUTPUT_FLOAT_MOTION as u32 => "u32";
    // Blender Z pass background sentinel, metres (finite infinity for EXR).
    REG_FLOAT_COLOR_PLANE: u32 = common::FLOAT_COLOR_PLANE as u32 => "u32";
    REG_FLOAT_DEPTH_PLANE: u32 = common::FLOAT_DEPTH_PLANE as u32 => "u32";
    REG_FLOAT_NORMAL_PLANE: u32 = common::FLOAT_NORMAL_PLANE as u32 => "u32";
    REG_FLOAT_MOTION_PLANE: u32 = common::FLOAT_MOTION_PLANE as u32 => "u32";
    // Object ray visibility; paired with constants.rs and the six v2 node fields.
    REG_RAY_CAMERA: u32 = common::RAY_CAMERA as u32 => "u32";
    REG_RAY_SHADOW: u32 = common::RAY_SHADOW as u32 => "u32";
    REG_RAY_DIFFUSE: u32 = common::RAY_DIFFUSE as u32 => "u32";
    REG_RAY_SPECULAR: u32 = common::RAY_SPECULAR as u32 => "u32";
    REG_RAY_TRANSMISSION: u32 = common::RAY_TRANSMISSION as u32 => "u32";
    REG_RAY_VOLUME: u32 = common::RAY_VOLUME as u32 => "u32";
    // One spare exact-integer f16 bit beside the two five-bit reflection fields.
    REG_VISIBILITY_DIFFUSE_OFFSET: f32 = REG_REFLECTION_PACK_BASE*REG_REFLECTION_PACK_BASE => "f32";
    // Outside WebGPU z in [0,w]; all vertices of a hidden instance clip away.
    REG_CULLED_CLIP: [f32; 4] = [0.,0.,2.,1.] => "vec4<f32>";
    // Match MATERIAL_INPUT_WORDS / MATERIAL_TAP_WORDS in constants.rs: the material record's
    // per-input tap bytes and the words of one texture tap in the material word buffer.
    REG_MATERIAL_TAP_WORDS: u32 = common::MATERIAL_TAP_WORDS as u32 => "u32";
    // Existing scalar light-record padding carries ray contribution flags.
    REG_LIGHT_CONTRIBUTION_WORD: u32 = common::LIGHT_CONTRIBUTION_WORD as u32 => "u32";
    // Triangle visibility metadata; matches LIGHT_RAY_* in constants.rs.
    REG_LIGHT_RAY_WORD: u32 = common::LIGHT_RAY_WORD as u32 => "u32";
    REG_LIGHT_RAY_LINK_WORD: u32 = common::LIGHT_RAY_LINK_WORD as u32 => "u32";
    REG_LIGHT_RAY_OPAQUE: u32 = common::LIGHT_RAY_OPAQUE as u32 => "u32";
    REG_LIGHT_RAY_SHADOWED: u32 = common::LIGHT_RAY_SHADOWED as u32 => "u32";
    REG_LIGHT_RAY_SHADOWED_BY_ALL: u32 = common::LIGHT_RAY_SHADOWED_BY_ALL as u32 => "u32";
    // Retained diffusion gets this many probes in addition to the exposure budget:
    // visibility samples do not average its one-per-receiver spatial estimate.
    REG_SUBSURFACE_BASE_PROBES: u32 = 64u32 => "u32";
    // Scheduling grain only: complete probe sequences continue across dispatches.
    // A 1024-probe L4 diffusion dispatch takes seconds; quarter-sized batches
    // bound invocation latency without reducing the integral's density.
    REG_SUBSURFACE_BATCH_PROBES: u32 = 256u32 => "u32";
    // RGB primary irradiance, ambient irradiance and normalization weights.
    REG_SUBSURFACE_ACCUMULATION_WORDS: u32 = 9u32 => "u32";
    // Cubic expm1 series: omitted fourth term is below float32 precision here.
    REG_SUBSURFACE_SERIES_MAX: f32 = 0.001 => "f32";
    // Alpha-table word of instances without alpha-tested geometry (rt.rs NO_ALPHA).
    REG_RAY_NO_ALPHA: u32 = 0xffffffffu32 => "u32";
    // Uniform sequence row offset above Frame.integrator.x sampled-method bit.
    REG_INTEGRATOR_RAY_SEQUENCE_SHIFT: u32 = common::INTEGRATOR_RAY_SEQUENCE_SHIFT => "u32";
    // Rank-one lattice step ratio; equals SAMPLE_LATTICE_RATIO in constants.rs.
    REG_SAMPLE_LATTICE_RATIO: f32 = common::SAMPLE_LATTICE_RATIO as f32 => "f32";
    // Named illumination links use the last scalar light-record padding word.
    REG_LIGHT_LINK_WORD: u32 = common::LIGHT_LINK_WORD as u32 => "u32";
    REG_LIGHT_LINK_ALL: u32 = common::LIGHT_LINK_ALL as u32 => "u32";
    // World-medium fallback quadrature; analytic library segments avoid these steps.
    REG_MEDIUM_STEPS: u32 = 8u32 => "u32";
    // Finite Beer-Lambert exponent and HG singularity bounds.
    REG_MEDIUM_EXP_LIMIT: f32 = 80. => "f32";
    // Degree-five (1-exp(-tau))/tau series remainder <= tau^6/5040:
    // at 1/8 it is below 8e-10, beneath f32 rounding error.
    REG_MEDIUM_INTEGRAL_SERIES_LIMIT: f32 = 0.125 => "f32";
    // Preserve the punctual-light one-millimetre centre-distance regularization.
    REG_LIGHT_MIN_DISTANCE_SQUARED: f32 = 1e-6 => "f32";
    // Streams 0/1 belong to reflections/GI; shadow lights share stream 2 with light-keyed seeds.
    REG_SAMPLING_STREAM_SHADOW: u32 = 2u32 => "u32";
    // Free ID bit31 and normal.w sign encode receiver world-light selection.
    REG_WORLD_SPECULAR_OFFSET: f32 = 2. => "f32";
    // Match FLAG_VERTEX_COLOR: glTF-only COLOR_0 base colour multiplier.
    REG_FLAG_GLTF_UNLIT: u32 = common::FLAG_GLTF_UNLIT as u32 => "u32";
    REG_FLAG_VERTEX_COLOR: u32 = common::FLAG_VERTEX_COLOR as u32 => "u32";
    // Screen-depth receiver agreement (independent of traversal completeness).
    REG_TRANSMISSION_DEPTH_TOLERANCE: f32 = 0.001 => "f32";
    // Relative world-space depth agreement.
    REG_MEDIUM_BOUND_WORDS: u32 = common::MEDIUM_BOUND_WORDS as u32 => "u32";
    REG_MEDIUM_TRIANGLE_WORDS: u32 = 9u32 => "u32";
    // Independent direction proposal for finite medium sources; never shares the path quadrature dimension.
    REG_SAMPLING_STREAM_MEDIUM_SOURCE: u32 = 22u32 => "u32";
    // Independent phase proposal for incident World light in participating media.
    REG_SAMPLING_STREAM_MEDIUM_ENVIRONMENT: u32 = 23u32 => "u32";
    REG_SAMPLING_STREAM_RAY_GI: u32 = 13u32 => "u32";
    REG_SAMPLING_STREAM_PROBE_EMITTER: u32 = 17u32 => "u32";
    REG_SAMPLING_STREAM_PROBE_TRIANGLE: u32 = 18u32 => "u32";
    REG_SAMPLING_STREAM_SECONDARY_EMITTER: u32 = 19u32 => "u32";
    REG_SAMPLING_STREAM_RAY_GI_LOBE: u32 = 7u32 => "u32";
    // Stable visibility prefixes process one fixed block per workgroup.
    REG_CAMERA_CULL_GROUP: u32 = common::CAMERA_CULL_GROUP => "u32";
    REG_CAMERA_CULL_BLOCK_VEC4S: u32 = common::CAMERA_CULL_BLOCK_VEC4S as u32 => "u32";
    REG_CAMERA_CULL_DRAW_VEC4S: u32 = common::CAMERA_CULL_DRAW_VEC4S as u32 => "u32";
    REG_CAMERA_INSTANCE_LOCATION: u32 = common::CAMERA_INSTANCE_LOCATION => "u32";
    // Prefix counts occupy the low bits; the high bit retains visibility.
    REG_CAMERA_VISIBLE_BIT: u32 = 0x80000000u32 => "u32";
    // Margin covers float roundoff in affine/projective interval bounds; rejection only.
    REG_CAMERA_CULL_ERROR: f32 = common::CAMERA_CULL_ERROR => "f32";
    // DDGI depth-moment lobe exponent and the distance kept for back-face hits
    // (Majercik et al. 2019), so probes inside geometry do not see through it.
    // Hash work per request; overflow grows storage and replays, never drops receivers.
    REG_PROBE_REQUEST_PROBES: u32 = 8u32 => "u32";
    REG_PROBE_DEPTH_SHARPNESS: f32 = 50. => "f32";
    REG_PROBE_BACKFACE_DISTANCE: f32 = 0.2 => "f32";
    // Below 1/8, eight GGX angular-moment series terms have <1e-9 truncation
    // error; above it the analytic expression avoids destructive cancellation.
    REG_GGX_MOMENT_SERIES_LIMIT: f32 = 0.125 => "f32";
    // Inverse projected-GGX mean cosine, Chebyshev coefficients for (1-M)/x
    // over log(1-M), validated on GPU against projected-NDF quadrature.
    // Tests bound relative alpha-squared error across the complete float domain.
    REG_GGX_NORMAL_INVERSE_DEGREE: i32 = REG_GGX_NORMAL_INVERSE.len() as i32 - 1 => "i32";
    // ln(2^-24): smallest positive deficit of a float below one.
    REG_GGX_NORMAL_LOG_MIN: f32 = -16.6355323 => "f32";
    // ln(1/3)-ln(2^-24): the alpha=1 mean cosine is 2/3.
    REG_GGX_NORMAL_LOG_RANGE: f32 = 15.53692 => "f32";
    REG_GGX_NORMAL_INVERSE: [f32; 19] = [
        4.81287476,
        -4.39369208,
        -0.106960362,
        -0.0128516757,
        0.0097766604,
        0.010878089,
        0.00706939291,
        0.00365507383,
        0.00162074556,
        0.000636114426,
        0.000224864427,
        7.24247609e-05,
        2.14470438e-05,
        5.88604181e-06,
        1.51088626e-06,
        3.65397979e-07,
        8.47586129e-08,
        1.85457945e-08,
        4.15156251e-09
    ] => "array<f32,19>";
    REG_LUMINANCE: [f32; 3] = [0.2126,0.7152,0.0722] => "vec3<f32>";
}

// Fixed quadrature tables must cover every dynamic tap budget.
