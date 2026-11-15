//! Versioned authoring-contract registry and every scene default.
//! Precedence: the area's standard, else Blender 5.2.1, else glTF 2.0, else a vork
//! choice recorded in docs/DEFAULTS.md. Identity transforms and white are neutral values.
use super::{OpenPbrKind, OpenPbrParam};
/// Blender 5.2.1 EEVEE offline render accumulation samples.
pub const RENDER_SAMPLES: u32 = 64;
/// Shipped camera budgets with OIDN denoising.
pub const DRAFT_SAMPLES: u32 = 4;
pub const FINAL_SAMPLES: u32 = 8;
/// Usable level-2 camera budget, independent of denoising.
pub const PATH_SAMPLES: u32 = 64;
/// Budgets when no fast GPU denoiser exists: more samples replace OIDN's cleanup.
pub const DRAFT_SAMPLES_WITHOUT_DENOISER: u32 = 16;
pub const FINAL_SAMPLES_WITHOUT_DENOISER: u32 = 32;
/// Previs spends four camera samples on silhouette coverage and ray shadows.
pub const PREVIS_SAMPLES: u32 = 4;
/// Built-in `--channels` output tokens in default order (the renderer parses
/// exactly these). EXR layers use the same names (`object_index.X`) and an AOV
/// writes `<name>.*`, so AOVs may not take these names.
pub const OUTPUT_CHANNELS: &[&str] = &["color", "depth", "object_index", "view_normal", "motion", "camera", "shadow_catcher"];
/// OpenPBR Surface 1.1.1 parameters, generated from `openpbr-params.md` (the spec's reference table).
pub const OPENPBR: &[OpenPbrParam] = include!(concat!(env!("OUT_DIR"), "/openpbr.rs"));

/// Blender 5.2.1 render defaults: 1920 x 1080 at 24 fps.
pub(super) const RESOLUTION: [u32; 2] = [1920, 1080];
pub(super) const FPS: crate::Time = crate::Time::integer(24);
/// Blender 5.2.1 motion blur: half-frame shutter (180 degrees).
pub const MOTION_BLUR_SHUTTER: f32 = 0.5;

/// vork: screen AO radius and strength.
/// Blender 5.2.1 spot: 45 degree cone (22.5 degree half angle), blend 0.15.
pub(super) const SPOT_BLEND: f32 = 0.15;
/// Blender 5.2.1 SpotLight.spot_size default, 45 degrees (full cone angle).
pub(super) const SPOT_SIZE: f32 = std::f32::consts::FRAC_PI_4;
/// Blender 5.2.1 Add Light colour; also an environment image's multiplier.
pub(super) const WHITE: [f32; 3] = [1.; 3];
/// Rejects zero-length rotations; zero scale is valid and hides geometry.
pub(super) const NONDEGENERATE_EPSILON: f32 = 1e-8;

/// Blender factory scene frame range 1..250 at 24 fps, expressed as seconds.
pub(super) const DURATION_SECONDS: &str = "125/12";
/// Blender area-light dimensions, metres; point source radius remains zero.
pub(super) const AREA_DIMENSION_M: f32 = 1.;
pub(super) fn area_dimension() -> f32 {
    AREA_DIMENSION_M
}

/// Blender 5.2.1 Add Light operators: sun irradiance 1 W/m², local power 10 W.
pub(super) const SUN_INTENSITY: f32 = 1.;
pub(super) const LOCAL_INTENSITY: f32 = 10.;

/// Avoid misleading spelling suggestions when every candidate is unrelated.
pub(super) const SUGGESTION_MIN_DISTANCE: usize = 1;
/// Allow proportionally more typos in longer identifiers.
pub(super) const SUGGESTION_LENGTH_DIVISOR: usize = 3;

/// Keep large node tables from turning one error into thousands of candidate names.
pub(super) const SUGGESTION_DISPLAY_LIMIT: usize = 5;
/// Blender 5.2.1 Background shader strength.
pub(super) const ENVIRONMENT_INTENSITY: f32 = 1.;

/// Blender CameraDOFSettings default anamorphic ratio.
/// Bound the offline accumulation budget; device upload limits can impose a smaller scene-dependent bound.
pub const MAX_RENDER_SAMPLES: u32 = 4096;

/// Blender factory World Background colour, scene linear.
pub const ENVIRONMENT_COLOR: [f32; 3] = [0.05; 3];
/// Blender 5.2.1 Light.temperature hard range in kelvin.
pub const LIGHT_TEMPERATURE_MIN: f32 = 800.;
pub const LIGHT_TEMPERATURE_MAX: f32 = 20000.;
/// vork: peak source radiance kept in the filtered environment when extracting a sun.

/// Unmodified animation layers use full weight and real-time speed.
const LAYER_UNIT: f32 = 1.;
pub(super) fn layer_unit() -> f32 {
    LAYER_UNIT
}
/// Node clock defaults: local time begins at content second 0, at real-time speed.
pub(super) fn clock_zero() -> vork_time::Clock { vork_time::Clock::from(0.) }
pub(super) fn clock_one() -> vork_time::Clock { vork_time::Clock::from(1.) }

/// Blender 5.2.1 sun full angular diameter, radians (0.526 degrees).
pub(super) fn sun_angle() -> f32 {
    0.526_f32.to_radians()
}

/// Blender 5.2.1 Subdivision Surface render defaults.
pub(super) fn subdivision_render_levels() -> u32 { 2 }
pub(super) fn subdivision_limit() -> bool { true }


pub(super) fn spot_blend() -> f32 { SPOT_BLEND }
pub(super) fn spot_size() -> f32 { SPOT_SIZE }

/// The geometry chain default; absence preserves existing scene behavior.
pub fn geometry_stages()->Vec<super::GeometryStage> {super::GeometryStage::DEFAULT.to_vec()}
/// SVG 2 stroke-width initial value.
pub(super) fn stroke_width() -> f32 { 1. }
/// SVG 2 stroke-miterlimit initial value.
pub(super) fn miter_limit() -> f32 { 4. }
