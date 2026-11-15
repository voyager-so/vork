use super::*;
/// An ordered image-graph step. Resource names remain shorthand for an unnamed step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum CompositeEntry {
    Material(#[schemars(with = "NameOr<Material>")] String),
    Node(CompositeNode),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompositeNode {
    #[schemars(with = "NameOr<Material>")]
    pub material: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork", "x-help-path" = "compositor.resolution", "x-help" = "[width,height] pixels or positive output scale; floor, minimum 1; null = output; last image resampled to output"))]
    pub resolution: Option<CompositeResolution>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum CompositeResolution { Absolute([u32; 2]), Relative(#[serde(deserialize_with="super::numeric")] f64) }
impl CompositeResolution {
    pub fn extent(&self, output: [u32; 2]) -> Result<[u32; 2]> {
        let size = match self {
            Self::Absolute(size) => *size,
            Self::Relative(scale) => {
                ensure!(scale.is_finite() && *scale > 0., "compositor resolution scale must be positive and finite");
                ensure!(output.iter().all(|v| f64::from(*v) * *scale <= f64::from(u32::MAX)), "compositor resolution overflows");
                output.map(|v| ((f64::from(v) * *scale).floor() as u32).max(1))
            }
        };
        ensure!(size.iter().all(|v| *v > 0), "compositor resolution must be nonzero");
        Ok(size)
    }
}
impl CompositeEntry {
    pub fn material(&self) -> &String { match self { Self::Material(m) => m, Self::Node(n) => &n.material } }
    pub fn name(&self) -> Option<&str> { match self { Self::Material(_) => None, Self::Node(n) => n.name.as_deref() } }
    pub fn extent(&self, output: [u32; 2]) -> Result<[u32; 2]> {
        match self { Self::Node(CompositeNode { resolution: Some(r), .. }) => r.extent(output), _ => Ok(output) }
    }
}
impl From<String> for CompositeEntry { fn from(s: String) -> Self { Self::Material(s) } }
impl From<&str> for CompositeEntry { fn from(s: &str) -> Self { Self::Material(s.into()) } }

/// Check normalized graph material slots before the untagged input decoder loses
/// the obsolete key. Inline materials and nested captures have already been hoisted.
pub(super) fn refuse_node_bindings(scene: &serde_json::Value) -> Result<()> {
    use serde_json::Value;
    fn node_compositions<'a>(nodes: Option<&'a Value>, out: &mut Vec<&'a Value>) {
        for node in nodes.and_then(Value::as_array).into_iter().flatten() {
            out.extend(node.get("composition"));
            node_compositions(node.get("children"), out);
        }
    }
    let mut compositions: Vec<&Value> = scene.get("composition").into_iter().chain(
        scene.get("images").and_then(Value::as_object).into_iter().flatten()
            .filter_map(|(_, image)| image.get("composition"))).collect();
    node_compositions(scene.get("nodes"), &mut compositions);
    for composition in compositions {
        for entry in composition.get("compositor").and_then(Value::as_array).into_iter().flatten() {
            let Some(name) = entry.get("material").unwrap_or(entry).as_str() else { continue };
            let inputs = scene.get("materials").and_then(|m| m.get(name))
                .and_then(|m| m.get("inputs")).and_then(Value::as_object);
            ensure!(!inputs.into_iter().flatten().any(|(_, input)| input.get("node").is_some()),
                "compositor {{node:NAME}} was renamed to {{step:NAME}}; see vork help upgrade");
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Render {
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "render.camera", "x-help" = "camera-node-name|null; null = the only authored camera, else first imported; a camera cut track may select"))]
    pub camera: Option<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "render.root", "x-help" = "node-name|null; only this subtree draws and is lit; objects outside still cast shadows; null = whole scene"))]
    pub root: Option<String>,
    /// Shader AOV declarations; color outputs contain scene-linear RGB.
    #[schemars(extend("x-source" = "Blender ViewLayer.aovs", "x-help-path" = "render.aovs", "x-help" = "[{name,type:value|color}]; output with --channels all or aov:<name>"))]
    pub aovs: Vec<Aov>,
    #[schemars(extend("x-source" = "Blender 5.2.1 RenderSettings.resolution_x/resolution_y", "x-help-path" = "render.resolution", "x-help" = "uvec2"))]
    pub resolution: [u32; 2],
    #[schemars(extend("x-source" = "Blender 5.2.1 RenderSettings.fps", "x-help-path" = "render.fps", "x-help" = "number|rational string; positive output frames/second"))]
    pub fps: crate::Time,
    #[schemars(extend("x-source" = "Blender 5.2.1 (frame range expressed in seconds)", "x-help-path" = "render.duration", "x-help" = "number|rational string; seconds"))]
    pub duration: vork_time::Clock,
    #[schemars(extend("x-source" = "Blender 5.2.1 render motion blur", "x-help-path" = "render.motion_blur", "x-help" = "MotionBlur|null; non-null enables; shutter in frames"))]
    pub motion_blur: Option<MotionBlur>,
    #[schemars(extend("x-source" = "vork (Blender RenderSettings.film_transparent plus a constant RGB)", "x-help-path" = "render.background", "x-help" = "environment|transparent|RGB|CSS colour; RGB and transparent apply to camera escapes and opacity passthrough; path glass transmission sees the World; a 2D canvas takes transparent or a CSS colour"))]
    pub background: Background,
    /// The image graph this render runs, resolved from its composition (`Scene::main_render`,
    /// `Image::resolved_render`). Authored as `composition.compositor`; empty in a decoded scene.
    #[serde(skip)]
    #[schemars(skip)]
    pub compositor: Vec<CompositeEntry>,
    /// How the shot is rendered, resolved from the profile and render-call values
    /// (`Profile::resolve`). Not scene content: scene files cannot carry it
    /// (`Scene::from_json` rejects it). Rust callers such as tests may preset it;
    /// Metadata records the resolved techniques.
    #[serde(default)]
    #[cfg_attr(not(any(test, feature="test-support")), serde(skip_deserializing))]
    #[schemars(skip)]
    pub settings: Settings,
    /// Final output quantization dither. Output encoding owns dither, so this is not scene content
    /// (`Scene::from_json` rejects it); Rust tests turn it off to assert exact codes.
    #[serde(skip_serializing_if = "is_true")]
    #[cfg_attr(not(any(test, feature="test-support")), serde(skip_deserializing))]
    #[schemars(skip)]
    pub dither: bool,
}
fn is_true(value: &bool) -> bool { *value }
impl Default for Render {
    fn default() -> Self {
        Self {
            camera: None,
            root: None,
            aovs: Vec::new(),
            resolution: constants::RESOLUTION,
            fps: constants::FPS,
            duration: constants::DURATION_SECONDS.parse().expect("default duration"),
            motion_blur: None,
            background: Background::Mode(BackgroundMode::Environment),
            compositor: Vec::new(),
            settings: Settings::default(),
            dither: true,
        }
    }
}
/// Internal resolved renderer techniques and budgets; never an authoring API.
/// Serialization records metadata; only tests deserialize provider choices.
#[derive(Clone, Debug, PartialEq, Serialize)]

pub struct Settings {
    /// Internal render-call intent; promotion may enable OIDN unless explicitly disabled.
    #[serde(skip)]
    pub promotion_denoise: Option<bool>,
    /// Draft-only native secondary compilation strategy; never an authored scene knob.
    #[serde(skip)]
    pub bounded_secondary_compilation: bool,
    /// Raster profiles use owning-cell admission instead of incident-neighborhood search.
    #[serde(skip)]
    pub approximate_field_boundaries: bool,
    /// Internal blocking-look policy, retained when an effect needs ray camera visibility.
    pub previs: bool,
    /// Internal scene-linear film normalization; independent of optional OIDN.
    pub reconstruct: bool,
    /// Camera integration budget shared with render-image captures.

    pub samples: u32,
    /// Internal sampling seed shared by all techniques.
    pub seed: u32,

    #[serde(flatten)]
    pub integrator: Integrator,
}
impl Default for Settings {
    fn default() -> Self { Self::with_integrator(Integrator::default()) }
}
impl Settings {
    /// Resolve quality/cost data into internal defaults; profile identity is irrelevant.
    pub fn for_level(level: u8, intent: RenderSettings) -> Self {
        let mut settings = Self::with_integrator(if level == 2 {
            Integrator::Path(PathIntegrator::default())
        } else { Integrator::Raster(RasterIntegrator::default()) });
        settings.samples = match level { 3 => constants::PREVIS_SAMPLES, 0 => constants::DRAFT_SAMPLES, 1 => constants::FINAL_SAMPLES, _ => constants::PATH_SAMPLES };
        if let Some(samples) = intent.samples { settings.samples = samples.get(); }
        settings.promotion_denoise = Some(intent.denoise);
        settings.seed = intent.seed;
        settings.bounded_secondary_compilation = level == 0;
        settings.approximate_field_boundaries = matches!(level,0|1|3);
        settings.previs = level == 3;
        settings.reconstruct = level == 1 && !intent.denoise;
        match &mut settings.integrator {
            Integrator::Path(path) => { path.denoiser = intent.denoise.then(Denoiser::default); }
            Integrator::Raster(raster) => {
                // Previs changes transport policy; camera optics, alpha and
                // scene refraction retain their existing providers.
                raster.fast_gi = (level != 3).then(|| FastGi { method: FastGiMethod::Auto, ..Default::default() });
                raster.reflections = if level == 3 { Reflections::Environment } else { Reflections::Auto };
                raster.refraction = Refraction::Auto;
                raster.probes = None;
                raster.denoiser = intent.denoise.then(Denoiser::default);
            }
        }
        settings
    }
    /// Stop denoising: retain film-filtered camera samples.
    pub fn disable_denoiser(&mut self) {
        self.promotion_denoise = Some(false);
        match &mut self.integrator {
            Integrator::Raster(raster) => { raster.denoiser = None; self.reconstruct = true; }
            Integrator::Path(path) => path.denoiser = None,
        }
    }
    /// The default look for a machine without a fast GPU denoiser: draft and final
    /// spend more samples instead. Authored sample budgets stay as they are.
    pub fn without_fast_denoiser(&mut self, level: u8, keep_samples: bool) {
        let promotion_denoise = self.promotion_denoise;
        self.disable_denoiser();
        self.promotion_denoise = promotion_denoise;
        if !keep_samples {
            self.samples = if level == 0 { constants::DRAFT_SAMPLES_WITHOUT_DENOISER } else { constants::FINAL_SAMPLES_WITHOUT_DENOISER };
        }
    }
    /// An integrator's settings with its default sample budget and aperture sampling.
    pub fn with_integrator(integrator: Integrator) -> Self {
        Self { promotion_denoise: None, bounded_secondary_compilation: false, approximate_field_boundaries: false, previs: false, reconstruct: false, seed: 0, samples: integrator.default_samples(), integrator }
    }
}
// Omitted `integrator` is raster; samples and aperture sampling default by integrator.
#[cfg(any(test, feature="test-support"))]
impl<'de> Deserialize<'de> for Settings {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        use serde::de::Error;
        let mut map = serde_json::Map::<String, serde_json::Value>::deserialize(d)?;
        let field = |map: &mut serde_json::Map<String, serde_json::Value>, name: &str| map.remove(name);
        let previs = field(&mut map, "previs").map(bool::deserialize).transpose().map_err(D::Error::custom)?.unwrap_or(false);
        let reconstruct = field(&mut map, "reconstruct").map(bool::deserialize).transpose().map_err(D::Error::custom)?.unwrap_or(false);
        let seed = field(&mut map, "seed").map(u32::deserialize).transpose().map_err(D::Error::custom)?.unwrap_or(0);
        let samples = field(&mut map, "samples").map(u32::deserialize).transpose().map_err(D::Error::custom)?;
        map.entry("integrator").or_insert_with(|| "raster".into());
        let value = serde_json::Value::Object(map);
        let integrator = Integrator::deserialize(&value)
            .map_err(|error| D::Error::custom(error))?;
        let defaults = Self::with_integrator(integrator);
        Ok(Self {
            promotion_denoise: Some(defaults.integrator.denoiser().is_some()),
            bounded_secondary_compilation: false, approximate_field_boundaries: false,
            previs,
            reconstruct,
            seed,
            samples: samples.unwrap_or(defaults.samples),
            integrator: defaults.integrator,
        })
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Background {
    Color(#[serde(deserialize_with="super::numeric")] [f32; 3]),
    Mode(BackgroundMode),
    /// A 2D canvas's background: a CSS colour behind every node.
    Css(String),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundMode {
    Environment,
    Transparent,
}
impl Background {
    pub fn transparent(&self) -> bool {
        matches!(self, Self::Mode(BackgroundMode::Transparent))
    }
    pub fn is_environment(&self) -> bool {
        matches!(self, Self::Mode(BackgroundMode::Environment))
    }
}
/// Indirect specular technique, independent of material roughness and metalness.
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Reflections {
    // Resolve scene reflection requirements before installation.
    Auto,
    #[default]
    Environment,
    // Shade the closest hit through scene ray queries.
    Ray,
}
/// Raster transmission intersection technique, independent of reflections.
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Refraction {
    // Prefiltered world lighting without scene tracing, as EEVEE with ray tracing disabled.
    #[default]
    Environment,
    Auto,
    Screen,
    Ray,
}
/// Light probes: sparse diffuse irradiance and filtered local reflection captures.
/// Irradiance replaces world diffuse light within supported receiver space.
/// Local directional captures supply glossy detail; escaped rays see the world.
/// Probes update only when lighting inputs
/// change (objects, lights, world, deformation, animated materials), never from
/// history, so a frame's probes depend only on the scene at that time.
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Probes {

    pub volume: Option<ProbeVolume>,

    pub gi_diffuse_bounces: u32,
}
impl Default for Probes {
    fn default() -> Self { Self { volume: Some(ProbeVolume::default()), gi_diffuse_bounces: 3 } }
}
/// Sparse local irradiance bricks allocated near receiver free space.
/// Axis resolutions bound vertex slots per canonical brick, not scene bounds.
/// Only vertices needed by receiver lookup stencils are resident.
/// Allocation is independent of camera visibility and capture scratch capacity.
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProbeVolume {

    pub resolution_x: u32,

    pub resolution_y: u32,

    pub resolution_z: u32,
    /// Ray budget per diffuse vertex and lighting generation, and per local
    /// directional capture and update. Captures update when lighting changes;
    /// unchanged scene inputs reuse the retained result.

    pub rays_per_probe: u32,

}
impl Default for ProbeVolume {
    fn default() -> Self { Self {
        resolution_x: 4, resolution_y: 4, resolution_z: 4, rays_per_probe: 2048,
    } }
}
impl ProbeVolume {
    pub fn resolution(&self) -> [u32; 3] { [self.resolution_x, self.resolution_y, self.resolution_z] }
}
/// Resolved transport provider.
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "integrator", rename_all = "snake_case")]

pub enum Integrator {

    Raster(RasterIntegrator),

    Path(PathIntegrator),
}
impl Default for Integrator {
    fn default() -> Self { Self::Raster(RasterIntegrator::default()) }
}
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PathIntegrator {
    /// Path defaults to a spatial homogeneous random walk with an index-matched
    /// diffuse boundary. Null retains local path reflection; raster promotes solid scattering to thickness.
    /// Thickness is a direct two-flux slab; random walks require closed solids.

    pub subsurface_method: Option<SubsurfaceMethod>,
    /// Optional scene-linear denoising before bloom, compositor and display conversion.
    /// Null disables OIDN entirely, including loading its runtime and guide storage.

    pub denoiser: Option<Denoiser>,
}
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct RasterIntegrator {

    /// Path defaults to a spatial homogeneous random walk with an index-matched
    /// diffuse boundary. Null retains local path reflection; raster promotes solid scattering to thickness.
    /// Thickness is a direct two-flux slab; random walks require closed solids.

    pub subsurface_method: Option<SubsurfaceMethod>,
    /// Optional scene-linear denoising before bloom, compositor and display conversion.
    /// Null disables OIDN entirely, including loading its runtime and guide storage.

    pub denoiser: Option<Denoiser>,
    /// Raster reconstruction uses EEVEE's radial Gaussian film filter at 1.5 px;
    /// path uses Cycles' separable Blackman–Harris at 1.5 px. One raster sample
    /// stays centered.

    pub fast_gi: Option<FastGi>,

    pub reflections: Reflections,

    /// Solid glass exits and opaque receivers use real geometry where supported.

    pub refraction: Refraction,

    pub probes: Option<Probes>,

    pub shadow_method: ShadowMethod,
}
impl Default for PathIntegrator {
    fn default() -> Self { Self {

            subsurface_method: Some(SubsurfaceMethod::RandomWalk),
            denoiser: Some(Denoiser::default()),
        } }
}
impl Default for RasterIntegrator {
    fn default() -> Self { Self {
            fast_gi: None,
            reflections: Reflections::default(),
            refraction: Refraction::default(),
            probes: None,
            shadow_method: ShadowMethod::default(),
            subsurface_method: None,
            denoiser: None,
        } }
}
/// Spatial subsurface transport technique, independent of material programs.
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SubsurfaceMethod { Thickness, RandomWalk }
impl Integrator {
    pub fn default_samples(&self) -> u32 { if self.is_path() {constants::PATH_SAMPLES} else {constants::RENDER_SAMPLES} }
    pub fn is_path(&self) -> bool {
        matches!(self, Self::Path(_))
    }
    pub fn path(&self) -> Option<&PathIntegrator> {
        if let Self::Path(p) = self {
            Some(p)
        } else {
            None
        }
    }
    pub fn raster(&self) -> Option<&RasterIntegrator> {
        if let Self::Raster(r) = self {
            Some(r)
        } else {
            None
        }
    }
    pub fn raster_mut(&mut self) -> Option<&mut RasterIntegrator> {
        if let Self::Raster(r) = self {
            Some(r)
        } else {
            None
        }
    }
    pub fn denoiser(&self) -> Option<&Denoiser> {
        match self {
            Self::Raster(r) => r.denoiser.as_ref(),
            Self::Path(p) => p.denoiser.as_ref(),
        }
    }
    pub fn subsurface_method(&self) -> Option<SubsurfaceMethod> {
        match self { Self::Raster(r) => r.subsurface_method, Self::Path(p) => p.subsurface_method }
    }
    // Path transport still uses native raster coverage for control outputs. Its
    // beauty samples lenses and shutters directly; it has no screen techniques.
    pub fn aa(&self) -> Antialiasing {
        if self.is_path() {Antialiasing::Msaa4} else {Antialiasing::Jitter}
    }
    pub fn fast_gi(&self) -> Option<&FastGi> {
        self.raster().and_then(|r| r.fast_gi.as_ref())
    }
    pub fn indirect(&self) -> Option<&FastGi> { self.fast_gi() }
    /// The irradiance volume, when probes and their volume are enabled.
    pub fn probe_volume(&self) -> Option<(ProbeVolume, u32)> {
        self.raster().and_then(|r| r.probes).and_then(|p| p.volume.map(|v| (v, p.gi_diffuse_bounces)))
    }
    pub fn reflections(&self) -> Option<Reflections> {
        self.raster().map(|r| r.reflections)
    }
    pub fn shadow_method(&self) -> ShadowMethod {
        self.raster().map_or(ShadowMethod::Ray, |r| r.shadow_method)
    }

}
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Debug, PartialEq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Antialiasing {
    Msaa4,
    #[default]
    Jitter,
}

#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShadowMethod {
    /// Renderer-owned triangle visibility for every shadowed local light.
    #[default]
    Auto,
    Ray,
}
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct FastGi {

    pub method: FastGiMethod,

}
impl Default for FastGi {
    fn default() -> Self {
        Self {
            method: FastGiMethod::Auto,
        }
    }
}
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Copy, Debug, Serialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FastGiMethod {
    /// Select supported retained diffuse lighting or scene queries; never replace
    /// offscreen indirect light with a screen-only estimator.
    #[default]
    Auto,
    Ray,
}

#[cfg(test)]
mod color_tests {
    use super::*;

    #[test]
    fn blender_render_look_suggests_display_programs() {
        let error = Scene::from_json(br#"{"version":2,"render":{"look":"None"}}"#).unwrap_err().to_string();
        assert!(error.contains("unknown field"), "{error}");
        assert!(error.contains("look") && error.contains("std/agx or blender/filmic"), "{error}");
    }

    #[test]
    fn retired_white_balance_uses_unknown_field_diagnostics() {
        for value in [serde_json::json!(null), serde_json::json!({"temperature":3200.,"tint":0.})] {
            let bytes = serde_json::to_vec(&serde_json::json!({"version":2,"render":{"white_balance":value}})).unwrap();
            let error = Scene::from_json(&bytes).unwrap_err().to_string();
            assert!(error.contains("unknown field") && error.contains("white_balance"), "{error}");
            assert!(error.contains("alid names"), "{error}");
        }
    }

    #[test]
    fn display_is_a_program_instance() {
        assert!(matches!(Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"}]}"#).unwrap().view(), NameOr::Inline(_)));
        let scene = Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"}],"composition":{"space":"3d","view":{"program":"std/standard","inputs":{"gamma":1.74}}}}"#).unwrap();
        scene.validate().unwrap();
        let NameOr::Name(name) = scene.view() else { panic!("inline lowering") };
        assert_eq!(scene.materials[&name].program, "std/standard");
        for field in ["view_transform", "exposure", "gamma", "dither_intensity", "look"] {
            assert!(serde_json::from_value::<Render>(serde_json::json!({field: 1})).is_err());
        }
    }

}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct MotionBlur {
    #[schemars(extend("x-source" = "Blender 5.2.1 motion_blur_shutter", "x-help-path" = "render.motion_blur.shutter", "x-help" = "number; frames"))]
    pub shutter: f32,
    #[schemars(extend("x-source" = "Blender 5.2.1 motion_blur_position", "x-help-path" = "render.motion_blur.position", "x-help" = "start|center|end"))]
    pub position: MotionBlurPosition,

}
impl Default for MotionBlur {
    fn default() -> Self {
        Self {
            shutter: constants::MOTION_BLUR_SHUTTER,
            position: MotionBlurPosition::Center,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MotionBlurPosition { Start, #[default] Center, End }
impl MotionBlurPosition {
    pub fn offset(self) -> f32 {
        match self {
            Self::Start => 0.5,
            Self::Center => 0.,
            Self::End => -0.5,
        }
    }
}

/// OIDN uses high quality, prefiltered albedo and shading-normal guides.
#[cfg_attr(any(test, feature="test-support"), derive(Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Denoiser {}

#[cfg(test)]
mod compositor_tests {
    use super::*;
    #[test]
    fn old_graph_binding_is_refused_with_one_upgrade_line() {
        use serde_json::json;
        for input in [json!({"node":"saved"}), json!({"node":"saved","step":"saved"})] {
            let material = json!({"program":"copy_program","inputs":{"source":input}});
            for entry in [json!("read"), json!({"material":"read"}),
                material.clone(), json!({"material":material})] {
                let graph = json!([{"name":"saved","material":"copy"}, entry]);
                for location in ["main", "capture", "inline-capture"] {
                    let mut scene = json!({"version":2,"nodes":[{"type":"camera"}],
                        "programs":{"copy_program":{"source":crate::v2::COPY_COMPOSITOR}},
                        "materials":{"copy":{"program":"copy_program"},"read":material}});
                    match location {
                        "main" => scene["composition"] = json!({"space":"3d","compositor":graph}),
                        "capture" => scene["images"] = json!({"capture":{"composition":{"compositor":graph},"render":{}}}),
                        _ => scene["composition"] = json!({"space":"3d","compositor":[{"program":"copy_program",
                            "inputs":{"source":{"image":{"composition":{"compositor":graph},"render":{}}}}}]}),
                    }
                    let error = Scene::from_json(&serde_json::to_vec(&scene).unwrap()).unwrap_err();
                    assert_eq!(format!("{error:#}"),
                        "compositor {node:NAME} was renamed to {step:NAME}; see vork help upgrade", "{location}");
                }
            }
        }
    }

    #[test]
    fn step_binding_has_one_serialized_spelling() {
        use serde_json::json;
        let input: Input = serde_json::from_value(json!({"step":"saved"})).unwrap();
        assert!(matches!(&input, Input::Step(binding) if binding.step == "saved"));
        assert_eq!(serde_json::to_value(input).unwrap(), json!({"step":"saved"}));
        for value in [json!({"node":"saved"}), json!({"step":"saved","node":"saved"}),
            json!({"step":"saved","image":"saved"}), json!({"step":1})] {
            assert!(serde_json::from_value::<Input>(value).is_err());
        }
    }

    #[test]
    fn step_names_resolve_within_each_capture_graph() {
        use serde_json::json;
        let graph = json!([{"name":"saved","material":"copy"}, "read"]);
        let mut document = json!({"version":2,"nodes":[{"type":"camera","name":"saved"}],
            "programs":{"copy_program":{"source":crate::v2::COPY_COMPOSITOR}},
            "materials":{"copy":{"program":"copy_program"},
                "read":{"program":"copy_program","inputs":{"source":{"step":"saved"}}}},
            "composition":{"space":"3d","compositor":graph},"render":{}, "images":{"capture":{"composition":{"compositor":graph},"render":{}}}});
        let scene = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap();
        assert!(matches!(&scene.materials["read"].inputs["source"], Input::Step(b) if b.step == "saved"));
        // Neither a main-graph name nor a scene-node name makes a capture edge valid.
        document["images"]["capture"]["composition"]["compositor"] = json!(["read"]);
        let error = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap_err();
        assert!(format!("{error:#}").contains("step saved must name an earlier entry"), "{error:#}");
    }

    #[test]
    fn graph_names_are_backward_and_local_and_extents_are_checked() {
        use serde_json::json;
        let scene = |entries: serde_json::Value| Scene::from_json(&serde_json::to_vec(&json!({
            "version":2,"programs":{"copy_program":{"source":crate::v2::COPY_COMPOSITOR}},"nodes":[{"type":"camera"}],
            "composition":{"space":"3d","compositor":entries},"render":{"resolution":[65,33]},
            "materials":{"copy":{"program":"copy_program"},
                "read":{"program":"copy_program","inputs":{"source":{"step":"saved"}}}}
        })).unwrap());
        let valid=scene(json!([{"material":"copy","name":"saved","resolution":0.5},"read"])).unwrap();
        assert_eq!(valid.compositor()[0].extent([65,33]).unwrap(),[32,16]);
        for entries in [
            json!(["read",{"material":"copy","name":"saved"}]),
            json!([{"material":"read","name":"saved"}]),
            json!([{"material":"copy","name":"saved"},{"material":"copy","name":"saved"}]),
            json!([{"material":"copy","name":""}]),
            json!([{"material":"copy","resolution":0}]),
            json!([{"material":"copy","resolution":[0,1]}]),
        ] { assert!(scene(entries).is_err()); }
        assert_eq!(CompositeResolution::Relative(0.001).extent([65,33]).unwrap(),[1,1]);
        assert_eq!(CompositeResolution::Relative(0.7).extent([1920,1080]).unwrap(),[1344,756]);
        assert!(CompositeResolution::Relative(f64::MAX).extent([65,33]).is_err());
        assert!(serde_json::from_value::<Input>(json!({"pass":"saved"})).is_err());
    }
    #[test]
    fn authored_compositors_are_ordered_materials_without_render_aliases() {
        assert!(serde_json::from_str::<Render>(r#"{"glare":{}}"#).is_err());
        let mut scene: Scene = serde_json::from_value(serde_json::json!({
            "version":2,"programs":{"copy_program":{"source":crate::v2::COPY_COMPOSITOR}},"nodes":[{"type":"camera"}],
            "materials":{"bloom":{"program":"copy_program","inputs":{"original":{"channel":"color"},"source":{"channel":"color"}}},"grade":{"program":"copy_program"}},
            "composition":{"space":"3d","compositor":["bloom","grade"]},"render":{}
        })).unwrap();
        scene.validate().unwrap();
        *scene.compositor_mut() = vec!["grade".into(), "bloom".into()];
        scene.validate().unwrap();
        *scene.compositor_mut() = vec!["bloom".into(), "bloom".into()];
        scene.validate().unwrap();
    }    #[test]
    fn aov_names_are_plain_channel_tokens() {
        let scene = |name: &str| -> Scene { serde_json::from_value(serde_json::json!({
            "version":2,"nodes":[{"type":"camera"}],
            "render":{"aovs":[{"name":name,"type":"value"}]}
        })).unwrap() };
        scene("heat_map-2").validate().unwrap();
        for name in ["", "a,b", "a.b", "a b", "a:b"] {
            assert!(scene(name).validate().is_err(), "{name:?}");
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Aov {
    #[schemars(extend("x-source" = "Blender ViewLayer AOV.name", "x-help-path" = "render.aovs.name", "x-help" = "name; EXR channels <name>.X or <name>.R/G/B"))]
    pub name: String,
    #[serde(rename="type")]
    #[schemars(extend("x-source" = "Blender ViewLayer AOV.type", "x-help-path" = "render.aovs.type", "x-help" = "value|color"))]
    pub kind: AovType,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all="snake_case")]
pub enum AovType { Value, Color }

#[cfg(test)]
mod beta_token_tests {
    use super::*;

    #[test]
    fn fast_gi_rejects_removed_settings_and_path_transport() {
        for field in ["ao", "indirect"] {
            let error = serde_json::from_value::<Settings>(serde_json::json!({field:{}})).unwrap_err().to_string();
            assert!(error.contains(field) && error.contains("unknown field"), "{error}");
        }
        assert!(serde_json::from_value::<Settings>(serde_json::json!({"integrator":"path","fast_gi":{}})).is_err());
        for field in ["fast_gi_ray_count", "fast_gi_step_count"] {
            assert!(serde_json::from_value::<FastGi>(serde_json::json!({field:2})).is_err());
        }
    }

    #[test]
    fn ao_strength_is_rejected_and_null_disables() {
        let error = serde_json::from_value::<Settings>(serde_json::json!({"fast_gi":{"strength":1}})).unwrap_err().to_string();
        assert!(error.contains("unknown field") && error.contains("strength"), "{error}");
        let disabled: Settings = serde_json::from_value(serde_json::json!({"fast_gi":null})).unwrap();
        assert!(disabled.integrator.indirect().is_none());
    }

    #[test]
    fn reflections_reject_null_and_preserve_omitted_default() {
        let omitted: Integrator = serde_json::from_value(serde_json::json!({"integrator":"raster"})).unwrap();
        assert_eq!(omitted.reflections(), Some(Reflections::Environment));
        assert!(serde_json::from_value::<Reflections>(serde_json::json!("screen")).is_err());
        for (token, expected) in [("auto", Reflections::Auto), ("ray", Reflections::Ray), ("environment", Reflections::Environment)] {
            let authored: Integrator = serde_json::from_value(serde_json::json!({"integrator":"raster", "reflections":token})).unwrap();
            assert_eq!(authored.reflections(), Some(expected));
        }
        let error = serde_json::from_value::<Settings>(serde_json::json!({"reflections":null})).unwrap_err().to_string();
        assert!(error.contains("invalid type"), "{error}");
        assert!(serde_json::from_value::<Integrator>(serde_json::json!({"integrator":"raster","reflections":"off"})).is_err());
    }

    #[test]
    fn removed_probe_settings_name_the_rejected_fields() {
        for (field, value) in [("visibility_fallback", "dark"), ("visibility_method", "ray"), ("emitter_sampling", "mis"), ("ray_correction", "extra_bounce"), ("screen_correction", "extra_bounce")] {
            let error = serde_json::from_value::<crate::Scene>(serde_json::json!({
                "version": 2, "render": {"settings": {"integrator": "raster", "probes": {"volume": {field: value}}}}
            })).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn removed_path_settings_name_the_rejected_fields() {
        for (field, value) in [("environment_method", "source"), ("coat_method", "albedo"), ("volume_method", "sampled")] {
            let error = serde_json::from_value::<crate::Scene>(serde_json::json!({
                "version": 2, "render": {"settings": {"integrator": "path", field: value}}
            })).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn removed_camera_distortion_names_the_rejected_field() {
        for value in [serde_json::json!({"k1": 0.1}), serde_json::Value::Null] {
            let error = serde_json::from_value::<crate::Scene>(serde_json::json!({
                "version": 2, "nodes": [{"type": "camera", "distortion": value}]
            })).unwrap_err();
            assert!(error.to_string().contains("distortion"), "{error}");
        }
    }

    #[test]
    fn enclosing_defaults_use_the_technique_enum_defaults() {
        let render: Render = serde_json::from_str("{}").unwrap();
        assert_eq!(render.settings.samples, 64);
        let raster = render.settings.integrator.raster().unwrap();
        assert_eq!(raster.refraction, Refraction::default());
        assert_eq!(raster.shadow_method, ShadowMethod::default());
        assert!(raster.fast_gi.is_none());
    }

}

#[cfg(test)]
mod subsurface_method_tests {
    use super::*;
    #[test]
    fn thickness_is_explicit_and_shared_between_integrators() {
        for kind in ["raster", "path"] {
            let default:Integrator=serde_json::from_value(serde_json::json!({"integrator":kind})).unwrap();
            assert_eq!(default.subsurface_method(),if kind=="path" {Some(SubsurfaceMethod::RandomWalk)} else {None});
            let authored:Integrator=serde_json::from_value(serde_json::json!({"integrator":kind,"subsurface_method":"thickness"})).unwrap();
            assert_eq!(authored.subsurface_method(),Some(SubsurfaceMethod::Thickness));
            assert_eq!(serde_json::from_value::<Integrator>(serde_json::to_value(&authored).unwrap()).unwrap(),authored);
            assert!(serde_json::from_value::<Integrator>(serde_json::json!({"integrator":kind,"subsurface_method":"random_walk"})).is_ok());
        }
    }
}
