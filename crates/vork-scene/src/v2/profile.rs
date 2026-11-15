//! Render profiles: named, versioned data files of settings-tier values.
//!
//! A profile never holds scene content. The render call picks one (default
//! `std/final`) and may add explicit values; precedence is built-in defaults <
//! profile < explicit values. Provider admission follows scene effects.
use super::*;

/// The default profile when a render call names none.
pub const DEFAULT_PROFILE: &str = "std/final";

/// User intent. Technique controls belong to the renderer.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
#[schemars(rename = "Settings")]
pub struct RenderSettings {
    /// Deterministic sampling seed, independent of the selected technique.
    #[schemars(extend("x-source" = "vork", "x-help-path" = "settings.seed", "x-help" = "u32; deterministic sampling seed"))]
    pub seed: u32,
    /// Override the profile sample budget; more samples means cleaner and slower.
    #[schemars(range(min = 1, max = 4096), extend("x-source" = "vork", "x-help-path" = "settings.samples", "x-help" = "u32; 1..4096; more samples means cleaner and slower"))]
    #[serde(deserialize_with = "deserialize_samples", skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub samples: Option<std::num::NonZeroU32>,
    /// Enable optional scene-linear OIDN. Off retains profile-owned reconstruction.
    #[schemars(extend("x-source" = "vork", "x-help-path" = "settings.denoise", "x-help" = "bool; optional OIDN on/off; off retains profile reconstruction"))]
    pub denoise: bool,
}

fn deserialize_samples<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<std::num::NonZeroU32>, D::Error> {
    let samples = std::num::NonZeroU32::deserialize(d)?;
    if samples.get() > constants::MAX_RENDER_SAMPLES {
        return Err(serde::de::Error::custom("samples must be an integer in 1..=4096"));
    }
    Ok(Some(samples))
}

/// Profile file contents.
#[derive(Clone, Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProfileFile {
    /// The profile's own revision, recorded in output metadata.
    #[schemars(extend("x-source" = "vork render profile", "x-help-path" = "profile.version", "x-help" = "u32"))]
    pub version: u32,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork render profile", "x-help-path" = "profile.description", "x-help" = "string"))]
    pub description: String,
    /// Quality/cost level, independent of the profile's name: 0 draft, 1 final, 2 reference, 3 previs (a separate blocking look).
    #[schemars(range(min = 0, max = 3), extend("x-source" = "vork", "x-help-path" = "profile.level", "x-help" = "u8; 0..3; quality/cost level; 3 is previs"))]
    pub level: u8,
    /// Sampling seed, sample budget and denoising intent.
    #[serde(default)]
    #[schemars(with = "RenderSettings")]
    #[schemars(extend("x-source" = "vork render profile", "x-help-path" = "profile.settings", "x-help" = "Settings; seed, samples and denoise only"))]
    pub settings: serde_json::Map<String, serde_json::Value>,
}

/// A loaded profile: its name (library `lib/name` or the file path given) and data.
#[derive(Clone, Debug)]
pub struct Profile {
    pub name: String,
    pub version: u32,
    pub description: String,
    pub settings: serde_json::Map<String, serde_json::Value>,
    pub level: u8,
}

impl Profile {
    /// A bundled library profile (`std/final`) or a local file path.
    pub fn load(spec: &str) -> Result<Self> {
        if let Some(text) = crate::libraries::profile(spec) {
            return Self::parse(spec, text.as_bytes());
        }
        let path = Path::new(spec);
        if path.extension().is_some_and(|e| e == "json") || path.exists() {
            let bytes = std::fs::read(path).map_err(|error| anyhow::anyhow!("profile {spec}: {error}"))?;
            return Self::parse(spec, &bytes);
        }
        anyhow::bail!("{} (vork help profiles)", diagnostics::unknown("profile", spec, crate::libraries::profile_names()))
    }
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Self> {
        let file: ProfileFile = serde_json::from_slice(diagnostics::without_bom(bytes))
            .map_err(|error| anyhow::anyhow!("profile {name}: {}", diagnostics::profile_error(diagnostics::without_bom(bytes), error)))?;
        // Settings errors name the profile, before any scene is involved.
        ensure!(file.level <= 3, "profile {name}: level must be 0, 1, 2 or 3 (std/draft, std/final, std/path, std/previs)");
        RenderSettings::deserialize(serde_json::Value::Object(file.settings.clone()))
            .map_err(|error| anyhow::anyhow!("profile {name}: settings: {error}; choose std/previs, std/draft, std/final or std/path (vork help profiles)"))?;
        Ok(Self { name: name.into(), version: file.version, description: file.description, settings: file.settings, level: file.level })
    }
    /// Built-in defaults < this profile < explicit render-call values (top-level keys replace).
    pub fn resolve(&self, explicit: &serde_json::Map<String, serde_json::Value>) -> Result<Settings> {
        let mut merged = self.settings.clone();
        merged.extend(explicit.clone());
        let promotion_denoise = merged.get("denoise").and_then(serde_json::Value::as_bool);
        let intent = RenderSettings::deserialize(serde_json::Value::Object(merged))
            .map_err(|error| anyhow::anyhow!("profile {}: {error}; choose std/previs, std/draft, std/final or std/path for quality", self.name))?;
        let mut settings = Settings::for_level(self.level, intent);
        settings.promotion_denoise = promotion_denoise;
        Ok(settings)
    }
}

/// The complete render-call field whitelist. Scene documents and tracks are
/// decoded without these values and are never rewritten by an override walker.
#[derive(Debug, Default)]
pub struct CallOverrides {
    pub settings: serde_json::Map<String, serde_json::Value>,
    pub resolution: Option<[u32;2]>,
    pub fps: Option<crate::Time>,
}
impl CallOverrides {
    pub const KEYS: &'static [&'static str] = &["resolution", "fps", "seed", "samples", "denoise"];
    pub fn parse(text:&str)->Result<(String,serde_json::Value)> {
        let (key,text)=text.split_once('=').context("expected KEY=VALUE")?;
        let value=if key=="resolution" && !text.starts_with('[') {
            let values=text.split(['x',',']).map(|v|v.trim().parse::<u32>()).collect::<std::result::Result<Vec<_>,_>>()?;
            serde_json::json!(values)
        } else {serde_json::from_str(text).with_context(||format!("--set {key}: expected a JSON value"))?};
        Self::resolve([(key.to_owned(),value.clone())])?;
        Ok((key.to_owned(),value))
    }
    pub fn resolve(values:impl IntoIterator<Item=(String,serde_json::Value)>)->Result<Self> {
        let mut result=Self::default();
        for (key,value) in values {
            match key.as_str() {
                "seed"=> {let _:u32=serde_json::from_value(value.clone()).context("--set seed: expected u32")?;result.settings.insert(key,value);}
                "samples"=> {
                    let samples:u32=serde_json::from_value(value.clone()).context("--set samples: expected integer in 1..=4096")?;
                    ensure!((1..=constants::MAX_RENDER_SAMPLES).contains(&samples), "--set samples: expected integer in 1..=4096");
                    result.settings.insert(key,value);
                }
                "denoise"=> {let _:bool=serde_json::from_value(value.clone()).context("--set denoise: expected bool")?;result.settings.insert(key,value);}
                "resolution"=> {
                    let size:[u32;2]=serde_json::from_value(value).context("--set resolution: expected [width,height] or WxH")?;
                    ensure!(size.iter().all(|v|*v>0),"--set resolution: dimensions must be positive");
                    result.resolution=Some(size);
                }
                "fps"=> {
                    let fps:crate::Time=serde_json::from_value(value).context("--set fps: expected positive number or rational string")?;
                    ensure!(fps>crate::Time::ZERO && fps.as_f32().is_finite(),"--set fps: must be positive and finite");result.fps=Some(fps);
                }
                _=>anyhow::bail!("--set {key}: allowed fields: {}; scene fields belong in the scene; quality uses --profile std/previs, std/draft, std/final or std/path", Self::KEYS.join(", ")),
            }
        }
        Ok(result)
    }
}

/// Internal test fixtures force providers without making them authoring settings.
#[cfg(any(test, feature = "test-support"))]
pub fn with_test_settings(scene: &str, settings: &str) -> serde_json::Result<serde_json::Value> {
    let mut scene: serde_json::Value = serde_json::from_str(scene)?;
    let mut settings: serde_json::Value = serde_json::from_str(settings)?;
    if let Some(fields) = settings.as_object_mut() { fields.entry("integrator").or_insert_with(|| "raster".into()); }
    scene["render"]["settings"] = settings;
    Ok(scene)
}

/// Resolve internal overrides for diagnostics compiled only with test support.
#[cfg(any(test, feature = "test-support"))]
pub fn test_settings(base: &Settings, overrides: &serde_json::Map<String, serde_json::Value>) -> Result<Settings> {
    let mut value = if let Some(integrator) = overrides.get("integrator") {
        serde_json::to_value(Settings::with_integrator(serde_json::from_value(serde_json::json!({"integrator":integrator}))?))?
    } else { serde_json::to_value(base)? };
    value.as_object_mut().unwrap().extend(overrides.clone());
    let mut settings: Settings = serde_json::from_value(value)?;
    if !overrides.contains_key("denoiser") {settings.promotion_denoise=base.promotion_denoise;}
    Ok(settings)
}

/// Testing aid: write a scene document holding preset `render.settings` as the
/// repo's convention, a scene file plus its sibling `X.profile.json`; returns that profile.
#[cfg(any(test, feature = "test-support"))]
pub fn write_authored(path: &Path, scene: &serde_json::Value) -> Result<Option<Profile>> {
    let mut scene = scene.clone();
    let settings = scene.get_mut("render").and_then(|r| r.as_object_mut()).and_then(|r| r.remove("settings"));
    std::fs::write(path, serde_json::to_vec(&scene)?)?;
    let Some(mut settings) = settings else { return Ok(None) };
    if let Some(fields) = settings.as_object_mut() { fields.entry("integrator").or_insert_with(|| "raster".into()); }
    let sibling = path.with_file_name(format!("{}.profile.json", path.file_stem().unwrap_or_default().to_string_lossy()));
    let forced = sibling.with_extension("test-settings.json");
    std::fs::write(forced, serde_json::to_vec(&settings)?)?;
    let level = if settings["integrator"] == "path" { 2 } else { 0 };
    let text = serde_json::to_vec(&serde_json::json!({"version": 1, "level":level, "settings": {"denoise":false}}))?;
    std::fs::write(&sibling, &text)?;
    Profile::parse(&sibling.to_string_lossy(), &text).map(Some)
}

/// Testing aid: replace a scene document's integrator settings, keeping its
/// sample budgets (as an integrator switch did before settings left the scene).
#[cfg(any(test, feature = "test-support"))]
pub fn replace_integrator(scene: &mut serde_json::Value, integrator: serde_json::Value) {
    let settings = &mut scene["render"]["settings"];
    let kept: serde_json::Map<_, _> = settings.as_object().into_iter().flatten()
        .filter(|(key, _)| matches!(key.as_str(), "seed" | "samples"))
        .map(|(key, value)| (key.clone(), value.clone())).collect();
    let mut replaced = integrator.as_object().cloned().unwrap_or_default();
    replaced.extend(kept);
    *settings = replaced.into();
}

impl Scene {
    /// The scene file's content: everything but the resolved render settings.
    pub fn content_json(&self) -> serde_json::Result<serde_json::Value> {
        let mut value = serde_json::to_value(self)?;
        value["render"].as_object_mut().map(|render| render.remove("settings"));
        Ok(value)
    }
    /// Apply resolved settings and check the scene's requirements against them.
    pub fn apply_settings(&mut self, settings: Settings, _profile: &str) -> Result<()> {
        self.render.settings = settings;
        self.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_strategies_are_level_based_and_inherited() {
        for level in 0..=3 {
            let profile=Profile::parse("renamed.json",format!(r#"{{"version":1,"level":{level},"settings":{{}}}}"#).as_bytes()).unwrap();
            for denoise in [false,true] {
                let intent=serde_json::from_value(serde_json::json!({"samples":64,"denoise":denoise})).unwrap();
                let settings=profile.resolve(&intent).unwrap();
                assert_eq!(settings.bounded_secondary_compilation,level==0);
                assert_eq!(settings.approximate_field_boundaries,level!=2);
                let mut scene=Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"}]}"#).unwrap();
                scene.apply_settings(settings.clone(),"renamed").unwrap();
                let capture:Image=serde_json::from_value(serde_json::json!({"render":{"resolution":[16,16]}})).unwrap();
                assert_eq!(capture.resolved_render(&scene.render).unwrap().settings.bounded_secondary_compilation,level==0);
                assert_eq!(capture.resolved_render(&scene.render).unwrap().settings.approximate_field_boundaries,level!=2);
                let mut fallback=settings;
                fallback.without_fast_denoiser(level,true);
                assert_eq!(fallback.bounded_secondary_compilation,level==0);
                assert_eq!(fallback.approximate_field_boundaries,level!=2);
            }
        }
        assert!(!Settings::default().bounded_secondary_compilation);
        assert!(!Settings::default().approximate_field_boundaries);
        assert!(!Profile::load(DEFAULT_PROFILE).unwrap().resolve(&Default::default()).unwrap().bounded_secondary_compilation);
        assert!(Profile::load(DEFAULT_PROFILE).unwrap().resolve(&Default::default()).unwrap().approximate_field_boundaries);
    }

    #[test]
    fn promotion_retains_explicit_denoise_intent() {
        for profile in ["std/previs", "std/draft", "std/final", "std/path"] {
            let profile=Profile::load(profile).unwrap();
            let default=profile.resolve(&Default::default()).unwrap();
            assert!(default.promotion_denoise.unwrap_or(true));
            for intent in [false,true] {
                let settings=profile.resolve(&serde_json::from_value(serde_json::json!({"denoise":intent})).unwrap()).unwrap();
                assert_eq!(settings.promotion_denoise,Some(intent));
            }
        }
        let profile=Profile::parse("custom.json",br#"{"version":1,"level":1,"settings":{"denoise":false}}"#).unwrap();
        assert_eq!(profile.resolve(&Default::default()).unwrap().promotion_denoise,Some(false));
    }

    #[test]
    fn previs_is_a_separate_look_with_inherited_capture_policy() {
        let settings=Profile::load("std/previs").unwrap().resolve(&Default::default()).unwrap();
        let raster=settings.integrator.raster().unwrap();
        assert!(settings.previs);
        assert!(raster.fast_gi.is_none() && raster.probes.is_none() && raster.denoiser.is_none());
        assert_eq!(raster.reflections, Reflections::Environment);
        assert_eq!(raster.refraction, Refraction::Auto);
        assert_eq!(settings.samples, constants::PREVIS_SAMPLES);
        let mut scene=Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"}]}"#).unwrap();
        scene.apply_settings(settings.clone(), "renamed-previs").unwrap();
        let capture:Image=serde_json::from_value(serde_json::json!({"render":{"resolution":[16,16]}})).unwrap();
        assert_eq!(capture.resolved_render(&scene.render).unwrap().settings, settings);
        let overrides=CallOverrides::resolve([("denoise".into(),serde_json::json!(true))]).unwrap();
        assert!(Profile::load("std/previs").unwrap().resolve(&overrides.settings).unwrap().integrator.denoiser().is_some());
    }

    #[test]
    fn final_oidn_default_and_reconstruction_opt_out_preserve_transport() {
        let profile=Profile::load(DEFAULT_PROFILE).unwrap();
        let default=profile.resolve(&Default::default()).unwrap();
        assert!(!default.reconstruct);
        assert!(default.integrator.denoiser().is_some());
        assert_eq!(default.samples, constants::FINAL_SAMPLES);
        let raw=profile.resolve(&serde_json::from_value(serde_json::json!({"denoise":false})).unwrap()).unwrap();
        assert!(raw.integrator.denoiser().is_none());
        assert!(raw.reconstruct);
        assert_eq!(raw.samples,default.samples);
        let clean=profile.resolve(&serde_json::from_value(serde_json::json!({"denoise":true})).unwrap()).unwrap();
        assert!(clean.integrator.denoiser().is_some());
        assert!(!clean.reconstruct);
        assert_eq!(clean.samples,default.samples);
        let mut scene=Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"}]}"#).unwrap();
        scene.apply_settings(default.clone(), "renamed-final").unwrap();
        let capture:Image=serde_json::from_value(serde_json::json!({"render":{"resolution":[16,16]}})).unwrap();
        assert_eq!(capture.resolved_render(&scene.render).unwrap().settings, default);
    }

    #[test]
    fn no_fast_denoiser_trades_oidn_for_samples_and_keeps_authored_budgets() {
        for (name, budget) in [("std/draft",constants::DRAFT_SAMPLES_WITHOUT_DENOISER),("std/final",constants::FINAL_SAMPLES_WITHOUT_DENOISER)] {
            let profile = Profile::load(name).unwrap();
            let mut settings = profile.resolve(&Default::default()).unwrap();
            settings.without_fast_denoiser(profile.level, false);
            assert!(settings.integrator.denoiser().is_none() && settings.reconstruct);
            assert_eq!(settings.samples, budget);
            let call = CallOverrides::resolve([("samples".into(), serde_json::json!(5))]).unwrap();
            let mut settings = profile.resolve(&call.settings).unwrap();
            settings.without_fast_denoiser(profile.level, true);
            assert!(settings.integrator.denoiser().is_none());
            assert_eq!(settings.samples, 5);
        }
    }

    #[test]
    fn shipped_budgets_and_sample_overrides() {
        for (name, budget) in [("std/draft",4),("std/final",8),("std/previs",4),("std/path",64)] {
            let profile = Profile::load(name).unwrap();
            let default = profile.resolve(&Default::default()).unwrap();
            assert_eq!(default.samples, budget);
            assert_eq!(default.integrator.denoiser().is_some(), name != "std/previs");
            let undenoised = profile.resolve(serde_json::json!({"denoise":false}).as_object().unwrap()).unwrap();
            assert_eq!(undenoised.samples, budget);
            for samples in [1, 7, 4096] {
                let call = CallOverrides::resolve([("samples".into(), serde_json::json!(samples))]).unwrap();
                let mut expected = default.clone();
                expected.samples = samples;
                assert_eq!(profile.resolve(&call.settings).unwrap(), expected);
            }
        }
        for denoise in [false, true] {
            let bytes = serde_json::to_vec(&serde_json::json!({"version":1,"level":2,"settings":{"denoise":denoise}})).unwrap();
            let custom = Profile::parse("custom", &bytes).unwrap();
            assert_eq!(custom.resolve(&Default::default()).unwrap().samples, 64);
        }
        let profile = Profile::parse("custom", br#"{"version":1,"level":1,"settings":{"samples":13}}"#).unwrap();
        assert_eq!(profile.resolve(&Default::default()).unwrap().samples, 13);
        let call = CallOverrides::resolve([("samples".into(), serde_json::json!(5))]).unwrap();
        assert_eq!(profile.resolve(&call.settings).unwrap().samples, 5);
        for value in [serde_json::json!(0), serde_json::json!(4097), serde_json::json!(null), serde_json::json!(1.5)] {
            let bytes = serde_json::to_vec(&serde_json::json!({"version":1,"level":1,"settings":{"samples":value}})).unwrap();
            assert!(Profile::parse("bad", &bytes).is_err());
        }
    }

    #[test]
    fn call_fields_are_explicit_and_typed() {
        for input in ["resolution=64x32","resolution=[64,32]","fps=24","seed=9","samples=1","samples=4096","denoise=true"] {CallOverrides::parse(input).unwrap();}
        for input in ["resolution=0x32","resolution=[1,2,3]","fps=0","samples=0","samples=-1","samples=1.5","samples=4097","samples=4294967296","samples=null","samples=true","seed=-1","denoise=1","frame_start=-1","frame_start=2","frame_end=8","camera=1","render.resolution=[1,2]","settings.seed=2","compositor=[]","exposure=2"] {assert!(CallOverrides::parse(input).is_err(),"{input}");}
    }
    #[test]
    fn profile_names_are_provenance_and_default_resolution_is_explicit() {
        assert_eq!(DEFAULT_PROFILE, "std/final");
        let empty = CallOverrides::resolve([]).unwrap().settings;
        for name in crate::libraries::profile_names() {
            let original = Profile::load(name).unwrap();
            let bytes = crate::libraries::profile(name).unwrap();
            let renamed = Profile::parse("arbitrary-renamed-profile.json", bytes.as_bytes()).unwrap();
            assert_ne!(original.name, renamed.name);
            assert_eq!(original.resolve(&empty).unwrap(), renamed.resolve(&empty).unwrap());
            for key in ["seed"] {
                let overrides = CallOverrides::resolve([(key.into(), serde_json::json!(7))]).unwrap();
                let resolved = original.resolve(&overrides.settings).unwrap();
                assert_eq!(resolved.seed, 7);
                assert_eq!(resolved, renamed.resolve(&overrides.settings).unwrap());
            }
        }
        let resolved = Profile::load(DEFAULT_PROFILE).unwrap().resolve(&empty).unwrap();
        let explicit_final = Profile::load("std/final").unwrap().resolve(&empty).unwrap();
        assert_eq!(resolved, explicit_final);
        // Bare Rust defaults are a separate API contract, not the default profile.
        assert_eq!(Settings::default().integrator.raster().unwrap().refraction, Refraction::Environment);
        assert_eq!(resolved.integrator.raster().unwrap().refraction, Refraction::Auto);
        let bare=Settings::default();
        assert!(bare.integrator.raster().unwrap().fast_gi.is_none());
        assert_eq!(bare.integrator.reflections(),Some(Reflections::Environment));
        for name in ["std/draft","std/final"] {
            let resolved=Profile::load(name).unwrap().resolve(&empty).unwrap();
            let raster=resolved.integrator.raster().unwrap();
            assert_eq!(raster.reflections,Reflections::Auto);
            assert_eq!(raster.fast_gi.as_ref().unwrap().method,FastGiMethod::Auto);
            assert!(raster.probes.is_none());
        }
    }

    #[test]
    fn intent_overrides_and_requirements() {
        let draft = Profile::load("std/draft").unwrap();
        let intent = CallOverrides::resolve([("seed".into(),serde_json::json!(42)),("denoise".into(),serde_json::json!(true))]).unwrap();
        let settings = draft.resolve(&intent.settings).unwrap();
        assert_eq!(settings.seed,42);
        assert!(settings.integrator.raster().unwrap().denoiser.is_some());
        let mut scene = Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"}]}"#).unwrap();
        scene.apply_settings(settings,&draft.name).unwrap();
        let path = Profile::load("std/path").unwrap();
        scene.apply_settings(path.resolve(&intent.settings).unwrap(), &path.name).unwrap();
        assert_eq!(scene.render.settings.seed,42);
        assert!(Profile::parse("bad",br#"{"version":1,"level":4}"#).is_err());
    }
    #[test]
    fn scene_files_carry_no_settings() {
        for text in [r#"{"version":2,"nodes":[{"type":"camera"}],"render":{"settings":{"seed":4}}}"#,
            r#"{"version":2,"nodes":[{"type":"camera"}],"render":{"samples":4}}"#] {
            let error = Scene::from_json(text.as_bytes()).unwrap_err().to_string();
            assert!(error.contains("--profile"), "{error}");
        }
    }
}

/// Normalize inline resource declarations without reading assets or validating a
/// complete scene, for CPU-only code-generation corpus tests.
#[cfg(any(test, feature = "test-support"))]
pub fn lower_inline_resources_for_test(value: &mut serde_json::Value) {
    super::inline::lower(value);
}

/// Test equivalent of the public scene reader's inline-resource lowering, with
/// internal render settings retained for provider-specific assertions.
#[cfg(any(test, feature = "test-support"))]
pub fn from_test_value<T: serde::de::DeserializeOwned>(mut value: serde_json::Value) -> serde_json::Result<T> {
    if std::any::type_name::<T>() == std::any::type_name::<super::Scene>() {
        super::inline::lower(&mut value);
    }
    serde_json::from_value(value)
}
#[cfg(any(test, feature = "test-support"))]
pub fn from_test_str<T: serde::de::DeserializeOwned>(value: &str) -> serde_json::Result<T> {
    from_test_value(serde_json::from_str(value)?)
}
