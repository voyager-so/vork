//! `vork setup` compiles the bundled profiles' reachable program and material
//! configurations. Library roles come from their stage declarations; uniform
//! and varying medium inputs share the normal compilation and cache paths.
use super::*;
use ::vork_scene::v2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Role {
    Surface,
    Hair,
    Medium,
    Environment,
    Camera,
    Composite,
    Display,
}

/// The role a std program's entry point declares; a module without one of these entries
/// is a surface.
pub(crate) fn role(source: &str) -> Role {
    let declares = |ret: &str, entry: &str, context: &str| {
        source.lines().any(|line| {
            let line = line.trim_start().trim_start_matches("public ");
            line.starts_with(ret) && line[ret.len()..].trim_start().starts_with(entry)
                && line.contains(context)
        })
    };
    if declares("CameraRay", "camera(", "CameraContext") {
        Role::Camera
    } else if declares("float3", "environment(", "EnvironmentContext") {
        Role::Environment
    } else if declares("MediumCoefficients", "medium(", "MediumContext") {
        Role::Medium
    } else if source.contains("CompositeContext") && declares("float4", "composite(", "CompositeContext") {
        Role::Composite
    } else if declares("ViewResult", "view(", "ViewContext") {
        Role::Display
    } else if declares("Hair", "surface(", "ShadingContext") {
        Role::Hair
    } else {
        Role::Surface
    }
}

/// The warm scenes for these std programs (each scene before its frame configurations).
pub(crate) fn scenes(r: &Renderer, programs: impl Iterator<Item = &'static str>) -> Result<Vec<serde_json::Value>> {
    let mut scenes = Vec::new();
    for name in programs {
        let source = vork_slang::module_source(name).with_context(|| format!("std module {name}"))?;
        scenes.extend(program_scenes(r, name, source)?);
    }
    Ok(scenes)
}

/// Scenes without programs: the built-in material table, opaque, as a transmissive
/// solid (refraction tracks its boundary; transmission changes every hit kernel), and
/// that solid beside a subsurface material (thickness subsurface transport).
pub(crate) fn table_scenes(r: &Renderer) -> Vec<serde_json::Value> {
    let triangle = |material: &str| serde_json::json!({"type": "mesh", "geometry": "triangle", "materials": [material]});
    let solid = serde_json::json!({"type": "mesh", "geometry": "box", "materials": ["glass"], "translation": [0, 0, -1]});
    let materials = serde_json::json!({"table": {"inputs": {"base_color": [0.8, 0.8, 0.8]}},
        "glass": {"inputs": {"transmission_weight": 1.0}},
        "skin": {"inputs": {"subsurface_weight": 0.5}}});
    let meshes = [vec![triangle("table")], vec![solid.clone()], vec![solid, triangle("skin")]];
    meshes.iter().flat_map(|meshes| shadows(r).into_iter().map(|sun| {
        let mut scene = scene(serde_json::json!([]), serde_json::Value::Null,
            meshes.clone(), serde_json::json!({"type": "camera", "name": "c", "translation": [0,0,3]}),
            serde_json::json!({"type": "light", "name": "world", "shape": "environment"}), materials.clone(), sun);
        scene["geometry"]["box"] = serde_json::json!({"std":"rounded_box","radius":0,"size": [1, 1, 1]});
        scene
    })).collect()
}

/// A process-owned, ordinary image binding shared by setup configurations.
fn setup_image() -> Result<PathBuf> {
    static IMAGE_LOCK:std::sync::Mutex<()>=std::sync::Mutex::new(());
    let _lock=IMAGE_LOCK.lock().unwrap();
    let file=std::env::temp_dir().join(format!("vork-setup-image-{}.png",std::process::id()));
    if !file.exists() {image::RgbaImage::from_pixel(4,4,image::Rgba([255;4])).save(&file)?;}
    Ok(file)
}

/// Process-owned one-point Gaussian captures, at SH degree 0 and 3 (the
/// standard 3DGS export), so setup reaches the kernels splat renders compile.
fn setup_splats() -> Result<Vec<PathBuf>> {
    static SPLAT_LOCK:std::sync::Mutex<()>=std::sync::Mutex::new(());
    let _lock=SPLAT_LOCK.lock().unwrap();
    [0usize, 45].into_iter().map(|rest| {
        let file=std::env::temp_dir().join(format!("vork-setup-splat-{rest}-{}.ply",std::process::id()));
        if !file.exists() {
            let mut ply=String::from("ply\nformat ascii 1.0\nelement vertex 1\n");
            let names=["x","y","z","scale_0","scale_1","scale_2","opacity","rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"];
            for name in names.iter().map(|n|n.to_string()).chain((0..rest).map(|i|format!("f_rest_{i}"))) {
                ply.push_str(&format!("property float {name}\n"));
            }
            // A unit-quaternion, half-opaque, grey point at the origin.
            ply.push_str("end_header\n0 0 0 -2 -2 -2 0 1 0 0 0 0.5 0.5 0.5");
            for _ in 0..rest {ply.push_str(" 0");}
            ply.push('\n');
            std::fs::write(&file,ply)?;
        }
        Ok(file)
    }).collect()
}

/// One minimal setup scene.
fn scene(compositor: serde_json::Value, motion_blur: serde_json::Value, meshes: Vec<serde_json::Value>, camera: serde_json::Value,
    world: serde_json::Value, materials: serde_json::Value, sun: serde_json::Value) -> serde_json::Value {
    let mut nodes = meshes;
    nodes.extend([camera, serde_json::json!({"type":"light","name":"sun","shape":"sun"}), world]);
    serde_json::json!({"version": 2,
        "composition":{"space":"3d","compositor":compositor},"render": {"resolution": [64, 64], "motion_blur": motion_blur, "settings": {"samples":1,"shadow_method":sun}},
        "buffers": {"p": {"type": "vec3", "data": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
            "s": {"type": "vec3", "data": [[0, -1, 0], [0, 1, 0]]},
            "w": {"type": "f32", "data": [0.1, 0.1]}, "c": {"type": "u32", "data": [2]}},
        "geometry": {"triangle": {"attributes": {"POSITION": "p"}},
            "strand": {"topology": "curves", "attributes": {"POSITION": "s"}, "widths": "w", "curve_vertex_counts": "c"}},
        "nodes": nodes,
        "materials": materials})
}

/// The scenes of one library program, in the role its entry point declares.
fn program_scenes(r: &Renderer, name: &'static str, source: &str) -> Result<Vec<serde_json::Value>> {
    if name == crate::splats::PROGRAM {
        // Its inputs are SH attributes only a capture supplies: the importer binds
        // them, promotes to path transport and selects the Gaussian intersection
        // kernels, exactly as a render of a splat asset does. Splats need ray queries.
        if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) { return Ok(Vec::new()); }
        return setup_splats()?.into_iter().map(|file| Ok(scene(serde_json::json!([]), serde_json::Value::Null,
            vec![serde_json::json!({"type": "asset", "name": "m", "file": file})],
            serde_json::json!({"type": "camera", "name": "c", "translation": [0,0,3]}),
            serde_json::json!({"type": "light", "name": "world", "shape": "environment"}),
            serde_json::json!({}), serde_json::json!("auto")))).collect();
    }
    let mut scenes = Vec::new();
    let mut materials = serde_json::Map::new();
    let mut light = serde_json::json!({"type": "light", "name": "world", "shape": "environment"});
    let mut mesh = serde_json::json!({"type": "mesh", "name": "m", "geometry": "triangle"});
    let mut compositor = serde_json::json!([]);
    let mut camera = serde_json::json!({"type": "camera", "name": "c", "translation": [0,0,3]});
    let mut images=serde_json::json!({});
    let material = true;
    let motion_blur = serde_json::Value::Null;
    let mut variants = vec![serde_json::json!({})];
    let mut suns = vec![serde_json::json!("auto")];
    match role(source) {
        Role::Surface => mesh["materials"] = serde_json::json!(["std"]),
        Role::Hair => mesh = serde_json::json!({"type": "mesh", "name": "m", "geometry": "strand", "materials": ["std"]}),
        Role::Camera => camera["material"]="std".into(),
        Role::Environment => {
            light["material"]="std".into();
            let program=environment_program::compiled(r,source,&PathBuf::from(format!("{name}.slang")))?;
            let mut inputs=serde_json::Map::new();
            for field in program.inputs.fields.iter().filter(|field|field.ty==vork_slang::Type::Image) {
                let file=setup_image()?;
                images["setup"]=serde_json::json!({"file":file,"color_space":"Non-Color","wrap_t":"clamp_to_edge"});
                inputs.insert(field.name.clone(),serde_json::json!({"image":"setup"}));
            }
            variants=vec![serde_json::Value::Object(inputs)];
        },
        Role::Medium => {
            light["medium"] = "std".into();
            // Varying scalars select the nonuniform kernel; defaults may be uniform.
            let program = medium::compiled(r, source, &PathBuf::from(format!("{name}.slang")))?;
            let varied: serde_json::Map<_, _> = program.inputs.fields.iter()
                .filter(|f| matches!(f.ty, vork_slang::Type::Float) && f.enumeration.is_none()
                    && f.range.as_ref().is_none_or(|range| range.min <= 0.5 && 0.5 <= range.max))
                .map(|f| (f.name.clone(), serde_json::json!(0.5)))
                .collect();
            variants.push(serde_json::Value::Object(varied));
            // Ray-traced shadows compile their own volume kernels.
            suns = shadows(r);
        }
        Role::Composite => {
            compositor=serde_json::json!(["std"]);

        },
        Role::Display => {},
    }
    for (inputs, sun) in variants.iter().flat_map(|inputs| suns.iter().map(move |sun| (inputs, sun))) {
        if material {
            materials.insert("std".into(), serde_json::json!({"program": name, "inputs": inputs}));
        }
        let mut value=scene(compositor.clone(), motion_blur.clone(), vec![mesh.clone()], camera.clone(), light.clone(),
            serde_json::Value::Object(materials.clone()), sun.clone());
        value["images"]=images.clone();
        scenes.push(value);
    }
    if role(source) == Role::Display {
        for scene in &mut scenes { scene["composition"] = serde_json::json!({"space":"3d","view":"std"}); }
    }
    Ok(scenes)
}

#[cfg(test)]
fn authored_medium_scenes(r: &Renderer) -> Result<Vec<serde_json::Value>> {
    let source=include_str!("../../vork-scene/tests/fixtures/medium.slang");
    let mut scenes=program_scenes(r,"test-medium",source)?;
    for scene in &mut scenes {
        scene["programs"]["test-medium"]=serde_json::json!({"source":source});
    }
    Ok(scenes)
}

/// Warm the triangle shadow provider where the device can execute it.
fn shadows(r:&Renderer)->Vec<serde_json::Value> {
    if r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {vec![serde_json::json!("ray")]} else {Vec::new()}
}

/// Compile each scene in every frame configuration that selects different passes:
/// what each bundled profile resolves to, both raster coverage counts, with and without
/// screen AO (which selects the unfused colour resolve).
pub(crate) fn compile(r: &Renderer, scenes: &[serde_json::Value]) -> Result<()> {
    run(r, compile_jobs(scenes)?.into_iter(), InstallOptions { variants: pipelines::VariantCompile::Setup, ..Default::default() })
}

/// Frame configurations: the profiles' own settings (a render's, before the renderer
/// resolves automatic techniques for the device), and the raster technique axes.
fn configurations() -> Result<Vec<v2::Settings>> {
    let mut settings: Vec<_> = vork_scene::libraries::profile_names()
        .map(|name|v2::Profile::load(name)?.resolve(&Default::default())).collect::<Result<_>>()?;
    for name in ["std/draft", "std/final"] {
        settings.push(v2::Profile::load(name)?.resolve(
            &serde_json::from_value(serde_json::json!({"denoise":false}))?)?);
    }
    Ok(settings)
}

fn compile_jobs(scenes: &[serde_json::Value]) -> Result<Vec<(serde_json::Value, v2::Settings)>> {
    let configurations = configurations()?;
    Ok(scenes.iter().flat_map(|scene| configurations.iter().map(|c| (scene.clone(), c.clone()))).collect())
}

/// A setup job's scene with its resolved settings, validated as a render would.
fn prepare(mut value: serde_json::Value, mut settings: v2::Settings) -> Result<v2::Scene> {
    let authored=value["render"].as_object_mut().unwrap().remove("settings").unwrap();
    if let Some(raster)=settings.integrator.raster_mut() {
        raster.shadow_method=match authored["shadow_method"].as_str() {
            Some("ray")=>v2::ShadowMethod::Ray,
            _=>raster.shadow_method,
        };
    }
    // Previs has a small shipped budget; retain it so setup reaches the merged
    // visibility path as well as the transmitting camera variants it will draw.
    if settings.integrator.is_path() { settings.samples=1; }
    vork_std::expand_scene(&mut value, Path::new("."))?;
    let mut scene = v2::Scene::from_json(&serde_json::to_vec(&value)?)?;
    scene.render.settings=settings;
    scene.validate()?;
    Ok(scene)
}

/// Install each (scene, integrator) and construct its frame resources. Constructors
/// compile all passes; submission would evaluate arbitrary library inputs as an
/// infinite world, which may have no finite emission integral. Setup certifies
/// compilation, not the semantics of those synthetic scenes.
/// Bound live installs: each job already fans out into compiler/driver workers.
/// A configuration the renderer refuses for a scene (path transport of an unbounded
/// sampled world medium) is one no render compiles: setup skips it.
fn run(r: &Renderer, jobs: impl Iterator<Item = (serde_json::Value, v2::Settings)>, options: InstallOptions) -> Result<()> {
    let jobs:Vec<_>=jobs.collect();
    let phase=if jobs.iter().all(|(_,settings)|settings.integrator.is_path()) {
        "Setup: warming path configurations"
    } else {"Setup: warming configurations"};
    vork_slang::progress::phase(phase,jobs.len());
    let jobs = std::sync::Mutex::new(jobs.into_iter());
    let completed = std::sync::atomic::AtomicUsize::new(0);
    // One configuration's failure does not stop the others warming; every
    // failure is reported, and the first keeps its error and failure kind.
    let failures = std::sync::Mutex::new(Vec::<(String, anyhow::Error)>::new());
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(4);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..workers).map(|_| scope.spawn(|| -> Result<()> {
            loop {
                let Some((scene, integrator)) = jobs.lock().unwrap().next() else { break };
                let name = scene["materials"]["std"]["program"].as_str().map(str::to_owned)
                    .or_else(|| scene["nodes"].as_array().unwrap().iter().any(|node| node["type"] == "asset")
                        .then(|| crate::splats::PROGRAM.to_owned()))
                    .unwrap_or_else(|| "built-in materials".into());
                let explicit_camera = scene["nodes"].as_array().unwrap().iter()
                    .any(|node| node["type"] == "camera" && !node["material"].is_null());
                // The explicit standard-camera jobs retain the broad hit keys.
                // Ordinary table jobs warm the smaller material-compatible keys
                // through the same install constructors, without extra installs.
                let variants = if explicit_camera && matches!(options.variants, pipelines::VariantCompile::SetupPlain) {
                    pipelines::VariantCompile::Setup
                } else { options.variants.clone() };
                let compile = || -> Result<()> {
                    let scene = prepare(scene, integrator)?;
                    let installed = r.install_with_options(scene, Path::new("."), InstallOptions {
                        variants, readback: options.readback, gpu_color: options.gpu_color,
                        ..Default::default() })?;
                    // Volume, resolve and composite pipelines belong to the frame slot.
                    let _frame = installed.create_frame_slot(r, Default::default())?;
                    Ok(())
                };
                match compile() {
                    // Sampled transport refuses a library medium it cannot bound; every other refusal is a failure.
                    Err(error) if crate::failure::cause::<crate::medium::SampledRefusal>(&error).is_some() => {}
                    Err(error) => failures.lock().unwrap().push((name, error)),
                    Ok(()) => {}
                }
                vork_slang::progress::completed(completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
            }
            Ok(())
        })).collect();
        workers.into_iter().try_for_each(pipelines::join)
    })?;
    let mut failures = failures.into_inner().unwrap().into_iter();
    let Some((name, error)) = failures.next() else { return Ok(()) };
    let others: Vec<String> = failures.map(|(name, error)| format!("{name}: {error:#}")).collect();
    Err(if others.is_empty() { error.context(format!("setup pipelines for {name}")) } else {
        error.context(format!("setup pipelines for {name}; {} more setup configuration(s) also failed: {}", others.len(), others.join("; ")))
    })
}

/// The exhaustive setup catalogue includes table materials with both camera links.
fn warm_raster_tables(r: &Renderer) -> Result<()> {
    let Some(opaque)=table_scenes(r).into_iter().next() else {return Ok(());};
    let mut scene=opaque.clone();
    if r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES) {
        scene["materials"]["glass"]["inputs"]["geometry_thin_walled"]=true.into();
        scene["nodes"].as_array_mut().unwrap().push(serde_json::json!({"type":"mesh","geometry":"box","materials":["glass"],"translation":[0,0,-1]}));
    }
    let mut jobs=Vec::new();
    for profile in ["std/draft", v2::DEFAULT_PROFILE, "std/previs"] {
        let settings=v2::Profile::load(profile)?.resolve(&Default::default())?;
        // Transmission changes the camera/shadow registry. Warm the opaque-only
        // default and previs variants too, so an ordinary first clip does not compile them.
        jobs.push((opaque.clone(),settings.clone()));
        jobs.push((scene.clone(),settings));
    }
    // Authored physical cameras use the bundled camera program and its ray
    // evaluator; warming only the implicit camera leaves a cold first clip.
    let camera_jobs:Vec<_>=jobs.iter().map(|(scene,settings)| {
        let mut scene=scene.clone();
        scene["materials"]["setup_camera"]=serde_json::json!({"program":"std/camera"});
        let camera=scene["nodes"].as_array_mut().unwrap().iter_mut().find(|n|n["type"]=="camera").unwrap();
        camera["material"]="setup_camera".into();
        (scene,settings.clone())
    }).collect();
    jobs.extend(camera_jobs);
    run(r,jobs.into_iter(), InstallOptions {variants:pipelines::VariantCompile::SetupPlain,..Default::default()})
}

/// The displays a canvas encodes for (`output_device::admit`).
fn canvas_displays() -> [crate::output_device::Target; 2] {
    [v2::OutputDisplay::Srgb, v2::OutputDisplay::DisplayP3].map(crate::output_device::Target::for_display)
}

/// A canvas runs its view in the display stage's picture form: the stage reads the canvas's picture as a
/// texture and writes the frame, a different shader from a world's stage for the same view program, and one
/// per display. Each std view is drawn as a root canvas's view for each display a canvas encodes for, and so
/// is the default view of a plain canvas, exactly as a render of that canvas draws it, the picture an EXR
/// destination reads included (an authored view's floating stage): a first canvas render, to any destination,
/// compiles neither the program nor its stage.
fn warm_canvas_views(r: &Renderer) -> Result<()> {
    let views = vork_scene::libraries::prebuilt_programs().filter(|name| vork_slang::module_source(name).is_some_and(|source| role(source) == Role::Display));
    for view in std::iter::once(None).chain(views.map(Some)) {
        let mut composition = serde_json::json!({"space": "2d"});
        if let Some(name) = view { composition["view"] = serde_json::json!({"program": name}); }
        let scene: v2::Scene = serde_json::from_value(serde_json::json!({"version": 2,
            "composition": composition, "render": {"resolution": [16, 16]}, "nodes": []}))?;
        for display in canvas_displays() {
            crate::canvas::warm(&scene, Path::new("."), r, display)
                .with_context(|| format!("setup canvas view {} for {}", view.unwrap_or("default"), display.display.name()))?;
        }
    }
    Ok(())
}

/// What a canvas draws, one node of each kind: Moda creates a drawing pipeline the first time a frame needs
/// it, so setup draws frames that need each. Two canvases, because how Moda draws a node depends on what
/// else is on the page: a group's opacity (and the like) splits the page, and on a split page a rounded shape
/// is drawn as a path, not by the shape pipelines a plain page uses. `effects: false` is the plain page:
/// solid and gradient fills, strokes and dashes, rounded shapes opaque and translucent. `effects: true` adds
/// image fills, blur, drop shadow, a clip path, a mask and a group's opacity. A blurred node is drawn into a
/// capture of one sample and a shadowed one into a capture of four, and a pipeline is created per sample
/// count, so the effects page also blurs a group that holds one node of each kind (`soft`). A filtered node
/// inside a filtered group is composited into the group's capture, at that capture's sample count: `soft` holds
/// a blurred and a shadowed node, and so does a shadowed group (`lifted`). Blend modes other
/// than normal are left to first use.
fn canvas_catalogue(float: bool, effects: bool) -> Result<v2::Scene> {
    let image = setup_image()?;
    let rect = "M4 4H28V28H4Z";
    let rounded = "M8 4H24A4 4 0 0 1 28 8V24A4 4 0 0 1 24 28H8A4 4 0 0 1 4 24V8A4 4 0 0 1 8 4Z";
    let circle = "M4 16A12 12 0 1 0 28 16A12 12 0 1 0 4 16Z";
    let mut composition = serde_json::json!({"space": "2d"});
    // An authored view makes the canvas float: Moda then draws the same nodes into Rgba16Float.
    if float { composition["view"] = serde_json::json!({"program": "std/standard"}); }
    let mut nodes = serde_json::json!([
            {"name": "solid", "type": "path", "d": rect, "fill": "#3a86ff", "stroke": "#ffffff", "stroke-width": 2},
            {"name": "linear", "type": "path", "d": rect, "fill": "linear-gradient(to bottom, #415a77, #0d1b2a)", "translation": [32, 0, 0]},
            {"name": "radial", "type": "path", "d": rect, "fill": "radial-gradient(closest-side, white, black)", "translation": [64, 0, 0]},
            {"name": "repeating", "type": "path", "d": rounded, "fill": "repeating-linear-gradient(45deg, #ffbe0b 0px, #ffbe0b 4px, #fb5607 4px, #fb5607 8px)", "translation": [0, 32, 0]},
            {"name": "plate", "type": "path", "d": rounded, "fill": "rgba(0, 0, 0, 0.6)", "translation": [32, 32, 0]},
            {"name": "card", "type": "path", "d": rounded, "fill": "#06d6a0", "translation": [64, 32, 0]},
            {"name": "ring", "type": "path", "d": circle, "fill": "none", "stroke": "#e0e1dd", "stroke-width": 3, "stroke-linecap": "round",
                "pathLength": 100, "stroke-dasharray": [60, 40], "stroke-dashoffset": 10, "translation": [0, 64, 0]}
    ]);
    if effects {
        nodes.as_array_mut().expect("nodes").extend(serde_json::json!([
            {"name": "photo", "type": "path", "d": rounded, "fill": {"image": "picture", "fit": "cover"}, "translation": [32, 64, 0]},
            {"name": "pattern", "type": "path", "d": rect, "fill": {"image": "picture"}, "stroke": "#ffffff", "stroke-width": 1, "stroke-opacity": 0.6, "translation": [64, 64, 0]},
            {"name": "shadowed", "type": "path", "d": rounded, "fill": {"image": "picture", "fit": "cover"}, "filter": "drop-shadow(0 2px 2px rgba(0, 0, 0, 0.6))", "translation": [32, 64, 0]},
            {"name": "blurred", "type": "path", "d": circle, "fill": "#ff006e", "opacity": 0.55, "filter": "blur(3px)", "translation": [16, 16, 0]},
            {"name": "glow", "type": "path", "d": "M4 16C10 4 18 28 28 16", "fill": "none", "stroke": "#36f9c5", "stroke-width": 2, "filter": "drop-shadow(0 0 3px #36f9c5)", "translation": [48, 16, 0]},
            {"name": "wipe", "type": "path", "d": circle},
            {"name": "clipped", "type": "path", "d": rect, "fill": "#8338ec", "clip-path": "wipe", "translation": [16, 48, 0]},
            {"name": "spot", "type": "path", "d": rect, "fill": "radial-gradient(closest-side, white, black)"},
            {"name": "masked", "type": "path", "d": rect, "fill": "linear-gradient(to right, #3a86ff, #8338ec, #ff006e)", "mask": "spot", "translation": [48, 48, 0]},
            {"name": "group", "opacity": 0.5, "translation": [8, 8, 0], "children": [
                {"name": "under", "type": "path", "d": rect, "fill": "#ffbe0b"},
                {"name": "over", "type": "path", "d": circle, "fill": "#fb5607", "translation": [8, 8, 0]}]},
            {"name": "soft", "filter": "blur(1px)", "children": [
                {"name": "soft-solid", "type": "path", "d": rect, "fill": "#3a86ff", "stroke": "#ffffff", "stroke-width": 2},
                {"name": "soft-linear", "type": "path", "d": rect, "fill": "linear-gradient(to bottom, #415a77, #0d1b2a)", "translation": [32, 0, 0]},
                {"name": "soft-radial", "type": "path", "d": rect, "fill": "radial-gradient(closest-side, white, black)", "translation": [64, 0, 0]},
                {"name": "soft-repeating", "type": "path", "d": rounded, "fill": "repeating-linear-gradient(45deg, #ffbe0b 0px, #ffbe0b 4px, #fb5607 4px, #fb5607 8px)", "translation": [0, 32, 0]},
                {"name": "soft-plate", "type": "path", "d": rounded, "fill": "rgba(0, 0, 0, 0.6)", "translation": [32, 32, 0]},
                {"name": "soft-card", "type": "path", "d": rounded, "fill": "#06d6a0", "translation": [64, 32, 0]},
                {"name": "soft-ring", "type": "path", "d": circle, "fill": "none", "stroke": "#e0e1dd", "stroke-width": 3, "stroke-linecap": "round",
                    "pathLength": 100, "stroke-dasharray": [60, 40], "stroke-dashoffset": 10, "translation": [0, 64, 0]},
                {"name": "soft-photo", "type": "path", "d": rounded, "fill": {"image": "picture", "fit": "cover"}, "translation": [32, 64, 0]},
                {"name": "soft-pattern", "type": "path", "d": rect, "fill": {"image": "picture"}, "translation": [64, 64, 0]},
                {"name": "soft-blurred", "type": "path", "d": circle, "fill": "#ff006e", "filter": "blur(1px)", "translation": [16, 16, 0]},
                {"name": "soft-shadowed", "type": "path", "d": rounded, "fill": "#06d6a0", "filter": "drop-shadow(0 1px 1px rgba(0, 0, 0, 0.6))", "translation": [48, 16, 0]}]},
            {"name": "lifted", "filter": "drop-shadow(0 1px 1px rgba(0, 0, 0, 0.6))", "children": [
                {"name": "lifted-blurred", "type": "path", "d": circle, "fill": "#ffbe0b", "filter": "blur(1px)", "translation": [16, 48, 0]},
                {"name": "lifted-shadowed", "type": "path", "d": rounded, "fill": "#3a86ff", "filter": "drop-shadow(0 1px 1px rgba(0, 0, 0, 0.6))", "translation": [48, 48, 0]}]}
        ]).as_array().expect("nodes").iter().cloned());
    }
    Ok(serde_json::from_value(serde_json::json!({"version": 2, "composition": composition,
        "render": {"resolution": [96, 96], "background": "#101820"},
        "images": {"picture": {"file": image}}, "nodes": nodes}))?)
}

/// The pipelines a canvas's drawing creates: both catalogue pages on an 8-bit and on a float canvas, drawn as
/// a render draws them, and text on each. Text has no font here; its batches are built in code
/// (`canvas_draw::warm_text`).
fn warm_canvas_drawing(r: &Renderer) -> Result<()> {
    for float in [false, true] {
        let precision = if float { "float" } else { "8-bit" };
        for effects in [false, true] {
            crate::canvas::warm(&canvas_catalogue(float, effects)?, Path::new("."), r, Default::default())
                .with_context(|| format!("setup canvas drawing ({precision}{})", if effects { ", effects" } else { "" }))?;
        }
        crate::canvas_draw::warm_text(r, float).with_context(|| format!("setup canvas text ({precision})"))?;
    }
    Ok(())
}

/// The exact init scene and the render command's default profile. Use ordinary
/// color delivery and the same foreground choices as a one-frame render.
fn starter_scene() -> Result<v2::Scene> {
    let mut value: serde_json::Value = serde_json::from_str(vork_scene::STARTER_SCENE)?;
    vork_std::expand_scene(&mut value, Path::new("."))?;
    let mut scene = v2::Scene::from_json(&serde_json::to_vec(&value)?)?;
    scene.render.settings = v2::Profile::load(v2::DEFAULT_PROFILE)?.resolve(&Default::default())?;
    scene.validate()?;
    Ok(scene)
}

pub(crate) fn warm_startup(r: &Renderer, variants: pipelines::VariantCompile) -> Result<()> {
    vork_slang::progress::phase("Setup: warming the init starter at the default profile", 1);
    let mut scene = starter_scene()?;
    if !r.release_preparation {
        let profile = v2::Profile::load(v2::DEFAULT_PROFILE)?;
        crate::denoise::resolve_default(r, &mut scene, &profile.name, profile.level, false);
    }
    let installed = r.install_with_options(scene, Path::new("."), InstallOptions {
        variants,
        readback: ReadbackOutputs { color: true, ..Default::default() },
        gpu_color: false,
        ..Default::default()
    })?;
    let _frame = installed.create_frame_slot(r, Default::default())?;
    vork_slang::progress::completed(1);
    Ok(())
}

pub(crate) fn warm(r: &Renderer) -> Result<()> {
    warm_raster_tables(r)?;
    // Retain exact starter keys even after the exhaustive catalogue recorded
    // generic alternatives. Portable preparation captures these artifacts too.
    warm_startup(r, pipelines::VariantCompile::Inline)?;
    let mut all = table_scenes(r);
    all.extend(scenes(r, vork_scene::libraries::prebuilt_programs())?);
    compile(r, &all)?;
    warm_canvas_views(r)?;
    // Drawing creates driver pipelines and compiles nothing a release carries: release preparation, whose
    // device executes nothing, has no use for it.
    if r.release_preparation { return Ok(()); }
    warm_canvas_drawing(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Release preparation's CPU-only adapter (ray queries enabled) at std/path.
    fn release_jobs(scenes: &[serde_json::Value]) -> Result<Vec<(serde_json::Value, v2::Settings)>> {
        let path = v2::Profile::load("std/path")?.resolve(&Default::default())?;
        Ok(scenes.iter().map(|scene| (scene.clone(), path.clone())).collect())
    }

    #[test]
    fn release_catalogue_warms_gaussian_radiance_through_a_capture() -> Result<()> {
        let r = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let source = vork_slang::module_source(crate::splats::PROGRAM).context("gaussian module")?;
        let scenes = program_scenes(&r, crate::splats::PROGRAM, source)?;
        assert_eq!(scenes.len(), 2, "SH degree 0 and degree 3 captures");
        run(&r, release_jobs(&scenes)?.into_iter(), InstallOptions { variants: pipelines::VariantCompile::Setup, ..Default::default() })
    }

    #[test]
    fn one_failed_configuration_reports_every_failure_without_stopping_the_rest() -> Result<()> {
        let r = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        // Bound to a triangle, the radiance program's SH attributes have no source.
        let unbound = scene(serde_json::json!([]), serde_json::Value::Null,
            vec![serde_json::json!({"type": "mesh", "name": "m", "geometry": "triangle", "materials": ["std"]})],
            serde_json::json!({"type": "camera", "name": "c", "translation": [0,0,3]}),
            serde_json::json!({"type": "light", "name": "world", "shape": "environment"}),
            serde_json::json!({"std": {"program": crate::splats::PROGRAM}}), serde_json::json!("auto"));
        let healthy = table_scenes(&r).remove(0);
        let jobs = release_jobs(&[unbound.clone(), healthy.clone(), unbound.clone()])?;
        let error = run(&r, jobs.into_iter(), InstallOptions { variants: pipelines::VariantCompile::Setup, ..Default::default() })
            .expect_err("unbound attributes refuse").to_string();
        assert!(error.starts_with("setup pipelines for std/gaussian; 1 more setup configuration(s) also failed: std/gaussian: "), "{error}");
        assert!(error.contains("Attribute requires an attribute binding"), "{error}");
        run(&r, release_jobs(&[healthy])?.into_iter(), InstallOptions { variants: pipelines::VariantCompile::Setup, ..Default::default() })
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn startup_plain_hits_match_exact_specializations() -> Result<()> {
        for metalness in [0., 1.] {
            // Explicit GPU runs require renderer creation and all assertions to succeed.
            let mut r = crate::test_renderer()?;
            // Isolate the device-key catalogue from disk and other tests. Setup
            // must be the sole source of known keys for the first install.
            let keys=r.disk_cache.path("test-plain-hit-keys",1,blake3::hash(b""));
            r.compiler = pipelines::Compiler::new(&r.device, r.cache.as_ref(), pipelines::Known::load(keys,false), r.compiler.blob());
            // This regression covers the exhaustive catalogue's raster hit selection.
            passes::decoupled::warm_shaders(&r);
            warm_raster_tables(&r)?;
            r.finish_shader_warmup()?;
            let mut scene = crate::test_scene(serde_json::json!({"version":2,
                "render":{"resolution":[64,48]},
                "nodes":[
                    {"type":"mesh","geometry":crate::test_geometry::latitude_longitude(1.,32,16),"materials":[{"inputs":{"base_color":[0.2,0.4,0.7],"base_metalness":metalness,"specular_roughness":0.3}}]},
                    {"type":"mesh","translation":[0,-1,0],"geometry":crate::test_geometry::plane([8.,8.],[1,1])},
                    {"type":"camera","translation":[3,2,5],"target":[0,0,0]},
                    {"type":"light","shape":"rectangle","translation":[-3,5,4],"target":[0,0,0],"energy":500,"size":3},
                    {"type":"light","shape":"sun","energy":0.1},
                    {"type":"light","shape":"point","translation":[2,3,2],"energy":10},
                    {"type":"light","shape":"spot","translation":[0,4,0],"target":[0,0,0],"energy":10}
                ]}), Path::new("."))?;
            scene.render.settings = v2::Profile::load(v2::DEFAULT_PROFILE)?.resolve(
                &serde_json::from_value(serde_json::json!({"denoise":false}))?)?;
            let draw = |mode, expect_plain| -> Result<Vec<[f32;4]>> {
                let gate = match &mode {pipelines::VariantCompile::Gated(gate) => Some(gate.clone()), _ => None};
                struct Release(Option<std::sync::Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>>);
                impl Drop for Release {
                    fn drop(&mut self) {if let Some(gate)=&self.0 {*gate.0.lock().unwrap()=true;gate.1.notify_all();}}
                }
                let _release = Release(gate.clone());
                r.compiler.hit_selections.lock().unwrap().clear();
                let mut installed = r.install_with_options(scene.clone(), Path::new("."), InstallOptions { variants: mode, ..Default::default() })?;
                let selections = r.compiler.hit_selections.lock().unwrap().clone();
                for is_gi in [true,false] {
                    let hits: Vec<_> = selections.iter().filter(|(requested,_)| if is_gi {
                        matches!(requested,pipelines::ProgramKernel::Gi {..})
                    } else {matches!(requested,pipelines::ProgramKernel::Reflections {..})}).collect();
                    assert_eq!(hits.len(),1,"must observe the installed hit pipeline: {selections:?}");
                    let (requested,selected)=*hits[0];
                    let plain=requested.plain_hit().expect("plain-compatible scene");
                    assert_ne!(requested,plain,"scene must distinguish exact and plain keys");
                    assert_eq!(selected,if expect_plain {plain} else {requested},"actual installed hit selection");
                }
                let mut slot = installed.create_frame_slot(&r, ReadbackPrecision::Float)?;
                let first = installed.render_frame(&r, &mut slot, 0.)?.linear_rgba.context("float readback")?;
                if let Some(gate) = gate {
                    assert!(!installed.variants_ready(), "hit variants must not block the first frame");
                    assert!(slot.view.variants_pending());
                    *gate.0.lock().unwrap() = true;
                    gate.1.notify_all();
                    r.settle_pipelines()?;
                    let promoted = installed.render_frame(&r, &mut slot, 0.)?.linear_rgba.context("promoted float readback")?;
                    assert!(!slot.view.variants_pending(), "the retained slot must promote at an exposure boundary");
                    crate::test_linear::assert_linear_eq(&first, &promoted);
                }
                Ok(first)
            };
            let plain = draw(pipelines::VariantCompile::Fallback, true)?;
            let gate = std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
            let background = draw(pipelines::VariantCompile::Gated(gate), true)?;
            r.settle_pipelines()?;
            let promoted = draw(pipelines::VariantCompile::Background, false)?;
            let exact = draw(pipelines::VariantCompile::Inline, false)?;
            crate::test_linear::assert_linear_eq(&plain, &exact);
            crate::test_linear::assert_linear_eq(&background, &exact);
            crate::test_linear::assert_linear_eq(&promoted, &exact);
        }
        Ok(())
    }

    #[test]
    fn setup_covers_draft_and_final_oidn_and_off_at_shipped_camera_budgets() -> Result<()> {
        let settings = configurations()?;
        for (name, budget) in [("std/draft",4), ("std/final",8)] {
            let shipped = v2::Profile::load(name)?.resolve(&Default::default())?;
            assert!(shipped.integrator.denoiser().is_some());
            assert_eq!(shipped.samples, budget);
            assert!(settings.contains(&shipped));
            let off = v2::Profile::load(name)?.resolve(
                &serde_json::from_value(serde_json::json!({"denoise":false}))?)?;
            assert!(settings.contains(&off));
            let value = serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"render":{"settings":{}}});
            let scene = prepare(value, shipped)?;
            assert_eq!(scene.render.settings.samples, budget);
        }
        Ok(())
    }

    #[test]
    fn every_std_program_has_a_role() {
        let roles: Vec<_> = vork_scene::libraries::prebuilt_programs()
            .map(|name| (name, role(vork_slang::module_source(name).unwrap()))).collect();
        for (name, expected) in [("std/sky", Role::Environment),
            ("std/hair", Role::Hair), ("std/camera",Role::Camera), ("std/environment_image",Role::Environment),
            ("std/agx", Role::Display), ("std/standard", Role::Display)] {
            assert!(roles.contains(&(name, expected)), "{name}: {roles:?}");
        }
        // Authored media are recognized by their entry signature.
        assert_eq!(role("MediumCoefficients medium(MediumContext ctx) {}"), Role::Medium);
        // Blender programs compile on first use.
        assert!(roles.iter().all(|(name, _)| name.starts_with("std/")), "{roles:?}");
    }

    /// The synthetic scenes still obey the scene schema and settings contract.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn setup_scenes_validate() -> Result<()> {
        let r = crate::test_renderer()?;
        let mut scenes = table_scenes(&r);
        scenes.extend(self::scenes(&r, vork_scene::libraries::prebuilt_programs())?);
        let mut count = 0;
        for (scene, integrator) in compile_jobs(&scenes)?.into_iter() {
            let name = scene["materials"]["std"]["program"].as_str().map_or_else(|| "built-in table".into(), str::to_owned);
            prepare(scene, integrator).with_context(|| format!("setup scene for {name}"))?;
            count += 1;
        }
        assert!(count > 0);
        Ok(())
    }

    /// Exercise setup's complete compile core, including authored medium variants. Running
    /// synthetic frames here used to fail the unbounded-emission diagnostic.
    #[test]
    #[ignore = "requires Vulkan/Metal and pinned Slang toolchain; complete setup warmup"]
    fn setup_compiles_all_library_pipelines() -> Result<()> {
        let r = Renderer::new()?;
        r.warm_shaders()?;
        // Compilation must also accept inputs whose infinite-world integral is
        // divergent. The same inputs must still be refused by a real frame.
        let mut scenes = authored_medium_scenes(&r)?;
        for scene in &mut scenes {
            scene["materials"]["std"]["inputs"] = serde_json::json!({"density": 0., "emission": [1., 1., 1.]});
        }
        compile(&r, &scenes)?;
        let (scene, integrator) = compile_jobs(&scenes)?.into_iter().next().context("fog compile job")?;
        let installed = r.install_with_options(prepare(scene, integrator)?, Path::new("."),
            InstallOptions { variants: pipelines::VariantCompile::Fallback, ..Default::default() })?;
        let mut frame = installed.create_frame_slot(&r, Default::default())?;
        frame.submit(&installed, &r, 0.)?;
        let error = frame.collect(&r).err().context("real frame accepted divergent world emission")?;
        assert!(format!("{error:#}").contains("unbounded medium emission"), "{error:#}");
        Ok(())
    }

    /// After setup's built-in-material scenes, a one-frame render of another scene
    /// with shipped programs (other materials, lights and glass) compiles no ray-hit or
    /// transmissive camera kernel: it draws the supersets setup stored, with the
    /// pixels of its own specializations.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn one_frame_renders_draw_setup_supersets() -> Result<()> {
        let mut r = crate::test_renderer()?;
        if !r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) { return Ok(()) }
        let root = std::env::temp_dir().join(format!("vork-setup-supersets-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root)?;
        r.disk_cache = disk_cache::Store::for_test(root.clone());
        // Warmth comes only from this store: drop keys loaded from the shared one.
        let known = pipelines::Known::load(r.disk_cache.path("pipeline-keys", 1, blake3::hash(b"")), false);
        r.compiler = pipelines::Compiler::new(&r.device, r.cache.as_ref(), known, r.compiler.blob());
        // Setup's shipped camera generics, then its built-in-material scenes.
        passes::decoupled::warm_shaders(&r);
        compile(&r, &table_scenes(&r))?;
        let variants = || -> Result<std::collections::BTreeSet<std::ffi::OsString>> {
            Ok(std::fs::read_dir(&root)?.flatten().map(|e| e.file_name())
                .filter(|n| n.to_string_lossy().contains(".slang-variant-")).collect())
        };
        let before = variants()?;
        let settings = v2::Profile::load(v2::DEFAULT_PROFILE)?.resolve(&serde_json::Map::new())?;
        let mut scene = crate::test_scene(serde_json::json!({"version": 2,
            "render": {"resolution": [48, 32]},
            "geometry": {"box": {"std":"rounded_box","radius":0,"size": [1, 1, 1]}, "floor": {"std":"rounded_box","radius":0,"size": [4, 0.1, 4]}},
            "nodes": [{"type": "mesh", "geometry": "box", "materials": ["glass"]},
                {"type": "mesh", "geometry": "floor", "materials": ["metal"], "translation": [0, -0.6, 0]},
                {"type": "camera", "translation": [0.5, 0.8, 3]},
                {"type": "light", "shape": "rectangle", "energy": 40, "translation": [1, 2, 1]},
                {"type": "light", "shape": "environment"}],
            "materials": {"glass": {"inputs": {"transmission_weight": 1.0, "coat_weight": 0.5}},
                "metal": {"inputs": {"base_color": [0.9, 0.6, 0.3], "base_metalness": 1.0, "specular_roughness": 0.3}}}}), Path::new("."))?;
        scene.render.settings = settings;
        let draw = |r: &Renderer, variants| -> Result<Vec<u8>> {
            let mut installed = r.install_with_options(scene.clone(), Path::new("."), InstallOptions { variants, ..Default::default() })?;
            let mut frame = installed.create_frame_slot(r, Default::default())?;
            // Setup warmth is independent of receiver-storage demand. Exercise
            // the same growth/retry path as a user's first rendered frame.
            Ok(installed.render_frame(r, &mut frame, 0.)?.rgba)
        };
        let superset = draw(&r, pipelines::VariantCompile::Fallback)?;
        let added: Vec<_> = variants()?.difference(&before).cloned().collect();
        assert!(added.is_empty(), "a one-frame render after setup compiled kernels: {added:?}");
        let specialized = draw(&r, pipelines::VariantCompile::Inline)?;
        assert!(variants()?.len() > before.len(), "the scene's own specializations are not setup's");
        let _ = std::fs::remove_dir_all(&root);
        let differs = superset.iter().zip(&specialized).filter(|(a, b)| a.abs_diff(**b) > 2).count();
        assert!(differs * 1000 < superset.len(), "superset and specialized kernels draw different pixels: {differs} channels");
        Ok(())
    }

    /// Setup prepares every std view as a canvas's view; a canvas with that view then finds its program compiled.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn setup_prepares_std_views_for_canvases() -> Result<()> {
        let r = crate::test_renderer()?;
        let views: Vec<_> = vork_scene::libraries::prebuilt_programs()
            .filter(|name| role(vork_slang::module_source(name).expect("std module")) == Role::Display).collect();
        assert!(views.contains(&"std/standard") && views.contains(&"std/agx"), "{views:?}");
        warm_canvas_views(&r)?;
        // Every std view now finds its compiled program: a canvas with other inputs packs different words only.
        for name in views {
            let source = vork_slang::module_source(name).context("std view module")?;
            let origin = PathBuf::from(format!("{name}.slang"));
            let key = crate::display_program::program_key(&programs::abi(), source, &origin)?;
            let _: vork_slang::DisplayProgram = programs::cached(&r, key, &vork_slang::Compiler::discover(), &programs::abi(), source, &origin,
                |_| bail!("{name} compiled again"))?;
        }
        let scene: v2::Scene = serde_json::from_value(serde_json::json!({"version": 2,
            "composition": {"space": "2d", "view": {"program": "std/standard", "inputs": {"exposure": 1}}}, "render": {"resolution": [32, 32]}, "nodes": []}))?;
        crate::canvas::check(&scene, Path::new("."), &r)
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn setup_prebuilds_std_worlds() -> Result<()> {
        let r = crate::test_renderer()?;
        compile(&r, &scenes(&r, ["std/sky"].into_iter())?)?;
        // Any std/sky world now finds its compiled program; only its input words differ.
        let source = vork_slang::module_source("std/sky").context("std/sky module")?;
        let key = environment_program::program_key(&programs::abi(), source, Path::new("std/sky.slang"))?;
        let _:vork_slang::EnvironmentProgram=programs::cached(&r, key, &vork_slang::Compiler::discover(), &programs::abi(), source, Path::new("std/sky.slang"), |_| bail!("std/sky compiled again"))?;
        let scene: v2::Scene = serde_json::from_value(serde_json::json!({"version": 2,
            "nodes": [{"type": "light", "shape": "environment", "material": "sky"}],
            "materials": {"sky": {"program": "std/sky", "inputs": {"sun_elevation": 0.5, "altitude": 10}}}}))?;
        let (image, program) = environment_program::install(&r, &scene, "sky", Path::new("."))?;
        assert!(program.is_none() && image.image.pixels().any(|p| p.0[2] > 0.), "a constant, lit sky");
        // Fog at its defaults and varied, under each shadow topology: every medium kernel.
        assert_eq!(authored_medium_scenes(&r)?.len(), 2 * shadows(&r).len());
        Ok(())
    }
}
