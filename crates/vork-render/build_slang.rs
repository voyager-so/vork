//! Build-time Slang compilation: every hand-written entry point becomes SPIR-V
//! (Vulkan) and MSL (Metal) plus Slang's reflection for both, embedded through
//! `$OUT_DIR/slang_shaders.rs`. Both targets compile on every host so the
//! layout tests check the Metal resource indices on Linux too.
use std::{
    fmt::Write,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

#[path = "src/forward_variant.rs"]
#[allow(dead_code)]
mod forward_variant;
use forward_variant::{ForwardKey, METAL_TARGET, NATIVE_OPTIMIZATION, SLANG_FLAGS, SPIRV_TARGET, Stage};

#[path = "src/resolve_native.rs"]
#[allow(dead_code)] // material_base serves the runtime resolve pass only.
mod resolve_native;
#[path = "src/camera_native.rs"]
mod camera_native;
#[path = "src/surface_native.rs"]
mod surface_native;
#[path = "src/camera_dispatch_native.rs"]
#[allow(dead_code)]
mod camera_dispatch_native;

/// Each shader owns its entry list beside its source (`<module>.entries`).
fn entries() -> Vec<(String, String, String)> {
    let mut files: Vec<_> = fs::read_dir("shaders")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "entries"))
        .collect();
    files.sort();
    let mut entries = Vec::new();
    for file in files {
        let module = file.file_stem().unwrap().to_str().unwrap();
        assert!(
            file.with_extension("slang").is_file(),
            "missing shader for {}",
            file.display()
        );
        for line in fs::read_to_string(&file).unwrap().lines() {
            let line = line.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let words: Vec<_> = line.split_whitespace().collect();
            assert!(
                words.len() == 2 && matches!(words[1], "vertex" | "fragment" | "compute"),
                "invalid entry in {}: {line}",
                file.display()
            );
            entries.push((module.to_owned(), words[0].to_owned(), words[1].to_owned()));
        }
    }
    entries
}

// A pass owns resource-register defines and compiler options beside its source.
fn module_options(command: &mut Command, module: &Path, target: Option<&str>) {
    let path = module.with_extension("defines");
    if let Ok(defines) = fs::read_to_string(&path) {
        for define in defines.split_whitespace() {
            command.arg(format!("-D{define}"));
        }
    }
    // Module validation emits target-independent IR. Options such as fp-mode
    // require a target and belong to the native entry compilation below.
    if let Some(target) = target {
        for define in target_defines(module, target) {
            command.arg(format!("-D{define}"));
        }
        if let Ok(flags) = fs::read_to_string(module.with_extension("flags")) {
            command.args(flags.split_whitespace());
        }
    }
}

// Target-specific compiler boundaries must not change the other backend.
fn target_defines(module: &Path, target: &str) -> Vec<String> {
    fs::read_to_string(module.with_extension(format!("{target}.defines")))
        .unwrap_or_default().split_whitespace().map(str::to_owned).collect()
}

// A module may restrict its registered native artifacts beside its entries.
// Modules may omit a target when their entry is not used by that backend.
fn targets(module: &Path) -> Vec<&'static str> {
    match fs::read_to_string(module.with_extension("targets")) {
        Ok(text) => {
            assert_eq!(text.trim(), "spirv", "unsupported target list for {}", module.display());
            vec!["spirv"]
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => vec!["spirv", "metal"],
        Err(e) => panic!("{}: {e}", module.display()),
    }
}

/// Keep shipped pipeline identity independent of when the code is hashed.
fn input_key(code: &[u8], entry: &str, metal: bool, threads: (u32, u32, u32)) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"shipped-entry-v1");
    hash.update(&[u8::from(metal)]);
    for part in [code, entry.as_bytes()] {
        hash.update(&(part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    let threads = if metal { threads } else { (0, 0, 0) };
    for size in [threads.0, threads.1, threads.2] {
        hash.update(&size.to_le_bytes());
    }
    *hash.finalize().as_bytes()
}

fn slangc() -> PathBuf {
    println!("cargo:rerun-if-env-changed=VORK_SLANGC");
    std::env::var_os("VORK_SLANGC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("slangc"))
}

/// Runs one compiler process with its diagnostics in `log`. Spawning briefly
/// holds a few descriptors (log file, /dev/null, exec pipe); one spawn at a time
/// keeps the build's open files independent of core count, and the parent holds
/// none while the compiler runs.
fn run(mut command: Command, log: &Path) -> Result<(), String> {
    static SPAWNING: Mutex<()> = Mutex::new(());
    let mut child = {
        let _guard = SPAWNING.lock().unwrap_or_else(|e| e.into_inner());
        let file = fs::File::create(log)
            .map_err(|e| format!("cannot create {}: {e}", log.display()))?;
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(file)
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", command.get_program().to_string_lossy()));
        // The command owns the parent's copy of the log file; close it now.
        drop(command);
        child?
    };
    let status = child.wait().map_err(|e| format!("waiting for slangc: {e}"))?;
    if status.success() {
        let _ = fs::remove_file(log);
        Ok(())
    } else {
        let diagnostics = fs::read_to_string(log).unwrap_or_default();
        let _ = fs::remove_file(log);
        Err(format!("{status}\n{diagnostics}"))
    }
}

#[path = "../vork-slang/src/renderer_registry.rs"]
#[allow(dead_code)]
mod renderer_registry;

/// Returns the compiler version and content identity of embedded runtime modules.
/// Targets this build compiles (`VORK_SHADER_TARGETS`, comma-separated:
/// spirv, metal). Default both, so layout tests check Metal indices on every
/// host; a Linux check that runs no Metal tests can set `spirv`.
fn host_targets() -> Vec<String> {
    println!("cargo:rerun-if-env-changed=VORK_SHADER_TARGETS");
    match std::env::var("VORK_SHADER_TARGETS") {
        Ok(list) if !list.trim().is_empty() => {
            let targets: Vec<String> = list.split(',').map(|t| t.trim().to_owned()).filter(|t| !t.is_empty()).collect();
            for t in &targets { assert!(t == "spirv" || t == "metal", "VORK_SHADER_TARGETS: unknown target {t}"); }
            assert!(std::env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") || targets.iter().any(|t| t == "metal"),
                "VORK_SHADER_TARGETS must include metal on Apple targets");
            targets
        }
        _ => vec!["spirv".into(), "metal".into()],
    }
}

pub fn compile(out: &Path, closure_dir: &Path) -> (String, blake3::Hash) {
    let slangc = slangc();
    let version = Command::new(&slangc)
        .arg("-version")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "the renderer's shaders need the pinned Slang compiler ({}): {e}; \
             install it with scripts/install-slang <dir> and set VORK_SLANGC=<dir>/bin/slangc",
                slangc.display()
            )
        });
    let version = String::from_utf8_lossy(&version.stderr).trim().to_owned();
    let pinned = fs::read_to_string("../../dist/slang.json").unwrap();
    assert!(
        pinned.contains(&format!("\"version\": \"{version}\"")),
        "Slang {version} at {} is not the version pinned in dist/slang.json",
        slangc.display()
    );
    let dir = out.join("slang");
    fs::create_dir_all(&dir).unwrap();
    // Generated modules live beside the outputs and are found through `-I`.
    let generated = out.join("slang-modules");
    fs::create_dir_all(&generated).unwrap();
    fs::write(generated.join("vork_object.slang"), crate::object_abi::slang()).unwrap();
    fs::write(generated.join("material_bindings.slang"),crate::binding_abi::material_source()).unwrap();
    fs::write(
        generated.join("vork_constants.slang"),
        renderer_registry::generate(true),
    )
    .unwrap();
    let trace_types="../vork-slang/src/trace_types.slang";
    println!("cargo:rerun-if-changed={trace_types}");
    fs::copy(trace_types,generated.join("program_trace.slang")).unwrap();
    let sampling="../vork-slang/src/vork_sampling.slang";
    println!("cargo:rerun-if-changed={sampling}");
    fs::copy(sampling,generated.join("vork_sampling.slang")).unwrap();
    // The same core and Rust-registry constants used by std programs and the oracle.
    let opacity = "shaders/std_opacity.slang";
    println!("cargo:rerun-if-changed={opacity}");
    fs::copy(opacity, generated.join("std_opacity.slang")).unwrap();
    let core = "../../libraries/std/openpbr_core.slang";
    println!("cargo:rerun-if-changed={core}");
    fs::copy(core, generated.join("openpbr_core.slang")).unwrap();
    fs::write(
        generated.join("vork_openpbr_constants.slang"),
        renderer_registry::generate(true),
    ).unwrap();
    fs::write(
        out.join("constants.wgsl"),
        renderer_registry::generate(false),
    )
    .unwrap();
    // Use the same reflected input decoder and dispatch bridge as installed cameras.
    let camera_source="../../libraries/std/camera.slang";
    println!("cargo:rerun-if-changed={camera_source}");
    println!("cargo:rerun-if-changed=src/camera_dispatch_native.rs");
    let abi=vork_slang::Abi {aov_count:0,context:vork_slang::Abi::spec_context(),closure:vork_slang::openpbr_fields()};
    let camera=vork_slang::Compiler::discover().compile_camera(&abi,&fs::read_to_string(camera_source).unwrap(),Path::new("std/camera.slang"),"shippedcamera").expect("compile shipped standard camera");
    let dispatch=camera_dispatch_native::generate(&abi,[(1,&camera)]);
    for (name,source) in dispatch.modules {
        let path=generated.join(format!("{name}.slang"));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path,source).unwrap();
    }
    // Entries, then the precompiled generic colour stages (one per stage and technique).
    let entries = entries();
    let mut units: Vec<(String, PathBuf, &str, &str)> = entries
        .iter()
        .map(|(module, entry, stage)| {
            (
                format!("{module}_{entry}"),
                Path::new("shaders").join(format!("{module}.slang")),
                entry.as_str(),
                stage.as_str(),
            )
        })
        .collect();
    for (name, path, entry, _) in &mut units {
        if path.file_stem().is_some_and(|s| s == "resolve") {
            let text = fs::read_to_string(&*path).unwrap();
            *path = generated.join(format!("{name}.slang"));
            fs::write(&*path, resolve_native::source(&text, entry)).unwrap();
        }
    }
    fs::write(generated.join("camera_config.slang"),camera_native::config_source()).unwrap();
    fs::write(generated.join("decoupled.slang"),camera_native::evaluation_source(&fs::read_to_string("shaders/decoupled.slang").unwrap())).unwrap();
    let camera_names:Vec<_>=[false,true].into_iter().flat_map(|msaa|[false,true].into_iter().flat_map(move |ray|["evaluate","evaluate_opaque","index","classify","prefix","scatter","combine","combine_diffusion"].into_iter().map(move |stage|(msaa,ray,stage)))).filter(|(msaa,_,stage)|!*msaa || matches!(*stage,"combine"|"combine_diffusion")).collect();
    for &(msaa,ray,stage) in &camera_names {
        let name=format!("camera_{}_{}_{}",u8::from(msaa),u8::from(ray),stage);
        let path=generated.join(format!("{name}.slang"));
        fs::write(&path,camera_native::entry_source(if stage=="combine_diffusion" {"combine"} else {stage})).unwrap();
        let mut defines=camera_native::defines(ray);
        if matches!(stage,"index"|"classify"|"prefix"|"scatter") {defines.push("VORK_CAMERA_CLASSIFY".into());}
        if matches!(stage,"combine"|"combine_diffusion") {defines.push("VORK_COMPOSE_STORED".into());}
        if stage=="combine_diffusion" {defines.push("VORK_RECEIVER_DIFFUSION".into());}
        if stage=="evaluate_opaque" {defines.push("VORK_CAMERA_OPAQUE".into());}
        if msaa {defines.push("VORK_RASTER_MSAA".into());}
        fs::write(path.with_extension("defines"),defines.join(" ")).unwrap();
        fs::write(path.with_extension("flags"),camera_native::COMPILER_FLAGS.join(" ")).unwrap();
        let entry=match stage {"evaluate"|"evaluate_opaque"=>"camera_evaluate","index"=>"camera_index","classify"=>"camera_classify","prefix"=>"camera_prefix","scatter"=>"camera_scatter","combine"|"combine_diffusion"=>"camera_combine",_=>"camera_receivers"};
        units.push((name,path,entry,"compute"));
    }
    let mut surfaces=Vec::new();
    for family in surface_native::SurfaceKernel::ALL {
        for rays in [false,true] {
            if rays && !family.camera() {continue;}
            for probes in [false,true] { for coverage in [false,true] { for boundaries in [0,1] {
                if boundaries>0 && (!rays || !coverage || !matches!(family,surface_native::SurfaceKernel::Evaluate|surface_native::SurfaceKernel::Occupancy)) {continue;}
                if family==surface_native::SurfaceKernel::Occupancy && boundaries==0 {continue;}
                if coverage && family.camera() && !rays {continue;}
                if family==surface_native::SurfaceKernel::Indirect && (!probes || !rays) {continue;}
                let name=format!("surface_{}_{}_{}_{}_{}",family.name(),u8::from(rays),u8::from(probes),u8::from(coverage),boundaries);
                let path=generated.join(format!("{name}.slang"));
                fs::write(&path,forward_variant::late_variant_module(&family.source(),family.camera() && rays)).unwrap();
                let mut defines=vec!["VORK_NO_IMPLICIT".to_owned(),"VORK_AOV_COUNT=0".to_owned(),"VORK_SURFACE_SLANG_OUTLINE".to_owned()];
                // All shipped surfaces require installation's no-containment proof.
                // Candidate record layout is independent: opaque connections keep
                // the compact reader without retaining the solid discovery walk.
                defines.push("VORK_NO_SHADOW_CONTAINMENT".into());
                if coverage {defines.push("VORK_PROGRAM_COVERAGE".into());}
                else {defines.push("VORK_NO_CONNECTION_TRANSMISSION".into());}
                if boundaries>0 {defines.push("VORK_RAY_REFRACTION".into());}
                if rays {defines.push("VORK_RAY_BINDINGS".into());}
                if probes {defines.extend(["VORK_PROBES".into(),format!("VORK_PROBE_BINDING={}",if family.camera() {9} else {0})]);}
                if family.camera() {
                    defines.extend(["VORK_CAMERA_SPLIT".into(),format!("VORK_REFRACTION_BOUNDARIES={boundaries}"),"VORK_FIBRE_BINDING=4".into()]);
                    fs::write(path.with_extension("flags"),camera_native::COMPILER_FLAGS.join(" ")).unwrap();
                }
                fs::write(path.with_extension("defines"),defines.join(" ")).unwrap();
                fs::write(path.with_extension("targets"),"spirv").unwrap();
                units.push((name.clone(),path,"surface_main","compute"));
                surfaces.push((family,rays,probes,coverage,boundaries,name));
            }}}
        }
    }
    // The prebuilt path engine, one unit per family, on both backends. Metal
    // takes the path kernel's registers; the unit's stem is not the module's.
    let mut path_engines=Vec::new();
    for engine in surface_native::PathEngine::ALL {
        let name=engine.name();
        let path=generated.join(format!("{name}.slang"));
        fs::write(&path,engine.source()).unwrap();
        fs::write(path.with_extension("defines"),surface_native::PathEngine::DEFINES.join(" ")).unwrap();
        let registers=forward_variant::hit_kernel_registers("path",engine.adaptive,false);
        fs::write(path.with_extension("metal.defines"),registers.iter().map(|define|define.trim_start_matches("-D")).collect::<Vec<_>>().join(" ")).unwrap();
        units.push((name.clone(),path,surface_native::PathEngine::ENTRY,"compute"));
        path_engines.push((engine,name));
    }
    let generics: Vec<_> = Stage::ALL.into_iter().flat_map(|stage| [false, true].map(|hashed| ForwardKey::generic(stage, hashed))).collect();
    // Built-in geometry stages vary only with the named-attribute count. Ship
    // every supported count, including table-material attribute reads.
    let mut streams = Vec::new();
    for attributes in 1..=8u32 {
        for (module, entry, stage) in [
            ("depth", "main_vs", "vertex"),
            ("depth", "shadow_vs", "vertex"),
            ("decoupled_visibility", "masked_vs", "vertex"),
            ("depth", "depth_fs", "fragment"),
            ("depth", "hashed_depth_fs", "fragment"),
            ("decoupled_visibility", "masked_fs", "fragment"),
            ("decoupled_visibility", "transparent_fragments_fs", "fragment"),
            ("decoupled_visibility", "transparent_triangles_fs", "fragment"),
        ] {
            let name = format!("stream_{attributes}_{module}_{entry}");
            let path = generated.join(format!("{name}.slang"));
            fs::write(&path, fs::read_to_string(format!("shaders/{module}.slang")).unwrap()).unwrap();
            fs::write(path.with_extension("defines"), format!("VORK_ATTRIBUTES={attributes}")).unwrap();
            streams.push((attributes, module, entry, name.clone()));
            units.push((name, path, entry, stage));
        }
    }
    let mut controls = Vec::new();
    for attributes in 0..=8u32 {
        for denoise in [false, true] {
            for key in &generics {
                let name = if attributes==0 && !denoise {key.name()} else {format!("{}_a{attributes}_d{}", key.name(), u8::from(denoise))};
                controls.push((attributes, denoise, key.stage as usize, key.hashed(), name.clone()));
                let path = generated.join(format!("{name}.slang"));
                fs::write(&path, key.module_with(None)).unwrap();
                let mut defines = format!("VORK_ATTRIBUTES={attributes}");
                if denoise {defines.push_str(" VORK_DENOISE");}
                fs::write(path.with_extension("defines"), defines).unwrap();
                let (entry, stage) = key.entry();
                units.push((name, path, entry, stage));
            }
        }
    }
    // Bundled libraries, for std.color's single blackbody (light_temperature.slang).
    println!("cargo:rerun-if-changed=../../libraries/std/color.slang");
    let include = [
        &generated,
        Path::new("shaders"),
        Path::new("shaders/generated"),
        closure_dir,
        Path::new("../../libraries"),
    ];
    // Compile shared modules even before their last WGSL consumer is ported.
    // This also validates every new per-file module without a central list.
    let mut modules: Vec<_> = fs::read_dir("shaders")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "slang"))
        .collect();
    modules.sort();
    for module in modules {
        let mut command = Command::new(&slangc);
        if module.file_stem().is_some_and(|s|s=="decoupled") {
            command.arg(generated.join("decoupled.slang")).arg("-DVORK_CAMERA_SPLIT");
        } else {command.arg(&module);}
        module_options(&mut command, &module, None);
        for path in include {
            command.arg("-I").arg(path);
        }
        let out = dir.join(module.file_name().unwrap()).with_extension("slang-module");
        command
            .arg("-DVORK_NO_GAUSSIANS")
            .args(["-warnings-as-errors", "all"])
            .arg("-o")
            .arg(&out);
        if let Err(e) = run(command, &out.with_extension("slang-module.log")) {
            panic!("slangc {}: {e}", module.display());
        }
    }
    let host = host_targets();
    let jobs: Vec<_> = units
        .iter()
        .flat_map(|u| [SPIRV_TARGET, METAL_TARGET].into_iter().filter(|t| host.iter().any(|h| h == t.1)).filter(move |t| targets(&u.1).contains(&t.1)).map(move |t| (u, t)))
        .collect();
    // Each compiler process holds its own linked IR. Respect Cargo's build
    // parallelism instead of launching every entry/target at once on small hosts.
    let parallelism = std::env::var("NUM_JOBS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let errors: Vec<String> = std::thread::scope(|scope| {
        let runs: Vec<_> = (0..parallelism.min(jobs.len()))
            .map(|_| {
                let (slangc, dir, include, jobs, next, failed) =
                    (&slangc, &dir, &include, &jobs, &next, &failed);
                scope.spawn(move || -> Result<(), String> {
                    while !failed.load(Ordering::Relaxed) {
                        let Some(&((name, path, entry, stage), (ext, target, flags))) =
                            jobs.get(next.fetch_add(1, Ordering::Relaxed))
                        else {
                            break;
                        };
                        let stem = dir.join(format!("{name}.{ext}"));
                        let mut command = Command::new(slangc);
                        command.arg(path);
                        module_options(&mut command, path, Some(target));
                        if target == "metal" {
                            if *stage == "vertex" {command.args(["-DVORK_MATERIAL_WORDS_REGISTER=t3","-DVORK_FRAME_LIGHTS_REGISTER=t30","-DVORK_FRAME_CLUSTERS_REGISTER=b29","-DVORK_POINT_WORDS_REGISTER=t2"]);}
                            command.args(forward_variant::hit_kernel_registers(
                                path.file_stem().unwrap().to_str().unwrap(),
                                false,path.file_stem().unwrap() == "probe_update"));
                        }
                        for dir in include {
                            command.arg("-I").arg(dir);
                        }
                        command
                            .args(["-target", target, "-entry", entry, "-stage", stage])
                            .args(SLANG_FLAGS)
                            .arg("-zero-initialize")
                            .arg(NATIVE_OPTIMIZATION)
                            .arg("-DVORK_NO_GAUSSIANS")
                            .args(["-warnings-as-errors", "all"])
                            .args(flags)
                            .arg("-o")
                            .arg(&stem)
                            .arg("-reflection-json")
                            .arg(stem.with_extension(format!("{ext}.json")));
                        let args:Vec<_>=command.get_args().collect();
                        let precise=args.windows(2).any(|pair|pair[0]=="-fp-mode" && pair[1]=="precise");
                        if let Err(e) = run(command, &stem.with_extension(format!("{ext}.log"))) {
                            failed.store(true, Ordering::Relaxed);
                            return Err(format!("slangc {name} ({target}): {e}"));
                        }
                        // These boundaries constrain Slang, not the driver. Normalize
                        // before embedding/hashing so shipped identities use actual code.
                        // The path engine gets what a scene's own path kernel gets
                        // at runtime (slang_forward.rs, driver_inline).
                        if target == "spirv" && (name.starts_with("path_engine_") || target_defines(path, target).iter()
                            .any(|define| define == "VORK_SURFACE_SLANG_OUTLINE")) {
                            let mut code = fs::read(&stem).map_err(|e| e.to_string())?;
                            vork_slang::spirv::allow_driver_inlining(&mut code).map_err(|e| e.to_string())?;
                            fs::write(&stem, code).map_err(|e| e.to_string())?;
                        }
                        // Slang's MSL source needs the same native FP contract
                        // as runtime shaders (vork_slang's precise emission).
                        if target=="metal" && name.starts_with("camera_") && name.ends_with("_evaluate") {
                            let source=camera_native::inline_metal_functions(&fs::read(&stem).map_err(|e|e.to_string())?);
                            fs::write(&stem,source).map_err(|e|e.to_string())?;
                        }
                        if precise && target=="metal" {
                            let mut source=b"#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n".to_vec();
                            source.extend(fs::read(&stem).map_err(|e|e.to_string())?);
                            fs::write(&stem,source).map_err(|e|e.to_string())?;
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        runs.into_iter()
            .filter_map(|run| run.join().unwrap().err())
            .collect()
    });
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    // Preserve active outputs while compiling: warm cached Rust builds can still
    // reference them. Prune only entries absent from this branch's complete set.
    let expected: std::collections::BTreeSet<_> = units.iter().flat_map(|u| {
        ["spv", "metal", "spv.json", "metal.json"].map(|ext| format!("{}.{ext}", u.0))
    }).collect();
    for file in fs::read_dir(&dir).unwrap() {
        let file = file.unwrap().path();
        if file.extension().is_some_and(|ext| ext == "spv" || ext == "metal" || ext == "json")
            && !expected.contains(file.file_name().unwrap().to_str().unwrap()) {
            fs::remove_file(file).unwrap();
        }
    }
    let mut rust = String::new();
    // Each distinct compiled module is one static, so the executable holds its
    // bytes once however many entries and tables name it. (A `const` holding
    // `include_bytes!` is copied into every codegen unit that uses it.)
    let mut embedded = std::collections::BTreeMap::<([u8; 32], bool), String>::new();
    let mut embed = |rust: &mut String, file: String, metal: bool| -> String {
        let code = fs::read(&file).unwrap();
        let next = embedded.len();
        embedded.entry((*blake3::hash(&code).as_bytes(), metal)).or_insert_with(|| {
            let name = format!("EMBEDDED_{next}");
            // A build embeds the form its platform runs: Metal source on Apple
            // (lib.rs supported_backends offers no Vulkan there), SPIR-V everywhere
            // else. Tests embed both. Both identities stay: they are computed here.
            rust.push_str(if metal { "#[cfg(any(target_vendor = \"apple\", test))] " }
                else { "#[cfg(any(not(target_vendor = \"apple\"), test))] " });
            writeln!(rust, "static {name}: [u8; {}] = *include_bytes!({file:?});", code.len()).unwrap();
            name
        }).clone()
    };
    for (name, module, entry, stage) in &units {
        let metal = targets(module).contains(&"metal") && host.iter().any(|h| h == "metal");
        if !metal {
            fs::write(dir.join(format!("{name}.metal")), "").unwrap();
            fs::write(dir.join(format!("{name}.metal.json")), "").unwrap();
        }
        let path = |ext: &str| dir.join(format!("{name}.{ext}")).display().to_string();
        // Metal takes the threadgroup size of a passthrough compute module from the
        // descriptor; wgpu names it num_workgroups. Zero makes every dispatch a no-op.
        let threads = if *stage == "compute" {
            let reflection: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path(if metal { "metal.json" } else { "spv.json" })).unwrap()).unwrap();
            let size = reflection["entryPoints"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["name"] == *entry)
                .and_then(|e| e["threadGroupSize"].as_array())
                .unwrap_or_else(|| panic!("{name}: no threadGroupSize in reflection"));
            let size: Vec<u32> = size.iter().map(|v| v.as_u64().unwrap().try_into().unwrap()).collect();
            (size[0], size[1], size[2])
        } else {
            (0, 0, 0)
        };
        let spirv_key = input_key(&fs::read(path("spv")).unwrap(), entry, false, threads);
        let metal_key = input_key(&fs::read(path("metal")).unwrap(), entry, true, threads);
        let empty_metal_key = input_key(&[], entry, true, threads);
        let spirv = embed(&mut rust, path("spv"), false);
        let msl = embed(&mut rust, path("metal"), true);
        writeln!(
            rust,
            "pub(crate) const {upper}: Compiled = Compiled {{ name: \"{name}\", entry: \"{entry}\", \
             spirv: {{ #[cfg(any(not(target_vendor = \"apple\"), test))] const SPIRV: &[u8] = &{spirv}; \
             #[cfg(not(any(not(target_vendor = \"apple\"), test)))] const SPIRV: &[u8] = &[]; SPIRV }}, \
             msl: {{ #[cfg(any(target_vendor = \"apple\", test))] const MSL: &str = match std::str::from_utf8(&{msl}) {{ Ok(source) => source, Err(_) => panic!(\"Metal source is UTF-8\") }}; \
             #[cfg(not(any(target_vendor = \"apple\", test)))] const MSL: &str = \"\"; MSL }}, \
             input_keys: [{spirv_key:?}, {{ \
             #[cfg(any(target_vendor = \"apple\", test))] const KEY: [u8; 32] = {metal_key:?}; \
             #[cfg(not(any(target_vendor = \"apple\", test)))] const KEY: [u8; 32] = {empty_metal_key:?}; KEY }}], \
             threads: {threads:?}, \
             #[cfg(test)] stage: \"{stage}\", #[cfg(test)] spirv_reflection: include_str!({:?}), \
             #[cfg(test)] msl_reflection: include_str!({:?}) }};",
            path("spv.json"),
            path("metal.json"),
            upper = name.to_uppercase(),
        )
        .unwrap();
    }
    let all: Vec<_> = units
        .iter()
        .map(|u| format!("&{}", u.0.to_uppercase()))
        .collect();
    writeln!(
        rust,
        "#[cfg(test)]\npub(crate) const ALL: &[&Compiled] = &[{}];",
        all.join(", ")
    )
    .unwrap();
    let cameras:Vec<_>=camera_names.iter().map(|(m,r,stage)|format!("({}, {}, \"{}\", &CAMERA_{}_{}_{})",u8::from(*m),u8::from(*r),stage,u8::from(*m),u8::from(*r),stage.to_uppercase())).collect();
    writeln!(rust,"pub(crate) const CAMERAS: &[(u8,u8,&str,&Compiled)]=&[{}];",cameras.join(",")).unwrap();
    let surfaces:Vec<_>=surfaces.iter().map(|(family,rays,probes,coverage,boundaries,name)|format!("(crate::surface_native::SurfaceKernel::{family:?},{rays},{probes},{coverage},{boundaries},&{})",name.to_uppercase())).collect();
    writeln!(rust,"pub(crate) const SURFACES: &[(crate::surface_native::SurfaceKernel,bool,bool,bool,u32,&Compiled)]=&[{}];",surfaces.join(",")).unwrap();
    let path_engines:Vec<_>=path_engines.iter().map(|(engine,name)|format!("(crate::surface_native::PathEngine {{adaptive:{},boundaries:{}}},&{})",engine.adaptive,engine.boundaries,name.to_uppercase())).collect();
    writeln!(rust,"pub(crate) const PATH_ENGINES: &[(crate::surface_native::PathEngine,&Compiled)]=&[{}];",path_engines.join(",")).unwrap();
    let stream_consts: Vec<_> = streams.iter().map(|(a,m,e,n)|format!("({a}, \"{m}\", \"{e}\", &{})",n.to_uppercase())).collect();
    writeln!(rust,"pub(crate) const STREAMS: &[(u32,&str,&str,&Compiled)] = &[{}];",stream_consts.join(",")).unwrap();
    let control_consts: Vec<_> = controls.iter().map(|(a,d,s,h,n)|format!("({a},{d},{s},{h},&{})",n.to_uppercase())).collect();
    writeln!(rust,"pub(crate) const CONTROL_GENERICS: &[(u32,bool,usize,bool,&Compiled)] = &[{}];",control_consts.join(",")).unwrap();
    // Every module a runtime variant compile imports, written to its workspace.
    let mut sources = Vec::new();
    let mut source_identity = blake3::Hasher::new();
    for dir in include {
        let mut files: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        files.sort();
        for file in files
            .into_iter()
            .filter(|f| f.extension().is_some_and(|e| e == "slang"))
        {
            let stem = file.file_stem().unwrap().to_string_lossy().into_owned();
            if dir==&generated && !["std_opacity","vork_object","vork_constants","openpbr_core","vork_openpbr_constants","vork_sampling","program_trace","camera_config","decoupled","material_bindings"].contains(&stem.as_str()) {
                continue;
            }
            let bytes = fs::read(&file).unwrap();
            source_identity.update(&(stem.len() as u64).to_le_bytes());
            source_identity.update(stem.as_bytes());
            source_identity.update(&(bytes.len() as u64).to_le_bytes());
            source_identity.update(&bytes);
            if sources.iter().any(|s:&String|s.starts_with(&format!("(\"{stem}\","))) {continue;}
            sources.push(format!(
                "(\"{stem}\", include_str!({:?}))",
                fs::canonicalize(&file).unwrap()
            ));
        }
    }
    // Table-only runtime variants also import the shared std normal filter.
    let name = "std/normal_filter";
    let file = Path::new("../../libraries/std/normal_filter.slang");
    let bytes = fs::read(file).unwrap();
    source_identity.update(&(name.len() as u64).to_le_bytes());
    source_identity.update(name.as_bytes());
    source_identity.update(&(bytes.len() as u64).to_le_bytes());
    source_identity.update(&bytes);
    sources.push(format!("({name:?}, include_str!({:?}))", fs::canonicalize(file).unwrap()));
    writeln!(
        rust,
        "pub(crate) const SOURCES: &[(&str, &str)] = &[{}];",
        sources.join(", ")
    )
    .unwrap();
    fs::write(out.join("slang_shaders.rs"), rust).unwrap();
    (version, source_identity.finalize())
}
