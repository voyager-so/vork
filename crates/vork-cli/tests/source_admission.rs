use serde_json::{Value, json};
use std::{io::Write, process::{Command, Output, Stdio}};

fn check(scene: &Value) -> Output { check_profile(scene, "std/path") }

fn check_profile(scene: &Value, profile: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
        .args(["check", "-", "--profile", profile])
        // A check must not consult device selection or try this invalid ICD.
        .env("VORK_GPU", "not-a-device")
        .env("VK_ICD_FILENAMES", "/missing/check-must-not-open-vulkan.json")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().unwrap();
    child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn source_admission_reports_independent_files_and_references_before_device_startup() {
    let scene = json!({"version":2,"nodes":[{"type":"camera","material":"filemat"},{"type":"light","shape":"point","image":"ies"},
        {"type":"light","shape":"environment","material":"two"},{"type":"asset","file":"missing-one.gltf"},{"type":"asset","file":"unsupported-two.obj"}],
        "buffers":{"first":{"type":"f32","file":"missing-first.bin"},"second":{"type":"f32","file":"missing-second.bin"}},
        "programs":{"file":{"file":"missing-program.slang"}},
        "images":{"ies":{"file":"missing-profile.ies"},"clip":{"file":"missing-movie.mp4"}},
        "materials":{"filemat":{"program":"file"},"one":{"program":"missing-first-program","inputs":{"map":{"image":"ies"}}},
            "two":{"program":"std/color","inputs":{"map":{"image":"clip"}}}}});
    for render in [false, true] {
        let output = if render {
            let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
                .args(["render", "-", "--out", "/tmp/vork-refuseearly-must-not-write.png"])
                .env("VORK_GPU", "not-a-device").env("VK_ICD_FILENAMES", "/missing/no-native-device.json")
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
            child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
            child.wait_with_output().unwrap()
        } else {check(&scene)};
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{error}");
        for expected in ["missing-one.gltf", "unsupported-two.obj", "missing-first.bin", "missing-second.bin",
            "missing-program.slang", "missing-profile.ies", "missing-movie.mp4", "missing-first-program", "library-only module"] {
            assert!(error.contains(expected), "missing {expected}: {error}");
        }
        assert!(!error.contains("not-a-device"), "native device selection ran: {error}");
        assert!(!error.contains("preparing scene, shaders and encoder"), "{error}");
    }
}

#[test]
fn cheap_refusals_keep_actionable_fixes() {
    for (scene, expected) in [
        (json!({"version":123,"nodes":[{"type":"camera"}]}), "rewrite the scene"),
        (json!({"version":2,"nodes":[{"type":"camera"},{"type":"asset","file":"scene.fbx"}]}), "glTF/GLB"),
        (json!({"version":2,"nodes":[{"type":"camera"}],"buffers":{"grid":{"type":"f32","grid":"density","file":"missing-volume.vdb"}}}), "restore the grid file"),
        (json!({"version":2,"nodes":[{"type":"camera"}],"materials":{"bad":{"program":"unknown-program"}}}), "choose a material program"),
    ] { let output=check(&scene);let error=String::from_utf8_lossy(&output.stderr);assert!(!output.status.success());assert!(error.contains(expected),"{error}"); }
}

#[test]
fn buffer_ranges_sequence_holes_and_light_ies_are_admitted_together() {
    let root = std::env::temp_dir().join(format!("vork-source-ranges-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("short.bin"), [0; 4]).unwrap();
    for n in [0, 2] {std::fs::create_dir_all(root.join(format!("plate{n:04}.png"))).unwrap();}
    std::fs::create_dir_all(root.join("buffer-directory.bin")).unwrap();
    let scene = json!({"version":2,"nodes":[{"type":"camera"},{"type":"light","shape":"point","image":"ies"},
        {"type":"mesh","geometry":{"std":"box"},"materials":["plate"]}],
        "buffers":{"truncated":{"type":"f32","file":root.join("short.bin"),"count":4},
            "directory":{"type":"f32","file":root.join("buffer-directory.bin"),"count":1}},
        "images":{"sequence":{"file":root.join("plate%04d.png")},"ies":{"file":root.join("missing-local.ies")}},
        "materials":{"plate":{"inputs":{"base_color":{"image":"sequence"}}}}});
    let output = check(&scene);
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    for expected in ["truncated", "too short", "missing frames: 1", "plate0000.png", "plate0002.png", "buffer-directory.bin", "missing-local.ies", "restore"] {
        assert!(error.contains(expected), "missing {expected}: {error}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn source_admission_keeps_unused_program_and_image_files_unopened() {
    let scene = json!({"version":2,"nodes":[{"type":"camera"}],
        "programs":{"unused":{"file":"unused-missing-program.slang"}},
        "materials":{"unused":{"program":"unused","inputs":{"map":{"image":"unused"}}}},
        "images":{"unused":{"file":"unused-missing-image.png"}}});
    let output = check(&scene);
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{error}");
    assert!(!error.contains("unused-missing"), "{error}");
}

#[test]
fn installation_refusals_keep_their_source_and_fix() {
    let root = std::env::temp_dir().join(format!("vork-install-refusals-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    // Self-contained valid PNG: renderer fixtures are private in source exports.
    std::fs::write(root.join("map.png"), &[
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196,
        137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 240,
        31, 0, 5, 0, 1, 255, 137, 153, 61, 29, 0, 0, 0, 0, 73, 69,
        78, 68, 174, 66, 96, 130,
    ]).unwrap();
    std::fs::write(root.join("tilt.ies"), b"IESNA:LM-63-2002\nTILT=INCLUDE\n").unwrap();
    for (scene, expected) in [
        (json!({"version":2,"nodes":[{"type":"camera"}],"programs":{"broken":{"source":"this is not Slang;"}},
            "composition":{"space":"3d","view":{"program":"broken"}},"render":{"resolution":[16,16]}}), "broken"),
        (json!({"version":2,"nodes":[{"type":"camera"},{"type":"light","shape":"point","image":"ies"}],
            "render":{"resolution":[16,16]},"images":{"ies":{"file":root.join("tilt.ies")}}}), "re-export the profile without a tilt table"),
        (json!({"version":2,"nodes":[{"type":"camera"},{"type":"mesh","geometry":"triangle","materials":["paint"]}],
            "buffers":{"p":{"type":"vec3","data":[[-1,-1,-3],[1,-1,-3],[0,1,-3]]}},
            "render":{"resolution":[16,16]},"geometry":{"triangle":{"attributes":{"POSITION":"p"}}},
            "images":{"map":{"file":root.join("map.png")}},"materials":{"paint":{"inputs":{"base_color":{"image":"map"}}}}}),
            "add that UV set to the geometry"),
    ] {
        let output = check(&scene);
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{scene}: {error}");
        assert!(error.contains(expected), "missing {expected}: {error}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn changing_image_displacement_keeps_its_warning() {
    let scene = json!({"version":2,"render":{"resolution":[16,16]},"nodes":[{"type":"camera","name":"camera"},{"type":"node","name":"empty"},
        {"type":"mesh","geometry":{"std":"box"},"materials":["surface"]}],
        "images":{"capture":{"render":{"camera":"camera","root":"empty"}}},
        "programs":{"surface":{"source":"struct Inputs {Image image;}; Closure surface(ShadingContext c, Inputs i) {return unlit(float3(.5));} float3 displace(DisplacementContext c, Inputs i) {return sample(i.image,c.POSITION.xy,c).rgb*.1;}"}},
        "materials":{"surface":{"program":"surface","inputs":{"image":{"image":"capture"}}}}});
    let output = check_profile(&scene, "std/final");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{error}");
    assert!(error.contains("displacement_image"), "{error}");
}

#[test]
fn missing_display_data_refuses_instead_of_substituting() {
    let scene = json!({"version":2,"nodes":[{"type":"camera"}],
        "composition":{"space":"3d","view":{"program":"std/agx","inputs":{"lut":{"image":"lut"}}}},"render":{},
        "images":{"lut":{"file":"missing-colour-transform.exr"}}});
    let output=check(&scene);let error=String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(error.contains("restore display image missing-colour-transform.exr"),"{error}");
    assert!(!error.contains("missing_image"),"{error}");
}

#[test]
fn imported_sources_collect_missing_and_truncated_buffers_before_native_startup() {
    let root = std::env::temp_dir().join(format!("vork-imported-sources-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("short buffer.bin"), [0; 4]).unwrap();
    std::fs::write(root.join("target.gltf"), json!({"asset":{"version":"2.0"}}).to_string()).unwrap();
    // A directory is an invalid image source on every platform, without relying
    // on permissions (which are ineffective when tests run as root).
    std::fs::create_dir_all(root.join("image-directory.png")).unwrap();
    std::fs::create_dir_all(root.join("environment.hdr")).unwrap();
    for name in ["mesh.gltf", "clip.gltf"] {
        std::fs::write(root.join(name), json!({"asset":{"version":"2.0"},
            "buffers":[{"uri":"missing-one.bin","byteLength":36},
                {"uri":"short%20buffer.bin","byteLength":36},
                {"uri":"missing-two.bin","byteLength":12}],
            "images":[{"uri":"image-directory.png"}]}).to_string()).unwrap();
    }
    for clip in [false, true] {
        let mut scene = json!({"version":2,"nodes":[{"type":"camera","name":"camera"}]});
        // The imported failures must aggregate with ordinary source failures.
        scene["nodes"].as_array_mut().unwrap().extend([
            json!({"type":"light","shape":"point","image":"ies"}),
            json!({"type":"light","shape":"environment","material":"environment"}),
            json!({"type":"mesh","geometry":{"std":"box"},"materials":["movie"]})
        ]);
        scene["images"] = json!({"ies":{"file":root.join("missing.ies")},
            "hdr":{"file":root.join("environment.hdr")},
            "movie":{"file":root.join("missing.mp4")}});
        scene["materials"] = json!({"movie":{"inputs":{"base_color":{"image":"movie"}}},
            "environment":{"program":"std/environment_image","inputs":{"image":{"image":"hdr"}}}});
        if clip {
            scene["nodes"].as_array_mut().unwrap().push(json!({"type":"asset","name":"rig","file":root.join("target.gltf")}));
            scene["animation"] = json!([{"clip":{"target":"rig","file":root.join("clip.gltf"),"index":0}}]);
        } else {
            scene["nodes"].as_array_mut().unwrap().push(json!({"type":"asset","file":root.join("mesh.gltf")}));
        }
        let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
            .args(["render", "-", "--out", "/tmp/vork-imported-sources-must-not-write.png"])
            .env("VORK_GPU", "not-a-device")
            .env("VK_ICD_FILENAMES", "/missing/no-native-device.json")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
        let output = child.wait_with_output().unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{error}");
        for expected in ["missing-one.bin", "missing-two.bin", "short%20buffer.bin", "expected 36 bytes",
            "image-directory.png", "environment.hdr", "missing.ies", "missing.mp4", "restore the buffer file", "restore the image file"] {
            assert!(error.contains(expected), "missing {expected}: {error}");
        }
        assert!(!error.contains("not-a-device"), "native device startup ran: {error}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[cfg(unix)]
fn nonregular_document_and_font_sources_refuse_without_blocking() {
    let root = std::env::temp_dir().join(format!("vork-nonregular-sources-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let pipe = root.join("source.gltf");
    assert!(Command::new("mkfifo").arg(&pipe).status().unwrap().success());
    let target = root.join("target.gltf");
    std::fs::write(&target, json!({"asset":{"version":"2.0"}}).to_string()).unwrap();
    let camera = json!({"type":"camera","name":"camera"});
    let scenes = [
        json!({"version":2,"nodes":[camera.clone(),{"type":"asset","file":pipe}]}),
        json!({"version":2,"nodes":[camera.clone(),{"type":"asset","name":"rig","file":target}],"animation":[{"clip":{"target":"rig","file":pipe,"index":0}}]}),
        json!({"version":2,"nodes":[camera,{"type":"mesh","geometry":{"std":"text","body":"A","font":pipe}}]}),
    ];
    for scene in scenes {
        let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
            .args(["render", "-", "--out", "/tmp/vork-nonregular-must-not-write.png"])
            .env("VORK_GPU", "not-a-device")
            .env("VK_ICD_FILENAMES", "/missing/no-native-device.json")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
        // Bound the child so a blocking open is a test failure, not a stuck suite.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while child.try_wait().unwrap().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if child.try_wait().unwrap().is_none() {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!("source admission blocked on {scene}: {}", String::from_utf8_lossy(&output.stderr));
        }
        let output = child.wait_with_output().unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(5), "{scene}: {error}");
        for expected in ["source.gltf", "not a regular file", "restore"] {
            assert!(error.contains(expected), "missing {expected}: {error}");
        }
        assert!(!error.contains("not-a-device"), "native startup ran: {error}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn sibling_io_failures_keep_io_exit_and_schema_failures_keep_precedence() {
    for count in [1, 2, 3] {
        let buffers: serde_json::Map<String, Value> = (0..count).map(|i|
            (format!("buffer{i}"), json!({"type":"f32","file":format!("missing-category-{i}.bin")}))).collect();
        let scene = json!({"version":2,"nodes":[{"type":"camera"}],"buffers":buffers});
        let output = check(&scene);
        let error = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(5), "{error}");
        for i in 0..count {
            assert!(error.contains(&format!("missing-category-{i}.bin")), "{error}");
        }
        assert!(error.contains("restore"), "{error}");
        let mut mixed = scene;
        mixed["materials"] = json!({"bad":{"program":"unknown-category-program"}});
        let output = check(&mixed);
        let error = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(3), "{error}");
        assert!(error.contains("unknown-category-program") && error.contains("missing-category-0.bin"), "{error}");
    }
}

#[test]
fn imported_buffer_io_keeps_the_importer_exit_category() {
    let root = std::env::temp_dir().join(format!("vork-imported-category-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for count in [1, 2] {
        let buffers: Vec<Value> = (0..count).map(|i| json!({"uri":format!("missing-{i}.bin"),"byteLength":4})).collect();
        let path = root.join("source.gltf");
        std::fs::write(&path, json!({"asset":{"version":"2.0"},"buffers":buffers}).to_string()).unwrap();
        let output = check(&json!({"version":2,"nodes":[{"type":"camera"},{"type":"asset","file":path}]}));
        let error = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(3), "{error}");
        for i in 0..count {assert!(error.contains(&format!("missing-{i}.bin")), "{error}");}
        assert!(error.contains("restore the buffer file"), "{error}");
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn fog_callback_refusal_uses_source_stages_before_device_startup() {
    let medium = "import vork; struct Inputs {}; MediumCoefficients medium(MediumContext c, Inputs i) { return MediumCoefficients(); }";
    let surface = "import vork; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return Closure(); }";
    for world in [false, true] {
        let mut nodes = vec![json!({"type":"camera"}), json!({"type":"mesh","geometry":{"std":"box"},"materials":["paint"]})];
        nodes.push(if world { json!({"type":"light","shape":"environment","medium":"mist"}) }
            else { json!({"type":"mesh","geometry":{"std":"box"},"materials":["mist"]}) });
        let scene = json!({"version":2,"nodes":nodes,"programs":{"volume":{"source":medium},"surface":{"source":surface}},
            "materials":{"mist":{"program":"volume"},"paint":{"program":"surface"}}});
        for render in [false, true] {
            let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
                .args(if render { vec!["render", "-", "--profile", "std/path", "--out", "/tmp/vork-fog-admission-must-not-write.png"] }
                    else { vec!["check", "-", "--profile", "std/path"] })
                .env("VORK_GPU", "not-a-device").env("VK_ICD_FILENAMES", "/missing/fog-must-not-open-vulkan.json")
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
            child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
            let output=child.wait_with_output().unwrap();
            let error=String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success(), "{error}");
            assert!(error.contains("material mist: fog with custom surface callbacks is not supported; remove its medium stage or use std/openpbr for surface materials"), "{error}");
            assert!(!error.contains("not-a-device") && !error.contains("preparing scene, shaders and encoder"), "{error}");
            assert_eq!(error.lines().filter(|line|line.starts_with("Error:")).count(),1,"{error}");
        }
    }
}

#[test]
fn fog_source_admission_keeps_unconsumed_slots_and_builtin_surfaces() {
    let medium = "import vork; struct Inputs {}; MediumCoefficients medium(MediumContext c, Inputs i) { return MediumCoefficients(); }";
    let surface = "import vork; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return Closure(); }";
    let mut scene=json!({"version":2,"nodes":[{"type":"camera"},
        {"type":"mesh","geometry":{"std":"box"},"materials":["paint","mist"]}],
        "programs":{"volume":{"source":medium},"surface":{"source":surface}},
        "materials":{"mist":{"program":"volume"},"paint":{"program":"surface"}}});
    let output=check(&scene);
    assert!(output.status.success(),"unused medium slot: {}",String::from_utf8_lossy(&output.stderr));
    scene["materials"]["paint"]=json!({"program":"std/openpbr"});
    scene["nodes"].as_array_mut().unwrap().push(json!({"type":"light","shape":"environment","medium":"mist"}));
    let output=check(&scene);
    assert!(output.status.success(),"builtin surfaces with fog: {}",String::from_utf8_lossy(&output.stderr));
}

#[test]
fn surface_only_source_admission_does_not_start_slang() {
    let root = std::env::temp_dir().join(format!("vork-dry-admission-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (index, source) in [
        "import vork; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return Closure(); }",
        "import \n vork ; import std.texture; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { Closure s = {}; s.base_color = float3(.2,.6,.8); return s; } // colour edit",
    ].into_iter().enumerate() {
        let file = root.join("paint.slang");
        std::fs::write(&file, source).unwrap();
        for program in [json!({"source":source}), json!({"file":file})] {
            let scene = json!({"version":2,"nodes":[{"type":"camera"},
                {"type":"mesh","geometry":{"std":"box"},"materials":["paint"]}],
                "materials":{"paint":{"program":program}}});
            let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
                .args(["render", "-", "--profile", "std/path", "--out"])
                .arg(root.join("must-not-render.png"))
                .env("VORK_SLANGC", root.join("missing-slangc"))
                .env("VORK_CACHE_DIR", root.join(format!("cache-{index}")))
                .env("WGPU_BACKEND", "not-a-backend")
                .env("VK_ICD_FILENAMES", "/missing/dry-admission-must-not-open-vulkan.json")
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
            child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
            let output = child.wait_with_output().unwrap();
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains("unsupported WGPU_BACKEND"), "source admission started the compiler: {error}");
            assert!(!root.join("must-not-render.png").exists());
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn indirect_medium_declarations_still_refuse_before_device_startup() {
    let root = std::env::temp_dir().join(format!("vork-indirect-fog-admission-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let medium = "import vork; struct Inputs {}; MediumCoefficients medium(MediumContext c, Inputs i) { return MediumCoefficients(); }";
    std::fs::write(root.join("mist_impl.slang"), medium).unwrap();
    for source in [
        "import mist_impl;".to_owned(),
        "import mist_impl; Closure surface(ShadingContext c, Inputs i) { return Closure(); }".to_owned(),
        "#include \"mist_impl.slang\"".to_owned(),
        format!("#define ENTRY medium\n{}", medium.replace("medium(", "ENTRY(")),
    ] {
        let file = root.join("carrier.slang");
        std::fs::write(&file, source).unwrap();
        let scene = json!({"version":2,"nodes":[{"type":"camera"},
            {"type":"mesh","geometry":{"std":"box"},"materials":["paint"]},
            {"children":[{"type":"mesh","geometry":{"std":"box"},"materials":["mist"]}]}],
            "materials":{"mist":{"program":{"file":file}},
                "paint":{"program":{"source":"import vork; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return Closure(); }"}}}});
        let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
            .args(["render", "-", "--profile", "std/path", "--out"])
            .arg(root.join("must-not-render.png"))
            .env("WGPU_BACKEND", "not-a-backend")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
        let output = child.wait_with_output().unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("material mist: fog with custom surface callbacks is not supported"), "{error}");
        assert!(!error.contains("unsupported WGPU_BACKEND"), "{error}");
        assert_eq!(error.lines().filter(|line| line.starts_with("Error:")).count(), 1, "{error}");
        assert!(!root.join("must-not-render.png").exists());
    }
    std::fs::remove_dir_all(root).unwrap();
}

// Keep the subprocess bound here: a regression must fail instead of hanging the suite.
fn check_canvas_bounded(scene: &Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
        .args(["check", "-"])
        .env("VORK_GPU", "not-a-device")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if child.try_wait().unwrap().is_some() { return child.wait_with_output().unwrap(); }
        if std::time::Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("canvas admission exceeded five seconds");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[test]
fn qa_2d_tiny_dash_refuses_excessive_expansion_in_one_line() {
    for (pattern, offset, length) in [
        (json!([1e-8, 1e-8]), 0., None),
        (json!([0., 1e-8, 1e-8]), 3., None),
        (json!([1., 1.]), -2., Some(1e12)),
    ] {
        let mut path = json!({"name":"box","type":"path","d":"M20 20H110V100H20Z",
            "fill":"red","stroke":"white","stroke-width":4,"stroke-dasharray":pattern,"stroke-dashoffset":offset});
        if let Some(length) = length { path["pathLength"] = json!(length); }
        let output = check_canvas_bounded(&json!({"version":2,"composition":{"space":"2d"},"nodes":[path]}));
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{error}");
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(error.contains("stroke-dasharray") && error.contains("budget"), "{error}");
    }
}

#[test]
fn qa_2d_reference_cycles_refuse_even_when_unused_or_invisible() {
    for (first, second) in [("clip-path", "clip-path"), ("mask", "mask"), ("clip-path", "mask")] {
        let mut a = json!({"name":"a","visible":false});
        let mut b = json!({"name":"b"});
        a[first] = json!("b"); b[second] = json!("a");
        let output = check_canvas_bounded(&json!({"version":2,"composition":{"space":"2d"},"nodes":[a,b]}));
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "cycle admitted");
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(error.contains("cycle") && (error.contains("node \"a\"") || error.contains("node \"b\"")), "{error}");
    }
    for field in ["clip-path", "mask"] {
        let mut node = json!({"name":"self"}); node[field] = json!("self");
        let output = check_canvas_bounded(&json!({"version":2,"composition":{"space":"2d"},"nodes":[node]}));
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success() && error.lines().count() == 1 && error.contains("cycle"), "{error}");
    }
    let output = check_canvas_bounded(&json!({"version":2,"composition":{"space":"2d"},
        "nodes":[{"name":"parent","children":[{"name":"child","mask":"parent"}]}]}));
    assert!(!output.status.success(), "cycle through a child admitted");
    assert!(String::from_utf8_lossy(&output.stderr).contains("cycle"));
    // Shared definitions form a DAG, which remains legal.
    let output = check_canvas_bounded(&json!({"version":2,"composition":{"space":"2d"},
        "nodes":[{"name":"a","mask":"shared"},{"name":"b","clip-path":"shared"},{"name":"shared"}]}));
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}
