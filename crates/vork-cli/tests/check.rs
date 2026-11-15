use serde_json::{Value, json};
use std::{io::Write, process::{Command, Output, Stdio}};

fn check(scene: &Value) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
        .args(["check", "-", "--profile", "std/path"])
        // A check must not consult device selection or try this invalid ICD.
        .env("VORK_GPU", "not-a-device")
        .env("VK_ICD_FILENAMES", "/missing/check-must-not-open-vulkan.json")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().unwrap();
    child.stdin.take().unwrap().write_all(scene.to_string().as_bytes()).unwrap();
    child.wait_with_output().unwrap()
}
fn starter() -> Value {
    serde_json::from_str(vork_scene::STARTER_SCENE).unwrap()
}
fn failure(scene: &Value, code: i32, text: &str) {
    let output = check(scene);
    let error = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(code), "{error}");
    assert!(error.contains(text), "expected {text:?}: {error}");
}

#[test]
fn authored_programs_can_use_retired_dotted_names() {
    for name in ["std.particles", "post.white_balance"] {
        let scene = json!({"version":2,"nodes":[{"type":"camera"}],
            "programs":{name:{"source":"import vork; struct Inputs {}; import std.view; ViewResult view(ViewContext c, Inputs i) { return finish_view(float3(1, 0, 0),c,1.); }"}},
            "composition":{"space":"3d","view":{"program":name}},"render":{}});
        let output = check(&scene);
        assert!(output.status.success(), "{name}: {}", String::from_utf8_lossy(&output.stderr));
    }
}

#[test]
fn retired_modules_refuse_in_one_line() {
    for (module, replacement) in [("post/white_balance", "std.white_balance"), ("std/particles", "fluid.particles")] {
        for imported in [false, true] {
            let mut scene = starter();
            let program = if imported {
                json!({"source": format!("import {}; struct Inputs {{}}; import std.view; ViewResult view(ViewContext c, Inputs i) {{return finish_view(float3(0.25),c,1.);}}", module.replace('/', "."))})
            } else { json!(module) };
            scene["composition"] = json!({"space":"3d","view":{"program":program}});
            let output = check(&scene);
            let error = String::from_utf8_lossy(&output.stderr);
            assert_eq!(output.status.code(), Some(3), "{error}");
            assert_eq!(error.lines().count(), 1, "{error}");
            assert!(error.contains(replacement) && error.contains("vork help upgrade"), "{error}");
        }
    }
}

#[test]
fn renamed_math_member_refuses_in_one_line() {
    let mut scene = starter();
    scene["nodes"][0]["materials"] = json!([{"program": {"source":
        "import blender.nodes; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return unlit(float3(Math(MathOperation.FRACTION, 1.25, 0., 0., false))); }"
    }}]);
    let output = check(&scene);
    let error = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(3), "{error}");
    assert_eq!(error.lines().count(), 1, "{error}");
    assert!(error.contains("FRACTION"), "{error}");
    assert!(error.split(|c: char| !c.is_ascii_alphanumeric()).any(|token| token == "FRACT"), "{error}");
}

#[test]
fn check_expands_and_installs_without_a_gpu() {
    let mut scene = starter();
    for name in ["box", "sphere", "grid", "cylinder", "cone", "torus", "capsule", "revolve"] {
        let geometry = if name == "revolve" {
            json!({"std":name,"profile":[[0,0],[1,0],[1,1],[0,1]]})
        } else { json!({"std":name}) };
        scene["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":geometry}));
    }
    let output = check(&scene);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "Scene checked successfully");
}

#[test]
fn check_rejects_schema_program_assets_and_timeline() {
    let mut scene = starter();
    scene["render"]["typo"] = json!(true);
    failure(&scene, 3, "typo");
    let mut scene = starter();
    scene["composition"] = json!({"space":"3d","view":{"program":"std/missing"}});
    failure(&scene, 3, "std/missing");
    let mut scene = starter();
    scene["render"]["duration"] = json!(-1);
    failure(&scene, 3, "render requires positive dimensions/time");
    let mut scene = starter();
    scene["nodes"].as_array_mut().unwrap().push(json!({"type":"asset","file":"missing-check-asset.glb"}));
    failure(&scene, 5, "missing-check-asset.glb");
}

#[test]
fn check_uses_fog_feature_admission() {
    let mut scene = starter();
    scene["materials"] = json!({"fog":{"program":{"source":"import vork; struct Inputs {}; MediumCoefficients medium(MediumContext ctx, Inputs inputs) { MediumCoefficients m = {}; m.scattering = float3(0.02); return m; }"}}});
    scene["nodes"].as_array_mut().unwrap().push(json!({"type":"light", "shape":"environment", "medium":"fog"}));
    // Implicit geometry is admitted without opening a GPU or compiling native shaders.
    scene["buffers"]["grid"] = json!({"type":"f32", "dims":[2,2,2], "data":[0,0,0,0,0,0,0,0]});
    scene["geometry"] = json!({"field":{"topology":"implicit", "attributes":{"density":"grid"}}});
    let output = check(&scene);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    scene["materials"]["custom"] = json!({"program":{"source":"import vork; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return Closure(); }"}});
    scene["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh", "geometry":"field", "materials":["custom"]}));
    failure(&scene, 3, "material fog: fog with custom surface callbacks is not supported; remove its medium stage or use std/openpbr for surface materials");
}

#[test]
fn file_base_and_optional_shader_compilation_share_render_installation() {
    let dir = std::env::temp_dir().join(format!("vork-check-files-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("display.slang"), "import vork; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs inputs) { return finish_view(ctx.color,ctx,1.); }").unwrap();
    let mut scene = starter();
    scene["composition"] = json!({"space":"3d","view":{"program":{"file":"display.slang"}}});
    std::fs::write(dir.join("scene.json"), scene.to_string()).unwrap();
    for shaders in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vork"));
        command.arg("check").arg(dir.join("scene.json"))
            .env("VORK_CACHE_DIR", dir.join("cache"))
            .env("VORK_GPU", "not-a-device");
        if shaders { command.arg("--shaders"); }
        let output = command.output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        for file in std::fs::read_dir(dir.join("cache")).unwrap() {
            let name = file.unwrap().file_name().to_string_lossy().into_owned();
            assert!(!name.contains("pipeline-keys") && !name.contains("shader-warm"), "check wrote driver cache ledger {name}");
            if !shaders { assert!(!name.contains("slang-variant"), "default check generated native shader {name}"); }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn water_example_checks_without_a_gpu() {
    let output = Command::new(env!("CARGO_BIN_EXE_vork"))
        .arg("check")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/water/scene.json"))
        .env("VORK_GPU", "not-a-device")
        .env("VK_ICD_FILENAMES", "/missing/check-must-not-open-vulkan.json")
        .output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "Scene checked successfully");
}

#[test]
fn missing_fonts_are_reported_together() {
    let scene = json!({"version":2,"nodes":[{"type":"camera"}],"geometry":{
        "one":{"std":"text","body":"first","font":"missing-first.ttf"},
        "two":{"std":"text","body":"second","font":"missing-second.otf"}}});
    let output = check(&scene);
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    for name in ["missing-first.ttf", "missing-second.otf", "restore the font file or correct its path"] {assert!(error.contains(name), "{error}");}
    assert_eq!(error.lines().count(), 1, "{error}");
    assert_eq!(error.matches("read font").count(), 2, "{error}");
}

#[test]
fn gaussian_default_material_names_do_not_depend_on_asset_paths() {
    let dir=std::env::temp_dir().join(format!("vork-check-splats-{}",std::process::id()));
    std::fs::create_dir_all(dir.join("captures with spaces")).unwrap();
    let mut ply=String::from("ply\nformat ascii 1.0\nelement vertex 1\n");
    for name in ["x","y","z","scale_0","scale_1","scale_2","opacity","rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"] {
        ply.push_str(&format!("property float {name}\n"));
    }
    ply.push_str("end_header\n0 0 0 0 0 -2 0 1 0 0 0 0.1 0.2 0.3\n");
    let relative="captures with spaces/point.ply";
    std::fs::write(dir.join(relative),ply).unwrap();
    let scene=json!({"version":2,"materials":{"Gaussian radiance":{},"Gaussian radiance_":{}},
        "nodes":[{"type":"asset","name":"relative","file":relative},
            {"type":"asset","name":"absolute","file":dir.join(relative)},
            {"type":"camera","translation":[0,0,3]}],"render":{"resolution":[8,8]}});
    std::fs::write(dir.join("scene.json"),scene.to_string()).unwrap();
    let output=Command::new(env!("CARGO_BIN_EXE_vork")).arg("check").arg(dir.join("scene.json"))
        .env("VORK_GPU","not-a-device").env("VK_ICD_FILENAMES","/missing/check-must-not-open-vulkan.json")
        .output().unwrap();
    assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    // Geometry-dependent closures must refuse before a shader can read a
    // triangle boundary that point geometry does not have.
    for (inputs,message) in [
        (json!({"transmission_weight":1}),"no closed dielectric boundary"),
        (json!({"subsurface_weight":1}),"no solid interior for subsurface scattering"),
    ] {
        let mut replacement=scene.clone();
        replacement["materials"]=json!({"surface":{"inputs":inputs}});
        for node in replacement["nodes"].as_array_mut().unwrap().iter_mut().take(2) {
            node["materials"]=json!({"*":"surface"});
        }
        std::fs::write(dir.join("refused.json"),replacement.to_string()).unwrap();
        let output=Command::new(env!("CARGO_BIN_EXE_vork")).arg("check").arg(dir.join("refused.json"))
            .env("VORK_GPU","not-a-device").env("VK_ICD_FILENAMES","/missing/check-must-not-open-vulkan.json")
            .output().unwrap();
        let error=String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success() && error.contains(message),"{error}");
        assert_eq!(error.lines().filter(|line|line.starts_with("Error:")).count(),1,"one refusal line");
    }
    // Program trace reports triangle boundaries only, so a scene with splats
    // refuses it instead of tracing straight through the points.
    std::fs::write(dir.join("probe.slang"),"import vork;\npublic struct Inputs {};\npublic Closure surface(ShadingContext ctx,Inputs i) {\n    let h=trace(ctx.POSITION,float3(0.,1.,0.),0.001,10.);\n    Closure c;c.base_color=h.hit ? float3(0.) : float3(1.);return c;\n}\n").unwrap();
    let mut traced=scene.clone();
    traced["programs"]=json!({"probe":{"file":"probe.slang"}});
    traced["materials"]=json!({"probe":{"program":"probe"}});
    traced["nodes"].as_array_mut().unwrap().push(json!({"type":"mesh","geometry":{"std":"sphere","radius":0.5},"materials":["probe"]}));
    std::fs::write(dir.join("traced.json"),traced.to_string()).unwrap();
    let output=Command::new(env!("CARGO_BIN_EXE_vork")).arg("check").arg(dir.join("traced.json"))
        .env("VORK_GPU","not-a-device").env("VK_ICD_FILENAMES","/missing/check-must-not-open-vulkan.json")
        .output().unwrap();
    let error=String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() && error.contains("trace does not support Gaussian splats"),"{error}");
    assert_eq!(error.lines().filter(|line|line.starts_with("Error:")).count(),1,"one refusal line");
    // The same program on the mesh alone remains valid.
    traced["nodes"].as_array_mut().unwrap().drain(0..2);
    std::fs::write(dir.join("traced-mesh.json"),traced.to_string()).unwrap();
    let output=Command::new(env!("CARGO_BIN_EXE_vork")).arg("check").arg(dir.join("traced-mesh.json"))
        .env("VORK_GPU","not-a-device").env("VK_ICD_FILENAMES","/missing/check-must-not-open-vulkan.json")
        .output().unwrap();
    assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn image_stroke_agrees_with_its_schema() {
    let image = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vork-render/tests/fixtures/shape-labels.png");
    let output = check(&json!({
        "version": 2, "composition": {"space": "2d"},
        "images": {"picture": {"file": image}},
        "nodes": [{"type": "path", "d": "M8 8H56V56H8Z", "stroke": {"image": "picture"}}],
        "render": {"resolution": [64, 64]}
    }));
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
}
