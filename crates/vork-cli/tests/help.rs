use std::process::Command;

fn help_args(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_vork"))
        .arg("help").args(args)
        .output()
        .unwrap();
    assert!(output.status.success(), "help {args:?}: {output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn help(topic: &str) -> String { help_args(&[topic]) }

#[test]
fn primitive_and_object_topics_resolve_generated_fields() {
    for topic in ["text", "path", "composition", "camera", "mesh", "light", "asset"] {
        let fields = help_args(&["field", topic]);
        assert_eq!(help(topic), fields, "{topic}");
        assert_eq!(help_args(&[topic, "--all"]), fields, "{topic}");
    }
    assert!(help("text").contains("node.font-size (text):"));
    assert!(!help("text").contains("node.d (path):"));
    assert!(help("path").contains("node.pathLength (path):"));
    assert!(!help("path").contains("node.font (text):"));
    assert!(help("composition").contains("composition.space:"));
}

#[test]
fn help_routes_concepts_fields_and_diagnostics_without_losing_the_schema() {
    for (topic, knob) in [("profiles", "settings.seed"), ("profiles", "settings.denoise")] {
        assert!(!help(topic).lines().any(|line| line.starts_with(knob)), "field line in the {topic} concept page");
        let all = help_args(&[topic, "--all"]);
        assert!(all.lines().any(|line| line.starts_with(knob) && line.contains("; default ")), "{topic} --all lost {knob}");
    }
    let root = help_args(&[]);
    for (topic, field, declaration) in [
        ("cameras", "node.material (camera):", "material: material-name|inline material|null = null; omitted = std/camera"),
        ("lights", "node.energy (light):", "energy: number|null; sun W/m²; local W; environment multiplier"),
        ("nodes", "node.geometry (mesh):", "geometry: geometry-name|inline geometry"),
    ] {
        assert!(root.contains(declaration), "root lost {field}");
        assert!(!help_args(&[topic, "--all"]).contains(field), "duplicate {field}");
    }
    assert!(help_args(&["profiles", "--all"]).contains("settings.seed:"));
    assert!(!help("render").contains("settings.seed:"));
    assert!(help("diagnostics").contains("VORK_CACHE_BYTES"));
    assert!(help_args(&["diagnostics", "--all"]).contains("VORK_CACHE_BYTES"));
    let outputs = help("outputs");
    assert!(root.contains("repeat --out PATH") && !outputs.contains("<!--"));
    // Decoding contracts an agent needs to read outputs stay in the default page.
    for contract in ["camera.json", "depth_unit_m", "view_normal RGB8", "motion RGB16", "EXR channels", "aov"] {
        assert!(outputs.contains(contract), "default outputs help lost {contract}");
    }
    let outputs_all = help_args(&["outputs", "--all"]);
    for detail in ["EXR channels:", "--exr-codec", "OUT.meta.json"] {
        assert!(outputs_all.contains(detail), "outputs --all lost {detail}");
    }
    assert!(root.contains("all includes render.aovs"));
    assert!(root.contains("applies to every --out"));
    let std_all = help_args(&["std", "--all"]);
    // Deep library signatures and inputs remain one --all away.
    for (library, detail) in [("std", "std.color: blackbody"), ("blender", "blender/filmic look:")] {
        assert!(!help(library).contains(detail), "{library} default shows {detail}");
        assert!(help_args(&[library, "--all"]).contains(detail), "{library} --all lost {detail}");
    }
    // Participating media remain internal; launch help must not advertise them.
    let blender_all = help_args(&["blender", "--all"]);
    for removed in ["absorption_color", "blender.volume", "blender/principled_volume"] {
        assert!(!blender_all.contains(removed), "blender --all advertises removed media: {removed}");
    }
    assert!(!std_all.contains("scatter:"));
    // Root keeps the fragment contract; retained signatures remain on demand.
    assert!(root.contains("Std builders return {buffers,geometry}"));
    let std_page = help("std");
    let programs_page = help("programs");
    for builder in ["box", "sphere", "grid", "cylinder", "cone", "torus", "capsule", "revolve", "sweep"] {
        let signature = format!("{builder}:");
        assert!(!root.contains(&signature), "root includes advanced signature {signature}");
        assert!(std_all.lines().any(|line| line.starts_with(&signature)), "std --all missing {builder}");
        assert!(std_page.contains(builder), "std page missing {builder}");
        assert!(programs_page.contains(builder), "programs page missing {builder}");
    }
    assert!(std_all.contains("rounded_box:"));
    for page in [&root, &std_all] {
        assert!(page.contains("omit name: the table key or inline position owns identity"));
    }
    assert!(std_all.contains("vork std expand - < scene.json > expanded.json"));
    assert!(std_all.contains("Expansion does not validate the whole scene"));
    assert!(!root.contains("rounded_box:"));
    let command_help = Command::new(env!("CARGO_BIN_EXE_vork")).args(["std", "--help"]).output().unwrap();
    assert!(command_help.status.success());
    assert!(!String::from_utf8_lossy(&command_help.stdout).contains("rounded_box:"));
    let schema: serde_json::Value = serde_json::from_str(include_str!("../../../packages/ts/scene.schema.json")).unwrap();
    assert_eq!(schema["$defs"]["Settings"]["properties"]["denoise"]["default"], false);
    assert!(schema["properties"]["render"]["default"].get("settings").is_none(), "render settings are not scene content");
    assert!(schema["$defs"]["Settings"].is_object());
}

#[test]
fn field_and_program_routes_and_topic_suggestions() {
    assert!(help_args(&["field", "node.target"]).contains("aims -Z"));
    assert!(help_args(&["field", "settings.seed"]).contains("u32"));
    for path in ["program.bindings", "mesh.instances.attributes"] {
        assert!(help_args(&["field", path]).contains("JSON shape: {name: string|Buffer object}"));
    }
    assert!(help_args(&["nodes", "--all"]).lines().any(|line|
        line.starts_with("node.instances.attributes (mesh):") && line.contains("JSON shape: {name: string|Buffer object}")));
    assert!(!help_args(&[]).contains("JSON shape:"));
    let abi = help_args(&["programs", "--all"]);
    assert!(abi.lines().any(|line| line.starts_with("program.bindings:") && line.contains("JSON shape: {name: string|Buffer object}")));
    for name in ["struct ShadingContext", "struct ViewContext", "struct Closure"] {
        assert!(abi.contains(name), "missing {name}");
    }
    let typo = Command::new(env!("CARGO_BIN_EXE_vork")).args(["help", "camras"]).output().unwrap();
    assert_eq!(typo.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&typo.stderr).contains("cameras"));
}

#[test]
fn styled_run_fields_are_listed_with_text_nodes() {
    let nodes = help_args(&["nodes", "--all"]);
    let runs = help_args(&["field", "run"]);
    let fields: Vec<_> = include_str!("../../vork-scene/src/v2/reference.txt")
        .lines().filter(|line| line.starts_with("run.")).collect();
    assert!(!fields.is_empty(), "styled-run field reference is missing");
    for field in fields {
        assert_eq!(nodes.lines().filter(|line| *line == field).count(), 1,
            "nodes --all lost styled-run field {field}");
        assert_eq!(runs.lines().filter(|line| *line == field).count(), 1,
            "run object lookup lost {field}");
    }
}

#[test]
fn every_field_remains_addressable_without_duplicate_topic_listings() {
    let topics: String = ["contract", "nodes", "cameras", "lights", "materials", "geometry", "buffers",
        "images", "animation", "render", "outputs", "profiles", "programs", "compositor", "std"]
        .iter().map(|topic| help_args(&[topic, "--all"])).collect();
    let root = help_args(&[]);
    // Parse API object blocks back to full wire paths, independent of field selection.
    let mut fields = std::collections::BTreeMap::<String, Vec<&str>>::new();
    let mut object = None;
    for line in root.lines() {
        if let Some(title) = line.strip_suffix(" {") {
            object = Some(match title {
                "Scene" => ("scene", ""), "Node" => ("node", ""),
                "Mesh (type: mesh)" => ("node", "mesh"),
                "Asset (type: asset)" => ("node", "asset"),
                "Camera (type: camera)" => ("node", "camera"),
                "Light (type: light)" => ("node", "light"),
                "Text (type: text, canvas)" => ("node", "text"),
                "Path (type: path, canvas)" => ("node", "path"),
                "Material" => ("material", ""), "Texture binding" => ("material.texture", ""),
                "Attribute binding" => ("material.attribute", ""), "AOV binding" => ("material.aov", ""),
                "Geometry" => ("geometry", ""), "Buffer" => ("buffer", ""), "Image" => ("image", ""),
                "Layer" => ("animation.layer", ""), "Track" => ("animation.track", ""), "Render" => ("render", ""),
                _ => panic!("unknown API object {title}"),
            });
        } else if line == "}" { object = None; }
        else if let (Some((prefix, variant)), Some(field)) = (object, line.strip_prefix("  ")) {
            let (name, declaration) = field.split_once(": ").unwrap();
            let path = if variant.is_empty() { format!("{prefix}.{name}") }
                else { format!("{prefix}.{name} ({variant})") };
            fields.entry(path).or_default().push(declaration);
        }
    }
    for line in include_str!("../../vork-scene/src/v2/reference.txt").lines().filter(|line| !line.is_empty()) {
        let (path, fact) = line.split_once(": ").unwrap();
        assert_eq!(help_args(&["field", path]).lines().filter(|entry| *entry == line).count(), 1, "field lookup lost {line}");
        let in_root = vork_scene::v2::api_field(line);
        assert_eq!(topics.lines().filter(|entry| *entry == line).count(), usize::from(!in_root && !vork_scene::v2::lookup_only_field(line)), "{line}");
        if !in_root { continue; }
        if path.starts_with("node [type=") {
            let variant = path.strip_prefix("node [type=\"").unwrap().strip_suffix("\"]").unwrap();
            assert!(root.contains(&format!("type: {variant}")) || variant == "node" && root.contains("type: node"));
        } else if path == "material.inputs (std/openpbr)" {
            assert!(root.contains(&fact.replace("OpenPBR Surface; shaded inputs:", "std/openpbr inputs:")));
        } else if !path.contains('.') {
            let spelling = match path {
                "coordinates" => "right-handed Y-up; camera local -Z",
                "names" => "Names exclude '/'", "paths" => "paths are scene-relative",
                "structure" => "Topology and program code are fixed",
                "normal maps" => "Normal maps: tangent-space +Y", "colour" => "linear Rec.709",
                _ => panic!("unverified root fact {path}"),
            };
            assert!(root.contains(spelling), "{line}");
        } else {
            let declarations = fields.get(path).unwrap_or_else(|| panic!("root lost {line}"));
            assert_eq!(declarations.len(), 1, "duplicate root field {path}");
            // The root page omits advanced tuning guidance and schema-shape detail.
            let fact = fact.split_once("; JSON shape:").map_or(fact, |(f, _)| f);
            let fact = fact.split_once("; when to change:").map_or(fact, |(f, _)| f);
            let (fact, default) = fact.rsplit_once("; default ").map_or((fact, None), |(f,d)| (f, Some(d)));
            let ty = fact.split("; ").next().unwrap();
            let expected = default.map_or_else(|| ty.to_owned(), |d| format!("{ty} = {d}"));
            assert_eq!(declarations[0].split("; ").next().unwrap(), expected, "{line}");
        }
    }
    assert!(!include_str!("../../vork-scene/src/v2/reference.txt").lines()
        .any(|line| line.starts_with("hair.")));
    let std = help_args(&["std", "--all"]);
    assert!(std.contains("std/hair.parametrization: enum color|melanin|absorption"));
    let retired = Command::new(env!("CARGO_BIN_EXE_vork"))
        .args(["help", "field", "hair.roughness"]).output().unwrap();
    assert_eq!(retired.status.code(), Some(2));
    assert!(root.contains("OpenPBR Surface 1.1.1") && root.contains("TOPIC [--all]"));
    assert!(root.contains("std/openpbr inputs: base_weight base_color"));
    assert!(!help("materials").contains("OpenPBR Surface 1.1.1"));
}

#[test]
fn image_render_help_matches_capture_fields_and_settings() {
    let help = help_args(&["field", "image.render"]);
    for field in ["camera", "root", "resolution", "background"] {
        assert!(help.contains(&format!("image.render.{field}:")), "{field}");
    }
    assert!(help_args(&["field", "image.composition"]).contains("image.composition.compositor:"));
    assert!(help.contains("sampling budget follows quality/cost level"));
    assert!(help.contains("closed shutter"));
    assert!(help_args(&["images", "--all"]).contains("aperture sampling is renderer-selected"));
    assert!(!help.contains("image.render.seed") && !help.contains("image.render.motion_blur"));
}

#[test]
fn removed_abi_flag_is_rejected() {
    let output = Command::new(env!("CARGO_BIN_EXE_vork"))
        .args(["help", "programs", "--abi"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unexpected argument '--abi'"));
    assert!(!help_args(&["--help"]).contains("--abi"));
}

fn field_error(path: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_vork"))
        .args(["help", "field", path]).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let error = String::from_utf8(output.stderr).unwrap();
    assert_eq!(error.lines().count(), 1, "{error}");
    error
}

#[test]
fn wrong_field_prefix_points_to_the_field_and_setting_override() {
    for (path, expected) in [
        ("render.seed", "settings.seed (profile setting; set with --set seed=...)"),
        ("camera.energy", "node.energy (light)"),
        ("render.denoise", "settings.denoise (profile setting; set with --set denoise=...)"),
    ] {
        let error = field_error(path);
        assert!(error.contains(expected), "{error}");
        assert!(!error.contains("Nearest valid names"), "{error}");
    }
}

#[test]
fn library_and_program_field_lookups_point_to_library_help() {
    for library in ["std", "blender"] {
        for path in [library.to_owned(), format!("{library}/unknown_program")] {
            let error = field_error(&path);
            assert!(error.contains("Nearest valid names"), "{error}");
            assert!(error.contains(&format!("vork help {library} --all`")), "{error}");
        }
    }
}

#[test]
fn field_typos_rank_full_paths_leaves_and_prefix_areas() {
    for (path, expected) in [
        ("composition.veiw", "composition.view"),
        ("node.enrgy", "node.energy (light)"),
        ("settings.sed", "settings.seed"),
        ("render.motion_blur.shuter", "render.motion_blur.shutter"),
    ] {
        let error = field_error(path);
        assert!(error.contains(&format!("did you mean {expected:?}?")), "{error}");
    }
}

#[test]
fn procedural_help_exposes_callable_signatures_and_enum_spelling() {
    assert!(help_args(&["blender", "--all"]).contains("blender.procedurals: noise"));
    let all = help_args(&["blender", "--all"]);
    for signature in ["noise(p, scale=5", "noise_color: same inputs", "voronoi(p, scale=5",
        "white_noise(p, dimensions=3", "gabor(p, scale=5",
        "Smooth_F1", "Chebychev",
        "NoiseType.FBM"] {
        assert!(all.contains(signature), "missing {signature}");
    }
    assert!(!all.contains("noise(ctx.rest_position)"));
    assert!(help_args(&["programs", "--all"]).contains("Closure|Blend|Hair surface(ShadingContext, Inputs)"));
}


#[test]
fn camera_enum_tokens_come_from_program_reflection() {
    for (field,tokens) in [("projection","persp|ortho"),("fit","auto|horizontal|vertical")] {
        let line=format!("std/camera.{field}: enum {tokens}");
        for text in [help("cameras"),help_args(&["std","--all"]),help_args(&["field","std/camera"]),help_args(&["field",&format!("std/camera.{field}")])] {
            assert!(text.contains(&line),"missing {line} in {text}");
        }
    }
}

#[test]
fn help_closed_pipe_exits_successfully() {
    use std::process::{Command, Stdio};
    let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
        .args(["help", "std", "--all"])
        .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.stderr.is_empty(), "{}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn optics_and_compositor_programs_are_directly_findable() {
    let camera = help("std/camera");
    assert!(camera.contains("std/camera.lens: number; mm; default 50"));
    assert!(camera.contains("std/camera.shift: vec2; default [0,0]"));
    for name in ["lens", "sensor_width", "ortho_scale", "shift", "clip_start", "use_dof", "aperture_ratio"] {
        let line = format!("std/camera.{name}:");
        assert!(camera.contains(&line), "{line}");
        assert!(help_args(&["field", &format!("std/camera.{name}")]).contains(&line));
    }
    assert!(help_args(&["field", "camera.material"]).contains("node.material (camera)"));
    let compositor=help_args(&["compositor","--all"]);
    assert!(compositor.contains("compositor_sample(image,uv)"));
    let root=help_args(&[]);
    assert!(!root.contains("post: colour adaptation"));
    assert!(!root.contains("blender: Blender Slang functions"));
}

#[cfg(target_os = "linux")]
#[test]
fn stdout_filesystem_failure_keeps_io_category() {
    let output = Command::new(env!("CARGO_BIN_EXE_vork")).arg("help")
        .stdout(std::fs::File::options().write(true).open("/dev/full").unwrap()).output().unwrap();
    assert_eq!(output.status.code(), Some(5), "{output:?}");
}

#[test]
fn bundled_program_contracts_are_exactly_addressable() {
    for program in ["std/hair", "std/sky", "std/text"] {
        let exact = help_args(&["field", program]);
        assert_eq!(help(program), exact);
        assert!(help_args(&["std", "--all"]).contains(&exact));
    }
    assert!(!help("cameras").contains("std/hair."));
    for leaf in ["emission_luminance", "subsurface_radius_scale"] {
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .args(["help", "field", &format!("std/openpbr.{leaf}")]).output().unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains(&format!("std/openpbr.{leaf}:")));
    }
}

#[test]
fn every_bundled_program_input_is_addressable() {
    for library in vork_scene::libraries::LIBRARIES {
        let listing = help_args(&[library.name, "--all"]);
        for module in library.modules.iter().filter(|module| module.material) {
            let fields = help_args(&["field", module.name]);
            assert!(!fields.is_empty(), "{}", module.name);
            for line in fields.lines() {
                let path = line.split_once(':').unwrap().0;
                assert_eq!(help_args(&["field", path]).trim(), line);
                assert!(listing.contains(line), "{path}");
            }
        }
    }
    for field in ["std/agx.exposure", "std/openpbr.emission_luminance", "std/openpbr.transmission_depth", "material.inputs.transmission_depth"] {
        assert!(help_args(&["field", field]).contains("default"));
    }
    assert!(help_args(&["field", "std/agx.exposure"]).contains("stops"));
    assert!(help_args(&["field", "std/agx.lut"]).contains("default bundled image"));
    let error = field_error("std/agx.exposur");
    assert!(error.contains("std/agx.exposure") && error.contains("vork help std --all"));
}

#[test]
fn std_filename_diagnostic_names_json_and_stdin() {
    let output = Command::new(env!("CARGO_BIN_EXE_vork")).args(["std", "expand", "scene.json"]).output().unwrap();
    assert_eq!(output.status.code(), Some(3));
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(message.contains("JSON argument") && message.contains("vork std expand - < scene.json"), "{message}");
    let output = Command::new(env!("CARGO_BIN_EXE_vork")).args(["std", "rounded_box", "{}"]).output().unwrap();
    assert!(output.status.success());
}

#[test]
fn quoted_help_arguments_match_separate_arguments() {
    for (quoted, separate) in [
        ("programs --all", vec!["programs", "--all"]),
        ("field node.target", vec!["field", "node.target"]),
        ("  profiles  ", vec!["profiles"]),
        ("--all", vec!["--all"]),
    ] {
        assert_eq!(help_args(&[quoted]), help_args(&separate));
    }
    for args in [vec!["nope --all"], vec!["programs extra"], vec!["field node.target extra"], vec!["field node.target", "other"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_vork")).arg("help").args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
    }
}

#[test]
fn deferred_help_contracts_are_explicit_on_public_routes() {
    for args in [vec!["field", "image.file"], vec!["images", "--all"], vec![]] {
        let text = help_args(&args);
        assert!(text.contains("half-float"), "missing image precision: {args:?}");
        assert!(!text.contains("exact data"));
    }
    for field in ["node.size", "node.size_y"] {
        assert!(help_args(&["field", field]).contains("only rectangle or ellipse"));
    }
    let cameras = help("cameras");
    for contract in ["full width", "full height", "width/height", "Horizontal", "Vertical", "Auto", "scene bounds"] {
        assert!(cameras.contains(contract), "camera help lost {contract}");
    }
    assert!(help("outputs").contains("EXR stills also require %06d"));
    assert!(help("profiles").contains("unchanged evaluated scene inputs, frame identity, render settings and backend"));
    assert!(help("sdk").contains("Regenerate SDK-written scenes with the matching SDK after upgrading."));
}

#[test]
fn fluid_help_names_geometry_and_buffer_output_namespaces() {
    for args in [&["fluid"][..], &["fluid", "--all"][..]] {
        let text = help_args(args);
        for key in [
            "geometry[\"water\"]", "buffers[\"surface\"]",
            "geometry[\"floor_film\"]", "buffers[\"film_surface\"]",
            "geometry[\"whitewater\"]", "buffers[\"white_draw_position\"]",
            "buffers[\"white_width\"]", "buffers[\"white_count\"]",
            "buffers[\"white_draw_id\"]", "whitewater_budget > 0",
        ] {
            assert!(text.contains(key), "{args:?} lost output contract {key}");
        }
    }
}

#[test]
fn active_help_describes_built_output() {
    let field = help_args(&["field", "node.active"]);
    assert!(field.contains("false removes picture and sound"), "{field}");
}

#[test]
fn path_stroke_shape_includes_supported_image_paint() {
    for field in ["node.stroke (path)", "node.fill (path)"] {
        let paint = help_args(&["field", field]);
        assert!(paint.contains("ImagePaint object"), "{paint}");
    }
}

#[test]
fn canvas_limits_are_explicit_where_the_features_are_advertised() {
    for topic in ["render", "sdk", "limits"] {
        for args in [vec![topic], vec![topic, "--all"]] {
            assert!(!help_args(&args).contains("Sessions reject canvas documents"), "{args:?}");
        }
    }
    for topic in ["ask", "limits"] {
        for args in [vec![topic], vec![topic, "--all"]] {
            assert!(help_args(&args).contains("Canvas documents support audio answers only"), "{args:?}");
        }
    }
    for topic in ["nodes", "images", "limits"] {
        for args in [vec![topic], vec![topic, "--all"]] {
            assert!(help_args(&args).contains("A canvas inside a canvas cannot set viewBox or preserveAspectRatio"), "{args:?}");
        }
    }
    for field in ["composition.viewBox", "composition.preserveAspectRatio"] {
        assert!(help_args(&["field", field]).contains("unsupported on a canvas inside a canvas"), "{field}");
    }
}
