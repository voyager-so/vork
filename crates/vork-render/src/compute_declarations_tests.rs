use super::*;

// Read the public interface from actual generated Slang, not a second expected
// symbol list. Bodies, resource declarations and private implementation names
// are deliberately excluded; return types, overloads and constants are retained.
fn interface(source: &str) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    let declarations = source
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    let mut depth = 0;
    let mut start = 0;
    for (at, c) in declarations.char_indices() {
        if depth == 0 && (c == ';' || c == '{') {
            let text = declarations[start..at].trim();
            let text = text.strip_prefix("extern ").unwrap_or(text);
            if let Some(text) = text.strip_prefix("static const ") {
                let words: Vec<_> = text.split_whitespace().collect();
                if words.len() >= 2 && !words[1].starts_with('_') {
                    result.insert(format!("{} {}", words[0], words[1]));
                }
            } else if let Some((head, parameters)) = text.split_once('(') {
                if !head.contains("[[") {
                    let words: Vec<_> = head.split_whitespace().collect();
                    if words.len() == 2
                        && !words[1].starts_with('_')
                        && !words[1].starts_with("vork_")
                    {
                        let types = parameters
                            .trim_end_matches(')')
                            .split(',')
                            .filter(|p| !p.trim().is_empty())
                            .map(|p| p.split_whitespace().next().unwrap())
                            .collect::<Vec<_>>()
                            .join(",");
                        result.insert(format!("{} {}({types})", words[0], words[1]));
                    }
                }
            }
            start = at + 1;
        }
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    start = at + 1;
                }
            }
            _ => {}
        }
    }
    for line in source.lines().filter(|line| line.starts_with("#define ")) {
        result.insert(line.split_whitespace().collect::<Vec<_>>().join(" "));
    }
    result
}

#[test]
fn catalog_matches_generated_compute_interface() -> Result<()> {
    let r = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
    let base = Path::new(".");
    // Every buffer element kind, dimensions, atomics, previous state, numbered
    // and semantic geometry attributes, optional streams, and Geometry overloads.
    let mut value = serde_json::json!({
        "version":2,"nodes":[{"type":"camera"},{"type":"mesh","name":"mesh","geometry":"proxy"}],"render":{},
        "buffers":{"proxy_positions":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
        "programs":{"all":{"source":"struct Inputs { Geometry mesh; }; Closure surface(ShadingContext c,Inputs i){return unlit(float3(.2,.4,.6));} void compute(ComputeContext c,Inputs i){}", "schedule":"install","dispatch":1,"bindings":{}}},
        "materials":{"all":{"program":"all","inputs":{"mesh":{"geometry":"mesh"}}}},
        "geometry":{"proxy":{"attributes":{"POSITION":"proxy_positions"}},"shape":{"attributes":{"POSITION":"v3","NORMAL":"v3","TANGENT":"v4","TEXCOORD_0":"v2","COLOR_1":"v4","JOINTS_2":"u4","WEIGHTS_3":"v4","TEXCOORD_bad":"f"},"widths":"f","ids":"u","count":"u"}}
    });
    use v2::BufferType::*;
    for (name, kind) in [
        ("f", F32),
        ("i", I32),
        ("u", U32),
        ("v2", Vec2),
        ("v3", Vec3),
        ("v4", Vec4),
        ("u4", Uvec4),
        ("matrix", Mat4),
        ("fluid_status", U32),
    ] {
        // An exhaustive match makes adding a buffer kind require test coverage.
        let ty = match kind {
            F32 => "f32",
            I32 => "i32",
            U32 => "u32",
            Vec2 => "vec2",
            Vec3 => "vec3",
            Vec4 => "vec4",
            Uvec4 => "uvec4",
            Mat4 => "mat4",
        };
        let datum = match kind {
            F32 | I32 | U32 => serde_json::json!(1),
            Vec2 => serde_json::json!([0, 0]),
            Vec3 => serde_json::json!([0, 0, 0]),
            Vec4 | Uvec4 => serde_json::json!([0, 0, 0, 1]),
            Mat4 => serde_json::json!([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        };
        value["buffers"][name] =
            serde_json::json!({"type":ty,"data":[datum.clone(),datum.clone(),datum]});
        let grid = format!("grid_{name}");
        value["buffers"][&grid] = serde_json::json!({"type":ty,"dims":[1,1,1]});
        value["programs"]["all"]["bindings"][&grid] = serde_json::json!(grid);
        value["programs"]["all"]["bindings"][name] = serde_json::json!(name);
    }
    value["buffers"]["live"] = serde_json::json!({"type":"u32","data":[3]});
    value["programs"]["all"]["bindings"]["live"] = serde_json::json!("live");
    value["geometry"]["shape"]["count"] = serde_json::json!("live");
    for points in [false, true] {
        let mut case = value.clone();
        if points {
            case["geometry"]["shape"]["topology"] = serde_json::json!("points");
            let attributes = case["geometry"]["shape"]["attributes"]
                .as_object_mut()
                .unwrap();
            for name in ["NORMAL", "TANGENT", "TEXCOORD_0"] {
                attributes.remove(name);
            }
        } else {
            let geometry = case["geometry"]["shape"].as_object_mut().unwrap();
            for name in ["widths", "ids", "count"] {
                geometry.remove(name);
            }
        }
        check_case(&r, base, case)?;
    }
    Ok(())
}

fn check_case(r: &Renderer, base: &Path, value: serde_json::Value) -> Result<()> {
    let mut scene = crate::test_scene(value, base)?;
    let decoded = buffers::SceneBuffers::decode(&scene, base)?;
    let named = ProgramBuffers::install(r, &scene, &decoded)?;
    let declarations = stage_declarations(&scene, &scene.programs["all"])?;
    let expected = prepare_program(
        r,
        &scene,
        base,
        &named,
        &named,
        Some(&scene.geometry["shape"]),
        false,
        "all",
        &scene.programs["all"],
    )?;
    let actual = interface(&declarations);
    eprintln!(
        "catalog/generated interface: {} signatures, constants and defines",
        actual.len()
    );
    assert_eq!(
        actual,
        interface(expected.header.strip_prefix(CONTEXT_BINDINGS).unwrap())
    );

    // Generate calls from the catalog itself: an added declaration must also
    // type-check in the unused compute body during surface compilation, and
    // have a real implementation when the same module compiles as compute.
    let mut body = String::new();
    for symbol in &actual {
        if symbol.starts_with("#define ") {
            let name = symbol.split_whitespace().nth(1).unwrap();
            writeln!(
                body,
                "\n#ifndef {name}\n#error missing generated define\n#endif"
            )?;
        } else if let Some((head, parameters)) = symbol.split_once('(') {
            let name = head.split_whitespace().nth(1).unwrap();
            let arguments = parameters
                .trim_end_matches(')')
                .split(',')
                .filter(|ty| !ty.is_empty())
                .map(|ty| {
                    if ty == "Geometry" {
                        "i.mesh".to_owned()
                    } else {
                        format!("({ty})0")
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            writeln!(body, "{name}({arguments});")?;
        } else {
            let name = symbol.split_whitespace().nth(1).unwrap();
            writeln!(body, "{{ let value = {name}; }}")?;
        }
    }
    let source = scene.programs["all"].source.as_ref().unwrap().replace(
        "void compute(ComputeContext c,Inputs i){}",
        &format!("void compute(ComputeContext c,Inputs i){{{body}}}"),
    );
    scene.programs.get_mut("all").unwrap().source = Some(source);
    let prepared = prepare_program(
        r,
        &scene,
        base,
        &named,
        &named,
        Some(&scene.geometry["shape"]),
        false,
        "all",
        &scene.programs["all"],
    )?;
    let (source, origin) = programs::material_source(&scene, base, "all")?;
    vork_slang::Compiler::discover().compile_surface(
        &programs::abi(),
        &source,
        &origin,
        "allaccessors",
    )?;
    let abi = programs::abi();
    let mut modules: Vec<_> = crate::native::SOURCES
        .iter()
        .map(|(name, source)| (name.to_string(), source.to_string()))
        .collect();
    modules.extend(abi.native_modules());
    let compute = prepared.artifact.slang;
    let surface = format!(
        "import vork;\n{source}\n[[vk::binding(0,0)]] RWStructuredBuffer<float4> result; [shader(\"compute\")][numthreads(1,1,1)] void main(){{result[0]=float4(surface((ShadingContext)0,(Inputs)0).base_color,1.);}}"
    );
    modules.push(("catalog_compute".into(), compute));
    modules.push(("catalog_surface".into(), surface));
    let modules: Vec<_> = modules
        .iter()
        .map(|(name, source)| (name.as_str(), source.as_str()))
        .collect();
    for (_, target, options) in [
        crate::forward_variant::SPIRV_TARGET,
        crate::forward_variant::METAL_TARGET,
    ] {
        let mut flags = vec!["-target", target, "-O2", "-matrix-layout-row-major"];
        flags.extend_from_slice(options);
        for (module, entry) in [
            ("catalog_compute", "vork_entry"),
            ("catalog_surface", "main"),
        ] {
            let code = vork_slang::Compiler::discover().compile_modules(
                &modules,
                module,
                &[(entry, "compute")],
                &flags,
            )?;
            assert!(!code.is_empty(), "{module} {target}");
            eprintln!("compiled {module} -> {target}: {} bytes", code.len());
        }
    }
    Ok(())
}
