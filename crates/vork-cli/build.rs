use std::process::Command;

fn program_input_help() {
    use std::{fmt::Write, path::Path};
    use vork_slang::{Abi, Compiler};
    let abi = Abi {
        aov_count: 0,
        context: Abi::spec_context(),
        closure: vork_slang::openpbr_fields(),
    };
    let compiler = Compiler::discover();
    println!("cargo:rerun-if-env-changed=VORK_SLANGC");
    let mut text = String::new();
    for module in vork_scene::libraries::LIBRARIES.iter().flat_map(|library| library.modules).filter(|module| module.material) {
        let program = module.name;
        let filename = format!("../../libraries/{program}.slang");
        let path = Path::new(&filename);
        println!("cargo:rerun-if-changed={}", path.display());
        let source = module.source;
        let inputs = compiler.reflect_inputs(&abi, source, path).expect("reflect bundled program Inputs");
        let declarations = source.split_once("struct Inputs {").unwrap().1.split_once("};").unwrap().0;
        for field in inputs.fields {
            // Read initializers from the bundled source rather than duplicating defaults.
            let declaration = declarations.lines().find_map(|line| {
                let (declaration, value) = line.split_once('=')?;
                (declaration.split_whitespace().last()? == field.name)
                    .then(|| (value.split_once(';').expect("bundled input initializer").0.trim(), line))
            });
            let initializer = declaration.map_or("zero (unbound resource)", |(value, _)| value);
            if let (vork_slang::Type::Float, Some(range)) = (&field.ty, &field.range) {
                let value = initializer.parse::<f64>().expect("numeric initializer");
                let valid = (value > range.min || (range.min_closed && value == range.min))
                    && (value < range.max || (range.max_closed && value == range.max));
                if !valid {
                    // Contextual defaults are internal sentinels, not authorable values.
                    let meaning = declaration.unwrap().1.split_once("// Unbound:")
                        .expect("out-of-range initializer needs an Unbound help sentence").1.trim();
                    assert!(!meaning.is_empty());
                    writeln!(text, "{program}.{}: {meaning}", field.name).unwrap();
                    continue;
                }
            }
            let default = if initializer == "zero (unbound resource)" {
                if vork_scene::libraries::LIBRARIES.iter().flat_map(|library| library.images)
                    .any(|data| data.program == program && data.input == field.name) {
                    "bundled image".to_owned()
                } else { "unbound".to_owned() }
            } else if field.enumeration.is_some() {
                initializer.split("::").last().unwrap().rsplit('.').next().unwrap().to_ascii_lowercase()
            } else if let vork_slang::Type::Vector(n) = field.ty {
                let value = initializer.split_once('(').expect("vector initializer").1.trim_end_matches(')');
                let mut values = value.split(',').map(|v| v.trim().parse::<f64>().expect("numeric vector initializer").to_string()).collect::<Vec<_>>();
                if values.len() == 1 { values.resize(n as usize, values[0].clone()); }
                assert_eq!(values.len(), n as usize);
                format!("[{}]", values.join(","))
            } else if matches!(field.ty, vork_slang::Type::Float) {
                initializer.parse::<f64>().expect("numeric initializer").to_string()
            } else { initializer.to_owned() };
            let units = match field.name.as_str() {
                "lens" | "sensor_width" | "sensor_height" => "; mm",
                "ortho_scale" | "clip_start" | "clip_end" | "focus_distance" => "; metres",
                "aperture_rotation" | "sun_elevation" | "sun_rotation" | "offset" => "; radians",
                "altitude" => "; metres",
                "exposure" => "; stops",
                "temperature" => "; kelvin",
                _ => "",
            };
            let range = field.range.as_ref().map_or_else(|| "no declared range".to_owned(), |r|
                format!("range {}{},{}{}", if r.min_closed { '[' } else { '(' }, r.min, r.max, if r.max_closed { ']' } else { ')' }));
            if let Some(enumeration) = field.enumeration {
                let tokens = enumeration
                    .cases
                    .iter()
                    .map(|case| case.to_ascii_lowercase())
                    .collect::<Vec<_>>()
                    .join("|");
                writeln!(
                    text,
                    "{program}.{}: enum {tokens}; {} (JSON tokens); default {default}; {range}.",
                    field.name, enumeration.ty
                )
                .unwrap();
            } else {
                let ty = match field.ty {
                    vork_slang::Type::Float => "number".to_owned(),
                    vork_slang::Type::Uint => "uint".to_owned(),
                    vork_slang::Type::Bool => "bool".to_owned(),
                    vork_slang::Type::Vector(n) => format!("vec{n}"),
                    ty => format!("{ty:?}"),
                };
                writeln!(text, "{program}.{}: {ty}{units}; default {default}; {range}.", field.name).unwrap();
            }
        }
    }
    text.push_str(&vork_std::text_input_help());
    std::fs::write(
        Path::new(&std::env::var("OUT_DIR").unwrap()).join("program-inputs.txt"),
        text,
    )
    .unwrap();
}

fn main() {
    program_input_help();
    // Resolve the worktree HEAD on every build, including packed refs and detached HEADs.
    println!("cargo:rerun-if-changed=build.rs");
    for arg in ["HEAD", "refs", "packed-refs"] {
        if let Ok(output) = Command::new("git")
            .args(["rev-parse", "--git-path", arg])
            .output()
        {
            if output.status.success() {
                println!(
                    "cargo:rerun-if-changed={}",
                    String::from_utf8_lossy(&output.stdout).trim()
                );
            }
        }
    }
    println!("cargo:rerun-if-env-changed=VORK_BUILD_COMMIT");
    let commit = std::env::var("VORK_BUILD_COMMIT")
        .ok()
        .filter(|commit| !commit.trim().is_empty())
        .map(|commit| commit.trim().to_owned())
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short=12", "HEAD"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        })
        .unwrap_or_else(|| "source-archive".into());
    println!(
        "cargo:rustc-env=VORK_VERSION={} (commit {}; API v2)",
        std::env::var("CARGO_PKG_VERSION").expect("Cargo package version"),
        commit
    );
}
