fn main() {
    println!("cargo:rerun-if-env-changed=VORK_SLANGC");
    println!("cargo:rerun-if-changed=../../dist/macos-minimum.txt");
    println!("cargo:rerun-if-changed=src/filesystem.cpp");
    let compiler = std::env::var_os("VORK_SLANGC")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("PATH").and_then(|paths| {
                std::env::split_paths(&paths)
                    .map(|p| {
                        p.join(if cfg!(windows) {
                            "slangc.exe"
                        } else {
                            "slangc"
                        })
                    })
                    .find(|p| p.is_file())
            })
        })
        .expect("install the pinned Slang toolchain and set VORK_SLANGC (docs/BUILD.md)");
    let compiler = compiler
        .canonicalize()
        .expect("canonical Slang compiler path");
    // MSVC's include search does not accept Rust's verbatim canonical prefix.
    #[cfg(windows)]
    let compiler = {
        let text = compiler.to_string_lossy();
        std::path::PathBuf::from(if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
            format!(r"\\{rest}")
        } else {
            text.strip_prefix(r"\\?\").unwrap_or(&text).to_owned()
        })
    };
    let include = compiler.parent().unwrap().parent().unwrap().join("include");
    assert!(
        include.join("slang.h").is_file(),
        "Slang toolchain must include headers: {}",
        include.display()
    );
    let tool = cc::Build::new().cpp(true).get_compiler();
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join(
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
            "vork-slang-filesystem.exe"
        } else {
            "vork-slang-filesystem"
        },
    );
    let mut command = tool.to_command();
    if tool.is_like_msvc() {
        command
            .arg("/std:c++17")
            .arg("/EHsc")
            .arg("/MT")
            .arg(format!("/I{}", include.display()))
            .arg("src/filesystem.cpp")
            .arg(format!("/Fe{}", output.display()))
            .arg(format!("/Fo{}", output.with_extension("obj").display()));
    } else {
        command
            .arg("-std=c++17")
            .arg("-O2")
            .arg("-I")
            .arg(include)
            .arg("src/filesystem.cpp")
            .arg("-o")
            .arg(&output);
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos" {
            command.arg(format!("-mmacosx-version-min={}", include_str!("../../dist/macos-minimum.txt").trim()));
        }
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "linux" {
            command.arg("-ldl");
        }
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
            command.arg("-municode");
        }
    }
    assert!(
        command
            .status()
            .expect("compile Slang filesystem adapter")
            .success(),
        "Slang filesystem adapter failed to compile"
    );
    // Cargo does not install non-Rust executables. Publish the adapter beside
    // the profile's binaries; release packaging includes this companion.
    let companion = output
        .parent()
        .unwrap()
        .ancestors()
        .nth(3)
        .unwrap()
        .join(output.file_name().unwrap());
    let temporary = companion.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::copy(&output, &temporary).expect("copy Slang adapter");
    if cfg!(windows) && companion.exists() {
        std::fs::remove_file(&companion).expect("replace Slang adapter");
    }
    std::fs::rename(temporary, &companion).expect("publish Slang adapter");
    println!(
        "cargo:rustc-env=VORK_SLANG_FILESYSTEM_ADAPTER={}",
        output.display()
    );

    // Explicit opt-in: directory layout must not change test admission.
    println!("cargo:rustc-check-cfg=cfg(vork_private_tests)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VORK_PRIVATE_TESTS");
    if std::env::var("VORK_PRIVATE_TESTS").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=vork_private_tests");
    }
}
