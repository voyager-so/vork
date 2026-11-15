//! Dependency contract, exercised against both native targets.
//! Every unchanged closure must validate warm; edits and resolver shadowing must
//! invalidate, and a dependency Slang can open must survive capture and replay.
use super::*;

#[test]
fn dependency_layout_matrix() -> Result<()> {
    let mut failures = Vec::new();
    let (mut passed, mut skips) = (0, 0);
    for target in ["spirv", "metal"] {
        for layout in [
            "sibling",
            "edit",
            "shadow",
            "alternate",
            "dotted",
            "file-link",
            "directory-link",
            "nested-link",
            "chain-link",
            "loop",
            "dangling",
            "temp-ancestor",
            "unreadable",
            "deep-unreadable",
            "home-library",
            "unicode",
            "case-spelling",
            "inline",
            "bundled",
        ] {
            let mut skipped = false;
            let result = (|| -> Result<()> {
                let compiler = Compiler::discover();
                let work = Workspace::new()?;
                let shot = work.0.join(if layout == "unicode" {
                    "shot café 色"
                } else {
                    "shot"
                });
                let search = work.0.join("search");
                fs::create_dir_all(&shot)?;
                fs::create_dir_all(&search)?;
                let value = "public float value(){return 0.25;}";
                let mut import = "palette";
                let mut helper = shot.join("palette.slang");
                if layout == "alternate" {
                    import = "my_palette";
                    helper = shot.join("my-palette.slang");
                }
                if layout == "dotted" || layout == "nested-link" {
                    import = "sub.palette";
                    helper = shot.join("sub/palette.slang");
                }
                fs::create_dir_all(helper.parent().unwrap())?;
                fs::write(&helper, value)?;
                if layout.ends_with("link")
                    || matches!(layout, "loop" | "dangling" | "temp-ancestor")
                {
                    #[cfg(unix)]
                    let link =
                        |a: &Path, b: &Path, _directory: bool| std::os::unix::fs::symlink(a, b);
                    #[cfg(windows)]
                    let link = |a: &Path, b: &Path, directory: bool| {
                        if directory {
                            std::os::windows::fs::symlink_dir(a, b)
                        } else {
                            std::os::windows::fs::symlink_file(a, b)
                        }
                    };
                    let probe = shot.join("privilege-probe");
                    if let Err(error) = link(&helper, &probe, false) {
                        eprintln!("SKIP {target}/{layout}: symlink unavailable: {error}");
                        skipped = true;
                        return Ok(());
                    }
                    fs::remove_file(probe)?;
                    match layout {
                        "file-link" | "nested-link" | "chain-link" => {
                            let implementation = shot.join("impl/palette_impl.slang");
                            fs::create_dir_all(implementation.parent().unwrap())?;
                            fs::rename(&helper, &implementation)?;
                            if layout == "chain-link" {
                                let intermediate = shot.join("impl/alias.slang");
                                link(&implementation, &intermediate, false)?;
                                link(&intermediate, &helper, false)?;
                            } else {
                                link(&implementation, &helper, false)?;
                            }
                        }
                        "directory-link" => {
                            fs::create_dir(shot.join("impl"))?;
                            fs::rename(&helper, shot.join("impl/palette.slang"))?;
                            link(&shot.join("impl"), &shot.join("sub"), true)?;
                            import = "sub.palette";
                        }
                        "loop" => link(
                            Path::new("palette.slang"),
                            &search.join("palette.slang"),
                            false,
                        )?,
                        "dangling" => {
                            link(Path::new("absent"), &search.join("palette.slang"), false)?
                        }
                        "temp-ancestor" => {
                            link(&std::env::temp_dir(), &shot.join("ancestor"), true)?
                        }
                        _ => unreachable!(),
                    }
                }
                let blocked = shot.join(match layout {
                    "unreadable" => "blocked",
                    "deep-unreadable" => "one/two/blocked",
                    _ => "home/Library",
                });
                let _permission_guard = if matches!(
                    layout,
                    "unreadable" | "deep-unreadable" | "home-library"
                ) {
                    fs::create_dir_all(&blocked)?;
                    let guard = UnreadableDirectory::new(&blocked)?;
                    if fs::read_dir(&blocked).is_ok() {
                        eprintln!("SKIP {target}/{layout}: this account bypasses directory read restrictions");
                        skipped = true;
                        return Ok(());
                    }
                    Some(guard)
                } else {
                    None
                };
                let prefix = match layout {
                    "inline" => value.to_owned(),
                    "bundled" => format!("import std.color; {value}"),
                    _ => format!("import {import};"),
                };
                let main = format!("{prefix} RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){{result[id.x]=value();}}");
                let search = search.to_string_lossy();
                let shot_text = shot.to_string_lossy();
                let args = ["-target", target, "-I", &search, "-I", &shot_text];
                let mut modules = vec![("main", main.as_str())];
                if layout == "bundled" {
                    modules.extend(vork_scene::libraries::modules().map(|m| (m.name, m.source)));
                }
                let run =
                    || compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args);
                let outcome = (|| -> Result<()> {
                    let first = run()?;
                    let code = first.compile()?;
                    ensure!(
                        code == compiler.compile_modules(
                            &modules,
                            "main",
                            &[("main", "compute")],
                            &args
                        )?,
                        "captured code differs from direct compiler resolution"
                    );
                    ensure!(!code.is_empty(), "empty result");
                    ensure!(first.receipt().valid(), "unchanged first receipt is cold");
                    let second = run()?;
                    ensure!(second.receipt().valid(), "unchanged second receipt is cold");
                    ensure!(
                        first.identity() == second.identity(),
                        "unchanged identity differs"
                    );
                    ensure!(code == second.compile()?, "unchanged code differs");
                    let changed = "public float value(){return 0.75;}";
                    match layout {
                        "edit" => fs::write(&helper, changed)?,
                        "shadow" => {
                            fs::write(Path::new(search.as_ref()).join("palette.slang"), changed)?
                        }
                        "alternate" => fs::write(shot.join("my_palette.slang"), changed)?,
                        "case-spelling" => {
                            let other_spelling = shot.join("Palette.slang");
                            let aliases_original = other_spelling.is_file();
                            fs::write(other_spelling, changed)?;
                            if !aliases_original {
                                ensure!(
                                    first.receipt().valid(),
                                    "distinct case-sensitive file made inputs cold"
                                );
                                ensure!(
                                    code == compiler.compile_modules(
                                        &modules,
                                        "main",
                                        &[("main", "compute")],
                                        &args
                                    )?,
                                    "case-sensitive import changed"
                                );
                                return Ok(());
                            }
                        }
                        _ => return Ok(()),
                    }
                    ensure!(
                        !first.receipt().valid(),
                        "changed resolution incorrectly warm"
                    );
                    let edited = run()?;
                    ensure!(
                        edited.identity() != first.identity(),
                        "changed identity reused"
                    );
                    ensure!(edited.compile()? != code, "changed result reused");
                    ensure!(
                        edited.compile()?
                            == compiler.compile_modules(
                                &modules,
                                "main",
                                &[("main", "compute")],
                                &args
                            )?,
                        "edited result differs from direct compiler resolution"
                    );
                    ensure!(first.compile()? == code, "snapshot changed after edit");
                    ensure!(edited.receipt().valid(), "edited result cannot warm");
                    Ok(())
                })();
                outcome
            })();
            match result {
                Ok(()) => {
                    if skipped {
                        skips += 1;
                    } else {
                        passed += 1;
                        eprintln!("PASS {target}/{layout}")
                    }
                }
                Err(error) => {
                    eprintln!("FAIL {target}/{layout}: {error:#}");
                    failures.push(format!("{target}/{layout}: {error:#}"));
                }
            }
        }
    }
    eprintln!(
        "Dependency matrix: {passed} passed, {skips} explicitly skipped, {} failed",
        failures.len()
    );
    ensure!(
        failures.is_empty(),
        "matrix failures:\n{}",
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn captured_observations_do_not_enumerate_unrelated_trees() -> Result<()> {
    let compiler = Compiler::discover();
    let work = Workspace::new()?;
    fs::write(
        work.0.join("palette.slang"),
        "public float value(){return 0.25;}",
    )?;
    let include = work.0.to_string_lossy();
    let modules = [("main", "import palette; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}")];
    let args = ["-target", "spirv", "-I", &include];
    let first = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
    let before = first.receipt();
    let tree = work.0.join("unrelated");
    for i in 0..200 {
        let directory = tree.join(format!("{i}/child"));
        fs::create_dir_all(&directory)?;
        for j in 0..100 {
            fs::write(directory.join(format!("{j}.slang")), "unrelated")?;
        }
    }
    let after = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
    assert_eq!(before.observations, after.receipt().observations);
    assert!(before.valid());
    assert!(!before
        .observations
        .keys()
        .any(|q| q.path.starts_with(&tree)));
    assert_eq!(first.compile()?, after.compile()?);
    Ok(())
}

#[test]
fn concurrent_proof_workspaces_keep_ast_and_inputs_isolated() -> Result<()> {
    let compiler = Compiler::discover();
    let work = Workspace::new()?;
    std::thread::scope(|scope| -> Result<()> {
        let jobs: Vec<_> = ["0.25", "0.75"].into_iter().map(|value| {
            let compiler = &compiler;
            let directory = work.0.join(value);
            scope.spawn(move || -> Result<Vec<u8>> {
                fs::create_dir(&directory)?;
                fs::write(directory.join("palette.slang"),format!("public float value(){{return {value};}}"))?;
                let source = "import palette; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx,Inputs i){return finish_view(float3(value()),ctx,1.);}";
                let (_,receipt) = compiler.capture_inputs(|compiler|compiler.compile_display(&Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()},source,&directory.join("main.slang"),"parallel"))?;
                assert!(receipt.valid());
                assert!(!directory.join("program.slang-ast").exists(),"AST escaped owned compiler workspace");
                Ok(receipt.observations.values().filter_map(Option::as_ref).flatten().copied().collect())
            })
        }).collect();
        let mut values = Vec::new();
        for job in jobs {
            values.push(job.join().unwrap()?);
        }
        assert_ne!(values[0], values[1]);
        Ok(())
    })
}

#[test]
fn captured_multi_entry_output_matches_cli() -> Result<()> {
    let compiler = Compiler::discover();
    let modules = [("main", "RWStructuredBuffer<float> output; [numthreads(1,1,1)] void first(uint3 id:SV_DispatchThreadID){output[id.x]=0.25;} [numthreads(1,1,1)] void second(uint3 id:SV_DispatchThreadID){output[id.x]=0.75;}")];
    let entries = [("first", "compute"), ("second", "compute")];
    for target in ["spirv", "metal"] {
        let args = ["-target", target];
        let prepared = compiler.prepare_modules(&modules, "main", &entries, &args)?;
        assert_eq!(
            prepared.compile()?,
            compiler.compile_modules(&modules, "main", &entries, &args)?,
            "{target}: every entry point must be emitted"
        );
    }
    Ok(())
}

#[test]
fn compiler_directory_imports_match_cli_precedence() -> Result<()> {
    let original = Compiler::discover().executable.canonicalize()?;
    let root = original.parent().unwrap().parent().unwrap();
    let work = Workspace::new()?;
    let bin = work.0.join("bin");
    fs::create_dir(&bin)?;
    let native = original.with_extension("real");
    let executable = bin.join(if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    });
    fs::copy(
        if native.is_file() { &native } else { &original },
        &executable,
    )?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(root.join("lib"), work.0.join("lib"))?;
    #[cfg(windows)]
    for directory in [root.join("bin"), root.join("lib")] {
        if !directory.is_dir() {
            continue;
        }
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("dll"))
            {
                fs::copy(&path, bin.join(path.file_name().unwrap()))?;
            }
        }
    }
    fs::write(
        bin.join("palette.slang"),
        "public float value(){return 0.25;}",
    )?;
    fs::write(
        work.0.join("palette.slang"),
        "public float value(){return 0.75;}",
    )?;
    let compiler = Compiler::new(executable);
    let include = work.0.to_string_lossy();
    let modules = [("main","import palette; RWStructuredBuffer<float> output; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){output[id.x]=value();}")];
    for target in ["spirv", "metal"] {
        let args = ["-target", target, "-I", &include];
        let prepared = compiler.prepare_modules(&modules, "main", &[("main", "compute")], &args)?;
        assert_eq!(
            prepared.compile()?,
            compiler.compile_modules(&modules, "main", &[("main", "compute")], &args)?
        );
        assert!(prepared.receipt().valid());
    }
    Ok(())
}

// Keep the permission fixture effective on Windows as well as Unix, and restore
// it even when a compile assertion fails so the workspace can always be removed.
struct UnreadableDirectory {
    path: PathBuf,
    #[cfg(windows)]
    account: String,
}
impl UnreadableDirectory {
    fn new(path: &Path) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0))?;
            Ok(Self { path: path.into() })
        }
        #[cfg(windows)]
        {
            let user = Command::new("whoami").output()?;
            ensure!(user.status.success(), "whoami failed for ACL fixture");
            let account = String::from_utf8(user.stdout)?.trim().to_owned();
            let result = Command::new("icacls")
                .arg(path)
                .arg("/deny")
                .arg(format!("{account}:(OI)(CI)(RX)"))
                .output()?;
            ensure!(
                result.status.success(),
                "cannot establish unreadable directory: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            Ok(Self {
                path: path.into(),
                account,
            })
        }
    }
}
impl Drop for UnreadableDirectory {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&self.path, fs::Permissions::from_mode(0o700));
        }
        #[cfg(windows)]
        {
            let _ = Command::new("icacls")
                .arg(&self.path)
                .arg("/remove:d")
                .arg(&self.account)
                .output();
        }
    }
}

#[test]
fn receipts_store_digests_instead_of_imported_source() -> Result<()> {
    let compiler = Compiler::discover();
    let work = Workspace::new()?;
    let helper = work.0.join("palette.slang");
    let source = format!(
        "/*{}*/ public float value(){{return 0.25;}}",
        "comment ".repeat(20000)
    );
    fs::write(&helper, &source)?;
    let include = work.0.to_string_lossy();
    let modules = [("main","import palette; RWStructuredBuffer<float> output; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){output[id.x]=value();}")];
    let prepared = compiler.prepare_modules(
        &modules,
        "main",
        &[("main", "compute")],
        &["-target", "spirv", "-I", &include],
    )?;
    let receipt = prepared.receipt();
    assert!(
        serde_json::to_vec(&receipt)?.len() < source.len() / 2,
        "receipt duplicated imported source"
    );
    assert!(receipt.valid());
    fs::write(helper, source.replace("0.25", "0.75"))?;
    assert!(!receipt.valid(), "digest must invalidate edited imports");
    Ok(())
}

#[test]
fn absolute_header_includes_preserve_retained_inputs() -> Result<()> {
    let compiler = Compiler::discover();
    let work = Workspace::new()?;
    let header = work.0.join("value.h");
    fs::write(&header, "float value(){return 0.25;}")?;
    let source = format!("#include \"{}\"\nRWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){{result[id.x]=value();}}",filesystem::compiler_path(&header));
    for target in ["spirv", "metal"] {
        let prepared = compiler.prepare_modules(&[("main", &source)], "main", &[("main", "compute")], &["-target", target])?;
        let original = prepared.compile()?;
        fs::write(&header, "float value(){return 0.75;}")?;
        assert!(!prepared.receipt().valid());
        assert_eq!(original, prepared.compile()?);
        let edited = compiler.prepare_modules(&[("main", &source)], "main", &[("main", "compute")], &["-target", target])?;
        assert_ne!(prepared.identity(), edited.identity());
        assert_ne!(original, edited.compile()?);
        fs::write(&header, "float value(){return 0.25;}")?;
    }
    Ok(())
}
