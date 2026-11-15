use std::process::Command;

#[test]
fn movie_rate_refusals_preserve_mixed_outputs() {
    let root = std::env::temp_dir().join(format!("vork-rate-refusal-{}", std::process::id()));
    std::fs::create_dir_all(root.join("frames")).unwrap();
    let scene = root.join("scene.json");
    let mut cases = vec![("5.64705898192515", "1", "1113587715/197197819")];
    if !cfg!(target_os = "macos") {
        cases.push((r#""1/2147483647""#, "2147483647", "1/2147483646"));
    }
    for (rate, duration, suggestion) in cases {
        std::fs::write(&scene, format!(r#"{{"version":2,"render":{{"resolution":[64,32],"fps":{rate},"duration":{duration}}},"nodes":[{{"type":"camera"}}]}}"#)).unwrap();
        for kind in ["mp4", "mov"] {
            let output = root.join(format!("clip.{kind}"));
            let preserved = [
                output.clone(),
                root.join(format!("clip.{kind}.meta.json")),
                root.join("frames/meta.json"),
                root.join("frames/000000.png"),
            ];
            for path in &preserved {
                std::fs::write(path, b"preserve").unwrap();
            }
            let result = Command::new(env!("CARGO_BIN_EXE_vork"))
                .arg("render")
                .arg(&scene)
                .arg("--out")
                .arg(output)
                .arg("--out")
                .arg(root.join("frames"))
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert_eq!(result.status.code(), Some(2), "{stderr}");
            let lines: Vec<_> = stderr
                .lines()
                .filter(|line| line.starts_with("Error:"))
                .collect();
            assert_eq!(lines.len(), 1, "{stderr}");
            assert!(
                lines[0].contains(&format!("{} cannot represent fps", kind.to_uppercase())),
                "{stderr}"
            );
            assert!(
                lines[0].ends_with(&format!("use fps \"{suggestion}\"")),
                "{stderr}"
            );
            for path in &preserved {
                assert_eq!(std::fs::read(path).unwrap(), b"preserve");
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "CPU movie-clock render contract; requires lavapipe, pinned Slang and FFmpeg"]
fn exact_movie_rate_container_matrix() {
    let root = std::env::temp_dir().join(format!("vork-movie-matrix-{}", std::process::id()));
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/movie_rates.py");
    let result = Command::new("python3")
        .env("VORK_GPU", "cpu")
        .arg(script)
        .arg("--binary")
        .arg(env!("CARGO_BIN_EXE_vork"))
        .arg("--out")
        .arg(&root)
        .status()
        .unwrap();
    assert!(result.success(), "matrix evidence: {}", root.display());
    std::fs::remove_dir_all(root).unwrap();
}
