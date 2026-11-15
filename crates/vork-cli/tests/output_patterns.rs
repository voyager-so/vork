#[cfg(all(unix, not(target_os = "macos")))]
#[path = "../../test_support.rs"]
mod test_support;

use std::process::Command;

#[test]
fn png_frame_patterns_are_usage_errors_even_for_one_still() {
    let dir = std::env::temp_dir().join(format!("vork-output-patterns-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.json");
    std::fs::write(&scene, r#"{"version":2,"nodes":[{"type":"camera"}]}"#).unwrap();
    for name in ["frame_####.png", "frame_%04d.png"] {
        let path = dir.join(name);
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .args(["render"]).arg(&scene).arg("--out").arg(&path)
            .args(["--at", "0"]).output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("NNNNNN.png") && stderr.contains("%06d.exr"), "{stderr}");
        assert!(!path.exists());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "renders images; requires a supported GPU, pinned Slang and OIDN"]
fn png_still_paths_preserve_literal_characters() {
    let dir = std::env::temp_dir().join(format!("vork-literal-output-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.json");
    std::fs::write(&scene, r#"{"version":2,"render":{"resolution":[8,8]},"nodes":[{"type":"camera"}]}"#).unwrap();
    for name in ["renders/review#2/still.png", "renders/100%/still.png", "part#1.png", "100%.png"] {
        for at in [false, true] {
            let path = dir.join(name);
            let mut command = Command::new(env!("CARGO_BIN_EXE_vork"));
            command.arg("render").arg(&scene).arg("--out").arg(&path).args(["--profile", "std/draft"]);
            if at { command.args(["--at", "0"]); }
            let output = command.output().unwrap();
            assert!(output.status.success(), "{}: {}", path.display(), String::from_utf8_lossy(&output.stderr));
            assert!(path.is_file());
            let metadata = path.with_file_name(format!("{}.meta.json", path.file_name().unwrap().to_string_lossy()));
            let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(metadata).unwrap()).unwrap();
            assert_eq!(metadata["profile"]["level"], 0, "{metadata}");
            std::fs::remove_file(path).unwrap();
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "renders images; requires a supported GPU, pinned Slang and OIDN"]
fn failed_overwrite_cannot_retain_success_metadata() {
    let dir = std::env::temp_dir().join(format!("vork-failed-overwrite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.json");
    std::fs::write(&scene, r#"{"version":2,"render":{"resolution":[8,8]},"nodes":[{"type":"camera"}]}"#).unwrap();
    let out = dir.join("frames");
    let render = |diagnostics| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vork"));
        command.arg("render").arg(&scene).arg("--out").arg(&out)
            .args(["--profile", "std/draft", "--at", "0"]);
        for name in ["still.png", "float-%06d.exr", "video.mp4"] {
            command.arg("--out").arg(dir.join(name));
        }
        if diagnostics { command.arg("--diagnostics"); }
        command.output().unwrap()
    };
    let first = render(false);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let manifests = [out.join("meta.json"), dir.join("still.png.meta.json"),
        dir.join("float.exr.meta.json"), dir.join("video.mp4.meta.json")];
    for manifest in &manifests { assert!(manifest.is_file(), "{}", manifest.display()); }
    // Every written render reports its timings beside its metadata.
    assert!(out.join("timings.json").is_file());
    // An output I/O failure, after a previous successful run.
    std::fs::remove_file(out.join("timings.json")).unwrap();
    std::fs::create_dir(out.join("timings.json")).unwrap();
    let failed = render(true);
    assert!(!failed.status.success());
    for manifest in &manifests {
        assert!(!manifest.exists(), "failed overwrite retained {}", manifest.display());
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
#[ignore = "renders images; requires a supported GPU, pinned Slang and OIDN"]
fn shorter_overwrite_and_benchmarks_do_not_claim_old_frames() {
    let dir = std::env::temp_dir().join(format!("vork-short-overwrite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.json");
    std::fs::write(&scene, r#"{"version":2,"render":{"resolution":[8,8],"duration":1,"fps":4},"nodes":[{"type":"camera"}]}"#).unwrap();
    let out = dir.join("frames");
    let render = |args: &[&str]| {
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .arg("render").arg(&scene).arg("--out").arg(&out)
            .args(["--profile", "std/draft"]).args(args).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    };
    render(&["--channels", "color,depth,camera", "--diagnostics"]);
    assert!(out.join("000003.png").exists());
    assert!(out.join("camera.json").exists());
    std::fs::write(out.join("notes.txt"), "keep").unwrap();
    let preserved: Vec<_> = [0, 2, 3].into_iter().flat_map(|frame| {
        [".png", ".depth.png"].into_iter().map(move |suffix| format!("{frame:06}{suffix}"))
    }).map(|name| {
        let bytes = std::fs::read(out.join(&name)).unwrap();
        (name, bytes)
    }).collect();
    render(&["--frame-start", "1", "--frame-end", "1"]);
    assert!(out.join("000001.png").exists());
    for (name, bytes) in &preserved {
        assert_eq!(std::fs::read(out.join(name)).unwrap(), *bytes, "preserve {name}");
    }
    for name in ["000001.depth.png", "camera.json"] {
        assert!(!out.join(name).exists(), "stale {name}");
    }
    // Every written render replaces the timing report with its own.
    let timing: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("timings.json")).unwrap()).unwrap();
    assert_eq!(timing["frames"], 1);
    assert!(timing["stages"]["wall_ms"].as_f64().unwrap() > 0.);
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta["frames"].as_array().unwrap().len(), 1);
    assert_eq!(meta["frames"][0]["index"], 1);
    // A shorter whole clip still replaces the entire generated namespace.
    std::fs::write(&scene, r#"{"version":2,"render":{"resolution":[8,8],"duration":0.25,"fps":4},"nodes":[{"type":"camera"}]}"#).unwrap();
    render(&[]);
    assert!(out.join("000000.png").exists());
    for name in ["000001.png", "000002.png", "000003.png", "000000.depth.png", "000001.depth.png", "000002.depth.png", "000003.depth.png", "camera.json"] {
        assert!(!out.join(name).exists(), "stale {name}");
    }
    assert!(out.join("notes.txt").exists());
    let meta = std::fs::read(out.join("meta.json")).unwrap();
    let timing = std::fs::read(out.join("timings.json")).unwrap();
    for flag in ["--no-write", "--gpu-only"] {
        render(&["--diagnostics", flag]);
        assert_eq!(std::fs::read(out.join("meta.json")).unwrap(), meta);
        assert_eq!(std::fs::read(out.join("timings.json")).unwrap(), timing);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
#[ignore = "renders images; requires a supported GPU, pinned Slang and OIDN"]
fn interrupted_overwrite_has_no_completion_record() {
    use std::os::unix::process::ExitStatusExt;
    let dir = std::env::temp_dir().join(format!("vork-interrupted-overwrite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.json");
    std::fs::write(&scene, r#"{"version":2,"render":{"resolution":[8,8],"duration":10000,"fps":24},"nodes":[{"type":"camera"}]}"#).unwrap();
    let out = dir.join("frames");
    let command = || {
        let mut c = Command::new(env!("CARGO_BIN_EXE_vork"));
        c.arg("render").arg(&scene).arg("--out").arg(&out).args(["--profile", "std/draft"]);
        c
    };
    let first = command().args(["--frame-end", "99"]).output().unwrap();
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    assert!(out.join("meta.json").exists());
    let previous = std::fs::read(out.join("000001.png")).unwrap();
    let mut changed: serde_json::Value = serde_json::from_slice(&std::fs::read(&scene).unwrap()).unwrap();
    changed["render"]["background"] = serde_json::json!([0.8, 0.1, 0.1]);
    std::fs::write(&scene, serde_json::to_vec(&changed).unwrap()).unwrap();
    let replaced = || std::fs::read(out.join("000001.png"))
        .is_ok_and(|bytes| !bytes.is_empty() && bytes != previous);
    let log = std::fs::File::create(dir.join("interrupted.log")).unwrap();
    let mut child = command().stdout(std::process::Stdio::null()).stderr(log).spawn().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while !replaced() && std::time::Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() { break; }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let wrote = replaced();
    let signalled = Command::new("kill").args(["-INT", &child.id().to_string()]).status().unwrap();
    assert!(signalled.success(), "{}", std::fs::read_to_string(dir.join("interrupted.log")).unwrap());
    let status = child.wait().unwrap();
    assert!(wrote, "{}", std::fs::read_to_string(dir.join("interrupted.log")).unwrap());
    assert_eq!(status.signal(), Some(2), "expected SIGINT: {}", std::fs::read_to_string(dir.join("interrupted.log")).unwrap());
    assert!(!out.join("meta.json").exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(windows)]
#[test]
#[ignore = "renders images; requires a Windows console, supported GPU, pinned Slang and OIDN"]
fn ctrl_break_closes_sequence_and_reports_saved_output() {
    use std::os::windows::process::CommandExt;
    #[link(name = "Kernel32")]
    unsafe extern "system" { fn GenerateConsoleCtrlEvent(event: u32, group: u32) -> i32; }
    let dir = std::env::temp_dir().join(format!("vork-break-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.json");
    std::fs::write(&scene, r#"{"version":2,"render":{"resolution":[8,8],"duration":10000,"fps":24},"nodes":[{"type":"camera"}]}"#).unwrap();
    let out = dir.join("frames");
    let log_path = dir.join("interrupt.log");
    let log = std::fs::File::create(&log_path).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vork"))
        .arg("render").arg(scene).arg("--out").arg(&out).args(["--profile", "std/previs"])
        .creation_flags(0x00000200) // CREATE_NEW_PROCESS_GROUP: target only the renderer.
        .stdout(log.try_clone().unwrap()).stderr(log).spawn().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(180);
    while !out.join("000001.png").exists() && std::time::Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() { break; }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(out.join("000001.png").exists(), "renderer did not begin writing");
    assert_ne!(unsafe { GenerateConsoleCtrlEvent(1, child.id()) }, 0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() { break status; }
        if std::time::Instant::now() > deadline { child.kill().unwrap(); panic!("Ctrl+Break did not finish output"); }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert_eq!(status.code(), Some(130));
    let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(out.join("meta.json")).unwrap()).unwrap();
    let frames = metadata["frames"].as_array().unwrap();
    assert!(!frames.is_empty() && frames.len() < 240000);
    for frame in frames {
        let path = out.join(format!("{:06}.png", frame["index"].as_u64().unwrap()));
        assert!(std::fs::read(path).unwrap().starts_with(b"\x89PNG\r\n\x1a\n"));
    }
    let log = std::fs::read_to_string(log_path).unwrap();
    assert!(log.contains("Interrupted: saved"));
    assert!(log.contains("output files closed cleanly"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn png16_requires_a_png_colour_destination() {
    for args in [vec!["--out", "movie.mp4"], vec!["--out", "frame%06d.exr"],
        vec!["--out", "frame.png", "--channels", "depth"],
        vec!["--out", "frame.png", "--no-write", "--diagnostics"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .args(["render", "missing.json", "--png-depth", "16"]).args(args).output().unwrap();
        assert_eq!(output.status.code(),Some(2),"{}",String::from_utf8_lossy(&output.stderr));
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires PNG colour output"));
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
#[test]
fn missing_ffmpeg_clock_capability_preserves_every_destination() {
    let root = std::env::temp_dir().join(format!("vork-clock-preflight-{}", std::process::id()));
    let bin = root.join("bin");
    let stills = root.join("stills");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::create_dir_all(&stills).unwrap();
    let ffmpeg = bin.join("ffmpeg");
    test_support::write_executable(&ffmpeg, "#!/bin/sh\nif [ \"$1\" = '-encoders' ]; then\n echo ' V..... prores_ks ProRes'\n echo ' V..... libx264 H.264'\nfi\n").unwrap();
    let scene = root.join("scene.json");
    std::fs::write(&scene, r#"{"version":2,"render":{"resolution":[64,32],"fps":24,"duration":1},"nodes":[{"type":"camera"}]}"#).unwrap();
    for extension in ["mov", "mp4"] {
        let movie = root.join(format!("clip.{extension}"));
        let mut preserved = vec![movie.clone(), stills.join("000000.png"), stills.join("meta.json")];
        for suffix in ["meta.json", "camera.json", "timings.json"] {
            preserved.push(root.join(format!("clip.{extension}.{suffix}")));
        }
        for path in &preserved { std::fs::write(path, b"previous completed render").unwrap(); }
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .env("PATH", &bin).args(["render"]).arg(&scene)
            .arg("--out").arg(&movie).arg("--out").arg(&stills)
            .output().unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(4), "{stderr}");
        let diagnostics: Vec<_> = stderr.lines().filter(|line| !line.starts_with('[')).collect();
        assert_eq!(diagnostics.len(), 1, "{stderr}");
        assert!(diagnostics[0].starts_with("Error:"), "{stderr}");
        assert!(stderr.contains("ffmpeg lacks") && stderr.contains("exact movie timing"), "{stderr}");
        for path in &preserved { assert_eq!(std::fs::read(path).unwrap(), b"previous completed render", "{}", path.display()); }
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(all(unix, not(target_os = "macos")))]
#[test]
fn ffmpeg_clock_probe_is_reused_across_processes_and_invalidated() {
    let root = std::env::temp_dir().join(format!("vork-ffmpeg-probe-{}", std::process::id()));
    let bin = root.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let ffmpeg = bin.join("ffmpeg");
    let script = "#!/bin/sh\ncase \"$*\" in\n '-version') echo version-one;;\n '-hide_banner -h full')\n echo probe >> \"$PROBE_LOG\"\n printf '%s\\n' '-fps_mode method' '-enc_time_base ratio' '-video_track_timescale int' '-movie_timescale int' 'setts_bsf AVOptions:' ' -pts string' ' -dts string' ' -duration string' ' -time_base rational' '';;\n *-encoders*) echo ' V..... prores_ks ProRes';;\nesac\n";
    test_support::write_executable(&ffmpeg, script).unwrap();
    let scene = root.join("invalid.json");
    // Stop immediately after dependency preflight, without a GPU.
    std::fs::write(&scene, "{").unwrap();
    let log = root.join("probes");
    let movie = root.join("clip.mp4");
    std::fs::write(&movie, b"preserve output").unwrap();
    let run = || {
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .env("PATH", &bin).env("VORK_CACHE_DIR", root.join("cache"))
            .env("PROBE_LOG", &log)
            .args(["render"]).arg(&scene).arg("--out").arg(&movie)
            .output().unwrap();
        assert_eq!(output.status.code(), Some(3), "{}", String::from_utf8_lossy(&output.stderr));
        assert_eq!(std::fs::read(&movie).unwrap(), b"preserve output");
    };
    let probes = || std::fs::read_to_string(&log).unwrap().lines().count();
    run();
    run();
    assert_eq!(probes(), 1, "fresh processes must reuse the full-help probe");
    // Replacement with the same version text must still invalidate the entry.
    test_support::write_executable(&ffmpeg, format!("{script}# rebuilt executable\n")).unwrap();
    run();
    assert_eq!(probes(), 2);
    // A changed runtime version (e.g. shared libraries) must invalidate it too.
    let runtime = root.join("runtime");
    std::fs::write(&runtime, b"one").unwrap();
    let script = script.replace("echo version-one", "/bin/cat \"$RUNTIME_VERSION\"");
    test_support::write_executable(&ffmpeg, &script).unwrap();
    let run_runtime = || {
        let output = Command::new(env!("CARGO_BIN_EXE_vork"))
            .env("PATH", &bin).env("VORK_CACHE_DIR", root.join("cache"))
            .env("PROBE_LOG", &log).env("RUNTIME_VERSION", &runtime)
            .args(["render"]).arg(&scene).arg("--out").arg(&movie)
            .output().unwrap();
        assert_eq!(output.status.code(), Some(3), "{}", String::from_utf8_lossy(&output.stderr));
    };
    run_runtime();
    std::fs::write(&runtime, b"two").unwrap();
    run_runtime();
    assert_eq!(probes(), 4);
    // A warmed timing cache must not conceal a missing selected codec.
    test_support::write_executable(&ffmpeg, script.replace(" V..... prores_ks", " V..... absent")).unwrap();
    let mov = root.join("clip.mov");
    std::fs::write(&mov, b"preserve output").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vork"))
        .env("PATH", &bin).env("VORK_CACHE_DIR", root.join("cache"))
        .args(["render"]).arg(&scene).arg("--out").arg(&mov)
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(4), "{stderr}");
    assert!(stderr.contains("ffmpeg encoder prores_ks unavailable"), "{stderr}");
    assert_eq!(stderr.lines().filter(|line| !line.starts_with('[')).count(), 1, "{stderr}");
    assert_eq!(std::fs::read(&mov).unwrap(), b"preserve output");
    std::fs::remove_dir_all(root).unwrap();
}
