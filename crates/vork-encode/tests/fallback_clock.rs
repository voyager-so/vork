//! A readback output that uses FFmpeg is validated before its old movie
//! is replaced (the renderer begins the output run only after this check).
#![cfg(unix)]

#[cfg(unix)]
#[path = "../../test_support.rs"]
mod test_support;

use std::process::Command;
use vork_encode::{Backend, EncodeSettings, Encoder};

#[test]
fn ffmpeg_fallback_clock_is_refused_before_output_replacement() {
    const CHILD: &str = "VORK_TEST_FALLBACK_CLOCK";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = std::path::PathBuf::from(root);
        for extension in ["mp4", "mov"] {
            let path = root.join(format!("existing.{extension}"));
            std::fs::write(&path, b"preserve output").unwrap();
            // Readback never prepares VideoToolbox, like an unavailable session.
            let prepared = Encoder::prepare(
                Backend::Readback,
                64,
                32,
                vork_encode::Time::integer(24),
                EncodeSettings::default(),
                &path,
            )
            .unwrap();
            let error = format!("{:#}", prepared.validate_fallback(1, None).unwrap_err());
            assert!(error.contains("required for exact movie timing"), "{error}");
            assert_eq!(error.lines().count(), 1, "{error}");
            assert_eq!(std::fs::read(&path).unwrap(), b"preserve output");
        }
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-fallback-clock-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    // Both readback codecs are available, but none of the exact-clock options are.
    let ffmpeg = root.join("ffmpeg");
    test_support::write_executable(
        &ffmpeg,
        b"#!/bin/sh\ncase \"$*\" in\n *-encoders*) printf '%s\\n' ' V....D libx264 H.264' ' VFS... prores_ks Apple ProRes';;\n *) echo 'Main options:';;\nesac\n",
    )
    .unwrap();

    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "ffmpeg_fallback_clock_is_refused_before_output_replacement",
            "--nocapture",
        ])
        .env(CHILD, &root)
        .env("PATH", &root)
        .output()
        .unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
