//! A restricted ffmpeg build must fail before any pixels enter its stdin.
#![cfg(unix)]

#[cfg(unix)]
#[path = "../../test_support.rs"]
mod test_support;

use std::process::Command;
use vork_encode::{Backend, DeviceInfo, EncodeSettings, Encoder};

#[test]
fn missing_software_encoder_is_reported_before_streaming() {
    const CHILD: &str = "VORK_TEST_MISSING_CODEC";
    if std::env::var_os(CHILD).is_some() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let root = std::path::PathBuf::from(std::env::var_os(CHILD).unwrap());
        // macOS Auto can encode natively without consulting ffmpeg.
        #[cfg(target_os = "macos")]
        let backends = [Backend::Readback];
        #[cfg(not(target_os = "macos"))]
        let backends = [Backend::Auto, Backend::Readback];
        for backend in backends {
            for extension in ["mp4", "mov"] {
                let path = root.join(format!("existing.{extension}"));
                std::fs::write(&path, b"preserve output").unwrap();
                let result = Encoder::open(
                    DeviceInfo {
                        device: device.clone(),
                        queue: queue.clone(),
                        backend,
                    },
                    64,
                    32,
                    vork_encode::Time::integer(24),
                    EncodeSettings::default(),
                    &path,
                    1,
                    3,
                );
                let error = match result {
                    Err(error) => error,
                    Ok(_) => panic!("missing encoder must fail before streaming"),
                };
                let diagnosis = format!("{error:#}");
                assert!(
                    diagnosis.contains("video encoding needs ffmpeg on PATH with libx264"),
                    "{diagnosis}"
                );
                assert!(
                    diagnosis.contains(if extension == "mp4" {
                        "encoder libx264 unavailable"
                    } else {
                        "encoder prores_ks unavailable"
                    }),
                    "{diagnosis}"
                );
                assert_eq!(std::fs::read(path).unwrap(), b"preserve output");
            }
        }
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-missing-codec-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let ffmpeg = root.join("ffmpeg");
    // Recognize options regardless of whether FFmpeg's banner is requested.
    // The listing must succeed so preflight diagnoses the absent codecs.
    test_support::write_executable(&ffmpeg, br#"#!/bin/sh
for arg in "$@"; do
  case "$arg" in
    -version) exit 0 ;;
    -encoders)
      echo ' V....D h264_nvenc NVIDIA NVENC'
      exit 0 ;;
  esac
done
exit 1
"#).unwrap();

    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "missing_software_encoder_is_reported_before_streaming",
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

#[cfg(target_os = "linux")]
#[test]
fn missing_ffmpeg_is_reported_before_native_nvenc_preparation() {
    const CHILD: &str = "VORK_TEST_NO_FFMPEG";
    if let Some(root) = std::env::var_os(CHILD) {
        for backend in [Backend::Auto, Backend::Readback] {
            let error = match Encoder::prepare(backend, 64, 32, vork_encode::Time::integer(24), EncodeSettings::default(), std::path::Path::new(&root).join("output.mp4")) {
                Err(error) => error,
                Ok(_) => panic!("ffmpeg is required even with native NVENC"),
            };
            let diagnosis = format!("{error:#}");
            assert!(diagnosis.contains("including MP4 with native NVENC"), "{diagnosis}");
            assert!(diagnosis.contains("sudo apt install ffmpeg"), "{diagnosis}");
            assert!(!std::path::Path::new(&root).join("output.mp4").exists());
        }
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-no-ffmpeg-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "missing_ffmpeg_is_reported_before_native_nvenc_preparation", "--nocapture"])
        .env(CHILD, &root).env("PATH", &root).output().unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(result.status.success(), "{}\n{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
}

#[test]
fn fallback_codec_is_checked_before_output_cleanup() {
    const CHILD: &str = "VORK_TEST_FALLBACK_CODEC";
    if let Some(root) = std::env::var_os(CHILD) {
        let path = std::path::Path::new(&root).join("existing.mp4");
        std::fs::write(&path, b"preserve output").unwrap();
        let prepared = Encoder::prepare(Backend::Readback, 64, 32,
            vork_encode::Time::integer(24), EncodeSettings::default(), &path).unwrap();
        let error = prepared.validate_fallback(2, None).expect_err("missing libx264 must refuse before cleanup");
        assert!(error.to_string().contains("encoder libx264 unavailable"), "{error:#}");
        assert_eq!(std::fs::read(path).unwrap(), b"preserve output");
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-fallback-codec-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let ffmpeg = root.join("ffmpeg");
    test_support::write_executable(&ffmpeg, b"#!/bin/sh\ncase \"$*\" in\n *-version*) exit 0;;\n *-encoders*) echo ' V....D h264_nvenc NVIDIA NVENC';;\n *) printf '%s\\n' '-fps_mode' '-enc_time_base' '-video_track_timescale' '-movie_timescale' 'setts_bsf AVOptions:' ' -pts' ' -dts' ' -duration' ' -time_base';;\nesac\n").unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fallback_codec_is_checked_before_output_cleanup", "--nocapture"])
        .env(CHILD, &root).env("PATH", &root).output().unwrap();
    std::fs::remove_dir_all(root).unwrap();
    assert!(result.status.success(), "{}\n{}", String::from_utf8_lossy(&result.stdout), String::from_utf8_lossy(&result.stderr));
}
