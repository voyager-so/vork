//! Late FFmpeg failure must not publish a partial movie or replace an existing one.
#![cfg(unix)]
#[cfg(unix)]
#[path = "../../test_support.rs"]
mod test_support;

use std::process::Command;
use vork_encode::{Backend, DeviceInfo, EncodeSettings, Encoder, Time};

#[test]
fn failed_ffmpeg_movie_is_not_published() {
    const CHILD: &str = "VORK_TEST_FAILED_MOVIE";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = std::path::PathBuf::from(root);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        for extension in ["mp4", "mov"] {
            for existing in [false, true] {
                let path = root.join(format!(
                    "{}.{extension}",
                    if existing { "existing" } else { "new" }
                ));
                if existing {
                    std::fs::write(&path, b"previous complete movie").unwrap();
                }
                let mut encoder = Encoder::open(
                    DeviceInfo {
                        device: device.clone(),
                        queue: queue.clone(),
                        backend: Backend::Readback,
                    },
                    64,
                    32,
                    Time::integer(if extension == "mov" { 7 } else { 24 }),
                    EncodeSettings::default(),
                    &path,
                    1,
                    1,
                )
                .unwrap();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: 64,
                        height: 32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    texture.as_image_copy(),
                    &[255; 64 * 32 * 4],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                    texture.size(),
                );
                let mut surface = encoder.create_surface().unwrap();
                let mut copy = device.create_command_encoder(&Default::default());
                surface.record_copy(&mut copy, &texture);
                let submission = queue.submit([copy.finish()]);
                encoder.push(0, texture, &mut surface, &submission).unwrap();
                let error = encoder.finish().unwrap_err();
                assert!(
                    format!("{error:#}").contains("driver stopped during encoding"),
                    "{error:#}"
                );
                drop(surface);
                drop(encoder);
                if existing {
                    assert_eq!(std::fs::read(&path).unwrap(), b"previous complete movie");
                } else {
                    assert!(!path.exists(), "failed movie was published");
                }
                assert!(!std::fs::read_dir(&root).unwrap().any(|entry| {
                    entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".vork-video-")
                }));
            }
        }
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let ffmpeg = root.path().join("ffmpeg");
    test_support::write_executable(&ffmpeg, r#"#!/bin/sh
case "$*" in
 *-version*) exit 0;;
 *-encoders*) printf '%s\n' ' V....D libx264 H.264' ' V....D prores_ks ProRes';;
 *rawvideo*)
   for last do :; done
   /bin/cat >/dev/null
   printf 'incomplete movie' > "$last"
   echo 'driver stopped during encoding' >&2
   exit 1;;
 *) printf '%s\n' '-fps_mode' '-enc_time_base' '-video_track_timescale' '-movie_timescale' 'setts_bsf AVOptions:' ' -pts' ' -dts';;
esac
"#).unwrap();

    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "failed_ffmpeg_movie_is_not_published",
            "--nocapture",
        ])
        .env(CHILD, root.path())
        .env("PATH", root.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
