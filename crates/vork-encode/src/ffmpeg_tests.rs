#[cfg(unix)]
#[path = "../../test_support.rs"]
mod test_support;

use super::*;
use std::io::Write;

fn finish(mut command: Command, input: &[u8]) {
    let mut child = command
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let write = child.stdin.take().unwrap().write_all(input);
    let result = child.wait_with_output().unwrap();
    assert!(
        result.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    write.unwrap();
}

fn check(path: &Path, fps: Time, frames: usize) {
    let oracle = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/movie_clock.py");
    let result = Command::new(if cfg!(windows) { "python" } else { "python3" })
        .arg(oracle)
        .arg(path)
        .args([
            fps.to_string(),
            frames.to_string(),
            "--identity-start".into(),
            "7".into(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{path:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn pixels(frames: usize) -> Vec<u8> {
    (7..frames + 7)
        .flat_map(|frame| {
            (0..64 * 32).flat_map(move |pixel| {
                let value = (((frame >> ((pixel % 64) / 4)) & 1) * 255) as u8;
                [value, value, value, 255]
            })
        })
        .collect()
}

fn encode(path: &Path, fps: Time, frames: usize) {
    rate::validate_ffmpeg_rate(fps, path).unwrap();
    validate_video_clock(fps, frames as u64, path).unwrap();
    let codec = if path.extension().unwrap() == "mov" {
        "prores_ks"
    } else {
        "libx264"
    };
    let gop = (fps.numerator() as u64 * 2).div_ceil(fps.denominator() as u64) as u32;
    finish(
        readback_command(64, 32, fps, frames as u64, EncodeSettings::default(), codec, gop, path).unwrap(),
        &pixels(frames),
    );
    check(path, fps, frames);
}

#[test]
fn exact_movie_clocks_and_order_survive_ffmpeg_transports() {
    let root = std::env::var_os("VORK_MOVIE_CLOCK_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(format!("vork-movie-clock-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    // Short/default clips, both parser-loss witnesses, codec limits and unrelated fractions.
    let mut rates = vec![
        "24",
        "30000/1001",
        "100000001/10000000",
        "1113587715/197197819",
        "2147483647/2147483629",
        "1/2147483646",
        "1073741823",
        "1234567891/987654319",
    ]
    .into_iter()
    .map(|s| s.parse::<Time>().unwrap())
    .collect::<Vec<_>>();
    // Broad deterministic fractions catch failures unrelated to known witnesses.
    let mut state = 0x9a713c4bu64;
    for _ in 0..16 {
        let mut component = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) % (i32::MAX as u64 - 1) + 1) as i64
        };
        rates.push(Time::new(component(), component()).unwrap());
    }
    for frames in [1, 32] {
        let source = root.join(format!("source-{frames}.mp4"));
        encode(&source, Time::integer(24), frames);
        let h264 = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(source)
            .args([
                "-c:v",
                "copy",
                "-bsf:v",
                "h264_mp4toannexb",
                "-f",
                "h264",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(h264.status.success());
        for fps in &rates {
            for extension in ["mp4", "mov"] {
                let path =
                    root.join(format!("readback-{fps}-{frames}.{extension}").replace('/', "_"));
                encode(&path, *fps, frames);
            }
            // The actual native-NVIDIA mux command, with a timestamp-free H.264 input.
            let path = root.join(format!("mux-{fps}-{frames}.mp4").replace('/', "_"));
            if fps.numerator() as u128 > fps.denominator() as u128 * 1_000_000 {
                let error = mux_command(*fps, &path).err().unwrap().to_string();
                assert!(error.contains("shorter than 1 microsecond"), "{error}");
                assert_eq!(error.lines().count(), 1);
                assert!(!path.exists());
                continue;
            }
            finish(mux_command(*fps, &path).unwrap(), &h264.stdout);
            check(&path, *fps, frames);
        }
    }
    for extension in ["mp4", "mov"] {
        encode(
            &root.join(format!("long.{extension}")),
            "1113587715/197197819".parse().unwrap(),
            4096,
        );
    }
    if std::env::var_os("VORK_MOVIE_CLOCK_OUTPUT").is_none() {
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn distribution_ffmpeg_movie_clocks_are_exact() {
    let root =
        std::env::temp_dir().join(format!("vork-distribution-ffmpeg-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for (rate, frames) in [("24", 24), ("25", 25), ("30000/1001", 30)] {
        let fps = rate.parse().unwrap();
        encode(&root.join("readback.mov"), fps, frames);
        let source = root.join("readback.mp4");
        encode(&source, fps, frames);
        let h264 = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&source)
            .args([
                "-c:v",
                "copy",
                "-bsf:v",
                "h264_mp4toannexb",
                "-f",
                "h264",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(
            h264.status.success(),
            "{}",
            String::from_utf8_lossy(&h264.stderr)
        );
        let path = root.join("mux.mp4");
        finish(mux_command(fps, &path).unwrap(), &h264.stdout);
        check(&path, fps, frames);
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn clock_preflight_refuses_unrepresentable_dependency_clocks() {
    let old_help = "-vsync method\n-enc_time_base ratio\n-video_track_timescale int\nsetts_bsf AVOptions:\n  -pts string\n  -dts string\n\n";
    let old = Clock::from_help(old_help).unwrap();
    Clock::from_help(&old_help.replace("\n", "\r\n"))
        .unwrap()
        .validate(Time::integer(24), 24, Path::new("clip.mp4"))
        .unwrap();
    assert!(
        Clock::from_help(&old_help.replace("-dts string", "-dts string\n-time_base rational"))
            .is_err()
    );
    for (rate, frames) in [("24", 24), ("25", 1), ("30000/1001", 30)] {
        for extension in ["mov", "mp4"] {
            old.validate(
                rate.parse().unwrap(),
                frames,
                &PathBuf::from(format!("clip.{extension}")),
            )
            .unwrap();
        }
    }
    for (rate, frames, extension, missing) in [
        ("24", 1, "mov", "movie_timescale"),
        ("24", 1, "mp4", "movie_timescale"),
        ("100000001/10000000", 1, "mov", "setts duration"),
        ("2997/100", 2997, "mp4", "setts duration"),
    ] {
        let error = old
            .validate(
                rate.parse().unwrap(),
                frames,
                &PathBuf::from(format!("clip.{extension}")),
            )
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(missing) && error.contains("install FFmpeg 5.1+"),
            "{error}"
        );
        assert_eq!(error.lines().count(), 1);
    }
    // Missing required capabilities refuse even if the executable reports success.
    for option in [
        "-vsync",
        "-enc_time_base",
        "-video_track_timescale",
        "-pts",
        "-dts",
    ] {
        let error = Clock::from_help(&old_help.replace(option, "-absent"))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(option), "{error}");
    }
    let modern = Clock::from_help(
        &old_help
            .replace(
                "-dts string",
                "-dts string\n-duration string\n-time_base rational",
            )
            .replace("-vsync method", "-vsync method\n-movie_timescale int"),
    )
    .unwrap();
    modern
        .validate(
            "1113587715/197197819".parse().unwrap(),
            32,
            Path::new("clip.mp4"),
        )
        .unwrap();
}

// Exercise the readback command itself: macOS intentionally skips the global
// FFmpeg preflight while native VideoToolbox is still an option.
#[cfg(unix)]
#[test]
fn readback_preflight_refuses_inexact_movie_clock() {
    const CHILD: &str = "VORK_TEST_READBACK_CLOCK";
    if let Some(root) = std::env::var_os(CHILD) {
        for extension in ["mp4", "mov"] {
            let path = PathBuf::from(&root).join(format!("existing.{extension}"));
            std::fs::write(&path, b"preserve output").unwrap();
            let error = readback_command(
                64,
                32,
                Time::integer(24),
                1,
                EncodeSettings::default(),
                if extension == "mov" {
                    "prores_ks"
                } else {
                    "libx264"
                },
                48,
                &path,
            )
            .unwrap_err()
            .to_string();
            assert_eq!(
                error,
                "ffmpeg lacks movie_timescale for exactly 1 frames at fps 24; install FFmpeg 5.1+ on PATH"
            );
            assert_eq!(error.lines().count(), 1);
            assert_eq!(std::fs::read(&path).unwrap(), b"preserve output");
            // The complete second fits the legacy clock; do not reject its rate.
            readback_command(
                64,
                32,
                Time::integer(24),
                24,
                EncodeSettings::default(),
                if extension == "mov" {
                    "prores_ks"
                } else {
                    "libx264"
                },
                48,
                &path,
            )
            .unwrap();
        }
        return;
    }
    let root = std::env::temp_dir().join(format!("vork-readback-clock-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    // FFmpeg 4.4's real timing capabilities, isolated from the process-wide cache.
    let ffmpeg = root.join("ffmpeg");
    test_support::write_executable(&ffmpeg, "#!/bin/sh\nprintf '%s\\n' '-vsync method' '-enc_time_base ratio' '-video_track_timescale int' 'setts_bsf AVOptions:' '  -pts string' '  -dts string' ''\n").unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "ffmpeg::tests::readback_preflight_refuses_inexact_movie_clock",
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

#[test]
fn clock_preflight_selects_fps_mode_on_ffmpeg_without_vsync() {
    // FFmpeg 9 help: per-stream options carry a stream specifier, -vsync is gone.
    let ffmpeg9 = "-enc_time_base[:<stream_spec>] <ratio>  set the desired time base\n-fps_mode[:<stream_spec>]  set framerate mode\n  -movie_timescale   <int>\n  -video_track_timescale <int>\nsetts_bsf AVOptions:\n  -pts <string>\n  -dts <string>\n  -duration <string>\n  -time_base <rational>\n\n";
    let modern = Clock::from_help(ffmpeg9).unwrap();
    assert!(modern.fps_mode && modern.packet_clock && modern.movie_timescale);
    modern.validate(Time::integer(24), 1, Path::new("clip.mp4")).unwrap();
    // FFmpeg 4.4 keeps -vsync.
    let old = Clock::from_help("-vsync method\n-enc_time_base ratio\n-video_track_timescale int\nsetts_bsf AVOptions:\n  -pts string\n  -dts string\n\n").unwrap();
    assert!(!old.fps_mode);
    let error = Clock::from_help(&ffmpeg9.replace("-fps_mode", "-absent")).err().unwrap().to_string();
    assert!(error.contains("-fps_mode or -vsync"), "{error}");
}

#[test]
fn stream_copy_mux_refuses_submicrosecond_frames() {
    for rate in ["24", "1000000", "2000000/2", "1000000000/1001"] {
        validate_mux_rate(rate.parse().unwrap()).unwrap();
    }
    for rate in ["1000001", "2000001/2", "1073741823"] {
        let error = validate_mux_rate(rate.parse().unwrap()).unwrap_err().to_string();
        assert!(error.contains("shorter than 1 microsecond") && error.contains("1000000 fps"), "{error}");
        assert_eq!(error.lines().count(), 1);
    }
}

#[test]
fn software_h264_targets_quality_and_names_its_primaries() {
    let command = readback_command(
        3840,
        2160,
        Time::new(30000, 1001).unwrap(),
        1,
        EncodeSettings { hardware_bitrate: 7_000_000, ..EncodeSettings::default() },
        "libx264",
        60,
        Path::new("out.mp4"),
    )
    .unwrap();
    let args: Vec<_> = command.get_args().map(|s| s.to_str().unwrap()).collect();
    let value = |key| args[args.iter().position(|s| *s == key).unwrap() + 1];
    // The portable default is a quality target; a bitrate belongs to hardware encoders only.
    assert_eq!((value("-crf"), value("-preset")), ("18", "veryfast"));
    assert!(!args.contains(&"-b:v"), "{args:?}");
    assert_eq!(value("-video_size"), "3840x2160");
    assert_eq!(value("-framerate"), "30000/1001");
    assert_eq!(value("-enc_time_base"), "1001:30000");
    assert_eq!((value("-g"), value("-bf"), value("-pix_fmt")), ("60", "0", "yuv420p"));
    assert_eq!(value("-color_primaries"), "bt709");
    assert!(value("-vf").contains("out_color_matrix=bt709:out_range=tv"));
    // A Display P3 picture names its primaries on the frames and the stream; the matrix, range and
    // transfer are the ones every picture is converted with.
    let wide = readback_command(
        1920,
        1080,
        Time::integer(24),
        1,
        EncodeSettings { primaries: Primaries::DisplayP3, ..EncodeSettings::default() },
        "libx264",
        48,
        Path::new("out.mp4"),
    )
    .unwrap();
    let args: Vec<_> = wide.get_args().map(|s| s.to_str().unwrap()).collect();
    let value = |key| args[args.iter().position(|s| *s == key).unwrap() + 1];
    assert_eq!(value("-color_primaries"), "smpte432");
    assert_eq!((value("-color_trc"), value("-colorspace"), value("-color_range")), ("bt709", "bt709", "tv"));
    assert!(value("-vf").contains("out_color_matrix=bt709:out_range=tv,setparams=range=limited:color_primaries=smpte432:color_trc=bt709:colorspace=bt709"));
    assert_eq!((Primaries::Bt709.h273(), Primaries::DisplayP3.h273()), (1, 12));
}
