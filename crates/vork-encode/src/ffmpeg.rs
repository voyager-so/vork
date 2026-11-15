//! FFmpeg transports preserve ordered frame identity across their rate parsers.
use super::*;

pub(super) fn readback_command(
    width: u32, height: u32, fps: Time, frames: u64, settings: EncodeSettings,
    codec: &str, gop_frames: u32, path: &Path,
) -> Result<Command> {
    let clock = Clock::query()?;
    clock.validate(fps, frames, path)?;
    let prores = codec == "prores_ks";
    let mut command = Command::new("ffmpeg");
    command.args([
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        &format!("{width}x{height}"),
        "-framerate",
        &fps.to_string(),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
    ]);
    if codec == "libx264" {
        // Honor CPU affinity and container quotas as well as bare-metal core count. FFmpeg's
        // automatic count can see every host CPU inside a container with a much smaller quota.
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).to_string();
        command.args([
            "-preset",
            SOFTWARE_PRESET,
            "-crf",
            &settings.software_crf.to_string(),
            "-threads",
            &threads,
        ]);
    } else if prores {
        command.args([
            "-profile:v",
            PRORES_PROFILE,
            "-alpha_bits",
            PRORES_ALPHA_BITS,
            "-threads",
            PRORES_THREADS,
        ]);
    } else {
        command.args(["-b:v", &settings.hardware_bitrate.to_string()]);
    }
    // Set both frame and stream properties: encoders may prefer the
    // converted frame's colour tags over the output codec options.
    let primaries = settings.primaries.ffmpeg();
    let filter = format!("scale=out_color_matrix=bt709:out_range=tv,setparams=range=limited:color_primaries={primaries}:color_trc=bt709:colorspace=bt709");
    command.args([
            "-vf",
            &filter,
            "-color_range",
            "tv",
            "-color_primaries",
            primaries,
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]);
    clock.apply(&mut command, fps);
    // The input rate parser can approximate p/q. Do not let that choose the
    // encoder clock or duplicate/drop frames before packet identity is restored.
    command.args([
        "-enc_time_base",
        &format!("{}:{}", fps.denominator(), fps.numerator()),
    ]);
    // FFmpeg 5.1 replaced -vsync with -fps_mode, and later releases removed
    // -vsync; both spellings pass every frame through unchanged.
    command.args(if clock.fps_mode { ["-fps_mode", "passthrough"] } else { ["-vsync", "0"] });
    command.args([
            "-pix_fmt",
            if prores {
                PRORES_PIXEL_FORMAT
            } else {
                "yuv420p"
            },
            "-bf",
            "0",
            "-g",
            &gop_frames.to_string(),
            "-map_metadata",
            "-1",
            "-f",
            if prores { "mov" } else { "mp4" },
        ])
        .arg(path);
    Ok(command)
}

// Capabilities select an exact clock representation before output invalidation.
// No command is retried with weaker timing after an encoder/muxer failure.
#[derive(Clone, Copy)]
struct Clock {
    packet_clock: bool,
    movie_timescale: bool,
    fps_mode: bool,
}

impl Clock {
    fn query() -> Result<Self> {
        Self::query_with(probe_video_clock_capabilities)
    }

    fn query_with(probe: impl FnOnce() -> Result<String>) -> Result<Self> {
        static CLOCK: std::sync::OnceLock<std::result::Result<Clock, String>> =
            std::sync::OnceLock::new();
        CLOCK
            .get_or_init(|| {
                probe().and_then(|help| Self::from_help(&help))
                    .map_err(|error| format!("{error:#}"))
            })
            .as_ref()
            .copied()
            .map_err(|error| anyhow::anyhow!("{error}"))
    }

    fn from_help(help: &str) -> Result<Self> {
        let help = help.replace("\r\n", "\n");
        // Newer releases list per-stream options as `-name[:<stream_spec>]`.
        let has = |text: &str, option: &str| {
            text.lines().any(|line| {
                line.split_whitespace().next().and_then(|name| name.split('[').next()) == Some(option)
            })
        };
        let fps_mode = has(&help, "-fps_mode");
        ensure!(
            fps_mode || has(&help, "-vsync"),
            "ffmpeg lacks -fps_mode or -vsync required for exact movie timing; install FFmpeg 5.1+ on PATH"
        );
        for option in ["-enc_time_base", "-video_track_timescale"] {
            ensure!(
                has(&help, option),
                "ffmpeg lacks {option} required for exact movie timing; install FFmpeg 5.1+ on PATH"
            );
        }
        let setts = help
            .split_once("setts_bsf AVOptions:")
            .map(|(_, options)| options.split("\n\n").next().unwrap_or(""))
            .unwrap_or("");
        for option in ["-pts", "-dts"] {
            ensure!(
                has(setts, option),
                "ffmpeg lacks setts {option} required for exact movie timing; install FFmpeg 5.1+ on PATH"
            );
        }
        let time_base = has(setts, "-time_base");
        let duration = has(setts, "-duration");
        ensure!(
            time_base == duration,
            "ffmpeg lacks a complete setts packet clock for exact movie timing; install FFmpeg 5.1+ on PATH"
        );
        Ok(Self {
            packet_clock: time_base,
            movie_timescale: has(&help, "-movie_timescale"),
            fps_mode,
        })
    }

    fn validate(self, fps: Time, frames: u64, path: &Path) -> Result<()> {
        let (p, q) = (fps.numerator() as u128, fps.denominator() as u128);
        // Without duration expressions the packet duration comes from the rate
        // parser (bounded to 1001000), and H.264's 1/1200000 demuxer clock.
        // That bound also keeps N*q exact in doubles for every u32 frame index.
        if !self.packet_clock {
            ensure!(
                p <= 1_001_000 && q <= 1_001_000,
                "ffmpeg lacks setts duration for exact fps {fps}; install FFmpeg 5.1+ on PATH"
            );
            let mov = path
                .extension()
                .is_some_and(|s| s.eq_ignore_ascii_case("mov"));
            ensure!(
                mov || 1_200_000 % p == 0,
                "ffmpeg lacks setts duration for exact H.264 muxing at fps {fps}; install FFmpeg 5.1+ on PATH"
            );
        }
        // Older MOV/MP4 muxers fix the movie/edit clock at 1000 Hz. The entire
        // selected clip must fit it exactly, including single-frame clips.
        ensure!(
            self.movie_timescale || (u128::from(frames) * q * 1000) % p == 0,
            "ffmpeg lacks movie_timescale for exactly {frames} frames at fps {fps}; install FFmpeg 5.1+ on PATH"
        );
        Ok(())
    }

    fn apply(self, command: &mut Command, fps: Time) {
        // One packet per input frame, with B frames disabled. FFmpeg's muxer
        // has already selected the track timebase 1/p before running setts.
        let q = fps.denominator();
        let expression = if self.packet_clock {
            format!(
                "setts=time_base={q}/{}:pts=N:dts=N:duration=1",
                fps.numerator()
            )
        } else {
            format!("setts=pts=N*{q}:dts=N*{q}")
        };
        command.args([
            "-bsf:v",
            &expression,
            "-video_track_timescale",
            &fps.numerator().to_string(),
        ]);
        if self.movie_timescale {
            command.args(["-movie_timescale", &fps.numerator().to_string()]);
        }
    }
}

/// Check the installed FFmpeg's timing capabilities before any output is removed.
pub fn validate_video_clock(fps: Time, frames: u64, path: &Path) -> Result<()> {
    if cfg!(target_os = "macos") {
        return Ok(());
    }
    rate::validate_ffmpeg_rate(fps, path)?;
    Clock::query()?.validate(fps, frames, path)
}

/// Check FFmpeg's timing capabilities for an output already known to encode
/// through FFmpeg, on every platform.
pub(super) fn validate_fallback_clock(fps: Time, frames: u64, path: &Path) -> Result<()> {
    rate::validate_ffmpeg_rate(fps, path)?;
    Clock::query()?.validate(fps, frames, path)
}

/// Probe and validate actual timing options. Successful help can be persisted by
/// the caller in its existing content-addressed store; failures are never cached.
pub fn probe_video_clock_capabilities() -> Result<String> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-h", "full"])
        .stdin(Stdio::null())
        .output()
        .context("query ffmpeg capabilities for exact movie timing")?;
    ensure!(output.status.success(), "query ffmpeg capabilities for exact movie timing failed");
    let help = format!("{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    Clock::from_help(&help)?;
    Ok(help)
}

/// Check timing options before parsing the scene or starting device work.
pub fn validate_video_clock_capabilities(probe: impl FnOnce() -> Result<String>) -> Result<()> {
    if cfg!(target_os = "macos") {
        return Ok(());
    }
    Clock::query_with(probe).map(|_| ())
}

/// FFmpeg's stream-copy MP4 mux keeps the movie duration on a microsecond
/// clock, so it cannot time a frame shorter than one microsecond exactly.
#[cfg(any(target_os = "linux", test))]
pub(super) fn validate_mux_rate(fps: Time) -> Result<()> {
    ensure!(
        (fps.numerator() as u128) <= fps.denominator() as u128 * 1_000_000,
        "fps {fps} makes frames shorter than 1 microsecond, which FFmpeg's stream-copy MP4 mux cannot time exactly; use at most 1000000 fps"
    );
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
pub(super) fn mux_command(fps: Time, path: &Path) -> Result<Command> {
    validate_mux_rate(fps)?;
    let clock = Clock::query()?;
    let mut command = Command::new("ffmpeg");
    command.args([
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-r",
        &fps.to_string(),
        "-f",
        "h264",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "copy",
    ]);
    clock.apply(&mut command, fps);
    command.args(["-map_metadata", "-1", "-f", "mp4"]).arg(path);
    Ok(command)
}

#[cfg(test)]
#[path = "ffmpeg_tests.rs"]
mod tests;
