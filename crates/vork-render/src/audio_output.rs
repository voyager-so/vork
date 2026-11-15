//! Soundtrack muxing. The scene mixer supplies PCM. Call after finishing the movie encoder and
//! before publishing complete output metadata. No audio is inferred from footage.
//!
//! Decoding/resampling uses user-installed ffmpeg/ffprobe, never linked or bundled.
//! FFmpeg's baseline is LGPL; the installed executable may enable other licences.
use anyhow::{Context, Result, ensure};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};
use vork_time::Time;

const RATE: i128 = 48_000;
const TOOLS: &str = "audio output needs ffmpeg and ffprobe on PATH; install FFmpeg and retry";
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(crate) fn sample_count(frames: u64, fps: Time) -> Result<i64> {
    let samples = sample_offset(frames, fps)?;
    ensure!(
        samples > 0,
        "audio output needs at least one sample of video duration"
    );
    Ok(samples)
}

pub(crate) fn sample_offset(frames: u64, fps: Time) -> Result<i64> {
    ensure!(
        fps > Time::ZERO,
        "audio output needs a positive frame count and fps"
    );
    let numerator = i128::from(frames)
        .checked_mul(i128::from(fps.denominator()))
        .and_then(|n| n.checked_mul(RATE))
        .context("audio duration exceeds the sample clock")?;
    let divisor = i128::from(fps.numerator());
    let samples = (numerator / divisor) + i128::from(numerator % divisor >= (divisor + 1) / 2);
    let samples = i64::try_from(samples).context("audio duration exceeds the sample clock")?;
    Ok(samples)
}

/// Offset is the source's onset on this movie's clock: positive inserts silence,
/// negative cuts the source's beginning. Frame i is at i/fps; the end is N/fps.
/// Durations and offsets round once to the nearest 48 kHz sample (ties away from
/// zero). Gain is finite linear amplitude, including zero or phase inversion.
pub(crate) fn mux_file(
    movie: &Path,
    file: &Path,
    gain: f64,
    offset: f64,
    frames: u64,
    fps: Time,
) -> Result<()> {
    // A sequence directory can itself have a .mov/.mp4 suffix.
    if super::shot::directory_marked(movie)
        || !movie
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("mp4") || e.eq_ignore_ascii_case("mov"))
    {
        return Ok(());
    }
    let samples = sample_count(frames, fps)?;
    ensure!(
        gain.is_finite(),
        "audio gain must be finite; supply a linear amplitude"
    );
    let shift = (offset * RATE as f64).round();
    ensure!(
        offset.is_finite() && shift.abs() < i64::MAX as f64,
        "audio offset exceeds the sample clock; supply a finite offset in seconds"
    );
    let shift = shift as i64;
    let unreadable = || {
        format!("cannot read audio {file:?}; convert it to a mono or stereo PCM WAV file and retry")
    };
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "json",
            "-i",
        ])
        .arg(file)
        .stdin(Stdio::null())
        .output()
        .map_err(|_| anyhow::anyhow!(TOOLS).context(crate::failure::FailureKind::Setup))?;
    ensure!(probe.status.success(), "{}", unreadable());
    let channels = serde_json::from_slice::<serde_json::Value>(&probe.stdout)
        .ok()
        .and_then(|v| v["streams"][0]["channels"].as_u64());
    ensure!(matches!(channels, Some(1 | 2)), "{}", unreadable());
    let layout = if channels == Some(1) {
        "pan=stereo|c0=c0|c1=c0,"
    } else {
        ""
    };
    let placement = if shift < 0 {
        format!("atrim=start_sample={},asetpts=N/SR/TB,", -shift)
    } else {
        // Delays past the delivery span are all silence; do not allocate a
        // delay buffer for time that cannot reach the output.
        format!("adelay={}S:all=1,", shift.min(samples))
    };
    let filter = format!(
        "[1:a:0]{layout}aresample=48000,asetpts=N/SR/TB,volume={gain}:precision=double,{placement}apad,atrim=end_sample={samples}[sound]"
    );
    let mov = movie.extension().unwrap().eq_ignore_ascii_case("mov");
    // Reserve a sibling using create_new; neither a failed subprocess nor a
    // concurrent output can overwrite another file. Drop cleans all exits.
    let temp = loop {
        let path = movie.with_file_name(format!(
            ".vork-audio-{}-{}.{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
            if mov { "mov" } else { "mp4" }
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => break Temporary(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e).context("reserve audio output beside the movie"),
        }
    };
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-y", "-xerror", "-i"])
        .arg(movie)
        .arg("-i")
        .arg(file)
        .args([
            "-filter_complex",
            &filter,
            "-map",
            "0:v:0",
            "-map",
            "[sound]",
            "-c:v",
            "copy",
            "-c:a",
            if mov { "pcm_s16le" } else { "aac" },
            "-ar",
            "48000",
            "-ac",
            "2",
        ])
        .arg(&temp.0)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|_| anyhow::anyhow!(TOOLS).context(crate::failure::FailureKind::Setup))?;
    // The frames are rendered and encoded: what fails here is the FFmpeg run that adds the sound.
    if !status.success() {
        return Err(anyhow::anyhow!("cannot mux audio {file:?}; restore a readable mono or stereo PCM WAV and a finished movie, and install FFmpeg with AAC/PCM encoding")
            .context(crate::failure::FailureKind::Device));
    }
    fs::rename(&temp.0, movie).context("replace finished movie with its audio mux")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Workspace(PathBuf);
    impl Workspace {
        fn new() -> Result<Self> {
            let parent = std::env::var_os("VORK_AUDIO_TEST_OUTPUT")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            fs::create_dir_all(&parent)?;
            let path = parent.join(format!(
                "vork-audio-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for Workspace {
        fn drop(&mut self) {
            if std::env::var_os("VORK_AUDIO_TEST_OUTPUT").is_none() {
                let _ = fs::remove_dir_all(&self.0);
            }
        }
    }

    fn run(command: &mut Command) -> Result<Vec<u8>> {
        let output = command.stdin(Stdio::null()).output()?;
        ensure!(
            output.status.success(),
            "{command:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(output.stdout)
    }
    // Write PCM directly so the decoder fixture does not depend on ffmpeg's
    // tone generator, volume defaults, or interpretation of channel layouts.
    fn wav(path: &Path, seconds: f64, onset: f64, channels: u16) -> Result<()> {
        let rate: u32 = 44_100;
        let count = (seconds * f64::from(rate)).round() as u32;
        let bytes = count * u32::from(channels) * 2;
        let mut data = Vec::new();
        data.extend(b"RIFF");
        data.extend((36 + bytes).to_le_bytes());
        data.extend(b"WAVEfmt ");
        data.extend(16_u32.to_le_bytes());
        data.extend(1_u16.to_le_bytes());
        data.extend(channels.to_le_bytes());
        data.extend(rate.to_le_bytes());
        data.extend((rate * u32::from(channels) * 2).to_le_bytes());
        data.extend((channels * 2).to_le_bytes());
        data.extend(16_u16.to_le_bytes());
        data.extend(b"data");
        data.extend(bytes.to_le_bytes());
        for i in 0..count {
            for c in 0..channels {
                let time = f64::from(i) / f64::from(rate);
                let sample = if time < onset {
                    0.
                } else {
                    0.2 / f64::from(c + 1) * ((time - onset) * 1000. * std::f64::consts::TAU).sin()
                };
                data.extend(((sample * 32767.).round() as i16).to_le_bytes());
            }
        }
        fs::write(path, data)?;
        Ok(())
    }
    fn video(path: &Path, frames: u64, fps: Time) -> Result<()> {
        let mov = path.extension().unwrap() == "mov";
        let mut command = Command::new("ffmpeg");
        command.args([
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=red:s=32x32:r={fps}"),
            "-frames:v",
            &frames.to_string(),
            "-an",
            "-c:v",
            if mov { "prores_ks" } else { "libx264" },
            "-threads",
            "1",
        ]);
        if mov {
            command.args(["-profile:v", "4444", "-pix_fmt", "yuva444p10le"]);
        }
        run(command.arg(path))?;
        Ok(())
    }
    fn probe(path: &Path) -> Result<serde_json::Value> {
        Ok(serde_json::from_slice(&run(Command::new("ffprobe")
            .args(["-v", "error", "-show_streams", "-of", "json"])
            .arg(path))?)?)
    }
    fn decode(path: &Path) -> Result<Vec<f32>> {
        let bytes = run(Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args(["-map", "0:a:0", "-f", "f32le", "-c:a", "pcm_f32le", "-"]))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn rms(samples: &[f32]) -> f64 {
        (samples.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
    }
    fn video_packets(path: &Path) -> Result<Vec<u8>> {
        run(Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args(["-map", "0:v:0", "-c:v", "copy", "-f", "data", "-"]))
    }

    #[test]
    fn movie_tone_sync_gain_padding_and_stream_copy() -> Result<()> {
        let root = Workspace::new()?;
        let source = root.path("tone.wav");
        wav(&source, 0.2, 0., 1)?;
        for fps in [Time::integer(24), Time::new(30_000, 1001)?] {
            for extension in ["mp4", "mov"] {
                let silent = root.path(&format!("silent.{extension}"));
                video(&silent, 30, fps)?;
                let packets = video_packets(&silent)?;
                let onset = 8. / fps.as_f64();
                let mut amplitudes = Vec::new();
                for gain in [1., 0.5] {
                    let output = root.path(&format!("gain-{gain}.{extension}"));
                    fs::copy(&silent, &output)?;
                    mux_file(&output, &source, gain, onset, 30, fps)?;
                    let streams = probe(&output)?;
                    let streams = streams["streams"].as_array().unwrap();
                    assert_eq!(streams.len(), 2);
                    let audio = streams.iter().find(|s| s["codec_type"] == "audio").unwrap();
                    let picture = streams.iter().find(|s| s["codec_type"] == "video").unwrap();
                    assert_eq!(
                        audio["codec_name"],
                        if extension == "mp4" {
                            "aac"
                        } else {
                            "pcm_s16le"
                        }
                    );
                    assert_eq!(audio["sample_rate"], "48000");
                    assert_eq!(audio["channels"], 2);
                    let duration = |s: &serde_json::Value| {
                        s["duration"].as_str().unwrap().parse::<f64>().unwrap()
                    };
                    assert!((duration(audio) - duration(picture)).abs() <= 1. / fps.as_f64());
                    assert!((duration(audio) - 30. / fps.as_f64()).abs() < 0.001);
                    assert_eq!(
                        video_packets(&output)?,
                        packets,
                        "mux must preserve every compressed video packet"
                    );
                    let pcm = decode(&output)?;
                    let left: Vec<_> = pcm.iter().step_by(2).copied().collect();
                    let detected = left.iter().position(|x| x.abs() > 0.005).unwrap();
                    let expected = (onset * 48_000.).round() as usize;
                    println!(
                        "{extension} fps={fps} gain={gain}: video={}s audio={}s onset={detected} expected={expected} samples",
                        duration(picture),
                        duration(audio)
                    );
                    assert!(
                        (detected as f64 - expected as f64).abs() <= 24_000. / fps.as_f64(),
                        "{extension} onset {detected}, expected {expected}"
                    );
                    let length = sample_count(30, fps)? as usize;
                    assert!(left.len() >= length);
                    assert!(
                        rms(&left[length - 2400..length]) < 0.0001,
                        "short source must pad silence"
                    );
                    amplitudes.push(rms(&left[expected + 2400..expected + 7200]));
                    if extension == "mov" {
                        assert!(
                            pcm.chunks_exact(2).all(|s| s[0] == s[1]),
                            "mono duplicates without attenuation"
                        );
                        assert!(
                            (amplitudes.last().unwrap() - 0.2 * gain / 2_f64.sqrt()).abs() < 0.001
                        );
                    }
                }
                assert!(
                    (amplitudes[1] / amplitudes[0] - 0.5).abs() < 0.02,
                    "{extension} gain {amplitudes:?}"
                );
                println!(
                    "{extension} fps={fps}: gain RMS ratio={}",
                    amplitudes[1] / amplitudes[0]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn offsets_cut_long_source_and_preserve_stereo() -> Result<()> {
        let root = Workspace::new()?;
        let source = root.path("stereo.wav");
        wav(&source, 2., 0.25, 2)?;
        for offset in [-0.125, -3., 2., 1e10, 0.] {
            let movie = root.path("offset.mov");
            video(&movie, 24, Time::integer(24))?;
            mux_file(&movie, &source, 1., offset, 24, Time::integer(24))?;
            let samples = decode(&movie)?;
            assert_eq!(
                samples.len(),
                48_000 * 2,
                "long input cuts exactly at video end"
            );
            if offset == -3. || offset >= 2. {
                assert!(
                    samples.iter().all(|x| *x == 0.),
                    "source outside the video span is silent"
                );
            } else {
                let detected = samples
                    .chunks_exact(2)
                    .position(|x| x[0].abs() > 0.005)
                    .unwrap();
                let expected = ((0.25 + offset) * 48_000.).round() as usize;
                assert!(
                    detected.abs_diff(expected) <= 1,
                    "PCM onset {detected}, expected {expected}"
                );
                let window = &samples[expected * 2 + 4800..expected * 2 + 9600];
                let left: Vec<_> = window.iter().step_by(2).copied().collect();
                let right: Vec<_> = window.iter().skip(1).step_by(2).copied().collect();
                assert!((rms(&right) / rms(&left) - 0.5).abs() < 0.001);
            }
        }
        Ok(())
    }

    #[test]
    fn unreadable_refuses_one_line_and_stills_never_open_audio() -> Result<()> {
        let root = Workspace::new()?;
        let movie = root.path("preserve.mp4");
        video(&movie, 1, Time::integer(24))?;
        let original = fs::read(&movie)?;
        let bad = root.path("unreadable.bin");
        fs::write(&bad, b"not audio")?;
        let error = mux_file(&movie, &bad, 1., 0., 1, Time::integer(24))
            .unwrap_err()
            .to_string();
        assert_eq!(error.lines().count(), 1);
        assert!(error.contains("cannot read audio") && error.contains("PCM WAV"));
        // Do not create control-character filenames (invalid on Windows), but
        // verify such an authored path still yields one escaped refusal line.
        let error = mux_file(
            &movie,
            &root.path("missing\nfile.wav"),
            1.,
            0.,
            1,
            Time::integer(24),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(error.lines().count(), 1);
        assert_eq!(fs::read(&movie)?, original);
        for name in [
            "still.png",
            "still.exr",
            "frames",
            "frames/",
            "frames.mov/",
            "frame_%06d.exr",
        ] {
            let path = root.path(name);
            // Deliberately invalid input and clocks prove the still/sequence
            // return happens before any audio work or filesystem access.
            mux_file(&path, &bad, f64::NAN, f64::INFINITY, 0, Time::ZERO)?;
            assert!(!path.exists());
        }
        for (gain, offset) in [(f64::NAN, 0.), (1., f64::INFINITY)] {
            assert!(mux_file(&movie, &bad, gain, offset, 1, Time::integer(24)).is_err());
            assert_eq!(fs::read(&movie)?, original);
        }
        assert_eq!(
            fs::read_dir(&root.0)?.count(),
            2,
            "no temporary output leaked"
        );
        Ok(())
    }

    #[test]
    fn external_decoder_formats_and_failed_mux_preserve_movie() -> Result<()> {
        let root = Workspace::new()?;
        let wav_source = root.path("tone.wav");
        wav(&wav_source, 0.2, 0., 1)?;
        for extension in ["mp3", "m4a", "flac"] {
            let source = root.path(&format!("tone.{extension}"));
            run(Command::new("ffmpeg")
                .args(["-v", "error", "-nostdin", "-y", "-i"])
                .arg(&wav_source)
                .arg(&source))?;
            let movie = root.path("decoded.mp4");
            video(&movie, 12, Time::integer(24))?;
            mux_file(&movie, &source, 1., 0., 12, Time::integer(24))?;
            assert!(
                rms(&decode(&movie)?[..4800]) > 0.1,
                "{extension} must decode"
            );
        }
        // A valid audio probe followed by a mux failure must also preserve the
        // original destination and remove the reserved temporary output.
        let broken = root.path("unfinished.mov");
        fs::write(&broken, b"unfinished video")?;
        let before = fs::read_dir(&root.0)?.count();
        let error = mux_file(&broken, &wav_source, 1., 0., 12, Time::integer(24)).unwrap_err();
        // The failed FFmpeg run is a device failure; failure.rs owns what exit code a category gets.
        assert_eq!(error.downcast_ref::<crate::failure::FailureKind>(), Some(&crate::failure::FailureKind::Device));
        let error = format!("{error:#}");
        assert_eq!(error.lines().count(), 1);
        assert!(error.contains("PCM WAV"));
        assert_eq!(fs::read(&broken)?, b"unfinished video");
        assert_eq!(fs::read_dir(&root.0)?.count(), before);
        Ok(())
    }

    #[test]
    fn frame_clock_rounds_once_and_refuses_unrepresentable_lengths() -> Result<()> {
        assert_eq!(sample_count(30, Time::new(30_000, 1001)?)?, 48_048);
        assert_eq!(sample_count(1, Time::new(24_000, 1001)?)?, 2002);
        assert_eq!(sample_count(1, Time::integer(29))?, 1655);
        assert!(sample_count(0, Time::ONE).is_err());
        assert!(sample_count(1, Time::ZERO).is_err());
        assert!(sample_count(u64::MAX, Time::new(1, i64::MAX)?).is_err());
        Ok(())
    }
}
