//! CPU sound samples and analysis. No rendering device or output-frame clock.
use anyhow::{Context, Result, ensure};
use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex},
};
pub const SAMPLE_RATE: u32 = 48_000;
pub type Frame = [f32; 2];

#[derive(Clone)]
pub struct Clip {
    pub rate: u32,
    pub frames: Vec<Frame>,
}

/// A sound file decoded front to back, only as far as a mix has needed it. One decoder runs
/// per source from the file's first frame and is kept where it stopped, so a longer need
/// continues it: no frame is decoded twice, and a prefix holds the very samples a whole decode
/// holds. Nothing seeks, so no codec or resampler has to land on the same sample grid twice.
pub struct Source {
    path: PathBuf,
    pub rate: u32,
    channels: u32,
    /// Sample frames in the file, when its header says: a WAV does, a decoder's stream does not.
    length: Option<usize>,
    decoded: Mutex<Decoded>,
}
struct Decoded {
    /// The frames decoded so far, from the first.
    clip: Arc<Clip>,
    /// The decoder, standing after the last frame of `clip`; none before the first read and
    /// after the end of the file.
    stream: Option<Stream>,
    whole: bool,
}
enum Stream {
    Wav(hound::WavReader<std::io::BufReader<std::fs::File>>),
    External(External),
}
/// The user's decoder, blocked on its output pipe between reads.
struct External {
    child: Child,
    stdout: ChildStdout,
    errors: Option<std::thread::JoinHandle<Vec<u8>>>,
}
impl Drop for External {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Source {
    fn wav(path: &Path) -> bool {
        path.extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("wav"))
    }
    fn new(path: &Path, rate: u32, channels: u32, length: Option<usize>) -> Self {
        Self {
            path: path.to_path_buf(),
            rate,
            channels,
            length,
            decoded: Mutex::new(Decoded {
                clip: Arc::new(Clip {
                    rate,
                    frames: Vec::new(),
                }),
                stream: None,
                whole: false,
            }),
        }
    }
    /// Read the file's layout and start its decoder: the first frame is decoded here, so a
    /// file that is missing, has no sound stream or does not decode is refused when it is
    /// opened, not when it is first mixed. The decoder then waits there for the first mix.
    pub fn open(path: &Path) -> Result<Self> {
        let source = Self::layout(path)?;
        source.prefix(1)?;
        Ok(source)
    }
    /// The file's rate, channels and, where its header says, length.
    fn layout(path: &Path) -> Result<Self> {
        if Self::wav(path) {
            let reader = hound::WavReader::open(path)
                .with_context(|| format!("audio: {}", path.display()))?;
            let spec = reader.spec();
            ensure!(
                matches!(spec.channels, 1 | 2),
                "audio: {} channels are not built yet; use mono or stereo",
                spec.channels
            );
            ensure!(spec.sample_rate > 0, "audio: sample rate must be positive");
            return Ok(Self::new(
                path,
                spec.sample_rate,
                spec.channels.into(),
                Some(reader.duration() as usize),
            ));
        }
        // The user's decoder is a separate executable, never linked or bundled.
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels",
                "-of",
                "csv=p=0",
            ])
            .arg(path)
            .output()
            .context("audio: install ffmpeg and ffprobe to decode this file")?;
        ensure!(
            probe.status.success(),
            "audio: cannot probe {}: {}",
            path.display(),
            String::from_utf8_lossy(&probe.stderr).trim()
        );
        let channels: u32 = String::from_utf8_lossy(&probe.stdout)
            .trim()
            .parse()
            .context("audio: file has no audio stream")?;
        ensure!(
            matches!(channels, 1 | 2),
            "audio: {channels} channels are not built yet; use mono or stereo"
        );
        Ok(Self::new(path, SAMPLE_RATE, channels, None))
    }
    /// The file's length in sample frames at `rate`. Only a decoder's stream end tells the
    /// length of a file without a WAV header, so that file is decoded to its end.
    pub fn frames(&self) -> Result<usize> {
        match self.length {
            Some(length) => Ok(length),
            None => Ok(self.prefix(usize::MAX)?.frames.len()),
        }
    }
    /// The file's first `frames` sample frames at least, or all of it when it is shorter.
    /// Decoding continues from where the last call stopped.
    pub fn prefix(&self, frames: usize) -> Result<Arc<Clip>> {
        let mut decoded = self.decoded.lock().unwrap();
        let decoded = &mut *decoded;
        let have = decoded.clip.frames.len();
        if decoded.whole || have >= frames {
            return Ok(decoded.clip.clone());
        }
        let wanted = frames - have;
        let more = match self.read(&mut decoded.stream, wanted) {
            Ok(more) => more,
            Err(error) => {
                // A failed decoder cannot continue: the next read starts the file again.
                decoded.stream = None;
                decoded.clip = Arc::new(Clip {
                    rate: self.rate,
                    frames: Vec::new(),
                });
                return Err(error);
            }
        };
        let ended = more.len() < wanted;
        // In place unless a mix still holds the shorter prefix.
        Arc::make_mut(&mut decoded.clip).frames.extend(more);
        if ended || self.length == Some(decoded.clip.frames.len()) {
            decoded.whole = true;
            decoded.stream = None;
        }
        Ok(decoded.clip.clone())
    }
    /// The next `wanted` frames of the stream, fewer only at the end of the file.
    fn read(&self, stream: &mut Option<Stream>, wanted: usize) -> Result<Vec<Frame>> {
        if stream.is_none() {
            *stream = Some(self.start()?);
        }
        match stream.as_mut().expect("started above") {
            Stream::Wav(reader) => {
                let spec = reader.spec();
                let channels = spec.channels as usize;
                let count = wanted.saturating_mul(channels);
                let samples = match spec.sample_format {
                    hound::SampleFormat::Float => reader
                        .samples::<f32>()
                        .take(count)
                        .collect::<Result<Vec<_>, _>>()?,
                    hound::SampleFormat::Int => {
                        let scale = 2_f64.powi(i32::from(spec.bits_per_sample) - 1);
                        reader
                            .samples::<i32>()
                            .take(count)
                            .map(|s| s.map(|x| (f64::from(x) / scale) as f32))
                            .collect::<Result<Vec<_>, _>>()?
                    }
                };
                ensure!(
                    samples.iter().all(|x| x.is_finite()),
                    "audio: nonfinite source sample"
                );
                Ok(samples
                    .chunks_exact(channels)
                    .map(|c| [c[0], c[channels - 1]])
                    .collect())
            }
            Stream::External(decoder) => {
                // The decoder writes the same stream however it is read: it waits on its
                // pipe after each read and is stopped when the source is dropped.
                let limit = (wanted as u64).saturating_mul(8);
                let mut bytes = Vec::new();
                (&mut decoder.stdout)
                    .take(limit)
                    .read_to_end(&mut bytes)?;
                if (bytes.len() as u64) < limit {
                    let status = decoder.child.wait()?;
                    let errors = decoder
                        .errors
                        .take()
                        .map(|errors| errors.join().unwrap_or_default())
                        .unwrap_or_default();
                    ensure!(
                        status.success(),
                        "audio: cannot decode {}: {}",
                        self.path.display(),
                        String::from_utf8_lossy(&errors).trim()
                    );
                    ensure!(
                        bytes.len() % 8 == 0,
                        "audio: incomplete decoded sample frame"
                    );
                }
                let frames: Vec<Frame> = bytes
                    .chunks_exact(8)
                    .map(|b| {
                        [
                            f32::from_le_bytes(b[..4].try_into().unwrap()),
                            f32::from_le_bytes(b[4..].try_into().unwrap()),
                        ]
                    })
                    .collect();
                ensure!(
                    frames.iter().flatten().all(|x| x.is_finite()),
                    "audio: nonfinite decoded sample"
                );
                Ok(frames)
            }
        }
    }
    /// Open the decoder at the file's first frame.
    fn start(&self) -> Result<Stream> {
        if Self::wav(&self.path) {
            let reader = hound::WavReader::open(&self.path)
                .with_context(|| format!("audio: {}", self.path.display()))?;
            let spec = reader.spec();
            ensure!(
                u32::from(spec.channels) == self.channels && spec.sample_rate == self.rate,
                "audio: {} changed while it was open",
                self.path.display()
            );
            return Ok(Stream::Wav(reader));
        }
        let mut decoder = Command::new("ffmpeg");
        decoder
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(&self.path);
        if self.channels == 1 {
            decoder.args(["-af", "pan=stereo|c0=c0|c1=c0"]);
        }
        let mut child = decoder
            .args([
                "-map", "0:a:0", "-f", "f32le", "-ac", "2", "-ar", "48000", "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("audio: install ffmpeg to decode this file")?;
        let stdout = child.stdout.take().expect("piped decoder output");
        let mut stderr = child.stderr.take().expect("piped decoder errors");
        // Drained beside the samples so a talkative decoder never blocks on its error pipe.
        let errors = std::thread::spawn(move || {
            let mut text = Vec::new();
            let _ = stderr.read_to_end(&mut text);
            text
        });
        Ok(Stream::External(External {
            child,
            stdout,
            errors: Some(errors),
        }))
    }
}

impl Clip {
    /// Decode a whole file.
    pub fn load(path: &Path) -> Result<Self> {
        let source = Source::open(path)?;
        let clip = source.prefix(usize::MAX)?;
        drop(source);
        Ok(Arc::try_unwrap(clip).unwrap_or_else(|clip| (*clip).clone()))
    }
    pub fn duration(&self) -> f64 {
        self.frames.len() as f64 / self.rate as f64
    }

    /// How many frames from the start of a source at `rate` `sample(seconds, speed)` can read:
    /// the frame under `seconds`, the reconstruction window after it and two frames of slack.
    pub fn reach(rate: u32, seconds: f64, speed: f64) -> usize {
        if seconds.is_nan() || seconds < 0. {
            return 0;
        }
        let radius = (16. / Self::cutoff(rate, speed)).ceil();
        (seconds * rate as f64 + radius + 2.) as usize
    }
    fn cutoff(rate: u32, speed: f64) -> f64 {
        (SAMPLE_RATE as f64 / (rate as f64 * speed.abs())).min(1.)
    }

    /// Windowed sinc reconstruction, with a rate-dependent antialias cutoff.
    pub fn sample(&self, seconds: f64, speed: f64) -> Frame {
        if seconds < 0. || seconds >= self.duration() || speed == 0. {
            return [0.; 2];
        }
        let position = seconds * self.rate as f64;
        if self.rate == SAMPLE_RATE
            && speed.abs() == 1.
            && (position - position.round()).abs() < 1e-7
        {
            return self
                .frames
                .get(position.round() as usize)
                .copied()
                .unwrap_or([0.; 2]);
        }
        let cutoff = Self::cutoff(self.rate, speed);
        let radius = (16. / cutoff).ceil() as i64;
        let center = position.floor() as i64;
        let mut sum = [0.; 2];
        let mut weight = 0.;
        for i in center - radius + 1..=center + radius {
            let x = (i as f64 - position) * cutoff;
            if x.abs() >= 16. {
                continue;
            }
            let sinc = if x.abs() < 1e-12 {
                1.
            } else {
                (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
            };
            let w = cutoff * sinc * (0.5 + 0.5 * (std::f64::consts::PI * x / 16.).cos());
            weight += w;
            if let Some(frame) = usize::try_from(i).ok().and_then(|i| self.frames.get(i)) {
                for c in 0..2 {
                    sum[c] += frame[c] as f64 * w;
                }
            }
        }
        sum.map(|x| (x / weight) as f32)
    }
}

pub fn write_wav(path: &Path, frames: &[Frame]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 2,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for &sample in frames.iter().flatten() {
        ensure!(sample.is_finite(), "audio: nonfinite mix sample");
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

pub struct Loudness {
    pub rms: f64,
    pub peak: f64,
}
pub fn loudness(frames: &[Frame]) -> Loudness {
    let mut energy = 0.;
    let mut peak: f64 = 0.;
    for &x in frames.iter().flatten() {
        energy += f64::from(x).powi(2);
        peak = peak.max(f64::from(x).abs());
    }
    Loudness {
        rms: if frames.is_empty() {
            0.
        } else {
            (energy / (frames.len() * 2) as f64).sqrt()
        },
        peak,
    }
}

/// Host analysis convention: Hann (N-1), mono average, 2/sum(window).
/// No temporal smoothing: the caller explicitly chooses the analysis window.
pub fn spectrum(frames: &[Frame]) -> Result<Vec<f64>> {
    use rustfft::{FftPlanner, num_complex::Complex};
    let n = frames.len();
    ensure!(
        n >= 4 && n.is_power_of_two() && n <= 65536,
        "audio spectrum: window must contain a power of two sample frames, 4..65536"
    );
    let mut window_sum = 0.;
    let mut input: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let w = 0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
            window_sum += w;
            Complex::new((f64::from(f[0]) + f64::from(f[1])) * 0.5 * w, 0.)
        })
        .collect();
    ensure!(
        window_sum > 0.,
        "audio spectrum: Hann window needs at least four samples"
    );
    FftPlanner::new().plan_fft_forward(n).process(&mut input);
    Ok(input[..=n / 2]
        .iter()
        .enumerate()
        .map(|(i, x)| x.norm() / window_sum * if i == 0 || i == n / 2 { 1. } else { 2. })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tone(hz: f64, rate: u32, n: usize) -> Clip {
        Clip {
            rate,
            frames: (0..n)
                .map(|i| {
                    [(2. * std::f64::consts::PI * hz * i as f64 / rate as f64).sin() as f32; 2]
                })
                .collect(),
        }
    }
    #[test]
    fn spectrum_and_loudness_measure_a_known_signal() {
        let clip = tone(1500., SAMPLE_RATE, 4096);
        let s = spectrum(&clip.frames).unwrap();
        assert!((s[128] - 1.).abs() < 1e-5);
        let level = loudness(&clip.frames);
        assert!((level.rms - 0.5_f64.sqrt()).abs() < 1e-6);
        assert!((level.peak - 1.).abs() < 1e-6);
        let cancelled: Vec<_> = clip.frames.iter().map(|x| [x[0], -x[0]]).collect();
        assert!(spectrum(&cancelled).unwrap().iter().all(|x| *x == 0.));
    }
    #[test]
    fn resampling_changes_pitch_and_suppresses_aliases() {
        let low = tone(3000., SAMPLE_RATE, 48000);
        let high = tone(18000., SAMPLE_RATE, 48000);
        let low_samples: Vec<_> = (1000..5096)
            .map(|i| low.sample(2. * i as f64 / SAMPLE_RATE as f64, 2.))
            .collect();
        let high_samples: Vec<_> = (1000..5096)
            .map(|i| high.sample(2. * i as f64 / SAMPLE_RATE as f64, 2.))
            .collect();
        let spectrum = spectrum(&low_samples).unwrap();
        assert!((spectrum[512] - 1.).abs() < 0.001);
        assert!(loudness(&high_samples).rms < 0.001);
    }
    #[test]
    #[ignore = "requires user-installed ffmpeg and ffprobe"]
    fn external_decoder_preserves_mono_amplitude() {
        let path = std::env::temp_dir().join(format!("vork-audio-mono-{}.bin", std::process::id()));
        let mut wav = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: SAMPLE_RATE,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for _ in 0..480 {
            wav.write_sample(0.25_f32).unwrap();
        }
        wav.finalize().unwrap();
        let decoded = Clip::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(decoded.frames.len(), 480);
        assert!(decoded.frames.iter().all(|f| *f == [0.25; 2]));
    }
    #[test]
    fn a_source_decodes_each_frame_once_as_far_as_it_is_needed() {
        let path =
            std::env::temp_dir().join(format!("vork-audio-source-{}.wav", std::process::id()));
        let frames: Vec<Frame> = (0..1000).map(|i| [i as f32, -(i as f32)]).collect();
        write_wav(&path, &frames).unwrap();
        let source = Source::open(&path).unwrap();
        assert_eq!((source.rate, source.frames().unwrap()), (SAMPLE_RATE, 1000));
        // Opening decoded the first frame and left the decoder there.
        assert_eq!(source.decoded.lock().unwrap().clip.frames, frames[..1]);
        assert!(source.decoded.lock().unwrap().stream.is_some());
        let prefix = source.prefix(100).unwrap();
        assert_eq!(prefix.frames, frames[..100]);
        assert!(Arc::ptr_eq(&prefix, &source.prefix(40).unwrap()));
        // A longer need continues the same decoder; a mix holding the shorter prefix keeps it.
        assert_eq!(source.prefix(101).unwrap().frames, frames[..101]);
        assert_eq!(prefix.frames.len(), 100);
        assert!(source.decoded.lock().unwrap().stream.is_some());
        let whole = source.prefix(usize::MAX).unwrap();
        assert_eq!(whole.frames, frames);
        assert!(source.decoded.lock().unwrap().stream.is_none());
        assert!(Arc::ptr_eq(&whole, &source.prefix(5000).unwrap()));
        drop(source);
        assert_eq!(Clip::load(&path).unwrap().frames, frames);
        std::fs::remove_file(path).unwrap();
        // Every frame a sample reads lies inside its reach.
        for (rate, speed) in [(SAMPLE_RATE, 1.), (44_100, 1.), (SAMPLE_RATE, -3.5), (8_000, 0.25)] {
            for seconds in [0., 0.01, 0.5] {
                let reach = Clip::reach(rate, seconds, speed);
                let clip = Clip {
                    rate,
                    frames: vec![[1.; 2]; reach + 64],
                };
                let mut short = clip.clone();
                short.frames.truncate(reach);
                short.frames.extend([[9.; 2]; 64]);
                assert_eq!(clip.sample(seconds, speed), short.sample(seconds, speed));
            }
        }
    }
    #[test]
    fn a_file_that_does_not_decode_is_refused_when_it_is_opened() {
        let path =
            std::env::temp_dir().join(format!("vork-audio-broken-{}.wav", std::process::id()));
        let frames = vec![[0.5, -0.5]; 3];
        let cut = |frames_kept: u64| {
            write_wav(&path, &frames).unwrap();
            let length = std::fs::metadata(&path).unwrap().len();
            let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.set_len(length - (3 - frames_kept) * 8).unwrap();
        };
        // The header promises three frames and none follow.
        cut(0);
        assert!(Source::open(&path).is_err());
        // Open reads the layout and the first frame, not the file: damage further in is
        // reported by the mix that reaches it.
        cut(1);
        let source = Source::open(&path).unwrap();
        assert!(source.prefix(3).is_err());
        // A header alone is not a sound file.
        std::fs::write(&path, b"RIFF").unwrap();
        assert!(Source::open(&path).is_err());
        std::fs::remove_file(&path).unwrap();
        assert!(Source::open(&path).is_err());
    }
    #[test]
    #[ignore = "requires user-installed ffmpeg and ffprobe"]
    fn an_undecodable_compressed_file_is_refused_when_it_is_opened() {
        let path =
            std::env::temp_dir().join(format!("vork-audio-broken-{}.mp3", std::process::id()));
        std::fs::write(&path, vec![0x55; 4096]).unwrap();
        assert!(Source::open(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn wav_round_trip_retains_unclamped_stereo() {
        let path = std::env::temp_dir().join(format!("vork-audio-test-{}.wav", std::process::id()));
        let frames = vec![[2., -3.], [0.125, -0.25], [0., 0.]];
        write_wav(&path, &frames).unwrap();
        let clip = Clip::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(clip.frames, frames);
        assert_eq!(clip.rate, SAMPLE_RATE);
    }
}
