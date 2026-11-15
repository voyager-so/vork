//! Unstable, internal: all Rust APIs in this crate are free to change.
//!
//! One ordered encoder per output. Input textures must be submitted on the supplied queue,
//! single-sampled RGBA8/BGRA8, with COPY_SRC usage. Indices begin at zero.
// Windows has only the portable pipe sink.
#![cfg_attr(
    not(any(target_os = "linux", target_os = "macos")),
    allow(irrefutable_let_patterns)
)]
mod constants;
mod ffmpeg;
mod rate;
pub use rate::validate_video_rate;
pub use ffmpeg::{probe_video_clock_capabilities, validate_video_clock, validate_video_clock_capabilities};
pub mod gpu_memory;
mod pipe;
pub use vork_time::Time;
use anyhow::{Context, Result, ensure};
use constants::*;
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

/// A frame beyond a working device's limits (not a setup failure).
#[derive(Debug)]
pub struct DeviceLimit(pub String);
impl std::fmt::Display for DeviceLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DeviceLimit {}

/// Check external video dependencies before device startup or scene installation.
/// Native macOS video does not need FFmpeg. PNG/EXR callers need no check.
/// Returns the dependency report for the persistent timing-capability identity.
pub fn validate_video_output(path: &Path) -> Result<Vec<u8>> {
    if cfg!(target_os = "macos") { return Ok(Vec::new()); }
    let mov = path.extension().is_some_and(|s| s.eq_ignore_ascii_case("mov"));
    if mov { return require_ffmpeg_encoder("prores_ks"); }
    // Linux native NVENC still requires FFmpeg as a muxer. Windows needs a
    // usable software fallback even if its FFmpeg advertises NVENC.
    if cfg!(target_os = "windows") { return require_ffmpeg_encoder("libx264"); }
    let output = Command::new("ffmpeg").arg("-version").stdin(Stdio::null()).output()
        .with_context(|| format!("video output needs ffmpeg on PATH: {FFMPEG_REQUIREMENT}"))?;
    ensure!(output.status.success(), "ffmpeg on PATH is unusable: {FFMPEG_REQUIREMENT}");
    let mut identity = output.stdout;
    identity.extend_from_slice(&output.stderr);
    Ok(identity)
}

/// The colour primaries of the display-encoded frames an encoder is given, which its stream names. Every
/// backend converts with the BT.709 matrix to limited range and tags the BT.709 transfer, as for sRGB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Primaries {
    /// sRGB pictures: BT.709 primaries.
    #[default]
    Bt709,
    /// Display P3 pictures: P3 primaries at D65 (SMPTE EG 432-1).
    DisplayP3,
}
impl Primaries {
    /// The ITU-T H.273 ColourPrimaries code, as H.264 VUI carries it.
    pub fn h273(self) -> u32 {
        match self { Self::Bt709 => 1, Self::DisplayP3 => 12 }
    }
    /// FFmpeg's name for the primaries.
    pub fn ffmpeg(self) -> &'static str {
        match self { Self::Bt709 => "bt709", Self::DisplayP3 => "smpte432" }
    }
}
/// H.264 output policy, translated by each backend without changing defaults.
#[derive(Clone, Copy, Debug)]
pub struct EncodeSettings {
    /// The primaries of the frames' display; the stream is tagged with them.
    pub primaries: Primaries,
    /// Emit encoder stage timings when CLI diagnostics are enabled.
    pub diagnostics: bool,
    /// Hardware bitrate target in bits per second (NVENC/VideoToolbox).
    pub hardware_bitrate: u32,
    /// Maximum keyframe spacing in seconds, shared by hardware and software.
    pub gop_seconds: u32,
    /// Software libx264 quality: 0 (lossless) through 51 (lowest quality).
    pub software_crf: u8,
}
impl Default for EncodeSettings {
    fn default() -> Self {
        Self {
            primaries: Primaries::default(),
            diagnostics: false,
            hardware_bitrate: HARDWARE_BITRATE,
            gop_seconds: GOP_SECONDS,
            software_crf: SOFTWARE_CRF,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Auto,
    Readback,
    HardwareReadback,
}
/// Handles belong to one device. Auto uses native Metal on macOS, Vulkan/CUDA NVENC on NVIDIA
/// Linux adapters, ffmpeg NVENC after readback on Windows, and portable readback (libx264 at
/// its quality target) on other Linux adapters and when hardware encoding is unavailable.
pub struct DeviceInfo {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub backend: Backend,
}
pub struct Encoder {
    gpu: DeviceInfo,
    width: u32,
    height: u32,
    path: PathBuf,
    next: u64,
    sink: Sink,
    closed: bool,
    staged_output: Option<tempfile::TempPath>,
    #[cfg(target_os = "linux")]
    diagnostics: bool,
}
/// Sink resources owned by one graph ring entry, never by the encoder queue.
/// Install once, record its copy after rendering, then push and release in order.
pub struct Surface {
    pub texture: Option<wgpu::Texture>,
    readback: Option<wgpu::Buffer>,
    pixels: Vec<u8>,
    returned: std::sync::mpsc::Sender<std::result::Result<Vec<u8>, String>>,
    completed: std::sync::Mutex<std::sync::mpsc::Receiver<std::result::Result<Vec<u8>, String>>>,
    in_flight: bool,
    pipe_wait_ms: f64,
    submitted: Option<u64>,
    #[cfg(target_os = "linux")]
    nvidia: Option<nvidia::Surface>,
}
impl Surface {
    /// Host wall time reclaiming the previous pipe payload in the last push.
    pub fn pipe_wait_ms(&self) -> f64 {
        self.pipe_wait_ms
    }

    /// Record the portable sink's copy in the frame's command buffer.
    pub fn record_copy(&self, encoder: &mut wgpu::CommandEncoder, source: &wgpu::Texture) {
        if let Some(buffer) = &self.readback {
            let stride = (source.width() * 4).div_ceil(COPY_ROW_ALIGNMENT as u32)
                * COPY_ROW_ALIGNMENT as u32;
            encoder.copy_texture_to_buffer(
                source.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(stride),
                        rows_per_image: None,
                    },
                },
                source.size(),
            );
        }
    }
}
enum Sink {
    #[cfg(target_os = "linux")]
    Nvidia(nvidia::Session),
    Pipe {
        child: Child,
        input: Option<pipe::Writer>,
        stderr: Option<pipe::Diagnostics>,
    },
    #[cfg(target_os = "macos")]
    Metal(native::Session),
}
/// Device-independent output preparation. Native codec/driver initialization
/// overlaps adapter discovery and GPU device creation.
/// Dropping an unfinished preparation cancels and reclaims its native session.
pub struct PreparedEncoder {
    backend: Backend,
    width: u32,
    height: u32,
    fps: Time,
    settings: EncodeSettings,
    path: PathBuf,
    gop_frames: u32,
    fallback_codec: std::sync::OnceLock<&'static str>,
    #[cfg(target_os = "macos")]
    metal: Option<native::Session>,
    #[cfg(target_os = "linux")]
    nvidia: Option<nvidia::Prepared>,
}
impl PreparedEncoder {
    /// Validate the selected FFmpeg codec and clock before replacing existing output.
    /// Native sessions do not need a readback codec unless attachment fails.
    /// Supply the rendering device for Linux hardware selection; software-only
    /// preparation and native macOS preparation need no device.
    pub fn validate_fallback(&self, frames: u64, device: Option<&wgpu::Device>) -> Result<()> {
        if let Some(device) = device { validate_device(device, self.width, self.height)?; }
        #[cfg(target_os = "macos")]
        if self.metal.is_some() { return Ok(()); }
        #[cfg(target_os = "linux")]
        if self.nvidia.is_some() && device.is_none_or(nvidia::is_adapter) { return ffmpeg::validate_mux_rate(self.fps); }
        if self.fallback_codec.get().is_none() {
            let codec = readback_codec(self.backend, self.width, self.height, &self.path, device)?;
            let _ = self.fallback_codec.set(codec);
        }
        ffmpeg::validate_fallback_clock(self.fps, frames, &self.path)
    }
    /// Attach the prepared output to the device used for rendering its frames.
    /// `frames` is the selected clip length, used to preflight FFmpeg movie timing.
    pub fn open(
        self,
        device: wgpu::Device,
        queue: wgpu::Queue,
        frames: u64,
        ring_depth: usize,
    ) -> Result<Encoder> {
        let Self {
            backend,
            width,
            height,
            fps,
            settings,
            path,
            gop_frames,
            fallback_codec,
            #[cfg(target_os = "macos")]
            metal,
            #[cfg(target_os = "linux")]
            nvidia,
        } = self;
        let gpu = DeviceInfo {
            device,
            queue,
            backend,
        };
        ensure!(
            ring_depth > 0 && ring_depth <= u32::MAX as usize,
            "invalid ring depth"
        );
        validate_device(&gpu.device, width, height)?;
        let prores = path
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("mov"));
        ensure!(
            !(prores && matches!(gpu.backend, Backend::HardwareReadback)),
            "HardwareReadback is not supported for MOV; use Auto for native Mac ProRes or Readback for software ProRes"
        );
        #[cfg(target_os = "linux")]
        if let Some(prepared) = nvidia.filter(|_| nvidia::is_adapter(&gpu.device)) {
            match nvidia::Session::open(
                &gpu,
                prepared,
                width,
                height,
                fps,
                settings.hardware_bitrate,
                gop_frames,
                settings.primaries.h273(),
                &path,
                ring_depth,
            ) {
                Ok(session) => {
                    eprintln!("Video encoder: native NVENC (GPU); FFmpeg muxing");
                    return Ok(Encoder {
                        gpu,
                        width,
                        height,
                        path,
                        next: 0,
                        sink: Sink::Nvidia(session),
                        closed: false,
                        staged_output: None,
                        #[cfg(target_os = "linux")]
                        diagnostics: settings.diagnostics,
                    });
                }
                Err(error) => eprintln!("GPU NVENC unavailable; using readback: {error:#}"),
            }
        }
        #[cfg(target_os = "macos")]
        if let Some(mut session) = metal {
            match session.attach(&path) {
                Ok(()) => {
                    return Ok(Encoder {
                        gpu,
                        width,
                        height,
                        path,
                        next: 0,
                        sink: Sink::Metal(session),
                        closed: false,
                        staged_output: None,
                        #[cfg(target_os = "linux")]
                        diagnostics: settings.diagnostics,
                    });
                }
                Err(error) => {
                    eprintln!("VideoToolbox output unavailable; using software encoding: {error:#}")
                }
            }
        }
        rate::validate_ffmpeg_rate(fps, &path)?;
        let codec = match fallback_codec.into_inner() {
            Some(codec) => codec,
            None => readback_codec(gpu.backend, width, height, &path, Some(&gpu.device))?,
        };
        eprintln!("Video encoder: FFmpeg {codec} (readback)");
        let mut staging = tempfile::Builder::new();
        staging.prefix(".vork-video-").suffix(if prores { ".mov" } else { ".mp4" });
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // Match normal FFmpeg output creation, including the caller's umask.
            staging.permissions(std::fs::Permissions::from_mode(0o666));
        }
        let staged_output = staging
            .tempfile_in(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))
            .with_context(|| format!("prepare video output {}", path.display()))?.into_temp_path();
        let mut child = ffmpeg::readback_command(width, height, fps, frames, settings, codec, gop_frames, &staged_output)?
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("start ffmpeg from PATH: {FFMPEG_REQUIREMENT}"))?;
        let input = match pipe::Writer::new(child.stdin.take().context("missing ffmpeg stdin")?) {
            Ok(writer) => Some(writer),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error).context("start encoder pipe writer");
            }
        };
        let stderr =
            match pipe::Diagnostics::new(child.stderr.take().context("missing ffmpeg stderr")?) {
                Ok(stderr) => Some(stderr),
                Err(error) => {
                    let _ = child.kill();
                    drop(input);
                    let _ = child.wait();
                    return Err(error).context("start ffmpeg stderr reader");
                }
            };
        Ok(Encoder {
            gpu,
            width,
            height,
            path,
            next: 0,
            sink: Sink::Pipe {
                child,
                input,
                stderr,
            },
            closed: false,
            staged_output: Some(staged_output),
                        #[cfg(target_os = "linux")]
                        diagnostics: settings.diagnostics,
        })
    }
}
/// Test FFmpeg's NVENC against the installed driver, using the same initialization
/// probe for setup and output selection. Native Linux NVENC has a separate ABI.
pub fn probe_ffmpeg_nvenc(width: u32, height: u32) -> Result<()> {
    // Modern NVENC rejects tiny frames; preserve small output dimensions so Auto
    // can select software for those outputs rather than accepting a larger probe.
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-f", "lavfi", "-i",
            &format!("color=size={}x{}", width.min(NVENC_PROBE_EXTENT), height.min(NVENC_PROBE_EXTENT)),
            "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"])
        .stdin(Stdio::null()).output().context("start FFmpeg NVENC probe")?;
    ensure!(output.status.success(), "FFmpeg h264_nvenc initialization failed ({}): {}",
        output.status, String::from_utf8_lossy(&output.stderr).trim());
    Ok(())
}

fn readback_codec(backend: Backend, width: u32, height: u32, path: &Path, device: Option<&wgpu::Device>) -> Result<&'static str> {
    #[cfg(not(target_os = "linux"))]
    let _ = device;
    #[cfg(target_os = "macos")]
    let _ = (width, height);
    let prores = path.extension().is_some_and(|s| s.eq_ignore_ascii_case("mov"));
    #[allow(unused_mut)]
    let mut codec = if prores { "prores_ks" } else { "libx264" };
    #[cfg(target_os = "macos")]
    if !prores && matches!(backend, Backend::HardwareReadback) {
        codec = "h264_videotoolbox";
    }
    // FFmpeg hardware H.264 follows the rendering adapter and encodes at a fixed bitrate. Only
    // NVIDIA has it; every other adapter's MP4 is libx264 at its quality target (CRF).
    #[cfg(target_os = "linux")]
    if !prores && !matches!(backend, Backend::Readback) {
        if nvidia::is_adapter(device.context("hardware H.264 selection requires the rendering device")?) {
            match probe_ffmpeg_nvenc(width, height) {
                // The probe initialised the encoder; listing FFmpeg's encoders again only costs a process.
                Ok(()) => return Ok("h264_nvenc"),
                Err(error) => {
                    ensure!(!matches!(backend, Backend::HardwareReadback), "hardware H.264 unavailable: {error:#}");
                    eprintln!("Hardware H.264 unavailable; using software libx264: {error:#}");
                }
            }
        } else {
            ensure!(
                !matches!(backend, Backend::HardwareReadback),
                "hardware H.264 unavailable: Linux hardware readback needs an NVIDIA rendering adapter"
            );
        }
    }
    #[cfg(windows)]
    if !prores && !matches!(backend, Backend::Readback) {
        match probe_ffmpeg_nvenc(width, height) {
            Ok(()) => codec = "h264_nvenc",
            Err(error) => {
                ensure!(!matches!(backend, Backend::HardwareReadback), "{error:#}");
                eprintln!("FFmpeg NVENC unavailable; using software libx264: {error:#}");
            }
        }
    }
    require_ffmpeg_encoder(codec)?;
    Ok(codec)
}

// Check before opening the rawvideo pipe so codec errors retain their cause.
fn require_ffmpeg_encoder(codec: &str) -> Result<Vec<u8>> {
    let output = Command::new("ffmpeg")
        .arg("-encoders")
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("query ffmpeg encoders: {FFMPEG_REQUIREMENT}"))?;
    validate_ffmpeg_encoder(output.status.success(), &output.stdout, codec)?;
    // The banner includes the runtime library versions as well as the build.
    let mut identity = output.stderr;
    identity.extend_from_slice(&output.stdout);
    Ok(identity)
}

fn validate_ffmpeg_encoder(success: bool, listing: &[u8], codec: &str) -> Result<()> {
    ensure!(
        success,
        "query ffmpeg encoders failed: {FFMPEG_REQUIREMENT}"
    );
    let listing = String::from_utf8_lossy(listing);
    ensure!(
        listing.lines().any(|line| {
            let mut fields = line.split_whitespace();
            fields.next().is_some_and(|flags| flags.starts_with('V'))
                && fields.next() == Some(codec)
        }),
        "ffmpeg encoder {codec} unavailable: {FFMPEG_REQUIREMENT}"
    );
    Ok(())
}

#[cfg(test)]
mod encoder_capability_tests {
    use super::*;

    #[test]
    fn checks_the_selected_video_encoder_before_streaming() {
        let listing = b"Encoders:\r\n V..... = Video\r\n V....D h264_nvenc NVIDIA NVENC H.264 encoder\r\n V....D libx264 H.264 encoder\r\n VFS... prores_ks Apple ProRes\r\n";
        for codec in ["h264_nvenc", "libx264", "prores_ks"] {
            validate_ffmpeg_encoder(true, listing, codec).unwrap();
        }
        for (success, listing, codec) in [
            (
                true,
                b" V....D h264_nvenc NVIDIA NVENC\n".as_slice(),
                "libx264",
            ),
            (true, b" V....D libx264 H.264\n".as_slice(), "prores_ks"),
            (
                true,
                b" A..... libx264 wrong media type\n".as_slice(),
                "libx264",
            ),
            (
                true,
                b" V....D libx264_extra wrong name\n".as_slice(),
                "libx264",
            ),
            (false, listing.as_slice(), "libx264"),
        ] {
            let error = validate_ffmpeg_encoder(success, listing, codec).unwrap_err();
            assert!(error.to_string().contains(FFMPEG_REQUIREMENT), "{error:#}");
        }
    }
}

fn validate_device(device: &wgpu::Device, width: u32, height: u32) -> Result<()> {
    validate_device_limits(&device.limits(), width, height)
}

fn validate_device_limits(limits: &wgpu::Limits, width: u32, height: u32) -> Result<()> {
    ensure!(
        width <= limits.max_texture_dimension_2d && height <= limits.max_texture_dimension_2d,
        gpu_memory::device_limit("encoder texture dimensions", format_args!("{width}x{height}"), format_args!("max_texture_dimension_2d={}", limits.max_texture_dimension_2d), "reduce the output resolution or use a GPU with larger texture limits")
    );
    let padded = (width as u64 * 4).div_ceil(COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
    ensure!(
        padded * height as u64 <= limits.max_buffer_size,
        gpu_memory::device_limit("encoder readback", format_args!("{} bytes at {width}x{height}", padded * height as u64), format_args!("max_buffer_size={} bytes", limits.max_buffer_size), "reduce the output resolution or use a GPU with a larger buffer limit")
    );
    Ok(())
}
impl Encoder {
    /// Open an output for a selected clip of `frames` frames.
    pub fn open(
        gpu: DeviceInfo,
        width: u32,
        height: u32,
        fps: Time,
        settings: EncodeSettings,
        path: impl AsRef<Path>,
        frames: u64,
        ring_depth: usize,
    ) -> Result<Self> {
        Self::prepare(gpu.backend, width, height, fps, settings, path)?
            .open(gpu.device, gpu.queue, frames, ring_depth)
    }
    /// Prepare CPU/native codec resources without creating or truncating output.
    /// Device validation and file creation happen in PreparedEncoder::open.
    pub fn prepare(
        backend: Backend,
        width: u32,
        height: u32,
        fps: Time,
        settings: EncodeSettings,
        path: impl AsRef<Path>,
    ) -> Result<PreparedEncoder> {
        ensure!(
            width > 0
                && height > 0
                && width % 2 == 0
                && height % 2 == 0,
            "dimensions must be positive/even"
        );
        validate_video_rate(fps, path.as_ref())?;
        if !matches!(backend, Backend::Auto) {
            rate::validate_ffmpeg_rate(fps, path.as_ref())?;
        }
        let gop_frames = u32::try_from((vork_time::exact::decimal(&fps.to_string())? * vork_time::exact::Rational::from_integer(settings.gop_seconds.into())).ceil().to_integer())?;
        ensure!(gop_frames > 0 && gop_frames <= i32::MAX as u32, "keyframe interval must fit a positive signed 32-bit frame count");
        ensure!(
            settings.hardware_bitrate > 0 && settings.software_crf <= MAX_SOFTWARE_CRF,
            "bitrate must be positive; software CRF must be 0..51"
        );
        let path = path.as_ref().to_owned();
        #[cfg(target_os = "linux")]
        {
            let output = Command::new("ffmpeg").arg("-version").output()
                .context("Linux video output needs ffmpeg on PATH, including MP4 with native NVENC (ffmpeg muxes H.264). Install it with: sudo apt install ffmpeg")?;
            ensure!(output.status.success(), "ffmpeg on PATH is unusable; install ffmpeg (sudo apt install ffmpeg). Linux MP4 needs ffmpeg even with NVENC");
        }
        #[cfg(target_os = "linux")]
        let nvidia = if matches!(backend, Backend::Auto)
            && !path.extension().is_some_and(|s| s.eq_ignore_ascii_case("mov"))
        {
            match nvidia::Prepared::new() {
                Ok(prepared) => Some(prepared),
                Err(error) => {
                    eprintln!("GPU NVENC unavailable; using readback: {error:#}");
                    None
                }
            }
        } else {
            None
        };
        #[cfg(target_os = "macos")]
        let metal = if matches!(backend, Backend::Auto) {
            match native::Session::prepare(
                width,
                height,
                fps,
                settings.hardware_bitrate,
                gop_frames,
                path.extension()
                    .is_some_and(|s| s.eq_ignore_ascii_case("mov")),
                settings.primaries,
            ) {
                Ok(session) => Some(session),
                Err(error) => {
                    eprintln!("VideoToolbox unavailable; using software encoding: {error:#}");
                    None
                }
            }
        } else {
            None
        };
        Ok(PreparedEncoder {
            backend,
            width,
            height,
            fps,
            settings,
            path,
            gop_frames,
            fallback_codec: std::sync::OnceLock::new(),
            #[cfg(target_os = "macos")]
            metal,
            #[cfg(target_os = "linux")]
            nvidia,
        })
    }
    /// Consume the next ring entry in frame order. The caller owns sink staging;
    /// `submission` is the render submission that recorded this surface's copy.
    pub fn push(
        &mut self,
        index: u64,
        texture: wgpu::Texture,
        surface: &mut Surface,
        submission: &wgpu::SubmissionIndex,
    ) -> Result<()> {
        ensure!(!self.closed, "encoder closed");
        ensure!(
            index == self.next,
            "expected frame {}, got {index}",
            self.next
        );
        ensure!(
            texture.width() == self.width
                && texture.height() == self.height
                && texture.depth_or_array_layers() == 1
                && texture.sample_count() == 1
                && texture.dimension() == wgpu::TextureDimension::D2,
            "texture dimensions/sample count mismatch"
        );
        ensure!(
            matches!(
                texture.format(),
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            ),
            "expected RGBA8Unorm/BGRA8Unorm texture"
        );
        ensure!(
            texture.usage().contains(wgpu::TextureUsages::COPY_SRC),
            "COPY_SRC required"
        );
        ensure!(
            surface.submitted.is_none(),
            "release the ring entry before resubmitting it"
        );
        if let Err(error) = self.feed(&texture, surface, submission) {
            self.closed = true;
            return Err(error);
        }
        surface.submitted = Some(index);
        self.next += 1;
        Ok(())
    }
    pub fn uses_pixel_readback(&self) -> bool {
        match &self.sink {
            #[cfg(target_os = "linux")]
            Sink::Nvidia(_) => false,
            Sink::Pipe { .. } => true,
            #[cfg(target_os = "macos")]
            Sink::Metal(_) => false,
        }
    }
    /// Acquire a native encoder surface for rendering, if this backend supports it.
    fn acquire_output(&mut self) -> Result<Option<wgpu::Texture>> {
        ensure!(!self.closed, "encoder closed");
        #[cfg(target_os = "macos")]
        if let Sink::Metal(session) = &mut self.sink {
            // SAFETY: this surface is allocated by this encoder on the same Metal device.
            // The encoder retains its CV objects through GPU completion and compression;
            // the imported HAL texture holds its own native texture retain.
            unsafe {
                let device = self
                    .gpu
                    .device
                    .as_hal::<wgpu::hal::api::Metal>()
                    .context("native encoder requires a Metal device")?;
                let raw =
                    session
                        .acquire(objc2::rc::Retained::as_ptr(device.raw_device())
                            as *mut std::ffi::c_void)?;
                let raw = objc2::rc::Retained::<
                    objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>,
                >::retain(raw.cast())
                .context("native output texture is null")?;
                let size = wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                };
                let texture = wgpu::hal::metal::Device::texture_from_raw(
                    raw,
                    wgpu::TextureFormat::Bgra8Unorm,
                    objc2_metal::MTLTextureType::Type2D,
                    1,
                    1,
                    size.into(),
                );
                return Ok(Some(
                    self.gpu
                        .device
                        .create_texture_from_hal::<wgpu::hal::api::Metal>(
                            texture,
                            &wgpu::TextureDescriptor {
                                label: Some("encoder IOSurface"),
                                size,
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Bgra8Unorm,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING
                                    | wgpu::TextureUsages::COPY_SRC
                                    | if self
                                        .gpu
                                        .device
                                        .features()
                                        .contains(wgpu::Features::BGRA8UNORM_STORAGE)
                                    {
                                        wgpu::TextureUsages::STORAGE_BINDING
                                    } else {
                                        wgpu::TextureUsages::empty()
                                    },
                                view_formats: &[],
                            },
                        ),
                ));
            }
        }
        Ok(None)
    }
    /// Allocate the sink's part of a graph ring entry before rendering begins.
    pub fn create_surface(&mut self) -> Result<Surface> {
        let (returned, completed) = std::sync::mpsc::channel();
        let readback = self.uses_pixel_readback().then(|| {
            self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ring encoder staging"),
                size: (u64::from(self.width) * 4).div_ceil(COPY_ROW_ALIGNMENT)
                    * COPY_ROW_ALIGNMENT
                    * u64::from(self.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        #[cfg(target_os = "linux")]
        let nvidia = if let Sink::Nvidia(session) = &mut self.sink {
            Some(session.create_surface()?)
        } else {
            None
        };
        Ok(Surface {
            texture: self.acquire_output()?,
            pixels: if readback.is_some() {
                Vec::with_capacity(self.width as usize * self.height as usize * 4)
            } else {
                Vec::new()
            },
            readback,
            returned,
            completed: std::sync::Mutex::new(completed),
            in_flight: false,
            pipe_wait_ms: 0.,
            submitted: None,
            #[cfg(target_os = "linux")]
            nvidia,
        })
    }
    /// Release native GPU consumers before ring reuse. Portable pipe payloads
    /// are independent of staging and are reclaimed at the next push.
    pub fn release(&mut self, index: u64, surface: &mut Surface) -> Result<()> {
        ensure!(
            surface.submitted == Some(index),
            "ring entry does not contain frame {index}"
        );
        #[cfg(target_os = "linux")]
        if let Sink::Nvidia(session) = &mut self.sink {
            session.release(
                index,
                surface
                    .nvidia
                    .as_mut()
                    .context("missing NVENC ring surface")?,
            )?;
        }
        #[cfg(target_os = "macos")]
        if let Sink::Metal(session) = &mut self.sink {
            let texture = surface.texture.as_ref().context("missing ring surface")?;
            let raw = unsafe { texture.as_hal::<wgpu::hal::api::Metal>() }
                .context("expected Metal surface")?;
            session.wait(raw.raw_handle() as *const _ as *mut std::ffi::c_void)?;
        }
        surface.submitted = None;
        Ok(())
    }
    fn feed(&mut self, texture: &wgpu::Texture, surface: &mut Surface, submission: &wgpu::SubmissionIndex) -> Result<()> {
        #[cfg(target_os = "linux")]
        if let Sink::Nvidia(session) = &mut self.sink {
            return session.push(
                &self.gpu,
                texture,
                self.next,
                surface
                    .nvidia
                    .as_mut()
                    .context("missing NVENC ring surface")?,
            );
        }
        #[cfg(target_os = "macos")]
        if let Sink::Metal(session) = &mut self.sink {
            ensure!(
                texture
                    .usage()
                    .contains(wgpu::TextureUsages::TEXTURE_BINDING),
                "Metal encoding requires TEXTURE_BINDING usage"
            );
            // Flush deferred queue.write_texture/write_buffer uploads before native access.
            self.gpu.queue.submit([]);
            // Both render and conversion use this queue: conversion precedes render-slot reuse.
            // SAFETY: guards keep the native objects alive while native submission retains them.
            let queue = unsafe { self.gpu.queue.as_hal::<wgpu::hal::api::Metal>() }
                .context("Auto requires a Metal queue")?;
            let raw = unsafe { texture.as_hal::<wgpu::hal::api::Metal>() }.context(
                "Auto requires Metal device on macOS; select Readback for other backends",
            )?;
            let surface = surface
                .texture
                .as_ref()
                .context("VideoToolbox requires an installed ring surface")?;
            let destination = unsafe { surface.as_hal::<wgpu::hal::api::Metal>() }
                .context("encoder surface must be Metal")?;
            return session.push(
                raw.raw_handle() as *const _ as *mut std::ffi::c_void,
                destination.raw_handle() as *const _ as *mut std::ffi::c_void,
                queue.as_raw() as *const _ as *mut std::ffi::c_void,
                self.next,
            );
        }
        // The pipe owns packed host pixels, not the staging buffer. Reclaim
        // them only after the successor render has been submitted, so pipe
        // backpressure cannot idle the GPU before ring reuse.
        let wait = std::time::Instant::now();
        if surface.in_flight {
            surface.pixels = surface
                .completed
                .get_mut()
                .unwrap()
                .recv()
                .context("encoder pipe writer stopped")?
                .map_err(anyhow::Error::msg)?;
            surface.in_flight = false;
        }
        surface.pipe_wait_ms = wait.elapsed().as_secs_f64() * 1000.;
        let buffer = surface
            .readback
            .as_ref()
            .context("portable encoding requires ring readback storage")?;
        let stride =
            (self.width * 4).div_ceil(COPY_ROW_ALIGNMENT as u32) * COPY_ROW_ALIGNMENT as u32;
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission.clone()),
            timeout: None,
        })?;
        rx.recv()??;
        let result = (|| -> Result<()> {
            let view = buffer.get_mapped_range(..);
            let Sink::Pipe { input, .. } = &mut self.sink else {
                unreachable!()
            };
            let input = input.as_mut().context("closed encoder")?;
            let mut packed = std::mem::take(&mut surface.pixels);
            packed.clear();
            if stride == self.width * 4 && texture.format() == wgpu::TextureFormat::Rgba8Unorm {
                packed.extend_from_slice(&view);
            } else {
                for bytes in view.chunks_exact(stride as usize) {
                    packed.extend_from_slice(&bytes[..self.width as usize * 4]);
                }
                if texture.format() == wgpu::TextureFormat::Bgra8Unorm {
                    for pixel in packed.chunks_exact_mut(4) {
                        pixel.swap(0, 2);
                    }
                }
            }
            input.send(packed, surface.returned.clone())?;
            surface.in_flight = true;
            Ok(())
        })();
        buffer.unmap();
        if result.is_err() {
            if let Sink::Pipe {
                child,
                input,
                stderr,
            } = &mut self.sink
            {
                pipe::finish(child, input, stderr)?;
            }
        }
        result
    }
    /// Finish every frame and mux the file, retaining native/GPU ownership until
    /// this encoder is dropped. No further frames or finish calls are accepted.
    /// Drop hardware encoders before std::process::exit/libc exit; only _exit
    /// may skip their drop without running driver exit hooks.
    pub fn finish(&mut self) -> Result<PathBuf> {
        ensure!(!self.closed, "encoder closed");
        ensure!(self.next > 0, "no frames");
        self.closed = true;
        match &mut self.sink {
            #[cfg(target_os = "linux")]
            Sink::Nvidia(session) => session.finish(self.diagnostics)?,
            Sink::Pipe {
                child,
                input,
                stderr,
            } => pipe::finish(child, input, stderr)?,
            #[cfg(target_os = "macos")]
            Sink::Metal(session) => session.finish()?,
        }
        if let Some(staged) = self.staged_output.take() {
            staged.persist(&self.path).map_err(|error| error.error).context("publish completed video")?;
        }
        Ok(self.path.clone())
    }
}
#[cfg(target_os = "linux")]
mod nvidia;
impl Drop for Encoder {
    fn drop(&mut self) {
        if let Sink::Pipe {
            child,
            input,
            stderr,
        } = &mut self.sink
        {
            let _ = child.kill();
            let _ = pipe::finish(child, input, stderr);
        }
    }
}
#[cfg(target_os = "macos")]
mod native {
    use super::*;
    use std::{
        ffi::{CString, c_void},
        os::unix::ffi::OsStrExt,
    };
    unsafe extern "C" {
        fn enc_prepare(
            w: u32,
            h: u32,
            fps_num: u32,
            fps_den: u32,
            bitrate: u32,
            gop_frames: u32,
            prores: bool,
            primaries: u32,
        ) -> *mut c_void;
        fn enc_attach(ctx: *mut c_void, path: *const i8) -> i32;
        fn enc_acquire(ctx: *mut c_void, device: *mut c_void) -> *mut c_void;
        fn enc_push(
            ctx: *mut c_void,
            texture: *mut c_void,
            destination: *mut c_void,
            queue: *mut c_void,
            index: u64,
        ) -> i32;
        fn enc_wait(ctx: *mut c_void, texture: *mut c_void) -> i32;
        fn enc_finish(ctx: *mut c_void) -> i32;
        fn enc_drop(ctx: *mut c_void);
    }
    pub struct Session(*mut c_void);
    // SAFETY: VideoToolbox/CoreVideo/AVAssetWriter objects are not thread-affine.
    // Ownership moves exclusively; all calls require &mut Session, and native
    // teardown drains the private handoff/writer queues before releasing objects.
    unsafe impl Send for Session {}
    impl Session {
        pub fn prepare(
            w: u32,
            h: u32,
            fps: Time,
            bitrate: u32,
            gop_frames: u32,
            prores: bool,
            primaries: Primaries,
        ) -> Result<Self> {
            let ptr = unsafe { enc_prepare(w, h, fps.numerator() as u32, fps.denominator() as u32, bitrate, gop_frames, prores, primaries.h273()) };
            ensure!(!ptr.is_null(), "native VideoToolbox initialization failed");
            Ok(Self(ptr))
        }
        pub fn attach(&mut self, path: &Path) -> Result<()> {
            let path = CString::new(path.as_os_str().as_bytes())?;
            let status = unsafe { enc_attach(self.0, path.as_ptr()) };
            ensure!(status == 0, "native output creation failed: {status}");
            Ok(())
        }
        pub fn acquire(&mut self, device: *mut c_void) -> Result<*mut c_void> {
            let raw = unsafe { enc_acquire(self.0, device) };
            ensure!(!raw.is_null(), "native output surface allocation failed");
            Ok(raw)
        }
        pub fn wait(&mut self, texture: *mut c_void) -> Result<()> {
            let status = unsafe { enc_wait(self.0, texture) };
            ensure!(status == 0, "native ring release failed: {status}");
            Ok(())
        }
        pub fn push(
            &mut self,
            texture: *mut c_void,
            destination: *mut c_void,
            queue: *mut c_void,
            index: u64,
        ) -> Result<()> {
            let s = unsafe { enc_push(self.0, texture, destination, queue, index) };
            ensure!(s == 0, "native encode failed: {s}");
            Ok(())
        }
        pub fn finish(&mut self) -> Result<()> {
            let s = unsafe { enc_finish(self.0) };
            ensure!(s == 0, "native finish failed: {s}");
            Ok(())
        }
    }
    impl Drop for Session {
        fn drop(&mut self) {
            unsafe { enc_drop(self.0) }
        }
    }
}

#[cfg(test)]
mod device_limit_tests {
    use super::*;
    #[test]
    fn device_refusals_report_demand_limit_and_recovery() {
        let mut limits = wgpu::Limits::default();
        limits.max_texture_dimension_2d = 16;
        limits.max_buffer_size = 256;
        for (w, h, demand, limit) in [(17, 8, "17x8", "max_texture_dimension_2d=16"), (8, 2, "512 bytes", "max_buffer_size=256")] {
            let error = validate_device_limits(&limits, w, h).unwrap_err();
            assert!(error.is::<DeviceLimit>());
            let message = error.to_string();
            for expected in [demand, limit, "reduce the output resolution", "vork help limits"] {
                assert!(message.contains(expected), "{message}");
            }
        }
        validate_device_limits(&limits, 8, 1).unwrap();
    }
}
