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
pub mod gpu_memory;
mod pipe;
#[cfg(target_os = "linux")]
mod vaapi;
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
pub fn validate_video_output(path: &Path) -> Result<()> {
    if cfg!(target_os = "macos") { return Ok(()); }
    let mov = path.extension().is_some_and(|s| s.eq_ignore_ascii_case("mov"));
    if mov { return require_ffmpeg_encoder("prores_ks"); }
    // Linux native NVENC still requires FFmpeg as a muxer. Windows needs a
    // usable software fallback even if its FFmpeg advertises NVENC.
    if cfg!(target_os = "windows") { return require_ffmpeg_encoder("libx264"); }
    let output = Command::new("ffmpeg").arg("-version").stdin(Stdio::null()).output()
        .with_context(|| format!("video output needs ffmpeg on PATH: {FFMPEG_REQUIREMENT}"))?;
    ensure!(output.status.success(), "ffmpeg on PATH is unusable: {FFMPEG_REQUIREMENT}");
    Ok(())
}

/// H.264 output policy, translated by each backend without changing defaults.
#[derive(Clone, Copy, Debug)]
pub struct EncodeSettings {
    /// Emit encoder stage timings when CLI diagnostics are enabled.
    pub diagnostics: bool,
    /// Hardware VBR target in bits per second (NVENC/VideoToolbox).
    pub hardware_bitrate: u32,
    /// Maximum keyframe spacing in seconds, shared by hardware and software.
    pub gop_seconds: u32,
    /// Software libx264 quality: 0 (lossless) through 51 (lowest quality).
    pub software_crf: u8,
}
impl Default for EncodeSettings {
    fn default() -> Self {
        Self {
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
/// Handles belong to one device. Auto uses native Metal on macOS, Vulkan/CUDA NVENC on Linux,
/// ffmpeg NVENC after readback on Windows, and falls back to portable readback (libx264)
/// when hardware encoding is unavailable.
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
    #[cfg(target_os = "macos")]
    metal: Option<native::Session>,
    #[cfg(target_os = "linux")]
    nvidia: Option<nvidia::Prepared>,
}
impl PreparedEncoder {
    /// Attach the prepared output to the device used for rendering its frames.
    pub fn open(
        self,
        device: wgpu::Device,
        queue: wgpu::Queue,
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
        if let Some(prepared) = nvidia {
            match nvidia::Session::open(
                &gpu,
                prepared,
                width,
                height,
                fps,
                settings.hardware_bitrate,
                gop_frames,
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
                        #[cfg(target_os = "linux")]
                        diagnostics: settings.diagnostics,
                    });
                }
                Err(error) => {
                    eprintln!("VideoToolbox output unavailable; using software encoding: {error:#}")
                }
            }
        }
        let mut codec = "libx264";
        #[cfg(target_os = "linux")]
        let mut vaapi_device = None;
        #[cfg(target_os = "macos")]
        if !prores && matches!(gpu.backend, Backend::HardwareReadback) {
            codec = "h264_videotoolbox";
        }
        // Windows has no zero-copy session; ffmpeg's NVENC still beats libx264 after readback.
        #[cfg(any(target_os = "linux", windows))]
        if !prores && !matches!(gpu.backend, Backend::Readback) {
            match probe_ffmpeg_nvenc(width, height) {
                Ok(()) => codec = "h264_nvenc",
                Err(error) => {
                    ensure!(!matches!(gpu.backend, Backend::HardwareReadback), "{error:#}");
                    eprintln!("FFmpeg NVENC unavailable; using software libx264: {error:#}");
                }
            }
        }
        // AMD has no NVENC; FFmpeg's VAAPI encoder is the hardware equivalent.
        #[cfg(target_os = "linux")]
        if !prores && codec == "libx264" {
            match vaapi::probe(width, height) {
                Ok(device) => {
                    codec = "h264_vaapi";
                    vaapi_device = Some(device);
                }
                Err(error) => eprintln!("FFmpeg VAAPI unavailable; using software libx264: {error:#}"),
            }
        }
        if prores {
            codec = "prores_ks";
        }
        require_ffmpeg_encoder(codec)?;
        eprintln!("Video encoder: FFmpeg {codec} (readback)");
        let mut command = Command::new("ffmpeg");
        #[cfg(target_os = "linux")]
        if let Some(device) = &vaapi_device {
            command.args(vaapi::device_arguments(device));
        }
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
            command.args([
                "-preset",
                SOFTWARE_PRESET,
                "-crf",
                &settings.software_crf.to_string(),
                "-threads",
                SOFTWARE_THREADS,
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
        // converted frame's colour tags over the output codec options. VAAPI
        // converts to NV12 and uploads inside the same filter chain.
        let mut filter = String::from(
            "scale=out_color_matrix=bt709:out_range=tv,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        );
        #[cfg(target_os = "linux")]
        let hardware_frames = vaapi_device.is_some();
        #[cfg(not(target_os = "linux"))]
        let hardware_frames = false;
        if hardware_frames {
            filter.push_str(",format=nv12,hwupload");
        }
        command.args([
            "-vf",
            filter.as_str(),
            "-color_range",
            "tv",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ]);
        if !hardware_frames {
            command.args([
                "-pix_fmt",
                if prores {
                    PRORES_PIXEL_FORMAT
                } else {
                    "yuv420p"
                },
            ]);
        }
        let mut child = command
            .args([
                "-bf",
                "0",
                "-g",
                &gop_frames.to_string(),
                "-video_track_timescale",
                &fps.numerator().to_string(),
                "-map_metadata",
                "-1",
                "-f",
                if prores { "mov" } else { "mp4" },
            ])
            .arg(&path)
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
/// Test FFmpeg's VAAPI against the installed driver, returning the render node it used.
#[cfg(target_os = "linux")]
pub fn probe_ffmpeg_vaapi(width: u32, height: u32) -> Result<std::path::PathBuf> {
    vaapi::probe(width, height)
}

// Check before opening the rawvideo pipe so codec errors retain their cause.
fn require_ffmpeg_encoder(codec: &str) -> Result<()> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("query ffmpeg encoders: {FFMPEG_REQUIREMENT}"))?;
    validate_ffmpeg_encoder(output.status.success(), &output.stdout, codec)
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
        DeviceLimit(gpu_memory::limit_message("encoder texture dimensions", format_args!("{width}x{height}"), format_args!("max_texture_dimension_2d={}", limits.max_texture_dimension_2d), "reduce the output resolution or use a GPU with larger texture limits"))
    );
    let padded = (width as u64 * 4).div_ceil(COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
    ensure!(
        padded * height as u64 <= limits.max_buffer_size,
        DeviceLimit(gpu_memory::limit_message("encoder readback", format_args!("{} bytes at {width}x{height}", padded * height as u64), format_args!("max_buffer_size={} bytes", limits.max_buffer_size), "reduce the output resolution or use a GPU with a larger buffer limit"))
    );
    Ok(())
}
impl Encoder {
    pub fn open(
        gpu: DeviceInfo,
        width: u32,
        height: u32,
        fps: Time,
        settings: EncodeSettings,
        path: impl AsRef<Path>,
        ring_depth: usize,
    ) -> Result<Self> {
        Self::prepare(gpu.backend, width, height, fps, settings, path)?
            .open(gpu.device, gpu.queue, ring_depth)
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
                && height % 2 == 0
                && fps > Time::ZERO
                && fps.numerator() <= i64::from(i32::MAX)
                && fps.denominator() <= i64::from(i32::MAX),
            "dimensions must be positive/even; fps must be positive"
        );
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
        ) -> Result<Self> {
            let ptr = unsafe { enc_prepare(w, h, fps.numerator() as u32, fps.denominator() as u32, bitrate, gop_frames, prores) };
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
