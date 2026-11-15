/// derived: wgpu COPY_BYTES_PER_ROW_ALIGNMENT, required by texture-to-buffer copies.
pub(crate) const COPY_ROW_ALIGNMENT: u64 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
/// quality/performance setting: caps the existing NVENC capability probe dimensions.
pub(crate) const NVENC_PROBE_EXTENT: u32 = 256;
/// quality/performance setting: existing portable H.264 encoding speed preset.
pub(crate) const SOFTWARE_PRESET: &str = "veryfast";
/// quality/performance setting: existing portable H.264 constant rate factor.
pub(crate) const SOFTWARE_CRF: u8 = 18;
/// quality/performance setting: existing hardware encoder bitrate in bits per second.
pub(crate) const HARDWARE_BITRATE: u32 = 12_000_000;
/// quality/performance setting: existing maximum keyframe spacing, in seconds.
pub(crate) const GOP_SECONDS: u32 = 2;
/// H.264/libx264 CRF domain upper endpoint (0 is lossless, 51 is lowest quality).
pub(crate) const MAX_SOFTWARE_CRF: u8 = 51;

/// ffmpeg auto-selects slice workers from available CPUs, independent of H.264 policy.
pub(crate) const PRORES_THREADS: &str = "0";
/// Bounded tail retains ffmpeg diagnostics without allowing unbounded stderr growth.
pub(crate) const FFMPEG_STDERR_BYTES: usize = 16 * 1024;

/// ProRes 4444 preserves an independent alpha plane in MOV on every ffmpeg platform.
pub const PRORES_PROFILE: &str = "4444";
/// Maximum prores_ks alpha precision; source coverage is RGBA8.
pub const PRORES_ALPHA_BITS: &str = "16";
/// Full chroma resolution and alpha, accepted by prores_ks.
pub const PRORES_PIXEL_FORMAT: &str = "yuva444p10le";

/// Shared diagnosis for every portable ffmpeg failure.
pub(crate) const FFMPEG_REQUIREMENT: &str =
    "video encoding needs ffmpeg on PATH with libx264 (MP4 fallback) and prores_ks (MOV); Linux MP4 needs ffmpeg even with native NVENC for muxing. Install: sudo apt install ffmpeg (Ubuntu), winget install Gyan.FFmpeg or extract ffmpeg-release-essentials.zip from https://www.gyan.dev/ffmpeg/builds/ and add bin to PATH (Windows), brew install ffmpeg (macOS fallback)";
