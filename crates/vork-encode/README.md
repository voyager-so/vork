# GPU frames to video

Unstable, internal Rust crate; everything in it is free to change.

Rust package `vork-encode`.
`Encoder::open(DeviceInfo { device, queue, backend }, width, height, fps, EncodeSettings::default(), path, frames, ring_depth)`,
`create_surface() -> Surface`, `push(index, texture, &mut surface)`, `release(index, &mut surface)`,
`finish() -> PathBuf`. Errors use `anyhow::Result`.

`Encoder::prepare(backend, width, height, fps, settings, path) -> PreparedEncoder`
creates device-independent codec resources without creating or truncating the output.
It may run on a worker while the render device starts. `PreparedEncoder::open(device,
queue, frames, ring_depth)` validates device limits, creates the output and attaches GPU resources.
The selected frame count lets FFmpeg readback validate exact movie timing before opening its destination, including when native macOS attachment fails.
Dropping a preparation without opening it cancels and reclaims its native session,
leaving an existing output untouched. On Linux, driver loading and CUDA initialization
run in prepare; context creation, Vulkan UUID matching and NVENC attachment remain in open.

Frames start at zero and must arrive in order; duplicate and future indices fail explicitly.
Install one `Surface` per graph ring entry and output. Record `surface.record_copy` after the
frame's resolve, submit on the supplied queue, then call `push`. Before reusing an entry,
call `release` to wait for its pipe write or native encode/file consumer. The graph also waits
for PNG/EXR consumers before resubmitting the entry. Surface storage is never replaced at collection.
Input is single-sample 2D RGBA8Unorm
or BGRA8Unorm, even dimensions, COPY_SRC usage; native Metal also requires TEXTURE_BINDING.
Texture values represent encoded sRGB components; do not supply linear-light colour values.

* macOS Auto: wgpu-hal Metal texture handle -> GPU compute conversion -> IOSurface-backed
  BGRA CVPixelBuffer -> one hardware-required VideoToolbox session -> AVAssetWriter passthrough
  MP4 mux. No CPU pixel readback. The renderer writes directly to installed BGRA IOSurfaces when supported;
  other sources use GPU conversion into the entry surface. CoreVideo retains the installed
  surfaces, and ring release waits through compression and file consumption.
  Rec.709 matrix/primaries/transfer are explicit. Metal completion hands off asynchronously.
* Linux Auto: wgpu Vulkan texture -> device-local, dedicated exportable buffer -> opaque fd
  imported into CUDA on the matching Vulkan device UUID -> one NVENC session. NVENC accepts
  RGBA/BGRA and performs its RGB-to-YUV conversion. No CPU pixel readback. Only compressed
  H.264 is piped through ffmpeg for MP4 muxing (`-c:v copy`). Export/import and resource
  registration happen once per slot. Input/bitstream capacity comes from the graph ring depth; each entry is drained in order
  before reuse, and finish
  drains the remaining slots. Vulkan ownership barriers cover each slot’s buffer range.
  Stage totals for copy/wait, NVENC submit/drain and mux writes/finish are printed at finish.
  If native initialization is unavailable, use readback NVENC or the reported libx264 fallback.
  `uses_pixel_readback()` reports which path opened; failures after the first frame are errors.
* Readback: the ring entry’s padded staging buffer -> libx264 through ffmpeg on every platform.
* HardwareReadback: readback -> ffmpeg VideoToolbox (Mac) or NVENC (NVIDIA), for measurement.

Linux MP4 muxing and portable fallback require ffmpeg on PATH (libx264 for software encoding).
Before opening a readback stream, hardware must pass its initialization probe;
software codecs must appear in ffmpeg’s video encoder list. Missing codecs fail
before writing pixels or touching the destination.
Native Linux loads libcuda.so.1 and libnvidia-encode.so.1 dynamically; no CUDA SDK is needed
to build. Native macOS needs no ffmpeg for encoding.
Hardware initialization errors can be handled by reopening with Readback. `finish` flushes hardware
callbacks before finalizing the mux. It borrows the encoder and closes frame submission;
GPU/native resources stay owned until the encoder is dropped. Retaining finished encoders
allows a one-shot process to exit after flushing its files without redundant driver teardown.
Drop hardware encoders before `std::process::exit`/libc `exit`; only `_exit` may
skip the drop without running driver exit hooks. Dropping an unfinished encoder
cancels it/reaps its child.
MP4 container bytes may include session metadata; determinism checks compare decoded framemd5.

## Encode policy

`EncodeSettings` selects the codec; H.264 is the implemented format.
Defaults preserve 12 Mbps hardware VBR, a two-second GOP and software libx264 CRF 18
with the `veryfast` preset. Callers can change bitrate, GOP seconds and software CRF;
all backends receive the same GOP, and native hardware receives the selected bitrate.
Hardware bitrate and software CRF are deliberately different rate-control modes.

Render concurrency (`vork_render::FRAMES_IN_FLIGHT`) bounds active GPU submissions. The
output ring has three entries per active frame: one GPU flight and two consumer flights.
PNG, EXR, portable video and native encoder surfaces share entry lifetimes; the encoder has
no independent reorder queue, staging recycler or surface-capacity policy.

All video outputs use limited-range Rec.709 YUV with matching matrix and transfer
tags on native VideoToolbox/NVENC and readback ffmpeg. The primaries tag is the
frames' display's (`EncodeSettings.primaries`): BT.709 for sRGB pictures, P3 at D65
(H.273 code 12) for Display P3 pictures. The frames are 8-bit, so they carry SDR displays only. The ffmpeg path explicitly converts RGB to limited-range BT.709
YUV before encoding, so its conversion agrees with the stream metadata.

MOV destinations select ProRes 4444 (full chroma plus alpha). Auto uses a native
VideoToolbox session and IOSurfaces on Mac, preferring hardware and allowing
Apple’s software encoder. VT initialization failure falls back to `prores_ks`
CPU readback, also used on Linux and Windows. Software ProRes automatically selects slice workers from available CPUs. An ordered pipe worker
overlaps pipe writes with rendering/readback; finish drains the worker before waiting for
the muxer. Each ring entry owns its CPU payload and gets it back on release; a bounded stderr
tail is included in failures.
`HardwareReadback` with MOV is rejected; use Auto or Readback. MP4 retains the
existing H.264 hardware path and cannot carry alpha. The renderer supplies straight RGBA when `render.background` is `"transparent"`.

The ordering test checks BT.709 tags and decodes with explicit BT.709/video range.
On Mac, set `VORK_REQUIRE_VT_PRORES=1` for that test to require Auto MOV to use
VideoToolbox (no pixel readback); without it, Auto may legitimately fall back.

Linux hardware encoding follows the rendering adapter. On NVIDIA, Auto uses NVENC.
On every other adapter Auto encodes MP4 with libx264 at its quality target (CRF):
hardware encoders run at a fixed bitrate, and the default must not look worse than
software on hard content. `PreparedEncoder::validate_fallback(frames, Some(&device))` performs selection
before output cleanup and retains the result. Software-only preflight accepts
`None` for the device. Hardware unavailability is reported before libx264 fallback;
HardwareReadback refuses. FFmpeg movies are staged beside their destination and
published only after successful finish, with no retry after streaming starts.
