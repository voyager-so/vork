//! Pinned constants mirrored from the production texture pipeline.
//! Every value cites its `file:line` source; the extractor manifest is the
//! authority for shader-side constants — these are the CPU-side ones.

/// Prefetch worker threads (local read + decode run in parallel).
pub const PREFETCH_CONCURRENCY: usize = 12;

/// The texture-manager max dimension AS CONFIGURED IN PRODUCTION — NOT the
/// `TEXTURE_CONSTANTS.MAX_TEXTURE_DIMENSION = 2048` initial value
/// (`renderers/webgpu/constants/WebGPUConstants.ts:214`). That 2048 default is
/// dead at runtime: `WebGPUDevice.requestDeviceForAdapter` requests
/// `maxTextureDimension2D = min(16384, adapter.limits.maxTextureDimension2D)`
/// (`renderers/webgpu/gpu/core/WebGPUDevice.ts:50`), and
/// `WebGPUShapeSystem.initialize` immediately calls
/// `textureManager.setTextureLimits({ maxTextureDimension: device.limits.maxTextureDimension2D })`
/// (`renderers/webgpu/rendering/systems/WebGPUShapeSystem.ts:353-357`), so every
/// image/pattern texture sizes against the DEVICE limit (16384 on the capture
/// fleet). A 5454×5454 photo therefore uploads at FULL resolution with
/// scaleFactor (1,1) and is bilinearly minified at draw — the goldens carry
/// that minification aliasing, not a 2048 Mitchell downscale (Q-89, found by
/// the WF5 real-canvas sweep). The matching wgpu device limit is requested in
/// `moda_engine::device::GpuContext::new`.
pub const MAX_TEXTURE_DIMENSION: u32 = 16384;

/// The LIVE granted `max_texture_dimension_2d` of the current GPU device,
/// published by `moda_engine::device::GpuContext` at device acquisition and
/// read by the sizing policy ([`crate::sizing`]). Browser parity: the classic
/// WebGPU renderer sizes textures against the DEVICE limit via
/// `textureManager.setTextureLimits({ maxTextureDimension:
/// device.limits.maxTextureDimension2D })` (see [`MAX_TEXTURE_DIMENSION`]'s
/// doc). On a healthy device the grant equals the [`MAX_TEXTURE_DIMENSION`]
/// opt-in and behavior is byte-identical to the constant; the values only
/// diverge when the requestDevice retry ladder lands on a lower tier
/// (WEB-NEXTJS-21G: an adapter advertising 16384 whose own requestDevice
/// grants only the spec-baseline 8192), where sizing MUST track the grant so
/// oversized rasters downscale at decode instead of failing texture creation
/// mid-render. Default = the constant, so every consumer with no live device
/// (tests, goldens, callers before device init) keeps the pinned behavior.
/// Process-global last-writer-wins like the engine's other device statics
/// (`DEVICE_LOST`): the wasm module owns exactly one device, and native fleet
/// processes acquire homogeneous devices.
static LIVE_MAX_TEXTURE_DIMENSION: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(MAX_TEXTURE_DIMENSION);

/// The sizing policy's effective per-axis texture cap — the pinned
/// [`MAX_TEXTURE_DIMENSION`] tightened by the live device grant (never above
/// the pin: the policy's 16384 ceiling is also a memory/quality decision, not
/// just a device bound).
pub fn live_max_texture_dimension() -> u32 {
    LIVE_MAX_TEXTURE_DIMENSION.load(std::sync::atomic::Ordering::Relaxed)
}

/// Record the granted `max_texture_dimension_2d` of a freshly acquired GPU
/// device. Values are clamped into `[1, MAX_TEXTURE_DIMENSION]` — a grant
/// above the pin does not raise the sizing ceiling (see
/// [`live_max_texture_dimension`]).
pub fn set_live_max_texture_dimension(granted: u32) {
    LIVE_MAX_TEXTURE_DIMENSION.store(
        granted.clamp(1, MAX_TEXTURE_DIMENSION),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// `TEXTURE_CONSTANTS.SMALL_IMAGE_THRESHOLD` (`WebGPUConstants.ts:220`).
pub const SMALL_IMAGE_THRESHOLD: u32 = 256;

/// `TEXTURE_CONSTANTS.MIN_SVG_DIMENSION` (`WebGPUConstants.ts:226`).
pub const MIN_SVG_DIMENSION: u32 = 512;

/// `TEXTURE_CONSTANTS.PIXEL_RATIO` (`WebGPUConstants.ts:232`) — a CONSTANT 2,
/// not the device pixel ratio. Drives the small-image/SVG raster upscale even
/// in 1x exports (the goldens were captured with this in effect).
pub const TEXTURE_PIXEL_RATIO: f64 = 2.0;

/// `TEXTURE_CONSTANTS.SVG_RASTER_ZOOM_HEADROOM` (`WebGPUConstants.ts`) — how far
/// past an SVG's on-screen footprint we still rasterize, so zooming in stays
/// crisp without re-decoding.
///
/// This is the multiplier on the footprint that
/// [`crate::sizing::resolve_target_size`] uses as a CEILING. The footprint
/// already carries [`TEXTURE_PIXEL_RATIO`], so 4 here is 8x the CSS-pixel box —
/// ~3 doublings of canvas zoom before an SVG softens, which no editor session
/// reaches on a pattern fill.
///
/// A ceiling is needed because the minimum-dimension policy multiplies an SVG's
/// NATURAL size by [`TEXTURE_PIXEL_RATIO`] with no reference to how large the
/// node actually draws. A Figma export pasted at a 4255x5971 artboard and drawn
/// in a 157x255 box rastered 8510x11942 (101.6 MP / 407 MB) for 0.2 MP of
/// output, and resvg had to run that asset's `feMorphology`/`feGaussianBlur`
/// chain over all of it — a main-thread hang, not just wasted memory
/// (canvas c4d6996f, 2026-08-20).
pub const SVG_RASTER_ZOOM_HEADROOM: f64 = 4.0;

/// `PATH_RENDERER_DEFAULTS.DEFAULT_PATTERN_BOUNDS` (`WebGPUConstants.ts:165`).
pub const DEFAULT_PATTERN_BOUNDS: [f32; 2] = [200.0, 200.0];

/// `SHAPE_DEFAULTS.pattern.defaultOverlayOpacity` (`core/config/shapeDefaults.ts:168`).
pub const DEFAULT_OVERLAY_OPACITY: f32 = 1.0;

/// `BLEND_MODE_DISABLED` (`renderers/webgpu/constants/blendModes.ts:73`).
pub const BLEND_MODE_DISABLED: f32 = -1.0;
