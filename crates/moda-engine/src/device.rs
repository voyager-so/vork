//! device.rs — headless wgpu device acquisition (spike recipe,
//! `research/spikes/wgpu-headless.md` §1; DECISION §1.5: one binary, three
//! backends by env/flag).
//!
//! Backend selection precedence: explicit `--backend` flag → `WGPU_BACKEND`
//! env var → auto (all backends, first adapter). `lavapipe` selects the
//! Vulkan backend and REQUIRES a CPU-type adapter (Mesa llvmpipe via
//! `VK_ICD_FILENAMES=…lvp_icd.x86_64.json`); it fails loudly when only a real
//! GPU is present so the M1L kill test can't silently measure the wrong tier.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, bail, Context, Result};

/// Set by wgpu's device-lost callback when the GPU device is genuinely lost.
/// Process-global because the callback is a `'static` closure and the engine is
/// process-global (moda-mobile-ffi's `engine_slot`). A lost device is permanent:
/// callers must rebuild the [`GpuContext`]/`Engine` rather than retry on it.
///
/// This is the signal of last resort. A loss can surface to the render caller
/// three ways: (1) a failed buffer map ([`is_device_lost_error`]); (2) a poll
/// that wgpu treats as fatal; (3) a later GPU call (e.g. `Queue::write_buffer`)
/// validating against the now-invalid device and PANICKING. Cases (2) and (3)
/// never return an `Err` the caller can classify, so this flag lets the caller
/// drop a dead engine even when the render unwound via a panic.
static DEVICE_LOST: AtomicBool = AtomicBool::new(false);

/// True once wgpu's device-lost callback has fired for the current device.
/// moda-mobile-ffi checks this after every render and drops its process-global
/// engine when set, so the next render rebuilds the device.
pub fn device_lost_flag() -> bool {
    DEVICE_LOST.load(Ordering::Acquire)
}

/// Clear the device-lost flag — call right after building a fresh
/// [`GpuContext`], so a loss recorded against the previous (now-dropped) device
/// does not immediately condemn the new one.
pub fn clear_device_lost_flag() {
    DEVICE_LOST.store(false, Ordering::Release);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Auto,
    Metal,
    Vulkan,
    /// Vulkan + require a software (CPU) adapter.
    Lavapipe,
}

impl BackendChoice {
    pub fn parse(s: &str) -> Result<BackendChoice> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(BackendChoice::Auto),
            "metal" => Ok(BackendChoice::Metal),
            "vulkan" => Ok(BackendChoice::Vulkan),
            "lavapipe" => Ok(BackendChoice::Lavapipe),
            other => bail!("unknown backend '{other}' (metal|vulkan|lavapipe|auto)"),
        }
    }

    /// flag (if given) → WGPU_BACKEND env → auto.
    pub fn resolve(flag: Option<&str>) -> Result<BackendChoice> {
        if let Some(f) = flag {
            return BackendChoice::parse(f);
        }
        if let Ok(env) = std::env::var("WGPU_BACKEND") {
            if !env.is_empty() {
                return BackendChoice::parse(&env);
            }
        }
        Ok(BackendChoice::Auto)
    }

    fn backends(self) -> wgpu::Backends {
        match self {
            BackendChoice::Auto => wgpu::Backends::all(),
            BackendChoice::Metal => wgpu::Backends::METAL,
            BackendChoice::Vulkan | BackendChoice::Lavapipe => wgpu::Backends::VULKAN,
        }
    }
}

#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    /// The largest MSAA sample count the adapter advertises as supported for
    /// ALL render-attachment formats the engine multisamples — the MSAA color
    /// (`Rgba8Unorm`, the default frame format) AND the depth (`Depth24Plus`) —
    /// intersected (a count is usable only if every multisampled attachment
    /// supports it, since they share one render pass). The frame planner clamps
    /// its requested `sample_count` to this so the color texture, depth
    /// texture, and every pipeline `MultisampleState.count` always agree on a
    /// value the device actually allows. Desktop/sim adapters advertise 4 (the
    /// parity path is unchanged); some Adreno/Android-Vulkan tiers do NOT
    /// advertise 4× for `Depth24Plus`, where this drops to 1 instead of failing
    /// the whole pass.
    ///
    /// This field is the `Rgba8Unorm`-frame count; an engine
    /// constructed with a `Bgra8Unorm` frame format must use
    /// [`GpuContext::max_sample_count_for`] instead, which consults
    /// [`GpuContext::max_sample_count_bgra`] — the same intersection computed
    /// for a BGRA color attachment.
    pub max_sample_count: u32,
    /// The [`GpuContext::max_sample_count`] intersection computed for
    /// a `Bgra8Unorm` MSAA color attachment (still ∩ `Depth24Plus`). Consumed
    /// by engines whose immutable frame format is BGRA (the live browser lane
    /// on preferred-`bgra8unorm` devices); read via
    /// [`GpuContext::max_sample_count_for`].
    pub max_sample_count_bgra: u32,
    /// The [`GpuContext::max_sample_count`] intersection computed for an `Rgba16Float` MSAA color attachment
    /// (still ∩ `Depth24Plus`): the pixel graph's float capture engine (motion-model.md §10.1).
    pub max_sample_count_f16: u32,
    /// Whether the PRESENTATION surface can be a `COPY_SRC` source.
    /// NOT a raw capability read — it's forced on for wasm32 (browsers back
    /// COPY_SRC for canvas textures even though wgpu's webgpu backend doesn't
    /// advertise it) and uses real advertised caps on native; see
    /// `new_async_for_surface`. When
    /// true, backdrop-effect frames present DIRECTLY into the swapchain and copy
    /// only the small backdrop region back out of it for the effect chain (the
    /// WebGPU renderer's approach — `WebGPUContext` configures the canvas with
    /// `RENDER_ATTACHMENT | COPY_SRC`). When false — notably Dawn's Android
    /// Vulkan swapchains, which can't be a copy source — those frames fall back
    /// to the offscreen target + full-frame blit. `false` on the native /
    /// headless `new` path (no surface to query).
    pub surface_copy_src: bool,
}

/// The depth format every multisampled engine pass attaches alongside its
/// color target — the color format is per-engine (`Rgba8Unorm` by default,
/// `Bgra8Unorm` for a preferred-BGRA live browser engine), so the
/// intersection is computed once per candidate color format. Keep in sync with
/// `frame.rs` `frame_targets` (MSAA color + depth) and `text_draw.rs` (depth).
const MULTISAMPLED_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

/// The largest sample count from the candidate set the adapter supports for
/// EVERY multisampled attachment format (`color_format` ∩ the shared depth).
/// Always ≥ 1 (single-sample is universal). Candidates are the engine-relevant
/// powers of two; we never request more than the planner's max (4), but probe
/// up so the intersection is honest if a future plan raises it.
///
/// The color format must also advertise `MULTISAMPLE_RESOLVE`: every engine
/// MSAA pass RESOLVES its color (into the offscreen target or, on the direct
/// path, straight into the swapchain view), so a count that can be sampled but
/// not resolved would still fail validation. Both frame formats carry
/// guaranteed `msaa_resolve` on the browser webgpu backend (no behavior change
/// there); this guards the native lanes, where advertised caps are real.
fn compute_max_sample_count(adapter: &wgpu::Adapter, color_format: wgpu::TextureFormat) -> u32 {
    let color_flags = adapter.get_texture_format_features(color_format).flags;
    if !color_flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE) {
        return 1;
    }
    let depth_flags = adapter
        .get_texture_format_features(MULTISAMPLED_DEPTH_FORMAT)
        .flags;
    // Descending so the first all-supported count wins.
    for &count in &[8u32, 4, 2] {
        if [color_flags, depth_flags]
            .iter()
            .all(|f| f.sample_count_supported(count))
        {
            return count;
        }
    }
    1
}

/// True when `err`'s chain shows the wgpu device is genuinely LOST — a state
/// that never recovers, so callers holding a long-lived [`GpuContext`]/`Engine`
/// must rebuild it instead of retrying (moda-mobile-ffi drops its process-global
/// engine slot on this). On Android a loss means a driver reset, GPU OOM, or
/// background GPU-work reclaim.
///
/// A failed buffer map ([`wgpu::BufferAsyncError`]) is the recoverable-vs-lost
/// boundary signal we key on: the map callback fires `Err` when the buffer's
/// device is lost or destroyed, and no retry on the same device will ever
/// succeed.
///
/// CRITICAL — a [`wgpu::PollError`] is NOT device loss. wgpu 29's `PollError`
/// has exactly two variants and neither is a loss:
///   * `PollError::Timeout` — "The requested Wait timed out before the
///     submission was completed." A SLOW frame (first-render shader compile on
///     mobile Vulkan, or the platform GPU watchdog returning `VK_TIMEOUT` on a
///     heavy draw) surfaces here. The device is healthy; the work just hasn't
///     finished. Treating this as loss drops a healthy engine, forcing the next
///     open to cold-rebuild and recompile shaders → another slow first render →
///     another spurious "loss" → a loop. (The mobile readback/submit polls use
///     `wait_indefinitely`, whose `None` timeout maps to a `u64::MAX` Vulkan
///     fence wait; a genuine `VK_ERROR_DEVICE_LOST` from that wait maps to
///     `DeviceError::Lost`, which wgpu handles as fatal and never returns to us
///     as a `PollError`.) `poll_until_complete_with_budget` in readback.rs
///     re-polls on `Timeout`, which only works if the classifier does not
///     condemn it.
///   * `PollError::WrongSubmissionIndex` — a caller bug (waiting on an index
///     that was never submitted), not a device condition.
/// So `PollError` must be excluded; only [`wgpu::BufferAsyncError`] classifies.
pub fn is_device_lost_error(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<wgpu::BufferAsyncError>().is_some())
}

impl GpuContext {
    pub fn new(choice: BackendChoice) -> Result<GpuContext> {
        Self::new_with_timestamps(choice, false)
    }

    pub fn new_profiled(choice: BackendChoice) -> Result<GpuContext> {
        Self::new_with_timestamps(choice, true)
    }

    fn new_with_timestamps(choice: BackendChoice, profile: bool) -> Result<GpuContext> {
        // Headless: no display handle, no surface (wgpu 29 dedicated ctor).
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = choice.backends();
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = match choice {
            BackendChoice::Lavapipe => {
                let adapters =
                    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
                adapters
                    .into_iter()
                    .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
                    .ok_or_else(|| {
                        anyhow!(
                            "backend 'lavapipe' requested but no CPU-type Vulkan adapter found \
                             (is Mesa llvmpipe installed and VK_ICD_FILENAMES set?)"
                        )
                    })?
            }
            // HighPerformance: with both a discrete ICD and lavapipe visible
            // (the Cloud Run GPU tier once the NVIDIA ICD manifest is baked —
            // Dockerfile.agent-worker), the default no-preference request does
            // not guarantee the discrete adapter wins; this makes the L4 the
            // deterministic pick. Applies to every GpuContext::new caller —
            // server render lane, CLI, moda-py, mobile FFI, desktop headless —
            // a no-op on single-GPU mobile SoCs; on dual-GPU laptops it biases
            // only export/headless work to the dGPU. The surface path
            // (new_async_for_surface, the browser lane) keeps the
            // no-preference default.
            _ => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("no GPU adapter for {choice:?}: {e}"))?,
        };

        let adapter_info = adapter.get_info();
        let descriptor = Self::device_descriptor(&adapter, profile);
        // Bounded retry ladder — see `fallback_descriptors` for why each tier
        // exists (adapter-parity for advertised-below-default adapters, spec
        // baseline for advertised-above-grantable ones, WEB-NEXTJS-21G).
        // Server-lane requests succeed on the first attempt, so the ladder
        // never runs there.
        let (device, queue) = match pollster::block_on(adapter.request_device(&descriptor)) {
            Ok(pair) => pair,
            Err(first_err) => {
                let [parity, baseline] = Self::fallback_descriptors(adapter.limits());
                match pollster::block_on(adapter.request_device(&parity)) {
                    Ok(pair) => pair,
                    Err(parity_err) => pollster::block_on(adapter.request_device(&baseline))
                        .with_context(|| {
                            format!(
                                "wgpu device request failed (parity and baseline limits also \
                                 rejected: {first_err}; parity: {parity_err})"
                            )
                        })?,
                }
            }
        };
        // Publish the GRANTED per-axis texture limit so the raster/SVG sizing
        // policy clamps to what this device can actually create — on a
        // baseline-tier device (8192) an oversized photo downscales at decode
        // instead of failing texture creation mid-render.
        moda_assets::constants::set_live_max_texture_dimension(
            device.limits().max_texture_dimension_2d,
        );

        // Record + log the loss ONCE. The flag lets the render caller drop a
        // dead engine even when the loss surfaces as a panic from a later GPU
        // call (e.g. `Queue::write_buffer` validating against the invalid
        // device) rather than as a classifiable `Err`. stderr is teed to
        // logcat under ModaNativeStdio on Android.
        device.set_device_lost_callback(|reason, message| {
            DEVICE_LOST.store(true, Ordering::Release);
            eprintln!("moda-engine: ERROR: GPU device lost ({reason:?}): {message}");
        });

        let max_sample_count = compute_max_sample_count(&adapter, wgpu::TextureFormat::Rgba8Unorm);
        let max_sample_count_bgra =
            compute_max_sample_count(&adapter, wgpu::TextureFormat::Bgra8Unorm);
        let max_sample_count_f16 =
            compute_max_sample_count(&adapter, wgpu::TextureFormat::Rgba16Float);

        Ok(GpuContext {
            device,
            queue,
            adapter_info,
            max_sample_count,
            max_sample_count_bgra,
            max_sample_count_f16,
            // No presentation surface on the blocking/native path.
            surface_copy_src: false,
        })
    }

    /// The adapter's MSAA ceiling for an engine whose frame color
    /// attachment is `frame_format` (∩ the shared `Depth24Plus` depth). Only
    /// `Bgra8Unorm` has its own precomputed ceiling; every other format maps
    /// to the `Rgba8Unorm` (default frame format) count. `Engine` construction
    /// rejects non-frame formats, so engine callers only ever pass the two —
    /// but the method itself is total and answers for any input.
    pub fn max_sample_count_for(&self, frame_format: wgpu::TextureFormat) -> u32 {
        match frame_format {
            wgpu::TextureFormat::Bgra8Unorm => self.max_sample_count_bgra,
            wgpu::TextureFormat::Rgba16Float => self.max_sample_count_f16,
            _ => self.max_sample_count,
        }
    }

    /// Mirror `WebGPUDevice.requestDeviceForAdapter`
    /// (`renderers/webgpu/gpu/core/WebGPUDevice.ts:50`): opt into
    /// `maxTextureDimension2D = min(16384, adapter limit)` so full-resolution
    /// photo pattern textures (browser uploads them un-downscaled up to the
    /// device limit — see `moda_assets::constants::MAX_TEXTURE_DIMENSION`)
    /// fit. wgpu's default limit is 8192, which would reject e.g. a 9000px
    /// upload that the browser accepts.
    fn device_descriptor(
        adapter: &wgpu::Adapter,
        profile: bool,
    ) -> wgpu::DeviceDescriptor<'static> {
        let mut limits = wgpu::Limits::default();
        limits.max_texture_dimension_2d = limits
            .max_texture_dimension_2d
            .max(16384.min(adapter.limits().max_texture_dimension_2d));
        let timestamps =
            profile || std::env::var("MODA_GPU_TIMESTAMPS").is_ok_and(|value| value == "1") || crate::prof::gpu();
        wgpu::DeviceDescriptor {
            required_limits: limits,
            required_features: Self::timestamp_features(
                adapter.get_info().backend,
                adapter.features(),
                timestamps,
                profile,
            ),
            ..Default::default()
        }
    }

    /// The timestamp features a device asks for: `TIMESTAMP_QUERY` when timing is on, plus
    /// `TIMESTAMP_QUERY_INSIDE_ENCODERS` for a profiled device (the format pass's readback timing), except on Metal.
    /// Apple GPUs sample counters only at encoder boundaries, so wgpu emulates `write_timestamp` with extra blit
    /// encoders; with them in the format pass's readback, that command buffer never completes and the wait for it
    /// hangs. Without the feature the format pass times its readback from a separate submission instead.
    fn timestamp_features(
        backend: wgpu::Backend,
        adapter: wgpu::Features,
        timestamps: bool,
        profile: bool,
    ) -> wgpu::Features {
        if !timestamps || !adapter.contains(wgpu::Features::TIMESTAMP_QUERY) {
            return wgpu::Features::empty();
        }
        let mut wanted = wgpu::Features::TIMESTAMP_QUERY;
        if profile && backend != wgpu::Backend::Metal {
            wanted |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        adapter & wanted
    }

    /// The bounded requestDevice retry ladder that follows a rejected
    /// [`Self::device_descriptor`] opt-in. Two descriptors, tried in order:
    ///
    /// 1. **Adapter parity** — exactly what the adapter advertises
    ///    (`adapter.limits()`). Recovers adapters that advertise LESS than the
    ///    WebGPU defaults (the iOS simulator's Metal adapter reports
    ///    `max_inter_stage_shader_variables = 15` < default 16), where any
    ///    default-based request over-asks. Shaders that genuinely exceed a
    ///    granted limit still fail at pipeline creation with a precise error.
    /// 2. **Spec baseline** — `wgpu::Limits::default()`, the WebGPU-spec
    ///    minimums every conformant implementation must grant. Recovers
    ///    adapters that advertise MORE than their own `requestDevice` grants
    ///    (WEB-NEXTJS-21G: `maxTextureDimension2D` advertised 16384, granted
    ///    8192 — and the parity tier re-requests the advertised lie, so it
    ///    deterministically fails the same way; the browser's actual grants
    ///    were exactly the spec baseline). The engine only ever opts in ABOVE
    ///    baseline for `max_texture_dimension_2d` (see
    ///    [`Self::device_descriptor`]), so landing on this tier costs only
    ///    texture headroom — and the sizing policy tracks the granted limit
    ///    (`moda_assets::constants::set_live_max_texture_dimension`), so
    ///    oversized rasters downscale instead of failing texture creation.
    fn fallback_descriptors(adapter_limits: wgpu::Limits) -> [wgpu::DeviceDescriptor<'static>; 2] {
        [
            wgpu::DeviceDescriptor {
                required_limits: adapter_limits,
                ..Default::default()
            },
            wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ]
    }

    /// WASM-EDITOR PROTOTYPE (research/wasm-editor/SPIKE-wasm-render.md): async
    /// device acquisition for environments where blocking is impossible
    /// (browser WebGPU via wgpu's webgpu backend). Caller owns the
    /// `wgpu::Instance` (it also creates the canvas surface) and may pass
    /// that surface so the adapter is presentation-compatible. Pure async —
    /// also works on native, where it is simply unused today.
    pub async fn new_async_for_surface(
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<GpuContext> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface,
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("no GPU adapter: {e}"))?;
        let adapter_info = adapter.get_info();
        let descriptor = Self::device_descriptor(&adapter, false);
        // Same bounded retry ladder as the blocking `new` path — see
        // `fallback_descriptors` (adapter parity for the iOS-sim Metal case,
        // spec baseline for the advertised-but-refused-limit case,
        // WEB-NEXTJS-21G).
        let (device, queue) = match adapter.request_device(&descriptor).await {
            Ok(pair) => pair,
            Err(first_err) => {
                let [parity, baseline] = Self::fallback_descriptors(adapter.limits());
                match adapter.request_device(&parity).await {
                    Ok(pair) => pair,
                    Err(parity_err) => {
                        adapter.request_device(&baseline).await.with_context(|| {
                            format!(
                                "wgpu device request failed (parity and baseline limits also \
                                 rejected: {first_err}; parity: {parity_err})"
                            )
                        })?
                    }
                }
            }
        };
        // Publish the GRANTED per-axis texture limit for the sizing policy —
        // see the blocking path's comment.
        moda_assets::constants::set_live_max_texture_dimension(
            device.limits().max_texture_dimension_2d,
        );
        let max_sample_count = compute_max_sample_count(&adapter, wgpu::TextureFormat::Rgba8Unorm);
        let max_sample_count_bgra =
            compute_max_sample_count(&adapter, wgpu::TextureFormat::Bgra8Unorm);
        let max_sample_count_f16 =
            compute_max_sample_count(&adapter, wgpu::TextureFormat::Rgba16Float);
        // Can the presentation surface be a copy SOURCE? Backdrop
        // frames then present direct and copy only their region back out of the
        // swapchain; otherwise they keep the offscreen + blit path.
        //
        // On the WEB (wgpu's webgpu backend) `get_capabilities` HARDCODES surface
        // usages to RENDER_ATTACHMENT and never introspects the canvas, but
        // `configure` passes the requested usage straight through to
        // `GPUCanvasContext.configure`, which every WebGPU browser backs for
        // COPY_SRC — the production WebGPU renderer (`WebGPUContext`) configures
        // the canvas `RENDER_ATTACHMENT | COPY_SRC` for ALL browsers. So the
        // canvas (not the under-reporting capability) is authoritative: force it
        // on for wasm32. On NATIVE backends the advertised usages are real —
        // honor them (Dawn's Android Vulkan swapchains, notably, can't be a copy
        // source). `false` when there is no surface (headless / readback).
        let surface_copy_src = compatible_surface
            .map(|s| {
                cfg!(target_arch = "wasm32")
                    || s.get_capabilities(&adapter)
                        .usages
                        .contains(wgpu::TextureUsages::COPY_SRC)
            })
            .unwrap_or(false);
        Ok(GpuContext {
            device,
            queue,
            adapter_info,
            max_sample_count,
            max_sample_count_bgra,
            max_sample_count_f16,
            surface_copy_src,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_never_requests_encoder_timestamps() {
        let ts = wgpu::Features::TIMESTAMP_QUERY;
        let all = ts | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let f = GpuContext::timestamp_features;
        // A profiled Metal device times the format pass, never with timestamps inside encoders.
        assert_eq!(f(wgpu::Backend::Metal, all, true, true), ts);
        // Other backends keep the readback timestamp.
        assert_eq!(f(wgpu::Backend::Vulkan, all, true, true), all);
        assert_eq!(f(wgpu::Backend::Dx12, all, true, true), all);
        // Bench timing (MODA_GPU_TIMESTAMPS) asks for pass timestamps only.
        assert_eq!(f(wgpu::Backend::Vulkan, all, true, false), ts);
        // Nothing without timing, or without adapter support.
        assert_eq!(f(wgpu::Backend::Vulkan, all, false, true), wgpu::Features::empty());
        let encoders_only = wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        assert_eq!(f(wgpu::Backend::Vulkan, encoders_only, true, true), wgpu::Features::empty());
        assert_eq!(f(wgpu::Backend::Vulkan, ts, true, true), ts);
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn max_sample_count_is_supported_and_clamps() {
        // Device-resilience clamp (Adreno/Android-Vulkan tiers lacking 4×
        // Depth24Plus): max_sample_count is always a real, adapter-supported
        // count for EVERY multisampled attachment format, and never exceeds the
        // engine's 4× ceiling. Desktop/sim adapters keep 4 (parity). Skips when
        // no GPU adapter exists (headless CI).
        let gpu = GpuContext::new(BackendChoice::Auto).expect("GPU test requires a working adapter");
        let n = gpu.max_sample_count;
        assert!(
            matches!(n, 1 | 2 | 4 | 8),
            "must be a real power-of-two MSAA count from the candidate set, got {n}"
        );
        // The render_plan clamp is `min(plan.sample_count, max).max(1)`; verify
        // it behaves on this adapter for the planner's default request of 4.
        let clamped = 4u32.min(n).max(1);
        assert!(
            clamped >= 1 && clamped <= n,
            "clamp must land at a supported count"
        );
        // Desktop/sim (macOS Metal, where this test runs locally) advertise 4;
        // a degraded mobile tier would report less. Either way the clamp never
        // exceeds the request.
        assert!(clamped <= 4, "clamp never raises above the requested 4");
    }

    #[test]
    #[ignore = "requires a working GPU adapter"]
    fn max_sample_count_for_selects_per_frame_format() {
        // A BGRA-frame engine must clamp its MSAA request against the
        // BGRA intersection, not the RGBA one — a mismatched clamp would create
        // MSAA color textures / pipelines at a count the adapter rejects for
        // that format. Any other (non-frame) format conservatively maps to the
        // default RGBA count.
        let gpu = GpuContext::new(BackendChoice::Auto).expect("GPU test requires a working adapter");
        assert_eq!(
            gpu.max_sample_count_for(wgpu::TextureFormat::Rgba8Unorm),
            gpu.max_sample_count
        );
        assert_eq!(
            gpu.max_sample_count_for(wgpu::TextureFormat::Bgra8Unorm),
            gpu.max_sample_count_bgra
        );
        assert_eq!(
            gpu.max_sample_count_for(wgpu::TextureFormat::R8Unorm),
            gpu.max_sample_count
        );
        for n in [gpu.max_sample_count, gpu.max_sample_count_bgra] {
            assert!(
                matches!(n, 1 | 2 | 4 | 8),
                "must be a real power-of-two MSAA count from the candidate set, got {n}"
            );
        }
    }

    #[test]
    fn buffer_map_failure_classifies_as_device_loss() {
        // The exact chain readback.rs produces when the map callback fires Err
        // because the buffer's device was lost or destroyed.
        let map = anyhow::Error::new(wgpu::BufferAsyncError).context("map_async channel");
        assert!(is_device_lost_error(&map));
    }

    #[test]
    fn poll_timeout_is_not_device_loss() {
        // REGRESSION GUARD (the bug that broke mobile canvas-open): a poll
        // timeout is a SLOW frame on a healthy device — a first-render shader
        // compile or a GPU watchdog `VK_TIMEOUT`. Misclassifying it as loss
        // drops a healthy engine, forcing a cold rebuild + recompile that times
        // out again — a loop. The exact chain readback.rs / frame.rs produce on
        // a slow poll:
        let poll =
            anyhow::Error::new(wgpu::PollError::Timeout).context("device poll during readback");
        assert!(!is_device_lost_error(&poll));
        let submit =
            anyhow::Error::new(wgpu::PollError::Timeout).context("device poll after submit");
        assert!(!is_device_lost_error(&submit));
    }

    #[test]
    fn poll_wrong_submission_index_is_not_device_loss() {
        // The other `PollError` variant — a caller bug, not a device condition.
        let wrong = anyhow::Error::new(wgpu::PollError::WrongSubmissionIndex(7, 3))
            .context("device poll after submit");
        assert!(!is_device_lost_error(&wrong));
    }

    #[test]
    fn device_lost_flag_round_trips() {
        // The flag is process-global; this test owns it within its own
        // set/clear pair (no other test touches it).
        clear_device_lost_flag();
        assert!(!device_lost_flag());
        DEVICE_LOST.store(true, Ordering::Release);
        assert!(device_lost_flag());
        clear_device_lost_flag();
        assert!(!device_lost_flag());
    }

    #[test]
    fn fallback_descriptors_are_parity_then_spec_baseline() {
        // WEB-NEXTJS-21G ladder shape: tier 1 re-requests exactly what the
        // adapter advertises (recovers advertised-below-default adapters);
        // tier 2 drops to the WebGPU-spec baseline every conformant
        // implementation must grant (recovers adapters whose requestDevice
        // refuses their own advertised limits — the parity tier re-requests
        // the lie, so only a baseline tier can recover that shape).
        let mut advertised = wgpu::Limits::default();
        advertised.max_texture_dimension_2d = 16384; // the advertised lie
        advertised.max_compute_invocations_per_workgroup = 1024;
        let [parity, baseline] = GpuContext::fallback_descriptors(advertised.clone());
        assert_eq!(parity.required_limits, advertised);
        assert_eq!(baseline.required_limits, wgpu::Limits::default());
        // The baseline tier must never itself opt in above spec defaults —
        // that is the whole point of the tier.
        assert_eq!(baseline.required_limits.max_texture_dimension_2d, 8192);
        assert_eq!(
            baseline
                .required_limits
                .max_compute_invocations_per_workgroup,
            256
        );
    }

    #[test]
    fn ordinary_errors_do_not_classify_as_device_loss() {
        assert!(!is_device_lost_error(&anyhow::anyhow!(
            "scene has no pages"
        )));
        let wrapped = anyhow::anyhow!("inner").context("GPU device init");
        assert!(!is_device_lost_error(&wrapped));
    }
}
