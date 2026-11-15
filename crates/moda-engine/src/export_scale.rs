//! Export scale-capping — the seam that keeps a `page × scale` render inside the
//! GPU's hard limits so a GIANT page down-scales to fit instead of failing
//! allocation.
//!
//! The native lane renders each page as ONE GPU texture whose size is
//! `ceil(page_w × scale) × ceil(page_h × scale)` (`frame.rs`) and reads it back
//! through mapped staging buffers of `padded_bytes_per_row(w) × rows` bytes
//! (`readback.rs`). Two device limits bound that at export scale
//! (`sample_count 4`, `scale ≥ 1`):
//!
//! - **per-axis texture** — `device.limits().max_texture_dimension_2d`
//!   (`min(16384, adapter)` as requested in `device::GpuContext::new`);
//! - **readback buffer** — `device.limits().max_buffer_size` (the wgpu default
//!   256 MB unless bumped), i.e. the 256-aligned row × height must fit ONE
//!   buffer.
//!
//! **The readback-buffer budget is no longer a hard requirement of the engine.**
//! `readback::read_texture_rgba_staged` now copies a finished texture out in
//! consecutive row BANDS, each sized to fit `max_buffer_size`, so a render whose
//! full readback would exceed one buffer no longer fails allocation — it reads
//! back in several bands, bit-identically. Passing `max_buffer_bytes` here is
//! therefore a *conservative* choice by the caller (it bounds peak host memory
//! and encode time at ≈64 MP), NOT the thing that keeps the render from
//! crashing; `0` disables it and leaves the axis cap in force. The per-axis
//! limit is the one hard wall left — a 16384² render is ~1 GB of pixels host-side
//! whichever way it is read back. Relaxing the callers' buffer budget wants the
//! encode/RSS measurement tracked alongside the banding change, not a bare edit.
//!
//! Rather than gate oversized pages to the browser (whose Canvas2D tile-stitch
//! fallback is an OOM source), we cap the effective scale so BOTH fit — for
//! slide/PPTX export this is strictly BETTER: nothing benefits from a >16k /
//! >64 MP embed, and giant rasters bloat the file.
//!
//! This is the single source of truth for the clamp: `moda-cli`'s render lanes
//! and `moda-mobile-ffi`'s session both call `clamp_export_scale`. (True tiling
//! — rendering a giant page in viewport chunks and stitching CPU-side to keep
//! full resolution — is the documented follow-up; scale-capping ships first.)

/// Per-axis GPU texture limit the engine requests (`device.rs`). Re-exported
/// from the canonical `moda_assets::constants` so the two can't drift — this is
/// the hardware floor (16384). Production callers pass the LIVE
/// `device.limits().max_texture_dimension_2d` instead of this constant, so the
/// clamp tracks the actual adapter; the constant is the shared default/floor
/// (used by tests and any caller without a device handle).
pub use moda_assets::constants::MAX_TEXTURE_DIMENSION;

/// Above this many OUTPUT pixels, MSAA (`sample_count 4`) is dropped to 1.
///
/// 4× MSAA multiplies the *transient* GPU allocation: on top of the resolve
/// target and the readback buffer (both `area × 4` bytes, unavoidable), the
/// multisampled color (`area × 4 × 4`) and the 4-sample depth (`area × 4 × 4`)
/// add ~`area × 32` bytes. At the axis cap (16384² ≈ 268 MP) that is ~8.6 GB
/// vs. ~2.1 GB at `sample_count 1` — a 4× swing that OOMs the render fleet.
///
/// Correctness-first tradeoff (directive item 3): a raster this large is viewed
/// heavily downscaled (or is already near 1:1 with high-resolution content), so
/// the lost geometric edge-AA is not visible, whereas an out-of-memory abort is
/// a hard failure. Normal large exports (e.g. a 4K slide at 2× = 33 MP, an
/// 8000×6000 poster = 48 MP) stay under this and keep 4× MSAA.
pub const MSAA_MAX_OUTPUT_PIXELS: u64 = 64_000_000;

/// A length in output pixels (`len × scale`) rounded up to whole pixels, after rounding to 1e-6 so float error
/// doesn't add one: 960 × (300 / 72) is 4000.0000000000005, and a plain `ceil` made a 960-wide page's 300 dpi PNG
/// 4001 px wide (a 1920-wide slide 8001). Every output size (`output_dims`, the camera's `viewport × dpr`, the
/// page plans) rounds with this, so they agree.
#[inline]
pub fn px_ceil(v: f64) -> f64 {
    ((v * 1e6).round() / 1e6).ceil()
}

/// Output pixel dimensions for a `(page w×h, scale)` render — [`px_ceil`] per axis,
/// mirroring `frame.rs`'s `width_px`/`height_px`.
#[inline]
pub fn output_dims(w: f64, h: f64, scale: f64) -> (u32, u32) {
    (
        px_ceil(w * scale).max(1.0) as u32,
        px_ceil(h * scale).max(1.0) as u32,
    )
}

/// Clamp `scale` so that, at the resulting output size, ALL of these hold:
/// each axis `≤ max_axis` (texture limit), `out_w × out_h ≤ max_pixels`
/// (optional pixel budget), and `padded_bytes_per_row(out_w) × out_h ≤
/// max_buffer_bytes` (readback-buffer limit).
///
/// Returns `scale` UNCHANGED when it already fits (the common case — a normal
/// page at a normal scale), so callers get byte-identical behavior below the
/// threshold. For an oversized page the result is the largest scale that fits,
/// preserving aspect (both axes shrink by the same factor — a single uniform
/// `scale` drives both).
///
/// `max_pixels == 0` / `max_buffer_bytes == 0` disable those respective budgets.
pub fn clamp_export_scale(
    scale: f64,
    w: f64,
    h: f64,
    max_axis: u32,
    max_pixels: u64,
    max_buffer_bytes: u64,
) -> f64 {
    // Guard degenerate dims so the analytic divisions below stay finite.
    let w = w.max(1e-6);
    let h = h.max(1e-6);
    let axis = max_axis as f64;

    // Analytic axis fit: shrink so neither axis exceeds the texture limit.
    let mut s = scale.min(axis / w).min(axis / h);

    let out_w = |s: f64| px_ceil(w * s);
    let out_h = |s: f64| px_ceil(h * s);
    // Exact readback-buffer bytes for a given scale (256-aligned rows), using
    // the canonical `readback::padded_bytes_per_row` so this can't drift from
    // the real allocation.
    let buffer_bytes = |s: f64| {
        crate::readback::padded_bytes_per_row(out_w(s).max(1.0) as u32) as u64
            * out_h(s).max(1.0) as u64
    };
    let fits = |s: f64| {
        let (ow, oh) = (out_w(s), out_h(s));
        ow <= axis
            && oh <= axis
            && (max_pixels == 0 || ow * oh <= max_pixels as f64)
            && (max_buffer_bytes == 0 || buffer_bytes(s) <= max_buffer_bytes)
    };

    // Analytic pre-shrink so the exact fits() loop below starts within a
    // rounding-nudge of fitting. Pixel budget: uniform-area sqrt shrink.
    if max_pixels != 0 {
        let area = out_w(s) * out_h(s);
        if area > max_pixels as f64 {
            s *= (max_pixels as f64 / area).sqrt();
        }
    }
    // Readback buffer: the exact bytes `padded_bytes_per_row(out_w)·out_h` are
    // bounded below by BOTH the wide-page regime (`4·w·h·s²`, padding
    // negligible) AND the thin-page regime (`256·h·s`, where a single
    // 256-padded row dominates), so a fitting `s` must satisfy both — cap by the
    // min of the two. This converges the analytic step for ANY aspect ratio; a
    // sqrt-only (wide-only) shrink under-corrects a thin/tall page and can leave
    // the bounded loop unable to converge. The thin term is defensive on real
    // hardware (the axis cap already bounds `out_h ≤ max_axis`, so `256·out_h`
    // can't reach a 256 MB buffer), but keeps the clamp correct for any
    // max_axis/max_buffer pair. The exact fits() loop then absorbs residual
    // ceil / padding rounding.
    if max_buffer_bytes != 0 {
        let max_buf = max_buffer_bytes as f64;
        let limit_wide = (max_buf / (4.0 * w * h)).sqrt();
        let limit_thin = max_buf / (256.0 * h);
        s = s.min(limit_wide).min(limit_thin);
    }

    // Absorb `ceil()` / row-padding / floating-point rounding against every
    // limit: `w × (axis / w)` can land a hair above `axis`, and the analytic
    // area shrink can land just above the exact padded-buffer size. Bounded so a
    // pathological cap can never spin — 64·ln(0.999) ≈ -0.064, i.e. at most a
    // ~6% extra shrink, far more than the rounding ever needs.
    for _ in 0..64 {
        if s <= 0.0 || fits(s) {
            break;
        }
        s *= 0.999;
    }
    s
}

/// MSAA sample count for a render whose OUTPUT is `out_w × out_h`: drop a
/// requested 4× to 1× above [`MSAA_MAX_OUTPUT_PIXELS`] to bound transient VRAM
/// (see that constant). `sample_count 1` is returned unchanged.
#[inline]
pub fn capped_sample_count(requested: u32, out_w: u32, out_h: u32) -> u32 {
    if requested > 1 && (out_w as u64) * (out_h as u64) > MSAA_MAX_OUTPUT_PIXELS {
        1
    } else {
        requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readback::padded_bytes_per_row;

    const CAP: u32 = MAX_TEXTURE_DIMENSION; // 16384

    /// A whole length at a density is that many pixels: 960 and 1920 at 300 dpi are 4000 and 8000 px (a plain
    /// `ceil` of 960 × (300 / 72) = 4000.0000000000005 gave 4001); a fractional one still rounds up.
    #[test]
    fn output_dims_ignore_float_error() {
        let dpi300 = 300.0 / 72.0;
        assert_eq!(output_dims(960.0, 540.0, dpi300), (4000, 2250));
        assert_eq!(output_dims(1920.0, 1080.0, dpi300), (8000, 4500));
        assert_eq!(output_dims(595.2756, 841.8898, dpi300), (2481, 3508));
        assert_eq!(output_dims(100.5, 10.0, 1.0), (101, 10));
    }
                                            // The CLI passes no separate pixel budget (0) and the device readback-buffer
                                            // limit (wgpu default 256 MB) as the binding area constraint.
    const NO_PIXELS: u64 = 0;
    const BUF_256MB: u64 = 256 * 1024 * 1024;
    // Axis² buffer budget — effectively "axis-only" (a 16384² readback is 1 GB,
    // so this permits everything the axis cap does).
    const HUGE_BUF: u64 = (CAP as u64) * (CAP as u64) * 4;

    fn buffer_bytes(w: f64, h: f64, s: f64) -> u64 {
        let (ow, oh) = output_dims(w, h, s);
        padded_bytes_per_row(ow) as u64 * oh as u64
    }

    /// A normal page at a normal scale is returned UNCHANGED (byte-identical
    /// path — no cap below the threshold), even with the 256 MB buffer budget.
    #[test]
    fn normal_page_scale_unchanged() {
        assert_eq!(
            clamp_export_scale(2.0, 1920.0, 1080.0, CAP, NO_PIXELS, BUF_256MB),
            2.0
        );
        assert_eq!(
            clamp_export_scale(1.0, 800.0, 600.0, CAP, NO_PIXELS, BUF_256MB),
            1.0
        );
        assert_eq!(
            clamp_export_scale(4.0, 1000.0, 1000.0, CAP, NO_PIXELS, BUF_256MB),
            4.0
        );
    }

    /// A page that lands EXACTLY on the axis cap at the requested scale is not
    /// shrunk when the buffer budget permits it: `ceil(16384 × 1) == 16384`.
    #[test]
    fn exactly_at_cap_unchanged() {
        let s = clamp_export_scale(1.0, 16384.0, 16384.0, CAP, NO_PIXELS, HUGE_BUF);
        assert_eq!(s, 1.0, "a page exactly at the cap must not be down-scaled");
        assert_eq!(output_dims(16384.0, 16384.0, s), (16384, 16384));
    }

    /// One logical pixel over the axis cap forces a down-scale so the ceil'd
    /// output fits, on BOTH axes.
    #[test]
    fn just_above_cap_shrinks_to_fit() {
        let s = clamp_export_scale(1.0, 16385.0, 16385.0, CAP, NO_PIXELS, HUGE_BUF);
        assert!(s < 1.0, "16385px @1x must cap below 1.0 (got {s})");
        let (w, h) = output_dims(16385.0, 16385.0, s);
        assert!(
            w <= CAP && h <= CAP,
            "capped output {w}x{h} still exceeds {CAP}"
        );
    }

    /// The requested export scale multiplies the page: 12000 × 2 = 24000 ≫ cap.
    /// With the 256 MB buffer this caps HARDER than the axis alone (≈8000², the
    /// binding constraint the giant-square render actually hit).
    #[test]
    fn oversized_by_scale_caps_and_fits() {
        let s = clamp_export_scale(2.0, 12000.0, 12000.0, CAP, NO_PIXELS, BUF_256MB);
        let (w, h) = output_dims(12000.0, 12000.0, s);
        assert!(w <= CAP && h <= CAP, "output {w}x{h} exceeds {CAP}");
        assert!(
            buffer_bytes(12000.0, 12000.0, s) <= BUF_256MB,
            "readback {w}x{h} exceeds the 256 MB buffer"
        );
        // Buffer-bound to ≈8000² ⇒ scale ≈0.66, well under both 2× and the axis.
        assert!(s > 0.0 && s < 1.0, "expected buffer-bound s<1, got {s}");
    }

    /// A thin, very wide page is pinned by the width axis only; neither area
    /// budget triggers, so the axis-ceil absorption alone must keep it in bounds.
    #[test]
    fn thin_wide_page_axis_only() {
        let s = clamp_export_scale(1.0, 30000.0, 200.0, CAP, NO_PIXELS, BUF_256MB);
        let (w, h) = output_dims(30000.0, 200.0, s);
        assert!(w <= CAP, "wide axis {w} exceeds {CAP}");
        assert!(h <= CAP, "short axis {h} exceeds {CAP}");
        assert!(w >= CAP - 1, "should render right up to the cap, got {w}");
    }

    /// The readback buffer is the binding limit for a big SQUARE page: a 16384²
    /// readback is 1 GB ≫ 256 MB, so the clamp must shrink until the exact
    /// 256-aligned buffer fits — this is the real wgpu error the fix prevents.
    #[test]
    fn readback_buffer_is_binding() {
        let s = clamp_export_scale(1.0, 16384.0, 16384.0, CAP, NO_PIXELS, BUF_256MB);
        assert!(
            s < 1.0,
            "square page must be buffer-capped below the axis, got {s}"
        );
        assert!(
            buffer_bytes(16384.0, 16384.0, s) <= BUF_256MB,
            "capped readback still exceeds the 256 MB buffer"
        );
    }

    /// Defensive (Gemini round-2): a very tall/thin page under a HYPOTHETICAL
    /// device whose axis limit is large enough that the readback buffer binds
    /// via the THIN regime (a single 256-padded row × a huge height). A
    /// wide-only (sqrt) analytic pre-shrink under-corrects here and the bounded
    /// loop can't converge; the thin-limit term must keep the result fitting.
    /// (Real hardware caps the axis at 16384, so `256·out_h ≤ 4 MB` and this
    /// regime is unreachable — but the clamp stays correct for any limits.)
    #[test]
    fn thin_tall_page_buffer_regime_converges() {
        let big_axis: u32 = 4_000_000; // hypothetical; real HW is 16384
        let (w, h) = (4.0_f64, 3_000_000.0_f64);
        let s = clamp_export_scale(1.0, w, h, big_axis, NO_PIXELS, BUF_256MB);
        let (ow, oh) = output_dims(w, h, s);
        assert!(ow <= big_axis && oh <= big_axis);
        assert!(
            buffer_bytes(w, h, s) <= BUF_256MB,
            "thin/tall readback {} exceeds the 256 MB buffer (s={s})",
            buffer_bytes(w, h, s)
        );
    }

    /// The result always yields an allocatable target (axis AND buffer) for a
    /// sweep of pathological sizes / scales (fuzz-lite boundary walk).
    #[test]
    fn always_fits_for_extreme_inputs() {
        for &(w, h, scale) in &[
            (16384.0, 16384.0, 4.0),
            (16383.0, 16385.0, 1.0),
            (50000.0, 1.0, 3.0),
            (1.0, 50000.0, 3.0),
            (25000.0, 25000.0, 1.0),
            (16385.0, 16385.0, 1.0),
            (13000.0, 13000.0, 2.0),
            (9000.0, 8000.0, 1.0),
        ] {
            let s = clamp_export_scale(scale, w, h, CAP, NO_PIXELS, BUF_256MB);
            let (ow, oh) = output_dims(w, h, s);
            assert!(
                ow <= CAP && oh <= CAP && buffer_bytes(w, h, s) <= BUF_256MB,
                "({w}x{h}@{scale}) -> {ow}x{oh} still out of bounds (s={s})"
            );
        }
    }

    /// Both budgets `== 0` ⇒ axis-only: the axis-exact square is allowed (its
    /// 1 GB readback would fail on a real device, but that is the buffer budget's
    /// job — this asserts the disable path).
    #[test]
    fn zero_budgets_is_axis_only() {
        let s = clamp_export_scale(1.0, 16384.0, 16384.0, CAP, 0, 0);
        assert_eq!(s, 1.0);
    }

    /// MSAA is dropped only above the pixel threshold, and never below.
    #[test]
    fn msaa_drop_threshold() {
        // 8000×8000 = 64 MP == threshold (not strictly greater) → keep 4×.
        assert_eq!(capped_sample_count(4, 8000, 8000), 4);
        // 8001×8000 > 64 MP → drop to 1×.
        assert_eq!(capped_sample_count(4, 8001, 8000), 1);
        // A 4K slide at 2× (7680×4320 = 33 MP) keeps MSAA.
        assert_eq!(capped_sample_count(4, 7680, 4320), 4);
        // The giant capped square drops MSAA.
        assert_eq!(capped_sample_count(4, 16384, 16384), 1);
        // sample_count 1 is never touched.
        assert_eq!(capped_sample_count(1, 16384, 16384), 1);
    }
}
