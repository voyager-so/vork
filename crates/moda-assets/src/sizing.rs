//! Raster sizing policy — line-for-line port of
//! `WebGPUTextureManager.resolveImageBitmap` / `calculateMinimumDimensions` /
//! `clampMinRasterSize` (`WebGPUTextureManager.ts:677-939`) and
//! `renderers/webgpu/utils/svgRasterizationSizing.ts`.
//!
//! COPY THE BUGS: the small-image/SVG minimum-dimension upscale applies even
//! at 1x export scale (TEXTURE_PIXEL_RATIO is the constant 2, not DPR), so a
//! 96x64 JPEG uploads as a 512x342 resampled texture with scaleFactor
//! (5.333…, 5.34375) — the golden's smoothing comes from this chain.
//!
//! …but not the unbounded one: [`svg_raster_ceiling`] caps an SVG raster to the
//! footprint it actually draws at. That was landed on BOTH sides at once, so the
//! port stays line-for-line — if you change one, change the other.

use crate::constants::{
    live_max_texture_dimension, MIN_SVG_DIMENSION, SMALL_IMAGE_THRESHOLD, SVG_RASTER_ZOOM_HEADROOM,
    TEXTURE_PIXEL_RATIO,
};

/// `calculateMinimumDimensions` (`WebGPUTextureManager.ts:693-720`).
/// JS `Math.round` = half away from zero.
fn calculate_minimum_dimensions(natural_w: u32, natural_h: u32, min_dimension: u32) -> (f64, f64) {
    let (nw, nh) = (natural_w as f64, natural_h as f64);
    let min = min_dimension as f64;
    if nw >= min && nh >= min {
        return (nw, nh);
    }
    let aspect = nw / nh;
    if aspect > 1.0 {
        let width = nw.max(min);
        let height = (width / aspect).round();
        (width, height)
    } else {
        let height = nh.max(min);
        let width = (height * aspect).round();
        (width, height)
    }
}

/// `clampMinRasterSize` (`WebGPUTextureManager.ts:677-688`), capped at the
/// LIVE device texture limit (browser parity — the TS manager's
/// `maxTextureDimension` is set from `device.limits.maxTextureDimension2D`;
/// see `constants::live_max_texture_dimension`).
pub fn clamp_min_raster_size(req: Option<(u32, u32)>) -> Option<(u32, u32)> {
    clamp_min_raster_size_with_max(req, live_max_texture_dimension())
}

fn clamp_min_raster_size_with_max(req: Option<(u32, u32)>, max_dim: u32) -> Option<(u32, u32)> {
    let (w, h) = req?;
    let w = w.max(1).min(max_dim);
    let h = h.max(1).min(max_dim);
    Some((w, h))
}

/// `resolveImageBitmap`'s target-size computation
/// (`WebGPUTextureManager.ts:892-920`). Returns the texture dimensions the
/// browser would upload for an image with the given natural size, capped at
/// the LIVE device texture limit (browser parity — see
/// [`clamp_min_raster_size`]). On a healthy device the live limit equals the
/// pinned 16384 and this is byte-identical to the historical behavior the
/// goldens capture; it diverges only when the requestDevice ladder granted a
/// lower tier (WEB-NEXTJS-21G), where an un-clamped target would fail texture
/// creation outright.
pub fn resolve_target_size(
    natural_w: u32,
    natural_h: u32,
    is_svg: bool,
    requested_min: Option<(u32, u32)>,
) -> (u32, u32) {
    resolve_target_size_with_max(
        natural_w,
        natural_h,
        is_svg,
        requested_min,
        live_max_texture_dimension(),
    )
}

fn resolve_target_size_with_max(
    natural_w: u32,
    natural_h: u32,
    is_svg: bool,
    requested_min: Option<(u32, u32)>,
    max_dim: u32,
) -> (u32, u32) {
    let is_small = natural_w < SMALL_IMAGE_THRESHOLD || natural_h < SMALL_IMAGE_THRESHOLD;

    let mut target_w = natural_w as f64;
    let mut target_h = natural_h as f64;

    if (is_svg || is_small) && natural_w > 0 && natural_h > 0 {
        let min_dim = if is_svg {
            MIN_SVG_DIMENSION
        } else {
            SMALL_IMAGE_THRESHOLD
        };
        let (w, h) = calculate_minimum_dimensions(natural_w, natural_h, min_dim);
        target_w = target_w.max(w * TEXTURE_PIXEL_RATIO);
        target_h = target_h.max(h * TEXTURE_PIXEL_RATIO);
    } else if natural_w > max_dim || natural_h > max_dim {
        let scale = (max_dim as f64 / natural_w as f64).min(max_dim as f64 / natural_h as f64);
        target_w = (natural_w as f64 * scale).floor();
        target_h = (natural_h as f64 * scale).floor();
    }

    if is_svg {
        if let Some((rw, rh)) = clamp_min_raster_size_with_max(requested_min, max_dim) {
            target_w = target_w.max(rw as f64);
            target_h = target_h.max(rh as f64);
        }
        // …then a CEILING from the same footprint. Everything above this point sizes
        // an SVG from its NATURAL dimensions, which say nothing about how large the
        // node draws: an SVG authored on a big artboard is upscaled x2 anyway (the
        // max_dim downscale is an `else if`, unreachable for any SVG), so a huge
        // artboard in a small box rasters orders of magnitude past what it displays
        // and resvg pays for every pixel of it — filter primitives included.
        //
        // Only applies when the caller supplied a footprint. A tiling pattern or a
        // node with no `fillPatternNaturalDimensions` passes None and keeps the
        // historical size exactly, since there is no displayed extent to cap to.
        //
        // The positivity condition matches the minimum-dimension branch above. This
        // is a `pub fn`, so a zero natural axis has to be total rather than merely
        // survivable: the aspect the ceiling is built from is undefined at zero, and
        // while `f64::max` happens to discard the resulting NaN (leaving the cap at
        // the footprint), depending on that is a landmine for anyone who rewrites
        // this as a comparison.
        if let Some(footprint) = requested_min.filter(|_| natural_w > 0 && natural_h > 0) {
            let (cap_w, cap_h) = svg_raster_ceiling(natural_w, natural_h, footprint);
            if target_w > cap_w || target_h > cap_h {
                // Uniform, like the max_dim branch: the pattern sampler is fed a
                // per-axis scale factor, but an anisotropic squeeze would resample
                // one axis harder than the other for no reason.
                let scale = (cap_w / target_w).min(cap_h / target_h);
                target_w = (target_w * scale).floor().max(1.0);
                target_h = (target_h * scale).floor().max(1.0);
            }
        }
    }

    let w = (target_w.floor() as u32).clamp(1, max_dim);
    let h = (target_h.floor() as u32).clamp(1, max_dim);
    (w, h)
}

/// The largest raster worth producing for an SVG that draws at `footprint`.
///
/// `footprint` is the on-screen extent in device pixels
/// ([`compute_desired_svg_raster_size`]), grown by [`SVG_RASTER_ZOOM_HEADROOM`]
/// for zoom. It is floored by the natural aspect scaled so the SHORTER axis
/// reaches `MIN_SVG_DIMENSION * TEXTURE_PIXEL_RATIO`, which is what keeps the
/// minimum-dimension policy intact: a 256x256 icon drawn at 11px still rasters
/// 1024x1024 exactly as before, because its ceiling floors to 1024 rather than
/// collapsing to the 88px its footprint alone would allow.
///
/// Note this floor is deliberately NOT [`calculate_minimum_dimensions`]: that
/// returns the natural size unchanged once both axes clear `MIN_SVG_DIMENSION`,
/// so using it would floor the ceiling at the very natural-size product this
/// cap exists to bound, and the cap could never bind.
///
/// Both natural axes must be non-zero — the caller enforces it, since the aspect
/// this scales is undefined at zero.
fn svg_raster_ceiling(natural_w: u32, natural_h: u32, footprint: (u32, u32)) -> (f64, f64) {
    let (nw, nh) = (natural_w as f64, natural_h as f64);
    let min_axis = MIN_SVG_DIMENSION as f64 * TEXTURE_PIXEL_RATIO;
    // Scale the natural aspect until the shorter axis lands on `min_axis`; `max`
    // picks the shorter axis's factor.
    let floor_scale = (min_axis / nw).max(min_axis / nh);
    (
        (footprint.0 as f64 * SVG_RASTER_ZOOM_HEADROOM).max(nw * floor_scale),
        (footprint.1 as f64 * SVG_RASTER_ZOOM_HEADROOM).max(nh * floor_scale),
    )
}

/// `getEffectiveSvgRasterPixelRatio` (`svgRasterizationSizing.ts:17-34`).
pub fn effective_svg_raster_pixel_ratio(export_pixel_ratio: Option<f64>) -> f64 {
    let screen = TEXTURE_PIXEL_RATIO;
    match export_pixel_ratio {
        Some(e) if e.is_finite() && e > 0.0 => screen.max(e),
        _ => screen,
    }
}

/// `computeDesiredSvgRasterSize` (`svgRasterizationSizing.ts:44-74`).
/// `pattern_scale`/`absolute_scale` per axis; returns None on the same
/// non-finite / non-positive guards as the TS.
pub fn compute_desired_svg_raster_size(
    natural_w: f64,
    natural_h: f64,
    pattern_scale: (f64, f64),
    absolute_scale: (f64, f64),
    pixel_ratio: f64,
    max_texture_dimension: Option<f64>,
) -> Option<(u32, u32)> {
    if !natural_w.is_finite() || !natural_h.is_finite() || natural_w <= 0.0 || natural_h <= 0.0 {
        return None;
    }
    if !pattern_scale.0.is_finite() || !pattern_scale.1.is_finite() {
        return None;
    }
    if !absolute_scale.0.is_finite() || !absolute_scale.1.is_finite() {
        return None;
    }
    if !pixel_ratio.is_finite() || pixel_ratio <= 0.0 {
        return None;
    }
    let scale_x = pattern_scale.0.abs() * absolute_scale.0.abs() * pixel_ratio;
    let scale_y = pattern_scale.1.abs() * absolute_scale.1.abs() * pixel_ratio;
    if !scale_x.is_finite() || !scale_y.is_finite() || scale_x <= 0.0 || scale_y <= 0.0 {
        return None;
    }
    let mut width = (natural_w * scale_x).ceil().max(1.0);
    let mut height = (natural_h * scale_y).ceil().max(1.0);
    if let Some(max_dim) = max_texture_dimension {
        if max_dim.is_finite() && max_dim > 0.0 {
            width = width.min(max_dim.floor()).max(1.0);
            height = height.min(max_dim.floor()).max(1.0);
        }
    }
    Some((width as u32, height as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_raster_downscales_to_the_reduced_device_limit() {
        // WEB-NEXTJS-21G: on a baseline-tier device (granted 8192) a photo the
        // pinned 16384 policy would pass through untouched must downscale
        // aspect-preserved to the GRANT, or texture creation fails outright.
        assert_eq!(
            resolve_target_size_with_max(10000, 5000, false, None, 8192),
            (8192, 4096)
        );
        // At the pinned max the same input is untouched (the healthy-device
        // behavior the goldens capture).
        assert_eq!(
            resolve_target_size_with_max(10000, 5000, false, None, 16384),
            (10000, 5000)
        );
    }

    #[test]
    fn svg_requested_min_clamps_to_the_reduced_device_limit() {
        // The SVG min-raster request rides through `clamp_min_raster_size`,
        // which must honor the live grant too.
        assert_eq!(
            clamp_min_raster_size_with_max(Some((16384, 16384)), 8192),
            Some((8192, 8192))
        );
        assert_eq!(
            resolve_target_size_with_max(400, 400, true, Some((16384, 16384)), 8192),
            (8192, 8192)
        );
    }

    #[test]
    fn small_jpeg_96x64_uploads_at_512x342() {
        // fit-align landscape-96x64.jpg: aspect 1.5 > 1 → width=max(96,256)=256,
        // height=round(256/1.5)=171 → target=(max(96,512), max(64,342)).
        assert_eq!(resolve_target_size(96, 64, false, None), (512, 342));
    }

    #[test]
    fn small_png_64x64_uploads_at_512x512() {
        assert_eq!(resolve_target_size(64, 64, false, None), (512, 512));
    }

    #[test]
    fn svg_80x80_uploads_at_1024() {
        // min SVG dim 512 → (512,512) → ×2 → 1024×1024; the 160×160 requested
        // min from computeDesiredSvgRasterSize does not raise it.
        assert_eq!(
            resolve_target_size(80, 80, true, Some((160, 160))),
            (1024, 1024)
        );
    }

    #[test]
    fn svg_requested_min_raises_when_larger() {
        assert_eq!(
            resolve_target_size(80, 80, true, Some((1500, 1500))),
            (1500, 1500)
        );
        // requested min only applies to SVG (resolveImageBitmap line 914)
        assert_eq!(
            resolve_target_size(300, 300, false, Some((1500, 1500))),
            (300, 300)
        );
    }

    #[test]
    fn big_artboard_svg_in_a_small_box_is_capped_to_its_footprint() {
        // The canvas c4d6996f lockup: a 4255x5971 Figma export drawn in a
        // 157x255 box (fillPatternScale 0.046 → a 392x550 device footprint).
        // The natural-size x2 policy alone asks for 8510x11942 = 101.6 MP /
        // 407 MB, and resvg runs the asset's feMorphology(166) +
        // feGaussianBlur(346/639/40) chain over every pixel of it — a
        // main-thread hang, not merely wasted memory.
        let (w, h) = resolve_target_size(4255, 5971, true, Some((392, 550)));
        assert_eq!((w, h), (1567, 2200));
        // Under 4 MP, i.e. ~1.5% of the area the uncapped policy asked for.
        assert!((w as f64 * h as f64) < 4_000_000.0);
        // Aspect preserved within a pixel of rounding, so the pattern sampler
        // is not fed an anisotropic squeeze.
        let (natural_aspect, capped_aspect) = (4255.0 / 5971.0, w as f64 / h as f64);
        assert!((natural_aspect - capped_aspect).abs() < 0.001);
    }

    #[test]
    fn small_icon_svg_keeps_its_full_min_dimension_raster() {
        // The regression this cap must not cause. `check.svg` is 256x256 drawn
        // at 11px, a 22x22 footprint — 4x that is 88px, far below the 1024 the
        // minimum-dimension policy has always produced. The ceiling's
        // shorter-axis floor is what holds it at 1024, so zooming into an icon
        // is exactly as crisp as before.
        assert_eq!(
            resolve_target_size(256, 256, true, Some((22, 22))),
            (1024, 1024)
        );
    }

    #[test]
    fn svg_drawn_near_its_natural_size_is_untouched() {
        // A 1064x585 SVG drawn at a 1062x584 footprint is already displaying
        // ~1:1; the cap must not touch the historical x2 raster.
        assert_eq!(
            resolve_target_size(1064, 585, true, Some((1062, 584))),
            (2128, 1170)
        );
    }

    #[test]
    fn zero_natural_axis_stays_finite_and_never_reaches_the_ceiling() {
        // `resolve_target_size` is public, so a degenerate decode must be total
        // rather than merely survivable. The ceiling's aspect is undefined at zero,
        // so it is skipped and the footprint floor alone decides — no NaN, no zero,
        // no panic on the `as u32` cast.
        // The floor is a per-axis `max`, so a zero axis takes the footprint while a
        // non-zero one keeps its natural size.
        for (nw, nh, want) in [
            (0u32, 5971u32, (392u32, 5971u32)),
            (4255, 0, (4255, 550)),
            (0, 0, (392, 550)),
        ] {
            let (w, h) = resolve_target_size(nw, nh, true, Some((392, 550)));
            assert_eq!((w, h), want, "natural {nw}x{nh}");
        }
        // And with no footprint either, the clamp floor still yields a usable 1x1
        // rather than a zero-sized texture.
        assert_eq!(resolve_target_size(0, 0, true, None), (1, 1));
    }

    #[test]
    fn svg_without_a_footprint_keeps_the_historical_size() {
        // Tiling patterns and nodes with no `fillPatternNaturalDimensions` pass
        // None: there is no displayed extent to cap against, so behavior must be
        // byte-identical to before the cap existed.
        assert_eq!(resolve_target_size(4255, 5971, true, None), (8510, 11942));
    }

    #[test]
    fn large_raster_scales_down_proportionally() {
        // Q-89: the runtime limit is the device limit (16384), not the dead
        // TEXTURE_CONSTANTS 2048 default — 4096x2048 uploads at full size...
        assert_eq!(resolve_target_size(4096, 2048, false, None), (4096, 2048));
        // ...and only beyond-device-limit rasters downscale proportionally.
        assert_eq!(
            resolve_target_size(20000, 10000, false, None),
            (16384, 8192)
        );
    }

    #[test]
    fn mid_size_raster_untouched() {
        assert_eq!(resolve_target_size(300, 400, false, None), (300, 400));
    }

    #[test]
    fn clamp_to_max_dimension() {
        // SVG 4000x100: aspect 40 → width=max(4000,512)=4000, h=round(4000/40)=100
        // → ×2 → (8000, 200); within the Q-89 device limit (16384) so no clamp.
        assert_eq!(resolve_target_size(4000, 100, true, None), (8000, 200));
        // The clamp itself still applies at the device limit — per-axis,
        // NOT proportional, copying the TS final clamp (10000x100 SVG →
        // min-dims ×2 → (20000, 200) → w clamps alone).
        assert_eq!(resolve_target_size(10000, 100, true, None), (16384, 200));
    }

    #[test]
    fn desired_svg_raster_size_ports_guards() {
        assert_eq!(
            compute_desired_svg_raster_size(80.0, 80.0, (1.0, 1.0), (1.0, 1.0), 2.0, Some(2048.0)),
            Some((160, 160))
        );
        assert_eq!(
            compute_desired_svg_raster_size(0.0, 80.0, (1.0, 1.0), (1.0, 1.0), 2.0, None),
            None
        );
        assert_eq!(
            compute_desired_svg_raster_size(80.0, 80.0, (0.0, 1.0), (1.0, 1.0), 2.0, None),
            None
        );
        // negative scales abs()'d
        assert_eq!(
            compute_desired_svg_raster_size(80.0, 80.0, (-1.0, 1.0), (1.0, -2.0), 2.0, None),
            Some((160, 320))
        );
    }

    #[test]
    fn effective_pixel_ratio_is_max_of_screen_and_export() {
        assert_eq!(effective_svg_raster_pixel_ratio(None), 2.0);
        assert_eq!(effective_svg_raster_pixel_ratio(Some(1.0)), 2.0);
        assert_eq!(effective_svg_raster_pixel_ratio(Some(4.0)), 4.0);
        assert_eq!(effective_svg_raster_pixel_ratio(Some(0.0)), 2.0);
    }
}
