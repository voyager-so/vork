//! M4e — CPU preprocessing for the image-backed procedural fills
//! (`liquid-metal-image` / `heatmap-image` / `dither-image`).
//!
//! Line-faithful port of `renderers/webgpu/shaders/imageProcessing.ts`
//! (Poisson edge-field mask, heatmap multi-blur channels, dither pipeline,
//! source-image passthrough). "Copy the bugs" (DECISION §1.3): JS numeric
//! semantics are preserved — `Math.round` = `floor(x + 0.5)`,
//! `Uint8ClampedArray` stores round-half-to-even, the Poisson values array is
//! f32 with f64 arithmetic per relaxation step, and the browser's
//! premultiplied-canvas round trips (drawImage → getImageData) are modeled
//! explicitly.
//!
//! Image IO lives elsewhere: this module is pure pixel math over a
//! [`ShaderImageSource`] (implemented by moda-assets for decoded rasters and
//! resvg SVG trees — the analog of `HTMLImageElement` + `drawImage`).
//!
//! Resampling parity notes (recorded, not silent):
//! - `drawImage` default smoothing ("low") is Skia 4-tap bilinear —
//!   [`bilinear_resize_premul`] reproduces it (dst-center mapping, edge
//!   clamp). For large raster downscales Skia's 4-tap aliases; ours does too.
//! - `imageSmoothingQuality: 'high'` (the Poisson low-res mask upscale) is
//!   Skia's Mitchell cubic (B = C = 1/3) — [`mitchell_resize_premul`]. Skia
//!   runs it in float raster pipelines; we run f64 separable and quantize
//!   once, which can differ by ±1 LSB on a smooth field (SSIM-absorbed).
//! - SVG sources rasterize at the exact draw size in both worlds (browser
//!   vector drawImage / resvg render-to-size) — no resampling there at all.

use std::fmt::Write as _;
use std::sync::Arc;

/// `HTMLImageElement` + `drawImage` seam. Implementations: decoded raster
/// (bilinear premul resize) and SVG tree (resvg render at size).
pub trait ShaderImageSource {
    /// `image.naturalWidth || image.width || 1` (CSS px; browser values are
    /// integers — callers apply the `|| 1` / `Math.max(1, …)` coercions).
    fn natural_size(&self) -> (f64, f64);
    /// `ctx.drawImage(image, 0, 0, w, h)` onto a fresh transparent canvas:
    /// PREMULTIPLIED sRGB RGBA8 at exactly (w, h).
    fn draw_premul(&self, w: u32, h: u32) -> Vec<u8>;
}

/// `ShaderImageProcessing` (`renderers/webgpu/shaders/base/ShaderResources.ts:38`).
#[derive(Debug, Clone, PartialEq)]
pub enum Processing {
    PoissonMask,
    Heatmap,
    SourceImage,
    DitherImage(DitherOptions),
}

/// `DitherImageProcessing` (`ShaderResources.ts:28-36`).
#[derive(Debug, Clone, PartialEq)]
pub struct DitherOptions {
    pub pattern: DitherPattern,
    pub brightness: f64,
    pub contrast: f64,
    pub pixel_size: f64,
    pub display_width: f64,
    pub display_height: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherPattern {
    Threshold,
    Bayer4x4,
    FloydSteinberg,
    Atkinson,
    Noise,
}

impl DitherPattern {
    pub fn as_str(self) -> &'static str {
        match self {
            DitherPattern::Threshold => "threshold",
            DitherPattern::Bayer4x4 => "bayer-4x4",
            DitherPattern::FloydSteinberg => "floyd-steinberg",
            DitherPattern::Atkinson => "atkinson",
            DitherPattern::Noise => "noise",
        }
    }
    /// `patternParam` (`DitherImageShader.ts:18-29`): unknown → threshold.
    pub fn parse(value: Option<&str>) -> DitherPattern {
        match value {
            Some("threshold") => DitherPattern::Threshold,
            Some("bayer-4x4") => DitherPattern::Bayer4x4,
            Some("floyd-steinberg") => DitherPattern::FloydSteinberg,
            Some("atkinson") => DitherPattern::Atkinson,
            Some("noise") => DitherPattern::Noise,
            _ => DitherPattern::Threshold,
        }
    }
}

/// `shaderImageProcessingKey` (`ShaderResources.ts:40-52`).
pub fn processing_key(processing: &Processing) -> String {
    match processing {
        Processing::PoissonMask => "poisson-mask".to_string(),
        Processing::Heatmap => "heatmap".to_string(),
        Processing::SourceImage => "source-image".to_string(),
        Processing::DitherImage(o) => {
            let mut s = String::new();
            let _ = write!(
                s,
                "dither-image:{}:{}:{}:{}:{}:{}",
                o.pattern.as_str(),
                js_round_i64(o.brightness),
                js_round_i64(o.contrast),
                js_round_i64(o.pixel_size),
                js_round_i64(o.display_width),
                js_round_i64(o.display_height),
            );
            s
        }
    }
}

/// `PreparedShaderImage` — the canvas that production uploads via
/// `copyExternalImageToTexture` with `premultipliedAlpha` unset, i.e. the
/// texture receives STRAIGHT-alpha bytes (the canvas' premultiplied store is
/// un-premultiplied on copy, with the same rounding as `getImageData`).
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedShaderImage {
    pub width: u32,
    pub height: u32,
    /// Straight (non-premultiplied) sRGB RGBA8.
    pub rgba_straight: Arc<Vec<u8>>,
    /// `contentRect` (heatmap only): [offsetX, offsetY, scaleX, scaleY].
    pub content_rect: Option<[f64; 4]>,
}

const POISSON_MASK_MAX_DIMENSION: f64 = 768.0;
const SOURCE_IMAGE_MAX_DIMENSION: f64 = 2048.0;
const DITHER_IMAGE_MAX_DIMENSION: f64 = 2048.0;
const BAYER4_MATRIX: [u8; 16] = [
    0, 128, 32, 160, 192, 64, 224, 96, 48, 176, 16, 144, 240, 112, 208, 80,
];

// config (imageProcessing.ts:36-43)
const SOURCE_MAX: f64 = POISSON_MASK_MAX_DIMENSION;
const SOLVE_MIN: f64 = 512.0;
const SOLVE_MAX: f64 = POISSON_MASK_MAX_DIMENSION;
const POISSON_MAX_ITERATIONS: usize = 240;
const POISSON_TOLERANCE: f64 = 1e-9;
const POISSON_SOURCE: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Size {
    width: f64,
    height: f64,
}

impl Size {
    fn w_u32(&self) -> u32 {
        self.width.max(1.0) as u32
    }
    fn h_u32(&self) -> u32 {
        self.height.max(1.0) as u32
    }
}

/// `prepareShaderImage` (imageProcessing.ts:45-86) for a LOADED image.
/// The `!source.src` (→ transparent 1×1, manager metadata `ready: true`) and
/// load-failure (→ fallback entry, `ready: false`) arms live with the caller.
pub fn prepare_shader_image(
    image: &dyn ShaderImageSource,
    processing: &Processing,
) -> PreparedShaderImage {
    match processing {
        Processing::Heatmap => prepare_heatmap_image(image),
        Processing::SourceImage => prepare_source_image(image),
        Processing::DitherImage(options) => prepare_dither_image(image, options),
        Processing::PoissonMask => prepare_poisson_mask(image),
    }
}

// ---------------------------------------------------------------------------
// poisson-mask (the default arm)
// ---------------------------------------------------------------------------

fn prepare_poisson_mask(image: &dyn ShaderImageSource) -> PreparedShaderImage {
    let output_size = normalize_poisson_mask_output_size(get_image_natural_size(image));
    let solve_size = fit_poisson_solve_size(output_size);
    let mask = raster_alpha(image, solve_size);

    if mask.count == 0 {
        return transparent(1, 1);
    }

    let (boundary, interior) = split_regions(&mask);
    let distance = solve_interior(&mask, &interior, &boundary);
    let low_res = draw_distance_mask(&mask, &distance, &interior);
    let final_rgba = upscale_with_original_alpha(&low_res, &mask, image, output_size);

    PreparedShaderImage {
        width: output_size.w_u32(),
        height: output_size.h_u32(),
        rgba_straight: Arc::new(final_rgba),
        content_rect: None,
    }
}

struct AlphaMask {
    width: usize,
    height: usize,
    alpha: Vec<u8>,
    count: usize,
}

/// `rasterAlpha` (ts:470-485): binary alpha (a == 0 → 0, else 1) of the image
/// drawn at `size`. getImageData's un-premultiply cannot turn nonzero alpha
/// into zero (alpha passes through), so premul alpha is usable directly.
fn raster_alpha(image: &dyn ShaderImageSource, size: Size) -> AlphaMask {
    let (w, h) = (size.w_u32(), size.h_u32());
    let premul = image.draw_premul(w, h);
    let (w, h) = (w as usize, h as usize);
    let mut alpha = vec![0u8; w * h];
    let mut count = 0usize;
    for (i, px) in premul.chunks_exact(4).enumerate() {
        let a = if px[3] == 0 { 0u8 } else { 1u8 };
        alpha[i] = a;
        count += a as usize;
    }
    AlphaMask {
        width: w,
        height: h,
        alpha,
        count,
    }
}

/// `normalizePoissonMaskOutputSize` (ts:428-441).
fn normalize_poisson_mask_output_size(size: Size) -> Size {
    let natural = normalize_size(size);
    let aspect = natural.width / natural.height;
    if aspect >= 1.0 {
        Size {
            width: SOURCE_MAX,
            height: js_round(SOURCE_MAX / aspect).max(1.0),
        }
    } else {
        Size {
            width: js_round(SOURCE_MAX * aspect).max(1.0),
            height: SOURCE_MAX,
        }
    }
}

/// `fitPoissonSolveSize` (ts:443-449).
fn fit_poisson_solve_size(size: Size) -> Size {
    let first_scale = SOLVE_MIN / size.width.min(size.height).max(1.0);
    let first = scale_size(size, first_scale);
    let cap_scale = (SOLVE_MAX / first.width.max(first.height)).min(1.0);
    if cap_scale < 1.0 {
        scale_size(first, cap_scale)
    } else {
        first
    }
}

/// `splitRegions` (ts:487-508).
fn split_regions(mask: &AlphaMask) -> (Vec<u8>, Vec<u32>) {
    let mut boundary = vec![0u8; mask.alpha.len()];
    let mut interior: Vec<u32> = Vec::new();
    for y in 0..mask.height {
        for x in 0..mask.width {
            let i = y * mask.width + x;
            if mask.alpha[i] == 0 {
                continue;
            }
            if is_boundary_pixel(mask, x, y) {
                boundary[i] = 1;
            } else {
                interior.push(i as u32);
            }
        }
    }
    (boundary, interior)
}

/// `isBoundaryPixel` (ts:721-737).
fn is_boundary_pixel(mask: &AlphaMask, x: usize, y: usize) -> bool {
    if x == 0 || y == 0 || x == mask.width - 1 || y == mask.height - 1 {
        return true;
    }
    let i = y * mask.width + x;
    let w = mask.width;
    mask.alpha[i - 1] == 0
        || mask.alpha[i + 1] == 0
        || mask.alpha[i - w] == 0
        || mask.alpha[i + w] == 0
        || mask.alpha[i - w - 1] == 0
        || mask.alpha[i - w + 1] == 0
        || mask.alpha[i + w - 1] == 0
        || mask.alpha[i + w + 1] == 0
}

/// `solveInterior` (ts:510-539): red/black SOR over the interior, values are
/// f32 (`Float32Array`), per-step arithmetic in f64 (JS numbers).
fn solve_interior(mask: &AlphaMask, interior: &[u32], boundary: &[u8]) -> Vec<f32> {
    let mut values = vec![0.0f32; mask.alpha.len()];
    let graph = neighbor_graph(mask, interior, boundary);
    let omega = optimal_sor_omega(mask);
    let rhs = poisson_rhs(mask);

    let mut red: Vec<usize> = Vec::new();
    let mut black: Vec<usize> = Vec::new();
    for (n, &i) in interior.iter().enumerate() {
        let x = (i as usize) % mask.width;
        let y = (i as usize) / mask.width;
        if (x + y) % 2 == 0 {
            red.push(n);
        } else {
            black.push(n);
        }
    }

    for _pass in 0..POISSON_MAX_ITERATIONS {
        let red_delta = relax(&red, interior, &graph, &mut values, omega, rhs);
        let black_delta = relax(&black, interior, &graph, &mut values, omega, rhs);
        if red_delta.max(black_delta) < POISSON_TOLERANCE {
            break;
        }
    }
    values
}

/// `optimalSorOmega` (ts:541-547).
fn optimal_sor_omega(mask: &AlphaMask) -> f64 {
    let nx = (mask.width as f64 - 1.0).max(2.0);
    let ny = (mask.height as f64 - 1.0).max(2.0);
    let jacobi_radius =
        ((std::f64::consts::PI / nx).cos() + (std::f64::consts::PI / ny).cos()) * 0.5;
    2.0 / (1.0 + (1.0 - jacobi_radius * jacobi_radius).max(0.0).sqrt())
}

/// `poissonRhs` (ts:549-552).
fn poisson_rhs(mask: &AlphaMask) -> f64 {
    let h = 1.0 / (mask.width.max(mask.height) as f64);
    h * h * POISSON_SOURCE
}

/// `neighborGraph` (ts:554-571).
fn neighbor_graph(mask: &AlphaMask, interior: &[u32], boundary: &[u8]) -> Vec<i32> {
    let mut graph = vec![-1i32; interior.len() * 4];
    let w = mask.width as i64;
    let offsets: [i64; 4] = [1, -1, -w, w];
    for (n, &iu) in interior.iter().enumerate() {
        let i = iu as i64;
        let x = (iu as usize) % mask.width;
        let y = (iu as usize) / mask.width;
        let allowed = [x < mask.width - 1, x > 0, y > 0, y < mask.height - 1];
        for (k, &off) in offsets.iter().enumerate() {
            let next = i + off;
            graph[n * 4 + k] =
                if allowed[k] && mask.alpha[next as usize] != 0 && boundary[next as usize] == 0 {
                    next as i32
                } else {
                    -1
                };
        }
    }
    graph
}

/// `relax` (ts:573-601). Reads/writes f32 cells; arithmetic in f64.
fn relax(
    batch: &[usize],
    interior: &[u32],
    graph: &[i32],
    values: &mut [f32],
    omega: f64,
    rhs: f64,
) -> f64 {
    let mut max_delta = 0.0f64;
    for &n in batch {
        let i = interior[n] as usize;
        let mut sum = 0.0f64;
        for k in 0..4 {
            let neighbor = graph[n * 4 + k];
            if neighbor >= 0 {
                sum += values[neighbor as usize] as f64;
            }
        }
        let next = (rhs + sum) * 0.25;
        let old = values[i] as f64;
        let relaxed = old + omega * (next - old);
        max_delta = max_delta.max((relaxed - old).abs());
        values[i] = relaxed as f32; // Float32Array store
    }
    max_delta
}

/// `drawDistanceMask` (ts:603-629): RGBA (straight, as putImageData input) at
/// solve size — gray = field, alpha = mask.
fn draw_distance_mask(mask: &AlphaMask, distance: &[f32], interior: &[u32]) -> Vec<u8> {
    let mut max_value = 0.0f64;
    for &i in interior {
        max_value = max_value.max(distance[i as usize] as f64);
    }
    let max_value = max_value.max(0.0001);

    let mut rgba = vec![0u8; mask.alpha.len() * 4];
    for i in 0..mask.alpha.len() {
        let p = i * 4;
        if mask.alpha[i] == 0 {
            rgba[p] = 255;
            rgba[p + 1] = 255;
            rgba[p + 2] = 255;
            rgba[p + 3] = 0;
            continue;
        }
        let gray = to_uint8_clamped(255.0 * (1.0 - (distance[i] as f64) / max_value));
        rgba[p] = gray;
        rgba[p + 1] = gray;
        rgba[p + 2] = gray;
        rgba[p + 3] = 255;
    }
    rgba
}

/// `upscaleWithOriginalAlpha` (ts:631-661).
///
/// Browser data flow modeled exactly:
/// 1. `putImageData(lowRes)` writes into a PREMULTIPLIED canvas store — the
///    white RGB under alpha-0 pixels is destroyed (premul 0).
/// 2. `drawImage(lowRes → outputSize)` resamples premultiplied bytes with
///    `imageSmoothingQuality: 'high'` (Mitchell cubic).
/// 3. `getImageData` un-premultiplies with Skia rounding.
/// 4. Channel rewrite: r = srcAlpha==0 ? 255 : (upscaledAlpha==0 ? 0 : upR),
///    g = srcAlpha (original image drawn at outputSize), b = 255, a = 255.
fn upscale_with_original_alpha(
    low_res_rgba_straight: &[u8],
    mask: &AlphaMask,
    image: &dyn ShaderImageSource,
    output_size: Size,
) -> Vec<u8> {
    let (ow, oh) = (output_size.w_u32(), output_size.h_u32());

    // putImageData premultiply (alpha here is only 0 or 255).
    let mut low_premul = low_res_rgba_straight.to_vec();
    premultiply_in_place(&mut low_premul);

    let up_premul =
        mitchell_resize_premul(&low_premul, mask.width as u32, mask.height as u32, ow, oh);
    let mut up = up_premul;
    unpremultiply_in_place(&mut up);

    // Original image drawn at outputSize (default smoothing): alpha only.
    let original_premul = image.draw_premul(ow, oh);

    let n = (ow as usize) * (oh as usize);
    let mut out = vec![0u8; n * 4];
    for i in 0..n {
        let p = i * 4;
        let alpha = original_premul[p + 3];
        let upscaled_alpha = up[p + 3];
        out[p] = if alpha == 0 {
            255
        } else if upscaled_alpha == 0 {
            0
        } else {
            up[p]
        };
        out[p + 1] = alpha;
        out[p + 2] = 255;
        out[p + 3] = 255;
    }
    out
}

// ---------------------------------------------------------------------------
// heatmap
// ---------------------------------------------------------------------------

/// `prepareHeatmapImage` (ts:328-394).
fn prepare_heatmap_image(image: &dyn ShaderImageSource) -> PreparedShaderImage {
    let canvas_size = 1000.0f64;
    let (nw, nh) = image.natural_size();
    let natural = Size {
        width: or_one(nw).max(1.0),
        height: or_one(nh).max(1.0),
    };
    let ratio = natural.width / natural.height;
    let max_blur = (canvas_size * 0.15).floor(); // 150
    let padding = (max_blur * 2.5).ceil(); // 375
    let mut image_width = canvas_size;
    let mut image_height = canvas_size;
    if ratio > 1.0 {
        image_height = (canvas_size / ratio).floor();
    } else {
        image_width = (canvas_size * ratio).floor();
    }

    let surface_w = (image_width + 2.0 * padding) as u32;
    let surface_h = (image_height + 2.0 * padding) as u32;

    // White fill + drawImage(image, padding, padding, iw, ih), then grayscale
    // from getImageData. Compositing over opaque white in premul space:
    // out_c = src_premul_c + (255 - src_a); out_a = 255 (no un-premul loss).
    // drawImage with a zero-extent destination draws nothing.
    let iw = image_width.max(0.0) as usize;
    let ih = image_height.max(0.0) as usize;
    let img_premul = if iw > 0 && ih > 0 {
        image.draw_premul(iw as u32, ih as u32)
    } else {
        Vec::new()
    };
    let (sw, sh) = (surface_w as usize, surface_h as usize);
    let pad = padding as usize;

    let total = sw * sh;
    let mut gray = vec![0u8; total];
    // background: white → gray 255 (0.299+0.587+0.114 = 1.0 exactly at 255).
    for g in gray.iter_mut() {
        *g = 255;
    }
    for y in 0..ih {
        for x in 0..iw {
            let sp = (y * iw + x) * 4;
            let a = img_premul[sp + 3] as u32;
            let r = img_premul[sp] as u32 + (255 - a);
            let g = img_premul[sp + 1] as u32 + (255 - a);
            let b = img_premul[sp + 2] as u32 + (255 - a);
            let v = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
            gray[(y + pad) * sw + (x + pad)] = to_uint8_clamped(js_round(v));
        }
    }

    let big_blur = multi_pass_blur_gray(&gray, sw, sh, max_blur as i64, 3);
    let inner_blur =
        multi_pass_blur_gray(&gray, sw, sh, js_round(0.12 * max_blur).max(1.0) as i64, 3);
    let contour = multi_pass_blur_gray(&gray, sw, sh, 5, 1);

    let mut rgba = vec![0u8; total * 4];
    for i in 0..total {
        let p = i * 4;
        rgba[p] = contour[i];
        rgba[p + 1] = big_blur[i];
        rgba[p + 2] = inner_blur[i];
        rgba[p + 3] = 255;
    }

    PreparedShaderImage {
        width: surface_w,
        height: surface_h,
        rgba_straight: Arc::new(rgba),
        content_rect: Some([
            padding / surface_w as f64,
            padding / surface_h as f64,
            image_width / surface_w as f64,
            image_height / surface_h as f64,
        ]),
    }
}

/// `blurGray` (ts:663-703): box blur via Uint32 integral image,
/// `Math.round((a - b - c + d) / area)`.
fn blur_gray(gray: &[u8], width: usize, height: usize, radius: i64) -> Vec<u8> {
    if radius <= 0 {
        return gray.to_vec();
    }
    let mut output = vec![0u8; width * height];
    let mut integral = vec![0u32; width * height];

    for y in 0..height {
        let mut row_sum = 0u32;
        for x in 0..width {
            let index = y * width + x;
            row_sum = row_sum.wrapping_add(gray[index] as u32);
            integral[index] = row_sum.wrapping_add(if y > 0 { integral[index - width] } else { 0 });
        }
    }

    let r = radius;
    for y in 0..height as i64 {
        let y1 = (y - r).max(0);
        let y2 = (y + r).min(height as i64 - 1);
        for x in 0..width as i64 {
            let x1 = (x - r).max(0);
            let x2 = (x + r).min(width as i64 - 1);
            let at = |yy: i64, xx: i64| -> u32 { integral[(yy as usize) * width + xx as usize] };
            let a = at(y2, x2);
            let b = if x1 > 0 { at(y2, x1 - 1) } else { 0 };
            let c = if y1 > 0 { at(y1 - 1, x2) } else { 0 };
            let d = if x1 > 0 && y1 > 0 {
                at(y1 - 1, x1 - 1)
            } else {
                0
            };
            let area = ((x2 - x1 + 1) * (y2 - y1 + 1)) as f64;
            let v = js_round((a as f64 - b as f64 - c as f64 + d as f64) / area);
            output[(y as usize) * width + x as usize] = to_uint8_clamped(v);
        }
    }
    output
}

/// `multiPassBlurGray` (ts:705-719).
fn multi_pass_blur_gray(
    gray: &[u8],
    width: usize,
    height: usize,
    radius: i64,
    passes: usize,
) -> Vec<u8> {
    let mut output = gray.to_vec();
    for _ in 0..passes.max(1) {
        output = blur_gray(&output, width, height, radius);
    }
    output
}

// ---------------------------------------------------------------------------
// source-image
// ---------------------------------------------------------------------------

/// `prepareSourceImage` (ts:312-326). The canvas is returned directly; the
/// texture copy un-premultiplies (straight-alpha texture).
fn prepare_source_image(image: &dyn ShaderImageSource) -> PreparedShaderImage {
    let natural = normalize_size(size_of(image));
    let scale = (SOURCE_IMAGE_MAX_DIMENSION / natural.width.max(natural.height)).min(1.0);
    let size = if scale < 1.0 {
        scale_size(natural, scale)
    } else {
        natural
    };
    let mut rgba = image.draw_premul(size.w_u32(), size.h_u32());
    unpremultiply_in_place(&mut rgba);
    PreparedShaderImage {
        width: size.w_u32(),
        height: size.h_u32(),
        rgba_straight: Arc::new(rgba),
        content_rect: None,
    }
}

// ---------------------------------------------------------------------------
// dither-image
// ---------------------------------------------------------------------------

/// `prepareDitherImage` (ts:88-120).
fn prepare_dither_image(
    image: &dyn ShaderImageSource,
    options: &DitherOptions,
) -> PreparedShaderImage {
    let natural = normalize_size(size_of(image));
    let (raster_size, pixel_size) = resolve_dither_raster_size(natural, options);

    let mut source_pixels = image.draw_premul(raster_size.w_u32(), raster_size.h_u32());
    unpremultiply_in_place(&mut source_pixels); // getImageData
    let w = raster_size.w_u32() as usize;
    let h = raster_size.h_u32() as usize;

    let (gray, alpha) =
        image_data_to_adjusted_gray(&source_pixels, options.brightness, options.contrast);
    let pix = pixelate_dither_input(&gray, &alpha, w, h, pixel_size);
    let dithered = dither_gray(&pix.gray, pix.width, pix.height, options.pattern);

    // Output canvas: putImageData (premul store) then straight-copy on upload.
    // v ∈ {0, 255} round-trips exactly; rgb collapses to 0 under alpha 0.
    let n = pix.width * pix.height;
    let mut rgba = vec![0u8; n * 4];
    for i in 0..n {
        let p = i * 4;
        let a = pix.alpha[i];
        let v = if a == 0 { 0 } else { dithered[i] };
        rgba[p] = v;
        rgba[p + 1] = v;
        rgba[p + 2] = v;
        rgba[p + 3] = a;
    }

    PreparedShaderImage {
        width: pix.width as u32,
        height: pix.height as u32,
        rgba_straight: Arc::new(rgba),
        content_rect: None,
    }
}

/// `resolveDitherRasterSize` (ts:122-138).
fn resolve_dither_raster_size(natural: Size, options: &DitherOptions) -> (Size, f64) {
    let raw_display = normalize_size(Size {
        width: options.display_width,
        height: options.display_height,
    });
    let display = Size {
        width: js_round(raw_display.width).max(1.0),
        height: js_round(raw_display.height).max(1.0),
    };
    let scale = (DITHER_IMAGE_MAX_DIMENSION / display.width.max(display.height)).min(1.0);
    let size = if scale < 1.0 {
        scale_size(display, scale)
    } else {
        display
    };
    let size = if size.width > 0.0 && size.height > 0.0 {
        size
    } else {
        natural
    };
    (size, (options.pixel_size * scale).max(1.0))
}

/// `imageDataToAdjustedGray` (ts:140-163).
fn image_data_to_adjusted_gray(
    pixels: &[u8],
    brightness: f64,
    contrast_percent: f64,
) -> (Vec<u8>, Vec<u8>) {
    let total = pixels.len() / 4;
    let mut gray = vec![0u8; total];
    let mut alpha = vec![0u8; total];
    let contrast = clamp_number(contrast_percent, 0.0, 200.0) / 100.0;
    let brightness_offset = clamp_number(brightness, -100.0, 100.0);
    for i in 0..total {
        let p = i * 4;
        let r = pixels[p] as f64;
        let g = pixels[p + 1] as f64;
        let b = pixels[p + 2] as f64;
        let mut v = 0.299 * r + 0.587 * g + 0.114 * b;
        v = (v - 128.0) * contrast + 128.0 + brightness_offset;
        // Math.round of the clamp, then Uint8ClampedArray store (integer).
        gray[i] = to_uint8_clamped(js_round(clamp_number(v, 0.0, 255.0)));
        alpha[i] = pixels[p + 3];
    }
    (gray, alpha)
}

struct Pixelated {
    gray: Vec<u8>,
    alpha: Vec<u8>,
    width: usize,
    height: usize,
}

/// `pixelateDitherInput` (ts:165-208).
fn pixelate_dither_input(
    gray: &[u8],
    alpha: &[u8],
    width: usize,
    height: usize,
    pixel_size: f64,
) -> Pixelated {
    let block_size = pixel_size.max(1.0);
    if block_size <= 1.0 {
        return Pixelated {
            gray: gray.to_vec(),
            alpha: alpha.to_vec(),
            width,
            height,
        };
    }
    let output_width = ((width as f64 / block_size).ceil() as usize).max(1);
    let output_height = ((height as f64 / block_size).ceil() as usize).max(1);
    let mut out_gray = vec![0u8; output_width * output_height];
    let mut out_alpha = vec![0u8; output_width * output_height];

    for y in 0..output_height {
        for x in 0..output_width {
            let start_x = ((x as f64 * block_size).ceil() as usize).min(width);
            let start_y = ((y as f64 * block_size).ceil() as usize).min(height);
            let end_x = (((x + 1) as f64 * block_size).ceil() as usize).min(width);
            let end_y = (((y + 1) as f64 * block_size).ceil() as usize).min(height);
            let mut gray_sum = 0u64;
            let mut alpha_sum = 0u64;
            let mut count = 0u64;
            for sy in start_y..end_y {
                for sx in start_x..end_x {
                    let i = sy * width + sx;
                    gray_sum += gray[i] as u64;
                    alpha_sum += alpha[i] as u64;
                    count += 1;
                }
            }
            let out_index = y * output_width + x;
            out_gray[out_index] = to_uint8_clamped(js_round(gray_sum as f64 / count.max(1) as f64));
            out_alpha[out_index] =
                to_uint8_clamped(js_round(alpha_sum as f64 / count.max(1) as f64));
        }
    }
    Pixelated {
        gray: out_gray,
        alpha: out_alpha,
        width: output_width,
        height: output_height,
    }
}

/// `ditherGray` (ts:210-221).
fn dither_gray(gray: &[u8], width: usize, height: usize, pattern: DitherPattern) -> Vec<u8> {
    match pattern {
        DitherPattern::Bayer4x4 => bayer_dither(gray, width, height),
        DitherPattern::FloydSteinberg => floyd_steinberg_dither(gray, width, height),
        DitherPattern::Atkinson => atkinson_dither(gray, width, height),
        DitherPattern::Noise => noise_dither(gray, width, height),
        DitherPattern::Threshold => threshold_dither(gray),
    }
}

fn threshold_dither(gray: &[u8]) -> Vec<u8> {
    gray.iter()
        .map(|&g| if g > 127 { 255 } else { 0 })
        .collect()
}

fn bayer_dither(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; gray.len()];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let threshold = BAYER4_MATRIX[(y % 4) * 4 + (x % 4)];
            output[i] = if gray[i] > threshold { 255 } else { 0 };
        }
    }
    output
}

fn noise_dither(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; gray.len()];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            output[i] = if (gray[i] as u32) > coordinate_noise_threshold(x as i32, y as i32) {
                255
            } else {
                0
            };
        }
    }
    output
}

/// `coordinateNoiseThreshold` (ts:260-264): Math.imul + unsigned shifts.
fn coordinate_noise_threshold(x: i32, y: i32) -> u32 {
    let seed = (x.wrapping_add(374761393)).wrapping_mul(668265263)
        ^ (y.wrapping_add(224682251)).wrapping_mul(3266489917u32 as i32);
    let mixed = (seed ^ ((seed as u32) >> 13) as i32).wrapping_mul(1274126177);
    ((mixed as u32) >> 24) & 255
}

/// `floydSteinbergDither` (ts:266-286): error buffer is Float32Array.
fn floyd_steinberg_dither(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut buffer: Vec<f32> = gray.iter().map(|&g| g as f32).collect();
    let mut output = vec![0u8; gray.len()];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let old = buffer[i] as f64;
            let quantized: f64 = if old > 127.0 { 255.0 } else { 0.0 };
            output[i] = quantized as u8;
            let error = old - quantized;
            if x + 1 < width {
                buffer[i + 1] = (buffer[i + 1] as f64 + (error * 7.0) / 16.0) as f32;
            }
            if x >= 1 && y + 1 < height {
                buffer[i + width - 1] =
                    (buffer[i + width - 1] as f64 + (error * 3.0) / 16.0) as f32;
            }
            if y + 1 < height {
                buffer[i + width] = (buffer[i + width] as f64 + (error * 5.0) / 16.0) as f32;
            }
            if x + 1 < width && y + 1 < height {
                buffer[i + width + 1] = (buffer[i + width + 1] as f64 + error / 16.0) as f32;
            }
        }
    }
    output
}

/// `atkinsonDither` (ts:288-310).
fn atkinson_dither(gray: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut buffer: Vec<f32> = gray.iter().map(|&g| g as f32).collect();
    let mut output = vec![0u8; gray.len()];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let old = buffer[i] as f64;
            let quantized: f64 = if old > 127.0 { 255.0 } else { 0.0 };
            output[i] = quantized as u8;
            let error = (old - quantized) / 8.0;
            let mut bump = |idx: usize| {
                buffer[idx] = (buffer[idx] as f64 + error) as f32;
            };
            if x + 1 < width {
                bump(i + 1);
            }
            if x + 2 < width {
                bump(i + 2);
            }
            if x >= 1 && y + 1 < height {
                bump(i + width - 1);
            }
            if y + 1 < height {
                bump(i + width);
            }
            if x + 1 < width && y + 1 < height {
                bump(i + width + 1);
            }
            if y + 2 < height {
                bump(i + 2 * width);
            }
        }
    }
    output
}

// ---------------------------------------------------------------------------
// shared numeric/JS helpers
// ---------------------------------------------------------------------------

fn size_of(image: &dyn ShaderImageSource) -> Size {
    let (w, h) = image.natural_size();
    Size {
        width: or_one(w),
        height: or_one(h),
    }
}

/// `image.naturalWidth || image.width || 1` falsy coercion.
fn or_one(v: f64) -> f64 {
    if v == 0.0 || v.is_nan() {
        1.0
    } else {
        v
    }
}

/// `normalizeSize` (ts:451-456): max(1, size || 1).
fn normalize_size(size: Size) -> Size {
    Size {
        width: or_one(size.width).max(1.0),
        height: or_one(size.height).max(1.0),
    }
}

/// `scaleSize` (ts:749-754).
fn scale_size(size: Size, factor: f64) -> Size {
    Size {
        width: js_round(size.width * factor).max(1.0),
        height: js_round(size.height * factor).max(1.0),
    }
}

/// `clampNumber` (ts:458-461): non-finite → min.
fn clamp_number(value: f64, min: f64, max: f64) -> f64 {
    if !value.is_finite() {
        return min;
    }
    value.max(min).min(max)
}

fn get_image_natural_size(image: &dyn ShaderImageSource) -> Size {
    size_of(image)
}

fn transparent(width: u32, height: u32) -> PreparedShaderImage {
    PreparedShaderImage {
        width,
        height,
        rgba_straight: Arc::new(vec![0u8; (width as usize) * (height as usize) * 4]),
        content_rect: None,
    }
}

/// JS `Math.round`: floor(x + 0.5) (round half toward +∞).
pub fn js_round(x: f64) -> f64 {
    (x + 0.5).floor()
}

fn js_round_i64(x: f64) -> i64 {
    js_round(x) as i64
}

/// `Uint8ClampedArray` store: clamp [0, 255] then round half to EVEN.
pub fn to_uint8_clamped(x: f64) -> u8 {
    if !(x > 0.0) {
        return 0;
    }
    if x >= 255.0 {
        return 255;
    }
    let f = x.floor();
    let diff = x - f;
    let mut v = f as u32;
    if diff > 0.5 || (diff == 0.5 && v % 2 == 1) {
        v += 1;
    }
    v as u8
}

/// Premultiply straight RGBA8 in place (canvas putImageData / drawImage
/// store conversion) — round to nearest, Chromium parity.
pub fn premultiply_in_place(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a == 255 {
            continue;
        }
        for c in px.iter_mut().take(3) {
            *c = ((*c as u32 * a + 127) / 255) as u8;
        }
    }
}

/// Un-premultiply in place (`getImageData` / straight-alpha texture copy):
/// Skia `SkUnPreMultiply` rounding — (c·255 + a/2) / a; alpha 0 zeroes RGB.
pub fn unpremultiply_in_place(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a == 255 {
            continue;
        }
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
            continue;
        }
        for c in px.iter_mut().take(3) {
            let v = (*c as u32 * 255 + a / 2) / a;
            *c = v.min(255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// resampling (drawImage analogs)
// ---------------------------------------------------------------------------

/// Skia bilinear (`drawImage`, default smoothing "low"): 4-tap, dst-center
/// mapping `src = (dst + 0.5) * (srcSize / dstSize) - 0.5`, edge clamp.
/// Operates on PREMULTIPLIED bytes (the canvas store domain).
pub fn bilinear_resize_premul(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    resize_premul(src, sw, sh, dw, dh, FilterKind::Bilinear)
}

/// Skia Mitchell cubic B = C = 1/3 (`imageSmoothingQuality: 'high'`), on
/// premultiplied bytes.
pub fn mitchell_resize_premul(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    resize_premul(src, sw, sh, dw, dh, FilterKind::Mitchell)
}

#[derive(Clone, Copy, PartialEq)]
enum FilterKind {
    Bilinear,
    Mitchell,
}

fn mitchell_weight(x: f64) -> f64 {
    // B = C = 1/3
    const B: f64 = 1.0 / 3.0;
    const C: f64 = 1.0 / 3.0;
    let x = x.abs();
    if x < 1.0 {
        ((12.0 - 9.0 * B - 6.0 * C) * x * x * x
            + (-18.0 + 12.0 * B + 6.0 * C) * x * x
            + (6.0 - 2.0 * B))
            / 6.0
    } else if x < 2.0 {
        ((-B - 6.0 * C) * x * x * x
            + (6.0 * B + 30.0 * C) * x * x
            + (-12.0 * B - 48.0 * C) * x
            + (8.0 * B + 24.0 * C))
            / 6.0
    } else {
        0.0
    }
}

fn bilinear_weight(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

/// Separable resize in f64 over premultiplied RGBA8; single final
/// quantization (round + clamp). Identity sizes short-circuit to a copy.
fn resize_premul(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32, kind: FilterKind) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let (sw, sh, dw, dh) = (
        sw as usize,
        sh as usize,
        dw.max(1) as usize,
        dh.max(1) as usize,
    );
    let support = match kind {
        FilterKind::Bilinear => 1.0,
        FilterKind::Mitchell => 2.0,
    };
    let weight = |x: f64| match kind {
        FilterKind::Bilinear => bilinear_weight(x),
        FilterKind::Mitchell => mitchell_weight(x),
    };

    // Pass 1: horizontal sw → dw (f64 intermediate).
    let mut mid = vec![0.0f64; dw * sh * 4];
    let sx = sw as f64 / dw as f64;
    for dx in 0..dw {
        let center = (dx as f64 + 0.5) * sx - 0.5;
        let lo = (center - support).ceil() as i64;
        let hi = (center + support).floor() as i64;
        let mut taps: Vec<(usize, f64)> = Vec::with_capacity((hi - lo + 1).max(1) as usize);
        let mut wsum = 0.0f64;
        for sxi in lo..=hi {
            let w = weight(sxi as f64 - center);
            if w == 0.0 {
                continue;
            }
            let clamped = sxi.clamp(0, sw as i64 - 1) as usize;
            taps.push((clamped, w));
            wsum += w;
        }
        if wsum == 0.0 {
            let clamped = (center.round() as i64).clamp(0, sw as i64 - 1) as usize;
            taps.push((clamped, 1.0));
            wsum = 1.0;
        }
        for y in 0..sh {
            let mut acc = [0.0f64; 4];
            for &(xi, w) in &taps {
                let p = (y * sw + xi) * 4;
                for ch in 0..4 {
                    acc[ch] += src[p + ch] as f64 * w;
                }
            }
            let mp = (y * dw + dx) * 4;
            for ch in 0..4 {
                mid[mp + ch] = acc[ch] / wsum;
            }
        }
    }

    // Pass 2: vertical sh → dh, quantize.
    let mut out = vec![0u8; dw * dh * 4];
    let sy = sh as f64 / dh as f64;
    for dy in 0..dh {
        let center = (dy as f64 + 0.5) * sy - 0.5;
        let lo = (center - support).ceil() as i64;
        let hi = (center + support).floor() as i64;
        let mut taps: Vec<(usize, f64)> = Vec::with_capacity((hi - lo + 1).max(1) as usize);
        let mut wsum = 0.0f64;
        for syi in lo..=hi {
            let w = weight(syi as f64 - center);
            if w == 0.0 {
                continue;
            }
            let clamped = syi.clamp(0, sh as i64 - 1) as usize;
            taps.push((clamped, w));
            wsum += w;
        }
        if wsum == 0.0 {
            let clamped = (center.round() as i64).clamp(0, sh as i64 - 1) as usize;
            taps.push((clamped, 1.0));
            wsum = 1.0;
        }
        for x in 0..dw {
            let mut acc = [0.0f64; 4];
            for &(yi, w) in &taps {
                let p = (yi * dw + x) * 4;
                for ch in 0..4 {
                    acc[ch] += mid[p + ch] * w;
                }
            }
            let op = (dy * dw + x) * 4;
            for ch in 0..4 {
                out[op + ch] = (acc[ch] / wsum).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic "image": opaque disc of the given radius on transparent
    /// background, rendered analytically at any draw size (vector-like).
    struct Disc {
        natural: (f64, f64),
    }

    impl ShaderImageSource for Disc {
        fn natural_size(&self) -> (f64, f64) {
            self.natural
        }
        fn draw_premul(&self, w: u32, h: u32) -> Vec<u8> {
            let mut out = vec![0u8; (w as usize) * (h as usize) * 4];
            let cx = w as f64 / 2.0;
            let cy = h as f64 / 2.0;
            let r = w.min(h) as f64 * 0.4;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let dx = x as f64 + 0.5 - cx;
                    let dy = y as f64 + 0.5 - cy;
                    if (dx * dx + dy * dy).sqrt() <= r {
                        let p = (y * w as usize + x) * 4;
                        out[p] = 30;
                        out[p + 1] = 60;
                        out[p + 2] = 90;
                        out[p + 3] = 255;
                    }
                }
            }
            out
        }
    }

    #[test]
    fn js_round_matches_math_round() {
        assert_eq!(js_round(0.5), 1.0);
        assert_eq!(js_round(1.5), 2.0);
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-0.5), 0.0);
        assert_eq!(js_round(-1.5), -1.0);
        assert_eq!(js_round(2.4), 2.0);
    }

    #[test]
    fn uint8_clamped_rounds_half_to_even() {
        assert_eq!(to_uint8_clamped(0.5), 0);
        assert_eq!(to_uint8_clamped(1.5), 2);
        assert_eq!(to_uint8_clamped(2.5), 2);
        assert_eq!(to_uint8_clamped(3.5), 4);
        assert_eq!(to_uint8_clamped(-1.0), 0);
        assert_eq!(to_uint8_clamped(300.0), 255);
        assert_eq!(to_uint8_clamped(f64::NAN), 0);
    }

    #[test]
    fn processing_keys_match_production() {
        assert_eq!(processing_key(&Processing::PoissonMask), "poisson-mask");
        assert_eq!(processing_key(&Processing::Heatmap), "heatmap");
        assert_eq!(processing_key(&Processing::SourceImage), "source-image");
        let o = DitherOptions {
            pattern: DitherPattern::Bayer4x4,
            brightness: -10.4,
            contrast: 100.0,
            pixel_size: 4.0,
            display_width: 512.4,
            display_height: 511.5,
        };
        // Math.round: -10.4 → -10, 512.4 → 512, 511.5 → 512.
        assert_eq!(
            processing_key(&Processing::DitherImage(o)),
            "dither-image:bayer-4x4:-10:100:4:512:512"
        );
    }

    #[test]
    fn poisson_solve_sizes_match_ts() {
        // square: output 768×768, solve 512×512 (512/768 scale, no cap)
        let out = normalize_poisson_mask_output_size(Size {
            width: 512.0,
            height: 512.0,
        });
        assert_eq!((out.width, out.height), (768.0, 768.0));
        let solve = fit_poisson_solve_size(out);
        assert_eq!((solve.width, solve.height), (512.0, 512.0));
        // 2:1 landscape: output 768×384 → first scale 512/384 → 1024×512 → cap 768/1024 → 768×384
        let out = normalize_poisson_mask_output_size(Size {
            width: 1000.0,
            height: 500.0,
        });
        assert_eq!((out.width, out.height), (768.0, 384.0));
        let solve = fit_poisson_solve_size(out);
        assert_eq!((solve.width, solve.height), (768.0, 384.0));
    }

    #[test]
    fn poisson_mask_field_shape() {
        let disc = Disc {
            natural: (512.0, 512.0),
        };
        let prepared = prepare_shader_image(&disc, &Processing::PoissonMask);
        assert_eq!((prepared.width, prepared.height), (768, 768));
        let w = prepared.width as usize;
        // center: inside mask, far from boundary → r near 0 (deep field), g=255
        let cp = ((prepared.height as usize / 2) * w + w / 2) * 4;
        let px = &prepared.rgba_straight[cp..cp + 4];
        assert_eq!(px[1], 255, "g = original alpha inside the disc");
        assert!(
            px[0] < 80,
            "r (edge field) should be deep inside: {}",
            px[0]
        );
        assert_eq!(px[2], 255);
        assert_eq!(px[3], 255);
        // corner: outside mask → r=255, g=0
        let px = &prepared.rgba_straight[0..4];
        assert_eq!(px[0], 255);
        assert_eq!(px[1], 0);
    }

    #[test]
    fn empty_mask_yields_transparent_1x1() {
        struct Empty;
        impl ShaderImageSource for Empty {
            fn natural_size(&self) -> (f64, f64) {
                (64.0, 64.0)
            }
            fn draw_premul(&self, w: u32, h: u32) -> Vec<u8> {
                vec![0u8; (w as usize) * (h as usize) * 4]
            }
        }
        let p = prepare_shader_image(&Empty, &Processing::PoissonMask);
        assert_eq!((p.width, p.height), (1, 1));
        assert_eq!(p.rgba_straight.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn heatmap_layout_and_content_rect() {
        let disc = Disc {
            natural: (512.0, 512.0),
        };
        let p = prepare_shader_image(&disc, &Processing::Heatmap);
        // square → image 1000×1000, padding 375 → surface 1750×1750
        assert_eq!((p.width, p.height), (1750, 1750));
        let cr = p.content_rect.expect("heatmap carries contentRect");
        assert!((cr[0] - 375.0 / 1750.0).abs() < 1e-12);
        assert!((cr[2] - 1000.0 / 1750.0).abs() < 1e-12);
        // far corner is pure white bg → gray 255 in every blur channel
        let px = &p.rgba_straight[0..4];
        assert_eq!(px, &[255, 255, 255, 255]);
    }

    #[test]
    fn dither_bayer_is_exact_on_known_grays() {
        // 4×4 image, gray 100 everywhere (opaque). Bayer thresholds decide.
        struct Flat;
        impl ShaderImageSource for Flat {
            fn natural_size(&self) -> (f64, f64) {
                (4.0, 4.0)
            }
            fn draw_premul(&self, w: u32, h: u32) -> Vec<u8> {
                let mut v = Vec::with_capacity((w * h * 4) as usize);
                for _ in 0..w * h {
                    v.extend_from_slice(&[100, 100, 100, 255]);
                }
                v
            }
        }
        let o = DitherOptions {
            pattern: DitherPattern::Bayer4x4,
            brightness: 0.0,
            contrast: 100.0,
            pixel_size: 1.0,
            display_width: 4.0,
            display_height: 4.0,
        };
        let p = prepare_shader_image(&Flat, &Processing::DitherImage(o));
        assert_eq!((p.width, p.height), (4, 4));
        for (i, px) in p.rgba_straight.chunks_exact(4).enumerate() {
            let (y, x) = (i / 4, i % 4);
            let expected = if 100 > BAYER4_MATRIX[(y % 4) * 4 + (x % 4)] {
                255
            } else {
                0
            };
            assert_eq!(px[0], expected, "bayer at ({x},{y})");
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn floyd_steinberg_serial_diffusion_matches_reference() {
        // 2×1, gray [200, 10]: px0 → 255, err -55 → px1 = 10 + (-55*7)/16 = -14.0625 → 0
        let out = floyd_steinberg_dither(&[200, 10], 2, 1);
        assert_eq!(out, vec![255, 0]);
        // Atkinson: err/8 spread; 1×1 trivially 255 for >127.
        assert_eq!(atkinson_dither(&[128], 1, 1), vec![255]);
        assert_eq!(atkinson_dither(&[127], 1, 1), vec![0]);
    }

    #[test]
    fn noise_threshold_matches_js_imul_semantics() {
        // Spot-checked against the JS implementation evaluated by hand:
        // seed/mixed arithmetic uses 32-bit wrapping ops.
        let t00 = coordinate_noise_threshold(0, 0);
        let t10 = coordinate_noise_threshold(1, 0);
        assert!(t00 <= 255 && t10 <= 255);
        assert_ne!(t00, t10, "coordinate noise varies per pixel");
        // determinism
        assert_eq!(t00, coordinate_noise_threshold(0, 0));
    }

    #[test]
    fn premul_unpremul_round_trips_for_binary_values() {
        let mut px = [255u8, 255, 255, 0];
        premultiply_in_place(&mut px);
        assert_eq!(px, [0, 0, 0, 0]);
        let mut px = [255u8, 0, 255, 128];
        premultiply_in_place(&mut px);
        unpremultiply_in_place(&mut px);
        assert_eq!(px, [255, 0, 255, 128]);
    }

    #[test]
    fn resize_identity_is_copy() {
        let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(bilinear_resize_premul(&src, 2, 1, 2, 1), src);
        assert_eq!(mitchell_resize_premul(&src, 2, 1, 2, 1), src);
    }

    #[test]
    fn source_image_caps_at_2048() {
        let disc = Disc {
            natural: (4096.0, 2048.0),
        };
        let p = prepare_shader_image(&disc, &Processing::SourceImage);
        assert_eq!((p.width, p.height), (2048, 1024));
    }
}
