//! M4e — shader-image sources for the image-backed procedural fills.
//!
//! The IO/rasterization half of `renderers/webgpu/shaders/imageProcessing.ts`
//! (the pixel math lives in `moda_shaders::runtime::image_processing`): this
//! module turns fetched bytes into a [`ShaderImageSource`] (the
//! `HTMLImageElement` + `drawImage` analog) and runs the processing pipeline.
//!
//! - Raster bytes (PNG/JPEG/WebP/GIF): decode + ICC→sRGB at NATURAL size
//!   (straight alpha), `drawImage` = premultiply + Skia-style 4-tap bilinear
//!   resize (default `imageSmoothingQuality`).
//! - SVG bytes: usvg tree; `drawImage` re-rasterizes the vector at the
//!   requested size via resvg (premultiplied output), exactly like Blink.
//!
//! Entries are keyed by the production resource key
//! (`createImageResourceKey`: `{processingKey}:{mime}:{src}`) so one source
//! used by two different processings (e.g. dither pixel sizes) caches
//! separately, like `ShaderImageResourceManager`.

use anyhow::{Context, Result};
use resvg::usvg;
use std::sync::Arc;

use moda_shaders::runtime::image_processing::{
    bilinear_resize_premul, premultiply_in_place, prepare_shader_image, PreparedShaderImage,
    Processing, ShaderImageSource,
};

/// Terminal state of one prefetched shader image (mirrors
/// `ShaderImageResourceManager` entry states). The texture fallback remains
/// transparent, while lifecycle metadata makes the renderer draw a visible
/// failed placeholder instead of dropping the node.
#[derive(Debug, Clone)]
pub enum ShaderImageEntry {
    Loaded(Arc<PreparedShaderImage>),
    /// Load/decode failed or timed out → terminal fallback entry —
    /// `ShaderImageResourceManager.load().catch`.
    Fallback,
}

/// Raster source: natural-size straight-alpha sRGB RGBA8.
pub struct RasterShaderSource {
    pub width: u32,
    pub height: u32,
    /// STRAIGHT alpha (HTMLImageElement holds the decoded image; premultiply
    /// happens at drawImage time).
    pub rgba_straight: Vec<u8>,
}

impl ShaderImageSource for RasterShaderSource {
    fn natural_size(&self) -> (f64, f64) {
        (self.width as f64, self.height as f64)
    }
    fn draw_premul(&self, w: u32, h: u32) -> Vec<u8> {
        let mut premul = self.rgba_straight.clone();
        premultiply_in_place(&mut premul);
        bilinear_resize_premul(&premul, self.width, self.height, w, h)
    }
}

/// SVG source: vector re-rasterized at every draw size (Blink parity).
pub struct SvgShaderSource {
    pub tree: usvg::Tree,
}

impl ShaderImageSource for SvgShaderSource {
    fn natural_size(&self) -> (f64, f64) {
        let (w, h) = crate::svg::svg_natural_size(&self.tree);
        (w as f64, h as f64)
    }
    fn draw_premul(&self, w: u32, h: u32) -> Vec<u8> {
        let Some(mut pixmap) = resvg::tiny_skia::Pixmap::new(w.max(1), h.max(1)) else {
            return vec![0u8; (w.max(1) as usize) * (h.max(1) as usize) * 4];
        };
        let sx = w.max(1) as f32 / self.tree.size().width();
        let sy = h.max(1) as f32 / self.tree.size().height();
        resvg::render(
            &self.tree,
            resvg::tiny_skia::Transform::from_scale(sx, sy),
            &mut pixmap.as_mut(),
        );
        pixmap.take()
    }
}

/// Decode fetched bytes into a shader-image source and run the processing
/// pipeline. `src` empty is the production `!source.src` arm: a 1×1
/// transparent prepared image whose manager metadata is `ready: true`.
pub fn load_shader_image(
    src: &str,
    bytes: &[u8],
    svg_hint: bool,
    processing: &Processing,
) -> Result<PreparedShaderImage> {
    if src.is_empty() {
        return Ok(PreparedShaderImage {
            width: 1,
            height: 1,
            rgba_straight: Arc::new(vec![0, 0, 0, 0]),
            content_rect: None,
        });
    }
    let is_svg = svg_hint || crate::source::sniff_svg_bytes(bytes);
    if is_svg {
        // SVG <text> lays out with the text engine's fonts (resolved only for an SVG).
        let tree = usvg::Tree::from_data(bytes, &crate::svg::usvg_options(&crate::svg::engine_fontdb()))
            .context("usvg parse (shader image)")?;
        Ok(prepare_shader_image(&SvgShaderSource { tree }, processing))
    } else {
        let (rgba, w, h, _icc) = crate::decode::decode_raster(bytes)?;
        Ok(prepare_shader_image(
            &RasterShaderSource {
                width: w,
                height: h,
                rgba_straight: rgba,
            },
            processing,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DISC_SVG: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><circle cx="256" cy="256" r="200" fill="#102030"/></svg>"##;

    #[test]
    fn svg_poisson_mask_pipeline_runs() {
        let p = load_shader_image("logo.svg", DISC_SVG, true, &Processing::PoissonMask).unwrap();
        assert_eq!((p.width, p.height), (768, 768));
        // center of the disc: g (original alpha) = 255, edge field deep.
        let w = p.width as usize;
        let cp = ((p.height as usize / 2) * w + w / 2) * 4;
        assert_eq!(p.rgba_straight[cp + 1], 255);
        assert!(p.rgba_straight[cp] < 80);
    }

    #[test]
    fn empty_src_is_transparent_1x1_ready() {
        let p = load_shader_image("", &[], false, &Processing::Heatmap).unwrap();
        assert_eq!((p.width, p.height), (1, 1));
        assert_eq!(p.rgba_straight.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn raster_dither_1to1_is_byte_exact() {
        // 8×8 opaque gradient PNG, displayed at natural size, threshold.
        let mut rgba = Vec::new();
        for y in 0..8u32 {
            for x in 0..8u32 {
                let v = (x * 32 + y) as u8;
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let img: image::RgbaImage = image::RgbaImage::from_raw(8, 8, rgba.clone()).unwrap();
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let o = moda_shaders::runtime::image_processing::DitherOptions {
            pattern: moda_shaders::runtime::image_processing::DitherPattern::Threshold,
            brightness: 0.0,
            contrast: 100.0,
            pixel_size: 1.0,
            display_width: 8.0,
            display_height: 8.0,
        };
        let p = load_shader_image("x.png", &png, false, &Processing::DitherImage(o)).unwrap();
        assert_eq!((p.width, p.height), (8, 8));
        for (i, px) in p.rgba_straight.chunks_exact(4).enumerate() {
            let v = rgba[i * 4] as f64;
            // gray of (v,v,v) = v; threshold > 127
            let expected = if v > 127.0 { 255 } else { 0 };
            assert_eq!(px[0], expected, "px {i}");
        }
    }
}
