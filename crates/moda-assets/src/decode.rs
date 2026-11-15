//! Raster decode → ICC→sRGB → premultiply → resize chain.
//!
//! Mirrors the browser pipeline the goldens were captured through:
//! Chromium decodes with color management to sRGB (moxcms supplies the ICC
//! transform natively), `createImageBitmap` premultiplies and resizes with
//! `resizeQuality: 'high'` (Skia kHigh = Mitchell cubic), and
//! `copyExternalImageToTexture({premultipliedAlpha: true})` uploads
//! premultiplied sRGB bytes (`WebGPUTextureManager.ts:818-826`).

use anyhow::{Context, Result};
use fast_image_resize as fir;
use image::ImageDecoder;
use std::io::Cursor;
use std::sync::Arc;

/// A decoded, color-managed, premultiplied, resize-applied image ready for
/// `wgpu::Queue::write_texture` as `rgba8unorm`.
#[derive(Clone)]
pub struct PreparedImage {
    /// Texel dimensions actually uploaded (post sizing policy).
    pub width: u32,
    pub height: u32,
    /// Natural (decode-time) dimensions.
    pub natural_width: u32,
    pub natural_height: u32,
    /// `CachedTexture.scaleFactor` (`WebGPUTextureManager.ts:851`):
    /// uploaded / natural, per axis. The pattern instance divides
    /// `fillPatternScale` by this (`WebGPUPathRenderer.ts:1338-1345`).
    pub scale_factor: (f64, f64),
    /// Premultiplied sRGB RGBA8, tightly packed (`straight`: unassociated instead).
    pub rgba_premul: Arc<Vec<u8>>,
    /// Whether `rgba_premul` holds straight (unassociated) colour: so for [`prepare_raster_natural`] and
    /// [`PreparedImage::from_straight`], whose consumer converts the texels to another encoding and associates
    /// them itself, in floating point. Associating in 8 bits first would round away the colour of a low-alpha
    /// texel.
    pub straight: bool,
    pub is_svg: bool,
    /// Whether an ICC profile was found and applied.
    pub icc_applied: bool,
    /// Non-zero-alpha pixels per mille; `0` = the decode SUCCEEDED but every
    /// pixel is FULLY TRANSPARENT (an asset that loads clean and paints
    /// nothing). Observability only — never consulted by plan building.
    ///
    /// CEILED (see [`coverage_permille`]), which is the load-bearing part: `0`
    /// is an EXACT predicate ("not one non-transparent pixel"), never a rounded
    /// one. A rounding-down permille would report `0` for a logo covering
    /// <0.05% of a large pixmap, and a false blank on a trust signal is worse
    /// than no signal at all.
    pub coverage_permille: u16,
    /// Low-cardinality reason a zero-coverage decode came out blank (one of the
    /// `BLANK_CAUSE_*` constants); `""` whenever `coverage_permille > 0`.
    pub blank_cause: &'static str,
}

impl PreparedImage {
    /// An image made in memory (the pixel graph's sink images): premultiplied sRGB RGBA8, `width × height`,
    /// uploaded as it is (no sizing policy, scale factor 1).
    pub fn from_premultiplied(rgba_premul: Vec<u8>, width: u32, height: u32) -> Self {
        let total = width as u64 * height as u64;
        let covered = rgba_premul.chunks_exact(4).filter(|p| p[3] != 0).count() as u64;
        Self {
            width,
            height,
            natural_width: width,
            natural_height: height,
            scale_factor: (1.0, 1.0),
            rgba_premul: Arc::new(rgba_premul),
            straight: false,
            is_svg: false,
            icc_applied: false,
            coverage_permille: coverage_permille(covered, total),
            blank_cause: "",
        }
    }

    /// [`PreparedImage::from_premultiplied`] for straight (unassociated) sRGB RGBA8: `straight`.
    pub fn from_straight(rgba: Vec<u8>, width: u32, height: u32) -> Self {
        Self { straight: true, ..Self::from_premultiplied(rgba, width, height) }
    }
}

impl std::fmt::Debug for PreparedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("natural", &(self.natural_width, self.natural_height))
            .field("scale_factor", &self.scale_factor)
            .field("is_svg", &self.is_svg)
            .field("icc_applied", &self.icc_applied)
            .field("coverage_permille", &self.coverage_permille)
            .field("blank_cause", &self.blank_cause)
            .finish()
    }
}

/// Generic zero-coverage cause: the decode produced only fully-transparent
/// pixels and nothing more specific is known.
pub const BLANK_CAUSE_FULLY_TRANSPARENT: &str = "fully_transparent";

/// The wasm editor lane rasterizes SVG with a deliberately EMPTY fontdb
/// (`crate::default_fontdb`), so a text-bearing SVG rasterizes to nothing at
/// all. The single most actionable blank cause we can name.
pub const BLANK_CAUSE_SVG_TEXT_NO_FONTDB: &str = "svg_text_no_fontdb";

/// Non-zero-alpha pixels per mille, CEILED and clamped to 1000.
///
/// Ceil rather than round: it makes `0` mean EXACTLY "zero non-transparent
/// pixels" instead of "fewer than 0.05% of pixels paint". Any pixel that paints
/// at all yields >= 1, so the blank verdict can never be a rounding artifact —
/// which is the whole reason this number is trustworthy enough to alert on.
/// The cost is a slight over-estimate at the low end; nothing reads the
/// magnitude for anything but human triage.
pub fn coverage_permille(non_transparent: u64, total: u64) -> u16 {
    if total == 0 {
        return 0;
    }
    (non_transparent * 1000).div_ceil(total).min(1000) as u16
}

/// Decode raster bytes (PNG/JPEG/WebP/GIF) to straight-alpha sRGB RGBA8,
/// applying an embedded ICC profile via moxcms when present.
///
/// JPEG bytes decode through mozjpeg (libjpeg-turbo) for bit parity with
/// Chromium's decoder — the browser routes every texture image through
/// Blink's HTMLImageElement decode (libjpeg-turbo, JDCT_ISLOW + fancy chroma
/// upsampling; the production load path is `WebGPUTextureManager.ts:753-767`).
/// The image crate's zune-jpeg differs by up to ±4/255 on a few percent of
/// pixels — invisible directly, but on a box-minified pattern fill those
/// deltas flip round-half unorm ties and the JPEG screenshot encode then
/// amplifies the 1-LSB flips into block noise (0ba834b3 p3: SSIM capped at
/// 0.97013; bit-exact 1.00000 with the libjpeg-turbo decode — quirk Q-93).
pub fn decode_raster(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32, bool)> {
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("image format sniff")?;
    let mut decoder = reader.into_decoder().map_err(|e| {
        // Keep the stable "image decoder:" prefix (backend telemetry substring-
        // matches it); a positive browser-undecodable sniff rides INSIDE the
        // context so the render worker can classify the miss as pixel parity
        // (Chromium bakes the identical failed-fill placeholder) instead of
        // bouncing the whole render to a fallback that fails the same way.
        //
        // Magic-byte sniff ONLY on this stage. `sniff_corrupt_complete_raster`
        // is deliberately not consulted here: a `Format` error while the
        // decoder constructs (`read_info`, every chunk up to the first IDAT) is
        // a different population from the demonstrated IDAT class — critical-
        // chunk faults libpng rejects too (PLTE CRC, invalid IHDR) mixed with
        // APNG frame-control (`fcTL`) faults and other header-stage strictness
        // the crate's private inner variant is the only way to tell apart, and
        // Chromium's own APNG handling is not the libpng story. No incident
        // evidence covers it, so it stays a plain decline (retry), as before.
        // (Malformed ANCILLARY metadata — gAMA/pHYs/sRGB/tEXt — never gets
        // here: png 0.18 swallows those as `BadAncillaryChunk`.)
        match sniff_browser_undecodable(bytes) {
            Some(kind) => anyhow::Error::new(BrowserUndecodable {
                kind,
                stage: "decoder",
                source: e,
            }),
            None => anyhow::Error::new(e).context("image decoder"),
        }
    })?;
    let icc = decoder.icc_profile().ok().flatten();
    // A camera JPEG's EXIF orientation: Chromium draws it upright (`createImageBitmap` and `<img>` default to
    // `from-image`), and the header sniffs report the upright size, so the pixels are turned here too.
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);

    let (mut rgba, mut w, mut h) = match decode_jpeg_turbo(bytes) {
        Some(out) => out,
        None => {
            // Same contract as the `into_decoder` arm: stable "image decode:"
            // prefix, positive sniff INSIDE the context. A decoder that
            // constructs (intact header chunks) and then trips a FORMAT error
            // in the pixel stream of a COMPLETE file is corrupt at rest, and
            // Chromium's `createImageBitmap` rejects the same bytes.
            let img = image::DynamicImage::from_decoder(decoder).map_err(|e| {
                let sniff = sniff_corrupt_complete_raster(bytes, &e);
                match sniff {
                    Some(kind) => anyhow::Error::new(BrowserUndecodable {
                        kind,
                        stage: "decode",
                        source: e,
                    }),
                    None => anyhow::Error::new(e).context("image decode"),
                }
            })?;
            let (w, h) = (img.width(), img.height());
            (img.into_rgba8().into_raw(), w, h)
        }
    };

    if orientation != image::metadata::Orientation::NoTransforms {
        let buffer = image::RgbaImage::from_raw(w, h, rgba).context("decoded raster size")?;
        let mut img = image::DynamicImage::ImageRgba8(buffer);
        img.apply_orientation(orientation);
        (w, h) = (img.width(), img.height());
        rgba = img.into_rgba8().into_raw();
    }

    let mut icc_applied = false;
    if let Some(icc_bytes) = icc {
        if let Ok(()) = apply_icc_to_srgb(&mut rgba, &icc_bytes) {
            icc_applied = true;
        }
    }
    Ok((rgba, w, h, icc_applied))
}

/// The TYPED marker for a positively-sniffed browser-undecodable decode
/// failure — the error [`decode_raster`] returns for one, wrapping the image
/// crate's error as its `source`. Its `Display` is the stable reason grammar
/// the backend substring-matches (`image {stage}: undecodable format (sniff:
/// {kind})`, with the cause following under `{e:#}` exactly as a plain
/// `.context()` would print it); the TYPE is what the Rust
/// `fail_on_missing_assets` gates read, via [`browser_undecodable_kind`], so an
/// exemption never has to parse a string it also emits. One authority for "is
/// this miss browser parity": the sniffs in this module.
///
/// A real error object rather than an anyhow `.context(..)` value on purpose:
/// anyhow stores a context inside a private `ContextError<C, E>`, so nothing in
/// `Error::chain()` ever has type `C`, and the marker would be unrecoverable
/// once a caller wrapped the error again.
#[derive(Debug)]
pub struct BrowserUndecodable {
    /// Sniff kind: `emf`, `wmf`, `html`, `corrupt-png`.
    pub kind: &'static str,
    /// Which decode stage failed: `decoder` (construction) or `decode` (pixels).
    stage: &'static str,
    source: image::ImageError,
}

impl std::fmt::Display for BrowserUndecodable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "image {}: undecodable format (sniff: {})",
            self.stage, self.kind
        )
    }
}

impl std::error::Error for BrowserUndecodable {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// The sniff kind carried anywhere in a decode error's chain, if any — `None`
/// for every failure that is not positively browser-undecodable (transport,
/// EOF / truncation, unknown junk, limits), which is what keeps those on the
/// decline/retry path.
pub fn browser_undecodable_kind(err: &anyhow::Error) -> Option<&'static str> {
    err.chain()
        .find_map(|c| c.downcast_ref::<BrowserUndecodable>())
        .map(|m| m.kind)
}

/// Best-effort magic-byte sniff for formats NO browser can decode either —
/// Chromium's `<img>` / `createImageBitmap` rejects EMF/WMF metafiles and an
/// HTML document exactly like this decoder, baking the identical failed-fill
/// placeholder. Prod census (30d): PPTX-import EMF leftovers + Next.js HTML
/// pages stored 200-as-image are ~77% of the "image format could not be
/// determined" class. Only runs after `into_decoder` already failed, so a
/// decodable image can never be misclassified; unknown junk returns `None`
/// (the backend keeps declining those so a transient bad body retains its
/// browser-side recovery chance).
pub fn sniff_browser_undecodable(bytes: &[u8]) -> Option<&'static str> {
    // EMF ([MS-EMF] 2.2.9): EMR_HEADER record type 1 (u32 LE) at offset 0 and
    // the " EMF" signature (0x464D4520 LE) at offset 40.
    if bytes.len() >= 44 && bytes[..4] == [0x01, 0x00, 0x00, 0x00] && bytes[40..44] == *b" EMF" {
        return Some("emf");
    }
    // WMF: the placeable (Aldus/APM) header magic D7 CD C6 9A. Standard
    // headerless WMF is deliberately NOT sniffed — its 01/02 00 09 00 prefix
    // is too weak a signature to gate a native accept on.
    if bytes.len() >= 4 && bytes[..4] == [0xD7, 0xCD, 0xC6, 0x9A] {
        return Some("wmf");
    }
    // HTML page served as an image asset (upstream data defect: an app/error
    // page persisted under an image content type). SVG never reaches here —
    // `sniff_svg_bytes` routes it to the SVG lane before raster decode.
    let head = &bytes[..bytes.len().min(1024)];
    let text = String::from_utf8_lossy(head);
    let t = text.trim_start_matches('\u{feff}').trim_start();
    // Allocation-free, case-insensitive prefix match. `str::get` is byte-indexed and
    // yields `None` on a non-char boundary (multi-byte lead), so this cannot panic.
    if ["<!doctype", "<html", "<head", "<body"].iter().any(|tag| {
        t.get(..tag.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(tag))
    }) {
        return Some("html");
    }
    None
}

/// Positive sniff for a raster whose bytes are a COMPLETE file of a known
/// format but whose PIXEL STREAM is corrupt at rest — the decoder constructed
/// (header chunks parsed) and then tripped a FORMAT error inside pixel decode
/// (IDAT: garbage filter bytes, a corrupt deflate stream, an IDAT CRC, image
/// data that ends early). Chromium's `createImageBitmap` rejects the same
/// bytes (demonstrated on the prod pair: an IDAT with garbage scanline filter
/// bytes + failed adler32/CRC), so the failed-fill placeholder is pixel parity,
/// same as EMF/WMF/HTML.
///
/// Consulted on the pixel-decode stage ONLY. `next_frame` consumes just the
/// image-data chunks, so a `Format` error there is image-data-class by
/// construction (IDAT for a still PNG; the `fdAT`/`fcTL` frame chunks of an
/// APNG are the one residual — Chromium decodes APNG through its own path and
/// no parity evidence exists for a damaged one). The header stage is a
/// different population and is not sniffed (see `decode_raster`).
///
/// Two gates, both required, both conservative toward `None`:
/// 1. The error is `ImageError::Decoding` — for PNG that is exactly
///    `png::DecodingError::Format` ("the input image was not a valid PNG"; the
///    inner variant enum is private, so no finer typed gate exists) — never an
///    IO/`UnexpectedEof` (a body that ran out mid-stream), a limits, or an
///    unsupported error.
/// 2. The byte buffer is structurally COMPLETE: for PNG, signature + a chunk
///    walk that lands exactly on an `IEND` at the end of the buffer. A short
///    or truncated download fails this and keeps its decline/retry path (the
///    next fetch may return the whole file). Every byte of a complete file is
///    already in hand, so a re-fetch returns the same bytes: nothing about
///    this class is transient.
///
/// PNG only. JPEG/WebP/GIF stay un-sniffed until the same browser evidence
/// is in hand — Chromium's JPEG decoder in particular paints partial rows on
/// a damaged scan rather than rejecting the bitmap.
pub fn sniff_corrupt_complete_raster(
    bytes: &[u8],
    err: &image::ImageError,
) -> Option<&'static str> {
    if !matches!(err, image::ImageError::Decoding(_)) {
        return None;
    }
    if png_is_structurally_complete(bytes) {
        return Some("corrupt-png");
    }
    None
}

/// PNG signature, then a chunk walk (`length u32be | type | data | crc`) that
/// reaches an `IEND` chunk ending exactly at the buffer's end, having seen an
/// `IHDR` first and at least one `IDAT`. CRCs are deliberately NOT checked —
/// a CRC mismatch is precisely the corruption this sniff exists to name.
///
/// Hand-rolled on purpose (review-guidelines 10.3 weighed): the question is
/// "is every byte of the file here?", which the png crate cannot answer for a
/// stream it has just rejected, and this walk is not a pre-check that must
/// agree with the decoder — the decoder already ruled. Its only failure
/// direction is toward `false` (no sniff → the pre-existing retryable
/// decline), since a truncated buffer cannot end in a 12-byte `IEND` chunk
/// whose declared length is 0.
fn png_is_structurally_complete(bytes: &[u8]) -> bool {
    const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];
    if !bytes.starts_with(&SIG) {
        return false;
    }
    let mut off = SIG.len();
    let mut first = true;
    let mut saw_idat = false;
    loop {
        let Some(header) = bytes.get(off..off + 8) else {
            return false;
        };
        let len = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let kind = &header[4..8];
        if first && kind != b"IHDR" {
            return false;
        }
        first = false;
        let Some(end) = off.checked_add(12).and_then(|o| o.checked_add(len)) else {
            return false;
        };
        if end > bytes.len() {
            return false;
        }
        match kind {
            b"IDAT" => saw_idat = true,
            b"IEND" => return saw_idat && len == 0 && end == bytes.len(),
            _ => {}
        }
        off = end;
    }
}

/// Chromium-parity JPEG decode: mozjpeg (libjpeg-turbo) with the Blink
/// settings — JDCT_ISLOW + fancy upsampling (the libjpeg defaults). Returns
/// `None` for non-JPEG bytes or any decode failure (the caller falls back to
/// the image crate, which also covers CMYK/exotic colorspaces).
/// mozjpeg is vendored C (no wasm32 build, needs the platform SDK on mobile)
/// and rides the `jpeg-parity` feature; without it the image-crate fallback
/// (the `not(feature = "jpeg-parity")` stub below + `decode_raster`'s `None`
/// path) decodes JPEG.
#[cfg(feature = "jpeg-parity")]
fn decode_jpeg_turbo(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    if bytes.len() < 2 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return None;
    }
    // mozjpeg signals errors by panicking (libjpeg setjmp/longjmp translated
    // to unwinds); contain them like `moda-engine/src/encode.rs::encode_jpeg`.
    std::panic::catch_unwind(|| -> Option<(Vec<u8>, u32, u32)> {
        let mut d = mozjpeg::Decompress::new_mem(bytes).ok()?;
        d.dct_method(mozjpeg::DctMethod::IntegerSlow);
        d.do_fancy_upsampling(true);
        let mut started = d.rgba().ok()?;
        let (w, h) = (started.width(), started.height());
        let px: Vec<[u8; 4]> = started.read_scanlines().ok()?;
        started.finish().ok()?;
        if px.len() != w * h {
            return None;
        }
        Some((bytemuck::cast_vec(px), w as u32, h as u32))
    })
    .ok()
    .flatten()
}

/// Without `jpeg-parity` (mobile cross-compile lane: mozjpeg-sys is a C
/// build script needing the platform SDK), every JPEG falls through to the
/// image crate's decoder in [`decode_raster`] — within ±4/255 of
/// libjpeg-turbo (quirk Q-93). Fine on device; never for server-parity lanes.
#[cfg(not(feature = "jpeg-parity"))]
fn decode_jpeg_turbo(_bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    None
}

/// ICC → sRGB in place on straight-alpha RGBA8 (alpha passes through).
/// Chromium parity: decode-time color management to the sRGB target.
fn apply_icc_to_srgb(rgba: &mut [u8], icc: &[u8]) -> Result<()> {
    use moxcms::{ColorProfile, Layout, TransformOptions};
    let src = ColorProfile::new_from_slice(icc).context("ICC parse")?;
    let dst = ColorProfile::new_srgb();
    let transform = src
        .create_transform_8bit(
            Layout::Rgba,
            &dst,
            Layout::Rgba,
            TransformOptions::default(),
        )
        .context("ICC transform build")?;
    let input = rgba.to_vec();
    transform
        .transform(&input, rgba)
        .context("ICC transform apply")?;
    Ok(())
}

/// Premultiply straight RGBA8 in place. Integer round-to-nearest
/// (`round(c·a/255)`), matching the GPU conversion within 1 LSB.
///
/// Returns the number of NON-TRANSPARENT pixels (`a != 0`) seen on the way
/// through — the alpha tally for [`PreparedImage::coverage_permille`], folded
/// into the walk this function already does so the coverage verdict costs zero
/// extra passes and zero allocation. Callers that don't want it ignore the
/// return value; the premultiply behavior is byte-for-byte unchanged.
pub fn premultiply_in_place(rgba: &mut [u8]) -> u64 {
    let mut non_transparent = 0u64;
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a != 0 {
            non_transparent += 1;
        }
        if a == 255 {
            continue;
        }
        for c in &mut px[..3] {
            *c = (((*c as u32) * a + 127) / 255) as u8;
        }
    }
    non_transparent
}

/// Resize PREMULTIPLIED RGBA8 with Mitchell cubic (Skia `kHigh`, the
/// `resizeQuality: 'high'` mapping used by `createImageBitmap`).
///
/// `use_alpha(false)` is load-bearing (quirk Q-94): fast_image_resize's
/// DEFAULT (`mul_div_alpha: true`) multiplies by alpha before convolving and
/// divides after — feeding it premultiplied bytes applies alpha TWICE,
/// crushing alpha-edge texels dark/heavy (beb9b4fd p2 logo read ~1.5 SSIM
/// points worse than the prod golden). Skia/Chromium convolve the
/// premultiplied N32 pixels directly, exactly once.
pub fn resize_premultiplied(
    rgba_premul: Vec<u8>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Result<Vec<u8>> {
    if (src_w, src_h) == (dst_w, dst_h) {
        return Ok(rgba_premul);
    }
    let src = fir::images::Image::from_vec_u8(src_w, src_h, rgba_premul, fir::PixelType::U8x4)
        .context("resize source image")?;
    let mut dst = fir::images::Image::new(dst_w, dst_h, fir::PixelType::U8x4);
    let mut resizer = fir::Resizer::new();
    resizer
        .resize(
            &src,
            &mut dst,
            &fir::ResizeOptions::new()
                .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Mitchell))
                .use_alpha(false),
        )
        .context("resize")?;
    Ok(dst.into_vec())
}

/// Full raster pipeline: decode → ICC → premultiply → sizing policy →
/// resize (premultiplied convolution, exactly once — Q-94).
pub fn prepare_raster(bytes: &[u8]) -> Result<PreparedImage> {
    prepare(bytes, true)
}

/// [`prepare_raster`] without the sizing policy and without 8-bit association: the decoded texels, straight
/// alpha (`PreparedImage::straight`), at their natural size, scale factor 1. For a host whose engine blends in
/// another encoding than sRGB (`moda_shaders::working_color`): that engine converts these texels when it
/// uploads them, associates them in floating point and resamples them on the GPU in its encoding, so nothing
/// may filter or round their sRGB bytes first. The texels are the image as decoded, whatever device draws
/// them: an engine refuses one its device's texture limit cannot hold when it uploads it. Stored under
/// [`crate::natural_key`].
pub fn prepare_raster_natural(bytes: &[u8]) -> Result<PreparedImage> {
    prepare(bytes, false)
}

fn prepare(bytes: &[u8], sized: bool) -> Result<PreparedImage> {
    let (mut rgba, nw, nh, icc_applied) = decode_raster(bytes)?;
    // Alpha tally rides the premultiply walk (zero extra pass). Counted at
    // NATURAL size, BEFORE the sizing-policy resize, which is correct for the
    // blank verdict: Mitchell convolution of an all-zero alpha channel is
    // all-zero, so "0 at natural size" ⟺ "0 after resize".
    let non_transparent = if sized { premultiply_in_place(&mut rgba) } else { rgba.chunks_exact(4).filter(|px| px[3] != 0).count() as u64 };
    let coverage = coverage_permille(non_transparent, (nw as u64) * (nh as u64));
    let (tw, th) = if sized { crate::sizing::resolve_target_size(nw, nh, false, None) } else { (nw, nh) };
    let rgba = resize_premultiplied(rgba, nw, nh, tw, th)?;
    Ok(PreparedImage {
        width: tw,
        height: th,
        natural_width: nw,
        natural_height: nh,
        scale_factor: (tw as f64 / nw.max(1) as f64, th as f64 / nh.max(1) as f64),
        rgba_premul: Arc::new(rgba),
        straight: !sized,
        is_svg: false,
        icc_applied,
        coverage_permille: coverage,
        blank_cause: if coverage == 0 {
            BLANK_CAUSE_FULLY_TRANSPARENT
        } else {
            ""
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_png(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
        let img: image::RgbaImage = image::RgbaImage::from_raw(w, h, rgba.to_vec()).unwrap();
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    /// Encode straight-alpha RGBA8 as a PNG carrying an embedded `iCCP` profile
    /// chunk — the same shape `decode_raster` reads back via the `image`
    /// crate's `icc_profile()`.
    fn encode_png_with_icc(rgba: &[u8], w: u32, h: u32, icc: &[u8]) -> Vec<u8> {
        use image::ImageEncoder;
        let mut out = Vec::new();
        let mut enc = image::codecs::png::PngEncoder::new(&mut out);
        enc.set_icc_profile(icc.to_vec())
            .expect("png encoder accepts an ICC profile");
        enc.write_image(rgba, w, h, image::ExtendedColorType::Rgba8)
            .unwrap();
        out
    }

    /// A real wide-gamut (Display-P3) ICC profile, round-tripped through
    /// moxcms — the same library `apply_icc_to_srgb` parses, so the embedded
    /// bytes are guaranteed to build a transform.
    fn display_p3_icc() -> Vec<u8> {
        moxcms::ColorProfile::new_display_p3()
            .encode()
            .expect("encode Display-P3 ICC profile")
    }

    /// A decode that succeeds but paints NOTHING must be
    /// distinguishable from a decode that paints something.
    #[test]
    fn fully_transparent_raster_reports_zero_coverage() {
        let png = encode_png(&[0u8, 0, 0, 0].repeat(8 * 8), 8, 8);
        let p = prepare_raster(&png).unwrap();
        assert_eq!(p.coverage_permille, 0);
        assert_eq!(p.blank_cause, BLANK_CAUSE_FULLY_TRANSPARENT);

        // A fully opaque image is the other end of the scale, and carries no
        // cause (an empty `blank_cause` is what "not blank" looks like).
        let opaque = encode_png(&[9u8, 9, 9, 255].repeat(8 * 8), 8, 8);
        let q = prepare_raster(&opaque).unwrap();
        assert_eq!(q.coverage_permille, 1000);
        assert_eq!(q.blank_cause, "");
    }

    /// The ANTI-STRIDE-SAMPLING test. A logo can occupy well under
    /// 0.1% of a large pixmap: one opaque pixel in 1024² is 1/1_048_576 of the
    /// image, which any stride sample would miss and any rounding-DOWN permille
    /// would report as `0`. Both would mean "blank" for a perfectly good asset —
    /// the most damaging failure this signal can have. The scan is full and
    /// sequential and the permille is CEILED, so this is exactly `1`.
    #[test]
    fn one_opaque_pixel_in_a_large_raster_is_never_blank() {
        let mut rgba = vec![0u8; 1024 * 1024 * 4];
        let center = ((512 * 1024) + 512) * 4;
        rgba[center..center + 4].copy_from_slice(&[255, 0, 0, 255]);
        let p = prepare_raster(&encode_png(&rgba, 1024, 1024)).unwrap();
        assert!(
            p.coverage_permille > 0,
            "one painted pixel must not read as blank; got {}",
            p.coverage_permille
        );
        assert_eq!(p.coverage_permille, 1, "ceiled permille of 1/1048576");
        assert_eq!(p.blank_cause, "");
    }

    #[test]
    fn coverage_permille_ceils_and_clamps() {
        // Zero non-transparent pixels is the ONLY input that yields 0.
        assert_eq!(coverage_permille(0, 1_000_000), 0);
        assert_eq!(coverage_permille(1, 1_000_000), 1);
        assert_eq!(coverage_permille(1000, 1_000_000), 1);
        assert_eq!(coverage_permille(1001, 1_000_000), 2);
        assert_eq!(coverage_permille(1_000_000, 1_000_000), 1000);
        // Degenerate inputs can't panic or overflow the u16.
        assert_eq!(coverage_permille(0, 0), 0);
        assert_eq!(coverage_permille(5, 1), 1000);
    }

    #[test]
    fn premultiply_rounds_to_nearest() {
        let mut px = vec![255, 128, 0, 128];
        premultiply_in_place(&mut px);
        // 255*128/255 = 128; 128*128/255 = 64.25 → 64; 0 → 0
        assert_eq!(px, vec![128, 64, 0, 128]);
        let mut opaque = vec![10, 20, 30, 255];
        premultiply_in_place(&mut opaque);
        assert_eq!(opaque, vec![10, 20, 30, 255]);
    }

    #[test]
    fn prepare_small_raster_applies_min_dimension_upscale() {
        // 4x4 red PNG → small-image policy → 512x512 texture, scaleFactor 128.
        let rgba = vec![255, 0, 0, 255].repeat(16);
        let png = encode_png(&rgba, 4, 4);
        let p = prepare_raster(&png).unwrap();
        assert_eq!((p.width, p.height), (512, 512));
        assert_eq!((p.natural_width, p.natural_height), (4, 4));
        assert_eq!(p.scale_factor, (128.0, 128.0));
        // Flat color survives Mitchell resampling exactly.
        assert_eq!(&p.rgba_premul[..4], &[255, 0, 0, 255]);
        assert!(!p.icc_applied);
    }

    /// A JPEG carrying an EXIF APP1 segment with this orientation tag, right after SOI.
    fn jpeg_with_orientation(img: image::RgbImage, orientation: u16) -> Vec<u8> {
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .unwrap();
        // "Exif\0\0", big-endian TIFF header, IFD0 at 8 with one SHORT entry (0x0112), no next IFD.
        let mut tiff = b"MM\0\x2a\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01".to_vec();
        tiff.extend_from_slice(&orientation.to_be_bytes());
        tiff.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        let mut app1 = b"Exif\0\0".to_vec();
        app1.extend_from_slice(&tiff);
        let mut out = jpeg[..2].to_vec();
        out.extend_from_slice(&[0xFF, 0xE1]);
        out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&app1);
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn decode_jpeg_turns_exif_orientation_upright() {
        // 32x16: left half red, right half blue. Orientation 6 is "turn 90° clockwise to view": 16x32, red on top.
        let img = image::RgbImage::from_fn(32, 16, |x, _| {
            if x < 16 { image::Rgb([255, 0, 0]) } else { image::Rgb([0, 0, 255]) }
        });
        let (rgba, w, h, _) = decode_raster(&jpeg_with_orientation(img.clone(), 6)).unwrap();
        assert_eq!((w, h), (16, 32));
        let top = px_at(&rgba, w, 8, 4);
        let bottom = px_at(&rgba, w, 8, 28);
        assert!(top[0] > 200 && top[2] < 60, "top is red: {top:?}");
        assert!(bottom[2] > 200 && bottom[0] < 60, "bottom is blue: {bottom:?}");
        // Orientation 1 keeps the stored pixels.
        let (_, w, h, _) = decode_raster(&jpeg_with_orientation(img, 1)).unwrap();
        assert_eq!((w, h), (32, 16));
    }

    #[test]
    fn decode_jpeg_works() {
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([0, 128, 255]));
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Jpeg)
            .unwrap();
        let (rgba, w, h, _) = decode_raster(&out).unwrap();
        assert_eq!((w, h), (8, 8));
        assert_eq!(rgba.len(), 8 * 8 * 4);
        assert_eq!(rgba[3], 255);
    }

    /// R2 caveat: an embedded ICC profile must be DETECTED (`icc_applied`)
    /// and APPLIED (the pixels actually move toward sRGB). Until now
    /// `icc_applied` was tracked but never asserted, and the decode tests
    /// used non-ICC images — so a regression that silently stopped applying
    /// the profile (or stopped detecting it) would not have been caught.
    #[test]
    fn decode_raster_applies_embedded_icc_profile() {
        // A saturated color in-gamut for both P3 and sRGB; the same numeric
        // RGB names a different color in each space, so the P3→sRGB transform
        // moves it (a neutral gray would not — shared D65 white point).
        let color = [20u8, 200, 80, 255];
        let rgba: Vec<u8> = color.repeat(4 * 4); // 4x4
        let icc = display_p3_icc();

        let with_icc = encode_png_with_icc(&rgba, 4, 4, &icc);
        let (px_icc, w, h, applied) = decode_raster(&with_icc).unwrap();
        assert_eq!((w, h), (4, 4));
        assert!(applied, "embedded ICC profile must be detected and applied");

        // Same pixels, NO profile → straight passthrough, icc_applied false.
        let no_icc = encode_png(&rgba, 4, 4);
        let (px_plain, _, _, applied_plain) = decode_raster(&no_icc).unwrap();
        assert!(!applied_plain, "no profile → icc_applied false");
        assert_eq!(&px_plain[..4], &color, "no-ICC decode is a passthrough");

        // The transform actually changed the pixels (not an identity no-op).
        let delta = (px_icc[0] as i32 - color[0] as i32).abs()
            + (px_icc[1] as i32 - color[1] as i32).abs()
            + (px_icc[2] as i32 - color[2] as i32).abs();
        assert!(
            delta >= 4,
            "P3→sRGB must measurably shift a saturated color; delta={delta} got={:?}",
            &px_icc[..4]
        );
        // Color management never touches alpha.
        assert_eq!(px_icc[3], 255);
    }

    /// A malformed/unparseable ICC profile must degrade gracefully: the
    /// profile IS present (so `decode_raster` reaches the ICC branch), moxcms
    /// fails to parse it, `apply_icc_to_srgb` returns Err, and `decode_raster`
    /// swallows it — `icc_applied` stays false and the raw pixels survive.
    #[test]
    fn decode_raster_tolerates_unparseable_icc() {
        let rgba: Vec<u8> = [10u8, 20, 30, 255].repeat(2 * 2);
        let png = encode_png_with_icc(&rgba, 2, 2, &[0u8; 8]); // not a valid ICC

        // Isolate the "detected but unparseable" path: prove the iCCP chunk is
        // actually embedded and read back by the decoder (otherwise this test
        // could pass merely because no profile was present at all).
        let embedded = image::ImageReader::new(Cursor::new(&png))
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap()
            .icc_profile()
            .ok()
            .flatten();
        assert_eq!(
            embedded.as_deref(),
            Some(&[0u8; 8][..]),
            "the garbage profile must be embedded + read back (ICC branch is exercised)"
        );

        let (px, w, h, applied) = decode_raster(&png).unwrap();
        assert_eq!((w, h), (2, 2));
        assert!(
            !applied,
            "unparseable ICC → not applied; decode still succeeds"
        );
        assert_eq!(&px[..4], &[10, 20, 30, 255]);
    }

    /// Q-93: JPEG bytes must route through the libjpeg-turbo decoder (the
    /// Chromium decode path) and never the image crate's zune-jpeg; PNG bytes
    /// must NOT take the turbo path.
    #[cfg(feature = "jpeg-parity")]
    #[test]
    fn jpeg_routes_through_libjpeg_turbo() {
        let img = image::RgbImage::from_fn(16, 16, |x, y| {
            image::Rgb([(x * 16) as u8, (y * 16) as u8, ((x + y) * 8) as u8])
        });
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(img.clone())
            .write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .unwrap();
        let turbo = decode_jpeg_turbo(&jpeg).expect("turbo path decodes JPEG");
        assert_eq!((turbo.1, turbo.2), (16, 16));
        let (via_raster, w, h, _) = decode_raster(&jpeg).unwrap();
        assert_eq!((w, h), (16, 16));
        // decode_raster must return exactly the turbo pixels (not zune-jpeg's,
        // which differ by a few LSBs on gradient content).
        assert_eq!(via_raster, turbo.0);

        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        assert!(decode_jpeg_turbo(&png).is_none());
    }

    /// Q-94: the sizing-policy resize convolves PREMULTIPLIED pixels exactly
    /// once (Skia/Chromium N32 premul resize). fast_image_resize's default
    /// `mul_div_alpha: true` would multiply by alpha a second time, crushing
    /// alpha-edge pixels: a seam pixel between opaque red and transparent
    /// black has premul red ≈ alpha (r/a unpremul = full red); the double-
    /// premultiply bug yielded r ≈ a²/255 (half the correct value at a=127).
    #[test]
    fn prepare_raster_resize_premultiplies_exactly_once() {
        // 2x2: left column opaque red, right column fully transparent.
        let px: Vec<u8> = vec![
            255, 0, 0, 255, /**/ 0, 0, 0, 0, //
            255, 0, 0, 255, /**/ 0, 0, 0, 0,
        ];
        let png = encode_png(&px, 2, 2);
        let p = prepare_raster(&png).unwrap();
        assert_eq!((p.width, p.height), (512, 512));
        let seam = px_at(&p.rgba_premul, p.width, 256, 256);
        assert!(seam[3] > 40 && seam[3] < 220, "seam alpha mid: {seam:?}");
        let (r, a) = (seam[0] as i32, seam[3] as i32);
        assert!(
            (r - a).abs() <= 2,
            "premul red must track alpha (single premultiply); \
             double-premultiply gives r≈a²/255: {seam:?}"
        );
    }

    fn px_at(rgba: &[u8], width: u32, x: usize, y: usize) -> [u8; 4] {
        let i = (y * width as usize + x) * 4;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    }

    /// A minimal EMF header: EMR_HEADER record type (1 u32 LE) at offset 0,
    /// " EMF" signature at offset 40 — the [MS-EMF] shape the prod PPTX-import
    /// leftovers carry.
    fn emf_bytes() -> Vec<u8> {
        let mut b = vec![0u8; 88];
        b[0] = 0x01; // record type EMR_HEADER
        b[40..44].copy_from_slice(b" EMF");
        b
    }

    #[test]
    fn sniff_browser_undecodable_kinds() {
        assert_eq!(sniff_browser_undecodable(&emf_bytes()), Some("emf"));
        assert_eq!(
            sniff_browser_undecodable(&[0xD7, 0xCD, 0xC6, 0x9A, 0, 0, 0, 0]),
            Some("wmf")
        );
        assert_eq!(
            sniff_browser_undecodable(b"<!DOCTYPE html><html><head></head></html>"),
            Some("html")
        );
        assert_eq!(
            sniff_browser_undecodable(b"\n  <html lang=\"en\"><body></body></html>"),
            Some("html")
        );
        // Unknown junk / real formats: no sniff — the backend keeps declining.
        assert_eq!(sniff_browser_undecodable(b"\x00\x01\x02 junk bytes"), None);
        assert_eq!(sniff_browser_undecodable(&[0x89, b'P', b'N', b'G']), None);
        // Truncated EMF-ish prefix without the signature: no sniff.
        assert_eq!(sniff_browser_undecodable(&[0x01, 0, 0, 0, 9, 9]), None);
    }

    /// The decode error for a positively-sniffed browser-undecodable format
    /// carries the sniff INSIDE the stable "image decoder:" prefix — the exact
    /// string the backend render worker gates its browser-parity accept on.
    #[test]
    fn decode_raster_error_carries_undecodable_sniff() {
        for (bytes, kind) in [
            (emf_bytes(), "emf"),
            (b"<!doctype html><html></html>".to_vec(), "html"),
        ] {
            let err = decode_raster(&bytes).unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.contains(&format!(
                    "image decoder: undecodable format (sniff: {kind})"
                )),
                "error must carry the {kind} sniff: {msg}"
            );
        }
        // Unknown junk keeps the plain prefix, with NO sniff marker.
        let err = decode_raster(b"\x00\x01\x02 junk bytes").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("image decoder:"),
            "stable prefix survives: {msg}"
        );
        assert!(
            !msg.contains("sniff:"),
            "no positive sniff on unknown junk: {msg}"
        );
    }

    /// A small 8-bit PALETTE PNG (color type 3, PLTE + tRNS, non-interlaced)
    /// — the exact shape of the clipboard-pasted prod pair. Encoded with
    /// `NoCompression` + `NoFilter`, so the IDAT payload is a single zlib
    /// STORED block whose scanline filter bytes sit at computable offsets.
    const PAL_W: u32 = 16;
    const PAL_H: u32 = 16;
    /// zlib header (2) + stored-block header (1 + LEN 2 + NLEN 2).
    const STORED_PREFIX: usize = 7;

    fn encode_palette_png() -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, PAL_W, PAL_H);
            enc.set_color(png::ColorType::Indexed);
            enc.set_depth(png::BitDepth::Eight);
            enc.set_palette(vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9]);
            enc.set_trns(vec![255, 255, 255, 0]);
            enc.set_compression(png::Compression::NoCompression);
            enc.set_filter(png::Filter::NoFilter);
            let mut w = enc.write_header().unwrap();
            let px: Vec<u8> = (0..PAL_W * PAL_H).map(|i| (i % 4) as u8).collect();
            w.write_image_data(&px).unwrap();
        }
        out
    }

    /// Byte range `[start, end)` of the IDAT chunk PAYLOAD in a single-IDAT PNG.
    fn idat_payload_range(png: &[u8]) -> (usize, usize) {
        let mut off = 8;
        let mut found = None;
        while off + 8 <= png.len() {
            let len =
                u32::from_be_bytes([png[off], png[off + 1], png[off + 2], png[off + 3]]) as usize;
            if &png[off + 4..off + 8] == b"IDAT" {
                assert!(found.is_none(), "fixture must carry exactly one IDAT");
                found = Some((off + 8, off + 8 + len));
            }
            off += 12 + len;
        }
        found.expect("IDAT present")
    }

    /// The prod corruption shape: a garbage scanline FILTER byte part-way down
    /// the image (the decoder's `UnknownFilterMethod`), which also invalidates
    /// the adler32 and the IDAT CRC. Chunk length + IEND are untouched, so the
    /// file is structurally complete — exactly what `createImageBitmap` rejects.
    fn corrupt_palette_png(row: usize, filter: u8) -> Vec<u8> {
        let mut png = encode_palette_png();
        let (start, _) = idat_payload_range(&png);
        let rowlen = PAL_W as usize + 1;
        let at = start + STORED_PREFIX + row * rowlen;
        assert_eq!(
            png[at], 0,
            "fixture row {row} must start with filter byte 0 (NoFilter)"
        );
        png[at] = filter;
        png
    }

    /// The parity case the 2026-09-08 PDF-export incident hit: a decoder that
    /// CONSTRUCTS (intact IHDR/PLTE/tRNS) and then trips a FORMAT error in the
    /// pixel stream of a COMPLETE file. It must carry the positive sniff on the
    /// stable "image decode:" prefix so the backend accepts the baked
    /// placeholder as browser parity instead of failing the export closed.
    #[test]
    fn corrupt_complete_palette_png_carries_corrupt_png_sniff() {
        // The pristine fixture decodes — proves the corruption below is the
        // only reason the decode fails.
        let (rgba, w, h, _) =
            decode_raster(&encode_palette_png()).expect("pristine fixture decodes");
        assert_eq!((w, h), (PAL_W, PAL_H));
        assert_eq!(&rgba[..4], &[255, 0, 0, 255]);
        assert_eq!(rgba[15], 0, "tRNS alpha applied to palette index 3");

        for (row, filter) in [(9usize, 5u8), (3, 8)] {
            let bad = corrupt_palette_png(row, filter);
            let err = decode_raster(&bad).unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.contains("image decode: undecodable format (sniff: corrupt-png)"),
                "complete-but-corrupt PNG must carry the corrupt-png sniff: {msg}"
            );
            assert!(
                msg.contains(&format!("Unknown filter method {filter}")),
                "the png crate's own cause survives in the chain: {msg}"
            );
        }
    }

    /// The marker the Rust `fail_on_missing_assets` gates read is TYPED: the
    /// same sniff the reason string carries must be recoverable from the anyhow
    /// chain by downcast (both `decoder` and `decode` stages emit the same type),
    /// and absent for every other failure.
    #[test]
    fn browser_undecodable_kind_is_recoverable_from_the_error_chain() {
        let emf = decode_raster(&emf_bytes()).unwrap_err();
        assert_eq!(browser_undecodable_kind(&emf), Some("emf"));
        let pixel_stage = decode_raster(&corrupt_palette_png(9, 5)).unwrap_err();
        assert_eq!(browser_undecodable_kind(&pixel_stage), Some("corrupt-png"));
        // A header-stage Format error on a complete PNG is NOT sniffed (see
        // `corrupt_header_chunk_of_complete_png_is_not_sniffed`).
        let mut bad_plte = encode_palette_png();
        let at = chunk_crc_offset(&bad_plte, b"PLTE");
        bad_plte[at] ^= 0xFF;
        let construction_stage = decode_raster(&bad_plte).unwrap_err();
        assert_eq!(browser_undecodable_kind(&construction_stage), None);
        // The Display of the marker IS the stable reason grammar (no drift
        // between what the backend substring-matches and what the gate reads).
        assert_eq!(
            format!("{pixel_stage:#}")
                .split(": Format error")
                .next()
                .unwrap(),
            "image decode: undecodable format (sniff: corrupt-png)"
        );

        let junk = decode_raster(b"\x00\x01\x02 junk bytes").unwrap_err();
        assert_eq!(browser_undecodable_kind(&junk), None);
        let pristine = encode_palette_png();
        let truncated = decode_raster(&pristine[..pristine.len() - 40]).unwrap_err();
        assert_eq!(browser_undecodable_kind(&truncated), None);
        // A context layer wrapped around the chain (the fetch lane's `{e:#}`
        // flattening happens AFTER this) does not hide the marker.
        let wrapped = decode_raster(&emf_bytes())
            .unwrap_err()
            .context("loading https://x/a");
        assert_eq!(browser_undecodable_kind(&wrapped), Some("emf"));
    }

    /// Byte offset of the 4-byte CRC of the first chunk of `kind`.
    fn chunk_crc_offset(png: &[u8], kind: &[u8; 4]) -> usize {
        let mut off = 8;
        while off + 8 <= png.len() {
            let len =
                u32::from_be_bytes([png[off], png[off + 1], png[off + 2], png[off + 3]]) as usize;
            if &png[off + 4..off + 8] == kind {
                return off + 8 + len;
            }
            off += 12 + len;
        }
        panic!("chunk {:?} present", std::str::from_utf8(kind));
    }

    /// PNG chunk CRC (ISO 3309), for building well-formed fixture chunks.
    fn png_crc(kind: &[u8; 4], data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for &b in kind.iter().chain(data) {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// Insert a well-formed chunk immediately after IHDR.
    fn insert_chunk_after_ihdr(png: &[u8], kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let ihdr_end = chunk_crc_offset(png, b"IHDR") + 4;
        let mut out = png[..ihdr_end].to_vec();
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        out.extend_from_slice(&png_crc(kind, data).to_be_bytes());
        out.extend_from_slice(&png[ihdr_end..]);
        out
    }

    /// A header-stage `Format` error on a COMPLETE PNG (a critical-chunk fault
    /// while the decoder constructs — here a PLTE CRC) is deliberately NOT
    /// sniffed: it is a different population from the demonstrated IDAT class
    /// and has no incident evidence, so it keeps the plain "image decoder:"
    /// decline. The two ancillary cases pin the leniency the pixel-stage premise
    /// rests on: a malformed ancillary VALUE (`gAMA` = 0, valid CRC) and an
    /// ancillary chunk with a bad CRC both decode clean (png 0.18 swallows
    /// ancillary `Format` errors as `BadAncillaryChunk` and skips ancillary CRC
    /// failures — libpng's warn-and-discard). A png bump that starts rejecting
    /// either would fail here and reopen the question.
    #[test]
    fn corrupt_header_chunk_of_complete_png_is_not_sniffed() {
        let mut bad_plte = encode_palette_png();
        let at = chunk_crc_offset(&bad_plte, b"PLTE");
        bad_plte[at] ^= 0xFF;
        assert!(png_is_structurally_complete(&bad_plte));
        let err = decode_raster(&bad_plte).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.starts_with("image decoder: "), "stable prefix: {msg}");
        assert!(msg.contains("CRC error"), "png crate cause survives: {msg}");
        assert!(
            !msg.contains("sniff:"),
            "header-stage fault is never sniffed: {msg}"
        );

        // Malformed ancillary metadata: valid-CRC gAMA chunk with gamma 0.
        let bad_gama = insert_chunk_after_ihdr(&encode_palette_png(), b"gAMA", &[0, 0, 0, 0]);
        assert!(png_is_structurally_complete(&bad_gama));
        let (rgba, w, h, _) =
            decode_raster(&bad_gama).expect("ancillary metadata fault is swallowed, not fatal");
        assert_eq!((w, h), (PAL_W, PAL_H));
        assert_eq!(&rgba[..4], &[255, 0, 0, 255]);
        // Sanity: the fixture surgery itself is sound (a valid gAMA decodes too).
        let good_gama =
            insert_chunk_after_ihdr(&encode_palette_png(), b"gAMA", &45455u32.to_be_bytes());
        decode_raster(&good_gama).expect("well-formed gAMA decodes");

        let mut bad_trns = encode_palette_png();
        let at = chunk_crc_offset(&bad_trns, b"tRNS");
        bad_trns[at] ^= 0xFF;
        let (rgba, w, h, _) =
            decode_raster(&bad_trns).expect("ancillary CRC fault is skipped, not fatal");
        assert_eq!((w, h), (PAL_W, PAL_H));
        assert_eq!(rgba.len(), (PAL_W * PAL_H * 4) as usize);
    }

    /// A TRUNCATED body must NOT be sniffed: the next fetch may return the whole
    /// file, so it keeps its decline/retry path. Both a clean truncation and a
    /// corrupt-then-truncated body fall on the un-sniffed side.
    #[test]
    fn truncated_png_keeps_plain_decode_error() {
        let pristine = encode_palette_png();
        let corrupt = corrupt_palette_png(9, 5);
        let (_, idat_end) = idat_payload_range(&pristine);
        for bytes in [
            &pristine[..idat_end - 40],       // mid-IDAT, no IEND
            &corrupt[..idat_end - 40],        // corrupt AND cut short
            &pristine[..pristine.len() - 12], // whole IDAT, IEND chunk missing
        ] {
            let err = decode_raster(bytes).unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.contains("image decode"),
                "stable prefix survives: {msg}"
            );
            assert!(
                !msg.contains("sniff:"),
                "a truncated body is never sniffed: {msg}"
            );
        }
    }

    #[test]
    fn png_structural_completeness_walk() {
        let png = encode_palette_png();
        assert!(png_is_structurally_complete(&png));
        assert!(
            png_is_structurally_complete(&corrupt_palette_png(9, 5)),
            "bad CRC/adler is still complete"
        );
        assert!(
            !png_is_structurally_complete(&png[..png.len() - 1]),
            "short IEND"
        );
        assert!(
            !png_is_structurally_complete(&png[..png.len() - 12]),
            "no IEND"
        );
        assert!(
            !png_is_structurally_complete(&png[..40]),
            "cut inside a chunk"
        );
        let mut trailing = png.clone();
        trailing.extend_from_slice(b"junk");
        assert!(!png_is_structurally_complete(&trailing), "bytes after IEND");
        assert!(
            !png_is_structurally_complete(b"\x89PNG\r\n\x1a\n"),
            "signature only"
        );
        assert!(!png_is_structurally_complete(&[]));
        // A JPEG is a raster too, but only PNG carries the sniff today.
        assert!(!png_is_structurally_complete(&[0xFF, 0xD8, 0xFF, 0xE0]));
    }

    /// The sniff gates on the ERROR class as well as the bytes: a complete PNG
    /// paired with a non-format error (IO/limits) is never called corrupt.
    #[test]
    fn corrupt_sniff_requires_a_decoding_error() {
        let png = encode_palette_png();
        let io =
            image::ImageError::IoError(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        assert_eq!(sniff_corrupt_complete_raster(&png, &io), None);
        let limits = image::ImageError::Limits(image::error::LimitError::from_kind(
            image::error::LimitErrorKind::InsufficientMemory,
        ));
        assert_eq!(sniff_corrupt_complete_raster(&png, &limits), None);
        let format = image::ImageError::Decoding(image::error::DecodingError::new(
            image::ImageFormat::Png.into(),
            "synthetic",
        ));
        assert_eq!(
            sniff_corrupt_complete_raster(&png, &format),
            Some("corrupt-png")
        );
        assert_eq!(
            sniff_corrupt_complete_raster(&png[..40], &format),
            None,
            "truncated"
        );
    }
}
