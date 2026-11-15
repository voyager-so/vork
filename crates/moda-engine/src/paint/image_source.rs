//! image_source.rs — minimal pattern-texture sourcing for M3b (data URIs).
//!
//! SEAM NOTE (M3b → M3c/M4): the real asset pipeline (HTTP fetch, ICC→sRGB
//! via moxcms, resvg SVG raster, disk cache, fit/align `imageFillProperties`)
//! belongs to `moda-assets` (M3c). At M3b time that crate is an empty stub
//! being built concurrently, so this module implements ONLY what the
//! patterns/distort-pattern fixture axes need: `data:` URI decode (PNG/JPEG via
//! the pinned `image` crate) + Chromium-parity premultiply. At integration
//! (M4) `resolve_pattern_source` should delegate non-data URLs to moda-assets
//! and keep the placeholder mapping below.
//!
//! Placeholder semantics (copy the bug — Q-22,
//! `WebGPUPathRenderer.ts:212-231, 1232-1454`;
//! `WebGPUConstants.ts PATH_RENDERER_DEFAULTS`):
//! - loading  → 1×1 grey (0.9, 0.9, 0.9), force-tiled, filters/perspective off
//! - failed   → 1×1 rose (0.85, 0.74, 0.74), same flags
//! - no src   → 1×1 white fallback, NOT a placeholder (filters still apply)

use std::sync::{Arc, OnceLock};

/// `PATH_RENDERER_DEFAULTS.IMAGE_LOADING_FILL` (premultiplied bytes; a = 1).
pub const LOADING_RGBA: [u8; 4] = [230, 230, 230, 255]; // round(0.9·255) = 230
/// `PATH_RENDERER_DEFAULTS.IMAGE_FAILED_FILL` ("rose").
pub const FAILED_RGBA: [u8; 4] = [217, 189, 189, 255]; // round(.85/.74/.74·255)
/// 1×1 white fallback texture (`WebGPUTextureManager.createFallbackTexture`).
pub const FALLBACK_RGBA: [u8; 4] = [255, 255, 255, 255];

/// Presentation state (`ImageFillState` / `PatternTextureState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternTextureState {
    Loaded,
    Loading,
    Failed,
    Fallback,
}

/// A decoded, premultiplied sRGB RGBA8 image ready for `write_texture`.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// Premultiplied RGBA8 (the browser uploads with
    /// `copyExternalImageToTexture(..., premultipliedAlpha: true)`). EMPTY when
    /// `external` is `Some`: the wasm editor lane keeps decoded pixels off the
    /// heap, so there are no bytes here — the texture is filled from the
    /// installed `ExternalImageUploader` at bind time.
    pub rgba_premultiplied: Arc<Vec<u8>>,
    /// Whether `rgba_premultiplied` holds straight (unassociated) colour instead
    /// (`moda_assets::PreparedImage::straight`): decoded texels kept for an engine that converts them to its
    /// working colour and associates them itself (`paint::draw`).
    pub straight: bool,
    /// `Some` → a browser-decoded, GPU-resident raster (wasm editor lane):
    /// `PatternTextureCache::bind_group` uploads it via the external uploader
    /// keyed by `key` instead of `write_texture`-ing `rgba_premultiplied`.
    pub external: Option<ExternalTextureRef>,
}

/// Reference to a browser-decoded image the engine uploads straight to a GPU
/// texture (wasm editor lane). Carries no pixels: `key` names the source in
/// the installed [`crate::external_texture::ExternalImageUploader`]'s registry,
/// and `upload_id` gives the retained texture cache its self-invalidating
/// identity (the heap lane uses the decoded-pixel `Arc` pointer instead).
#[derive(Debug, Clone)]
pub struct ExternalTextureRef {
    pub key: String,
    pub upload_id: u64,
}

/// A resolved pattern texture: either real decoded bytes or a 1×1 placeholder.
#[derive(Debug, Clone)]
pub struct PatternTexture {
    pub state: PatternTextureState,
    /// Cache key for GPU texture/bind-group dedup (the src string, or a
    /// placeholder tag).
    pub key: String,
    pub image: DecodedImage,
    /// M3c: uploaded / natural texel ratio per axis — the production
    /// `CachedTexture.scaleFactor`. `fillPatternScale` is divided by this
    /// at instance packing (`WebGPUPathRenderer.ts:1338-1345`). (1, 1) for
    /// the inline data-URI fallback path and all placeholders.
    pub scale_factor: (f64, f64),
}

fn shared_one_by_one_bytes(rgba: [u8; 4]) -> Arc<Vec<u8>> {
    match rgba {
        FALLBACK_RGBA => {
            static BYTES: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
            Arc::clone(BYTES.get_or_init(|| Arc::new(FALLBACK_RGBA.to_vec())))
        }
        FAILED_RGBA => {
            static BYTES: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
            Arc::clone(BYTES.get_or_init(|| Arc::new(FAILED_RGBA.to_vec())))
        }
        LOADING_RGBA => {
            static BYTES: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
            Arc::clone(BYTES.get_or_init(|| Arc::new(LOADING_RGBA.to_vec())))
        }
        _ => Arc::new(rgba.to_vec()),
    }
}

fn one_by_one(state: PatternTextureState, key: &str, rgba: [u8; 4]) -> PatternTexture {
    PatternTexture {
        state,
        key: key.to_string(),
        image: DecodedImage {
            width: 1,
            height: 1,
            // Shared static Arc for the known placeholder RGBAs — no per-call
            // allocation (buffer-churn fix, main e85f7c1f).
            rgba_premultiplied: shared_one_by_one_bytes(rgba),
            straight: false,
            external: None,
        },
        scale_factor: (1.0, 1.0),
    }
}

/// Resolve a pattern fill's texture from its `fillPatternImageSrc`.
///
/// - `None` / empty → white fallback (state `Fallback`, filters apply),
/// - `data:image/...` → decode now; decode failure → `Failed` placeholder,
/// - any other scheme → `Loading` placeholder (what an un-arrived texture
///   bakes into a browser export on timeout). M3c replaces this arm.
pub fn resolve_pattern_source(src: Option<&str>) -> PatternTexture {
    let Some(src) = src.filter(|s| !s.is_empty()) else {
        return one_by_one(PatternTextureState::Fallback, "__fallback", FALLBACK_RGBA);
    };
    // M3c seam delegation: when the CLI has prefetched + installed an
    // `AssetStore` (`moda_assets::engine_seam::install_store`), EVERY source
    // resolves through the full asset pipeline — data URIs included, so the
    // production sizing-policy resize (small-image/SVG min-dimension
    // upscale + scale-factor compensation) and resvg SVG rasterization apply
    // exactly as in the goldens. Without a store (engine unit tests,
    // store-less embedding) the inline data-URI fallback below keeps M3b's
    // behavior.
    // A paint layer's surface (raster spike): its texture, not an asset.
    if let Some(texture) = crate::surface::pattern_texture(src) {
        return texture;
    }
    if let Some(resolved) = moda_assets::engine_seam::resolve(src) {
        use moda_assets::ImageFillState;
        return match resolved.state {
            ImageFillState::Loaded => loaded_to_pattern_texture(src, resolved),
            ImageFillState::Failed => {
                one_by_one(PatternTextureState::Failed, "__failed", FAILED_RGBA)
            }
            ImageFillState::Loading => {
                one_by_one(PatternTextureState::Loading, "__loading", LOADING_RGBA)
            }
        };
    }
    if let Some(rest) = src.strip_prefix("data:") {
        match decode_data_uri(rest) {
            Ok(img) => PatternTexture {
                state: PatternTextureState::Loaded,
                key: src.to_string(),
                image: img,
                scale_factor: (1.0, 1.0),
            },
            Err(_) => {
                moda_assets::report::record(moda_assets::report::RuntimeEvent::PlaceholderBaked {
                    src: src.to_string(),
                    state: "failed".into(),
                    shader_image: false,
                });
                one_by_one(PatternTextureState::Failed, "__failed", FAILED_RGBA)
            }
        }
    } else {
        // SEAM: store-less fallback — non-data URLs bake the loading
        // placeholder (what an un-arrived texture bakes into a browser
        // export on timeout).
        moda_assets::report::record(moda_assets::report::RuntimeEvent::PlaceholderBaked {
            src: src.to_string(),
            state: "never-prefetched".into(),
            shader_image: false,
        });
        one_by_one(PatternTextureState::Loading, "__loading", LOADING_RGBA)
    }
}

/// Map a store-resolved `Loaded` asset to a `PatternTexture`. Heap bytes
/// (native / SVG / shader) carry pixels; an external (browser-decoded) raster
/// or live video frame carries only its upload reference — `rgba_premultiplied`
/// stays empty and the texture is filled from the external uploader at bind time.
/// Factored out so the image (`resolve_pattern_source`) and video
/// streaming (`resolve_video_source`) paths share identical texel handling.
fn loaded_to_pattern_texture(
    key: &str,
    resolved: moda_assets::engine_seam::ResolvedAsset,
) -> PatternTexture {
    use moda_assets::engine_seam::ResolvedPixels;
    let (rgba_premultiplied, external) = match resolved.pixels {
        ResolvedPixels::Heap(bytes) => (bytes, None),
        ResolvedPixels::External { key, upload_id } => (
            Arc::new(Vec::new()),
            Some(ExternalTextureRef { key, upload_id }),
        ),
    };
    PatternTexture {
        state: PatternTextureState::Loaded,
        key: key.to_string(),
        image: DecodedImage {
            width: resolved.width,
            height: resolved.height,
            rgba_premultiplied,
            straight: resolved.straight,
            external,
        },
        scale_factor: resolved.scale_factor,
    }
}

/// Resolve a `'video'` fill's texture: prefer the LIVE streaming frame installed
/// under the per-node key `video:<nodeId>` (by the VideoPlaybackController),
/// falling back to the poster URL when none has arrived — the
/// poster-vs-streaming rule in `docs/roadmap/canvas/video-nodes.md`.
///
/// Phase 0 (poster-only) always took the poster arm because the streaming key
/// was never installed; Phase 1 lights it up. With no live frame this is
/// byte-identical to `resolve_pattern_source(poster_src)`, so an idle video, a
/// never-played video, and a static export all still render the poster exactly
/// as before — which is what keeps the `mutation_replay` plan byte-exact for any
/// fixture that doesn't install a live frame.
pub fn resolve_video_source(stream_key: &str, poster_src: Option<&str>) -> PatternTexture {
    if let Some(resolved) = moda_assets::engine_seam::resolve_streaming(stream_key) {
        return loaded_to_pattern_texture(stream_key, resolved);
    }
    resolve_pattern_source(poster_src)
}

fn decode_data_uri(rest: &str) -> Result<DecodedImage, String> {
    // data:[<mediatype>][;base64],<data>
    let (meta, payload) = rest.split_once(',').ok_or("malformed data URI")?;
    let bytes = if meta.ends_with(";base64") || meta.contains(";base64;") {
        base64_decode(payload)?
    } else {
        // Percent-decoded text payloads (e.g. data: SVG) are out of M3b scope.
        return Err("non-base64 data URI".into());
    };
    decode_image_bytes(&bytes)
}

/// Decode PNG/JPEG bytes to premultiplied RGBA8 (sRGB).
pub fn decode_image_bytes(bytes: &[u8]) -> Result<DecodedImage, String> {
    let img = image::load_from_memory(bytes).map_err(|e| e.to_string())?;
    let rgba = img.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    let mut data = rgba.into_raw();
    premultiply_in_place(&mut data);
    Ok(DecodedImage {
        width,
        height,
        rgba_premultiplied: Arc::new(data),
        straight: false,
        external: None,
    })
}

/// Premultiply straight RGBA8 in place, round-to-nearest (Chromium's
/// `copyExternalImageToTexture(premultipliedAlpha: true)` conversion).
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

/// Tiny RFC-4648 base64 decoder (standard alphabet, '=' padding, whitespace
/// tolerated) — avoids a new dependency for fixture data URIs.
pub fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Result<u32, String> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 byte {c}")),
        }
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        if c.is_ascii_whitespace() || c == b'=' {
            continue;
        }
        acc = (acc << 6) | val(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 16×16 PNG data URI used by the patterns/basic-repeat fixtures.
    const FIXTURE_PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAARElEQVR42mO8Y2TEAANhGqvg7BN5PnC23iOEGiYGEgHtNTAaRd0h6O4tHTcGtR9+ndAg6G7k+BmEfmAhxt3I8TMI/QAA7mAX9A/NHxMAAAAASUVORK5CYII=";

    #[test]
    fn base64_round_trip() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8h").unwrap(), b"hello!");
        assert_eq!(base64_decode("aA==").unwrap(), b"h");
        assert!(base64_decode("a-b").is_err());
    }

    #[test]
    fn decodes_fixture_png_data_uri() {
        let tex = resolve_pattern_source(Some(FIXTURE_PNG));
        assert_eq!(tex.state, PatternTextureState::Loaded);
        assert_eq!(tex.image.width, 16);
        assert_eq!(tex.image.height, 16);
        assert_eq!(tex.image.rgba_premultiplied.len(), 16 * 16 * 4);
    }

    #[test]
    fn placeholder_mapping() {
        let none = resolve_pattern_source(None);
        let none_again = resolve_pattern_source(None);
        assert_eq!(none.state, PatternTextureState::Fallback);
        assert_eq!(&*none.image.rgba_premultiplied, &FALLBACK_RGBA);
        assert!(Arc::ptr_eq(
            &none.image.rgba_premultiplied,
            &none_again.image.rgba_premultiplied
        ));

        let bad = resolve_pattern_source(Some("data:image/png;base64,!!!!"));
        let bad_again = resolve_pattern_source(Some("data:image/png;base64,!!!!"));
        assert_eq!(bad.state, PatternTextureState::Failed);
        assert_eq!(&*bad.image.rgba_premultiplied, &FAILED_RGBA);
        assert!(Arc::ptr_eq(
            &bad.image.rgba_premultiplied,
            &bad_again.image.rgba_premultiplied
        ));

        let http = resolve_pattern_source(Some("https://example.com/a.png"));
        let http_again = resolve_pattern_source(Some("https://example.com/a.png"));
        assert_eq!(http.state, PatternTextureState::Loading);
        assert_eq!(&*http.image.rgba_premultiplied, &LOADING_RGBA);
        assert!(Arc::ptr_eq(
            &http.image.rgba_premultiplied,
            &http_again.image.rgba_premultiplied
        ));
    }

    #[test]
    fn video_source_falls_back_to_poster_without_a_live_frame() {
        // No store installed ⇒ no streaming entry for the per-node key, so a
        // `'video'` fill resolves its poster exactly like a pattern fill (the
        // P0 poster-only behavior, and the static-export / idle path in P1).
        // This is the branch that keeps `mutation_replay` byte-exact when no
        // live frame is fed.
        let stream_key = "video:node-abc";
        let via_video = resolve_video_source(stream_key, Some(FIXTURE_PNG));
        let via_poster = resolve_pattern_source(Some(FIXTURE_PNG));
        assert_eq!(via_video.state, PatternTextureState::Loaded);
        assert_eq!(
            (via_video.image.width, via_video.image.height),
            (via_poster.image.width, via_poster.image.height)
        );
        assert_eq!(
            &*via_video.image.rgba_premultiplied,
            &*via_poster.image.rgba_premultiplied
        );
        // No poster + no live frame ⇒ the white fallback (filters still apply),
        // not a placeholder — identical to a pattern fill with no src.
        assert_eq!(
            resolve_video_source(stream_key, None).state,
            PatternTextureState::Fallback
        );
    }

    #[test]
    fn premultiply_rounds_to_nearest() {
        let mut px = [200u8, 100, 0, 128];
        premultiply_in_place(&mut px);
        // 200·128/255 = 100.39 → 100; 100·128/255 = 50.19 → 50.
        assert_eq!(px, [100, 50, 0, 128]);
    }
}
