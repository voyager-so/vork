//! M3f — procedural-fill instance packing runtime.
//!
//! Ports the CPU side of `defineProceduralShader.packInstanceData`
//! (`renderers/webgpu/shaders/base/defineProceduralShader.ts:237-292`) plus
//! the per-shader custom `packInstanceData` / `resolve` callbacks (see
//! `packs.rs`), driven by the extraction manifest's packing tables
//! (`manifest.modules.<id>.packing` — the only authority per DECISION §1.3).
//!
//! Output is the `shaderParams` (64 f32) + `colorData` (64 f32) tail of the
//! 672-byte `CustomShaderData` ABI; the engine owns the shared header fields
//! (model/fill/stroke/pathInfo/bounds/pageClip) and splices these in via
//! `seams::PaintInstance`.
//!
//! Determinism: the production render context pins `shaderTimeSeconds: 0`
//! for exports/screenshots (`WebGPUPathRenderer.ts:260, 367`); the CLI's
//! `--shader-time-s` flows in here as `shader_time_s`. Session-random
//! descriptor defaults (Q-29) are replaced by the manifest's seeded-PRNG
//! values — fixtures must persist explicit seeds for golden parity.

pub mod image_processing;
pub mod jscolor;
pub mod jsval;
mod packs;

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::Value;

use jscolor::color_to_normalized_rgba;
use jsval::{as_finite_number, as_string_array, get_path_value};

pub const SHADER_PARAM_FLOAT_CAPACITY: usize = 64;
pub const COLOR_DATA_FLOAT_CAPACITY: usize = 64;

/// Packed procedural-fill instance tail.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedFill {
    /// Resolved module id (post `shader_id_aliases`); keys the WGSL module,
    /// pipeline descriptors, and `<id>.vector-text-fill` variant.
    pub module_id: String,
    pub shader_params: [f32; SHADER_PARAM_FLOAT_CAPACITY],
    pub color_data: [f32; COLOR_DATA_FLOAT_CAPACITY],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FillPackError {
    /// Not in the registry (after alias resolution) — manifest `unsupported`.
    UnknownShader(String),
}

impl std::fmt::Display for FillPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FillPackError::UnknownShader(id) => write!(f, "unknown shader id '{id}'"),
        }
    }
}

/// M4e — `ShaderImageResource` metadata for image-backed shaders
/// (`renderers/webgpu/shaders/base/ShaderResources.ts:8-20`). `ready: false`
/// is the manager's 1×1 fallback entry (also what the `fillShaderId`-only
/// route ALWAYS gets — see `image_source_quirk` note below).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShaderImageMeta {
    pub width: f64,
    pub height: f64,
    pub aspect_ratio: f64,
    pub ready: bool,
    /// Terminal load/decode failure. False with `ready == false` means the
    /// resource is still loading or was never prefetched.
    pub failed: bool,
    /// heatmap `contentRect`: [offsetX, offsetY, scaleX, scaleY].
    pub content_rect: Option<[f64; 4]>,
}

impl ShaderImageMeta {
    /// `createFallbackEntry` metadata (`ShaderImageResourceManager.ts:137-159`).
    pub fn fallback() -> ShaderImageMeta {
        ShaderImageMeta {
            width: 1.0,
            height: 1.0,
            aspect_ratio: 1.0,
            ready: false,
            failed: false,
            content_rect: None,
        }
    }

    pub fn failed() -> ShaderImageMeta {
        ShaderImageMeta {
            failed: true,
            ..ShaderImageMeta::fallback()
        }
    }

    pub fn loaded(width: u32, height: u32, content_rect: Option<[f64; 4]>) -> ShaderImageMeta {
        ShaderImageMeta {
            width: width as f64,
            height: height as f64,
            aspect_ratio: width as f64 / (height as f64).max(1.0),
            ready: true,
            failed: false,
            content_rect,
        }
    }
}

impl std::error::Error for FillPackError {}

// ---------------------------------------------------------------------------
// Typed packing tables (deserialized from `ModuleMeta.packing`)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct PackingMeta {
    pub authoring_mode: String,
    pub clock_param_key: Option<String>,
    pub colors: Option<ColorsMeta>,
    pub defaults: Value,
    pub has_custom_pack: bool,
    pub params: Vec<ParamMeta>,
    pub primary_image: Option<PrimaryImageMeta>,
    pub uses_renderer_clock: bool,
}

#[derive(Debug, Deserialize)]
pub struct PrimaryImageMeta {
    pub control_key: String,
    pub processing: Value,
    #[serde(default)]
    pub placement: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ParamMeta {
    pub key: String,
    pub offset: usize,
    pub fallback: f64,
    pub has_resolve: bool,
}

#[derive(Debug, Deserialize)]
pub struct ColorsMeta {
    pub key: String,
    pub max: usize,
    pub pad_with_defaults: bool,
    pub fallback: Option<Vec<String>>,
}

/// Packing tables for every `procedural-fill` module, parsed once.
pub fn packing_tables() -> &'static BTreeMap<String, PackingMeta> {
    static TABLES: OnceLock<BTreeMap<String, PackingMeta>> = OnceLock::new();
    TABLES.get_or_init(|| {
        let manifest = crate::manifest();
        manifest
            .modules
            .iter()
            .filter(|(_, meta)| meta.class == "procedural-fill")
            .filter_map(|(id, meta)| {
                let packing = meta.packing.as_ref()?;
                let parsed: PackingMeta = serde_json::from_value(packing.clone())
                    .unwrap_or_else(|e| panic!("packing table for '{id}' deserializes: {e}"));
                Some((id.clone(), parsed))
            })
            .collect()
    })
}

/// `resolveShaderId` (`core/shaders/shaderIdAliases.ts:13-17`), backed by the
/// manifest's extracted alias map.
pub fn resolve_shader_id(raw: &str) -> &str {
    crate::manifest()
        .shader_id_aliases
        .get(raw)
        .map(String::as_str)
        .unwrap_or(raw)
}

/// `ShaderManager.supportsTextShaderFill` (`ShaderManager.ts:181-184`,
/// `defineProceduralShader.ts:207-209`): can the shader registered under this
/// (possibly legacy-alias) id drive a text fill? True iff the resolved id is
/// in the manifest's extracted eligibility set (descriptor `shade` shader
/// without a `primaryImage` resource). Unknown ids return false without
/// logging, exactly like the production wrapper.
pub fn supports_vector_text_fill(raw_shader_id: &str) -> bool {
    static ELIGIBLE: OnceLock<std::collections::HashSet<&'static str>> = OnceLock::new();
    let set = ELIGIBLE.get_or_init(|| {
        crate::manifest()
            .vector_text_fill
            .eligible_shader_ids
            .iter()
            .map(String::as_str)
            .collect()
    });
    set.contains(resolve_shader_id(raw_shader_id))
}

/// Module id of the `<id>.vector-text-fill` WGSL variant for a text-fill
/// eligible shader (post alias resolution).
pub fn vector_text_fill_module_id(raw_shader_id: &str) -> String {
    format!("{}.vector-text-fill", resolve_shader_id(raw_shader_id))
}

/// `resolveShaderTimeSeconds` (`ShaderBufferLayout.ts:14-17`).
fn resolve_shader_time_seconds(time_s: f64, fallback: f64) -> f64 {
    if time_s.is_finite() {
        time_s
    } else {
        fallback
    }
}

// ---------------------------------------------------------------------------
// Pack context — the closures handed to packInstanceData, as a struct
// ---------------------------------------------------------------------------

pub(crate) struct Ctx<'a> {
    /// Raw node params (`node.fillShaderParams ?? {}`).
    pub params: &'a Value,
    /// Descriptor defaults (manifest packing `defaults`).
    pub defaults: &'a Value,
    /// `usesRendererClock(descriptor) ? (clock.paramKey ?? 'time') : null`.
    pub clock_key: Option<String>,
    /// `--shader-time-s` (export context pins 0).
    pub time_s: f64,
    /// M4e: `resources.primaryImage` metadata (image-backed shaders only).
    pub image: Option<ShaderImageMeta>,
}

impl Ctx<'_> {
    /// `getNumberParam` (`defineProceduralShader.ts:251-256`): clock key →
    /// renderer time; else params path → defaults path → fallback (finite
    /// numbers only).
    pub fn get_number_param(&self, key: &str, fallback: f64) -> f64 {
        if self.clock_key.as_deref() == Some(key) {
            // rendererClockTimeSeconds = resolveShaderTimeSeconds(ctx): our
            // time is always defined, mirroring the export render context.
            return resolve_shader_time_seconds(self.time_s, 0.0);
        }
        self.get_number_param_value(key, fallback)
    }

    /// `getNumberParamValue` (`defineProceduralShader.ts:686-693`).
    pub fn get_number_param_value(&self, key: &str, fallback: f64) -> f64 {
        if let Some(v) = get_path_value(self.params, key).and_then(as_finite_number) {
            return v;
        }
        get_path_value(self.defaults, key)
            .and_then(as_finite_number)
            .unwrap_or(fallback)
    }

    /// `getColorArrayParamValue` (`defineProceduralShader.ts:695-717`).
    pub fn get_color_array_param(
        &self,
        key: &str,
        max: usize,
        fallback: Option<&[String]>,
    ) -> Vec<String> {
        if let Some(v) = get_path_value(self.params, key).and_then(as_string_array) {
            if !v.is_empty() {
                return v.into_iter().take(max).collect();
            }
        }
        if let Some(v) = get_path_value(self.defaults, key).and_then(as_string_array) {
            if !v.is_empty() {
                return v.into_iter().take(max).collect();
            }
        }
        if let Some(fb) = fallback {
            if !fb.is_empty() {
                return fb.iter().take(max).cloned().collect();
            }
        }
        vec!["#ffffff".to_string()]
    }

    /// `packNumberParamEntries` (`defineProceduralShader.ts:651-663`):
    /// per-entry `resolve` callback first (used when finite), else
    /// `getNumberParam(key, fallback)`.
    pub fn pack_number_params(
        &self,
        shader_id: &str,
        entries: &[ParamMeta],
        shader_params: &mut [f32; SHADER_PARAM_FLOAT_CAPACITY],
    ) {
        for entry in entries {
            let resolved = if entry.has_resolve {
                packs::resolve_param(shader_id, &entry.key, self)
            } else {
                None
            };
            let value = match resolved {
                Some(r) if r.is_finite() => r,
                _ => self.get_number_param(&entry.key, entry.fallback),
            };
            shader_params[entry.offset] = value as f32;
        }
    }

    /// `packColorArrayEntry` (`defineProceduralShader.ts:665-684`): alpha is
    /// ALWAYS packed as 1.0; unparsable colors become opaque white.
    pub fn pack_color_array(
        &self,
        entry: &ColorsMeta,
        color_data: &mut [f32; COLOR_DATA_FLOAT_CAPACITY],
    ) {
        let fallback = entry.fallback.as_deref();
        let colors = self.get_color_array_param(&entry.key, entry.max, fallback);
        // fallbackColors: same lookup with empty params (defaults/fallback).
        let empty = Value::Object(serde_json::Map::new());
        let fallback_ctx = Ctx {
            params: &empty,
            defaults: self.defaults,
            clock_key: None,
            time_s: self.time_s,
            image: self.image,
        };
        let fallback_colors = fallback_ctx.get_color_array_param(&entry.key, entry.max, fallback);
        let count = if entry.pad_with_defaults {
            entry.max
        } else {
            colors.len()
        };
        for i in 0..count {
            let color = colors
                .get(i)
                .or_else(|| fallback_colors.get(i % fallback_colors.len().max(1)))
                .map(String::as_str)
                .unwrap_or("#ffffff");
            let rgba = color_to_normalized_rgba(color);
            let offset = i * 4;
            color_data[offset] = rgba.map(|c| c.r).unwrap_or(1.0) as f32;
            color_data[offset + 1] = rgba.map(|c| c.g).unwrap_or(1.0) as f32;
            color_data[offset + 2] = rgba.map(|c| c.b).unwrap_or(1.0) as f32;
            color_data[offset + 3] = 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Pack one procedural fill: `raw_shader_id` (pre-alias), the node's
/// `fillShaderParams` (any JSON; non-objects behave like `{}` exactly as
/// `isRecord(nodeData) ? nodeData : {}`), and the deterministic shader clock.
///
/// Image-backed shaders (M4e) pack with the FALLBACK image metadata here
/// (`ready: false` — exactly what the `fillShaderId`-only production route
/// gets, since `WebGPUPathRenderer` only resolves an image source from a
/// shaderEffect: `WebGPUPathRenderer.ts:2050`). Callers with a resolved
/// shader-image resource use [`pack_fill_with_image`].
pub fn pack_fill(
    raw_shader_id: &str,
    params: &Value,
    shader_time_s: f64,
) -> Result<PackedFill, FillPackError> {
    pack_fill_with_image(raw_shader_id, params, shader_time_s, None)
}

/// M4e: [`pack_fill`] with `resources.primaryImage` metadata (the
/// `ShaderImageResourceManager` binding's `metadata` — width/height/aspect/
/// ready/contentRect) for image-backed shaders. `None` packs the manager's
/// fallback entry (`ready: 0`).
pub fn pack_fill_with_image(
    raw_shader_id: &str,
    params: &Value,
    shader_time_s: f64,
    image: Option<ShaderImageMeta>,
) -> Result<PackedFill, FillPackError> {
    let id = resolve_shader_id(raw_shader_id).to_string();
    let Some(meta) = packing_tables().get(&id) else {
        return Err(FillPackError::UnknownShader(raw_shader_id.to_string()));
    };

    // `isRecord(nodeData) ? nodeData : {}` (defineProceduralShader.ts:243).
    let empty = Value::Object(serde_json::Map::new());
    let params = if params.is_object() { params } else { &empty };

    // usesRendererClock(descriptor) ? (descriptor.clock?.paramKey ?? 'time').
    let clock_key = if meta.uses_renderer_clock {
        Some(
            meta.clock_param_key
                .clone()
                .unwrap_or_else(|| "time".to_string()),
        )
    } else {
        None
    };

    let ctx = Ctx {
        params,
        defaults: &meta.defaults,
        clock_key,
        time_s: shader_time_s,
        image: if meta.primary_image.is_some() {
            Some(image.unwrap_or_else(ShaderImageMeta::fallback))
        } else {
            None
        },
    };

    let mut shader_params = [0.0f32; SHADER_PARAM_FLOAT_CAPACITY];
    let mut color_data = [0.0f32; COLOR_DATA_FLOAT_CAPACITY];

    if meta.has_custom_pack {
        packs::custom_pack(&id, &ctx, &meta.params, &mut shader_params, &mut color_data)
            .unwrap_or_else(|| panic!("custom pack for '{id}' not ported (manifest drift?)"));
    } else {
        ctx.pack_number_params(&id, &meta.params, &mut shader_params);
        if let Some(colors) = &meta.colors {
            ctx.pack_color_array(colors, &mut color_data);
        }
    }

    Ok(PackedFill {
        module_id: id,
        shader_params,
        color_data,
    })
}

/// Shader playback: the `shader_params` float offsets that carry the
/// renderer clock for this fill — the ONLY slots that change when the shader
/// clock advances — so the live camera present path can re-apply a new clock to
/// a CACHED plan's shader instance (`render_plan_to_texture_impl`) without a
/// full plan rebuild.
///
/// Detected generically by packing the fill at several distinct finite clocks
/// and diffing: every ported pack writes the clock as a direct `sp[i] = time`
/// copy (standard `pack_number_params` clock-key path + every `runtime::packs`
/// custom pack), so a clock-driven slot equals its clock at every sample.
///
/// Returns:
/// * `Some(offsets)` — re-threadable: overwrite exactly these `shader_params`
///   slots with `time_s as f32` at present (empty = the fill has no clock).
/// * `None` — a clock-driven slot is NOT a direct copy (a non-linear clock use,
///   or a clock-dependent `color_data`): the caller must NOT re-thread and
///   should keep the clock in its cache key (per-frame rebuild) to stay correct.
///   No current shader hits this; it is a self-validating guard against manifest
///   drift. An unknown shader id also returns `None`.
///
/// Image metadata is clock-independent, so this packs with the manager fallback
/// image (`pack_fill`) — image-driven slots are identical at every sample and so
/// never appear in the diff.
pub fn clock_param_offsets(raw_shader_id: &str, params: &Value) -> Option<Vec<u16>> {
    // Sample at several distinct finite clocks. The direct-copy invariant means
    // a clock slot reads back exactly its input clock, so a driven slot must
    // equal its clock at EVERY sample. Three non-collinear samples reject a
    // non-linear clock use that merely passes through two of them (e.g. a
    // `time*time` slot through (0,0),(1,1) is caught at t=2 → 4 ≠ 2).
    const SAMPLES: [f64; 3] = [0.0, 1.0, 2.0];
    let packs: Vec<PackedFill> = SAMPLES
        .iter()
        .map(|&t| pack_fill(raw_shader_id, params, t))
        .collect::<Result<_, _>>()
        .ok()?;
    let base = &packs[0];
    // The clock never drives color_data; if a future pack makes it do so we
    // cannot re-thread (we only patch shader_params at present).
    if packs.iter().any(|p| p.color_data != base.color_data) {
        return None;
    }
    let mut offsets = Vec::new();
    for i in 0..SHADER_PARAM_FLOAT_CAPACITY {
        // Bit-compare so a non-clock slot that is NaN at every sample (e.g. a
        // non-finite mesh-rbf center) counts as UNCHANGED rather than "differing"
        // — `NaN == NaN` is false, which would otherwise mis-flag it as a
        // non-direct-copy clock use and needlessly defeat re-threading.
        if packs
            .iter()
            .all(|p| p.shader_params[i].to_bits() == base.shader_params[i].to_bits())
        {
            continue;
        }
        // A clock-driven slot must be a direct copy of the clock at EVERY sample
        // for `slot = time as f32` re-threading to reproduce a fresh build
        // byte-for-byte. Anything else is a non-linear clock use we won't patch.
        let direct_copy = packs
            .iter()
            .zip(SAMPLES)
            .all(|(p, t)| (p.shader_params[i] as f64 - t).abs() <= 1e-6);
        if !direct_copy {
            return None;
        }
        offsets.push(i as u16);
    }
    Some(offsets)
}

// ---------------------------------------------------------------------------
// M4e — image-backed fill helpers (processing spec + resource key)
// ---------------------------------------------------------------------------

/// `numberParam` (`mesh/imageShaderHelpers.ts:16-18`): finite number or
/// fallback — RAW params only (no descriptor-defaults fallback).
fn raw_number_param(params: &Value, key: &str, fallback: f64) -> f64 {
    params
        .get(key)
        .and_then(as_finite_number)
        .unwrap_or(fallback)
}

/// `ditherImageProcessingParams` (`DitherImageShader.ts:31-46`).
fn dither_image_processing_params(params: &Value) -> image_processing::DitherOptions {
    use image_processing::{DitherOptions, DitherPattern};
    let source_width = raw_number_param(params, "sourceWidth", 1.0);
    let source_height = raw_number_param(params, "sourceHeight", 1.0);
    let scale = raw_number_param(params, "scale", 1.0);
    let scale_x = raw_number_param(params, "scaleX", scale).abs();
    let scale_y = raw_number_param(params, "scaleY", scale).abs();
    DitherOptions {
        pattern: DitherPattern::parse(params.get("pattern").and_then(Value::as_str)),
        brightness: raw_number_param(params, "brightness", 0.0),
        contrast: raw_number_param(params, "contrast", 100.0),
        pixel_size: raw_number_param(params, "pixelSize", 1.0),
        display_width: (source_width * scale_x.max(0.0001)).max(1.0),
        display_height: (source_height * scale_y.max(0.0001)).max(1.0),
    }
}

/// `getResourceRequirements(rawParams).primaryImage.processing`
/// (`defineProceduralShader.ts:186-197`): the processing spec for an
/// image-backed fill, or `None` when the (alias-resolved) shader has no
/// primary image. `raw_params` follows production precedence:
/// `shaderEffect?.params ?? node.fillShaderParams ?? {}`.
pub fn shader_image_processing(
    raw_shader_id: &str,
    raw_params: &Value,
) -> Option<image_processing::Processing> {
    use image_processing::Processing;
    let id = resolve_shader_id(raw_shader_id);
    let meta = packing_tables().get(id)?;
    let primary_image = meta.primary_image.as_ref()?;
    let empty = Value::Object(serde_json::Map::new());
    let params = if raw_params.is_object() {
        raw_params
    } else {
        &empty
    };
    if primary_image.processing.get("dynamic") == Some(&Value::Bool(true)) {
        return match id {
            "dither-image" => Some(Processing::DitherImage(dither_image_processing_params(
                params,
            ))),
            other => panic!("manifest drift: unported dynamic image processing for '{other}'"),
        };
    }

    match primary_image.processing.as_str() {
        Some("poisson-mask") => Some(Processing::PoissonMask),
        Some("heatmap") => Some(Processing::Heatmap),
        Some("source-image") => Some(Processing::SourceImage),
        Some(other) => panic!("manifest drift: unknown image processing '{other}' for '{id}'"),
        None => panic!("manifest drift: invalid image processing metadata for '{id}'"),
    }
}

/// Whether an image shader reads placement from the node's live
/// `fillPattern*` fields instead of effect-local params.
pub fn shader_uses_source_fill_placement(raw_shader_id: &str) -> bool {
    let id = resolve_shader_id(raw_shader_id);
    packing_tables()
        .get(id)
        .and_then(|meta| meta.primary_image.as_ref())
        .and_then(|image| image.placement.as_deref())
        == Some("source-fill")
}

/// `inferImageMimeType` (`WebGPUPathRenderer.ts:138-148`).
pub fn infer_image_mime_type(src: &str) -> Option<String> {
    if let Some(rest) = src.strip_prefix("data:") {
        let end = rest.find([';', ',']).unwrap_or(rest.len());
        let mime = &rest[..end];
        return if mime.is_empty() {
            // `/^data:([^;,]+)/` requires 1+ chars; no match → undefined.
            None
        } else {
            Some(mime.to_string())
        };
    }
    let lower = src.to_lowercase();
    if lower.ends_with(".svg") {
        Some("image/svg+xml".to_string())
    } else if lower.ends_with(".png") {
        Some("image/png".to_string())
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        Some("image/jpeg".to_string())
    } else {
        None
    }
}

/// `createImageResourceKey` (`ShaderImageResourceManager.ts:192-194`):
/// `${processingKey}:${mimeType ?? ''}:${src}`.
pub fn image_resource_key(src: &str, processing: &image_processing::Processing) -> String {
    format!(
        "{}:{}:{}",
        image_processing::processing_key(processing),
        infer_image_mime_type(src).unwrap_or_default(),
        src
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn alias_resolution_matches_production() {
        // core/shaders/shaderIdAliases.ts:1-9
        assert_eq!(resolve_shader_id("mesh-spot"), "mesh-gradient-advanced");
        assert_eq!(resolve_shader_id("mesh-spot-simple"), "mesh-gradient");
        assert_eq!(resolve_shader_id("shape-grid-dots"), "shape-grid");
        assert_eq!(resolve_shader_id("shape-grid-tiles"), "shape-tiles");
        assert_eq!(resolve_shader_id("clouds"), "clouds");
    }

    #[test]
    fn all_49_procedural_fills_have_packing_tables() {
        assert_eq!(packing_tables().len(), 49);
    }

    #[test]
    fn unknown_shaders_error_and_image_backed_pack_with_fallback_meta() {
        let p = json!({});
        assert!(matches!(
            pack_fill("not-a-shader", &p, 0.0),
            Err(FillPackError::UnknownShader(_))
        ));
        // Image-backed shaders are un-gated; packing without an image uses the
        // fallback metadata (ready = 0 → the shader draws its background).
        for id in [
            "liquid-metal-image",
            "heatmap-image",
            "dither-image",
            "gradient-map-image",
        ] {
            let packed = pack_fill(id, &p, 0.0).unwrap();
            assert_eq!(packed.module_id, id);
        }
        // liquid-metal fallback: imageReady at float index 11 must be 0,
        // aspect (10) 1, source dims (13, 14) fall back to prepared 1×1.
        let packed = pack_fill("liquid-metal-image", &p, 0.0).unwrap();
        assert_eq!(packed.shader_params[11], 0.0);
        assert_eq!(packed.shader_params[10], 1.0);
        assert_eq!(packed.shader_params[13], 1.0);
        assert_eq!(packed.shader_params[14], 1.0);
    }

    /// The load-bearing invariant for live shader-playback cache
    /// reuse — every renderer-clock shader writes its clock as a DIRECT copy
    /// into `shader_params` (no `color_data` clock dependence), so the live
    /// camera present path can re-thread a new clock onto a cached plan with a
    /// plain `slot = time as f32`. Proven across the whole shader set: clock
    /// shaders expose >=1 re-threadable offset; the rest expose none. If a
    /// future pack breaks the direct-copy invariant this test fails LOUDLY
    /// (rather than silently freezing or corrupting a playback shader).
    #[test]
    fn clock_param_offsets_cover_every_clock_shader() {
        let p = json!({});
        let mut clock_shaders = 0;
        for (id, meta) in packing_tables() {
            let offsets = clock_param_offsets(id, &p)
                .unwrap_or_else(|| panic!("'{id}' clock is not a direct copy — cannot re-thread"));
            if meta.uses_renderer_clock {
                clock_shaders += 1;
                assert!(
                    !offsets.is_empty(),
                    "'{id}' uses the renderer clock but exposed no re-threadable offset"
                );
                // Re-threading reproduces a fresh build at an arbitrary clock.
                let fresh = pack_fill(id, &p, 7.5).unwrap();
                let mut rethreaded = pack_fill(id, &p, 0.0).unwrap();
                for &off in &offsets {
                    rethreaded.shader_params[off as usize] = 7.5_f32;
                }
                assert_eq!(
                    rethreaded.shader_params, fresh.shader_params,
                    "'{id}' re-thread != fresh build"
                );
            } else {
                assert!(
                    offsets.is_empty(),
                    "'{id}' has no renderer clock but exposed clock offsets {offsets:?}"
                );
            }
        }
        // Sanity: the manifest carries the 39 clock shaders we expect.
        assert_eq!(clock_shaders, 39, "renderer-clock shader count drifted");
    }

    #[test]
    fn clock_param_offsets_unknown_shader_is_none() {
        assert_eq!(clock_param_offsets("not-a-shader", &json!({})), None);
    }

    /// Clock detection must be robust to NON-default params — the
    /// renderer clock overrides any param-supplied `time`, and other params move
    /// DIFFERENT slots (constant across the time samples), so neither changes the
    /// detected clock offset or the re-thread == fresh-build invariant. Covers a
    /// standard-pack shader (clouds) and an image-backed clock shader
    /// (liquid-metal-image; image metadata sits in slots disjoint from the clock).
    #[test]
    fn clock_param_offsets_robust_to_non_default_params() {
        let params = json!({ "time": 5.0, "scale": 2.0, "speed": 3.0, "coverage": 0.9 });
        let offsets = clock_param_offsets("clouds", &params).expect("re-threadable");
        assert_eq!(
            offsets,
            vec![0u16],
            "clouds clock stays at shader_params[0]"
        );
        let fresh = pack_fill("clouds", &params, 9.0).unwrap();
        let mut rethreaded = pack_fill("clouds", &params, 0.0).unwrap();
        for &off in &offsets {
            rethreaded.shader_params[off as usize] = 9.0_f32;
        }
        assert_eq!(
            rethreaded.shader_params, fresh.shader_params,
            "re-thread == fresh build under non-default params"
        );

        let img_params = json!({ "speed": 2.0, "distortionScale": 0.5 });
        let img_offsets =
            clock_param_offsets("liquid-metal-image", &img_params).expect("re-threadable");
        assert!(
            !img_offsets.is_empty(),
            "an image-backed clock shader still exposes its clock slot under non-default params"
        );
    }

    /// A non-clock param that packs to NaN (mesh-rbf coerces a
    /// non-numeric center coord to NaN) must NOT defeat re-threading. The diff
    /// bit-compares, so a slot that is NaN at every sample counts as unchanged
    /// (not a non-direct-copy clock use). mesh-rbf has no clock, so the result is
    /// `Some([])` (re-threadable, nothing to patch) rather than `None`.
    #[test]
    fn clock_param_offsets_tolerates_nan_non_clock_params() {
        let params = json!({ "centers": [{ "x": "not-a-number", "y": 0.5 }] });
        // Confirm the slot really is NaN (guards against the fixture going stale).
        let packed = pack_fill("mesh-rbf", &params, 0.0).unwrap();
        assert!(
            packed.shader_params[4].is_nan(),
            "expected a NaN center slot"
        );
        assert_eq!(clock_param_offsets("mesh-rbf", &params), Some(vec![]));
    }

    #[test]
    fn image_resource_key_matches_production() {
        use image_processing::Processing;
        assert_eq!(
            image_resource_key("https://x/y.svg", &Processing::PoissonMask),
            "poisson-mask:image/svg+xml:https://x/y.svg"
        );
        assert_eq!(
            image_resource_key("data:image/png;base64,AAAA", &Processing::Heatmap),
            "heatmap:image/png:data:image/png;base64,AAAA"
        );
        assert_eq!(
            image_resource_key("https://x/y.webp", &Processing::SourceImage),
            "source-image::https://x/y.webp"
        );
    }

    #[test]
    fn dither_processing_params_from_raw_params() {
        use image_processing::{DitherPattern, Processing};
        let params = json!({
            "pattern": "floyd-steinberg", "brightness": 5, "contrast": 120,
            "pixelSize": 4, "sourceWidth": 512, "sourceHeight": 256,
            "scaleX": 1.0, "scaleY": 2.0
        });
        let Some(Processing::DitherImage(o)) = shader_image_processing("dither-image", &params)
        else {
            panic!("dither-image must yield a DitherImage spec");
        };
        assert_eq!(o.pattern, DitherPattern::FloydSteinberg);
        assert_eq!(o.display_width, 512.0);
        assert_eq!(o.display_height, 512.0);
        assert_eq!(o.pixel_size, 4.0);
        // non-image shader → None
        assert!(shader_image_processing("clouds", &json!({})).is_none());
    }

    #[test]
    fn static_image_processing_comes_from_the_extraction_manifest() {
        use image_processing::Processing;
        assert_eq!(
            shader_image_processing("liquid-metal-image", &json!({})),
            Some(Processing::PoissonMask)
        );
        assert_eq!(
            shader_image_processing("heatmap-image", &json!({})),
            Some(Processing::Heatmap)
        );
        assert_eq!(
            shader_image_processing("gradient-map-image", &json!({})),
            Some(Processing::SourceImage)
        );
    }

    #[test]
    fn gradient_map_packing_matches_the_shared_source_image_abi() {
        let packed = pack_fill_with_image(
            "gradient-map-image",
            &json!({
                "scaleX": 1.2, "scaleY": 0.8, "rotation": 0.25,
                "offsetX": 10, "offsetY": -5,
                "sourceWidth": 640, "sourceHeight": 480,
                "offset": 0.3, "colorCount": 3,
                "colors": ["#000000", "#ff0000", "#ffffff"]
            }),
            0.0,
            Some(ShaderImageMeta::loaded(640, 480, None)),
        )
        .unwrap();

        assert_eq!(
            &packed.shader_params[..12],
            &[
                4.0 / 3.0,
                1.0,
                1.2,
                0.8,
                0.25,
                10.0,
                -5.0,
                640.0,
                480.0,
                0.3,
                3.0,
                1.0
            ]
        );

        let loading = pack_fill_with_image(
            "gradient-map-image",
            &json!({}),
            0.0,
            Some(ShaderImageMeta::fallback()),
        )
        .unwrap();
        let failed = pack_fill_with_image(
            "gradient-map-image",
            &json!({}),
            0.0,
            Some(ShaderImageMeta::failed()),
        )
        .unwrap();
        assert_eq!(loading.shader_params[11], 0.0);
        assert_eq!(failed.shader_params[11], 2.0);
        assert_eq!(
            &packed.color_data[..12],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn clock_param_substitution() {
        // clouds: clock paramKey 'time' at offset 0; params.time is OVERRIDDEN
        // by the renderer clock (export context pins it).
        let p = json!({"time": 99.0, "coverage": 0.5});
        let packed = pack_fill("clouds", &p, 1.5).unwrap();
        assert_eq!(packed.shader_params[0], 1.5);
        assert_eq!(packed.shader_params[6], 0.5);
    }

    #[test]
    fn auto_pack_defaults_and_overrides() {
        // mesh-glass-rods: pure auto-pack with padWithDefaults colors.
        let p = json!({"rods": 7});
        let packed = pack_fill("mesh-glass-rods", &p, 0.0).unwrap();
        assert_eq!(packed.shader_params[9], 7.0); // rods override
        assert_eq!(packed.shader_params[8], 6.0); // frequency default
                                                  // colors fallback ['#FDE08A', ...] padded to 4, alpha forced 1.
        assert_eq!(packed.color_data[3], 1.0);
        assert!((packed.color_data[0] - 0xFD as i32 as f32 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn mesh_gradient_colors_resolve_clamps_count() {
        // resolve: max(1, min(colors.length, 10)) at offset 0
        // (MeshGradientShader.ts params table).
        let p = json!({"colors": ["#111111", "#222222", "#333333"]});
        let packed = pack_fill("mesh-gradient", &p, 0.0).unwrap();
        assert_eq!(packed.shader_params[0], 3.0);
        // default 6 colors when unset
        let packed = pack_fill("mesh-gradient", &json!({}), 0.0).unwrap();
        assert_eq!(packed.shader_params[0], 6.0);
    }
}
