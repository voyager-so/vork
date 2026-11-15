//! Hand-ported per-shader `resolve` callbacks and custom `packInstanceData`
//! bodies (PLAN M3f). Every function cites its TS source; ports are
//! line-faithful including fallback colors, alpha handling differences,
//! offset gaps, and dead alias branches ("copy the bugs", DECISION §1.3).
//!
//! Session-random fallbacks (Q-29): where production falls back to
//! `Math.random()` (seeds when a doc carries no explicit value; warped-grid
//! per-warp centers/angles/phases), this port substitutes the manifest's
//! seeded-PRNG default (for seeds) or a fixed constant (for warp fields) —
//! those browser values are random per session and are NOT part of the
//! render contract; fixtures persist explicit values.

use serde_json::Value;

use super::jscolor::{color_to_normalized_rgba, NormRgba};
use super::jsval::{as_finite_number, js_round, num, pad_color_array};
use super::{Ctx, ParamMeta, COLOR_DATA_FLOAT_CAPACITY, SHADER_PARAM_FLOAT_CAPACITY};

type Sp = [f32; SHADER_PARAM_FLOAT_CAPACITY];
type Cd = [f32; COLOR_DATA_FLOAT_CAPACITY];

// ---------------------------------------------------------------------------
// resolve callbacks (params-table entries with `has_resolve`)
// ---------------------------------------------------------------------------

/// Dispatch for descriptor `resolve` callbacks. Returns `None` when the
/// (shader, key) pair has no ported callback (treated as non-finite by the
/// caller, which then falls back to `getNumberParam` — same as production
/// when `resolve` returns a non-finite value).
pub(crate) fn resolve_param(shader_id: &str, key: &str, ctx: &Ctx) -> Option<f64> {
    match (shader_id, key) {
        // MeshGradientShader.ts: Math.max(1, Math.min(getColorArrayParam('colors', 10).length, 10))
        ("mesh-gradient", "colors") => {
            let len = ctx.get_color_array_param("colors", 10, None).len() as f64;
            Some(len.min(10.0).max(1.0))
        }
        // AnalogGrainGradientShader.ts: Math.max(2, Math.min(Math.round(getNumberParam('colorCount', 6)), 6))
        ("analog-grain-gradient", "colorCount") => {
            let v = js_round(ctx.get_number_param("colorCount", 6.0));
            Some(v.min(6.0).max(2.0))
        }
        // DitheredWaveRibbonShader.ts: Math.min(Math.max(Math.round(getNumberParam('ribbonColorCount', 4)), 2), 4)
        ("dithered-wave-ribbon", "ribbonColorCount") => {
            let v = js_round(ctx.get_number_param("ribbonColorCount", 4.0));
            Some(v.max(2.0).min(4.0))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

/// `writeColorAt` (`mesh/colorPacking.ts:26-33`): alpha ALWAYS 1, opaque
/// white when the string fails to parse.
fn write_color_at(cd: &mut Cd, index: usize, hex: &str) {
    let rgba = color_to_normalized_rgba(hex);
    let offset = index * 4;
    cd[offset] = rgba.map(|c| c.r).unwrap_or(1.0) as f32;
    cd[offset + 1] = rgba.map(|c| c.g).unwrap_or(1.0) as f32;
    cd[offset + 2] = rgba.map(|c| c.b).unwrap_or(1.0) as f32;
    cd[offset + 3] = 1.0;
}

/// Write a parsed color keeping ITS alpha, with explicit per-channel
/// fallbacks (the bespoke packs each choose their own).
fn write_rgba_at(cd: &mut Cd, index: usize, parsed: Option<NormRgba>, fallback: [f64; 4]) {
    let offset = index * 4;
    cd[offset] = parsed.map(|c| c.r).unwrap_or(fallback[0]) as f32;
    cd[offset + 1] = parsed.map(|c| c.g).unwrap_or(fallback[1]) as f32;
    cd[offset + 2] = parsed.map(|c| c.b).unwrap_or(fallback[2]) as f32;
    cd[offset + 3] = parsed.map(|c| c.a).unwrap_or(fallback[3]) as f32;
}

/// `{ ...makeDefaults(), ...params }` view: params key wins when PRESENT
/// (including explicit null — JS spread semantics).
struct Merged<'a> {
    params: &'a Value,
    base: Value,
}

impl Merged<'_> {
    fn get(&self, key: &str) -> Option<&Value> {
        self.params
            .as_object()
            .and_then(|o| o.get(key))
            .or_else(|| self.base.get(key))
    }
    fn num(&self, key: &str, fallback: f64) -> f64 {
        num(self.get(key), fallback)
    }
    fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }
}

/// The pinned stand-in for a `Math.random()` seed fallback: the manifest
/// defaults carry the extraction-time seeded-PRNG value (Q-29).
fn pinned_seed(ctx: &Ctx) -> f64 {
    ctx.defaults
        .get("seed")
        .and_then(as_finite_number)
        .unwrap_or(0.0)
}

fn time_of(ctx: &Ctx, fallback: f64) -> f64 {
    // resolveShaderTimeSeconds(renderContext, fallback)
    if ctx.time_s.is_finite() {
        ctx.time_s
    } else {
        fallback
    }
}

// ---------------------------------------------------------------------------
// custom packInstanceData ports
// ---------------------------------------------------------------------------

/// Returns `Some(())` when `shader_id` has a ported custom pack; `None` lets
/// the caller fail loudly (manifest says custom, port missing = drift).
pub(crate) fn custom_pack(
    shader_id: &str,
    ctx: &Ctx,
    table: &[ParamMeta],
    sp: &mut Sp,
    cd: &mut Cd,
) -> Option<()> {
    match shader_id {
        "band-dither-panels" => band_dither_panels(ctx, table, sp, cd),
        "dithered-wave-ribbon" => dithered_wave_ribbon(ctx, table, sp, cd),
        "metaball-field" => metaball_field(ctx, table, sp, cd),
        "simplex-blob" => simplex_blob(ctx, table, sp, cd),
        "soft-glass-panels" => soft_glass_panels(ctx, table, sp, cd),
        "prismatic-swirl-panels" => prismatic_swirl_panels(ctx, table, sp, cd),
        "segment-display-field" => segment_display_field(ctx, table, sp, cd),
        "mesh-gradient-advanced" => mesh_gradient_advanced(ctx, sp, cd),
        "mesh-grid" => mesh_grid(ctx, sp, cd),
        "mesh-rbf" => mesh_rbf(ctx, sp, cd),
        "perspective-grid" => perspective_grid(ctx, sp, cd),
        "shape-grid" => shape_grid(ctx, sp, cd),
        "shape-tiles" => shape_tiles(ctx, sp, cd),
        "warped-grid" => warped_grid(ctx, sp, cd),
        "light-rays" => light_rays(ctx, sp, cd),
        "light-rays-advanced" => light_rays_advanced(ctx, sp, cd),
        // Image-backed shaders; image metadata rides `Ctx.image`.
        "liquid-metal-image" => liquid_metal_image(ctx, sp, cd),
        "heatmap-image" => heatmap_image(ctx, sp, cd),
        "dither-image" => dither_image(ctx, sp),
        "gradient-map-image" => gradient_map_image(ctx, sp, cd),
        "halftone-image"
        | "cross-hatching-image"
        | "slice-shift-image"
        | "pixel-stretch-image"
        | "grid-displacement-image"
        | "lens-distortion-image" => source_image_effect(shader_id, ctx, table, sp, cd),
        _ => return None,
    }
    Some(())
}

fn p<'a>(ctx: &'a Ctx, key: &str) -> Option<&'a Value> {
    ctx.params.as_object().and_then(|o| o.get(key))
}

/// BandDitherPanelsShader.ts:254-263.
fn band_dither_panels(ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    ctx.pack_number_params("band-dither-panels", table, sp);
    let background = pad_color_array(p(ctx, "backgroundColor"), &["#210388"], 1);
    let panel = pad_color_array(p(ctx, "panelColor"), &["#C24DA7"], 1);
    write_color_at(cd, 0, &background[0]);
    write_color_at(cd, 1, &panel[0]);
}

/// DitheredWaveRibbonShader.ts:356-366 (maxBackgroundColors 2, maxRibbonColors 4).
fn dithered_wave_ribbon(ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    ctx.pack_number_params("dithered-wave-ribbon", table, sp);
    let bg = pad_color_array(p(ctx, "backgroundColors"), &["#02040A", "#141923"], 2);
    let ribbon = pad_color_array(
        p(ctx, "ribbonColors"),
        &["#F7F2E9", "#69D7C8", "#7B8CFF", "#C97834"],
        4,
    );
    for (i, hex) in bg.iter().enumerate() {
        write_color_at(cd, i, hex);
    }
    for (i, hex) in ribbon.iter().enumerate() {
        write_color_at(cd, 2 + i, hex);
    }
}

/// MetaballFieldShader.ts:418-431 (maxBackgroundColors 3, maxMetaballColors 2).
fn metaball_field(ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    ctx.pack_number_params("metaball-field", table, sp);
    let bg = pad_color_array(
        p(ctx, "backgroundColors"),
        &["#121416", "#314763", "#273038"],
        3,
    );
    let metaball = pad_color_array(p(ctx, "metaballColors"), &["#1239af", "#6d5f4f"], 2);
    for (i, hex) in bg.iter().enumerate() {
        write_color_at(cd, i, hex);
    }
    for (i, hex) in metaball.iter().enumerate() {
        write_color_at(cd, 3 + i, hex);
    }
}

/// SimplexBlobShader.ts:373-385 (max 2 + 2).
fn simplex_blob(ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    ctx.pack_number_params("simplex-blob", table, sp);
    let bg = pad_color_array(p(ctx, "backgroundColors"), &["#181818", "#7F24F0"], 2);
    let blob = pad_color_array(p(ctx, "blobColors"), &["#181818", "#7F24F0"], 2);
    for (i, hex) in bg.iter().enumerate() {
        write_color_at(cd, i, hex);
    }
    for (i, hex) in blob.iter().enumerate() {
        write_color_at(cd, 2 + i, hex);
    }
}

/// SoftGlassPanelsShader.ts:397-407.
fn soft_glass_panels(ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    ctx.pack_number_params("soft-glass-panels", table, sp);
    let background = pad_color_array(p(ctx, "backgroundColor"), &["#F5F3EE"], 1);
    let blob = pad_color_array(p(ctx, "blobColor"), &["#000000"], 1);
    write_color_at(cd, 0, &background[0]);
    write_color_at(cd, 1, &blob[0]);
}

/// PrismaticSwirlPanelsShader.ts:490-499 (maxSwirlColors 3, maxBlobColors 3).
fn prismatic_swirl_panels(ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    ctx.pack_number_params("prismatic-swirl-panels", table, sp);
    let swirl = pad_color_array(p(ctx, "swirlColors"), &["#15EAD0", "#2868F0", "#E845A9"], 3);
    let blob = pad_color_array(p(ctx, "blobColors"), &["#E845A9", "#FF8F2E", "#FFE76A"], 3);
    for (i, hex) in swirl.iter().enumerate() {
        write_color_at(cd, i, hex);
    }
    for (i, hex) in blob.iter().enumerate() {
        write_color_at(cd, 3 + i, hex);
    }
}

/// SegmentDisplayFieldShader.ts:462-479. Public colors are
/// [background, glyph] but the WGSL reads [glyph, background] — pickColors
/// swaps them (copy the swap).
fn segment_display_field(ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    ctx.pack_number_params("segment-display-field", table, sp);
    let incoming: Vec<&str> = p(ctx, "colors")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let background = incoming.first().copied().unwrap_or("#000000");
    let glyph = incoming.get(1).copied().unwrap_or("#6DFFF0");
    write_color_at(cd, 0, glyph);
    write_color_at(cd, 1, background);
}

/// MeshGradientAdvancedShader.ts:275-321. No defaults merge — inline `num`
/// fallbacks; DEFAULT_COLORS when `params.colors` is not an array.
fn mesh_gradient_advanced(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const DEFAULT_COLORS: [&str; 6] = [
        "#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF", "#06FFB4",
    ];
    let colors: Vec<String> = match p(ctx, "colors").and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .take(10)
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect(),
        None => DEFAULT_COLORS.iter().map(|s| s.to_string()).collect(),
    };
    let count = (colors.len() as f64).min(10.0).max(1.0);
    let gp = |k: &str, fb: f64| num(p(ctx, k), fb);
    sp[0] = count as f32;
    sp[1] = gp("distortion", 0.08) as f32;
    sp[2] = gp("swirl", 2.0) as f32;
    sp[3] = time_of(ctx, gp("time", 0.0)) as f32;
    sp[4] = gp("positions", 25.0) as f32;
    sp[5] = gp("waveX", 0.25) as f32;
    sp[6] = gp("waveXShift", 0.15) as f32;
    sp[7] = gp("waveY", 0.3) as f32;
    sp[8] = gp("waveYShift", 0.4) as f32;
    sp[9] = gp("mixing", 0.5) as f32;
    sp[10] = gp("grainMixer", 0.3) as f32;
    sp[11] = gp("grainOverlay", 0.15) as f32;
    for i in 0..(count as usize) {
        // `colors[i] || '#ffffff'` — missing/empty → white; alpha forced 1.
        let hex = colors
            .get(i)
            .filter(|s| !s.is_empty())
            .map(String::as_str)
            .unwrap_or("#ffffff");
        write_color_at(cd, i, hex);
    }
}

/// MeshGridShader.ts:175-203. gridColors used only when EXACTLY 9 entries.
fn mesh_grid(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const DEFAULT_GRID_COLORS: [&str; 9] = [
        "#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF", "#06FFB4", "#FF4365", "#00F5FF",
        "#8000FF",
    ];
    let grid_colors: Vec<String> = match p(ctx, "gridColors").and_then(Value::as_array) {
        Some(arr) if arr.len() == 9 => arr
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect(),
        _ => DEFAULT_GRID_COLORS.iter().map(|s| s.to_string()).collect(),
    };
    sp[0] = 9.0;
    sp[1] = num(p(ctx, "distortionAmount"), 0.08) as f32;
    sp[2] = num(p(ctx, "swirlStrength"), 2.0) as f32;
    sp[3] = 1.0;
    for i in 0..9usize {
        let row = i / 3;
        let col = i % 3;
        sp[4 + i * 2] = ((col as f64 + 0.5) / 3.0) as f32;
        sp[4 + i * 2 + 1] = ((row as f64 + 0.5) / 3.0) as f32;
        // colorToNormalizedRgba(gridColors[i]) — white on parse failure,
        // alpha forced 1 (MeshGridShader.ts:196-200).
        write_color_at(cd, i, &grid_colors[i]);
    }
}

/// MeshRBFShader.ts:101-130. Centers are `{x, y, color}` records packed as
/// (x,y) pairs from float offset 4; `c?.x ?? 0.5` keeps JS ToNumber coercion
/// at the Float32Array store.
fn mesh_rbf(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    let empty: Vec<Value> = Vec::new();
    let centers = p(ctx, "centers")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let count = centers.len().min(16);
    sp[0] = count as f32;
    sp[1] = num(p(ctx, "falloff"), 0.35) as f32;
    sp[2] = num(p(ctx, "gamma"), 1.0) as f32;
    sp[3] = 0.0;
    for (i, c) in centers.iter().take(count).enumerate() {
        let coord = |key: &str| -> f64 {
            match c.get(key) {
                // `c?.x ?? 0.5` then ToNumber at the typed-array store.
                Some(v) => as_finite_number(v).unwrap_or(f64::NAN),
                None => 0.5,
            }
        };
        sp[4 + i * 2] = coord("x") as f32;
        sp[4 + i * 2 + 1] = coord("y") as f32;
        // `c?.color || '#ffffff'` — missing/empty → white.
        let hex = c
            .get("color")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("#ffffff");
        write_color_at(cd, i, hex);
    }
}

/// PerspectiveGridShader.ts:135-167. Colors written ONLY when they parse —
/// otherwise the colorData slot keeps its zero initialization (copy the bug).
fn perspective_grid(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    // `params.X ?? defaults.X` (nullish — explicit null falls through).
    let pick = |key: &str| -> Option<Value> {
        match p(ctx, key) {
            Some(Value::Null) | None => ctx.defaults.get(key).cloned(),
            Some(v) => Some(v.clone()),
        }
    };
    let background_color = pick("backgroundColor");
    let grid_color = pick("gridColor");
    let dnum = |key: &str| -> f64 {
        match p(ctx, key).and_then(as_finite_number) {
            Some(v) => v,
            None => ctx
                .defaults
                .get(key)
                .and_then(as_finite_number)
                .unwrap_or(0.0),
        }
    };
    sp[0] = dnum("horizonY") as f32;
    sp[1] = dnum("gridSpacing") as f32;
    sp[2] = dnum("lineWidth") as f32;
    let parse = |v: &Option<Value>| -> Option<NormRgba> {
        v.as_ref()
            .and_then(Value::as_str)
            .and_then(color_to_normalized_rgba)
    };
    if let Some(c) = parse(&background_color) {
        cd[0] = c.r as f32;
        cd[1] = c.g as f32;
        cd[2] = c.b as f32;
        cd[3] = 1.0;
    }
    if let Some(c) = parse(&grid_color) {
        cd[4] = c.r as f32;
        cd[5] = c.g as f32;
        cd[6] = c.b as f32;
        cd[7] = 1.0;
    }
}

/// ShapeGridShader.ts:490-621 (+ makeDefaults `:13-40` + coerceParams `:49-61`).
fn shape_grid(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    let d = Merged {
        params: ctx.params,
        base: serde_json::json!({
            "lattice": "square",
            "shape": "circle",
            "cellSize": 0.06,
            "jitter": 0.0,
            "strokeWidth": 0.0,
            "radiusBase": 0.35,
            "radiusGradientAmount": 0.0,
            "radiusGradientAngle": 0.0,
            "sizeGradientMode": "uniform",
            "sizeGradient": { "start": { "x": 0.0, "y": 0.5 }, "end": { "x": 1.0, "y": 0.5 } },
            "sizeGradientStartValue": 1.0,
            "sizeGradientEndValue": 0.0,
            "rippleAmplitude": 0.0,
            "rippleWavelength": 0.4,
            "rippleSpeed": 0.8,
            "rippleAngle": 0.785398163,
            "colorMode": "uniform",
            "colors": ["#222222", "#888888"],
            "gradientAngle": 0.0,
            "gradientRepeat": 1.0,
            "background": "#00000000",
            "time": 0,
            "seed": pinned_seed(ctx),
        }),
    };

    // coerceParams: enum coercion in place.
    let coerce = |key: &str, valid: &[&str], fallback: &'static str| -> String {
        match d.str(key) {
            Some(s) if valid.contains(&s) => s.to_string(),
            _ => fallback.to_string(),
        }
    };
    let lattice_s = coerce("lattice", &["square", "hex", "tri"], "square");
    let shape_s = coerce("shape", &["circle", "square"], "circle");
    let color_mode_s = coerce("colorMode", &["uniform", "gradient", "hashed"], "uniform");
    let size_gradient_mode_s = coerce(
        "sizeGradientMode",
        &["uniform", "linear", "radial"],
        "uniform",
    );

    let lattice = match lattice_s.as_str() {
        "hex" => 1.0,
        "tri" => 2.0,
        _ => 0.0,
    };
    let shape = if shape_s == "square" { 1.0 } else { 0.0 };
    let color_mode = match color_mode_s.as_str() {
        "gradient" => 1.0,
        "hashed" => 2.0,
        _ => 0.0,
    };
    let size_gradient_mode = match size_gradient_mode_s.as_str() {
        "linear" => 1.0,
        "radial" => 2.0,
        _ => 0.0,
    };

    let cell_size = d.num("cellSize", 0.06);
    let jitter = d.num("jitter", 0.0);
    let stroke_width = d.num("strokeWidth", 0.0);
    let radius_base = d.num("radiusBase", 0.35);
    let radius_gradient_amount = d.num("radiusGradientAmount", 0.0);
    let radius_gradient_angle = d.num("radiusGradientAngle", 0.0);
    let ripple_amplitude = d.num("rippleAmplitude", 0.0);
    let ripple_wavelength = d.num("rippleWavelength", 0.4);
    let ripple_speed = d.num("rippleSpeed", 0.8);
    let ripple_angle = d.num("rippleAngle", 0.785398163);
    let gradient_angle = d.num("gradientAngle", 0.0);
    let gradient_repeat = d.num("gradientRepeat", 1.0);
    let time = time_of(ctx, d.num("time", 0.0));
    let seed = d.num("seed", pinned_seed(ctx));

    let size_gradient = d.get("sizeGradient");
    let nested = |outer: Option<&Value>, key: &str| -> Option<Value> {
        outer.and_then(|v| v.get(key)).cloned()
    };
    let sg_start = nested(size_gradient, "start");
    let sg_end = nested(size_gradient, "end");
    let sg_start_x = num(sg_start.as_ref().and_then(|v| v.get("x")), 0.0);
    let sg_start_y = num(sg_start.as_ref().and_then(|v| v.get("y")), 0.5);
    let sg_end_x = num(sg_end.as_ref().and_then(|v| v.get("x")), 1.0);
    let sg_end_y = num(sg_end.as_ref().and_then(|v| v.get("y")), 0.5);
    let sg_start_val = num(
        size_gradient.and_then(|v| v.get("startValue")),
        d.num("sizeGradientStartValue", 1.0),
    );
    let sg_end_val = num(
        size_gradient.and_then(|v| v.get("endValue")),
        d.num("sizeGradientEndValue", 0.0),
    );

    // Size stops (2..4), sorted stable by pos.
    let raw_stop_count = match d.get("sizeStopCount").and_then(as_finite_number) {
        Some(v) => v.floor(),
        None => 2.0,
    };
    let stop_count = raw_stop_count.min(4.0).max(2.0) as usize;
    let read_stop = |i: usize| -> (f64, f64) {
        let stop = d.get(&format!("sizeStop{i}"));
        let pos_fb = match i {
            0 => 0.0,
            1 => 1.0,
            2 => 0.5,
            _ => 0.75,
        };
        let val_fb = if i == 0 { sg_start_val } else { sg_end_val };
        let pos = num(stop.and_then(|v| v.get("pos")), pos_fb);
        let value = num(stop.and_then(|v| v.get("value")), val_fb);
        (pos.min(1.0).max(0.0), value.min(1.0).max(0.0))
    };
    let mut stops: Vec<(f64, f64)> = (0..stop_count).map(read_stop).collect();
    stops.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Colors (coerceParams filtered string entries, sliced to 8).
    const DEFAULT_COLORS: [&str; 2] = ["#222222", "#888888"];
    let raw_colors: Vec<String> = match d.get("colors").and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .take(8)
            .collect(),
        None => DEFAULT_COLORS.iter().map(|s| s.to_string()).collect(),
    };
    let color_count = raw_colors.len().min(8).max(1);

    sp[0] = 0.0; // dots-only mode
    sp[1] = lattice as f32;
    sp[2] = shape as f32;
    sp[3] = time as f32;
    sp[4] = cell_size as f32;
    sp[6] = jitter as f32;
    sp[7] = stroke_width as f32;
    sp[8] = radius_base as f32;
    sp[9] = radius_gradient_amount as f32;
    sp[10] = radius_gradient_angle.cos() as f32;
    sp[11] = radius_gradient_angle.sin() as f32;
    sp[12] = ripple_wavelength as f32;
    sp[13] = ripple_speed as f32;
    sp[14] = ripple_angle.cos() as f32;
    sp[15] = ripple_angle.sin() as f32;
    sp[16] = color_mode as f32;
    sp[17] = gradient_angle.cos() as f32;
    sp[18] = gradient_repeat as f32;
    sp[19] = color_count as f32;
    sp[20] = size_gradient_mode as f32;
    sp[24] = sg_start_x as f32;
    sp[25] = sg_start_y as f32;
    sp[26] = sg_end_x as f32;
    sp[27] = sg_end_y as f32;
    sp[28] = ripple_amplitude as f32;
    sp[29] = seed as f32;
    sp[30] = gradient_angle.sin() as f32;
    sp[32] = stop_count as f32;
    let s0 = stops.first().copied().unwrap_or((0.0, sg_start_val));
    let s1 = stops.get(1).copied().unwrap_or((1.0, sg_end_val));
    let s2 = stops.get(2).copied().unwrap_or((0.5, sg_end_val));
    let s3 = stops.get(3).copied().unwrap_or((0.75, sg_end_val));
    sp[33] = s0.0 as f32;
    sp[34] = s0.1 as f32;
    sp[35] = s1.0 as f32;
    sp[36] = s1.1 as f32;
    sp[37] = s2.0 as f32;
    sp[38] = s2.1 as f32;
    sp[39] = s3.0 as f32;
    sp[40] = s3.1 as f32;

    // Palette: `rawColors[i] || '#000000'`, parsed alpha KEPT, black on fail.
    for i in 0..color_count {
        let hex = raw_colors
            .get(i)
            .filter(|s| !s.is_empty())
            .map(String::as_str)
            .unwrap_or("#000000");
        write_rgba_at(cd, i, color_to_normalized_rgba(hex), [0.0, 0.0, 0.0, 1.0]);
    }
    // Background at slot 14: string || '#ffffff', parsed alpha kept.
    let bg_hex = d
        .str("background")
        .filter(|s| !s.is_empty())
        .unwrap_or("#ffffff");
    write_rgba_at(
        cd,
        14,
        color_to_normalized_rgba(bg_hex),
        [1.0, 1.0, 1.0, 1.0],
    );
}

/// ShapeTilesShader.ts:292-345 (+ sanitizeParams `:57-91`).
fn shape_tiles(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const DEFAULT_COLORS: [&str; 2] = ["#222222", "#888888"];
    // sanitizeParams: per-field typeof checks over makeDefaults.
    let pn = |key: &str, fb: f64| num(p(ctx, key), fb);
    let cell_size = pn("cellSize", 0.06);
    let stroke_width = pn("strokeWidth", 0.0);
    let ripple_amplitude = pn("rippleAmplitude", 0.0);
    let ripple_wavelength = pn("rippleWavelength", 0.4);
    let ripple_speed = pn("rippleSpeed", 0.8);
    let ripple_angle = pn("rippleAngle", std::f64::consts::PI / 4.0);
    let gradient_angle = pn("gradientAngle", 0.0);
    let gradient_repeat = pn("gradientRepeat", 1.0);
    let time_param = pn("time", 0.0);
    let seed = pn("seed", pinned_seed(ctx));
    // colorMode: rawCm = p.colorMode ?? 'gradient'; valid else 'gradient'.
    let color_mode_s = match p(ctx, "colorMode").and_then(Value::as_str) {
        Some(s) if s == "uniform" || s == "gradient" || s == "hashed" => s,
        _ => "gradient",
    };
    // colors: strict isStringArray(p.colors) → slice 8, else defaults.
    let colors: Vec<String> = match p(ctx, "colors").and_then(super::jsval::as_string_array) {
        Some(v) => v.into_iter().take(8).collect(),
        None => DEFAULT_COLORS.iter().map(|s| s.to_string()).collect(),
    };
    let background = p(ctx, "background")
        .and_then(Value::as_str)
        .unwrap_or("#00000000");

    let color_mode = match color_mode_s {
        "gradient" => 1.0,
        "hashed" => 2.0,
        _ => 0.0,
    };
    let time = time_of(ctx, time_param);
    let color_count = colors.len().min(8).max(1);

    sp[0] = 1.0; // tiles mode marker
    sp[1] = 2.0; // lattice tri
    sp[2] = 3.0; // shape tri
    sp[3] = time as f32;
    sp[4] = cell_size as f32;
    sp[7] = stroke_width as f32;
    sp[11] = ripple_amplitude as f32;
    sp[12] = ripple_wavelength as f32;
    sp[13] = ripple_speed as f32;
    sp[14] = ripple_angle as f32;
    sp[15] = seed as f32;
    sp[16] = color_mode as f32;
    sp[17] = gradient_angle as f32;
    sp[18] = gradient_repeat as f32;
    sp[19] = color_count as f32;

    for i in 0..color_count {
        let hex = colors
            .get(i)
            .filter(|s| !s.is_empty())
            .map(String::as_str)
            .unwrap_or("#000000");
        write_rgba_at(cd, i, color_to_normalized_rgba(hex), [0.0, 0.0, 0.0, 1.0]);
    }
    let bg_hex = if background.is_empty() {
        "#ffffff"
    } else {
        background
    };
    write_rgba_at(
        cd,
        14,
        color_to_normalized_rgba(bg_hex),
        [1.0, 1.0, 1.0, 1.0],
    );
}

/// WarpedGridShader.ts:194-280 (+ makeDefaults `:10-22`).
///
/// Production per-warp fallbacks for missing `center`/`angle`/`phase` are
/// `Math.random()` — session-random and unreproducible even browser-side
/// (Q-29). This port pins them (0.5 / 0.5 / 0.0 / 0.0); documents are
/// expected to persist full warp records (the fixture corpus does).
fn warped_grid(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    let d = Merged {
        params: ctx.params,
        base: serde_json::json!({
            "spacing": { "x": 0.08, "y": 0.08 },
            "thickness": 0.004,
            "aaWidth": 0.002,
            "time": 0,
            "noise": { "amplitude": 0.0, "frequency": 2.0 },
            "warps": [],
            "lineColor": "#ffffff",
            "background": "#000000",
            "seed": pinned_seed(ctx),
        }),
    };
    let spacing = d.get("spacing");
    let noise = d.get("noise");
    let sx = num(spacing.and_then(|v| v.get("x")), 0.08);
    let sy = num(spacing.and_then(|v| v.get("y")), 0.08);
    let thickness = d.num("thickness", 0.004);
    let aa_width = d.num("aaWidth", 0.002);
    let time = time_of(ctx, d.num("time", 0.0));
    let noise_amp = num(noise.and_then(|v| v.get("amplitude")), 0.0);
    let noise_freq = num(noise.and_then(|v| v.get("frequency")), 2.0);
    let seed = d.num("seed", pinned_seed(ctx));

    let empty: Vec<Value> = Vec::new();
    let warps: Vec<&Value> = d
        .get("warps")
        .and_then(Value::as_array)
        .unwrap_or(&empty)
        .iter()
        .take(3)
        .collect();

    sp[0] = sx as f32;
    sp[1] = sy as f32;
    sp[2] = thickness as f32;
    sp[3] = time as f32;
    sp[4] = aa_width as f32;
    sp[5] = noise_amp as f32;
    sp[6] = noise_freq as f32;
    sp[7] = warps.len() as f32;
    sp[8] = seed as f32;
    sp[9] = 0.0;
    sp[10] = 0.0;
    sp[11] = 0.0;

    let mut warp_base = 32usize;
    for w in &warps {
        let center = w.get("center");
        // Pinned stand-ins for Math.random() fallbacks (see fn doc).
        let cx = num(center.and_then(|v| v.get("x")), 0.5);
        let cy = num(center.and_then(|v| v.get("y")), 0.5);
        let radius = num(w.get("radius"), 0.25);
        let amp = num(w.get("amplitude"), 0.02);
        let speed = num(w.get("speed"), 2.0);
        let angle = num(w.get("angle"), 0.0);
        let falloff = num(w.get("falloff"), 2.0);
        let phase = num(w.get("phase"), 0.0);
        sp[warp_base] = cx as f32;
        sp[warp_base + 1] = cy as f32;
        sp[warp_base + 2] = radius as f32;
        sp[warp_base + 3] = amp as f32;
        sp[warp_base + 4] = speed as f32;
        sp[warp_base + 5] = angle as f32;
        sp[warp_base + 6] = falloff as f32;
        sp[warp_base + 7] = phase as f32;
        warp_base += 8;
    }

    // lineColor at slot 0 (alpha kept, white fallback); background at 14
    // (alpha kept, black fallback).
    let line_hex = d
        .str("lineColor")
        .filter(|s| !s.is_empty())
        .unwrap_or("#ffffff");
    write_rgba_at(
        cd,
        0,
        color_to_normalized_rgba(line_hex),
        [1.0, 1.0, 1.0, 1.0],
    );
    let bg_hex = d
        .str("background")
        .filter(|s| !s.is_empty())
        .unwrap_or("#000000");
    write_rgba_at(
        cd,
        14,
        color_to_normalized_rgba(bg_hex),
        [0.0, 0.0, 0.0, 1.0],
    );
}

/// Shared light-rays alias normalization over the merged view
/// (`light/lightRaysAliases.ts:9-27`). NOTE: pack-time normalization runs on
/// `{...makeDefaults(), ...params}`, whose canonical keys are ALWAYS defined
/// — the alias remaps are pack-time inert for whole keys and only the
/// partial-`centerGlow` branch can fire (copy the bug; quirk Q-43).
struct LightRaysParams {
    time: f64,
    ray_frequency: f64,
    ray_sharpness: f64,
    speckle: f64,
    center_size: f64,
    center_strength: f64,
    additive_mix: f64,
    origin_x: f64,
    origin_y: f64,
    seed: f64,
}

fn light_rays_common(ctx: &Ctx, d: &Merged) -> LightRaysParams {
    // normalizeLightRaysAliases(d): each alias applies only when the
    // canonical key is undefined on the merged record.
    let canon = |canonical: &str, alias: &str| -> Option<Value> {
        match d.get(canonical) {
            Some(v) => Some(v.clone()),
            None => d.get(alias).cloned(),
        }
    };
    let ray_frequency = num(canon("rayFrequency", "density").as_ref(), 0.9);
    let ray_sharpness = num(canon("raySharpness", "intensity").as_ref(), 0.7);
    let speckle = num(canon("speckle", "spotty").as_ref(), 0.6);
    let additive_mix = num(canon("additiveMix", "bloom").as_ref(), 0.35);

    // centerGlow: glow = {...centerGlow}, then midSize/midIntensity fill
    // UNDEFINED size/strength only.
    let center_glow = d.get("centerGlow");
    let glow_obj = center_glow.filter(|v| v.is_object());
    let mid_active = d.get("midSize").is_some() || d.get("midIntensity").is_some();
    let glow_size = match glow_obj.and_then(|v| v.get("size")) {
        Some(v) => Some(v.clone()),
        None if mid_active => d.get("midSize").cloned(),
        None => None,
    };
    let glow_strength = match glow_obj.and_then(|v| v.get("strength")) {
        Some(v) => Some(v.clone()),
        None if mid_active => d.get("midIntensity").cloned(),
        None => None,
    };
    let center_size = num(glow_size.as_ref(), 0.35);
    let center_strength = num(glow_strength.as_ref(), 0.8);

    // origin: built from originX/originY only when `origin` is undefined.
    let origin = d.get("origin");
    let (origin_x, origin_y) = match origin {
        Some(o) => (num(o.get("x"), 0.5), num(o.get("y"), 0.5)),
        None if d.get("originX").is_some() || d.get("originY").is_some() => {
            (num(d.get("originX"), 0.5), num(d.get("originY"), 0.5))
        }
        None => (0.5, 0.5),
    };

    LightRaysParams {
        time: time_of(ctx, d.num("time", 0.0)),
        ray_frequency,
        ray_sharpness,
        speckle,
        center_size,
        center_strength,
        additive_mix,
        origin_x,
        origin_y,
        seed: d.num("seed", pinned_seed(ctx)),
    }
}

/// LightRaysShader.ts:228-287 (+ makeDefaults `:13-26`).
fn light_rays(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const DEFAULT_COLORS: [&str; 3] = ["#FFBE0B", "#FB5607", "#FF006E"];
    let d = Merged {
        params: ctx.params,
        base: serde_json::json!({
            "colors": DEFAULT_COLORS,
            "time": 0,
            "origin": { "x": 0.5, "y": 0.5 },
            "rayFrequency": 0.9,
            "raySharpness": 0.7,
            "speckle": 0.6,
            "centerGlow": { "size": 0.35, "strength": 0.8 },
            "additiveMix": 0.35,
            "seed": pinned_seed(ctx),
        }),
    };
    let v = light_rays_common(ctx, &d);

    // Always exactly 3 colors, cycling provided ones; `|| '#ffffff'` on
    // falsy entries (and on the empty-array modulo case).
    let raw_colors: Vec<String> = match d.get("colors").and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .map(|c| c.as_str().unwrap_or("").to_string())
            .collect(),
        None => DEFAULT_COLORS.iter().map(|s| s.to_string()).collect(),
    };
    let mut colors: Vec<String> = Vec::with_capacity(3);
    for i in 0..3usize {
        let c = if raw_colors.is_empty() {
            String::new()
        } else {
            raw_colors[i % raw_colors.len()].clone()
        };
        colors.push(if c.is_empty() {
            "#ffffff".to_string()
        } else {
            c
        });
    }

    sp[0] = v.time as f32;
    sp[1] = v.ray_frequency as f32;
    sp[2] = v.ray_sharpness as f32;
    sp[3] = v.speckle as f32;
    sp[4] = v.center_size as f32;
    sp[5] = v.center_strength as f32;
    sp[6] = v.additive_mix as f32;
    sp[7] = v.seed as f32;
    sp[8] = v.origin_x as f32;
    sp[9] = v.origin_y as f32;

    // Alpha from the PARSED color is kept here (unlike writeColorAt).
    for (i, hex) in colors.iter().enumerate() {
        write_rgba_at(cd, i, color_to_normalized_rgba(hex), [1.0, 1.0, 1.0, 1.0]);
    }
}

/// LightRaysAdvancedShader.ts:172-238 (+ makeDefaults `:13-26`).
fn light_rays_advanced(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const DEFAULT_COLORS: [&str; 3] = ["#FFBE0B", "#FB5607", "#FF006E"];
    let d = Merged {
        params: ctx.params,
        base: serde_json::json!({
            "colors": DEFAULT_COLORS,
            "background": "#000000",
            "time": 0,
            "origin": { "x": 0.5, "y": 0.5 },
            "rayFrequency": 0.9,
            "raySharpness": 0.7,
            "speckle": 0.6,
            "centerGlow": { "size": 0.35, "strength": 0.8 },
            "additiveMix": 0.35,
            "seed": pinned_seed(ctx),
        }),
    };
    let v = light_rays_common(ctx, &d);

    let raw_colors: Vec<String> = match d.get("colors").and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .take(5)
            .map(|c| c.as_str().unwrap_or("").to_string())
            .collect(),
        None => DEFAULT_COLORS.iter().map(|s| s.to_string()).collect(),
    };
    let count = raw_colors.len().min(5).max(1);

    sp[0] = count as f32;
    sp[1] = v.time as f32;
    sp[2] = v.ray_frequency as f32;
    sp[3] = v.ray_sharpness as f32;
    sp[4] = v.speckle as f32;
    sp[5] = v.center_size as f32;
    sp[6] = v.center_strength as f32;
    sp[7] = v.additive_mix as f32;
    sp[8] = v.origin_x as f32;
    sp[9] = v.origin_y as f32;
    sp[10] = v.seed as f32;
    sp[11] = 0.0;

    for i in 0..count {
        let hex = raw_colors
            .get(i)
            .filter(|s| !s.is_empty())
            .map(String::as_str)
            .unwrap_or("#ffffff");
        write_rgba_at(cd, i, color_to_normalized_rgba(hex), [1.0, 1.0, 1.0, 1.0]);
    }

    // Background at slot 14, alpha kept, BLACK rgb fallback (`:227-232`).
    let bg_hex = d
        .str("background")
        .filter(|s| !s.is_empty())
        .unwrap_or("#000000");
    write_rgba_at(
        cd,
        14,
        color_to_normalized_rgba(bg_hex),
        [0.0, 0.0, 0.0, 1.0],
    );
}

// ---------------------------------------------------------------------------
// M4e — image-backed shaders (mesh/imageShaderHelpers.ts + shader packers)
// ---------------------------------------------------------------------------

/// `primaryImageMetrics(resources)` (`base/ShaderResources.ts:54-64`):
/// (imageAspect, imageReady) — fallback entries report (1, 0).
fn primary_image_metrics(ctx: &Ctx) -> (f64, f64) {
    match &ctx.image {
        Some(img) if img.ready => (img.aspect_ratio, 1.0),
        _ => (1.0, 0.0),
    }
}

/// `resources?.primaryImage?.width ?? 1` (the prepared-image dims fallback
/// for `sourceWidth`/`sourceHeight`). NOTE: production uses the metadata
/// dims even when `ready` is false (the 1×1 fallback entry) — copy that.
fn primary_image_dims(ctx: &Ctx) -> (f64, f64) {
    match &ctx.image {
        Some(img) => (img.width, img.height),
        None => (1.0, 1.0),
    }
}

/// `numberParam(value, fallback)` (`mesh/imageShaderHelpers.ts:16-18`) over a
/// raw param key — finite JSON number or fallback, NO defaults lookup.
fn img_num(ctx: &Ctx, key: &str, fallback: f64) -> f64 {
    num(p(ctx, key), fallback)
}

/// `colorArrayParam(value, fallback)` (`imageShaderHelpers.ts:26-29`):
/// exactly ONE color — the user array's head when non-empty, else fallback.
/// Non-string / empty entries behave as opaque white in `packColors`
/// (`colors[i] || '#ffffff'`, then parse-fail → white) — both collapse to
/// "#ffffff" here.
fn color_array_param(value: Option<&Value>, fallback: &[&str]) -> Vec<String> {
    let user: Vec<String> = match value.and_then(Value::as_array) {
        Some(arr) if !arr.is_empty() => arr.iter().map(color_entry).collect(),
        _ => fallback.iter().map(|s| s.to_string()).collect(),
    };
    let mut all = user;
    all.extend(fallback.iter().map(|s| s.to_string()));
    all.truncate(1);
    all
}

/// `padColors(colors, defaults, max)` (`imageShaderHelpers.ts:36-38`).
fn pad_colors(colors: &[String], defaults: &[&str], max: usize) -> Vec<String> {
    let mut all: Vec<String> = colors.to_vec();
    all.extend(defaults.iter().map(|s| s.to_string()));
    all.truncate(max);
    all
}

/// One entry of a colors array: strings pass through; falsy / non-string
/// values land on opaque white exactly like `colors[i] || '#ffffff'`
/// followed by a parse-fail-→-white in `packColors`.
fn color_entry(v: &Value) -> String {
    match v.as_str() {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => "#ffffff".to_string(),
    }
}

/// `packColors(colors, max)` (`imageShaderHelpers.ts:45-56`): one RGBA vec4
/// per slot, alpha ALWAYS 1, parse failure → opaque white.
fn pack_colors(cd: &mut Cd, colors: &[String], max: usize) {
    for (i, hex) in colors.iter().take(max).enumerate() {
        write_color_at(cd, i, if hex.is_empty() { "#ffffff" } else { hex });
    }
}

/// `params.colors` for the image shaders: `Array.isArray(params.colors) ?
/// params.colors.slice(0, maxColors) : defaultColors`.
fn image_shader_colors(ctx: &Ctx, defaults: &[&str], max: usize) -> Vec<String> {
    match p(ctx, "colors").and_then(Value::as_array) {
        Some(arr) => arr.iter().take(max).map(color_entry).collect(),
        None => defaults.iter().map(|s| s.to_string()).collect(),
    }
}

/// LiquidMetalImageShader.ts:312-368.
fn liquid_metal_image(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const MAX_COLORS: usize = 4;
    const DEFAULT_BG: [&str; 1] = ["#000000"];
    const DEFAULT_COLORS: [&str; 4] = ["#000000", "#0051FF", "#00A6FF", "#BDE4FF"];
    let (image_aspect, image_ready) = primary_image_metrics(ctx);
    let (img_w, img_h) = primary_image_dims(ctx);

    let colors = image_shader_colors(ctx, &DEFAULT_COLORS, MAX_COLORS);
    let color_count = js_round(img_num(ctx, "colorCount", 4.0));
    let count = color_count.min(MAX_COLORS as f64).max(2.0);

    let scale = img_num(ctx, "scale", 1.0);
    sp[0] = ctx.get_number_param("time", 0.0) as f32;
    sp[1] = img_num(ctx, "repetition", 1.0) as f32;
    sp[2] = img_num(ctx, "speed", 1.0) as f32;
    sp[3] = count as f32;
    sp[4] = img_num(ctx, "angle", 18.0) as f32;
    sp[5] = img_num(ctx, "distortion", 1.0) as f32;
    sp[6] = img_num(ctx, "scaleX", scale) as f32;
    sp[7] = img_num(ctx, "rotation", 0.0) as f32;
    sp[8] = img_num(ctx, "offsetX", 0.0) as f32;
    sp[9] = img_num(ctx, "offsetY", 0.0) as f32;
    sp[10] = image_aspect as f32;
    sp[11] = image_ready as f32;
    sp[12] = img_num(ctx, "scaleY", scale) as f32;
    sp[13] = img_num(ctx, "sourceWidth", img_w) as f32;
    sp[14] = img_num(ctx, "sourceHeight", img_h) as f32;

    let mut all = color_array_param(p(ctx, "backgroundColor"), &DEFAULT_BG);
    all.extend(pad_colors(&colors, &DEFAULT_COLORS, MAX_COLORS));
    pack_colors(cd, &all, MAX_COLORS + 1);
}

/// HeatmapImageShader.ts:438-512.
fn heatmap_image(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const MAX_COLORS: usize = 7;
    const DEFAULT_BG: [&str; 1] = ["#000000"];
    const DEFAULT_COLORS: [&str; 7] = [
        "#112069", "#1f3ca3", "#3265e7", "#6bd8ff", "#ffe77a", "#ff9a1f", "#ff4d00",
    ];
    let (image_aspect, image_ready) = primary_image_metrics(ctx);
    let (img_w, img_h) = primary_image_dims(ctx);

    let colors = image_shader_colors(ctx, &DEFAULT_COLORS, MAX_COLORS);
    let color_count = js_round(img_num(ctx, "colorCount", 7.0));
    let count = color_count.min(MAX_COLORS as f64).max(2.0);

    let scale = img_num(ctx, "scale", 1.0);
    sp[0] = ctx.get_number_param("time", 0.0) as f32;
    sp[1] = img_num(ctx, "repetition", 1.0) as f32;
    sp[2] = img_num(ctx, "speed", 1.0) as f32;
    sp[3] = count as f32;
    sp[4] = img_num(ctx, "angle", 270.0) as f32;
    sp[5] = img_num(ctx, "innerGlow", 1.0) as f32;
    sp[6] = img_num(ctx, "outerGlow", 0.34) as f32;
    sp[7] = img_num(ctx, "scaleX", scale) as f32;
    sp[8] = img_num(ctx, "rotation", 0.0) as f32;
    sp[9] = img_num(ctx, "offsetX", 0.0) as f32;
    sp[10] = img_num(ctx, "offsetY", 0.0) as f32;
    sp[11] = image_aspect as f32;
    sp[12] = image_ready as f32;
    sp[13] = img_num(ctx, "analogGrain", 0.5) as f32;
    sp[14] = img_num(ctx, "scaleY", scale) as f32;
    sp[15] = img_num(ctx, "sourceWidth", img_w) as f32;
    sp[16] = img_num(ctx, "sourceHeight", img_h) as f32;
    // heatmapPlacement = resources?.primaryImage?.contentRect ?? identity.
    let placement = ctx
        .image
        .as_ref()
        .and_then(|i| i.content_rect)
        .unwrap_or([0.0, 0.0, 1.0, 1.0]);
    sp[17] = placement[0] as f32;
    sp[18] = placement[1] as f32;
    sp[19] = placement[2] as f32;
    sp[20] = placement[3] as f32;

    let mut all = color_array_param(p(ctx, "backgroundColor"), &DEFAULT_BG);
    all.extend(pad_colors(&colors, &DEFAULT_COLORS, MAX_COLORS));
    pack_colors(cd, &all, MAX_COLORS + 1);
}

/// DitherImageShader.ts:171-205 (colorData intentionally all-zero; dithering
/// is baked into the texture CPU-side — `prepareDitherImage`).
fn dither_image(ctx: &Ctx, sp: &mut Sp) {
    let (image_aspect, image_ready) = primary_image_metrics(ctx);
    let (img_w, img_h) = primary_image_dims(ctx);
    let scale = img_num(ctx, "scale", 1.0);
    sp[4] = img_num(ctx, "scaleX", scale) as f32;
    sp[5] = img_num(ctx, "rotation", 0.0) as f32;
    sp[6] = img_num(ctx, "offsetX", 0.0) as f32;
    sp[7] = img_num(ctx, "offsetY", 0.0) as f32;
    sp[8] = img_num(ctx, "sourceWidth", img_w) as f32;
    sp[9] = img_num(ctx, "sourceHeight", img_h) as f32;
    sp[10] = img_num(ctx, "scaleY", scale) as f32;
    sp[11] = image_ready as f32;
    sp[12] = image_aspect as f32;
}

/// Shared `packSourceImagePlacement` ABI (`imageShaderHelpers.ts`): aspect,
/// ready, scale X/Y, rotation, offset X/Y, then source width/height.
fn source_image_placement(ctx: &Ctx, sp: &mut Sp, offset: usize) {
    let (image_aspect, image_ready) = primary_image_metrics(ctx);
    let (img_w, img_h) = primary_image_dims(ctx);
    let scale = img_num(ctx, "scale", 1.0);
    sp[offset] = image_aspect as f32;
    sp[offset + 1] = image_ready as f32;
    sp[offset + 2] = img_num(ctx, "scaleX", scale) as f32;
    sp[offset + 3] = img_num(ctx, "scaleY", scale) as f32;
    sp[offset + 4] = img_num(ctx, "rotation", 0.0) as f32;
    sp[offset + 5] = img_num(ctx, "offsetX", 0.0) as f32;
    sp[offset + 6] = img_num(ctx, "offsetY", 0.0) as f32;
    sp[offset + 7] = img_num(ctx, "sourceWidth", img_w) as f32;
    sp[offset + 8] = img_num(ctx, "sourceHeight", img_h) as f32;
}

/// Gradient Map placeholder ABI: loading=0, ready=1, terminal failure=2.
fn primary_image_state(ctx: &Ctx) -> f32 {
    match ctx.image {
        Some(image) if image.failed => 2.0,
        Some(image) if image.ready => 1.0,
        _ => 0.0,
    }
}

/// GradientMapImageShader.ts: descriptor defaults + custom packing map.
fn gradient_map_image(ctx: &Ctx, sp: &mut Sp, cd: &mut Cd) {
    const MAX_COLORS: usize = 8;
    const DEFAULT_COLORS: [&str; 8] = [
        "#050816", "#142B77", "#2458D6", "#23A6D5", "#6EE7B7", "#FDE68A", "#FB923C", "#F8FAFC",
    ];

    source_image_placement(ctx, sp, 0);
    sp[9] = img_num(ctx, "offset", 0.0) as f32;
    sp[10] = js_round(img_num(ctx, "colorCount", MAX_COLORS as f64))
        .max(2.0)
        .min(MAX_COLORS as f64) as f32;
    sp[11] = primary_image_state(ctx);

    let provided = p(ctx, "colors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .take(MAX_COLORS)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            DEFAULT_COLORS
                .iter()
                .map(|color| color.to_string())
                .collect()
        });
    let mut colors = if provided.len() >= 2 {
        provided
    } else {
        DEFAULT_COLORS
            .iter()
            .map(|color| color.to_string())
            .collect()
    };
    let last = colors
        .last()
        .cloned()
        .unwrap_or_else(|| "#ffffff".to_string());
    colors.resize(MAX_COLORS, last);
    pack_colors(cd, &colors, MAX_COLORS);
}

/// The image shader effects ported from the shader sandbox (docs/qa/shader-sandbox-port.md): the shared
/// source-image placement ABI in [0..8] (the node's image fill), the image state in [11] (as Gradient Map), then
/// the manifest's number table (offsets from 12, renderer clock included) and its colours.
fn source_image_effect(shader_id: &str, ctx: &Ctx, table: &[ParamMeta], sp: &mut Sp, cd: &mut Cd) {
    source_image_placement(ctx, sp, 0);
    sp[11] = primary_image_state(ctx);
    ctx.pack_number_params(shader_id, table, sp);
    if let Some(colors) = super::packing_tables()
        .get(shader_id)
        .and_then(|meta| meta.colors.as_ref())
    {
        ctx.pack_color_array(colors, cd);
    }
}
