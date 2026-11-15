//! page_render.rs — port of the page-quad fill resolution the export /
//! screenshot paths bake under every page (`pages_getPageRenderData`,
//! `core/scene/scene.pages.ts:249-270`), plus its helpers
//! `sampleRepresentativeShaderColor` (scene.pages.ts:210-237) and
//! `averageHexColors` (scene.pages.ts:185-208).
//!
//! WHY THIS EXISTS (WF5 v3 grind, shaders/page-background): the browser's
//! export coordinator gets `pageData` from `scene.getPageRenderData()` and
//! bakes `createPageBackgroundData(pageData)` for both the export and the
//! agent-screenshot configs (`renderers/webgpu/export/WebGPUExportCoordinator.ts:267-331,605-650`).
//! For a page whose top background layer is a shader paint, the quad UNDER
//! the transient shader rectangle is NOT a stored colour — it is the
//! "representative" color sampled from the shader params (the average of the
//! first known palette key). A page-background shader whose own `background`
//! param is the default transparent `#00000000` (shape-grid / shape-grid-dots)
//! therefore composites its dots over that representative color, not over
//! the page colour. Rendering the page colour instead (the old native behavior) made
//! same-color dots invisible (canvas d463d8b5 page 4: dots #0B0C10 over page
//! fill #0b0c10 vs the browser's avg(#0B0C10,#00F5FF) ≈ teal #058188 quad).
//! Opaque page-bg shaders mask the divergence, which is why only the
//! shape-grid family exposed it.

use crate::value::Value;

use crate::types::RawPage;

/// `PAGE_DEFAULTS.backgroundColor` (`core/config/canvasDefaults.ts:147`).
pub const PAGE_DEFAULT_BACKGROUND: &str = "#ffffff";

/// `KNOWN_SHADER_PALETTE_KEYS` (`core/procedural/patterns/shaderPaletteKeys.ts:14-28`) —
/// order matters: the first key with a usable average wins.
const KNOWN_SHADER_PALETTE_KEYS: [&str; 13] = [
    "colors",
    "gridColors",
    "palette",
    "swirlColors",
    "metalColors",
    "baseColor",
    "backgroundColor",
    "backgroundColors",
    "panelColor",
    "blobColor",
    "ribbonColors",
    "blobColors",
    "metaballColors",
];

/// `averageHexColors` (`core/scene/scene.pages.ts:185-208`): only entries
/// matching `^#?[0-9a-fA-F]{6}$` (after trim) count; channel-wise mean,
/// `Math.round`ed, emitted as lowercase `#rrggbb`. None when nothing matched.
fn average_hex_colors<'a>(colors: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut sr: u64 = 0;
    let mut sg: u64 = 0;
    let mut sb: u64 = 0;
    let mut count: u64 = 0;
    for color in colors {
        let t = color.trim();
        // /^#?([0-9a-fA-F]{6})$/ — at most one leading '#', exactly 6 hex digits.
        let hex = t.strip_prefix('#').unwrap_or(t);
        if hex.len() != 6 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            continue;
        }
        sr += u64::from_str_radix(&hex[0..2], 16).ok()?;
        sg += u64::from_str_radix(&hex[2..4], 16).ok()?;
        sb += u64::from_str_radix(&hex[4..6], 16).ok()?;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    // JS Math.round(value / count) on non-negative values == round half up.
    let avg = |s: u64| -> u64 { ((s as f64) / (count as f64)).round() as u64 };
    Some(format!("#{:02x}{:02x}{:02x}", avg(sr), avg(sg), avg(sb)))
}

/// `sampleRepresentativeShaderColor` (`core/scene/scene.pages.ts:210-237`).
///
/// 1. `background ?? backgroundColor` if it is a STRING (nullish-coalescing:
///    an explicit JSON `null` background falls through to `backgroundColor`;
///    any other non-string `background` value still shadows it).
/// 2. First `KNOWN_SHADER_PALETTE_KEYS` array whose string entries average to
///    a color (6-digit-hex entries only — 8-digit/rgba entries are skipped).
/// 3. `centers[*].color` strings averaged (mesh-rbf style params).
pub fn sample_representative_shader_color(shader_params: Option<&Value>) -> Option<String> {
    // `!shaderParams || typeof shaderParams !== 'object'` — JS arrays are
    // typeof 'object' but have none of the looked-up keys, so collapsing
    // arrays to None is behavior-identical.
    let obj = shader_params?.as_object()?;

    let background = match obj.get("background") {
        Some(Value::Null) | None => obj.get("backgroundColor"),
        other => other,
    };
    if let Some(Value::String(s)) = background {
        return Some(s.clone());
    }

    for key in KNOWN_SHADER_PALETTE_KEYS {
        if let Some(Value::Array(arr)) = obj.get(key) {
            if let Some(rep) = average_hex_colors(arr.iter().filter_map(Value::as_str)) {
                return Some(rep);
            }
        }
    }

    if let Some(Value::Array(centers)) = obj.get("centers") {
        // `typeof value === 'object' && value !== null && 'color' in value`,
        // collect string colors; return the average DIRECTLY (no fallthrough).
        return average_hex_colors(
            centers
                .iter()
                .filter_map(|v| v.as_object())
                .filter_map(|o| o.get("color"))
                .filter_map(Value::as_str),
        );
    }

    None
}

/// The `fill` field of `PageRenderData` (`pageRenderFill`,
/// `core/scene/scene.pages.ts`): the page quad's solid colour. The top
/// background layer decides: a solid's colour string, a shader's
/// representative colour (else the default), and the default under a
/// gradient or image (the page's background rectangle draws over it). A page
/// without a background has no quad: the empty string (the page system's
/// `if (page.fill)` skips it).
/// A colour string that may carry alpha (`#rgba`, `#rrggbbaa`, `rgba()`, `hsla()`, `transparent`).
fn may_be_translucent(color: &str) -> bool {
    let c = color.trim().to_ascii_lowercase();
    let hex = c.strip_prefix('#').map(str::len);
    matches!(hex, Some(4 | 8)) || c.starts_with("rgba") || c.starts_with("hsla") || c == "transparent"
}

pub fn page_render_fill(page: &RawPage) -> String {
    let Some(fill) = crate::paint::top_page_fill(page) else {
        return String::new();
    };
    // A stack of background layers (E3.1 P6) draws on the page's background
    // rectangle, over the default: unless its top layer is an opaque solid
    // (which hides the rest), the quad must not show the top layer under the
    // layers below it.
    let visible = crate::paint::page_fills(page)
        .iter()
        .filter(|f| crate::paint::shown(f.visible, f.opacity))
        .count();
    let top_covers = fill.opacity.unwrap_or(1.0) >= 1.0
        && fill.blend_mode.as_deref().is_none_or(|m| m == "normal")
        && crate::paint::solid_color(&fill.paint).is_some_and(|c| !may_be_translucent(c));
    if visible > 1 && !top_covers {
        return PAGE_DEFAULT_BACKGROUND.to_string();
    }
    match &fill.paint {
        crate::paint::Paint::Solid(s) => s
            .color
            .as_str()
            .map_or_else(|| PAGE_DEFAULT_BACKGROUND.to_string(), str::to_string),
        crate::paint::Paint::Shader(s) => {
            let params = s.params.clone().map(Value::Object);
            sample_representative_shader_color(params.as_ref())
                .unwrap_or_else(|| PAGE_DEFAULT_BACKGROUND.to_string())
        }
        _ => PAGE_DEFAULT_BACKGROUND.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn page(v: Value) -> RawPage {
        RawPage(v.as_object().unwrap().clone())
    }

    /// A page whose background is the one `paint`.
    fn page_with(paint: Value) -> RawPage {
        page(json!({"id": "p", "fills": [{"id": "fill-0", "paint": paint}]}))
    }

    fn shader_page(params: Value) -> RawPage {
        page_with(json!({"type": "shader", "shaderId": "shape-grid-dots", "params": params}))
    }

    #[test]
    fn a_background_stack_draws_over_the_default() {
        let p = page(json!({"id": "p", "fills": [
            {"id": "a", "paint": {"type": "solid", "color": "#0b0c10"}},
            {"id": "b", "paint": {"type": "solid", "color": "rgba(255,0,0,0.5)"}}]}));
        assert_eq!(page_render_fill(&p), PAGE_DEFAULT_BACKGROUND);
    }

    #[test]
    fn solid_background_uses_its_color() {
        let p = page_with(json!({"type": "solid", "color": "#0b0c10"}));
        assert_eq!(page_render_fill(&p), "#0b0c10");
    }

    #[test]
    fn solid_non_string_color_is_white() {
        // Scene.pages.ts `typeof fill === 'string'` guard (an unresolved
        // variable reference).
        let p = page_with(json!({"type": "solid", "color": {"type": "variable", "variableId": "v1"}}));
        assert_eq!(page_render_fill(&p), "#ffffff");
    }

    #[test]
    fn the_top_visible_layer_decides() {
        let p = page(json!({"id": "p", "fills": [
            {"id": "fill-0", "paint": {"type": "solid", "color": "#111111"}},
            {"id": "fill-1", "paint": {"type": "solid", "color": "#222222"}},
            {"id": "fill-2", "visible": false, "paint": {"type": "solid", "color": "#333333"}}
        ]}));
        assert_eq!(page_render_fill(&p), "#222222");
    }

    #[test]
    fn gradient_and_image_backgrounds_are_white() {
        let stops = json!([{"offset": 0.0, "color": "#0b0c10"}, {"offset": 1.0, "color": "#ffffff"}]);
        for paint in [
            json!({"type": "linear", "start": {"x": 0.0, "y": 50.0}, "end": {"x": 100.0, "y": 50.0}, "stops": stops}),
            json!({"type": "radial", "center": {"x": 50.0, "y": 50.0}, "radiusX": 50.0, "radiusY": 50.0, "rotation": 0.0, "stops": stops}),
            json!({"type": "conic", "center": {"x": 50.0, "y": 50.0}, "angle": 0.0, "stops": stops}),
            json!({"type": "image", "src": "a.png", "placement": {}}),
        ] {
            assert_eq!(page_render_fill(&page_with(paint.clone())), "#ffffff", "paint {paint}");
        }
    }

    #[test]
    fn shader_averages_first_palette_key() {
        // The d463d8b5 page-4 case: avg(#0B0C10, #00F5FF) = #058188-ish teal.
        let p = shader_page(json!({
            "colors": ["#0B0C10", "#00F5FF"],
            "blobColors": ["#00F5FF", "#00FFA3"],
            "backgroundColors": ["#0B0C10", "#0a1628"]
        }));
        // (11+0)/2=5.5→6, (12+245)/2=128.5→129, (16+255)/2=135.5→136
        assert_eq!(page_render_fill(&p), "#068188");
    }

    #[test]
    fn shader_background_string_wins() {
        let p = shader_page(json!({"background": "#00000000", "colors": ["#ff0000"]}));
        // Even a non-6-digit background string is returned VERBATIM (the page
        // system's colorToNormalizedRgb then fails on 8-digit hex → white).
        assert_eq!(page_render_fill(&p), "#00000000");
    }

    #[test]
    fn shader_null_background_falls_to_background_color() {
        let p = shader_page(json!({"background": null, "backgroundColor": "#abcdef"}));
        assert_eq!(page_render_fill(&p), "#abcdef");
    }

    #[test]
    fn shader_palette_key_order_and_skipping() {
        // 8-digit entries are skipped by averageHexColors; an array that
        // averages to nothing falls through to the NEXT key.
        let p = shader_page(json!({
            "colors": ["#00000000", "rgba(1,2,3,1)"],
            "gridColors": ["#102030", "#304050"]
        }));
        assert_eq!(page_render_fill(&p), "#203040");
    }

    #[test]
    fn shader_centers_fallback() {
        let p = shader_page(json!({"centers": [{"color": "#102030"}, {"color": "#304050"}, {"x": 1}]}));
        assert_eq!(page_render_fill(&p), "#203040");
    }

    #[test]
    fn shader_no_usable_params_is_white() {
        for params in [json!({}), json!({"colors": "nope"})] {
            assert_eq!(page_render_fill(&shader_page(params.clone())), "#ffffff", "params {params}");
        }
        let p = page_with(json!({"type": "shader", "shaderId": "shape-grid-dots"}));
        assert_eq!(page_render_fill(&p), "#ffffff");
    }

    #[test]
    fn no_background_has_no_quad() {
        // WebGPUPageSystem.ts `if (page.fill)`: the ENGINE skips the quad for
        // '' (JS falsy). A page with no visible background layer (`fills: []`,
        // every layer hidden) has none [Q5]; the white default of a colour
        // page lives in the document codemod (024-paint-stacks).
        assert_eq!(page_render_fill(&page(json!({"id": "p", "fills": []}))), "");
        let hidden = page(json!({"id": "p", "fills": [
            {"id": "fill-0", "visible": false, "paint": {"type": "solid", "color": "#123456"}}
        ]}));
        assert_eq!(page_render_fill(&hidden), "");
    }

    #[test]
    fn empty_string_color_passes_through() {
        // The rule returns an empty solid colour verbatim (the engine skips it).
        let p = page_with(json!({"type": "solid", "color": ""}));
        assert_eq!(page_render_fill(&p), "");
    }
}
