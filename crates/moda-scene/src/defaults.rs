//! Render-time defaults table (scene-schema.md §15) as typed accessors over
//! `RawNode`. Every value cites its TS source line; the unit tests below pin
//! each one (PLAN M0c: "defaults table with per-value unit tests citing TS
//! source lines").
//!
//! NOTE on creation vs render defaults: values like `strokeWidth` have
//! *creation* defaults (shapes 0, lines/paths 2 — `core/config/shapeDefaults.ts:13,117-118,135-136`)
//! but at render time an ABSENT field simply means "no stroke"
//! (scene-schema.md §3.6). Accessors here implement the RENDER defaults.

use crate::types::RawNode;
use crate::value::Value;

impl RawNode {
    /// `opacity ?? 1` (nodes.ts:113-159 BaseNodeProperties; displayList.ts:202).
    pub fn opacity(&self) -> f64 {
        self.f64("opacity").unwrap_or(1.0)
    }

    /// `hidden ?? false` (nodes.ts BaseNodeProperties).
    pub fn hidden(&self) -> bool {
        self.bool("hidden").unwrap_or(false)
    }

    /// `locked ?? false` (nodes.ts BaseNodeProperties).
    pub fn locked(&self) -> bool {
        self.bool("locked").unwrap_or(false)
    }

    /// `rotation || 0` (nodeTransformUtils.ts:257 reads `node.rotation || 0`).
    pub fn rotation(&self) -> f64 {
        self.f64("rotation").unwrap_or(0.0)
    }

    /// `skewX` in degrees (`nodeSkewX`, transform2d.ts): 0 when absent or not finite.
    pub fn skew_x(&self) -> f64 {
        match self.f64("skewX") {
            Some(v) if v.is_finite() => v.clamp(-89.0, 89.0),
            _ => 0.0,
        }
    }

    /// Uniform `scale || 1` (nodeTransformUtils.ts:264-268).
    pub fn scale(&self) -> f64 {
        let s = self.f64("scale").unwrap_or(1.0);
        if s == 0.0 {
            1.0 // `node.scale || 1` — JS falsy 0 coerces to 1
        } else {
            s
        }
    }

    /// Group `scaleX || 1` / `scaleY || 1` (nodeTransformUtils.ts:261-263).
    pub fn scale_xy(&self) -> (f64, f64) {
        if self.has("scaleX") || self.has("scaleY") {
            let sx = self.f64("scaleX").unwrap_or(1.0);
            let sy = self.f64("scaleY").unwrap_or(1.0);
            (
                if sx == 0.0 { 1.0 } else { sx },
                if sy == 0.0 { 1.0 } else { sy },
            )
        } else {
            let s = self.scale();
            (s, s)
        }
    }


    /// `getRoundedRectCornerRadii` clone (nodes.ts:1819-1827): scalar
    /// broadcast; array entries beyond its length read as 0.
    pub fn corner_radii(&self) -> [f64; 4] {
        match self.json_field("cornerRadius") {
            Some(Value::Number(n)) => {
                let r = n.as_f64().unwrap_or(0.0);
                [r, r, r, r]
            }
            Some(Value::Array(a)) => {
                let g = |i: usize| a.get(i).and_then(Value::as_f64).unwrap_or(0.0);
                [g(0), g(1), g(2), g(3)]
            }
            _ => [0.0; 4],
        }
    }

    /// The top stroke's `align ?? 'center'` (scene-schema.md §15).
    pub fn stroke_align(&self) -> String {
        crate::paint::top_stroke(self)
            .and_then(|s| s.align)
            .unwrap_or_else(|| "center".to_string())
    }

    /// `fillRule ?? 'nonzero'` (nodes.ts:804-840 PathNode).
    pub fn fill_rule(&self) -> &str {
        self.str("fillRule").unwrap_or("nonzero")
    }

    /// path `closed ?? false` (nodes.ts:804-840).
    pub fn path_closed(&self) -> bool {
        self.bool("closed").unwrap_or(false)
    }

    /// line `routing ?? 'straight'` (nodes.ts:726-753).
    pub fn line_routing(&self) -> &str {
        self.str("routing").unwrap_or("straight")
    }

    /// `pointerLength` / `pointerWidth` doc default 10 (nodes.ts:726-753;
    /// editor creates with 20 — shapeDefaults.ts:125-126 — but the RENDER
    /// default for an absent field is 10).
    pub fn pointer_length(&self) -> f64 {
        self.f64("pointerLength").unwrap_or(10.0)
    }
    pub fn pointer_width(&self) -> f64 {
        self.f64("pointerWidth").unwrap_or(10.0)
    }

    /// Shape text frame: `textPadding ?? default`, where the DEFAULT is capped at
    /// `floor(min(width, height) * 0.15)` so a badge-sized shape keeps a usable text area
    /// (`getTextPadding`, textInShapeHelpers.ts:284-315; shapeDefaults.ts:186-199). A flat 8
    /// eats the whole inscribed rect of a 36px circle. The cap reaches 8 at ~54px, so larger
    /// shapes are unchanged, and an explicit `textPadding` is always used as given.
    ///
    /// Paths are exempt: they carry no width/height, and their text box comes from the
    /// centroid's distance to the nearest edge rather than from a bounding box.
    pub fn text_frame_padding(&self) -> f64 {
        const BASE: f64 = 8.0;
        if let Some(explicit) = self.f64("textPadding") {
            return explicit;
        }
        if self.node_type() == "path" {
            return BASE;
        }
        // Validate each dimension BEFORE taking the minimum: `f64::min` ignores NaN
        // (`NAN.min(10.0) == 10.0`) where JS `Math.min` propagates it, so folding first and
        // checking after would let a NaN width cap the padding here while `getTextPadding`
        // falls back to 8 — exactly the silent drift this mirror exists to prevent.
        let w = self.f64("width").unwrap_or(0.0);
        let h = self.f64("height").unwrap_or(0.0);
        if !w.is_finite() || !h.is_finite() || w <= 0.0 || h <= 0.0 {
            return BASE;
        }
        BASE.min((w.min(h) * 0.15).floor())
    }

    /// Frame `textHorizontalAlign ?? 'center'` (nodes.ts:416-458).
    pub fn text_frame_h_align(&self) -> &str {
        self.str("textHorizontalAlign").unwrap_or("center")
    }

    /// Frame `textVerticalAlign ?? 'middle'` (nodes.ts:416-458).
    pub fn text_frame_v_align(&self) -> &str {
        self.str("textVerticalAlign").unwrap_or("middle")
    }

    /// Frame `textResizeMode ?? 'no-resize'` (nodes.ts:416-458).
    pub fn text_frame_resize_mode(&self) -> &str {
        self.str("textResizeMode").unwrap_or("no-resize")
    }

    /// richtext `verticalAlign ?? 'top'` (nodes.ts:1063-1128).
    pub fn richtext_vertical_align(&self) -> &str {
        self.str("verticalAlign").unwrap_or("top")
    }

    /// `blendMode ?? 'normal'` (nodes.ts:7; scene-schema.md §3.7).
    pub fn blend_mode(&self) -> &str {
        self.str("blendMode").unwrap_or("normal")
    }

    /// Connector label `textPosition ?? 0.5` (nodes.ts:470-543).
    pub fn connector_text_position(&self) -> f64 {
        self.f64("textPosition").unwrap_or(0.5)
    }

    /// Connector label `textPadding ?? 4` (nodes.ts:470-543).
    pub fn connector_text_padding(&self) -> f64 {
        self.f64("textPadding").unwrap_or(4.0)
    }

    // Drop / inner shadow param defaults (a missing `opacity` is 1, a missing
    // `spread` 0) live on the effect types: `effect::DropShadow::opacity`,
    // `effect::InnerShadow::opacity` / `spread`.

    // --- Brushes / variable-width strokes (the stroke's `profile` / `taper` /
    // `wave` / `dynamic` / `brush`, `PathNode.inkChannels`; brushes.md §4).
    // They are read from the MODEL stroke (`paint::model_stroke`: the top-most
    // stroke with a width profile or brush, else the top stroke). Absent or
    // unrepairable ⇒ `None`, which renders the plain uniform stroke. Present
    // objects get the same defaults and clamps as `core/scene/strokeModel.ts`
    // (see stroke_model.rs). Paths carry the whole model; lines carry the width
    // model (profile, taper, wave; §4.9).

    /// The model stroke's `miterLimit` — absent ⇒ 4, clamped to 1–500.
    pub fn miter_limit(&self) -> f64 {
        crate::paint::model_stroke(self)
            .and_then(|s| s.miter_limit)
            .filter(|v| v.is_finite())
            .map_or(4.0, |v| v.clamp(1.0, 500.0))
    }

    /// The width profile — absent ⇒ uniform width.
    pub fn stroke_profile(&self) -> Option<crate::stroke_model::WidthProfile> {
        self.variable_stroke_value(StrokeObject::Profile)
            .as_ref()
            .and_then(crate::stroke_model::parse_width_profile)
    }

    /// The taper — `units ?? 'percent'`, missing numbers 0.
    pub fn stroke_taper(&self) -> Option<crate::stroke_model::StrokeTaper> {
        self.variable_stroke_value(StrokeObject::Taper)
            .as_ref()
            .and_then(crate::stroke_model::parse_stroke_taper)
    }

    /// The wave — `units ?? 'px'`, `wavelength` 20 px / 1 cycle when not positive.
    pub fn stroke_wave(&self) -> Option<crate::stroke_model::StrokeWave> {
        self.variable_stroke_value(StrokeObject::Wave)
            .as_ref()
            .and_then(crate::stroke_model::parse_stroke_wave)
    }

    /// The dynamic wobble — `frequency ?? 1` clamped to 0.01..20.
    pub fn stroke_dynamic(&self) -> Option<crate::stroke_model::StrokeDynamic> {
        self.variable_stroke_value(StrokeObject::Dynamic)
            .as_ref()
            .and_then(crate::stroke_model::parse_stroke_dynamic)
    }

    /// The brush — requires a non-empty `brushId`.
    pub fn stroke_brush(&self) -> Option<crate::stroke_model::BrushApplication> {
        self.variable_stroke_value(StrokeObject::Brush)
            .as_ref()
            .and_then(crate::stroke_model::parse_brush_application)
    }

    /// `inkChannels` — path pen samples.
    pub fn ink_channels(&self) -> Option<crate::stroke_model::InkChannels> {
        if self.node_type() != "path" {
            return None;
        }
        self.json_field("inkChannels")
            .filter(|v| !v.is_null())
            .and_then(crate::stroke_model::parse_ink_channels)
    }

    /// Seed for a dynamic stroke: `strokeDynamic.seed ?? strokeBrush.seed ?? 0`.
    pub fn stroke_dynamic_seed(&self) -> u32 {
        self.stroke_dynamic()
            .and_then(|d| d.seed)
            .or_else(|| self.stroke_brush().and_then(|b| b.seed))
            .unwrap_or(0)
    }

    fn variable_stroke_value(&self, key: StrokeObject) -> Option<Value> {
        // Paths carry the whole model; lines the width model only (brushes.md §4.9).
        let allowed = match self.node_type() {
            "path" => true,
            "line" => matches!(key, StrokeObject::Profile | StrokeObject::Taper | StrokeObject::Wave),
            _ => false,
        };
        if !allowed {
            return None;
        }
        let stroke = crate::paint::model_stroke(self)?;
        match key {
            StrokeObject::Profile => stroke.profile,
            StrokeObject::Taper => stroke.taper,
            StrokeObject::Wave => stroke.wave,
            StrokeObject::Dynamic => stroke.dynamic,
            StrokeObject::Brush => stroke.brush,
        }
        .filter(|v| !v.is_null())
    }
}

/// The stroke-scoped model objects (`Stroke.profile` … `Stroke.brush`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrokeObject {
    Profile,
    Taper,
    Wave,
    Dynamic,
    Brush,
}


/// Page defaults.
pub mod page {
    /// `PAGE_DEFAULTS.backgroundColor` (`core/config/canvasDefaults.ts:147`).
    pub const BACKGROUND_COLOR: &str = "#ffffff";
    /// `PAGE_DEFAULTS.pageSpacing` (`core/config/canvasDefaults.ts:152`).
    pub const PAGE_SPACING: f64 = 100.0;
    /// `DEFAULT_PAGES_LAYOUT_SPACING` (`core/config/layoutDefaults.ts:14`),
    /// forced when layoutMode === 'pages' (`core/scene/scene.layout.ts:33-35`).
    pub const PAGES_LAYOUT_SPACING: f64 = 2500.0;
}

/// Table defaults (`core/scene/tableExpansion.ts:54-55`).
pub mod table {
    pub const STROKE: &str = "#000000";
    pub const STROKE_WIDTH: f64 = 1.0;
    /// `PLACEMENT_DEFAULTS.table.cellWidth` (`core/config/placementDefaults.ts:41`).
    /// Used by the load-boundary table migration to synthesize a MISSING
    /// `columns` axis entry, so a healed table lands on the same geometry the
    /// table tool would have created.
    pub const CELL_WIDTH: f64 = 150.0;
    /// `PLACEMENT_DEFAULTS.table.cellHeight` (`core/config/placementDefaults.ts:42`).
    pub const CELL_HEIGHT: f64 = 50.0;
    /// `PLACEMENT_DEFAULTS.table.htmlContent` (`core/config/placementDefaults.ts:43`).
    /// The `\"` escapes in the TS literal are single-quoted-string noise — the
    /// authored value carries plain double quotes, and `table_cell_to_richtext`
    /// parses it as such, so the bytes must match exactly.
    pub const CELL_HTML_CONTENT: &str =
        r#"<p style="font-size: 20px; font-family: Inter; color: #000000; font-weight: 400"></p>"#;
    /// `SHAPE_DEFAULTS.table.cellTextPadding` (`core/config/shapeDefaults.ts:173`).
    pub const CELL_TEXT_PADDING: f64 = 8.0;
}

/// Generated-graphic source defaults (`core/scene/nodes.ts`).
pub mod generated {
    /// nodes.ts:1182.
    pub const QR_ERROR_CORRECTION: &str = "M";
    /// `DEFAULT_QR_FOREGROUND_COLOR` (nodes.ts).
    pub const QR_FOREGROUND_COLOR: &str = "#000000";
    /// `DEFAULT_QR_BACKGROUND_COLOR` (nodes.ts).
    pub const QR_BACKGROUND_COLOR: &str = "#ffffff";
    /// nodes.ts:1186.
    pub const LATEX_FONT_SIZE: f64 = 32.0;
    /// nodes.ts:1187.
    pub const LATEX_COLOR: &str = "#111111";
    /// Single-asset placeholder fill (`generatedGraphicExpansion.ts:96`).
    pub const PLACEHOLDER_FILL: &str = "#f0f0f0";
    /// Single-asset placeholder stroke (`generatedGraphicExpansion.ts:98`).
    pub const PLACEHOLDER_STROKE: &str = "#cccccc";
    /// Chart placeholder fill (`chartExpansion.ts:145`).
    pub const CHART_PLACEHOLDER_FILL: &str = "#f5f5f5";
    /// Chart placeholder stroke (`chartExpansion.ts:146`).
    pub const CHART_PLACEHOLDER_STROKE: &str = "#cccccc";

    /// The chart config `createDefaultChartConfig` builds (nodes.ts), the one default for new and imported charts.
    ///
    /// DUPLICATED, like every other value in this module, and for the same
    /// reason: there is no codegen between the TS scene schema and this port.
    /// Unlike the others it is 76 fields, so a hand-kept copy would drift in
    /// silence — `default_chart_config_matches_the_typescript` below reads
    /// `core/scene/nodes.ts` and fails when it does, which makes the drift loud
    /// instead. Held as JSON text rather than a `json!` literal because the
    /// macro exceeds the recursion limit at this width.
    ///
    /// `r##` and not `r#`: the config is full of hex colours, and a `"#` inside
    /// the text closes an `r#"…"#` string early.
    const DEFAULT_CHART_CONFIG_JSON: &str = r##"
    {
        "fontFamily": "Inter",
        "fontSize": 12,
        "labelColor": "#666666",
        "titleColor": "#1a1a1a",
        "subtitleColor": "#666666",
        "titleSize": 16,
        "titleWeight": 600,
        "titleAlign": "left",
        "subtitleSize": 12,
        "palette": [
            "#3b82f6",
            "#ef4444",
            "#22c55e",
            "#f59e0b",
            "#8b5cf6",
            "#06b6d4",
            "#ec4899",
            "#64748b"
        ],
        "background": "transparent",
        "showGrid": true,
        "gridColor": "#e5e5e5",
        "gridStyle": "solid",
        "gridWidth": 1,
        "showXAxis": true,
        "showYAxis": true,
        "axisColor": "#cccccc",
        "axisWidth": 1,
        "showXLabels": true,
        "showYLabels": true,
        "yTickCount": 5,
        "yFormat": "auto",
        "yScale": "linear",
        "barRadius": 0,
        "barGap": 0.2,
        "barGroupGap": 0.1,
        "showLabels": false,
        "labelPosition": "top",
        "labelFormat": "auto",
        "valueFormat": "auto",
        "valueDensity": "auto",
        "valueLabelColor": "#666666",
        "valueLabelSize": 11,
        "showLegend": "auto",
        "legendPosition": "bottom",
        "legendAlign": "center",
        "legendGap": 8,
        "legendMarker": "square",
        "legendMarkerSize": 12,
        "legendColor": "#666666",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "curve": "linear",
        "strokeCap": "round",
        "strokeJoin": "round",
        "showPoints": false,
        "pointSize": 8,
        "pointShape": "circle",
        "pointFill": "",
        "pointStroke": "",
        "pointStrokeWidth": 0,
        "fillOpacity": 0.3,
        "baseline": 0,
        "pointOpacity": 0.7,
        "minPointSize": 8,
        "maxPointSize": 40,
        "xTickCount": 5,
        "innerRadius": 0,
        "startAngle": -90,
        "direction": "cw",
        "padAngle": 0,
        "explode": [],
        "explodeDistance": 10,
        "explodeAll": false,
        "centerLabel": "",
        "centerSublabel": "",
        "centerColor": "#1a1a1a",
        "centerLabelColor": "#1a1a1a",
        "centerSublabelColor": "#666666",
        "centerSize": 24,
        "pieLabelPosition": "outside",
        "pieLabelFormat": "percent",
        "minLabelAngle": 10,
        "padding": 16,
        "cornerRadius": 0
    }
    "##;

    /// Parsed once. The text above is a compile-time constant, so a parse
    /// failure is a bug in this file and not in any document.
    pub fn default_chart_config() -> &'static crate::value::Value {
        static CONFIG: std::sync::OnceLock<crate::value::Value> = std::sync::OnceLock::new();
        CONFIG.get_or_init(|| {
            crate::value::from_str(DEFAULT_CHART_CONFIG_JSON)
                .expect("default chart config is valid JSON")
        })
    }
}

/// Rich text default inline style (`utils/richtext/types.ts:30-39`).
pub mod richtext {
    pub const FONT_FAMILY: &str = "sans-serif";
}

#[cfg(test)]
mod tests {
    use crate::types::RawNode;
    use crate::value::json;

    fn node(v: crate::value::Value) -> RawNode {
        RawNode::from_value(v).unwrap()
    }

    #[test]
    fn opacity_defaults_to_1() {
        // nodes.ts BaseNodeProperties `opacity?: number` 0–1, default 1
        // (scene-schema.md §15; displayList.ts:202 `node.opacity ?? 1`).
        assert_eq!(node(json!({"id":"a","type":"rectangle"})).opacity(), 1.0);
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","opacity":0.25})).opacity(),
            0.25
        );
    }

    #[test]
    fn hidden_and_locked_default_false() {
        // nodes.ts:113-159 (scene-schema.md §3.1).
        let n = node(json!({"id":"a","type":"rectangle"}));
        assert!(!n.hidden());
        assert!(!n.locked());
    }

    #[test]
    fn rotation_defaults_to_0_and_scale_to_1() {
        // nodeTransformUtils.ts:257 `node.rotation || 0`;
        // nodeTransformUtils.ts:264-268 `node.scale || 1`.
        let n = node(json!({"id":"a","type":"rectangle"}));
        assert_eq!(n.rotation(), 0.0);
        assert_eq!(n.scale(), 1.0);
        // `scale: 0` is falsy in JS → || 1 (copy the bug).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","scale":0})).scale(),
            1.0
        );
    }

    #[test]
    fn group_scale_xy_defaults() {
        // nodeTransformUtils.ts:261-263 `node.scaleX || 1` / `node.scaleY || 1`.
        let n = node(json!({"id":"g","type":"group","scaleX":2.0}));
        assert_eq!(n.scale_xy(), (2.0, 1.0));
    }

    #[test]
    fn corner_radius_scalar_array_and_short_array() {
        // nodes.ts:1819-1827 getRoundedRectCornerRadii: short arrays read
        // missing entries as 0 (scene-schema.md §4.1).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle"})).corner_radii(),
            [0.0; 4]
        );
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","cornerRadius":5})).corner_radii(),
            [5.0, 5.0, 5.0, 5.0]
        );
        assert_eq!(
            node(json!({"id":"a","type":"rectangle","cornerRadius":[1,2]})).corner_radii(),
            [1.0, 2.0, 0.0, 0.0]
        );
    }

    #[test]
    fn stroke_align_defaults_to_center() {
        // nodes.ts:378-383 (scene-schema.md §3.6/§15).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle"})).stroke_align(),
            "center"
        );
    }

    #[test]
    fn path_defaults() {
        // nodes.ts:804-840: fillRule 'nonzero', closed false (scene-schema.md §4.5).
        let n = node(json!({"id":"p","type":"path","points":[]}));
        assert_eq!(n.fill_rule(), "nonzero");
        assert!(!n.path_closed());
    }

    #[test]
    fn line_defaults() {
        // nodes.ts:726-753: routing 'straight'; pointerLength/Width default 10
        // (doc default; creation default is 20 per shapeDefaults.ts:125-126).
        let n = node(json!({"id":"l","type":"line","points":[0,0,1,1]}));
        assert_eq!(n.line_routing(), "straight");
        assert_eq!(n.pointer_length(), 10.0);
        assert_eq!(n.pointer_width(), 10.0);
    }

    #[test]
    fn text_frame_defaults() {
        // nodes.ts:416-458: padding 8, h center, v middle, no-resize
        // (scene-schema.md §3.10/§15).
        let n = node(json!({"id":"a","type":"rectangle","textContent":"<p>x</p>"}));
        assert_eq!(n.text_frame_padding(), 8.0);
        assert_eq!(n.text_frame_h_align(), "center");
        assert_eq!(n.text_frame_v_align(), "middle");
        assert_eq!(n.text_frame_resize_mode(), "no-resize");
    }

    /// The DEFAULT padding is capped at `floor(min(w,h) * 0.15)` so badge-sized shapes keep a
    /// usable text area (textInShapeHelpers.ts:284-315). Must match `getTextPadding` exactly:
    /// a mismatch places shape text differently in the primary and fallback renderers.
    #[test]
    fn text_frame_padding_small_shape_cap() {
        let big = node(json!({"id":"a","type":"ellipse","width":100.0,"height":100.0}));
        assert_eq!(big.text_frame_padding(), 8.0);

        // 54px is the first size where floor(0.15 * side) reaches the flat 8.
        let at_cap = node(json!({"id":"b","type":"ellipse","width":54.0,"height":54.0}));
        assert_eq!(at_cap.text_frame_padding(), 8.0);
        let below_cap = node(json!({"id":"c","type":"ellipse","width":53.0,"height":53.0}));
        assert_eq!(below_cap.text_frame_padding(), 7.0);

        // 36px badge: 5, not 8 — the reported floor-to-8px-numeral case.
        let badge = node(json!({"id":"d","type":"ellipse","width":36.0,"height":36.0}));
        assert_eq!(badge.text_frame_padding(), 5.0);

        // The SHORTER side drives the cap.
        let wide = node(json!({"id":"e","type":"rectangle","width":400.0,"height":20.0}));
        assert_eq!(wide.text_frame_padding(), 3.0);

        // An explicit textPadding is never capped, and 0 stays 0.
        let explicit =
            node(json!({"id":"f","type":"ellipse","width":36.0,"height":36.0,"textPadding":8.0}));
        assert_eq!(explicit.text_frame_padding(), 8.0);
        let zero =
            node(json!({"id":"g","type":"ellipse","width":36.0,"height":36.0,"textPadding":0.0}));
        assert_eq!(zero.text_frame_padding(), 0.0);

        // Paths (no width/height) and degenerate sizes fall back to the flat default.
        let path = node(json!({"id":"h","type":"path","points":[{"x":0.0,"y":0.0}]}));
        assert_eq!(path.text_frame_padding(), 8.0);
        let zero_size = node(json!({"id":"i","type":"rectangle","width":0.0,"height":0.0}));
        assert_eq!(zero_size.text_frame_padding(), 8.0);

        // ONE bad dimension is enough to fall back, matching JS `Math.min` propagation. Folding
        // with `f64::min` first would hide it, since `NAN.min(x) == x`.
        let one_bad = node(json!({"id":"j","type":"rectangle","width":40.0,"height":0.0}));
        assert_eq!(one_bad.text_frame_padding(), 8.0);
        let missing_height = node(json!({"id":"k","type":"rectangle","width":40.0}));
        assert_eq!(missing_height.text_frame_padding(), 8.0);
    }

    #[test]
    fn connector_text_defaults() {
        // nodes.ts:470-543: textPosition 0.5, textPadding 4 (scene-schema.md §4.4).
        let n = node(json!({"id":"l","type":"line","points":[0,0,1,1],"textContent":"hi"}));
        assert_eq!(n.connector_text_position(), 0.5);
        assert_eq!(n.connector_text_padding(), 4.0);
    }

    #[test]
    fn richtext_vertical_align_defaults_top() {
        // nodes.ts:1063-1128 (scene-schema.md §4.9/§15).
        assert_eq!(
            node(json!({"id":"t","type":"richtext","htmlContent":"<p></p>"}))
                .richtext_vertical_align(),
            "top"
        );
    }

    #[test]
    fn blend_mode_defaults_normal() {
        // nodes.ts:7 BlendModeType (scene-schema.md §3.7/§15).
        assert_eq!(
            node(json!({"id":"a","type":"rectangle"})).blend_mode(),
            "normal"
        );
    }


    #[test]
    fn variable_stroke_defaults() {
        // Absent ⇒ None (plain stroke). nodes.ts VariableStrokeProperties.
        let plain = node(json!({"type": "path", "points": []}));
        assert!(plain.stroke_profile().is_none());
        assert!(plain.stroke_taper().is_none());
        assert!(plain.ink_channels().is_none());
        assert_eq!(plain.stroke_dynamic_seed(), 0);

        let p = node(json!({
            "type": "path",
            "strokes": [{
                "id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1,
                "taper": {"startLength": 20},
                "wave": {"amount": 0.2},
                "dynamic": {"wiggle": 3},
                "brush": {"brushId": "b1", "seed": 42},
                "profile": {"preset": "eye"}
            }]
        }));
        let taper = p.stroke_taper().unwrap();
        assert_eq!(taper.units, crate::stroke_model::TaperUnits::Percent);
        assert_eq!((taper.start_length, taper.end_length), (20.0, 0.0));
        let wave = p.stroke_wave().unwrap();
        assert_eq!(wave.units, crate::stroke_model::WaveUnits::Px);
        assert_eq!(wave.wavelength, 20.0);
        let dynamic = p.stroke_dynamic().unwrap();
        assert_eq!((dynamic.frequency, dynamic.smoothen), (1.0, 0.0));
        assert_eq!(p.stroke_dynamic_seed(), 42); // falls back to the brush seed
        assert_eq!(p.stroke_profile().unwrap().preset.as_deref(), Some("eye"));

        // Only paths carry these fields.
        let rect = node(json!({"type": "rectangle", "strokes": [{
            "id": "stroke-0", "paint": {"type": "solid", "color": "#000"}, "width": 1,
            "taper": {"startLength": 20}}]}));
        assert!(rect.stroke_taper().is_none());
    }
}
