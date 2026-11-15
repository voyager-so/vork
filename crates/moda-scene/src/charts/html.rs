//! HTML helpers for chart label nodes:
//!  - `escapeHtml` (utils/html/escape.ts)
//!  - `buildStyledHtml` / `buildStyleDeclarations`
//!    (lib/designAgent/toolHandlers/sharedTextHelpers.ts:75-197)
//!  - the legend label's hand-rolled style string (charts/legend.ts:149-154,
//!    which uses a DIFFERENT declaration order than buildStyleDeclarations).

use super::jsmath::js_num_to_string;

/// `escapeHtml`.
pub fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// `sanitizeStyleValue` (sharedTextHelpers.ts:91-93).
fn sanitize_style_value(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[derive(Default)]
pub struct TextStyleProps<'a> {
    pub font_size: Option<f64>,
    pub font_weight: Option<f64>,
    pub font_family: Option<&'a str>,
    pub text_align: Option<&'a str>,
    pub color: Option<&'a str>,
}

/// `buildStyleDeclarations` order: font-size, font-weight, font-family,
/// text-align, color (sharedTextHelpers.ts:101-139; the chart helpers never
/// set letter-spacing/line-height/font-style/text-decoration).
fn build_style_declarations(props: &TextStyleProps) -> Vec<String> {
    let mut decls = Vec::new();
    if let Some(fs) = props.font_size {
        decls.push(format!("font-size: {}px", js_num_to_string(fs)));
    }
    if let Some(fw) = props.font_weight {
        decls.push(format!("font-weight: {}", js_num_to_string(fw)));
    }
    if let Some(ff) = props.font_family {
        if !ff.is_empty() {
            let clean = sanitize_style_value(ff.trim_matches(|c| c == '"' || c == '\''));
            let quoted = if clean.contains(' ') {
                format!("&quot;{clean}&quot;")
            } else {
                clean
            };
            decls.push(format!("font-family: {quoted}"));
        }
    }
    if let Some(ta) = props.text_align {
        decls.push(format!("text-align: {ta}"));
    }
    if let Some(c) = props.color {
        if !c.is_empty() {
            decls.push(format!("color: {}", sanitize_style_value(c)));
        }
    }
    decls
}

/// RAW persisted numeric → style prop. A missing config field reads as NaN
/// through cfg.rs, but the TS side sees `undefined` and
/// `buildStyleDeclarations`' `!== undefined` gate DROPS the declaration —
/// `Some(NaN)` would print "font-size: NaNpx" where the TS emits nothing.
/// Use this ONLY where the TS counterpart passes the config value straight
/// through (axes/tick labels, titles, value labels, slice labels). A
/// COMPUTED size (scatter's quadrant `fontSize - 2`, pie's
/// `max(10, floor(centerSize·scale))`) must stay a plain `Some(...)`: the TS
/// computes NaN there, NaN passes its `!== undefined` gate, and both mirrors
/// print "NaNpx" in agreement. Pinned by the chart-partial-fontsize fixture;
/// identity for every finite config. (`legend_label_style` below is a third
/// case: its TS hand-rolls the string and really prints "undefinedpx" —
/// unrepresentable here, deliberately left divergent.)
pub fn raw_size(value: f64) -> Option<f64> {
    Some(value).filter(|v| v.is_finite())
}

/// `buildStyledHtml(props, content)` (sharedTextHelpers.ts:191-197).
/// `content` must already be escaped by the caller (matching the TS call
/// sites which pass `escapeHtml(...)` or pre-built anchor HTML).
pub fn build_styled_html(props: &TextStyleProps, content: &str) -> String {
    let decls = build_style_declarations(props);
    if !decls.is_empty() {
        format!("<p style=\"{}\">{}</p>", decls.join("; "), content)
    } else {
        format!("<p>{content}</p>")
    }
}

/// charts/legend.ts:149-154 — hand-rolled declaration order:
/// font-family, font-size, text-align: left, color. (Unsanitized in the TS;
/// values come from config strings.)
pub fn legend_label_style(font_family: &str, font_size: f64, legend_color: &str) -> String {
    format!(
        "font-family: {font_family}; font-size: {}px; text-align: left; color: {legend_color}",
        js_num_to_string(font_size)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn styled_html_matches_ts() {
        let props = TextStyleProps {
            font_size: Some(12.0),
            font_family: Some("Inter"),
            text_align: Some("center"),
            color: Some("#666666"),
            ..Default::default()
        };
        assert_eq!(
            build_styled_html(&props, "Q1"),
            "<p style=\"font-size: 12px; font-family: Inter; text-align: center; color: #666666\">Q1</p>"
        );
        let title = TextStyleProps {
            font_size: Some(16.0),
            font_weight: Some(600.0),
            font_family: Some("DM Sans"),
            text_align: Some("left"),
            color: Some("#1a1a1a"),
        };
        assert_eq!(
            build_styled_html(&title, "T"),
            "<p style=\"font-size: 16px; font-weight: 600; font-family: &quot;DM Sans&quot;; text-align: left; color: #1a1a1a\">T</p>"
        );
    }

    #[test]
    fn escape() {
        assert_eq!(
            escape_html("a<b> & 'c'\""),
            "a&lt;b&gt; &amp; &#39;c&#39;&quot;"
        );
    }
}
