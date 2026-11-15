//! Variables in the Rust lane: literals only.
//!
//! Variables v2 (docs/variables-contract.md, design-systems.md §4) keeps a cached
//! literal next to every binding, and the editor resolves bindings (modes,
//! aliases) before it feeds nodes to the engine. So the engine never resolves a
//! variable: every bound colour, number and string it reads is already a plain
//! value. The one thing left is CSS: a rich-text colour run is stored as
//! `var(--nf-variable-<id>, <literal>)`, and the text shaper does not evaluate
//! custom properties, so [`resolve_node`] rewrites each such reference to its
//! fallback literal (a reference without one, which only a malformed document
//! carries, becomes `#000000`). It covers rich-text `htmlContent`, a node's
//! `textContent` and table cells' `textContent`, and runs where the resolved
//! record is derived from the canonical node (admission, re-resolution,
//! re-expansion), so `moda-render` draws a saved document as the editor does.

use crate::types::RawNode;
use crate::value::Value;
use std::borrow::Cow;

const NEEDLE: &str = "--nf-variable-";
/// What a reference without a fallback renders as.
const NO_FALLBACK: &str = "#000000";

fn is_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ':'
}

/// Parse a `var(--nf-variable-<id>[, <fallback>])` starting at `s` (which
/// begins with `var(`). Returns the replacement text and the bytes consumed,
/// or `None` when `s` is not such a reference (it is then kept verbatim).
fn parse_reference(s: &str) -> Option<(&str, usize)> {
    let mut i = "var(".len();
    i += s[i..].len() - s[i..].trim_start().len();
    if !s[i..].starts_with(NEEDLE) {
        return None;
    }
    i += NEEDLE.len();
    let id_len = s[i..].find(|c: char| !is_id_char(c)).unwrap_or(s.len() - i);
    if id_len == 0 {
        return None;
    }
    i += id_len;
    i += s[i..].len() - s[i..].trim_start().len();
    match s[i..].chars().next()? {
        ')' => Some((NO_FALLBACK, i + 1)),
        ',' => {
            let start = i + 1;
            let mut depth = 0usize;
            for (off, c) in s[start..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' if depth == 0 => {
                        let fallback = s[start..start + off].trim();
                        let fallback = if fallback.is_empty() { NO_FALLBACK } else { fallback };
                        return Some((fallback, start + off + 1));
                    }
                    ')' => depth -= 1,
                    _ => {}
                }
            }
            None
        }
        _ => None,
    }
}

/// Replace every `var(--nf-variable-<id>, <fallback>)` in `html` with its
/// fallback (`#000000` when it has none). Borrows when there is nothing to do.
pub fn resolve_html_fallbacks(html: &str) -> Cow<'_, str> {
    if !html.contains(NEEDLE) {
        return Cow::Borrowed(html);
    }
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    let mut changed = false;
    while let Some(at) = rest.find("var(") {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        match parse_reference(tail) {
            Some((literal, used)) => {
                out.push_str(literal);
                rest = &tail[used..];
                changed = true;
            }
            None => {
                out.push_str("var(");
                rest = &tail["var(".len()..];
            }
        }
    }
    if !changed {
        return Cow::Borrowed(html);
    }
    out.push_str(rest);
    Cow::Owned(out)
}

/// Rewrite the string at `slot` in place when it holds a reference.
fn resolve_string(slot: &mut Value) -> bool {
    let Value::String(s) = slot else { return false };
    match resolve_html_fallbacks(s) {
        Cow::Borrowed(_) => false,
        Cow::Owned(o) => {
            *s = o;
            true
        }
    }
}

fn node_has_references(node: &RawNode) -> bool {
    let has = |v: Option<&Value>| v.and_then(Value::as_str).is_some_and(|s| s.contains(NEEDLE));
    if has(node.json_field("htmlContent")) || has(node.json_field("textContent")) {
        return true;
    }
    node.node_type() == "table"
        && node.json_field("cells").and_then(Value::as_array).is_some_and(|rows| {
            rows.iter().filter_map(Value::as_array).flatten().any(|c| has(c.get("textContent")))
        })
}

/// The node the engine draws: `node` with every CSS variable reference in its
/// HTML replaced by the reference's fallback literal. Nodes without one are
/// cloned untouched.
pub fn resolve_node(node: &RawNode) -> RawNode {
    let mut resolved = node.clone();
    if !node_has_references(node) {
        return resolved;
    }
    for key in ["htmlContent", "textContent"] {
        if let Some(slot) = resolved.get_mut(key) {
            resolve_string(slot);
        }
    }
    if resolved.node_type() == "table" {
        if let Some(Value::Array(rows)) = resolved.get_mut("cells") {
            for row in rows.iter_mut() {
                let Value::Array(cells) = row else { continue };
                for cell in cells.iter_mut() {
                    if let Some(slot) = cell.as_object_mut().and_then(|c| c.get_mut("textContent")) {
                        resolve_string(slot);
                    }
                }
            }
        }
    }
    resolved
}

/// [`resolve_node`] for a node that references a variable; `None` (no clone)
/// for the rest, which the engine draws as they are.
pub fn resolve_node_if_referenced(node: &RawNode) -> Option<RawNode> {
    node_has_references(node).then(|| resolve_node(node))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    #[test]
    fn references_become_their_fallbacks() {
        assert_eq!(
            resolve_html_fallbacks("<p style=\"color: var(--nf-variable-var_a, #ff006e)\">a</p>"),
            "<p style=\"color: #ff006e\">a</p>"
        );
        // Nested parentheses, whitespace, several references, and one without a fallback.
        assert_eq!(
            resolve_html_fallbacks(
                "color: var( --nf-variable-b ,  rgba(0, 0, 0, 0.5) ); x: var(--nf-variable-c); y: var(--nf-variable-d,)"
            ),
            "color: rgba(0, 0, 0, 0.5); x: #000000; y: #000000"
        );
    }

    #[test]
    fn everything_else_is_kept_verbatim() {
        for html in [
            "<p>plain</p>",
            "var(--other, #fff)",
            "var(--nf-variable-)",
            "var(--nf-variable-a, rgba(0,0,0",
            "--nf-variable-a outside var()",
        ] {
            assert!(matches!(resolve_html_fallbacks(html), Cow::Borrowed(_)), "{html}");
        }
        assert_eq!(
            resolve_html_fallbacks("var(--x) then var(--nf-variable-a, red)"),
            "var(--x) then red"
        );
    }

    #[test]
    fn rich_text_and_table_cells_resolve() {
        let text = RawNode::from_value(json!({
            "id": "t", "type": "richtext",
            "htmlContent": "<p style=\"color: var(--nf-variable-v, #123456)\">x</p>"
        }))
        .unwrap();
        assert_eq!(
            resolve_node(&text).str("htmlContent"),
            Some("<p style=\"color: #123456\">x</p>")
        );
        let table = RawNode::from_value(json!({
            "id": "tbl", "type": "table",
            "cells": [[{"id": "c", "textContent": "<p style=\"color: var(--nf-variable-v, #abcdef)\">c</p>"}, {"id": "d", "textContent": "<p>d</p>"}]]
        }))
        .unwrap();
        let out = resolve_node(&table);
        assert_eq!(out.json_field("cells").unwrap()[0][0]["textContent"], "<p style=\"color: #abcdef\">c</p>");
        assert_eq!(out.json_field("cells").unwrap()[0][1]["textContent"], "<p>d</p>");
        // No references: an untouched copy.
        let plain = RawNode::from_value(json!({"id": "r", "type": "rectangle"})).unwrap();
        assert_eq!(resolve_node(&plain), plain);
    }
}
