//! Canonical plain-text normalization for KineticText unit indexing.
//!
//! `unitTransforms` is a DENSE array addressed by **code-point index of the
//! node's canonical normalized plain text**, which this module defines (it
//! began as a port of the animation-v3 TS lane's `buildParsedText`, deleted with
//! v3; its JS whitespace and entity semantics are kept). The native vector renderer
//! (Slice B) maps each laid-out glyph → its source code point and
//! indexes into the SAME space.
//!
//! The native `parser.rs` deliberately does NOT do this: it builds structured
//! `Paragraph`/`InlineRun`s for layout and does not decode entities or collapse
//! whitespace. So the unit-index contract gets its own faithful normalizer here,
//! pinned by `fixtures/text-plain-parity/cases.json` (Rust test:
//! `crates/moda-text/tests/plain_text_parity.rs`).
//!
//! This is a hand port (NOT regex) so JS `\s` / `[^\S\n]` semantics are matched
//! exactly — the `regex` crate's Unicode `\s` includes code points (e.g. U+0085)
//! that JS `\s` does not, which would silently diverge. The steps below are
//! `buildParsedText`'s, 1:1.

/// JS `\s` (ECMAScript WhiteSpace ∪ LineTerminator). Used for the whitespace
/// collapse (minus `\n`) and `trim`, matching `buildParsedText` exactly.
fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'      // tab
        | '\u{000A}'    // line feed
        | '\u{000B}'    // vertical tab
        | '\u{000C}'    // form feed
        | '\u{000D}'    // carriage return
        | '\u{0020}'    // space
        | '\u{00A0}'    // no-break space
        | '\u{1680}'
        | '\u{2000}'
            ..='\u{200A}'
        | '\u{2028}'    // line separator
        | '\u{2029}'    // paragraph separator
        | '\u{202F}'
        | '\u{205F}'
        | '\u{3000}'
        | '\u{FEFF}' // zero-width no-break space (BOM)
    )
}

/// Closing block tags `buildParsedText` turns into a newline:
/// `(?:p|div|li|h[1-6]|ul|ol|blockquote)` (case-insensitive).
fn is_block_tag_name(name: &str) -> bool {
    matches!(
        name,
        "p" | "div" | "li" | "ul" | "ol" | "blockquote" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
    )
}

/// `parseTextContent(html).plain` for the native lane — the canonical plain text
/// `unitTransforms` is indexed by. See the module docs. Empty input → empty
/// string (matching `parseTextContent`'s `html.length === 0` early return and
/// the all-whitespace collapse).
pub fn parse_plain_text(html: &str) -> String {
    if html.is_empty() {
        return String::new();
    }
    let chars: Vec<char> = html.chars().collect();
    // Step 1–3: tag handling, applied as three SEQUENTIAL passes exactly like the
    // TS `.replace(...).replace(...).replace(...)` chain (the order matters: a
    // `<br>` is turned into `\n` before the generic strip can swallow it inside a
    // larger malformed tag).
    let after_br = replace_tag_matches(&chars, match_br, Some('\n'));
    let after_block = replace_tag_matches(&after_br, match_closing_block, Some('\n'));
    let stripped = replace_tag_matches(&after_block, match_generic_tag, None);
    // Step 4: decode HTML entities.
    let decoded = decode_entities(&stripped);
    // Steps 5–7: whitespace collapse + trim.
    normalize_whitespace(&decoded)
}

/// Number of code points in the canonical plain text — the length of the dense
/// `unitTransforms` index space (`Array.from(plain).length` in the TS lane).
pub fn plain_text_code_point_count(html: &str) -> usize {
    parse_plain_text(html).chars().count()
}

/// Try to match `<\s*br\s*/?>` (case-insensitive) starting at `i` (where
/// `chars[i] == '<'`). Returns the exclusive end index on success.
fn match_br(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    j = skip_ws(chars, j);
    j = expect_ci(chars, j, 'b')?;
    j = expect_ci(chars, j, 'r')?;
    j = skip_ws(chars, j);
    // optional `/`
    if chars.get(j) == Some(&'/') {
        j += 1;
    }
    if chars.get(j) == Some(&'>') {
        Some(j + 1)
    } else {
        None
    }
}

/// Try to match `<\s*/\s*(p|div|li|h[1-6]|ul|ol|blockquote)\s*>` (ci) at `i`.
fn match_closing_block(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    j = skip_ws(chars, j);
    if chars.get(j) != Some(&'/') {
        return None;
    }
    j += 1;
    j = skip_ws(chars, j);
    // Read the tag-name identifier (letters + digits), matching the regex's
    // whole-alternative behavior (reading the full run avoids matching `p` in
    // `pre`; membership + the required trailing `\s*>` reproduce the regex).
    let start = j;
    while let Some(&c) = chars.get(j) {
        if c.is_ascii_alphanumeric() {
            j += 1;
        } else {
            break;
        }
    }
    if j == start {
        return None;
    }
    let name: String = chars[start..j]
        .iter()
        .collect::<String>()
        .to_ascii_lowercase();
    if !is_block_tag_name(&name) {
        return None;
    }
    j = skip_ws(chars, j);
    if chars.get(j) == Some(&'>') {
        Some(j + 1)
    } else {
        None
    }
}

/// Try to match `<[^>]*>` at `i` — a generic tag (any chars up to the first
/// `>`). Returns None if there is no closing `>` (a lone `<` is left literal,
/// matching the TS regex).
fn match_generic_tag(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while let Some(&c) = chars.get(j) {
        if c == '>' {
            return Some(j + 1);
        }
        j += 1;
    }
    None
}

/// Walk `chars` left-to-right; at every `<`, try `matcher`. On a match, emit
/// `replacement` (or nothing if `None`) and skip the matched span; otherwise
/// copy the char through. Mirrors a global `String.replace(/<.../g, ...)`.
fn replace_tag_matches(
    chars: &[char],
    matcher: fn(&[char], usize) -> Option<usize>,
    replacement: Option<char>,
) -> Vec<char> {
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            if let Some(end) = matcher(chars, i) {
                if let Some(r) = replacement {
                    out.push(r);
                }
                i = end;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn skip_ws(chars: &[char], mut j: usize) -> usize {
    while let Some(&c) = chars.get(j) {
        // Tag-internal `\s` — ASCII whitespace in practice; use the JS set for
        // faithfulness.
        if is_js_whitespace(c) {
            j += 1;
        } else {
            break;
        }
    }
    j
}

fn expect_ci(chars: &[char], j: usize, lower: char) -> Option<usize> {
    match chars.get(j) {
        Some(&c) if c.to_ascii_lowercase() == lower => Some(j + 1),
        _ => None,
    }
}

/// Decode HTML entities exactly like `decodeEntities`
/// (the v3 TS lane's, kept): named `amp/lt/gt/quot/apos/nbsp`
/// (case-insensitive) and numeric `&#nn;` / `&#xnn;`, with NUL / UTF-16
/// surrogate / out-of-range code points mapped to U+FFFD. A token that doesn't
/// match the entity grammar (e.g. `&#1a;`, `&amp` with no `;`) is left literal.
///
/// CONTRACT: this named set is the v3 TS lane's `NAMED_ENTITIES`, kept as is.
/// It is the *canonical* `unitTransforms` index space, so
/// expanding it to the full HTML5 named set (`&copy;`, `&mdash;`, …) would BREAK
/// the cross-lane unit-index parity even though the LAYOUT/shaping path
/// (`parser.rs` via the `scraper`/html5ever DOM, which resolves all named
/// character references at parse time) decodes that larger set. Consequence of
/// the intentional canon-vs-layout gap: a glyph from an exotic named entity is
/// shaped from the layout-decoded code point (e.g. `©`) but is left literal here
/// (`&copy;`), so Slice B's glyph→unit mapping can't reconcile them and degrades
/// that span to identity (statically correct; kinetic transform silently
/// inert). Not reachable through the editor — `utils/html/escape.ts` only ever
/// emits the five basic entities, all covered here — only paste/import or
/// code-driver-built HTML with exotic entities. Pinned by the
/// `unknown-named-entity-*` cases in the shared `text-plain-parity` fixture.
fn decode_entities(input: &[char]) -> String {
    if !input.contains(&'&') {
        return input.iter().collect();
    }
    let mut out = String::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] == '&' {
            if let Some((decoded, end)) = match_entity(input, i) {
                out.push_str(&decoded);
                i = end;
                continue;
            }
        }
        out.push(input[i]);
        i += 1;
    }
    out
}

/// Match one `&...;` entity at `i`. Returns (decoded string, exclusive end).
fn match_entity(chars: &[char], i: usize) -> Option<(String, usize)> {
    // chars[i] == '&'
    let mut j = i + 1;
    if chars.get(j) == Some(&'#') {
        j += 1;
        let is_hex = matches!(chars.get(j), Some('x') | Some('X'));
        if is_hex {
            j += 1;
        }
        let digits_start = j;
        while let Some(&c) = chars.get(j) {
            let ok = if is_hex {
                c.is_ascii_hexdigit()
            } else {
                c.is_ascii_digit()
            };
            if ok {
                j += 1;
            } else {
                break;
            }
        }
        if j == digits_start {
            return None; // no digits → not a numeric entity
        }
        if chars.get(j) != Some(&';') {
            return None; // missing terminator → left literal
        }
        let body: String = chars[digits_start..j].iter().collect();
        let radix = if is_hex { 16 } else { 10 };
        // Faithful to JS `parseInt` (which accumulates in f64): a value past f64's
        // finite range becomes Infinity, and TS `decodeEntities` leaves a
        // non-finite ref LITERAL (`if (!Number.isFinite(code)) return match`)
        // rather than mapping it to U+FFFD. A finite value > 0x10FFFF (or 0 / a
        // surrogate) still maps to U+FFFD. Mirroring the Infinity→literal branch
        // keeps the canonical code-point count identical to TS for pathological
        // huge numeric entities (otherwise a 400-digit `&#9…9;` is 1 char `�` in
        // Rust but the literal string in TS, shifting every later unit index).
        let code = parse_js_int_f64(&body, radix);
        if !code.is_finite() {
            return None; // non-finite → left literal (TS `!Number.isFinite`)
        }
        let decoded = if code == 0.0
            || (55296.0..=57343.0).contains(&code) // 0xD800..=0xDFFF surrogates
            || code > 1_114_111.0
        // > 0x10FFFF
        {
            '\u{FFFD}'.to_string()
        } else {
            // In-range, non-surrogate → always a valid scalar value.
            char::from_u32(code as u32).map_or_else(|| '\u{FFFD}'.to_string(), |c| c.to_string())
        };
        return Some((decoded, j + 1));
    }
    // Named entity: [a-zA-Z][a-zA-Z0-9]* ;
    let name_start = j;
    if !matches!(chars.get(j), Some(c) if c.is_ascii_alphabetic()) {
        return None;
    }
    j += 1;
    while let Some(&c) = chars.get(j) {
        if c.is_ascii_alphanumeric() {
            j += 1;
        } else {
            break;
        }
    }
    if chars.get(j) != Some(&';') {
        return None;
    }
    let name: String = chars[name_start..j]
        .iter()
        .collect::<String>()
        .to_ascii_lowercase();
    let decoded = match name.as_str() {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => " ",
        _ => return None, // unknown named entity → left literal
    };
    Some((decoded.to_string(), j + 1))
}

/// Parse `digits` in `radix` like JS `parseInt` — accumulate in f64 so a value
/// past f64's finite range yields `f64::INFINITY` (JS numbers ARE f64). Every
/// code point of interest is < 2^21, where f64 represents integers exactly, so
/// this matches JS bit-for-bit across the entire valid range; the f64 path only
/// matters at the Infinity boundary, where the caller (like TS) leaves the entity
/// literal. Digits are guaranteed valid by the caller's grammar scan.
fn parse_js_int_f64(digits: &str, radix: u32) -> f64 {
    let mut value = 0.0f64;
    let r = radix as f64;
    for ch in digits.chars() {
        let d = ch.to_digit(radix).unwrap_or(0) as f64;
        value = value * r + d;
    }
    value
}

/// Steps 5–7 of `buildParsedText`: collapse non-newline whitespace runs to a
/// single space (`[^\S\n]+` → ` `), collapse space-padded newline runs to a
/// single newline (` *\n+ *` → `\n`), then trim.
fn normalize_whitespace(input: &str) -> String {
    // Step 5: collapse maximal runs of JS-whitespace-except-newline → single ' '.
    let mut collapsed: Vec<char> = Vec::with_capacity(input.len());
    let mut in_ws_run = false;
    for c in input.chars() {
        if c != '\n' && is_js_whitespace(c) {
            if !in_ws_run {
                collapsed.push(' ');
                in_ws_run = true;
            }
        } else {
            collapsed.push(c);
            in_ws_run = false;
        }
    }
    // Step 6: ` *\n+ *` → `\n`. After step 5, spaces are singletons, but the
    // matcher handles general runs for faithfulness.
    let mut out: Vec<char> = Vec::with_capacity(collapsed.len());
    let mut i = 0;
    while i < collapsed.len() {
        // Try to match ` *\n+ *` starting at i.
        let mut j = i;
        while collapsed.get(j) == Some(&' ') {
            j += 1;
        }
        if collapsed.get(j) == Some(&'\n') {
            while collapsed.get(j) == Some(&'\n') {
                j += 1;
            }
            while collapsed.get(j) == Some(&' ') {
                j += 1;
            }
            out.push('\n');
            i = j;
        } else {
            out.push(collapsed[i]);
            i += 1;
        }
    }
    // Step 7: trim (JS `\s` = the same whitespace set).
    let s: String = out.iter().collect();
    s.trim_matches(is_js_whitespace).to_string()
}

#[cfg(test)]
mod tests {
    use super::parse_plain_text;

    #[test]
    fn numeric_entity_overflow_to_infinity_is_left_literal() {
        // JS `parseInt` of a ref past f64's finite range is Infinity, and TS
        // `decodeEntities` leaves a non-finite ref LITERAL. A 400-digit ref is
        // unambiguously Infinity in any f64 accumulation, so canon must keep the
        // entity verbatim (its full code-point span), NOT collapse it to U+FFFD —
        // otherwise the canonical unit index shifts vs the TS lane.
        let huge = "9".repeat(400);
        let html = format!("<p>A&#{huge};B</p>");
        let expected = format!("A&#{huge};B");
        assert_eq!(parse_plain_text(&html), expected);
    }

    #[test]
    fn numeric_entity_finite_but_out_of_range_is_replacement() {
        // A finite value above 0x10FFFF still maps to U+FFFD (matches TS); only
        // the Infinity boundary is left literal.
        assert_eq!(parse_plain_text("<p>A&#9999999999;B</p>"), "A\u{FFFD}B");
    }
}
