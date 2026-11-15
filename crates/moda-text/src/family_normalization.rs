//! Conservative font-family name normalization — the Rust mirror of
//! `utils/fonts/familyNormalization.ts`.
//!
//! A family name that misses an exact catalog lookup is very often a name for a
//! family we DO host, spelled the way some importer wrote it: a weight encoded as
//! a suffix (`Open Sans Light` is `Open Sans` at 300), a PostScript name
//! (`Arial-BoldMT`, `OpenSans-Light` — PDF and PPTX imports carry these verbatim,
//! subset prefix and all), or a family Google Fonts has since renamed
//! (`Source Sans Pro` → `Source Sans 3`).
//!
//! ## The conservatism contract
//!
//! Candidates are only consulted AFTER an exact lookup fails, and a candidate is
//! only used if IT matches the catalog exactly — so a real family whose own name
//! ends in a weight word (`Archivo Black`) is never collapsed. Every rule is
//! name-preserving: a suffix that genuinely encodes a weight/style, or a documented
//! 1:1 rename. There is NO lookalike matching, so licensed families we do not host
//! (Avenir, Gotham) keep missing.
//!
//! `font_resolver.py` is no longer quite this verbatim: it also matches a family
//! whose spelling differs only in spacing or hyphens, so a PDF's camelCase-split
//! `Jet Brains Mono` finds the catalog's `JetBrains Mono`. That belongs to the
//! importer, which writes the CATALOG's spelling into the canvas — a stored node
//! never carries the fuzzy spelling, so this path never resolves one. Do not
//! mirror it here: at render time it would apply to names no importer vetted.
//!
//! The weight table mirrors `parse_font_family_weight`
//! (`backend/app/services/pptx/font_utils.py`) 1:1.
//!
//! **This file and its TypeScript twin MUST stay in step**: the WASM/native renderer
//! resolves families here while the editor UI and the export lane resolve them in
//! TS. A divergence means the canvas and the exported file render different fonts.

/// How faithful a candidate is to what the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateKind {
    /// Same typeface, different spelling (suffix, PostScript name, rename).
    Normalized,
    /// NOT the requested typeface — the request was unresolvable (theme token).
    Substituted,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FamilyCandidate {
    /// Family name to look up in the catalog.
    pub name: String,
    /// Weight the removed suffix encoded (`Open Sans Light` → 300).
    pub weight: Option<i64>,
    /// Style the removed suffix encoded (`"italic"`).
    pub style: Option<&'static str>,
    pub kind: CandidateKind,
}

/// `_WEIGHT_VARIANTS` (`backend/app/services/pptx/font_utils.py`), lowercased and
/// ordered longest-first so "extra bold" wins over "bold".
const WEIGHT_SUFFIXES: &[(&str, i64)] = &[
    ("extra light", 200),
    ("ultra light", 200),
    ("extralight", 200),
    ("ultralight", 200),
    ("extra bold", 800),
    ("ultra bold", 800),
    ("extrabold", 800),
    ("ultrabold", 800),
    ("semi bold", 600),
    ("demi bold", 600),
    ("hairline", 100),
    ("semibold", 600),
    ("demibold", 600),
    ("regular", 400),
    ("medium", 500),
    ("normal", 400),
    ("light", 300),
    ("black", 900),
    ("heavy", 900),
    ("thin", 100),
    ("book", 400),
    ("bold", 700),
];

const ITALIC_SUFFIXES: &[&str] = &["italic", "oblique"];

/// `RENAMED_FAMILY_ALIASES` (`familyNormalization.ts`). Sources are lowercase;
/// every target is a family the catalog actually hosts. These are RENAMES of the
/// same typeface, never lookalike substitutions.
const RENAMED_FAMILY_ALIASES: &[(&str, &str)] = &[
    ("source sans pro", "Source Sans 3"),
    ("source serif pro", "Source Serif 4"),
    ("muli", "Mulish"),
    ("andada", "Andada Pro"),
    ("droid sans", "Noto Sans"),
    ("droid serif", "Noto Serif"),
    ("droid sans mono", "Noto Sans Mono"),
];

/// Family used for an OOXML theme reference we cannot expand
/// (`THEME_TOKEN_FALLBACK_FAMILY`).
pub const THEME_TOKEN_FALLBACK_FAMILY: &str = "Inter";

fn alias_for(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    RENAMED_FAMILY_ALIASES
        .iter()
        .find(|(source, _)| *source == lower)
        .map(|(_, target)| *target)
}

/// `+mj-lt` / `+mn-lt` (and the `-ea` / `-cs` script variants) are raw PowerPoint
/// theme references that name no font, so a canvas carrying one as a node family
/// renders in Noto Sans. DEFENSIVE RENDERING ONLY, and still needed after the
/// importer fix (#8528): that only helps the NEXT import and cannot repair a canvas
/// that already stores the token. The original theme font is not recoverable
/// client-side (the source `.pptx` is not retained and `canvas_data` carries no
/// theme), so the honest resolution is the default family.
pub fn is_theme_font_token(raw: &str) -> bool {
    let lower = raw.trim().to_lowercase();
    let Some(rest) = lower.strip_prefix('+') else {
        return false;
    };
    let Some((major_minor, script)) = rest.split_once('-') else {
        return false;
    };
    matches!(major_minor, "mj" | "mn") && matches!(script, "lt" | "ea" | "cs")
}

/// `ABCDEF+Roboto` — the six-letter subset tag PDF embedders prepend.
/// `>= 7` (not `> 7`) so a bare `"ABCDEF+"` strips to empty exactly as the TS
/// regex does; a shorter/longer tag or a lowercase letter is not one.
fn strip_subset_prefix(name: &str) -> &str {
    let bytes = name.as_bytes();
    if bytes.len() >= 7 && bytes[6] == b'+' && bytes[..6].iter().all(|b| b.is_ascii_uppercase()) {
        &name[7..]
    } else {
        name
    }
}

/// Foundry tags on PostScript names, only ever stripped after a lowercase letter
/// (the TS `(?<=[a-z])(PSMT|MT|PS)$`).
fn strip_vendor_suffix(name: &str) -> &str {
    for tag in ["PSMT", "MT", "PS"] {
        if let Some(head) = name.strip_suffix(tag) {
            if head
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_lowercase())
            {
                return head;
            }
        }
    }
    name
}

/// `TimesNewRoman` → `Times New Roman`, `IBMPlexSans` → `IBM Plex Sans`.
///
/// Mirrors the two TS regexes: split lower/digit→Upper, then acronym→Titlecase.
/// The case tests are deliberately ASCII-ONLY, because the TS side uses the ASCII
/// character classes `([a-z0-9])([A-Z])` / `([A-Z]+)([A-Z][a-z])`. Using Unicode
/// `char::is_lowercase` here would split `"CaféSans"` in Rust but not in TS, and a
/// space-free family name with a non-ASCII letter would then resolve in the primary
/// renderer while missing in the editor and the export lane.
fn de_camel(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 && c.is_ascii_uppercase() {
            let prev = chars[i - 1];
            let split = (prev.is_ascii_lowercase() || prev.is_ascii_digit())
                || (prev.is_ascii_uppercase()
                    && chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase()));
            if split {
                out.push(' ');
            }
        }
        out.push(c);
    }
    out.trim().to_string()
}

#[derive(Debug, Clone, PartialEq)]
struct SuffixSplit {
    base: String,
    weight: Option<i64>,
    style: Option<&'static str>,
}

/// Strip `" <suffix>"` (ASCII, case-insensitive) off the end of `base`.
/// Byte-boundary safe: the needle is ASCII, so a non-ASCII tail simply declines.
fn strip_trailing_word(base: &str, suffix: &str) -> Option<String> {
    let needle_len = suffix.len() + 1;
    if base.len() <= needle_len {
        return None;
    }
    let split = base.len() - needle_len;
    if !base.is_char_boundary(split) {
        return None;
    }
    let (head, tail) = base.split_at(split);
    if !tail.starts_with(' ') || !tail[1..].eq_ignore_ascii_case(suffix) {
        return None;
    }
    Some(head.trim_end().to_string())
}

/// Peel a trailing italic marker and/or weight word off a space-separated name.
/// Handles either order and at most one of each (`Open Sans Light Italic`).
fn strip_style_suffixes(name: &str) -> SuffixSplit {
    let mut base = name.trim().to_string();
    let mut weight: Option<i64> = None;
    let mut italic = false;

    for _ in 0..2 {
        if !italic {
            if let Some(head) = ITALIC_SUFFIXES
                .iter()
                .find_map(|s| strip_trailing_word(&base, s))
            {
                base = head;
                italic = true;
                continue;
            }
        }
        if weight.is_none() {
            if let Some((head, w)) = WEIGHT_SUFFIXES
                .iter()
                .find_map(|(s, w)| strip_trailing_word(&base, s).map(|head| (head, *w)))
            {
                base = head;
                weight = Some(w);
                continue;
            }
        }
        break;
    }

    SuffixSplit {
        base,
        weight,
        style: if italic { Some("italic") } else { None },
    }
}

/// Parse the style half of a PostScript name (`BoldItalic`, `SemiBold`, `Regular`).
/// `None` means "not a style part at all" — never invent a family from it.
fn parse_style_words(raw: &str) -> Option<(Option<i64>, Option<&'static str>)> {
    let expanded = de_camel(strip_vendor_suffix(raw)).to_lowercase();
    let words: Vec<&str> = expanded.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    let italic = words.iter().any(|w| ITALIC_SUFFIXES.contains(w));
    let rest = words
        .iter()
        .filter(|w| !ITALIC_SUFFIXES.contains(*w))
        .copied()
        .collect::<Vec<_>>()
        .join(" ");

    let weight = if rest.is_empty() {
        None
    } else {
        match WEIGHT_SUFFIXES.iter().find(|(s, _)| *s == rest) {
            Some((_, w)) => Some(*w),
            // Unparseable leftovers (e.g. the "serif" of `sans-serif`) — decline.
            None => return None,
        }
    };
    if !italic && weight.is_none() {
        return None;
    }
    Some((weight, if italic { Some("italic") } else { None }))
}

/// Interpret a space-free name as a PostScript name: `Family[-Style]` plus foundry
/// tag. `None` when it reads as an ordinary name.
fn post_script_split(name: &str) -> Option<SuffixSplit> {
    if name.chars().any(char::is_whitespace) {
        return None;
    }
    let (family_part, style_part) = match name.split_once('-') {
        Some((f, s)) => (f, Some(s)),
        None => (name, None),
    };
    let family = de_camel(strip_vendor_suffix(family_part));
    if family.chars().count() < 2 {
        return None;
    }

    match style_part {
        None => {
            if family == name {
                None
            } else {
                Some(SuffixSplit {
                    base: family,
                    weight: None,
                    style: None,
                })
            }
        }
        // A hyphen whose right-hand side is not a style is part of the family name
        // (`sans-serif`, `system-ui`) — never split it.
        Some(style) => {
            parse_style_words(&style.replace('-', "")).map(|(weight, style)| SuffixSplit {
                base: family,
                weight,
                style,
            })
        }
    }
}

/// Ordered, conservative reinterpretations of a family name that missed the
/// catalog. `familyNameCandidates` (`familyNormalization.ts`).
pub fn family_name_candidates(raw: &str) -> Vec<FamilyCandidate> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if is_theme_font_token(trimmed) {
        return vec![FamilyCandidate {
            name: THEME_TOKEN_FALLBACK_FAMILY.to_string(),
            weight: None,
            style: None,
            kind: CandidateKind::Substituted,
        }];
    }

    let start = strip_subset_prefix(trimmed);
    let mut out: Vec<FamilyCandidate> = Vec::new();
    let mut seen: Vec<String> = vec![trimmed.to_lowercase()];

    let mut push = |name: &str, weight: Option<i64>, style: Option<&'static str>| {
        if name.chars().count() < 2 {
            return;
        }
        let lower = name.to_lowercase();
        let key = format!("{lower}|{}|{}", weight.unwrap_or(-1), style.unwrap_or(""));
        if seen.iter().any(|s| *s == lower || *s == key) {
            return;
        }
        seen.push(key);
        out.push(FamilyCandidate {
            name: name.to_string(),
            weight,
            style,
            kind: CandidateKind::Normalized,
        });
    };

    let mut push_with_alias = |name: &str, weight: Option<i64>, style: Option<&'static str>| {
        push(name, weight, style);
        if let Some(alias) = alias_for(name) {
            push(alias, weight, style);
        }
    };

    if let Some(alias) = alias_for(start) {
        push_with_alias(alias, None, None);
    }

    let spaced = strip_style_suffixes(start);
    if spaced.base != start {
        push_with_alias(&spaced.base, spaced.weight, spaced.style);
    }

    if let Some(ps) = post_script_split(start) {
        push_with_alias(&ps.base, ps.weight, ps.style);
        // `OpenSansLight` → `Open Sans Light` → `Open Sans` at 300.
        let ps_spaced = strip_style_suffixes(&ps.base);
        if ps_spaced.base != ps.base {
            push_with_alias(
                &ps_spaced.base,
                ps.weight.or(ps_spaced.weight),
                ps.style.or(ps_spaced.style),
            );
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(raw: &str) -> Vec<String> {
        family_name_candidates(raw)
            .into_iter()
            .map(|c| c.name)
            .collect()
    }

    fn candidate(raw: &str, expected: &str) -> Option<FamilyCandidate> {
        family_name_candidates(raw)
            .into_iter()
            .find(|c| c.name.eq_ignore_ascii_case(expected))
    }

    #[test]
    fn strips_weight_suffixes_and_keeps_the_weight() {
        // Mirrors the TS suite; the weight table is the backend's.
        for (raw, base, weight) in [
            ("Open Sans Light", "Open Sans", 300),
            ("Roboto Thin", "Roboto", 100),
            ("Roboto Extra Light", "Roboto", 200),
            ("Roboto Medium", "Roboto", 500),
            ("Playfair Display SemiBold", "Playfair Display", 600),
            ("Roboto Extra Bold", "Roboto", 800),
            ("Roboto Black", "Roboto", 900),
            ("roboto bold", "roboto", 700),
        ] {
            let c = candidate(raw, base).unwrap_or_else(|| panic!("{raw}: no candidate {base}"));
            assert_eq!(c.weight, Some(weight), "{raw}");
        }
    }

    #[test]
    fn carries_italic_alone_and_after_a_weight() {
        assert_eq!(
            candidate("Open Sans Italic", "Open Sans").unwrap().style,
            Some("italic")
        );
        let c = candidate("Open Sans Light Italic", "Open Sans").unwrap();
        assert_eq!((c.weight, c.style), (Some(300), Some("italic")));
    }

    #[test]
    fn parses_postscript_names() {
        assert_eq!(names("ArialMT"), vec!["Arial"]);
        assert_eq!(
            candidate("Arial-BoldMT", "Arial").unwrap().weight,
            Some(700)
        );
        assert!(candidate("TimesNewRomanPSMT", "Times New Roman").is_some());
        let c = candidate("TimesNewRomanPS-BoldItalicMT", "Times New Roman").unwrap();
        assert_eq!((c.weight, c.style), (Some(700), Some("italic")));
        let c = candidate("ABCDEF+Roboto-Bold", "Roboto").unwrap();
        assert_eq!(c.weight, Some(700));
        assert!(names("PlayfairDisplay").contains(&"Playfair Display".to_string()));
        assert!(names("IBMPlexSans").contains(&"IBM Plex Sans".to_string()));
    }

    #[test]
    fn applies_renames_including_to_a_weight_stripped_base() {
        assert!(candidate("Source Sans Pro", "Source Sans 3").is_some());
        assert_eq!(
            candidate("Source Sans Pro Light", "Source Sans 3")
                .unwrap()
                .weight,
            Some(300)
        );
        assert!(candidate("Muli", "Mulish").is_some());
        assert!(candidate("Droid Sans Mono", "Noto Sans Mono").is_some());
    }

    #[test]
    fn resolves_ooxml_theme_tokens_to_the_default_family() {
        for token in ["+mn-lt", "+mj-lt", "+mn-ea", "+MJ-CS"] {
            assert!(is_theme_font_token(token), "{token}");
            assert_eq!(
                family_name_candidates(token),
                vec![FamilyCandidate {
                    name: THEME_TOKEN_FALLBACK_FAMILY.to_string(),
                    weight: None,
                    style: None,
                    kind: CandidateKind::Substituted,
                }],
                "{token}"
            );
        }
        assert!(!is_theme_font_token("+xyz"));
        assert!(names("+xyz").is_empty());
    }

    #[test]
    fn licensed_families_we_do_not_host_keep_missing() {
        // Never invent a lookalike for a deliberately absent family.
        for family in ["Avenir", "Gotham", "Gotham Pro", "Helvetica Neue"] {
            assert!(names(family).is_empty(), "{family} → {:?}", names(family));
        }
        // A weight suffix is still stripped, but only ever to the SAME typeface.
        for family in ["Avenir Next Medium", "Helvetica Neue Light"] {
            for c in family_name_candidates(family) {
                assert!(
                    family.to_lowercase().starts_with(&c.name.to_lowercase()),
                    "{family} → {}",
                    c.name
                );
            }
        }
    }

    #[test]
    fn css_generics_and_degenerate_input_are_never_normalized() {
        for generic in ["sans-serif", "serif", "monospace", "system-ui"] {
            assert!(names(generic).is_empty(), "{generic}");
        }
        assert!(names("").is_empty());
        assert!(names("   ").is_empty());
        assert!(names("Bold").is_empty());
        assert!(names("MT").is_empty());
    }

    /// The TS `deCamel` uses ASCII character classes, so a non-ASCII lowercase
    /// letter must NOT trigger a split here either — otherwise a space-free family
    /// name resolves in the primary renderer and misses in the editor/export lane.
    #[test]
    fn de_camel_splits_on_ascii_only_matching_the_ts_regexes() {
        assert_eq!(de_camel("TimesNewRoman"), "Times New Roman");
        assert_eq!(de_camel("IBMPlexSans"), "IBM Plex Sans");
        assert_eq!(de_camel("CaféSans"), "CaféSans");
        assert_eq!(de_camel("ÉcoleSans"), "École Sans");
        assert!(names("CaféSans").is_empty());
    }

    #[test]
    fn subset_prefix_bound_matches_the_ts_regex() {
        assert_eq!(strip_subset_prefix("ABCDEF+Roboto"), "Roboto");
        assert_eq!(strip_subset_prefix("ABCDEF+"), "");
        assert_eq!(strip_subset_prefix("ABCDE+Roboto"), "ABCDE+Roboto");
        assert_eq!(strip_subset_prefix("abcdef+Roboto"), "abcdef+Roboto");
        assert!(names("ABCDEF+").is_empty());
    }

    #[test]
    fn never_returns_the_input_itself() {
        for raw in ["Roboto", "Open Sans Light", "ArialMT", "Source Sans Pro"] {
            assert!(
                !names(raw).iter().any(|n| n.eq_ignore_ascii_case(raw)),
                "{raw}"
            );
        }
    }
}
