//! Emoji segmentation — port of `utils/msdf/emoji.ts:40-61`.
//!
//! A cluster is either a keycap sequence or a run of ZWJ-joined elements, where
//! each element is a pictograph optionally trailed by a skin-tone modifier
//! (U+1F3FB–1F3FF) and/or a variation selector (emoji.ts:6). Skin-tone
//! modifiers and the keycap combiner (U+20E3) are NOT `Extended_Pictographic`,
//! so they are matched explicitly here — otherwise a cluster breaks mid-sequence
//! (e.g. 🕵🏻‍♀️ shatters into base face + stray text + a lone ♀).
//!
//! `Extended_Pictographic` membership is the vendored table in `ext_pict.rs`,
//! generated from the same Node runtime that produced the oracle dumps, so
//! membership is bit-identical to V8's `\p{…}`. Regional-indicator PAIRS
//! cluster greedily left-to-right (platform RI-pairing semantics) when their
//! waved-flag SVG ships (`flags.rs`, generated from the emoji pack and
//! mirrored to the CDN); an unshipped pair (e.g. 🇽🇽) and a lone regional
//! indicator stay text so they never request a 404 SVG. Tag-sequence
//! subdivision flags (🏴󠁧󠁢󠁳󠁣󠁴󠁿 = U+1F3F4 + a tag run terminated by U+E007F)
//! cluster the same way, gated on `flags.rs`'s `VALID_TAG_SEQUENCES`; a
//! sequence whose SVG doesn't ship degrades to the bare 🏴, whose SVG does.
//!
//! A LONE pictograph is gated the same way, against `flags.rs`'s
//! `SHIPPED_SINGLE_EMOJI`: the property spans far more codepoints than Noto
//! draws, and routing an undrawable one to the image lane renders blank space
//! (`ops.rs` drops the quad, layout keeps the advance). Uncovered ones demote to
//! text and reach the font + script-fallback chain instead. Gating on the
//! SHIPPED SET rather than on `Extended_Pictographic` also keeps the outcome
//! stable across Unicode versions — see Q-131.
//!
//! Codepoints that are `Emoji=Yes, Emoji_Presentation=No` default to TEXT and become
//! emoji only with VS16. Noto ships an SVG for ~120 of them, so
//! `ships_single_codepoint_emoji` is true and the demote-to-text hatch above cannot save
//! them: each leaves the text run, drops its node off the vector engine, and paints as
//! fixed Noto artwork ignoring the run's family, weight and fill. That once
//! reached a customer through TM, whose artwork is fixed grey.
//!
//! `TEXT_PRESENTATION_OVERRIDES` below is the answer, and it is HAND-CURATED rather than
//! derived from the Unicode property: that property also covers the heart, the
//! warning sign, the aeroplane, the sun and the scissors, which users type wanting the
//! picture. The line drawn instead — a MARK SET IN RUNNING TEXT (punctuation, list bullet,
//! arrow, check, card suit) is text; anything DEPICTING AN OBJECT stays emoji.
//!
//! Two tests hold it up: `text_presentation_overrides_ts_mirror_is_identical` pins the
//! hand-mirrored TS and Rust tables together, and
//! `every_text_presentation_override_resolves_without_declining_the_node`
//! (`vector::script_fallback`) proves each entry actually resolves once it is text, so
//! widening the table cannot silently trade a colour image for a whole-node decline.

use crate::ext_pict::EXT_PICT_RANGES;
use crate::flags::{is_valid_flag_pair, is_valid_tag_sequence, ships_single_codepoint_emoji};

/// The base of a tag-sequence subdivision flag (🏴 U+1F3F4).
const TAG_FLAG_BASE: char = '\u{1F3F4}';

/// Skin-tone modifiers (Emoji_Modifier, U+1F3FB–1F3FF) — not
/// Extended_Pictographic, so matched explicitly.
fn is_skin_tone_modifier(cp: u32) -> bool {
    (0x1F3FB..=0x1F3FF).contains(&cp)
}

fn is_variation_selector(c: char) -> bool {
    c == '\u{FE0F}' || c == '\u{FE0E}'
}

fn is_keycap_base(c: char) -> bool {
    matches!(c, '0'..='9' | '#' | '*')
}

/// Regional indicators (U+1F1E6–1F1FF) — not Extended_Pictographic; pairs are
/// matched explicitly (emoji.ts `RI_PAIR`).
fn is_regional_indicator(cp: u32) -> bool {
    (0x1F1E6..=0x1F1FF).contains(&cp)
}

/// Tag characters (U+E0020–E007F) — the payload of a tag-sequence subdivision
/// flag. Not Extended_Pictographic, so matched explicitly (emoji.ts
/// `TAG_SEQUENCE`).
fn is_tag_char(cp: u32) -> bool {
    (0xE0020..=0xE007F).contains(&cp)
}

/// End index (exclusive) of a well-formed tag run starting at `from`: one or
/// more U+E0020–E007E followed by the U+E007F terminator. `None` when there is
/// no run, or it is unterminated — an unterminated run is not part of any
/// cluster, so the base flag stands alone and the loose tag chars are stripped
/// by `push_text`.
fn tag_run_end(chars: &[char], from: usize) -> Option<usize> {
    let mut j = from;
    while j < chars.len() && (0xE0020..=0xE007E).contains(&(chars[j] as u32)) {
        j += 1;
    }
    if j > from && j < chars.len() && chars[j] as u32 == 0xE007F {
        Some(j + 1)
    } else {
        None
    }
}

/// Consume an element's trailers after its base pictograph: an optional
/// skin-tone modifier then an optional variation selector. Shared by the base
/// element and each ZWJ-joined element.
fn consume_element_trailers(chars: &[char], i: &mut usize) {
    if *i < chars.len() && is_skin_tone_modifier(chars[*i] as u32) {
        *i += 1;
    }
    if *i < chars.len() && is_variation_selector(chars[*i]) {
        *i += 1;
    }
}

pub fn is_extended_pictographic(cp: u32) -> bool {
    // binary search over the sorted, disjoint ranges
    let mut lo = 0usize;
    let mut hi = EXT_PICT_RANGES.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (a, b) = EXT_PICT_RANGES[mid];
        if cp < a {
            hi = mid;
        } else if cp > b {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    Text(String),
    Emoji(String),
}

/// `TEXT_PRESENTATION_OVERRIDES` (`utils/msdf/emoji.ts`) — hand-mirrored, and pinned to it
/// by `text_presentation_overrides_ts_mirror_is_identical`. Add a codepoint to BOTH: that
/// test fails on either side alone.
///
/// Unicode marks a large set `Emoji=Yes, Emoji_Presentation=No`: emoji-capable but TEXT by
/// default, becoming emoji only with VS16. Noto ships an SVG for ~120 of them, so
/// `ships_single_codepoint_emoji` is true and the demote-to-text hatch never fires — each
/// one leaves the text run, drops its node off the vector engine, and paints as fixed Noto
/// artwork ignoring the run's family, weight and fill. That once reached a
/// customer through TM, whose artwork is fixed grey.
///
/// HAND-CURATED, not the Unicode property: that set also holds the heart, the
/// warning sign, the aeroplane, the sun and the scissors — pictographs users type wanting
/// the picture. The line we draw instead: a MARK SET IN RUNNING TEXT (punctuation, a list
/// bullet, an arrow, a check) belongs here; anything DEPICTING AN OBJECT stays emoji.
///
/// U+203C and U+2049 are MEASURED OUT: they classify `Unknown`, so they get an empty
/// fallback chain, and a font without the glyph would decline the whole node to Canvas2D.
/// U+3030 is left out for lack of demonstrated benefit: adding and removing it produced
/// byte-identical renders.
///
/// Sorted — `is_text_presentation_override` binary-searches it, and the test asserts the
/// ordering so a careless insert cannot silently go unfound.
pub(crate) const TEXT_PRESENTATION_OVERRIDES: &[u32] = &[
    0x00A9, // (c) COPYRIGHT SIGN
    0x00AE, // (R) REGISTERED SIGN
    0x2122, // TM TRADE MARK SIGN            — the customer case
    0x2194, // <-> LEFT RIGHT ARROW
    0x2195, // UP DOWN ARROW
    0x2196, // NORTH WEST ARROW
    0x2197, // NORTH EAST ARROW
    0x2198, // SOUTH EAST ARROW
    0x2199, // SOUTH WEST ARROW
    0x21A9, // LEFTWARDS ARROW WITH HOOK
    0x21AA, // RIGHTWARDS ARROW WITH HOOK
    0x25AA, // BLACK SMALL SQUARE            — list bullet
    0x25AB, // WHITE SMALL SQUARE            — list bullet
    0x25B6, // BLACK RIGHT-POINTING TRIANGLE — list marker
    0x25C0, // BLACK LEFT-POINTING TRIANGLE  — list marker
    0x25FB, // WHITE MEDIUM SQUARE
    0x25FC, // BLACK MEDIUM SQUARE
    0x2611, // BALLOT BOX WITH CHECK         — checklist
    0x2660, // BLACK SPADE SUIT
    0x2663, // BLACK CLUB SUIT
    0x2665, // BLACK HEART SUIT
    0x2666, // BLACK DIAMOND SUIT
    0x2714, // HEAVY CHECK MARK              — the "done" bullet
    0x2716, // HEAVY MULTIPLICATION X
    0x2733, // EIGHT SPOKED ASTERISK         — footnote mark
    0x2734, // EIGHT POINTED BLACK STAR      — footnote mark
    0x27A1, // BLACK RIGHTWARDS ARROW
    0x2934, // ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS
    0x2935, // ARROW POINTING RIGHTWARDS THEN CURVING DOWNWARDS
    0x2B05, // LEFTWARDS BLACK ARROW
    0x2B06, // UPWARDS BLACK ARROW
    0x2B07, // DOWNWARDS BLACK ARROW
];

/// True when `cp` defaults to TEXT presentation and so needs VS16 to be an emoji.
fn is_text_presentation_override(cp: u32) -> bool {
    TEXT_PRESENTATION_OVERRIDES.binary_search(&cp).is_ok()
}

/// `isEmojiAfterOverrides` (`utils/msdf/emoji.ts`).
fn is_emoji_after_overrides(cluster: &str) -> bool {
    let mut cps = cluster.chars().map(|c| c as u32);
    let Some(base) = cps.next() else {
        return false;
    };
    let has_vs16 = cluster.chars().any(|c| c == '\u{FE0F}');
    if is_text_presentation_override(base) {
        return has_vs16;
    }
    let mut significant = cluster.chars().filter(|c| !is_variation_selector(*c));
    let lone = significant.next().filter(|_| significant.next().is_none());
    // A lone pictograph with VS15 (U+FE0E) and no VS16 asks for TEXT presentation, as
    // Unicode defines the selector: it goes to the font + script-fallback chain, where
    // `split_text_into_segments` drops its selector. On the image lane the selector
    // reached the colour face, which maps no glyph for it: the cluster drew nothing and
    // failed export (✈ + VS15). Mirrors `isEmojiAfterOverrides` (emoji.ts).
    if lone.is_some() && !has_vs16 && cluster.chars().any(|c| c == '\u{FE0E}') {
        return false;
    }
    // A LONE pictograph whose SVG doesn't ship would draw NOTHING: the planning
    // site (`ops.rs`) drops the emoji quad on an `emoji_rgba` miss while layout
    // keeps its ~1em box, so the glyph becomes blank space — and the browser lane
    // bakes the same transparent hole. Demote it to text instead, so it flows
    // through the font + script-fallback chain like any other symbol. Exactly the
    // rule an unshipped flag pair already follows, one codepoint wider.
    //
    // `Extended_Pictographic` covers far more than Noto draws: ~1.4k codepoints
    // ship a single-codepoint SVG, and the property's non-emoji pictographs (⚑
    // U+2691, ★ U+2605, ♪ U+266A, the chess pieces, the dice, the mahjong tiles)
    // have none. Gating on the SHIPPED SET rather than the property also keeps
    // this independent of the runtime's Unicode version, which is what actually
    // decides which of those the segmenter even offers here.
    //
    // Single-codepoint clusters ONLY (variation selectors ride along — they are
    // stripped from the Noto filename). A keycap, skin-toned, flag or ZWJ cluster
    // resolves to a MULTI-codepoint filename this table does not describe, and its
    // parts cannot be reshaped as text anyway.
    match lone {
        Some(only) => ships_single_codepoint_emoji(only as u32),
        None => true,
    }
}

/// `splitTextIntoSegments` (emoji.ts:40-61): scan for regex matches; each
/// match becomes an emoji segment unless the ©/®/™ override demotes it back to
/// text. Non-matching spans become text segments.
pub fn split_text_into_segments(input: &str) -> Vec<Segment> {
    split_text_into_segments_with_ranges(input).into_iter().map(|(_, segment)| segment).collect()
}

/// As above, retaining source code-point ranges even where text controls are stripped.
/// Run mapping uses these ranges to apply styles after whole-paragraph classification.
pub(crate) fn split_text_into_segments_with_ranges(input: &str) -> Vec<(std::ops::Range<usize>, Segment)> {
    if input.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = input.chars().collect();
    let mut out = Vec::new();
    let mut last = 0usize; // char index of the end of the previous match
    let mut i = 0usize;
    // A stray ZWJ or tag char (one not consumed into an emoji cluster) must not
    // survive in a TEXT segment. U+200D doubles as the emoji-run placeholder
    // sentinel downstream (`runs::EMOJI_PLACEHOLDER`), so a literal ZWJ left in
    // text would be miscounted as an emoji at layout time; tag chars are invisible
    // format characters everywhere but are not treated as control/whitespace, so a
    // survivor is shaped as notdef tofu. An emoji presentation selector (VS15
    // U+FE0E, VS16 U+FE0F) left in text has done its job (it chose text over the
    // image lane, as in `✔` + VS15), and few faces map it: kept, it failed the coverage
    // probe and declined the whole node, whose flags then drew as boxed letters.
    // Strip all of them from text spans only — inside `Segment::Emoji` values the
    // ZWJ is the join char, the tag run selects the subdivision flag and the
    // selectors ride along, so there they must survive.
    let push_text = |out: &mut Vec<_>, range: std::ops::Range<usize>| {
        if let Some(segment) = text_segment(&chars[range.clone()]) {
            out.push((range, segment));
        }
    };
    while i < chars.len() {
        // Keycap: [0-9#*] VS16? U+20E3. Checked before the pictograph test so a
        // bare digit stays text and only clusters when the combiner follows.
        if is_keycap_base(chars[i]) {
            let mut j = i + 1;
            if j < chars.len() && chars[j] == '\u{FE0F}' {
                j += 1;
            }
            if j < chars.len() && chars[j] == '\u{20E3}' {
                let start = i;
                i = j + 1;
                if start > last {
                    push_text(&mut out, last..start);
                }
                out.push((start..i, Segment::Emoji(chars[start..i].iter().collect())));
                last = i;
                continue;
            }
        }
        // Regional-indicator pair (emoji.ts `RI_PAIR` alternative): greedy
        // left-to-right pairing. A pair with a shipped waved-flag SVG becomes
        // an emoji cluster; an unshipped pair is consumed but demoted to text
        // (mirrors the regex, which matches ANY pair before the validity
        // gate). A lone RI falls through as plain text below.
        if is_regional_indicator(chars[i] as u32) {
            if i + 1 < chars.len() && is_regional_indicator(chars[i + 1] as u32) {
                let start = i;
                i += 2;
                if start > last {
                    push_text(&mut out, last..start);
                }
                if is_valid_flag_pair(chars[start] as u32, chars[start + 1] as u32) {
                    out.push((start..i, Segment::Emoji(chars[start..i].iter().collect())));
                } else {
                    push_text(&mut out, start..i);
                }
                last = i;
            } else {
                i += 1;
            }
            continue;
        }
        if !is_extended_pictographic(chars[i] as u32) {
            i += 1;
            continue;
        }
        // match start
        let start = i;
        i += 1;
        // Tag-sequence subdivision flag (🏴󠁧󠁢󠁳󠁣󠁴󠁿 = U+1F3F4 + a terminated tag run;
        // emoji.ts `TAG_SEQUENCE`). Tag chars are not Extended_Pictographic, so
        // without this the cluster stopped at the base and the run leaked into the
        // next TEXT segment. Keyed off a well-formed run rather than the base alone,
        // so 🏴 and 🏴‍☠️ — both with their own shipped SVG — fall through to the
        // ordinary trailer/ZWJ walk below.
        if chars[start] == TAG_FLAG_BASE {
            if let Some(run_end) = tag_run_end(&chars, i) {
                let cluster: String = chars[start..run_end].iter().collect();
                if start > last {
                    push_text(&mut out, last..start);
                }
                // Only GB-ENG/SCT/WLS ship. For anything else Unicode prescribes
                // drawing the base flag and ignoring the tags — and 🏴 has its own
                // SVG, so it stays on the image lane instead of being stranded in
                // text (the unshipped-RI-pair rule) where the native host carries no
                // color-emoji face.
                out.push((start..run_end, Segment::Emoji(if is_valid_tag_sequence(&cluster) {
                    cluster
                } else {
                    TAG_FLAG_BASE.to_string()
                })));
                i = run_end;
                last = i;
                continue;
            }
        }
        // base element trailers: (?:[🏻-🏿])?(?:️|︎)?
        consume_element_trailers(&chars, &mut i);
        // (?:‍ ELEMENT)* — each ZWJ-joined element may carry its own modifier/VS
        while i + 1 < chars.len()
            && chars[i] == '\u{200D}'
            && is_extended_pictographic(chars[i + 1] as u32)
        {
            i += 2;
            consume_element_trailers(&chars, &mut i);
        }
        let m: String = chars[start..i].iter().collect();
        if start > last {
            push_text(&mut out, last..start);
        }
        if is_emoji_after_overrides(&m) {
            out.push((start..i, Segment::Emoji(m)));
        } else {
            push_text(&mut out, start..i);
        }
        last = i;
    }
    if last < chars.len() {
        push_text(&mut out, last..chars.len());
    }
    out
}

/// Text controls have already affected classification; they have no drawable glyph.
pub(crate) fn is_text_character(c: char) -> bool {
    c != '\u{200D}' && !is_tag_char(c as u32) && !is_variation_selector(c)
}

pub(crate) fn text_segment(span: &[char]) -> Option<Segment> {
    let text: String = span.iter().copied().filter(|&c| is_text_character(c)).collect();
    (!text.is_empty()).then_some(Segment::Text(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copyright_needs_vs16() {
        // `TEXT_PRESENTATION_OVERRIDES` (`utils/msdf/emoji.ts`).
        assert_eq!(
            split_text_into_segments("©"),
            vec![Segment::Text("©".to_string())]
        );
        assert_eq!(
            split_text_into_segments("©\u{FE0F}"),
            vec![Segment::Emoji("©\u{FE0F}".to_string())]
        );
    }

    #[test]
    fn text_presentation_overrides_is_sorted() {
        // `is_text_presentation_override` binary-searches the table, so an
        // out-of-order insert would simply never be found — a silent no-op that
        // reintroduces the mis-render for that codepoint.
        let mut sorted = TEXT_PRESENTATION_OVERRIDES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted,
            TEXT_PRESENTATION_OVERRIDES.to_vec(),
            "TEXT_PRESENTATION_OVERRIDES must stay sorted and duplicate-free"
        );
    }

    #[test]
    #[ignore = "needs unported Moda fixture packages/editor/src/utils/msdf/emoji.ts"]
    fn text_presentation_overrides_ts_mirror_is_identical() {
        // This table is HAND-mirrored from emoji.ts — unlike the
        // generated flag tables, nothing regenerates it. Drift means the browser
        // and an export of the same canvas disagree about whether a codepoint is
        // emoji, so one of them paints Noto artwork where the other paints a font
        // glyph. The 4.1b back-pointer in emoji.ts points here; this is the check
        // that makes the pointer mean something.
        let ts_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .join("packages/editor/src/utils/msdf/emoji.ts");
        let src =
            std::fs::read_to_string(&ts_path).unwrap_or_else(|e| panic!("read {ts_path:?}: {e}"));
        let block = src
            .split_once("const TEXT_PRESENTATION_OVERRIDES")
            .expect("TEXT_PRESENTATION_OVERRIDES not found in emoji.ts")
            .1
            .split_once("]);")
            .expect("unterminated TEXT_PRESENTATION_OVERRIDES literal")
            .0;
        let mut ts: Vec<u32> = Vec::new();
        for line in block.lines() {
            // Entry rows are `  0xNNNN, // NAME`. Prose in the surrounding comments
            // cites codepoints as `U+NNNN`, never `0x`, so it cannot parse as an entry.
            let Some(rest) = line.trim().strip_prefix("0x") else {
                continue;
            };
            let Some((hex, _)) = rest.split_once(',') else {
                continue;
            };
            if let Ok(cp) = u32::from_str_radix(hex.trim(), 16) {
                ts.push(cp);
            }
        }
        assert!(
            ts.len() >= 30,
            "parsed only {} entries from emoji.ts — parser out of step with the table?",
            ts.len()
        );
        // Compare as SETS, not sequences. The TS side is a `Set`, where order carries
        // no meaning, and it is grouped semantically (legal marks, punctuation, arrows,
        // bullets, checks, footnote marks, card suits) because that is what makes it
        // reviewable. The Rust side must be SORTED because `is_text_presentation_override`
        // binary-searches it — an invariant `text_presentation_overrides_is_sorted` owns.
        // Comparing sequences here would force one of those two properties to give way.
        let mut ts_sorted = ts.clone();
        ts_sorted.sort_unstable();
        let mut rs_sorted = TEXT_PRESENTATION_OVERRIDES.to_vec();
        rs_sorted.sort_unstable();
        assert_eq!(
            ts_sorted, rs_sorted,
            "utils/msdf/emoji.ts TEXT_PRESENTATION_OVERRIDES drifted from the Rust mirror \
             in emoji.rs — the two are hand-kept and must be edited together"
        );
        assert_eq!(
            ts.len(),
            ts_sorted
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "emoji.ts TEXT_PRESENTATION_OVERRIDES has a duplicate entry"
        );
    }

    #[test]
    fn typographic_marks_are_text_and_pictographs_stay_emoji() {
        // Every override is a MARK, and must set in the run's own font…
        for &cp in TEXT_PRESENTATION_OVERRIDES {
            let ch = char::from_u32(cp).expect("valid scalar");
            let bare = ch.to_string();
            assert_eq!(
                split_text_into_segments(&bare),
                vec![Segment::Text(bare.clone())],
                "U+{cp:04X} must be TEXT on its own"
            );
            // …while VS16 still opts explicitly into the colour emoji.
            let with_vs16 = format!("{ch}\u{FE0F}");
            assert_eq!(
                split_text_into_segments(&with_vs16),
                vec![Segment::Emoji(with_vs16.clone())],
                "U+{cp:04X} with VS16 must still be EMOJI"
            );
        }

        // …and the pictographs deliberately left OUT stay on the emoji lane. This is
        // the half that stops a future "just derive it from Emoji_Presentation=No"
        // change: that property would sweep all of these in and turn a heart, a
        // warning sign and an aeroplane into monochrome font glyphs.
        for cp in [
            0x2764_u32, // heart
            0x26A0,     // warning sign
            0x2708,     // aeroplane
            0x2600,     // sun
            0x2702,     // scissors
            0x2603,     // snowman
            0x2709,     // envelope
            0x2699,     // gear
            0x2139,     // information source — deliberately excluded, see emoji.ts
            0x2744,     // snowflake — deliberately excluded
        ] {
            let ch = char::from_u32(cp).expect("valid scalar");
            let bare = ch.to_string();
            assert_eq!(
                split_text_into_segments(&bare),
                vec![Segment::Emoji(bare.clone())],
                "U+{cp:04X} is a pictograph and must stay EMOJI"
            );
        }
    }

    #[test]
    fn trademark_needs_vs16() {
        // U+2122 is `Emoji=Yes, Emoji_Presentation=No` like © / ®, and Noto
        // ships emoji_u2122.svg, so the SHIPPED_SINGLE_EMOJI gate cannot demote it —
        // without the override the sign leaves the text run and paints as fixed grey
        // artwork that ignores the run's family, weight and fill colour.
        assert!(ships_single_codepoint_emoji(0x2122));

        // The claim is equivalence with ®: whatever shape the working override produces,
        // ™ must produce it too. Both split into ADJACENT TEXT segments rather than one
        // merged run — the pre-existing shape for © / ®, and harmless: only an `Emoji`
        // segment forces the engine swap.
        let kinds = |s: &str| {
            split_text_into_segments(s)
                .into_iter()
                .map(|seg| matches!(seg, Segment::Emoji(_)))
                .collect::<Vec<_>>()
        };
        assert_eq!(kinds("Instinct™ MI400"), kinds("Instinct® MI400"));
        assert_eq!(kinds("™"), kinds("®"));
        assert!(kinds("Instinct™ MI400").iter().all(|is_emoji| !is_emoji));

        // ...and the text survives the round trip intact.
        let joined: String = split_text_into_segments("Instinct™ MI400")
            .into_iter()
            .map(|seg| match seg {
                Segment::Text(t) | Segment::Emoji(t) => t,
            })
            .collect();
        assert_eq!(joined, "Instinct™ MI400");

        // The colour emoji stays reachable on explicit request, as it does for © / ®.
        assert_eq!(
            split_text_into_segments("™\u{FE0F}"),
            vec![Segment::Emoji("™\u{FE0F}".to_string())]
        );
    }

    #[test]
    fn skin_tone_modifier_joins_base() {
        // A skin-tone modifier attaches to its base pictograph as one cluster.
        assert_eq!(
            split_text_into_segments("👋🏽"),
            vec![Segment::Emoji("👋🏽".to_string())]
        );
    }

    #[test]
    fn plain_zwj_sequence_stays_one_cluster() {
        assert_eq!(
            split_text_into_segments("👨\u{200D}👩\u{200D}👧"),
            vec![Segment::Emoji("👨\u{200D}👩\u{200D}👧".to_string())]
        );
    }

    #[test]
    fn skin_toned_zwj_profession_is_one_cluster() {
        // 🕵🏻‍♀️ = U+1F575 U+1F3FB U+200D U+2640 U+FE0F — the per-element
        // modifier (skin tone on the base) + trailing VS on the ZWJ element must
        // all stay in one cluster.
        let detective = "\u{1F575}\u{1F3FB}\u{200D}\u{2640}\u{FE0F}";
        assert_eq!(
            split_text_into_segments(detective),
            vec![Segment::Emoji(detective.to_string())]
        );
        // 🧑🏽‍💻 skin-toned technologist.
        let technologist = "\u{1F9D1}\u{1F3FD}\u{200D}\u{1F4BB}";
        assert_eq!(
            split_text_into_segments(technologist),
            vec![Segment::Emoji(technologist.to_string())]
        );
    }

    #[test]
    fn profession_zwj_is_one_cluster() {
        // 👩‍🔧 = U+1F469 U+200D U+1F527 (woman + ZWJ + wrench) — a
        // pictograph-ZWJ-pictograph profession sequence must stay one cluster so
        // the combined Noto SVG is looked up, not woman + wrench separately.
        let mechanic = "\u{1F469}\u{200D}\u{1F527}";
        assert_eq!(
            split_text_into_segments(mechanic),
            vec![Segment::Emoji(mechanic.to_string())]
        );
    }

    #[test]
    fn multi_person_skin_tone_kiss_is_one_cluster() {
        // 👩🏽‍❤️‍💋‍👩🏼 = U+1F469 U+1F3FD U+200D U+2764 U+FE0F U+200D U+1F48B
        // U+200D U+1F469 U+1F3FC — the longest common ZWJ sequence: two
        // skin-toned people joined through a VS16-carrying heart and a kiss mark.
        // Every per-element skin-tone modifier and VS16 must stay inside the one
        // cluster.
        let kiss =
            "\u{1F469}\u{1F3FD}\u{200D}\u{2764}\u{FE0F}\u{200D}\u{1F48B}\u{200D}\u{1F469}\u{1F3FC}";
        assert_eq!(
            split_text_into_segments(kiss),
            vec![Segment::Emoji(kiss.to_string())]
        );
    }

    #[test]
    fn stray_zwj_in_text_is_stripped() {
        // A ZWJ not consumed into an emoji cluster must NOT survive in a text
        // segment — U+200D doubles as the downstream emoji placeholder sentinel,
        // so a literal one would be miscounted as an emoji at layout time.
        assert_eq!(
            split_text_into_segments("a\u{200D}b"),
            vec![Segment::Text("ab".to_string())]
        );
    }

    #[test]
    fn presentation_selector_in_text_is_stripped() {
        // `✔` + VS15 is text; its selector must not reach the text run, where no
        // face maps it and it declined the whole node. A flag beside it stays emoji.
        let flag = "\u{1F1F0}\u{1F1EA}";
        assert_eq!(
            split_text_into_segments(&format!("ok \u{2714}\u{FE0E} {flag}")),
            vec![
                Segment::Text("ok ".to_string()),
                Segment::Text("\u{2714}".to_string()),
                Segment::Text(" ".to_string()),
                Segment::Emoji(flag.to_string()),
            ]
        );
        // A text-default codepoint with VS16 stays one emoji cluster, selector kept.
        assert_eq!(
            split_text_into_segments("\u{2764}\u{FE0F}"),
            vec![Segment::Emoji("\u{2764}\u{FE0F}".to_string())]
        );
        assert_eq!(
            split_text_into_segments("a\u{FE0F}b\u{FE0E}"),
            vec![Segment::Text("ab".to_string())]
        );
    }

    #[test]
    fn lone_pictograph_with_vs15_is_text() {
        // ✈ + VS15 asks for text presentation. On the image lane the colour face mapped no
        // glyph for the selector, so the cluster drew nothing and failed export. Mirrors
        // emojiPresentationSelector.test.ts.
        assert_eq!(
            split_text_into_segments("Airport \u{2708}\u{FE0E}"),
            vec![
                Segment::Text("Airport ".to_string()),
                Segment::Text("\u{2708}".to_string()),
            ]
        );
        assert_eq!(
            split_text_into_segments("\u{2708}\u{FE0F}"),
            vec![Segment::Emoji("\u{2708}\u{FE0F}".to_string())]
        );
        assert_eq!(
            split_text_into_segments("\u{2708}"),
            vec![Segment::Emoji("\u{2708}".to_string())]
        );
    }

    #[test]
    fn zwj_cluster_between_text_keeps_its_joiner() {
        // Text/emoji/text: the surrounding text is plain, the emoji segment keeps
        // its ZWJ so the combined Noto SVG is looked up.
        let mechanic = "\u{1F469}\u{200D}\u{1F527}";
        assert_eq!(
            split_text_into_segments(&format!("x{mechanic}y")),
            vec![
                Segment::Text("x".to_string()),
                Segment::Emoji(mechanic.to_string()),
                Segment::Text("y".to_string()),
            ]
        );
    }

    #[test]
    fn keycap_clusters_only_with_combiner() {
        // 1️⃣ = '1' U+FE0F U+20E3 → one emoji cluster.
        assert_eq!(
            split_text_into_segments("1\u{FE0F}\u{20E3}"),
            vec![Segment::Emoji("1\u{FE0F}\u{20E3}".to_string())]
        );
        // A bare digit stays text.
        assert_eq!(
            split_text_into_segments("5"),
            vec![Segment::Text("5".to_string())]
        );
    }

    #[test]
    fn valid_flag_pairs_cluster_as_emoji() {
        // 🇺🇸 — the waved-flag SVG ships (pack + CDN), so the RI pair is one
        // emoji cluster routed to the image pipeline.
        let us = "\u{1F1FA}\u{1F1F8}";
        assert_eq!(
            split_text_into_segments(us),
            vec![Segment::Emoji(us.to_string())]
        );
        // Greedy left-to-right pairing: 🇺🇸🇯🇵 → two flags.
        let jp = "\u{1F1EF}\u{1F1F5}";
        assert_eq!(
            split_text_into_segments(&format!("{us}{jp}")),
            vec![
                Segment::Emoji(us.to_string()),
                Segment::Emoji(jp.to_string())
            ]
        );
        // Mixed with text.
        assert_eq!(
            split_text_into_segments(&format!("go {us}!")),
            vec![
                Segment::Text("go ".to_string()),
                Segment::Emoji(us.to_string()),
                Segment::Text("!".to_string()),
            ]
        );
    }

    #[test]
    fn invalid_pairs_and_lone_regional_indicators_stay_text() {
        // 🇽🇽 has no shipped flag SVG — the pair is consumed (greedy) but
        // demoted to text, rendering via the fallback text lane instead of a
        // transparent hole.
        let xx = "\u{1F1FD}\u{1F1FD}";
        assert_eq!(
            split_text_into_segments(xx),
            vec![Segment::Text(xx.to_string())]
        );
        // A lone regional indicator never clusters.
        assert_eq!(
            split_text_into_segments("\u{1F1FA}"),
            vec![Segment::Text("\u{1F1FA}".to_string())]
        );
        // Greedy pairing consumes an invalid leading pair before a valid one
        // could form off its second RI: 🇽🇺🇸 → XU (text) + lone S (text) —
        // bit-identical to the TS regex, which matches ANY pair then demotes.
        assert_eq!(
            split_text_into_segments("\u{1F1FD}\u{1F1FA}\u{1F1F8}"),
            vec![
                Segment::Text("\u{1F1FD}\u{1F1FA}".to_string()),
                Segment::Text("\u{1F1F8}".to_string()),
            ]
        );
    }

    const GB_ENG: &str = "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}";
    const GB_SCT: &str = "\u{1F3F4}\u{E0067}\u{E0062}\u{E0073}\u{E0063}\u{E0074}\u{E007F}";
    const GB_WLS: &str = "\u{1F3F4}\u{E0067}\u{E0062}\u{E0077}\u{E006C}\u{E0073}\u{E007F}";

    #[test]
    fn tag_sequence_flags_cluster_as_emoji() {
        // Tag chars are not Extended_Pictographic, so the cluster used to stop at
        // the base 🏴 and the six tag chars leaked into the next TEXT segment,
        // where they shape as notdef tofu.
        for flag in [GB_ENG, GB_SCT, GB_WLS] {
            assert_eq!(
                split_text_into_segments(flag),
                vec![Segment::Emoji(flag.to_string())]
            );
        }
        // Adjacent sequences each terminate at their own U+E007F — no run-on.
        assert_eq!(
            split_text_into_segments(&format!("{GB_ENG}{GB_SCT}{GB_WLS}")),
            vec![
                Segment::Emoji(GB_ENG.to_string()),
                Segment::Emoji(GB_SCT.to_string()),
                Segment::Emoji(GB_WLS.to_string()),
            ]
        );
        assert_eq!(
            split_text_into_segments(&format!("go {GB_SCT}!")),
            vec![
                Segment::Text("go ".to_string()),
                Segment::Emoji(GB_SCT.to_string()),
                Segment::Text("!".to_string()),
            ]
        );
    }

    #[test]
    fn unshipped_tag_sequences_degrade_to_the_bare_flag() {
        // Only GB-ENG/SCT/WLS ship. Unicode says draw the base flag and ignore
        // unrecognized tags; 🏴 has its own SVG so it stays on the image lane.
        let us_ca = "\u{1F3F4}\u{E0075}\u{E0073}\u{E0063}\u{E0061}\u{E007F}";
        assert_eq!(
            split_text_into_segments(us_ca),
            vec![Segment::Emoji("\u{1F3F4}".to_string())]
        );
        // An unterminated run (no U+E007F) is not a sequence: the base clusters
        // alone and the loose tag chars are stripped, never shaped.
        assert_eq!(
            split_text_into_segments("\u{1F3F4}\u{E0067}\u{E0062}"),
            vec![Segment::Emoji("\u{1F3F4}".to_string())]
        );
        // Tag chars with no base flag at all are dropped from the text run.
        assert_eq!(
            split_text_into_segments("x\u{E0067}\u{E007F}y"),
            vec![Segment::Text("xy".to_string())]
        );
    }

    #[test]
    fn other_waved_flag_clusters_are_untouched() {
        // Both have their own shipped SVG. The tag branch keys off a well-formed
        // tag run, not off the U+1F3F4 base, so neither is demoted.
        assert_eq!(
            split_text_into_segments("\u{1F3F4}"),
            vec![Segment::Emoji("\u{1F3F4}".to_string())]
        );
        let pirate = "\u{1F3F4}\u{200D}\u{2620}\u{FE0F}";
        assert_eq!(
            split_text_into_segments(pirate),
            vec![Segment::Emoji(pirate.to_string())]
        );
    }

    #[test]
    fn flag_table_mirrors_are_identical() {
        // moda-scene (assess gate) carries its own generated copy of the
        // valid-flag-pair table (it must stay dependency-light). Both are
        // emitted by `gen-flag-tables` — pin them byte-identical so a partial
        // regen can't make segmentation and assessment disagree.
        assert_eq!(
            crate::flags::VALID_FLAG_PAIRS,
            moda_scene::flags::VALID_FLAG_PAIRS
        );
        assert_eq!(
            crate::flags::VALID_TAG_SEQUENCES,
            moda_scene::flags::VALID_TAG_SEQUENCES
        );
    }

    #[test]
    #[ignore = "needs unported Moda fixture packages/editor/src/utils/msdf/emojiFlags.ts"]
    fn flag_table_ts_mirror_is_identical() {
        // The THIRD generated copy lives in the frontend
        // (utils/msdf/emojiFlags.ts — the browser segmenter's membership
        // gate). Parse its `\u{...}\u{...}` pair literals and pin them to the
        // Rust table, so a partial regen or hand-edit can't make browser
        // segmentation disagree with native clustering/assessment.
        let ts_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .join("packages/editor/src/utils/msdf/emojiFlags.ts");
        let src =
            std::fs::read_to_string(&ts_path).unwrap_or_else(|e| panic!("read {ts_path:?}: {e}"));
        // Entry lines are `  '\u{XXXX}\u{YYYY}…',`. Doc-comment examples start with
        // `*` after trimming, so they never parse as entries — which is why the RI
        // table reads as 259 here and not the 260 a naive grep reports.
        let mut ts_pairs: Vec<(u32, u32)> = Vec::new();
        let mut ts_tag_seqs: Vec<String> = Vec::new();
        for line in src.lines() {
            let Some(rest) = line.trim().strip_prefix("'\\u{") else {
                continue;
            };
            let cps: Option<Vec<u32>> = rest
                .split("\\u{")
                .map(|p| u32::from_str_radix(p.split('}').next().unwrap_or(""), 16).ok())
                .collect();
            let Some(cps) = cps else { continue };
            // Two codepoints = an RI pair; anything longer = a tag sequence.
            if let [a, b] = cps[..] {
                ts_pairs.push((a, b));
            } else if cps.len() > 2 {
                ts_tag_seqs.push(cps.iter().filter_map(|cp| char::from_u32(*cp)).collect());
            }
        }
        let drifted = "utils/msdf/emojiFlags.ts drifted from the generated Rust tables — \
             re-run `cargo run -p moda-fonts --example gen-flag-tables`";
        assert_eq!(
            ts_pairs,
            crate::flags::VALID_FLAG_PAIRS.to_vec(),
            "{drifted}"
        );
        assert_eq!(
            ts_tag_seqs,
            crate::flags::VALID_TAG_SEQUENCES.to_vec(),
            "{drifted}"
        );
    }

    #[test]
    #[ignore = "needs unported Moda fixture packages/editor/src/utils/msdf/emojiFlags.ts"]
    fn shipped_single_emoji_ts_mirror_is_identical() {
        // Same drift guard as `flag_table_ts_mirror_is_identical`, for the
        // coverage table — and it needs its OWN parser: that test reads
        // `'\u{...}'` string entries, so it skips these `[0x…, 0x…],` numeric
        // range rows entirely and would stay green through a partial regen.
        //
        // Drift here is not cosmetic. This table decides whether a lone
        // pictograph goes to the SVG lane or the text lane, so a Rust/TS
        // disagreement means the browser and an export of the same canvas
        // disagree about whether a codepoint is emoji — and one of them draws
        // nothing.
        let ts_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .unwrap()
            .join("packages/editor/src/utils/msdf/emojiFlags.ts");
        let src =
            std::fs::read_to_string(&ts_path).unwrap_or_else(|e| panic!("read {ts_path:?}: {e}"));
        let mut ts_ranges: Vec<(u32, u32)> = Vec::new();
        for line in src.lines() {
            // Entry rows are `  [0xLO, 0xHI],`. The doc comment's prose `[lo, hi]`
            // has no `0x`, so it can never parse as an entry.
            let Some(rest) = line.trim().strip_prefix("[0x") else {
                continue;
            };
            let Some((lo, tail)) = rest.split_once(", 0x") else {
                continue;
            };
            let Some(hi) = tail.strip_suffix("],") else {
                continue;
            };
            match (u32::from_str_radix(lo, 16), u32::from_str_radix(hi, 16)) {
                (Ok(lo), Ok(hi)) => ts_ranges.push((lo, hi)),
                _ => continue,
            }
        }
        assert!(
            ts_ranges.len() > 100,
            "parsed only {} ranges from emojiFlags.ts — parser out of step with the generator?",
            ts_ranges.len()
        );
        assert_eq!(
            ts_ranges,
            crate::flags::SHIPPED_SINGLE_EMOJI.to_vec(),
            "utils/msdf/emojiFlags.ts SHIPPED_SINGLE_EMOJI drifted from the generated Rust \
             table — re-run `cargo run -p moda-fonts --example gen-flag-tables`"
        );
    }

    #[test]
    fn symbol_codepoints_split_on_extended_pictographic() {
        // The symbol-script fallback (`vector::script_fallback`) only ever sees
        // what emoji segmentation leaves as TEXT, so the boundary between the
        // two lanes is load-bearing: an Extended_Pictographic symbol must leave
        // as an emoji cluster (image lane), a non-pictographic one must stay
        // text (vector lane, where the Noto Sans Math / Symbols 2 chain covers
        // it). Pin both sides.
        //
        // ✨ U+2728, ★ U+2605, ❤ U+2764, ⚕ U+2695 are all
        // Extended_Pictographic despite living in the Misc Symbols / Dingbats
        // blocks the symbol ranges claim. Membership alone no longer decides the
        // lane, though: ★ ships no SVG, so the coverage gate demotes it back to
        // text (`lone_pictographs_without_a_shipped_svg_stay_text`) — which is
        // the fix, since as an emoji it drew nothing at all. The other three
        // ship and stay on the image lane.
        assert_eq!(
            split_text_into_segments("\u{2728}\u{2605}\u{2764}\u{2695}"),
            vec![
                Segment::Emoji("\u{2728}".to_string()),
                Segment::Text("\u{2605}".to_string()),
                Segment::Emoji("\u{2764}".to_string()),
                Segment::Emoji("\u{2695}".to_string()),
            ]
        );
        // ✓ U+2713 and ☆ U+2606 are NOT Extended_Pictographic — one text
        // segment, handed to the vector lane whole.
        assert_eq!(
            split_text_into_segments("\u{2713}\u{2606}"),
            vec![Segment::Text("\u{2713}\u{2606}".to_string())]
        );
    }

    #[test]
    fn lone_pictographs_without_a_shipped_svg_stay_text() {
        // The bug: `Extended_Pictographic` is much wider than the Noto SVG set,
        // and `ops.rs` drops an emoji quad whose SVG is missing while layout
        // still reserves its ~1em box — so a lone pictograph with no SVG
        // rendered as BLANK SPACE. ~1.4k codepoints ship a single-codepoint
        // SVG; the property's non-emoji (E0.0) pictographs ship none.
        //
        // ⚑ U+2691 BLACK FLAG (the reported case), ★ U+2605 BLACK STAR,
        // ♪ U+266A EIGHTH NOTE, ♜ U+265C BLACK CHESS ROOK, ⚀ U+2680 DIE FACE-1,
        // 🀀 U+1F000 MAHJONG TILE EAST WIND — all assigned, all in real content,
        // none with an SVG. They now leave as text and reach the font +
        // script-fallback chain.
        for cluster in [
            "\u{2691}",
            "\u{2605}",
            "\u{266A}",
            "\u{265C}",
            "\u{2680}",
            "\u{1F000}",
        ] {
            assert!(
                !ships_single_codepoint_emoji(cluster.chars().next().unwrap() as u32),
                "{cluster:?} unexpectedly has a shipped SVG — fixture assumption broken"
            );
            assert_eq!(
                split_text_into_segments(cluster),
                vec![Segment::Text(cluster.to_string())],
                "{cluster:?} must not be routed to the emoji lane"
            );
        }
        // Adjacent demoted pictographs stay SEPARATE text segments — each was
        // its own regex match before the gate demoted it. That is the shape the
        // TS lane produces for the same input (and the shape the unshipped-RI
        // rule already produces, see `invalid_pairs_and_lone_regional_indicators
        // _stay_text`), so parity, not coalescing, is the correct behaviour.
        assert_eq!(
            split_text_into_segments("\u{2691}\u{2605}\u{266A}"),
            vec![
                Segment::Text("\u{2691}".to_string()),
                Segment::Text("\u{2605}".to_string()),
                Segment::Text("\u{266A}".to_string()),
            ]
        );
        // Demoted pictographs merge into an ADJACENT text span, so surrounding
        // prose is not fragmented.
        assert_eq!(
            split_text_into_segments("Rated \u{2605} highly"),
            vec![
                Segment::Text("Rated ".to_string()),
                Segment::Text("\u{2605}".to_string()),
                Segment::Text(" highly".to_string()),
            ]
        );
    }

    #[test]
    fn lone_pictographs_with_a_shipped_svg_are_untouched() {
        // The other half of the gate — the guarantee that this is additive.
        // ✅ U+2705 shares a block with ⚑, 🀄 U+1F004 shares one with 🀀, and
        // both ship, so both stay on the image lane.
        for cluster in [
            "\u{2705}",
            "\u{1F004}",
            "\u{2728}",
            "\u{2764}",
            "\u{2695}",
            "🎨",
        ] {
            assert_eq!(
                split_text_into_segments(cluster),
                vec![Segment::Emoji(cluster.to_string())],
                "{cluster:?} ships an SVG and must stay an emoji"
            );
        }
        // A VS16 does not change the resolved filename, so it must not change
        // the verdict either: ☑ U+2611 ships, with or without the selector.
        assert_eq!(
            split_text_into_segments("\u{2611}\u{FE0F}"),
            vec![Segment::Emoji("\u{2611}\u{FE0F}".to_string())]
        );
        // Multi-codepoint clusters resolve to filenames this table does not
        // describe and must bypass the gate entirely: a skin-toned thumbs-up
        // (👍🏻) and a ZWJ family both keep clustering.
        let skin_toned = "\u{1F44D}\u{1F3FB}";
        assert_eq!(
            split_text_into_segments(skin_toned),
            vec![Segment::Emoji(skin_toned.to_string())]
        );
        let family = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}";
        assert_eq!(
            split_text_into_segments(family),
            vec![Segment::Emoji(family.to_string())]
        );
    }

    #[test]
    fn shipped_single_emoji_ranges_are_sorted_and_disjoint() {
        // `ships_single_codepoint_emoji` binary-searches, so a generator that
        // emitted overlapping or unsorted ranges would silently answer wrong.
        let table = crate::flags::SHIPPED_SINGLE_EMOJI;
        assert!(table.len() > 100, "table looks truncated: {}", table.len());
        for w in table.windows(2) {
            assert!(w[0].0 <= w[0].1, "range {:?} is inverted", w[0]);
            assert!(
                w[0].1 < w[1].0,
                "ranges {:?} and {:?} overlap or are unsorted",
                w[0],
                w[1]
            );
        }
        // Boundary probes: every range's own endpoints hit, and the codepoint
        // just past a range's end misses unless the next range starts there.
        for (i, (lo, hi)) in table.iter().enumerate() {
            assert!(ships_single_codepoint_emoji(*lo));
            assert!(ships_single_codepoint_emoji(*hi));
            let next_starts_here = table.get(i + 1).is_some_and(|(n, _)| *n == hi + 1);
            assert_eq!(ships_single_codepoint_emoji(hi + 1), next_starts_here);
        }
    }

    #[test]
    fn mixed_text_and_emoji() {
        let segs = split_text_into_segments("hi 🎨 there");
        assert_eq!(
            segs,
            vec![
                Segment::Text("hi ".to_string()),
                Segment::Emoji("🎨".to_string()),
                Segment::Text(" there".to_string()),
            ]
        );
    }
}
