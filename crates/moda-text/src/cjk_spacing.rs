//! CJK spacing (`text-spacing-trim`, `text-autospace`; `crate::paragraph_format::CjkSpacing`). Mirrored by
//! `utils/richtext/cjkSpacing.ts` (same classes, same adjustments, same lines).
//!
//! Fullwidth punctuation is drawn in the middle or on one side of a square em; trimming removes its blank half:
//! - opening marks (`「『（【〔〈《［｛〖〘〚｟〝`) are blank on the left, closing marks (`」』）】〕〉》］｝〗〙〛｠〞〟`) and the
//!   stops `、。，．` on the right; in Traditional Chinese (`lang` zh-Hant, zh-TW, zh-HK) the stops are centred and never
//!   trimmed;
//! - **adjacent pairs** (every value but `space-all`): a closing mark before a closing mark or a middle dot (`・：；`)
//!   loses its right half; an opening mark after an opening, closing or middle mark moves left by half an em (the mark
//!   before it gives up its blank half);
//! - **line start** (`space-first`, `trim-start`, `trim-both`): a line that starts with an opening mark starts half an em
//!   further left (`Line::indent_start`), so the mark's ink sits on the edge (`space-first` leaves the paragraph's first
//!   line, and the first after a hard break, alone);
//! - **line end** (`trim-both`): a closing mark that ends a line loses its right half.
//!
//! Autospace adds an eighth of an em between an ideograph (Han, kana) and a letter of a space-separated script
//! (`ideograph-alpha`) or a digit (`ideograph-numeric`); not across spaces or punctuation, and not at a line's end.
//!
//! The pair adjustments are made before the wrap, so line breaks account for them: the character before the gap is set
//! apart in its own run with its letter spacing changed (spread over the code points of its cluster). The line edges
//! are adjusted after the wrap; a pair the wrap split across two lines takes its pair adjustment back.

use crate::paragraph_format::{Autospace, ParagraphFormat, SpacingTrim};
use crate::types::{InlineRun, Line, LineSegment, Paragraph};
use crate::vector::cjk::{is_cjk_breakable_codepoint, is_cluster_continuation, js_whitespace};

const OPENING: &str = "「『（［｛〈《【〔〖〘〚｟〝";
const CLOSING_BRACKETS: &str = "」』）］｝〉》】〕〗〙〛｠〞〟";
const STOPS: &str = "、。，．";
const MIDDLE: &str = "・：；";

/// The blank half of a fullwidth mark, as a fraction of its font size.
const HALF: f64 = 0.5;
/// Autospace, as a fraction of the ideograph's font size.
const AUTOSPACE: f64 = 0.125;

fn in_set(set: &str, ch: char) -> bool {
    set.chars().any(|c| c == ch)
}

/// A fullwidth opening mark (blank on the left).
pub fn is_opening(ch: char) -> bool {
    in_set(OPENING, ch)
}

/// A fullwidth closing mark or stop, as context for an adjacent pair (centred stops included).
pub fn is_closing(ch: char) -> bool {
    in_set(CLOSING_BRACKETS, ch) || in_set(STOPS, ch)
}

/// A closing mark whose right half is blank in `lang`: every closing bracket, and the stops but in Traditional
/// Chinese, which centres them.
pub fn is_trimmable_closing(ch: char, lang: Option<&str>) -> bool {
    in_set(CLOSING_BRACKETS, ch)
        || (in_set(STOPS, ch)
            && lang.and_then(crate::lang::CjkRegion::of) != Some(crate::lang::CjkRegion::TraditionalChinese))
}

fn is_middle(ch: char) -> bool {
    in_set(MIDDLE, ch)
}

/// Han and kana (CSS's ideographs for autospace; Hangul is written with spaces and isn't one).
pub fn is_ideograph(ch: char) -> bool {
    matches!(ch as u32,
        0x3005..=0x3007 | 0x3041..=0x3096 | 0x30a1..=0x30fa | 0x30fc..=0x30ff | 0x31f0..=0x31ff
        | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x3ffff)
}

/// A letter of a space-separated script: alphabetic, and not CJK.
fn is_alpha(ch: char) -> bool {
    ch.is_alphabetic() && !is_cjk_breakable_codepoint(ch as u32)
}

/// A decimal digit, not a fullwidth one.
fn is_numeric(ch: char) -> bool {
    use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};
    ch.general_category() == GeneralCategory::DecimalNumber && !is_cjk_breakable_codepoint(ch as u32)
}

/// One side of a pair: its base character, font size and language.
#[derive(Clone, Copy)]
struct Side<'a> {
    base: char,
    font_size: f64,
    lang: Option<&'a str>,
}

/// The advance change (px) of the cluster `a` that `b` follows directly (no space between), under `trim` and
/// `autospace`: negative to trim, positive for autospace, 0 for none (`pairAdjustPx`).
fn pair_adjust(a: Side<'_>, b: Side<'_>, trim: SpacingTrim, autospace: Autospace) -> f64 {
    let mut adj = 0.0;
    if trim != SpacingTrim::SpaceAll {
        if is_trimmable_closing(a.base, a.lang) && (is_closing(b.base) || is_middle(b.base)) {
            adj -= HALF * a.font_size;
        } else if is_opening(b.base) && (is_opening(a.base) || is_closing(a.base) || is_middle(a.base)) {
            adj -= HALF * a.font_size;
        }
    }
    let wants = |c: char| (autospace.alpha && is_alpha(c)) || (autospace.numeric && is_numeric(c));
    if is_ideograph(a.base) && wants(b.base) {
        adj += AUTOSPACE * a.font_size;
    } else if is_ideograph(b.base) && wants(a.base) {
        adj += AUTOSPACE * b.font_size;
    }
    adj
}

/// A cluster of a paragraph: a base character and the marks that belong to it, within one run.
struct Cluster {
    run: usize,
    start: usize,
    end: usize,
    base: char,
    chars: usize,
}

/// `para` with its adjacent-pair adjustments made (`applyCjkCharSpacing`): each adjusted cluster in a run of its own
/// whose letter spacing carries the change. `None` when nothing changes.
pub fn apply_char_spacing(para: &Paragraph) -> Option<Paragraph> {
    let spacing = para.format.spacing;
    if spacing.trim == SpacingTrim::SpaceAll && spacing.autospace == Autospace::NONE {
        return None;
    }
    let mut clusters: Vec<Cluster> = Vec::new();
    for (ri, run) in para.runs.iter().enumerate() {
        for (b, ch) in run.text.char_indices() {
            match clusters.last_mut() {
                Some(c) if c.run == ri && is_cluster_continuation(ch) => {
                    c.end = b + ch.len_utf8();
                    c.chars += 1;
                }
                _ => clusters.push(Cluster {
                    run: ri,
                    start: b,
                    end: b + ch.len_utf8(),
                    base: ch,
                    chars: 1,
                }),
            }
        }
    }
    let side = |c: &Cluster| Side {
        base: c.base,
        font_size: para.runs[c.run].style.font_size,
        lang: para.runs[c.run].style.lang.as_deref(),
    };
    let mut adjust = vec![0.0f64; clusters.len()];
    let mut any = false;
    for k in 0..clusters.len().saturating_sub(1) {
        let (a, b) = (&clusters[k], &clusters[k + 1]);
        if js_whitespace(a.base) || js_whitespace(b.base) {
            continue;
        }
        let adj = pair_adjust(side(a), side(b), spacing.trim, spacing.autospace);
        if adj != 0.0 {
            adjust[k] = adj;
            any = true;
        }
    }
    if !any {
        return None;
    }
    let mut runs: Vec<InlineRun> = Vec::with_capacity(para.runs.len());
    let mut k = 0usize;
    for (ri, run) in para.runs.iter().enumerate() {
        let mut cp_at = vec![0; run.text.len()+1];
        let mut cp = 0;
        for (byte, _) in run.text.char_indices() { cp_at[byte] = cp; cp += 1; }
        cp_at[run.text.len()] = cp;
        let mut plain_from = 0usize;
        while k < clusters.len() && clusters[k].run == ri {
            let c = &clusters[k];
            if adjust[k] != 0.0 {
                if c.start > plain_from {
                    runs.push(InlineRun {
                        text: run.text[plain_from..c.start].to_string(),
                        style: run.style.source_slice(cp_at[plain_from]..cp_at[c.start]),
                    });
                }
                let mut style = run.style.source_slice(cp_at[c.start]..cp_at[c.end]);
                style.letter_spacing += adjust[k] / c.chars as f64;
                runs.push(InlineRun {
                    text: run.text[c.start..c.end].to_string(),
                    style,
                });
                plain_from = c.end;
            }
            k += 1;
        }
        if plain_from < run.text.len() || run.text.is_empty() {
            runs.push(InlineRun {
                text: run.text[plain_from..].to_string(),
                style: run.style.source_slice(cp_at[plain_from]..cp),
            });
        }
    }
    Some(Paragraph {
        runs,
        ..para.clone()
    })
}

/// The first cluster of a line's text: its base character and the segment it's in.
fn first_side(line: &Line) -> Option<(char, &LineSegment)> {
    let seg = line.segments.first()?;
    Some((seg.text.chars().next()?, seg))
}

/// The base character of a segment's last cluster, and how many code points the cluster has.
fn last_cluster(text: &str) -> Option<(char, usize, usize)> {
    let mut base = None;
    let mut start = 0usize;
    let mut chars = 0usize;
    for (b, ch) in text.char_indices() {
        if base.is_some() && is_cluster_continuation(ch) {
            chars += 1;
        } else {
            base = Some(ch);
            start = b;
            chars = 1;
        }
    }
    base.map(|c| (c, start, chars))
}

/// Make a line's last cluster a segment of its own (when it isn't), returning its index.
fn split_last_cluster(line: &mut Line) -> Option<usize> {
    let seg = line.segments.last()?;
    if seg.tab.is_some() || seg.hyphen {
        return None;
    }
    let (_, start, _) = last_cluster(&seg.text)?;
    if start > 0 {
        let seg = line.segments.pop().unwrap();
        let mut head = seg.clone();
        head.text = seg.text[..start].to_string();
        let cp = head.text.chars().count();
        head.style.source = seg.style.source.slice(0..cp);
        let mut tail = seg;
        tail.style.source = tail.style.source.slice(cp..tail.text.chars().count());
        tail.text = tail.text[start..].to_string();
        line.segments.push(head);
        line.segments.push(tail);
    }
    Some(line.segments.len() - 1)
}

/// Adjust the lines of a paragraph laid out from [`apply_char_spacing`]'s runs (`trimCjkLines`): a pair split across
/// a soft wrap takes its adjustment back, and the line edges are trimmed per `text-spacing-trim`.
pub fn trim_lines(lines: &mut [Line], format: &ParagraphFormat) {
    let spacing = format.spacing;
    if spacing.is_default() {
        return;
    }
    let trim = spacing.trim;
    for i in 0..lines.len() {
        // A pair the wrap split: the cluster before the gap ends this line, the one after starts the next.
        let next_first = if lines[i].soft_wrap {
            lines.get(i + 1).and_then(first_side).map(|(c, s)| Side {
                base: c,
                font_size: s.style.font_size,
                lang: None,
            })
        } else {
            None
        };
        if let Some(next) = next_first.filter(|n| !js_whitespace(n.base)) {
            let line = &mut lines[i];
            if let Some(seg) = line.segments.last() {
                if let Some((base, _, chars)) = last_cluster(&seg.text) {
                    if !js_whitespace(base) {
                        let a = Side {
                            base,
                            font_size: seg.style.font_size,
                            lang: seg.style.lang.as_deref(),
                        };
                        let adj = pair_adjust(a, next, trim, spacing.autospace);
                        if adj != 0.0 {
                            if let Some(j) = split_last_cluster(line) {
                                line.segments[j].style.letter_spacing -= adj / chars as f64;
                            }
                        }
                    }
                }
            }
        }
        // Line end: a closing mark loses its blank right half.
        if trim == SpacingTrim::TrimBoth {
            let line = &mut lines[i];
            if let Some(seg) = line.segments.last() {
                if let Some((base, _, chars)) = last_cluster(&seg.text) {
                    if is_trimmable_closing(base, seg.style.lang.as_deref()) {
                        let font_size = seg.style.font_size;
                        if let Some(j) = split_last_cluster(line) {
                            line.segments[j].style.letter_spacing -= HALF * font_size / chars as f64;
                        }
                    }
                }
            }
        }
        // Line start: an opening mark's blank left half hangs before the edge.
        let starts = match trim {
            SpacingTrim::TrimStart | SpacingTrim::TrimBoth => true,
            SpacingTrim::SpaceFirst => i > 0 && lines[i - 1].soft_wrap,
            _ => false,
        };
        if starts {
            if let Some((c, seg)) = first_side(&lines[i]) {
                if is_opening(c) && seg.tab.is_none() {
                    let shift = HALF * seg.style.font_size;
                    lines[i].indent_start -= shift;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paragraph_format::CjkSpacing;

    #[test]
    fn pairs() {
        let s = |base| Side {
            base,
            font_size: 20.0,
            lang: None,
        };
        let both = Autospace {
            alpha: true,
            numeric: true,
        };
        assert_eq!(pair_adjust(s('。'), s('」'), SpacingTrim::Normal, Autospace::NONE), -10.0);
        assert_eq!(pair_adjust(s('」'), s('「'), SpacingTrim::Normal, Autospace::NONE), -10.0);
        assert_eq!(pair_adjust(s('（'), s('「'), SpacingTrim::Normal, Autospace::NONE), -10.0);
        assert_eq!(pair_adjust(s('」'), s('あ'), SpacingTrim::TrimBoth, Autospace::NONE), 0.0);
        assert_eq!(pair_adjust(s('」'), s('「'), SpacingTrim::SpaceAll, Autospace::NONE), 0.0);
        assert_eq!(pair_adjust(s('漢'), s('A'), SpacingTrim::SpaceAll, both), 2.5);
        assert_eq!(pair_adjust(s('1'), s('か'), SpacingTrim::SpaceAll, both), 2.5);
        assert_eq!(pair_adjust(s('漢'), s('한'), SpacingTrim::SpaceAll, both), 0.0);
        assert_eq!(
            pair_adjust(
                s('漢'),
                s('1'),
                SpacingTrim::SpaceAll,
                Autospace {
                    alpha: true,
                    numeric: false
                }
            ),
            0.0
        );
        let hant = Side {
            base: '。',
            font_size: 20.0,
            lang: Some("zh-TW"),
        };
        assert_eq!(pair_adjust(hant, s('」'), SpacingTrim::Normal, Autospace::NONE), 0.0);
        let _ = CjkSpacing::DEFAULT;
    }
}
