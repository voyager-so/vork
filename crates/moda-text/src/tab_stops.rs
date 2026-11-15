//! Tab stops with leaders (Illustrator / InDesign Tabs panel): a paragraph's `\t` advances to its next stop, and a
//! stop's leader fills the gap with a repeated string (`Soup ......... 6.50`). Mirrored by
//! `utils/richtext/tabStops.ts` (same parse, same resolve, same constants), so the editor measures what the engine
//! draws.
//!
//! - `\t` goes to the first stop right of the pen; past the last stop, default stops every `interval` px (36 unless
//!   set), left-aligned without a leader.
//! - Left: the text after the tab starts at the stop. Right: it ends there. Center: it centres on it. Decimal: its
//!   first decimal character ('.' unless set) sits on it (none: it ends there). The field is the text up to the next
//!   tab or the line's end; it never moves left of the pen.
//! - A leader repeats copies on a grid anchored at the paragraph's content left edge (copy k at k × its width), so
//!   dots line up down a column, and fills the gap: the whole copies inside it, and at each end the characters of the
//!   copy the gap cuts that lie wholly inside it (a leader longer than the gap still fills it).
//! - A paragraph with tabs and no stops of its own takes [`default_tab_stops`] (the parser sets them): its tabs go to
//!   the default stops, every 36 px, as in Illustrator, InDesign and Word. A tab is never a space's width.
//!
//! Stored as the inherited CSS custom property `--nf-tab-stops`, e.g.
//! `240 right leader '.', 300 decimal char ',', every 48` (strings single-quoted, CSS hex escapes inside).

use crate::types::InlineStyle;

pub const TAB_STOPS_PROPERTY: &str = "--nf-tab-stops";
/// InDesign's default: half an inch at 72 px/in.
pub const DEFAULT_TAB_INTERVAL_PX: f64 = 36.0;
pub const MIN_TAB_INTERVAL_PX: f64 = 1.0;
/// Most copies one leader draws.
pub const MAX_LEADER_COPIES: usize = 2000;
pub const MAX_TAB_STOPS: usize = 64;
const MAX_LEADER_CHARS: usize = 16;
const EPS: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabAlign {
    Left,
    Center,
    Right,
    Decimal,
}

impl TabAlign {
    fn parse(word: &str) -> Option<TabAlign> {
        Some(match word {
            "left" => TabAlign::Left,
            "center" => TabAlign::Center,
            "right" => TabAlign::Right,
            "decimal" => TabAlign::Decimal,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TabStop {
    /// px from the paragraph's content left edge.
    pub position: f64,
    pub align: TabAlign,
    pub leader: Option<String>,
    /// Decimal stops: the aligned character; `None` = '.'.
    pub decimal_char: Option<char>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TabStops {
    /// Sorted by position, positions distinct.
    pub stops: Vec<TabStop>,
    /// Default interval past the last stop; `None` = [`DEFAULT_TAB_INTERVAL_PX`].
    pub interval: Option<f64>,
}

/// A paragraph with tabs and no stops of its own: only the default stops (`DEFAULT_TAB_STOPS`).
pub fn default_tab_stops() -> std::sync::Arc<TabStops> {
    std::sync::Arc::new(TabStops {
        stops: Vec::new(),
        interval: None,
    })
}

/// A resolved tab's leader: `count` whole copies of `text`, the first `offset` px after the tab's start, `unit` apart,
/// and the parts of the copies the gap cuts at either end (`head` before the whole copies, `tail` after them).
#[derive(Debug, Clone, PartialEq)]
pub struct Leader {
    pub text: String,
    pub offset: f64,
    pub unit: f64,
    pub count: usize,
    pub head: Option<LeaderPiece>,
    pub tail: Option<LeaderPiece>,
}

/// Part of a leader copy: its characters inside the gap, `offset` px after the tab's start.
#[derive(Debug, Clone, PartialEq)]
pub struct LeaderPiece {
    pub text: String,
    pub offset: f64,
}

/// What a resolved tab occupies on its line.
#[derive(Debug, Clone, PartialEq)]
pub struct TabFill {
    pub advance: f64,
    pub leader: Option<Leader>,
}

/// 4 decimals, the precision every markup length keeps (`round4` in `tabStops.ts`): a right stop at a text's width
/// lands on the width, not a rounding past it.
fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Tidy stops (`normalizeTabStops`): finite positions clamped to [0, 100000], sorted, one per position (the last
/// wins), capped; leaders trimmed to 16 chars and dropped when blank.
pub fn normalize_tab_stops(stops: Vec<TabStop>, interval: Option<f64>) -> Option<TabStops> {
    let mut by_pos: Vec<TabStop> = Vec::new();
    for s in stops {
        if !s.position.is_finite() {
            continue;
        }
        let position = round4(s.position.clamp(0.0, 100000.0));
        let leader = s
            .leader
            .map(|l| l.chars().take(MAX_LEADER_CHARS).collect::<String>())
            .filter(|l| !l.is_empty() && !l.chars().all(char::is_whitespace));
        let decimal_char = if s.align == TabAlign::Decimal {
            s.decimal_char.filter(|c| *c != '.')
        } else {
            None
        };
        let stop = TabStop {
            position,
            align: s.align,
            leader,
            decimal_char,
        };
        match by_pos.iter_mut().find(|t| t.position == position) {
            Some(t) => *t = stop,
            None => by_pos.push(stop),
        }
    }
    by_pos.sort_by(|a, b| a.position.total_cmp(&b.position));
    by_pos.truncate(MAX_TAB_STOPS);
    let interval = interval
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| round4(v).max(MIN_TAB_INTERVAL_PX))
        .filter(|v| *v != DEFAULT_TAB_INTERVAL_PX);
    if by_pos.is_empty() && interval.is_none() {
        return None;
    }
    Some(TabStops {
        stops: by_pos,
        interval,
    })
}

enum Token {
    Word(String),
    Str(String),
    Comma,
}

fn tokenize(value: &str) -> Option<Vec<Token>> {
    let chars: Vec<char> = value.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch == ',' {
            tokens.push(Token::Comma);
            i += 1;
        } else if ch == '\'' || ch == '"' {
            let q = ch;
            let mut s = String::new();
            i += 1;
            let mut closed = false;
            while i < chars.len() {
                let c = chars[i];
                if c == q {
                    closed = true;
                    i += 1;
                    break;
                }
                if c == '\\' && i + 1 < chars.len() {
                    // CSS escape: 1–6 hex digits and one optional space, or the next character itself.
                    let mut hex = String::new();
                    let mut j = i + 1;
                    while j < chars.len() && hex.len() < 6 && chars[j].is_ascii_hexdigit() {
                        hex.push(chars[j]);
                        j += 1;
                    }
                    if !hex.is_empty() {
                        if j < chars.len() && chars[j] == ' ' {
                            j += 1;
                        }
                        let cp = u32::from_str_radix(&hex, 16).unwrap_or(0);
                        s.push(
                            char::from_u32(cp)
                                .filter(|_| cp > 0)
                                .unwrap_or('\u{FFFD}'),
                        );
                        i = j;
                    } else {
                        s.push(chars[i + 1]);
                        i += 2;
                    }
                    continue;
                }
                s.push(c);
                i += 1;
            }
            if !closed {
                return None;
            }
            tokens.push(Token::Str(s));
        } else {
            let mut w = String::new();
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !matches!(chars[i], ',' | '\'' | '"')
            {
                w.push(chars[i]);
                i += 1;
            }
            tokens.push(Token::Word(w.to_lowercase()));
        }
    }
    Some(tokens)
}

/// px per absolute length unit at 72 px per inch (`LENGTH_UNIT_PX` in `markup/units.ts`).
fn unit_px(unit: &str) -> Option<f64> {
    Some(match unit {
        "" | "px" | "pt" => 1.0,
        "pc" => 12.0,
        "in" => 72.0,
        "cm" => 72.0 / 2.54,
        "mm" => 72.0 / 25.4,
        "q" => 72.0 / 101.6,
        _ => return None,
    })
}

/// `^(-?(?:\d+\.?\d*|\.\d+))([a-z]*)$` with a known unit (`parseTabLength`).
fn parse_length(word: &str) -> Option<f64> {
    let word = word.trim().to_ascii_lowercase();
    let split = word.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(word.len());
    let (num, unit) = word.split_at(split);
    if !unit.chars().all(|c| c.is_ascii_lowercase()) {
        return None;
    }
    let per = unit_px(unit)?;
    let digits = num.strip_prefix('-').unwrap_or(num);
    let (int, frac) = match digits.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (digits, None),
    };
    let all_digits = |s: &str| s.chars().all(|c| c.is_ascii_digit());
    let ok = match frac {
        None => !int.is_empty() && all_digits(int),
        Some(f) => all_digits(int) && all_digits(f) && (!int.is_empty() || !f.is_empty()),
    };
    if !ok {
        return None;
    }
    num.parse::<f64>().ok().map(|v| v * per).filter(|v| v.is_finite())
}

/// Parse a `--nf-tab-stops` value (`parseTabStops`); `none`, '' or garbage → `None`. Unknown items are skipped.
pub fn parse_tab_stops(value: &str) -> Option<TabStops> {
    if value.trim().is_empty() || value.trim().eq_ignore_ascii_case("none") {
        return None;
    }
    let tokens = tokenize(value)?;
    let mut items: Vec<Vec<Token>> = vec![Vec::new()];
    for t in tokens {
        match t {
            Token::Comma => items.push(Vec::new()),
            t => items.last_mut().unwrap().push(t),
        }
    }
    let mut stops = Vec::new();
    let mut interval = None;
    for item in &items {
        let Some(Token::Word(head)) = item.first() else {
            continue;
        };
        if head == "every" {
            if let Some(Token::Word(w)) = item.get(1) {
                if let Some(v) = parse_length(w).filter(|v| *v > 0.0) {
                    interval = Some(v);
                }
            }
            continue;
        }
        let Some(position) = parse_length(head) else {
            continue;
        };
        let mut stop = TabStop {
            position,
            align: TabAlign::Left,
            leader: None,
            decimal_char: None,
        };
        let mut k = 1;
        while k < item.len() {
            if let Token::Word(w) = &item[k] {
                if let Some(a) = TabAlign::parse(w) {
                    stop.align = a;
                } else if w == "leader" || w == "char" {
                    if let Some(Token::Str(s)) = item.get(k + 1) {
                        if w == "leader" {
                            stop.leader = Some(s.clone());
                        } else {
                            stop.decimal_char = s.chars().next();
                        }
                        k += 1;
                    }
                }
            }
            k += 1;
        }
        stops.push(stop);
    }
    normalize_tab_stops(stops, interval)
}

/// The stop a tab at pen `x` goes to (`nextTabStop`).
pub fn next_tab_stop(ts: &TabStops, x: f64) -> TabStop {
    if let Some(s) = ts.stops.iter().find(|s| s.position > x + EPS) {
        return s.clone();
    }
    let interval = ts
        .interval
        .unwrap_or(DEFAULT_TAB_INTERVAL_PX)
        .max(MIN_TAB_INTERVAL_PX);
    TabStop {
        position: ((x / interval + EPS).floor() + 1.0) * interval,
        align: TabAlign::Left,
        leader: None,
        decimal_char: None,
    }
}

/// Where a field starts for `stop` from pen `x` (`tabFieldStart`).
pub fn tab_field_start(stop: &TabStop, x: f64, field_width: f64, before_decimal: f64) -> f64 {
    let p = stop.position;
    let target = match stop.align {
        TabAlign::Left => p,
        TabAlign::Right => p - field_width,
        TabAlign::Center => p - field_width / 2.0,
        TabAlign::Decimal => p - before_decimal,
    };
    x.max(target)
}

/// Whole leader copies on the grid anchored at 0 inside [from, to] (`leaderGrid`): the first index and how many.
pub fn leader_grid(from: f64, to: f64, unit: f64) -> (i64, usize) {
    if !(unit > 0.01) || !(to > from) {
        return (0, 0);
    }
    let first = (from / unit - EPS).ceil() as i64;
    let end = (to / unit + EPS).floor() as i64;
    (first, (end - first).clamp(0, MAX_LEADER_COPIES as i64) as usize)
}

/// The characters of the leader copy at grid x `copy_x` that lie wholly inside [from, to], and the x of the first.
fn leader_piece<F>(
    text: &str,
    style: &InlineStyle,
    copy_x: f64,
    from: f64,
    to: f64,
    measure: &mut F,
) -> Option<(String, f64)>
where
    F: FnMut(&str, &InlineStyle) -> f64,
{
    let mut kept = String::new();
    let mut first_x = None;
    let mut before = 0.0;
    for (i, ch) in text.char_indices() {
        let end = i + ch.len_utf8();
        let after = measure(&text[..end], style);
        let (a, b) = (copy_x + before, copy_x + after);
        before = after;
        if a >= from - 1e-6 && b <= to + 1e-6 {
            first_x.get_or_insert(a);
            kept.push(ch);
        } else if first_x.is_some() {
            break;
        }
    }
    if kept.trim().is_empty() {
        return None;
    }
    first_x.map(|x| (kept, x))
}

pub fn is_tab_text(text: &str) -> bool {
    text == "\t"
}

/// Resolve every tab of one line (`resolveLineTabs`): each tab segment's fill (by segment index) and the line's width.
pub fn resolve_line_tabs<'a, F>(
    segments: &'a [(&'a str, &'a InlineStyle)],
    ts: &TabStops,
    mut measure: F,
) -> (f64, Vec<(usize, TabFill)>)
where
    F: FnMut(&str, &InlineStyle) -> f64,
{
    let mut fills = Vec::new();
    let mut x = 0.0;
    for (i, (text, style)) in segments.iter().enumerate() {
        if !is_tab_text(text) {
            x += measure(text, style);
            continue;
        }
        let stop = next_tab_stop(ts, x);
        let decimal_char = stop.decimal_char.unwrap_or('.');
        let mut field_width = 0.0;
        let mut before_decimal: Option<f64> = None;
        for (t, s) in segments[i + 1..].iter().take_while(|(t, _)| !is_tab_text(t)) {
            if stop.align == TabAlign::Decimal && before_decimal.is_none() {
                if let Some(at) = t.find(decimal_char) {
                    let pre = if at > 0 { measure(&t[..at], s) } else { 0.0 };
                    before_decimal = Some(field_width + pre);
                }
            }
            field_width += measure(t, s);
        }
        let start = tab_field_start(&stop, x, field_width, before_decimal.unwrap_or(field_width));
        let mut fill = TabFill {
            advance: start - x,
            leader: None,
        };
        if let Some(leader) = &stop.leader {
            let unit = measure(leader, style);
            let (first, count) = leader_grid(x, start, unit);
            // The copies the gap cuts: the one ending where the whole copies start, and the one after them.
            let mut piece = |k: i64| {
                leader_piece(leader, style, k as f64 * unit, x, start, &mut measure)
                    .map(|(text, at)| LeaderPiece { text, offset: at - x })
            };
            let valid = unit > 0.01 && start > x;
            let head = if valid && first as f64 * unit > x + EPS {
                piece(first - 1)
            } else {
                None
            };
            let tail = if valid && (first + count as i64) as f64 * unit < start - EPS {
                piece(first + count as i64)
            } else {
                None
            };
            if count > 0 || head.is_some() || tail.is_some() {
                fill.leader = Some(Leader {
                    text: leader.clone(),
                    offset: first as f64 * unit - x,
                    unit,
                    count,
                    head,
                    tail,
                });
            }
        }
        fills.push((i, fill));
        x = start;
    }
    (snap_width(x), fills)
}

/// A line's resolved width on a micro-pixel grid. A right or decimal field starts at the stop less its width, and
/// adding its segments back can end a few ulps past the stop (250.00000000000003 for a 250 stop), so a field ending
/// exactly at a stop on the box edge wrapped as if it overflowed.
fn snap_width(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// What a tab segment that starts at `segment_x` draws of its leader, left to right: the text of each piece (the
/// head, every whole copy, the tail) and its x.
pub fn leader_pieces(fill: Option<&TabFill>, segment_x: f64) -> Vec<(&str, f64)> {
    let Some(l) = fill.and_then(|f| f.leader.as_ref()) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(l.count + 2);
    if let Some(h) = &l.head {
        out.push((h.text.as_str(), segment_x + h.offset));
    }
    for k in 0..l.count {
        out.push((l.text.as_str(), segment_x + l.offset + k as f64 * l.unit));
    }
    if let Some(t) = &l.tail {
        out.push((t.text.as_str(), segment_x + t.offset));
    }
    out
}

/// Every leader copy of a positioned line, as segments to draw (the leader in its tab's style) and their x.
pub fn leader_segments(
    line: &crate::types::Line,
    positions: &[crate::positioning::SegmentPosition],
) -> Vec<(crate::types::LineSegment, f64)> {
    let mut out = Vec::new();
    for (seg, pos) in line.segments.iter().zip(positions) {
        for (text, x) in leader_pieces(seg.tab.as_ref(), pos.start_x) {
            out.push((crate::types::LineSegment::new(text.to_string(), seg.style.without_source()), x));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::default_style;

    fn mono(text: &str, _s: &InlineStyle) -> f64 {
        text.chars().count() as f64 * 10.0
    }

    fn resolve(stops: &str, parts: &[&str]) -> (f64, Vec<(usize, TabFill)>) {
        let style = default_style();
        let segs: Vec<(&str, &InlineStyle)> = parts.iter().map(|p| (*p, &style)).collect();
        resolve_line_tabs(&segs, &parse_tab_stops(stops).unwrap(), mono)
    }

    #[test]
    fn parses_like_the_ts_twin() {
        let t = parse_tab_stops("240 right leader '.', 100 left, 300 decimal char ',' leader ' . ', every 48").unwrap();
        assert_eq!(t.stops.len(), 3);
        assert_eq!(t.stops[0].position, 100.0);
        assert_eq!(t.stops[1].leader.as_deref(), Some("."));
        assert_eq!(t.stops[2].align, TabAlign::Decimal);
        assert_eq!(t.stops[2].decimal_char, Some(','));
        assert_eq!(t.stops[2].leader.as_deref(), Some(" . "));
        assert_eq!(t.interval, Some(48.0));
        assert_eq!(
            parse_tab_stops("50 left leader 'a\\3b b\\3a \\27 c'").unwrap().stops[0].leader.as_deref(),
            Some("a;b:'c")
        );
        assert!(parse_tab_stops("none").is_none());
        assert!(parse_tab_stops("50 left leader 'open").is_none());
        assert!(parse_tab_stops("every 36").is_none());
    }

    #[test]
    fn positions_take_length_units_and_keep_four_decimals() {
        let t = parse_tab_stops("77mm right leader '.', 1in left, 2.5cm center, 30pt, 1pc, every 0.25in").unwrap();
        let at: Vec<f64> = t.stops.iter().map(|s| s.position).collect();
        // 77 mm is 218.26771… px, kept at 218.2677 like the markup width `77mm`, not rounded past it to 218.268.
        assert_eq!(at, vec![12.0, 30.0, 70.8661, 72.0, 218.2677]);
        assert_eq!(t.interval, Some(18.0));
        assert_eq!(parse_tab_stops("3em right, 10 left").unwrap().stops.len(), 1);
    }

    #[test]
    fn stop_positions() {
        let t = parse_tab_stops("100 left, 200 right").unwrap();
        assert_eq!(next_tab_stop(&t, 0.0).position, 100.0);
        assert_eq!(next_tab_stop(&t, 100.0).position, 200.0);
        assert_eq!(next_tab_stop(&t, 250.0).position, 252.0);
        let every = parse_tab_stops("every 50").unwrap();
        assert_eq!(next_tab_stop(&every, 50.0).position, 100.0);
    }

    #[test]
    fn alignment_and_leaders() {
        // Right: "6.50" ends at 200; 12 dots from 40 to 160.
        let (w, fills) = resolve("200 right leader '.'", &["Soup", "\t", "6.50"]);
        assert_eq!(w, 200.0);
        assert_eq!(fills[0].1.advance, 120.0);
        let l = fills[0].1.leader.as_ref().unwrap();
        assert_eq!((l.offset, l.unit, l.count), (0.0, 10.0, 12));
        let (_, fills) = resolve("200 right leader ' . '", &["Soup", "\t", "6.50"]);
        let l = fills[0].1.leader.as_ref().unwrap();
        assert_eq!((l.offset, l.unit, l.count), (20.0, 30.0, 3));
        // The copy the gap cuts at its start gives its characters inside the gap (". " from 40); at the end only a
        // space of the copy at 150 is inside, which draws nothing.
        assert_eq!(
            leader_pieces(Some(&fills[0].1), 40.0),
            vec![(". ", 40.0), (" . ", 60.0), (" . ", 90.0), (" . ", 120.0)]
        );
        // A leader longer than the gap fills it: a 16-char leader (160 px) over a 120 px gap draws the characters
        // of the copies at 0 and 160 that fall inside [40, 160] ("efghijklmnop" from 40), where it drew nothing.
        let (_, fills) = resolve("200 right leader 'abcdefghijklmnop'", &["Soup", "\t", "6.50"]);
        let l = fills[0].1.leader.as_ref().expect("a leader");
        assert_eq!(l.count, 0);
        assert_eq!(leader_pieces(Some(&fills[0].1), 40.0), vec![("efghijklmnop", 40.0)]);
        // Longer than a copy but not a whole number of them: whole copies, then the cut ones at both ends.
        let (_, fills) = resolve("400 right leader 'abcdefghijklmnop'", &["Soup", "\t", "6.50"]);
        assert_eq!(
            leader_pieces(Some(&fills[0].1), 40.0),
            vec![("efghijklmnop", 40.0), ("abcdefghijklmnop", 160.0), ("abcd", 320.0)]
        );
        // Center, decimal, decimal char, no decimal, too wide.
        assert_eq!(resolve("100 center", &["a", "\t", "abcd"]).1[0].1.advance, 70.0);
        assert_eq!(resolve("100 decimal", &["a", "\t", "12.5"]).1[0].1.advance, 70.0);
        assert_eq!(resolve("100 decimal", &["a", "\t", "3.25"]).1[0].1.advance, 80.0);
        assert_eq!(resolve("100 decimal char ','", &["a", "\t", "1234,5"]).1[0].1.advance, 50.0);
        assert_eq!(resolve("100 decimal", &["a", "\t", "42"]).1[0].1.advance, 70.0);
        assert_eq!(resolve("50 right", &["abcd", "\t", "efgh"]).1[0].1.advance, 0.0);
        assert_eq!(leader_grid(40.0, 45.0, 10.0), (4, 0));
    }

    /// A field that ends at its stop measures the stop, though placing it at the stop less its width and adding its
    /// segments back ends 250.00000000000003 here: the wrap's fit test against a 250 box saw an overflow.
    #[test]
    fn a_field_ending_at_its_stop_measures_the_stop() {
        let style = default_style();
        let segs: Vec<(&str, &InlineStyle)> = ["name", "\t", "a", "b"].iter().map(|p| (*p, &style)).collect();
        let widths = |t: &str, _: &InlineStyle| match t {
            "name" => 30.0,
            "a" => 0.02,
            "b" => 0.013,
            _ => 1.0,
        };
        let (w, _) = resolve_line_tabs(&segs, &parse_tab_stops("250 right leader '.'").unwrap(), widths);
        assert_eq!(w, 250.0);
    }
}
