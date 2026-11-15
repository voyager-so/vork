//! Greedy first-fit wrap — line-for-line port of `layoutParagraph`
//! (`utils/richtext/layout.ts:17-192`) and `breakTextByGrapheme`
//! (`utils/richtext/measurement.ts:172-195`), in f64 (DECISION §1.4: the TS
//! engine is f64; f32 can flip fit decisions near threshold).
//!
//! Grapheme clusters come from `unicode-segmentation` (UAX#29 extended
//! clusters — the same algorithm class as the production `Intl.Segmenter`
//! with granularity 'grapheme', measurement.ts:3-6).

use crate::hyphenate::{word_breaks, WordBreak, SOFT_HYPHEN};
use crate::measure::TextMeasurer;
use crate::paragraph_format::{Hyphenation, ParagraphFormat};
use crate::tab_stops::{resolve_line_tabs, TabFill, TabStops};
use crate::types::{InlineStyle, Line, LineSegment, Paragraph};
use std::collections::HashMap;
use unicode_segmentation::UnicodeSegmentation;

/// Per-layout-pass measurement cache (`layout.ts:31-42`). The key format is
/// the TS template string; Rust f64 `Display` is shortest-round-trip like JS
/// `Number#toString`, and the key only needs to be injective per (style,text).
fn key_of(text: &str, style: &InlineStyle) -> String {
    // Vertical text: an upright, sideways or combined piece measures differently from the same text horizontal.
    let vertical = style.vertical.cache_key();
    let ruby = style.ruby.cache_key();
    if !vertical.is_empty() || !ruby.is_empty() {
        return format!("{}{vertical}{ruby}", key_of_horizontal(text, style));
    }
    key_of_horizontal(text, style)
}

fn key_of_horizontal(text: &str, style: &InlineStyle) -> String {
    if !style.font_variations.is_empty() {
        // E6.3: axis settings change advances too.
        return format!(
            "{}|fv:{}",
            key_of_features(text, style),
            crate::font_variations::serialize_font_variation_settings(&style.font_variations)
        );
    }
    key_of_features(text, style)
}

fn key_of_features(text: &str, style: &InlineStyle) -> String {
    if !style.font_features.is_empty() {
        // F7.3: features change advances (`tnum`, ligatures), so they key the
        // cache too. Only when set, so the pre-F7.3 key is unchanged.
        return format!(
            "{}|{}|{}|{}|ls:{}|ff:{}|{}",
            style.font_family,
            style.font_size,
            style.font_weight,
            style.font_style,
            style.letter_spacing,
            crate::font_features::serialize_font_feature_settings(&style.font_features),
            text
        );
    }
    format!(
        "{}|{}|{}|{}|ls:{}|{}",
        style.font_family,
        style.font_size,
        style.font_weight,
        style.font_style,
        style.letter_spacing,
        text
    )
}

pub struct CachedMeasurer<'a, M: TextMeasurer> {
    measurer: &'a M,
    cache: HashMap<String, f64>,
}

impl<'a, M: TextMeasurer> CachedMeasurer<'a, M> {
    pub fn new(measurer: &'a M) -> Self {
        CachedMeasurer {
            measurer,
            cache: HashMap::new(),
        }
    }
    pub fn measure(&mut self, text: &str, style: &InlineStyle) -> f64 {
        let key = key_of(text, style);
        if let Some(w) = self.cache.get(&key) {
            return *w;
        }
        let w = self.measurer.measure(text, style);
        self.cache.insert(key, w);
        w
    }

    /// The line box `style` asks for (`InlineStyle::line_box_px`), Auto from the measurer's font metrics.
    pub fn line_box_px(&self, style: &InlineStyle) -> f64 {
        style.line_box_px(|s| self.measurer.auto_line_height_px(s))
    }
}

/// Tab stops: a line's width resolved over its segments (`crate::tab_stops`), and each tab's fill.
fn resolve_segments<M: TextMeasurer>(
    segments: &[LineSegment],
    ts: &TabStops,
    cache: &mut CachedMeasurer<'_, M>,
) -> (f64, Vec<(usize, TabFill)>) {
    let parts: Vec<(&str, &InlineStyle)> = segments
        .iter()
        .map(|s| (s.text.as_str(), &s.style))
        .collect();
    resolve_line_tabs(&parts, ts, |t, s| cache.measure(t, s))
}

/// Split whitespace into pieces with each tab alone (`whitespacePieces`, tab-stop paragraphs only).
fn whitespace_pieces(text: &str, tabs: bool) -> Vec<String> {
    if !tabs || !text.contains('\t') {
        return vec![text.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch == '\t' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push("\t".to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// `breakTextByGrapheme` (measurement.ts): cluster split, then the longest
/// fitting prefix per fragment; minimum 1 cluster per fragment.
pub fn break_text_by_grapheme<F: FnMut(&str) -> f64>(
    text: &str,
    max_width: f64,
    mut measure_fn: F,
) -> Vec<String> {
    let clusters: Vec<&str> = text.graphemes(true).collect();
    let mut fragments: Vec<String> = Vec::new();
    let mut start = 0usize;
    while start < clusters.len() {
        let count = fit_grapheme_prefix(&clusters, start, max_width, &mut measure_fn);
        fragments.push(clusters[start..start + count].concat());
        start += count;
    }
    fragments
}

/// `fitGraphemePrefix` (measurement.ts): how many of `clusters` from `start`
/// make the longest prefix within `max_width` (at least one). Doubling, then
/// bisecting, measures O(k log k) text for a k-cluster fragment however long
/// the rest is; a bisection over the whole rest measured it all for every
/// line, and a ~1M-character word hung layout (#2138).
pub fn fit_grapheme_prefix<F: FnMut(&str) -> f64>(
    clusters: &[&str],
    start: usize,
    max_width: f64,
    measure_fn: &mut F,
) -> usize {
    if start >= clusters.len() {
        return 0;
    }
    let rest = clusters.len() - start;
    let mut fits = |count: usize| measure_fn(&clusters[start..start + count].concat()) <= max_width;
    let mut low = 0usize; // the longest prefix known to fit
    let mut high = rest + 1; // the shortest known not to
    let mut probe = 1usize;
    loop {
        if !fits(probe) {
            high = probe;
            break;
        }
        low = probe;
        if probe == rest {
            break;
        }
        probe = (probe * 2).min(rest);
    }
    while high - low > 1 {
        let mid = (low + high) / 2;
        if fits(mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    low.max(1)
}

struct WrapState<'a, 'c, M: TextMeasurer> {
    lines: Vec<Line>,
    current: Line,
    current_width: f64,
    pending_space: bool,
    pending_space_style: Option<InlineStyle>,
    pending_space_text: Option<String>,
    current_word_segments: Vec<LineSegment>,
    /// E6.3: the wrap width of line `i` (the greedy wrap uses one width for
    /// every line; balance and pretty narrow some).
    widths: &'c dyn Fn(usize) -> f64,
    /// A word was broken between graphemes because it fit no line.
    broke_word: bool,
    /// Area type (`crate::area`): asked with (line index, word width) when a word doesn't fit an empty line; true
    /// leaves that line empty (a later line of the shape is wide enough) — Illustrator doesn't split a word at the
    /// narrow top of an ellipse. `None` for box text.
    skip: Option<&'c dyn Fn(usize, f64) -> bool>,
    /// The indices of the lines `skip` left empty.
    skipped: Vec<usize>,
    cache: &'c mut CachedMeasurer<'a, M>,
    /// The paragraph's tab stops: a `\t` is a segment of its own and the line's width is resolved, not summed.
    tab_stops: Option<&'c TabStops>,
    /// The paragraph's indents and hyphenation (`crate::paragraph_format`).
    format: ParagraphFormat,
    /// Consecutive lines that ended in a hyphen, for the hyphen limit.
    hyphen_run: u32,
    /// The line being filled ends at a hyphenation break.
    line_hyphenated: bool,
}

impl<'a, M: TextMeasurer> WrapState<'a, '_, M> {
    /// The width of a line of `segments` with its tabs resolved.
    fn extent(&mut self, segments: &[LineSegment]) -> f64 {
        match self.tab_stops {
            Some(ts) => resolve_segments(segments, ts, self.cache).0,
            None => 0.0,
        }
    }

    /// `addWhitespace`: the run's pieces, each tab alone.
    fn add_whitespace(&mut self, text: &str, style: &InlineStyle) {
        let mut offset = 0;
        for piece in whitespace_pieces(text, self.tab_stops.is_some()) {
            let end = offset + piece.chars().count();
            self.add_segment(&piece, &style.source_slice(offset..end));
            offset = end;
        }
    }

    /// The wrap width of the line being filled: its container's, less the paragraph's indents.
    fn max_width(&self) -> f64 {
        let w = (self.widths)(self.lines.len());
        if self.format.has_indents() {
            w - self.inset_now()
        } else {
            w
        }
    }

    /// How much the paragraph's indents take from the line being filled (the first-line indent on its first line;
    /// area type's skipped lines don't count).
    fn inset_now(&self) -> f64 {
        self.format.inset(self.lines.len() == self.skipped.len())
    }

    /// `pushLine` (layout.ts:44-53). `soft`: the wrap ended the line (see
    /// `Line::soft_wrap`).
    fn push_line(&mut self, soft: bool) {
        let hyphenated = std::mem::take(&mut self.line_hyphenated);
        if self.current.segments.is_empty() {
            return;
        }
        self.hyphen_run = if hyphenated { self.hyphen_run + 1 } else { 0 };
        self.current.soft_wrap = soft;
        if self.current.line_height_px == 0.0 {
            let s = &self.current.segments[0].style;
            self.current.line_height_px = self.cache.line_box_px(s);
        }
        self.lines.push(std::mem::replace(
            &mut self.current,
            Line::new(Vec::new(), 0.0, false),
        ));
        self.current_width = 0.0;
    }

    /// `addSegment` (layout.ts:55-61).
    fn add_segment(&mut self, text: &str, style: &InlineStyle) {
        let lh = self.cache.line_box_px(style);
        if self.current.line_height_px < lh {
            self.current.line_height_px = lh;
        }
        self.current
            .segments
            .push(LineSegment::new(text, style.clone()));
        if self.tab_stops.is_some() {
            let segs = std::mem::take(&mut self.current.segments);
            self.current_width = self.extent(&segs);
            self.current.segments = segs;
        } else {
            self.current_width += self.cache.measure(text, style);
        }
    }

    /// Area type: leave empty the lines too narrow for a word `width` wide while a later one takes it.
    fn skip_unfit_lines(&mut self, width: f64) {
        let Some(skip) = self.skip else { return };
        while self.current.segments.is_empty()
            && width > self.max_width()
            && skip(
                self.lines.len(),
                if self.format.has_indents() {
                    width + self.inset_now()
                } else {
                    width
                },
            )
        {
            self.skipped.push(self.lines.len());
            self.lines.push(Line::new(Vec::new(), 0.0, false));
            self.current_width = 0.0;
        }
    }

    /// `measureWordGroup` (layout.ts:63-69).
    fn measure_word_group(&mut self) -> f64 {
        let segs = self.current_word_segments.clone();
        let mut w = 0.0;
        for seg in &segs {
            w += self.cache.measure(&seg.text, &seg.style);
        }
        w
    }

    /// `addWordGroup` (layout.ts:71-73).
    fn add_word_group(&mut self) {
        let segs = std::mem::take(&mut self.current_word_segments);
        for seg in &segs {
            self.add_segment(&seg.text, &seg.style);
        }
        self.current_word_segments = segs;
    }

    /// `breakWordGroupOnEmptyLine` (layout.ts).
    fn break_word_group_on_empty_line(&mut self) {
        // Normal wrapping has no emergency break: an unbreakable group overflows intact.
        use crate::paragraph_format::{LineBreak, WordBreak};
        if self.format.breaking.word_break != WordBreak::BreakAll
            && self.format.breaking.line_break != LineBreak::Anywhere
        {
            self.add_word_group();
            return;
        }
        let segs = std::mem::take(&mut self.current_word_segments);
        for seg in &segs {
            let clusters: Vec<&str> = seg.text.graphemes(true).collect();
            let mut start = 0usize;
            let mut cp_offset = 0;
            while start < clusters.len() {
                let available = (self.max_width() - self.current_width).max(0.0);
                if available <= 0.0 {
                    self.push_line(true);
                }
                let budget = (self.max_width() - self.current_width).max(0.0);
                let style = seg.style.clone();
                let count = {
                    let cache = &mut *self.cache;
                    fit_grapheme_prefix(&clusters, start, budget, &mut |t: &str| cache.measure(t, &style))
                };
                if start + count < clusters.len() {
                    self.broke_word = true;
                }
                let fragment = clusters[start..start + count].concat();
                let cp_end = cp_offset + fragment.chars().count();
                self.add_segment(&fragment, &seg.style.source_slice(cp_offset..cp_end));
                cp_offset = cp_end;
                start += count;
                if start < clusters.len() {
                    self.push_line(true);
                }
            }
        }
        self.current_word_segments = segs;
    }

    /// `flushWordGroup` (layout.ts:93-127).
    fn flush_word_group(&mut self) {
        if self.current_word_segments.is_empty() {
            return;
        }
        if let Some(h) = hyphenation_for(&self.format, &self.current_word_segments) {
            self.flush_word_group_hyphenating(h);
            return;
        }
        let group_width = self.measure_word_group();
        let mut space_width = 0.0;
        if self.pending_space && !self.current.segments.is_empty() {
            if let Some(style) = self.pending_space_style.clone() {
                let space_run = self
                    .pending_space_text
                    .clone()
                    .unwrap_or_else(|| " ".to_string());
                space_width = self.cache.measure(&space_run, &style);
            }
        }

        let max_width = self.max_width();
        // With tab stops the candidate line's width is resolved whole (`flushWordGroup`).
        let fits_current = if self.tab_stops.is_some() {
            let mut candidate = self.current.segments.clone();
            if self.pending_space && !self.current.segments.is_empty() {
                if let Some(style) = &self.pending_space_style {
                    let text = self.pending_space_text.as_deref().unwrap_or(" ");
                    for piece in whitespace_pieces(text, true) {
                        candidate.push(LineSegment::new(piece, style.clone()));
                    }
                }
            }
            candidate.extend(self.current_word_segments.iter().cloned());
            self.extent(&candidate) <= max_width
        } else {
            self.current_width + space_width + group_width <= max_width
        };
        if !self.current.segments.is_empty() && fits_current {
            if self.pending_space {
                if let Some(style) = self.pending_space_style.clone() {
                    let text = self
                        .pending_space_text
                        .clone()
                        .unwrap_or_else(|| " ".to_string());
                    self.add_whitespace(&text, &style);
                }
            }
            self.add_word_group();
        } else if self.current.segments.is_empty() {
            self.skip_unfit_lines(group_width);
            if group_width <= self.max_width() {
                self.add_word_group();
            } else {
                self.break_word_group_on_empty_line();
            }
        } else {
            let tabs = self.pending_tabs();
            self.push_line(true);
            self.carry_tabs(tabs, &self.current_word_segments.clone());
            if !self.current.segments.is_empty() {
                self.add_word_group();
            } else {
                self.skip_unfit_lines(group_width);
                if group_width <= self.max_width() {
                    self.add_word_group();
                } else {
                    self.break_word_group_on_empty_line();
                }
            }
        }

        self.current_word_segments.clear();
        self.pending_space = false;
        self.pending_space_style = None;
        self.pending_space_text = None;
    }

    /// Whether `extra` fits after the line being filled (and its pending space): `flushWordGroup`'s fit test.
    fn fits_current(&mut self, extra: &[LineSegment]) -> bool {
        let max_width = self.max_width();
        if self.tab_stops.is_some() {
            let mut candidate = self.current.segments.clone();
            if self.pending_space && !self.current.segments.is_empty() {
                if let Some(style) = &self.pending_space_style {
                    let text = self.pending_space_text.as_deref().unwrap_or(" ");
                    for piece in whitespace_pieces(text, true) {
                        candidate.push(LineSegment::new(piece, style.clone()));
                    }
                }
            }
            candidate.extend(extra.iter().cloned());
            return self.extent(&candidate) <= max_width;
        }
        let mut space_width = 0.0;
        if self.pending_space && !self.current.segments.is_empty() {
            if let Some(style) = self.pending_space_style.clone() {
                let text = self
                    .pending_space_text
                    .clone()
                    .unwrap_or_else(|| " ".to_string());
                space_width = self.cache.measure(&text, &style);
            }
        }
        let mut w = 0.0;
        for seg in extra {
            w += self.cache.measure(&seg.text, &seg.style);
        }
        self.current_width + space_width + w <= max_width
    }

    /// Add the pending space (if any) before a word on a line with content.
    fn add_pending_space(&mut self) {
        if self.pending_space {
            if let Some(style) = self.pending_space_style.clone() {
                let text = self
                    .pending_space_text
                    .clone()
                    .unwrap_or_else(|| " ".to_string());
                self.add_whitespace(&text, &style);
            }
        }
    }

    /// The tabs of the pending whitespace (tab-stop paragraphs only), with its style.
    fn pending_tabs(&self) -> Option<(String, InlineStyle)> {
        self.tab_stops?;
        let text = self
            .pending_space_text
            .as_deref()
            .filter(|_| self.pending_space)?;
        let tabs: String = text.chars().filter(|&c| c == '\t').collect();
        let mut style = self.pending_space_style.clone()?;
        style.source = style.source.filter(text, |c| c == '\t');
        (!tabs.is_empty()).then_some((tabs, style))
    }

    /// A word that wraps after a tab takes the tab to the new line, so it still reaches its stop (and leader): a
    /// right-aligned price that doesn't fit beside its label lands at the stop on the next line, not at its start.
    /// Only when the tab and the word fit that line; otherwise the word moves down alone, as before.
    fn carry_tabs(&mut self, tabs: Option<(String, InlineStyle)>, word: &[LineSegment]) {
        let Some((tabs, style)) = tabs else { return };
        if !self.current.segments.is_empty() {
            return;
        }
        let mut candidate: Vec<LineSegment> = tabs
            .chars().enumerate()
            .map(|(i, t)| LineSegment::new(t.to_string(), style.source_slice(i..i+1)))
            .collect();
        candidate.extend(word.iter().cloned());
        if self.extent(&candidate) <= self.max_width() {
            self.add_whitespace(&tabs, &style);
        }
    }

    fn clear_pending_space(&mut self) {
        self.pending_space = false;
        self.pending_space_style = None;
        self.pending_space_text = None;
    }

    /// The largest break of `segs` whose head (and hyphen) fits: on the line being filled (`on_current`) or on an
    /// empty line. Returns (head with its hyphen, tail, the tail's breaks).
    fn best_split(
        &mut self,
        segs: &[LineSegment],
        breaks: &[WordBreak],
        on_current: bool,
    ) -> Option<(Vec<LineSegment>, Vec<LineSegment>, Vec<WordBreak>)> {
        for (i, b) in breaks.iter().enumerate().rev() {
            let (mut head, tail) = split_segments(segs, b.offset);
            if head.is_empty() || tail.is_empty() {
                continue;
            }
            if b.hyphen {
                let style = head
                    .last()
                    .map(|s| s.style.clone())
                    .unwrap_or_else(crate::types::default_style);
                let mut hy = LineSegment::new("-", style.without_source());
                hy.hyphen = true;
                head.push(hy);
            }
            let fits = if on_current {
                self.fits_current(&head)
            } else {
                let mut w = 0.0;
                for seg in &head {
                    w += self.cache.measure(&seg.text, &seg.style);
                }
                w <= self.max_width()
            };
            if fits {
                let rest = breaks[i + 1..]
                    .iter()
                    .map(|r| WordBreak {
                        offset: r.offset - b.offset,
                        hyphen: r.hyphen,
                    })
                    .collect();
                return Some((head, tail, rest));
            }
        }
        None
    }

    /// Add a hyphenated head (its hyphen flagged) and end the line there.
    fn add_hyphenated_head(&mut self, head: &[LineSegment]) {
        for seg in head {
            self.add_segment(&seg.text, &seg.style);
            if seg.hyphen {
                if let Some(last) = self.current.segments.last_mut() {
                    last.hyphen = true;
                }
            }
        }
        self.line_hyphenated = true;
        self.push_line(true);
    }

    /// `flushWordGroup` in a hyphenating paragraph (`crate::hyphenate`): the soft hyphens come out of the word, and a
    /// word that doesn't fit breaks at its last hyphenation point that does (the hyphen limit allowing) before it
    /// moves down whole, or, on an empty line, before it breaks between letters.
    fn flush_word_group_hyphenating(&mut self, h: Hyphenation) {
        let raw = std::mem::take(&mut self.current_word_segments);
        let text: String = raw.iter().map(|s| s.text.as_str()).collect();
        let (_, mut breaks) = word_breaks(&text, &h, self.format.lang);
        let mut segs: Vec<LineSegment> = raw
            .into_iter()
            .map(|mut s| {
                if s.text.contains(SOFT_HYPHEN) {
                    s.style.source = s.style.source.filter(&s.text, |c| c != SOFT_HYPHEN);
                    s.text = s.text.replace(SOFT_HYPHEN, "");
                }
                s
            })
            .filter(|s| !s.text.is_empty())
            .collect();
        if segs.is_empty() {
            return;
        }
        loop {
            if !self.current.segments.is_empty() {
                if self.fits_current(&segs) {
                    self.add_pending_space();
                    for seg in &segs {
                        self.add_segment(&seg.text, &seg.style);
                    }
                    break;
                }
                if h.allows(self.hyphen_run) {
                    if let Some((head, tail, rest)) = self.best_split(&segs, &breaks, true) {
                        self.add_pending_space();
                        self.add_hyphenated_head(&head);
                        self.clear_pending_space();
                        segs = tail;
                        breaks = rest;
                        continue;
                    }
                }
                let tabs = self.pending_tabs();
                self.push_line(true);
                self.clear_pending_space();
                self.carry_tabs(tabs, &segs);
                if !self.current.segments.is_empty() {
                    for seg in &segs {
                        self.add_segment(&seg.text, &seg.style);
                    }
                    break;
                }
            }
            let mut group_width = 0.0;
            for seg in &segs {
                group_width += self.cache.measure(&seg.text, &seg.style);
            }
            self.skip_unfit_lines(group_width);
            if group_width <= self.max_width() {
                for seg in &segs {
                    self.add_segment(&seg.text, &seg.style);
                }
                break;
            }
            if h.allows(self.hyphen_run) {
                if let Some((head, tail, rest)) = self.best_split(&segs, &breaks, false) {
                    self.add_hyphenated_head(&head);
                    segs = tail;
                    breaks = rest;
                    continue;
                }
            }
            self.current_word_segments = segs;
            self.break_word_group_on_empty_line();
            break;
        }
        self.current_word_segments.clear();
        self.clear_pending_space();
    }
}

/// Split a word group's segments after `offset` chars: (head, tail).
fn split_segments(segs: &[LineSegment], offset: usize) -> (Vec<LineSegment>, Vec<LineSegment>) {
    let mut head = Vec::new();
    let mut tail = Vec::new();
    let mut seen = 0usize;
    for seg in segs {
        let n = seg.text.chars().count();
        if seen + n <= offset {
            head.push(seg.clone());
        } else if seen >= offset {
            tail.push(seg.clone());
        } else {
            let k = offset - seen;
            let byte = seg
                .text
                .char_indices()
                .nth(k)
                .map(|(b, _)| b)
                .unwrap_or(seg.text.len());
            head.push(LineSegment::new(&seg.text[..byte], seg.style.source_slice(0..k)));
            tail.push(LineSegment::new(&seg.text[byte..], seg.style.source_slice(k..n)));
        }
        seen += n;
    }
    (head, tail)
}

/// Vertical text (`crate::vertical::split_runs`) cuts a run into upright, sideways and combined pieces, often inside a
/// word ("年、HTML。"): between two pieces the wrap may break where the segmenter would break the text whole (kinsoku
/// included). `prev` is the word group's last segment, `word` the piece's first word. Horizontal runs keep their
/// historical behaviour (a word continues across runs).
fn break_between_pieces(
    prev: Option<&LineSegment>,
    style: &InlineStyle,
    word: &str,
    word_segmenter: Option<WordSegmenter<'_>>,
    rules: crate::paragraph_format::LineBreaking,
) -> bool {
    let (Some(seg), Some(prev)) = (word_segmenter, prev) else {
        return false;
    };
    let piece = |s: &InlineStyle| s.vertical.glyph.is_some() || s.ruby.text.is_some();
    if !piece(&prev.style) && !piece(style) {
        return false;
    }
    let (Some(a), Some(b)) = (prev.text.chars().last(), word.chars().next()) else {
        return false;
    };
    seg(&format!("{a}{b}"), rules).len() > 1
}

/// `WordSegmenter` (`layout.ts:6-14`) — optional sub-segmenter for
/// whitespace-free word runs (the vector engine's CJK/kinsoku tokenizer,
/// `cjkLineBreaking.segmentTextForWrap`), under the paragraph's `word-break` /
/// `line-break` (`ParagraphFormat::breaking`). Tokens must be non-empty and
/// concatenate back to the input word. `None` ⇒ words wrap whole,
/// byte-for-byte the historical (MSDF) behavior.
pub type WordSegmenter<'a> =
    &'a dyn Fn(&str, crate::paragraph_format::LineBreaking) -> Vec<String>;

/// Whether a line may break between the end of `before` and the start of `after`, two pieces of one space-free word
/// in different runs (`breaksBetween`): what the segmenter says of their two characters. Never after whitespace (a
/// no-break space inside the word).
fn breaks_between(
    seg: WordSegmenter<'_>,
    rules: crate::paragraph_format::LineBreaking,
    before: &str,
    after: &str,
) -> bool {
    let (Some(a), Some(b)) = (before.chars().last(), after.chars().next()) else {
        return false;
    };
    if crate::vector::cjk::js_whitespace(a) || crate::vector::cjk::js_whitespace(b) {
        return false;
    }
    let mut pair = String::with_capacity(8);
    pair.push(a);
    pair.push(b);
    seg(&pair, rules).len() > 1
}

/// `layoutParagraph` (layout.ts:17-192) without a word segmenter — the MSDF
/// path (the optional `wordSegmenter` is vector-engine-only, layout.ts:6-14).
pub fn layout_paragraph<M: TextMeasurer>(
    para: &Paragraph,
    max_width: f64,
    measurer: &M,
) -> Vec<Line> {
    layout_paragraph_with_segmenter(para, max_width, measurer, None)
}

/// `layoutParagraph` (layout.ts:17-192) with the optional `wordSegmenter`
/// (layout.ts:172-179): each whitespace-delimited word is sub-segmented and a
/// break opportunity (`flushWordGroup`) is inserted between tokens. The
/// paragraph's `text-wrap` (E6.3) picks greedy, balanced or pretty lines
/// (`crate::text_wrap`).
pub fn layout_paragraph_with_segmenter<M: TextMeasurer>(
    para: &Paragraph,
    max_width: f64,
    measurer: &M,
    word_segmenter: Option<WordSegmenter<'_>>,
) -> Vec<Line> {
    layout_paragraph_with_policy(para, max_width, measurer, word_segmenter, Default::default())
}

/// Wrap with the caller's line-box convention; shaping and line breaking are unchanged.
pub(crate) fn layout_paragraph_with_policy<M: TextMeasurer>(
    para: &Paragraph,
    max_width: f64,
    measurer: &M,
    word_segmenter: Option<WordSegmenter<'_>>,
    policy: crate::types::LineHeightPolicy,
) -> Vec<Line> {
    // CJK spacing (`crate::cjk_spacing`): pair adjustments before the wrap, the line edges after it.
    let spaced = crate::cjk_spacing::apply_char_spacing(para);
    let para = spaced.as_ref().unwrap_or(para);
    let mut cache = CachedMeasurer::new(measurer);
    // The every-line composer (`crate::composer`) replaces the greedy wrap and `text-wrap`; tab stops keep it.
    if let Some(mut lines) = compose_paragraph(para, &|_| max_width, 0, &mut cache, word_segmenter)
    {
        for line in &mut lines {
            line.line_height_px = policy.resolve(line.line_height_px);
        }
        crate::cjk_spacing::trim_lines(&mut lines, &para.format);
        return lines;
    }
    let mut wrap =
        |widths: &dyn Fn(usize) -> f64| greedy_wrap(para, widths, &mut cache, word_segmenter);
    let mut lines = crate::text_wrap::wrap_lines(para.wrap, max_width, &mut wrap);
    // Box text follows its caller's convention. Area type keeps exact
    // leading (Illustrator): it wraps through `greedy_wrap_area`.
    for line in &mut lines {
        line.line_height_px = policy.resolve(line.line_height_px);
    }
    crate::cjk_spacing::trim_lines(&mut lines, &para.format);
    lines
}

/// One greedy pass with a per-line width: the lines, and whether a word had
/// to break between graphemes.
fn greedy_wrap<M: TextMeasurer>(
    para: &Paragraph,
    widths: &dyn Fn(usize) -> f64,
    cache: &mut CachedMeasurer<'_, M>,
    word_segmenter: Option<WordSegmenter<'_>>,
) -> (Vec<Line>, bool) {
    let (lines, broke, _) = greedy_wrap_skipping(para, widths, None, cache, word_segmenter);
    (lines, broke)
}

/// The greedy wrap of area type (`crate::area`): per-line widths from the shape, and `skip` to leave a line empty
/// when a word doesn't fit it but fits a later one. Returns the lines (skipped ones empty, in place) and the
/// skipped indices.
pub(crate) fn greedy_wrap_area<M: TextMeasurer>(
    para: &Paragraph,
    widths: &dyn Fn(usize) -> f64,
    skip: &dyn Fn(usize, f64) -> bool,
    // The first line past the shape: from it on every line is the overflow width (`usize::MAX`: unknown).
    uniform_from: usize,
    cache: &mut CachedMeasurer<'_, M>,
    word_segmenter: Option<WordSegmenter<'_>>,
) -> (Vec<Line>, Vec<usize>) {
    // The every-line composer fits the shape's widths line by line too; when no breaks fit (a word wider than a
    // line of the shape), the greedy wrap leaves that line empty instead.
    if let Some(lines) = compose_paragraph(para, widths, uniform_from, cache, word_segmenter) {
        return (lines, Vec::new());
    }
    let (lines, _, skipped) = greedy_wrap_skipping(para, widths, Some(skip), cache, word_segmenter);
    (lines, skipped)
}

/// Each line's indents (`Line::indent_start` / `indent_end`): the first line not skipped takes the first-line indent.
fn set_indents(lines: &mut [Line], format: &ParagraphFormat, skipped: &[usize]) {
    if !format.has_indents() {
        return;
    }
    let first = (0..lines.len()).find(|i| !skipped.contains(i));
    for (i, line) in lines.iter_mut().enumerate() {
        line.indent_start = format.start_indent(Some(i) == first);
        line.indent_end = format.indent_right;
    }
}

/// A composer token: what the wrap's tokenizer finds between break opportunities.
enum Token {
    /// Whitespace at a chunk's start, kept on its first line (the greedy wrap keeps it too).
    Lead(LineSegment),
    /// A run of spaces between words.
    Space(LineSegment),
    /// A word group: segments the wrap keeps together (a zero-width break between two adjacent ones: CJK tokens).
    Word(Vec<LineSegment>),
}

/// What an item came from, to rebuild the line's segments.
#[derive(Clone)]
enum Source {
    /// A piece of a word segment; pieces of one segment (`origin`) join again on a line.
    Piece {
        seg: LineSegment,
        origin: usize,
    },
    Space(LineSegment),
    /// A hyphenation point: the hyphen's style when it draws one.
    Break(Option<crate::types::InlineStyle>),
    End,
}

/// The every-line composer (`crate::composer`) over a paragraph whose line `k` is `widths(k)` wide (less its
/// indents): `None` when the paragraph uses the single-line composer, has tab stops, or no breaks fit. Mirrors
/// `composeParagraph` (`utils/richtext/layout.ts`).
pub(crate) fn compose_paragraph<M: TextMeasurer>(
    para: &Paragraph,
    widths: &dyn Fn(usize) -> f64,
    // The paragraph line from which `widths` is one width (0: the box's, every line; area type: past its shape's
    // lines; `usize::MAX`: varies to the end). Past it the composer's work stays linear in the paragraph.
    uniform_from: usize,
    cache: &mut CachedMeasurer<'_, M>,
    word_segmenter: Option<WordSegmenter<'_>>,
) -> Option<Vec<Line>> {
    use crate::paragraph_format::Composer;
    if para.format.composer != Composer::EveryLine || para.tab_stops.is_some() {
        return None;
    }
    let mut lines: Vec<Line> = Vec::new();
    let mut tokens: Vec<Token> = Vec::new();
    let mut group: Vec<LineSegment> = Vec::new();
    fn flush(group: &mut Vec<LineSegment>, tokens: &mut Vec<Token>) {
        if !group.is_empty() {
            tokens.push(Token::Word(std::mem::take(group)));
        }
    }
    for run in &para.runs {
        let mut chunk_offset = 0;
        for (ci, chunk) in run.text.split('\n').enumerate() {
            let chunk_base = chunk_offset;
            chunk_offset += chunk.chars().count() + 1;
            if ci > 0 {
                flush(&mut group, &mut tokens);
                if !has_content(&tokens) {
                    tokens.clear();
                    // A hard break after nothing: an empty line (the greedy wrap's too).
                    let mut line = Line::new(Vec::new(), cache.line_box_px(&run.style), false);
                    line.indent_start = para.format.start_indent(lines.is_empty());
                    line.indent_end = para.format.indent_right;
                    lines.push(line);
                } else {
                    compose_chunk(
                        para,
                        std::mem::take(&mut tokens),
                        widths,
                        uniform_from,
                        cache,
                        &mut lines,
                    )?;
                }
            }
            let chars: Vec<char> = chunk.chars().collect();
            let mut i = 0usize;
            while i < chars.len() {
                let ch = chars[i];
                // U+200B is a free break, never a measured or drawn segment.
                if ch == '\u{200B}' {
                    flush(&mut group, &mut tokens);
                    i += 1;
                    continue;
                }
                if ch == '\u{00A0}' {
                    let mut j = i + 1;
                    while j < chars.len() && chars[j] == '\u{00A0}' {
                        j += 1;
                    }
                    group.push(LineSegment::new(
                        chars[i..j].iter().collect::<String>(),
                        run.style.source_slice(chunk_base+i..chunk_base+j),
                    ));
                    i = j;
                    continue;
                }
                if ch == ' ' || ch == '\t' {
                    let mut j = i + 1;
                    while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
                        j += 1;
                    }
                    flush(&mut group, &mut tokens);
                    let seg =
                        LineSegment::new(chars[i..j].iter().collect::<String>(), run.style.source_slice(chunk_base+i..chunk_base+j));
                    match tokens.last() {
                        None => tokens.push(Token::Lead(seg)),
                        // Spaces in two runs in a row: the later run's (the greedy wrap keeps one pending space).
                        Some(Token::Space(_)) => {
                            tokens.pop();
                            tokens.push(Token::Space(seg));
                        }
                        Some(_) => tokens.push(Token::Space(seg)),
                    }
                    i = j;
                    continue;
                }
                let mut j = i + 1;
                while j < chars.len()
                    && chars[j] != ' '
                    && chars[j] != '\t'
                    && chars[j] != '\u{00A0}'
                    && chars[j] != '\u{200B}'
                {
                    j += 1;
                }
                let word: String = chars[i..j].iter().collect();
                if i == 0 && break_between_pieces(group.last(), &run.style, &word, word_segmenter, para.format.breaking) {
                    flush(&mut group, &mut tokens);
                }
                match word_segmenter.filter(|_| !run.style.unbreakable()) {
                    Some(seg) => {
                        let mut token_offset = chunk_base + i;
                        for (k, token) in seg(&word, para.format.breaking).into_iter().enumerate() {
                            let token_end = token_offset + token.chars().count();
                            let token_style = run.style.source_slice(token_offset..token_end);
                            token_offset = token_end;
                            if k > 0
                                || group.last().is_some_and(|last| {
                                    breaks_between(seg, para.format.breaking, &last.text, &token)
                                })
                            {
                                flush(&mut group, &mut tokens);
                            }
                            group.push(LineSegment::new(token, token_style));
                        }
                    }
                    None => group.push(LineSegment::new(word, run.style.source_slice(chunk_base+i..chunk_base+j))),
                }
                i = j;
            }
        }
    }
    flush(&mut group, &mut tokens);
    if has_content(&tokens) {
        compose_chunk(para, tokens, widths, uniform_from, cache, &mut lines)?;
    }
    Some(lines)
}

/// How a word group hyphenates: as its paragraph does, else at its own soft hyphens alone (CSS `hyphens: manual`,
/// the initial value: a word with soft hyphens breaks only there, `crate::hyphenate`), with no hyphen limit. A word
/// with none, in a paragraph that doesn't hyphenate, doesn't break.
fn hyphenation_for(format: &ParagraphFormat, word: &[LineSegment]) -> Option<Hyphenation> {
    format.hyphenate.or_else(|| {
        word.iter()
            .any(|s| s.text.contains(SOFT_HYPHEN))
            .then_some(Hyphenation {
                limit: 0,
                ..Hyphenation::DEFAULT
            })
    })
}

/// Whether a chunk's tokens lay anything out (a word of soft hyphens alone doesn't, as in the greedy wrap).
fn has_content(tokens: &[Token]) -> bool {
    tokens.iter().any(|t| match t {
        Token::Lead(_) => true,
        Token::Space(_) => false,
        Token::Word(segs) => segs
            .iter()
            .any(|s| s.text.chars().any(|c| c != SOFT_HYPHEN)),
    })
}

/// Compose one chunk (the text between hard breaks) onto `lines`.
fn compose_chunk<M: TextMeasurer>(
    para: &Paragraph,
    mut tokens: Vec<Token>,
    widths: &dyn Fn(usize) -> f64,
    uniform_from: usize,
    cache: &mut CachedMeasurer<'_, M>,
    lines: &mut Vec<Line>,
) -> Option<()> {
    use crate::composer::{
        break_lines, Item, ItemKind, Params, HYPHEN_PENALTY, RAGGED_STRETCH_EM, SPACE_EM,
    };
    let format = &para.format;
    let justify = para.align == "justify";
    let ws = format.word_spacing;
    let factor = ws.desired / 100.0;
    // Whether any word may break at a hyphen (a hyphenating paragraph's, or a word's soft hyphens): the composer's
    // passes that use those breaks run.
    let mut hyphen_points = format.hyphenate.is_some();
    // Trailing spaces never reach a line (the greedy wrap drops a pending space at a break too).
    while matches!(tokens.last(), Some(Token::Space(_))) {
        tokens.pop();
    }
    let mut items: Vec<Item> = Vec::new();
    let mut sources: Vec<Source> = Vec::new();
    let mut origin = 0usize;
    let mut prev_word = false;
    for token in tokens {
        match token {
            Token::Lead(seg) => {
                items.push(Item::boxed(cache.measure(&seg.text, &seg.style)));
                sources.push(Source::Piece { seg, origin });
                origin += 1;
                prev_word = false;
            }
            Token::Space(seg) => {
                let w0 = cache.measure(&seg.text, &seg.style);
                let (stretch, shrink) = if justify {
                    (
                        w0 * ((ws.max - ws.desired) / 100.0),
                        w0 * ((ws.desired - ws.min) / 100.0),
                    )
                } else {
                    (0.0, 0.0)
                };
                items.push(Item::glue(w0 * factor, stretch, shrink));
                sources.push(Source::Space(seg));
                prev_word = false;
            }
            Token::Word(raw) => {
                let (segs, breaks) = match hyphenation_for(format, &raw) {
                    Some(h) => {
                        let text: String = raw.iter().map(|s| s.text.as_str()).collect();
                        let (_, breaks) = word_breaks(&text, &h, format.lang);
                        hyphen_points |= !breaks.is_empty();
                        let segs: Vec<LineSegment> = raw
                            .into_iter()
                            .map(|mut s| {
                                if s.text.contains(SOFT_HYPHEN) {
                                    s.style.source = s.style.source.filter(&s.text, |c| c != SOFT_HYPHEN);
                    s.text = s.text.replace(SOFT_HYPHEN, "");
                                }
                                s
                            })
                            .filter(|s| !s.text.is_empty())
                            .collect();
                        (segs, breaks)
                    }
                    None => (raw, Vec::new()),
                };
                if segs.is_empty() {
                    continue;
                }
                if prev_word {
                    // Two word groups with no space between (CJK tokens): a free break.
                    items.push(Item::penalty(0.0, 0.0, false, false));
                    sources.push(Source::Break(None));
                }
                // Pieces between the breaks, with a flagged penalty at each; a piece keeps its segment's origin.
                let spans: Vec<(usize, usize)> = {
                    let mut at = 0usize;
                    segs.iter()
                        .map(|seg| {
                            let n = seg.text.chars().count();
                            at += n;
                            (at - n, at)
                        })
                        .collect()
                };
                let total = spans.last().map_or(0, |s| s.1);
                let mut cuts: Vec<usize> = breaks.iter().map(|b| b.offset).collect();
                cuts.push(total);
                let mut from = 0usize;
                for (r, &to) in cuts.iter().enumerate() {
                    for (idx, (seg, &(s0, s1))) in segs.iter().zip(&spans).enumerate() {
                        let (a, b) = (from.max(s0), to.min(s1));
                        if a >= b {
                            continue;
                        }
                        let text: String = seg.text.chars().skip(a - s0).take(b - a).collect();
                        let piece = LineSegment::new(text, seg.style.source_slice(a-s0..b-s0));
                        items.push(Item::boxed(cache.measure(&piece.text, &piece.style)));
                        sources.push(Source::Piece {
                            seg: piece,
                            origin: origin + idx,
                        });
                    }
                    if let Some(brk) = breaks.get(r) {
                        // The hyphen takes the style of the letter before the break.
                        let style = segs
                            .iter()
                            .zip(&spans)
                            .find(|(_, &(s0, s1))| s0 < brk.offset && brk.offset <= s1)
                            .map(|(seg, _)| seg.style.clone())
                            .filter(|_| brk.hyphen);
                        let width = style.as_ref().map_or(0.0, |st| cache.measure("-", st));
                        items.push(Item::penalty(width, HYPHEN_PENALTY, true, brk.hyphen));
                        sources.push(Source::Break(style));
                    }
                    from = to;
                }
                origin += segs.len();
                prev_word = true;
            }
        }
    }
    items.push(Item::penalty(0.0, f64::NEG_INFINITY, false, false));
    sources.push(Source::End);

    let first_line = lines.len();
    let base_size = para
        .runs
        .first()
        .map(|r| r.style.line_font_size())
        .or_else(|| para.base_style.as_ref().map(|s| s.line_font_size()))
        .unwrap_or(16.0);
    let params = Params {
        justify,
        ragged_stretch: RAGGED_STRETCH_EM * base_size,
        hyphen_limit: format.hyphenate.map_or(0, |h| h.limit),
        // One space's stretch (at least 1 % of one), for a line with no spaces.
        lone_stretch: SPACE_EM * base_size * ((ws.max - ws.desired) / 100.0).max(0.01),
        // One width from the paragraph's line `uniform_from` on, less the first line's indent (line 0), in this
        // chunk's line numbers.
        uniform_from: uniform_from.max(1).saturating_sub(first_line),
    };
    let line_width = |k: usize| {
        let i = first_line + k;
        widths(i) - format.inset(i == 0)
    };
    let breaks = break_lines(&items, &line_width, &params, hyphen_points)?;

    let mut start = 0usize;
    for (k, &b) in breaks.iter().enumerate() {
        let mut segs: Vec<LineSegment> = Vec::new();
        let mut last_origin: Option<usize> = None;
        for src in &sources[start..b] {
            match src {
                Source::Piece { seg, origin } => {
                    if last_origin == Some(*origin) {
                        let last = segs.last_mut().unwrap();
                        last.text.push_str(&seg.text);
                        last.style.source.append(&seg.style.source);
                    } else {
                        segs.push(seg.clone());
                    }
                    last_origin = Some(*origin);
                }
                Source::Space(seg) => {
                    segs.push(seg.clone());
                    last_origin = None;
                }
                Source::Break(_) | Source::End => {}
            }
        }
        if let (ItemKind::Penalty, Source::Break(Some(style))) = (items[b].kind, &sources[b]) {
            let mut hy = LineSegment::new("-", style.without_source());
            hy.hyphen = true;
            segs.push(hy);
        }
        let soft = k + 1 < breaks.len();
        let mut line_height: f64 = 0.0;
        for seg in &segs {
            line_height = line_height.max(cache.line_box_px(&seg.style));
        }
        let i = first_line + k;
        let mut line = Line::new(segs, line_height, soft);
        if format.has_indents() {
            line.indent_start = format.start_indent(i == 0);
            line.indent_end = format.indent_right;
        }
        // Word spacing: a justified line's spaces fill it (stretched or shrunk); the others take the desired
        // spacing, except a justified paragraph's last line that the composer shrank to fit (TeX shrinks it too).
        let n = crate::positioning::justify_opportunities(&line);
        if n > 0 {
            let mut natural = 0.0;
            for seg in &line.segments {
                natural += cache.measure(&seg.text, &seg.style);
            }
            let mut extra = 0.0;
            if ws.desired != 100.0 {
                for seg in &line.segments {
                    if crate::positioning::space_count(&seg.text) > 0 {
                        extra += cache.measure(&seg.text, &seg.style) * (factor - 1.0);
                    }
                }
            }
            if justify && (soft || natural + extra > line_width(k)) {
                line.space_px = Some((line_width(k) - natural) / n as f64);
            } else if ws.desired != 100.0 {
                line.space_px = Some(extra / n as f64);
            }
        }
        lines.push(line);
        start = b + 1;
        while start < items.len() && items[start].kind != ItemKind::Box {
            start += 1;
        }
    }
    Some(())
}

fn greedy_wrap_skipping<M: TextMeasurer>(
    para: &Paragraph,
    widths: &dyn Fn(usize) -> f64,
    skip: Option<&dyn Fn(usize, f64) -> bool>,
    cache: &mut CachedMeasurer<'_, M>,
    word_segmenter: Option<WordSegmenter<'_>>,
) -> (Vec<Line>, bool, Vec<usize>) {
    let mut st = WrapState {
        lines: Vec::new(),
        current: Line::new(Vec::new(), 0.0, false),
        current_width: 0.0,
        pending_space: false,
        pending_space_style: None,
        pending_space_text: None,
        current_word_segments: Vec::new(),
        widths,
        broke_word: false,
        skip,
        skipped: Vec::new(),
        cache,
        tab_stops: para.tab_stops.as_deref(),
        format: para.format,
        hyphen_run: 0,
        line_hyphenated: false,
    };

    for run in &para.runs {
        let chunks: Vec<&str> = run.text.split('\n').collect();
        let mut chunk_offset = 0;
        for (ci, chunk) in chunks.into_iter().enumerate() {
            let chunk_base = chunk_offset;
            chunk_offset += chunk.chars().count() + 1;
            if ci > 0 {
                // Hard break (layout.ts): every '\n' ends a line — an empty one too, so two breaks in a row
                // leave a blank line as the browser does (a lone trailing break adds none: nothing follows it).
                st.flush_word_group();
                if st.current.segments.is_empty() {
                    st.current.line_height_px = st.cache.line_box_px(&run.style);
                    st.lines.push(std::mem::replace(
                        &mut st.current,
                        Line::new(Vec::new(), 0.0, false),
                    ));
                    st.current_width = 0.0;
                    st.hyphen_run = 0;
                } else {
                    st.push_line(false);
                }
            }
            if chunk.is_empty() {
                continue;
            }
            // Manual tokenizer (layout.ts:141-184). Boundary chars are all
            // BMP singletons, so char-wise iteration matches the TS
            // UTF-16-unit scan.
            let chars: Vec<char> = chunk.chars().collect();
            let mut i = 0usize;
            while i < chars.len() {
                let ch = chars[i];
                if ch == '\u{200B}' {
                    st.flush_word_group();
                    i += 1;
                    continue;
                }
                if ch == '\u{00A0}' {
                    // NBSP run — non-breaking (layout.ts:145-151)
                    let mut j = i + 1;
                    while j < chars.len() && chars[j] == '\u{00A0}' {
                        j += 1;
                    }
                    st.current_word_segments.push(LineSegment::new(
                        chars[i..j].iter().collect::<String>(),
                        run.style.source_slice(chunk_base+i..chunk_base+j),
                    ));
                    i = j;
                    continue;
                }
                if ch == ' ' || ch == '\t' {
                    // ASCII space/tab run — break opportunity (layout.ts:153-167)
                    let mut j = i + 1;
                    while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
                        j += 1;
                    }
                    st.flush_word_group();
                    let space_text: String = chars[i..j].iter().collect();
                    if !st.current.segments.is_empty() {
                        st.pending_space = true;
                        st.pending_space_style = Some(run.style.source_slice(chunk_base+i..chunk_base+j));
                        st.pending_space_text = Some(space_text);
                    } else {
                        // Leading whitespace at line start — preserved
                        st.add_whitespace(&space_text, &run.style.source_slice(chunk_base+i..chunk_base+j));
                    }
                    i = j;
                    continue;
                }
                // Word run (layout.ts:168-183)
                let mut j = i + 1;
                while j < chars.len()
                    && chars[j] != ' '
                    && chars[j] != '\t'
                    && chars[j] != '\u{00A0}'
                    && chars[j] != '\u{200B}'
                {
                    j += 1;
                }
                let word: String = chars[i..j].iter().collect();
                // Vertical text cuts runs into pieces mid-word: the break the segmenter allows between them.
                if i == 0
                    && break_between_pieces(st.current_word_segments.last(), &run.style, &word, word_segmenter, para.format.breaking)
                {
                    st.flush_word_group();
                }
                // A combined vertical piece (tate-chu-yoko) stays whole: it is one em of its column.
                match word_segmenter.filter(|_| !run.style.unbreakable()) {
                    Some(seg) => {
                        // layout.ts:172-179 — flushing between tokens creates a
                        // break opportunity with no inter-token space.
                        let mut token_offset = chunk_base + i;
                        for (k, token) in seg(&word, para.format.breaking).into_iter().enumerate() {
                            let token_end = token_offset + token.chars().count();
                            let token_style = run.style.source_slice(token_offset..token_end);
                            token_offset = token_end;
                            // A run boundary (a style change) inside a word breaks where the text would.
                            if k > 0
                                || st.current_word_segments.last().is_some_and(|last| {
                                    breaks_between(seg, para.format.breaking, &last.text, &token)
                                })
                            {
                                st.flush_word_group();
                            }
                            st.current_word_segments
                                .push(LineSegment::new(token, token_style));
                        }
                    }
                    None => st
                        .current_word_segments
                        .push(LineSegment::new(word, run.style.source_slice(chunk_base+i..chunk_base+j))),
                }
                i = j;
            }
        }
    }

    st.flush_word_group();
    st.push_line(false);
    set_indents(&mut st.lines, &para.format, &st.skipped);
    if let Some(ts) = st.tab_stops {
        // Every tab segment carries its width and leader (`layoutParagraph`).
        for line in &mut st.lines {
            for (i, fill) in resolve_segments(&line.segments, ts, st.cache).1 {
                line.segments[i].tab = Some(fill);
            }
        }
    }
    (st.lines, st.broke_word, st.skipped)
}
