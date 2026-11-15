//! E6.3 text wrap: CSS `text-wrap` on a paragraph, `auto` (greedy),
//! `balance` or `pretty`, like Figma's and Paper's Wrap options. Mirrored by
//! `utils/richtext/textWrap.ts` (same search, same constants), so the editor
//! measures what the engine draws.
//!
//! Both run the greedy wrap (`crate::wrap`) again with narrower line widths:
//!
//! - **Balance** spreads the text evenly over the lines greedy wrapping needs:
//!   the narrowest single width that keeps the same number of lines (and
//!   breaks no word the greedy wrap didn't), found by bisection. Best for
//!   headlines.
//! - **Pretty** avoids an orphan, a last line holding a single word: when the
//!   greedy wrap leaves one, the line before it (then the two, then the three
//!   before it: at most the last four lines) is narrowed just enough to bring
//!   a second word down, keeping the line count. The other lines keep their
//!   greedy breaks. When that can't be done the greedy lines stay.
//!
//! Alignment still uses the box width, as in CSS: a balanced centred headline
//! stays centred in its box.

use crate::types::Line;

/// A paragraph's CSS `text-wrap` style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextWrap {
    #[default]
    Auto,
    Balance,
    Pretty,
}

impl TextWrap {
    /// Parse a CSS `text-wrap` / `text-wrap-style` value: `balance` and
    /// `pretty` (alone or after `wrap`); anything else is `auto`.
    pub fn parse(value: &str) -> TextWrap {
        let v = value.to_ascii_lowercase();
        if v.split_whitespace().any(|t| t == "balance") {
            TextWrap::Balance
        } else if v.split_whitespace().any(|t| t == "pretty") {
            TextWrap::Pretty
        } else {
            TextWrap::Auto
        }
    }

    pub fn css(self) -> &'static str {
        match self {
            TextWrap::Auto => "auto",
            TextWrap::Balance => "balance",
            TextWrap::Pretty => "pretty",
        }
    }
}

/// Bisection steps (and the width tolerance they stop at) for both searches.
const SEARCH_STEPS: usize = 24;
const SEARCH_EPSILON_PX: f64 = 0.25;
/// How many trailing lines `pretty` may re-flow (Figma: "adjusts the last four lines").
const PRETTY_TAIL_LINES: usize = 4;

/// A greedy wrap with a width per line index: the lines, and whether a word
/// had to break between graphemes.
pub type WrapFn<'a> = dyn FnMut(&dyn Fn(usize) -> f64) -> (Vec<Line>, bool) + 'a;

/// Whether `line` is an orphan: one word (no break opportunity inside it).
pub fn is_orphan(line: &Line) -> bool {
    let text: String = line.segments.iter().map(|s| s.text.as_str()).collect();
    let t = text.trim_matches(|c| c == ' ' || c == '\t');
    !t.is_empty() && !t.contains([' ', '\t'])
}

/// Lay a paragraph out at `width` with `wrap`'s line breaking.
pub fn wrap_lines(wrap: TextWrap, width: f64, greedy: &mut WrapFn<'_>) -> Vec<Line> {
    let (base, base_broke) = greedy(&|_| width);
    let n = base.len();
    if n < 2 || !(width.is_finite() && width > 0.0) {
        return base;
    }
    match wrap {
        TextWrap::Auto => base,
        TextWrap::Balance => {
            let (mut lo, mut hi) = (0.0f64, width);
            let mut best = base;
            for _ in 0..SEARCH_STEPS {
                if hi - lo <= SEARCH_EPSILON_PX {
                    break;
                }
                let mid = (lo + hi) / 2.0;
                let (lines, broke) = greedy(&|_| mid);
                if lines.len() == n && (base_broke || !broke) {
                    hi = mid;
                    best = lines;
                } else {
                    lo = mid;
                }
            }
            best
        }
        TextWrap::Pretty => {
            if !base.last().is_some_and(is_orphan) {
                return base;
            }
            // Narrow the line before the last, then the two before it, …, up
            // to the last four lines: the fewest lines that fix the orphan.
            for k in 1..PRETTY_TAIL_LINES.min(n) {
                let first = n - 1 - k;
                if let Some(lines) = pretty_tail(greedy, width, n, base_broke, first) {
                    return lines;
                }
            }
            base
        }
    }
}

/// The widest narrowing of lines `first..n-1` (the last line keeps the full
/// width) that leaves no orphan, the same line count and no newly broken word.
fn pretty_tail(
    greedy: &mut WrapFn<'_>,
    width: f64,
    n: usize,
    base_broke: bool,
    first: usize,
) -> Option<Vec<Line>> {
    // `fixed(w)` (no orphan, or gone past fixing it) is true for small `w`
    // and false at `width`: bisect for the largest true `w`.
    let (mut lo, mut hi) = (0.0f64, width);
    let mut best: Option<Vec<Line>> = None;
    for _ in 0..SEARCH_STEPS {
        if hi - lo <= SEARCH_EPSILON_PX {
            break;
        }
        let mid = (lo + hi) / 2.0;
        let (lines, broke) = greedy(&|i| if i >= first && i + 1 < n { mid } else { width });
        let too_far = lines.len() != n || (broke && !base_broke);
        if too_far || !lines.last().is_some_and(is_orphan) {
            lo = mid;
            if !too_far {
                best = Some(lines);
            }
        } else {
            hi = mid;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::TextMeasurer;
    use crate::parser::parse_rich_text_to_paragraphs;
    use crate::types::InlineStyle;
    use crate::wrap::layout_paragraph;

    /// Every char is 10px wide.
    struct Mono;
    impl TextMeasurer for Mono {
        fn measure(&self, text: &str, _style: &InlineStyle) -> f64 {
            text.chars().count() as f64 * 10.0
        }
    }

    fn lines(html: &str, width: f64) -> Vec<String> {
        let para = &parse_rich_text_to_paragraphs(html)[0];
        layout_paragraph(para, width, &Mono)
            .iter()
            .map(|l| l.segments.iter().map(|s| s.text.as_str()).collect())
            .collect()
    }

    const TEXT: &str = "Text balances the lines of this headline";

    #[test]
    fn the_wrap_rides_through_the_runs_and_is_inherited() {
        let html = r#"<div style="text-wrap: pretty"><p>a</p></div><p style="text-wrap: balance">b</p><p>c</p>"#;
        let wraps =
            |paras: &[crate::types::Paragraph]| paras.iter().map(|p| p.wrap).collect::<Vec<_>>();
        let want = [TextWrap::Pretty, TextWrap::Balance, TextWrap::Auto];
        assert_eq!(wraps(&parse_rich_text_to_paragraphs(html)), want);
        let catalog = crate::catalog::Catalog::default();
        let runs = crate::runs::extract_msdf_runs_by_paragraph(html, &catalog);
        assert_eq!(
            wraps(&crate::runs::convert_msdf_runs_to_paragraphs(&runs)),
            want
        );
    }

    #[test]
    fn parses_css_values() {
        assert_eq!(TextWrap::parse("balance"), TextWrap::Balance);
        assert_eq!(TextWrap::parse("wrap pretty"), TextWrap::Pretty);
        assert_eq!(TextWrap::parse("wrap"), TextWrap::Auto);
        assert_eq!(TextWrap::parse("nowrap"), TextWrap::Auto);
    }

    #[test]
    fn balance_evens_out_the_lines_and_keeps_their_count() {
        let auto = lines(&format!("<p>{TEXT}</p>"), 300.0);
        assert_eq!(auto, ["Text balances the lines of", "this headline"]);
        let balanced = lines(
            &format!("<p style=\"text-wrap: balance\">{TEXT}</p>"),
            300.0,
        );
        assert_eq!(balanced, ["Text balances the", "lines of this headline"]);
        let widths: Vec<usize> = balanced.iter().map(|l| l.chars().count()).collect();
        assert!(widths[0].abs_diff(widths[1]) <= 5, "{widths:?}");
    }

    #[test]
    fn balance_leaves_one_line_and_hard_breaks_alone() {
        assert_eq!(
            lines("<p style=\"text-wrap: balance\">Short title</p>", 300.0),
            ["Short title"]
        );
        assert_eq!(
            lines(
                "<p style=\"text-wrap: balance\">Title<br>A much longer subtitle line</p>",
                300.0
            ),
            ["Title", "A much longer subtitle line"]
        );
    }

    #[test]
    fn pretty_brings_a_second_word_down_and_keeps_early_lines_full() {
        // 20 chars a line: five full lines, then the orphan "end".
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll \
                    mmmm nnnn oooo pppp qqqq rrrr ssss tttt end";
        let auto = lines(&format!("<p>{text}</p>"), 200.0);
        assert_eq!(auto.len(), 6);
        assert_eq!(auto[5], "end");
        let pretty = lines(&format!("<p style=\"text-wrap: pretty\">{text}</p>"), 200.0);
        assert_eq!(pretty.len(), 6, "same line count: {pretty:?}");
        assert_eq!(pretty[5], "tttt end");
        // Only the line before the last changed.
        assert_eq!(pretty[..4], auto[..4]);
        assert_eq!(pretty[4], "qqqq rrrr ssss");
    }

    #[test]
    fn pretty_leaves_text_without_an_orphan_alone() {
        let text = "one two three four five six";
        assert_eq!(
            lines(&format!("<p style=\"text-wrap: pretty\">{text}</p>"), 200.0),
            lines(&format!("<p>{text}</p>"), 200.0)
        );
    }
}
