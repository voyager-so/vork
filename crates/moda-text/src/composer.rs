//! Adobe's Every-line Composer: Knuth–Plass total-fit line breaking ("Breaking Paragraphs into Lines", 1981) over
//! boxes, glue and penalties. Mirrored by `utils/richtext/composer.ts` (same passes, same constants, the same
//! arithmetic in the same order), so both lanes pick the same breaks.
//!
//! The wrap (`crate::wrap::compose_paragraph`) turns a paragraph into items: a box per word piece, glue per space
//! (natural width the desired word spacing; justified text stretches it to the maximum and shrinks it to the
//! minimum), a flagged penalty at each hyphenation point (its width the hyphen's) and a zero penalty between CJK
//! tokens. This module picks the breaks that minimise the paragraph's total demerits:
//!
//! - A line's adjustment ratio `r` is how far its glue must stretch (`r > 0`) or shrink (`r < 0`) to fill it;
//!   `r < -1` is overfull and never allowed. Ragged text (left, centre, right) has no glue stretch: each line gets
//!   the fixed stretch [`Params::ragged_stretch`] (TeX's `\raggedright`, 2 em), so the composer evens the lines'
//!   slack out instead of filling each one greedily. A chunk's last line (before a hard break or the paragraph's
//!   end) is free to be short (`\parfillskip`).
//! - Badness `100 |r|³`; demerits `(line penalty + badness)² ± penalty²`, plus `ADJ_DEMERITS` when a line's
//!   fitness class (tight, decent, loose, very loose) is two or more from the line before it, `DOUBLE_HYPHEN_DEMERITS`
//!   for two hyphenated lines in a row and `FINAL_HYPHEN_DEMERITS` for a hyphen on the second-to-last line (plain
//!   TeX's values).
//! - Illustrator's Hyphen Limit is a hard limit: a node remembers how many hyphenated lines end in a row there.
//! - A justified line with nothing to stretch (a lone word) is scored as if it had one space's stretch
//!   ([`Params::lone_stretch`]), so the composer never isolates a word on a line to dodge a loose one.
//! - Three passes, as TeX: without hyphenation at badness ≤ 100, with it at ≤ 200, then with it at any badness
//!   (justified text with Illustrator's narrow 80–133 % spacing usually lands here). Badness is not capped in that
//!   pass (TeX caps it at 10000): a line twice as loose costs eight times as much, so the composer hyphenates, within
//!   the limit, where that tightens the spacing. Looseness 0.
//!
//! `None` when no set of breaks fits (a box wider than its line): the wrap falls back to the greedy wrap, which
//! breaks such a word between letters.

/// A box (unbreakable width), glue (a breakable space) or a penalty (a break opportunity with a cost).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ItemKind {
    Box,
    Glue,
    Penalty,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Item {
    pub kind: ItemKind,
    pub width: f64,
    pub stretch: f64,
    pub shrink: f64,
    /// Penalties only: the cost of breaking here; `f64::NEG_INFINITY` forces a break.
    pub penalty: f64,
    /// Penalties only: a hyphenated break (consecutive ones cost `DOUBLE_HYPHEN_DEMERITS` and count toward the
    /// hyphen limit).
    pub flagged: bool,
    /// Penalties only: an automatic hyphenation point, which the first pass doesn't use.
    pub auto_hyphen: bool,
}

impl Item {
    pub fn boxed(width: f64) -> Item {
        Item {
            kind: ItemKind::Box,
            width,
            stretch: 0.0,
            shrink: 0.0,
            penalty: 0.0,
            flagged: false,
            auto_hyphen: false,
        }
    }
    pub fn glue(width: f64, stretch: f64, shrink: f64) -> Item {
        Item {
            kind: ItemKind::Glue,
            width,
            stretch,
            shrink,
            penalty: 0.0,
            flagged: false,
            auto_hyphen: false,
        }
    }
    pub fn penalty(width: f64, penalty: f64, flagged: bool, auto_hyphen: bool) -> Item {
        Item {
            kind: ItemKind::Penalty,
            width,
            stretch: 0.0,
            shrink: 0.0,
            penalty,
            flagged,
            auto_hyphen,
        }
    }
}

/// Plain TeX's `\linepenalty`.
pub const LINE_PENALTY: f64 = 10.0;
/// `\hyphenpenalty` / `\exhyphenpenalty`.
pub const HYPHEN_PENALTY: f64 = 50.0;
/// `\adjdemerits`.
pub const ADJ_DEMERITS: f64 = 10000.0;
/// `\doublehyphendemerits`.
pub const DOUBLE_HYPHEN_DEMERITS: f64 = 10000.0;
/// `\finalhyphendemerits`.
pub const FINAL_HYPHEN_DEMERITS: f64 = 5000.0;
/// The badness of a line that must stretch with no stretchability (TeX's `inf_bad`).
pub const INF_BAD: f64 = 10000.0;
/// `\pretolerance` and `\tolerance`: the first two passes' badness limits.
pub const PRETOLERANCE: f64 = 100.0;
pub const TOLERANCE: f64 = 200.0;
/// The ragged stretch per line, in ems of the paragraph's size (plain TeX's `\raggedright`).
pub const RAGGED_STRETCH_EM: f64 = 2.0;
/// A space's width in ems, for [`Params::lone_stretch`] (a text face's space is about a quarter em).
pub const SPACE_EM: f64 = 0.25;

#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Justified: glue stretches and shrinks; else ragged.
    pub justify: bool,
    /// Ragged text's per-line stretch, px.
    pub ragged_stretch: f64,
    /// The hyphen limit (0: none).
    pub hyphen_limit: u32,
    /// The stretch a justified line with no spaces is scored with, px: one space's (never 0).
    pub lone_stretch: f64,
    /// The first line from which every line is as wide as the next (TeX's "easy line"): breaks reached in different
    /// numbers of lines past it are one node, so a long paragraph's active list stays a few nodes a break.
    /// `usize::MAX`: the widths may vary to the end (area type), and every line count is its own node.
    pub uniform_from: usize,
}

#[derive(Clone, Copy)]
struct Node {
    /// The break's item index (`usize::MAX`: the paragraph's start).
    position: usize,
    /// Lines before this break.
    line: usize,
    fitness: u8,
    /// Consecutive hyphenated lines ending here.
    hyphens: u32,
    flagged: bool,
    total_width: f64,
    total_stretch: f64,
    total_shrink: f64,
    demerits: f64,
    /// The node this break follows (index into the node arena).
    prev: Option<usize>,
}

fn fitness_class(r: f64) -> u8 {
    if r < -0.5 {
        0
    } else if r <= 0.5 {
        1
    } else if r <= 1.0 {
        2
    } else {
        3
    }
}

/// Whether item `i` is a legal break: a penalty that isn't infinite, or glue right after a box.
fn is_break(items: &[Item], i: usize, allow_auto: bool) -> bool {
    let it = &items[i];
    match it.kind {
        ItemKind::Penalty => it.penalty < f64::INFINITY && (allow_auto || !it.auto_hyphen),
        ItemKind::Glue => i > 0 && items[i - 1].kind == ItemKind::Box,
        ItemKind::Box => false,
    }
}

/// The breaks (item indices, the last the final forced penalty) that minimise total demerits for `items`, whose
/// line `k` is `width_of(k)` wide. `items` must end with a forced penalty.
pub fn break_lines(
    items: &[Item],
    width_of: &dyn Fn(usize) -> f64,
    params: &Params,
    hyphenate: bool,
) -> Option<Vec<usize>> {
    if items.is_empty() {
        return None;
    }
    let passes: [(bool, f64); 3] = [
        (false, PRETOLERANCE),
        (hyphenate, TOLERANCE),
        (hyphenate, f64::INFINITY),
    ];
    for (allow_auto, tolerance) in passes {
        if let Some(breaks) = pass(items, width_of, params, allow_auto, tolerance) {
            return Some(breaks);
        }
    }
    None
}

/// Badness: `100 |r|³` (`INF_BAD` for a line that must stretch with nothing to stretch).
fn badness(r: f64) -> f64 {
    if r == f64::INFINITY {
        return INF_BAD;
    }
    100.0 * r.abs() * r.abs() * r.abs()
}

fn pass(
    items: &[Item],
    width_of: &dyn Fn(usize) -> f64,
    params: &Params,
    allow_auto: bool,
    tolerance: f64,
) -> Option<Vec<usize>> {
    let n = items.len();
    // Running sums up to (not including) each item.
    let mut sum_w = vec![0.0f64; n + 1];
    let mut sum_y = vec![0.0f64; n + 1];
    let mut sum_z = vec![0.0f64; n + 1];
    for (i, it) in items.iter().enumerate() {
        let (w, y, z) = match it.kind {
            ItemKind::Penalty => (0.0, 0.0, 0.0),
            _ => (it.width, it.stretch, it.shrink),
        };
        sum_w[i + 1] = sum_w[i] + w;
        sum_y[i + 1] = sum_y[i] + y;
        sum_z[i + 1] = sum_z[i] + z;
    }
    let mut arena: Vec<Node> = vec![Node {
        position: usize::MAX,
        line: 0,
        fitness: 1,
        hyphens: 0,
        flagged: false,
        total_width: 0.0,
        total_stretch: 0.0,
        total_shrink: 0.0,
        demerits: 0.0,
        prev: None,
    }];
    let mut active: Vec<usize> = vec![0];

    for b in 0..n {
        if !is_break(items, b, allow_auto) {
            continue;
        }
        let item = items[b];
        let forced = item.kind == ItemKind::Penalty && item.penalty == f64::NEG_INFINITY;
        let flagged = item.kind == ItemKind::Penalty && item.flagged;
        let pen_w = if item.kind == ItemKind::Penalty {
            item.width
        } else {
            0.0
        };
        // Candidates keyed by (line, fitness, hyphen run), first found wins a tie; lines past `uniform_from` are one.
        let mut candidates: Vec<Node> = Vec::new();
        let mut keep: Vec<usize> = Vec::with_capacity(active.len());
        for &a in &active {
            let node = arena[a];
            let width = sum_w[b] - node.total_width + pen_w;
            let stretch = sum_y[b] - node.total_stretch;
            let shrink = sum_z[b] - node.total_shrink;
            let len = width_of(node.line);
            let last = forced && b == n - 1;
            let r = if width < len {
                if last {
                    0.0
                } else if params.justify {
                    if stretch > 0.0 {
                        (len - width) / stretch
                    } else if params.lone_stretch > 0.0 {
                        (len - width) / params.lone_stretch
                    } else {
                        f64::INFINITY
                    }
                } else if params.ragged_stretch > 0.0 {
                    (len - width) / params.ragged_stretch
                } else {
                    f64::INFINITY
                }
            } else if width > len {
                if params.justify && shrink > 0.0 {
                    (len - width) / shrink
                } else {
                    f64::NEG_INFINITY
                }
            } else {
                0.0
            };
            if r < -1.0 || forced {
                // Overfull from here on (or past a forced break): this node is done.
            } else {
                keep.push(a);
            }
            if r < -1.0 {
                continue;
            }
            let badness = badness(r);
            if badness > tolerance {
                continue;
            }
            // With no limit, only whether the line before was hyphenated matters to what follows.
            let hyphens = match (flagged, params.hyphen_limit) {
                (false, _) => 0,
                (true, 0) => 1,
                (true, _) => node.hyphens + 1,
            };
            if flagged && params.hyphen_limit > 0 && hyphens > params.hyphen_limit {
                continue;
            }
            let base = LINE_PENALTY + badness;
            let mut d = base * base;
            if item.kind == ItemKind::Penalty {
                if item.penalty >= 0.0 {
                    d += item.penalty * item.penalty;
                } else if item.penalty > f64::NEG_INFINITY {
                    d -= item.penalty * item.penalty;
                }
            }
            if flagged && node.flagged {
                d += DOUBLE_HYPHEN_DEMERITS;
            }
            if last && node.flagged {
                d += FINAL_HYPHEN_DEMERITS;
            }
            let fitness = fitness_class(r);
            if node.position != usize::MAX && (fitness as i32 - node.fitness as i32).abs() > 1 {
                d += ADJ_DEMERITS;
            }
            let total = node.demerits + d;
            let line = (node.line + 1).min(params.uniform_from);
            let key = (line, fitness, hyphens);
            match candidates
                .iter_mut()
                .find(|c| (c.line, c.fitness, c.hyphens) == key)
            {
                Some(c) => {
                    if total < c.demerits {
                        c.demerits = total;
                        c.prev = Some(a);
                    }
                }
                None => candidates.push(Node {
                    position: b,
                    line,
                    fitness,
                    hyphens,
                    flagged,
                    total_width: 0.0,
                    total_stretch: 0.0,
                    total_shrink: 0.0,
                    demerits: total,
                    prev: Some(a),
                }),
            }
        }
        active = keep;
        if !candidates.is_empty() {
            // The next line starts after the break's discardable items (glue and penalties up to the next box).
            let mut after = b + 1;
            while after < n && items[after].kind != ItemKind::Box {
                if items[after].kind == ItemKind::Penalty
                    && items[after].penalty == f64::NEG_INFINITY
                {
                    break;
                }
                after += 1;
            }
            for mut c in candidates {
                c.total_width = sum_w[after];
                c.total_stretch = sum_y[after];
                c.total_shrink = sum_z[after];
                arena.push(c);
                active.push(arena.len() - 1);
            }
        }
        if active.is_empty() {
            return None;
        }
    }
    // The best node at the final forced break.
    let mut best: Option<usize> = None;
    for &a in &active {
        if arena[a].position == n - 1 && best.is_none_or(|b| arena[a].demerits < arena[b].demerits)
        {
            best = Some(a);
        }
    }
    let mut breaks = Vec::new();
    let mut cur = best;
    while let Some(i) = cur {
        if arena[i].position == usize::MAX {
            break;
        }
        breaks.push(arena[i].position);
        cur = arena[i].prev;
    }
    breaks.reverse();
    (!breaks.is_empty()).then_some(breaks)
}

/// The total demerits of a set of breaks over `items`, as the composer scores them (tests compare greedy and
/// total-fit breaks with it). `None` when a line is overfull.
pub fn score_breaks(
    items: &[Item],
    breaks: &[usize],
    width_of: &dyn Fn(usize) -> f64,
    params: &Params,
) -> Option<f64> {
    let mut total = 0.0;
    let mut start = 0usize;
    let mut prev_fitness: Option<u8> = None;
    let mut prev_flagged = false;
    for (line, &b) in breaks.iter().enumerate() {
        let item = items[b];
        let pen_w = if item.kind == ItemKind::Penalty {
            item.width
        } else {
            0.0
        };
        let (mut w, mut y, mut z) = (pen_w, 0.0, 0.0);
        for it in &items[start..b] {
            if it.kind != ItemKind::Penalty {
                w += it.width;
                y += it.stretch;
                z += it.shrink;
            }
        }
        let len = width_of(line);
        let last = line + 1 == breaks.len();
        let r = if w < len {
            if last {
                0.0
            } else if params.justify {
                if y > 0.0 {
                    (len - w) / y
                } else {
                    (len - w) / params.lone_stretch
                }
            } else {
                (len - w) / params.ragged_stretch
            }
        } else if w > len {
            if params.justify && z > 0.0 {
                (len - w) / z
            } else {
                return None;
            }
        } else {
            0.0
        };
        if r < -1.0 {
            return None;
        }
        let badness = badness(r);
        let flagged = item.kind == ItemKind::Penalty && item.flagged;
        let mut d = (LINE_PENALTY + badness).powi(2);
        if item.kind == ItemKind::Penalty && item.penalty.is_finite() {
            d += item.penalty.signum() * item.penalty * item.penalty;
        }
        if flagged && prev_flagged {
            d += DOUBLE_HYPHEN_DEMERITS;
        }
        if last && prev_flagged {
            d += FINAL_HYPHEN_DEMERITS;
        }
        let fitness = fitness_class(r);
        if prev_fitness.is_some_and(|f| (f as i32 - fitness as i32).abs() > 1) {
            d += ADJ_DEMERITS;
        }
        total += d;
        prev_fitness = Some(fitness);
        prev_flagged = flagged;
        start = b + 1;
        while start < items.len() && items[start].kind != ItemKind::Box {
            start += 1;
        }
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Words of `w` px, spaces of 10 px (stretch 5, shrink 3), ending in `\parfillskip` and a forced break.
    fn items(words: &[f64]) -> Vec<Item> {
        let mut out = Vec::new();
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                out.push(Item::glue(10.0, 5.0, 3.0));
            }
            out.push(Item::boxed(*w));
        }
        out.push(Item::penalty(0.0, f64::NEG_INFINITY, false, false));
        out
    }

    #[test]
    fn one_line_when_everything_fits() {
        let it = items(&[30.0, 40.0]);
        let p = Params {
            justify: true,
            ragged_stretch: 32.0,
            hyphen_limit: 0,
            lone_stretch: 1.0,
            uniform_from: usize::MAX,
        };
        assert_eq!(break_lines(&it, &|_| 200.0, &p, false), Some(vec![3]));
    }

    #[test]
    fn total_fit_beats_first_fit() {
        // First fit: [50 50 50] [100] [...]: a very loose second line. Total fit moves a word down.
        let words = [50.0, 50.0, 50.0, 100.0, 60.0, 60.0];
        let it = items(&words);
        let p = Params {
            justify: true,
            ragged_stretch: 32.0,
            hyphen_limit: 0,
            lone_stretch: 1.0,
            uniform_from: usize::MAX,
        };
        let width = |_: usize| 175.0;
        let kp = break_lines(&it, &width, &p, false).unwrap();
        // Greedy: 50+10+50+10+50 = 170 fits, 100 doesn't; 100+10+60 = 170, next 60 doesn't.
        let greedy = vec![5, 9, 11];
        let kp_score = score_breaks(&it, &kp, &width, &p).unwrap();
        let greedy_score = score_breaks(&it, &greedy, &width, &p).unwrap();
        assert!(
            kp_score <= greedy_score,
            "{kp:?} {kp_score} vs {greedy_score}"
        );
    }

    #[test]
    fn nothing_fits_a_box_wider_than_the_line() {
        let it = items(&[300.0]);
        let p = Params {
            justify: true,
            ragged_stretch: 32.0,
            hyphen_limit: 0,
            lone_stretch: 1.0,
            uniform_from: usize::MAX,
        };
        assert_eq!(break_lines(&it, &|_| 200.0, &p, false), None);
    }

    /// A long paragraph whose loose line sends it to the last (any badness) pass: with one width throughout, a break
    /// keeps a few nodes, not one per line count, so 4,000 words compose in milliseconds (were minutes, and GBs).
    #[test]
    fn a_long_paragraph_on_the_last_pass_stays_linear() {
        let mut words = vec![180.0];
        words.extend((0..4000).map(|i| [20.0, 35.0, 50.0, 28.0][i % 4]));
        let it = items(&words);
        let p = Params {
            justify: true,
            ragged_stretch: 32.0,
            hyphen_limit: 0,
            lone_stretch: 1.0,
            uniform_from: 1,
        };
        let width = |k: usize| if k == 0 { 190.0 } else { 200.0 };
        let start = std::time::Instant::now();
        let breaks = break_lines(&it, &width, &p, false).expect("breaks");
        assert!(breaks.len() > 100);
        assert!(start.elapsed().as_secs_f64() < 5.0, "{:?}", start.elapsed());
        // The same optimum as keeping every line count apart, on a paragraph short enough for that.
        let short = items(&words[..300]);
        let all = Params {
            uniform_from: usize::MAX,
            ..p
        };
        let a = break_lines(&short, &width, &p, false).unwrap();
        let b = break_lines(&short, &width, &all, false).unwrap();
        assert_eq!(
            score_breaks(&short, &a, &width, &p),
            score_breaks(&short, &b, &width, &p)
        );
    }
}
