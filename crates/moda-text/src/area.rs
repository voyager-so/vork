//! Area type: Illustrator's Area Type, Threaded Text and Text Wrap, for the vector lane.
//!
//! Box text wraps every line at the box width and stacks the lines from the top. Area type fits each line into the
//! shape that holds it instead:
//!
//! - **The shape.** A frame's outline (`AreaFrame::shape`, any closed path: an ellipse, a star, a boolean's
//!   result; the node box when absent) less its **inset**. A line occupies a horizontal band, from its baseline
//!   minus the line's ascent to its baseline plus its descent; the room for it is where the shape covers the
//!   whole band (the scanline intervals of the shape at every height in the band, intersected). A band the shape
//!   splits (a crescent, a wrap object in the middle of a column) holds one line per interval, left to right, on
//!   one baseline, as in Illustrator.
//! - **Text wrap.** Objects with Text Wrap above the text cut their outline, grown by their offset, out of those
//!   intervals (`WrapExclusion`); an inverted wrap keeps only what is inside it.
//! - **Rows and columns** split the frame's inset box into a grid of regions with gutters; text fills them column
//!   by column (or row by row, `TextFlow::Rows`).
//! - **First baseline.** A region's first line sits with its baseline below the region's top by the line's ascent,
//!   cap height, leading, x height, em box or a fixed amount, but no less than the minimum (Area Type Options).
//!   Every later line sits a line height below the one before it (as box text does), plus the paragraph spacing
//!   at a paragraph's start.
//! - **Where a line can't go.** A band with no interval wide enough moves down a pixel at a time; a band past the
//!   region's bottom moves to the next region, then the next frame of the thread. A word too wide for a line
//!   leaves it empty and waits for a wider one (the narrow top of an ellipse stays blank rather than splitting a
//!   word; past the last frame the word overflows whole); only a word wider than the whole region breaks between
//!   letters. Text left over when the last frame is full is the overflow (the red "+" of the out port).
//! - **Threads.** A story flows through `AreaSpec::frames`, every frame of the thread up to and including the
//!   one being drawn (`AreaSpec::index`): the frames before it only consume text, so each frame draws its own
//!   lines without knowing the rest of the thread.
//! - **Vertical alignment** (top, centre, bottom, justify) applies to rectangular regions without wrap objects,
//!   as in Illustrator.
//!
//! The line breaking is box text's greedy wrap (`crate::wrap`) with a width per line. A line's band depends on its
//! own metrics (a 48 px word needs a taller band than 16 px text), and its metrics on what the wrap put on it, so
//! each paragraph is wrapped again with the metrics the previous pass found until they agree (a line's slot only
//! depends on the lines before it, so this settles in at most one pass per line; mixed sizes take two or three).

use std::cell::RefCell;

use serde_json::Value;

use crate::line_metrics::{compute_baseline_for_line_with_fallback, FontMetricsProvider};
use crate::measure::TextMeasurer;
use crate::paragraph_layout::{layout_paragraphs_with, ParagraphLayoutResult};
use crate::types::{InlineStyle, Line, LineSegment, Paragraph};
use crate::wrap::{greedy_wrap_area, CachedMeasurer, WordSegmenter};

/// A closed polygon, frame-local px.
pub type Ring = Vec<(f64, f64)>;

/// Where a region's first baseline sits (Illustrator's Area Type Options ▸ First Baseline).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FirstBaseline {
    /// The line's ascent: the tallest glyphs touch the top.
    #[default]
    Ascent,
    CapHeight,
    /// The line's leading (line height).
    Leading,
    XHeight,
    /// The top of the em box.
    EmBox,
    /// Exactly the minimum (`AreaOptions::min_baseline`).
    Fixed,
    /// Box text's own placement, half the leading above the line's ascent (CSS): box text that a wrap object
    /// reaches keeps its first line where it was.
    HalfLeading,
}

impl FirstBaseline {
    pub fn parse(s: &str) -> FirstBaseline {
        match s {
            "cap-height" => FirstBaseline::CapHeight,
            "leading" => FirstBaseline::Leading,
            "x-height" => FirstBaseline::XHeight,
            "em-box" => FirstBaseline::EmBox,
            "fixed" => FirstBaseline::Fixed,
            "half-leading" => FirstBaseline::HalfLeading,
            _ => FirstBaseline::Ascent,
        }
    }

    pub fn css(self) -> &'static str {
        match self {
            FirstBaseline::Ascent => "ascent",
            FirstBaseline::CapHeight => "cap-height",
            FirstBaseline::Leading => "leading",
            FirstBaseline::XHeight => "x-height",
            FirstBaseline::EmBox => "em-box",
            FirstBaseline::Fixed => "fixed",
            FirstBaseline::HalfLeading => "half-leading",
        }
    }
}

/// The order text fills a grid of rows and columns (Area Type Options ▸ Text Flow).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextFlow {
    /// Down each column, then the next column to the right.
    #[default]
    Columns,
    /// Across each row, then the next row down.
    Rows,
}

/// Vertical alignment of area type (rectangular regions only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AreaVAlign {
    #[default]
    Top,
    Center,
    Bottom,
    /// The lines spread to fill the region.
    Justify,
}

impl AreaVAlign {
    /// The node's `verticalAlign`: `top`, `middle`, `bottom`, `justify`.
    pub fn parse(s: &str) -> AreaVAlign {
        match s {
            "middle" | "center" => AreaVAlign::Center,
            "bottom" => AreaVAlign::Bottom,
            "justify" => AreaVAlign::Justify,
            _ => AreaVAlign::Top,
        }
    }
}

/// Area Type Options.
#[derive(Debug, Clone, PartialEq)]
pub struct AreaOptions {
    /// Inset spacing: how far the text keeps from the outline.
    pub inset: f64,
    pub first_baseline: FirstBaseline,
    /// The first baseline sits at least this far below the region top.
    pub min_baseline: f64,
    pub rows: usize,
    pub row_gutter: f64,
    pub columns: usize,
    pub column_gutter: f64,
    pub flow: TextFlow,
    pub valign: AreaVAlign,
}

/// Illustrator's default gutter (18 pt).
pub const DEFAULT_GUTTER: f64 = 18.0;

impl Default for AreaOptions {
    fn default() -> Self {
        AreaOptions {
            inset: 0.0,
            first_baseline: FirstBaseline::Ascent,
            min_baseline: 0.0,
            rows: 1,
            row_gutter: DEFAULT_GUTTER,
            columns: 1,
            column_gutter: DEFAULT_GUTTER,
            flow: TextFlow::Columns,
            valign: AreaVAlign::Top,
        }
    }
}

/// An object with Text Wrap, in the frame's local px.
#[derive(Debug, Clone, PartialEq)]
pub struct WrapExclusion {
    pub rings: Vec<Ring>,
    /// How far the text keeps from the outline.
    pub offset: f64,
    /// Invert Wrap: the text flows inside the outline instead of around it.
    pub invert: bool,
}

/// One text container: its box, outline, wrap objects and options.
#[derive(Debug, Clone, PartialEq)]
pub struct AreaFrame {
    pub width: f64,
    /// `INFINITY` for text that grows downward (auto height).
    pub height: f64,
    /// The outline, frame-local px; `None` = the box.
    pub shape: Option<Vec<Ring>>,
    pub exclusions: Vec<WrapExclusion>,
    pub options: AreaOptions,
}

impl AreaFrame {
    /// A box frame (no outline, no wrap objects).
    pub fn rect(width: f64, height: f64) -> AreaFrame {
        AreaFrame {
            width,
            height,
            shape: None,
            exclusions: Vec::new(),
            options: AreaOptions::default(),
        }
    }
}

/// A node's area layout input: the frames its story flows through, up to and including its own.
#[derive(Debug, Clone, PartialEq)]
pub struct AreaSpec {
    pub frames: Vec<AreaFrame>,
    /// Which frame is the node being laid out (the last one, normally).
    pub index: usize,
}

impl AreaSpec {
    pub fn single(frame: AreaFrame) -> AreaSpec {
        AreaSpec {
            frames: vec![frame],
            index: 0,
        }
    }

    /// The frame being laid out.
    pub fn frame(&self) -> &AreaFrame {
        &self.frames[self.index.min(self.frames.len().saturating_sub(1))]
    }

    /// Parse the resolved area input the renderers and the editor both build (see `moda_engine::text_area` and
    /// `utils/text/areaTextResolve.ts`):
    ///
    /// ```json
    /// { "index": 1, "frames": [ { "width": 200, "height": 300, "autoHeight": false, "verticalAlign": "top",
    ///     "area": { "points": [PathPoint…], "boxWidth": 200, "boxHeight": 300, "inset": 6,
    ///               "firstBaseline": "ascent", "minBaseline": 0, "rows": 1, "rowGutter": 18,
    ///               "columns": 2, "columnGutter": 18, "flow": "columns" },
    ///     "exclusions": [ { "points": [PathPoint…], "offset": 6, "invert": false } ] } ] }
    /// ```
    ///
    /// `area.points` scale with the frame (they were drawn at `boxWidth` × `boxHeight`); exclusion points are
    /// already frame-local. `None` without a usable frame.
    pub fn from_json(v: &Value) -> Option<AreaSpec> {
        let frames: Vec<AreaFrame> = v
            .get("frames")?
            .as_array()?
            .iter()
            .filter_map(frame_from_json)
            .collect();
        if frames.is_empty() {
            return None;
        }
        let index = v
            .get("index")
            .and_then(Value::as_u64)
            .map(|i| i as usize)
            .unwrap_or(frames.len() - 1)
            .min(frames.len() - 1);
        Some(AreaSpec { frames, index })
    }
}

fn num(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|n| n.is_finite())
}

fn frame_from_json(v: &Value) -> Option<AreaFrame> {
    let width = num(v.get("width"))?.max(0.0);
    let auto_height = v
        .get("autoHeight")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let height = if auto_height {
        f64::INFINITY
    } else {
        num(v.get("height")).unwrap_or(0.0).max(0.0)
    };
    let area = v.get("area").filter(|a| a.is_object());
    let mut options = AreaOptions::default();
    let mut shape = None;
    if let Some(a) = area {
        options = options_from_json(a);
        if let Some(points) = a.get("points").and_then(Value::as_array) {
            let sx = match num(a.get("boxWidth")) {
                Some(bw) if bw > 1e-9 && width > 0.0 => width / bw,
                _ => 1.0,
            };
            let sy = match num(a.get("boxHeight")) {
                Some(bh) if bh > 1e-9 && height.is_finite() && height > 0.0 => height / bh,
                _ => 1.0,
            };
            let rings = path_points_to_rings(points, sx, sy);
            if !rings.is_empty() {
                shape = Some(rings);
            }
        }
    }
    if let Some(va) = v.get("verticalAlign").and_then(Value::as_str) {
        options.valign = AreaVAlign::parse(va);
    }
    if area
        .and_then(|a| a.get("verticalJustify"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        options.valign = AreaVAlign::Justify;
    }
    let exclusions = v
        .get("exclusions")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|e| {
                    let rings = path_points_to_rings(e.get("points")?.as_array()?, 1.0, 1.0);
                    if rings.is_empty() {
                        return None;
                    }
                    Some(WrapExclusion {
                        rings,
                        offset: num(e.get("offset")).unwrap_or(0.0).max(0.0),
                        invert: e.get("invert").and_then(Value::as_bool).unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(AreaFrame {
        width,
        height,
        shape,
        exclusions,
        options,
    })
}

/// Area Type Options from a node's `area` record (missing fields keep their defaults).
pub fn options_from_json(a: &Value) -> AreaOptions {
    let d = AreaOptions::default();
    let count = |k: &str| {
        num(a.get(k))
            .map(|n| n.floor().clamp(1.0, 64.0) as usize)
            .unwrap_or(1)
    };
    AreaOptions {
        inset: num(a.get("inset")).unwrap_or(d.inset).max(0.0),
        first_baseline: a
            .get("firstBaseline")
            .and_then(Value::as_str)
            .map(FirstBaseline::parse)
            .unwrap_or_default(),
        min_baseline: num(a.get("minBaseline")).unwrap_or(0.0).max(0.0),
        rows: count("rows"),
        row_gutter: num(a.get("rowGutter")).unwrap_or(d.row_gutter).max(0.0),
        columns: count("columns"),
        column_gutter: num(a.get("columnGutter"))
            .unwrap_or(d.column_gutter)
            .max(0.0),
        flow: match a.get("flow").and_then(Value::as_str) {
            Some("rows") => TextFlow::Rows,
            _ => TextFlow::Columns,
        },
        valign: d.valign,
    }
}

/// PathPoint JSON (`move` / `line` / `curve` / `quadratic`; an `arc` follows its chord) to closed rings, one per
/// subpath, curves flattened to about 0.1 px; coordinates scaled by (`sx`, `sy`).
pub fn path_points_to_rings(points: &[Value], sx: f64, sy: f64) -> Vec<Ring> {
    use crate::text_path::PathCmd;
    let xy =
        |v: &Value| -> Option<(f64, f64)> { Some((num(v.get("x"))? * sx, num(v.get("y"))? * sy)) };
    let mut cmds: Vec<PathCmd> = Vec::with_capacity(points.len());
    for (i, p) in points.iter().enumerate() {
        let Some((x, y)) = xy(p) else { continue };
        let kind = p.get("type").and_then(Value::as_str).unwrap_or("line");
        let cmd = match kind {
            _ if i == 0 => PathCmd::Move(x, y),
            "move" => PathCmd::Move(x, y),
            "curve" => match (p.get("cp1").and_then(xy), p.get("cp2").and_then(xy)) {
                (Some(c1), Some(c2)) => PathCmd::Cubic(c1.0, c1.1, c2.0, c2.1, x, y),
                _ => PathCmd::Line(x, y),
            },
            "quadratic" => match p.get("cp").and_then(xy) {
                Some(c) => PathCmd::Quad(c.0, c.1, x, y),
                None => PathCmd::Line(x, y),
            },
            _ => PathCmd::Line(x, y),
        };
        cmds.push(cmd);
    }
    cmds_to_rings(&cmds)
}

/// Path commands to closed rings, one per subpath (each `Move` starts one), curves flattened to about 0.1 px.
pub fn cmds_to_rings(cmds: &[crate::text_path::PathCmd]) -> Vec<Ring> {
    use crate::text_path::PathCmd;
    let mut subpaths: Vec<Vec<PathCmd>> = Vec::new();
    for cmd in cmds {
        if matches!(cmd, PathCmd::Move(..)) || subpaths.is_empty() {
            subpaths.push(Vec::new());
        }
        subpaths.last_mut().unwrap().push(*cmd);
    }
    subpaths
        .iter()
        .map(|cmds| {
            let mut ring = crate::text_path::flatten(cmds);
            if ring.len() > 1 && ring.first() == ring.last() {
                ring.pop();
            }
            ring
        })
        .filter(|r| r.len() >= 3)
        .collect()
}

// ---------------------------------------------------------------------------
// Scanline geometry
// ---------------------------------------------------------------------------

type Intervals = Vec<(f64, f64)>;

const EPS: f64 = 1e-6;

/// The inside of `rings` (nonzero winding) along the horizontal line at `y`.
pub fn intervals_at(rings: &[Ring], y: f64) -> Intervals {
    let mut xs: Vec<(f64, i32)> = Vec::new();
    for ring in rings {
        let n = ring.len();
        for i in 0..n {
            let (x0, y0) = ring[i];
            let (x1, y1) = ring[(i + 1) % n];
            if y0 == y1 {
                continue;
            }
            let (lo, hi, dir) = if y0 < y1 { (y0, y1, 1) } else { (y1, y0, -1) };
            if y < lo || y >= hi {
                continue;
            }
            let t = (y - y0) / (y1 - y0);
            xs.push((x0 + t * (x1 - x0), dir));
        }
    }
    xs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Intervals = Vec::new();
    let mut winding = 0;
    let mut start = 0.0;
    for (x, d) in xs {
        let was = winding != 0;
        winding += d;
        let is = winding != 0;
        if !was && is {
            start = x;
        } else if was && !is && x > start {
            out.push((start, x));
        }
    }
    merge(out)
}

fn merge(mut v: Intervals) -> Intervals {
    v.retain(|(a, b)| b > a);
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Intervals = Vec::with_capacity(v.len());
    for (a, b) in v {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

fn intersect(a: &Intervals, b: &Intervals) -> Intervals {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        if hi > lo {
            out.push((lo, hi));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn subtract(a: &Intervals, b: &Intervals) -> Intervals {
    let mut out = Vec::new();
    for &(a0, a1) in a {
        let mut cur = a0;
        for &(b0, b1) in b {
            if b1 <= cur || b0 >= a1 {
                continue;
            }
            if b0 > cur {
                out.push((cur, b0));
            }
            cur = cur.max(b1);
            if cur >= a1 {
                break;
            }
        }
        if cur < a1 {
            out.push((cur, a1));
        }
    }
    out
}

fn grow(v: &Intervals, d: f64) -> Intervals {
    merge(v.iter().map(|&(a, b)| (a - d, b + d)).collect())
}

fn shrink(v: &Intervals, d: f64) -> Intervals {
    v.iter()
        .map(|&(a, b)| (a + d, b - d))
        .filter(|(a, b)| b > a)
        .collect()
}

/// The heights to sample a band at: its ends (just inside) and just either side of every vertex in it. Between
/// vertices every edge is straight, so an interval's ends move linearly and these samples see their extremes.
fn band_samples(rings: &[Ring], y0: f64, y1: f64, extra_step: Option<f64>) -> Vec<f64> {
    let (a, b) = if y1 - y0 > 2.0 * EPS {
        (y0 + EPS, y1 - EPS)
    } else {
        let m = (y0 + y1) / 2.0;
        (m, m)
    };
    let mut ys = vec![a, b];
    for ring in rings {
        for &(_, y) in ring {
            if y > a && y < b {
                ys.push(y - EPS);
                ys.push(y + EPS);
            }
        }
    }
    if let Some(step) = extra_step {
        let mut y = a + step;
        while y < b {
            ys.push(y);
            y += step;
        }
    }
    ys.sort_by(f64::total_cmp);
    ys.dedup();
    ys
}

/// Where the inside of `rings` spans the whole band [y0, y1].
pub fn interior_over_band(rings: &[Ring], y0: f64, y1: f64) -> Intervals {
    let mut acc: Option<Intervals> = None;
    for y in band_samples(rings, y0, y1, None) {
        let at = intervals_at(rings, y);
        acc = Some(match acc {
            None => at,
            Some(prev) => intersect(&prev, &at),
        });
        if acc.as_ref().is_some_and(Vec::is_empty) {
            break;
        }
    }
    acc.unwrap_or_default()
}

/// Where the inside of `rings` reaches anywhere in the band [y0, y1].
pub fn union_over_band(rings: &[Ring], y0: f64, y1: f64) -> Intervals {
    let ys = band_samples(rings, y0, y1, Some(2.0));
    let mut all: Intervals = Vec::new();
    let mut prev: Option<Intervals> = None;
    for y in ys {
        let at = intervals_at(rings, y);
        if let Some(p) = &prev {
            // An interval that slides between two samples sweeps what lies between them.
            for &(a0, a1) in p {
                for &(b0, b1) in &at {
                    if a0 < b1 && b0 < a1 {
                        all.push((a0.min(b0), a1.max(b1)));
                    }
                }
            }
        }
        all.extend(at.iter().copied());
        prev = Some(at);
    }
    merge(all)
}

/// One cell of a frame's row/column grid, frame-local px.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl AreaFrame {
    /// The grid cells text fills, in flow order.
    pub fn regions(&self) -> Vec<Region> {
        let o = &self.options;
        let inset = o.inset;
        let (x0, x1) = (inset, (self.width - inset).max(inset));
        let (y0, y1) = if self.height.is_finite() {
            (inset, (self.height - inset).max(inset))
        } else {
            (inset, f64::INFINITY)
        };
        let cols = o.columns.max(1);
        let rows = if self.height.is_finite() {
            o.rows.max(1)
        } else {
            1
        };
        let cg = if cols > 1 { o.column_gutter } else { 0.0 };
        let rg = if rows > 1 { o.row_gutter } else { 0.0 };
        let cw = ((x1 - x0 - cg * (cols - 1) as f64) / cols as f64).max(0.0);
        let rh = if y1.is_finite() {
            ((y1 - y0 - rg * (rows - 1) as f64) / rows as f64).max(0.0)
        } else {
            f64::INFINITY
        };
        let cell = |r: usize, c: usize| {
            let cx = x0 + c as f64 * (cw + cg);
            // (A single row may be unbounded: 0 × ∞ is NaN.)
            let cy = if r == 0 {
                y0
            } else {
                y0 + r as f64 * (rh + rg)
            };
            Region {
                x0: cx,
                y0: cy,
                x1: cx + cw,
                y1: if rh.is_finite() {
                    cy + rh
                } else {
                    f64::INFINITY
                },
            }
        };
        let mut out = Vec::with_capacity(rows * cols);
        match o.flow {
            TextFlow::Columns => {
                for c in 0..cols {
                    for r in 0..rows {
                        out.push(cell(r, c));
                    }
                }
            }
            TextFlow::Rows => {
                for r in 0..rows {
                    for c in 0..cols {
                        out.push(cell(r, c));
                    }
                }
            }
        }
        out
    }

    /// Whether vertical alignment applies (Illustrator: rectangular area type only).
    fn is_rectangular(&self) -> bool {
        self.shape.is_none() && self.exclusions.is_empty() && self.height.is_finite()
    }

    /// The horizontal intervals a line in the band [y0, y1] of `region` may use.
    pub fn available(&self, region: &Region, y0: f64, y1: f64) -> Vec<(f64, f64)> {
        let inset = self.options.inset;
        let mut room = match &self.shape {
            // The box less the inset is the region grid already.
            None => vec![(region.x0, region.x1)],
            Some(rings) => intersect(
                &shrink(&interior_over_band(rings, y0 - inset, y1 + inset), inset),
                &vec![(region.x0, region.x1)],
            ),
        };
        for ex in &self.exclusions {
            if room.is_empty() {
                break;
            }
            let off = ex.offset;
            if ex.invert {
                room = intersect(
                    &room,
                    &shrink(&interior_over_band(&ex.rings, y0 - off, y1 + off), off),
                );
            } else {
                room = subtract(
                    &room,
                    &grow(&union_over_band(&ex.rings, y0 - off, y1 + off), off),
                );
            }
        }
        room
    }
}

// ---------------------------------------------------------------------------
// Flow
// ---------------------------------------------------------------------------

/// What placing a line needs to know about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineMetrics {
    /// Line height (leading).
    pub lh: f64,
    /// Baseline below the line's top (half-leading, as box text).
    pub bft: f64,
    pub ascent: f64,
    pub descent: f64,
    pub cap: f64,
    pub xh: f64,
    pub font_size: f64,
}

impl LineMetrics {
    fn same(&self, o: &LineMetrics) -> bool {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        close(self.lh, o.lh)
            && close(self.bft, o.bft)
            && close(self.ascent, o.ascent)
            && close(self.descent, o.descent)
            && close(self.cap, o.cap)
            && close(self.xh, o.xh)
    }
}

/// A line's metrics (`fallback_font_size` for a line without text).
pub fn line_metrics<P: FontMetricsProvider>(
    line: &Line,
    fallback_font_size: f64,
    provider: &P,
) -> LineMetrics {
    let b = compute_baseline_for_line_with_fallback(
        line,
        line.line_height_px,
        fallback_font_size,
        provider,
    );
    let (mut cap, mut xh, mut fs) = (0.0f64, 0.0f64, 0.0f64);
    for seg in &line.segments {
        let (c, x) = provider.cap_x_heights(&seg.style);
        cap = cap.max(c);
        xh = xh.max(x);
        fs = fs.max(seg.style.line_font_size());
    }
    if line.segments.is_empty() {
        fs = fallback_font_size;
        cap = fallback_font_size * 0.7;
        xh = fallback_font_size * 0.5;
    }
    LineMetrics {
        lh: line.line_height_px,
        bft: b.baseline_from_top_px,
        ascent: b.ascent,
        descent: b.descent,
        cap,
        xh,
        font_size: fs,
    }
}

/// The metrics a line in `style` alone would have: the guess for a paragraph's lines before they are wrapped.
fn style_metrics<P: FontMetricsProvider>(style: &InlineStyle, provider: &P) -> LineMetrics {
    let line = Line {
        segments: vec![LineSegment {
            text: String::new(),
            style: style.clone(),
            tab: None,
            hyphen: false,
        }],
        line_height_px: style.line_font_size() * style.line_height,
        soft_wrap: false,
        ..Default::default()
    };
    line_metrics(&line, style.line_font_size(), provider)
}

/// Where the first baseline sits below a region's top.
fn first_baseline_offset(o: &AreaOptions, m: &LineMetrics) -> f64 {
    let v = match o.first_baseline {
        FirstBaseline::Ascent => m.ascent,
        FirstBaseline::CapHeight => m.cap,
        FirstBaseline::Leading => m.lh,
        FirstBaseline::XHeight => m.xh,
        FirstBaseline::EmBox => {
            let ad = m.ascent + m.descent;
            if ad > 0.0 {
                m.font_size * m.ascent / ad
            } else {
                m.font_size * 0.8
            }
        }
        FirstBaseline::Fixed => return o.min_baseline,
        FirstBaseline::HalfLeading => m.bft,
    };
    v.max(o.min_baseline)
}

/// Where one line goes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSlot {
    /// The frame (index into `AreaSpec::frames`).
    pub frame: usize,
    /// The grid cell within the frame.
    pub region: usize,
    /// The interval's left edge and width, frame-local px: the line aligns within it.
    pub x: f64,
    pub width: f64,
    pub baseline: f64,
    /// The line's own metrics.
    pub bft: f64,
    pub ascent: f64,
    pub descent: f64,
    pub line_height: f64,
}

impl LineSlot {
    /// The line's top (its baseline less its baseline-from-top), as box text reports `lineTopY`.
    pub fn top(&self) -> f64 {
        self.baseline - self.bft
    }
}

#[derive(Debug, Clone)]
struct Row {
    baseline: f64,
    /// Intervals not used yet, last first.
    rest: Vec<(f64, f64)>,
    /// The lowest line bottom (top + line height) on the row so far.
    bottom: f64,
    frame: usize,
    region: usize,
}

#[derive(Debug, Clone)]
struct Cursor {
    frame: usize,
    region: usize,
    /// `None` at a region's start (the first-baseline rule places the line); else the top of the next row.
    next_top: Option<f64>,
    /// At a region's start, how far below the first-baseline position the search for room has moved.
    start_drop: f64,
    row: Option<Row>,
    done: bool,
}

/// A band too narrow for this many ems holds no line.
const MIN_INTERVAL_EM: f64 = 0.5;
/// How far a band that finds no room moves down before it looks again.
const STEP_PX: f64 = 1.0;
/// Search bound per row (a region taller than this many steps moves on).
const MAX_STEPS: usize = 20_000;
/// Wrap passes per paragraph before the metrics must agree.
const MAX_PASSES: usize = 24;

struct Flow<'a> {
    spec: &'a AreaSpec,
    regions: Vec<Vec<Region>>,
    paragraph_spacing: f64,
}

impl Flow<'_> {
    /// The next line's slot, or `None` when every frame is full.
    fn next_slot(&self, cur: &mut Cursor, m: &LineMetrics, para_start: bool) -> Option<LineSlot> {
        if cur.done {
            return None;
        }
        if let Some(row) = cur.row.as_mut() {
            if !para_start {
                if let Some((x0, x1)) = row.rest.pop() {
                    row.bottom = row.bottom.max(row.baseline - m.bft + m.lh);
                    return Some(LineSlot {
                        frame: row.frame,
                        region: row.region,
                        x: x0,
                        width: x1 - x0,
                        baseline: row.baseline,
                        bft: m.bft,
                        ascent: m.ascent,
                        descent: m.descent,
                        line_height: m.lh,
                    });
                }
            }
            cur.next_top = Some(row.bottom);
            cur.row = None;
        }
        let mut spacing = if para_start && cur.next_top.is_some() {
            self.paragraph_spacing
        } else {
            0.0
        };
        let min_w = (m.font_size * MIN_INTERVAL_EM).max(1.0);
        let mut steps = 0usize;
        loop {
            let Some(frame) = self.spec.frames.get(cur.frame) else {
                cur.done = true;
                return None;
            };
            let regions = &self.regions[cur.frame];
            let Some(reg) = regions.get(cur.region).copied() else {
                cur.frame += 1;
                cur.region = 0;
                cur.next_top = None;
                cur.start_drop = 0.0;
                spacing = 0.0;
                steps = 0;
                continue;
            };
            let baseline = match cur.next_top {
                None => reg.y0 + first_baseline_offset(&frame.options, m) + cur.start_drop,
                Some(t) => t + spacing + m.bft,
            };
            let (b0, b1) = if m.ascent + m.descent > 0.0 {
                (baseline - m.ascent, baseline + m.descent)
            } else {
                (baseline - m.lh * 0.8, baseline + m.lh * 0.2)
            };
            if !(b1 <= reg.y1 + 1e-6) || steps > MAX_STEPS || reg.x1 - reg.x0 < min_w {
                cur.region += 1;
                cur.next_top = None;
                cur.start_drop = 0.0;
                spacing = 0.0;
                steps = 0;
                continue;
            }
            let ivs: Vec<(f64, f64)> = frame
                .available(&reg, b0, b1)
                .into_iter()
                .filter(|(a, b)| b - a >= min_w)
                .collect();
            if ivs.is_empty() {
                match cur.next_top {
                    None => cur.start_drop += STEP_PX,
                    Some(t) => cur.next_top = Some(t + spacing + STEP_PX),
                }
                spacing = 0.0;
                steps += 1;
                continue;
            }
            let mut rest = ivs;
            rest.reverse();
            let (x0, x1) = rest.pop().unwrap();
            cur.row = Some(Row {
                baseline,
                rest,
                bottom: baseline - m.bft + m.lh,
                frame: cur.frame,
                region: cur.region,
            });
            return Some(LineSlot {
                frame: cur.frame,
                region: cur.region,
                x: x0,
                width: x1 - x0,
                baseline,
                bft: m.bft,
                ascent: m.ascent,
                descent: m.descent,
                line_height: m.lh,
            });
        }
    }

    /// The width overflow text wraps at (it isn't drawn; it only has to break somewhere): the last region's.
    fn overflow_width(&self) -> f64 {
        self.regions
            .iter()
            .rev()
            .find_map(|r| r.last())
            .map(|r| (r.x1 - r.x0).max(1.0))
            .unwrap_or(1.0)
    }
}

/// The slots of one paragraph's lines under assumed metrics, generated on demand (the wrap asks for line `i`'s
/// width; the word-skip looks ahead).
struct SlotGen<'f, 'a> {
    flow: &'f Flow<'a>,
    /// Metrics per line index (the previous pass's lines); later lines take the last one's.
    assumed: &'f [LineMetrics],
    base: LineMetrics,
    cursor: RefCell<Cursor>,
    slots: RefCell<Vec<Option<LineSlot>>>,
    /// The cursor after each slot.
    after: RefCell<Vec<Cursor>>,
    start: Cursor,
}

impl SlotGen<'_, '_> {
    fn metrics_at(&self, i: usize) -> LineMetrics {
        self.assumed
            .get(i)
            .or(self.assumed.last())
            .copied()
            .unwrap_or(self.base)
    }

    fn get(&self, i: usize) -> Option<LineSlot> {
        let mut slots = self.slots.borrow_mut();
        while slots.len() <= i {
            let n = slots.len();
            let m = self.metrics_at(n);
            let mut cur = self.cursor.borrow_mut();
            let s = self.flow.next_slot(&mut cur, &m, n == 0);
            slots.push(s);
            self.after.borrow_mut().push(cur.clone());
        }
        slots[i]
    }

    /// Whether a word `need` wide that doesn't fit line `i` should leave it empty and move on: when it fits the
    /// region's full width (a later line, or the next region or frame, will take it; past the last frame it
    /// overflows whole). Only a word wider than the region itself breaks between letters.
    fn skip_for(&self, i: usize, need: f64) -> bool {
        let Some(si) = self.get(i) else { return false };
        let reg = &self.flow.regions[si.frame][si.region];
        need <= reg.x1 - reg.x0 + 1e-9
    }

    /// The cursor once `n` lines are placed.
    fn cursor_after(&self, n: usize) -> Cursor {
        if n == 0 {
            return self.start.clone();
        }
        self.get(n - 1);
        self.after.borrow()[n - 1].clone()
    }
}

/// An area layout: every line's slot (`None` = overflow, past the last frame).
#[derive(Debug, Clone, Default)]
pub struct AreaLayout {
    /// The frame being drawn (`AreaSpec::index`).
    pub index: usize,
    /// Per paragraph, per line (the lines of the `ParagraphLayoutResult` it came with).
    pub slots: Vec<Vec<Option<LineSlot>>>,
    /// Text is left over past the last frame.
    pub overflow: bool,
}

impl AreaLayout {
    /// Line `l` of paragraph `p`, when it is drawn in this frame.
    pub fn visible(&self, p: usize, l: usize) -> Option<&LineSlot> {
        self.slots
            .get(p)
            .and_then(|ls| ls.get(l))
            .and_then(Option::as_ref)
            .filter(|s| s.frame == self.index)
    }

    /// Whether any line lands in this frame.
    pub fn has_visible_lines(&self) -> bool {
        self.slots
            .iter()
            .flatten()
            .flatten()
            .any(|s| s.frame == self.index)
    }
}

/// The first line of `para` from which every line it can reach is `widths`' one width, for the every-line composer
/// (which then treats those lines as one, so a long story composes in linear, not cubic, time).
///
/// The shape's lines run out at the first slot that isn't one (`in_shape` false: past the last frame, for good —
/// `next_slot` stays done), and every line after takes the overflow width. An auto-height frame never runs out, so
/// the scan stops at the most lines the paragraph can take (each holds at least one character); widths past that
/// are never asked for.
fn uniform_width_from(para: &Paragraph, widths: &dyn Fn(usize) -> f64, in_shape: impl Fn(usize) -> bool) -> usize {
    let max_lines = para.runs.iter().map(|r| r.text.chars().count()).sum::<usize>() + 1;
    let mut from = 0usize;
    let mut last = f64::NAN;
    for i in 0..=max_lines {
        if !in_shape(i) {
            // Every later line is the overflow width; it starts the run if this one doesn't continue it.
            return if widths(i) == last { from } else { i };
        }
        let w = widths(i);
        if w != last {
            from = i;
            last = w;
        }
    }
    from
}

/// Lay paragraphs out as area type: `layout_paragraphs_to_lines`' result (skipped lines removed) and each line's
/// slot. `metrics` must be the provider the draw computes baselines with.
pub fn layout_area<M: TextMeasurer, P: FontMetricsProvider>(
    paragraphs: &[Paragraph],
    spec: &AreaSpec,
    paragraph_spacing_px: f64,
    measurer: &M,
    metrics: &P,
    word_segmenter: Option<WordSegmenter<'_>>,
) -> (ParagraphLayoutResult, AreaLayout) {
    let flow = Flow {
        spec,
        regions: spec.frames.iter().map(AreaFrame::regions).collect(),
        paragraph_spacing: paragraph_spacing_px.max(0.0),
    };
    let mut cursor = Cursor {
        frame: 0,
        region: 0,
        next_top: None,
        start_drop: 0.0,
        row: None,
        done: false,
    };
    // Area type's Auto leading is Illustrator's 120 % of the size, not the font's own spacing (box text, Figma).
    let measurer = &crate::measure::MultipleLeading(measurer);
    let mut cache = CachedMeasurer::new(measurer);
    let mut all_slots: Vec<Vec<Option<LineSlot>>> = Vec::new();
    let result = layout_paragraphs_with(
        paragraphs,
        paragraph_spacing_px,
        measurer,
        false,
        |para, list_indent, base_fs| {
            let indent = list_indent.unwrap_or(0.0);
            let base_style = para
                .runs
                .first()
                .map(|r| r.style.clone())
                .or_else(|| para.base_style.clone())
                .unwrap_or_else(crate::types::default_style);
            let base = style_metrics(&base_style, metrics);
            let base_lh = base_style.line_font_size() * base_style.line_height;
            let start = cursor.clone();
            let mut assumed: Vec<LineMetrics> = Vec::new();
            let mut pass = 0;
            loop {
                pass += 1;
                let gen = SlotGen {
                    flow: &flow,
                    assumed: &assumed,
                    base,
                    cursor: RefCell::new(start.clone()),
                    slots: RefCell::new(Vec::new()),
                    after: RefCell::new(Vec::new()),
                    start: start.clone(),
                };
                let overflow_w = flow.overflow_width();
                let widths =
                    |i: usize| gen.get(i).map(|s| s.width).unwrap_or(overflow_w).max(0.0) - indent;
                let skip = |i: usize, need: f64| gen.skip_for(i, need + indent);
                let uniform_from = uniform_width_from(para, &widths, |i| gen.get(i).is_some());
                // CJK spacing (`crate::cjk_spacing`): pair adjustments before the wrap, the line edges after it.
                let spaced = crate::cjk_spacing::apply_char_spacing(para);
                let para = spaced.as_ref().unwrap_or(para);
                let (mut lines, skipped) =
                    greedy_wrap_area(para, &widths, &skip, uniform_from, &mut cache, word_segmenter);
                crate::cjk_spacing::trim_lines(&mut lines, &para.format);
                if lines.is_empty() {
                    // An empty paragraph still takes a line (`layoutParagraphsToLines`).
                    lines.push(Line {
                        segments: Vec::new(),
                        line_height_px: base_lh,
                        soft_wrap: false,
                        ..Default::default()
                    });
                }
                let actual: Vec<LineMetrics> = lines
                    .iter()
                    .enumerate()
                    .map(|(i, l)| {
                        if skipped.contains(&i) {
                            gen.metrics_at(i)
                        } else {
                            line_metrics(l, base_fs, metrics)
                        }
                    })
                    .collect();
                let agree = actual
                    .iter()
                    .enumerate()
                    .all(|(i, m)| m.same(&gen.metrics_at(i)));
                if agree || pass >= MAX_PASSES {
                    let mut slots: Vec<Option<LineSlot>> =
                        (0..lines.len()).map(|i| gen.get(i)).collect();
                    cursor = gen.cursor_after(lines.len());
                    // The line's own metrics, where the pass assumed others (only when the passes ran out).
                    for (s, m) in slots.iter_mut().zip(&actual) {
                        if let Some(s) = s {
                            s.bft = m.bft;
                            s.ascent = m.ascent;
                            s.descent = m.descent;
                            s.line_height = m.lh;
                        }
                    }
                    // Skipped lines only held a slot empty.
                    let mut kept_lines = Vec::with_capacity(lines.len());
                    let mut kept_slots = Vec::with_capacity(lines.len());
                    for (i, (line, slot)) in lines.into_iter().zip(slots).enumerate() {
                        if !skipped.contains(&i) {
                            kept_lines.push(line);
                            kept_slots.push(slot);
                        }
                    }
                    all_slots.push(kept_slots);
                    return kept_lines;
                }
                assumed = actual;
            }
        },
    );
    let mut layout = AreaLayout {
        index: spec.index.min(spec.frames.len().saturating_sub(1)),
        overflow: all_slots.iter().flatten().any(Option::is_none),
        slots: all_slots,
    };
    align_vertically(spec, &flow.regions, &mut layout);
    (result, layout)
}

/// Vertical alignment within rectangular regions: shift (or, justified, spread) each region's lines into the
/// room below its last line.
fn align_vertically(spec: &AreaSpec, regions: &[Vec<Region>], layout: &mut AreaLayout) {
    for (f, frame) in spec.frames.iter().enumerate() {
        let valign = frame.options.valign;
        if valign == AreaVAlign::Top || !frame.is_rectangular() {
            continue;
        }
        for (r, reg) in regions[f].iter().enumerate() {
            let mut members: Vec<&mut LineSlot> = layout
                .slots
                .iter_mut()
                .flatten()
                .flatten()
                .filter(|s| s.frame == f && s.region == r)
                .collect();
            let Some(last) = members.last() else { continue };
            let slack = reg.y1 - (last.baseline + last.descent);
            if !(slack > 0.0) {
                continue;
            }
            let n = members.len();
            for (k, s) in members.iter_mut().enumerate() {
                s.baseline += match valign {
                    AreaVAlign::Center => slack / 2.0,
                    AreaVAlign::Bottom => slack,
                    AreaVAlign::Justify if n > 1 => slack * k as f64 / (n - 1) as f64,
                    _ => 0.0,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Ring {
        vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    }

    #[test]
    fn scanline_of_a_rectangle_and_a_hole() {
        let rings = vec![rect(0.0, 0.0, 100.0, 100.0), {
            let mut r = rect(40.0, 40.0, 60.0, 60.0);
            r.reverse();
            r
        }];
        assert_eq!(intervals_at(&rings, 10.0), vec![(0.0, 100.0)]);
        assert_eq!(intervals_at(&rings, 50.0), vec![(0.0, 40.0), (60.0, 100.0)]);
        // Over a band the hole touches, the hole cuts the whole band.
        assert_eq!(
            interior_over_band(&rings, 30.0, 45.0),
            vec![(0.0, 40.0), (60.0, 100.0)]
        );
        assert_eq!(
            union_over_band(&[rect(40.0, 40.0, 60.0, 60.0)], 30.0, 45.0),
            vec![(40.0, 60.0)]
        );
        assert!(union_over_band(&[rect(40.0, 40.0, 60.0, 60.0)], 0.0, 30.0).is_empty());
    }

    #[test]
    fn a_triangle_band_takes_its_narrowest_height() {
        // Apex at the top: narrower above.
        let tri = vec![vec![(50.0, 0.0), (100.0, 100.0), (0.0, 100.0)]];
        let iv = interior_over_band(&tri, 40.0, 60.0);
        assert_eq!(iv.len(), 1);
        assert!(
            (iv[0].0 - 30.0).abs() < 1e-3 && (iv[0].1 - 70.0).abs() < 1e-3,
            "{iv:?}"
        );
    }

    #[test]
    fn interval_set_ops() {
        let a = vec![(0.0, 10.0), (20.0, 30.0)];
        assert_eq!(
            subtract(&a, &vec![(5.0, 25.0)]),
            vec![(0.0, 5.0), (25.0, 30.0)]
        );
        assert_eq!(
            intersect(&a, &vec![(5.0, 25.0)]),
            vec![(5.0, 10.0), (20.0, 25.0)]
        );
        assert_eq!(grow(&a, 6.0), vec![(-6.0, 36.0)]);
        assert_eq!(shrink(&a, 5.0), Vec::<(f64, f64)>::new());
    }

    #[test]
    fn regions_split_the_inset_box() {
        let mut f = AreaFrame::rect(236.0, 100.0);
        f.options.inset = 10.0;
        f.options.columns = 2;
        f.options.column_gutter = 16.0;
        let r = f.regions();
        assert_eq!(r.len(), 2);
        assert_eq!((r[0].x0, r[0].x1), (10.0, 110.0));
        assert_eq!((r[1].x0, r[1].x1), (126.0, 226.0));
        assert_eq!((r[0].y0, r[0].y1), (10.0, 90.0));
        f.options.rows = 2;
        f.options.row_gutter = 10.0;
        f.options.flow = TextFlow::Rows;
        let r = f.regions();
        // Across the first row, then down.
        assert_eq!((r[1].x0, r[1].y0), (126.0, 10.0));
        assert_eq!((r[2].x0, r[2].y0), (10.0, 55.0));
    }

    #[test]
    fn parses_the_resolved_input() {
        let v: Value = serde_json::json!({
            "index": 0,
            "frames": [{
                "width": 200, "height": 100, "verticalAlign": "bottom",
                "area": {
                    "boxWidth": 100, "boxHeight": 50, "inset": 4, "firstBaseline": "cap-height", "columns": 3,
                    "points": [
                        {"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 100, "y": 0},
                        {"type": "line", "x": 100, "y": 50}, {"type": "line", "x": 0, "y": 50}
                    ]
                },
                "exclusions": [{ "offset": 3, "invert": true, "points": [
                    {"type": "move", "x": 10, "y": 10}, {"type": "line", "x": 20, "y": 10}, {"type": "line", "x": 20, "y": 20}
                ]}]
            }]
        });
        let spec = AreaSpec::from_json(&v).unwrap();
        let f = spec.frame();
        assert_eq!(f.options.columns, 3);
        assert_eq!(f.options.first_baseline, FirstBaseline::CapHeight);
        assert_eq!(f.options.valign, AreaVAlign::Bottom);
        assert_eq!(
            f.shape.as_ref().unwrap()[0][2],
            (200.0, 100.0),
            "the outline scales with the frame"
        );
        assert!(f.exclusions[0].invert);
        assert_eq!(f.exclusions[0].offset, 3.0);
    }
}
