//! Variable-width stroke outlines (brushes.md §5, plan PR P5).
//!
//! Given a path node's geometry (`points`, `closed`, `cornerRadius` — the same
//! inputs as the engine's `path_cmds_for_node`), its stroke weight, caps and
//! joins, and the variable-stroke objects (`strokeProfile`, `strokeTaper`,
//! `strokeWave`, …; parsed by `stroke_model.rs` with the same repairs as the
//! TS boundary), produce the stroke's OUTLINE as path commands: closed
//! subpaths, to be filled with the stroke's paint under the **nonzero** rule.
//!
//! This is the single implementation: the editor reaches it through the
//! `moda-wasm` export `stroke_outline` (lane: `core/geometry/strokeOutline.ts`
//! → `lib/strokeOutline/`), the engine (P8) and moda-render call
//! [`stroke_outline`] natively, so canvas equals export by construction.
//!
//! Pipeline: [`centerline`] samples the path adaptively (arc-length steps sized
//! to the width model's features, ≤ ~4° of turn per step, exact samples at
//! profile knots and taper ends, sample pairs at discontinuities) → per-sample
//! half-widths → [`offset`] builds the sides (curvature clamp, joins, caps,
//! loop trimming) → [`crate::path_geom::fit::fit_polyline`] fits cubics back
//! within `tolerance` (so the fill stays smooth at any zoom; pass a
//! zoom-appropriate tolerance and cache per zoom band).
//!
//! Width at a sample = `strokeWidth / 2 × profile(t) × taper(s) × wave(s)` per
//! side (1 = the plain stroke). Marker `t` is a fraction of the total length of
//! all subpaths (anchored markers follow their anchor, resolved on the
//! unrounded path); presets, taper and wave apply per subpath.
//!
//! SAFETY: the wasm release profile is `panic = "abort"`. Inputs are scrubbed
//! (non-finite or huge coordinates dropped, counts capped, work budgets on
//! sampling and loop trimming) and `stroke_outline_fuzz_no_panic` exercises
//! hostile inputs; the native build also wraps the solve in `catch_unwind`.

pub(crate) mod centerline;
pub mod nib;
pub(crate) mod offset;
pub mod width;
pub mod windows;
pub mod wobble;

#[cfg(test)]
mod gallery;
#[cfg(test)]
pub(crate) mod raster;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_anchors;
#[cfg(test)]
mod tests_brush;
#[cfg(test)]
mod tests_windows;

use crate::value::{json, Map, Value};

use crate::path_geom::fit::fit_polyline;
use crate::path_geom::vec2::V2;
use crate::stroke_model::{InkChannels, StrokeDynamic, StrokeTaper, StrokeWave};
use crate::types::RawNode;
use centerline::{node_segments, sample_subpath, Density, Subpath};
use nib::{hash01, ink_at, InkAt, Nib};
use offset::{dot_contour, outline_subpath, Cap, Contour, Join, Pen, Style};
use width::{taper_events, taper_factor, taper_min_length, wave_factor, wave_length, ProfileCurve};

pub use crate::blend::BlendPathPoint as PathPoint;
pub use windows::{stroke_outline_windows, subpath_spans, ArcWindow, SubpathSpan};

/// Largest `points` array accepted.
pub const MAX_POINTS: usize = 100_000;
/// Soft budget of centerline samples for the whole path.
pub const MAX_SAMPLES: usize = 20_000;
/// Default fit tolerance (path units).
pub const DEFAULT_TOLERANCE: f64 = 0.25;
/// Default miter limit (SVG / lyon default).
pub const DEFAULT_MITER_LIMIT: f64 = 4.0;
const MAX_STROKE_WIDTH: f64 = 1e6;
/// Smallest semi-minor axis of a calligraphic nib, in page units at 1×: a
/// 0.5-unit hairline, as Illustrator draws a flat nib moving along its own
/// edge. A nib smaller than that stays circular (the floor never exceeds the
/// semi-major axis), and width profiles still scale it to 0 at a taper's tip.
pub const MIN_NIB_HALF: f64 = 0.25;

/// Options beyond the node's own fields.
#[derive(Debug, Clone, Copy)]
pub struct OutlineOptions {
    /// Fit tolerance in path units; `0` returns polylines (lines only).
    pub tolerance: f64,
    /// Miter limit (ratio of miter length to stroke width).
    pub miter_limit: f64,
    /// Return the centerline width samples too (hit testing, bounds, UI).
    pub samples: bool,
}

impl Default for OutlineOptions {
    fn default() -> Self {
        OutlineOptions {
            tolerance: DEFAULT_TOLERANCE,
            miter_limit: DEFAULT_MITER_LIMIT,
            samples: false,
        }
    }
}

/// Centerline width samples (what the outline was built from).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WidthSamples {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    /// Fraction of the total length (the marker `t`).
    pub t: Vec<f64>,
    /// Half-widths in path units, left / right of travel.
    pub left: Vec<f64>,
    pub right: Vec<f64>,
    /// Index of the first sample of each subpath.
    pub subpath_starts: Vec<usize>,
}

impl WidthSamples {
    /// Half-widths `(left, right)` at fraction `t`, linearly interpolated.
    pub fn width_at(&self, t: f64) -> (f64, f64) {
        width_at(&self.t, &self.left, &self.right, t)
    }
}

/// Half-widths `(left, right)` at the start and the end of an open, single-subpath
/// round-pen stroke of arc length `len` (a line's body): the profile, taper and
/// wave exactly as the outline evaluates them there, without building it.
/// Markers are plain fractions (no anchors). `None` for a brushed stroke (its
/// ends are the nib's, not a cap).
pub fn open_end_half_widths(node: &RawNode, len: f64) -> Option<[(f64, f64); 2]> {
    let model = WidthModel::from_node(node, None, &[], false);
    if model.nib.is_some() || !(len > 0.0) || !len.is_finite() {
        return None;
    }
    let side = |u: f64, after: bool, s: f64| match model.pen_at(u, after, s, len, u, None) {
        offset::Pen::Round { l, r } => (l, r),
        _ => (0.0, 0.0),
    };
    Some([side(0.0, true, 0.0), side(1.0, false, len)])
}

/// Linear interpolation of half-widths over sorted `ts` (a pure helper shared
/// by hit testing; mirrors `widthAt` in `core/geometry/strokeOutline.ts`).
pub fn width_at(ts: &[f64], left: &[f64], right: &[f64], t: f64) -> (f64, f64) {
    let n = ts.len().min(left.len()).min(right.len());
    if n == 0 || !t.is_finite() {
        return (0.0, 0.0);
    }
    let i = ts[..n].partition_point(|x| *x < t);
    if i == 0 {
        return (left[0], right[0]);
    }
    if i >= n {
        return (left[n - 1], right[n - 1]);
    }
    let (t0, t1) = (ts[i - 1], ts[i]);
    let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
    (
        left[i - 1] + (left[i] - left[i - 1]) * f,
        right[i - 1] + (right[i] - right[i - 1]) * f,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

#[derive(Debug, Clone)]
pub enum Outline {
    /// Closed subpaths; fill nonzero.
    Shape {
        points: Vec<PathPoint>,
        subpath_count: usize,
        /// Bounds of the outline's samples (exact on the stroke). The fitted
        /// cubics stay within `tolerance` of them, so a caller that needs a
        /// strict box for the curves pads by `tolerance`.
        bounds: Bounds,
        /// Largest half-width anywhere on the stroke (bounds / hit slop).
        max_half_width: f64,
        samples: Option<WidthSamples>,
    },
    /// Nothing to draw (no stroke weight, no length, all widths zero).
    Empty,
}

#[derive(Debug, Clone)]
pub struct OutlineError {
    pub code: String,
    pub message: String,
}

fn err(code: &str, message: &str) -> OutlineError {
    OutlineError {
        code: code.to_string(),
        message: message.to_string(),
    }
}

/// Everything that sets the width at a sample.
struct WidthModel {
    base_half: f64,
    profile: ProfileCurve,
    taper: Option<StrokeTaper>,
    wave: Option<StrokeWave>,
    /// Calligraphic nib (from the brush definition), scaled by the weight.
    nib: Option<Nib>,
    /// Stroke weight (the nib's scale).
    weight: f64,
    ink: Option<InkChannels>,
    /// Per-stroke random draws for the nib's angle / roundness / size.
    random: [f64; 3],
    dynamic: Option<(StrokeDynamic, u32)>,
}

impl WidthModel {
    fn from_node(
        node: &RawNode,
        brush: Option<&Value>,
        anchor_t: &[Option<f64>],
        periodic: bool,
    ) -> WidthModel {
        let sw = crate::paint::model_stroke(node)
            .map(|s| s.width)
            .filter(|w| w.is_finite())
            .unwrap_or(0.0)
            .clamp(0.0, MAX_STROKE_WIDTH);
        let application = node.stroke_brush();
        // The definition must be the one the stroke names (when it has an id).
        let nib = application.as_ref().and_then(|app| {
            let def = brush?;
            let id = def.get("id").and_then(Value::as_str);
            if id.is_some_and(|id| id != app.brush_id) {
                return None;
            }
            Nib::from_brush(def).map(|n| n.with_overrides(app.overrides.as_ref()))
        });
        let brush_seed = application.as_ref().and_then(|a| a.seed).unwrap_or(0) as u64;
        let random = [0u64, 1, 2].map(|k| hash01((brush_seed << 8) ^ (0xB5 + k)));
        let dynamic = node
            .stroke_dynamic()
            .filter(|d| d.wiggle > 0.0)
            .map(|d| (d, node.stroke_dynamic_seed()));
        WidthModel {
            base_half: if nib.is_some() { sw } else { sw * 0.5 },
            profile: node
                .stroke_profile()
                .map(|p| ProfileCurve::from_model(&p, anchor_t, periodic))
                .unwrap_or_else(ProfileCurve::uniform),
            taper: node.stroke_taper(),
            wave: node.stroke_wave(),
            ink: if nib.is_some_and(|n| n.uses_ink()) {
                node.ink_channels()
            } else {
                None
            },
            nib,
            weight: sw,
            random,
            dynamic,
        }
    }

    /// Largest possible half-width factor before the profile (for "is
    /// anything drawn at all").
    fn drawable(&self) -> bool {
        let base = match &self.nib {
            Some(n) => self.weight * n.max_half(),
            None => self.base_half,
        };
        base > 0.0 && self.profile.max_side() > 0.0
    }

    /// Largest arc-length step that still resolves the width features.
    fn max_step(&self, sub_len: f64, total: f64) -> f64 {
        let mut step = (total / 256.0).max(0.25);
        let span = if self.profile.per_subpath {
            sub_len
        } else {
            total
        };
        step = step.min(span * self.profile.min_gap() / 8.0);
        if let Some(t) = &self.taper {
            step = step.min(taper_min_length(t, sub_len) / 12.0);
        }
        if let Some(w) = &self.wave {
            if w.amount > 0.0 {
                step = step.min(wave_length(w, sub_len) / 16.0);
            }
        }
        if let Some((d, _)) = &self.dynamic {
            step = step.min(wobble::wavelength(d) / 8.0);
        }
        if let Some(ink) = &self.ink {
            if ink.t.len() > 1 {
                step = step.min(total / (2.0 * ink.t.len() as f64));
            }
        }
        step
    }

    /// The pen at sample `i` (the sample's position along the path decides
    /// profile, taper, wave and the ink channels).
    fn pen(&self, sp: &Subpath, i: usize, total: f64) -> Pen {
        self.pen_heading(sp, i, total, heading(&sp.samples[i]))
    }

    /// [`Self::pen`] with the path direction a path-relative nib takes given.
    fn pen_heading(&self, sp: &Subpath, i: usize, total: f64, dir: V2) -> Pen {
        let smp = &sp.samples[i];
        let t = if self.profile.per_subpath {
            if sp.length > 0.0 {
                smp.s_local / sp.length
            } else {
                0.0
            }
        } else if total > 0.0 {
            smp.s / total
        } else {
            0.0
        };
        self.pen_at(
            t,
            smp.after,
            smp.s_local,
            sp.length,
            if total > 0.0 { smp.s / total } else { 0.0 },
            Some(dir),
        )
    }

    /// `u` = profile parameter, `s_local` / `len` = position in the subpath,
    /// `t_path` = fraction of the whole path (ink channels), `heading` = the
    /// path's direction there (a path-relative nib turns with it).
    fn pen_at(
        &self,
        u: f64,
        after: bool,
        s_local: f64,
        len: f64,
        t_path: f64,
        heading: Option<V2>,
    ) -> Pen {
        let (pl, pr) = self.profile.eval(u, after);
        let mut g = 1.0;
        if let Some(tp) = &self.taper {
            g *= taper_factor(tp, s_local, len);
        }
        if let Some(w) = &self.wave {
            g *= wave_factor(w, s_local, len);
        }
        match &self.nib {
            Some(nib) => {
                let ink = match &self.ink {
                    Some(ink) => ink_at(ink, t_path),
                    // No pen channels: only Fade (the position along the path) varies.
                    None => InkAt::at_t(t_path),
                };
                let (a, b, u) = nib.ellipse(&ink, self.random, heading);
                let a = a * self.weight;
                Pen::Ellipse {
                    a,
                    // A flat nib keeps a hairline (Illustrator draws ~0.5 page
                    // units at 1× where the path runs along its edge): minor
                    // axis ≥ 0.5 units, never above the major axis.
                    b: (b * self.weight).max(MIN_NIB_HALF.min(a)),
                    u,
                    fl: pl * g,
                    fr: pr * g,
                }
            }
            None => Pen::Round {
                l: self.base_half * pl * g,
                r: self.base_half * pr * g,
            },
        }
    }
}

fn style_of(node: &RawNode, opts: &OutlineOptions) -> Style {
    let stroke = crate::paint::model_stroke(node);
    let stroke = stroke.as_ref();
    Style {
        cap: match stroke.and_then(|s| s.cap.as_deref()) {
            Some("round") => Cap::Round,
            Some("square") => Cap::Square,
            _ => Cap::Butt,
        },
        join: match stroke.and_then(|s| s.join.as_deref()) {
            Some("round") => Join::Round,
            Some("bevel") => Join::Bevel,
            _ => Join::Miter,
        },
        // The stroke's own miter limit wins; else the option (default 4).
        miter_limit: if stroke.is_some_and(|s| s.miter_limit.is_some()) {
            node.miter_limit()
        } else if opts.miter_limit.is_finite() {
            opts.miter_limit.clamp(1.0, 500.0)
        } else {
            DEFAULT_MITER_LIMIT
        },
        // Half the fit tolerance: the dropped sliver plus the fit's own error stay within it.
        fan_min: if opts.tolerance.is_finite() {
            (opts.tolerance * 0.5).clamp(0.0, 1e4)
        } else {
            0.0
        },
    }
}

/// The path's direction at a sample: its tangent, or at a corner the
/// bisector of the incoming and outgoing tangents (a path-relative nib's
/// legs end with their own directions: [`corner_legs`]).
pub(crate) fn heading(smp: &centerline::Sample) -> V2 {
    if smp.corner {
        (smp.t_in + smp.t_out).unit().unwrap_or(smp.t_out)
    } else {
        smp.t_out
    }
}

/// A path-relative nib arrives at a corner in one orientation and leaves in
/// another, so it is outlined leg by leg: the subpath is cut at its corners
/// into open legs, each ending on the corner with the nib turned to its own
/// direction (`pen_at(i, dir)`: the pen at sample `i` for direction `dir`).
/// The turn between them is `offset::nib_turn_contour`. A closed ring's legs
/// run from corner to corner round the seam (`s` kept increasing).
fn corner_legs(
    sub: &Subpath,
    pens: &[Pen],
    pen_at: impl Fn(usize, V2) -> Pen,
) -> Vec<(Subpath, Vec<Pen>)> {
    let n = sub.samples.len();
    let corners: Vec<usize> = (0..n).filter(|&i| sub.samples[i].corner).collect();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    if sub.closed {
        for k in 0..corners.len() {
            ranges.push((corners[k], corners[(k + 1) % corners.len()]));
        }
    } else {
        let mut start = 0;
        for &c in &corners {
            if c > start {
                ranges.push((start, c));
            }
            start = c;
        }
        if start + 1 < n {
            ranges.push((start, n - 1));
        }
    }
    let mut legs = Vec::with_capacity(ranges.len());
    for (a, b) in ranges {
        let idx: Vec<usize> = if b > a {
            (a..=b).collect()
        } else {
            (a..n).chain(0..=b).collect()
        };
        let mut samples = Vec::with_capacity(idx.len());
        let mut leg_pens = Vec::with_capacity(idx.len());
        let last = idx.len() - 1;
        for (k, &i) in idx.iter().enumerate() {
            let mut smp = sub.samples[i];
            if k > 0 && i < idx[k - 1] {
                smp.s += sub.length;
            }
            let pen = if k == 0 && smp.corner {
                smp.t_in = smp.t_out;
                smp.corner = false;
                smp.brk = true;
                pen_at(i, smp.t_out)
            } else if k == last && smp.corner {
                smp.t_out = smp.t_in;
                smp.corner = false;
                smp.brk = true;
                pen_at(i, smp.t_in)
            } else {
                pens[i]
            };
            samples.push(smp);
            leg_pens.push(pen);
        }
        let moves = samples.windows(2).any(|w| w[0].p.dist(w[1].p) > 1e-9);
        if samples.len() >= 2 && moves {
            legs.push((
                Subpath {
                    samples,
                    closed: false,
                    length: sub.length,
                    index: sub.index,
                },
                leg_pens,
            ));
        }
    }
    legs
}

/// Outline a path node's stroke. `node` is a `PathNode` (its `type` is not
/// checked here; callers route only paths). Pure; never panics on native (the
/// solve runs under `catch_unwind`).
///
/// `brush` is the `BrushDefinition` the node's `strokeBrush.brushId` names
/// (from the document's brush table); a calligraphic definition sets the nib,
/// any other type (art / scatter / pattern, plan P17–P19) is ignored here.
pub fn stroke_outline(
    node: &RawNode,
    brush: Option<&Value>,
    opts: &OutlineOptions,
) -> Result<Outline, OutlineError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve(node, brush, opts, None)
        })) {
            Ok(r) => r,
            Err(_) => Err(err(
                "solver_failed",
                "the stroke outliner panicked on this input",
            )),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        solve(node, brush, opts, None)
    }
}

/// `windows`: outline only these arc-length windows (see [`windows`]).
fn solve(
    node: &RawNode,
    brush: Option<&Value>,
    opts: &OutlineOptions,
    windows: Option<&[ArcWindow]>,
) -> Result<Outline, OutlineError> {
    let raw_points = node
        .points()
        .map_or(std::borrow::Cow::Borrowed(&[][..]), |p| p.json());
    let raw_points: &[Value] = &raw_points;
    if raw_points.len() > MAX_POINTS {
        return Err(err("too_many_points", "at most 100000 path points"));
    }
    let closed = node.bool("closed").unwrap_or(false);
    let corners = crate::path_geom::round_corners::PathCorners::of_node(node);
    let (seg_paths, anchor_t, periodic) = node_segments(raw_points, closed, &corners);
    let model = WidthModel::from_node(node, brush, &anchor_t, periodic);
    if !model.drawable() {
        return Ok(Outline::Empty);
    }
    let style = style_of(node, opts);
    let tolerance = if opts.tolerance.is_finite() {
        opts.tolerance.clamp(0.0, 1e4)
    } else {
        DEFAULT_TOLERANCE
    };

    let lengths: Vec<f64> = seg_paths.iter().map(|sp| sp.lens.iter().sum()).collect();
    let total: f64 = lengths.iter().sum();
    // Global density: never more than MAX_SAMPLES for the whole path.
    let floor_step = total / MAX_SAMPLES as f64;

    let mut contours: Vec<Contour> = Vec::new();
    let mut samples_out = opts.samples.then(WidthSamples::default);
    let mut max_half: f64 = 0.0;
    // Hard sample budget for the whole path: every subpath gets a share of
    // MAX_SAMPLES by its estimate (arc-length steps + turn bound), and the
    // sampler coarsens to fit — memory is bounded whatever the input.
    let steps: Vec<f64> = seg_paths
        .iter()
        .zip(&lengths)
        .map(|(_, len)| model.max_step(*len, total).max(floor_step).max(1e-6))
        .collect();
    let estimates: Vec<f64> = seg_paths
        .iter()
        .zip(&steps)
        .map(|(sp, st)| centerline::subpath_estimate(sp, *st))
        .collect();
    let est_total: f64 = estimates.iter().sum();
    let mut s0 = 0.0;
    for (index, sp) in seg_paths.iter().enumerate() {
        let len = lengths[index];
        if sp.segs.is_empty() {
            if sp.has_commands && !sp.closed && windows.is_none() {
                // Zero-length subpath: a dot for round / square caps.
                let t = if total > 0.0 { s0 / total } else { 0.0 };
                let pen = model.pen_at(
                    if model.profile.per_subpath { 0.0 } else { t },
                    true,
                    0.0,
                    0.0,
                    t,
                    None,
                );
                max_half = max_half.max(pen.reach());
                if let Some(c) = dot_contour(&style, &pen, sp.start) {
                    contours.push(c);
                }
            }
            continue;
        }
        let step = steps[index];
        let share = if est_total > MAX_SAMPLES as f64 {
            MAX_SAMPLES as f64 * estimates[index] / est_total
        } else {
            2.0 * estimates[index]
        };
        let budget = share.floor() as usize + sp.segs.len() + 2;
        // Width events inside this subpath (global arc length).
        let mut events: Vec<(f64, bool)> = Vec::new();
        for (t, disc) in model.profile.events() {
            let s = if model.profile.per_subpath {
                s0 + t * len
            } else {
                t * total
            };
            if s > s0 && s < s0 + len {
                events.push((s, disc));
            }
        }
        if let Some(tp) = &model.taper {
            for s in taper_events(tp, len) {
                events.push((s0 + s, false));
            }
        }
        if let Some(ws) = windows {
            windows::push_events(
                ws,
                index,
                s0,
                len,
                sp.closed,
                model.taper.as_ref(),
                &mut events,
            );
        }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        events.dedup_by(|b, a| {
            if (a.0 - b.0).abs() <= 1e-9 {
                a.1 |= b.1;
                true
            } else {
                false
            }
        });
        let mut sub = sample_subpath(
            sp,
            index,
            s0,
            Density {
                max_step: step,
                budget,
            },
            &events,
        );
        if let Some((d, seed)) = &model.dynamic {
            wobble::displace(&mut sub, d, *seed);
        }
        let mut pens: Vec<Pen> = (0..sub.samples.len())
            .map(|i| model.pen(&sub, i, total))
            .collect();
        let (hl, hr): (Vec<f64>, Vec<f64>) = sub
            .samples
            .iter()
            .zip(&pens)
            .map(|(smp, pen)| {
                pen.normal_extent(
                    (smp.t_in.left_normal() + smp.t_out.left_normal())
                        .unit()
                        .unwrap_or(smp.t_out.left_normal()),
                )
            })
            .unzip();
        let sub_max = pens.iter().map(Pen::reach).fold(0.0, f64::max);
        max_half = max_half.max(sub_max);
        if let Some(ws) = samples_out.as_mut() {
            ws.subpath_starts.push(ws.t.len());
            for (i, smp) in sub.samples.iter().enumerate() {
                ws.x.push(smp.p.x);
                ws.y.push(smp.p.y);
                ws.t.push(if total > 0.0 { smp.s / total } else { 0.0 });
                ws.left.push(hl[i]);
                ws.right.push(hr[i]);
            }
        }
        s0 += len;
        if sub_max <= 0.0 {
            continue;
        }
        if let Some(ws) = windows {
            contours.extend(windows::outline_windows(
                &sub, &pens, &model, &style, ws, total, step,
            ));
            continue;
        }
        // A path-relative nib turns in place at each corner: the region it
        // sweeps there (from its incoming to its outgoing orientation).
        let turns: Vec<Contour> = if model.nib.is_some_and(|n| n.angle_relative_to_path) {
            (0..sub.samples.len())
                .filter(|&i| sub.samples[i].corner)
                .filter_map(|i| {
                    let smp = &sub.samples[i];
                    let pin = model.pen_heading(&sub, i, total, smp.t_in);
                    let pout = model.pen_heading(&sub, i, total, smp.t_out);
                    offset::nib_turn_contour(smp.p, &pin, &pout)
                })
                .collect()
        } else {
            Vec::new()
        };
        let window = 6.0 * sub_max + 4.0 * step;
        if !turns.is_empty() {
            let legs = corner_legs(&sub, &pens, |i, dir| model.pen_heading(&sub, i, total, dir));
            let body: Vec<Contour> = legs
                .iter()
                .flat_map(|(leg, leg_pens)| outline_subpath(leg, leg_pens, &style, window))
                .collect();
            // Wound like the body, so nonzero fill unions them.
            let sign = body
                .iter()
                .map(offset::signed_area)
                .fold(0.0, |acc: f64, a| if a.abs() > acc.abs() { a } else { acc });
            contours.extend(body);
            for mut t in turns {
                if offset::signed_area(&t) * sign < 0.0 {
                    t.pts.reverse();
                    t.brk.reverse();
                }
                contours.push(t);
            }
        } else {
            let sub = rotate_closed_seam(sub, &mut pens);
            contours.extend(outline_subpath(&sub, &pens, &style, window));
        }
    }
    if contours.is_empty() {
        return Ok(Outline::Empty);
    }
    let mut points: Vec<PathPoint> = Vec::new();
    let mut bounds = Bounds {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    let mut subpath_count = 0;
    for c in &contours {
        let segs = fit_polyline(&c.pts, &c.brk, true, tolerance);
        let Some(first) = segs.first() else { continue };
        let p = first.start();
        points.push(PathPoint::Move { x: p.x, y: p.y });
        for s in &segs {
            points.push(s.to_path_point());
        }
        // Bounds of the outline samples; the fitted curve stays within
        // `tolerance` of them.
        for q in &c.pts {
            bounds.min_x = bounds.min_x.min(q.x);
            bounds.min_y = bounds.min_y.min(q.y);
            bounds.max_x = bounds.max_x.max(q.x);
            bounds.max_y = bounds.max_y.max(q.y);
        }
        subpath_count += 1;
    }
    if subpath_count == 0 {
        return Ok(Outline::Empty);
    }
    Ok(Outline::Shape {
        points,
        subpath_count,
        bounds,
        max_half_width: max_half,
        samples: samples_out,
    })
}

/// Closed subpaths: start the rings at a smooth, unclamped sample so no loop
/// straddles the seam (loop trimming works on a linear sequence).
fn rotate_closed_seam(mut sub: Subpath, pens: &mut [Pen]) -> Subpath {
    let n = sub.samples.len();
    if !sub.closed || n < 3 {
        return sub;
    }
    // Pick the smooth sample with the least turn relative to its width.
    let mut best = None;
    let mut best_score = f64::INFINITY;
    for i in 0..n {
        let smp = &sub.samples[i];
        if smp.corner || smp.brk {
            continue;
        }
        let a = sub.samples[(i + n - 1) % n].p;
        let b = sub.samples[(i + 1) % n].p;
        let (Some(d0), Some(d1)) = ((smp.p - a).unit(), (b - smp.p).unit()) else {
            continue;
        };
        let score = d0.cross(d1).abs() * pens[i].reach();
        if score < best_score {
            best_score = score;
            best = Some(i);
        }
    }
    let Some(k) = best else { return sub };
    sub.samples.rotate_left(k);
    pens.rotate_left(k);
    // Keep `s` increasing along the ring for the loop-search window.
    let len = sub.length;
    let base = sub.samples[0].s;
    for smp in sub.samples.iter_mut() {
        if smp.s < base {
            smp.s += len;
        }
    }
    sub
}

// ---------------------------------------------------------------------------
// JSON surface (wasm export `stroke_outline`)
// ---------------------------------------------------------------------------

fn bounds_json(b: &Bounds) -> Value {
    json!({"minX": b.min_x, "minY": b.min_y, "maxX": b.max_x, "maxY": b.max_y})
}

fn error_json(e: &OutlineError) -> String {
    json!({"status": "error", "code": e.code, "message": e.message}).to_string()
}

/// Node fields the outliner reads; everything else in the request is ignored.
const NODE_KEYS: &[&str] = &["points", "closed", "cornerRadius", "cornerStyle", "strokes", "inkChannels"];

/// JSON in / JSON out; always an envelope.
///
/// ```jsonc
/// // request: the PathNode's own fields + options. The stroke is the node's
/// // model stroke (width, cap, join, miterLimit, profile, taper, wave,
/// // dynamic, brush) as the only entry of `strokes`.
/// { "points": PathPoint[], "closed"?, "cornerRadius"?, "strokes": [Stroke],
///   "inkChannels"?,
///   "miterLimit"?: 4,                // the option, used when the stroke has none (1–500)
///   "brush"?: BrushDefinition,       // the definition the stroke's brush.brushId names

///   "tolerance"?: 0.25,              // fit tolerance, path units; 0 = polylines
///   "samples"?: false }
/// // response
/// { "status": "ok", "points": PathPoint[], "fillRule": "nonzero", "closed": true,
///   "subpathCount", "bounds": {minX,minY,maxX,maxY}, "maxHalfWidth",
///   "samples"?: { x[], y[], t[], left[], right[], subpathStarts[] } }
/// { "status": "empty" }
/// { "status": "error", "code", "message" }
/// ```
pub fn stroke_outline_json(request_json: &str) -> String {
    stroke_outline_json_with(request_json, &|_, _| None)
}

/// [`stroke_outline_json`] restricted to arc-length windows a caller resolves from the request (the engine's
/// trim: `moda_engine::tess::outline::stroke_outline_trimmed_json`). `windows` sees the outliner's node and the
/// whole request; `None` outlines the whole stroke.
pub fn stroke_outline_json_with(
    request_json: &str,
    windows: &dyn Fn(&RawNode, &Map) -> Option<Vec<ArcWindow>>,
) -> String {
    if request_json.len() > 64 * 1024 * 1024 {
        return error_json(&err("request_too_large", "request exceeds 64 MiB"));
    }
    let value: Value = match crate::value::from_str(request_json) {
        Ok(v) => v,
        Err(e) => {
            return error_json(&err(
                "invalid_json",
                &format!("request is not valid JSON: {e}"),
            ))
        }
    };
    let Some(obj) = value.as_object() else {
        return error_json(&err("invalid_request", "request must be an object"));
    };
    if obj.get("points").is_some_and(|p| !p.is_array()) {
        return error_json(&err("invalid_request", "points must be an array"));
    }
    let mut m = Map::new();
    m.insert("type".into(), json!("path"));
    for k in NODE_KEYS {
        if let Some(v) = obj.get(*k) {
            m.insert((*k).into(), v.clone());
        }
    }
    let node = RawNode::new(m);
    let d = OutlineOptions::default();
    let opts = OutlineOptions {
        tolerance: obj
            .get("tolerance")
            .and_then(Value::as_f64)
            .unwrap_or(d.tolerance),
        miter_limit: obj
            .get("miterLimit")
            .and_then(Value::as_f64)
            .unwrap_or(d.miter_limit),
        samples: obj.get("samples").and_then(Value::as_bool).unwrap_or(false),
    };
    let brush = obj.get("brush").filter(|b| b.is_object());
    let result = match windows(&node, obj) {
        Some(ws) => stroke_outline_windows(&node, brush, &opts, &ws),
        None => stroke_outline(&node, brush, &opts),
    };
    match result {
        Ok(Outline::Empty) => json!({"status": "empty"}).to_string(),
        Ok(Outline::Shape {
            points,
            subpath_count,
            bounds,
            max_half_width,
            samples,
        }) => {
            let mut out = json!({
                "status": "ok",
                "points": points.iter().map(PathPoint::to_json).collect::<Vec<_>>(),
                "fillRule": "nonzero",
                "closed": true,
                "subpathCount": subpath_count,
                "bounds": bounds_json(&bounds),
                "maxHalfWidth": max_half_width,
            });
            if let Some(s) = samples {
                out["samples"] = json!({
                    "x": s.x, "y": s.y, "t": s.t, "left": s.left, "right": s.right,
                    "subpathStarts": s.subpath_starts,
                });
            }
            out.to_string()
        }
        Err(e) => error_json(&e),
    }
}
