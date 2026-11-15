//! Arc-length windows on a variable-width outline (plan P8: trim paths and
//! dashes on profiled / brushed strokes).
//!
//! The engine resolves trim and dash into windows of each subpath's arc length
//! (measured by [`subpath_spans`], the outliner's own lengths, so the windows
//! and the width model agree), and [`stroke_outline_windows`] outlines only
//! those windows. Widths come from the WHOLE path — profile markers, the
//! calligraphic nib, ink channels and the wobble keep their full-path `t` — so
//! a dash of a tapered stroke is a slice of the tapered outline, not a new
//! taper per dash. One exception, After Effects-style: under TRIM, taper is
//! evaluated against the visible (trimmed) run ([`ArcWindow::taper_extent`]),
//! so a write-on keeps its tapered tip (that is what px taper is for); the
//! dashes of a trimmed run carry the run as their extent.
//!
//! Each window is an open piece: its ends get the node's `lineCap` (a
//! calligraphic nib gets its half-nib cap, like any open end of its sweep).
//! A window that covers a whole closed subpath (a dash that never turns off)
//! stays a closed ring.

use crate::types::RawNode;
use crate::value::Value;

use super::centerline::{node_segments, Sample, Subpath};
use super::offset::{outline_subpath, Contour, Pen, Style};
use super::width::taper_events;
use super::{err, rotate_closed_seam, solve, Outline, OutlineError, OutlineOptions, WidthModel};
use crate::stroke_model::StrokeTaper;

/// A window of one subpath's arc length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcWindow {
    /// Index of the subpath, as [`subpath_spans`] lists them.
    pub subpath: usize,
    /// Arc length from the subpath's start. On a closed subpath `end` may
    /// exceed the length (the window runs through the seam); `start <= 0 &&
    /// end >= length` keeps the closed ring.
    pub start: f64,
    pub end: f64,
    /// Evaluate taper over this run (`start`, `end`, same arc-length measure
    /// as the window, containing it) instead of the whole subpath: the trimmed
    /// run a trim window or its dashes belong to. `None` = the subpath's own
    /// ends (untrimmed dashes).
    pub taper_extent: Option<(f64, f64)>,
}

/// One subpath as the outliner measures it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubpathSpan {
    pub length: f64,
    pub closed: bool,
}

/// The outliner's subpaths (points order, including zero-length ones) and
/// their arc lengths.
pub fn subpath_spans(node: &RawNode) -> Vec<SubpathSpan> {
    let raw_points = node.points().map_or(std::borrow::Cow::Borrowed(&[][..]), |p| p.json());
    let raw_points: &[Value] = &raw_points;
    if raw_points.len() > super::MAX_POINTS {
        return Vec::new();
    }
    let closed = node.bool("closed").unwrap_or(false);
    let corners = crate::path_geom::round_corners::PathCorners::of_node(node);
    let (paths, _, _) = node_segments(raw_points, closed, &corners);
    paths
        .iter()
        .map(|sp| SubpathSpan {
            length: sp.lens.iter().sum(),
            closed: sp.closed,
        })
        .collect()
}

/// [`super::stroke_outline`] restricted to `windows`. No windows → `Empty`.
pub fn stroke_outline_windows(
    node: &RawNode,
    brush: Option<&Value>,
    opts: &OutlineOptions,
    windows: &[ArcWindow],
) -> Result<Outline, OutlineError> {
    if windows.is_empty() {
        return Ok(Outline::Empty);
    }
    if windows.len() > 100_000 {
        return Err(err("too_many_windows", "at most 100000 windows"));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve(node, brush, opts, Some(windows))
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
        solve(node, brush, opts, Some(windows))
    }
}

const EPS: f64 = 1e-9;

fn is_ring(w: &ArcWindow, len: f64, closed: bool) -> bool {
    closed && w.start <= EPS && w.end >= len - 1e-7
}

/// Normalised `(start, end)` of a window in the subpath's local arc length:
/// open subpaths CLAMP to `[0, len]` (an open path has no seam to wrap
/// through — the engine splits a wrapping trim into two runs itself); closed
/// ones wrap `start` into `[0, len)` and keep `end - start <= len`. `None` =
/// empty.
fn span_of(w: &ArcWindow, len: f64, closed: bool) -> Option<(f64, f64)> {
    if !(w.start.is_finite() && w.end.is_finite() && len.is_finite() && len > 0.0) {
        return None;
    }
    if closed {
        let width = (w.end - w.start).min(len);
        if width <= EPS {
            return None;
        }
        let a = w.start.rem_euclid(len);
        Some((a, a + width))
    } else {
        let (a, b) = (w.start.max(0.0), w.end.min(len));
        (b - a > EPS).then_some((a, b))
    }
}

/// How far [`span_of`] moved the window to normalise it: a whole number of
/// laps on a closed subpath (the taper extent moves with it), zero on an open
/// one (clamping trims the window but must not move its taper extent).
fn wrap_shift(w: &ArcWindow, a: f64, closed: bool) -> f64 {
    if closed {
        a - w.start
    } else {
        0.0
    }
}

/// Window ends as sampling events (global arc length) for subpath `index`
/// starting at `s0`, so the pieces start and end on exact samples.
pub(super) fn push_events(
    windows: &[ArcWindow],
    index: usize,
    s0: f64,
    len: f64,
    closed: bool,
    taper: Option<&StrokeTaper>,
    events: &mut Vec<(f64, bool)>,
) {
    let mut push = |s: f64| {
        let s = if closed { s.rem_euclid(len) } else { s };
        if s > EPS && s < len - EPS {
            events.push((s0 + s, false));
        }
    };
    for w in windows.iter().filter(|w| w.subpath == index) {
        if is_ring(w, len, closed) {
            continue;
        }
        let Some((a, b)) = span_of(w, len, closed) else {
            continue;
        };
        push(a);
        push(b);
        // Under trim, taper knots move with the trimmed run: sample them
        // exactly too.
        if let (Some(tp), Some((e0, e1))) = (taper, w.taper_extent) {
            if e0.is_finite() && e1.is_finite() && e1 > e0 {
                let shift = wrap_shift(w, a, closed);
                for k in taper_events(tp, e1 - e0) {
                    push(e0 + shift + k);
                }
            }
        }
    }
}

/// Outline the windows of one sampled subpath. `pens` are the full-path pens
/// (one per sample).
#[allow(clippy::too_many_arguments)]
pub(super) fn outline_windows(
    sub: &Subpath,
    pens: &[Pen],
    model: &WidthModel,
    style: &Style,
    windows: &[ArcWindow],
    total: f64,
    step: f64,
) -> Vec<Contour> {
    let mut out = Vec::new();
    let len = sub.length;
    let n = sub.samples.len();
    if n < 2 {
        return out;
    }
    let seq = unroll(sub);
    for w in windows.iter().filter(|w| w.subpath == sub.index) {
        if is_ring(w, len, sub.closed) {
            let mut ring_pens = pens.to_vec();
            let ring = rotate_closed_seam(sub.clone(), &mut ring_pens);
            let reach = ring_pens.iter().map(Pen::reach).fold(0.0, f64::max);
            if reach > 0.0 {
                out.extend(outline_subpath(
                    &ring,
                    &ring_pens,
                    style,
                    6.0 * reach + 4.0 * step,
                ));
            }
            continue;
        }
        let Some((a, b)) = span_of(w, len, sub.closed) else {
            continue;
        };
        let Some((piece, idx)) = slice(sub, &seq, a, b) else {
            continue;
        };
        // Pens: full-path ones by index, boundary samples re-evaluated; under
        // trim, taper is re-evaluated against the trimmed run.
        let shift = wrap_shift(w, a, sub.closed);
        let extent = w
            .taper_extent
            .filter(|(e0, e1)| e0.is_finite() && e1.is_finite() && e1 > e0)
            .filter(|_| model.taper.is_some())
            .map(|(e0, e1)| (e0 + shift, e1 + shift));
        let piece_pens: Vec<Pen> = piece
            .samples
            .iter()
            .zip(&idx)
            .map(|(smp, k)| match (k, extent) {
                (Some(k), None) => pens[*k],
                (_, Some((e0, e1))) => pen_for(
                    model,
                    sub,
                    smp,
                    (smp.s_local - e0).clamp(0.0, e1 - e0),
                    e1 - e0,
                    total,
                ),
                (None, None) => pen_for(
                    model,
                    sub,
                    smp,
                    smp.s_local.rem_euclid(len.max(EPS)),
                    len,
                    total,
                ),
            })
            .collect();
        let reach = piece_pens.iter().map(Pen::reach).fold(0.0, f64::max);
        if reach <= 0.0 {
            continue;
        }
        out.extend(outline_subpath(
            &piece,
            &piece_pens,
            style,
            6.0 * reach + 4.0 * step,
        ));
    }
    out
}

/// The pen at `smp` with taper evaluated at `local` of `span`.
fn pen_for(
    model: &WidthModel,
    sub: &Subpath,
    smp: &Sample,
    local: f64,
    span: f64,
    total: f64,
) -> Pen {
    // `s` of a sample past the seam of a closed ring was shifted by +len;
    // fold it back for the whole-path fractions.
    let mut s = smp.s;
    let s0 = sub.samples[0].s - sub.samples[0].s_local;
    if s > s0 + sub.length + EPS {
        s -= sub.length;
    }
    let s_local = s - s0;
    let t_path = if total > 0.0 { s / total } else { 0.0 };
    let u = if model.profile.per_subpath {
        if sub.length > 0.0 {
            s_local / sub.length
        } else {
            0.0
        }
    } else {
        t_path
    };
    model.pen_at(u, smp.after, local, span, t_path, Some(super::heading(smp)))
}

/// The ring unrolled for window slicing, built once per subpath: the samples,
/// then (closed) the ring again shifted by `len`, then its first sample at
/// `2 len` — every window of a closed subpath (start in `[0, len)`, width
/// `≤ len`) lies inside it. Each entry keeps its index in `sub.samples`.
fn unroll(sub: &Subpath) -> Vec<(Sample, usize)> {
    let n = sub.samples.len();
    let len = sub.length;
    let mut seq: Vec<(Sample, usize)> = sub.samples.iter().copied().zip(0..n).collect();
    if sub.closed {
        for lap in 1..=2 {
            let shift = len * lap as f64;
            for (k, smp) in sub.samples.iter().enumerate() {
                if lap == 2 && k > 0 {
                    break;
                }
                let mut q = *smp;
                q.s_local += shift;
                q.s += shift;
                seq.push((q, k));
            }
        }
    }
    seq
}

/// The samples of `sub` in local arc length `[a, b]` (b may pass the seam of a
/// closed ring) as an open subpath, with the index of each sample in `sub`
/// (`None` for an interpolated end). `seq` = [`unroll`]`(sub)`; the window is
/// found by binary search, so a window costs O(log n + its own samples).
fn slice(
    sub: &Subpath,
    seq: &[(Sample, usize)],
    a: f64,
    b: f64,
) -> Option<(Subpath, Vec<Option<usize>>)> {
    let at = |target: f64| -> Option<(Sample, Option<usize>)> {
        // An exact sample (window ends are sampling events) or a lerp.
        let i = seq.partition_point(|(q, _)| q.s_local < target - 1e-7);
        if let Some((q, k)) = seq.get(i) {
            if (q.s_local - target).abs() <= 1e-7 {
                return Some((*q, Some(*k)));
            }
        }
        if i == 0 || i >= seq.len() {
            return None;
        }
        let (p, q) = (seq[i - 1].0, seq[i].0);
        let f = ((target - p.s_local) / (q.s_local - p.s_local).max(EPS)).clamp(0.0, 1.0);
        let dir = (q.p - p.p).unit().unwrap_or(p.t_out);
        Some((
            Sample {
                p: p.p.lerp(q.p, f),
                s: p.s + (q.s - p.s) * f,
                s_local: target,
                t_in: dir,
                t_out: dir,
                corner: false,
                after: false,
                brk: false,
                cusp: false,
            },
            None,
        ))
    };
    let (first, k0) = at(a)?;
    let (last, k1) = at(b)?;
    let mut samples = vec![first];
    let mut idx = vec![k0];
    let lo = seq.partition_point(|(q, _)| q.s_local <= a + 1e-7);
    let hi = seq.partition_point(|(q, _)| q.s_local < b - 1e-7);
    for (q, k) in &seq[lo..hi.max(lo)] {
        samples.push(*q);
        idx.push(Some(*k));
    }
    samples.push(last);
    idx.push(k1);
    if samples.len() < 2 {
        return None;
    }
    // Open ends: a corner sample at an end takes the direction of travel into
    // the piece; a discontinuity pair at an end keeps only the inner side.
    if samples.len() > 2 && samples[0].p.dist(samples[1].p) <= EPS && samples[1].after {
        samples.remove(0);
        idx.remove(0);
    }
    let m = samples.len();
    if m > 2 && samples[m - 1].p.dist(samples[m - 2].p) <= EPS && samples[m - 1].after {
        samples.pop();
        idx.pop();
    }
    let m = samples.len();
    samples[0].t_in = samples[0].t_out;
    samples[0].corner = false;
    samples[0].cusp = false;
    samples[m - 1].t_out = samples[m - 1].t_in;
    samples[m - 1].corner = false;
    samples[m - 1].cusp = false;
    Some((
        Subpath {
            samples,
            closed: false,
            length: b - a,
            index: sub.index,
        },
        idx,
    ))
}
