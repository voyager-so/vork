//! Offsetting: sampled centerline + per-sample half-widths → closed outline
//! contours (polylines with fitting breaks).
//!
//! Construction, per subpath:
//!
//! 1. Each side point is `c ± n·h`. On the side a curve bends TOWARD, the offset
//!    is clamped to the local radius of curvature (the side follows the evolute
//!    instead of folding back through a swallowtail), and the point is flagged.
//! 2. Corners: the outer side gets the join (miter / round / bevel); the inner
//!    side is routed through the corner point itself (the pivot), as strokers do.
//! 3. Local loops that contain a flagged point (clamped or pivot) and wind the
//!    same way as the band are cut at their self-intersection. They enclose
//!    area that is inside the stroke anyway, so the cut only removes overlaps.
//! 4. Open subpaths: left side, end cap, right side reversed, start cap → one
//!    contour. Closed subpaths: left ring and reversed right ring.
//!
//! Every contour winds the same way (the band is always on its right in
//! traversal order), so overlaps between pieces add up under the nonzero rule:
//! the fill is the union, with no holes where pieces overlap.

use super::centerline::{Subpath, EPS_LEN};
use super::nib::{ellipse_point, support, support_param};
use crate::path_geom::vec2::{v2, V2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Butt,
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Join {
    Miter,
    Round,
    Bevel,
}

/// Point flags.
pub(crate) const BREAK: u8 = 1;
/// Inner-join pivot or curvature-clamped point: loops through it may be cut.
pub(crate) const SOFT: u8 = 2;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SidePt {
    pub p: V2,
    /// Centerline arc length this point came from (loop-search window).
    pub s: f64,
    pub flags: u8,
}

/// A closed contour polyline (last point connects to the first).
#[derive(Debug, Clone, Default)]
pub(crate) struct Contour {
    pub pts: Vec<V2>,
    pub brk: Vec<bool>,
}

impl Contour {
    fn push(&mut self, p: V2, brk: bool) {
        if let Some(last) = self.pts.last() {
            if last.dist(p) <= EPS_LEN {
                let k = self.brk.len() - 1;
                self.brk[k] |= brk;
                return;
            }
        }
        self.pts.push(p);
        self.brk.push(brk);
    }
}

/// Round arcs are sampled at most this many radians per step.
const ARC_STEP: f64 = 0.15;

fn rotate(v: V2, a: f64) -> V2 {
    let (s, c) = a.sin_cos();
    v2(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// Signed angle from `a` to `b` (radians, −π..π].
fn angle_between(a: V2, b: V2) -> f64 {
    a.cross(b).atan2(a.dot(b))
}

/// Arc points strictly between `from` and `to` around `c`, radius blending
/// `r0 → r1`, sweeping `sweep` radians.
fn arc_points(c: V2, dir0: V2, sweep: f64, r0: f64, r1: f64, out: &mut Vec<V2>) {
    let steps = ((sweep.abs() / ARC_STEP).ceil() as usize).clamp(1, 256);
    for i in 1..steps {
        let f = i as f64 / steps as f64;
        out.push(c + rotate(dir0, sweep * f) * (r0 + (r1 - r0) * f));
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Style {
    pub cap: Cap,
    pub join: Join,
    pub miter_limit: f64,
    /// A clamped run whose evolute never strays further than this from the naive offset needs no fan: the sliver
    /// it would fill is within the fit tolerance (a radius-1 arc under a 2-wide stroke clamps by rounding noise,
    /// and one quad per sample there was 56 stray subpaths on the outlined lucide bell).
    pub fan_min: f64,
}

/// The pen at one sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Pen {
    /// Round pen (plain / profile / taper / wave strokes): half-widths left
    /// and right of travel, measured along the normal. Joins and caps follow
    /// the node's `lineJoin` / `lineCap`.
    Round { l: f64, r: f64 },
    /// Convex elliptical nib (calligraphic brush): semi-axes `a` (along `u`)
    /// and `b`, scaled by `fl` on the left and `fr` on the right (profile ×
    /// taper × wave). The outline is the exact sweep (Minkowski sum) of the
    /// nib: each side sits at the nib's SUPPORT point for the path normal (with
    /// its tangential component), corners get the nib's own boundary arc and
    /// open ends a half-nib cap — `lineJoin` / `lineCap` do not apply.
    Ellipse {
        a: f64,
        b: f64,
        u: V2,
        fl: f64,
        fr: f64,
    },
}

impl Pen {
    /// Offset from the centerline to the side (`left` or right) for the left
    /// normal `n`.
    pub(crate) fn offset(&self, n: V2, left: bool) -> V2 {
        match *self {
            Pen::Round { l, r } => {
                if left {
                    n * l
                } else {
                    n * -r
                }
            }
            Pen::Ellipse { a, b, u, fl, fr } => {
                if left {
                    support(a, b, u, n) * fl
                } else {
                    support(a, b, u, -n) * fr
                }
            }
        }
    }

    /// Half-widths along the normal `n` (what `samples` reports).
    pub(crate) fn normal_extent(&self, n: V2) -> (f64, f64) {
        (
            self.offset(n, true).dot(n).max(0.0),
            (-self.offset(n, false).dot(n)).max(0.0),
        )
    }

    /// Largest distance from the centerline anywhere on this pen.
    pub(crate) fn reach(&self) -> f64 {
        match *self {
            Pen::Round { l, r } => l.max(r),
            Pen::Ellipse { a, b, fl, fr, .. } => a.max(b) * fl.max(fr),
        }
    }
}

/// Points of the nib boundary strictly between the support points for
/// directions `d0` and `d1` (scaled `s0 → s1`), turning `turn` radians with
/// the directions (the nib's own arc — exact Minkowski at a corner or cap).
#[allow(clippy::too_many_arguments)]
fn nib_arc(c: V2, a: f64, b: f64, u: V2, d0: V2, turn: f64, s0: f64, s1: f64, out: &mut Vec<V2>) {
    let phi0 = support_param(a, b, u, d0);
    let d1 = rotate(d0, turn);
    let mut dphi = (support_param(a, b, u, d1) - phi0).rem_euclid(std::f64::consts::TAU);
    // The parameter turns the same way as the direction: pick that branch.
    if turn < 0.0 && dphi > 0.0 {
        dphi -= std::f64::consts::TAU;
    }
    if turn.abs() >= std::f64::consts::PI - 1e-9 && dphi.abs() < 1e-9 {
        dphi = turn.signum() * std::f64::consts::TAU;
    }
    let steps = ((dphi.abs() / ARC_STEP).ceil() as usize).clamp(1, 256);
    for i in 1..steps {
        let f = i as f64 / steps as f64;
        out.push(c + ellipse_point(a, b, u, phi0 + dphi * f) * (s0 + (s1 - s0) * f));
    }
}

/// Build the side polylines of one subpath. `hl[i]`, `hr[i]` are the half
/// widths at sample `i` (already including every factor).
/// Per sample and side: `(evolute point K, naive offset N)`; equal when the
/// side was not clamped. `None` at corners (they break fan runs).
type FanPt = Option<(V2, V2)>;

fn build_sides(
    sp: &Subpath,
    pens: &[Pen],
    style: &Style,
    window: f64,
    band: &Band,
    margin: f64,
) -> (Vec<SidePt>, Vec<SidePt>, Vec<FanPt>, Vec<FanPt>) {
    let smp = &sp.samples;
    let n = smp.len();
    // Index of each inner-corner pivot in `left` / `right`.
    let (mut piv_l, mut piv_r): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut left: Vec<SidePt> = Vec::with_capacity(n + 16);
    let mut right: Vec<SidePt> = Vec::with_capacity(n + 16);
    let mut fan_l: Vec<FanPt> = Vec::with_capacity(n);
    let mut fan_r: Vec<FanPt> = Vec::with_capacity(n);
    // Discrete curvature (signed: + bends left) for the clamp.
    let neighbour = |i: usize, dir: isize| -> Option<usize> {
        let mut j = i as isize;
        for _ in 0..n {
            j += dir;
            if j < 0 || j >= n as isize {
                if !sp.closed {
                    return None;
                }
                j = j.rem_euclid(n as isize);
            }
            let jj = j as usize;
            if jj == i {
                return None;
            }
            if smp[jj].p.dist(smp[i].p) > EPS_LEN {
                return Some(jj);
            }
        }
        None
    };
    for i in 0..n {
        let c = smp[i].p;
        let pen = pens[i];
        let brk = if smp[i].brk { BREAK } else { 0 };
        if !smp[i].corner {
            let nl = smp[i].t_out.left_normal();
            let (ol, or) = (pen.offset(nl, true), pen.offset(nl, false));
            // Normal components: what the clamp compares with the radius.
            let (l, r) = (ol.dot(nl).max(0.0), (-or.dot(nl)).max(0.0));
            let (mut cl, mut cr) = (l, r);
            let (mut fl, mut fr) = (brk, brk);
            if let (Some(a), Some(b)) = (neighbour(i, -1), neighbour(i, 1)) {
                let d0 = c - smp[a].p;
                let d1 = smp[b].p - c;
                let turn = angle_between(d0, d1);
                let ds = 0.5 * (d0.len() + d1.len());
                if turn.abs() > 1e-9 && ds > 0.0 {
                    let rho = ds / turn.abs();
                    // `left_normal` points left of travel; a turn toward it
                    // (dot > 0) bends left.
                    let bends_left = d1.dot(d0.left_normal()) > 0.0;
                    if bends_left && cl > rho {
                        cl = rho;
                        fl |= SOFT;
                    } else if !bends_left && cr > rho {
                        cr = rho;
                        fr |= SOFT;
                    }
                }
            }
            // A clamped side shrinks its whole offset (the segment from the
            // centre to the pen's support point lies inside the pen).
            let kl = if l > 0.0 { ol * (cl / l) } else { ol };
            let kr = if r > 0.0 { or * (cr / r) } else { or };
            left.push(SidePt {
                p: c + kl,
                s: smp[i].s,
                flags: fl,
            });
            right.push(SidePt {
                p: c + kr,
                s: smp[i].s,
                flags: fr,
            });
            fan_l.push(Some((c + kl, c + ol)));
            fan_r.push(Some((c + kr, c + or)));
            continue;
        }
        fan_l.push(None);
        fan_r.push(None);
        // Corner.
        let (ti, to) = (smp[i].t_in, smp[i].t_out);
        // A width step AT a corner (discontinuous marker on the anchor, or a
        // closed seam's end/start widths): the coincident pair would give two
        // joins. The incoming sample only ends its leg and steps into the
        // pivot; the outgoing sample draws the one join, at its width.
        let next = if i + 1 < n {
            Some(i + 1)
        } else if sp.closed {
            Some(0)
        } else {
            None
        };
        if let Some(j) = next.filter(|&j| j != i && smp[j].corner && smp[j].p.dist(c) <= EPS_LEN) {
            let ni = ti.left_normal();
            let s = smp[i].s;
            for (side, is_left) in [(&mut left, true), (&mut right, false)] {
                let p = c + pen.offset(ni, is_left);
                side.push(SidePt { p, s, flags: BREAK });
                // No step (a closed seam at an even width): the outgoing join starts right here, so a detour
                // through the pivot would only be a spike.
                if pens[j].offset(ni, is_left).dist(p - c) > EPS_LEN {
                    side.push(SidePt {
                        p: c,
                        s,
                        flags: BREAK | SOFT,
                    });
                }
            }
            continue;
        }
        // Path vertices take the node's join; a cusp inside a curve is round.
        let join = if smp[i].cusp { Join::Round } else { style.join };
        let (ni, no) = (ti.left_normal(), to.left_normal());
        let bends_left = to.dot(ni) > 0.0;
        let s = smp[i].s;
        let inner = |side: &mut Vec<SidePt>, pivots: &mut Vec<usize>, is_left: bool| {
            side.push(SidePt {
                p: c + pen.offset(ni, is_left),
                s,
                flags: BREAK,
            });
            pivots.push(side.len());
            side.push(SidePt {
                p: c,
                s,
                flags: BREAK | SOFT,
            });
            side.push(SidePt {
                p: c + pen.offset(no, is_left),
                s,
                flags: BREAK,
            });
        };
        let outer = |side: &mut Vec<SidePt>, is_left: bool| {
            let sign = if is_left { 1.0 } else { -1.0 };
            let a = ni * sign;
            let b = no * sign;
            let mut sweep = angle_between(a, b);
            // A full reversal: sweep through the forward direction.
            if (sweep.abs() - std::f64::consts::PI).abs() < 1e-9 {
                sweep = if rotate(a, sweep * 0.5).dot(ti) >= 0.0 {
                    sweep
                } else {
                    -sweep
                };
            }
            if let Pen::Ellipse {
                a: ea,
                b: eb,
                u,
                fl,
                fr,
            } = pen
            {
                // The nib's own boundary between the two support points.
                let f = if is_left { fl } else { fr };
                side.push(SidePt {
                    p: c + pen.offset(ni, is_left),
                    s,
                    flags: BREAK,
                });
                let mut pts = Vec::new();
                nib_arc(c, ea, eb, u, a, sweep, f, f, &mut pts);
                for p in pts {
                    side.push(SidePt { p, s, flags: 0 });
                }
                side.push(SidePt {
                    p: c + pen.offset(no, is_left),
                    s,
                    flags: BREAK,
                });
                return;
            }
            let h = pen.offset(ni, is_left).len();
            side.push(SidePt {
                p: c + a * h,
                s,
                flags: BREAK,
            });
            if h > EPS_LEN {
                match join {
                    Join::Bevel => {}
                    Join::Miter => {
                        let half = sweep.abs() * 0.5;
                        let ratio = 1.0 / half.cos().max(1e-12);
                        if ratio <= style.miter_limit && half < std::f64::consts::FRAC_PI_2 - 1e-6 {
                            if let Some(bis) = (a + b).unit() {
                                side.push(SidePt {
                                    p: c + bis * (h * ratio),
                                    s,
                                    flags: BREAK,
                                });
                            }
                        }
                    }
                    Join::Round => {
                        let mut pts = Vec::new();
                        arc_points(c, a, sweep, h, h, &mut pts);
                        for p in pts {
                            side.push(SidePt { p, s, flags: 0 });
                        }
                    }
                }
            }
            side.push(SidePt {
                p: c + b * h,
                s,
                flags: BREAK,
            });
        };
        if bends_left {
            inner(&mut left, &mut piv_l, true);
            outer(&mut right, false);
        } else {
            outer(&mut left, true);
            inner(&mut right, &mut piv_r, false);
        }
    }
    let left = meet_inner_corners(
        left,
        &piv_l,
        window,
        band,
        margin,
        sp.closed.then_some(sp.length),
    );
    let right = meet_inner_corners(
        right,
        &piv_r,
        window,
        band,
        margin,
        sp.closed.then_some(sp.length),
    );
    (left, right, fan_l, fan_r)
}

/// Segment intersection with both parameters in `[0, 1]`.
fn seg_cross(a0: V2, a1: V2, b0: V2, b1: V2) -> Option<(f64, f64)> {
    let r = a1 - a0;
    let s = b1 - b0;
    let den = r.cross(s);
    if den.abs() <= 1e-12 * (r.len() * s.len()).max(1e-300) {
        return None;
    }
    let q = b0 - a0;
    let ta = q.cross(s) / den;
    let tb = q.cross(r) / den;
    const E: f64 = 1e-9;
    ((-E..=1.0 + E).contains(&ta) && (-E..=1.0 + E).contains(&tb)).then_some((ta, tb))
}

/// Inner sides of a corner meet where the incoming and outgoing offsets cross, as Illustrator's outline does,
/// instead of running through the pivot (the centreline point, an overlap only the nonzero fill hid, and two
/// extra anchors). Each pivot's incoming side is searched back and its outgoing side forward (within `window` of
/// centreline arc length, never past another pivot) for the crossing nearest the corner; the run between is
/// replaced by it. Where the offsets don't cross (legs shorter than the half width, a hairpin wider than the
/// window) or the run it would drop is not all inside the stroke (a leg curling round so the other side crosses
/// it far off, enclosing real outline), the pivot stays: it is inside the stroke, so the fill is still right.
fn meet_inner_corners(
    side: Vec<SidePt>,
    pivots: &[usize],
    window: f64,
    band: &Band,
    margin: f64,
    period: Option<f64>,
) -> Vec<SidePt> {
    let n = side.len();
    if pivots.is_empty() || n < 5 {
        return side;
    }
    if let Some(len) = period {
        // A ring: start it midway between the two pivots farthest apart, so no corner sits at its ends, meet the
        // corners, then turn it back to start where it did (the least `s`: it grows round the ring from there).
        let mut ps = pivots.to_vec();
        ps.sort_unstable();
        let (mut gap, mut at) = (0, 0);
        for (k, &q) in ps.iter().enumerate() {
            let nx = if k + 1 < ps.len() {
                ps[k + 1]
            } else {
                ps[0] + n
            };
            if nx - q > gap {
                gap = nx - q;
                at = (q + (nx - q) / 2) % n;
            }
        }
        let mut turned = side;
        turned.rotate_left(at);
        let mut ps: Vec<usize> = ps.iter().map(|&q| (q + n - at) % n).collect();
        ps.sort_unstable();
        let mut out = meet_ring(turned, &ps, window, band, margin, len);
        let first = (0..out.len())
            .min_by(|&x, &y| {
                out[x]
                    .s
                    .partial_cmp(&out[y].s)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        out.rotate_left(first);
        return out;
    }
    meet_ring(side, pivots, window, band, margin, f64::INFINITY)
}

/// [`meet_inner_corners`] on a linear run of side points whose `s` repeats every `period` (a ring's length;
/// infinite for an open side).
fn meet_ring(
    side: Vec<SidePt>,
    pivots: &[usize],
    window: f64,
    band: &Band,
    margin: f64,
    period: f64,
) -> Vec<SidePt> {
    let n = side.len();
    let near = |s: f64, sc: f64| {
        let d = (s - sc).abs();
        d.min((period - d).abs()) <= window
    };
    let is_pivot = |k: usize| side[k].flags & (BREAK | SOFT) == BREAK | SOFT;
    // Total segment tests, whatever the input.
    let mut budget: usize = 64 * n + 65_536;
    // (first index dropped, first index kept after, crossing).
    let mut cuts: Vec<(usize, usize, V2, f64)> = Vec::new();
    for &p in pivots {
        if p < 2 || p + 2 >= n {
            continue;
        }
        let sc = side[p].s;
        // Incoming segments `side[a-1] → side[a]`, a = p-1, p-2, …; outgoing `side[b] → side[b+1]`, b = p+1, ….
        let mut ins: Vec<usize> = Vec::new();
        let mut a = p - 1;
        while a >= 1 && !is_pivot(a - 1) && near(side[a].s, sc) && ins.len() < 512 {
            ins.push(a);
            a -= 1;
        }
        let mut outs: Vec<usize> = Vec::new();
        let mut b = p + 1;
        while b + 1 < n && !is_pivot(b + 1) && near(side[b].s, sc) && outs.len() < 512 {
            outs.push(b);
            b += 1;
        }
        let mut best: Option<(usize, usize, usize, V2)> = None;
        'search: for (ki, &a) in ins.iter().enumerate() {
            if best.is_some_and(|(d, ..)| ki > d) {
                break;
            }
            for (mi, &b) in outs.iter().enumerate() {
                if budget == 0 {
                    break 'search;
                }
                budget -= 1;
                if best.is_some_and(|(d, ..)| ki + mi >= d) {
                    break;
                }
                if let Some((ta, _)) = seg_cross(side[a - 1].p, side[a].p, side[b].p, side[b + 1].p)
                {
                    best = Some((ki + mi, a, b + 1, side[a - 1].p.lerp(side[a].p, ta)));
                }
            }
        }
        if let Some((_, a, b, x)) = best {
            let (lo, hi) = (sc - 2.0 * window, sc + 2.0 * window);
            let shifts: &[f64] = if period.is_finite() {
                &[0.0, period, -period]
            } else {
                &[0.0]
            };
            let inside = side[a..b].iter().all(|q| {
                q.flags & SOFT != 0
                    || shifts
                        .iter()
                        .any(|d| band.covered(q.p, lo + d, hi + d, margin, &mut budget))
            });
            if inside {
                cuts.push((a, b, x, sc));
            }
        }
    }
    // Pivots come in order, so cuts do; skip any that overlaps the one before (two corners sharing a short leg).
    let mut out: Vec<SidePt> = Vec::with_capacity(n);
    let mut next = 0;
    for (a, b, x, s) in cuts {
        if a < next {
            continue;
        }
        out.extend_from_slice(&side[next..a]);
        out.push(SidePt {
            p: x,
            s,
            flags: BREAK | SOFT,
        });
        #[cfg(test)]
        LOOP_CUTS.with(|c| c.set(c.get() + 1));
        next = b;
    }
    out.extend_from_slice(&side[next..]);
    out
}

/// Where the clamp moved a side point to the evolute, the band between the
/// evolute and the naive offset is still inside the stroke (it is covered by
/// the sample's own normal segment) but no longer inside the side contour.
/// Put it back as extra contours: one polygon per clamped run when its pieces
/// all turn the same way, otherwise one quad per sample pair, each wound like
/// the band so the nonzero fill stays a union.
fn fan_contours(fan: &[FanPt], closed: bool, min: f64, out: &mut Vec<Contour>, budget: &mut usize) {
    let n = fan.len();
    let clamped = |i: usize| fan[i].is_some_and(|(k, nv)| k.dist(nv) > EPS_LEN);
    if n < 2 || !(0..n).any(clamped) {
        return;
    }
    // Runs of clamped samples, extended by one neighbour on each side (where
    // K = N) so the fan closes onto the unclamped side.
    let mut i = 0;
    let all = (0..n).all(clamped);
    while i < n {
        if !clamped(i) {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && clamped(i) {
            i += 1;
        }
        let mut idx: Vec<usize> = Vec::new();
        if start > 0 {
            idx.push(start - 1);
        } else if closed && !all {
            idx.push(n - 1);
        }
        idx.extend(start..i);
        if i < n {
            idx.push(i);
        } else if closed {
            idx.push(0);
        }
        let run: Vec<(V2, V2)> = idx.iter().filter_map(|&k| fan[k]).collect();
        if run.iter().all(|(k, nv)| k.dist(*nv) <= min) {
            continue;
        }
        if run.len() < 2 || *budget < run.len() * 4 {
            continue;
        }
        *budget -= run.len() * 4;
        let quad = |a: (V2, V2), b: (V2, V2)| [a.0, b.0, b.1, a.1];
        let signs: Vec<f64> = run
            .windows(2)
            .map(|w| shoelace(&quad(w[0], w[1])))
            .collect();
        let pos = signs.iter().all(|a| *a >= 0.0);
        let neg = signs.iter().all(|a| *a <= 0.0);
        if pos || neg {
            let mut poly: Vec<V2> = run.iter().map(|p| p.0).collect();
            poly.extend(run.iter().rev().map(|p| p.1));
            if shoelace(&poly) < 0.0 {
                poly.reverse();
            }
            let mut c = Contour::default();
            for p in poly {
                c.push(p, true);
            }
            if c.pts.len() >= 3 {
                out.push(c);
            }
        } else {
            for w in run.windows(2) {
                let mut q = quad(w[0], w[1]).to_vec();
                if shoelace(&q) < 0.0 {
                    q.reverse();
                }
                let mut c = Contour::default();
                for p in q {
                    c.push(p, true);
                }
                if c.pts.len() >= 3 && shoelace(&c.pts).abs() > EPS_LEN {
                    out.push(c);
                }
            }
        }
    }
}

/// The band as per-chord trapezoids (sorted by `s`): "is this point strictly
/// inside the stroke?" for deciding whether a loop is an overlap (removable)
/// or real outline (a curl's outer bump).
pub(crate) struct Band {
    pieces: Vec<(f64, f64, [V2; 4])>,
    /// Per piece (same order): its centreline chord and, for a round pen, the smaller half-width at its ends (the
    /// capsule round the chord is inside the stroke).
    capsules: Vec<(V2, V2, f64)>,
}

impl Band {
    pub(crate) fn new(sp: &Subpath, pens: &[Pen]) -> Band {
        let smp = &sp.samples;
        let n = smp.len();
        let mut pieces = Vec::with_capacity(n);
        let mut capsules = Vec::with_capacity(n);
        let count = if sp.closed { n } else { n.saturating_sub(1) };
        for i in 0..count {
            let j = (i + 1) % n;
            let (a, b) = (smp[i].p, smp[j].p);
            let Some(d) = (b - a).unit() else { continue };
            let nl = d.left_normal();
            let (s0, mut s1) = (smp[i].s, smp[j].s);
            if s1 < s0 {
                s1 = s0 + (b - a).len();
            }
            let h = match (pens[i], pens[j]) {
                (Pen::Round { l: l0, r: r0 }, Pen::Round { l: l1, r: r1 }) => {
                    l0.min(r0).min(l1).min(r1)
                }
                _ => 0.0,
            };
            capsules.push((a, b, h));
            pieces.push((
                s0,
                s1,
                [
                    a + pens[i].offset(nl, true),
                    b + pens[j].offset(nl, true),
                    b + pens[j].offset(nl, false),
                    a + pens[i].offset(nl, false),
                ],
            ));
        }
        let mut order: Vec<usize> = (0..pieces.len()).collect();
        order.sort_by(|&x, &y| {
            pieces[x]
                .0
                .partial_cmp(&pieces[y].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Band {
            pieces: order.iter().map(|&k| pieces[k]).collect(),
            capsules: order.iter().map(|&k| capsules[k]).collect(),
        }
    }

    /// [`Band::interior`], or strictly inside the capsule of a round-pen piece whose `s` range meets `[lo, hi]`
    /// (covers the wedges between a curve's trapezoids, which a round pen fills).
    fn covered(&self, p: V2, lo: f64, hi: f64, margin: f64, budget: &mut usize) -> bool {
        if self.interior(p, lo, hi, margin, budget) {
            return true;
        }
        let start = self.pieces.partition_point(|x| x.0 < lo).saturating_sub(1);
        for (k, (s0, s1, _)) in self.pieces.iter().enumerate().skip(start) {
            if *budget == 0 || *s0 > hi {
                return false;
            }
            *budget -= 1;
            if *s1 < lo {
                continue;
            }
            let (a, b, h) = self.capsules[k];
            if h <= margin {
                continue;
            }
            let d = b - a;
            let len2 = d.dot(d);
            let t = if len2 > 0.0 {
                ((p - a).dot(d) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if p.dist(a + d * t) < h - margin {
                return true;
            }
        }
        false
    }

    /// Strictly inside some trapezoid whose `s` range meets `[lo, hi]`, by at
    /// least `margin` from its edges.
    fn interior(&self, p: V2, lo: f64, hi: f64, margin: f64, budget: &mut usize) -> bool {
        // Pieces are sorted by start; a piece spans at most one sample step, so
        // starting a little before `lo` catches every piece that reaches it.
        let start = self.pieces.partition_point(|x| x.0 < lo).saturating_sub(1);
        for (s0, s1, q) in &self.pieces[start..] {
            if *budget == 0 {
                return false;
            }
            *budget -= 1;
            if *s0 > hi {
                break;
            }
            if *s1 < lo {
                continue;
            }
            // Convex quad, either orientation: inside iff on the same side of
            // every edge by `margin`.
            let mut sign = 0.0;
            let mut inside = true;
            for k in 0..4 {
                let (e0, e1) = (q[k], q[(k + 1) % 4]);
                let Some(e) = (e1 - e0).unit() else { continue };
                let d = e.cross(p - e0);
                if d.abs() <= margin {
                    inside = false;
                    break;
                }
                if sign == 0.0 {
                    sign = d.signum();
                } else if d.signum() != sign {
                    inside = false;
                    break;
                }
            }
            if inside && sign != 0.0 {
                return true;
            }
        }
        false
    }
}

fn seg_intersect(a0: V2, a1: V2, b0: V2, b1: V2) -> Option<(f64, f64)> {
    let r = a1 - a0;
    let s = b1 - b0;
    let den = r.cross(s);
    if den.abs() <= 1e-18 * (r.len2() + s.len2()).max(1e-300) {
        return None;
    }
    let q = b0 - a0;
    let ta = q.cross(s) / den;
    let tb = q.cross(r) / den;
    const E: f64 = 1e-12;
    (ta > E && ta <= 1.0 + E && tb >= -E && tb < 1.0 - E).then_some((ta, tb))
}

fn shoelace(pts: &[V2]) -> f64 {
    let n = pts.len();
    let mut a = 0.0;
    for i in 0..n {
        let (p, q) = (pts[i], pts[(i + 1) % n]);
        a += p.x * q.y - q.x * p.y;
    }
    0.5 * a
}

#[cfg(test)]
thread_local! {
    /// Loops `trim_loops` cut (and inner corners `meet_inner_corners` met) on this thread (tests read it to see the cutting happen).
    pub(crate) static LOOP_CUTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Cut local loops that pass through a SOFT point and wind like the band
/// (positive shoelace on a y-down canvas, see module docs). `pts` must be in
/// contour order. `window` bounds the centerline arc length a loop may span.
pub(crate) fn trim_loops(pts: &[SidePt], window: f64, band: &Band, margin: f64) -> Vec<SidePt> {
    let n = pts.len();
    if n < 4 {
        return pts.to_vec();
    }
    // soft_prefix[k] = number of SOFT points in pts[..k].
    let mut soft_prefix = Vec::with_capacity(n + 1);
    soft_prefix.push(0usize);
    for p in pts {
        let last = *soft_prefix.last().unwrap_or(&0);
        soft_prefix.push(last + usize::from(p.flags & SOFT != 0));
    }
    if soft_prefix[n] == 0 {
        return pts.to_vec();
    }
    let mut out: Vec<SidePt> = Vec::with_capacity(n);
    out.push(pts[0]);
    let mut cur = pts[0].p;
    let mut i = 0;
    // Total work (segment tests + band lookups) is bounded, whatever the input.
    let mut budget: usize = 256 * n + 65_536;
    while i + 1 < n {
        let a1 = pts[i + 1].p;
        let (amin, amax) = (
            v2(cur.x.min(a1.x), cur.y.min(a1.y)),
            v2(cur.x.max(a1.x), cur.y.max(a1.y)),
        );
        let mut best: Option<(usize, V2)> = None;
        let s0 = pts[i + 1].s;
        let mut j = i + 2;
        // Only worth scanning if a SOFT point lies ahead within the window.
        while j + 1 < n && (pts[j].s - s0).abs() <= window && budget > 0 {
            budget -= 1;
            if soft_prefix[j + 1] - soft_prefix[i + 1] == 0 {
                j += 1;
                continue;
            }
            let (b0, b1) = (pts[j].p, pts[j + 1].p);
            if b0.x.max(b1.x) < amin.x
                || b0.x.min(b1.x) > amax.x
                || b0.y.max(b1.y) < amin.y
                || b0.y.min(b1.y) > amax.y
            {
                j += 1;
                continue;
            }
            if let Some((ta, _tb)) = seg_intersect(cur, a1, b0, b1) {
                let x = cur.lerp(a1, ta);
                // Loop: x, pts[i+1..=j], back to x.
                let mut poly: Vec<V2> = Vec::with_capacity(j - i + 1);
                poly.push(x);
                poly.extend(pts[i + 1..=j].iter().map(|p| p.p));
                if shoelace(&poly) > 0.0
                    && loop_is_overlap(&pts[i + 1..=j], band, window, margin, &mut budget)
                {
                    best = Some((j, x));
                }
            }
            j += 1;
        }
        if let Some((j, x)) = best {
            #[cfg(test)]
            LOOP_CUTS.with(|c| c.set(c.get() + 1));
            #[cfg(test)]
            if std::env::var("STROKE_DEBUG").is_ok() {
                eprintln!(
                    "cut loop {}..{} at ({:.2},{:.2}) s {:.1}..{:.1}",
                    i + 1,
                    j,
                    x.x,
                    x.y,
                    pts[i + 1].s,
                    pts[j].s
                );
            }
            out.push(SidePt {
                p: x,
                s: pts[j].s,
                flags: BREAK,
            });
            cur = x;
            i = j;
        } else {
            out.push(pts[i + 1]);
            cur = a1;
            i += 1;
        }
    }
    out
}

/// A loop is an overlap (safe to cut) when (nearly) all of its vertices lie
/// strictly inside the band — pivots and clamped points are inside by
/// construction. Ears and swallowtails fold back into the stroke; a curl's
/// outer bump runs along the true outline and is kept.
fn loop_is_overlap(
    pts: &[SidePt],
    band: &Band,
    window: f64,
    margin: f64,
    budget: &mut usize,
) -> bool {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in pts {
        lo = lo.min(p.s);
        hi = hi.max(p.s);
    }
    let (lo, hi) = (lo - window, hi + window);
    let outside = pts
        .iter()
        .filter(|p| p.flags & SOFT == 0 && !band.interior(p.p, lo, hi, margin, budget))
        .count();
    outside * 20 <= pts.len()
}

/// An open end. `from` / `to` are the side points being joined; `from` is on
/// the left (`from_left`) or right side; `n` is the left normal there and `t`
/// points away from the stroke.
#[allow(clippy::too_many_arguments)]
fn end_cap(
    style: &Style,
    pen: &Pen,
    c: V2,
    n: V2,
    from_left: bool,
    from: V2,
    to: V2,
    t: V2,
    contour: &mut Contour,
) {
    if let Pen::Ellipse { a, b, u, fl, fr } = *pen {
        // Half the nib, from the support point on the `from` side around the
        // back (through `t`) to the opposite one.
        let (d0, s0, s1) = if from_left { (n, fl, fr) } else { (-n, fr, fl) };
        let turn = if rotate(d0, std::f64::consts::FRAC_PI_2).dot(t) >= 0.0 {
            std::f64::consts::PI
        } else {
            -std::f64::consts::PI
        };
        let mut pts = Vec::new();
        nib_arc(c, a, b, u, d0, turn, s0, s1, &mut pts);
        for p in pts {
            contour.push(p, false);
        }
        return;
    }
    cap_points(style, from, to, t, contour);
}

fn cap_points(style: &Style, from: V2, to: V2, t: V2, contour: &mut Contour) {
    // `from` → `to` across the end, bulging along `t`.
    let d = to - from;
    let r = d.len() * 0.5;
    if r <= EPS_LEN {
        return;
    }
    match style.cap {
        Cap::Butt => {}
        Cap::Square => {
            contour.push(from + t * r, true);
            contour.push(to + t * r, true);
        }
        Cap::Round => {
            let m = from.lerp(to, 0.5);
            let a = (from - m) * (1.0 / r);
            // Sweep from `from` through `t` to `to`.
            let sweep = if angle_between(a, t) >= 0.0 {
                std::f64::consts::PI
            } else {
                -std::f64::consts::PI
            };
            let mut pts = Vec::new();
            arc_points(m, a, sweep, r, r, &mut pts);
            for p in pts {
                contour.push(p, false);
            }
        }
    }
}

/// Twice the signed area of a contour (shoelace, canvas coordinates).
pub(crate) fn signed_area(c: &Contour) -> f64 {
    let n = c.pts.len();
    (0..n)
        .map(|i| {
            let (p, q) = (c.pts[i], c.pts[(i + 1) % n]);
            p.x * q.y - q.x * p.y
        })
        .sum()
}

/// The region a path-relative nib covers turning in place at a corner, from
/// its incoming orientation (`pin`) to its outgoing one (`pout`) the shorter
/// way: the union of the turning ellipse, a polygon star-shaped around `c`
/// (the farthest reach of the nib in each direction over the turn).
pub(crate) fn nib_turn_contour(c: V2, pin: &Pen, pout: &Pen) -> Option<Contour> {
    let (
        Pen::Ellipse {
            a,
            b,
            u: u0,
            fl: fl0,
            fr: fr0,
        },
        Pen::Ellipse {
            u: u1,
            fl: fl1,
            fr: fr1,
            ..
        },
    ) = (*pin, *pout)
    else {
        return None;
    };
    let f = 0.25 * (fl0 + fr0 + fl1 + fr1);
    if !(a * f > EPS_LEN) {
        return None;
    }
    let th0 = u0.y.atan2(u0.x);
    let turn = u0.cross(u1).atan2(u0.dot(u1));
    // Radius of the ellipse in a direction `alpha` from its major axis.
    let rho = |alpha: f64| {
        let (s, co) = alpha.sin_cos();
        let den = ((b * co).powi(2) + (a * s).powi(2)).sqrt();
        if den > 0.0 {
            a * b / den
        } else {
            a
        }
    };
    const TURN_STEPS: usize = 32;
    const DIRECTIONS: usize = 96;
    let mut out = Contour::default();
    for j in 0..DIRECTIONS {
        let phi = std::f64::consts::TAU * j as f64 / DIRECTIONS as f64;
        let r = (0..=TURN_STEPS)
            .map(|k| rho(phi - (th0 + turn * k as f64 / TURN_STEPS as f64)))
            .fold(0.0, f64::max);
        out.push(c + v2(phi.cos(), phi.sin()) * (r * f), false);
    }
    (out.pts.len() >= 3).then_some(out)
}

/// A dot for a zero-length subpath (SVG: round and square caps draw one).
pub(crate) fn dot_contour(style: &Style, pen: &Pen, c: V2) -> Option<Contour> {
    let h = match *pen {
        Pen::Round { l, r } => 0.5 * (l + r),
        Pen::Ellipse { a, b, u, fl, fr } => {
            // A click with a nib prints the nib.
            let f = 0.5 * (fl + fr);
            if a * f <= EPS_LEN {
                return None;
            }
            let mut out = Contour::default();
            for i in 0..48 {
                let phi = std::f64::consts::TAU * i as f64 / 48.0;
                out.push(c + ellipse_point(a, b, u, phi) * f, false);
            }
            return Some(out);
        }
    };
    if h <= EPS_LEN {
        return None;
    }
    let mut out = Contour::default();
    match style.cap {
        Cap::Butt => return None,
        Cap::Square => {
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                out.push(c + v2(x, y) * h, true);
            }
        }
        Cap::Round => {
            let steps = 32;
            for i in 0..steps {
                let a = std::f64::consts::TAU * i as f64 / steps as f64;
                out.push(c + v2(a.cos(), -a.sin()) * h, false);
            }
        }
    }
    Some(out)
}

fn push_side(contour: &mut Contour, side: &[SidePt]) {
    for p in side {
        contour.push(p.p, p.flags & BREAK != 0);
    }
}

/// Offset one subpath into contours. `window` bounds loop trimming (centerline
/// arc length).
pub(crate) fn outline_subpath(
    sp: &Subpath,
    pens: &[Pen],
    style: &Style,
    window: f64,
) -> Vec<Contour> {
    let n = sp.samples.len();
    if n < 2 {
        return Vec::new();
    }
    let band = Band::new(sp, pens);
    let hmax = pens.iter().map(Pen::reach).fold(0.0, f64::max);
    let margin = 1e-3 * hmax + 1e-9;
    let (left, mut right, fan_l, fan_r) = build_sides(sp, pens, style, window, &band, margin);
    let mut fans: Vec<Contour> = Vec::new();
    let mut fan_budget = 8 * n + 64;
    fan_contours(&fan_l, sp.closed, style.fan_min, &mut fans, &mut fan_budget);
    fan_contours(&fan_r, sp.closed, style.fan_min, &mut fans, &mut fan_budget);
    let left = trim_loops(&left, window, &band, margin);
    right.reverse();
    let right = trim_loops(&right, window, &band, margin);
    if sp.closed {
        let mut a = Contour::default();
        push_side(&mut a, &left);
        let mut b = Contour::default();
        push_side(&mut b, &right);
        let mut out: Vec<Contour> = [a, b].into_iter().filter(|c| c.pts.len() >= 3).collect();
        out.extend(fans);
        return out;
    }
    let mut c = Contour::default();
    push_side(&mut c, &left);
    let (Some(le), Some(re)) = (left.last(), right.first()) else {
        return Vec::new();
    };
    let last = &sp.samples[n - 1];
    if let Some(k) = c.brk.last_mut() {
        *k = true;
    }
    end_cap(
        style,
        &pens[n - 1],
        last.p,
        last.t_out.left_normal(),
        true,
        le.p,
        re.p,
        last.t_out,
        &mut c,
    );
    let before = c.pts.len();
    push_side(&mut c, &right);
    if let Some(k) = c.brk.get_mut(before) {
        *k = true;
    }
    if let Some(k) = c.brk.last_mut() {
        *k = true;
    }
    let first = &sp.samples[0];
    let (Some(rs), Some(ls)) = (right.last(), left.first()) else {
        return Vec::new();
    };
    end_cap(
        style,
        &pens[0],
        first.p,
        first.t_in.left_normal(),
        false,
        rs.p,
        ls.p,
        -first.t_in,
        &mut c,
    );
    if let Some(k) = c.brk.first_mut() {
        *k = true;
    }
    // Drop a closing duplicate.
    if c.pts.len() > 1 && c.pts[0].dist(c.pts[c.pts.len() - 1]) <= EPS_LEN {
        c.pts.pop();
        let b = c.brk.pop().unwrap_or(false);
        c.brk[0] |= b;
    }
    if c.pts.len() < 3 {
        return fans;
    }
    let mut out = vec![c];
    out.extend(fans);
    out
}
