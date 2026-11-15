//! The second set of Illustrator live effects (E8.1b): Scribble (Stylize),
//! Tweak and Free Distort (Distort & Transform), and Convert to Shape. Each
//! was measured against Illustrator 30.8 through its scripting DOM
//! (`applyEffect` with the plug-in's own dictionary keys, then Expand
//! Appearance); the findings are in the doc comment of each function and in
//! `docs/roadmap/design-editor/live-effects.md`.

use crate::value::{json, Value};

use super::{bounds_of, map_shape, Bounds, Contour, Seg, Shape, SplitMix64};
use crate::blend::BlendPathPoint;
use crate::effect::{ConvertToShapeEffect, FreeDistortEffect, ScribbleEffect, TweakEffect};
use crate::path_geom::vec2::{v2, V2};
use crate::stroke_outline::{stroke_outline, Outline, OutlineOptions};
use crate::types::RawNode;

/// The circle's cubic handle length, as a fraction of the radius.
const KAPPA: f64 = 0.552_284_749_830_793_4;

fn fin(v: f64, d: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        d
    }
}

fn opt(v: Option<f64>, d: f64) -> f64 {
    v.filter(|v| v.is_finite()).unwrap_or(d)
}

fn seeded(seed: Option<f64>, salt: u64) -> SplitMix64 {
    let s = opt(seed, 0.0).abs().floor() as u64;
    SplitMix64(s ^ salt)
}

// ---------------------------------------------------------------------------
// Free Distort
// ---------------------------------------------------------------------------

/// The default Free Distort corners: the bounds themselves (no change).
pub const FREE_DISTORT_IDENTITY: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

/// The four target corners (TL, TR, BR, BL, in the bounds' unit space), or the
/// identity when the entry does not hold four finite pairs.
fn free_distort_corners(e: &FreeDistortEffect) -> [[f64; 2]; 4] {
    if e.corners.len() != 4 || e.corners.iter().flatten().any(|v| !v.is_finite()) {
        return FREE_DISTORT_IDENTITY;
    }
    [e.corners[0], e.corners[1], e.corners[2], e.corners[3]]
}

/// Free Distort, Illustrator's way: a BILINEAR map of the art's bounds onto the
/// four corners, applied to every anchor and handle with no subdivision (an
/// ellipse keeps its 4 anchors; a straight segment stays straight). Measured:
/// a circle in a trapezoid lands its side anchors and handles exactly on the
/// bilinear images of the originals (a projective map would move them).
pub fn free_distort(shape: &Shape, e: &FreeDistortEffect, bounds: Option<Bounds>) -> Shape {
    let Some(b) = bounds else {
        return shape.clone();
    };
    let (w, h) = (b.width(), b.height());
    if !(w > 0.0) || !(h > 0.0) {
        return shape.clone();
    }
    let c = free_distort_corners(e);
    let q = c.map(|[u, v]| v2(b.min.x + u * w, b.min.y + v * h));
    let f = move |p: V2| {
        let u = (p.x - b.min.x) / w;
        let v = (p.y - b.min.y) / h;
        let top = q[0].lerp(q[1], u);
        let bottom = q[3].lerp(q[2], u);
        top.lerp(bottom, v)
    };
    map_shape(shape, &f, true)
}

// ---------------------------------------------------------------------------
// Tweak
// ---------------------------------------------------------------------------

/// Tweak: every anchor, "In" handle (the one before the anchor along the
/// path) and "Out" handle moves by its own uniform random offset in
/// [−amount, amount] on each axis. Relative amounts are a percentage of the
/// bounds' width (horizontal) and height (vertical); Illustrator: 10 % on a
/// 160 × 60 box moves points up to 16 px across and 6 px down. Moving only the
/// anchors leaves the handles where they were (a straight segment then bends,
/// its handles still on the old anchors), as Illustrator does. A seed always
/// gives the same shape; every draw is made whether or not its part moves, so
/// toggling a checkbox never reshuffles the others.
pub fn tweak(
    shape: &Shape,
    e: &TweakEffect,
    bounds: Option<Bounds>,
    rng: &mut SplitMix64,
) -> Shape {
    let absolute = e.size_mode.as_deref() == Some("absolute");
    let (bw, bh) = bounds
        .map(|b| (b.width(), b.height()))
        .unwrap_or((0.0, 0.0));
    let ax = fin(e.horizontal, 0.0).abs() * if absolute { 1.0 } else { bw / 100.0 };
    let ay = fin(e.vertical, 0.0).abs() * if absolute { 1.0 } else { bh / 100.0 };
    let (do_anchor, do_in, do_out) = (
        e.anchors != Some(false),
        e.in_handles != Some(false),
        e.out_handles != Some(false),
    );
    let mut draw = |on: bool| {
        let d = v2(rng.signed_unit() * ax, rng.signed_unit() * ay);
        if on {
            d
        } else {
            v2(0.0, 0.0)
        }
    };
    shape
        .iter()
        .map(|c| {
            let n = c.segs.len();
            if n == 0 {
                return c.clone();
            }
            // Anchor k = the start of segment k (a ring's last end is its start).
            let anchors = if c.closed { n } else { n + 1 };
            let mut da = Vec::with_capacity(anchors);
            let mut din = Vec::with_capacity(anchors);
            let mut dout = Vec::with_capacity(anchors);
            for _ in 0..anchors {
                da.push(draw(do_anchor));
                din.push(draw(do_in));
                dout.push(draw(do_out));
            }
            let anchor_at = |k: usize| {
                if k < n {
                    c.segs[k].start()
                } else {
                    c.segs[n - 1].end()
                }
            };
            let segs = (0..n)
                .map(|i| {
                    let s = c.segs[i];
                    let [p0, p1, p2, p3] = s.c.0;
                    let (h1, h2) = if s.line { (p0, p3) } else { (p1, p2) };
                    let j = if c.closed { (i + 1) % n } else { i + 1 };
                    let a0 = anchor_at(i) + da[i];
                    let a1 = anchor_at(j) + da[j];
                    Seg::curve(a0, h1 + dout[i], h2 + din[j], a1)
                })
                .collect::<Vec<_>>();
            Contour {
                start: segs[0].start(),
                segs,
                closed: c.closed,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Convert to Shape
// ---------------------------------------------------------------------------

fn rect_contour(x0: f64, y0: f64, x1: f64, y1: f64) -> Contour {
    let (a, b, c, d) = (v2(x0, y0), v2(x1, y0), v2(x1, y1), v2(x0, y1));
    Contour {
        start: a,
        segs: vec![
            Seg::line(a, b),
            Seg::line(b, c),
            Seg::line(c, d),
            Seg::line(d, a),
        ],
        closed: true,
    }
}

fn ellipse_contour(cx: f64, cy: f64, rx: f64, ry: f64) -> Contour {
    let (kx, ky) = (rx * KAPPA, ry * KAPPA);
    let top = v2(cx, cy - ry);
    let right = v2(cx + rx, cy);
    let bottom = v2(cx, cy + ry);
    let left = v2(cx - rx, cy);
    Contour {
        start: top,
        segs: vec![
            Seg::curve(top, top + v2(kx, 0.0), right - v2(0.0, ky), right),
            Seg::curve(right, right + v2(0.0, ky), bottom + v2(kx, 0.0), bottom),
            Seg::curve(bottom, bottom - v2(kx, 0.0), left + v2(0.0, ky), left),
            Seg::curve(left, left - v2(0.0, ky), top - v2(kx, 0.0), top),
        ],
        closed: true,
    }
}

fn rounded_rect_contour(x0: f64, y0: f64, x1: f64, y1: f64, radius: f64) -> Contour {
    let r = radius.min((x1 - x0) / 2.0).min((y1 - y0) / 2.0).max(0.0);
    if r <= 0.0 {
        return rect_contour(x0, y0, x1, y1);
    }
    let k = r * KAPPA;
    let p = [
        v2(x0 + r, y0),
        v2(x1 - r, y0),
        v2(x1, y0 + r),
        v2(x1, y1 - r),
        v2(x1 - r, y1),
        v2(x0 + r, y1),
        v2(x0, y1 - r),
        v2(x0, y0 + r),
    ];
    // Each corner's arc, starting at p[2i + 1] and ending at p[2i + 2].
    let arcs = [
        Seg::curve(p[1], p[1] + v2(k, 0.0), p[2] - v2(0.0, k), p[2]),
        Seg::curve(p[3], p[3] + v2(0.0, k), p[4] + v2(k, 0.0), p[4]),
        Seg::curve(p[5], p[5] - v2(k, 0.0), p[6] + v2(0.0, k), p[6]),
        Seg::curve(p[7], p[7] - v2(0.0, k), p[0] - v2(k, 0.0), p[0]),
    ];
    let mut segs = Vec::with_capacity(8);
    for i in 0..4 {
        let (a, b) = (p[2 * i], p[2 * i + 1]);
        if a.dist(b) > 1e-9 {
            segs.push(Seg::line(a, b));
        }
        segs.push(arcs[i]);
    }
    Contour {
        start: p[0],
        segs,
        closed: true,
    }
}

/// Convert to Shape over the art's bounds (Illustrator 30.8): Absolute draws
/// `width` × `height` about the bounds' centre; Relative grows the bounds by
/// `extraWidth` on the left and right and `extraHeight` on the top and bottom.
/// A relative ellipse goes through the corners of that box (its axes are √2 ×
/// the box's); an absolute one is `width` × `height`. The rounded rectangle's
/// corner radius is clamped to half the shorter side (circular arcs).
pub fn convert_to_shape(shape: &Shape, e: &ConvertToShapeEffect) -> Shape {
    let Some(b) = bounds_of(std::iter::once(shape)) else {
        return shape.clone();
    };
    let c = b.center();
    let absolute = e.size_mode.as_deref() == Some("absolute");
    let (hw, hh) = if absolute {
        (
            opt(e.width, b.width()).abs() / 2.0,
            opt(e.height, b.height()).abs() / 2.0,
        )
    } else {
        (
            (b.width() / 2.0 + opt(e.extra_width, 0.0)).max(0.0),
            (b.height() / 2.0 + opt(e.extra_height, 0.0)).max(0.0),
        )
    };
    let contour = match e.shape.as_str() {
        "ellipse" => {
            let f = if absolute {
                1.0
            } else {
                std::f64::consts::SQRT_2
            };
            ellipse_contour(c.x, c.y, hw * f, hh * f)
        }
        "rounded-rectangle" => rounded_rect_contour(
            c.x - hw,
            c.y - hh,
            c.x + hw,
            c.y + hh,
            opt(e.corner_radius, 0.0),
        ),
        _ => rect_contour(c.x - hw, c.y - hh, c.x + hw, c.y + hh),
    };
    vec![contour]
}

// ---------------------------------------------------------------------------
// Scribble
// ---------------------------------------------------------------------------

/// Most hatch lines one scribble draws (a runaway guard: 1 px spacing over a
/// 10 000 px shape).
const MAX_SCRIBBLE_LINES: usize = 20_000;

/// A contour flattened to a closed polygon (an open contour closes, as a fill does).
fn flatten(c: &Contour) -> Vec<V2> {
    let mut out = vec![c.start];
    for s in &c.segs {
        if s.line {
            out.push(s.end());
            continue;
        }
        let n = ((s.len() / 2.0).ceil() as usize).clamp(4, 128);
        for i in 1..=n {
            out.push(s.eval(i as f64 / n as f64));
        }
    }
    out
}

/// The inside intervals of the line `p · n = c`, as `[s0, s1]` along `d`.
fn line_intervals(polys: &[Vec<V2>], n: V2, d: V2, c: f64, even_odd: bool) -> Vec<(f64, f64)> {
    let mut hits: Vec<(f64, i32)> = Vec::new();
    for poly in polys {
        let m = poly.len();
        if m < 2 {
            continue;
        }
        for i in 0..m {
            let p0 = poly[i];
            let p1 = poly[(i + 1) % m];
            let a0 = p0.dot(n) - c;
            let a1 = p1.dot(n) - c;
            if (a0 < 0.0) == (a1 < 0.0) {
                continue;
            }
            let t = a0 / (a0 - a1);
            let p = p0.lerp(p1, t);
            hits.push((p.dot(d), if a1 > a0 { 1 } else { -1 }));
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::new();
    let mut w = 0i32;
    let mut start = 0.0;
    for (s, dir) in hits {
        let was = if even_odd { w & 1 != 0 } else { w != 0 };
        w += dir;
        let now = if even_odd { w & 1 != 0 } else { w != 0 };
        if !was && now {
            start = s;
        } else if was && !now && s - start > 1e-6 {
            out.push((start, s));
        }
    }
    out
}

/// The scribble's centre lines: open contours zig-zagging across the region.
///
/// Illustrator 30.8, measured with every variation at 0 (it is then
/// deterministic): hatch lines run along `d` = (cos angle, −sin angle) on
/// screen, `spacing` apart, the first `strokeWidth / 4 + 0.03` px inside the
/// region's extent across them (so they start at the top for 0°, the left for
/// 90°). The path runs along each line in the direction of `d`, then straight
/// back to the start of the next (the zig-zag), each end pushed `pathOverlap`
/// past the edge. Curviness gives every anchor handles across the lines, ±h
/// along the lines' normal, h = curviness × half the length of the crossing
/// the anchor ends (a short crossing near a star's tip turns tightly). A line that crosses the region more than once starts a new path for
/// each crossing no earlier path can continue (the other leg of a star).
/// Variations draw uniformly in ±variation per line (spacing), per end
/// (overlap) and per anchor (curviness).
pub fn scribble_centrelines(
    shape: &Shape,
    e: &ScribbleEffect,
    even_odd: bool,
    rng: &mut SplitMix64,
) -> Vec<Contour> {
    let polys: Vec<Vec<V2>> = shape.iter().map(flatten).collect();
    let th = fin(e.angle, 0.0).to_radians();
    let d = v2(th.cos(), -th.sin());
    let n = v2(-d.y, d.x);
    let (mut lo_n, mut hi_n) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in polys.iter().flatten() {
        lo_n = lo_n.min(p.dot(n));
        hi_n = hi_n.max(p.dot(n));
    }
    if !(lo_n <= hi_n) {
        return Vec::new();
    }
    let sw = fin(e.stroke_width, 0.0).max(0.0);
    let spacing = fin(e.spacing, 0.0).max(0.1);
    let spacing_var = fin(e.spacing_variation, 0.0).abs();
    let overlap = fin(e.path_overlap, 0.0);
    let overlap_var = fin(e.path_overlap_variation, 0.0).abs();
    let curv = fin(e.curviness, 0.0).max(0.0) / 100.0;
    let curv_var = fin(e.curviness_variation, 0.0).abs() / 100.0;
    struct Run {
        pts: Vec<V2>,
        /// Per anchor: the length of the crossing it ends (its turn's handles scale with it).
        lens: Vec<f64>,
        last: (f64, f64),
        open: bool,
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut c = lo_n + sw / 4.0 + 0.03;
    let mut lines = 0usize;
    while c <= hi_n && lines < MAX_SCRIBBLE_LINES {
        let intervals = line_intervals(&polys, n, d, c, even_odd);
        let mut continued = vec![false; runs.len()];
        for (s0, s1) in intervals {
            let a = s0 - (overlap + rng.signed_unit() * overlap_var);
            let b = s1 + (overlap + rng.signed_unit() * overlap_var);
            let base = n * c;
            let (pa, pb) = (base + d * a, base + d * b);
            let target = runs
                .iter()
                .enumerate()
                .position(|(i, r)| r.open && !continued[i] && r.last.0.max(s0) < r.last.1.min(s1));
            match target {
                Some(i) => {
                    continued[i] = true;
                    runs[i].pts.push(pa);
                    runs[i].pts.push(pb);
                    runs[i].lens.extend([s1 - s0, s1 - s0]);
                    runs[i].last = (s0, s1);
                }
                None => {
                    runs.push(Run {
                        pts: vec![pa, pb],
                        lens: vec![s1 - s0, s1 - s0],
                        last: (s0, s1),
                        open: true,
                    });
                    continued.push(true);
                }
            }
        }
        // A run no crossing of this line continued is finished.
        for (r, cont) in runs.iter_mut().zip(&continued) {
            if !cont {
                r.open = false;
            }
        }
        c += (spacing + rng.signed_unit() * spacing_var).max(0.1);
        lines += 1;
    }
    runs.into_iter()
        .filter(|r| r.pts.len() >= 2)
        .map(|r| {
            let hs: Vec<f64> = r
                .lens
                .iter()
                .map(|len| (curv + rng.signed_unit() * curv_var).max(0.0) * len / 2.0)
                .collect();
            let segs = r
                .pts
                .windows(2)
                .enumerate()
                .map(|(i, w)| {
                    let (h0, h1) = (hs[i], hs[i + 1]);
                    if h0 <= 0.0 && h1 <= 0.0 {
                        Seg::line(w[0], w[1])
                    } else {
                        Seg::curve(w[0], w[0] + n * h0, w[1] - n * h1, w[1])
                    }
                })
                .collect();
            Contour {
                start: r.pts[0],
                segs,
                closed: false,
            }
        })
        .collect()
}

/// A set of open centre lines stroked `width` px wide with round caps and
/// joins (Illustrator's scribble stroke), as one closed nonzero outline.
pub fn stroke_centrelines(lines: &[Contour], width: f64) -> Shape {
    if lines.is_empty() || !(width > 0.0) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for line in lines {
        let pts: Vec<Value> = super::points_from_contours(std::slice::from_ref(line))
            .iter()
            .map(BlendPathPoint::to_json)
            .collect();
        let node = RawNode::new(
            json!({
                "id": "scribble",
                "type": "path",
                "points": pts,
                "closed": false,
                "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000000"}, "width": width, "cap": "round", "join": "round"}],
            })
            .as_object()
            .cloned()
            .unwrap_or_default(),
        );
        let opts = OutlineOptions {
            tolerance: 0.1,
            ..OutlineOptions::default()
        };
        if let Ok(Outline::Shape { points, .. }) = stroke_outline(&node, None, &opts) {
            out.extend(super::contours_from_points(&points, true));
        }
    }
    out
}

/// Scribble one region (a copy's fill, or its stroke's band) into the outline
/// of the scribble stroke.
pub fn scribble(shape: &Shape, e: &ScribbleEffect, even_odd: bool, rng: &mut SplitMix64) -> Shape {
    let lines = scribble_centrelines(shape, e, even_odd, rng);
    stroke_centrelines(&lines, fin(e.stroke_width, 0.0).max(0.0))
}

/// The salt a Scribble / Tweak stream mixes into its seed.
pub const SCRIBBLE_SALT: u64 = 0x6D6F_6461_5363_7262; // "modaScrb"
pub const TWEAK_SALT: u64 = 0x6D6F_6461_5477_6B31; // "modaTwk1"

/// The effect's own random stream.
pub fn scribble_rng(e: &ScribbleEffect) -> SplitMix64 {
    seeded(e.seed, SCRIBBLE_SALT)
}

pub fn tweak_rng(e: &TweakEffect) -> SplitMix64 {
    seeded(e.seed, TWEAK_SALT)
}
