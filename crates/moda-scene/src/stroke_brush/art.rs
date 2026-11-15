//! Art brush (brushes.md §2.1, §4.4, §4.6; plan P17): the brush artwork
//! warped along the stroke — artwork x along the arc length, artwork y across
//! the normal.
//!
//! # Mapping
//!
//! 1. **Direction** turns the artwork so its "along" axis points down the
//!    path: `right` keeps it (art left edge at the path start, art top on the
//!    path's left), `left` turns it 180°, `up` runs it bottom → top, `down`
//!    top → bottom (all rotations, so the art is never mirrored by direction).
//! 2. The turned artwork's geometric bounds give its length `La`; it is
//!    centred across the path.
//! 3. **Fit** maps art distance `u ∈ [0, La]` to path distance `s ∈ [0, Lp]`
//!    and fixes the cross scale `k` (path px per art unit across):
//!    - `stretch`: `s = u·Lp/La`; `k = scale × strokeWidth` — Illustrator's
//!      rule (lead decision, P17 review): the artwork's own height is its
//!      thickness at weight 1 and scale 1, and the stroke weight multiplies
//!      it (as the calligraphic nib's `size × strokeWidth`).
//!    - `proportional`: `s = u·Lp/La`; `k = (Lp/La) × scale × strokeWidth` —
//!      the art scales uniformly to the path's length and keeps its aspect
//!      ratio, so longer paths draw thicker art (Illustrator's "Scale
//!      Proportionately"), and the stroke weight multiplies the thickness as
//!      in the other modes (Illustrator 30.8: Floral Brush 11 on a 300 px line
//!      is 37 px thick at weight 1 and 74 px at weight 2).
//!    - `guides`: the art before `guides.start` and after `guides.end` keeps
//!      its proportions at the stretch mode's `k` for that end; the middle
//!      stretches to fill. Guides are fractions of the artwork's extent along
//!      the direction's axis in the UNTURNED artwork, from its left (right /
//!      left) or top (up / down) edge, so changing the direction never
//!      changes which part is protected. `end` must exceed `start`; otherwise
//!      the guides are ignored (a `guides_ignored` warning) and `guides` fits
//!      like `stretch`. When the two ends are longer than the path they
//!      shrink together and the middle vanishes.
//! 4. **Flip along** mirrors the placed art end to end (`s → Lp − s`, so
//!    guide-protected ends travel with the art); **flip across** mirrors it
//!    over the path.
//! 5. `scale` is a `Dyn`: random draws once per stroke from the seed; pen
//!    channels vary the thickness along the path (the along mapping is not
//!    affected). With `overrides.widthFrom = 'profile'` the stroke's width
//!    profile multiplies the thickness per side (art above its centre line
//!    follows the profile's left width, below it the right).
//!
//! # Corners and overlap
//!
//! The centerline is made G1 before warping by filleting each corner
//! (`geom::fillet_corners`):
//! - `overlap: 'none'` uses a tiny radius (2 % of the art's half-thickness,
//!   0.05–1 px), so at a sharp corner the outer side of the art fans around
//!   the corner and the inner side folds over itself — the self-overlap
//!   Illustrator shows.
//! - `overlap: 'adjust'` fillets with the art's largest half-thickness `a`
//!   (half its visual height, strokes included, times the cross scale), but
//!   never trims more than `a` from either side of a corner: right-angle and
//!   obtuse corners get the full radius (no inner fold), acute ones a
//!   tighter fillet so the art still reaches into the corner. The inner side
//!   of the art is then compressed wherever the path's radius of curvature
//!   is smaller than the art's half-thickness (a soft knee at 0.85 R
//!   saturating below R) — which is what keeps acute corners from folding.
//!   An approximation of Illustrator's "Adjust corners and folds to prevent
//!   overlap".
//!
//! Work is bounded (`MAX_WORK_PIECES`): an input that would need more (e.g.
//! a tiny tolerance on a huge path) fails with `too_complex`.
//!
//! Compound stroke paths get the whole artwork on each subpath. Closed paths
//! start and end the art at the first anchor. Output is cubic-only, in the
//! stroke node's local coordinates, each placed shape keeping its own fill
//! and stroke (colourised per `colorize`, stroke widths scaled by the cross
//! scale at mid-path). Gradient geometry in artwork paints is not warped.

use crate::value::{Map, Value};

use super::artwork::{collect_artwork, default_geometry, Artwork, GeometryFn};
use super::clip::{clip_x, clip_x_open};
use super::colorize::{self, parse_colorize, Colorize};
use super::common::{
    error, node_stroke, parse_request, placement_json, subpath_length, with_json_cap, Request,
    MAX_WORK_PIECES,
};
pub use super::common::{BrushStroke, PlacedAt, Placement};
use super::dyn_value::{draw, ink_at, largest, parse_dyn, Dyn};
use super::geom::{curve_bounds, fillet_corners_mapped, Bounds, Subpath};
use super::spine::Spine;
use super::warp::{crossings, WarpBudget, Warper};
use crate::path_geom::vec2::V2;
pub use crate::stroke_outline::ArcWindow;
use crate::transform;
use crate::types::Affine;

/// Default fitting tolerance, px.
pub const DEFAULT_TOLERANCE: f64 = 0.25;
const SALT_SCALE: u64 = 0xA27;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtFit {
    Stretch,
    Proportional,
    Guides,
}

impl ArtFit {
    fn parse(v: Option<&Value>) -> Option<ArtFit> {
        Some(match v.and_then(Value::as_str)? {
            "stretch" => ArtFit::Stretch,
            "proportional" => ArtFit::Proportional,
            "guides" => ArtFit::Guides,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtDirection {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overlap {
    Adjust,
    None,
}

/// `ArtBrushDefinition` (nodes.ts), normalised like `normalizeBrushDefinition`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtBrush {
    pub id: String,
    pub name: String,
    /// `artwork.nodes`, entries without a string `id` and `type` dropped.
    pub artwork: Vec<Value>,
    pub scale: Dyn,
    pub fit: ArtFit,
    /// `(start, end)`, `0 ≤ start < end ≤ 1`, in the unturned artwork.
    pub guides: Option<(f64, f64)>,
    /// `guides` were given but `end ≤ start`: ignored, with a warning.
    pub guides_ignored: bool,
    pub direction: ArtDirection,
    pub flip_along: bool,
    pub flip_across: bool,
    pub overlap: Overlap,
    pub colorize: Colorize,
    /// `scaleWithWeight`: Illustrator's sizing — the stroke weight multiplies
    /// the brush's designed size. Without it (the default, as in Figma) the
    /// stroke width is the width of the mark: the design is scaled to it.
    pub scale_with_weight: bool,
}

/// `artwork.nodes` with the entries `normalizeBrushDefinition` keeps (a
/// string `id` and `type`); `None` without a `nodes` array.
pub fn artwork_nodes(raw: Option<&Value>) -> Option<Vec<Value>> {
    let nodes = raw?.as_object()?.get("nodes")?.as_array()?;
    Some(
        nodes
            .iter()
            .filter(|n| {
                n.get("id").is_some_and(Value::is_string)
                    && n.get("type").is_some_and(Value::is_string)
            })
            .cloned()
            .collect(),
    )
}

/// Parse an art brush definition; `None` when it is not a repairable art
/// brush (no id, other type, no `artwork.nodes` array).
pub fn parse_art_brush(raw: &Value) -> Option<ArtBrush> {
    let o = raw.as_object()?;
    let id = o
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    if o.get("type").and_then(Value::as_str) != Some("art") {
        return None;
    }
    let artwork = artwork_nodes(o.get("artwork"))?;
    let finite = |v: Option<&Value>| v.and_then(Value::as_f64).filter(|x| x.is_finite());
    let raw_guides = o.get("guides").and_then(Value::as_object).and_then(|g| {
        let start = finite(g.get("start"))?.clamp(0.0, 1.0);
        let end = finite(g.get("end"))?.clamp(0.0, 1.0);
        Some((start, end))
    });
    let guides = raw_guides.filter(|(a, b)| b > a);
    let guides_ignored = raw_guides.is_some() && guides.is_none();
    Some(ArtBrush {
        id: id.to_string(),
        name: o
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .to_string(),
        artwork,
        scale: parse_dyn(o.get("scale"), 1.0, 0.0, f64::INFINITY),
        fit: ArtFit::parse(o.get("fit")).unwrap_or(ArtFit::Stretch),
        guides,
        guides_ignored,
        direction: match o.get("direction").and_then(Value::as_str) {
            Some("left") => ArtDirection::Left,
            Some("up") => ArtDirection::Up,
            Some("down") => ArtDirection::Down,
            _ => ArtDirection::Right,
        },
        flip_along: o.get("flipAlong").and_then(Value::as_bool) == Some(true),
        flip_across: o.get("flipAcross").and_then(Value::as_bool) == Some(true),
        overlap: match o.get("overlap").and_then(Value::as_str) {
            Some("adjust") => Overlap::Adjust,
            _ => Overlap::None,
        },
        colorize: parse_colorize(o.get("colorize")),
        scale_with_weight: super::common::scale_with_weight(o),
    })
}

/// `BrushApplication.overrides` as they apply to an art brush.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArtOverrides {
    pub scale: Option<Dyn>,
    pub flip_along: Option<bool>,
    pub flip_across: Option<bool>,
    pub colorize: Option<Colorize>,
    /// Pattern-only fits are ignored.
    pub fit: Option<ArtFit>,
    pub width_from_profile: bool,
    pub overlap: Option<Overlap>,
}

pub fn parse_art_overrides(raw: Option<&Map>) -> ArtOverrides {
    let Some(o) = raw else {
        return ArtOverrides::default();
    };
    ArtOverrides {
        scale: o
            .get("scale")
            .map(|v| parse_dyn(Some(v), 1.0, 0.0, f64::INFINITY)),
        flip_along: o.get("flipAlong").and_then(Value::as_bool),
        flip_across: o.get("flipAcross").and_then(Value::as_bool),
        colorize: o.get("colorize").map(|v| parse_colorize(Some(v))),
        fit: ArtFit::parse(o.get("fit")),
        width_from_profile: o.get("widthFrom").and_then(Value::as_str) == Some("profile"),
        overlap: match o.get("overlap").and_then(Value::as_str) {
            Some("adjust") => Some(Overlap::Adjust),
            Some("none") => Some(Overlap::None),
            _ => None,
        },
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ArtOptions<'a> {
    /// Max distance between the output cubics and the exact warp, px.
    pub tolerance: f64,
    /// Trim / dash: draw only these arc-length windows (`ArcWindow`s in the
    /// outliner's measure, as the engine resolves trim and dash). The art is
    /// laid out along the WHOLE path and then clipped (a reveal, not a
    /// re-stretch). `None` = the whole stroke; `Some(&[])` = nothing.
    pub windows: Option<&'a [ArcWindow]>,
}

impl Default for ArtOptions<'_> {
    fn default() -> Self {
        ArtOptions {
            tolerance: DEFAULT_TOLERANCE,
            windows: None,
        }
    }
}

/// A window's `[a, b]` pieces on a subpath of length `len`, the way
/// `stroke_outline` normalises them: an open subpath clamps to `[0, len]`; a
/// closed one wraps `start` into `[0, len)`, keeps at most one lap, and splits
/// a run through the seam in two. A window covering a closed ring from its
/// start (the outliner's `is_ring`) stays one whole piece.
pub(crate) fn window_pieces(w: &ArcWindow, len: f64, closed: bool) -> Vec<(f64, f64)> {
    const EPS: f64 = 1e-9;
    if !(w.start.is_finite() && w.end.is_finite() && len.is_finite() && len > 0.0) {
        return Vec::new();
    }
    if closed {
        if w.start <= EPS && w.end >= len - 1e-7 {
            return vec![(0.0, len)];
        }
        let width = (w.end - w.start).min(len);
        if width <= EPS {
            return Vec::new();
        }
        let a = w.start.rem_euclid(len);
        let b = a + width;
        if b <= len + EPS {
            vec![(a, b.min(len))]
        } else {
            vec![(a, len), (0.0, b - len)]
        }
    } else {
        let (a, b) = (w.start.max(0.0), w.end.min(len));
        if b - a > EPS {
            vec![(a, b)]
        } else {
            Vec::new()
        }
    }
}

/// A paint with only its fills (strokes removed) or only its strokes (fills
/// removed): a clipped shape fills its closed cut polygon but strokes only the
/// original outline.
pub(crate) fn paint_part(paint: &Map, stroke_only: bool) -> Map {
    let mut p = paint.clone();
    p.remove(if stroke_only { "fills" } else { "strokes" });
    p
}

fn direction_affine(d: ArtDirection) -> Affine {
    let (a, b, c, dd) = match d {
        ArtDirection::Right => (1.0, 0.0, 0.0, 1.0),
        ArtDirection::Left => (-1.0, 0.0, 0.0, -1.0),
        // u = -y, v = x
        ArtDirection::Up => (0.0, 1.0, -1.0, 0.0),
        // u = y, v = -x
        ArtDirection::Down => (0.0, -1.0, 1.0, 0.0),
    };
    Affine {
        a,
        b,
        c,
        d: dd,
        e: 0.0,
        f: 0.0,
    }
}

/// Piecewise-linear art distance → path distance.
#[derive(Debug, Clone)]
struct AlongMap {
    /// Breakpoints (u, s), increasing in u.
    knots: Vec<(f64, f64)>,
    flip: bool,
    lp: f64,
}

impl AlongMap {
    fn s_of(&self, u: f64) -> f64 {
        let k = &self.knots;
        let i = k.partition_point(|q| q.0 <= u).clamp(1, k.len() - 1);
        let (a, b) = (k[i - 1], k[i]);
        let slope = if b.0 > a.0 {
            (b.1 - a.1) / (b.0 - a.0)
        } else {
            0.0
        };
        let s = a.1 + (u - a.0) * slope;
        if self.flip {
            self.lp - s
        } else {
            s
        }
    }

    /// The art distance at path distance `s`, by bisection (the map is
    /// monotone): collapsed pieces resolve to one of their ends.
    fn u_at(&self, s: f64, la: f64) -> f64 {
        let increasing = !self.flip;
        let (mut lo, mut hi) = (0.0, la);
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            if (self.s_of(mid) < s) == increasing {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    }

    /// Art distances where the path is at distance `s` (none on a collapsed piece).
    fn u_of(&self, s: f64) -> Option<f64> {
        let s = if self.flip { self.lp - s } else { s };
        self.knots.windows(2).find_map(|w| {
            let (a, b) = (w[0], w[1]);
            (b.1 > a.1 + 1e-12 && s >= a.1 && s <= b.1)
                .then(|| a.0 + (s - a.1) / (b.1 - a.1) * (b.0 - a.0))
        })
    }
}

fn compress_inner(a: f64, curvature: f64) -> f64 {
    if curvature * a <= 0.0 {
        return a;
    }
    let r = 1.0 / curvature.abs();
    let (knee, cap) = (0.85 * r, 0.98 * r);
    let m = a.abs();
    if m <= knee {
        return a;
    }
    let c = knee + (cap - knee) * ((m - knee) / (cap - knee)).tanh();
    c.copysign(a)
}

/// Place an art brush along a stroke. Pure and total: never panics, bounded
/// work on any input.
pub fn art_brush_geometry(
    stroke: &BrushStroke,
    brush: &ArtBrush,
    overrides: &ArtOverrides,
    artwork: &Artwork,
    opts: &ArtOptions,
) -> Placement {
    let mut out = Placement {
        skipped: artwork.skipped.clone(),
        bounds: Bounds::empty(),
        ..Placement::default()
    };
    let tol = if opts.tolerance.is_finite() {
        opts.tolerance.clamp(0.01, 100.0)
    } else {
        DEFAULT_TOLERANCE
    };
    let scale = overrides.scale.unwrap_or(brush.scale);
    let fit = overrides.fit.unwrap_or(brush.fit);
    let overlap = overrides.overlap.unwrap_or(brush.overlap);
    let flip_along = overrides.flip_along.unwrap_or(brush.flip_along);
    let flip_across = overrides.flip_across.unwrap_or(brush.flip_across);
    let colorize = overrides.colorize.as_ref().unwrap_or(&brush.colorize);
    let profile = if overrides.width_from_profile {
        stroke.profile
    } else {
        None
    };
    let mut stroke_width = if stroke.stroke_width.is_finite() {
        stroke.stroke_width.max(0.0)
    } else {
        0.0
    };

    // Artwork in the art frame: u along [0, La], v across centred on 0.
    let turned = artwork.transformed(direction_affine(brush.direction));
    let b = turned.bounds();
    if b.is_empty() {
        return out;
    }
    let la = b.width();
    if !la.is_finite() || la <= 1e-9 {
        return out;
    }
    let cy = (b.min.y + b.max.y) / 2.0;
    let art = turned.transformed(transform::translate(-b.min.x, -cy));
    // Largest distance of the art (strokes included) from its centre line.
    let vb = turned.visual_bounds();
    let reach = (vb.max.y - cy).abs().max((vb.min.y - cy).abs());
    // Sizing: the artwork's height across the path (its painted height for a
    // hairline) is the stroke width, unless the weight multiplies it.
    let designed = if b.height() > 1e-9 { b.height() } else { vb.height() };
    stroke_width *= super::common::weight_unit(brush.scale_with_weight, designed);
    if fit == ArtFit::Guides && brush.guides_ignored {
        out.warnings
            .push("guides_ignored: guides.end must be greater than guides.start".into());
    }
    let random = draw(stroke.seed, SALT_SCALE, 0);
    let ctx = colorize::resolve(colorize, stroke.stroke_color, &artwork.weighted_colors());
    let total_len: f64 = stroke.subpaths.iter().map(subpath_length).sum();
    if !total_len.is_finite() || total_len <= 1e-9 {
        return out;
    }
    let max_side = match profile {
        Some(p) => (0..=64)
            .map(|i| {
                let (l, r) = p.width_at(i as f64 / 64.0, None);
                l.max(r)
            })
            .fold(0.0, f64::max),
        None => 1.0,
    };
    let mut budget = WarpBudget::new(MAX_WORK_PIECES);
    let mut cum = 0.0;
    for (si, sp) in stroke.subpaths.iter().enumerate() {
        let orig_len = subpath_length(sp);
        let t_start = cum / total_len;
        cum += orig_len;
        let t_span = orig_len / total_len;
        if !orig_len.is_finite() || orig_len <= 1e-9 {
            continue;
        }
        let base_k = |lp: f64| match fit {
            ArtFit::Proportional => lp / la * stroke_width,
            _ => stroke_width,
        };
        // Largest half-thickness → fillet radius (and its trim cap).
        let half = reach * base_k(orig_len) * largest(&scale) * max_side;
        let (radius, max_trim) = match overlap {
            Overlap::Adjust => {
                let r = (half * 1.02).max(tol * 0.5);
                (r, half.max(tol * 0.5))
            }
            Overlap::None => {
                let r = (half * 0.02).clamp(0.05, 1.0);
                (r, f64::INFINITY)
            }
        };
        let (adjusted, arc_map) = fillet_corners_mapped(sp, radius, max_trim);
        let Some(spine) = Spine::new(&adjusted, (tol * 0.1).min(0.05)) else {
            continue;
        };
        if spine.truncated {
            out.too_complex = Some("the stroke path needs more than 200 000 samples".into());
            return out;
        }
        let lp = spine.length;
        let k0 = base_k(lp);
        let t_of = |s: f64| t_start + (s / lp).clamp(0.0, 1.0) * t_span;
        let k_at = |s: f64| {
            let ink = if scale.uses_ink() {
                ink_at(stroke.ink, t_of(s))
            } else {
                Default::default()
            };
            k0 * scale.eval(&ink, random).max(0.0)
        };

        // Along map.
        let mut knots = vec![(0.0, 0.0), (la, lp)];
        let mut cut_us: Vec<f64> = Vec::new();
        if fit == ArtFit::Guides {
            if let Some((g0, g1)) = brush.guides {
                // Guides are measured in the unturned artwork from its left /
                // top edge; `left` and `up` run the art from the other end.
                let (g0, g1) = match brush.direction {
                    ArtDirection::Left | ArtDirection::Up => (1.0 - g1, 1.0 - g0),
                    _ => (g0, g1),
                };
                let (u0, u1) = (g0 * la, g1 * la);
                // The protected ends keep the art's proportions at that end's
                // stretch-mode cross scale (flip along swaps which end is which).
                let (k_first, k_last) = if flip_along {
                    (k_at(lp), k_at(0.0))
                } else {
                    (k_at(0.0), k_at(lp))
                };
                let (mut ls, mut le) = (u0 * k_first, (la - u1) * k_last);
                if ls + le > lp {
                    let f = lp / (ls + le);
                    ls *= f;
                    le *= f;
                }
                knots = vec![(0.0, 0.0), (u0, ls), (u1, lp - le), (la, lp)];
                knots.dedup_by(|b, a| b.0 <= a.0 + 1e-12 && (b.1 - a.1).abs() < 1e-12);
                cut_us.extend([u0, u1]);
            }
        }
        let along = AlongMap {
            knots,
            flip: flip_along,
            lp,
        };
        cut_us.extend(spine.kinks.iter().filter_map(|s| along.u_of(*s)));
        cut_us.retain(|u| *u > 1e-9 && *u < la - 1e-9);
        cut_us.sort_by(|a, b| a.total_cmp(b));
        cut_us.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        // Trim / dash windows → art-space ranges to keep (clip before warping).
        let measured = stroke
            .measured
            .get(si)
            .copied()
            .filter(|m| m.is_finite() && *m > 1e-9)
            .unwrap_or(orig_len);
        // (id suffix, run): pieces named by input window index, `.0` / `.1`
        // for the two halves of a run through a closed seam.
        let runs: Vec<(String, (f64, f64))> = match opts.windows {
            None => vec![(String::new(), (0.0, measured))],
            Some(ws) => ws
                .iter()
                .enumerate()
                .filter(|(_, w)| w.subpath == si)
                .flat_map(|(wi, w)| {
                    let pieces = window_pieces(w, measured, sp.closed);
                    let split = pieces.len() > 1;
                    pieces.into_iter().enumerate().map(move |(pi, r)| {
                        let id = if split {
                            format!(".w{wi}.{pi}")
                        } else {
                            format!(".w{wi}")
                        };
                        (id, r)
                    })
                })
                .collect(),
        };
        if runs.is_empty() {
            continue;
        }
        // Window ends → spine distance through the fillets' arc-length map
        // (exact on the kept pieces), then → art distance.
        let own_total: f64 = orig_len;
        let spine_scale = if arc_map.total_out() > 0.0 {
            lp / arc_map.total_out()
        } else {
            1.0
        };
        let to_spine = |m: f64| arc_map.to_out(m * own_total / measured) * spine_scale;
        // A run whose spine range wraps past the spine's end (the filleted
        // closed path starts a few px after the seam) splits there too.
        type Clip = (String, (f64, f64), Option<(f64, f64)>);
        let mut clips: Vec<Clip> = Vec::new();
        for (suffix, (a, b)) in &runs {
            if opts.windows.is_none() || (*a <= 1e-9 && *b >= measured - 1e-9) {
                clips.push((suffix.clone(), (*a, *b), None));
                continue;
            }
            let (sa, sb) = (to_spine(*a), to_spine(*b));
            let spans: Vec<(f64, f64)> = if sp.closed && sb < sa - 1e-9 {
                [(sa, lp), (0.0, sb)]
                    .into_iter()
                    .filter(|(x, y)| y - x > 1e-9)
                    .collect()
            } else {
                vec![(sa, sb.max(sa))]
            };
            let n = spans.len();
            for (k, (x, y)) in spans.into_iter().enumerate() {
                let (ua, ub) = (along.u_at(x, la), along.u_at(y, la));
                let id = if n > 1 {
                    format!("{suffix}~{k}")
                } else {
                    suffix.clone()
                };
                clips.push((id, (*a, *b), Some((ua.min(ub), ua.max(ub)))));
            }
        }

        let adjust = overlap == Overlap::Adjust;
        let map = |p: V2| -> V2 {
            let s = along.s_of(p.x);
            let q = spine.eval(s);
            let v = if flip_across { -p.y } else { p.y };
            let mut a = v * k_at(s);
            if let Some(prof) = profile {
                let (l, r) = prof.width_at(t_of(s), None);
                a *= if v > 0.0 { r } else { l };
            }
            if adjust {
                a = compress_inner(a, q.curvature);
            }
            q.pos + q.right() * a
        };
        let warper = Warper {
            map: &map,
            tolerance: tol,
            step: (la.max(reach) * 1e-5).max(1e-9),
        };
        let k_mid = k_at(lp / 2.0)
            * profile.map_or(1.0, |p| {
                let (l, r) = p.width_at(t_of(lp / 2.0), None);
                (l + r) / 2.0
            });
        'windows: for (suffix, run, clip) in &clips {
            for shape in &art.shapes {
                let at = PlacedAt {
                    subpath: si,
                    s: (run.0 + run.1) / 2.0,
                    s_span: *run,
                };
                let id = format!("{}#{}{}", shape.source_id, si, suffix);
                let warp_all = |subs: &[Subpath], budget: &mut WarpBudget| -> Vec<Subpath> {
                    let mut out = Vec::new();
                    for asp in subs {
                        let mut segs = Vec::new();
                        for c in &asp.segs {
                            let cuts = crossings(c, |p| p.x, &cut_us, 256);
                            warper.warp(c, &cuts, &mut segs, budget);
                        }
                        if !segs.is_empty() {
                            out.push(Subpath {
                                segs,
                                closed: asp.closed,
                            });
                        }
                    }
                    out
                };
                let stroke_scale = shape.transform_scale * k_mid;
                let Some((u0, u1)) = clip else {
                    let subpaths = warp_all(&shape.subpaths, &mut budget);
                    out.emit(
                        id,
                        &shape.source_id,
                        &shape.paint,
                        subpaths,
                        stroke_scale,
                        ctx.as_ref(),
                        at,
                    );
                    continue;
                };
                // Shapes wholly outside the window cost nothing.
                let mut sb = Bounds::empty();
                for asp in &shape.subpaths {
                    sb.union(&curve_bounds(&asp.segs));
                }
                if sb.is_empty() || sb.max.x < *u0 || sb.min.x > *u1 {
                    continue;
                }
                let filled = clip_x(&clip_x(&shape.subpaths, *u0, false), *u1, true);
                let stroked = crate::paint::map_strokes_visibly(&shape.paint);

                let cut = sb.min.x < *u0 || sb.max.x > *u1;
                if stroked && cut {
                    // Fill the closed cut polygon; stroke only the original outline.
                    let runs = clip_x_open(&clip_x_open(&shape.subpaths, *u0, false), *u1, true);
                    let fill_paths = warp_all(&filled, &mut budget);
                    out.emit(
                        id.clone(),
                        &shape.source_id,
                        &paint_part(&shape.paint, false),
                        fill_paths,
                        stroke_scale,
                        ctx.as_ref(),
                        at,
                    );
                    let stroke_paths = warp_all(&runs, &mut budget);
                    out.emit(
                        format!("{id}.stroke"),
                        &shape.source_id,
                        &paint_part(&shape.paint, true),
                        stroke_paths,
                        stroke_scale,
                        ctx.as_ref(),
                        at,
                    );
                } else {
                    let subpaths = warp_all(&filled, &mut budget);
                    out.emit(
                        id,
                        &shape.source_id,
                        &shape.paint,
                        subpaths,
                        stroke_scale,
                        ctx.as_ref(),
                        at,
                    );
                }
                if budget.exhausted {
                    break 'windows;
                }
            }
        }
        if budget.exhausted {
            out.too_complex = Some(format!(
                "art brush placement needs more than {MAX_WORK_PIECES} warp steps (raise the tolerance or simplify the path / artwork)"
            ));
            return out;
        }
    }
    out
}

/// Place an art brush on a path node. `geometry` turns artwork leaf nodes
/// into paths ([`default_geometry`] when the caller has nothing richer).
pub fn art_brush_for_node(
    node: &Map,
    brush: &ArtBrush,
    geometry: GeometryFn,
    opts: &ArtOptions,
) -> Placement {
    let ns = node_stroke(node);
    let artwork = collect_artwork(&brush.artwork, geometry);
    let stroke = BrushStroke {
        subpaths: ns.subpaths,
        stroke_width: ns.stroke_width,
        stroke_color: ns.stroke_color.as_ref(),
        profile: ns.profile.as_ref(),
        ink: ns.ink.as_ref(),
        seed: ns.seed,
        measured: ns.measured,
    };
    let overrides = parse_art_overrides(ns.overrides.as_ref());
    art_brush_geometry(&stroke, brush, &overrides, &artwork, opts)
}

/// JSON in / JSON out (the WASM `art_brush` export); always an envelope,
/// never panics.
///
/// ```jsonc
/// // request
/// { "node": PathNode,            // points, closed, cornerRadius, strokeWidth, stroke,
///                                // strokeBrush (seed, overrides), strokeProfile, inkChannels
///   "brush": ArtBrushDefinition,
///   "tolerance"?: 0.25 }
/// // response — node-local coordinates, drawing order
/// { "status": "ok", "nodes": [{ "id", "type": "path", "sourceId", "x": 0, "y": 0,
///     "points": PathPoint[], "closed", ...paint }],
///   "bounds": { "minX", "minY", "maxX", "maxY" }, "skipped": string[] }
/// { "status": "empty" }
/// { "status": "error", "code": "invalid_json" | "invalid_request" | "invalid_brush"
///     | "request_too_large" | "too_many_points", "message" }
/// ```
pub fn art_brush_json(request_json: &str) -> String {
    art_brush_json_with(request_json, &default_geometry)
}

/// [`art_brush_json`] with a caller-supplied artwork geometry callback.
pub fn art_brush_json_with(request_json: &str, geometry: GeometryFn) -> String {
    match parse_request(request_json) {
        Ok(req) => art_request(req, geometry),
        Err(e) => e,
    }
}

pub(crate) fn art_request(req: Request, geometry: GeometryFn) -> String {
    let Some(brush) = parse_art_brush(&req.brush) else {
        return error(
            "invalid_brush",
            "brush must be an art brush definition with artwork.nodes",
        );
    };
    let opts = ArtOptions {
        tolerance: req.tolerance.unwrap_or(DEFAULT_TOLERANCE),
        windows: req.windows.as_deref(),
    };
    placement_json(&with_json_cap(|| {
        art_brush_for_node(&req.node, &brush, geometry, &opts)
    }))
}
