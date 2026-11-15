//! Pattern brush (brushes.md §2.1, §4.4, §4.6; plan P19): tiles laid along
//! the stroke — side tiles along each run between corners, corner tiles at
//! corners, start and end tiles at the ends of open paths.
//!
//! # Tiles and scale
//!
//! Every tile's artwork is normalised to its geometric bounds (along = x from
//! 0, across centred). One scale `k = scale × strokeWidth` applies to all
//! tiles — Illustrator's rule, as for art and scatter brushes: at weight 1
//! and scale 1 every tile is its native size; the stroke weight multiplies.
//!
//! A `scale` that follows a pen channel (pressure, stylus wheel, tilt, …)
//! sizes each tile from the ink at its position, as Illustrator does: the
//! tiles are laid end to end, each at the scale read at its centre (never
//! below 5 % of the brush's largest scale), and the fit then works on those
//! lengths — stretch scales the whole run to the leg, add-space spreads the
//! slack, approximate-path centres it. Start / end tiles read the scale at
//! the leg's ends and corners at the corner (their depth and the legs' cut
//! follow). A fixed or random scale is one value per stroke.
//!
//! With `overrides.widthFrom = 'profile'` (Illustrator's "Width Points /
//! Profile") the stroke's width profile multiplies each tile's thickness
//! point by point (the path's right side by the profile's right width, the
//! left by its left), and corner tiles by its mean at the corner; tile
//! lengths are unchanged.
//!
//! # Runs and corners
//!
//! The path splits into **legs** at anchors where it turns by more than 15°.
//! With the side tile's bottom on the path's right (the inside of a
//! clockwise frame; flip across swaps it), a right turn is an **outer**
//! corner and a left turn an **inner** one. A corner kind with no tile and
//! no auto mode stays **empty**, as in Illustrator: the legs stop where a
//! corner tile would begin and nothing fills the corner.
//!
//! Corner handling (`outerCorner` / `innerCorner`):
//! - **artwork** and **`centered`** (which uses the side tile): the legs stop
//!   `d = a·tan(θ/2)` before the corner (`a` = half thickness, θ the turn;
//!   `tan(θ/2)` capped at the 150° value for hairpins, used consistently by
//!   cuts, quads and miters). Each corner's `d` is decided once, capped at
//!   half of either neighbouring leg, and both legs use it. The tile fills the corner quad — incoming edge,
//!   outer miter, outgoing edge, inner intersection — by a bilinear map. At
//!   90° this is Illustrator's layout exactly (matched against stock pattern
//!   brushes): both corner tiles are authored with the incoming side on their
//!   bottom edge and the outgoing side on their right edge (the outer tile is
//!   a clockwise frame's top-left corner). The outer tile turns with the path;
//!   the inner tile is placed mirrored across the incoming leg's line.
//! - **`between`**: the legs run into the corner and meet along the miter;
//!   whole tiles fit on each leg. The tiles nearest the corner are sheared so
//!   the leg ends on the miter line; the shear starts on a tile boundary far
//!   enough back that the map never folds (slope ≥ 0.1), and a leg too short
//!   for that clamps the miter.
//! - **`sliced`**: as `between`, but each leg's tiling is shifted half a
//!   tile at the corner and the tiles are cut at the miter, so one tile
//!   appears folded around the corner like a picture-frame joint.
//! - **`overlap`**: the legs run straight on past the corner by `a`; the
//!   end tiles overlap in the corner square.
//!
//! # Fit (per leg, on the length left after start / end / corner tiles)
//!
//! With tile length `Ls` and gap `spacing × Ls`:
//! - **stretch**: `n = round(L / (Ls + gap))` (≥ 1) tiles, tiles and gaps
//!   scaled to fill exactly.
//! - **add-space**: `n = floor(L / (Ls + gap))` (≥ 1) tiles at their size,
//!   the slack spread evenly as extra space around every tile.
//! - **approximate-path**: `n = round(...)` tiles at their size, centred on
//!   the leg; the run may overshoot or fall short by under half a tile (the
//!   tiles extend along the leg's end tangents) — our stand-in for
//!   Illustrator moving the tiles slightly off the path. Next to a corner tile
//!   it rounds down (never overhanging the corner); on a closed loop without
//!   corners it spreads the remainder like add-space (seamless).
//!
//! Flip along mirrors every side tile end to end and swaps the start and end
//! tiles; corner tiles are mirrored across the diagonal that swaps their
//! incoming and outgoing join edges (so the joins stay put). Flip across
//! mirrors every tile over the path (and swaps outer and inner corners).
//! Every tile — side, corner, start, end — counts toward `MAX_TILES` and the
//! segment budget; over either, the placement is refused (`too_complex`).
//! Placed tiles carry their arc length (`PlacedAt`). Closed paths with
//! corners start at their first corner; without corners the whole loop is
//! one leg (stretch keeps it seamless). Output ids: `<node>#<subpath>.
//! <leg>.<tile>` for side tiles, `.start`, `.end`, `.c<corner>`.

use crate::value::{Map, Value};

use super::art::{paint_part, window_pieces, ArcWindow};
use super::artwork::{collect_artwork, default_geometry, Artwork, GeometryFn};
use super::clip::{clip_x, clip_x_open};
use super::colorize::{self, parse_colorize, Colorize};
use super::common::{
    error, node_stroke, parse_request, placement_json, subpath_length, with_json_cap, BrushStroke,
    PlacedAt, Placement, Request, MAX_OUTPUT_SEGMENTS, MAX_WORK_PIECES,
};
use super::dyn_value::{draw, ink_at, largest, parse_dyn, Dyn};
use super::geom::{Bounds, Subpath};
use super::spine::Spine;
use super::warp::{crossings, WarpBudget, Warper};
use crate::path_geom::vec2::{v2, Cubic, V2};
use crate::types::Affine;

/// Turns sharper than this (degrees) are corners.
pub const CORNER_MIN_DEGREES: f64 = 15.0;
/// Most tiles per stroke.
pub const MAX_TILES: usize = 10_000;
const DEFAULT_TOLERANCE: f64 = 0.25;
const MAX_TURN: f64 = 150.0 * std::f64::consts::PI / 180.0;
const SALT_SCALE: u64 = 0x61;
/// A pen-driven tile never shrinks below this share of the largest scale
/// (keeps the tile count bounded where the pen barely touches).
const MIN_TILE_SCALE: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoCorner {
    Centered,
    Between,
    Sliced,
    Overlap,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CornerTile {
    Art(Vec<Value>),
    Auto(AutoCorner),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternFit {
    Stretch,
    AddSpace,
    ApproximatePath,
}

impl PatternFit {
    fn parse(v: Option<&Value>) -> Option<PatternFit> {
        Some(match v.and_then(Value::as_str)? {
            "stretch" => PatternFit::Stretch,
            "add-space" => PatternFit::AddSpace,
            "approximate-path" => PatternFit::ApproximatePath,
            _ => return None,
        })
    }
}

/// `PatternBrushDefinition` (nodes.ts), normalised like `normalizeBrushDefinition`.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternBrush {
    pub id: String,
    pub name: String,
    pub side: Vec<Value>,
    pub outer_corner: Option<CornerTile>,
    pub inner_corner: Option<CornerTile>,
    pub start: Option<Vec<Value>>,
    pub end: Option<Vec<Value>>,
    pub scale: Dyn,
    pub spacing: f64,
    pub fit: PatternFit,
    pub flip_along: bool,
    pub flip_across: bool,
    pub colorize: Colorize,
    /// `scaleWithWeight`: Illustrator's sizing — the stroke weight multiplies
    /// the brush's designed size. Without it (the default, as in Figma) the
    /// stroke width is the width of the mark: the design is scaled to it.
    pub scale_with_weight: bool,
}

fn corner_tile(raw: Option<&Value>) -> Option<CornerTile> {
    match raw? {
        Value::String(s) => Some(CornerTile::Auto(match s.as_str() {
            "centered" => AutoCorner::Centered,
            "between" => AutoCorner::Between,
            "sliced" => AutoCorner::Sliced,
            "overlap" => AutoCorner::Overlap,
            _ => return None,
        })),
        v => super::art::artwork_nodes(Some(v)).map(CornerTile::Art),
    }
}

pub fn parse_pattern_brush(raw: &Value) -> Option<PatternBrush> {
    let o = raw.as_object()?;
    let id = o
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())?;
    if o.get("type").and_then(Value::as_str) != Some("pattern") {
        return None;
    }
    let tiles = o.get("tiles")?.as_object()?;
    let side = super::art::artwork_nodes(tiles.get("side"))?;
    Some(PatternBrush {
        id: id.to_string(),
        name: o
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or(id)
            .to_string(),
        side,
        outer_corner: corner_tile(tiles.get("outerCorner")),
        inner_corner: corner_tile(tiles.get("innerCorner")),
        start: super::art::artwork_nodes(tiles.get("start")),
        end: super::art::artwork_nodes(tiles.get("end")),
        scale: parse_dyn(o.get("scale"), 1.0, 0.0, f64::INFINITY),
        spacing: o
            .get("spacing")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
            .max(0.0),
        fit: PatternFit::parse(o.get("fit")).unwrap_or(PatternFit::Stretch),
        flip_along: o.get("flipAlong").and_then(Value::as_bool) == Some(true),
        flip_across: o.get("flipAcross").and_then(Value::as_bool) == Some(true),
        colorize: parse_colorize(o.get("colorize")),
        scale_with_weight: super::common::scale_with_weight(o),
    })
}

/// `BrushApplication.overrides` as they apply to a pattern brush.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PatternOverrides {
    pub scale: Option<Dyn>,
    pub flip_along: Option<bool>,
    pub flip_across: Option<bool>,
    pub colorize: Option<Colorize>,
    /// Art-only fits are ignored.
    pub fit: Option<PatternFit>,
    /// Tile thickness follows the stroke's width profile.
    pub width_from_profile: bool,
}

pub fn parse_pattern_overrides(raw: Option<&Map>) -> PatternOverrides {
    let Some(o) = raw else {
        return PatternOverrides::default();
    };
    PatternOverrides {
        scale: o
            .get("scale")
            .map(|v| parse_dyn(Some(v), 1.0, 0.0, f64::INFINITY)),
        flip_along: o.get("flipAlong").and_then(Value::as_bool),
        flip_across: o.get("flipAcross").and_then(Value::as_bool),
        colorize: o.get("colorize").map(|v| parse_colorize(Some(v))),
        fit: PatternFit::parse(o.get("fit")),
        width_from_profile: o.get("widthFrom").and_then(Value::as_str) == Some("profile"),
    }
}

/// Collected tile artworks.
#[derive(Debug, Clone, Default)]
pub struct PatternArtwork {
    pub side: Artwork,
    pub outer: Option<Artwork>,
    pub inner: Option<Artwork>,
    pub start: Option<Artwork>,
    pub end: Option<Artwork>,
}

pub fn collect_pattern_artwork(brush: &PatternBrush, geometry: GeometryFn) -> PatternArtwork {
    let corner = |c: &Option<CornerTile>| match c {
        Some(CornerTile::Art(nodes)) => Some(collect_artwork(nodes, geometry)),
        _ => None,
    };
    PatternArtwork {
        side: collect_artwork(&brush.side, geometry),
        outer: corner(&brush.outer_corner),
        inner: corner(&brush.inner_corner),
        start: brush.start.as_ref().map(|n| collect_artwork(n, geometry)),
        end: brush.end.as_ref().map(|n| collect_artwork(n, geometry)),
    }
}

/// A tile normalised to x ∈ [0, w], y centred, flips applied.
#[derive(Debug, Clone)]
struct Tile {
    art: Artwork,
    w: f64,
    h: f64,
}

fn tile(art: &Artwork, flip_along: bool, flip_across: bool) -> Option<Tile> {
    let b = art.bounds();
    if b.is_empty() {
        return None;
    }
    let (w, h) = (b.width(), b.height());
    if !w.is_finite() || !h.is_finite() || w.max(h) <= 1e-9 {
        return None;
    }
    let cy = (b.min.y + b.max.y) / 2.0;
    let sx = if flip_along { -1.0 } else { 1.0 };
    let sy = if flip_across { -1.0 } else { 1.0 };
    // x' = sx·(x − min.x) (+ w when mirrored), y' = sy·(y − cy)
    let m = Affine {
        a: sx,
        b: 0.0,
        c: 0.0,
        d: sy,
        e: if flip_along { b.max.x } else { -b.min.x },
        f: -sy * cy,
    };
    Some(Tile {
        art: art.transformed(m),
        w: w.max(1e-9),
        h,
    })
}

/// How a leg ends.
#[derive(Debug, Clone, Copy, PartialEq)]
enum End {
    /// Path start / end, or a closed loop's seam without corners.
    Open,
    /// At corner `j`: cut back by `d` (corner tile / centered).
    Cut(usize, f64),
    /// Sheared to the miter; `tan` = tan(θ/2) signed by the turn.
    Miter { tan: f64, sliced: bool },
    /// Straight on past the corner by `ext`.
    Extend(f64),
    /// Not a split (unused on real legs).
    Through,
}

struct Corner {
    pos: V2,
    t_in: V2,
    t_out: V2,
    /// Signed turn, radians; positive turns right (clockwise on screen).
    turn: f64,
    kind: CornerKind,
}

#[derive(Clone)]
enum CornerKind {
    Tile(Tile),
    Auto(AutoCorner),
    /// No tile for this corner kind: the legs stop as for a corner tile and
    /// the corner stays bare (Illustrator).
    Empty,
}

fn left(t: V2) -> V2 {
    v2(t.y, -t.x)
}

/// Bilinear map of the unit square onto a quad `[p00, p10, p11, p01]`
/// (u along x, v along y).
fn bilinear(q: &[V2; 4], u: f64, v: f64) -> V2 {
    let a = q[0].lerp(q[1], u);
    let b = q[3].lerp(q[2], u);
    a.lerp(b, v)
}

fn transform_subpaths(subpaths: &[Subpath], m: Affine) -> Vec<Subpath> {
    let ap = |p: V2| v2(m.a * p.x + m.c * p.y + m.e, m.b * p.x + m.d * p.y + m.f);
    subpaths
        .iter()
        .map(|sp| Subpath {
            segs: sp
                .segs
                .iter()
                .map(|c| Cubic([ap(c.0[0]), ap(c.0[1]), ap(c.0[2]), ap(c.0[3])]))
                .collect(),
            closed: sp.closed,
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct PatternOptions<'a> {
    /// Max distance between the output cubics and the exact warp, px.
    pub tolerance: f64,
    /// Trim / dash: draw only these arc-length windows (as [`super::art::ArtOptions::windows`]).
    /// Tiles are laid out along the WHOLE path, then side / start / end tiles
    /// are clipped at the window ends in leg space before warping (a reveal);
    /// corner tiles stay whole and show when their joint is in a window.
    /// `None` = the whole stroke; `Some(&[])` = nothing.
    pub windows: Option<&'a [ArcWindow]>,
}

impl Default for PatternOptions<'_> {
    fn default() -> Self {
        PatternOptions {
            tolerance: DEFAULT_TOLERANCE,
            windows: None,
        }
    }
}

/// A subpath's window runs in its measured arc length, for a placement whose
/// arc lengths may run past the closed seam (a leg through the start, an
/// overlap extension before 0): pieces are repeated one lap either side and
/// runs that touch are merged, so a tile across the seam is one piece. Each
/// run keeps the lowest window index that feeds it (its id suffix). `None`:
/// the subpath is drawn whole (no windows, or a window covering all of it).
fn subpath_runs(
    windows: Option<&[ArcWindow]>,
    si: usize,
    len: f64,
    closed: bool,
) -> Option<Vec<(usize, f64, f64)>> {
    let ws = windows?;
    let mut runs: Vec<(usize, f64, f64)> = Vec::new();
    for (wi, w) in ws.iter().enumerate().filter(|(_, w)| w.subpath == si) {
        for (a, b) in window_pieces(w, len, closed) {
            if a <= 1e-9 && b >= len - 1e-7 {
                return None;
            }
            let laps: &[f64] = if closed { &[-1.0, 0.0, 1.0] } else { &[0.0] };
            for k in laps {
                runs.push((wi, a + k * len, b + k * len));
            }
        }
    }
    runs.sort_by(|x, y| x.1.total_cmp(&y.1));
    let mut merged: Vec<(usize, f64, f64)> = Vec::new();
    for r in runs {
        match merged.last_mut() {
            Some(m) if r.1 <= m.2 + 1e-9 => {
                m.2 = m.2.max(r.2);
                m.0 = m.0.min(r.0);
            }
            _ => merged.push(r),
        }
    }
    // Windows that together cover the whole subpath (a ring window starting
    // anywhere, or overlapping dashes) draw it whole: same ids as no windows.
    if merged
        .iter()
        .any(|(_, a, b)| *a <= 1e-9 && *b >= len - 1e-7)
    {
        return None;
    }
    Some(merged)
}

/// A tile's piece under trim / dash: id suffix, arc-length span, lower /
/// upper clip in leg x (`None` = uncut on that side).
type TilePiece = (String, (f64, f64), (Option<f64>, Option<f64>));

/// Place a pattern brush along a stroke. Pure and total.
pub fn pattern_brush_geometry(
    stroke: &BrushStroke,
    brush: &PatternBrush,
    overrides: &PatternOverrides,
    artwork: &PatternArtwork,
    opts: &PatternOptions,
) -> Placement {
    let tolerance = opts.tolerance;
    let mut out = Placement {
        skipped: artwork.side.skipped.clone(),
        bounds: Bounds::empty(),
        ..Placement::default()
    };
    let tol = if tolerance.is_finite() {
        tolerance.clamp(0.01, 100.0)
    } else {
        DEFAULT_TOLERANCE
    };
    let flip_along = overrides.flip_along.unwrap_or(brush.flip_along);
    let flip_across = overrides.flip_across.unwrap_or(brush.flip_across);
    let fit = overrides.fit.unwrap_or(brush.fit);
    let scale = overrides.scale.unwrap_or(brush.scale);
    let colorize = overrides.colorize.as_ref().unwrap_or(&brush.colorize);
    let ctx = colorize::resolve(
        colorize,
        stroke.stroke_color,
        &artwork.side.weighted_colors(),
    );
    let Some(side) = tile(&artwork.side, flip_along, flip_across) else {
        return out;
    };
    let mk = |a: &Option<Artwork>| a.as_ref().and_then(|a| tile(a, flip_along, flip_across));
    let (mut start_tile, mut end_tile) = (mk(&artwork.start), mk(&artwork.end));
    if flip_along {
        std::mem::swap(&mut start_tile, &mut end_tile);
    }
    // Corner tiles keep their own orientation: flip along mirrors them across
    // the diagonal that swaps their incoming and outgoing edges (see below).
    let mk_corner = |a: &Artwork| tile(a, false, flip_across);
    let kind_for = |c: &Option<CornerTile>, art: &Option<Artwork>| -> Option<CornerKind> {
        match c {
            Some(CornerTile::Art(_)) => art.as_ref().and_then(mk_corner).map(CornerKind::Tile),
            Some(CornerTile::Auto(AutoCorner::Centered)) => {
                mk_corner(&artwork.side).map(CornerKind::Tile)
            }
            Some(CornerTile::Auto(a)) => Some(CornerKind::Auto(*a)),
            None => None,
        }
    };
    let mut outer = kind_for(&brush.outer_corner, &artwork.outer);
    let mut inner = kind_for(&brush.inner_corner, &artwork.inner);
    if flip_across {
        std::mem::swap(&mut outer, &mut inner);
    }
    // Sizing: the side tile's height is the stroke width, unless the weight
    // multiplies it.
    let stroke_width = if stroke.stroke_width.is_finite() {
        stroke.stroke_width.max(0.0) * super::common::weight_unit(brush.scale_with_weight, side.h)
    } else {
        0.0
    };
    let total_len: f64 = stroke.subpaths.iter().map(subpath_length).sum();
    if !total_len.is_finite() || total_len <= 1e-9 {
        return out;
    }
    // The tile scale at a whole-path fraction `t`: one value per stroke unless
    // it follows the pen (then per tile, floored at MIN_TILE_SCALE).
    let per_tile = scale.uses_ink();
    let random = draw(stroke.seed, SALT_SCALE, 0);
    let k_fixed = stroke_width * scale.eval(&Default::default(), random).max(0.0);
    let k_ref = if per_tile {
        stroke_width * largest(&scale).max(0.0)
    } else {
        k_fixed
    };
    if !k_ref.is_finite() || k_ref <= 1e-12 {
        return out;
    }
    let k_at = |t: f64| -> f64 {
        if !per_tile {
            return k_fixed;
        }
        let k = stroke_width * scale.eval(&ink_at(stroke.ink, t), random).max(0.0);
        if k.is_finite() {
            k.max(k_ref * MIN_TILE_SCALE)
        } else {
            k_ref
        }
    };
    let profile = if overrides.width_from_profile {
        stroke.profile
    } else {
        None
    };
    // Mean profile width at `t` (corner tiles), 1 without a profile.
    let profile_mean = |t: f64| {
        profile.map_or(1.0, |p| {
            let (l, r) = p.width_at(t, None);
            ((l + r) / 2.0).max(0.0)
        })
    };
    // Tiles and segments placed so far, every kind counted (side, corner, start, end).
    let mut tiles_placed = 0usize;
    let mut segs_placed = 0usize;
    let mut budget = WarpBudget::new(MAX_WORK_PIECES);
    let too_many = || {
        format!("pattern brush placement needs more than {MAX_TILES} tiles or {MAX_OUTPUT_SEGMENTS} segments (raise scale or spacing)")
    };
    // `tan(θ/2)` capped at the 150° turn: corner cuts, quads and miters use
    // the real tangents with this one cap, consistently.
    let max_tan = (MAX_TURN / 2.0).tan();
    let half_tan = |turn: f64| (turn / 2.0).tan().clamp(-max_tan, max_tan);

    let mut cum = 0.0;
    for (si, sp) in stroke.subpaths.iter().enumerate() {
        let sub_start = cum;
        cum += subpath_length(sp);
        let segs: Vec<Cubic> = sp
            .segs
            .iter()
            .copied()
            .filter(|c| c.0.iter().any(|p| p.dist(c.0[0]) > 1e-12))
            .collect();
        if segs.is_empty() {
            continue;
        }
        // Corners at joins (joint j = start of segs[j]; j = 0 only when closed).
        let n = segs.len();
        let mut corners: Vec<(usize, Corner)> = Vec::new();
        let first_joint = if sp.closed { 0 } else { 1 };
        for j in first_joint..n {
            let prev = &segs[(j + n - 1) % n];
            let (Some(t_in), Some(t_out)) = (prev.tangent(1.0), segs[j].tangent(0.0)) else {
                continue;
            };
            let turn = t_in.cross(t_out).atan2(t_in.dot(t_out));
            if turn.abs() < CORNER_MIN_DEGREES.to_radians() {
                continue;
            }
            let kind = if turn > 0.0 { &outer } else { &inner };
            let kind = kind.clone().unwrap_or(CornerKind::Empty);
            corners.push((
                j,
                Corner {
                    pos: segs[j].0[0],
                    t_in,
                    t_out,
                    turn,
                    kind,
                },
            ));
        }
        // Legs: runs of segments between corners.
        let mut legs: Vec<(Vec<Cubic>, Option<usize>, Option<usize>)> = Vec::new();
        // Arc length (own measure) from the subpath start to each joint, and
        // the scale to the outliner's measure (`PlacedAt`).
        let seg_len: Vec<f64> = segs.iter().map(|c| c.arc_len()).collect();
        let own_total: f64 = seg_len.iter().sum();
        let joint_s: Vec<f64> = std::iter::once(0.0)
            .chain(seg_len.iter().scan(0.0, |acc, l| {
                *acc += l;
                Some(*acc)
            }))
            .collect();
        let measured = stroke
            .measured
            .get(si)
            .copied()
            .filter(|m| m.is_finite() && *m > 1e-9)
            .unwrap_or(own_total);
        let to_m = if own_total > 0.0 {
            measured / own_total
        } else {
            1.0
        };
        let runs = subpath_runs(opts.windows, si, measured, sp.closed);
        if runs.as_ref().is_some_and(|r| r.is_empty()) {
            continue;
        }
        let runs = runs.as_deref();
        let mut leg_start: Vec<f64> = Vec::new();
        if corners.is_empty() {
            legs.push((segs.clone(), None, None));
            leg_start.push(0.0);
        } else if sp.closed {
            let m = corners.len();
            for c in 0..m {
                let (j0, j1) = (corners[c].0, corners[(c + 1) % m].0);
                let mut run = Vec::new();
                let mut j = j0;
                loop {
                    run.push(segs[j]);
                    j = (j + 1) % n;
                    if j == j1 {
                        break;
                    }
                }
                legs.push((run, Some(c), Some((c + 1) % m)));
                leg_start.push(joint_s[j0]);
            }
        } else {
            let mut j0 = 0;
            for (c, (j, _)) in corners.iter().enumerate() {
                legs.push((
                    segs[j0..*j].to_vec(),
                    if c == 0 { None } else { Some(c - 1) },
                    Some(c),
                ));
                leg_start.push(joint_s[j0]);
                j0 = *j;
            }
            legs.push((segs[j0..].to_vec(), Some(corners.len() - 1), None));
            leg_start.push(joint_s[j0]);
        }
        // Per-corner cut, decided once for both legs (first pass): the corner
        // tile's half-depth, never more than half of either neighbouring leg.
        let leg_len: Vec<f64> = legs
            .iter()
            .map(|(segs, _, _)| segs.iter().map(|c| c.arc_len()).sum())
            .collect();
        // Whole-path fraction at a subpath arc length (own measure).
        let t_of_own = |s: f64| ((sub_start + s) / total_len).clamp(0.0, 1.0);
        // Each corner's half thickness: the tile scale (and profile) there.
        let corner_a: Vec<f64> = corners
            .iter()
            .map(|(j, _)| {
                let t = t_of_own(joint_s[*j]);
                side.h * k_at(t) / 2.0 * profile_mean(t)
            })
            .collect();
        let mut cut_at: Vec<f64> = corners
            .iter()
            .zip(&corner_a)
            .map(|((_, c), a)| a * half_tan(c.turn).abs())
            .collect();
        for (li, (_, c0, c1)) in legs.iter().enumerate() {
            for c in [c0, c1].into_iter().flatten() {
                cut_at[*c] = cut_at[*c].min(leg_len[li] / 2.0);
            }
        }
        let end_of = |c: Option<usize>| -> End {
            let Some(c) = c else {
                return End::Open;
            };
            let corner = &corners[c].1;
            let tan = half_tan(corner.turn);
            match &corner.kind {
                CornerKind::Tile(_) | CornerKind::Empty => End::Cut(c, cut_at[c]),
                CornerKind::Auto(AutoCorner::Between) => End::Miter { tan, sliced: false },
                CornerKind::Auto(AutoCorner::Sliced) => End::Miter { tan, sliced: true },
                CornerKind::Auto(AutoCorner::Overlap) => End::Extend(corner_a[c]),
                CornerKind::Auto(AutoCorner::Centered) => End::Through,
            }
        };
        let open_path = !sp.closed;
        let leg_count = legs.len();
        for (li, (leg_segs, c0, c1)) in legs.iter().enumerate() {
            let leg = Subpath {
                segs: leg_segs.clone(),
                closed: sp.closed && corners.is_empty(),
            };
            let Some(spine) = Spine::new(&leg, (tol * 0.1).min(0.05)) else {
                continue;
            };
            if spine.truncated {
                out.too_complex = Some("the stroke path needs more than 200 000 samples".into());
                return out;
            }
            let l = spine.length;
            // Leg x → the subpath's arc length in the outliner's measure.
            let (ls0, leg_own) = (leg_start[li], leg_len[li].max(1e-12));
            let s_of = |x: f64| (ls0 + x * leg_own / l) * to_m;
            let x_of = |s: f64| (s / to_m - ls0) * l / leg_own;
            // Leg x → whole-path fraction (ink, profile) and the tile scale there.
            let t_of_x = |x: f64| t_of_own(ls0 + x.clamp(0.0, l) * leg_own / l);
            let k_x = |x: f64| k_at(t_of_x(x));
            let (e0, e1) = (end_of(*c0), end_of(*c1));
            // Half thickness at each end (the corner's; an open end never uses it).
            let a_end = |c: &Option<usize>, x: f64| {
                c.map_or_else(|| side.h * k_x(x) / 2.0, |c| corner_a[c])
            };
            let (a0, a1) = (a_end(c0, 0.0), a_end(c1, l));
            let cut = |e: End| match e {
                End::Cut(_, d) => d.min(l / 2.0),
                _ => 0.0,
            };
            let mut x0 = cut(e0);
            let mut x1 = l - cut(e1);
            if let End::Extend(e) = e0 {
                x0 -= e;
            }
            if let End::Extend(e) = e1 {
                x1 += e;
            }
            let sliced0 = matches!(e0, End::Miter { sliced: true, .. });
            let sliced1 = matches!(e1, End::Miter { sliced: true, .. });
            let cut0 = matches!(e0, End::Cut(..));
            let cut1 = matches!(e1, End::Cut(..));

            // Layout first (start / end tiles reserve their length), then the map.
            let start = if open_path && li == 0 {
                start_tile.as_ref()
            } else {
                None
            };
            let end = if open_path && li + 1 == leg_count {
                end_tile.as_ref()
            } else {
                None
            };
            let start_x = x0;
            let k_start = k_x(x0);
            if let Some(t) = start {
                x0 += t.w * k_start;
            }
            let k_end = k_x(x1);
            if let Some(t) = end {
                x1 -= t.w * k_end;
            }
            let end_x = x1;
            if sliced0 {
                x0 -= side.w * k_x(x0) / 2.0;
            }
            if sliced1 {
                x1 += side.w * k_x(x1) / 2.0;
            }
            let avail = x1 - x0;
            let seamless = leg.closed;
            let (stretch_fit, spread) = match fit {
                PatternFit::Stretch => (true, false),
                PatternFit::AddSpace => (false, true),
                // A closed loop has no ends to overshoot: approximate-path
                // spreads the remainder like add-space so it stays seamless.
                PatternFit::ApproximatePath => (false, seamless),
            };
            // Tiles end to end from x0, each `(k, unit)` at the scale read at
            // its centre; the last one reaches past x1 (it covers `frac` of its
            // unit before x1). Then the fit keeps round(n) (stretch,
            // approximate-path) or floor(n) (add-space; approximate-path next
            // to a corner tile, never overhanging it) of them, at least one.
            let mut units: Vec<(f64, f64)> = Vec::new();
            let mut frac = 1.0;
            if avail > 1e-9 {
                let mut x = x0;
                while x < x1 - 1e-9 {
                    // Budget before any allocation (a huge leg of tiny tiles).
                    if units.len() >= MAX_TILES {
                        out.too_complex = Some(too_many());
                        return out;
                    }
                    let kk = if per_tile {
                        k_x(x + side.w * k_x(x) * (1.0 + brush.spacing) / 2.0)
                    } else {
                        k_fixed
                    };
                    let unit = side.w * kk * (1.0 + brush.spacing);
                    if !(unit > 1e-12) {
                        break;
                    }
                    units.push((kk, unit));
                    frac = (x1 - x) / unit;
                    x += unit;
                }
            }
            if !units.is_empty() {
                let n = units.len();
                let round_down = fit == PatternFit::AddSpace
                    || (fit == PatternFit::ApproximatePath && !seamless && (cut0 || cut1));
                let whole = if round_down { frac >= 1.0 - 1e-9 } else { frac >= 0.5 };
                units.truncate(if whole { n } else { (n - 1).max(1) });
            }
            // Side tiles: (leg x of the tile's start, scale along, scale across).
            let sum: f64 = units.iter().map(|u| u.1).sum();
            let n_tiles = units.len().max(1) as f64;
            let mut slots: Vec<(f64, f64, f64)> = Vec::with_capacity(units.len());
            let mut pos = if stretch_fit || spread {
                x0
            } else {
                x0 + (avail - sum) / 2.0
            };
            for &(kk, unit) in &units {
                let ls = side.w * kk;
                let gap = brush.spacing * ls;
                if stretch_fit {
                    let f = avail / sum;
                    slots.push((pos + gap * f / 2.0, kk * f, kk));
                    pos += unit * f;
                } else if spread {
                    let cell = unit + (avail - sum) / n_tiles;
                    slots.push((pos + (cell - ls) / 2.0, kk, kk));
                    pos += cell;
                } else {
                    slots.push((pos + gap / 2.0, kk, kk));
                    pos += unit;
                }
            }
            let extra = start.is_some() as usize + end.is_some() as usize;
            let count = slots.len();
            let side_segs = side.art.segment_count().max(1);
            tiles_placed = tiles_placed.saturating_add(count + extra);
            segs_placed = segs_placed
                .saturating_add(count.saturating_mul(side_segs))
                .saturating_add(start.map_or(0, |t| t.art.segment_count()))
                .saturating_add(end.map_or(0, |t| t.art.segment_count()));
            if tiles_placed > MAX_TILES || segs_placed > MAX_OUTPUT_SEGMENTS {
                out.too_complex = Some(too_many());
                return out;
            }

            // Mitered ends shear the tiles nearest the corner so the leg ends on
            // the miter line. The shear starts on a tile boundary, far enough
            // back that the map never folds (slope ≥ 0.1: ramp ≥ a·|tan| / 0.9);
            // a leg too short for that shears over half its length with the
            // miter clamped.
            let bounds_x: Vec<f64> = slots
                .iter()
                .flat_map(|&(xs, sx, _)| [xs, xs + side.w * sx])
                .filter(|x| *x > 1e-9 && *x < l - 1e-9)
                .collect();
            let need = |e: End, a: f64| match e {
                End::Miter { tan, .. } => a * tan.abs() / 0.9,
                _ => 0.0,
            };
            let pick = |dists: Vec<f64>, need: f64| {
                dists
                    .into_iter()
                    .filter(|d| *d >= need - 1e-9 && *d <= l / 2.0 + 1e-9)
                    .fold(f64::INFINITY, f64::min)
                    .min(l / 2.0)
                    .max(1e-9)
            };
            let ramp0 = pick(bounds_x.clone(), need(e0, a0));
            let ramp1 = pick(bounds_x.iter().map(|x| l - x).collect(), need(e1, a1));
            let miter = |e: End, ramp: f64, a: f64| match e {
                End::Miter { tan, .. } => {
                    let lim = 0.9 * ramp / a.max(1e-12);
                    tan.clamp(-lim, lim)
                }
                _ => 0.0,
            };
            let (m0, m1) = (miter(e0, ramp0, a0), miter(e1, ramp1, a1));
            // Leg coordinates (x along the centerline, a across, + = right) → page.
            let map = |p: V2| -> V2 {
                let (x, mut a) = (p.x, p.y);
                if let Some(prof) = profile {
                    let (pl, pr) = prof.width_at(t_of_x(x), None);
                    a *= if a > 0.0 { pr } else { pl };
                }
                let mut s = x;
                if m0 != 0.0 && x < ramp0 {
                    s -= -a * m0 * (ramp0 - x) / ramp0;
                }
                if m1 != 0.0 && x > l - ramp1 {
                    s += -a * m1 * (x - (l - ramp1)) / ramp1;
                }
                let q = spine.eval(s);
                q.pos + q.right() * a
            };
            let ramp_cuts: Vec<f64> = {
                // Kinks (corners without a corner tile) and the miter ramps.
                let mut v = spine.kinks.clone();
                if m0 != 0.0 {
                    v.push(ramp0);
                }
                if m1 != 0.0 {
                    v.push(l - ramp1);
                }
                v.sort_by(|a, b| a.total_cmp(b));
                v
            };
            let warper = Warper {
                map: &map,
                tolerance: tol,
                step: (l * 1e-6).max(1e-6),
            };
            let mut over_budget = false;
            let mut place =
                |t: &Tile, id: String, x_start: f64, sx: f64, sy: f64, clip: (bool, bool)| {
                    let m = Affine {
                        a: sx,
                        b: 0.0,
                        c: 0.0,
                        d: sy,
                        e: x_start,
                        f: 0.0,
                    };
                    let len = t.w * sx;
                    let (sa, sb) = (s_of(x_start), s_of(x_start + len));
                    // (id suffix, s span, lower / upper clip in leg x): the sliced
                    // corner cuts at the leg ends, window ends cut inside the tile.
                    let slice = (clip.0.then_some(0.0), clip.1.then_some(l));
                    // Runs are sorted and disjoint: find the first that ends past
                    // the tile's start, then walk only while they start before its end.
                    let pieces: Vec<TilePiece> = match runs {
                        None => vec![(String::new(), (sa, sb), slice)],
                        Some(runs) => runs[runs.partition_point(|r| r.2 <= sa + 1e-9)..]
                            .iter()
                            .take_while(|r| r.1 < sb - 1e-9)
                            .filter(|(_, a, b)| a.max(sa) < b.min(sb) - 1e-9)
                            .map(|(wi, a, b)| {
                                let lo = (*a > sa + 1e-9).then(|| x_of(*a));
                                let hi = (*b < sb - 1e-9).then(|| x_of(*b));
                                let lo = match (lo, slice.0) {
                                    (Some(x), Some(y)) => Some(x.max(y)),
                                    (x, y) => x.or(y),
                                };
                                let hi = match (hi, slice.1) {
                                    (Some(x), Some(y)) => Some(x.min(y)),
                                    (x, y) => x.or(y),
                                };
                                (format!(".w{wi}"), (a.max(sa), b.min(sb)), (lo, hi))
                            })
                            .collect(),
                    };
                    // One tile cut by two runs fed by the same window (a
                    // wrapping window's two laps): `~k` keeps the ids unique.
                    let mut pieces = pieces;
                    let bases: Vec<String> = pieces.iter().map(|p| p.0.clone()).collect();
                    for (i, p) in pieces.iter_mut().enumerate() {
                        let k = bases[..i].iter().filter(|b| **b == bases[i]).count();
                        if k > 0 {
                            p.0 = format!("{}~{k}", bases[i]);
                        }
                    }
                    // Each piece past the first is one more placed tile.
                    let more = pieces.len().saturating_sub(1);
                    if more > 0 {
                        tiles_placed = tiles_placed.saturating_add(more);
                        segs_placed =
                            segs_placed.saturating_add(more.saturating_mul(t.art.segment_count()));
                        if tiles_placed > MAX_TILES || segs_placed > MAX_OUTPUT_SEGMENTS {
                            over_budget = true;
                        }
                    }
                    if over_budget {
                        return;
                    }
                    let mut warp_all = |subs: &[Subpath]| -> Vec<Subpath> {
                        let mut warped = Vec::new();
                        for s in subs {
                            let mut segs = Vec::new();
                            for c in &s.segs {
                                let cuts = crossings(c, |p| p.x, &ramp_cuts, 8);
                                warper.warp(c, &cuts, &mut segs, &mut budget);
                            }
                            if !segs.is_empty() {
                                warped.push(Subpath {
                                    segs,
                                    closed: s.closed,
                                });
                            }
                        }
                        warped
                    };
                    for (suffix, span, (lo, hi)) in &pieces {
                        let at = PlacedAt {
                            subpath: si,
                            s: (span.0 + span.1) / 2.0,
                            s_span: *span,
                        };
                        for shape in &t.art.shapes {
                            let whole = transform_subpaths(&shape.subpaths, m);
                            let name = format!("{}#{}.{}{}", shape.source_id, si, id, suffix);
                            let scale = shape.transform_scale * sy;
                            if lo.is_none() && hi.is_none() {
                                let warped = warp_all(&whole);
                                out.emit(
                                    name,
                                    &shape.source_id,
                                    &shape.paint,
                                    warped,
                                    scale,
                                    ctx.as_ref(),
                                    at,
                                );
                                continue;
                            }
                            // Sliced corner / window end: fill the closed cut polygon;
                            // stroke only the original outline (a cut is never stroked).
                            let cut = |subs: &[Subpath], open: bool| {
                                let f = if open { clip_x_open } else { clip_x };
                                let mut s = subs.to_vec();
                                if let Some(x) = lo {
                                    s = f(&s, *x, false);
                                }
                                if let Some(x) = hi {
                                    s = f(&s, *x, true);
                                }
                                s
                            };
                            let stroked = crate::paint::map_strokes_visibly(&shape.paint);

                            if stroked {
                                let fill = warp_all(&cut(&whole, false));
                                out.emit(
                                    name.clone(),
                                    &shape.source_id,
                                    &paint_part(&shape.paint, false),
                                    fill,
                                    scale,
                                    ctx.as_ref(),
                                    at,
                                );
                                let runs = warp_all(&cut(&whole, true));
                                out.emit(
                                    format!("{name}.stroke"),
                                    &shape.source_id,
                                    &paint_part(&shape.paint, true),
                                    runs,
                                    scale,
                                    ctx.as_ref(),
                                    at,
                                );
                            } else {
                                let fill = warp_all(&cut(&whole, false));
                                out.emit(
                                    name,
                                    &shape.source_id,
                                    &shape.paint,
                                    fill,
                                    scale,
                                    ctx.as_ref(),
                                    at,
                                );
                            }
                        }
                    }
                };
            if let Some(t) = start {
                place(t, "start".into(), start_x, k_start, k_start, (false, false));
            }
            if let Some(t) = end {
                place(t, "end".into(), end_x, k_end, k_end, (false, false));
            }
            for (i, &(xs, sx, sy)) in slots.iter().enumerate() {
                let clip = (sliced0 && i == 0, sliced1 && i + 1 == count);
                place(&side, format!("{li}.{i}"), xs, sx, sy, clip);
            }
            if over_budget {
                out.too_complex = Some(too_many());
                return out;
            }
        }
        // Corner tiles into their quads.
        for (ci, (j, corner)) in corners.iter().enumerate() {
            let CornerKind::Tile(t) = &corner.kind else {
                continue;
            };
            // Corner tiles stay whole: shown when their joint is in a window
            // (the runs repeat a lap either side of a closed seam).
            let sc = joint_s[*j] * to_m;
            if let Some(runs) = runs {
                if !runs
                    .iter()
                    .any(|(_, a, b)| sc >= a - 1e-9 && sc <= b + 1e-9)
                {
                    continue;
                }
            }
            tiles_placed = tiles_placed.saturating_add(1);
            segs_placed = segs_placed.saturating_add(t.art.segment_count());
            if tiles_placed > MAX_TILES || segs_placed > MAX_OUTPUT_SEGMENTS {
                out.too_complex = Some(too_many());
                return out;
            }
            let d = cut_at[ci];
            let (c, t1, t2, a) = (corner.pos, corner.t_in, corner.t_out, corner_a[ci]);
            let (l1, l2) = (left(t1), left(t2));
            let outer_turn = corner.turn > 0.0;
            // Illustrator's corner tile orientation (matched shape by shape
            // against stock pattern brushes, Modern decisive): both corner
            // tiles are authored in one frame — the incoming side joins the
            // tile's BOTTOM edge and the outgoing side its RIGHT edge (the
            // outer tile is a clockwise frame's top-left corner). At a right
            // (outer) turn the tile turns with the path; at a left (inner) turn
            // Illustrator places it MIRRORED across the incoming leg's line:
            // the same frame, with the outside on the path's right.
            // [LT, RT, RB, LB], outside `o` = left normal (outer) / right (inner):
            // [outer miter, outgoing cut outside, inner intersection, incoming cut outside].
            let (o1, o2) = if outer_turn {
                (l1, l2)
            } else {
                (l1 * -1.0, l2 * -1.0)
            };
            let quad = [
                c + t1 * d + o1 * a,
                c + t2 * d + o2 * a,
                c - t1 * d - o1 * a,
                c - t1 * d + o1 * a,
            ];
            // Tile coords (x ∈ [0, w], y ∈ [−h/2, h/2]) → unit square → quad.
            // Flip along mirrors across the diagonal that swaps the tile's two
            // join edges (bottom ↔ right, so they stay on the joins): (u, v) → (v, u).
            let (tw, th) = (t.w, t.h.max(1e-9));
            let map = |p: V2| {
                let (u, v) = (p.x / tw, p.y / th + 0.5);
                let (u, v) = if flip_along { (v, u) } else { (u, v) };
                bilinear(&quad, u, v)
            };
            let warper = Warper {
                map: &map,
                tolerance: tol,
                step: (tw.max(th) * 1e-6).max(1e-9),
            };
            let at = PlacedAt {
                subpath: si,
                s: sc,
                s_span: ((joint_s[*j] - d) * to_m, (joint_s[*j] + d) * to_m),
            };
            for shape in &t.art.shapes {
                let mut warped = Vec::new();
                for s in &shape.subpaths {
                    let mut segs = Vec::new();
                    for c in &s.segs {
                        warper.warp(c, &[], &mut segs, &mut budget);
                    }
                    if !segs.is_empty() {
                        warped.push(Subpath {
                            segs,
                            closed: s.closed,
                        });
                    }
                }
                out.emit(
                    format!("{}#{}.c{}", shape.source_id, si, ci),
                    &shape.source_id,
                    &shape.paint,
                    warped,
                    shape.transform_scale * (2.0 * a / th),
                    ctx.as_ref(),
                    at,
                );
            }
        }
    }
    if budget.exhausted {
        out.too_complex = Some(format!(
            "pattern brush placement needs more than {MAX_WORK_PIECES} warp steps (raise the tolerance or simplify the path)"
        ));
    }
    out
}

/// Place a pattern brush on a path node (the whole stroke).
pub fn pattern_brush_for_node(
    node: &Map,
    brush: &PatternBrush,
    geometry: GeometryFn,
    tolerance: f64,
) -> Placement {
    pattern_brush_for_node_with(
        node,
        brush,
        geometry,
        &PatternOptions {
            tolerance,
            windows: None,
        },
    )
}

/// Place a pattern brush on a path node with [`PatternOptions`] (trim / dash windows).
pub fn pattern_brush_for_node_with(
    node: &Map,
    brush: &PatternBrush,
    geometry: GeometryFn,
    opts: &PatternOptions,
) -> Placement {
    let ns = node_stroke(node);
    let artwork = collect_pattern_artwork(brush, geometry);
    let stroke = BrushStroke {
        subpaths: ns.subpaths,
        stroke_width: ns.stroke_width,
        stroke_color: ns.stroke_color.as_ref(),
        profile: ns.profile.as_ref(),
        ink: ns.ink.as_ref(),
        seed: ns.seed,
        measured: ns.measured,
    };
    let overrides = parse_pattern_overrides(ns.overrides.as_ref());
    pattern_brush_geometry(&stroke, brush, &overrides, &artwork, opts)
}

/// JSON in / JSON out, same envelope as [`super::art::art_brush_json`] with a
/// `PatternBrushDefinition` as `brush`.
pub fn pattern_brush_json(request_json: &str) -> String {
    pattern_brush_json_with(request_json, &default_geometry)
}

pub fn pattern_brush_json_with(request_json: &str, geometry: GeometryFn) -> String {
    match parse_request(request_json) {
        Ok(req) => pattern_request(req, geometry),
        Err(e) => e,
    }
}

pub(crate) fn pattern_request(req: Request, geometry: GeometryFn) -> String {
    let Some(brush) = parse_pattern_brush(&req.brush) else {
        return error(
            "invalid_brush",
            "brush must be a pattern brush definition with tiles.side.nodes",
        );
    };
    let opts = PatternOptions {
        tolerance: req.tolerance.unwrap_or(DEFAULT_TOLERANCE),
        windows: req.windows.as_deref(),
    };
    placement_json(&with_json_cap(|| {
        pattern_brush_for_node_with(&req.node, &brush, geometry, &opts)
    }))
}
