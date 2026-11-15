//! Vector patterns (Illustrator's pattern swatches, E8.5): the document's pattern table
//! (`canvas.patterns`), its tile lattice and its tile artwork. The paint that uses one is
//! `paint::Paint::Pattern` (`{ type: 'pattern', patternId, transform? }`).
//!
//! A pattern definition, as stored (the mirror of `packages/editor/src/core/scene/patterns.ts`):
//!
//! ```jsonc
//! { "id": "pat-dots", "key": "…", "name": "Dots",
//!   "tile": { "type": "grid" | "brick-row" | "brick-column" | "hex-row" | "hex-column",
//!             "width": 20, "height": 20,          // the tile box, px (> 0)
//!             "spacingX": 0, "spacingY": 0,       // added between tiles (may be negative)
//!             "brickOffset": 0.5,                 // brick types: the shift, a fraction of the pitch
//!             "overlap": { "x": "left" | "right", "y": "top" | "bottom" } },  // which copy is in front
//!   "artwork": { "nodes": [ … ] } }               // the tile's art in tile space (a flat node list, like
//!                                                 // a brush's artwork: groups list their children)
//! ```
//!
//! **The lattice.** Copy `(i, j)` of the tile sits at [`Tile::origin`]: `(i·px, j·py)` for a grid, where the
//! pitch is the tile size plus the spacing; brick by row shifts row `j` by `frac(j·offset)·px`, brick by column
//! shifts column `i` by `frac(i·offset)·py`; hex by column steps columns by `¾·width + spacingX` and moves odd
//! columns down half a row, hex by row steps rows by `¾·height + spacingY` and moves odd rows right half a column
//! (Illustrator's hex tiles, whose boxes interlock).
//!
//! **Overlap.** Art may reach past its tile box into its neighbours. Copies draw rows outermost; `overlap.y`
//! `top` (the default) draws the bottom rows first so upper rows are in front, `bottom` the reverse; within a row,
//! `overlap.x` `left` (the default) draws right to left so left copies are in front, `right` the reverse.
//!
//! Every renderer enumerates copies with [`Tile::copies`], so they agree on which copies draw and in what order:
//! the engine clips the copies covering a layer to its region; the SVG and PDF writers put the copies covering
//! one [`Tile::period`] cell in a native pattern cell (the TS twin `core/scene/patternLattice.ts`, checked
//! against the shared cases in `fixtures/patterns/lattice-cases.json`).

use crate::value::Value;

use crate::stroke_brush::artwork::{collect_artwork, Artwork, GeometryFn};
use crate::stroke_brush::common::{PlacedAt, Placement};
use crate::types::Affine;

/// How copies of the tile are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileType {
    Grid,
    BrickRow,
    BrickColumn,
    HexRow,
    HexColumn,
}

impl TileType {
    pub fn from_str(s: Option<&str>) -> TileType {
        match s {
            Some("brick-row") => TileType::BrickRow,
            Some("brick-column") => TileType::BrickColumn,
            Some("hex-row") => TileType::HexRow,
            Some("hex-column") => TileType::HexColumn,
            _ => TileType::Grid,
        }
    }
}

/// A pattern's tile settings (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tile {
    pub kind: TileType,
    pub width: f64,
    pub height: f64,
    pub spacing_x: f64,
    pub spacing_y: f64,
    /// Brick types: the shift per row (column), a fraction of the pitch in `(0, 1)`.
    pub brick_offset: f64,
    /// `true`: left copies in front (the default); `false`: right.
    pub left_in_front: bool,
    /// `true`: upper copies in front (the default); `false`: lower.
    pub top_in_front: bool,
}

/// Why a tile lays out no copies.
#[derive(Debug, Clone, PartialEq)]
pub enum CopiesError {
    /// The pitch is not positive (the spacing cancels the tile size).
    NoPitch,
    /// More copies than the cap would draw (the count, or a lower bound of it).
    TooMany(usize),
}

/// The longest brick cycle a [`Tile::period`] cell holds (an offset of `k/n`, `n` up to this).
pub const MAX_BRICK_PERIOD: u32 = 16;

impl Tile {
    /// The tile settings of a definition (`None`: the size is missing, zero or not finite).
    pub fn parse(tile: &Value) -> Option<Tile> {
        let f = |k: &str| tile.get(k).and_then(Value::as_f64).filter(|v| v.is_finite());
        let (width, height) = (f("width")?, f("height")?);
        if !(width > 0.0 && height > 0.0) {
            return None;
        }
        let offset = f("brickOffset").filter(|o| *o > 0.0 && *o < 1.0).unwrap_or(0.5);
        let overlap = tile.get("overlap");
        let side = |k: &str| overlap.and_then(|o| o.get(k)).and_then(Value::as_str);
        Some(Tile {
            kind: TileType::from_str(tile.get("type").and_then(Value::as_str)),
            width,
            height,
            spacing_x: f("spacingX").unwrap_or(0.0),
            spacing_y: f("spacingY").unwrap_or(0.0),
            brick_offset: offset,
            left_in_front: side("x") != Some("right"),
            top_in_front: side("y") != Some("bottom"),
        })
    }

    /// The distance between neighbouring copies along x and y (columns and rows).
    pub fn pitch(&self) -> (f64, f64) {
        match self.kind {
            TileType::HexColumn => (0.75 * self.width + self.spacing_x, self.height + self.spacing_y),
            TileType::HexRow => (self.width + self.spacing_x, 0.75 * self.height + self.spacing_y),
            _ => (self.width + self.spacing_x, self.height + self.spacing_y),
        }
    }

    /// The top-left corner of copy `(i, j)`'s tile box, in pattern space.
    pub fn origin(&self, i: i64, j: i64) -> (f64, f64) {
        let (px, py) = self.pitch();
        let (x, y) = (i as f64 * px, j as f64 * py);
        let odd = |k: i64| k.rem_euclid(2) as f64;
        match self.kind {
            TileType::Grid => (x, y),
            TileType::BrickRow => (x + (j as f64 * self.brick_offset).rem_euclid(1.0) * px, y),
            TileType::BrickColumn => (x, y + (i as f64 * self.brick_offset).rem_euclid(1.0) * py),
            TileType::HexColumn => (x, y + odd(i) * py / 2.0),
            TileType::HexRow => (x + odd(j) * px / 2.0, y),
        }
    }

    /// The lattice's rectangular period: the size of a cell that tiles the plane by plain translation (what an
    /// SVG `<pattern>` or a PDF tiling pattern repeats). `None` when the pitch is not positive, or a brick offset
    /// is not a fraction `k/n` with `n ≤ MAX_BRICK_PERIOD`.
    pub fn period(&self) -> Option<(f64, f64)> {
        let (px, py) = self.pitch();
        if !(px > 1e-9 && py > 1e-9) {
            return None;
        }
        let cycle = || (1..=MAX_BRICK_PERIOD).find(|n| {
            let k = self.brick_offset * *n as f64;
            (k - k.round()).abs() < 1e-9
        });
        match self.kind {
            TileType::Grid => Some((px, py)),
            TileType::BrickRow => cycle().map(|n| (px, py * n as f64)),
            TileType::BrickColumn => cycle().map(|n| (px * n as f64, py)),
            TileType::HexColumn => Some((2.0 * px, py)),
            TileType::HexRow => Some((px, 2.0 * py)),
        }
    }

    /// The copies whose art reaches into `region` (`[min_x, min_y, max_x, max_y]`, pattern space), in drawing
    /// order (the overlap order, see the module docs). `art` is the art's bounds relative to a copy's origin (the
    /// tile box when the art stays inside it). Refuses past `cap` copies.
    pub fn copies(&self, region: [f64; 4], art: [f64; 4], cap: usize) -> Result<Vec<(i64, i64)>, CopiesError> {
        let (px, py) = self.pitch();
        if !(px > 1e-9 && py > 1e-9) {
            return Err(CopiesError::NoPitch);
        }
        let [x0, y0, x1, y1] = region;
        let [ax0, ay0, ax1, ay1] = art;
        if !(x0 < x1 && y0 < y1 && ax0 < ax1 && ay0 < ay1) || [x0, y0, x1, y1, ax0, ay0, ax1, ay1].iter().any(|v| !v.is_finite()) {
            return Ok(Vec::new());
        }
        // Candidate ranges: a copy's origin is at most one pitch past its unshifted lattice point.
        let range = |lo: f64, hi: f64, p: f64| {
            let a = ((lo / p).floor() - 2.0).max(-1e15);
            let b = ((hi / p).ceil() + 2.0).min(1e15);
            (a as i64, b as i64)
        };
        let (i0, i1) = range(x0 - ax1, x1 - ax0, px);
        let (j0, j1) = range(y0 - ay1, y1 - ay0, py);
        let (ni, nj) = ((i1 - i0 + 1).max(0) as u128, (j1 - j0 + 1).max(0) as u128);
        // Each candidate row / column holds at most one extra copy per side: past 4× the cap, refuse unseen.
        let candidates = ni * nj;
        if candidates > (cap as u128 + 16) * 4 + ni + nj {
            let estimate = ((ni.saturating_sub(4)) * (nj.saturating_sub(4))).min(usize::MAX as u128) as usize;
            return Err(CopiesError::TooMany(estimate.max(cap + 1)));
        }
        let rows: Vec<i64> = if self.top_in_front { (j0..=j1).rev().collect() } else { (j0..=j1).collect() };
        let cols: Vec<i64> = if self.left_in_front { (i0..=i1).rev().collect() } else { (i0..=i1).collect() };
        let mut out = Vec::new();
        for &j in &rows {
            for &i in &cols {
                let (ox, oy) = self.origin(i, j);
                if ox + ax1 > x0 && ox + ax0 < x1 && oy + ay1 > y0 && oy + ay0 < y1 {
                    if out.len() >= cap {
                        return Err(CopiesError::TooMany(out.len() + 1));
                    }
                    out.push((i, j));
                }
            }
        }
        Ok(out)
    }
}

/// A parsed pattern definition (borrowing its artwork).
#[derive(Debug, Clone)]
pub struct PatternDef<'a> {
    pub id: &'a str,
    pub tile: Tile,
    /// The artwork's node list (`artwork.nodes`).
    pub artwork: &'a [Value],
}

/// Parse a pattern definition (`None`: no string id, or no usable tile size).
pub fn parse_pattern(def: &Value) -> Option<PatternDef<'_>> {
    let id = def.get("id").and_then(Value::as_str)?;
    let tile = Tile::parse(def.get("tile")?)?;
    let artwork = def
        .get("artwork")
        .and_then(|a| a.get("nodes"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    Some(PatternDef { id, tile, artwork })
}

/// The pattern ids a node's paints name (fills, strokes and a rich text's glyph fills), in order, repeats kept.
pub fn node_pattern_ids(node: &crate::types::RawNode) -> Vec<String> {
    let mut out = Vec::new();
    for key in ["fills", "strokes", "textFills"] {
        if let Some(Value::Array(layers)) = node.json_field(key) {
            for l in layers {
                let paint = l.get("paint");
                if paint.and_then(|p| p.get("type")).and_then(Value::as_str) == Some("pattern") {
                    if let Some(id) = paint.and_then(|p| p.get("patternId")).and_then(Value::as_str) {
                        out.push(id.to_string());
                    }
                }
            }
        }
    }
    out
}

/// A pattern's tile artwork, prepared once per definition: its shapes in tile space with their own paints.
#[derive(Debug, Clone, Default)]
pub struct TileArt {
    /// The artwork's shapes (tile space). A pattern paint inside the art is left out (no recursion).
    pub art: Artwork,
    /// The art's bounds in tile space, strokes included (`[min_x, min_y, max_x, max_y]`), relative to a copy's
    /// origin. The tile box when there is no art.
    pub bounds: [f64; 4],
    /// Artwork nodes that have no geometry (text, images) and paint layers left out (a pattern inside a pattern).
    pub skipped: Vec<String>,
}

/// Prepare a pattern's tile artwork (`geometry` turns artwork leaves into paths, as for brushes).
pub fn tile_art(def: &PatternDef, geometry: GeometryFn) -> TileArt {
    let mut art = collect_artwork(def.artwork, geometry);
    let mut skipped: Vec<String> = art.skipped.clone();
    for shape in &mut art.shapes {
        for key in ["fills", "strokes"] {
            if let Some(Value::Array(layers)) = shape.paint.get_mut(key) {
                let before = layers.len();
                layers.retain(|l| l.get("paint").and_then(|p| p.get("type")).and_then(Value::as_str) != Some("pattern"));
                if layers.len() != before {
                    skipped.push(format!("{} (a pattern inside a pattern)", shape.source_id));
                }
            }
        }
    }
    let b = art.visual_bounds();
    let tile_box = [0.0, 0.0, def.tile.width, def.tile.height];
    let bounds = if b.is_empty() || ![b.min.x, b.min.y, b.max.x, b.max.y].iter().all(|v| v.is_finite()) {
        tile_box
    } else {
        // Miter joins reach past `visual_bounds`: pad by the widest stroke (a miter at the default limit of 4
        // reaches 2 half-widths). Over-including a copy only draws art the clip removes.
        let pad = art
            .shapes
            .iter()
            .filter_map(|s| crate::paint::top_stroke_of(&s.paint).map(|st| st.width * s.transform_scale))
            .filter(|w| w.is_finite() && *w > 0.0)
            .fold(0.0f64, f64::max)
            * 2.0;
        [b.min.x - pad, b.min.y - pad, b.max.x + pad, b.max.y + pad]
    };
    TileArt { art, bounds, skipped }
}

impl TileArt {
    /// One copy's shapes: path nodes (`x = y = 0`) whose points are the tile's mapped by `m` (tile space → the
    /// space they are drawn in) in f64, so neighbouring copies computed with the same map share their edges
    /// exactly. Stroke widths scale by `m`'s area scale. Ids are `<id_prefix><k>`.
    pub fn copy(&self, m: Affine, id_prefix: &str) -> Vec<crate::types::RawNode> {
        let scale = (m.a * m.d - m.b * m.c).abs().sqrt();
        let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        let mut placement = Placement::default();
        for (k, shape) in self.art.transformed(m).shapes.into_iter().enumerate() {
            placement.emit(
                format!("{id_prefix}{k}"),
                &shape.source_id,
                &shape.paint,
                shape.subpaths,
                shape.transform_scale * scale,
                None,
                PlacedAt { subpath: 0, s: 0.0, s_span: (0.0, 0.0) },
            );
        }
        placement.paths.iter().map(|p| p.to_node(p.id.clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn tile(v: Value) -> Tile {
        Tile::parse(&v).expect("tile")
    }

    #[test]
    fn parses_defaults_and_refuses_bad_sizes() {
        let t = tile(json!({"width": 10, "height": 20}));
        assert_eq!(t.kind, TileType::Grid);
        assert_eq!((t.spacing_x, t.spacing_y, t.brick_offset), (0.0, 0.0, 0.5));
        assert!(t.left_in_front && t.top_in_front);
        assert!(Tile::parse(&json!({"width": 0, "height": 20})).is_none());
        assert!(Tile::parse(&json!({"width": 10})).is_none());
        let t = tile(json!({"type": "brick-row", "width": 10, "height": 20, "brickOffset": 1.5,
            "overlap": {"x": "right", "y": "bottom"}}));
        assert_eq!(t.brick_offset, 0.5, "an offset outside (0, 1) falls back to a half");
        assert!(!t.left_in_front && !t.top_in_front);
    }

    #[test]
    fn origins_follow_each_tile_type() {
        let grid = tile(json!({"width": 10, "height": 20, "spacingX": 2, "spacingY": 4}));
        assert_eq!(grid.origin(1, 1), (12.0, 24.0));
        assert_eq!(grid.origin(-1, -2), (-12.0, -48.0));
        let brick = tile(json!({"type": "brick-row", "width": 10, "height": 10, "brickOffset": 0.5}));
        assert_eq!(brick.origin(0, 1), (5.0, 10.0));
        assert_eq!(brick.origin(0, 2), (0.0, 20.0));
        assert_eq!(brick.origin(0, -1), (5.0, -10.0));
        let third = tile(json!({"type": "brick-column", "width": 10, "height": 30, "brickOffset": 1.0 / 3.0}));
        assert!((third.origin(1, 0).1 - 10.0).abs() < 1e-9);
        assert!((third.origin(2, 0).1 - 20.0).abs() < 1e-9);
        let hexc = tile(json!({"type": "hex-column", "width": 20, "height": 10}));
        assert_eq!(hexc.pitch(), (15.0, 10.0));
        assert_eq!(hexc.origin(1, 0), (15.0, 5.0));
        assert_eq!(hexc.origin(2, 0), (30.0, 0.0));
        let hexr = tile(json!({"type": "hex-row", "width": 10, "height": 20}));
        assert_eq!(hexr.origin(0, 1), (5.0, 15.0));
    }

    #[test]
    fn periods_tile_the_lattice_by_translation() {
        for v in [
            json!({"type": "grid", "width": 10, "height": 7}),
            json!({"type": "brick-row", "width": 10, "height": 7, "brickOffset": 0.25}),
            json!({"type": "brick-column", "width": 10, "height": 7, "brickOffset": 0.2}),
            json!({"type": "hex-row", "width": 10, "height": 7, "spacingY": 1}),
            json!({"type": "hex-column", "width": 10, "height": 7, "spacingX": -1}),
        ] {
            let t = tile(v.clone());
            let (pw, ph) = t.period().expect("period");
            // Shifting by the period maps every copy origin onto another copy origin.
            let origins: Vec<(f64, f64)> = (-12..12).flat_map(|i| (-12..12).map(move |j| (i, j))).map(|(i, j)| t.origin(i, j)).collect();
            for &(x, y) in origins.iter().filter(|(x, y)| x.abs() < 20.0 && y.abs() < 20.0) {
                let hit = origins.iter().any(|&(u, v)| (u - x - pw).abs() < 1e-9 && (v - y).abs() < 1e-9);
                let hit_y = origins.iter().any(|&(u, v)| (u - x).abs() < 1e-9 && (v - y - ph).abs() < 1e-9);
                assert!(hit && hit_y, "{v}: ({x}, {y}) + period ({pw}, {ph})");
            }
        }
        let odd = tile(json!({"type": "brick-row", "width": 10, "height": 7, "brickOffset": 0.123456}));
        assert_eq!(odd.period(), None);
        assert_eq!(tile(json!({"width": 10, "height": 7, "spacingX": -10})).period(), None);
    }

    #[test]
    fn copies_cover_the_region_in_overlap_order() {
        let t = tile(json!({"width": 10, "height": 10}));
        let c = t.copies([0.0, 0.0, 20.0, 10.0], [0.0, 0.0, 10.0, 10.0], 100).unwrap();
        // Top in front: bottom rows first; left in front: right to left.
        assert_eq!(c, vec![(1, 0), (0, 0)]);
        let t = tile(json!({"width": 10, "height": 10, "overlap": {"x": "right", "y": "bottom"}}));
        let c = t.copies([0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 10.0, 10.0], 100).unwrap();
        assert_eq!(c, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        // Art past the tile box pulls in the neighbours it reaches.
        let c = t.copies([0.0, 0.0, 10.0, 10.0], [-2.0, 0.0, 12.0, 10.0], 100).unwrap();
        assert_eq!(c, vec![(-1, 0), (0, 0), (1, 0)]);
        assert!(matches!(t.copies([0.0, 0.0, 1000.0, 1000.0], [0.0, 0.0, 10.0, 10.0], 50), Err(CopiesError::TooMany(n)) if n > 50));
        assert_eq!(t.copies([0.0, 0.0, 80.0, 80.0], [0.0, 0.0, 10.0, 10.0], 50), Err(CopiesError::TooMany(51)));
        assert!(matches!(t.copies([0.0, 0.0, 1e9, 1e9], [0.0, 0.0, 10.0, 10.0], 50), Err(CopiesError::TooMany(_))));
        let flat = tile(json!({"width": 10, "height": 10, "spacingY": -10}));
        assert_eq!(flat.copies([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], 50), Err(CopiesError::NoPitch));
    }

    #[test]
    fn brick_copies_include_the_shifted_rows() {
        let t = tile(json!({"type": "brick-row", "width": 10, "height": 10, "brickOffset": 0.5, "overlap": {"x": "right", "y": "bottom"}}));
        let c = t.copies([0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 10.0, 10.0], 100).unwrap();
        let origins: Vec<(f64, f64)> = c.iter().map(|&(i, j)| t.origin(i, j)).collect();
        assert_eq!(origins, vec![(0.0, 0.0), (10.0, 0.0), (-5.0, 10.0), (5.0, 10.0), (15.0, 10.0)]);
    }

    #[test]
    fn tile_copies_map_the_art_and_drop_nested_patterns() {
        let def_json = json!({"id": "p", "tile": {"width": 10, "height": 10}, "artwork": {"nodes": [
            {"id": "a", "type": "rectangle", "x": 2, "y": 2, "width": 6, "height": 6,
             "fills": [{"id": "f", "paint": {"type": "solid", "color": "#ff0000"}}],
             "strokes": [{"id": "s", "paint": {"type": "solid", "color": "#000000"}, "width": 1}]},
            {"id": "b", "type": "rectangle", "x": 0, "y": 0, "width": 1, "height": 1,
             "fills": [{"id": "f", "paint": {"type": "pattern", "patternId": "p"}}]},
            {"id": "t", "type": "richtext", "x": 0, "y": 0, "width": 5, "height": 5}
        ]}});
        let def = parse_pattern(&def_json).unwrap();
        let geometry = crate::stroke_brush::artwork::default_geometry;
        let art = tile_art(&def, &geometry);
        assert!(art.skipped.iter().any(|s| s.contains("pattern inside a pattern")));
        assert!(art.skipped.iter().any(|s| s == "t"));
        // Bounds are in tile space, padded by the stroke.
        assert!(art.bounds[0] < 1.5 && art.bounds[2] > 8.5, "{:?}", art.bounds);
        let m = crate::transform::compose(
            crate::transform::from_trs(crate::types::Transform2D { x: 0.0, y: 0.0, rotation: 0.0, scale_x: 2.0, scale_y: 2.0 }),
            crate::transform::translate(10.0, 0.0),
        );
        let shapes = art.copy(m, "p#");
        assert_eq!(shapes.len(), 2);
        assert_eq!(shapes[0].id(), "p#0");
        let s = crate::paint::node_strokes(&shapes[0]);
        assert_eq!(s[0].width, 2.0, "stroke widths scale with the map");
        assert!(crate::paint::node_fills(&shapes[1]).is_empty(), "a nested pattern fill is dropped");
        let first = shapes[0].points().and_then(|p| p.first()).map(|p| p.json().into_owned()).unwrap();
        // The rectangle's outline starts at its top-right corner: ((2 + 6) + 10) × 2, 2 × 2.
        assert_eq!((first["x"].as_f64(), first["y"].as_f64()), (Some(36.0), Some(4.0)), "{first}");
    }

    #[test]
    fn node_pattern_ids_reads_every_paint_list() {
        let n = crate::types::RawNode::from_value(json!({"id": "n", "type": "rectangle",
            "fills": [{"id": "a", "paint": {"type": "pattern", "patternId": "p1"}}, {"id": "b", "paint": {"type": "solid", "color": "#fff"}}],
            "strokes": [{"id": "c", "width": 1, "paint": {"type": "pattern", "patternId": "p2"}}]})).unwrap();
        assert_eq!(node_pattern_ids(&n), vec!["p1".to_string(), "p2".to_string()]);
    }
}
