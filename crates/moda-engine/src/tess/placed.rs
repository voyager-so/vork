//! Art, scatter, pattern and bristle brushes drawn by the engine (brush plan
//! P8 follow-up to P17/P18): a path whose stroke names one of them draws its
//! body fill as usual and, instead of its stroke, the shapes that
//! `moda_scene::stroke_brush` places along it — each placed shape a path with
//! its own paint (colourised artwork, or a translucent bristle strand), drawn
//! like a group in the stroke node's slot.
//! One implementation for the WASM canvas and native moda-render.

use moda_scene::stroke_brush::art::ArtOptions;
use moda_scene::stroke_brush::common::PlacedAt;
use moda_scene::stroke_brush::{
    art_brush_for_node, bristle_brush_for_node, parse_art_brush, parse_bristle_brush,
    parse_pattern_brush, parse_scatter_brush,
    pattern_brush_for_node_with, scatter_brush_for_node, PatternOptions, PlacedPath,
};
use moda_scene::stroke_outline::{ArcWindow, SubpathSpan};

use crate::brush_artwork::brush_artwork_geometry;
use moda_scene::types::RawNode;
use moda_scene::value::{json, Value};

use super::outline::brush_id;

/// Which placed brush a node's stroke uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacedKind {
    Art,
    Scatter,
    Pattern,
    /// Strands generated from the brush's own parameters (no artwork).
    Bristle,
}

/// The placed (art / scatter / pattern / bristle) brush definition a path's
/// stroke names, if any.
pub fn placed_brush<'a>(
    scene: &'a moda_scene::Scene,
    node: &RawNode,
) -> Option<(PlacedKind, &'a Value)> {
    if node.node_type() != "path" {
        return None;
    }
    let def = scene.doc().brush(&brush_id(node)?)?;
    let kind = PlacedKind::from_brush_type(def.get("type").and_then(Value::as_str)?)?;
    Some((kind, def))
}

impl PlacedKind {
    /// The placed kind a brush definition's `type` names (`None`: not placed).
    pub fn from_brush_type(brush_type: &str) -> Option<Self> {
        match brush_type {
            "art" => Some(PlacedKind::Art),
            "scatter" => Some(PlacedKind::Scatter),
            "pattern" => Some(PlacedKind::Pattern),
            "bristle" => Some(PlacedKind::Bristle),
            _ => None,
        }
    }
}

/// Whether the zoom band changes this placement (art and pattern warps are
/// fitted to a tolerance; scatter copies are exact affine images of the
/// artwork; bristle strands are sampled in the brush's own units).
pub fn band_dependent(kind: PlacedKind) -> bool {
    matches!(kind, PlacedKind::Art | PlacedKind::Pattern)
}

/// Whether trim / dash windows are applied by the placer itself rather than
/// by filtering the full placement per frame ([`in_windows`]): art clips its
/// artwork in art space and pattern clips its tiles in leg space before
/// warping (a reveal); bristle clips its strands; scatter shows or hides
/// whole copies.
pub fn windows_in_placer(kind: PlacedKind) -> bool {
    matches!(
        kind,
        PlacedKind::Art | PlacedKind::Pattern | PlacedKind::Bristle
    )
}

/// Whether a scatter copy is drawn under trim / dash `windows`:
/// its anchor `at.s` lies in a window on its subpath (on a closed subpath a
/// window past the seam wraps). Copies and tiles are shown or hidden whole.
///
/// Windows are half-open, `[start, end)`: a copy exactly on a dash boundary
/// belongs to the window that starts there, never to both (copies now sit on
/// exact multiples of the step from the path's start, so boundaries are
/// common). A window that ends exactly at the subpath's end includes it, so the
/// copy centred on an open path's end still draws; a dash running through a
/// closed seam (`end > len`) keeps its end exclusive. `spans` are the
/// outliner's subpaths (`subpath_spans`).
pub fn in_windows(at: &PlacedAt, windows: &[ArcWindow], spans: &[SubpathSpan]) -> bool {
    let (len, closed) = spans
        .get(at.subpath)
        .map_or((0.0, false), |s| (s.length, s.closed));
    const EPS: f64 = 1e-9;
    // Placements past a closed seam (a pattern leg through the start, a
    // corner at the closing joint) carry `s ≥ len`: wrap them first. On an
    // open subpath `s = len` is the end, not the start: a copy there stays
    // (wrapping it to 0 would show it under a trim that hides the end).
    let s = if len > 0.0 && (closed || at.s < 0.0 || at.s > len + EPS) {
        at.s.rem_euclid(len)
    } else {
        at.s
    };
    windows.iter().filter(|w| w.subpath == at.subpath).any(|w| {
        // Inclusive only for a window ending exactly at the subpath's end.
        let to_end = len > 0.0 && (w.end - len).abs() <= EPS;
        let inside = |x: f64| x >= w.start && (x < w.end || (to_end && x <= w.end + EPS));
        inside(s) || (len > 0.0 && w.end > len && inside(s + len))
    })
}

/// The node's body without its strokes (fill, text and effects unchanged):
/// the brush replaces the stroke band.
pub fn without_stroke(node: &RawNode) -> RawNode {
    let mut m = node.fields().clone();
    moda_scene::paint::clear_strokes(&mut m);
    node.with_fields(m)
}


/// A placement as the engine draws it.
#[derive(Debug, Clone, Default)]
pub struct Placed {
    /// Synthetic path nodes (node-local coordinates, drawing order); ids
    /// `<node id>::brush::<i>`; paints are the artwork's own (colourised) —
    /// node opacity and blend are applied at draw time ([`with_node_paint`]),
    /// so paint-only edits of the stroke node never re-place.
    pub nodes: Vec<RawNode>,
    /// Each shape's geometry hash (precomputed: the plan never re-hashes the
    /// synthetic nodes).
    pub geom_hashes: Vec<u64>,
    /// `skipped`-class notes: artwork without geometry, placement warnings
    /// (e.g. `guides_ignored`), a refused (`too_complex`) placement.
    pub notes: Vec<String>,
    /// The placement was refused: draw the plain stroke instead.
    pub too_complex: bool,
    /// Where each shape sits along the stroke (for per-frame trim / dash).
    pub at: Vec<PlacedAt>,
    /// Whether every shape carries the first one's paint (`frame::strand_run_body`'s check, made once per
    /// placement rather than on every build of the stroke).
    pub one_paint: std::sync::OnceLock<bool>,
}

/// The stroke node a placed shape (`<node id>::brush::<i>`) belongs to; any other id as it is.
pub fn placed_owner_id(item_id: &str) -> &str {
    match item_id.rsplit_once("::brush::") {
        Some((owner, k)) if !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()) => owner,
        _ => item_id,
    }
}

/// Paint fields of the stroke node a placement depends on (colourisation
/// reads the stroke colour): part of the placement cache key.
pub fn placement_paint_hash(node: &RawNode) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    moda_scene::paint::with_top_stroke(node, |s| s.and_then(|s| moda_scene::value::to_string(&s.paint).ok()))
        .hash(&mut h);
    h.finish()
}

/// A placed shape with the stroke node's opacity (multiplied) and blend mode
/// (when the shape has none); borrowed as-is when there is nothing to apply.
pub fn with_node_paint<'a>(shape: &'a RawNode, node: &RawNode) -> std::borrow::Cow<'a, RawNode> {
    let opacity = node.f64("opacity").filter(|o| o.is_finite()).unwrap_or(1.0);
    let blend = node.json_field("blendMode").filter(|b| !b.is_null());
    let needs_blend = blend.is_some() && shape.json_field("blendMode").is_none();
    if opacity == 1.0 && !needs_blend {
        return std::borrow::Cow::Borrowed(shape);
    }
    let mut m = shape.fields().clone();
    let own = m.get("opacity").and_then(Value::as_f64).unwrap_or(1.0);
    m.insert("opacity".into(), json!(own * opacity));
    if needs_blend {
        m.insert("blendMode".into(), blend.cloned().unwrap_or(Value::Null));
    }
    std::borrow::Cow::Owned(shape.with_fields(m))
}

/// The node without its brush (a refused placement draws the plain stroke).
pub fn without_brush(node: &RawNode) -> RawNode {
    let mut m = node.fields().clone();
    moda_scene::paint::clear_stroke_brush(&mut m);
    node.with_fields(m)
}

/// Place the brush along `node` with the same artwork geometry as the WASM
/// `stroke_brush` export (`brush_artwork_geometry`), so canvas = export.
pub fn place(
    node: &RawNode,
    kind: PlacedKind,
    def: &Value,
    tolerance: f64,
    windows: Option<&[ArcWindow]>,
) -> Placed {
    let geometry = &brush_artwork_geometry;
    let placement = match kind {
        PlacedKind::Art => match parse_art_brush(def) {
            Some(b) => {
                art_brush_for_node(&node.to_json_map(), &b, geometry, &ArtOptions { tolerance, windows })
            }
            None => return Placed::default(),
        },
        PlacedKind::Scatter => match parse_scatter_brush(def) {
            Some(b) => scatter_brush_for_node(&node.to_json_map(), &b, geometry),
            None => return Placed::default(),
        },
        PlacedKind::Pattern => match parse_pattern_brush(def) {
            Some(b) => pattern_brush_for_node_with(
                &node.to_json_map(),
                &b,
                geometry,
                &PatternOptions { tolerance, windows },
            ),
            None => return Placed::default(),
        },
        PlacedKind::Bristle => match parse_bristle_brush(def) {
            Some(b) => bristle_brush_for_node(&node.to_json_map(), &b, windows),
            None => return Placed::default(),
        },
    };
    let mut notes: Vec<String> = placement
        .skipped
        .iter()
        .map(|id| format!("brush artwork {id} has no geometry"))
        .chain(placement.warnings.iter().cloned())
        .collect();
    if let Some(why) = &placement.too_complex {
        notes.push(format!(
            "brush placement too complex ({why}); drew the plain stroke"
        ));
        return Placed {
            notes,
            too_complex: true,
            ..Placed::default()
        };
    }
    let nodes: Vec<RawNode> = placement
        .paths
        .iter()
        .enumerate()
        .map(|(i, p)| synth(node, i, p))
        .collect();
    let geom_hashes = nodes
        .iter()
        .map(|n| crate::residency::node_hashes(n).geom)
        .collect();
    Placed {
        nodes,
        geom_hashes,
        notes,
        too_complex: false,
        at: placement.paths.iter().map(|p| p.at).collect(),
        one_paint: Default::default(),
    }
}

fn synth(node: &RawNode, i: usize, p: &PlacedPath) -> RawNode {
    p.to_node(format!("{}::brush::{i}", node.id()))
}

/// The z range placed shapes spread over inside the stroke node's slot: `[z, z + span)`.
pub const PLACED_Z_SPAN: f64 = 0.45;

thread_local! {
    static Z_SPAN: std::cell::Cell<f64> = const { std::cell::Cell::new(PLACED_Z_SPAN) };
}

/// Run `f` with placed shapes squeezed into `[z, z + span)`: a stroke layer of a paint stack (E3.1 P6) owns
/// only its slice of the node's slot, so its artwork stays under the layers above it.
pub fn with_z_span<R>(span: f64, f: impl FnOnce() -> R) -> R {
    let prev = Z_SPAN.with(|c| c.replace(span));
    let out = f();
    Z_SPAN.with(|c| c.set(prev));
    out
}

/// Fractional z of placed shape `k` of `n` inside the stroke node's slot:
/// above the body fill, below shape text / inner shadow (+0.5).
pub fn placed_z(z: f64, k: usize, n: usize) -> f64 {
    let span = Z_SPAN.with(std::cell::Cell::get);
    if span == PLACED_Z_SPAN {
        return z + 0.05 + 0.4 * k as f64 / n.max(1) as f64;
    }
    z + span * (0.05 + 0.4 * k as f64 / n.max(1) as f64) / PLACED_Z_SPAN
}

/// The keys a placed shape's node sets itself, after its paint (`PlacedPath::to_node`'s order).
const SHAPE_OWN: [&str; 10] = [
    "id", "type", "sourceId", "x", "y", "points", "closed", "subpath", "s", "sSpan",
];

/// A placement in flat arrays, for a host worker to hand back to the main thread (typed arrays transfer; the
/// shapes' paint, shared by most of them, goes once). [`placement_wire`] encodes one; [`placed_from_wire`] rebuilds
/// the same [`Placed`] (the same shape nodes, key for key, and the worker's geometry hashes).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlacementWire {
    /// JSON: `{ notes, tooComplex, paints: [paint map], sources: [source id] }`.
    pub meta: String,
    /// Per shape: its paint's and source id's index in `meta`, and whether it is closed (0 / 1).
    pub paint: Vec<u32>,
    pub source: Vec<u32>,
    pub closed: Vec<u8>,
    /// Per shape: `subpath, s, sSpan.0, sSpan.1` (its [`PlacedAt`]).
    pub at: Vec<f64>,
    /// Per shape: its point count; all points' kinds (`CompactKind`) and coordinates, shape after shape.
    pub counts: Vec<u32>,
    pub kinds: Vec<u8>,
    pub coords: Vec<f64>,
    /// Per shape: its geometry hash as two u32 (low, high).
    pub hashes: Vec<u32>,
}

/// [`PlacementWire`] of `placed` (the shapes `stroke_id`'s placement made), or `None` when a shape isn't in the
/// canonical form `PlacedPath::to_node` writes (compact points, its own keys after the paint).
pub fn placement_wire(stroke_id: &str, placed: &Placed) -> Option<PlacementWire> {
    let mut w = PlacementWire::default();
    let mut paints: Vec<moda_scene::value::Map> = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    if placed.nodes.len() != placed.geom_hashes.len() || placed.nodes.len() != placed.at.len() {
        return None;
    }
    for (i, n) in placed.nodes.iter().enumerate() {
        let f = n.fields();
        let pts = n.compact_points()?;
        // The own keys, in order, after the paint.
        let own_at = f.len().checked_sub(SHAPE_OWN.len())?;
        if !f.keys().skip(own_at).eq(SHAPE_OWN.iter().copied())
            || f.keys().take(own_at).any(|k| SHAPE_OWN.contains(&k.as_str()))
            || f.get("id").and_then(Value::as_str) != Some(format!("{stroke_id}::brush::{i}").as_str())
        {
            return None;
        }
        let paint: moda_scene::value::Map =
            f.iter().take(own_at).map(|(k, v)| (k.clone(), v.clone())).collect();
        let pi = match paints.iter().position(|p| *p == paint) {
            Some(k) => k,
            None => {
                paints.push(paint);
                paints.len() - 1
            }
        };
        let source = f.get("sourceId")?.as_str()?;
        let si = match sources.iter().position(|s| s == source) {
            Some(k) => k,
            None => {
                sources.push(source.to_string());
                sources.len() - 1
            }
        };
        w.paint.push(pi as u32);
        w.source.push(si as u32);
        w.closed.push(u8::from(f.get("closed")?.as_bool()?));
        let at = &placed.at[i];
        w.at.extend([at.subpath as f64, at.s, at.s_span.0, at.s_span.1]);
        w.counts.push(pts.len() as u32);
        for k in 0..pts.len() {
            let (kind, c) = pts.point(k);
            w.kinds.push(kind as u8);
            w.coords.extend_from_slice(c);
        }
        let h = placed.geom_hashes[i];
        w.hashes.extend([h as u32, (h >> 32) as u32]);
    }
    w.meta = json!({
        "notes": placed.notes,
        "tooComplex": placed.too_complex,
        "paints": paints,
        "sources": sources,
    })
    .to_string();
    Some(w)
}

/// The [`Placed`] a [`PlacementWire`] carries for stroke `stroke_id` (`None`: malformed).
pub fn placed_from_wire(stroke_id: &str, w: &PlacementWire) -> Option<Placed> {
    use moda_scene::path_points::{CompactKind, CompactPoints};
    let meta: Value = moda_scene::value::from_str(&w.meta).ok()?;
    let paints: Vec<&moda_scene::value::Map> = meta
        .get("paints")?
        .as_array()?
        .iter()
        .map(Value::as_object)
        .collect::<Option<_>>()?;
    let sources: Vec<&str> = meta
        .get("sources")?
        .as_array()?
        .iter()
        .map(Value::as_str)
        .collect::<Option<_>>()?;
    let notes: Vec<String> = meta
        .get("notes")?
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect::<Option<_>>()?;
    let n = w.counts.len();
    if w.paint.len() != n
        || w.source.len() != n
        || w.closed.len() != n
        || w.at.len() != 4 * n
        || w.hashes.len() != 2 * n
    {
        return None;
    }
    let num = |v: f64| moda_scene::feed_points::js_number(if v.is_finite() { v } else { 0.0 });
    let (mut kind_at, mut coord_at) = (0usize, 0usize);
    let mut nodes = Vec::with_capacity(n);
    let mut at = Vec::with_capacity(n);
    for i in 0..n {
        let count = w.counts[i] as usize;
        let kinds: Vec<CompactKind> = w
            .kinds
            .get(kind_at..kind_at + count)?
            .iter()
            .map(|&k| CompactKind::try_from_u8(k))
            .collect::<Option<_>>()?;
        kind_at += count;
        let ncoords: usize = kinds.iter().map(|k| k.coords()).sum();
        let points = CompactPoints::from_js_coords(&kinds, w.coords.get(coord_at..coord_at + ncoords)?)?;
        coord_at += ncoords;
        let placed_at = PlacedAt {
            subpath: w.at[4 * i] as usize,
            s: w.at[4 * i + 1],
            s_span: (w.at[4 * i + 2], w.at[4 * i + 3]),
        };
        // As `PlacedPath::to_node`: the paint, then the shape's own keys.
        let paint = paints.get(w.paint[i] as usize)?;
        let mut m = moda_scene::value::Map::with_capacity(paint.len() + SHAPE_OWN.len());
        for (k, v) in paint.iter() {
            m.insert(k.clone(), v.clone());
        }
        m.insert("id".into(), Value::String(format!("{stroke_id}::brush::{i}")));
        m.insert("type".into(), Value::String("path".into()));
        m.insert("sourceId".into(), Value::String((*sources.get(w.source[i] as usize)?).to_string()));
        m.insert("x".into(), Value::from(0u64));
        m.insert("y".into(), Value::from(0u64));
        m.insert("points".into(), Value::Null);
        m.insert("closed".into(), Value::Bool(w.closed[i] != 0));
        m.insert("subpath".into(), Value::from(placed_at.subpath as u64));
        m.insert("s".into(), num(placed_at.s));
        m.insert(
            "sSpan".into(),
            Value::Array(vec![num(placed_at.s_span.0), num(placed_at.s_span.1)]),
        );
        nodes.push(RawNode::with_compact_points(m, points));
        at.push(placed_at);
    }
    if kind_at != w.kinds.len() || coord_at != w.coords.len() {
        return None;
    }
    Some(Placed {
        nodes,
        geom_hashes: w
            .hashes
            .chunks_exact(2)
            .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
            .collect(),
        notes,
        too_complex: meta.get("tooComplex")?.as_bool()?,
        at,
        one_paint: Default::default(),
    })
}

/// The one body a bristle stroke's strands draw as (`frame::strand_run_body`): every strand's triangles, at the
/// current zoom band. `None` when a strand's tessellation refused its fill or carries stroke UVs (the strands
/// then draw one by one).
pub fn strand_run_geometry(placed: &Placed) -> Option<crate::seams::TessellatedGeometry> {
    use super::moda_engine_path_types as pt;
    let mut all = crate::seams::TessellatedGeometry {
        positions: Vec::new(),
        vertex_types: Vec::new(),
        vertex_count: 0,
        bounds: [0.0; 4],
        fill_dropped: None,
        stroke_dropped: None,
        stroke_uv: Vec::new(),
    };
    // `[x0, y0, x1, y1]` while accumulating; geometry bounds are `[x, y, w, h]`.
    let mut b = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    for shape in &placed.nodes {
        let style = moda_scene::paint::node_stroke_style(shape);
        let g = super::with_miter_limit(shape.miter_limit(), || {
            super::tessellate_combined_with_dash_trim(
                &super::path_cmds_for_node(shape),
                0.0,
                shape.bool("closed").unwrap_or(false),
                None,
                None,
                0.0,
                pt::line_cap_from_str(style.cap.as_deref()),
                pt::line_join_from_str(style.join.as_deref()),
                pt::fill_rule_from_str(shape.str("fillRule")),
                pt::stroke_align_from_str(style.align.as_deref()),
            )
        });
        if g.fill_dropped.is_some() || g.stroke_dropped.is_some() || !g.stroke_uv.is_empty() {
            return None;
        }
        if g.vertex_count == 0 {
            continue;
        }
        all.positions.extend_from_slice(&g.positions);
        all.vertex_types.extend_from_slice(&g.vertex_types);
        all.vertex_count += g.vertex_count;
        let [x, y, w, h] = g.bounds;
        b = [b[0].min(x), b[1].min(y), b[2].max(x + w), b[3].max(y + h)];
    }
    if b[0] <= b[2] {
        all.bounds = [b[0], b[1], b[2] - b[0], b[3] - b[1]];
    }
    Some(all)
}

/// The outline-cache `base` of a stroke's strand body: the same strands at any zoom band (they refine per band
/// like outlines).
pub fn strand_run_base(placed: &Placed) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    placed.geom_hashes.hash(&mut h);
    0x57A4_u16.hash(&mut h);
    h.finish()
}

/// The tess-cache key of a stroke's strand body ([`strand_run_geometry`]) at zoom band `band`.
pub fn strand_run_key(placed: &Placed, band: i32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    placed.geom_hashes.hash(&mut h);
    band.hash(&mut h);
    h.finish()
}

/// A host worker's placement of one bristle or art stroke (`WasmEngine::take_outline_requests`, `task: "place"`):
/// the placement as a [`PlacementWire`], and its strands' one body at the request's band (`None`: they draw one by
/// one). Pure: no engine instance.
pub struct PlacementJob {
    pub wire: PlacementWire,
    pub strands: Option<crate::seams::TessellatedGeometry>,
}

/// Whether a worker can place a `kind` stroke as the main thread does (`frame::build_placed_brush_body`): bristle
/// and art, without trim / dash (`windowed`).
pub fn worker_placeable(kind: PlacedKind, windowed: bool) -> bool {
    matches!(kind, PlacedKind::Bristle | PlacedKind::Art) && !windowed
}

/// Run a `task: "place"` request (`{ id, key, base, band, kind, node, brush }`,
/// [`crate::residency::PlanCaches::placement_request`]): place the stroke as the main thread would (a band-dependent
/// kind, art, at the band's tolerance: its warp refits cubics to it), and tessellate its strand body at `band` when
/// its shapes share one paint (`frame::strand_run_body` draws them as one body then). `kind` defaults to bristle.
pub fn placement_job(req: &Value) -> Result<PlacementJob, String> {
    let node = req
        .get("node")
        .and_then(Value::as_object)
        .map(|m| RawNode::new(m.clone()))
        .ok_or("no node")?;
    let def = req.get("brush").filter(|b| b.is_object()).ok_or("no brush")?;
    let band = req.get("band").and_then(Value::as_i64).unwrap_or(0) as i32;
    let kind = match req.get("kind").and_then(Value::as_str) {
        None => PlacedKind::Bristle,
        Some(k) => PlacedKind::from_brush_type(k).filter(|&k| worker_placeable(k, false)).ok_or("not a worker kind")?,
    };
    let tolerance = super::outline::tolerance_for_band(if band_dependent(kind) { band } else { 0 });
    let placed = place(&node, kind, def, tolerance, None);
    let wire = placement_wire(node.id(), &placed).ok_or("placement not in wire form")?;
    // One paint (the wire's shapes all share its first) and no stroke band: what `strand_run_body` draws as one body.
    let one_body = placed.nodes.len() >= 2
        && wire.paint.iter().all(|&p| p == 0)
        && crate::paint::painted_stroke_width(&placed.nodes[0]) == 0.0;
    let strands = if one_body {
        let _band = super::outline::enter_camera(Some(2f64.powi(band)));
        strand_run_geometry(&placed)
    } else {
        None
    };
    Ok(PlacementJob { wire, strands })
}

/// Run a `task: "strands"` request (`{ id, key, base, band, node, brush }`,
/// `crate::residency::PlanCaches::strands_request`): place the bristle stroke as the main thread did and tessellate
/// its strands as one body at `band` ([`strand_run_geometry`]). An error when the placement's strand key at `band`
/// isn't the request's `key` (the main thread placed something else: a trim, a dash, another brush) or the strands
/// don't draw as one body. Pure: no engine instance.
pub fn strands_job(req: &Value) -> Result<crate::seams::TessellatedGeometry, String> {
    let node = req
        .get("node")
        .and_then(Value::as_object)
        .map(|m| RawNode::new(m.clone()))
        .ok_or("no node")?;
    let def = req.get("brush").filter(|b| b.is_object()).ok_or("no brush")?;
    let band = req.get("band").and_then(Value::as_i64).unwrap_or(0) as i32;
    let key = req
        .get("key")
        .and_then(Value::as_str)
        .and_then(|k| k.parse::<u64>().ok())
        .ok_or("no key")?;
    let placed = place(&node, PlacedKind::Bristle, def, super::outline::tolerance_for_band(0), None);
    if placed.too_complex || placed.nodes.len() < 2 || strand_run_key(&placed, band) != key {
        return Err("not this placement's strands".into());
    }
    let _band = super::outline::enter_camera(Some(2f64.powi(band)));
    strand_run_geometry(&placed).ok_or_else(|| "strands don't draw as one body".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A worker's placement crosses to the main thread as flat arrays and comes back as the same `Placed`: every
    /// shape node key for key (paint, own keys, compact points), the geometry hashes (which re-hash to the same),
    /// the anchors and the notes.
    #[test]
    fn a_bristle_placement_round_trips_through_the_wire() {
        let node = RawNode::new(
            json!({"id": "b", "type": "path", "x": 10.0, "y": 10.0, "width": 300.0, "height": 40.0,
                   "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#223344"}, "width": 6,
                                "brush": {"brushId": "round", "seed": 5}}],
                   "points": [{"type": "move", "x": 0, "y": 0},
                              {"type": "curve", "x": 300, "y": 20, "cp1": {"x": 100, "y": -40}, "cp2": {"x": 200, "y": 60}},
                              {"type": "line", "x": 280, "y": 40}]})
            .as_object()
            .unwrap()
            .clone(),
        );
        let def = json!({"id": "round", "name": "Round", "type": "bristle", "shape": "flat-fan", "size": 3,
                         "length": 1.5, "density": 0.5, "thickness": 0.5, "opacity": 0.8, "stiffness": 0.4});
        let placed = place(&node, PlacedKind::Bristle, &def, super::super::outline::tolerance_for_band(0), None);
        assert!(placed.nodes.len() > 4, "strands placed");
        let wire = placement_wire("b", &placed).expect("wire form");
        assert_eq!(wire.meta.matches("\"fills\"").count(), 1, "one shared paint: {}", wire.meta);
        let back = placed_from_wire("b", &wire).expect("decoded");
        assert_eq!(back.nodes.len(), placed.nodes.len());
        for (a, b) in placed.nodes.iter().zip(&back.nodes) {
            assert_eq!(a.to_json_map(), b.to_json_map());
            assert_eq!(
                a.fields().keys().collect::<Vec<_>>(),
                b.fields().keys().collect::<Vec<_>>(),
                "key order"
            );
            assert_eq!(
                crate::residency::node_hashes(a).geom,
                crate::residency::node_hashes(b).geom
            );
        }
        assert_eq!(back.geom_hashes, placed.geom_hashes);
        assert_eq!(format!("{:?}", back.at), format!("{:?}", placed.at));
        assert_eq!(back.notes, placed.notes);
        assert_eq!(back.too_complex, placed.too_complex);
        // And the worker's job: the same wire, and the strand body the main thread would tessellate.
        let job = placement_job(&json!({"id": "b", "key": "1", "base": "2", "band": 0,
                                        "node": Value::Object(node.to_json_map()), "brush": def}))
        .expect("job");
        assert_eq!(job.wire, wire);
        let main = strand_run_geometry(&placed).expect("strand body");
        let got = job.strands.expect("strand body");
        assert_eq!(got.positions, main.positions);
        assert_eq!(got.vertex_types, main.vertex_types);
        assert!(placed_from_wire("b", &PlacementWire { counts: vec![1], ..wire.clone() }).is_none(), "malformed");
    }

    #[test]
    fn windows_are_half_open() {
        let at = |s: f64| PlacedAt {
            subpath: 0,
            s,
            s_span: (s - 1.0, s + 1.0),
        };
        let w = |start: f64, end: f64| ArcWindow {
            subpath: 0,
            start,
            end,
            taper_extent: None,
        };
        let dashes = [w(0.0, 10.0), w(20.0, 30.0), w(40.0, 50.0)];
        let span = |length: f64, closed: bool| SubpathSpan { length, closed };
        let lens = [span(50.0, false)];
        let ring = [span(50.0, true)];
        // A copy exactly on a boundary belongs to the window starting there.
        assert!(in_windows(&at(0.0), &dashes, &lens));
        assert!(!in_windows(&at(10.0), &dashes, &lens), "end is exclusive");
        assert!(in_windows(&at(20.0), &dashes, &lens), "start is inclusive");
        assert!(!in_windows(&at(30.0), &dashes, &lens));
        // A window ending at the subpath's end keeps the copy centred on it.
        assert!(in_windows(&at(50.0), &dashes, &lens));
        // …but the end copy is not the start copy: a trim that hides the end hides it.
        assert!(!in_windows(&at(50.0), &[w(0.0, 25.0)], &lens));
        // A dash through a closed seam ([45, 55) on a 50-long ring) keeps its
        // end exclusive: the copy at 5 (= 55 wrapped) is outside, 4 inside.
        let seam = [w(45.0, 55.0)];
        assert!(in_windows(&at(4.0), &seam, &ring));
        assert!(
            !in_windows(&at(5.0), &seam, &ring),
            "wrapped end is exclusive"
        );
        assert!(in_windows(&at(45.0), &seam, &ring));
        // Other subpaths are not matched.
        assert!(!in_windows(
            &PlacedAt {
                subpath: 1,
                ..at(5.0)
            },
            &dashes,
            &lens
        ));
    }

    #[test]
    fn a_placed_shape_belongs_to_its_stroke() {
        assert_eq!(placed_owner_id("s::brush::12"), "s");
        assert_eq!(placed_owner_id("n::paint::1::brush::0"), "n::paint::1");
        assert_eq!(placed_owner_id("s::brush::x"), "s::brush::x");
        assert_eq!(placed_owner_id("plain"), "plain");
    }

    #[test]
    fn scatter_places_colourised_copies() {
        let node = RawNode::from_value(json!({
            "id": "s", "type": "path", "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#ff0000"}, "width": 1, "brush": {"brushId": "dots", "seed": 3}}], "opacity": 0.5,
            "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 200, "y": 0}]
        }))
        .unwrap();
        let def = json!({"id": "dots", "name": "Dots", "type": "scatter",
            "artwork": {"nodes": [{"id": "d", "type": "ellipse", "x": 0, "y": 0, "width": 10, "height": 10, "fills": [{"id": "fill-0", "paint": {"type": "solid", "color": "#000000"}}]}]},
            "size": {"value": 1}, "spacing": {"value": 1}, "scatter": {"value": 0}, "rotation": {"value": 0},
            "rotationRelativeTo": "path", "colorize": {"method": "tints"}});
        let placed = place(&node, PlacedKind::Scatter, &def, 0.2, None);
        assert!(placed.notes.is_empty(), "{:?}", placed.notes);
        let nodes = placed.nodes;
        assert!(nodes.len() >= 5, "{}", nodes.len());
        assert_eq!(nodes[0].id(), "s::brush::0");
        assert_eq!(
            nodes[0].f64("opacity"),
            None,
            "node opacity is applied at draw time"
        );
        assert_eq!(with_node_paint(&nodes[0], &node).f64("opacity"), Some(0.5));
        assert!(placed_z(3.0, 0, 5) > 3.0 && placed_z(3.0, 4, 5) < 3.5);
    }

    #[test]
    fn bristle_places_translucent_strands() {
        let node = RawNode::from_value(json!({
            "id": "b", "type": "path", "strokes": [{"id": "stroke-0", "paint": {"type": "solid", "color": "#112233"}, "width": 2, "opacity": 0.5, "brush": {"brushId": "round", "seed": 1}}],
            "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 200, "y": 0}]
        }))
        .unwrap();
        let def = json!({"id": "round", "name": "Round", "type": "bristle", "shape": "round-point", "size": 3,
            "length": 1, "density": 0.33, "thickness": 0.5, "opacity": 0.75, "stiffness": 0.5});
        assert!(!band_dependent(PlacedKind::Bristle) && windows_in_placer(PlacedKind::Bristle));
        let placed = place(&node, PlacedKind::Bristle, &def, 0.2, None);
        assert!(placed.notes.is_empty(), "{:?}", placed.notes);
        assert_eq!(placed.nodes.len(), 12);
        let n = &placed.nodes[0];
        assert_eq!(n.node_type(), "path");
        // Strand opacity × the stroke layer's own opacity.
        let a = 1.0 - 0.25f64.powf(1.0 / 12.0);
        assert!((n.f64("opacity").unwrap() - a * 0.5).abs() < 1e-9);
        let trimmed = place(
            &node,
            PlacedKind::Bristle,
            &def,
            0.2,
            Some(&[ArcWindow { subpath: 0, start: 0.0, end: 50.0, taper_extent: None }]),
        );
        assert_eq!(trimmed.nodes.len(), 12);
        assert!(trimmed.at.iter().all(|at| at.s_span.1 <= 60.0));
    }

    #[test]
    fn window_membership_wraps_the_closed_seam() {
        let at = |s: f64| PlacedAt {
            subpath: 0,
            s,
            s_span: (s - 5.0, s + 5.0),
        };
        let w = |a: f64, b: f64| ArcWindow {
            subpath: 0,
            start: a,
            end: b,
            taper_extent: None,
        };
        let lens = [SubpathSpan {
            length: 800.0,
            closed: true,
        }];
        // A tile past the seam (s = 880 ≡ 80) is inside [0, 120].
        assert!(in_windows(&at(880.0), &[w(0.0, 120.0)], &lens));
        assert!(!in_windows(&at(880.0), &[w(100.0, 200.0)], &lens));
        // A corner at the closing joint (s = len ≡ 0).
        assert!(in_windows(&at(800.0), &[w(0.0, 10.0)], &lens));
        // A window wrapping the seam [700, 900] keeps s = 50 and s = 750.
        assert!(in_windows(&at(50.0), &[w(700.0, 900.0)], &lens));
        assert!(in_windows(&at(750.0), &[w(700.0, 900.0)], &lens));
        assert!(!in_windows(&at(400.0), &[w(700.0, 900.0)], &lens));
        // Other subpaths never match.
        assert!(!in_windows(
            &PlacedAt {
                subpath: 1,
                s: 50.0,
                s_span: (45.0, 55.0)
            },
            &[w(0.0, 120.0)],
            &[
                lens[0],
                SubpathSpan {
                    length: 300.0,
                    closed: true
                }
            ]
        ));
    }
}
