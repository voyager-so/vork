//! Repeat objects (Illustrator's Object ▸ Repeat ▸ Radial / Grid / Mirror): a group whose `repeat` field draws its
//! children (the SOURCE) several times, each copy under its own transform in the group's local space.
//!
//! - `{"type": "radial", "count", "radius", "startAngle"?, "reverseOverlap"?, "hidden"?}`: `count` copies around a
//!   circle. The source stays where it is and sits on the circle at `startAngle` degrees from its centre (−90, the
//!   default, puts the source at the top: the centre is `radius` below the source's centre); copy `i` is the source
//!   rotated `i × 360 / count` degrees about that centre (clockwise on screen). Copies stack in index order, the
//!   last on top; `reverseOverlap` stacks them the other way. `hidden` lists copies not drawn.
//! - `{"type": "grid", "rows", "columns", "spacingX"?, "spacingY"?, "pattern"?, "offset"?, "flipRow"?,
//!   "flipColumn"?, "hidden"?}`: a rows × columns grid of tiles the size of the source's box plus the spacing (a
//!   gap; negative overlaps). `pattern` `brick-row` shifts every other row by `offset` (a fraction of a tile, default
//!   ½), `brick-column` every other column; `flipRow` / `flipColumn` (`horizontal`, `vertical` or `both`) reflect
//!   every other row / column about the tile's centre, as Illustrator's grid types. Copies are row-major.
//! - `{"type": "mirror", "axis": {"x", "y", "angle"}}`: the source and its reflection across the line through
//!   (`x`, `y`) at `angle` degrees (90 = vertical), in the group's local space.
//!
//! The SOURCE BOX is the geometric bounds of the group's visible children in its local space, leaf by leaf (each
//! leaf's own box: a path's point bounds, a line's end points, a shape's width × height; frames with their content
//! unless they clip it; booleans as their operands), the same rule as the editor's `core/scene/repeat.ts`, so the
//! two place copies identically. Radial and grid placements follow the source: edit the source and every copy
//! follows, as Illustrator's repeats do.
//!
//! **How it draws.** Like an envelope ([`crate::envelope`]), a repeat resolves at scene load (after booleans, live
//! effects and the envelopes inside it; innermost live group first) into TRANSIENT nodes drawn in place of its
//! children ([`crate::scene::Scene::visit_z`]): per copy, a transient group `<repeat>~r<i>` carrying the copy's
//! transform, holding a transient copy of every drawn node of the source (`<repeat>~r<i>~<id>`) with its own
//! transform, paint, effects, clip masks and opacity. A boolean copies as its result, a table or chart as its
//! drawn parts, an inner envelope or repeat as what it draws. Every lane (canvas, PNG export, masks, isolation)
//! draws them like any other node, and each copy's shapes key the tessellation cache by their source's id
//! ([`REPEAT_SOURCE_KEY`]), so the source is tessellated once however many copies draw it.
//!
//! This is the instance lane of the planned Distribute primitive (docs/design/live-nodes.md §7.5, "v1: the engine
//! expands each instance to a transient copy of the source subtree under a transform"): the spec is Distribute's
//! placement for the radial and grid layouts, plus a reflect layout.

use crate::value::{json, Map, Value};

use crate::envelope::Resolved;
use crate::scene::Scene;
use crate::transform;
use crate::types::{Affine, NodeId, RawNode};

/// The group field that makes a group a repeat.
pub const REPEAT_KEY: &str = "repeat";
/// On every transient a repeat makes: the repeat's id.
pub const REPEAT_OF_KEY: &str = "_repeatOf";
/// On every transient a repeat makes: the id of the node it copies (the original source node, through nested
/// repeats), which its shapes key the tessellation cache by.
pub const REPEAT_SOURCE_KEY: &str = "_repeatSource";

/// Most copies one repeat draws.
pub const MAX_REPEAT_INSTANCES: usize = 10_000;
/// Most transient nodes one repeat makes (copies × drawn source nodes); past it, the last copies are left out.
pub const MAX_REPEAT_NODES: usize = 20_000;
/// Most radial copies (Illustrator's Radial Repeat takes up to 360 here too).
pub const MAX_RADIAL_COUNT: usize = 360;
/// Most grid rows or columns.
pub const MAX_GRID_DIVISIONS: usize = 1_000;

/// Reflect every other row / column of a grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flip {
    None,
    Horizontal,
    Vertical,
    Both,
}

impl Flip {
    fn parse(v: Option<&Value>) -> Flip {
        match v.and_then(Value::as_str) {
            Some("horizontal") => Flip::Horizontal,
            Some("vertical") => Flip::Vertical,
            Some("both") => Flip::Both,
            _ => Flip::None,
        }
    }

    fn scale(self) -> (f64, f64) {
        match self {
            Flip::None => (1.0, 1.0),
            Flip::Horizontal => (-1.0, 1.0),
            Flip::Vertical => (1.0, -1.0),
            Flip::Both => (-1.0, -1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPattern {
    Grid,
    BrickRow,
    BrickColumn,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RepeatSpec {
    Radial { count: usize, radius: f64, start_angle: f64, reverse_overlap: bool, hidden: Vec<usize> },
    Grid {
        rows: usize,
        columns: usize,
        spacing_x: f64,
        spacing_y: f64,
        pattern: GridPattern,
        offset: f64,
        flip_row: Flip,
        flip_column: Flip,
        hidden: Vec<usize>,
    },
    Mirror { x: f64, y: f64, angle: f64 },
}

fn num(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_f64).filter(|x| x.is_finite())
}

fn whole(v: Option<&Value>) -> Option<usize> {
    let x = num(v)?;
    (x >= 1.0 && x.fract() == 0.0).then_some(x as usize)
}

fn hidden_of(v: Option<&Value>) -> Vec<usize> {
    let mut out: Vec<usize> = v
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_f64()).filter(|x| *x >= 0.0 && x.fract() == 0.0).map(|x| x as usize).collect())
        .unwrap_or_default();
    out.sort_unstable();
    out.dedup();
    out
}

/// Parse a `repeat` spec (see the module docs); `None` when it isn't a valid one (the group then draws as a plain
/// group).
pub fn parse_spec(v: &Value) -> Option<RepeatSpec> {
    match v.get("type")?.as_str()? {
        "radial" => {
            let count = whole(v.get("count"))?;
            if count > MAX_RADIAL_COUNT {
                return None;
            }
            let radius = num(v.get("radius"))?;
            if radius < 0.0 {
                return None;
            }
            Some(RepeatSpec::Radial {
                count,
                radius,
                start_angle: num(v.get("startAngle")).unwrap_or(-90.0),
                reverse_overlap: v.get("reverseOverlap").and_then(Value::as_bool).unwrap_or(false),
                hidden: hidden_of(v.get("hidden")),
            })
        }
        "grid" => {
            let rows = whole(v.get("rows"))?;
            let columns = whole(v.get("columns"))?;
            if rows > MAX_GRID_DIVISIONS || columns > MAX_GRID_DIVISIONS {
                return None;
            }
            let pattern = match v.get("pattern").and_then(Value::as_str) {
                Some("brick-row") => GridPattern::BrickRow,
                Some("brick-column") => GridPattern::BrickColumn,
                _ => GridPattern::Grid,
            };
            Some(RepeatSpec::Grid {
                rows,
                columns,
                spacing_x: num(v.get("spacingX")).unwrap_or(0.0),
                spacing_y: num(v.get("spacingY")).unwrap_or(0.0),
                pattern,
                offset: num(v.get("offset")).unwrap_or(0.5),
                flip_row: Flip::parse(v.get("flipRow")),
                flip_column: Flip::parse(v.get("flipColumn")),
                hidden: hidden_of(v.get("hidden")),
            })
        }
        "mirror" => {
            let axis = v.get("axis")?;
            Some(RepeatSpec::Mirror { x: num(axis.get("x"))?, y: num(axis.get("y"))?, angle: num(axis.get("angle")).unwrap_or(90.0) })
        }
        _ => None,
    }
}

/// Whether a node is a repeat: a group with a valid `repeat` spec.
pub fn is_repeat(node: &RawNode) -> bool {
    node.node_type() == "group" && node.json_field(REPEAT_KEY).and_then(parse_spec).is_some()
}

/// An axis-aligned box, `[min_x, min_y, max_x, max_y]`.
pub type Box4 = [f64; 4];

fn rotation_about(deg: f64, cx: f64, cy: f64) -> Affine {
    let r = deg.to_radians();
    let (s, c) = r.sin_cos();
    Affine { a: c, b: s, c: -s, d: c, e: cx - (c * cx - s * cy), f: cy - (s * cx + c * cy) }
}

fn scale_about(sx: f64, sy: f64, cx: f64, cy: f64) -> Affine {
    Affine { a: sx, b: 0.0, c: 0.0, d: sy, e: cx - sx * cx, f: cy - sy * cy }
}

/// The reflection across the line through (`x`, `y`) at `angle` degrees.
pub fn reflection(x: f64, y: f64, angle: f64) -> Affine {
    let t = (2.0 * angle).to_radians();
    let (s, c) = t.sin_cos();
    Affine { a: c, b: s, c: s, d: -c, e: x - (c * x + s * y), f: y - (s * x - c * y) }
}

/// The centre a radial repeat turns about, group-local.
pub fn radial_center(radius: f64, start_angle: f64, b: Box4) -> (f64, f64) {
    let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
    let a = start_angle.to_radians();
    (cx - radius * a.cos(), cy - radius * a.sin())
}

/// Every copy a repeat draws, as `(index, transform)` in drawing order (bottom first), hidden copies left out.
/// Transforms are group-local (source space → group space). `b` is the source box; `None` (an empty source)
/// draws nothing for radial and grid repeats.
pub fn instance_transforms(spec: &RepeatSpec, b: Option<Box4>) -> Vec<(usize, Affine)> {
    match spec {
        RepeatSpec::Radial { count, radius, start_angle, reverse_overlap, hidden } => {
            let Some(b) = b else { return Vec::new() };
            let (cx, cy) = radial_center(*radius, *start_angle, b);
            let step = 360.0 / *count as f64;
            let mut out: Vec<(usize, Affine)> = (0..*count)
                .filter(|i| !hidden.contains(i))
                .map(|i| (i, if i == 0 { transform::identity() } else { rotation_about(step * i as f64, cx, cy) }))
                .collect();
            if *reverse_overlap {
                out.reverse();
            }
            out
        }
        RepeatSpec::Grid { rows, columns, spacing_x, spacing_y, pattern, offset, flip_row, flip_column, hidden } => {
            let Some(b) = b else { return Vec::new() };
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            let (px, py) = (w + spacing_x, h + spacing_y);
            let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
            let mut out = Vec::new();
            for r in 0..*rows {
                for c in 0..*columns {
                    let i = r * columns + c;
                    if hidden.contains(&i) || out.len() >= MAX_REPEAT_INSTANCES {
                        continue;
                    }
                    let mut tx = c as f64 * px;
                    let mut ty = r as f64 * py;
                    match pattern {
                        GridPattern::BrickRow if r % 2 == 1 => tx += offset * px,
                        GridPattern::BrickColumn if c % 2 == 1 => ty += offset * py,
                        _ => {}
                    }
                    let (mut sx, mut sy) = (1.0, 1.0);
                    if r % 2 == 1 {
                        let (fx, fy) = flip_row.scale();
                        sx *= fx;
                        sy *= fy;
                    }
                    if c % 2 == 1 {
                        let (fx, fy) = flip_column.scale();
                        sx *= fx;
                        sy *= fy;
                    }
                    let a = transform::compose(transform::translate(tx, ty), scale_about(sx, sy, cx, cy));
                    out.push((i, if i == 0 { transform::identity() } else { a }));
                }
            }
            out
        }
        RepeatSpec::Mirror { x, y, angle } => vec![(0, transform::identity()), (1, reflection(*x, *y, *angle))],
    }
}

/// A group transform (`x`, `y`, `rotation`, `scaleX`, `scaleY`) equal to `a`, which is a rotation, a reflection
/// and a translation (every copy transform is).
pub fn trs_of(a: &Affine) -> (f64, f64, f64, f64, f64) {
    let sx = a.a.hypot(a.b);
    let det = a.a * a.d - a.b * a.c;
    let sy = if sx > 0.0 { det / sx } else { 1.0 };
    let rot = a.b.atan2(a.a).to_degrees();
    (a.e, a.f, rot, if sx > 0.0 { sx } else { 1.0 }, if sy != 0.0 { sy } else { 1.0 })
}

fn invert(a: &Affine) -> Option<Affine> {
    let det = a.a * a.d - a.b * a.c;
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    let (ia, ib, ic, id) = (a.d / det, -a.b / det, -a.c / det, a.a / det);
    Some(Affine { a: ia, b: ib, c: ic, d: id, e: -(ia * a.e + ic * a.f), f: -(ib * a.e + id * a.f) })
}

fn visible(node: &RawNode) -> bool {
    !node.truthy("hidden") && node.f64("opacity") != Some(0.0)
}

fn children_ids(scene: &Scene, id: &str, node: &RawNode) -> Vec<NodeId> {
    scene
        .children_by_parent
        .get(id)
        .cloned()
        .or_else(|| node.array("children").map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()))
        .unwrap_or_default()
}

fn grow(acc: &mut Option<Box4>, p: (f64, f64)) {
    if !(p.0.is_finite() && p.1.is_finite()) {
        return;
    }
    *acc = Some(match *acc {
        None => [p.0, p.1, p.0, p.1],
        Some(b) => [b[0].min(p.0), b[1].min(p.1), b[2].max(p.0), b[3].max(p.1)],
    });
}

fn grow_box(acc: &mut Option<Box4>, local: Box4, a: &Affine) {
    for p in [(local[0], local[1]), (local[2], local[1]), (local[2], local[3]), (local[0], local[3])] {
        grow(acc, transform::apply_point(p, *a));
    }
}

/// The source box of repeat `rep_id` (see the module docs), group-local.
pub fn source_box(scene: &Scene, rep_id: &str) -> Option<Box4> {
    let node = scene.node(rep_id)?;
    let to_local = scene.accumulated_affine(rep_id).and_then(|(a, _)| invert(&a))?;
    let mut acc = None;
    for id in children_ids(scene, rep_id, node) {
        box_of(scene, rep_id, &id, &to_local, &mut acc, 0);
    }
    acc
}

fn box_of(scene: &Scene, rep_id: &str, id: &str, to_local: &Affine, acc: &mut Option<Box4>, depth: u32) {
    let Some(node) = scene.node(id) else { return };
    if !visible(node) || depth > 64 {
        return;
    }
    let rel = |nid: &str| scene.accumulated_affine(nid).map(|(a, _)| transform::compose(*to_local, a));
    match node.node_type() {
        "group" | "boolean" => {
            for c in children_ids(scene, id, node) {
                box_of(scene, rep_id, &c, to_local, acc, depth + 1);
            }
        }
        "container" | "section" => {
            if let (Some(w), Some(h), Some(a)) = (node.f64("width"), node.f64("height"), rel(id)) {
                grow_box(acc, [0.0, 0.0, w, h], &a);
            }
            if !node.truthy("clipContent") {
                for c in children_ids(scene, id, node) {
                    box_of(scene, rep_id, &c, to_local, acc, depth + 1);
                }
            }
        }
        "path" => {
            if let (Some(b), Some(a)) = (node.points().and_then(transform::bounds_from_points), rel(id)) {
                grow_box(acc, [b.min_x, b.min_y, b.max_x, b.max_y], &a);
            }
        }
        "line" => {
            // A line's points are in its parent's space.
            let Some(pts) = node.array("points") else { return };
            let n = |i: usize| pts.get(i).and_then(Value::as_f64);
            let parent = scene.parent_map.get(id).map(String::as_str).unwrap_or(rep_id);
            let a = if parent == rep_id { Some(transform::identity()) } else { rel(parent) };
            if let (Some(x1), Some(y1), Some(x2), Some(y2), Some(a)) = (n(0), n(1), n(2), n(3), a) {
                grow(acc, transform::apply_point((x1, y1), a));
                grow(acc, transform::apply_point((x2, y2), a));
            }
        }
        _ => {
            if let (Some(w), Some(h), Some(a)) = (node.f64("width"), node.f64("height"), rel(id)) {
                grow_box(acc, [0.0, 0.0, w, h], &a);
            }
        }
    }
}

/// Ids of the repeats in `scene`.
pub fn repeat_ids(scene: &Scene) -> Vec<NodeId> {
    scene.node_order.iter().filter(|id| scene.node(id).is_some_and(is_repeat)).cloned().collect()
}

struct Copier<'a> {
    scene: &'a Scene,
    rep_id: &'a str,
    out: Resolved,
    /// Transients left to make before the budget stops the copies.
    budget: usize,
    /// Warned about a group's appearance once.
    appearance_noted: bool,
}

/// The keys a copy never carries: the live-group specs (an inner envelope or repeat copies as what it draws), and
/// the tree links the scene derives from canonical nodes.
const DROPPED_KEYS: &[&str] = &["parentId", "repeat", "envelope", "imageTrace", "_transformSourceId"];

impl Copier<'_> {
    fn stamp(&self, m: &mut Map, src: &RawNode, new_id: &str) {
        m.insert("id".into(), json!(new_id));
        for k in DROPPED_KEYS {
            m.remove(*k);
        }
        m.insert(REPEAT_OF_KEY.into(), json!(self.rep_id));
        let source = src.str(REPEAT_SOURCE_KEY).unwrap_or(src.id()).to_string();
        m.insert(REPEAT_SOURCE_KEY.into(), json!(source));
    }

    /// A transient group with `template`'s compositing keys and the given transform, holding `ids`' copies.
    fn wrapper(&mut self, parent: &str, new_id: String, src: &RawNode, trs: (f64, f64, f64, f64, f64), ids: &[NodeId], prefix: &str) -> Option<String> {
        let mut m = Map::new();
        m.insert("type".into(), json!("group"));
        for k in ["opacity", "blendMode", "effects", "name", "metadata", "maskType", "maskInvert", "maskFeather", "hidden"] {
            if let Some(v) = src.json_field(k) {
                m.insert(k.into(), v.clone());
            }
        }
        let (x, y, r, sx, sy) = trs;
        for (k, v) in [("x", x), ("y", y), ("rotation", r), ("scaleX", sx), ("scaleY", sy)] {
            m.insert(k.into(), json!(v));
        }
        self.stamp(&mut m, src, &new_id);
        self.push_group(parent, new_id, m, ids, prefix, None)
    }

    /// Push group record `m` (id `new_id`) under `parent`, then its children's copies; its `children` (and clip mask)
    /// name the copies.
    fn push_group(&mut self, parent: &str, new_id: String, mut m: Map, ids: &[NodeId], prefix: &str, mask: Option<&str>) -> Option<String> {
        if self.budget == 0 {
            return None;
        }
        self.budget -= 1;
        let at = self.out.transients.len();
        self.out.transients.push((parent.to_string(), RawNode::new(Map::new())));
        let mut kids = Vec::new();
        let mut mask_copy = None;
        for cid in ids {
            if let Some(k) = self.copy(&new_id, cid, prefix) {
                if mask == Some(cid.as_str()) {
                    mask_copy = Some(k.clone());
                }
                kids.push(Value::String(k));
            }
        }
        m.insert("children".into(), Value::Array(kids));
        match mask_copy {
            Some(k) => {
                m.insert("clipMaskNodeId".into(), json!(k));
                self.out.mask_sources.push(k);
            }
            None => {
                m.remove("clipMaskNodeId");
            }
        }
        self.out.transients[at].1 = RawNode::new(m);
        Some(new_id)
    }

    /// Copy node `id` (and what it draws) under transient `parent`; returns the copy's id.
    fn copy(&mut self, parent: &str, id: &str, prefix: &str) -> Option<String> {
        let scene = self.scene;
        let canonical = scene.canonical_or_transient(id)?;
        if !visible(canonical) || self.budget == 0 {
            return None;
        }
        let node = scene.effective_node(id)?;
        let new_id = format!("{prefix}{id}");
        // An inner envelope or repeat: what it draws.
        if let Some(ids) = scene.envelope_transients.get(id) {
            let _ = ids;
            let kids = scene.transient_by_parent.get(id).cloned().unwrap_or_default();
            let mut m = node.to_json_map();
            for k in ["fills", "strokes", "clipMaskNodeId", "children"] {
                m.remove(k);
            }
            self.stamp(&mut m, node, &new_id);
            return self.push_group(parent, new_id, m, &kids, prefix, None);
        }
        match canonical.node_type() {
            "group" | "container" | "section" => {
                if canonical.node_type() == "group" && scene.group_appearance(id).is_some() && !self.appearance_noted {
                    self.appearance_noted = true;
                    self.out.skipped.push(format!("{id} (a group's own fills and strokes)"));
                }
                let kids = if canonical.node_type() == "section" && canonical.truthy("contentsHidden") {
                    Vec::new()
                } else {
                    children_ids(scene, id, canonical)
                };
                let mask = canonical.str("clipMaskNodeId").filter(|m| !m.is_empty()).map(str::to_string);
                let mut m = node.to_json_map();
                if canonical.node_type() == "group" {
                    m.remove("fills");
                    m.remove("strokes");
                }
                self.stamp(&mut m, node, &new_id);
                self.push_group(parent, new_id, m, &kids, prefix, mask.as_deref())
            }
            "table" | "generated" | "generator" => {
                // The host's drawn parts. A generator's (and a native-children chart's) are in its local space; a
                // table's and a proxied chart's carry the host's transform themselves.
                let local = canonical.node_type() == "generator"
                    || (canonical.node_type() == "generated" && canonical.str("renderMode") == Some("native-children"));
                let trs = if local {
                    let (sx, sy) = node.scale_xy();
                    (node.f64("x").unwrap_or(0.0), node.f64("y").unwrap_or(0.0), node.rotation(), sx, sy)
                } else {
                    (0.0, 0.0, 0.0, 1.0, 1.0)
                };
                let kids = scene.transient_by_parent.get(id).cloned().unwrap_or_default();
                self.wrapper(parent, new_id, node, trs, &kids, prefix)
            }
            _ => {
                // A leaf, or a boolean's result (its resolved record is the path it draws).
                self.budget -= 1;
                let mut m = node.to_json_map();
                self.stamp(&mut m, node, &new_id);
                self.out.transients.push((parent.to_string(), RawNode::new(m)));
                Some(new_id)
            }
        }
    }
}

/// How many transients one copy of the source makes.
fn copy_size(scene: &Scene, ids: &[NodeId], depth: u32) -> usize {
    let mut n = 0;
    for id in ids {
        let Some(c) = scene.canonical_or_transient(id) else { continue };
        if !visible(c) || depth > 64 {
            continue;
        }
        n += 1;
        let kids = if scene.envelope_transients.contains_key(id) || matches!(c.node_type(), "table" | "generated" | "generator") {
            scene.transient_by_parent.get(id).cloned().unwrap_or_default()
        } else if matches!(c.node_type(), "group" | "container" | "section") {
            children_ids(scene, id, c)
        } else {
            Vec::new()
        };
        n += copy_size(scene, &kids, depth + 1);
    }
    n
}

/// Resolve repeat `rep_id` of `scene` (see the module docs). Inner envelopes and repeats must be resolved already.
pub fn resolve(scene: &Scene, rep_id: &str) -> Resolved {
    let mut out = Resolved::default();
    let Some(node) = scene.node(rep_id) else { return out };
    let Some(spec) = node.json_field(REPEAT_KEY).and_then(parse_spec) else { return out };
    let b = source_box(scene, rep_id);
    let mut instances = instance_transforms(&spec, b);
    let ids = children_ids(scene, rep_id, node);
    let per_copy = copy_size(scene, &ids, 0) + 1;
    let fits = (MAX_REPEAT_NODES / per_copy).max(1);
    if instances.len() > fits {
        out.skipped.push(format!(
            "{} of {} copies (a repeat draws at most {MAX_REPEAT_NODES} layers)",
            instances.len() - fits,
            instances.len()
        ));
        instances.truncate(fits);
    }
    out.content_bounds = b.map(|b| crate::live_effects::Bounds {
        min: crate::path_geom::vec2::v2(b[0], b[1]),
        max: crate::path_geom::vec2::v2(b[2], b[3]),
    });
    let mut copier = Copier { scene, rep_id, out, budget: MAX_REPEAT_NODES, appearance_noted: false };
    for (i, a) in &instances {
        // The copy's group carries its transform only: the repeat's own compositing stays on the repeat.
        let inst_id = format!("{rep_id}~r{i}");
        let prefix = format!("{inst_id}~");
        let mut m = Map::new();
        m.insert("id".into(), json!(inst_id));
        m.insert("type".into(), json!("group"));
        m.insert("name".into(), json!(format!("Copy {}", i + 1)));
        let (x, y, r, sx, sy) = trs_of(a);
        for (k, v) in [("x", x), ("y", y), ("rotation", r), ("scaleX", sx), ("scaleY", sy)] {
            m.insert(k.into(), json!(v));
        }
        m.insert(REPEAT_OF_KEY.into(), json!(rep_id));
        copier.push_group(rep_id, inst_id, m, &ids, &prefix, None);
    }
    let mut out = copier.out;
    if let Some(b) = b {
        let mut acc: Option<Box4> = None;
        for (_, a) in &instances {
            grow_box(&mut acc, b, a);
        }
        out.bounds = acc.map(|b| crate::live_effects::Bounds {
            min: crate::path_geom::vec2::v2(b[0], b[1]),
            max: crate::path_geom::vec2::v2(b[2], b[3]),
        });
    }
    out
}

#[cfg(test)]
#[path = "repeat_tests.rs"]
mod tests;
