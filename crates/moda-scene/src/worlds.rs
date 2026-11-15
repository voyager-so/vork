//! The scene's derived transform columns: per canonical row (a node's index in `doc.nodes`), its accumulated
//! transform, with the hierarchy as integer arrays the columns are computed over.
//!
//! One rule composes every accumulated transform in the scene, parent first:
//!
//! ```text
//! frame(parent)  = frame(grandparent) ∘ local-as-parent(parent)
//! world(node)    = frame(parent) ∘ local(node)
//! ```
//!
//! ([`Placed::under`]). The columns hold that for the rows whose whole transform ancestry is canonical nodes; the
//! walk by name ([`Scene::accumulated_affine`]) computes the same composition, in the same order, for every other
//! node (an expansion's transients, a node a derived record stands in for). So a matrix read from a column and a
//! matrix walked by name are the same `f64`s, and a reader never needs to know which it got.
//!
//! The columns are derived data under the stamps rule ([`crate::stamps`]): [`Scene::refresh_worlds`] brings them
//! up to the scene's clock, recomputing the hierarchy when a structure stamp is newer than it and otherwise only
//! the subtrees of the rows written since (the scene's own dirty lists: a placement mark flows to descendants
//! through the parent-before-child order). They answer only while the clock stands where they were refreshed;
//! after any later write a reader walks by name until the next refresh. Nothing else invalidates them.

use crate::scene::{is_parent_node, HashMap, Scene};
use crate::stamps::{Group, Stamps, Subscription};
use crate::transform::{self, compose};
use crate::types::{Affine, NodeId, RawNode, Transform2D};

/// A transform with what [`Transform2D`] accumulates beside the matrix (the rotations summed, the scales
/// multiplied): a node's local transform, or one composed under the transforms above it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placed {
    pub affine: Affine,
    pub rotation: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}

impl Placed {
    /// The node's own local transform (`calculateAccumulatedTransform`, nodeTransformUtils.ts:251-311).
    pub(crate) fn own(node: &RawNode) -> Placed {
        let (scale_x, scale_y) = node.scale_xy();
        Placed::local(node, scale_x, scale_y)
    }

    /// The node's local transform as the transform parent of what is under it:
    /// `parent.type === 'group' ? parent.scaleX||1 : parent.scale||1` (nodeTransformUtils.ts:288-289). A boolean's
    /// canonical node is a group-like transform parent (E7.1).
    pub(crate) fn parent(node: &RawNode, boolean_source: bool) -> Placed {
        let (scale_x, scale_y) = if node.node_type() == "group" || boolean_source {
            let sx = node.f64("scaleX").unwrap_or(1.0);
            let sy = node.f64("scaleY").unwrap_or(1.0);
            (if sx == 0.0 { 1.0 } else { sx }, if sy == 0.0 { 1.0 } else { sy })
        } else {
            let s = node.scale();
            (s, s)
        };
        Placed::local(node, scale_x, scale_y)
    }

    fn local(node: &RawNode, scale_x: f64, scale_y: f64) -> Placed {
        let [x, y] = [node.f64("x").unwrap_or(0.0), node.f64("y").unwrap_or(0.0)];
        let rotation = node.rotation();
        Placed {
            affine: transform::from_trsk(Transform2D { x, y, rotation, scale_x, scale_y }, node.skew_x()),
            rotation,
            scale_x,
            scale_y,
        }
    }

    /// This local transform under the accumulated transform above it: the one composition rule. With nothing
    /// above, the local transform is the accumulated one as it is (it is not multiplied by an identity).
    pub(crate) fn under(self, above: Option<&Placed>) -> Placed {
        match above {
            None => self,
            Some(above) => Placed {
                affine: compose(above.affine, self.affine),
                rotation: above.rotation + self.rotation,
                scale_x: above.scale_x * self.scale_x,
                scale_y: above.scale_y * self.scale_y,
            },
        }
    }

    /// What [`Scene::accumulated_affine`] answers: the matrix and the decomposed transform at its origin.
    pub(crate) fn answer(&self) -> (Affine, Transform2D) {
        let origin = transform::apply_point((0.0, 0.0), self.affine);
        (
            self.affine,
            Transform2D { x: origin.0, y: origin.1, rotation: self.rotation, scale_x: self.scale_x, scale_y: self.scale_y },
        )
    }
}

/// No row: a chain that ends, a row that keeps no frame, a row not yet ordered.
const NONE: u32 = u32::MAX;

/// The derived transform columns of one scene, and the hierarchy by row they are computed over. See the module
/// text for the rule; [`Scene::world_rows`] hands it out while it is current.
#[derive(Default)]
pub struct Worlds {
    /// The scene's dirty lists for these columns: rows whose placement, and rows whose content, was written since
    /// the last refresh (an unrestricted write to a row stamps both, and may have changed any field).
    lists: Option<[Subscription; 2]>,
    /// The scene clock the columns were last brought up to. They answer while the clock stands there.
    current_at: Option<u64>,
    /// The scene clock the hierarchy was read at.
    read_at: Option<u64>,
    /// Whether a derived record stood in for any node when the hierarchy was read: there a content write can
    /// change who a node's transform ancestors are, so the hierarchy also falls with a content stamp.
    derived: bool,

    // The hierarchy, by row.
    /// Rows whose whole transform ancestor chain is canonical nodes: the rows the columns answer for.
    plain: Vec<bool>,
    /// Row `r`'s transform parent: the next link of [`Scene::transform_parent_chain`], as a row, or `NONE` where
    /// the chain ends. One link a row.
    link: Vec<u32>,
    /// The position of the row's page.
    page: Vec<(f64, f64)>,
    /// Every row once, each parent before its children: row `r` and everything under it are
    /// `order[at[r]..end[r]]`, `r` first.
    order: Vec<u32>,
    at: Vec<u32>,
    end: Vec<u32>,
    /// Rows under which every node is a row: no expansion, and no child that is not a canonical node.
    whole: Vec<bool>,
    /// Rows that hold children ([`is_parent_node`]).
    parent: Vec<bool>,
    /// Per row, where its frame sits in `frames`, or `NONE`: plain rows that hold children keep one.
    frame_of: Vec<u32>,

    // The columns.
    /// Per plain row, the node's accumulated transform in its page: [`Scene::accumulated_affine`]'s answer.
    own: Vec<Placed>,
    /// Per frame-keeping row, its accumulated transform as a transform parent.
    frames: Vec<Placed>,

    /// Scratch for a refresh: the written rows' ranges of `order`.
    ranges: Vec<(u32, u32)>,
}

impl Worlds {
    /// Subscribe these columns to the scene's writes. Called once, when the scene is made, before any other
    /// consumer subscribes: a scene that replaces another takes over its subscriptions by position
    /// ([`Stamps::adopt_subscriptions`]), and these are the first of their groups in every scene.
    pub(crate) fn subscribe(&mut self, stamps: &mut Stamps) {
        self.lists = Some([stamps.subscribe(Group::Placement), stamps.subscribe(Group::Content)]);
    }

    /// Whether the columns stand for the scene at `clock`.
    fn current(&self, clock: u64) -> bool {
        self.current_at == Some(clock)
    }

    /// The accumulated transform of the node at `row` in its page, when the columns are current at `clock` and
    /// answer for the row.
    pub(crate) fn placed(&self, clock: u64, row: usize) -> Option<&Placed> {
        if !self.current(clock) || !*self.plain.get(row)? {
            return None;
        }
        self.own.get(row)
    }

    /// The position of the page of the node at `row` (of a row the columns answer for).
    pub(crate) fn page(&self, row: usize) -> (f64, f64) {
        self.page.get(row).copied().unwrap_or((0.0, 0.0))
    }

    /// The scene clock the hierarchy was read at: it names one hierarchy. Something made over these rows (a
    /// plan's spans by row) stands while this does.
    pub fn hierarchy(&self) -> u64 {
        self.read_at.unwrap_or(0)
    }

    /// How many rows the columns cover.
    pub fn len(&self) -> usize {
        self.plain.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plain.is_empty()
    }

    /// Where `row` and everything under it sit in the parent-before-child order: positions `.0..(.1)`. One
    /// row's range lies wholly inside or wholly outside another's.
    pub fn range(&self, row: u32) -> Option<(u32, u32)> {
        let row = row as usize;
        let (at, end) = (*self.at.get(row)?, *self.end.get(row)?);
        (at != NONE).then_some((at, end))
    }

    /// The rows at positions `from..to` of the parent-before-child order.
    pub fn ordered(&self, from: u32, to: u32) -> &[u32] {
        self.order.get(from as usize..to as usize).unwrap_or(&[])
    }

    /// Whether everything under `row` is a row: nothing under it is an expansion's transient (a table's cells, a
    /// chart's parts) or a child that is not a canonical node. Those are found by name.
    pub fn whole(&self, row: u32) -> bool {
        self.whole.get(row as usize).copied().unwrap_or(false)
    }

    /// Whether the row holds children: a move of it is a move of its subtree, and it need not draw itself.
    pub fn is_parent(&self, row: u32) -> bool {
        self.parent.get(row as usize).copied().unwrap_or(false)
    }

    /// Bring the columns up to `scene`.
    fn refresh(&mut self, scene: &Scene) {
        let clock = scene.stamps.clock();
        if self.current(clock) {
            return;
        }
        let derived = self.derived || !scene.resolved_by_id.is_empty() || !scene.boolean_source_by_id.is_empty();
        let stands = self.plain.len() == scene.doc.nodes.len()
            && self.read_at.is_some_and(|at| {
                scene.stamps.latest(Group::Structure) <= at && (!derived || scene.stamps.latest(Group::Content) <= at)
            });
        if stands {
            // The subtrees of the written rows, as the union of their ranges: a range starts inside an earlier
            // one only when it lies wholly inside it, so every row is computed once, after its parent.
            self.ranges.clear();
            for list in self.lists.iter().flatten() {
                for &row in scene.stamps.dirty(*list) {
                    if let Some(range) = self.range(row) {
                        self.ranges.push(range);
                    }
                }
            }
            self.ranges.sort_unstable();
            let mut covered = 0u32;
            for i in 0..self.ranges.len() {
                let (from, to) = self.ranges[i];
                if to <= covered {
                    continue;
                }
                for position in from.max(covered)..to {
                    self.place(scene, self.order[position as usize] as usize);
                }
                covered = to;
            }
        } else {
            self.read_hierarchy(scene);
            for position in 0..self.order.len() {
                self.place(scene, self.order[position] as usize);
            }
        }
        self.current_at = Some(clock);
    }

    /// Compute the columns of `row` from its node and the frame of its transform parent, which is current.
    fn place(&mut self, scene: &Scene, row: usize) {
        if !self.plain[row] {
            return;
        }
        let node = &scene.doc.nodes[row];
        let above = match self.link[row] {
            NONE => None,
            parent => Some(self.frames[self.frame_of[parent as usize] as usize]),
        };
        self.own[row] = Placed::own(node).under(above.as_ref());
        let frame = self.frame_of[row];
        if frame != NONE {
            // A plain row is its own resolved record and no boolean's source.
            self.frames[frame as usize] = Placed::parent(node, false).under(above.as_ref());
        }
    }

    /// Read the scene's hierarchy into the row arrays.
    fn read_hierarchy(&mut self, scene: &Scene) {
        let rows = scene.doc.nodes.len();
        let identity = Placed { affine: transform::translate(0.0, 0.0), rotation: 0.0, scale_x: 1.0, scale_y: 1.0 };
        self.read_at = Some(scene.stamps.clock());
        self.derived = !scene.resolved_by_id.is_empty() || !scene.boolean_source_by_id.is_empty();
        fn reset<T: Clone>(column: &mut Vec<T>, rows: usize, value: T) {
            column.clear();
            column.resize(rows, value);
        }
        reset(&mut self.plain, rows, false);
        reset(&mut self.link, rows, NONE);
        reset(&mut self.page, rows, (0.0, 0.0));
        reset(&mut self.at, rows, NONE);
        reset(&mut self.end, rows, 0);
        reset(&mut self.whole, rows, true);
        reset(&mut self.parent, rows, false);
        reset(&mut self.frame_of, rows, NONE);
        reset(&mut self.own, rows, identity);
        self.order.clear();
        self.order.reserve(rows);
        self.frames.clear();
        // A repeated id draws its last copy; only that row is the node.
        let canonical = |row: usize| scene.node_index.get(scene.doc.nodes[row].id()) == Some(&row);
        // Each row's first link of `transform_parent_chain`, by the same rules. A parent that holds children
        // (or is a boolean's source) carries its transform; one that does not ends the chain. A link through
        // an expansion (a table's cell, a chart's part) is not a row: such rows are not answered here.
        // Where a derived record stands in for a node, the node that draws is not the canonical one: that row, and
        // every row under it, is not answered. This is decided here, once; the hierarchy is read again when
        // content changes in a scene that has such records, so a reader looks no name up.
        let derived = self.derived;
        let mut answered = vec![false; rows];
        for (row, node) in scene.doc.nodes.iter().enumerate() {
            if !canonical(row) {
                continue;
            }
            let id = node.id();
            self.parent[row] = is_parent_node(node);
            if derived && (scene.resolved_by_id.contains_key(id) || scene.boolean_source_by_id.contains_key(id)) {
                continue;
            }
            match scene.parent_map.get(id) {
                Some(p) => match scene.effective_node(p) {
                    Some(parent) if is_parent_node(parent) || scene.boolean_source_by_id.contains_key(p) => {
                        if let Some(&parent) = scene.node_index.get(p) {
                            self.link[row] = parent as u32;
                            answered[row] = true;
                        }
                    }
                    _ => answered[row] = true,
                },
                None => answered[row] = !scene.transient_parent_of.contains_key(id),
            }
        }
        // A row is plain when every link above it is: decided once a row, following links that are not decided
        // yet and stopping at any that is. (A hierarchy has no cycles; one is refused on load. A walk longer
        // than the rows is treated as one.)
        let mut decided = vec![false; rows];
        let mut path: Vec<usize> = Vec::new();
        for start in 0..rows {
            if decided[start] {
                continue;
            }
            path.clear();
            let mut row = start;
            let plain = loop {
                if decided[row] {
                    break self.plain[row];
                }
                if !answered[row] || path.len() > rows {
                    break false;
                }
                path.push(row);
                match self.link[row] {
                    NONE => break true,
                    parent => row = parent as usize,
                }
            };
            for &row in &path {
                self.plain[row] = plain;
                decided[row] = true;
            }
            decided[row] = true;
        }
        // Each page's position once (the first page of an id, as the walk by name finds it), then each row its
        // page's.
        let mut positions: HashMap<&str, (f64, f64)> = HashMap::default();
        for (i, page) in scene.doc.pages.iter().enumerate() {
            positions.entry(page.id()).or_insert_with(|| scene.page_position(i));
        }
        for (row, node) in scene.doc.nodes.iter().enumerate() {
            if self.plain[row] {
                self.page[row] =
                    scene.page_of(node.id()).and_then(|pid| positions.get(pid.as_str()).copied()).unwrap_or((0.0, 0.0));
            }
        }
        // Parent-before-child order, by the child lists the display walk reads. Rows with a parent are reached
        // from it; whatever is left after that (a row on no list) starts its own range.
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for pass in 0..2 {
            for root in 0..rows {
                if self.at[root] != NONE {
                    continue;
                }
                if pass == 0 && (!canonical(root) || scene.parent_map.contains_key(scene.doc.nodes[root].id())) {
                    continue;
                }
                self.at[root] = self.order.len() as u32;
                self.order.push(root as u32);
                stack.push((root, 0));
                while let Some(&(row, next)) = stack.last() {
                    let id = scene.doc.nodes[row].id();
                    if next == 0 && (!canonical(row) || scene.transient_by_parent.contains_key(id)) {
                        self.whole[row] = false;
                    }
                    // Only the node a name stands for has that name's children.
                    let kids: &[NodeId] =
                        if canonical(row) { scene.children_by_parent.get(id).map(Vec::as_slice).unwrap_or(&[]) } else { &[] };
                    if let Some(kid) = kids.get(next) {
                        stack.last_mut().expect("the entry just read").1 += 1;
                        match scene.node_index.get(kid) {
                            Some(&child) if self.at[child] == NONE => {
                                self.at[child] = self.order.len() as u32;
                                self.order.push(child as u32);
                                stack.push((child, 0));
                            }
                            // Not a row, or a row already placed under another parent.
                            _ => self.whole[row] = false,
                        }
                    } else {
                        self.end[row] = self.order.len() as u32;
                        stack.pop();
                        if let Some(&(above, _)) = stack.last() {
                            if !self.whole[row] {
                                self.whole[above] = false;
                            }
                        }
                    }
                }
            }
        }
        // A plain row that holds children keeps a frame for them.
        for row in 0..rows {
            if self.plain[row] && self.parent[row] {
                self.frame_of[row] = self.frames.len() as u32;
                self.frames.push(identity);
            }
        }
        // The columns are computed along the order: a row's transform parent must keep a frame and hold the row
        // inside its range, so that it is computed first and a write to it reaches the row. A document whose
        // child lists and parent links disagree about that is answered by the walk by name alone.
        let agrees = (0..rows).all(|row| {
            !self.plain[row] || match self.link[row] {
                NONE => true,
                parent => {
                    let parent = parent as usize;
                    self.frame_of[parent] != NONE && self.at[parent] < self.at[row] && self.at[row] < self.end[parent]
                }
            }
        });
        if !agrees {
            reset(&mut self.plain, rows, false);
        }
    }
}

impl Scene {
    /// Bring the scene's world columns up to its clock ([`crate::worlds`]): the hierarchy again when structure
    /// changed, else the subtrees of the rows written since the last refresh. Until the next write, every
    /// accumulated transform of a row the columns answer for is read from them, and [`Scene::world_rows`] hands
    /// out the hierarchy by row.
    pub fn refresh_worlds(&mut self) {
        if self.worlds.current(self.stamps.clock()) {
            return;
        }
        let mut worlds = std::mem::take(&mut self.worlds);
        worlds.refresh(self);
        for list in worlds.lists.iter().flatten() {
            self.stamps.clear(*list);
        }
        self.worlds = worlds;
    }

    /// The world columns and the hierarchy by row, while they are current: `None` after any write since
    /// [`Scene::refresh_worlds`].
    pub fn world_rows(&self) -> Option<&Worlds> {
        self.worlds.current(self.stamps.clock()).then_some(&self.worlds)
    }

    /// [`Scene::node_canvas_affine`] of the canonical node at `row`, read from the world columns: no name is
    /// looked up. `None` when the columns are not current or do not answer for the row (a derived record stands
    /// in for it or for one of its transform ancestors, or an ancestor is not a canonical node); the walk by name
    /// answers then, with the same composition.
    pub fn row_canvas_affine(&self, row: u32) -> Option<Affine> {
        let placed = self.worlds.placed(self.stamps.clock(), row as usize)?;
        let page = self.worlds.page(row as usize);
        Some(compose(transform::translate(page.0, page.1), placed.affine))
    }

    /// The accumulated transform as a transform parent of everything above the node named `id`, composed parent
    /// first along [`Scene::transform_parent_chain`]; `None` when nothing is above it.
    pub(crate) fn frame_above(&self, id: &str) -> Option<Placed> {
        let mut frame: Option<Placed> = None;
        for pid in self.transform_parent_chain(id).iter().rev() {
            let Some(parent) = self.effective_node(pid) else {
                continue;
            };
            frame = Some(Placed::parent(parent, self.boolean_source_by_id.contains_key(pid)).under(frame.as_ref()));
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::NodePlacement;
    use crate::value::json;

    fn bits(a: &Affine) -> [u64; 6] {
        [a.a, a.b, a.c, a.d, a.e, a.f].map(f64::to_bits)
    }

    /// What the walk by name answers for each of `ids` (the columns are stale), then what the columns answer
    /// once refreshed: the same numbers, bit for bit, by row and by name.
    fn check(scene: &mut Scene, ids: &[&str]) {
        assert!(scene.world_rows().is_none(), "a write leaves the columns unanswering until they are refreshed");
        let walked: Vec<_> = ids.iter().map(|id| scene.node_canvas_affine_and_transform(id).unwrap()).collect();
        scene.refresh_worlds();
        assert!(scene.world_rows().is_some());
        for (id, walked) in ids.iter().zip(&walked) {
            let row = scene.row_of(id).unwrap();
            assert_eq!(bits(&scene.row_canvas_affine(row).unwrap()), bits(&walked.0), "{id} by row");
            let (affine, trs) = scene.node_canvas_affine_and_transform(id).unwrap();
            assert_eq!(bits(&affine), bits(&walked.0), "{id} by name");
            assert_eq!(format!("{trs:?}"), format!("{:?}", walked.1), "{id} transform");
        }
    }

    #[test]
    fn the_columns_hold_what_the_walk_by_name_composes() {
        let d = json!({"nodes": [
            {"id": "g", "type": "group", "x": 10.5, "y": -3.25, "rotation": 30, "scaleX": 1.5, "scaleY": 0.75, "children": ["h", "a"]},
            {"id": "h", "type": "group", "x": 2, "y": 7, "rotation": -12.5, "scaleX": -1, "scaleY": 2, "skewX": 10, "children": ["b"]},
            {"id": "a", "type": "path", "x": 1, "y": 2, "rotation": 45,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 10, "y": 0}, {"type": "line", "x": 10, "y": 10}], "closed": true},
            {"id": "b", "type": "path", "x": 3.3, "y": 4.4, "scaleX": 2, "scaleY": 0.3,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 5, "y": 0}, {"type": "line", "x": 5, "y": 5}], "closed": true},
            {"id": "c", "type": "path", "x": 9, "y": 9,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 5, "y": 0}, {"type": "line", "x": 5, "y": 5}], "closed": true}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["g", "c"]}]});
        let mut scene = Scene::load(&d).unwrap();
        let all = ["g", "h", "a", "b", "c"];
        check(&mut scene, &all);
        // Parent first: a node two groups deep is its local transform under (g ∘ h), not g ∘ (h ∘ local).
        {
            let node = |id: &str| scene.node(id).unwrap();
            let above = Placed::parent(node("h"), false).under(Some(&Placed::parent(node("g"), false)));
            let b = Placed::own(node("b")).under(Some(&above));
            assert_eq!(bits(&scene.accumulated_affine("b").unwrap().0), bits(&b.affine));
        }
        // The hierarchy by row: a row and everything under it are one range, parents first.
        {
            let rows = scene.world_rows().unwrap();
            let names = |id: &str| -> Vec<String> {
                let (from, to) = rows.range(scene.row_of(id).unwrap()).unwrap();
                rows.ordered(from, to).iter().map(|&row| scene.row(row).unwrap().id().to_string()).collect()
            };
            assert_eq!(names("g"), ["g", "h", "b", "a"]);
            assert_eq!(names("h"), ["h", "b"]);
            assert_eq!(names("c"), ["c"]);
            let (g, a) = (scene.row_of("g").unwrap(), scene.row_of("a").unwrap());
            assert!(rows.is_parent(g) && !rows.is_parent(a) && rows.whole(g));
        }
        // A typed placement write to an ancestor reaches everything under it, through the same hierarchy.
        let hierarchy = scene.world_rows().unwrap().hierarchy();
        let number = |v: f64| crate::value::Number::from_f64(v).unwrap();
        let g = scene.row_of("g").unwrap();
        scene.write_placement(g, Some(NodePlacement { x: number(40.25), y: number(1.0), rotation: number(77.0), scale_x: number(0.5), scale_y: number(3.0) }), None);
        check(&mut scene, &all);
        // So does a rotation written as a field, and a write to a leaf alone.
        let h = scene.row_of("h").unwrap();
        scene.write_row(h, |node| node.set("rotation", json!(33.5))).unwrap();
        check(&mut scene, &all);
        let b = scene.row_of("b").unwrap();
        scene.write_placement(b, Some(NodePlacement { x: number(-7.125), y: number(0.5), rotation: number(3.0), scale_x: number(1.0), scale_y: number(1.0) }), None);
        check(&mut scene, &all);
        assert_eq!(scene.world_rows().unwrap().hierarchy(), hierarchy, "writes to rows do not read the hierarchy again");
        // After a whole-node edit too, nothing answers until the next refresh.
        let mut moved = scene.node("a").unwrap().clone();
        moved.set("x", json!(-55.5));
        scene.apply_node_edits(vec![moved], &[], &[], None).unwrap();
        assert!(scene.world_rows().is_none());
        check(&mut scene, &all);
        // A structural edit moves rows: the hierarchy is read again.
        scene.apply_node_edits(vec![], &["g".to_string()], &[], None).unwrap();
        check(&mut scene, &["c"]);
        assert_ne!(scene.world_rows().unwrap().hierarchy(), hierarchy);
    }

    thread_local! {
        /// What the probe below read.
        static READ: std::cell::Cell<Option<f64>> = const { std::cell::Cell::new(None) };
    }

    /// A write stamps what it wrote after deriving from it. A derivation in that window reads the written node:
    /// columns refreshed before the write do not answer it with the transform the node held. (Without the clock
    /// advance in `Scene::derive_written` the probe reads the old position.)
    #[test]
    fn a_derivation_inside_a_write_reads_the_written_transform() {
        let d = json!({"nodes": [
            {"id": "a", "type": "path", "x": 1, "y": 2,
             "points": [{"type": "move", "x": 0, "y": 0}, {"type": "line", "x": 10, "y": 0}, {"type": "line", "x": 10, "y": 10}], "closed": true}],
            "pages": [{"id": "p", "width": 100, "height": 100, "nodeIds": ["a"]}]});
        fn probe(scene: &Scene) {
            READ.with(|read| read.set(scene.accumulated_affine("a").map(|(affine, _)| affine.e)));
        }
        for whole in [false, true] {
            let mut scene = Scene::load(&d).unwrap();
            scene.refresh_worlds();
            assert_eq!(scene.accumulated_affine("a").unwrap().0.e, 1.0, "warm: read from the column");
            let x = if whole { 70.0 } else { 40.0 };
            crate::scene::DERIVATION_PROBE.with(|slot| slot.set(Some(probe as fn(&Scene))));
            if whole {
                let mut node = scene.node("a").unwrap().clone();
                node.set("x", json!(x));
                scene.apply_node_edits(vec![node], &[], &[], None).unwrap();
            } else {
                let row = scene.row_of("a").unwrap();
                scene.write_row(row, |node| node.set("x", json!(x))).unwrap();
            }
            crate::scene::DERIVATION_PROBE.with(|slot| slot.set(None));
            assert_eq!(READ.with(|read| read.take()), Some(x), "whole-node edit: {whole}");
        }
    }
}
