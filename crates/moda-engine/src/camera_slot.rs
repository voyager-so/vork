//! Scale RFC stage C (`docs/roadmap/scale-architecture.md`): the retained camera
//! plan — the editor viewport's plan, kept across edits and camera moves and
//! patched in place instead of rebuilt.
//!
//! A camera plan is a list of segments (each visible page, then the freeform
//! canvas root), and each segment a list of [`PlanUnit`]s: a display-list node
//! with its items, a subtree capture's whole range, or a culled range. A full
//! build records the units (`frame::CameraPlanAux`); from then on:
//!
//! - **A property edit** (move, resize, recolour, text, a group transform, a drag
//!   preview) re-plans the edited nodes' units — the node's own, or every unit of
//!   an edited group's subtree — at their render-z, through the same node loop a
//!   full build runs ([`crate::frame::rebuild_segment_range`]), and splices the
//!   items in. O(edited units).
//! - **A structural edit** (add, delete, reorder, reparent, hide/show, a parent's
//!   non-transform key, a table's re-expansion) re-plans only the top-level
//!   subtrees it touched: the segment's top-level list is diffed against the one
//!   the plan was built from, and the range between the common prefix and suffix
//!   (widened to every top-level subtree holding a structurally edited node) is
//!   re-planned from a fresh display list of just those subtrees. Render-z is
//!   baked into every item's depth, so when the range consumes a different z
//!   than before the rest of the segment is re-planned after it: an add or
//!   delete near the top of the z-order (a paste, its undo, a new layer) costs
//!   O(what changed), one near the bottom O(what is above it).
//! - **A camera move** keeps the plan while the cull window serves the view. A
//!   segment that splits (a background blur, a split blend mode) re-plans only
//!   the units whose items are projected into the camera target, and a zoom that
//!   crosses a capture zoom step only the units holding layer captures.
//!
//! Every re-planned range goes through the code a full build runs, under the
//! same contexts, and its scratch mask table is mapped onto the plan's in
//! first-use order, so a patched plan is byte-identical to a from-scratch one:
//! [`plan_divergence`] is the oracle, and the fuzz tests
//! (`camera_slot_tests.rs`) run thousands of random edits through it. An edit
//! to a blend's object or spine, or to a group's clip mask, re-plans that
//! consumer too, as if it were edited itself ([`Deps::expand`]). Whatever
//! the slot cannot prove — a mask the table would need mid-way or no longer
//! first-uses where it did, a segment whose split or capture context appears or goes,
//! a scene-wide flag that re-keys tessellation, a render-z shift that would move
//! a later segment — falls back to the full rebuild, which re-seeds the slot.
//!
//! Still O(n) per patch: `Arc::make_mut` copies the plan's item vector while the
//! engine's prepared frame holds the previous plan (a pointer bump or a small box
//! per item), a length-changing splice shifts the later units' offsets, and a
//! patch that changes how many units a segment has re-indexes the later units'
//! nodes. A cull-window change (a pan past half a viewport, a zoom past 2×)
//! rebuilds, as before.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use moda_scene::scene::is_parent_node;
use moda_scene::types::{compute_page_bg_id, is_page_bg_id, PaintItem};
use moda_scene::Scene;

use crate::cull::Aabb;
use crate::frame::{
    rebuild_segment_range, CameraParams, CameraPlanAux, CameraSegment, CaptureView, PlanUnit,
    PlanWarnings, RangeBuild, RenderOptions, SegmentKind,
};
use crate::residency::{NodeHashes, PlanCaches, SceneDirt};
use crate::seams::{CoveragePlan, DrawItem, MaskPlan};

/// [`PlanUnit::flags`]: the unit's items are projected into the camera target
/// (a backdrop effect or a blend split), so a camera move re-plans it.
pub(crate) const UNIT_CAMERA: u8 = 1;
/// [`PlanUnit::flags`]: the unit holds a layer capture, planned under the
/// capture view (a zoom step and a window), so a capture-view change re-plans it.
pub(crate) const UNIT_CAPTURE: u8 = 2;

/// The dirty-log entry naming a page's top-level list (`ResidentScene`).
/// A unit planned again, to splice over unit `u` (`CameraSlot::splice_unit_batch`): its units and items, the root
/// it belongs to, whether its warnings changed, whether it bakes the shader clock.
struct UnitRepl {
    u: usize,
    root: usize,
    units: Vec<PlanUnit>,
    items: Vec<DrawItem>,
    warnings_changed: bool,
    shader: bool,
}

pub(crate) fn page_list_key(page_id: &str) -> String {
    format!("\u{0}list:{page_id}")
}

/// The dirty-log entry naming the canvas root's top-level list.
pub(crate) const CANVAS_LIST_KEY: &str = "\u{0}list:canvas-root";

/// One top-level subtree's share of a segment.
#[derive(Debug, Clone)]
struct RootSpan {
    id: String,
    dl_start: usize,
    dl_end: usize,
    unit_start: usize,
    unit_end: usize,
}

/// The visible nodes of a segment that switch its planning contexts on: each set
/// holds the nodes satisfying one page-level predicate of the plan build.
#[derive(Debug, Default, Clone)]
struct SegmentFlags {
    /// `backdrop::has_active_background_blur`.
    backdrop: HashSet<String>,
    /// `backdrop::blend::node_splits` over an opaque target.
    split: HashSet<String>,
    /// A node's own split blend mode (`split_blend_mode_over`, opaque).
    own_split: HashSet<String>,
    /// `layer_blur::plan::needs_layer_capture_in`.
    capture: HashSet<String>,
}

impl SegmentFlags {
    fn remove(&mut self, id: &str) {
        self.backdrop.remove(id);
        self.split.remove(id);
        self.own_split.remove(id);
        self.capture.remove(id);
    }

    fn add(&mut self, scene: &Scene, id: &str) {
        let Some(n) = scene.effective_node(id) else {
            return;
        };
        if crate::backdrop::plan::has_active_background_blur(n) {
            self.backdrop.insert(id.to_string());
        }
        if crate::backdrop::blend::node_splits(scene, id, false) {
            self.split.insert(id.to_string());
        }
        if moda_scene::scene::is_opacity_group(n)
            || crate::backdrop::blend::split_blend_mode_over(n, false).is_some()
            || crate::backdrop::blend::composite_blend_split_in(scene, id, false).is_some()
        {
            self.own_split.insert(id.to_string());
        }
        if crate::layer_blur::plan::needs_layer_capture_in(scene, n) {
            self.capture.insert(id.to_string());
        }
    }

    /// The split context a build derives from these flags: `Some(keep rect /
    /// ellipse bodies on the SDF path)` when the segment splits.
    fn backdrop_spec(&self, kind: SegmentKind) -> Option<bool> {
        let effect = !self.backdrop.is_empty();
        let split = !self.split.is_empty();
        if !(effect || split) {
            return None;
        }
        Some(match kind {
            SegmentKind::Page(_) => split && !effect && self.own_split.is_empty(),
            SegmentKind::CanvasRoot => false,
        })
    }
}

/// Scene-wide flags (every node, visible or not, transients included) that
/// re-key the camera plan's tessellation: outlined strokes and curves.
#[derive(Debug, Default, Clone)]
struct SceneFlags {
    outline: HashSet<String>,
    curve: HashSet<String>,
}

impl SceneFlags {
    fn recheck(&mut self, scene: &Scene, id: &str) {
        match scene.effective_node(id) {
            Some(n) => {
                if crate::tess::outline::node_has_outline(scene, n) {
                    self.outline.insert(id.to_string());
                } else {
                    self.outline.remove(id);
                }
                if crate::tess::outline::node_has_curves(n) {
                    self.curve.insert(id.to_string());
                } else {
                    self.curve.remove(id);
                }
            }
            None => {
                self.outline.remove(id);
                self.curve.remove(id);
            }
        }
    }

    /// Whether any member still exists, pruning members whose node went away
    /// without an edit naming them (a host's dropped transients).
    fn any_live(set: &mut HashSet<String>, scene: &Scene) -> bool {
        let mut dead: Vec<String> = Vec::new();
        let mut live = false;
        for id in set.iter() {
            if scene.effective_node(id).is_some() {
                live = true;
                break;
            }
            dead.push(id.clone());
        }
        for d in dead {
            set.remove(&d);
        }
        live
    }

    fn has(&mut self, scene: &Scene) -> (bool, bool) {
        (
            Self::any_live(&mut self.outline, scene),
            Self::any_live(&mut self.curve, scene),
        )
    }
}

/// Nodes whose edit changes OTHER nodes' items, by what they feed: a blend
/// node's objects and spine feed the blend (it draws their geometry and paint),
/// and a `clipMaskNodeId` target feeds the group that clips with it (its mask
/// plan, and the box hints its descendants' items bake). Supersets: edits add
/// entries, nothing removes them (a stale entry only re-plans a consumer that
/// did not need it) until the next full build re-seeds the slot.
#[derive(Debug, Default, Clone)]
struct Deps {
    /// Source (object or spine) → the blend nodes drawing from it.
    blend_consumers: HashMap<String, HashSet<String>>,
    /// Mask node → the groups naming it as their `clipMaskNodeId`, and a clip-to-below base (or a clipped layer) →
    /// the clipped layers stacked directly on it.
    mask_consumers: HashMap<String, HashSet<String>>,
    /// Whether any node noted so far clips to the layer below (`clipToBelow`): only then can an added or moved layer
    /// become a base ([`Deps::note_clip_run`]).
    any_clipped: bool,
}

impl Deps {
    fn note(&mut self, scene: &Scene, id: &str) {
        let Some(node) = scene.effective_node(id) else {
            return;
        };
        if node.node_type() == "blend" {
            for src in moda_scene::blend::dependency_ids(node) {
                self.blend_consumers
                    .entry(src.to_string())
                    .or_default()
                    .insert(id.to_string());
            }
        }
        if let Some(m) = node.str("clipMaskNodeId") {
            if !m.is_empty() {
                self.mask_consumers
                    .entry(m.to_string())
                    .or_default()
                    .insert(id.to_string());
            }
        }
        // Clip to the layer below: the base's coverage clips this layer; and the clipped layers stacked on this one
        // share its base, so a toggle of this one moves theirs (they re-plan when it changes).
        if let Some(base) = scene.clip_below_base(id) {
            self.mask_consumers
                .entry(base.to_string())
                .or_default()
                .insert(id.to_string());
        }
        if node.truthy("clipToBelow") {
            self.any_clipped = true;
            // The layer directly below it too: a list edit that puts another layer there changes the base without
            // naming this one, and names (re-plans) that slot's old layer.
            if let Some(siblings) = scene.sibling_ids(id) {
                if let Some(at) = siblings.iter().position(|s| s == id).filter(|&at| at > 0) {
                    self.mask_consumers
                        .entry(siblings[at - 1].to_string())
                        .or_default()
                        .insert(id.to_string());
                }
            }
            self.note_clip_run(scene, id);
        }
    }

    /// Clip to the layer below: the clipped layers stacked directly on `id` (the run above it, up to the first layer
    /// that isn't clipped) re-plan when it changes. A clipped `id` shares its base with them, so its toggle moves
    /// theirs; an unclipped one is their base, which an edit that adds or moves a layer under a clipped run makes it
    /// without the run's layers being edited (the edit's own nodes run this, [`CameraSlot`]'s dirty pass).
    fn note_clip_run(&mut self, scene: &Scene, id: &str) {
        let Some(siblings) = scene.sibling_ids(id) else {
            return;
        };
        let Some(at) = siblings.iter().position(|s| s == id) else {
            return;
        };
        for above in &siblings[at + 1..] {
            if !scene.canonical_or_transient(above).is_some_and(|n| n.truthy("clipToBelow")) {
                break;
            }
            self.mask_consumers
                .entry(id.to_string())
                .or_default()
                .insert(above.to_string());
        }
    }

    fn is_empty(&self) -> bool {
        self.blend_consumers.is_empty() && self.mask_consumers.is_empty() && !self.any_clipped
    }

    fn consumers_of<'a>(&'a self, id: &str) -> impl Iterator<Item = &'a String> + 'a {
        self.blend_consumers
            .get(id)
            .into_iter()
            .chain(self.mask_consumers.get(id))
            .flatten()
    }

    /// Add to `dirty` the consumers its edits reach, each as an edit of its own,
    /// so it re-plans through the lane a direct edit takes: a node feeds a
    /// consumer when it, an ancestor (a leaf inside a source group) or a
    /// descendant (a moved group with a source inside) is that consumer's
    /// source. A structural edit's consumer re-plans structurally (a source that
    /// went away or appeared takes a mask out of the table or puts one in); a
    /// property edit's as a property edit, which reaches its whole subtree (a
    /// mask group's descendants bake its mask). A consumer can be a source in
    /// turn. Only `dirty[from..]` are walked (the rest were), and `seeds` as
    /// structural edits without joining `dirty` (the nodes of a re-planned
    /// top-level range: a source moved to another page or out of a group
    /// with the list edit that moved it). Returns whether `dirty` changed;
    /// `None` = a runaway (corrupt) walk.
    fn expand(
        &self,
        scene: &Scene,
        dirty: &mut Vec<(String, bool)>,
        from: usize,
        seeds: &[String],
    ) -> Option<bool> {
        if self.blend_consumers.is_empty() && self.mask_consumers.is_empty() {
            return Some(false);
        }
        let mut at: HashMap<String, usize> = dirty
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.clone(), i))
            .collect();
        let mut queue: Vec<(String, bool)> = dirty[from..].to_vec();
        queue.extend(seeds.iter().map(|s| (s.clone(), true)));
        let mut changed = false;
        let mut found: Vec<String> = Vec::new();
        while let Some((id, structural)) = queue.pop() {
            found.clear();
            // The node and its ancestors.
            let mut seen: HashSet<&str> = HashSet::new();
            let mut cursor: &str = id.as_str();
            loop {
                if !seen.insert(cursor) {
                    return None; // a cycle
                }
                found.extend(self.consumers_of(cursor).cloned());
                match scene
                    .group_parent_of(cursor)
                    .map(String::as_str)
                    .or_else(|| scene.transient_parent_of().get(cursor).map(String::as_str))
                {
                    Some(p) => cursor = p,
                    None => break,
                }
            }
            // Its descendants.
            let mut sub = Vec::new();
            CameraSlot::walk(scene, &id, &mut sub)?;
            for d in sub.iter().filter(|d| **d != id) {
                found.extend(self.consumers_of(d).cloned());
            }
            for c in found.drain(..) {
                match at.get(&c) {
                    Some(&j) => {
                        if structural && !dirty[j].1 {
                            dirty[j].1 = true;
                            changed = true;
                            queue.push(dirty[j].clone());
                        }
                    }
                    None => {
                        at.insert(c.clone(), dirty.len());
                        changed = true;
                        queue.push((c.clone(), structural));
                        dirty.push((c, structural));
                    }
                }
            }
        }
        Some(changed)
    }
}

/// The display list's `text-mask-source` input (`clip_mask_source_ids`), kept
/// current as groups change: group id → the mask node it names.
#[derive(Debug, Default, Clone)]
struct ClipSources {
    by_owner: HashMap<String, String>,
    counts: HashMap<String, usize>,
}

impl ClipSources {
    fn seed(scene: &Scene) -> ClipSources {
        let mut c = ClipSources::default();
        for id in scene.node_order() {
            c.recheck(scene, id);
        }
        c
    }

    fn recheck(&mut self, scene: &Scene, id: &str) {
        if let Some(old) = self.by_owner.remove(id) {
            if let Some(n) = self.counts.get_mut(&old) {
                *n -= 1;
                if *n == 0 {
                    self.counts.remove(&old);
                }
            }
        }
        if let Some(target) = moda_scene::display_list::clip_mask_source_of(scene, id) {
            *self.counts.entry(target.clone()).or_insert(0) += 1;
            self.by_owner.insert(id.to_string(), target);
        }
    }

    fn set(&self) -> HashSet<String> {
        self.counts.keys().cloned().collect()
    }
}

struct Segment {
    seg: CameraSegment,
    roots: Vec<RootSpan>,
    flags: SegmentFlags,
    /// Mask owner → its entry in the segment's table (relative), built on the
    /// first unit re-plan that needs it and dropped when a region re-plans the
    /// table.
    mask_owners: Option<HashMap<String, usize>>,
}

/// The parts of a slot that follow the scene, not the camera: kept when a camera
/// move past the cull window rebuilds the plan at the same scene generation, so
/// the new slot's seed is a walk of its display list, not of every node's
/// predicates.
pub(crate) struct SceneIndex {
    pub generation: u64,
    visible_pages: Vec<usize>,
    /// Per segment: its kind, its top-level roots' display-list spans (their unit
    /// spans follow the next plan's units) and its flags.
    segments: Vec<(SegmentKind, Vec<RootSpan>, SegmentFlags)>,
    /// Node → (segment, display-list index): the display list is the scene's,
    /// not the window's.
    by_node: HashMap<String, (usize, usize)>,
    deps: Deps,
    scene_flags: SceneFlags,
    clip_sources: ClipSources,
}

/// What a patch did.
#[derive(Debug)]
pub(crate) enum SlotOutcome {
    /// Nothing changed since the plan was built: serve it as is.
    Unchanged,
    /// Patched; serve the slot's plan (`units` re-planned). `drop_after`: a
    /// re-planned unit bakes the shader clock, so the slot serves this frame
    /// and the next one rebuilds.
    Patched { units: usize, drop_after: bool },
    /// The slot cannot serve: rebuild (and re-seed).
    Fallback(&'static str),
}

/// What [`CameraSlot::prepare`] found: the scene flags this frame plans under,
/// and the patch [`CameraSlot::apply`] will make.
pub(crate) struct Prepared {
    /// (has outlines, has curves): the zoom-band keying of tessellation.
    pub scene_flags: (bool, bool),
    /// Whether the visible segments split (the plan is camera-dependent).
    pub camera_dependent: bool,
    /// Whether the visible segments hold layer captures.
    pub has_captures: bool,
    dirty: Vec<(String, bool)>,
    regions: Vec<Region>,
    /// Unchanged nodes whose units re-plan in place because their segment's split context changed
    /// ([`CameraSlot::respec_segment`]).
    respec: Vec<String>,
}

impl Prepared {
    /// Also re-plan the units of these unchanged nodes, in place (a refinement: a change of zoom band re-plans
    /// its banded units, a budget's worth this frame).
    pub(crate) fn add_unit_refines(&mut self, ids: impl IntoIterator<Item = String>) {
        self.respec.extend(ids);
    }
}

/// A structural re-plan of one segment: its top-level roots `old` (a range of
/// the slot's root list) become `new` (ids of the scene's current list).
struct Region {
    segment: usize,
    old: std::ops::Range<usize>,
    new: Vec<String>,
}

/// Regions that touch, as one region each (they never overlap: a whole-root re-plan inside the diff is the diff's).
/// Each splice shifts the roots after it, so an insert just before a whole-root re-plan (the diff `k..k` and the root
/// `k..k+1`, both starting at `k`) would otherwise re-plan whatever the insert put at `k`: the root was then planned
/// twice and the inserted node lost.
fn merge_touching(mut regions: Vec<Region>) -> Vec<Region> {
    regions.sort_by(|a, b| {
        (a.segment, a.old.start, a.old.end).cmp(&(b.segment, b.old.start, b.old.end))
    });
    let mut merged: Vec<Region> = Vec::with_capacity(regions.len());
    for r in regions {
        match merged.last_mut() {
            Some(m) if m.segment == r.segment && r.old.start <= m.old.end => {
                debug_assert_eq!(r.old.start, m.old.end, "overlapping regions");
                m.old.end = r.old.end.max(m.old.end);
                m.new.extend(r.new);
            }
            _ => merged.push(r),
        }
    }
    merged
}

/// The retained camera plan (see the module doc).
pub(crate) struct CameraSlot {
    pub visible_pages: Vec<usize>,
    pub opts_key: u64,
    /// The options key without the zoom band (`None`: unknown): a change of band alone keeps the slot and refines
    /// its units at the new band ([`CameraSlot::band_refine_candidates`]).
    pub bandless_key: Option<u64>,
    pub resource_key: u64,
    /// The cull window the plan was built with (`None` = unculled).
    pub cull: Option<Aabb>,
    /// The capture view its layer captures were planned under.
    pub capture: Option<CaptureView>,
    /// The camera its split items are projected for.
    cam: CameraParams,
    pub built_at_generation: u64,
    /// `Some(shader clock bits)`: the plan holds items baking a clock the present can't re-thread, so it serves
    /// only at that clock (`residency::camera_plan_retained`).
    pub clock: Option<u64>,
    pub plan: Arc<CoveragePlan>,
    pub warnings: Arc<PlanWarnings>,
    segments: Vec<Segment>,
    /// Node id → (segment, display-list index of its primary item); its unit is
    /// found by binary search ([`Self::locate`]), so the index follows the
    /// display list (the scene), not the units (the cull window).
    by_node: HashMap<String, (usize, usize)>,
    deps: Deps,
    scene_flags: SceneFlags,
    clip_sources: ClipSources,
    /// (has outlines, has curves) the plan was built under.
    built_scene_flags: (bool, bool),
    /// Marginal heap of the plan (`residency::plan_bytes`), kept by deltas.
    pub bytes: usize,
}

/// The camera fields a split projection reads.
fn same_projection(a: &CameraParams, b: &CameraParams) -> bool {
    a.camera_x == b.camera_x
        && a.camera_y == b.camera_y
        && a.zoom == b.zoom
        && a.viewport_w == b.viewport_w
        && a.viewport_h == b.viewport_h
        && a.dpr == b.dpr
}

/// A segment's top-level ids in the scene as it is now (a page's background
/// first).
fn top_level_ids(scene: &Scene, kind: SegmentKind) -> Option<Vec<String>> {
    match kind {
        SegmentKind::Page(i) => {
            let page_id = scene.doc().pages.get(i)?.id().to_string();
            let mut ids = vec![compute_page_bg_id(&page_id)];
            ids.extend(scene.page_top_level_ids(&page_id));
            Some(ids)
        }
        SegmentKind::CanvasRoot => Some(scene.doc().canvas_root_node_ids.clone()),
    }
}

/// The flattened z-order of one top-level id (a page background visits as
/// itself, as the page's z-order starts with it).
fn flattened_root(scene: &Scene, root: &str, out: &mut Vec<String>) {
    if is_page_bg_id(root) {
        if scene.effective_node(root).is_some() {
            out.push(root.to_string());
        }
    } else {
        out.extend(scene.flattened_z_order_for_ids(&[root.to_string()]));
    }
}

/// The top-level ancestor of `id` in the scene as it is now.
fn top_of<'a>(scene: &'a Scene, id: &'a str) -> &'a str {
    let mut cursor = id;
    for _ in 0..256 {
        match scene
            .group_parent_of(cursor)
            .map(String::as_str)
            .or_else(|| scene.transient_parent_of().get(cursor).map(String::as_str))
        {
            // A page is a parent in the index but not a node: its child is top level.
            Some(p) if scene.canonical_or_transient(p).is_some() => cursor = p,
            _ => return cursor,
        }
    }
    cursor
}

/// Whether a unit's nodes make its items camera- or capture-dependent: a node
/// in the segment's backdrop or split set (a background blur, or a blend split
/// — its own blend mode, or its paint stack's at its effective opacity) has its
/// split items projected into the camera target; one in its capture set is
/// planned under the capture view.
fn unit_flags(flags: &SegmentFlags, backdrop: bool, layer_blur: bool, dl: &[PaintItem]) -> u8 {
    if !backdrop && !layer_blur {
        return 0;
    }
    let mut out = 0;
    for item in dl.iter().filter(|i| i.is_primary_node_item) {
        let id = item.node_id.as_str();
        if backdrop && (flags.backdrop.contains(id) || flags.split.contains(id)) {
            out |= UNIT_CAMERA;
        }
        if layer_blur && flags.capture.contains(id) {
            out |= UNIT_CAPTURE;
        }
    }
    out
}

/// Top-level root index of every node in these roots' flattened z-order.
fn root_map(scene: &Scene, roots: &[String]) -> HashMap<String, usize> {
    let mut root_of: HashMap<String, usize> = HashMap::new();
    let mut flat = Vec::new();
    for (ri, r) in roots.iter().enumerate() {
        flat.clear();
        flattened_root(scene, r, &mut flat);
        for id in flat.drain(..) {
            root_of.insert(id, ri);
        }
    }
    root_of
}

/// Tile `roots` (in order) over a display-list range and its units:
/// `dl[dl_base..dl_end]`, `units[unit_base..]` whose `dl_start`s lie in the
/// range. `None` when the items don't tile by root (not contiguous, an
/// unknown node) or a unit straddles two roots.
fn tile_roots(
    scene: &Scene,
    roots: &[String],
    dl: &[PaintItem],
    dl_base: usize,
    dl_end: usize,
    units: &[PlanUnit],
    unit_base: usize,
) -> Option<Vec<RootSpan>> {
    tile_roots_with(&root_map(scene, roots), roots, dl, dl_base, dl_end, units, unit_base)
}

/// [`tile_roots`] with the roots' node → root map already built.
fn tile_roots_with(
    root_of: &HashMap<String, usize>,
    roots: &[String],
    dl: &[PaintItem],
    dl_base: usize,
    dl_end: usize,
    units: &[PlanUnit],
    unit_base: usize,
) -> Option<Vec<RootSpan>> {
    let mut spans: Vec<RootSpan> = roots
        .iter()
        .map(|id| RootSpan {
            id: id.clone(),
            dl_start: dl_base,
            dl_end: dl_base,
            unit_start: unit_base,
            unit_end: unit_base,
        })
        .collect();
    if spans.is_empty() {
        return (dl_base == dl_end).then_some(spans);
    }
    let mut cursor = 0usize;
    spans[0].dl_start = dl_base;
    for (k, item) in dl.iter().enumerate().take(dl_end).skip(dl_base) {
        if !item.is_primary_node_item {
            continue;
        }
        let ri = *root_of.get(&item.node_id)?;
        if ri < cursor {
            return None;
        }
        while cursor < ri {
            spans[cursor].dl_end = k;
            cursor += 1;
            spans[cursor].dl_start = k;
        }
    }
    while cursor < spans.len() {
        spans[cursor].dl_end = dl_end;
        cursor += 1;
        if cursor < spans.len() {
            spans[cursor].dl_start = dl_end;
        }
    }
    let mut ui = unit_base;
    for r in &mut spans {
        r.unit_start = ui;
        while ui < units.len() && units[ui].dl_start < r.dl_end {
            if units[ui].dl_start < r.dl_start || units[ui].dl_end > r.dl_end {
                return None;
            }
            ui += 1;
        }
        r.unit_end = ui;
    }
    Some(spans)
}

/// Point roots' unit spans at `units` (their display-list spans already hold).
fn assign_unit_spans(roots: &mut [RootSpan], units: &[PlanUnit]) -> Option<()> {
    let mut ui = 0usize;
    for r in roots.iter_mut() {
        r.unit_start = ui;
        while ui < units.len() && units[ui].dl_start < r.dl_end {
            if units[ui].dl_start < r.dl_start || units[ui].dl_end > r.dl_end {
                return None;
            }
            ui += 1;
        }
        r.unit_end = ui;
    }
    (ui == units.len()).then_some(())
}

/// Map a scratch mask table (a re-planned range's own, in first-use order) onto
/// a segment's table `table` (`base` = where it starts in the combined one).
/// A scratch mask whose owner has an entry in `table[..limit]` must equal it,
/// or replaces it when `replaceable(owner)`; any other is appended when
/// `append`, else the mapping fails. Returns scratch index → combined index.
fn map_masks(
    scratch: &[MaskPlan],
    table: &mut Vec<MaskPlan>,
    base: usize,
    limit: usize,
    append: bool,
    replaceable: &dyn Fn(&str) -> bool,
) -> Option<Vec<usize>> {
    let mut owners: HashMap<String, usize> = HashMap::with_capacity(limit);
    for (j, m) in table[..limit].iter().enumerate() {
        owners.entry(m.owner_node_id.clone()).or_insert(j);
    }
    let mut map: Vec<usize> = Vec::with_capacity(scratch.len());
    for m in scratch {
        let mut mapped = m.clone();
        mapped.ancestor = match m.ancestor {
            Some(a) => Some(*map.get(a)?),
            None => None,
        };
        match owners.get(&m.owner_node_id) {
            Some(&j) => {
                if table[j] != mapped {
                    if !replaceable(&m.owner_node_id) {
                        return None;
                    }
                    table[j] = mapped;
                }
                map.push(base + j);
            }
            None if append => {
                table.push(mapped);
                owners.insert(m.owner_node_id.clone(), table.len() - 1);
                map.push(base + table.len() - 1);
            }
            None => return None,
        }
    }
    Some(map)
}

/// Point items planned against a scratch mask table at the combined table
/// (`map`). Items inside a layer-capture group keep segment-local ids, as a
/// full build leaves them (the combine rebases only top-level items; a capture
/// renders its items with no mask table), so they map to `map[j] - base`.
fn remap_item_masks(items: &mut [DrawItem], map: &[usize], base: usize) -> bool {
    remap_masks_at(items, map, 0, true) && {
        let mut ok = true;
        for item in items.iter_mut() {
            if let DrawItem::LayerBlurComposite { group, .. } = item {
                if group_has_masks(group) {
                    let g = Arc::make_mut(group);
                    ok &= remap_nested(&mut g.items, map, base);
                }
            }
        }
        ok
    }
}

fn group_has_masks(g: &crate::seams::LayerBlurGroup) -> bool {
    // A capture with its own mask table: its items index that table.
    if !g.masks.is_empty() {
        return false;
    }
    g.items.iter().any(|i| match i {
        DrawItem::Body { mask, .. } => mask.is_some(),
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        DrawItem::Text(_) => false,
        DrawItem::LayerBlurComposite { group, .. } => group_has_masks(group),
        _ => false,
    })
}

fn remap_nested(items: &mut [DrawItem], map: &[usize], base: usize) -> bool {
    if !remap_masks_at(items, map, base, false) {
        return false;
    }
    for item in items.iter_mut() {
        if let DrawItem::LayerBlurComposite { group, .. } = item {
            if group_has_masks(group) && !remap_nested(&mut Arc::make_mut(group).items, map, base) {
                return false;
            }
        }
    }
    true
}

/// Map the top-level mask refs of `items` through `map`, less `base`.
fn remap_masks_at(items: &mut [DrawItem], map: &[usize], base: usize, _top: bool) -> bool {
    for item in items.iter_mut() {
        let Some(id) = item_mask_id(item) else { continue };
        match map.get(id as usize) {
            Some(&j) => set_item_mask_id(item, (j - base) as u64),
            None => return false,
        }
    }
    true
}

/// The mask a draw item is clipped by (bodies, text and capture composites
/// carry one).
pub(crate) fn item_mask_id(item: &DrawItem) -> Option<u64> {
    match item {
        DrawItem::Body { mask, .. } => mask.as_ref().map(|m| m.mask_id),
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        DrawItem::Text(_) => None,
        DrawItem::LayerBlurComposite { mask, .. } => mask.as_ref().map(|m| m.mask_id),
        _ => None,
    }
}

/// Point a masked draw item at mask `id`. A text batch is shared (`Arc`), and the prepared frame keeps its prepared
/// text while the batch is the same `Arc`: it is copied only when the id actually changes.
pub(crate) fn set_item_mask_id(item: &mut DrawItem, id: u64) {
    match item {
        DrawItem::Body { mask: Some(m), .. } => m.mask_id = id,
        DrawItem::LayerBlurComposite { mask: Some(m), .. } => m.mask_id = id,
        // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
        _ => {}
    }
}

/// Whether moving `item` by `dz` in render-z (its z, and the depth baked from
/// it) gives exactly what planning it at the new z does: every depth it bakes
/// is `linear_depth` of its z, and its z moves exactly — an integer or quarter
/// z always does; another fraction (a copy's or a placed brush's share of a
/// slot) only within one binade, where `(z + f) + dz` rounds as `(z + dz) + f`.
fn z_shiftable(item: &DrawItem, dz: i64) -> bool {
    let z_ok = |z: f64| {
        let a = z.floor();
        if (z * 4.0).fract() == 0.0 {
            return true;
        }
        let b = a + dz as f64;
        (a >= 1.0 && b >= 1.0 && (a as u64).ilog2() == (b as u64).ilog2()) || (a == 0.0 && dz > 0)
    };
    let depth_ok = |z: f64, d: f32| crate::depth::linear_depth(z) as f32 == d;
    match item {
        DrawItem::Body { z, instance, paint, .. } => {
            z_ok(*z)
                && depth_ok(*z, instance.path_info[3])
                && match paint {
                    crate::seams::BodyPaint::Pattern(p) => depth_ok(*z, p.instance.path_info[3]),
                    crate::seams::BodyPaint::Gradient(g) | crate::seams::BodyPaint::PostShaderGradientStroke(g) => {
                        depth_ok(*z, g.path_info[2])
                    }
                    _ => true,
                }
        }
        DrawItem::Shadow { z, instance, .. } => z_ok(*z) && depth_ok(*z, instance.shape_data[3]),
        DrawItem::SolidShapeSdf { z, instance, .. } => z_ok(*z) && depth_ok(*z, instance.dimensions[2]),
        DrawItem::GradientShapeSdf { z, instance, .. } => {
            z_ok(*z) && depth_ok(*z, instance.dimensions[2])
        }
        // Text bakes its depth at present, from `z`.
        DrawItem::Text(b) => z_ok(b.z),
        DrawItem::LayerBlurComposite { z, group, .. } => {
            z_ok(*z)
                && group.items.iter().all(|i| z_shiftable(i, dz))
                && group.coverage.iter().all(|i| z_shiftable(i, dz))
        }
        DrawItem::BackdropSplit(_) | DrawItem::PageFill { .. } => true,
    }
}

/// Move `item` by `dz` in render-z ([`z_shiftable`] must hold).
fn shift_item_z(item: &mut DrawItem, dz: i64) {
    let d = dz as f64;
    let depth = |z: f64| crate::depth::linear_depth(z) as f32;
    match item {
        DrawItem::Body { z, instance, paint, .. } => {
            *z += d;
            instance.path_info[3] = depth(*z);
            // A pattern or gradient body draws from its paint's own instance, which bakes the depth too (an image
            // fill kept its old depth after a z shift: found by the raster spike's paint layers in the fuzz).
            match paint {
                crate::seams::BodyPaint::Pattern(p) => p.instance.path_info[3] = depth(*z),
                crate::seams::BodyPaint::Gradient(g) | crate::seams::BodyPaint::PostShaderGradientStroke(g) => g.path_info[2] = depth(*z),
                _ => {}
            }
        }
        DrawItem::Shadow { z, instance, .. } => {
            *z += d;
            instance.shape_data[3] = depth(*z);
        }
        DrawItem::SolidShapeSdf { z, instance, .. } => {
            *z += d;
            instance.dimensions[2] = depth(*z);
        }
        DrawItem::GradientShapeSdf { z, instance, .. } => {
            *z += d;
            instance.dimensions[2] = depth(*z);
        }
        DrawItem::Text(b) if dz != 0 => std::sync::Arc::make_mut(b).z += d,
        DrawItem::Text(_) => {}
        DrawItem::LayerBlurComposite { z, group, .. } => {
            *z += d;
            let g = Arc::make_mut(group);
            // The noise's silhouette items too (they bake the depth as the body's do): a noise-filtered layer
            // shifted in render-z kept its silhouette at the old z (found by the slot fuzz's shader backgrounds).
            for i in g.items.iter_mut().chain(g.coverage.iter_mut()) {
                shift_item_z(i, dz);
            }
        }
        DrawItem::BackdropSplit(_) | DrawItem::PageFill { .. } => {}
    }
}

/// Shift the mask references of a later segment's items after an earlier
/// segment's table grew or shrank by `delta` (entries from `from` on moved).
fn shift_item_masks(items: &mut [DrawItem], from: usize, delta: i64) {
    for item in items.iter_mut() {
        if let Some(id) = item_mask_id(item) {
            if id as usize >= from {
                set_item_mask_id(item, (id as i64 + delta) as u64);
            }
        }
    }
}

/// Merge `rest` (a range re-planned right after `head`) into `head`: its
/// display-list indices and items follow `head`'s, and its scratch masks map
/// onto `head`'s table by owner (first-use order is kept). `head_dl` is the
/// length of `head`'s display list.
fn merge_ranges(
    head: &mut RangeBuild,
    head_dl: usize,
    rest: RangeBuild,
) -> Result<(), &'static str> {
    let base_masks = head.masks.len();
    let item_off = head.items.len();
    let mut map: Vec<usize> = Vec::with_capacity(rest.masks.len());
    // Merged table length after each of `rest`'s masks.
    let mut len_after: Vec<usize> = Vec::with_capacity(rest.masks.len());
    for m in &rest.masks {
        let mut mapped = m.clone();
        mapped.ancestor = match m.ancestor {
            Some(a) => Some(*map.get(a).ok_or("mask ancestor")?),
            None => None,
        };
        match head
            .masks
            .iter()
            .position(|x| x.owner_node_id == m.owner_node_id)
        {
            Some(j) => {
                if head.masks[j] != mapped {
                    return Err("one owner, two masks");
                }
                map.push(j);
            }
            None => {
                head.masks.push(mapped);
                map.push(head.masks.len() - 1);
            }
        }
        len_after.push(head.masks.len());
    }
    let mut items = rest.items;
    if !remap_item_masks(&mut items, &map, 0) {
        return Err("mask ref");
    }
    head.items.extend(items);
    for mut unit in rest.units {
        unit.dl_start += head_dl;
        unit.dl_end += head_dl;
        unit.item_start += item_off;
        unit.masks_before = match unit.masks_before {
            0 => base_masks,
            k => len_after[k - 1],
        };
        head.units.push(unit);
    }
    head.end_z = rest.end_z;
    head.mask_warnings.absorb(rest.mask_warnings);
    Ok(())
}

impl CameraSlot {
    /// Seed a slot from a full build and what it recorded. `Err(why)` when the
    /// recorded plan cannot be indexed (a display list that doesn't tile by
    /// top-level subtree): the lane then rebuilds on every change, as before.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn seed(
        scene: &Scene,
        generation: u64,
        plan: Arc<CoveragePlan>,
        warnings: Arc<PlanWarnings>,
        aux: CameraPlanAux,
        visible_pages: &[usize],
        opts_key: u64,
        resource_key: u64,
        cull: Option<Aabb>,
        capture: Option<CaptureView>,
        cam: CameraParams,
        built_scene_flags: (bool, bool),
        bytes: usize,
        reuse: Option<SceneIndex>,
    ) -> Result<CameraSlot, &'static str> {
        // The scene-side index of an earlier slot at this generation and view.
        let reuse = reuse.filter(|x| {
            x.generation == generation
                && x.visible_pages == visible_pages
                && x.segments.len() == aux.segments.len()
                && x.segments.iter().zip(&aux.segments).all(|(a, b)| a.0 == b.kind)
        });
        let (mut reused_segments, reused_rest, reused_nodes) = match reuse {
            Some(x) => (
                x.segments.into_iter().map(Some).collect::<Vec<_>>(),
                Some((x.deps, x.scene_flags, x.clip_sources)),
                Some(x.by_node),
            ),
            None => (Vec::new(), None, None),
        };
        let index_nodes = reused_nodes.is_none();
        let mut segments = Vec::with_capacity(aux.segments.len());
        let mut by_node: HashMap<String, (usize, usize)> = reused_nodes.unwrap_or_default();
        for (si, mut seg) in aux.segments.into_iter().enumerate() {
            let dl = seg.dl.clone();
            let (roots, flags) = match reused_segments.get_mut(si).and_then(Option::take) {
                // The same display list: its root spans hold; the units are new.
                Some((_, mut roots, flags)) => {
                    if roots.last().map_or(0, |r| r.dl_end) != dl.items.len() {
                        return Err("seed: the scene index is not this display list's");
                    }
                    assign_unit_spans(&mut roots, &seg.units)
                        .ok_or("seed: units do not tile the roots")?;
                    (roots, flags)
                }
                None => {
                    let tops = top_level_ids(scene, seg.kind).ok_or("seed: page gone")?;
                    let root_of = root_map(scene, &tops);
                    let roots =
                        tile_roots_with(&root_of, &tops, &dl.items, 0, dl.items.len(), &seg.units, 0)
                            .ok_or("seed: the display list does not tile by root")?;
                    if roots.last().map_or(0, |r| r.unit_end) != seg.units.len() {
                        return Err("seed: units past the last root");
                    }
                    let mut f = SegmentFlags::default();
                    for item in dl.items.iter().filter(|i| i.is_primary_node_item) {
                        f.add(scene, &item.node_id);
                    }
                    (roots, f)
                }
            };
            let (backdrop, layer_blur) = (seg.backdrop.is_some(), seg.layer_blur);
            for unit in seg.units.iter_mut() {
                // A culled range draws nothing a camera move could re-project.
                if !unit.culled {
                    let items = &dl.items[unit.dl_start..unit.dl_end];
                    unit.flags = unit_flags(&flags, backdrop, layer_blur, items);
                }
            }
            if index_nodes {
                for (d, item) in dl.items.iter().enumerate() {
                    if item.is_primary_node_item {
                        by_node.insert(item.node_id.clone(), (si, d));
                    }
                }
            } else if cfg!(debug_assertions) {
                // The kept index was patched along with the slot's display list;
                // the fresh build's must be the same sequence.
                for (d, item) in dl.items.iter().enumerate() {
                    if item.is_primary_node_item && by_node.get(&item.node_id) != Some(&(si, d)) {
                        return Err("seed: the kept node index disagrees with the display list");
                    }
                }
            }
            // The flags must reproduce the contexts the build used.
            if flags.backdrop_spec(seg.kind) != seg.backdrop
                || flags.capture.is_empty() == seg.layer_blur
            {
                return Err("seed: segment flags disagree with the build's contexts");
            }
            segments.push(Segment {
                seg,
                roots,
                flags,
                mask_owners: None,
            });
        }
        let (deps, scene_flags, clip_sources) = match reused_rest {
            Some(rest) => rest,
            None => {
                let mut deps = Deps::default();
                let mut scene_flags = SceneFlags::default();
                for id in scene.node_order().iter().chain(scene.transient_by_id().keys()) {
                    deps.note(scene, id);
                    scene_flags.recheck(scene, id);
                }
                (deps, scene_flags, ClipSources::seed(scene))
            }
        };
        let mut slot = CameraSlot {
            visible_pages: visible_pages.to_vec(),
            opts_key,
            bandless_key: None,
            resource_key,
            cull,
            capture,
            cam,
            built_at_generation: generation,
            clock: None,
            plan,
            warnings,
            segments,
            by_node,
            deps,
            scene_flags,
            clip_sources,
            built_scene_flags,
            bytes,
        };
        if slot.scene_flags.has(scene) != built_scene_flags {
            return Err("seed: scene flags disagree with the build's");
        }
        // The recorded warnings must reassemble to the build's.
        let mut a = slot.assemble_warnings();
        let mut b = (*slot.warnings).clone();
        a.skipped.sort();
        a.dropped.sort();
        b.skipped.sort();
        b.dropped.sort();
        if a.skipped != b.skipped || a.dropped != b.dropped {
            return Err("seed: warnings do not reassemble");
        }
        slot.warnings = Arc::new(slot.assemble_warnings());
        Ok(slot)
    }

    /// The scene-side index, for the next seed at this generation (a camera
    /// move past the cull window).
    pub(crate) fn into_index(self) -> SceneIndex {
        SceneIndex {
            generation: self.built_at_generation,
            visible_pages: self.visible_pages,
            segments: self
                .segments
                .into_iter()
                .map(|s| (s.seg.kind, s.roots, s.flags))
                .collect(),
            by_node: self.by_node,
            deps: self.deps,
            scene_flags: self.scene_flags,
            clip_sources: self.clip_sources,
        }
    }

    /// After a change of zoom band alone: every unit whose items could differ at another band (a tessellated
    /// body: curves, round joins and caps, strands, outlines; text; or nothing drawn at this band), by its first node,
    /// with the
    /// canvas box of its bodies (viewport-first refinement). They re-plan as refinements, a budget's worth a frame;
    /// until then they draw what they drew at the old band, as a rebuild's stale-while-refine would. SDF shapes,
    /// shadows and page fills don't depend on the band.
    pub(crate) fn band_refine_candidates(&self) -> Vec<(String, [f64; 4])> {
        let mut out = Vec::new();
        for s in &self.segments {
            let at = s.seg.item_start + s.seg.prefix;
            for unit in &s.seg.units {
                if unit.culled {
                    continue;
                }
                let items = &self.plan.items[at + unit.item_start..at + unit.item_start + unit.item_len];
                // A unit that drew nothing may draw at another band (a stroke whose strands flattened away).
                let mut banded = items.is_empty();
                let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for item in items {
                    match item {
                        DrawItem::Body { geometry, instance, .. } => {
                            banded = true;
                            let [x, y, w, h] = geometry.bounds;
                            let m = instance.model_matrix;
                            for (px, py) in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)] {
                                let (cx, cy) = (
                                    (m[0] * px + m[4] * py + m[12]) as f64,
                                    (m[1] * px + m[5] * py + m[13]) as f64,
                                );
                                b = [b[0].min(cx), b[1].min(cy), b[2].max(cx), b[3].max(cy)];
                            }
                        }
                        DrawItem::Text(_) => banded = true,
                        _ => {}
                    }
                }
                if !banded {
                    continue;
                }
                let Some(id) = s.seg.dl.items[unit.dl_start..unit.dl_end]
                    .iter()
                    .find(|i| i.is_primary_node_item)
                    .map(|i| i.node_id.clone())
                else {
                    continue;
                };
                if !(b[0] <= b[2] && b.iter().all(|v| v.is_finite())) {
                    b = [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY];
                }
                out.push((id, b));
            }
        }
        out
    }

    /// Whether the plan has units a camera move re-plans.
    pub(crate) fn camera_dependent(&self) -> bool {
        self.segments.iter().any(|s| s.seg.backdrop.is_some())
    }

    /// The (segment, unit) holding `id`'s primary display-list item.
    fn locate(&self, id: &str) -> Option<(usize, usize)> {
        let &(si, d) = self.by_node.get(id)?;
        let units = &self.segments.get(si)?.seg.units;
        let u = units.partition_point(|x| x.dl_end <= d);
        (u < units.len() && units[u].dl_start <= d).then_some((si, u))
    }

    /// The (segment, root) holding unit `u` of segment `si`.
    fn root_of_unit(&self, si: usize, u: usize) -> Option<usize> {
        let roots = &self.segments[si].roots;
        let ri = roots.partition_point(|r| r.unit_end <= u);
        (ri < roots.len() && roots[ri].unit_start <= u).then_some(ri)
    }

    /// The nodes a property edit to `id` re-plans: `id`, and for a group, frame
    /// or transient host its whole subtree. `None` = a runaway (corrupt) walk.
    fn walk(scene: &Scene, id: &str, out: &mut Vec<String>) -> Option<()> {
        let Some(node) = scene.effective_node(id) else {
            return Some(());
        };
        if !(is_parent_node(node) || scene.transient_by_parent().contains_key(id)) {
            out.push(id.to_string());
            return Some(());
        }
        let cap = scene.node_order().len() + scene.transient_by_id().len() + 16;
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = vec![id];
        while let Some(n) = stack.pop() {
            if seen.len() > cap {
                return None;
            }
            if !seen.insert(n) {
                continue;
            }
            out.push(n.to_string());
            if let Some(kids) = scene.children_by_parent().get(n) {
                stack.extend(kids.iter().map(String::as_str));
            }
            if let Some(kids) = scene.transient_by_parent().get(n) {
                stack.extend(kids.iter().map(String::as_str));
            }
        }
        Some(())
    }

    /// Whether re-planning the units of `walked` (the nodes one property edit
    /// re-plans) without a cull window is exactly what a culled full build draws:
    /// nothing in the edited subtree was culled, each re-planned node was drawn,
    /// and its paint still meets the window through every clipping frame above it —
    /// so every node there is still drawn (a group through any descendant that is).
    /// Culled nodes elsewhere in its top-level subtree are untouched by the edit.
    /// Otherwise the top-level subtree is re-planned whole, under the window.
    fn unit_path_exact(
        &self,
        scene: &Scene,
        si: usize,
        walked: &[String],
        window: &Aabb,
        clip_sources: &HashSet<String>,
    ) -> bool {
        let s = &self.segments[si];
        // A group's edit reaches every node below it, culled ones included (a move can bring one into view, and a
        // culled node has no unit to re-plan): when a culled range starts inside its subtree, the subtree is
        // re-planned whole. Culled nodes elsewhere in the top-level subtree are untouched by the edit.
        if walked.len() > 1 {
            let inside: HashSet<&str> = walked.iter().map(String::as_str).collect();
            let culled_inside = s.seg.units.iter().any(|x| {
                (x.culled || x.part_culled)
                    && s.seg
                        .dl
                        .items
                        .get(x.dl_start)
                        .is_some_and(|i| inside.contains(i.node_id.as_str()))
            });
            if culled_inside {
                return false;
            }
        }
        let mut ok: HashMap<&str, bool> = HashMap::with_capacity(walked.len());
        for id in walked {
            let Some((sj, u)) = self.locate(id.as_str()) else {
                continue; // not drawn (hidden): nothing to re-plan
            };
            if sj != si {
                return false;
            }
            if self.root_of_unit(si, u).is_none() {
                return false;
            }
            // A re-planned node must have been drawn: a culled one's unit is its whole culled range.
            let unit = &s.seg.units[u];
            if unit.culled || unit.part_culled {
                return false;
            }
            let items = moda_scene::display_list::display_list_items_for(
                scene,
                std::slice::from_ref(id),
                0,
                clip_sources,
            );
            let Some(item) = items.iter().find(|i| i.is_primary_node_item) else {
                ok.insert(id.as_str(), false);
                continue;
            };
            let own = crate::cull::item_visual_bounds(scene, item);
            if own[0] > own[2] || own[1] > own[3] {
                continue; // paints nothing itself (a group): decided below
            }
            let mut b = [
                own[0].max(window[0]),
                own[1].max(window[1]),
                own[2].min(window[2]),
                own[3].min(window[3]),
            ];
            // Through every clipping frame above it.
            let escapes = scene
                .effective_node(id)
                .is_some_and(crate::cull::escapes_container_clip);
            let mut cursor: &str = id;
            for _ in 0..256 {
                let Some(p) = scene
                    .group_parent_of(cursor)
                    .or_else(|| scene.transient_parent_of().get(cursor))
                else {
                    break;
                };
                if !escapes {
                    if let Some(c) = scene
                        .effective_node(p)
                        .and_then(|pn| crate::effects::container_clip_aabb(scene, pn))
                    {
                        b = [
                            b[0].max(c[0]),
                            b[1].max(c[1]),
                            b[2].min(c[2]),
                            b[3].min(c[3]),
                        ];
                    }
                }
                cursor = p;
            }
            ok.insert(id.as_str(), b[0] <= b[2] && b[1] <= b[3]);
        }
        if ok.values().any(|v| !v) {
            return false;
        }
        // A re-planned node that paints nothing itself is drawn through a
        // descendant that is.
        for id in walked {
            if ok.contains_key(id.as_str()) || !self.by_node.contains_key(id.as_str()) {
                continue;
            }
            let mut sub = Vec::new();
            if Self::walk(scene, id, &mut sub).is_none() {
                return false;
            }
            if !sub.iter().any(|d| ok.get(d.as_str()) == Some(&true)) {
                return false;
            }
        }
        true
    }

    /// Bring the slot's flags up to the scene and plan the patch. `Err` = fall
    /// back (the caller drops the slot).
    /// Per segment, the structural diff range of its top-level list (old root
    /// indices) and the new roots it becomes: the list diffed against the one
    /// the plan was built from, widened to every root holding a structurally
    /// edited node.
    #[allow(clippy::type_complexity)]
    fn structural_diffs(
        &self,
        scene: &Scene,
        dirty: &[(String, bool)],
    ) -> Result<Vec<Option<(std::ops::Range<usize>, Vec<String>)>>, &'static str> {
        let mut diffs: Vec<Option<(std::ops::Range<usize>, Vec<String>)>> =
            vec![None; self.segments.len()];
        let structural: Vec<&str> = dirty
            .iter()
            .filter(|(_, s)| *s)
            .map(|(id, _)| id.as_str())
            .collect();
        if structural.is_empty() {
            return Ok(diffs);
        }
        let dirty_new: HashSet<&str> = structural
            .iter()
            .filter(|id| scene.canonical_or_transient(id).is_some())
            .map(|id| top_of(scene, id))
            .collect();
        for (si, s) in self.segments.iter().enumerate() {
            let new_tops = top_level_ids(scene, s.seg.kind).ok_or("page gone")?;
            let dirty_old: HashSet<usize> = structural
                .iter()
                .filter_map(|id| self.locate(id))
                .filter(|(sj, _)| *sj == si)
                .filter_map(|(_, u)| self.root_of_unit(si, u))
                .collect();
            let old_tops: Vec<&str> = s.roots.iter().map(|r| r.id.as_str()).collect();
            let (m, k) = (old_tops.len(), new_tops.len());
            let clean = |i: usize, j: usize| {
                old_tops[i] == new_tops[j]
                    && !dirty_old.contains(&i)
                    && !dirty_new.contains(new_tops[j].as_str())
            };
            let mut p = 0;
            while p < m && p < k && clean(p, p) {
                p += 1;
            }
            let mut q = 0;
            while q < m - p && q < k - p && clean(m - 1 - q, k - 1 - q) {
                q += 1;
            }
            if p + q == m && p + q == k {
                continue;
            }
            diffs[si] = Some((p..m - q, new_tops[p..k - q].to_vec()));
        }
        Ok(diffs)
    }

    pub(crate) fn prepare(
        &mut self,
        scene: &Scene,
        dirt: SceneDirt,
        extra_dirty: &[String],
    ) -> Result<Prepared, &'static str> {
        let mut dirty: Vec<(String, bool)> = match dirt {
            SceneDirt::Hard => return Err("hard change"),
            SceneDirt::Unchanged => Vec::new(),
            SceneDirt::Nodes(v) => v,
        };
        // Whether each segment had a backdrop effect as built (its flags change below).
        let backdrop_before: Vec<bool> = self
            .segments
            .iter()
            .map(|s| !s.flags.backdrop.is_empty())
            .collect();
        // Refinement re-plans (`extra_dirty`: a stale outline or deferred text): the node is unchanged, so the cull
        // decides it as before and its unit re-plans in place.
        let mut refine_only: HashSet<&str> = HashSet::new();
        for id in extra_dirty {
            if !dirty.iter().any(|(d, _)| d == id) {
                dirty.push((id.clone(), false));
                refine_only.insert(id.as_str());
            }
        }
        // Scene-wide flags and reverse dependencies, for every node an edit
        // named (a re-expanded host's transients too).
        for (id, _) in &dirty {
            self.scene_flags.recheck(scene, id);
            self.clip_sources.recheck(scene, id);
            self.deps.note(scene, id);
            if let Some(kids) = scene.transient_by_parent().get(id) {
                for k in kids {
                    self.scene_flags.recheck(scene, k);
                    self.deps.note(scene, k);
                }
            }
        }
        // A layer added or moved under a clipped run is its new base (after every note, so a clipped layer added
        // with it counts).
        if self.deps.any_clipped {
            for (id, _) in &dirty {
                self.deps.note_clip_run(scene, id);
            }
        }
        // An edited blend or mask source (or a node inside one, or a group with
        // one inside) changes another node's items: that consumer re-plans too.
        self.deps.expand(scene, &mut dirty, 0, &[]).ok_or("runaway subtree")?;
        // The plan builds a canvas-root segment only while the canvas has roots: a slot built without one
        // (a new, empty document) has nowhere to splice the first root, so it rebuilds.
        if scene.is_freeform()
            && !scene.doc().canvas_root_node_ids.is_empty()
            && !self
                .segments
                .iter()
                .any(|s| s.seg.kind == SegmentKind::CanvasRoot)
        {
            return Err("canvas roots without a segment");
        }
        let clip_sources = self.clip_sources.set();
        // Per segment: old root indices to re-plan whole (under the cull window),
        // and the structural diff range with its new roots.
        let n_seg = self.segments.len();
        let mut whole_roots: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_seg];
        let mut diffs = self.structural_diffs(scene, &dirty)?;
        // A structural range re-plans whole top-level subtrees, some named by no
        // edit (a list edit names the list): a source among them feeds its
        // consumers too, which can widen the ranges.
        let mut walked_to = dirty.len();
        let mut seeded: HashSet<String> = HashSet::new();
        let mut settled = self.deps.is_empty();
        for _ in 0..64 {
            if settled {
                break;
            }
            let mut seeds: Vec<String> = Vec::new();
            for (si, d) in diffs.iter().enumerate() {
                let Some((old, new)) = d else { continue };
                let s = &self.segments[si];
                let mut ids: Vec<String> = Vec::new();
                for r in &s.roots[old.clone()] {
                    ids.extend(
                        s.seg.dl.items[r.dl_start..r.dl_end]
                            .iter()
                            .filter(|i| i.is_primary_node_item)
                            .map(|i| i.node_id.clone()),
                    );
                    Self::walk(scene, &r.id, &mut ids).ok_or("runaway subtree")?;
                }
                for r in new {
                    Self::walk(scene, r, &mut ids).ok_or("runaway subtree")?;
                }
                // The range's top layer is under the first unchanged one: a clipped run there has it as its base now.
                if self.deps.any_clipped {
                    if let Some(top) = new.last() {
                        self.deps.note_clip_run(scene, top);
                    }
                }
                seeds.extend(ids.into_iter().filter(|id| seeded.insert(id.clone())));
            }
            let changed = self
                .deps
                .expand(scene, &mut dirty, walked_to, &seeds)
                .ok_or("runaway subtree")?;
            walked_to = dirty.len();
            if changed {
                diffs = self.structural_diffs(scene, &dirty)?;
            } else {
                settled = true;
            }
        }
        if !settled {
            return Err("blend / mask consumers did not settle");
        }
        // A consumer re-planned here (a clipped layer whose base a list edit changed) keeps its new sources for the
        // next edit.
        if self.deps.any_clipped {
            for (id, _) in &dirty {
                self.deps.note(scene, id);
            }
        }
        // Property edits: their units, unless culling could decide differently
        // now — then their whole top-level subtree, under the window.
        if let Some(window) = self.cull {
            for (id, structural) in &dirty {
                if *structural || refine_only.contains(id.as_str()) {
                    continue;
                }
                let Some((si, u)) = self.locate(id.as_str()) else {
                    continue;
                };
                let mut walked = Vec::new();
                Self::walk(scene, id, &mut walked).ok_or("runaway subtree")?;
                if !self.unit_path_exact(
                    scene,
                    si,
                    &walked,
                    &window,
                    &clip_sources,
                ) {
                    let ri = self.root_of_unit(si, u).ok_or("unit without a root")?;
                    whole_roots[si].insert(ri);
                }
            }
        }
        // Regions: the diff range, and each whole-root re-plan outside it.
        let mut regions: Vec<Region> = Vec::new();
        for si in 0..n_seg {
            let diff = diffs[si].take();
            if let Some((old, new)) = &diff {
                regions.push(Region {
                    segment: si,
                    old: old.clone(),
                    new: new.clone(),
                });
            }
            for &ri in &whole_roots[si] {
                if diff.as_ref().is_some_and(|(old, _)| old.contains(&ri)) {
                    continue;
                }
                regions.push(Region {
                    segment: si,
                    old: ri..ri + 1,
                    new: vec![self.segments[si].roots[ri].id.clone()],
                });
            }
        }
        // Segment flags: the nodes a region replaces, then the property-edited
        // nodes and their subtrees (a group's opacity reaches its descendants'
        // split predicate).
        let mut flat = Vec::new();
        for r in &regions {
            let s = &mut self.segments[r.segment];
            let d0 = s
                .roots
                .get(r.old.start)
                .map_or(s.seg.dl.items.len(), |x| x.dl_start);
            let d1 = if r.old.is_empty() {
                d0
            } else {
                s.roots[r.old.end - 1].dl_end
            };
            for item in s.seg.dl.items[d0..d1]
                .iter()
                .filter(|i| i.is_primary_node_item)
            {
                s.flags.remove(&item.node_id);
            }
            flat.clear();
            for root in &r.new {
                flattened_root(scene, root, &mut flat);
            }
            for id in &flat {
                s.flags.add(scene, id);
            }
        }
        for (id, structural) in &dirty {
            if *structural {
                continue; // its region re-flagged it
            }
            let Some(&(si, _)) = self.by_node.get(id.as_str()) else {
                continue;
            };
            // A group's clip mask decides whether its other children capture (`needs_mask_capture`).
            let root = scene
                .group_parent_of(id)
                .filter(|p| scene.effective_node(p).and_then(|g| g.str("clipMaskNodeId")) == Some(id.as_str()))
                .map_or(id.as_str(), String::as_str);
            let mut walked = Vec::new();
            Self::walk(scene, root, &mut walked).ok_or("runaway subtree")?;
            for n in &walked {
                if self
                    .by_node
                    .get(n.as_str())
                    .is_some_and(|&(sj, _)| sj == si)
                {
                    let flags = &mut self.segments[si].flags;
                    flags.remove(n);
                    flags.add(scene, n);
                }
            }
        }
        // Members whose node went away without an edit naming it.
        for s in &mut self.segments {
            for set in [
                &mut s.flags.backdrop,
                &mut s.flags.split,
                &mut s.flags.own_split,
                &mut s.flags.capture,
            ] {
                if !set.is_empty() {
                    set.retain(|id| scene.effective_node(id).is_some());
                }
            }
        }
        // The contexts must stay as the plan was built, except a split context that turns on or off (or changes
        // whether rect / ellipse bodies keep the SDF path) with no backdrop effect before or after: a blend split
        // changes no plan-level state, so only the units whose items read the context re-plan.
        let mut respec = Vec::new();
        for si in 0..n_seg {
            let s = &self.segments[si];
            if s.flags.capture.is_empty() == s.seg.layer_blur {
                return Err("capture context changed");
            }
            let spec = s.flags.backdrop_spec(s.seg.kind);
            if spec == s.seg.backdrop {
                continue;
            }
            if backdrop_before[si] || !s.flags.backdrop.is_empty() {
                return Err("split context changed");
            }
            let covered: HashSet<usize> = regions
                .iter()
                .filter(|r| r.segment == si)
                .flat_map(|r| r.old.clone())
                .collect();
            let whole = self.respec_segment(scene, si, spec, &covered, &mut respec);
            for ri in whole {
                regions.push(Region {
                    segment: si,
                    old: ri..ri + 1,
                    new: vec![self.segments[si].roots[ri].id.clone()],
                });
            }
        }
        let scene_flags = self.scene_flags.has(scene);
        if scene_flags != self.built_scene_flags {
            return Err("outline / curve keying changed");
        }
        Ok(Prepared {
            scene_flags,
            camera_dependent: self.camera_dependent(),
            has_captures: self.segments.iter().any(|s| s.seg.layer_blur),
            dirty,
            regions: merge_touching(regions),
            respec,
        })
    }

    /// Switch segment `si` to split context `spec` (neither side with a backdrop effect). A node's items read
    /// the split context only when it could split or SDF-draw: a rect / ellipse body (the context decides its
    /// SDF path), a blend mode on the node or on one of its fills or strokes, a translucent paint stack
    /// (grouped through a split), an effect, an appearance stack, a node in the segment's split sets, or items
    /// that already split. Every other node plans the same items either way. The units of the nodes that read
    /// it re-plan in place (pushed to `respec`); a blended group's (its subtree capture opens only under the
    /// context) or a unit with a culled range inside re-plans its whole root (returned, `covered` excepted).
    /// The segment's unit flags follow the new context.
    fn respec_segment(
        &mut self,
        scene: &Scene,
        si: usize,
        spec: Option<bool>,
        covered: &HashSet<usize>,
        respec: &mut Vec<String>,
    ) -> Vec<usize> {
        use crate::seams::BlendMode;
        let s = &self.segments[si];
        let blended = |v: Option<&str>| BlendMode::from_node_value(v) != BlendMode::Normal;
        let reads_context = |id: &str| -> bool {
            let Some(n) = scene.effective_node(id) else {
                return true;
            };
            let paints = |k: &str| n.array(k).map_or(0, |a| a.len());
            let paint_blended = |k: &str| {
                n.array(k).is_some_and(|a| {
                    a.iter()
                        .any(|e| blended(e.get("blendMode").and_then(moda_scene::value::Value::as_str)))
                })
            };
            let translucent = n.f64("opacity").unwrap_or(1.0) * scene.ancestor_opacity_product(id) < 1.0;
            crate::paint::ShapeType::from_node_type(n.node_type()).is_some()
                || blended(n.str("blendMode"))
                || paint_blended("fills")
                || paint_blended("strokes")
                || (translucent
                    && (paints("fills") + paints("strokes") > 1
                        || n.json_field("vectorNetwork").is_some_and(|v| !v.is_null())))
                || n.array("effects").is_some_and(|e| !e.is_empty())
                || crate::paint::layers::has_entry_effects(n)
                || n.json_field(moda_scene::appearance::APPEARANCE_ORDER_KEY).is_some()
                || s.flags.backdrop.contains(id)
                || s.flags.split.contains(id)
                || s.flags.own_split.contains(id)
                || s.flags.capture.contains(id)
        };
        let opens_capture = |id: &str| {
            scene.effective_node(id).is_some_and(|n| {
                moda_scene::scene::is_opacity_group(n)
                    || (crate::backdrop::blend::is_blend_subtree_node(n)
                        && crate::backdrop::blend::composite_blend_split(n).is_some())
            })
        };
        let at = s.seg.item_start + s.seg.prefix;
        let mut whole = Vec::new();
        for (ri, root) in s.roots.iter().enumerate() {
            if covered.contains(&ri) {
                continue;
            }
            let mut in_place = Vec::new();
            let mut replan_root = false;
            for unit in &s.seg.units[root.unit_start..root.unit_end] {
                if unit.culled {
                    continue;
                }
                let items = &self.plan.items[at + unit.item_start..at + unit.item_start + unit.item_len];
                let split_items = items.iter().any(|i| match i {
                    DrawItem::BackdropSplit(_) => true,
                    DrawItem::Body { blend_mode, .. } => *blend_mode != BlendMode::Normal,
                    _ => false,
                });
                let nodes = s.seg.dl.items[unit.dl_start..unit.dl_end]
                    .iter()
                    .filter(|i| i.is_primary_node_item);
                let mut first = None;
                let mut reads = split_items;
                for item in nodes {
                    first.get_or_insert(item.node_id.as_str());
                    if opens_capture(&item.node_id) {
                        replan_root = true;
                    }
                    reads = reads || reads_context(&item.node_id);
                }
                if !reads {
                    continue;
                }
                match first {
                    Some(id) if !unit.part_culled => in_place.push(id.to_string()),
                    _ => replan_root = true,
                }
                if replan_root {
                    break;
                }
            }
            if replan_root {
                whole.push(ri);
            } else {
                respec.extend(in_place);
            }
        }
        #[cfg(test)]
        RESPECS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let s = &mut self.segments[si];
        s.seg.backdrop = spec;
        let (backdrop, layer_blur) = (spec.is_some(), s.seg.layer_blur);
        for unit in s.seg.units.iter_mut() {
            if !unit.culled {
                let items = &s.seg.dl.items[unit.dl_start..unit.dl_end];
                unit.flags = unit_flags(&s.flags, backdrop, layer_blur, items);
            }
        }
        whole
    }

    /// Re-plan regions, the last first (a region shifts only what follows it).
    /// Every root they re-plan joins `rebuilt`.
    #[allow(clippy::too_many_arguments)]
    fn replan_regions(
        &mut self,
        mut regions: Vec<Region>,
        scene: &Scene,
        hashes: &crate::residency::HashMap<String, NodeHashes>,
        caches: &mut PlanCaches,
        opts: &RenderOptions,
        cam: &CameraParams,
        capture: Option<CaptureView>,
        rebuilt: &mut HashSet<(usize, String)>,
    ) -> Result<(usize, bool), &'static str> {
        if regions.is_empty() {
            return Ok((0, false));
        }
        let clip_sources = self.clip_sources.set();
        regions.sort_by(|a, b| (b.segment, b.old.start).cmp(&(a.segment, a.old.start)));
        let (mut n, mut shader) = (0, false);
        for r in regions {
            let (k, sh) = self.replan_region(
                r,
                scene,
                hashes,
                caches,
                opts,
                cam,
                capture,
                &clip_sources,
                rebuilt,
            )?;
            n += k;
            shader |= sh;
        }
        Ok((n, shader))
    }

    /// Apply a prepared patch for this frame's camera and capture view.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply(
        &mut self,
        prep: Prepared,
        scene: &Scene,
        generation: u64,
        hashes: &crate::residency::HashMap<String, NodeHashes>,
        caches: &mut PlanCaches,
        opts: &RenderOptions,
        cam: &CameraParams,
        capture: Option<CaptureView>,
    ) -> SlotOutcome {
        let camera_moved = self.camera_dependent() && !same_projection(cam, &self.cam);
        let capture_moved = capture != self.capture;
        if prep.dirty.is_empty()
            && prep.regions.is_empty()
            && prep.respec.is_empty()
            && !camera_moved
            && !capture_moved
        {
            self.built_at_generation = generation;
            return SlotOutcome::Unchanged;
        }
        let mut replanned = 0usize;
        let mut shader = false;
        let mut warnings_changed = false;
        // 1. Regions: structural edits, and property edits a cull decision may
        //    follow. `rebuilt` = (segment, root id) re-planned whole.
        let mut rebuilt: HashSet<(usize, String)> = HashSet::new();
        if !prep.regions.is_empty() {
            match self.replan_regions(
                prep.regions,
                scene,
                hashes,
                caches,
                opts,
                cam,
                capture,
                &mut rebuilt,
            ) {
                Ok((n, sh)) => {
                    replanned += n;
                    shader |= sh;
                    warnings_changed = true;
                }
                Err(why) => return SlotOutcome::Fallback(why),
            }
        }
        // 2. Units whose items follow the camera or the capture view; one with
        //    a culled range inside re-plans its whole root under the window.
        let mut units: BTreeSet<(usize, usize)> = BTreeSet::new();
        if camera_moved || capture_moved {
            let mut roots: Vec<Region> = Vec::new();
            for (si, s) in self.segments.iter().enumerate() {
                let mut last_root = None;
                for (u, unit) in s.seg.units.iter().enumerate() {
                    let flagged = (camera_moved && unit.flags & UNIT_CAMERA != 0)
                        || (capture_moved && unit.flags & UNIT_CAPTURE != 0);
                    if !flagged || unit.culled {
                        continue;
                    }
                    if unit.part_culled {
                        let Some(ri) = self.root_of_unit(si, u) else {
                            return SlotOutcome::Fallback("unit without a root");
                        };
                        let id = &s.roots[ri].id;
                        if last_root != Some(ri) && !rebuilt.contains(&(si, id.clone())) {
                            roots.push(Region {
                                segment: si,
                                old: ri..ri + 1,
                                new: vec![id.clone()],
                            });
                        }
                        last_root = Some(ri);
                    } else {
                        units.insert((si, u));
                    }
                }
            }
            if !roots.is_empty() {
                // Their indices shift under these regions: collect afresh below.
                units.clear();
                match self.replan_regions(
                    roots,
                    scene,
                    hashes,
                    caches,
                    opts,
                    cam,
                    capture,
                    &mut rebuilt,
                ) {
                    Ok((n, sh)) => {
                        replanned += n;
                        shader |= sh;
                        warnings_changed = true;
                    }
                    Err(why) => return SlotOutcome::Fallback(why),
                }
                for (si, s) in self.segments.iter().enumerate() {
                    for (u, unit) in s.seg.units.iter().enumerate() {
                        let flagged = (camera_moved && unit.flags & UNIT_CAMERA != 0)
                            || (capture_moved && unit.flags & UNIT_CAPTURE != 0);
                        if flagged && !unit.culled && !unit.part_culled {
                            units.insert((si, u));
                        }
                    }
                }
            }
        }
        // 3. Units of property-edited nodes; a group's edit reaches every unit of
        //    its subtree. `walked` = the nodes reached (their masks may change).
        let mut walked_all: HashSet<String> = HashSet::new();
        for (id, structural) in &prep.dirty {
            if *structural {
                continue;
            }
            let mut walked = Vec::new();
            if Self::walk(scene, id, &mut walked).is_none() {
                return SlotOutcome::Fallback("runaway subtree");
            }
            for n in walked {
                if let Some(su) = self.locate(n.as_str()) {
                    units.insert(su);
                }
                walked_all.insert(n);
            }
        }
        // Unchanged nodes whose units read a split context that changed.
        for id in &prep.respec {
            if let Some((si, u)) = self.locate(id) {
                if !self.segments[si].seg.units[u].culled {
                    units.insert((si, u));
                }
            }
        }
        // Plan every unit first (none moves until the splice), then splice each segment's units in one pass.
        let mut by_segment: std::collections::BTreeMap<usize, Vec<UnitRepl>> = Default::default();
        for &(si, u) in units.iter() {
            let in_rebuilt = !rebuilt.is_empty()
                && self.root_of_unit(si, u).is_some_and(|ri| {
                    rebuilt.contains(&(si, self.segments[si].roots[ri].id.clone()))
                });
            if in_rebuilt {
                continue;
            }
            match self.replan_unit_at(si, u, None, scene, hashes, caches, opts, cam, capture, &walked_all) {
                Ok(r) => {
                    replanned += 1;
                    shader |= r.shader;
                    warnings_changed |= r.warnings_changed;
                    by_segment.entry(si).or_default().push(r);
                }
                Err(why) => return SlotOutcome::Fallback(why),
            }
        }
        for (si, repls) in by_segment {
            self.splice_unit_batch(si, repls);
        }
        for mask in &self.plan.masks {
            if let Some(source) = mask.image.as_ref().and_then(|image| image.rendered.as_ref()) {
                if source.source_fingerprint != crate::effects::mask::rendered_mask_fingerprint(scene, &source.node_id) {
                    return SlotOutcome::Fallback("rendered mask source changed");
                }
            }
        }
        if warnings_changed {
            self.warnings = Arc::new(self.assemble_warnings());
        }
        if camera_moved || capture_moved {
            let plan = Arc::make_mut(&mut self.plan);
            plan.camera = [cam.camera_x, cam.camera_y];
            plan.camera_frame = Some(crate::seams::CameraFrameView {
                camera_x: cam.camera_x,
                camera_y: cam.camera_y,
                zoom: cam.zoom,
                viewport_w: cam.viewport_w,
                viewport_h: cam.viewport_h,
            });
        }
        self.cam = *cam;
        self.capture = capture;
        self.built_at_generation = generation;
        SlotOutcome::Patched {
            units: replanned,
            drop_after: shader,
        }
    }

    /// Splice a re-planned range over units `u0..u1` of segment `si`: its items
    /// (masks already mapped onto the combined table) over theirs, its units and
    /// display list over theirs, and every later unit's and segment's offsets
    /// shifted. The node index follows; root spans are the caller's. Returns
    /// (units the range now has, unit count delta, display-list length delta).
    #[allow(clippy::too_many_arguments)]
    fn splice_units(
        &mut self,
        si: usize,
        u0: usize,
        u1: usize,
        new_units: Vec<PlanUnit>,
        items: Vec<DrawItem>,
        new_dl: Option<Vec<PaintItem>>,
    ) -> (usize, i64, i64) {
        let plan = Arc::make_mut(&mut self.plan);
        let s = &mut self.segments[si];
        let body_len = s.seg.item_len - s.seg.prefix;
        let i0 = s.seg.units.get(u0).map_or(body_len, |x| x.item_start);
        let i1 = s.seg.units.get(u1).map_or(body_len, |x| x.item_start);
        let d0 = s
            .seg
            .units
            .get(u0)
            .map_or(s.seg.dl.items.len(), |x| x.dl_start);
        let d1 = s
            .seg
            .units
            .get(u1)
            .map_or(s.seg.dl.items.len(), |x| x.dl_start);
        let at = s.seg.item_start + s.seg.prefix;
        let delta_bytes = crate::residency::items_bytes(&items) as i64
            - crate::residency::items_bytes(&plan.items[at + i0..at + i1]) as i64;
        self.bytes = (self.bytes as i64 + delta_bytes).max(0) as usize;
        let item_delta = items.len() as i64 - (i1 - i0) as i64;
        if items.len() == i1 - i0 {
            for (slot, item) in plan.items[at + i0..at + i1].iter_mut().zip(items) {
                *slot = item;
            }
        } else {
            plan.items.splice(at + i0..at + i1, items);
        }
        let dl_delta = match new_dl {
            Some(dl) => {
                // The old range's nodes leave the index — unless a node has moved
                // to another segment's (already re-planned) range meanwhile — and
                // the new range's, and every later one's when the length moved,
                // take their display-list positions.
                for (k, item) in s.seg.dl.items[d0..d1].iter().enumerate() {
                    if item.is_primary_node_item
                        && self.by_node.get(&item.node_id) == Some(&(si, d0 + k))
                    {
                        self.by_node.remove(&item.node_id);
                    }
                }
                let delta = dl.len() as i64 - (d1 - d0) as i64;
                let new_len = dl.len();
                Arc::make_mut(&mut s.seg.dl).items.splice(d0..d1, dl);
                let reindex_to = if delta != 0 {
                    s.seg.dl.items.len()
                } else {
                    d0 + new_len
                };
                for (k, item) in s.seg.dl.items[d0..reindex_to].iter().enumerate() {
                    if item.is_primary_node_item {
                        self.by_node.insert(item.node_id.clone(), (si, d0 + k));
                    }
                }
                delta
            }
            None => 0,
        };
        let n_new = new_units.len();
        let unit_delta = n_new as i64 - (u1 - u0) as i64;
        s.seg.units.splice(u0..u1, new_units);
        if item_delta != 0 || dl_delta != 0 {
            for later in &mut s.seg.units[u0 + n_new..] {
                later.item_start = (later.item_start as i64 + item_delta) as usize;
                later.dl_start = (later.dl_start as i64 + dl_delta) as usize;
                later.dl_end = (later.dl_end as i64 + dl_delta) as usize;
            }
        }
        s.seg.item_len = (s.seg.item_len as i64 + item_delta) as usize;
        let (backdrop, layer_blur) = (s.seg.backdrop.is_some(), s.seg.layer_blur);
        for k in u0..u0 + n_new {
            let (a, b) = (s.seg.units[k].dl_start, s.seg.units[k].dl_end);
            s.seg.units[k].flags = unit_flags(&s.flags, backdrop, layer_blur, &s.seg.dl.items[a..b]);
        }
        if item_delta != 0 {
            for later in &mut self.segments[si + 1..] {
                later.seg.item_start = (later.seg.item_start as i64 + item_delta) as usize;
            }
        }
        (n_new, unit_delta, dl_delta)
    }

    /// Shift the spans of roots `from..` of segment `si` (after a splice before
    /// them).
    fn shift_roots(&mut self, si: usize, from: usize, unit_delta: i64, dl_delta: i64) {
        if unit_delta == 0 && dl_delta == 0 {
            return;
        }
        for r in &mut self.segments[si].roots[from..] {
            r.unit_start = (r.unit_start as i64 + unit_delta) as usize;
            r.unit_end = (r.unit_end as i64 + unit_delta) as usize;
            r.dl_start = (r.dl_start as i64 + dl_delta) as usize;
            r.dl_end = (r.dl_end as i64 + dl_delta) as usize;
        }
    }

    /// Move units `from..` of segment `si` by `dz` in render-z: each item's z and
    /// the depth baked from it, when that is exact (`shift_item_z`), else the
    /// unit is re-planned at its new z.
    #[allow(clippy::too_many_arguments)]
    fn shift_units(
        &mut self,
        si: usize,
        from: usize,
        dz: i64,
        scene: &Scene,
        hashes: &crate::residency::HashMap<String, NodeHashes>,
        caches: &mut PlanCaches,
        opts: &RenderOptions,
        cam: &CameraParams,
        capture: Option<CaptureView>,
    ) -> Result<(), &'static str> {
        let n = self.segments[si].seg.units.len();
        let mut replan: Vec<usize> = Vec::new();
        {
            let plan = Arc::make_mut(&mut self.plan);
            let s = &mut self.segments[si];
            let base = s.seg.item_start + s.seg.prefix;
            for u in from..n {
                let unit = &mut s.seg.units[u];
                let items = &mut plan.items[base + unit.item_start..base + unit.item_start + unit.item_len];
                if items.iter().all(|i| z_shiftable(i, dz)) {
                    for i in items.iter_mut() {
                        shift_item_z(i, dz);
                    }
                    unit.z = (unit.z as i64 + dz) as u64;
                } else {
                    replan.push(u);
                }
            }
        }
        let walked = HashSet::new();
        let mut repls = Vec::with_capacity(replan.len());
        for &u in &replan {
            let z = (self.segments[si].seg.units[u].z as i64 + dz) as u64;
            repls.push(self.replan_unit_at(si, u, Some(z), scene, hashes, caches, opts, cam, capture, &walked)?);
        }
        self.splice_unit_batch(si, repls);
        Ok(())
    }

    /// Plan unit `u` of segment `si` again at render-z `z` (`None` = where it is), for
    /// [`Self::splice_unit_batch`] to splice. A unit that
    /// moves keeps its cull decisions: a subtree capture with a culled range
    /// inside re-plans under the window (its range is the whole subtree, so the
    /// decisions are the full build's); anything else draws what it drew.
    #[allow(clippy::too_many_arguments)]
    fn replan_unit_at(
        &mut self,
        si: usize,
        u: usize,
        z: Option<u64>,
        scene: &Scene,
        hashes: &crate::residency::HashMap<String, NodeHashes>,
        caches: &mut PlanCaches,
        opts: &RenderOptions,
        cam: &CameraParams,
        capture: Option<CaptureView>,
        walked: &HashSet<String>,
    ) -> Result<UnitRepl, &'static str> {
        let mut unit = self.segments[si].seg.units[u].clone();
        let cull = if z.is_some() && unit.part_culled {
            self.cull
        } else {
            None
        };
        if let Some(z) = z {
            unit.z = z;
        }
        let build = {
            let s = &self.segments[si];
            rebuild_segment_range(
                scene,
                &s.seg,
                &s.seg.dl.items[unit.dl_start..unit.dl_end],
                opts,
                cam,
                capture,
                Some((caches, hashes)),
                unit.z,
                cull.as_ref(),
            )
        };
        if build.end_z - unit.z != unit.z_consumed {
            return Err("unit render-z changed");
        }
        if !build.mask_warnings.is_empty() {
            return Err("mask warnings");
        }
        let mut items = build.items;
        // The table is in first-use order: the unit must still first-use exactly
        // the masks it did (a mask source edited into a degenerate box drops its
        // mask; a full build's table would then lose it), and use none first used
        // after it.
        let (mb_u, mb_next) = {
            let s = &self.segments[si].seg;
            let next = s.units.get(u + 1).map_or(s.mask_len, |x| x.masks_before);
            (unit.masks_before, next)
        };
        if build.masks.is_empty() && mb_next > mb_u {
            return Err("mask first use moved");
        }
        // Masks: every scratch mask is one the plan has; a changed one only when
        // its owner's whole subtree is being re-planned.
        if !build.masks.is_empty() {
            let (base, len) = (self.segments[si].seg.mask_start, self.segments[si].seg.mask_len);
            if self.segments[si].mask_owners.is_none() {
                let mut owners: HashMap<String, usize> = HashMap::with_capacity(len);
                for (j, m) in self.plan.masks[base..base + len].iter().enumerate() {
                    owners.entry(m.owner_node_id.clone()).or_insert(j);
                }
                self.segments[si].mask_owners = Some(owners);
            }
            let owners = self.segments[si].mask_owners.as_ref().expect("built above");
            let plan = Arc::make_mut(&mut self.plan);
            let mut map: Vec<usize> = Vec::with_capacity(build.masks.len());
            for m in &build.masks {
                let mut mapped = m.clone();
                mapped.ancestor = match m.ancestor {
                    Some(a) => Some(*map.get(a).ok_or("mask ancestor")?),
                    None => None,
                };
                let j = base + *owners.get(&m.owner_node_id).ok_or("mask changed")?;
                if plan.masks[j] != mapped {
                    if !walked.contains(&m.owner_node_id) {
                        return Err("mask changed");
                    }
                    plan.masks[j] = mapped;
                }
                map.push(j);
            }
            let mut first_used = 0usize;
            for &j in &map {
                let rel = j - base;
                if rel >= mb_next {
                    return Err("mask first use moved");
                }
                first_used += usize::from(rel >= mb_u);
            }
            // `map` is one entry per distinct owner.
            if first_used != mb_next - mb_u {
                return Err("mask first use moved");
            }
            if !remap_item_masks(&mut items, &map, base) {
                return Err("mask ref");
            }
        }
        // Items that bake a shader clock the present can't re-thread drop the slot after this frame (it is served
        // once, rebuilt next frame, which keeps the exact-clock cache behaviour). Re-threadable ones keep it, as a
        // seed with them does: a pan over a page whose split units hold procedural fills re-planned those units
        // and then rebuilt the whole plan on every other frame.
        let shader = crate::residency::items_use_shader_time(&items)
            && !crate::residency::items_shader_time_rethreadable(&items);
        let mut new_units = build.units;
        let first_masks = unit.masks_before;
        for nu in &mut new_units {
            nu.dl_start += unit.dl_start;
            nu.dl_end += unit.dl_start;
            nu.item_start += unit.item_start;
            // Its masks are the plan's already: none is first used here anew.
            nu.masks_before = first_masks;
        }
        let warnings_changed = new_units.len() != 1
            || new_units[0].skipped != unit.skipped
            || new_units[0].dropped != unit.dropped;
        let root = self.root_of_unit(si, u).ok_or("unit without a root")?;
        Ok(UnitRepl {
            u,
            root,
            units: new_units,
            items,
            warnings_changed,
            shader,
        })
    }

    /// Splice re-planned units of segment `si` over theirs in one pass: the plan's items and the segment's units
    /// are rebuilt once, however many units changed (the text a first visit queued re-plans thousands of units in
    /// one frame; a splice each moved the plan's tail every time). `repls` are in ascending unit order, distinct.
    fn splice_unit_batch(&mut self, si: usize, repls: Vec<UnitRepl>) {
        if repls.is_empty() {
            return;
        }
        let plan = Arc::make_mut(&mut self.plan);
        let s = &mut self.segments[si];
        let body_len = s.seg.item_len - s.seg.prefix;
        let at = s.seg.item_start + s.seg.prefix;
        let old_items = std::mem::take(&mut plan.items);
        let added: usize = repls.iter().map(|r| r.items.len()).sum();
        let mut items: Vec<DrawItem> = Vec::with_capacity(old_items.len() + added);
        let mut old = old_items.into_iter();
        let mut pos = 0usize;
        let old_units = std::mem::take(&mut s.seg.units);
        let mut units: Vec<PlanUnit> = Vec::with_capacity(old_units.len() + repls.len());
        // The old units are owned here: retain them by moving, including their warning vectors.
        let mut old_units = old_units.into_iter().peekable();
        let mut unit_cursor = 0usize;
        let mut item_shift: i64 = 0;
        let mut bytes_delta: i64 = 0;
        let mut new_ranges: Vec<(usize, usize)> = Vec::with_capacity(repls.len());
        let mut root_deltas: Vec<(usize, i64)> = Vec::new();
        for r in repls {
            while unit_cursor < r.u {
                let mut unit = old_units.next().expect("unit in range");
                unit.item_start = (unit.item_start as i64 + item_shift) as usize;
                units.push(unit);
                unit_cursor += 1;
            }
            let replaced = old_units.next().expect("replaced unit in range");
            let i0 = replaced.item_start;
            let i1 = old_units.peek().map_or(body_len, |x| x.item_start);
            while pos < at + i0 {
                items.push(old.next().expect("item in range"));
                pos += 1;
            }
            let removed: Vec<DrawItem> = old.by_ref().take(i1 - i0).collect();
            pos += i1 - i0;
            bytes_delta += crate::residency::items_bytes(&r.items) as i64 - crate::residency::items_bytes(&removed) as i64;
            drop(removed);
            let delta = r.items.len() as i64 - (i1 - i0) as i64;
            items.extend(r.items);
            let first = units.len();
            for mut unit in r.units {
                unit.item_start = (unit.item_start as i64 + item_shift) as usize;
                units.push(unit);
            }
            new_ranges.push((first, units.len()));
            let unit_delta = (units.len() - first) as i64 - 1;
            if unit_delta != 0 {
                root_deltas.push((r.root, unit_delta));
            }
            item_shift += delta;
            unit_cursor = r.u + 1;
        }
        items.extend(old);
        for mut unit in old_units {
            unit.item_start = (unit.item_start as i64 + item_shift) as usize;
            units.push(unit);
        }
        plan.items = items;
        s.seg.units = units;
        s.seg.item_len = (s.seg.item_len as i64 + item_shift) as usize;
        let (backdrop, layer_blur) = (s.seg.backdrop.is_some(), s.seg.layer_blur);
        for (a, b) in new_ranges {
            for k in a..b {
                let (d0, d1) = (s.seg.units[k].dl_start, s.seg.units[k].dl_end);
                s.seg.units[k].flags = unit_flags(&s.flags, backdrop, layer_blur, &s.seg.dl.items[d0..d1]);
            }
        }
        // Roots: a root's end moves by its own units' change, and every root after it by those before.
        if !root_deltas.is_empty() {
            let mut per_root = vec![0i64; s.roots.len()];
            for (ri, d) in root_deltas {
                per_root[ri] += d;
            }
            let mut running = 0i64;
            for (r, d) in s.roots.iter_mut().zip(per_root) {
                r.unit_start = (r.unit_start as i64 + running) as usize;
                running += d;
                r.unit_end = (r.unit_end as i64 + running) as usize;
            }
        }
        self.bytes = (self.bytes as i64 + bytes_delta).max(0) as usize;
        if item_shift != 0 {
            for later in &mut self.segments[si + 1..] {
                later.seg.item_start = (later.seg.item_start as i64 + item_shift) as usize;
            }
        }
    }

    /// Re-plan a structural region: the old top-level roots become the new ones
    /// (and, when the region's render-z or mask order moved, the rest of the
    /// segment after them). Returns (units re-planned, shader clock used).
    #[allow(clippy::too_many_arguments)]
    fn replan_region(
        &mut self,
        r: Region,
        scene: &Scene,
        hashes: &crate::residency::HashMap<String, NodeHashes>,
        caches: &mut PlanCaches,
        opts: &RenderOptions,
        cam: &CameraParams,
        capture: Option<CaptureView>,
        clip_sources: &HashSet<String>,
        rebuilt: &mut HashSet<(usize, String)>,
    ) -> Result<(usize, bool), &'static str> {
        let si = r.segment;
        let cull = self.cull;
        let (u0, u1, n_units, n_roots) = {
            let s = &self.segments[si];
            let at = |ri: usize| s.roots.get(ri).map_or(s.seg.units.len(), |x| x.unit_start);
            (
                at(r.old.start),
                at(r.old.end),
                s.seg.units.len(),
                s.roots.len(),
            )
        };
        let mut flat = Vec::new();
        for root in &r.new {
            flattened_root(scene, root, &mut flat);
        }
        let mut new_dl =
            moda_scene::display_list::display_list_items_for(scene, &flat, 0, clip_sources);
        let (z0, z1, mb0, mb1) = {
            let s = &self.segments[si].seg;
            let z_at = |u: usize| s.units.get(u).map_or(s.z_end, |x| x.z);
            let mb_at = |u: usize| s.units.get(u).map_or(s.mask_len, |x| x.masks_before);
            (z_at(u0), z_at(u1), mb_at(u0), mb_at(u1))
        };
        let mut build = {
            let s = &self.segments[si].seg;
            rebuild_segment_range(
                scene,
                s,
                &new_dl,
                opts,
                cam,
                capture,
                Some((caches, hashes)),
                z0,
                cull.as_ref(),
            )
        };
        // Keep the rest of the segment when the region ends at the z it did, and
        // its masks first used anew are exactly the ones it first used before (so
        // the table keeps its first-use order).
        let (table_before, table_region): (HashSet<String>, Vec<String>) = {
            let s = &self.segments[si].seg;
            let t = &self.plan.masks[s.mask_start..s.mask_start + s.mask_len];
            (
                t[..mb0].iter().map(|m| m.owner_node_id.clone()).collect(),
                t[mb0..mb1]
                    .iter()
                    .map(|m| m.owner_node_id.clone())
                    .collect(),
            )
        };
        let fresh_owners = |b: &RangeBuild| -> Vec<String> {
            b.masks
                .iter()
                .map(|m| m.owner_node_id.clone())
                .filter(|o| !table_before.contains(o))
                .collect()
        };
        // A region whose masks keep their first-use order splices in the middle,
        // and what follows it moves by the render-z it gained or lost
        // (`shift_units`); otherwise the rest of the segment is re-planned after it.
        let tail = fresh_owners(&build) != table_region;
        let dz = build.end_z as i64 - z1 as i64;
        let old_z_end = self.segments[si].seg.z_end;
        let (end_u, new_roots) = if tail && u1 < n_units {
            // Re-plan the rest of the segment after the region, from where the
            // region's z ended.
            let rest_dl: Vec<PaintItem> = {
                let s = &self.segments[si].seg;
                s.dl.items[s.units[u1].dl_start..].to_vec()
            };
            let rest = {
                let s = &self.segments[si].seg;
                rebuild_segment_range(
                    scene,
                    s,
                    &rest_dl,
                    opts,
                    cam,
                    capture,
                    Some((caches, hashes)),
                    build.end_z,
                    cull.as_ref(),
                )
            };
            merge_ranges(&mut build, new_dl.len(), rest)?;
            new_dl.extend(rest_dl);
            let mut roots = r.new.clone();
            roots.extend(
                self.segments[si].roots[r.old.end..n_roots]
                    .iter()
                    .map(|x| x.id.clone()),
            );
            (n_units, roots)
        } else {
            (u1, r.new.clone())
        };
        let root_end = if end_u == n_units && tail {
            n_roots
        } else {
            r.old.end
        };
        if !build.mask_warnings.is_empty() {
            return Err("mask warnings");
        }
        // Masks: first-used-before-the-region ones must be unchanged; the rest are
        // the region's own (mapped in place in the middle, re-appended in a tail).
        let (map, new_mask_len, old_mask_len, base) = {
            let s = &self.segments[si].seg;
            let base = s.mask_start;
            let mut table: Vec<MaskPlan> = self.plan.masks[base..base + s.mask_len].to_vec();
            let map = if tail {
                table.truncate(mb0);
                map_masks(&build.masks, &mut table, base, mb0, true, &|_| false)
                    .ok_or("mask before the region changed")?
            } else {
                let region_owned: HashSet<&str> = table_region.iter().map(String::as_str).collect();
                let limit = table.len();
                map_masks(&build.masks, &mut table, base, limit, false, &|o| {
                    region_owned.contains(o)
                })
                .ok_or("mask changed")?
            };
            let old_len = s.mask_len;
            let new_len = table.len();
            let plan = Arc::make_mut(&mut self.plan);
            plan.masks.splice(base..base + old_len, table);
            self.segments[si].mask_owners = None;
            (map, new_len, old_len, base)
        };
        let mut items = build.items;
        if !remap_item_masks(&mut items, &map, base) {
            return Err("mask ref");
        }
        // Items that bake a shader clock the present can't re-thread drop the slot after this frame (it is served
        // once, rebuilt next frame, which keeps the exact-clock cache behaviour). Re-threadable ones keep it, as a
        // seed with them does: a pan over a page whose split units hold procedural fills re-planned those units
        // and then rebuilt the whole plan on every other frame.
        let shader = crate::residency::items_use_shader_time(&items)
            && !crate::residency::items_shader_time_rethreadable(&items);
        // Units' first-use mask counts, in the segment's table: masks before the
        // region, plus the region's scratch masks first used anew before each.
        let mut fresh_prefix: Vec<usize> = Vec::with_capacity(build.masks.len() + 1);
        fresh_prefix.push(0);
        for m in &build.masks {
            let last = *fresh_prefix.last().expect("seeded");
            fresh_prefix.push(last + usize::from(!table_before.contains(&m.owner_node_id)));
        }
        let new_units: Vec<PlanUnit> = build
            .units
            .into_iter()
            .map(|mut unit| {
                unit.masks_before = mb0 + fresh_prefix[unit.masks_before.min(build.masks.len())];
                unit
            })
            .collect();
        // Region-relative offsets become segment-relative in the splice below.
        let (d0, i0) = {
            let s = &self.segments[si].seg;
            let body_len = s.item_len - s.prefix;
            (
                s.units.get(u0).map_or(s.dl.items.len(), |x| x.dl_start),
                s.units.get(u0).map_or(body_len, |x| x.item_start),
            )
        };
        let new_units: Vec<PlanUnit> = new_units
            .into_iter()
            .map(|mut unit| {
                unit.dl_start += d0;
                unit.dl_end += d0;
                unit.item_start += i0;
                unit
            })
            .collect();
        let (n_new, unit_delta, dl_delta) =
            self.splice_units(si, u0, end_u, new_units, items, Some(new_dl));
        // Roots after the region move with the splice; the region's own (and, in
        // a tail, every later one) are re-tiled over its new display-list range.
        self.shift_roots(si, root_end, unit_delta, dl_delta);
        {
            let s = &mut self.segments[si];
            let dl_end = s
                .seg
                .units
                .get(u0 + n_new)
                .map_or(s.seg.dl.items.len(), |x| x.dl_start);
            let tiled = tile_roots(
                scene,
                &new_roots,
                &s.seg.dl.items,
                d0,
                dl_end,
                &s.seg.units,
                u0,
            )
            .ok_or("region does not tile by root")?;
            if tiled.last().map_or(u0, |x| x.unit_end) != u0 + n_new {
                return Err("region units do not tile");
            }
            s.roots.splice(r.old.start..root_end, tiled);
            s.seg.mask_len = new_mask_len;
        }
        for r in &new_roots {
            rebuilt.insert((si, r.clone()));
        }
        // Later segments: their masks moved with this table's length (before any
        // of their units re-plans against the table below).
        let mask_delta = new_mask_len as i64 - old_mask_len as i64;
        if mask_delta != 0 {
            let first_later = base + old_mask_len;
            let plan = Arc::make_mut(&mut self.plan);
            for later in &mut self.segments[si + 1..] {
                let (a, b) = (
                    later.seg.item_start,
                    later.seg.item_start + later.seg.item_len,
                );
                shift_item_masks(&mut plan.items[a..b], first_later, mask_delta);
                later.seg.mask_start = (later.seg.mask_start as i64 + mask_delta) as usize;
            }
            for m in plan.masks[base + new_mask_len..].iter_mut() {
                if let Some(a) = m.ancestor.as_mut() {
                    if *a >= first_later {
                        *a = (*a as i64 + mask_delta) as usize;
                    }
                }
            }
        }
        // Render-z: what follows the region moves with it; in a tail the region
        // already ends the segment.
        if tail {
            self.segments[si].seg.z_end = build.end_z;
        } else if dz != 0 {
            self.shift_units(si, u0 + n_new, dz, scene, hashes, caches, opts, cam, capture)?;
            let seg = &mut self.segments[si].seg;
            seg.z_end = (seg.z_end as i64 + dz) as u64;
        }
        // A full build refuses a segment that draws an item past what depth orders
        // (`depth::ensure_render_z`); the build this falls back to decides by the
        // items themselves. Here: the end of the last unit that drew anything.
        let past_depth_order = |seg: &CameraSegment| {
            seg.units
                .iter()
                .rev()
                .find(|u| u.item_len > 0)
                .is_some_and(|u| u.z + u.z_consumed > crate::depth::MAX_RENDER_Z)
        };
        if past_depth_order(&self.segments[si].seg) {
            return Err("render-z past what depth orders");
        }
        // Freeform pages band their z cumulatively and the canvas root sits above
        // every page, so later segments move too (other layouts start each page
        // at 0).
        let seg_dz = self.segments[si].seg.z_end as i64 - old_z_end as i64;
        if seg_dz != 0 && scene.is_freeform() && si + 1 < self.segments.len() {
            for sj in si + 1..self.segments.len() {
                {
                    let seg = &mut self.segments[sj].seg;
                    seg.z_start = (seg.z_start as i64 + seg_dz) as u64;
                    seg.z_end = (seg.z_end as i64 + seg_dz) as u64;
                }
                self.shift_units(sj, 0, seg_dz, scene, hashes, caches, opts, cam, capture)?;
                if past_depth_order(&self.segments[sj].seg) {
                    return Err("render-z past what depth orders");
                }
            }
        }
        Ok((n_new, shader))
    }

    /// Test diagnostics: where the slot holds `id`.
    #[cfg(test)]
    pub(crate) fn describe(&self, id: &str) -> String {
        let segs: Vec<String> = self
            .segments
            .iter()
            .map(|s| {
                format!(
                    "{:?} items {}+{} masks {}+{} z {}..{}",
                    s.seg.kind,
                    s.seg.item_start,
                    s.seg.item_len,
                    s.seg.mask_start,
                    s.seg.mask_len,
                    s.seg.z_start,
                    s.seg.z_end
                )
            })
            .collect();
        format!("{segs:?}; {}", self.describe_node(id))
    }

    #[cfg(test)]
    fn describe_node(&self, id: &str) -> String {
        match self.locate(id) {
            Some((si, u)) => {
                let s = &self.segments[si];
                let unit = &s.seg.units[u];
                format!(
                    "seg {si} ({:?}, backdrop {:?}, capture {}), unit {u}/{} flags {} dl {}..{} items +{}..+{} z {} (+{}) culled {}; cam {:?}",
                    s.seg.kind,
                    s.seg.backdrop,
                    s.seg.layer_blur,
                    s.seg.units.len(),
                    unit.flags,
                    unit.dl_start,
                    unit.dl_end,
                    unit.item_start,
                    unit.item_len,
                    unit.z,
                    unit.z_consumed,
                    unit.culled,
                    (self.cam.camera_x, self.cam.camera_y, self.cam.zoom),
                )
            }
            None => format!("{id}: not indexed"),
        }
    }

    /// The plan's warnings from its segments and units, in build order.
    fn assemble_warnings(&self) -> PlanWarnings {
        let mut w = PlanWarnings::default();
        for s in &self.segments {
            w.absorb(s.seg.head_warnings.clone());
            for u in &s.seg.units {
                w.skipped.extend(u.skipped.iter().cloned());
                w.dropped.extend(u.dropped.iter().cloned());
            }
            w.absorb(s.seg.tail_warnings.clone());
        }
        w
    }
}

/// The oracle: the first difference between a served camera plan and a
/// from-scratch build of the same frame (items and masks, under the same cull
/// window and capture view), or `None`. Items compare with the re-threaded
/// shader clock normalised.
#[allow(clippy::too_many_arguments)]
pub fn plan_divergence(
    scene: &Scene,
    visible_pages: &[usize],
    cam: &CameraParams,
    opts: &RenderOptions,
    hashes: &crate::residency::HashMap<String, NodeHashes>,
    cull: Option<&Aabb>,
    capture: Option<CaptureView>,
    got: &CoveragePlan,
) -> Option<String> {
    let build = |window: Option<&Aabb>| {
        let mut fresh = PlanCaches::default();
        let mut out = crate::frame::CullOut::default();
        crate::frame::build_camera_plan_culled_with(
            scene,
            visible_pages,
            cam,
            opts,
            Some((&mut fresh, hashes)),
            None,
            window.map(|w| (w, &mut out)),
            capture,
            None,
        )
    };
    let full = match build(cull) {
        Ok(Some((p, _))) => p,
        Ok(None) => return Some("fresh build returned no plan".into()),
        Err(e) => return Some(format!("fresh build failed: {e}")),
    };
    let fp = crate::residency::drawitem_fingerprint_verify;
    let same_items = |a: &[DrawItem], b: &[DrawItem]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| fp(x) == fp(y))
    };
    if same_items(&got.items, &full.items) && got.masks == full.masks {
        return None;
    }
    let n = got.items.len().min(full.items.len());
    let first = (0..n).find(|&i| fp(&got.items[i]) != fp(&full.items[i]));
    let ids = |items: &[DrawItem], i: usize| -> Vec<String> {
        items[i.saturating_sub(3)..(i + 4).min(items.len())]
            .iter()
            .map(|it| {
                let z = match it {
                    DrawItem::Body { z, .. }
                    | DrawItem::Shadow { z, .. }
                    | DrawItem::SolidShapeSdf { z, .. }
                    | DrawItem::GradientShapeSdf { z, .. }
                    | DrawItem::LayerBlurComposite { z, .. } => *z,
                    DrawItem::Text(b) => b.z,
                    _ => -1.0,
                };
                let m = match it {
                    DrawItem::Body { mask, .. } => mask.as_ref().map(|m| m.mask_id as i64),
                    // vork: a text batch carries no mask (moda-text-draw's TextQuadBatch; see PROVENANCE.md).
                    DrawItem::Text(_) => None,
                    _ => None,
                };
                format!(
                    "{}@{z}m{m:?}",
                    crate::residency::drawitem_node_id(it).unwrap_or("-")
                )
            })
            .collect()
    };
    let debug_diff = |a: &DrawItem, b: &DrawItem| -> String {
        let (a, b) = (format!("{a:?}"), format!("{b:?}"));
        let k = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
        let lo = k.saturating_sub(120);
        format!(
            "…{}… / …{}…",
            a.chars().skip(lo).take(240).collect::<String>(),
            b.chars().skip(lo).take(240).collect::<String>()
        )
    };
    let owners = |p: &CoveragePlan| -> Vec<String> {
        p.masks
            .iter()
            .map(|m| format!("{}<{:?}", m.owner_node_id, m.ancestor))
            .collect()
    };
    Some(match first {
        Some(i) => format!(
            "node <{}> item {i} of {} (fresh {}): got {:?} / fresh {:?}; {}; masks got {:?} / fresh {:?}",
            crate::residency::drawitem_node_id(&got.items[i])
                .unwrap_or("-")
                .split("::")
                .next()
                .unwrap_or("-"),
            got.items.len(),
            full.items.len(),
            ids(&got.items, i),
            ids(&full.items, i),
            debug_diff(&got.items[i], &full.items[i]),
            owners(got),
            owners(&full),
        ),
        None if got.items.len() != full.items.len() => format!(
            "item counts differ: got {} / fresh {}",
            got.items.len(),
            full.items.len()
        ),
        None => {
            let j = got.masks.iter().zip(&full.masks).position(|(a, b)| a != b);
            let source = |plan: &CoveragePlan| j.and_then(|index| plan.masks.get(index))
                .and_then(|mask| mask.image.as_ref()).and_then(|image| image.rendered.as_ref())
                .map(|mask| (mask.node_id.clone(), mask.source_fingerprint));
            format!(
                "masks differ (got {} / fresh {}) at {:?}; sources got {:?} / fresh {:?}",
                got.masks.len(),
                full.masks.len(),
                j,
                source(got),
                source(&full)
            )
        }
    })
}

/// Split-context switches the slot made in place (tests check the fuzz reaches them).
#[cfg(test)]
static RESPECS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
#[path = "camera_slot_tests.rs"]
mod tests;
