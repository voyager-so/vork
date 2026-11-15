//! The 2D canvas (END-STATE-API §7): a scene with `composition: {space: "2d"}` is a canvas of
//! `render.resolution` pixels (or its `viewBox` placed in them), origin top-left, y down, whose group, text
//! and path nodes paint in tree order. Each frame is drawn by Moda's 2D engine (`canvas_draw`): the canvas's
//! nodes as one Moda document, planned and drawn with Moda's pipelines into a 4x MSAA target. Text is laid
//! out by Moda's text engine (crates `moda-text`, `moda-text-draw`): each text node becomes Moda
//! `TextNodeSpec`s (its styled runs spans of its HTML), its glyph, word and line tracks Moda's per-code-point
//! unit transforms, and its batches are drawn where Moda's frame walk places the text node. Text Moda's
//! vector lane declines (right-to-left and complex scripts) is drawn by its degraded tier, a cosmic-text
//! raster. The canvas blends in encoded sRGB, as CSS and SVG do, or in the encoding its composition names
//! (`composition.color_space`): its authored colours and the pictures it paints are sRGB code values, which
//! Moda's engine converts where they enter it, before it filters or blends them (`canvas_draw::working_color`).
//! Frames are delivered to the destinations a 3D shot writes.
//!
//! Node clocks (§8) are the 3D renderer's: `active`, `time` and a node's own `animation` are evaluated by the
//! same animation plan, so an absent node draws nothing, children included, and a node's tracks run on its
//! local time. A world (`composition: {space: "3d"}`) is drawn by the 3D renderer at its node's local time and
//! placed as one picture (`canvas_world`).
use crate::failure::{FailureKind, device, panicked, scene as scene_error, setup};
use crate::shot::{self, Output, RenderedShot};
use crate::Renderer;
use anyhow::{Context, Result, bail, ensure};
use glam::{DMat2, DVec2, Quat};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use vork_encode::{Backend, EncodeSettings, Encoder, Surface};
use vork_scene::v2::{self, NodeKind, TextUnit, TrackTarget, UnitProperty};
use vork_scene::Scene;
use crate::canvas_draw::{PaintForm, PaintSource};

macro_rules! ensure_usage {
    ($condition:expr, $($message:tt)*) => {
        if !$condition { return Err(anyhow::anyhow!($($message)*).context(FailureKind::Usage)); }
    };
}

/// The levels of pieces, innermost first: a glyph's transform applies first, then its word's, then its line's.
const LEVELS: [TextUnit; 3] = [TextUnit::Glyphs, TextUnit::Words, TextUnit::Lines];

/// One node of the canvas tree, flattened in paint order.
struct Item {
    name: String,
    visible: bool,
    text: Option<TextItem>,
    /// How Moda draws the node.
    node: crate::canvas_draw::Node,
}
/// What an item's evaluation starts from, held apart from the item's larger parts (its name, text and node) so
/// evaluating an item that only moves reads this alone: its parent, its placement and opacity at rest, a path's
/// keyed numbers at rest, and whether it is text.
struct Rest {
    parent: Option<usize>,
    transform: v2::Transform,
    opacity: f32,
    path: Option<crate::canvas_draw::PathValues>,
    text: bool,
}
/// Every item's children, in paint order, as one array.
struct Children {
    /// Item `i`'s children are `items[first[i]..first[i + 1]]`.
    first: Vec<usize>,
    items: Vec<usize>,
}
impl Children {
    /// From each item's parent. Items are in paint order, a parent before its children, so each item's
    /// children come out ascending.
    fn new(rest: &[Rest]) -> Self {
        let mut first = vec![0; rest.len() + 1];
        for parent in rest.iter().filter_map(|item| item.parent) { first[parent + 1] += 1; }
        for item in 0..rest.len() { first[item + 1] += first[item]; }
        let mut items = vec![0; first[rest.len()]];
        let mut next = first.clone();
        for (item, at_rest) in rest.iter().enumerate() {
            if let Some(parent) = at_rest.parent {
                items[next[parent]] = item;
                next[parent] += 1;
            }
        }
        Self { first, items }
    }
    fn of(&self, item: usize) -> &[usize] {
        &self.items[self.first[item]..self.first[item + 1]]
    }
}
/// A node of the canvas's scene, where the scene holds it, with its parent's item.
struct SceneNode<'a> {
    node: &'a v2::Node,
    parent: Option<usize>,
}
impl crate::animation::ClockNode for SceneNode<'_> {
    fn node(&self) -> &v2::Node { self.node }
    fn parent(&self) -> Option<usize> { self.parent }
    /// The mark is the 3D installer's; the canvas evaluates its items from their tracks.
    fn set_dynamic(&mut self) {}
}
/// A paint: straight-alpha sRGB, or `none`.
type Paint = Option<[f32; 4]>;
/// A text node laid out once: Moda's spec at rest, and, in the code points of Moda's canonical plain text
/// (the space its unit transforms index), each glyph, word and line, with the ink box it covers at rest.
#[derive(Clone)]
struct TextItem {
    /// The node's HTML (each run a span with its font, naming its paint; the HTML as drawn, colours and all, when
    /// the degraded tier rasters it), its runs' colours, box and wrapping; no stroke, no motion.
    spec: moda_text::ops::TextNodeSpec,
    /// Each code point's run. This and the three tables below are of the text's content and layout, not its
    /// paints: text that takes new paints shares them.
    run_of: Arc<Vec<usize>>,
    code_points: usize,
    clip_source: bool,
    units: Arc<BTreeMap<TextUnit, Vec<Vec<usize>>>>,
    /// Per level, the piece each code point belongs to.
    piece_of: Arc<BTreeMap<TextUnit, Vec<Option<usize>>>>,
    /// Per level, each piece's ink box at rest (node px: left, top, right, bottom).
    boxes: Arc<BTreeMap<TextUnit, Vec<Option<[f64; 4]>>>>,
    /// The centre of all the node's ink at rest: Moda turns every code point about it when all form one unit.
    centre: DVec2,
    /// Each code point's fill and stroke at rest (its run's, else the node's).
    fill: Vec<PaintSource>,
    stroke: Vec<PaintSource>,
    stroke_width: f64,
    /// Whether tracks address this node's glyphs, words or lines.
    pieces_keyed: bool,
    /// Whether tracks key a glyph's, word's or line's fill: each code point then draws in its own colour.
    fill_keyed: bool,
    /// The text laid out: the one layout its box, lines, ink boxes and draws read.
    layout: Arc<Laid>,
    degraded_paints: Vec<(PaintSource, String)>,
    /// The laid-out box (px): its width and the text's height.
    size: [f64; 2],
}
impl TextItem {
    /// Moda's vector lane declined this text (right-to-left or complex script); its degraded tier draws it.
    fn degraded(&self) -> bool { self.layout.vector.is_none() }
}
/// A text node's accepted layout, made once for its content and shared by every node with the same content: the
/// same HTML without its colours (text, runs, fonts and their axes, size, spacing, alignment, white space), box
/// width and wrapping, on this document's fonts. Where a node draws and in what colours is no part of it.
struct Laid {
    /// Moda's vector layout; `None` when its vector lane declines the text and its degraded tier draws it.
    vector: Option<moda_text::vector::VectorLayout>,
    /// The box's width (px): the widest line when the text has no width of its own.
    width: f64,
    /// The text's height (px).
    height: f64,
    /// Per line, the source code points its segments hold, in order.
    lines: Vec<Vec<usize>>,
    /// A hash of the content this layout is of (`LaidKey`), made once when the text is laid out: what names the
    /// layout where the key itself is too big to carry. Equal content has an equal hash within a process.
    content: u64,
}
/// What lays text out: its HTML with paints named, not coloured (`TextHtml::layout`), its width and whether it wraps. The line-height policy is CSS for every node.
type LaidKey = (Arc<str>, Option<u64>, bool);
/// One glyph's, word's or line's animated fields, in the text node's space.
#[derive(Clone, PartialEq)]
struct UnitValue {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
    opacity: f32,
    /// Paint layers resolve after the sampled text layout is known.
    fill: Vec<UnitPaint>,
    stroke: Vec<UnitPaint>,
    origin: Option<[(f32, f32); 2]>,
}
#[derive(Clone, PartialEq)]
enum UnitPaint {
    Css(Arc<crate::animation::style::StyleValue>, f32),
    Numeric([f32;4], f32, bool),
}
impl Default for UnitValue {
    fn default() -> Self {
        Self { translation: [0.; 3], rotation: [0., 0., 0., 1.], scale: [1.; 3], opacity: 1., fill: Vec::new(), stroke: Vec::new(), origin: None }
    }
}
/// One batch Moda draws: a text node's item, which of the node's batches it is, its spec at an instant and its
/// model matrix (canvas px, where Moda's frame walk places the node), and the layout it draws.
#[derive(Clone)]
struct Draw {
    item: usize,
    sub: usize,
    spec: Arc<moda_text::ops::TextNodeSpec>,
    model: [f32; 16],
    layout: Arc<Laid>,
}

/// A canvas document: its text laid out, its fonts loaded, its tracks compiled.
pub(crate) struct Document {
    /// The canvas's scene, kept when its image graph or a view stage is prepared from it: it keeps a float
    /// picture (`float`), or it blends in another encoding than sRGB, which only its view turns into code
    /// values a reader or a display takes.
    view: Option<Scene>,
    /// `float`.
    float: bool,
    /// The encoding the canvas blends in (`composition.color_space`).
    working: v2::WorkingColorSpace,
    /// The image files the canvas's image graph may read, by name, each with its fixed time when it has one
    /// (`images.X.time`): every frame carries their instants (`Frame::images`).
    graph_images: Vec<(String, Option<vork_time::Clock>)>,
    images: v2::Table<v2::Image>,
    base: PathBuf,
    font_lists: Vec<Vec<PathBuf>>,
    keyed: BTreeMap<String, BTreeSet<UnitProperty>>,
    glyphs_keyed: BTreeSet<String>,
    /// What a restyling track starts from, for each item one names (`restyled_field`); `None` for every other
    /// item. By item.
    authored: Vec<Option<Box<Authored>>>,
    styles: RefCell<Vec<Option<StyledItem>>>,
    size: [u32; 2],
    items: Vec<Item>,
    /// Each item's rest values, by item.
    rest: Vec<Rest>,
    children: Children,
    /// Each named node's item.
    by_name: HashMap<String, usize>,
    fonts: Fonts,
    plan: crate::animation::Plan,
    /// The tracks' samples, reused from frame to frame.
    samples: RefCell<crate::animation::Samples>,
    /// The output page Moda draws on: its size, the viewBox placement and the background.
    page: crate::canvas_draw::Page,
    /// The worlds in the canvas: each one's item and its scene, drawn at `size` px.
    worlds: Vec<WorldPart>,
    media: RefCell<crate::canvas_draw::Media>,
    retained: RefCell<Option<crate::canvas_draw::Retained>>,
    text_specs: RefCell<Vec<Option<TextSpecs>>>,
    evaluated: RefCell<Option<Evaluated>>,
    /// Whether the installed scene is owed the whole evaluated table. A frame writes the scene as it ends
    /// (`write_rows`), so this is set only before the first install, after a failure part way through a write,
    /// and, for a canvas with worlds, after a frame that needs a whole write: those wait for the plan, which has
    /// the worlds' pictures.
    stale: std::cell::Cell<bool>,
    /// The text builder made of the evaluated table's batches; `None` when a frame changed a batch since.
    text: RefCell<Option<crate::canvas_draw::CanvasTextBuilder>>,
    /// Which document this is, unique in the process, and how many snapshots tests have taken of it.
    #[cfg(test)]
    id: u64,
    #[cfg(test)]
    snapshots: std::cell::Cell<u64>,
    /// Each track's kind, by flat track index (every track of a canvas is authored, so that is `Plan::sampled`
    /// order); `None` for a control track (a layer's weight).
    kinds: Vec<Option<ChangeKind>>,
    /// Each track's item, when it targets a node of this canvas or a piece of its text.
    track_items: Vec<Option<usize>>,
    /// What a change of each track changes of its item (`canvas_draw::change`), by flat track index.
    track_parts: Vec<u8>,
    /// Each item's tracks, ascending: item `i`'s are `item_tracks[item_tracks_first[i]..item_tracks_first[i + 1]]`.
    /// An item is evaluated by blending these from its rest values, and visits no other track.
    item_tracks_first: Vec<u32>,
    item_tracks: Vec<u32>,
    /// The tracks of the page's background: the tracks that target no item and still draw.
    page_tracks: Vec<u32>,
    /// Scratch for naming the changed items: per item, 0 or one more than its slot among them.
    item_marks: RefCell<Vec<u32>>,
    /// Frames by what changed since the frame before: unchanged, then each `ChangeKind`.
    changes: std::cell::Cell<[u64; 5]>,
}
/// A world inside the canvas.
pub(crate) struct WorldPart {
    item: usize,
    pub scene: Scene,
}
struct StyledItem {
    fields: BTreeMap<String, Arc<crate::animation::style::StyleValue>>,
    draw: Arc<crate::canvas_draw::Node>,
    text: Option<Arc<TextItem>>,
    revision: u64,
    text_revision: u64,
}
struct TextSpecs {
    revision: u64,
    style_revision: u64,
    units: BTreeMap<(TextUnit, usize), UnitValue>,
    specs: Vec<Arc<moda_text::ops::TextNodeSpec>>,
    composition: Vec<crate::canvas_draw::TextPart>,
    clip: Option<Arc<moda_text::ops::TextNodeSpec>>,
}
/// The scene's fonts as Moda's text lanes read them, shared with the text builder Moda's frame walk calls.
#[derive(Clone)]
struct Fonts {
    host: Arc<moda_text::TextHost>,
    store: std::rc::Rc<moda_text::vector::VectorFontStore>,
    /// Moda's degraded tier on the scene's fonts, built when a node needs it.
    degraded: std::rc::Rc<std::cell::OnceCell<moda_text::degraded::DegradedEngine>>,
    /// The layouts in use, by content: an entry lives as long as a text item or draw holds it.
    layouts: std::rc::Rc<RefCell<Layouts>>,
}
#[derive(Default)]
struct Layouts {
    by_content: HashMap<LaidKey, std::sync::Weak<Laid>>,
    /// The entry count at which dropped layouts are next swept out.
    sweep_at: usize,
}
/// One instant of a canvas, as asked for. Its values are the document's evaluated table (`Evaluated`), which
/// stands at this instant until the next frame is evaluated.
struct Frame {
    /// The instant asked for, exactly: an authored view's `ctx.time`, and what its animated inputs are sampled
    /// at, so a view's keys fall on the same side of a boundary as the canvas's own.
    request: vork_time::Request,
    /// World clocks advance even when their surrounding canvas is unchanged.
    worlds: Vec<Option<vork_time::Request>>,
    /// The instant each image of the canvas's image graph shows (`Document::graph_images`): scene time, or the
    /// image's own fixed or keyed time.
    images: Vec<(String, vork_time::exact::Germ)>,
}
/// Every item's values at one instant, and the text batches to draw.
#[derive(Clone)]
struct FrameState {
    background: Paint,
    nodes: Vec<Option<Arc<crate::canvas_draw::Node>>>,
    values: Vec<crate::canvas_draw::Values>,
    /// Each item's text batches, in the order it draws them (`Draw::sub`); empty for an item that draws none.
    draws: Vec<Vec<Draw>>,
    /// Which nodes draw at all: visible, present on their clocks, and so is every ancestor.
    shown: Vec<bool>,
    /// Of those, the ones no zero scale at or above them collapses: a text node's batches are built only then.
    drawing: Vec<bool>,
}
impl FrameState {
    /// Every text batch, in item order.
    fn batches(&self) -> impl Iterator<Item = &Draw> {
        self.draws.iter().flatten()
    }
}
/// The document's evaluated table: every item's values at the last frame evaluated, written in place by each
/// frame. There is one, it is never copied, and it has no revision: the document's samples stand at the same
/// frame, so the tracks the next sampling reports changed are what differs from it.
struct Evaluated {
    present: Vec<bool>,
    state: FrameState,
}
/// What a frame changed of the evaluated table.
enum Changed {
    /// Nothing: the table stands.
    Nothing,
    /// These items, each with what of it changed (`canvas_draw::change`); and whether every changed track draws
    /// through an item of the canvas.
    Items(Vec<(usize, u8)>, bool),
    /// Every item was evaluated.
    Every,
}
/// A test's copy of the evaluated table at one frame: tests keep frames side by side, read their values and edit
/// them by hand, which the document's own table (one, in place) does not allow.
#[cfg(test)]
struct Snapshot {
    state: FrameState,
    /// The document it was taken of (`Document::id`), and which snapshot of that document it is, counted from one.
    document: u64,
    serial: u64,
    edited: bool,
}
#[cfg(test)]
impl std::ops::Deref for Snapshot {
    type Target = FrameState;
    fn deref(&self) -> &FrameState { &self.state }
}
#[cfg(test)]
impl std::ops::DerefMut for Snapshot {
    /// A test edits a frame's values by hand: the result is a state no evaluation made.
    fn deref_mut(&mut self) -> &mut FrameState {
        self.edited = true;
        &mut self.state
    }
}
/// What a track's change asks of a frame, decided once at install. Ordered by how much a frame redoes:
/// a frame's kind is the largest kind among the tracks whose sample or weight changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ChangeKind {
    /// A node's translation, rotation, scale or opacity: its placement, and its descendants'.
    Placement,
    /// A path's colour, the stops and geometry of a gradient of one form, or paint opacity: values of the paint
    /// it already has.
    PaintValue,
    /// Geometry, stroke shape, text, a paint's kind or a filter's parameters: that node is prepared again.
    Content,
    /// Which nodes draw, the page, and clip and mask relations.
    Structure,
}
impl ChangeKind {
    /// A track's kind, or `None` for a control track that draws nothing itself (a layer's weight): its effect
    /// is the changed weight of the tracks it drives, which carry their own kinds.
    /// `form` is the one paint form the track's values keep, if they do; `rest(node, fill)` the form of the
    /// path's paint at rest (a colour over a gradient, an image or `none` changes the paint's kind or presence;
    /// so does a gradient over one of another form). `path(node)` says the node is a path that owns its paints.
    fn of(target: &TrackTarget, numeric: bool, form: Option<PaintForm>, rest: impl Fn(&str, bool) -> Option<PaintForm>,
        path: impl Fn(&str) -> bool) -> Option<Self> {
        use v2::NodeProperty as P;
        Some(match target {
            TrackTarget::LayerWeight { .. } => return None,
            TrackTarget::Node { property: P::Translation | P::Rotation | P::Scale | P::Opacity, .. } if numeric => Self::Placement,
            // The writer drops an invisible node from the generated hierarchy.
            TrackTarget::Node { property: P::Visibility(_), .. } => Self::Structure,
            TrackTarget::Node { node, property: property @ (P::Fill | P::Stroke) } if form.is_some() && rest(node, *property == P::Fill) == form => Self::PaintValue,
            // Opacity scales whatever paint is there: a colour, a gradient or an image.
            TrackTarget::Node { node, property: P::FillOpacity | P::StrokeOpacity } if numeric && path(node) => Self::PaintValue,
            TrackTarget::Node { property: P::Style(p), .. } if matches!(p.name(), "clip-path" | "mask") => Self::Structure,
            TrackTarget::Render(v2::RenderProperty::Background) => Self::Structure,
            _ => Self::Content,
        })
    }
}

/// The node and the authored field a track restyles: a style, paint or stroke-width track's value is mixed from
/// that field as authored, and a changed content field makes the node again with the new value in it. `None`
/// for every other track. (A stroke-width track restyles a text node; a path's stroke width is a number of its
/// own.)
fn restyled_field(target: &TrackTarget) -> Option<(&str, &'static str)> {
    use v2::NodeProperty as P;
    match target {
        TrackTarget::Node { node, property: P::Style(p) } => Some((node.as_str(), p.name())),
        TrackTarget::Node { node, property: P::Fill } => Some((node.as_str(), "fill")),
        TrackTarget::Node { node, property: P::Stroke } => Some((node.as_str(), "stroke")),
        TrackTarget::Node { node, property: P::StrokeWidth } => Some((node.as_str(), "stroke-width")),
        _ => None,
    }
}
/// An item some track restyles, as authored: the typed node (no children, no layers of its own), which a
/// restyle copies and writes its changed fields into, and the authored value of each field such a track names,
/// which that track's value is mixed from.
struct Authored {
    node: v2::Node,
    fields: Vec<(&'static str, serde_json::Value)>,
}
impl Authored {
    /// The authored value of a field a track names; null for one the node does not write.
    fn field(&self, name: &str) -> &serde_json::Value {
        static NULL: serde_json::Value = serde_json::Value::Null;
        self.fields.iter().find(|(field, _)| *field == name).map_or(&NULL, |(_, value)| value)
    }
}

/// A CSS colour (or `none`) as straight-alpha sRGB.
fn paint(here: &str, value: &str) -> Result<Paint> {
    v2::text::paint(value).with_context(|| here.to_string())
}
/// A paint as the CSS colour Moda reads, quantised to 8 bits a channel.
fn css(paint: [f32; 4]) -> String {
    let [r, g, b, a] = paint.map(|c| (c.clamp(0., 1.) * 255.).round() as u8);
    format!("rgba({r}, {g}, {b}, {})", f32::from(a) / 255.)
}
fn quantised(paint: Paint) -> Option<[u8; 4]> {
    paint.filter(|p| p[3] > 0.).map(|p| p.map(|c| (c.clamp(0., 1.) * 255.).round() as u8))
}

/// The HTML Moda's parser reads for a text node: one paragraph per line of `pre-line` text, else one
/// paragraph; each stretch of one run a span with the node's CSS properties and the run's own. When tracks
/// address the node's glyphs, every letter is its own glyph: ligatures are off (`liga`, `clig`, `dlig` and
/// `hlig` 0, after the authored features), so a ligature spanning addressed glyphs draws as its letters.
fn html(kind: &NodeKind, family: &dyn Fn(Option<&v2::FontFiles>) -> String, separate_letters: bool) -> Result<String> {
    Ok(text_markup(kind, family, separate_letters, false)?.0)
}
/// What lays a text node out, and its runs' colours.
struct TextHtml {
    /// The node's HTML with each run naming its paint (`--nf-paint`: the run's index; the empty line's is the run
    /// count) where the drawn HTML (`html`) has its `color`. Runs and shaping are those of the drawn HTML, and text
    /// that differs only in its colours has one layout.
    layout: String,
    /// The paints: each run's colour as Moda draws that `color`, then the empty line's (the node's own).
    colors: Vec<[f32; 4]>,
}
fn text_html(kind: &NodeKind, family: &dyn Fn(Option<&v2::FontFiles>) -> String, separate_letters: bool) -> Result<TextHtml> {
    let (layout, colors) = text_markup(kind, family, separate_letters, true)?;
    Ok(TextHtml { layout, colors })
}
/// A run's flat fill as the CSS colour Moda reads: its own or the node's; white under a gradient or an image
/// (the paint is drawn through the glyphs' coverage), transparent for `none`.
fn run_fill(run: &v2::Run, fill: &v2::Paint) -> Result<String> {
    let fill = match run.fill.as_ref().unwrap_or(fill) {
        v2::Paint::None(_) => None,
        v2::Paint::Css(css) if !css.contains("gradient(") => v2::text::paint(css)?,
        _ => Some([1.; 4]),
    };
    Ok(fill.map(css).unwrap_or_else(|| "rgba(0, 0, 0, 0)".into()))
}
/// A text node's paints as a draw supplies them (`TextNodeSpec::run_colors`): each run's colour, then the empty
/// line's. What `text_markup` returns beside the HTML, without making the HTML.
fn text_colors(kind: &NodeKind) -> Result<Vec<[f32; 4]>> {
    let NodeKind::Text { content, fill, .. } = kind else { unreachable!() };
    content.runs().iter().chain([&v2::Run::default()]).map(|run| Ok(moda_text::vector::run_color(&run_fill(run, fill)?))).collect()
}
/// A text node's HTML and its runs' colours. `named`: each run names its paint (the layout's HTML); else each
/// run's `color` is its flat fill (the HTML as drawn, which the degraded tier rasters).
fn text_markup(kind: &NodeKind, family: &dyn Fn(Option<&v2::FontFiles>) -> String, separate_letters: bool, named: bool) -> Result<(String, Vec<[f32; 4]>)> {
    let NodeKind::Text { content, font_size, font_weight, font_style, font_variation_settings, font_feature_settings, letter_spacing,
        line_height, text_align, direction, white_space, fill, .. } = kind else { unreachable!() };
    let runs = content.runs();
    let style = |run: &v2::Run, paint: Option<usize>| -> Result<(String, String)> {
        let mut style = format!("font-family: '{}'; font-size: {}px; font-weight: {}; font-style: {}; letter-spacing: {}px", family(run.font.as_ref()),
            run.font_size.unwrap_or(*font_size), run.font_weight.unwrap_or(*font_weight).value(),
            match run.font_style.unwrap_or(*font_style) { v2::FontStyle::Normal => "normal", v2::FontStyle::Italic => "italic", v2::FontStyle::Oblique => "oblique" },
            run.letter_spacing.unwrap_or(*letter_spacing));
        match line_height {
            v2::LineHeight::Normal(_) => style.push_str("; line-height: normal"),
            v2::LineHeight::Multiple(multiple) => style.push_str(&format!("; line-height: {multiple}")),
        }
        let variations = run.font_variation_settings.as_ref().unwrap_or(font_variation_settings);
        let features = run.font_feature_settings.as_ref().unwrap_or(font_feature_settings);
        if variations != "normal" { style.push_str(&format!("; font-variation-settings: {variations}")); }
        const NO_LIGATURES: &str = "\"liga\" 0, \"clig\" 0, \"dlig\" 0, \"hlig\" 0";
        match (features != "normal", separate_letters) {
            (true, true) => style.push_str(&format!("; font-feature-settings: {features}, {NO_LIGATURES}")),
            (true, false) => style.push_str(&format!("; font-feature-settings: {features}")),
            (false, true) => style.push_str(&format!("; font-feature-settings: {NO_LIGATURES}")),
            (false, false) => {}
        }
        let fill = run_fill(run, fill)?;
        match paint {
            Some(paint) => style.push_str(&format!("; {}: {paint}", moda_text::types::PAINT_PROPERTY)),
            None => style.push_str(&format!("; color: {fill}")),
        }
        Ok((vork_std::moda::escape_html(&style).replace('"', "&quot;"), fill))
    };
    // Each run's style and colour as drawn, then the empty line's (the node's own).
    let styled = runs.iter().chain([&v2::Run::default()]).enumerate()
        .map(|(paint, run)| style(run, named.then_some(paint))).collect::<Result<Vec<_>>>()?;
    let colors: Vec<[f32; 4]> = styled.iter().map(|(_, fill)| moda_text::vector::run_color(fill)).collect();
    let styles: Vec<String> = styled.into_iter().map(|(style, _)| style).collect();
    let span = |run: usize, text: &str| format!("<span style=\"{}\">{}</span>", styles[run], vork_std::moda::escape_html(text));
    let block = format!("text-align: {}; direction: {}", text_align.css(), match direction { v2::Direction::Ltr => "ltr", v2::Direction::Rtl => "rtl" });
    let laid = v2::laid_out_runs(content, *white_space);
    let html = laid.split(|(c, _)| *c == '\n')
        .map(|line| {
            let mut spans = String::new();
            let mut start = 0;
            while start < line.len() {
                let run = line[start].1;
                let end = line[start..].iter().position(|(_, r)| *r != run).map_or(line.len(), |n| start + n);
                spans.push_str(&span(run, &line[start..end].iter().map(|(c, _)| *c).collect::<String>()));
                start = end;
            }
            if line.is_empty() {
                // An empty line keeps the height of the node's text.
                spans = format!("<span style=\"{}\"></span>", styles[runs.len()]);
            }
            format!("<p style=\"{block}\">{spans}</p>")
        })
        .collect();
    Ok((html, colors))
}

/// For each line of a layout (the source code points its segments hold), the code points of `plain` it holds:
/// its characters in order, the white space at its breaks with it.
fn line_code_points(layout: &[Vec<usize>], plain: &[char]) -> Vec<Vec<usize>> {
    let mut lines = Vec::new();
    let mut next = 0;
    for line in layout {
        let mut held = Vec::new();
        for &cp in line {
            if cp >= plain.len() { continue; }
            held.extend((next..=cp).filter(|&i| plain[i] != '\n'));
            next = next.max(cp+1);
        }
        lines.push(held);
    }
    if let Some(last) = lines.last_mut() { last.extend((next..plain.len()).filter(|&i| plain[i] != '\n')); }
    lines
}
fn union(a: Option<[f64; 4]>, b: [f64; 4]) -> [f64; 4] {
    match a { None => b, Some(a) => [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])] }
}

fn build_text(
    node: &v2::Node, images: &v2::Table<v2::Image>, base: &Path,
    lists: &[Vec<PathBuf>], fonts: &Fonts,
    keyed: Option<&BTreeSet<UnitProperty>>, glyphs_keyed: bool,
    styles: &BTreeMap<String, Arc<crate::animation::style::StyleValue>>, media: &crate::canvas_draw::Media,
) -> Result<Option<TextItem>> {
    v2::Scene::validate_text(node, images)?;
    let files = |font: &v2::FontFiles| -> Vec<PathBuf> {font.files().iter().map(|file|base.join(file)).collect()};
    Ok(match &node.kind {
                NodeKind::Text { font, content, width, white_space, stroke_width, fill, .. } => {
                    let family = |run_font: Option<&v2::FontFiles>| {
                        let list = files(run_font.unwrap_or(font));
                        vork_std::moda::family(lists.iter().position(|l| *l == list).expect("collected font list"))
                    };
                    let TextHtml { layout: layout_html, colors } = text_html(&node.kind, &family, glyphs_keyed)?;
                    let layout_html: Arc<str> = layout_html.into();
                    let here = format!("node {:?}", node.name);
                    let laid = v2::laid_out_runs(content, *white_space);
                    let plain_text: String = laid.iter().map(|(ch, _)| *ch).collect();
                    let nowrap = width.is_none() || *white_space == v2::WhiteSpace::Nowrap;
                    let mut spec = moda_text::ops::TextNodeSpec {
                        node_id: node.name.clone(), html: layout_html.clone(), width: width.map(f64::from), opacity: 1., auto_width: nowrap, line_height_policy: moda_text::types::LineHeightPolicy::Css, ..Default::default()
                    };
                    // The layout is of the text without its colours: each run names its paint, and the draw has them.
                    // So the spec's HTML is the layout's, and a change of colour leaves it as it is.
                    spec.run_colors = Arc::new(colors);
                    // The degraded tier lays out and rasters the HTML as drawn, colours and all: it is made only for
                    // text that tier draws, and there the spec holds it.
                    let drawn = || self::html(&node.kind, &family, glyphs_keyed);
                    let accepted = fonts.lay_out(&spec, layout_html, &here, &drawn)?;
                    let is_degraded = accepted.vector.is_none();
                    if is_degraded { spec.html = drawn()?.into(); }
                    // Without a width the box is the widest line, so text-align places lines within it.
                    if width.is_none() { spec.width = Some(accepted.width); }
                    let plain: Vec<char> = plain_text.chars().collect();
                    let mut units = BTreeMap::new();
                    for unit in [TextUnit::Glyphs, TextUnit::Words] {
                        let ranges = v2::unit_ranges(&plain_text, unit);
                        units.insert(unit, ranges.into_iter().map(|range| {
                            let start = plain_text[..range.start].chars().count();
                            (start..start + plain_text[range].chars().count()).collect::<Vec<usize>>()
                        }).collect::<Vec<_>>());
                    }
                    units.insert(TextUnit::Lines, line_code_points(&accepted.lines, &plain));
                    let mut piece_of = BTreeMap::new();
                    for (unit, pieces) in &units {
                        let mut of = vec![None; plain.len()];
                        for (index, code_points) in pieces.iter().enumerate() {
                            for &cp in code_points { of[cp] = Some(index); }
                        }
                        piece_of.insert(*unit, of);
                    }
                    let run_of: Vec<usize> = laid.iter().map(|(_, run)| *run).collect();
                    let height = accepted.height;
                    // The box Moda's frame walk bounds the node by: its width, and the text's height.
                    let size = [spec.width.unwrap_or(0.), height];
                    let (fills, strokes) = text_paints(node, images, base, styles, media, size, &run_of)?;
                    // A paint as `text_paints` resolves it, for the degraded tier's coverage below.
                    let bounds = Some([0., 0., size[0], size[1]]);
                    let resolve = |field:&str, value:Option<&v2::Paint>, node_value:&v2::Paint| -> Result<PaintSource> {
                        if value.is_none() {
                            if let Some(crate::animation::style::StyleValue::Paint(paint))=styles.get(field).map(|v|v.as_ref()) {
                                return crate::canvas_draw::resolve_animated_paint(images,base,bounds.unwrap(),paint,media);
                            }
                        }
                        crate::canvas_draw::resolve_paint(images,base,value.unwrap_or(node_value),bounds,&format!("{here}: {field}"),media)
                    };
                    let mut degraded_paints = Vec::new();
                    if is_degraded && fills.iter().any(|p| matches!(p, PaintSource::Moda(_))) {
                        for source in &fills {
                            if matches!(source, PaintSource::None) || degraded_paints.iter().any(|(p, _)| p == source) { continue; }
                            let mut coverage = node.kind.clone();
                            if let NodeKind::Text { content, fill: coverage_fill, .. } = &mut coverage {
                                let selected = |value: Option<&v2::Paint>| -> Result<v2::Paint> {
                                    Ok(if resolve("fill",value,fill)? == *source {
                                        v2::Paint::Css("white".into())
                                    } else { v2::Paint::NONE })
                                };
                                if let v2::Content::Runs(runs) = content {
                                    for run in runs { run.fill = Some(selected(run.fill.as_ref())?); }
                                }
                                *coverage_fill = selected(None)?;
                            }
                            degraded_paints.push((source.clone(), self::html(&coverage, &family, false)?));
                        }
                    }
                    let keys = keyed;
                    let pieces_keyed = keys.is_some();
                    if is_degraded {
                        ensure!(!pieces_keyed, "{here}: glyph, word and line tracks on right-to-left or complex-script text are not built yet");
                        ensure!(strokes.iter().all(|s| matches!(s, PaintSource::None)), "{here}: stroke on right-to-left or complex-script text is not built yet");
                    }
                    let (boxes, centre) = if pieces_keyed && !is_degraded {
                        let cp_boxes = accepted.vector.as_ref().map_or_else(Vec::new, |vector| vector.code_point_boxes(plain.len()));
                        let mut boxes = BTreeMap::new();
                        for (unit, pieces) in &units {
                            boxes.insert(*unit, pieces.iter().map(|cps| cps.iter().filter_map(|&cp| cp_boxes[cp]).fold(None, |a, b| Some(union(a, b)))).collect::<Vec<_>>());
                        }
                        let all = cp_boxes.iter().flatten().fold(None, |a, b| Some(union(a, *b)));
                        (boxes, all.map_or(DVec2::ZERO, |b| DVec2::new((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)))
                    } else {
                        (BTreeMap::new(), DVec2::ZERO)
                    };
                    Some(TextItem {
                        spec, run_of: Arc::new(run_of), code_points: plain.len(), clip_source: false, units: Arc::new(units), piece_of: Arc::new(piece_of), boxes: Arc::new(boxes), centre, fill: fills, stroke: strokes, stroke_width: f64::from(*stroke_width),
                        pieces_keyed, fill_keyed: keys.is_some_and(|k| k.contains(&UnitProperty::Fill)), layout: accepted, degraded_paints, size,
                    })
                }
                NodeKind::Node {} | NodeKind::Path { .. } => None,
                _ => bail!("node {:?}: only group, text and path nodes are built in a 2D canvas", node.name),
            })
}

/// Each code point's fill and stroke: its run's, else the node's (a paint track's value, in `styles`, when the run
/// has none of its own), resolved in the text's box. `run_of` is each code point's run.
fn text_paints(node: &v2::Node, images: &v2::Table<v2::Image>, base: &Path,
    styles: &BTreeMap<String, Arc<crate::animation::style::StyleValue>>, media: &crate::canvas_draw::Media,
    size: [f64; 2], run_of: &[usize]) -> Result<(Vec<PaintSource>, Vec<PaintSource>)> {
    let NodeKind::Text { content, stroke, stroke_width, fill, .. } = &node.kind else { unreachable!() };
    let here = format!("node {:?}", node.name);
    let runs = content.runs();
    let bounds = Some([0., 0., size[0], size[1]]);
    let resolve = |field:&str, value:Option<&v2::Paint>, node_value:&v2::Paint| -> Result<PaintSource> {
        if value.is_none() {
            if let Some(crate::animation::style::StyleValue::Paint(paint))=styles.get(field).map(|v|v.as_ref()) {
                return crate::canvas_draw::resolve_animated_paint(images,base,bounds.unwrap(),paint,media);
            }
        }
        crate::canvas_draw::resolve_paint(images,base,value.unwrap_or(node_value),bounds,&format!("{here}: {field}"),media)
    };
    let paints = |field: &str, node_value: &v2::Paint, run_value: &dyn Fn(&v2::Run) -> Option<v2::Paint>| -> Result<Vec<PaintSource>> {
        let per_run = runs.iter().map(|run| resolve(field,run_value(run).as_ref(),node_value)).collect::<Result<Vec<_>>>()?;
        resolve(field,None,node_value)?;
        Ok(run_of.iter().map(|&run|per_run[run].clone()).collect())
    };
    let fills = paints("fill", fill, &|run| run.fill.clone())?;
    let strokes = if *stroke_width > 0. { paints("stroke", stroke, &|run| run.stroke.clone())? } else { vec![PaintSource::None; run_of.len()] };
    Ok((fills, strokes))
}

/// The text `kept` with the paints `node` has: what `build_text` makes of `node` when only its fill or stroke
/// differs from the node `kept` was made from, for text Moda's vector lane draws. Its layout, box, pieces and HTML
/// do not depend on those, so none is made again or copied (the tables are shared): the runs' colours and each
/// code point's paints are made.
fn repainted_text(node: &v2::Node, kept: &TextItem, images: &v2::Table<v2::Image>, base: &Path,
    styles: &BTreeMap<String, Arc<crate::animation::style::StyleValue>>, media: &crate::canvas_draw::Media) -> Result<TextItem> {
    v2::Scene::validate_text(node, images)?;
    let (fill, stroke) = text_paints(node, images, base, styles, media, kept.size, &kept.run_of)?;
    let mut spec = kept.spec.clone();
    spec.run_colors = Arc::new(text_colors(&node.kind)?);
    // Every part is named, so a part added to the text is placed here as shared or as made again.
    Ok(TextItem {
        spec, fill, stroke,
        run_of: kept.run_of.clone(), units: kept.units.clone(), piece_of: kept.piece_of.clone(), boxes: kept.boxes.clone(), layout: kept.layout.clone(),
        code_points: kept.code_points, clip_source: kept.clip_source, centre: kept.centre, stroke_width: kept.stroke_width,
        pieces_keyed: kept.pieces_keyed, fill_keyed: kept.fill_keyed, degraded_paints: kept.degraded_paints.clone(), size: kept.size,
    })
}

/// A canvas scene's font lists, one family each: every text node's and every run's, authored or keyed.
fn font_lists(scene: &Scene, base: &Path) -> Result<Vec<Vec<PathBuf>>> {
    let mut lists: Vec<Vec<PathBuf>> = Vec::new();
    let files = |font: &v2::FontFiles| -> Vec<PathBuf> { font.files().iter().map(|file| base.join(file)).collect() };
    fn collect(nodes: &[v2::Node], files: &dyn Fn(&v2::FontFiles) -> Vec<PathBuf>, lists: &mut Vec<Vec<PathBuf>>) {
        for node in nodes {
            if let NodeKind::Text { font, content, .. } = &node.kind {
                let mut fonts = vec![font.clone()];
                fonts.extend(content.runs().iter().filter_map(|run| run.font.clone()));
                for font in &fonts {
                    let list = files(font);
                    if !lists.contains(&list) { lists.push(list); }
                }
            }
            collect(&node.children, files, lists);
        }
    }
    collect(&scene.nodes, &files, &mut lists);
    let owned=scene.node_owned_layers()?;
    for tracks in scene.animation.iter().map(|l|&l.tracks).chain(owned.iter().map(|l|&l.layer.tracks)) {
        for track in tracks.iter().flatten() {
            if track.path.last().is_some_and(|p|p=="font") {
                if let v2::Values::Inline(values)=&track.values {for value in values {
                    let font:v2::FontFiles=serde_json::from_value(value.clone())?;
                    let list=files(&font);if !lists.contains(&list){lists.push(list);}
                }}
            }
            if track.path.last().is_some_and(|p|p=="content") {
                if let v2::Values::Inline(values)=&track.values {
                    for value in values {
                        let content:v2::Content=serde_json::from_value(value.clone())?;
                        for run in content.runs().iter() {if let Some(font)=&run.font {let list=files(font);if !lists.contains(&list){lists.push(list);}}}
                    }
                }
            }
        }
    }
    Ok(lists)
}

/// The files a canvas reads that the scene's other sources do not list: its font files and the files of the
/// images its paints name, authored or keyed. False when one cannot be listed (a painted movie, frame sequence or
/// rendered image, tracks that do not resolve), so a session keeps nothing of such a document and reads it again
/// for every render. An image no paint names is not read and is not a source.
pub(crate) fn sources(scene: &Scene, base: &Path, files: &mut BTreeSet<PathBuf>) -> bool {
    let mut complete = true;
    // Nodes inside nested worlds' canvases are read too: more files than this document's own, never fewer.
    match font_lists(scene, base) {
        Ok(lists) => files.extend(lists.into_iter().flatten()),
        Err(_) => complete = false,
    }
    let mut painted: BTreeSet<&str> = BTreeSet::new();
    fn authored<'a>(nodes: &'a [v2::Node], painted: &mut BTreeSet<&'a str>) {
        for node in nodes {
            let paints: Vec<&v2::Paint> = match &node.kind {
                NodeKind::Path { fill, stroke, .. } => vec![fill, stroke],
                NodeKind::Text { fill, stroke, content, .. } => {
                    let mut paints = vec![fill, stroke];
                    if let v2::Content::Runs(runs) = content {
                        paints.extend(runs.iter().flat_map(|run| run.fill.iter().chain(&run.stroke)));
                    }
                    paints
                }
                _ => Vec::new(),
            };
            for paint in paints {
                if let v2::Paint::Image(binding) = paint { painted.insert(&binding.image); }
            }
            authored(&node.children, painted);
        }
    }
    authored(&scene.nodes, &mut painted);
    // A keyed paint or keyed text content names its image inside a track value: every `image` name in any
    // track value counts, which lists more than the paints' own and never fewer.
    fn keyed<'a>(value: &'a serde_json::Value, painted: &mut BTreeSet<&'a str>) {
        match value {
            serde_json::Value::Object(fields) => {
                if let Some(name) = fields.get("image").and_then(serde_json::Value::as_str) { painted.insert(name); }
                for value in fields.values() { keyed(value, painted); }
            }
            serde_json::Value::Array(values) => for value in values { keyed(value, painted); },
            _ => {}
        }
    }
    let owned = scene.node_owned_layers();
    if owned.is_err() { complete = false; }
    let owned = owned.unwrap_or_default();
    for tracks in scene.animation.iter().map(|layer| &layer.tracks).chain(owned.iter().map(|owned| &owned.layer.tracks)) {
        for track in tracks.iter().flatten() {
            if let v2::Values::Inline(values) = &track.values { for value in values { keyed(value, &mut painted); } }
        }
    }
    for name in painted {
        // An unknown name is the document's error to report.
        let Some(image) = scene.images.get(name) else { continue };
        if image.render.is_some() { complete = false; }
        let Some(file) = image.file.as_deref() else { continue };
        let path = base.join(file);
        if crate::footage::is_media(&path) { complete = false; } else { files.insert(path); }
    }
    complete
}

/// Whether any of `nodes` or their descendants is a 3D world (`composition: {space: "3d"}`).
pub(crate) fn has_worlds(nodes: &[v2::Node]) -> bool {
    nodes.iter().any(|node| node.composition.as_ref().is_some_and(|c| c.space == v2::Space::World) || has_worlds(&node.children))
}

impl Document {
    /// Lay out every text node of a canvas scene and compile its tracks.
    pub(crate) fn new(scene: &Scene, base: &Path) -> Result<Self> {
        ensure!(scene.is_canvas(), "not a 2D canvas");
        let [w, h] = scene.render.resolution;
        let composition = scene.composition.as_ref().expect("a canvas has a composition");
        let aspect = v2::AspectRatio::parse(composition.preserve_aspect_ratio.as_deref()).map_err(anyhow::Error::msg)?;
        let (scale, offset) = aspect.placement(scene.canvas_view_box(), [w, h]);
        // The worlds' insides are documents of their own; this one holds their nodes, empty.
        let whole = scene;
        let outer = scene.without_compositions()?;
        let scene: &Scene = &outer;
        let lists = font_lists(scene, base)?;
        let host = Arc::new(vork_std::moda::host(&vork_std::moda::read(&lists)?)?);
        let fonts = Fonts { store: std::rc::Rc::new(host.vector_font_store_lenient()), host, degraded: Default::default(), layouts: Default::default() };
        // Every track with its target: the scene's and the nodes' own.
        let mut tracks: Vec<(&v2::Track, TrackTarget)> = Vec::new();
        for (layer, targets) in scene.animation.iter().zip(scene.track_targets()?) {
            tracks.extend(layer.tracks.iter().flatten().zip(targets));
        }
        for owned in scene.node_owned_layers()? {
            tracks.extend(owned.layer.tracks.iter().flatten().zip(owned.targets));
        }
        // Which pieces' fields each text node's tracks key, and CSS colour keys read once.
        let mut keyed: BTreeMap<String, BTreeSet<UnitProperty>> = BTreeMap::new();
        let mut glyphs_keyed: BTreeSet<String> = BTreeSet::new();
        for (_, target) in &tracks {
            let TrackTarget::Text { node, unit, property, .. } = target else { continue };
            keyed.entry(node.clone()).or_default().insert(*property);
            if *unit == TextUnit::Glyphs { glyphs_keyed.insert(node.clone()); }

        }
        let media = crate::canvas_draw::Media::new(scene, base)?;
        // The scene's nodes in paint order, each where the scene holds it, with its parent's item: a walk with
        // its own stack, so the tree's depth costs no call stack.
        let mut flat: Vec<SceneNode> = Vec::new();
        let mut open = vec![(None, scene.nodes.iter())];
        while let Some((parent, siblings)) = open.last_mut() {
            match siblings.next() {
                Some(node) => {
                    let parent = *parent;
                    open.push((Some(flat.len()), node.children.iter()));
                    flat.push(SceneNode { node, parent });
                }
                None => { open.pop(); }
            }
        }
        let mut items = Vec::with_capacity(flat.len());
        // The fields each node's tracks restyle.
        let mut restyled: HashMap<&str, Vec<&'static str>> = HashMap::new();
        for (node, field) in tracks.iter().filter_map(|(_, target)| restyled_field(target)) {
            let fields = restyled.entry(node).or_default();
            if !fields.contains(&field) { fields.push(field); }
        }
        let mut authored: Vec<Option<Box<Authored>>> = Vec::with_capacity(flat.len());
        let mut references = Vec::with_capacity(flat.len());
        let mut worlds = Vec::new();
        let mut rest: Vec<Rest> = Vec::with_capacity(flat.len());
        for &SceneNode { node, parent } in &flat {
            let text = build_text(node, &scene.images, base, &lists, &fonts, keyed.get(&node.name), glyphs_keyed.contains(&node.name), &BTreeMap::new(), &media)?;
            // A stroke-width track restyles a text node only.
            let fields: Vec<&'static str> = restyled.get(node.name.as_str()).into_iter().flatten().copied()
                .filter(|field| *field != "stroke-width" || text.is_some()).collect();
            authored.push(if fields.is_empty() { None } else {
                let mut source_node=node.without_children();
                source_node.animation.clear();
                // The fields' authored values, as the node writes them: read from the written node, which is not kept.
                let mut source=serde_json::to_value(&source_node)?;
                // These computed defaults are omitted by the authoring serializer.
                if let NodeKind::Text {font_variation_settings,font_feature_settings,..}=&source_node.kind {
                    source["font-variation-settings"]=serde_json::json!(font_variation_settings);
                    source["font-feature-settings"]=serde_json::json!(font_feature_settings);
                }
                let fields = fields.into_iter().filter_map(|field| Some((field, source.get(field)?.clone()))).collect();
                Some(Box::new(Authored { node: source_node, fields }))
            });
            let world = node.composition.as_ref().is_some_and(|c| c.space == v2::Space::World);
            let kind = match (&node.kind, &text) {
                (_, Some(text)) => crate::canvas_draw::Kind::Text { size: text.size },
                _ if world => {
                    // The world's picture fills the canvas's box, at the output's pixel density.
                    let [x, y, vw, vh] = whole.canvas_view_box().map(f64::from);
                    let px = [(vw * scale[0]).round().max(1.) as u32, (vh * scale[1]).round().max(1.) as u32];
                    worlds.push(WorldPart { item: items.len(), scene: whole.world_part(&node.name, px)? });
                    crate::canvas_draw::Kind::World { bounds: [x, y, x + vw, y + vh] }
                }
                (NodeKind::Path { .. }, _) => crate::canvas_draw::Kind::Path(Arc::new(crate::canvas_draw::path_item(&scene.images, base, node, &media)?)),
                _ => crate::canvas_draw::Kind::Group,
            };
            let filter = match &node.filter {
                Some(filter) => crate::canvas_draw::parse_filter(filter).with_context(|| format!("node {:?}", node.name))?,
                None => Default::default(),
            };
            let draw = crate::canvas_draw::Node { parent, clip_path: None, mask: None, filter, kind };
            rest.push(Rest { parent, transform: node.transform.clone(), opacity: node.opacity, text: text.is_some(),
                path: match &draw.kind { crate::canvas_draw::Kind::Path(path) => Some(path.values), _ => None } });
            items.push(Item { name: node.name.clone(), visible: node.visibility.visible, text, node: draw });
            references.push((node.clip_path.clone(), node.mask.clone()));
        }
        let by_name: HashMap<String, usize> = items.iter().enumerate().filter(|(_, item)| !item.name.is_empty()).map(|(i, item)| (item.name.clone(), i)).collect();
        // clip-path and mask name nodes; the scene checked that they exist.
        for (i, (clip_path, mask)) in references.into_iter().enumerate() {
            items[i].node.clip_path = clip_path.and_then(|name| by_name.get(&name).copied());
            items[i].node.mask = mask.and_then(|name| by_name.get(&name).copied());
        }
        let children = Children::new(&rest);
        let mut clip_sources = vec![false; items.len()];
        let mut pending: Vec<_> = items.iter().filter_map(|item| item.node.clip_path).collect();
        for (track,target) in &tracks {
            if matches!(target,TrackTarget::Node{property:v2::NodeProperty::Style(p),..} if p.name()=="clip-path") {
                if let v2::Values::Inline(values)=&track.values {pending.extend(values.iter().filter_map(|v|v.as_str().and_then(|name|by_name.get(name).copied())));}
            }
        }
        while let Some(source) = pending.pop() {
            if std::mem::replace(&mut clip_sources[source], true) { continue; }
            pending.extend(children.of(source).iter().copied());
        }
        for (source, _) in clip_sources.into_iter().enumerate().filter(|(_,used)| *used) {
            if let Some(text) = &mut items[source].text { text.clip_source = true; }
        }
        // Static line counts are authoritative only if no track can change layout.
        // Animated layouts are checked against the sampled pieces in frame().
        let dynamic_lines: BTreeSet<_> = tracks.iter()
            .filter_map(|(_, target)| target.changes_text_pieces(TextUnit::Lines)).collect();
        for (track, target) in &tracks {
            let TrackTarget::Text { node, unit: TextUnit::Lines, index, .. } = target else { continue };
            if dynamic_lines.contains(node.as_str()) { continue; }
            let Some(text) = by_name.get(node).and_then(|&i| items[i].text.as_ref()) else { continue };
            let count = text.units[&TextUnit::Lines].len();
            ensure!(*index < count, "track {}: node {node:?} has {count} lines; index {index} is past the end", v2::display_path(&track.path));
        }
        let mut plan = crate::animation::Plan::compile_with_buffers(scene, base, None)?;
        plan.mark_canvas_nodes(&mut flat)?;
        let samples = RefCell::new(plan.samples_naming_changes());
        let mut kinds: Vec<Option<ChangeKind>> = {
            // Paths own their paints; text paints are part of glyph batches.
            let path = |name: &str| by_name.get(name).and_then(|&i| match &items[i].node.kind {
                crate::canvas_draw::Kind::Path(path) if items[i].text.is_none() => Some(path),
                _ => None,
            });
            let samples = samples.borrow();
            plan.sampled(&samples).zip(plan.sampled_paint_forms()).map(|((target, sample, _, _), form)|
                ChangeKind::of(target, matches!(sample, crate::animation::Sample::Numeric(_)), form,
                    |name, fill| path(name).and_then(|path| path.paint_form(fill)), |name| path(name).is_some())).collect()
        };
        // A paint that any track can turn into another form is content for every track on it: layers mix.
        {
            let samples = samples.borrow();
            let paint_of = |target: &TrackTarget| match target {
                TrackTarget::Node { node, property: property @ (v2::NodeProperty::Fill | v2::NodeProperty::Stroke) } =>
                    Some((node.clone(), *property == v2::NodeProperty::Fill)),
                _ => None,
            };
            let unstable: BTreeSet<(String, bool)> = plan.sampled(&samples).zip(&kinds)
                .filter(|(_, kind)| **kind != Some(ChangeKind::PaintValue))
                .filter_map(|((target, ..), _)| paint_of(target)).collect();
            for ((target, ..), kind) in plan.sampled(&samples).zip(&mut kinds) {
                if paint_of(target).is_some_and(|paint| unstable.contains(&paint)) { *kind = Some(ChangeKind::Content); }
            }
        }
        let track_items: Vec<Option<usize>> = plan.sampled(&samples.borrow()).map(|(target, ..)| match target {
            TrackTarget::Node { node, .. } | TrackTarget::Text { node, .. } => by_name.get(node).copied(),
            _ => None,
        }).collect();
        let track_parts: Vec<u8> = plan.sampled(&samples.borrow()).map(|(target, ..)| match target {
            TrackTarget::Node { property: v2::NodeProperty::Translation | v2::NodeProperty::Rotation | v2::NodeProperty::Scale, .. } => crate::canvas_draw::change::TRANSFORM,
            TrackTarget::Node { property: v2::NodeProperty::Opacity, .. } => crate::canvas_draw::change::OPACITY,
            _ => crate::canvas_draw::change::REST,
        }).collect();
        // The sampler names tracks by flat index, and these tables are in authored-track order: the same order
        // when every track is authored, as a canvas's are (only an imported clip's channels have no target).
        ensure!(kinds.len() == plan.track_count() && track_items.len() == plan.track_count(),
            "internal: a canvas track without an authored target");
        let mut item_tracks_first = vec![0u32; items.len() + 1];
        for item in track_items.iter().flatten() { item_tracks_first[item + 1] += 1; }
        for item in 0..items.len() { item_tracks_first[item + 1] += item_tracks_first[item]; }
        let mut item_tracks = vec![0u32; item_tracks_first[items.len()] as usize];
        {
            let mut next = item_tracks_first.clone();
            for (track, item) in track_items.iter().enumerate() {
                if let Some(item) = item {
                    item_tracks[next[*item] as usize] = track as u32;
                    next[*item] += 1;
                }
            }
        }
        let page_tracks: Vec<u32> = plan.sampled(&samples.borrow()).enumerate()
            .filter(|(_, (target, ..))| matches!(target, TrackTarget::Render(v2::RenderProperty::Background)))
            .map(|(track, _)| track as u32).collect();
        let item_marks = RefCell::new(vec![0; items.len()]);
        let background = match &scene.render.background {
            v2::Background::Css(color) => Some(vork_std::moda::css_color(color).context("render.background")?),
            _ => None,
        };
        let working = composition.working_space();
        let page = crate::canvas_draw::Page { size: [w, h], scale, offset, background, working: crate::canvas_draw::working_color(working) };
        let text_specs = RefCell::new((0..items.len()).map(|_| None).collect());
        let styles = RefCell::new((0..items.len()).map(|_| None).collect());
        // The precision rule: the canvas holds a world, it authors a view or an image graph, or it blends in
        // linear light.
        let float = !worlds.is_empty() || composition.view.is_some() || !composition.compositor.is_empty() || working.is_linear();
        let graph_images = whole.compositor_image_reads(whole.compositor()).into_iter()
            .filter_map(|name| whole.images.get(&name).map(|image| image.time.clone()).map(|time| (name, time))).collect();
        Ok(Self { view: (float || working != v2::WorkingColorSpace::Srgb).then(|| whole.clone()), float, working, graph_images, images: scene.images.clone(), base: base.into(), font_lists: lists, keyed, glyphs_keyed, authored, styles, size: [w, h], items, rest, children, by_name, fonts, plan, samples, page, worlds, media: RefCell::new(media),
            retained: RefCell::new(None), text_specs, evaluated: Default::default(), stale: std::cell::Cell::new(true), text: Default::default(),
            #[cfg(test)]
            id: { static IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1); IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) },
            #[cfg(test)]
            snapshots: Default::default(),
            kinds, changes: Default::default(), track_items, track_parts, item_tracks_first, item_tracks, page_tracks, item_marks })
    }
    /// Item `i` as authored. Only an item a track restyles is read this way, and `new` kept exactly those.
    fn authored(&self, i: usize) -> Result<&Authored> {
        self.authored[i].as_deref().with_context(|| format!("internal: node {:?} is restyled by a track and was not kept as authored", self.items[i].name))
    }
    /// Read static images once into Moda's asset store.
    pub(crate) fn install_images(&self) -> Result<()> {
        let nodes: Vec<&crate::canvas_draw::Node> = self.items.iter().map(|item| &item.node).collect();
        let paints: Vec<_> = self.items.iter().filter_map(|item| item.text.as_ref())
            .flat_map(|text| text.fill.iter().chain(&text.stroke)).collect();
        crate::canvas_draw::install_images(&nodes, &paints)
    }
    /// Whether the canvas keeps a float picture (RGBA16F) instead of the 8-bit one a browser's canvas keeps:
    /// only when something in it needs the precision. It holds a world, whose viewed picture arrives floating
    /// point; it authors a view or an image graph (`composition.compositor`), which read and return float
    /// colour; or it blends in linear light (`Linear Rec.709`, `Linear Rec.2020`, `ACEScg`), whose dark
    /// values eight bits cannot hold. An encoded space (sRGB, Display P3) holds its code values in eight bits.
    /// A float picture is rounded once, where it reaches an 8-bit destination; an 8-bit picture is copied as
    /// it is.
    pub(crate) fn float(&self) -> bool {
        self.float
    }
    /// The encoding the canvas blends in.
    pub(crate) fn working(&self) -> v2::WorkingColorSpace {
        self.working
    }
    /// The canvas's picture at `request` as the world that reads it takes it: associated sRGB code values,
    /// Rgba8Unorm as drawn or, where the picture is not that already (a float canvas, an authored view,
    /// another working encoding), Rgba16Float with its image graph and its view applied
    /// (`Pictures::reader_stage`). `errors`: the shader diagnostics of the frame the picture is drawn for, where its graph and its
    /// view record a result they cannot use; that frame refuses it where it is collected. Without them each
    /// one's own error word is read back here, which waits for the GPU.
    pub(crate) fn draw(&self, engine: &crate::canvas_draw::Engine, worlds: &Pictures, renderer: &Renderer, request: &vork_time::Request,
        errors: Option<&wgpu::Buffer>) -> Result<wgpu::Texture> {
        let frame = self.frame(request)?;
        let drawn = worlds.draw(renderer, self, &frame, engine, errors)?;
        if let (Some(graph), None) = (&worlds.graph, errors) { graph.check(renderer)?; }
        let Some(view) = &worlds.view else { return Ok(drawn) };
        // Each draw's viewed picture is a texture of its own: its readers are recorded in commands submitted
        // later (a world's shutter samples, a frame's finish), after further draws.
        let viewed = crate::gpu_memory::texture(&renderer.device, &wgpu::TextureDescriptor {
            label: Some("canvas viewed picture"), size: drawn.size(), mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format: view.format(renderer),
            // COPY_SRC: tests read the picture back.
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        });
        let mut encoder = renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("canvas view") });
        view.encode(renderer, &mut encoder, &drawn, &viewed, &frame.request, errors)?;
        renderer.queue.submit([encoder.finish()]);
        if errors.is_none() { view.check(renderer)?; }
        Ok(viewed)
    }

    /// Frames evaluated so far, by what changed since the frame before.
    pub(crate) fn change_counters(&self) -> serde_json::Value {
        let [unchanged, placement, paint_value, content, structure] = self.changes.get();
        serde_json::json!({"unchanged": unchanged, "placement": placement, "paint_value": paint_value,
            "content": content, "structure": structure})
    }

    /// One instant: every node's animated values (transform, visibility, opacity, a path's keyed fields), and the
    /// batches of each text node in paint order: its Moda specs with its pieces' motion and paint.
    ///
    /// The document's samples and its evaluated table move together: the sampler reports the tracks that changed
    /// since its last call, and that is what differs from the table. A frame that fails after sampling leaves no
    /// table, so the next one is evaluated whole. The values are written into the table; the frame returned is the
    /// instant and its worlds' clocks.
    fn frame(&self, request: &vork_time::Request) -> Result<Frame> {
        let frame = self.evaluate(request).and_then(|(frame, changed)| {
            self.write_rows(&changed)?;
            Ok(frame)
        });
        if frame.is_err() { self.evaluated.borrow_mut().take(); }
        frame
    }

    /// Write what a frame evaluated into the installed scene, from the evaluated table: the frame's values are in
    /// the scene when it returns, and the scene's stamps are the account of what changed. A frame that evaluated
    /// some items writes their rows; one that evaluated every item, or changed something no item names, or an
    /// item's form, brings the scene whole (`Retained::update`).
    ///
    /// Two cases wait for the plan, and leave `stale` set for it: there is no scene yet (the first install), and a
    /// whole write of a canvas with worlds. A world's picture is drawn after the frame is evaluated and is part of
    /// how its item is lowered, so lowering before the plan has it would lower the canvas twice.
    fn write_rows(&self, changed: &Changed) -> Result<()> {
        let items = match changed {
            Changed::Nothing => return Ok(()),
            Changed::Items(items, true) => Some(&items[..]),
            Changed::Items(_, false) | Changed::Every => None,
        };
        let mut retained = self.retained.borrow_mut();
        // Until the scene is written it is owed everything: a failure below, or a wait for the plan, leaves it so.
        let stale = self.stale.replace(true);
        let Some(scene) = retained.as_mut() else { return Ok(()) };
        let items = items.filter(|_| !stale);
        if items.is_none() && !self.worlds.is_empty() { return Ok(()); }
        let _profile = moda_engine::prof::span("canvas.rows");
        let evaluated = self.evaluated.borrow();
        let state = &evaluated.as_ref().expect("the frame's table").state;
        let mut page = self.page.clone();
        page.background = state.background;
        let node = |i: usize| state.nodes[i].as_deref().unwrap_or(&self.items[i].node);
        let written = match items {
            Some(items) => scene.write(&page, node, &state.values, items)?,
            None => false,
        };
        if !written {
            if !self.worlds.is_empty() { return Ok(()); }
            let nodes: Vec<&crate::canvas_draw::Node> = (0..self.items.len()).map(node).collect();
            scene.update(&page, &nodes, &state.values)?;
        }
        self.stale.set(false);
        Ok(())
    }

    /// Evaluate into the table. Returns the frame and what of the table it changed.
    fn evaluate(&self, request: &vork_time::Request) -> Result<(Frame, Changed)> {
        let _frame_profile = moda_engine::prof::span("canvas.evaluate");
        let time = request.numeric()?;
        let mut samples = self.samples.borrow_mut();
        {
            let _sample = moda_engine::prof::span("canvas.evaluate.sample");
            self.plan.sample_request_into(time, request, &mut samples)?;
        }
        let samples = &*samples;
        // The tracks whose sample or weight changed, control tracks included, are the frame's account of what
        // changed: none means the evaluated state stands (the reuse test), and the frame's kind is the largest
        // kind among those that draw; `None` when none does. Continuously advancing clocks are not among them.
        let changed_tracks = samples.changed();
        let (same, change) = match self.evaluated.borrow().as_ref() {
            None => (false, Some(ChangeKind::Structure)),
            Some(old) => {
                let kind = changed_tracks.iter().map(|&track| self.kinds[track as usize]).max().flatten();
                // A node arriving or leaving on its clock changes which nodes draw.
                let same_presence = !self.plan.has_clocks() || (old.present.len() == samples.clocks.len()
                    && old.present.iter().zip(&samples.clocks).all(|(a,b)| *a == b.is_some()));
                if same_presence { (changed_tracks.is_empty(), kind) } else { (false, Some(ChangeKind::Structure)) }
            }
        };
        let mut counts = self.changes.get();
        counts[change.map_or(0, |kind| kind as usize + 1)] += 1;
        self.changes.set(counts);
        let scene_instant = request.exact()?;
        // The image graph's images follow scene time (a movie shown under a node's clock is not also one of
        // them: `validate_media_clocks`), as a world's compositor images do.
        let images = self.graph_images.iter().map(|(name, time)| {
            let base = time.as_ref().map(|time| vork_time::exact::Germ::point(time.exact())).unwrap_or_else(|| scene_instant.clone());
            Ok((name.clone(), self.plan.image_time_exact(samples, &scene_instant, name, base)?))
        }).collect::<Result<Vec<_>>>()?;
        // World clocks advance even when the canvas around them stands.
        let worlds = |shown: &[bool]| -> Vec<Option<vork_time::Request>> {
            self.worlds.iter().map(|world| shown[world.item]
                .then(|| self.plan.local_instant(samples, world.item, &scene_instant)).flatten().map(vork_time::Request::Exact)).collect()
        };
        if same {
            // The table stands, and the scene is owed nothing new.
            let evaluated = self.evaluated.borrow();
            let state = &evaluated.as_ref().expect("the state that stands").state;
            self.media.borrow_mut().update(&self.plan, samples, &scene_instant, &state.shown)?;
            return Ok((Frame { request: request.clone(), worlds: worlds(&state.shown), images }, Changed::Nothing));
        }
        // Do not publish a partially evaluated state after an error.
        let previous = self.evaluated.borrow_mut().take();
        let mut styled = self.styles.borrow_mut();
        let mut specs = self.text_specs.borrow_mut();
        // Which nodes draw, and the page, stand until a track of that kind changes.
        let (background, shown, present) = match previous {
            Some(mut old) if change < Some(ChangeKind::Structure) => {
                let _profile = moda_engine::prof::span("canvas.evaluate.changed");
                self.media.borrow_mut().update(&self.plan, samples, &scene_instant, &old.state.shown)?;
                // What changed, named by the tracks that changed: each one's item, evaluated again in place.
                let (items, bound) = self.changed_items(samples);
                if let Some(same_draws) = self.evaluate_changed(samples, &mut old.state, &items, &mut styled, &mut specs)? {
                    #[cfg(debug_assertions)]
                    self.assert_whole(samples, &old.state, &old.present, &mut styled, &mut specs)?;
                    let worlds = worlds(&old.state.shown);
                    if !same_draws { self.text.borrow_mut().take(); }
                    *self.evaluated.borrow_mut() = Some(old);
                    return Ok((Frame { request: request.clone(), worlds, images }, Changed::Items(items, bound)));
                }
                // A scale reached or left zero: which nodes draw changed after all, and every item is evaluated.
                (old.state.background, std::mem::take(&mut old.state.shown), old.present)
            }
            previous => {
                let mut present = previous.map(|old| old.present).unwrap_or_default();
                present.clear();
                if self.plan.has_clocks() { present.extend(samples.clocks.iter().map(Option::is_some)); }
                let shown = self.shown(samples);
                self.media.borrow_mut().update(&self.plan, samples, &scene_instant, &shown)?;
                (self.background(samples), shown, present)
            }
        };
        let _nodes_profile = moda_engine::prof::span("canvas.evaluate.nodes");
        let count = self.items.len();
        let mut state = FrameState { background, nodes: Vec::with_capacity(count), values: Vec::with_capacity(count), draws: Vec::with_capacity(count),
            shown, drawing: Vec::with_capacity(count) };
        for i in 0..count {
            // Items are parents first.
            let parent_drawing = self.rest[i].parent.is_none_or(|p| state.drawing[p]);
            state.nodes.push(None);
            state.values.push(self.unevaluated(i));
            state.draws.push(Vec::new());
            let drawing = self.evaluate_item(samples, i, state.shown[i], parent_drawing, &mut styled, &mut specs,
                &mut state.nodes[i], &mut state.values[i], &mut state.draws[i])?;
            state.drawing.push(drawing);
        }
        self.check_references(&state.nodes)?;
        let worlds = worlds(&state.shown);
        // Every item was evaluated: the text builder is owed its batches.
        self.text.borrow_mut().take();
        *self.evaluated.borrow_mut() = Some(Evaluated { present, state });
        Ok((Frame { request: request.clone(), worlds, images }, Changed::Every))
    }

    /// The values an item's slot holds before its first evaluation writes them.
    fn unevaluated(&self, i: usize) -> crate::canvas_draw::Values {
        crate::canvas_draw::Values { style_revision: 0, transform: self.rest[i].transform.clone(), visible: false, opacity: 0., path: None, texts: 0,
            text_composition: Vec::new(), text_revision: 0, world: None }
    }

    /// Item `i`'s tracks, ascending: the order they blend in.
    fn tracks_of(&self, i: usize) -> &[u32] {
        &self.item_tracks[self.item_tracks_first[i] as usize..self.item_tracks_first[i + 1] as usize]
    }

    /// The page's background at this sample.
    fn background(&self, samples: &crate::animation::Samples) -> Paint {
        let mut background = self.page.background;
        for &track in &self.page_tracks {
            let Some((_, crate::animation::Sample::Style(value), weight, _)) = self.plan.sampled_at(samples, track as usize) else { continue };
            if weight == 0. { continue; }
            let crate::animation::style::StyleValue::Paint(crate::animation::style::Paint::Color(color)) = value.as_ref() else { unreachable!("canvas backgrounds key CSS colours") };
            background = Some(crate::animation::style::color_mix(background.unwrap_or([0.; 4]), *color, weight));
        }
        background
    }

    /// Which nodes draw at this sample: each item's visibility through its visibility tracks, false when the node
    /// is absent on its clock (§8 rule 1) or its parent does not draw. Items are parents first.
    fn shown(&self, samples: &crate::animation::Samples) -> Vec<bool> {
        let mut shown: Vec<bool> = Vec::with_capacity(self.items.len());
        for (i, item) in self.items.iter().enumerate() {
            let mut visible = item.visible;
            for &track in self.tracks_of(i) {
                if self.kinds[track as usize] != Some(ChangeKind::Structure) { continue; }
                if let Some((TrackTarget::Node { property: v2::NodeProperty::Visibility(v2::Ray::All), .. }, crate::animation::Sample::Bool(value), weight, additive)) = self.plan.sampled_at(samples, track as usize) {
                    if weight >= 0.5 && !additive { visible = *value; }
                }
            }
            let draws = visible && !self.plan.absent(samples, i) && self.rest[i].parent.is_none_or(|p| shown[p]);
            shown.push(draws);
        }
        shown
    }

    /// The items the tracks the sampler reports changed target, each once, parents before children, with what of
    /// it those tracks change (`track_parts`); and whether every one of those tracks draws through an item of
    /// this canvas.
    fn changed_items(&self, samples: &crate::animation::Samples) -> (Vec<(usize, u8)>, bool) {
        let tracks = samples.changed();
        let mut marks = self.item_marks.borrow_mut();
        let mut items: Vec<(usize, u8)> = Vec::with_capacity(tracks.len());
        let mut bound = true;
        // Items are parents first, so ascending is parents first; tracks authored in node order name them so.
        let mut ascending = true;
        for &track in tracks {
            let i = track as usize;
            // A control track draws nothing itself: the tracks it drives changed weight, and are named themselves.
            if self.kinds.get(i).copied().flatten().is_none() { continue; }
            let Some(item) = self.track_items.get(i).copied().flatten() else { bound = false; continue };
            let what = self.track_parts[i];
            if marks[item] == 0 {
                ascending &= items.last().is_none_or(|&(last, _)| last < item);
                items.push((item, what));
                marks[item] = items.len() as u32;
            } else {
                items[marks[item] as usize - 1].1 |= what;
            }
        }
        for (item, _) in &items { marks[*item] = 0; }
        if !ascending { items.sort_unstable_by_key(|&(item, _)| item); }
        (items, bound)
    }

    /// Bring `state` to this sample by evaluating `items` alone, parents before children: each one's node and values
    /// are written in place, its text batches replace the ones the state held when they differ, and no other item
    /// is visited. An item reads whether its
    /// parent draws from the state, so a changed parent is brought to this sample before its changed child. Returns whether the state still draws the text
    /// batches it drew; `None` when an item's scale reached or left zero, which changes what draws below it: the
    /// caller evaluates every item.
    fn evaluate_changed(&self, samples: &crate::animation::Samples, state: &mut FrameState, items: &[(usize, u8)],
        styled: &mut [Option<StyledItem>], specs: &mut [Option<TextSpecs>]) -> Result<Option<bool>> {
        let zero = |t: &v2::Transform| t.scale[0] == 0. || t.scale[1] == 0.;
        let mut draws: Vec<Draw> = Vec::new();
        let mut same_draws = true;
        for &(i, what) in items {
            draws.clear();
            let parent_drawing = self.rest[i].parent.is_none_or(|p| state.drawing[p]);
            let was_zero = zero(&state.values[i].transform);
            #[cfg(debug_assertions)]
            let (old_node, old) = (state.nodes[i].clone(), state.values[i].clone());
            // The item's values are written where the state holds them.
            let drawing = self.evaluate_item(samples, i, state.shown[i], parent_drawing, styled, specs,
                &mut state.nodes[i], &mut state.values[i], &mut draws)?;
            if zero(&state.values[i].transform) != was_zero { return Ok(None); }
            // The tracks' account names what of the item changed: the scene writes the named parts alone.
            #[cfg(debug_assertions)]
            {
                use crate::canvas_draw::change;
                let values = &state.values[i];
                let same_node = match (&old_node, &state.nodes[i]) {
                    (None, None) => true,
                    (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                    _ => false,
                };
                let rest = !same_node || old.style_revision != values.style_revision || old.visible != values.visible || old.path != values.path
                    || old.texts != values.texts || old.text_revision != values.text_revision || old.text_composition != values.text_composition;
                debug_assert!(drawing == state.drawing[i] && (old.transform == values.transform || what & change::TRANSFORM != 0)
                    && (old.opacity == values.opacity || what & change::OPACITY != 0) && (!rest || what & change::REST != 0),
                    "item {:?} changed in a way its changed tracks do not name (named {what:#b})", self.items[i].name);
            }
            let _ = (what, drawing);
            // Only a text node has batches.
            let same = !self.rest[i].text || (state.draws[i].len() == draws.len() && state.draws[i].iter().zip(&draws)
                .all(|(a, b)| Arc::ptr_eq(&a.spec, &b.spec) && Arc::ptr_eq(&a.layout, &b.layout)));
            if !same {
                // The item owns its batches: replacing them moves no other item's.
                std::mem::swap(&mut state.draws[i], &mut draws);
                same_draws = false;
            }
        }
        Ok(Some(same_draws))
    }

    /// The test of a state evaluated in place: it equals a state evaluated whole at this sample. Nothing of the
    /// state is taken on trust: which nodes draw, the background and the clocks' presence are made again from the
    /// samples, and every item is evaluated from those. The style and text caches are shared with the evaluation
    /// under test, and cannot hide a stale item: they are keyed by the item's sampled fields, so an item whose
    /// samples differ from the ones its kept node or batches were made from is made again, as a new node or new
    /// batches, and differs here.
    #[cfg(debug_assertions)]
    fn assert_whole(&self, samples: &crate::animation::Samples, state: &FrameState, present: &[bool],
        styled: &mut [Option<StyledItem>], specs: &mut [Option<TextSpecs>]) -> Result<()> {
        debug_assert!(!self.plan.has_clocks() || (present.len() == samples.clocks.len()
            && present.iter().zip(&samples.clocks).all(|(a, b)| *a == b.is_some())), "a node arrived or left on its clock");
        debug_assert!(state.background == self.background(samples), "the background changed");
        let shown = self.shown(samples);
        debug_assert!(shown == state.shown, "which nodes draw changed");
        let mut all_drawing: Vec<bool> = Vec::with_capacity(self.items.len());
        let mut draws: Vec<Draw> = Vec::new();
        for i in 0..self.items.len() {
            draws.clear();
            let parent_drawing = self.rest[i].parent.is_none_or(|p| all_drawing[p]);
            let (mut node, mut values) = (None, self.unevaluated(i));
            // A world's picture is given by a plan and kept in the table; no evaluation makes or changes it.
            values.world = state.values[i].world.clone();
            let drawing = self.evaluate_item(samples, i, shown[i], parent_drawing, styled, specs, &mut node, &mut values, &mut draws)?;
            all_drawing.push(drawing);
            let same_node = match (&state.nodes[i], &node) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            };
            let same_draws = draws.len() == state.draws[i].len() && draws.iter().zip(&state.draws[i])
                .all(|(a, b)| a.item == b.item && a.sub == b.sub && Arc::ptr_eq(&a.spec, &b.spec) && Arc::ptr_eq(&a.layout, &b.layout));
            debug_assert!(same_node && same_draws && values == state.values[i] && drawing == state.drawing[i],
                "item {:?} differs from its evaluation at this sample: no changed track names it", self.items[i].name);
        }
        Ok(())
    }

    /// One item at this sample, written where the caller holds it: `node` (its node when style tracks restyle
    /// it), `values` (its animated values) and, appended to `draws`, its text batches. Returns whether it draws
    /// (its scale and every ancestor's is not zero). The item's own tracks blend from its rest values in track
    /// order. It reads no other item's values: `shown` (whether the node draws at all) and `parent_drawing` are
    /// the caller's, from the evaluated state. Every frame evaluates an item through this function: every item on
    /// the first frame and when which nodes draw changed, else the items the changed tracks name. Nothing is
    /// built for what an item does not have: an item that only moves reads its rest values and its tracks, and
    /// writes its transform and opacity.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_item(&self, samples: &crate::animation::Samples, i: usize, shown: bool, parent_drawing: bool,
        styled: &mut [Option<StyledItem>], specs: &mut [Option<TextSpecs>],
        node: &mut Option<Arc<crate::canvas_draw::Node>>, values: &mut crate::canvas_draw::Values, draws: &mut Vec<Draw>) -> Result<bool> {
        let item = &self.items[i];
        let at_rest = &self.rest[i];
        let mut transform = at_rest.transform.clone();
        let mut opacity = at_rest.opacity;
        let mut path = at_rest.path;
        // Made when a track writes one.
        let mut fields: Option<BTreeMap<String, Arc<crate::animation::style::StyleValue>>> = None;
        let mut units: Option<BTreeMap<(TextUnit, usize), UnitValue>> = None;
        for &track in self.tracks_of(i) {
            let Some((target, sample, weight, additive)) = self.plan.sampled_at(samples, track as usize) else { continue };
            if weight == 0. { continue; }
            match (target, sample) {
                (TrackTarget::Node { property, .. }, crate::animation::Sample::Style(value)) => {
                    let field=match property {v2::NodeProperty::Style(p)=>p.name(),v2::NodeProperty::Fill=>"fill",v2::NodeProperty::Stroke=>"stroke",_=>unreachable!()};
                    let kind=target.style_interpolation().unwrap();
                    let rest=fields.as_ref().and_then(|f|f.get(field)).cloned().map(Ok).unwrap_or_else(|| {
                        let mut value=self.authored(i)?.field(field).clone();
                        if field=="filter" && value.is_null() {value=serde_json::json!("none");}
                        crate::animation::style::StyleValue::parse(kind,&value).map(Arc::new)
                    })?;
                    fields.get_or_insert_with(BTreeMap::new).insert(field.into(),Arc::new(rest.mix(value,weight,kind)));
                }
                (TrackTarget::Text { unit, index, property, .. }, crate::animation::Sample::Style(value)) => {
                    let piece=units.get_or_insert_with(BTreeMap::new).entry((*unit,*index)).or_default();
                    if let crate::animation::style::StyleValue::Origin(origin)=value.as_ref() {
                        let rest=piece.origin.unwrap_or([(0.5,0.);2]);
                        piece.origin=Some(std::array::from_fn(|i|(rest[i].0+(origin[i].0-rest[i].0)*weight,rest[i].1+(origin[i].1-rest[i].1)*weight)));
                        continue;
                    }
                    let slot=match property {UnitProperty::Fill=>&mut piece.fill,UnitProperty::Stroke=>&mut piece.stroke,_=>unreachable!()};
                    slot.push(UnitPaint::Css(value.clone(),weight));
                }
                (TrackTarget::Node { property: property @ (v2::NodeProperty::Fill | v2::NodeProperty::Stroke), .. }, crate::animation::Sample::Numeric(value)) => {
                    let field=if *property==v2::NodeProperty::Fill {"fill"} else {"stroke"};
                    let rest=match fields.as_ref().and_then(|f|f.get(field)) {Some(v)=>v.json(),None=>self.authored(i)?.field(field).clone()};
                    let mut color=rest.as_str().and_then(|v|paint(field,v).ok().flatten()).unwrap_or([0.;4]);
                    crate::animation::blend(&mut color,value,weight,additive,crate::animation::BlendKind::Add);
                    fields.get_or_insert_with(BTreeMap::new).insert(field.into(),Arc::new(crate::animation::style::StyleValue::Paint(crate::animation::style::Paint::Color(color.map(|v|v.clamp(0.,1.))))));
                }
                (TrackTarget::Node { node, property }, crate::animation::Sample::Numeric(value)) => {
                    let t = &mut transform;
                    let add = crate::animation::BlendKind::Add;
                    match property {
                        v2::NodeProperty::Style(p) => {
                            use crate::animation::style::StyleValue;
                            let field=p.name();
                            let rest=fields.as_ref().and_then(|f|f.get(field)).cloned().map(Ok).unwrap_or_else(||StyleValue::parse(p.interpolation(),self.authored(i)?.field(field)).map(Arc::new))?;
                            let mut number=value[0];
                            let sample=if additive {
                                let base=rest.json().as_f64().with_context(||format!("node {node:?}: {field} cannot combine from a keyword"))? as f32;
                                number=base+number*weight;StyleValue::Json(serde_json::json!(number))
                            } else {rest.mix(&StyleValue::Json(serde_json::json!(number)),weight,p.interpolation())};
                            fields.get_or_insert_with(BTreeMap::new).insert(field.into(),Arc::new(sample));
                        }
                        v2::NodeProperty::Translation => crate::animation::blend(&mut t.translation, value, weight, additive, add),
                        v2::NodeProperty::Scale => crate::animation::blend(&mut t.scale, value, weight, additive, crate::animation::BlendKind::Multiply),
                        v2::NodeProperty::Rotation => crate::animation::blend(t.rotation_mut(), value, weight, additive, crate::animation::BlendKind::Rotation),
                        v2::NodeProperty::Opacity => crate::animation::blend(std::slice::from_mut(&mut opacity), value, weight, additive, add),
                        v2::NodeProperty::StrokeWidth if at_rest.text => {
                            let authored_width=self.authored(i)?.field("stroke-width").as_f64().unwrap();
                            let mut width=fields.as_ref().and_then(|f|f.get("stroke-width")).and_then(|s|s.json().as_f64()).unwrap_or(authored_width) as f32;
                            crate::animation::blend(std::slice::from_mut(&mut width),value,weight,additive,add);
                            fields.get_or_insert_with(BTreeMap::new).insert("stroke-width".into(),Arc::new(crate::animation::style::StyleValue::Json(serde_json::json!(width))));
                        }
                        property => {
                            let Some(path) = &mut path else { continue };
                            match property {
                                v2::NodeProperty::StrokeWidth => crate::animation::blend(std::slice::from_mut(&mut path.stroke_width), value, weight, additive, add),
                                v2::NodeProperty::StrokeDashoffset => crate::animation::blend(std::slice::from_mut(&mut path.stroke_dashoffset), value, weight, additive, add),
                                v2::NodeProperty::FillOpacity => crate::animation::blend(std::slice::from_mut(&mut path.fill_opacity), value, weight, additive, add),
                                v2::NodeProperty::StrokeOpacity => crate::animation::blend(std::slice::from_mut(&mut path.stroke_opacity), value, weight, additive, add),
                                _ => {}
                            }
                        }
                    }
                }
                (TrackTarget::Text { unit, index, property, .. }, crate::animation::Sample::Numeric(value)) => {
                    if !at_rest.text { continue; }
                    let piece = units.get_or_insert_with(BTreeMap::new).entry((*unit, *index)).or_default();
                    match property {
                        UnitProperty::Translation => crate::animation::blend(&mut piece.translation, value, weight, additive, crate::animation::BlendKind::Add),
                        UnitProperty::Scale => crate::animation::blend(&mut piece.scale, value, weight, additive, crate::animation::BlendKind::Multiply),
                        UnitProperty::Rotation => crate::animation::blend(&mut piece.rotation, value, weight, additive, crate::animation::BlendKind::Rotation),
                        UnitProperty::Opacity => crate::animation::blend(std::slice::from_mut(&mut piece.opacity), value, weight, additive, crate::animation::BlendKind::Add),
                        UnitProperty::Fill | UnitProperty::Stroke => {
                            let slot=if *property==UnitProperty::Fill {&mut piece.fill}else{&mut piece.stroke};
                            slot.push(UnitPaint::Numeric(value.as_slice().try_into().expect("validated RGBA"),weight,additive));
                        }
                        UnitProperty::TransformOrigin => {}
                    }
                }
                // Visibility is which nodes draw (`shown`).
                _ => {}
            }
        }
        // Only a text node has kept batches.
        let text_revision = |specs: &[Option<TextSpecs>]| if at_rest.text { specs[i].as_ref().map_or(0, |c| c.revision) } else { 0 };
        // A node that does not draw (hidden, or absent on its clock: §8 rule 1) has no picture.
        if !shown {
            if node.is_some() { *node = None; }
            values.style_revision = 0;
            values.transform = at_rest.transform.clone();
            values.visible = false;
            values.opacity = 0.;
            values.path = None;
            values.texts = 0;
            values.text_composition.clear();
            values.text_revision = text_revision(&*specs);
            return Ok(false);
        }
        // Style tracks restyle the node: it is parsed again when a content field differs from the one kept.
        let restyled = fields.is_some();
        if let Some(fields) = fields {
            if styled[i].as_ref().is_none_or(|cached|cached.fields!=fields) {
                let content_fields=|fields:&BTreeMap<String,Arc<crate::animation::style::StyleValue>>| fields.iter().filter(|(field,_)|!matches!(field.as_str(),"filter"|"clip-path"|"mask")).map(|(field,value)|(field.clone(),value.clone())).collect::<BTreeMap<_,_>>();
                let content_changed=styled[i].as_ref().map_or_else(||!content_fields(&fields).is_empty(),|old|content_fields(&old.fields)!=content_fields(&fields));
                let node=if content_changed {
                    // The node as authored, with each field a track wrote in it: written into the typed node, field by field.
                    let mut node=self.authored(i)?.node.clone();
                    for (field,value) in &fields {if field!="d" {node.set_field(field,&value.json())?;}}
                    Some(node)
                } else {None};
                // The kept text when only its paints differ: the fields other than fill and stroke are the ones it
                // was made from, and Moda's vector lane draws it (the degraded tier rasters its colours).
                let kept=styled[i].as_ref().filter(|_|node.is_some()).and_then(|old| {
                    let other=|fields:&BTreeMap<String,Arc<crate::animation::style::StyleValue>>| fields.iter().filter(|(field,_)|!matches!(field.as_str(),"fill"|"stroke"|"filter"|"clip-path"|"mask")).map(|(field,value)|(field.clone(),value.clone())).collect::<BTreeMap<_,_>>();
                    old.text.clone().filter(|text|!text.degraded() && other(&old.fields)==other(&fields))
                });
                let mut text=match (&node,kept) {
                    (Some(node),Some(kept))=>Some(Arc::new(repainted_text(node,&kept,&self.images,&self.base,&fields,&self.media.borrow())?)),
                    (Some(node),None)=>build_text(node,&self.images,&self.base,&self.font_lists,&self.fonts,self.keyed.get(&node.name),self.glyphs_keyed.contains(&node.name),&fields,&self.media.borrow())?.map(Arc::new),
                    (None,_)=>styled[i].as_ref().and_then(|old|old.text.clone()),
                };
                if let Some(text)=&mut text {Arc::make_mut(text).clip_source=item.text.as_ref().is_some_and(|t|t.clip_source);}
                let mut draw=styled[i].as_ref().filter(|_|!content_changed).map_or_else(||item.node.clone(),|old|(*old.draw).clone());
                if let Some(node)=node.as_ref().filter(|node|matches!(node.kind,NodeKind::Path{..})) {
                    let mut path=crate::canvas_draw::sampled_path_item(&self.images,&self.base,node,fields.get("d").map(|v|v.as_ref()),&self.media.borrow())?;
                    for (field,value) in &fields {if let crate::animation::style::StyleValue::Paint(p)=value.as_ref(){path.set_paint(&self.images,&self.base,field,p,&self.media.borrow())?;}}
                    draw.kind=crate::canvas_draw::Kind::Path(Arc::new(path));
                }
                let reference=|field:&str,rest:Option<usize>| fields.get(field).map_or(rest,|value|value.json().as_str().and_then(|name|self.by_name.get(name).copied()));
                draw.clip_path=reference("clip-path",item.node.clip_path);
                draw.mask=reference("mask",item.node.mask);
                if let Some(text)=&text {draw.kind=crate::canvas_draw::Kind::Text{size:text.size};}
                draw.filter=if let Some(crate::animation::style::StyleValue::Filter(filter))=fields.get("filter").map(|v|v.as_ref()) {filter.clone()} else {item.node.filter.clone()};
                let revision=styled[i].as_ref().map_or(1,|c|c.revision+1);
                let text_paints:Vec<_>=text.iter().flat_map(|t|t.fill.iter().chain(&t.stroke)).collect();
                crate::canvas_draw::install_images(&[&draw],&text_paints)?;
                let text_revision=styled[i].as_ref().map_or(0,|old|old.text_revision)+u64::from(content_changed && text.is_some());
                styled[i]=Some(StyledItem{fields,draw:Arc::new(draw),text,revision,text_revision});
            }
            // The restyled path's values, with the four keyed numbers the tracks blended.
            if let crate::canvas_draw::Kind::Path(p)=&styled[i].as_ref().unwrap().draw.kind {
                let old=path;
                path=Some(p.values);
                if let (Some(old),Some(p))=(old,&mut path) {
                    p.stroke_width=old.stroke_width;p.stroke_dashoffset=old.stroke_dashoffset;p.fill_opacity=old.fill_opacity;p.stroke_opacity=old.stroke_opacity;
                }
            }
        }
        let style = if restyled { styled[i].as_ref() } else { None };
        let q = Quat::from_array(transform.resolved_rotation()).normalize();
        ensure!(transform.translation[2] == 0. && transform.scale[2] == 1., "node {:?}: translation z and scale z are not built in a 2D canvas; key x and y", item.name);
        ensure!(q.x.abs() < 1e-5 && q.y.abs() < 1e-5, "node {:?}: a rotation off the canvas's Z axis is not built in a 2D canvas", item.name);
        let mut texts = 0;
        let drawing = transform.scale[0] != 0. && transform.scale[1] != 0. && parent_drawing;
        let text = if at_rest.text { style.and_then(|s| s.text.as_deref()).or(item.text.as_ref()) } else { None };
        if let Some(text) = text {
            if drawing {
                // The text node consumes its own samples, cached-spec hits included.
                let keyed = units.unwrap_or_default();
                for ((unit,index),_) in &keyed {
                    let count=text.units[unit].len();
                    ensure!(*index<count,"node {:?}: {} has {count} pieces at this time; index {index} is past the end",item.name,unit.segment());
                }
                let slot = &mut specs[i];
                let style_revision = style.map_or(0, |s| s.text_revision);
                if slot.as_ref().is_none_or(|c| c.units != keyed || c.style_revision != style_revision) {
                    let (text_specs, mut composition) = if text.degraded() && !text.degraded_paints.is_empty() {
                        let text_specs = text.degraded_paints.iter().map(|(_, html)| {
                            let mut spec = text.spec.clone(); spec.html = html.clone().into(); spec
                        }).collect();
                        let parts = text.degraded_paints.iter().enumerate().map(|(sub, (paint, _))|
                            crate::canvas_draw::TextPart::Paint { sub, paint: paint.json().unwrap(), size: text.size, bounds: [0., 0., text.size[0], text.size[1]] }).collect();
                        (text_specs, parts)
                    } else if text.degraded() { (vec![text.spec.clone()], Vec::new()) } else { self.batches(&item.name, text, &keyed)? };
                    fn coverage_bounds(parts: &mut [crate::canvas_draw::TextPart], specs: &[moda_text::ops::TextNodeSpec],
                        fonts: &Fonts, item: usize, layout: &Arc<Laid>) -> Result<()> {
                        for part in parts {
                            match part {
                                crate::canvas_draw::TextPart::Paint { sub, bounds, .. } => {
                                    let draw = Draw { item, sub: *sub, spec: Arc::new(specs[*sub].clone()),
                                        model: glam::Mat4::IDENTITY.to_cols_array(), layout: layout.clone() };
                                    if let Some(batch) = fonts.build(&draw, 0., [f32::MIN, f32::MIN, f32::MAX, f32::MAX], 1.)? {
                                        if let Some(b) = batch.local_bounds() { *bounds = b; }
                                    }
                                }
                                crate::canvas_draw::TextPart::Group { children, .. } => coverage_bounds(children, specs, fonts, item, layout)?,
                                _ => {}
                            }
                        }
                        Ok(())
                    }
                    coverage_bounds(&mut composition, &text_specs, &self.fonts, i, &text.layout)?;
                    let revision = slot.as_ref().map_or(1, |cached| cached.revision + 1);
                    let clip = if text.clip_source {
                        let mut geometry = text.clone();
                        geometry.fill = vec![PaintSource::Color([1.; 4]); text.code_points];
                        geometry.stroke = vec![PaintSource::None; text.code_points];
                        geometry.fill_keyed = false;
                        geometry.spec.fill_override = Some("white".into());
                        let clip_units = keyed.iter().map(|(key, value)| {
                            let mut value = value.clone();
                            value.opacity = 1.; value.fill.clear(); value.stroke.clear();
                            (*key, value)
                        }).collect();
                        Some(Arc::new(self.batches(&item.name, &geometry, &clip_units)?.0.remove(0)))
                    } else { None };
                    *slot = Some(TextSpecs { revision, style_revision, units: keyed, composition, clip, specs: text_specs.into_iter().map(Arc::new).collect() });
                }
                let cached = slot.as_ref().unwrap();
                for (sub, spec) in cached.specs.iter().enumerate() {
                    draws.push(Draw { item: i, sub, spec: spec.clone(), model: glam::Mat4::IDENTITY.to_cols_array(), layout: text.layout.clone() });
                    texts += 1;
                }
                if let Some(clip) = &cached.clip {
                    draws.push(Draw { item: i, sub: texts, spec: clip.clone(),
                        model: glam::Mat4::IDENTITY.to_cols_array(), layout: text.layout.clone() });
                }
            }
        }
        if node.is_some() || style.is_some() { *node = style.map(|s| s.draw.clone()); }
        values.style_revision = style.map_or(0, |s| s.revision);
        values.transform = transform;
        values.visible = true;
        values.opacity = opacity;
        values.path = path;
        values.texts = texts;
        values.text_revision = text_revision(&*specs);
        match if at_rest.text { specs[i].as_ref() } else { None } {
            Some(cached) => values.text_composition.clone_from(&cached.composition),
            None => values.text_composition.clear(),
        }
        Ok(drawing)
    }

    /// A clip-path or mask a style track names must not lead back to its own node.
    fn check_references(&self, nodes: &[Option<Arc<crate::canvas_draw::Node>>]) -> Result<()> {
        if !nodes.iter().enumerate().any(|(i,n)|n.as_ref().is_some_and(|n|n.clip_path!=self.items[i].node.clip_path || n.mask!=self.items[i].node.mask)) { return Ok(()); }
        let mut states=vec![0u8;self.items.len()];
        // Each installed child/reference edge is visited once. Scanning
        // the document for each node made this cycle check quadratic.
        let mut pending = Vec::new();
        for root in 0..self.items.len() {
            if states[root] == 2 { continue; }
            pending.push((root, false));
            while let Some((i, exit)) = pending.pop() {
                if exit { states[i] = 2; continue; }
                ensure!(states[i] != 1, "clip-path/mask dependency cycle at node {:?}", self.items[i].name);
                if states[i] == 2 { continue; }
                states[i] = 1;
                pending.push((i, true));
                pending.extend(self.children.of(i).iter().rev().map(|&child| (child, false)));
                let node = nodes[i].as_deref().unwrap_or(&self.items[i].node);
                pending.extend([node.mask, node.clip_path].into_iter().flatten().map(|target| (target, false)));
            }
        }
        Ok(())
    }

    /// One text node's Moda specs at an instant. Every code point's transform is its glyph's, then its word's,
    /// then its line's, each about its own origin; its opacity their product; its fill and stroke the innermost
    /// piece's that keys them, else its own. Text in one fill (its runs' colours) and one stroke on every code point (or none) is one
    /// spec; otherwise each stroke colour is a spec drawing only the outlines of its code points, and each fill
    /// colour a spec drawing only the fill of its code points, fills first.
    fn batches(&self, name: &str, text: &TextItem, units: &BTreeMap<(TextUnit, usize), UnitValue>) -> Result<(Vec<moda_text::ops::TextNodeSpec>, Vec<crate::canvas_draw::TextPart>)> {
        let n = text.code_points;
        let mut base = text.spec.clone();
        let paint_bounds=[0.,0.,text.size[0],text.size[1]];
        let apply_paints=|layers:&[UnitPaint],mut paint:PaintSource|->Result<PaintSource> {
            for layer in layers {
                paint=match layer {
                    UnitPaint::Css(value,weight)=>{
                        let crate::animation::style::StyleValue::Paint(value)=value.as_ref() else {unreachable!()};
                        crate::canvas_draw::blend_paint(&paint,&crate::canvas_draw::resolve_animated_paint(&self.images,&self.base,paint_bounds,value,&self.media.borrow())?,*weight,paint_bounds)?
                    },
                    UnitPaint::Numeric(value,weight,additive)=>{
                        let mut color=paint.colour().unwrap_or([0.;4]);
                        crate::animation::blend(&mut color,value,*weight,*additive,crate::animation::BlendKind::Add);
                        PaintSource::Color(color.map(|c|c.clamp(0.,1.)))
                    }
                };
            }
            Ok(paint)
        };
        let mut unit_paints=BTreeMap::new();
        for (&(unit,index),value) in units {
            let first=text.units[&unit].get(index).and_then(|cps|cps.first()).copied();
            let paint=|layers:&[UnitPaint],rest:&[PaintSource]|->Result<Option<PaintSource>> {
                if layers.is_empty(){Ok(None)}else{apply_paints(layers,first.map_or(PaintSource::None,|cp|rest[cp].clone())).map(Some)}
            };
            unit_paints.insert((unit,index),(paint(&value.fill,&text.fill)?,paint(&value.stroke,&text.stroke)?));
        }
        let mut fills = text.fill.clone();
        let mut strokes = text.stroke.clone();
        if text.pieces_keyed {
            let mut xforms = Vec::with_capacity(n);
            for cp in 0..n {
                let (mut m, mut b, mut opacity) = (DMat2::IDENTITY, DVec2::ZERO, 1f32);
                for level in LEVELS {
                    let Some(index) = text.piece_of[&level][cp] else { continue };
                    let Some(value) = units.get(&(level, index)) else { continue };
                    let l = linear(value).with_context(|| format!("node {name:?}: {} {index}", level.segment()))?;
                    let pivot = text.boxes.get(&level).and_then(|b| b[index]).map_or(text.centre, |b| {
                        let [(fx, px), (fy, py)] = value.origin.unwrap_or([(0.5, 0.), (0.5, 0.)]);
                        DVec2::new(b[0] + f64::from(fx) * (b[2] - b[0]) + f64::from(px), b[1] + f64::from(fy) * (b[3] - b[1]) + f64::from(py))
                    });
                    // This level's map, l·(p − P) + P + t, after the inner levels' m·p + b.
                    m = l * m;
                    b = l * (b - pivot) + pivot + DVec2::new(f64::from(value.translation[0]), f64::from(value.translation[1]));
                    opacity *= value.opacity.clamp(0., 1.);
                }
                for level in LEVELS.into_iter().rev() {
                    let Some((fill,stroke)) = text.piece_of[&level][cp].and_then(|index| unit_paints.get(&(level,index))) else {continue};
                    if let Some(fill)=fill {fills[cp]=fill.clone();}
                    if let Some(stroke)=stroke {strokes[cp]=stroke.clone();}
                }
                // Moda maps each glyph p to lin·(p − W) + W + off, W the centre of all the node's ink.
                let off = b + (m - DMat2::IDENTITY) * text.centre;
                xforms.push(moda_text::vector::ops::GlyphXform {
                    lin: [m.x_axis.x as f32, m.y_axis.x as f32, m.x_axis.y as f32, m.y_axis.y as f32],
                    off: [off.x as f32, off.y as f32],
                    opacity,
                });
            }
            base.unit_transforms = (xforms).into();
            if n > 0 { base.unit_groups = (vec![0; n]).into(); }
        }
        if text.stroke_width <= 0. { strokes.iter_mut().for_each(|s| *s = PaintSource::None); }
        crate::canvas_draw::install_images(&[],&fills.iter().chain(&strokes).collect::<Vec<_>>())?;
        if fills.iter().chain(&strokes).any(|p| matches!(p, PaintSource::Moda(_))) {
            return Ok(self.painted_batches(text, units, base, fills, strokes));
        }
        let fills: Vec<_> = fills.iter().map(PaintSource::colour).collect();
        let strokes: Vec<_> = strokes.iter().map(PaintSource::colour).collect();
        // Keep every previously supported frame on its original batch path. Only the formerly
        // refused fill/stroke overlap needs unit pictures, composed by Moda's opacity groups.
        let compose = base.unit_transforms.iter().enumerate().any(|(cp, xform)| {
            xform.opacity > 0. && xform.opacity < 1.
                && quantised(fills[cp]).is_some() && quantised(strokes[cp]).is_some()
        });
        let finish = |specs| {
            if compose { self.compose_units(text, units, specs, &[]) } else { (specs, Vec::new()) }
        };
        let mut stroke_colours: Vec<[u8; 4]> = Vec::new();
        for s in strokes.iter().filter_map(|s| quantised(*s)) {
            if !stroke_colours.contains(&s) { stroke_colours.push(s); }
        }
        let rgba = |q: [u8; 4]| q.map(|c| f32::from(c) / 255.);
        // One spec when every code point has the same stroke, or none.
        let one_stroke = strokes.iter().all(|s| quantised(*s) == stroke_colours.first().copied());
        if !text.fill_keyed && one_stroke {
            let mut spec = base;
            if let Some(&colour) = stroke_colours.first() {
                spec.stroke_color = Some(css(rgba(colour)));
                spec.stroke_width = Some(text.stroke_width);
            }
            return Ok(finish(vec![spec]));
        }
        if base.unit_transforms.is_empty() {
            base.unit_transforms = (vec![moda_text::vector::ops::GlyphXform::IDENTITY; n]).into();
        }
        let mut specs = Vec::new();
        if !text.fill_keyed {
            specs.push(base.clone());
        } else {
            let mut fill_colours: Vec<[u8; 4]> = Vec::new();
            for f in fills.iter().filter_map(|f| quantised(*f)) {
                if !fill_colours.contains(&f) { fill_colours.push(f); }
            }
            for colour in fill_colours {
                let mut spec = base.clone();
                spec.fill_override = Some(css(rgba(colour)));
                spec.unit_fill = ((0..n).map(|cp| if quantised(fills[cp]) == Some(colour) { 1. } else { 0. }).collect::<Vec<_>>()).into();
                specs.push(spec);
            }
        }
        for colour in stroke_colours {
            let mut spec = base.clone();
            spec.stroke_color = Some(css(rgba(colour)));
            spec.stroke_width = Some(text.stroke_width);
            spec.unit_fill = (vec![0.; n]).into();
            for (cp, xform) in std::sync::Arc::make_mut(&mut spec.unit_transforms).iter_mut().enumerate() {
                if quantised(strokes[cp]) != Some(colour) { xform.opacity = 0.; }
            }
            specs.push(spec);
        }
        Ok(finish(specs))
    }

    /// Paint each distinct run/glyph coverage with the same paint the path renderer consumes.
    fn painted_batches(&self, text: &TextItem, units: &BTreeMap<(TextUnit, usize), UnitValue>,
        mut base: moda_text::ops::TextNodeSpec, fills: Vec<PaintSource>, strokes: Vec<PaintSource>)
        -> (Vec<moda_text::ops::TextNodeSpec>, Vec<crate::canvas_draw::TextPart>) {
        let n = text.code_points;
        if base.unit_transforms.is_empty() {
            base.unit_transforms = Arc::new(vec![moda_text::vector::ops::GlyphXform::IDENTITY; n]);
        }
        let mut specs = Vec::new();
        let mut paints = Vec::new();
        for (stroke, sources) in [(false, fills), (true, strokes)] {
            let mut distinct = Vec::new();
            for source in &sources {
                if !matches!(source, PaintSource::None) && !distinct.contains(source) { distinct.push(source.clone()); }
            }
            for source in distinct {
                let mut spec = base.clone();
                spec.fill_override = Some("white".into());
                spec.unit_fill = Arc::new(vec![if stroke { 0. } else { 1. }; n]);
                spec.stroke_color = stroke.then(|| "white".into());
                spec.stroke_width = stroke.then_some(text.stroke_width);
                for (cp, xform) in Arc::make_mut(&mut spec.unit_transforms).iter_mut().enumerate() {
                    if sources[cp] != source { xform.opacity = 0.; }
                }
                paints.push(source.json());
                specs.push(spec);
            }
        }
        self.compose_units(text, units, specs, &paints)
    }

    /// Keep independently faded unit pictures separate; range batches share their full layout.
    /// A word/line contains its glyph pictures; its opacity applies after those children compose.
    /// Paint order inside each picture remains all fills followed by all strokes.
    fn compose_units(&self, text: &TextItem, units: &BTreeMap<(TextUnit, usize), UnitValue>,
        mut specs: Vec<moda_text::ops::TextNodeSpec>, paints: &[Option<moda_scene::value::Value>]) -> (Vec<moda_text::ops::TextNodeSpec>, Vec<crate::canvas_draw::TextPart>) {
        use crate::canvas_draw::TextPart;
        // Fractional factors live on the composition groups; preserve zero selection masks.
        for spec in &mut specs {
            for xform in Arc::make_mut(&mut spec.unit_transforms).iter_mut() {
                if xform.opacity > 0. { xform.opacity = 1.; }
            }
        }
        let paths: Vec<Vec<_>> = (0..text.code_points).map(|cp| LEVELS.into_iter().rev().filter_map(|level| {
            let index = text.piece_of[&level][cp]?;
            let opacity = units.get(&(level, index))?.opacity.clamp(0., 1.);
            (opacity > 0. && opacity < 1.).then_some((level, index, opacity))
        }).collect()).collect();
        fn emit(paths: &[Vec<(TextUnit, usize, f32)>], range: std::ops::Range<usize>, depth: usize,
            specs: &[moda_text::ops::TextNodeSpec], paints: &[Option<moda_scene::value::Value>], size: [f64; 2], out: &mut Vec<moda_text::ops::TextNodeSpec>) -> Vec<TextPart> {
            let mut parts = Vec::new();
            let mut start = range.start;
            while start < range.end {
                let key = paths[start].get(depth);
                let end = (start + 1..range.end).find(|&cp| paths[cp].get(depth) != key).unwrap_or(range.end);
                if let Some(&(_, _, opacity)) = key {
                    parts.push(TextPart::Group { opacity, children: emit(paths, start..end, depth + 1, specs, paints, size, out) });
                } else {
                    for (index, spec) in specs.iter().enumerate() {
                        let mut spec = spec.clone();
                        spec.unit_range = Some(start..end);
                        if spec.unit_transforms[start..end].iter().any(|x| x.opacity > 0.) {
                            parts.push(match paints.get(index).and_then(Option::as_ref) {
                                Some(paint) => TextPart::Paint { sub: out.len(), paint: paint.clone(), size, bounds: [0., 0., size[0], size[1]] },
                                None => TextPart::Batch(out.len()),
                            });
                            out.push(spec);
                        }
                    }
                }
                start = end;
            }
            parts
        }
        let mut out = Vec::new();
        let composition = emit(&paths, 0..paths.len(), 0, &specs, paints, text.size, &mut out);
        (out, composition)
    }

    /// Bring the installed scene to the evaluated table and plan it through Moda's retained geometry caches. Text
    /// batches keep their geometry until their content or AA/raster scale changes; Moda places them anew.
    /// `worlds` are the pictures of the canvas's worlds at the frame the table stands at.
    fn plan(&self, worlds: &[Option<(String, [u32; 2])>]) -> Result<crate::canvas_draw::Plan> {
        let mut evaluated = self.evaluated.borrow_mut();
        let evaluated = evaluated.as_mut().context("internal: a canvas planned before a frame was evaluated")?;
        self.bring(&mut evaluated.state, worlds)
    }

    /// Give the scene its worlds' pictures, bring it whole to `state` if it is owed that, and plan it.
    fn bring(&self, state: &mut FrameState, worlds: &[Option<(String, [u32; 2])>]) -> Result<crate::canvas_draw::Plan> {
        let _profile = moda_engine::prof::span("canvas.plan");
        let update_profile = moda_engine::prof::span("canvas.plan.update");
        let mut page=self.page.clone();page.background=state.background;
        // Until the plan is made the scene is owed everything, and the text builder is not kept: a failure below
        // leaves it so.
        let stale = self.stale.replace(true);
        let text = self.text.borrow_mut().take();
        let mut retained = self.retained.borrow_mut();
        // A world's picture is a new one each frame, so its item's row is written here. The table keeps the
        // picture: a later frame that moves the world's item writes its row from the table, and lowering the
        // canvas again reads it there.
        let mut pictures: Vec<(usize, u8)> = Vec::new();
        for (i, world) in self.worlds.iter().enumerate() {
            let picture = worlds.get(i).cloned().flatten();
            if state.values[world.item].world != picture {
                state.values[world.item].world = picture;
                pictures.push((world.item, crate::canvas_draw::change::REST));
            }
        }
        let state = &*state;
        let node = |i: usize| state.nodes[i].as_deref().unwrap_or(&self.items[i].node);
        let written = match retained.as_mut() {
            Some(scene) if !stale => pictures.is_empty() || scene.write(&page, node, &state.values, &pictures)?,
            _ => false,
        };
        if !written {
            let nodes: Vec<&crate::canvas_draw::Node> = (0..self.items.len()).map(node).collect();
            match retained.as_mut() {
                Some(scene) => scene.update(&page, &nodes, &state.values)?,
                None => *retained = Some(crate::canvas_draw::Retained::new(&page, &nodes, &state.values)?),
            }
        }
        drop(update_profile);
        // A frame that changed no text batch draws the ones planned: their models come from the plan.
        let text = match text {
            Some(text) => text,
            None => {
                let fonts = self.fonts.clone();
                let draws: HashMap<_, _> = state.batches().map(|d| ((d.item, d.sub), d.clone())).collect();
                let text = std::rc::Rc::new(move |item: usize, sub: usize, model: [f32; 16], z: f64, clip: [f32; 4], scale: f64, opacity: f64| {
                    let draw = draws.get(&(item, sub)).context("a text batch the frame did not plan")?;
                    let mut placed = draw.clone();
                    placed.model = model;
                    Arc::make_mut(&mut placed.spec).opacity = opacity;
                    fonts.build(&placed, z, clip, scale)
                });
                text as crate::canvas_draw::CanvasTextBuilder
            }
        };
        let result = crate::canvas_draw::plan(retained.as_mut().unwrap(), &page, text.clone())?;
        *self.text.borrow_mut() = Some(text);
        self.stale.set(false);
        Ok(result)
    }

    /// A test's copy of the evaluated table at `request`.
    #[cfg(test)]
    fn snapshot(&self, request: &vork_time::Request) -> Result<Snapshot> {
        self.frame(request)?;
        let serial = self.snapshots.get() + 1;
        self.snapshots.set(serial);
        Ok(Snapshot { state: self.evaluated.borrow().as_ref().expect("a frame was evaluated").state.clone(), document: self.id, serial, edited: false })
    }

    /// Plan a test's snapshot. The document's own table when the snapshot is this document's latest and unedited; else the
    /// scene is brought whole to the snapshot's state, and is then owed the whole table again.
    #[cfg(test)]
    fn plan_snapshot(&self, snapshot: &Snapshot, worlds: &[Option<(String, [u32; 2])>]) -> Result<crate::canvas_draw::Plan> {
        if !snapshot.edited && snapshot.document == self.id && snapshot.serial == self.snapshots.get() { return self.plan(worlds); }
        let mut state = snapshot.state.clone();
        self.stale.set(true);
        self.text.borrow_mut().take();
        let plan = self.bring(&mut state, worlds);
        self.stale.set(true);
        self.text.borrow_mut().take();
        plan
    }
}

impl Fonts {
    /// The accepted layout of `spec`'s text, made once for its content (`LaidKey`, with `layout_html` the text
    /// without its colours) on these fonts: Moda's vector
    /// layout, or, for text its vector lane declines, its degraded tier's lines and box.
    fn lay_out(&self, spec: &moda_text::ops::TextNodeSpec, layout_html: Arc<str>, here: &str, drawn: &dyn Fn() -> Result<String>) -> Result<Arc<Laid>> {
        let key: LaidKey = (layout_html, spec.width.map(f64::to_bits), spec.auto_width);
        if let Some(laid) = self.layouts.borrow().by_content.get(&key).and_then(std::sync::Weak::upgrade) { return Ok(laid); }
        let _profile = moda_engine::prof::span("canvas.text.layout");
        // The port has no scene-owned emoji image provider. Classify the same sanitized
        // paragraphs as drawing: removing controls can bring a base and selector together.
        let paragraphs = moda_text::parser::parse_rich_text_to_paragraphs(&spec.html);
        ensure!(paragraphs.iter().all(|paragraph| {
            let text: String = paragraph.runs.iter().map(|run| run.text.as_str()).collect();
            moda_text::emoji::split_text_into_segments(&text).iter()
                .all(|segment| !matches!(segment, moda_text::emoji::Segment::Emoji(_)))
        }), "{here}: emoji presentation is not built yet");
        // Without a width the box is the widest line, so text-align places lines within it.
        let fit = spec.width.is_none();
        let outlines = moda_text::vector::layout_counts()[1];
        let content = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hasher);
            hasher.finish()
        };
        let uncoloured = moda_text::ops::TextNodeSpec { html: key.0.clone(), ..spec.clone() };
        let laid = match moda_text::vector::accept_vector_layout(&uncoloured, &self.store, &self.host.catalog, moda_text::vector::VectorPlacement::VectorParity, fit) {
            Ok(vector) => Laid {
                width: vector.box_width(), height: vector.height(),
                lines: vector.lines().iter().map(|line| line.source.iter().flat_map(|source| source.values()).flatten().collect()).collect(),
                vector: Some(vector), content,
            },
            Err(moda_text::vector::VectorDecline { reason, cause, families, outlines: true }) =>
                bail!("{here}: Moda's vector text lane declined this text ({reason} ({cause}; families {families:?}))"),
            Err(_) => {
                let engine = self.degraded.get_or_init(|| self.host.degraded_engine());
                let mut spec = spec.clone();
                spec.html = drawn()?.into();
                let mut layout = engine.layout_json(&spec).map_err(|error| anyhow::anyhow!("{here}: {error}"))?;
                if fit {
                    spec.width = Some(vork_std::moda::widest_line(&layout));
                    layout = engine.layout_json(&spec).map_err(|error| anyhow::anyhow!("{here}: {error}"))?;
                }
                let lines = || layout["paragraphs"].as_array().into_iter().flatten().flat_map(|p| p["lines"].as_array().into_iter().flatten());
                Laid {
                    vector: None, content, width: spec.width.unwrap_or(0.),
                    height: layout["totalHeightPx"].as_f64().unwrap_or_else(|| lines()
                        .map(|line| line["lineTopY"].as_f64().unwrap_or(0.) + line["lineHeightPx"].as_f64().unwrap_or(0.)).fold(0., f64::max)),
                    lines: lines().map(|line| line["segments"].as_array().into_iter().flatten()
                        .flat_map(|segment| segment["sourceIndices"].as_array().into_iter().flatten())
                        .filter_map(|v| v.as_u64().map(|i| i as usize)).collect()).collect(),
                }
            }
        };
        if moda_engine::prof::on() {
            moda_engine::prof::add_work("canvas.text.outlines", 0, moda_text::vector::layout_counts()[1] - outlines);
        }
        let laid = Arc::new(laid);
        let mut layouts = self.layouts.borrow_mut();
        if layouts.by_content.len() >= layouts.sweep_at {
            layouts.by_content.retain(|_, laid| laid.strong_count() > 0);
            layouts.sweep_at = 2 * layouts.by_content.len() + 1;
        }
        layouts.by_content.insert(key, Arc::downgrade(&laid));
        Ok(laid)
    }

    /// Moda's batch for one draw of its layout: its vector lane's, or, for text its degraded tier draws, its raster.
    fn build(&self, draw: &Draw, z: f64, clip: [f32; 4], scale: f64) -> Result<Option<Arc<moda_text_draw::TextQuadBatch>>> {
        let outcome = match &draw.layout.vector {
            None => {
                let engine = self.degraded.get_or_init(|| self.host.degraded_engine());
                moda_text_draw::build_degraded_batch(engine, &draw.spec, draw.model, z, clip, scale)
                    .with_context(|| format!("node {:?}: Moda's degraded text lane", draw.spec.node_id))?
            }
            Some(layout) => match moda_text_draw::build_vector_batch(layout, &self.store, &self.host, &draw.spec, draw.model, z, clip, scale)? {
                Ok(outcome) => outcome,
                Err(reason) => bail!("node {:?}: Moda's vector text lane declined this text ({reason})", draw.spec.node_id),
            },
        };
        match outcome {
            moda_text_draw::TextBuildOutcome::Item(batch) => Ok(Some(batch)),
            moda_text_draw::TextBuildOutcome::Empty => Ok(None),
            moda_text_draw::TextBuildOutcome::Degraded { reason } => bail!("node {:?}: {reason}", draw.spec.node_id),
        }
    }
}

/// A piece's rotation and scale as a 2×2 map (rotation about the canvas's Z axis after a scale).
fn linear(value: &UnitValue) -> Result<DMat2> {
    let [_, _, tz] = value.translation;
    let [sx, sy, sz] = value.scale;
    let q = Quat::from_array(value.rotation).normalize();
    ensure!(tz == 0. && sz == 1., "translation z and scale z are not built in a 2D canvas; key x and y");
    ensure!(q.x.abs() < 1e-5 && q.y.abs() < 1e-5, "a rotation off the canvas's Z axis is not built in a 2D canvas");
    let angle = 2. * f64::from(q.z).atan2(f64::from(q.w));
    let (sin, cos) = angle.sin_cos();
    let (sx, sy) = (f64::from(sx), f64::from(sy));
    Ok(DMat2::from_cols(DVec2::new(sx * cos, sx * sin), DVec2::new(-sy * sin, sy * cos)))
}

/// The GPU side: Moda's engine, which draws each frame's plan into its own 4x MSAA target and resolve, and the
/// straight-alpha frame the outputs read: video encoders on the GPU, the exposure report reduced on the GPU
/// (`display_stats`), PNG files through a readback.
struct Target {
    engine: crate::canvas_draw::Engine,
    device: wgpu::Device,
    size: [u32; 2],
    /// The last frame Moda drew: associated (premultiplied) colour, Rgba8Unorm or (`Document::float`) Rgba16Float.
    drawn: RefCell<Option<wgpu::Texture>>,
    frame_format: wgpu::TextureFormat,
    /// `finish` pipelines by destination format and by how the destination is written.
    finish: std::cell::RefCell<HashMap<(wgpu::TextureFormat, Finish), wgpu::ComputePipeline>>,
    /// Creates `finish` pipelines through the persistent pipeline cache.
    compiler: crate::pipelines::Compiler,
    stride: u32,
    pictures: Pictures,
    /// The view stages this canvas's renders have asked for, and the one that finishes this render's frames
    /// (`set_display`).
    display: RefCell<Stages>,
    /// Whether the drawn picture is float (`Document::float`).
    float: bool,
    /// The stage that writes the canvas's picture as a reader takes it (`Pictures::reader_stage`), built when a
    /// render first writes that picture to EXR (`set_raw`).
    picture: RefCell<Option<crate::display_program::PictureStage>>,
}
/// A root canvas's view stages, each built when a render first encodes for its display and kept with the
/// canvas: a float canvas has one for any display, an 8-bit canvas for a display other than the one its
/// encoding is the code values of (sRGB's for an sRGB canvas, Display P3's for a Display P3 canvas). A render
/// to Display P3 builds no sRGB stage beside it, and a session that goes back and forth between displays
/// builds each once.
#[derive(Default)]
struct Stages {
    built: Vec<(crate::output_device::Target, crate::display_program::PictureStage)>,
    /// The stage for the display in use; `None` when the picture needs none.
    current: Option<usize>,
}
impl Stages {
    fn stage(&self) -> Option<&crate::display_program::PictureStage> {
        self.current.map(|index| &self.built[index].1)
    }
}
/// How `Target::finish` writes a destination.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Finish {
    /// An 8-bit picture: its associated bytes as straight alpha, with the rounding of an 8-bit divide.
    Unassociate,
    /// A frame the view stage already finished: its code values as they are.
    Copy,
}
/// Installed world pictures shared by standalone canvases and material pictures, and the canvas's image graph
/// and view stage.
pub(crate) struct Pictures {
    /// The canvas's image graph (`composition.compositor`), run over each drawn picture before any view.
    graph: Option<crate::compositor::CanvasGraph>,
    /// The view stage of a canvas a world reads (`floating`, `reader_stage`): it writes that world's float
    /// picture. A root canvas's stage belongs to its `Target`, which builds it for the display a render
    /// encodes for.
    view: Option<crate::display_program::PictureStage>,
    worlds: RefCell<Vec<crate::canvas_world::World>>,
    pool: moda_engine::pixel::pool::TexturePool,
    convert: crate::canvas_world::Convert,
}
impl Pictures {
    /// `floating`: the picture is read by a world, so an authored view delivers it floating point.
    pub(crate) fn new(renderer: &Renderer, document: &Document, base: &Path, floating: bool) -> Result<Self> {
        document.media.borrow_mut().prepare_gpu(renderer)?;
        let worlds = document.worlds.iter().map(|world| crate::canvas_world::World::install(renderer, world.scene.clone(), base)
            .with_context(|| format!("node {:?}", document.items[world.item].name))).collect::<Result<Vec<_>>>()?;
        let view = if floating { Self::reader_stage(renderer, document, base)? } else { None };
        let graph = document.view.as_ref().map(|scene| crate::compositor::CanvasGraph::new(renderer, scene, base)).transpose()?.flatten();
        Ok(Self { graph, view, worlds: RefCell::new(worlds), pool: moda_engine::pixel::pool::TexturePool::new(&renderer.device, moda_engine::pixel::pool::TexturePool::budget_from_env()),
            convert: crate::canvas_world::Convert::into_canvas(document.working()) })
    }
    /// The stage that writes the canvas's picture as a reader takes it: its view's code values for sRGB,
    /// floating point, neither clamped nor rounded. Every picture that leaves a canvas has passed its view; a
    /// world reads the canvas so, and an EXR stores it so, whatever encoding the canvas blends in. A canvas
    /// whose view is the identity for sRGB (the default view of an sRGB canvas) has no stage: its drawn
    /// picture is that picture.
    fn reader_stage(renderer: &Renderer, document: &Document, base: &Path) -> Result<Option<crate::display_program::PictureStage>> {
        let reader = crate::output_device::Target::default();
        document.view.as_ref().filter(|scene| !crate::display_program::Prepared::identity(scene, reader))
            .map(|scene| crate::display_program::PictureStage::new(renderer, scene, base, reader, true)).transpose()
    }
    /// The canvas's picture at `frame` before its view: what Moda drew, then the canvas's image graph over it.
    /// `errors`: the shader diagnostics of the frame the picture is drawn for, where the graph records a result
    /// that is not finite; without them the graph's own word (`CanvasGraph::errors`).
    fn draw(&self, renderer: &Renderer, document: &Document, frame: &Frame, engine: &crate::canvas_draw::Engine, errors: Option<&wgpu::Buffer>) -> Result<wgpu::Texture> {
        // Each present world is drawn by the 3D renderer at its local time, and its frame crosses into a pool
        // texture Moda's engine samples as an image: on the GPU, in queue order before Moda's drawing.
        let worlds_profile = (!frame.worlds.is_empty()).then(|| moda_engine::prof::span("canvas.worlds"));
        let mut captures = Vec::new();
        let mut pictures = vec![None; frame.worlds.len()];
        let mut stored = Vec::new();
        for ((world, request), picture) in self.worlds.borrow_mut().iter_mut().zip(&frame.worlds).zip(&mut pictures) {
            let Some(request) = request else { continue };
            let [w, h] = world.size;
            let output = world.draw(renderer, request).map_err(device)?;
            let texture = self.pool.acquire(w, h, moda_engine::pixel::image::Storage::F16).map_err(|error| device(anyhow::anyhow!("a world's picture: {error}")))?;
            let mut encoder = renderer.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("world picture") });
            self.convert.encode(renderer, &mut encoder, output, texture.view(), texture.texture().format(), crate::canvas_world::Crossing::ToCanvas);
            renderer.queue.submit([encoder.finish()]);
            let capture = moda_engine::external_texture::CaptureTexture::new(texture);
            stored.push((capture.key().to_string(), [w, h], capture.upload_id));
            *picture = Some((capture.key().to_string(), [w, h]));
            captures.push(capture);
        }
        if !frame.worlds.is_empty() { crate::canvas_draw::install_captures(&stored); }
        drop(worlds_profile);
        let plan = document.plan(&pictures).map_err(scene_error)?;
        let drawn = {
            let _profile = moda_engine::prof::span("canvas.draw");
            engine.render_to_texture(&plan).map_err(device)?
        };
        drop(captures);
        match &self.graph {
            Some(graph) => graph.draw(renderer, &drawn, &frame.request, &frame.images, errors),
            None => Ok(drawn),
        }
    }
}

/// One frame in flight: its finished pixels stay on the GPU until acceptance and encoder release.
struct Slot {
    frame: wgpu::Texture,
    stats: crate::display_stats::Stats,
    /// The slot's entry in the submission ring, as a 3D frame slot has: the encoders' surfaces, the frame they
    /// still hold, the PNG staging and the credit that bounds its consumers.
    ring: crate::ring::Entry,
    /// Whether this render reads pixels back into the entry's staging (PNG files).
    readback: bool,
    /// This frame's copies of the error words of the picture's steps: the image graph's, the frame's view
    /// stage's, then the EXR picture's view stage's (four bytes each; a step the frame has not leaves its word
    /// zero), read with the frame's other results.
    picture_errors: Option<wgpu::Buffer>,
    /// This render's read of the canvas's picture after its view, for EXR.
    raw: Option<RawRead>,
}
/// A slot's read of the canvas's picture after its view, as a reader takes it: associated sRGB code values,
/// RGBA16F or (an 8-bit sRGB canvas, whose drawn picture it is) RGBA8.
struct RawRead {
    /// Where the reader's floating stage writes the picture; without one the drawn picture is read.
    picture: Option<wgpu::Texture>,
    format: wgpu::TextureFormat,
    stride: u32,
    readback: wgpu::Buffer,
}
impl Target {
    fn new(renderer: &Renderer, size: [u32; 2], document: &Document, base: &Path) -> Result<Self> {
        let limit = renderer.device.limits().max_texture_dimension_2d;
        ensure!(size[0] <= limit && size[1] <= limit,
            crate::gpu_memory::device_limit("render resolution", format_args!("{}x{} pixels", size[0], size[1]), format_args!("max_texture_dimension_2d={limit} pixels per axis"), "reduce render.resolution"));
        let engine = crate::canvas_draw::Engine::new(renderer, document.float(), document.working()).map_err(device)?;
        let stride = (size[0] * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        Ok(Self { engine, device: renderer.device.clone(), size, drawn: RefCell::new(None), frame_format: renderer.output_format(), finish: Default::default(), compiler: renderer.compiler.clone(), stride,
            pictures: Pictures::new(renderer, document, base, false)?, display: Default::default(), float: document.float(), picture: Default::default() })
    }

    /// Prepare this render's EXR picture: the canvas's picture after its view, as a reader takes it
    /// (`Pictures::reader_stage`). Its floating stage is built once and kept with the canvas; a canvas with
    /// none (the default view of an sRGB canvas) has its drawn picture read as it is, 8-bit or float.
    fn set_raw(&self, renderer: &Renderer, document: &Document, base: &Path) -> Result<()> {
        let mut picture = self.picture.borrow_mut();
        if picture.is_none() { *picture = Pictures::reader_stage(renderer, document, base)?; }
        Ok(())
    }

    /// Encode this render's frames for `display`. An 8-bit canvas's picture is the code values of the display
    /// its encoding names (sRGB, Display P3): for that display its default view is the identity and it needs
    /// no stage. A float canvas is finished by its view stage for any display, and an 8-bit canvas for any
    /// other display than its own: the canvas's view runs in the display stage with that display as its
    /// target, over the drawn picture, as a world's view does. `scene` is the canvas's root scene.
    fn set_display(&self, renderer: &Renderer, document: &Document, scene: &Scene, base: &Path, display: crate::output_device::Target) -> Result<()> {
        let mut stages = self.display.borrow_mut();
        let scene = document.view.as_ref().unwrap_or(scene);
        let current = if !document.float() && crate::display_program::Prepared::identity(scene, display) {
            None
        } else if let Some(index) = stages.built.iter().position(|(built, _)| *built == display) {
            Some(index)
        } else {
            let stage = crate::display_program::PictureStage::new(renderer, scene, base, display, false)?;
            stages.built.push((display, stage));
            Some(stages.built.len() - 1)
        };
        stages.current = current;
        Ok(())
    }

    /// `readback`: this render reads the 8-bit frame back (PNG files). `raw`: it reads the picture after its
    /// view back (EXR files; `set_raw` came first).
    fn slot(&self, renderer: &Renderer, readback: bool, raw: bool, surfaces: Vec<Surface>) -> Slot {
        let size = wgpu::Extent3d { width: self.size[0], height: self.size[1], depth_or_array_layers: 1 };
        let viewed = self.picture.borrow().is_some();
        let raw = raw.then(|| {
            let format = if viewed || self.float { wgpu::TextureFormat::Rgba16Float } else { wgpu::TextureFormat::Rgba8Unorm };
            let texel = if format == wgpu::TextureFormat::Rgba16Float { 8 } else { 4 };
            let stride = (self.size[0] * texel).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            RawRead {
                picture: viewed.then(|| crate::gpu_memory::texture(&renderer.device, &wgpu::TextureDescriptor {
                    label: Some("canvas viewed picture"), size, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
                })),
                format, stride,
                readback: crate::gpu_memory::buffer(&renderer.device, &wgpu::BufferDescriptor {
                    label: Some("canvas picture readback"), size: u64::from(stride) * u64::from(self.size[1]),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
                }),
            }
        });
        let staging = if readback { u64::from(self.stride) * u64::from(self.size[1]) } else { 0 };
        let mut ring = crate::ring::Entry::staging(&renderer.device, staging);
        ring.encoder_surfaces = surfaces;
        Slot {
            frame: renderer.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("canvas frame"),
                size: wgpu::Extent3d { width: self.size[0], height: self.size[1], depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format: self.frame_format,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            }),
            stats: crate::display_stats::Stats::new(&renderer.device),
            ring,
            readback,
            picture_errors: (self.pictures.graph.is_some() || self.display.borrow().stage().is_some() || raw.as_ref().is_some_and(|raw| raw.picture.is_some()))
                .then(|| crate::gpu_memory::buffer(&renderer.device, &wgpu::BufferDescriptor {
                    label: Some("canvas picture errors"), size: 12,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
                })),
            raw,
        }
    }

    /// Draw one frame with Moda's engine (it submits its own work), then record the straight-alpha frame, its
    /// exposure reduction and, with a readback, the copy of its pixels.
    fn encode(&self, renderer: &Renderer, document: &Document, frame: &Frame, slot: &mut Slot) -> Result<wgpu::CommandEncoder> {
        *self.drawn.borrow_mut() = Some(self.pictures.draw(renderer, document, frame, &self.engine, None)?);
        let _profile = moda_engine::prof::span("canvas.output.convert_record");
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("canvas frame") });
        self.finish(renderer, &mut encoder, &slot.frame, &frame.request, None)?;
        // Each step's error word travels with this frame, before the next frame's graph and view clear them.
        let display = self.display.borrow();
        if let Some(errors) = &slot.picture_errors {
            if let Some(graph) = &self.pictures.graph { encoder.copy_buffer_to_buffer(graph.errors(), 0, errors, 0, 4); }
            if let Some(view) = display.stage() { encoder.copy_buffer_to_buffer(view.errors(), 0, errors, 4, 4); }
        }
        // The picture after its view, for EXR: the reader's floating stage writes it; without one it is the
        // drawn picture. It is read before the next draw reuses the drawn picture.
        if let Some(raw) = &slot.raw {
            let drawn = self.drawn.borrow();
            let drawn = drawn.as_ref().context("no frame drawn")?;
            let stage = self.picture.borrow();
            let picture = match (stage.as_ref(), &raw.picture) {
                (Some(stage), Some(picture)) => {
                    stage.encode(renderer, &mut encoder, drawn, picture, &frame.request, None)?;
                    if let Some(errors) = &slot.picture_errors { encoder.copy_buffer_to_buffer(stage.errors(), 0, errors, 8, 4); }
                    picture
                }
                _ => drawn,
            };
            ensure!(picture.format() == raw.format, "the canvas's picture changed its format after the render began");
            encoder.copy_texture_to_buffer(picture.as_image_copy(), wgpu::TexelCopyBufferInfo {
                buffer: &raw.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(raw.stride), rows_per_image: None },
            }, picture.size());
        }
        slot.stats.encode(renderer, &mut encoder, crate::display_stats::Display::Texture(&slot.frame), None);
        if slot.readback {
            encoder.copy_texture_to_buffer(slot.frame.as_image_copy(), wgpu::TexelCopyBufferInfo {
                buffer: &slot.ring.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.stride), rows_per_image: None },
            }, slot.frame.size());
        }
        Ok(encoder)
    }

    /// Write the resolved frame into `destination` (storage-bindable) as straight alpha: the blend leaves associated
    /// colour (src-alpha colour, one alpha), and every output is straight alpha. A native encoder surface takes
    /// storage writes, not copies, so it is written this way too. An 8-bit picture for the canvas's own display is
    /// unassociated with the rounding of an 8-bit divide, for every destination. A float picture, and any
    /// picture for another display (`set_display`), is finished once, into the frame
    /// (`frame` is `None`), at the frame's `request`, by its view stage, where every view's result is clamped
    /// and rounded; an encoder surface then takes that finished `frame`'s code values.
    fn finish(&self, renderer: &Renderer, encoder: &mut wgpu::CommandEncoder, destination: &wgpu::Texture, request: &vork_time::Request, frame: Option<&wgpu::Texture>) -> Result<()> {
        ensure!(destination.usage().contains(wgpu::TextureUsages::STORAGE_BINDING), "the video encoder's surface cannot be written on this device");
        let device = &self.device;
        let drawn = self.drawn.borrow();
        let drawn = drawn.as_ref().context("no frame drawn")?;
        let display = self.display.borrow();
        let (kind, source) = match (display.stage(), frame) {
            (None, _) => (Finish::Unassociate, drawn),
            (Some(view), None) => return view.encode(renderer, encoder, drawn, destination, request, None),
            (Some(_), Some(frame)) => (Finish::Copy, frame),
        };
        let source = source.create_view(&Default::default());
        let mut pipelines = self.finish.borrow_mut();
        let pipeline = pipelines.entry((destination.format(), kind)).or_insert_with(|| {
            let format = match destination.format() { wgpu::TextureFormat::Bgra8Unorm => "bgra8unorm", _ => "rgba8unorm" };
            let texel = match kind {
                Finish::Unassociate => "let c = vec4<u32>(round(textureLoad(source, id.xy, 0) * 255.0)); var rgb = c.rgb;\n\
                    if (c.a > 0u && c.a < 255u) { rgb = min((c.rgb * 255u + c.a / 2u) / c.a, vec3<u32>(255u)); }\n\
                    textureStore(destination, id.xy, vec4<f32>(vec3<f32>(rgb), f32(c.a)) / 255.0);",
                Finish::Copy => "textureStore(destination, id.xy, textureLoad(source, id.xy, 0));",
            };
            let module = crate::timing::shader_module(device, wgpu::ShaderModuleDescriptor {
                label: Some("canvas finish"),
                source: wgpu::ShaderSource::Wgsl(format!("@group(0) @binding(0) var source: texture_2d<f32>;\n\
                    @group(0) @binding(1) var destination: texture_storage_2d<{format}, write>;\n\
                    @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
                    let size = textureDimensions(source); if (id.x >= size.x || id.y >= size.y) {{ return; }}\n\
                    {texel} }}").into()),
            });
            self.compiler.compute_cached(&wgpu::ComputePipelineDescriptor {
                label: Some("canvas finish"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None,
            })
        });
        let destination_view = destination.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas finish"), layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&source) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&destination_view) }],
        });
        let timestamps = moda_engine::gpu_profile::pass_timestamps();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("canvas finish"), timestamp_writes: moda_engine::gpu_profile::compute_writes(&timestamps) });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.size[0].div_ceil(8), self.size[1].div_ceil(8), 1);
        Ok(())
    }

    /// A slot's frame once its submission is done: its exposure, with a readback its RGBA8 pixels, and with a
    /// raw read the picture after its view as associated float RGBA.
    fn collect(&self, renderer: &Renderer, slot: &mut Slot, submission: wgpu::SubmissionIndex) -> Result<(crate::Exposure, Option<Vec<u8>>, Option<Vec<[f32; 4]>>)> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let mapped = [slot.stats.pending().context("canvas exposure was not recorded")?].into_iter().chain(slot.readback.then_some(&slot.ring.readback)).chain(&slot.picture_errors).chain(slot.raw.as_ref().map(|raw| &raw.readback)).collect::<Vec<_>>();
        for buffer in &mapped {
            let sender = sender.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| { let _ = sender.send(result); });
        }
        renderer.wait(Some(submission))?;
        for _ in &mapped { receiver.recv().context("canvas readback").map_err(device)?.context("canvas readback").map_err(device)?; }
        let picture_errors = slot.picture_errors.as_ref().map(|errors| {
            let codes = { let mapped = errors.get_mapped_range(..); [0, 4, 8].map(|at| u32::from_le_bytes(mapped[at..at + 4].try_into().expect("four bytes"))) };
            errors.unmap();
            codes
        });
        // The frame's completion is proved before its results are read as the scene's: every pixel
        // counted, and nothing recorded by the device. Only then are the picture's error words the scene's.
        let exposure = slot.stats.read(u64::from(self.size[0]) * u64::from(self.size[1]))?.context("canvas exposure was not recorded")?;
        renderer.errors.check()?;
        // Every step's word is read before the frame's error is chosen. A view written for another display
        // is wrong at every frame, so it outranks a non-finite result, which is this frame's refusal, as a
        // world's is: a sequence of per-frame files skips that.
        let codes = picture_errors.into_iter().flatten().filter(|&code| code != 0).collect::<Vec<_>>();
        let wrong_display = codes.contains(&vork_slang::registry::VIEW_DESTINATION);
        let code = if wrong_display { vork_slang::registry::VIEW_DESTINATION } else { codes.first().copied().unwrap_or(0) };
        if let Err(error) = crate::display_program::refuse(code) {
            // The frame delivers nothing: let go of its pixels, so its slot can take another frame.
            if slot.readback { slot.ring.readback.unmap(); }
            if let Some(raw) = &slot.raw { raw.readback.unmap(); }
            return Err(scene_error(if wrong_display { error } else { anyhow::Error::new(crate::FrameRefused(error.to_string())) }));
        }
        let pixels = slot.readback.then_some(&slot.ring.readback).map(|readback| {
            let mut rgba = Vec::with_capacity((self.size[0] * self.size[1] * 4) as usize);
            {
                let mapped = readback.get_mapped_range(..);
                for row in mapped.chunks(self.stride as usize).take(self.size[1] as usize) {
                    rgba.extend_from_slice(&row[..(self.size[0] * 4) as usize]);
                }
            }
            readback.unmap();
            if slot.frame.format() == wgpu::TextureFormat::Bgra8Unorm {
                for px in rgba.chunks_exact_mut(4) { px.swap(0, 2); }
            }
            rgba
        });
        let raw = slot.raw.as_ref().map(|raw| {
            let mut rgba = Vec::with_capacity((self.size[0] * self.size[1]) as usize);
            {
                let mapped = raw.readback.get_mapped_range(..);
                for row in mapped.chunks(raw.stride as usize).take(self.size[1] as usize) {
                    if raw.format == wgpu::TextureFormat::Rgba16Float {
                        rgba.extend(row[..(self.size[0] * 8) as usize].chunks_exact(8).map(|texel|
                            std::array::from_fn(|c| half::f16::from_bits(u16::from_le_bytes([texel[2 * c], texel[2 * c + 1]])).to_f32())));
                    } else {
                        rgba.extend(row[..(self.size[0] * 4) as usize].chunks_exact(4).map(|texel| std::array::from_fn(|c| f32::from(texel[c]) / 255.)));
                    }
                }
            }
            raw.readback.unmap();
            rgba
        });
        Ok((exposure, pixels, raw))
    }
}

/// `vork check` on a canvas: fonts load, images decode, and the first frame lays out and plans with Moda's engine
/// (no GPU).
pub(crate) fn check(scene: &Scene, base: &Path, renderer: &Renderer) -> Result<()> {
    // Everything here reads, lays out and installs the document: its refusals are the scene's,
    // except what the device hands back, which keeps its own category.
    check_document(scene, base, renderer).map_err(scene_error)
}
fn check_document(scene: &Scene, base: &Path, renderer: &Renderer) -> Result<()> {
    let document = Document::new(scene, base)?;
    document.install_images()?;
    document.media.borrow_mut().prepare_gpu(renderer)?;
    document.frame(&vork_time::Request::Numeric(0.))?;
    document.plan(&[])?;
    // A float canvas's image graph and view stage: their programs, inputs and images, as a render prepares them.
    if let Some(scene) = &document.view {
        crate::compositor::CanvasGraph::new(renderer, scene, base)?;
        crate::display_program::PictureStage::new(renderer, scene, base, Default::default(), false)?;
    }
    // Each world installs as its own 3D scene.
    for world in &document.worlds {
        renderer.install_with_options(world.scene.clone(), base, crate::InstallOptions { readback: Default::default(), ..Default::default() })
            .with_context(|| format!("node {:?}", document.items[world.item].name))?;
    }
    Ok(())
}

/// Draw one frame of `scene` as a render to `display` does, and keep nothing of it: the document, Moda's
/// drawing, the frame's finish and view stage, its exposure reduction, and the picture after its view that an
/// EXR destination reads (`set_raw`: an authored view's floating stage, a different shader from the frame's).
/// `vork setup` runs it over its canvas catalogue, so the driver pipelines a canvas render creates at its
/// first draw, to PNG, video or EXR, are already in the pipeline cache. It is the render path itself
/// (`Target::encode`), not a list of pipelines kept beside it. Nothing is read back: a pipeline exists once its
/// pass is encoded, and setup has no use for the frame's statistics or pixels, so only what the device itself
/// reports can fail it.
///
/// Release shader preparation has no device that executes anything. What it keeps of a canvas is what was
/// compiled before any drawing, the view's program and its stages for this display and for the EXR picture,
/// so it stops there.
pub(crate) fn warm(scene: &Scene, base: &Path, renderer: &Renderer, display: crate::output_device::Target) -> Result<()> {
    let document = Document::new(scene, base)?;
    let target = Target::new(renderer, scene.render.resolution, &document, base)?;
    target.set_display(renderer, &document, scene, base, display)?;
    target.set_raw(renderer, &document, base)?;
    if renderer.release_preparation { return Ok(()); }
    document.install_images()?;
    let mut slot = target.slot(renderer, true, true, Vec::new());
    let frame = document.frame(&vork_time::Request::Numeric(0.))?;
    let mut command = target.encode(renderer, &document, &frame, &mut slot)?;
    // A clip's native encoder takes the finished frame through `finish` as well, into its own surface: a
    // storage texture in the frame's format. Finishing into one here creates that pipeline (the copy of a
    // frame its view stage finished, or the same unassociate the frame took).
    let surface = crate::gpu_memory::texture(&renderer.device, &wgpu::TextureDescriptor {
        label: Some("setup encoder surface"), size: slot.frame.size(), mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format: slot.frame.format(),
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
    });
    target.finish(renderer, &mut command, &surface, &frame.request, Some(&slot.frame))?;
    let submission = renderer.queue.submit([command.finish()]);
    renderer.wait(Some(submission))?;
    renderer.errors.check()
}

// Test-only observation of the real scheduler, without timing thresholds or encoder substitutes.
#[cfg(test)]
std::thread_local! {
    static FRAME_EVENTS: RefCell<Option<Vec<(&'static str, usize)>>> = const { RefCell::new(None) };
}
#[cfg(test)]
fn frame_event(event: &'static str, frame: usize) {
    FRAME_EVENTS.with_borrow_mut(|events| {
        if let Some(events) = events { events.push((event, frame)); }
    });
}

/// What a session keeps of a canvas between renders: the document (its text laid out, its fonts loaded, its tracks
/// compiled) and the GPU side (Moda's engine with its pipelines and caches, the installed worlds). It belongs to the
/// renderer that made it and to one resolved scene, which `identity` names; the session proves the scene's files.
pub(crate) struct Kept {
    identity: blake3::Hash,
    document: Document,
    target: Target,
}
impl Kept {
    /// The resolved scene and the directory its files resolve against: everything the document and the target are
    /// built from. Outputs, frame ranges and destinations are each render's own. The scene's content is hashed
    /// as the typed scene is walked, once (`crate::identity`).
    fn identity(scene: &Scene, base: &Path) -> Result<blake3::Hash> {
        let mut hash = blake3::Hasher::new();
        hash.update(base.as_os_str().as_encoded_bytes());
        hash.update(&[0]);
        crate::identity::write(&mut hash, scene)?;
        Ok(hash.finalize())
    }
    /// Build what a render of `scene` builds before its first frame, with a render's errors, then evaluate and
    /// plan the first frame as `check` does: a session proves a candidate document this way before it publishes
    /// it. A canvas with worlds is planned by its first render instead, when the worlds' pictures exist; their
    /// scenes are installed here, in the target.
    pub(crate) fn prepare(renderer: &Renderer, scene: &Scene, base: &Path, display: crate::output_device::Target) -> Result<Self> {
        let identity = Self::identity(scene, base)?;
        let document = Document::new(scene, base).map_err(scene_error)?;
        let target = Target::new(renderer, scene.render.resolution, &document, base).map_err(scene_error)?;
        // The view stage for the display the session's renders encode for: an authored view's program, inputs
        // and images are proved here, and the render finds the stage built.
        target.set_display(renderer, &document, scene, base, display).map_err(scene_error)?;
        document.install_images().map_err(scene_error)?;
        document.frame(&vork_time::Request::Numeric(0.)).map_err(scene_error)?;
        if document.worlds.is_empty() {
            document.plan(&[]).map_err(scene_error)?;
        }
        Ok(Self { identity, document, target })
    }
}

/// What the shot entry hands over for a canvas.
pub(crate) struct Request {
    pub sound: Option<crate::audio::MovieSound>,
    pub scene: Scene,
    pub base: PathBuf,
    pub out: Vec<PathBuf>,
    pub outputs: Vec<Output>,
    pub at: Option<vork_time::Clock>,
    pub frame_start: Option<usize>,
    pub frame_end: Option<usize>,
    pub png_depth: crate::PngDepth,
    pub exr_codec: crate::ExrCodec,
    /// Each destination's EXR storage (`output.color_space`), by path.
    pub raw_colors: BTreeMap<PathBuf, crate::output_device::RawColor>,
    /// The display the render call's destinations share.
    pub display: crate::output_device::Target,
    pub no_write: bool,
    pub gpu_only: bool,
    pub diagnostics: bool,
    pub stop: Option<Arc<std::sync::atomic::AtomicBool>>,
    pub profile_record: serde_json::Value,
    pub device_reused: bool,
    /// A session's document and GPU side from an earlier render on the same renderer, reused when this render
    /// resolves to the same scene.
    pub kept: Option<Kept>,
}

/// Render a canvas to the destinations a 3D shot writes: a PNG still, a directory of frames, EXR frames, MP4 or MOV.
pub(crate) fn render(request: Request, device_worker: std::thread::ScopedJoinHandle<'_, Result<Renderer>>, warnings: &crate::warnings::Collector, wall: Instant) -> Result<RenderedShot> {
    let Request { scene, base, out, outputs, at: authored_at, frame_start, frame_end, png_depth, exr_codec, raw_colors, display, no_write, gpu_only, diagnostics, stop, profile_record, device_reused, sound, kept } = request;
    let parse_start = Instant::now();
    let [w, h] = scene.render.resolution;
    let fps = scene.render.fps;
    ensure_usage!(!out.is_empty(), "at least one output destination is required");
    ensure_usage!(outputs.iter().all(|o| *o == Output::Color), "a 2D canvas has one channel, color; depth, motion, object IDs and AOVs are 3D outputs");
    ensure_usage!(png_depth == crate::PngDepth::Eight, "--png-depth 16: a 2D canvas is 8-bit");
    let has_format = |path: &PathBuf, extension: &str| !shot::directory_marked(path) && path.extension().is_some_and(|s| s.eq_ignore_ascii_case(extension));
    let is_video = |path: &PathBuf| shot::video_destination(path);
    let is_png = |path: &PathBuf| has_format(path, "png");
    let is_exr = |path: &PathBuf| has_format(path, "exr");
    for path in &out {
        ensure_usage!(shot::recognized_destination(path),
            "--out {}: unrecognized extension; use .png (one still), .exr (with %06d), .mp4, .mov, or a directory (no extension, or end the path with /)", path.display());
        ensure_usage!(!is_exr(path) || path.file_name().is_some_and(|name| name.to_string_lossy().contains("%06d")), "EXR destination requires a %06d frame placeholder");
        ensure_usage!(!is_png(path) || !shot::has_frame_pattern(path), "--out {}: PNG frame patterns are unsupported; use a directory (every frame as NNNNNN.png)", path.display());
    }
    for (i, path) in out.iter().enumerate() {
        ensure_usage!(!out[..i].contains(path), "duplicate output path: {}", path.display());
    }
    let video_paths: Vec<PathBuf> = out.iter().filter(|p| is_video(p)).cloned().collect();
    let directory_paths: Vec<PathBuf> = out.iter().filter(|p| !is_video(p)).cloned().collect();
    let video = !video_paths.is_empty();
    if directory_paths.iter().any(|path| is_exr(path)) {
        i32::try_from(fps.numerator()).context("EXR rate numerator exceeds i32")?;
        u32::try_from(fps.denominator()).context("EXR rate denominator exceeds u32")?;
    }
    let clip_frames = shot::clip_frame_count(&scene.render)?;
    let at = authored_at.as_ref().map(|time| vork_time::Request::from(time).numeric()).transpose().map_err(|error| error.context(FailureKind::Usage))?;
    if let Some(t) = at {
        ensure_usage!(t.is_finite() && authored_at.as_ref().is_none_or(|t| t.exact() >= Default::default()), "--at must be nonnegative and finite");
    }
    let implicit_still = at.is_none() && frame_start.is_none() && frame_end.is_none() && clip_frames > 1
        && !directory_paths.is_empty() && directory_paths.iter().all(|p| is_png(p));
    let at = if implicit_still { Some(0.) } else { at };
    ensure_usage!(at.is_none() || (frame_start.is_none() && frame_end.is_none()), "--at selects one still; it cannot be combined with --frame-start or --frame-end");
    let first_frame = frame_start.unwrap_or(0);
    let last_frame = frame_end.unwrap_or(clip_frames.saturating_sub(1));
    ensure_usage!(first_frame <= last_frame && last_frame < clip_frames,
        "frame range {first_frame}..{last_frame} must lie within the clip's frames 0..{} (render.fps x render.duration) with start <= end", clip_frames.saturating_sub(1));
    ensure_usage!(at.is_none() || !video, "--at selects one still; a video renders every frame");
    let frames = if at.is_some() { 1 } else { last_frame + 1 - first_frame };
    let index = |frame: usize| first_frame + frame;
    let frame_time = |frame: usize| -> Result<f32> {
        match at { Some(t) => Ok(t), None => vork_time::Request::Frame(index(frame) as u32, fps).numeric() }
    };
    let still_index = at.map(|t| (f64::from(t) * fps.as_f64() + 0.5).floor() as u32);
    let output_index = |frame: usize| still_index.unwrap_or(index(frame) as u32);
    if video {
        ensure_usage!(!no_write && !gpu_only, "video output cannot be combined with --no-write or --gpu-only");
        ensure_usage!(w % 2 == 0 && h % 2 == 0, "video output requires even dimensions");
    }
    for path in &video_paths {
        ensure_usage!(!path.is_dir(), "video output path is a directory: {}", path.display());
        std::fs::create_dir_all(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))?;
    }
    for path in &directory_paths {
        ensure_usage!(!is_png(path) || frames == 1, "--out {}: a PNG file is one still; add --at SECONDS, or use a directory for every frame", path.display());
        if is_png(path) || is_exr(path) {
            std::fs::create_dir_all(path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")))?;
        } else {
            std::fs::create_dir_all(path)?;
        }
    }
    // A kept document and GPU side serve the same resolved scene; any other is dropped here and built again.
    // The scene's identity is what decides that. With nothing kept there is nothing to decide: the scene is
    // hashed by the worker that prepares the encoders, beside its document's construction, not before it.
    let known = match &kept {
        Some(_) => Some(Kept::identity(&scene, &base)?),
        None => None,
    };
    let kept = kept.filter(|kept| Some(kept.identity) == known);
    let install_reused = kept.is_some();
    let (kept_document, kept_target) = kept.map_or((None, None), |kept| (Some(kept.document), Some(kept.target)));
    // The video encoders' sessions open while the text is laid out.
    let (document, prepared) = std::thread::scope(|scope| {
        let prepared = scope.spawn(|| {
            let prepared = video_paths.iter().map(|path| {
                Encoder::prepare(Backend::Auto, w, h, fps, EncodeSettings { diagnostics, primaries: display.video_primaries(), ..EncodeSettings::default() }, path).map_err(device)
            }).collect::<Result<Vec<_>>>();
            let identity = match known {
                Some(identity) => Ok(identity),
                None => Kept::identity(&scene, &base),
            };
            (prepared, identity)
        });
        let document = match kept_document {
            Some(document) => Ok(document),
            None => Document::new(&scene, &base).map_err(scene_error),
        };
        (document, prepared.join().map_err(|panic| panicked("encoder preparation worker", panic)))
    });
    let document = document?;
    let scene_parse_ms = parse_start.elapsed().as_secs_f64() * 1000.;
    let (prepared, identity) = prepared?;
    let prepared = prepared?;
    let identity = identity?;
    let renderer = device_worker.join().map_err(|panic| panicked("device creation worker", panic))??;
    // Receipts describe this request: timings from earlier renders on a retained device are excluded.
    let pipeline_timings_before: std::collections::BTreeMap<String, f64> =
        if device_reused { renderer.pipeline_timings().into_iter().collect() } else { Default::default() };
    let gpu_profile = moda_engine::prof::gpu();
    if gpu_profile {
        let available = moda_engine::gpu_profile::enable(&renderer.device, &renderer.queue);
        eprintln!("canvas GPU timestamps available: {available}; query resolution serializes frames");
    }
    // Old outputs are removed only after every encoder can be written.
    for prepared in &prepared { prepared.validate_fallback(frames as u64, Some(&renderer.device)).map_err(device)?; }
    let output_run = if no_write || gpu_only { None } else {
        let selected = if let Some(index) = still_index { Some(index..=index) }
            else if frame_start.is_some() || frame_end.is_some() { Some(first_frame as u32..=last_frame as u32) }
            else { None };
        Some(crate::output_run::OutputRun::begin(&out, selected)?)
    };
    // Frames in flight: the CPU builds the next frames while the GPU draws and the encoders take the earlier ones.
    let in_flight = crate::constants::FRAMES_IN_FLIGHT.min(frames);
    let mut encoders = prepared.into_iter().map(|prepared| prepared.open(renderer.device.clone(), renderer.queue.clone(), frames as u64, in_flight).map_err(device)).collect::<Result<Vec<_>>>()?;
    // Each slot's ring entry on every encoder, installed in frame order so entry indices match the slots.
    let surfaces = (0..in_flight).map(|_| encoders.iter_mut().map(|encoder| encoder.create_surface().map_err(setup)).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
    let install_start = Instant::now();
    let target = match kept_target {
        Some(target) => target,
        None => Target::new(&renderer, [w, h], &document, &base).map_err(scene_error)?,
    };
    target.set_display(&renderer, &document, &scene, &base, display).map_err(scene_error)?;
    // An 8-bit canvas shows its authored display colours as they are, or (sRGB code values on a wider display)
    // re-expressed with nothing lost. Display P3 code values shown as sRGB are not: the view clips the colours
    // sRGB cannot show.
    let as_authored = !document.float() && (document.working() == v2::WorkingColorSpace::Srgb || document.working().display() == Some(display.display));
    // Images already in Moda's store are not read again.
    document.install_images().map_err(scene_error)?;
    // PNG files read the 8-bit frame back; EXR files read the picture after its view.
    let pngs = !no_write && !gpu_only && directory_paths.iter().any(|path| !is_exr(path));
    let raws = !no_write && !gpu_only && directory_paths.iter().any(|path| is_exr(path));
    if raws { target.set_raw(&renderer, &document, &base)?; }
    let mut slots: Vec<Slot> = surfaces.into_iter().map(|surfaces| target.slot(&renderer, pngs, raws, surfaces)).collect();
    let png_writers: Vec<_> = directory_paths.iter().filter(|path| pngs && !is_exr(path)).map(|path|
        crate::png_output::FrameOutputWriter::canvas(path.clone(), w, h, in_flight, output_index(0) as usize, display.display)).collect();
    let raw_writers: Vec<_> = directory_paths.iter().filter(|path| raws && is_exr(path)).map(|path|
        crate::png_output::FrameOutputWriter::canvas_exr(path.clone(), w, h, exr_codec, in_flight, output_index(0) as usize,
            (fps.numerator() as i32, fps.denominator() as u32), raw_colors[path])).collect();
    let install_ms = install_start.elapsed().as_secs_f64() * 1000.;
    eprintln!("render: canvas {w}x{h}; {frames} frames; {} text and {} path node(s), {} world(s)", document.items.iter().filter(|item| item.text.is_some()).count(),
        document.items.iter().filter(|item| matches!(item.node.kind, crate::canvas_draw::Kind::Path(_))).count(), document.worlds.len());
    vork_slang::progress::phase("Rendering frames", frames);
    let loop_start = Instant::now();
    let mut metadata_frames = Vec::new();
    let mut frame_exposures = Vec::new();
    let mut samples = Vec::new();
    let mut memory_frames = Vec::new();
    let mut rendered_frames = 0;
    let mut first_frame_ms = 0.;
    let mut interrupted = false;
    let mp4 = video_paths.iter().any(|path| has_format(path, "mp4"));
    let mut refused = shot::Refusals::default();
    let result = (|| -> Result<()> {
        let mut pending: std::collections::VecDeque<(usize, f32, Instant, wgpu::SubmissionIndex)> = std::collections::VecDeque::new();
        // How many submitted frames wait for acceptance. A ring entry is busy from its frame's submission until
        // every consumer lets go of it. PNG writers take a copy at acceptance, so the whole ring can be on the
        // GPU. A video encoder holds the entry after acceptance until the entry is reused, and reuse waits for
        // it: the ring is split, half for frames the GPU is drawing and the rest for frames the encoders are
        // draining, so neither wait lands on a frame that was only just queued.
        let window = if encoders.is_empty() { in_flight } else { in_flight.div_ceil(2) };
        let mut collect = |(frame, time, frame_start, submission): (usize, f32, Instant, wgpu::SubmissionIndex), slots: &mut [Slot], encoders: &mut [Encoder]| -> Result<()> {
            // Everything done to accept one frame; `canvas.output.wait_readback` and `canvas.output.deliver` are
            // inside it, so what they leave is this span's own.
            let _accept_profile = moda_engine::prof::span("canvas.accept");
            let slot = &mut slots[frame % in_flight];
            // A credit bounds raw pixels and compressed output until every destination finishes writing.
            let png_lease = (pngs || raws).then(|| std::sync::Arc::new(slot.ring.credit.acquire()));
            let collected = {
                let _profile = moda_engine::prof::span("canvas.output.wait_readback");
                target.collect(&renderer, slot, submission.clone())
            };
            let (mut e, rgba, raw) = match collected {
                Ok(collected) => collected,
                // A video cannot skip a frame; per-frame files can, so the rest of the range renders.
                Err(error) if encoders.is_empty() && refused.skip(output_index(frame), &error) => {
                    for writer in png_writers.iter().chain(&raw_writers) { writer.skip(output_index(frame) as usize)?; }
                    vork_slang::progress::completed(rendered_frames + refused.count());
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
            // A canvas shown as authored (`as_authored`) has no view that clips between its colours and the
            // display: a full channel is the colour asked for. A float canvas (a world, an authored view, an
            // image graph, linear light) and Display P3 code values shown as sRGB pass a view whose result can
            // run out of display range.
            if as_authored { e.clipped = None; }
            let _profile = moda_engine::prof::span("canvas.output.deliver");
            if let Some(path) = video_paths.iter().find(|path| mp4 && has_format(path, "mp4")) {
                ensure_usage!(e.coverage == 1.,
                    "--out {}: MP4 needs an opaque picture and this canvas leaves pixels transparent at {time} s; write .mov (ProRes 4444 keeps transparency) or PNG frames",
                    path.display());
            }
            if !encoders.is_empty() {
                let started = Instant::now();
                for (encoder, surface) in encoders.iter_mut().zip(&mut slot.ring.encoder_surfaces) {
                    let source = surface.texture.clone().unwrap_or_else(|| slot.frame.clone());
                    encoder.push(frame as u64, source, surface, &submission).map_err(device)?;
                }
                crate::timing::encoded(started);
                slot.ring.frame = Some(frame);
                #[cfg(test)]
                frame_event("push", frame);
            }
            if let Some(rgba) = rgba {
                let rgba = std::sync::Arc::new(rgba);
                let lease = png_lease.as_ref().expect("PNG readback has a writer credit");
                for writer in &png_writers {
                    writer.push_canvas(output_index(frame) as usize, rgba.clone(), lease.clone())?;
                }
            }
            if let Some(raw) = raw {
                let raw = std::sync::Arc::new(raw);
                let lease = png_lease.as_ref().expect("picture readback has a writer credit");
                for writer in &raw_writers {
                    writer.push_canvas_float(output_index(frame) as usize, raw.clone(), lease.clone())?;
                }
            }
            if diagnostics && (frame == 0 || frame + 1 == frames) {
                if let Some(memory) = renderer.device.generate_allocator_report() {
                    memory_frames.push(serde_json::json!({"frame": output_index(frame),
                        "allocated_bytes": memory.total_allocated_bytes,
                        "reserved_bytes": memory.total_reserved_bytes}));
                }
            }
            if rendered_frames == 0 { first_frame_ms = loop_start.elapsed().as_secs_f64() * 1000.; }
            metadata_frames.push(serde_json::json!({"index": output_index(frame), "time": time, "camera": null, "exposure": e.json()}));
            frame_exposures.push((index(frame), e));
            samples.push(serde_json::json!({"frame": index(frame), "time": time, "frame_wall_ms": frame_start.elapsed().as_secs_f64() * 1000.}));
            rendered_frames += 1;
            vork_slang::progress::completed(rendered_frames + refused.count());
            Ok(())
        };
        for frame in 0..frames {
            if stop.as_ref().is_some_and(|stop| stop.load(std::sync::atomic::Ordering::Relaxed)) && frame > 0 { interrupted = true; break; }
            // Frames stay in flight: the oldest is accepted only when the window is full, so the GPU draws it while
            // the next ones are built.
            if pending.len() == window { collect(pending.pop_front().expect("a frame in flight"), &mut slots, &mut encoders)?; }
            let frame_start = Instant::now();
            // Everything done to build and submit one frame: `canvas.evaluate`, `canvas.worlds`, `canvas.plan`,
            // `canvas.draw`, `canvas.output.convert_record` and `canvas.output.release_convert_submit` are inside
            // it, so a changing frame's time that none of them covers shows as this span's remainder.
            let frame_profile = moda_engine::prof::span("canvas.frame");
            let time = frame_time(frame).map_err(scene_error)?;
            let request = match at { Some(t) => vork_time::Request::Numeric(t), None => vork_time::Request::Frame(index(frame) as u32, fps) };
            let instant = document.frame(&request).map_err(scene_error)?;
            let slot = &mut slots[frame % in_flight];
            // Moda submits the draw independently of the output surfaces. Start it before
            // waiting for this ring entry's previous encoder payload to drain.
            #[cfg(test)]
            frame_event("draw", frame);
            let mut command = target.encode(&renderer, &document, &instant, slot)?;
            let output_profile = moda_engine::prof::span("canvas.output.release_convert_submit");
            if let Some(previous) = slot.ring.frame.take() {
                #[cfg(test)]
                frame_event("release", previous);
                for (encoder, surface) in encoders.iter_mut().zip(&mut slot.ring.encoder_surfaces) { encoder.release(previous as u64, surface).map_err(device)?; }
            }
            for surface in &slot.ring.encoder_surfaces {
                match &surface.texture {
                    Some(destination) => target.finish(&renderer, &mut command, destination, &instant.request, Some(&slot.frame)).map_err(device)?,
                    None => surface.record_copy(&mut command, &slot.frame),
                }
            }
            let submission = renderer.queue.submit([command.finish()]);
            drop(output_profile);
            drop(frame_profile);
            if gpu_profile {
                moda_engine::gpu_profile::resolve_into_prof(&renderer.device, &renderer.queue);
            }
            if moda_engine::prof::on() {
                eprintln!("canvas profile frame {frame} cumulative: {}", moda_engine::prof::report_json());
                eprintln!("canvas residency frame {frame} cumulative: {}", target.engine.profile_counters());
                eprintln!("canvas change kinds frame {frame} cumulative: {}", document.change_counters());
                if let Some(retained) = document.retained.borrow().as_ref() {
                    eprintln!("canvas plan cache frame {frame} cumulative: {}", retained.profile_counters());
                }
            }
            pending.push_back((frame, time, frame_start, submission));
        }
        while let Some(frame) = pending.pop_front() { collect(frame, &mut slots, &mut encoders)?; }
        Ok(())
    })();
    // Join all output workers even when a render or another destination failed.
    let png_finished: Vec<_> = png_writers.into_iter().chain(raw_writers).map(|writer| writer.finish()).collect();
    result?;
    png_finished.into_iter().collect::<Result<()>>()?;
    let finish_profile = moda_engine::prof::span("canvas.output.finish");
    let finish_start = Instant::now();
    // Finish every accepted stream before reporting an encoder error.
    let finished: Vec<Result<()>> = encoders.iter_mut().map(|encoder| encoder.finish().map(|_| ()).map_err(device)).collect();
    if !encoders.is_empty() { crate::timing::encoded(finish_start); }
    let encoder_finish_ms = finish_start.elapsed().as_secs_f64() * 1000.;
    finished.into_iter().collect::<Result<()>>()?;
    let frames = if interrupted { rendered_frames } else { frames };
    drop(finish_profile);
    renderer.errors.check().map_err(device)?;
    // Every frame is drawn, so the pipelines this canvas needed all exist (a later frame can be the
    // first to use one): save what it compiled, once (`Renderer::save_pipelines_soon`). A kept-alive
    // worker saves while it waits for the next request, and a one-shot's exit finishes the save
    // after delivery; optional scene specialization never delays completion.
    renderer.save_pipelines_soon();
    // The other frames are written; withholding meta.json leaves the output marked incomplete, and the
    // exit status reports the refusals.
    // An interrupted render's count is the frames it delivered; the refused ones were rendered too.
    refused.finish(if interrupted { frames + refused.count() } else { frames })?;
    if gpu_profile {
        eprintln!("canvas counted GPU work: {}", serde_json::to_string(&moda_engine::gpu_profile::take_counters())?);
    }
    if moda_engine::prof::on() {
        eprintln!("canvas host profile (inclusive process totals): {}", moda_engine::prof::report_json());
    }
    if let Some(sound) = &sound {
        sound.mux(&out, frames as u64)?;
    }
    let mut sidecars = Vec::new();
    let mut exposures = Vec::new();
    let mut pending_metadata = Vec::new();
    let render_settings = serde_json::json!({"profile": profile_record, "resolution": [w, h], "frames": frames, "canvas": true});
    for path in out.iter().filter(|_| output_run.is_some()) {
        let meta_path = if is_video(path) || is_png(path) || is_exr(path) { shot::sidecar(path, "meta.json") } else { path.join("meta.json") };
        // The exposure report and the display's name describe the frame encoded for the display: PNG and video.
        // An EXR holds the picture a reader takes, unclamped, and names its colour itself.
        let viewed = !is_exr(path);
        let frame_rows: Vec<serde_json::Value> = metadata_frames.iter().cloned().map(|mut frame| {
            if !viewed { frame["exposure"] = serde_json::Value::Null; }
            frame
        }).collect();
        let mut meta = serde_json::json!({
            "version": 3, "fps": fps, "frames": frame_rows, "object_ids": {},
            "warnings": warnings.snapshot(), "profile": profile_record, "render_settings": render_settings, "canvas": {"size": [w, h]},
        });
        if let Some(summary) = shot::exposure_summary(&frame_exposures).filter(|_| viewed) {
            meta["exposure"] = summary.clone();
            exposures.push(serde_json::json!({"output": path, "exposure": summary}));
        }
        // Every viewed picture's sidecar names the display it is encoded for, sRGB included, as its PNG's colour
        // chunk and its video stream do. An EXR is not encoded for a display and names its colour itself.
        if viewed {
            meta["output_display"] = serde_json::to_value(display.display)?;
            if !is_video(path) { meta["output_png_depth"] = 8.into(); }
        }
        if let Some(parent) = meta_path.parent() { std::fs::create_dir_all(parent)?; }
        pending_metadata.push((meta_path.clone(), serde_json::to_vec_pretty(&meta)?));
        sidecars.push(meta_path);
    }
    let pipeline_creation_ms: Vec<_> = renderer.pipeline_timings().into_iter().filter_map(|(name, ms)| {
        let delta = ms - pipeline_timings_before.get(&name).copied().unwrap_or(0.);
        (delta > 0.).then_some((name, delta))
    }).collect();
    let stages_end = Instant::now();
    let mut report = serde_json::json!({
        "version": 4, "device": renderer.adapter_info.name, "backend": format!("{:?}", renderer.adapter_info.backend),
        "pipeline_cache_supported": renderer.pipeline_cache_supported(), "pipeline_cache_status": renderer.pipeline_cache_status,
        "pipeline_cache_input_bytes": renderer.pipeline_cache_input_bytes, "pipeline_cache_rejection": renderer.pipeline_cache_rejection,
        "pipeline_creation_ms": pipeline_creation_ms,
        "frames": frames, "frame_start": at.is_none().then_some(first_frame), "still_time": at, "resolution": [w, h], "canvas": true,
        "png_write": pngs, "exr_write": raws, "video_write": video, "output": out,
        "encoder_finish_ms": encoder_finish_ms, "scene_parse_ms": scene_parse_ms, "install_ms": install_ms,
        "device_ms": if device_reused { 0. } else { renderer.startup.as_secs_f64() * 1000. },
        "first_frame_ms": first_frame_ms, "ready_ms": (loop_start - wall).as_secs_f64() * 1000.,
        "loop_wall_ms": (stages_end - loop_start).as_secs_f64() * 1000., "wall_ms": (stages_end - wall).as_secs_f64() * 1000., "samples": samples,
    });
    // Canvas painting records no per-pass GPU spans; its compile and encode stages are timed.
    let (stages, pipeline_cache_saves) = crate::timing::stages(&renderer.device, wall, stages_end, crate::timing::StageTotals {
        setup_ms: (loop_start - wall).as_secs_f64() * 1000., first_frame_ms, frames_ms: (stages_end - loop_start).as_secs_f64() * 1000.,
        gpu_spans_ms: Vec::new(), denoise_busy_ms: 0., denoise_wait_ms: 0.,
    });
    report["stages"] = stages;
    report["pipeline_cache_saves"] = serde_json::to_value(pipeline_cache_saves)?;
    report["pipeline_cache_save_pending"] = renderer.compiler.save_pending().into();
    if diagnostics {
        report["memory_frames"] = memory_frames.into();
        report["retained_gpu_bytes"] = renderer.residency.retained_bytes().into();
    }
    report["render_settings"] = render_settings;
    report["device_reused"] = device_reused.into();
    report["install_reused"] = install_reused.into();
    report["interrupted"] = interrupted.into();
    report["warnings"] = serde_json::to_value(warnings.snapshot())?;
    report["sidecars"] = serde_json::to_value(sidecars)?;
    report["exposure"] = exposures.into();
    // Every written render reports where its time went, beside its metadata.
    if output_run.is_some() {
        let mut timing_paths = Vec::new();
        for path in &out {
            let timing_path = if is_video(path) || is_png(path) || is_exr(path) { shot::sidecar(path, "timings.json") } else { path.join("timings.json") };
            std::fs::write(&timing_path, serde_json::to_vec_pretty(&report)?)?;
            timing_paths.push(timing_path);
        }
        if let Some(sidecars) = report["sidecars"].as_array_mut() {
            sidecars.extend(timing_paths.iter().map(|path| serde_json::json!(path)));
        }
    }
    if let Some(run) = &output_run { run.complete(&pending_metadata)?; }
    Ok(RenderedShot::canvas(report, encoders, renderer, Kept { identity, document, target }))
}

#[cfg(test)]
mod kept_identity_tests {
    use super::*;

    /// A kept canvas serves the scenes whose sorted JSON is the same text, under the same directory, and no other.
    #[test]
    fn a_kept_canvas_serves_exactly_the_scenes_with_the_same_content() -> Result<()> {
        let documents = [
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            // The same document with its keys in another order.
            r#"{"nodes":[
                {"fill":"red","d":"M0 0H20V20H0Z","type":"path","name":"a"},
                {"children":[{"fill":"blue","name":"b","d":"M30 0H50V20H30Z","type":"path"}],"name":"g"}],
                "render":{"resolution":[100,100]},"composition":{"space":"2d"},"version":2}"#,
            // A paint, a geometry, a name, a placement in the tree, an order, a render setting, a track.
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"lime"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V21H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"c","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"},
                {"name":"g","children":[]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]},
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,101]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}]}"#,
            r#"{"version":2,"composition":{"space":"2d"},"render":{"resolution":[100,100]},"nodes":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"g","children":[{"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"blue"}]}],
                "animation":[{"tracks":[{"path":["nodes","a","opacity"],"keys":[[0,1],[1,0]]}]}]}"#,
        ];
        let scenes = documents.iter().map(|document| Scene::from_json(document.as_bytes())).collect::<Result<Vec<_>>>()?;
        let base = Path::new("/a");
        let mut shared = 0;
        for (i, a) in scenes.iter().enumerate() {
            for (j, b) in scenes.iter().enumerate() {
                let same = serde_json::to_vec(a)? == serde_json::to_vec(b)?;
                assert_eq!(Kept::identity(a, base)? == Kept::identity(b, base)?, same, "documents {i} and {j}");
                shared += usize::from(same && i != j);
            }
        }
        // Only the reordered document shares, and every edit is its own scene.
        assert_eq!(shared, 2);
        // The directory the scene's files resolve against is part of what was built.
        assert_ne!(Kept::identity(&scenes[0], base)?, Kept::identity(&scenes[0], Path::new("/b"))?);
        Ok(())
    }
}

#[cfg(test)]
mod text_qa_tests {
    use super::*;

    fn document(content: &str, extra: serde_json::Value) -> Result<Document> {
        let mut node = serde_json::json!({"name":"title","type":"text","content":content,
            "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf","font-size":40});
        node.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},"nodes":[node]
        }))?)?;
        Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))
    }

    #[test]
    fn text_shadows_have_a_capable_owner() -> Result<()> {
        for (content, font) in [("مرحبا", "NotoSansArabic-Regular.ttf"), ("AB", "LiberationSans-Regular.ttf")] {
            let doc = document(content, serde_json::json!({
                "font": format!("../vork-std/tests/fonts/{font}"), "filter":"drop-shadow(8px 4px 0px red)"
            }))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            let nodes = doc.items.iter().map(|item| &item.node).collect::<Vec<_>>();
            let lowered = crate::canvas_draw::document(&doc.page, &nodes, &frame.values)?;
            // Vector and raster text alike: the group's captured picture casts the shadow, from the text's own alpha.
            let owner = lowered["nodes"].as_array().unwrap().iter().find(|n| n["effects"].as_array().is_some_and(|fx| !fx.is_empty())).unwrap();
            assert_eq!(owner["type"], "group", "{content}: {lowered}");
            assert_eq!(owner["effects"][0]["type"], "filter-shadow", "{content}: {lowered}");
        }
        Ok(())
    }

    #[test]
    fn text_units_preserve_unicode_spaces_and_blank_lines() -> Result<()> {
        for content in ["A\u{a0}\u{a0}B", "\u{2007}A\u{202f}", "A\n\nB"] {
            let doc = document(content, serde_json::json!({"white-space":"pre-line"}))?;
            let text = doc.items[0].text.as_ref().unwrap();
            assert_eq!(text.code_points, content.chars().count(), "{content:?}");
            assert_eq!(text.units[&TextUnit::Glyphs].len(), v2::unit_ranges(content, TextUnit::Glyphs).len(), "{content:?}");
        }
        Ok(())
    }

    #[test]
    fn preserved_unicode_glyph_targets_reach_the_draw() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"title","type":"text","content":"A\u{a0}\u{a0}B",
                "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf","font-size":40}],
            "animation":[{"tracks":[{"path":["nodes","title","glyphs",3,"translation"],"keys":[[0,[30,0,0]]]}]}]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let first = frame.batches().next().unwrap();
        assert_eq!(first.spec.unit_transforms[3].off[0], 30.);
        assert_eq!(first.spec.unit_transforms[2].off[0], 0.);
        Ok(())
    }

    #[test]
    fn sanitized_controls_do_not_lose_the_following_ink_box() -> Result<()> {
        let content = format!("A{}B", "\u{2060}".repeat(16));
        let doc = document(&content, serde_json::json!({}))?;
        let boxes = doc.items[0].text.as_ref().unwrap().layout.vector.as_ref().unwrap().code_point_boxes(content.chars().count());
        assert!(boxes[17].is_some(), "B keeps its authored character index after sanitization");
        Ok(())
    }

    #[test]
    fn variable_weight_ink_boxes_use_the_drawn_instance() -> Result<()> {
        let make = |extra: serde_json::Value| {
            let mut node = serde_json::json!({"font":"../moda-text/tests/fonts/NotoSans-WeightSubset.ttf"});
            node.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            document("Hamburgefonts", node)
        };
        let regular = make(serde_json::json!({}))?;
        let heavy = make(serde_json::json!({"font-weight":800}))?;
        let explicit = make(serde_json::json!({"font-variation-settings":"'wght' 800"}))?;
        let boxes = |doc: &Document| doc.items[0].text.as_ref().unwrap().layout.vector.as_ref().unwrap().code_point_boxes("Hamburgefonts".chars().count());
        assert_ne!(boxes(&regular), boxes(&heavy));
        assert_eq!(boxes(&heavy), boxes(&explicit));
        Ok(())
    }

    #[test]
    fn emoji_presentation_refuses_before_rendering() {
        for text in ["🙂", "A ☃ B", "©\u{fe0f}"] {
            let error = match document(text, serde_json::json!({})) {
                Ok(_) => panic!("emoji presentation must not silently disappear: {text:?}"),
                Err(error) => format!("{error:#}"),
            };
            assert_eq!(error, "node \"title\": emoji presentation is not built yet");
        }
        assert!(document("©\u{fe0e}", serde_json::json!({})).is_ok());
    }

    #[test]
    fn emoji_presentation_across_styled_runs_refuses() {
        for content in [
            serde_json::json!([{"content":"©","fill":"blue"},{"content":"\u{fe0f}","fill":"red"}]),
            serde_json::json!([{"content":"1","fill":"blue"},{"content":"\u{fe0f}","fill":"red"},{"content":"\u{20e3}","fill":"green"}]),
        ] {
            let error = match document("", serde_json::json!({"content":content})) {
                Ok(_) => panic!("emoji presentation across styles must refuse"),
                Err(error) => format!("{error:#}"),
            };
            assert_eq!(error, "node \"title\": emoji presentation is not built yet");
        }
        assert!(document("", serde_json::json!({"content":[
            {"content":"©","fill":"blue"},{"content":"\u{fe0e}","fill":"red"}
        ]})).is_ok());
        assert!(document("", serde_json::json!({"white-space":"pre-line","content":[
            {"content":"©\n","fill":"blue"},{"content":"\u{fe0f}","fill":"red"}
        ]})).is_ok());
    }

    #[test]
    fn emoji_presentation_created_by_sanitization_refuses() {
        for separator in ["\u{2060}", "\u{feff}", "\u{200c}"] {
            for content in [
                serde_json::json!(format!("A©{separator}\u{fe0f}B")),
                serde_json::json!([{"content":"A©","fill":"blue"},
                    {"content":format!("{separator}\u{fe0f}B"),"fill":"red"}]),
            ] {
                let error = match document("", serde_json::json!({"content":content})) {
                    Ok(_) => panic!("sanitization must not bypass the emoji refusal: {separator:?}"),
                    Err(error) => format!("{error:#}"),
                };
                assert_eq!(error, "node \"title\": emoji presentation is not built yet");
            }
        }
    }

    #[test]
    fn text_presentation_across_styled_runs_stays_text() {
        let catalog = moda_text::catalog::Catalog::from_catalog_json(&serde_json::json!({}));
        let paragraphs = moda_text::parser::parse_rich_text_to_paragraphs(
            "<p><span style='color: blue'>A✈</span><span style='color: red'>\u{fe0e}B</span></p>"
        );
        let runs = moda_text::runs::map_paragraphs_to_msdf_runs(&paragraphs, &catalog);
        assert!(runs.iter().all(|run| run.kind == moda_text::runs::RunKind::Text), "{runs:?}");
        assert_eq!(runs.iter().map(|run| run.text.as_str()).collect::<String>(), "A✈B");
        for run in &runs {
            assert_eq!(run.color, if run.text.contains('B') { "red" } else { "blue" });
        }
    }

    #[test]
    fn emoji_classification_ignores_style_boundaries() {
        use moda_text::{emoji::Segment, runs::RunKind};
        let catalog = moda_text::catalog::Catalog::from_catalog_json(&serde_json::json!({}));
        for text in ["A✈\u{fe0e}B", "©\u{fe0f}", "1\u{fe0f}\u{20e3}", "👍🏽", "👩\u{200d}💻", "🇺🇸", "🏴\u{e0067}\u{e0062}\u{e007f}"] {
            let expected = moda_text::emoji::split_text_into_segments(text);
            for split in text.char_indices().skip(1).map(|(i, _)| i) {
                let paragraphs = moda_text::parser::parse_rich_text_to_paragraphs(&format!(
                    "<p><span style='color: blue'>{}</span><span style='color: red'>{}</span></p>",
                    &text[..split], &text[split..]
                ));
                let runs = moda_text::runs::map_paragraphs_to_msdf_runs(&paragraphs, &catalog);
                let mut actual = Vec::new();
                for run in runs {
                    match (run.kind, actual.last_mut()) {
                        (RunKind::Text, Some(Segment::Text(previous))) => previous.push_str(&run.text),
                        (RunKind::Text, _) => actual.push(Segment::Text(run.text)),
                        (RunKind::Emoji, _) => {
                            assert_eq!(run.color, "blue", "emoji uses its base's style");
                            actual.push(Segment::Emoji(run.text));
                        }
                    }
                }
                let mut canonical = Vec::new();
                for segment in &expected {
                    match (segment, canonical.last_mut()) {
                        (Segment::Text(text), Some(Segment::Text(previous))) => previous.push_str(text),
                        _ => canonical.push(segment.clone()),
                    }
                }
                assert_eq!(actual, canonical, "{text:?} split at byte {split}");
            }
        }
    }

    #[test]
    fn missing_glyph_refusal_identifies_character() {
        let error = match document("A \u{10ffff} B", serde_json::json!({})) {
            Ok(_) => panic!("missing glyph must refuse instead of drawing notdef"),
            Err(error) => format!("{error:#}"),
        };
        assert_eq!(error.lines().count(), 1, "{error}");
        for part in ["node \"title\"", "\\u{10ffff}"] {
            assert!(error.contains(part), "{error}");
        }
        assert!(!error.contains("lane"), "{error}");
    }

    #[test]
    fn complex_text_stroke_refuses_without_internal_lane_details() {
        let error = match document("مرحبا", serde_json::json!({
            "font":"../vork-std/tests/fonts/NotoSansArabic-Regular.ttf", "stroke":"red","stroke-width":1
        })) {
            Ok(_) => panic!("complex text strokes are not built"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains("node \"title\""), "{error}");
        assert!(error.contains("not built yet"), "{error}");
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(!error.contains("Moda") && !error.contains("tier"), "{error}");
    }
}

#[cfg(test)]
mod paint_tests {
    use super::*;

    #[test]
    fn scrolling_rich_text_through_path_clips_plans_without_refusal() -> Result<()> {
        // Reduced from the author-trial feed: nested scrolling rows, rich
        // spans, and an SVG clip whose bounds do not begin at the origin.
        for path in ["M0 80 h960 v460 h-960 Z", "M0 80 H960 L0 540 Z"] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "render":{"resolution":[960,540], "duration":4},
                "nodes":[
                    {"type":"path", "name":"feed-clip", "d":path, "fill":"white"},
                    {"name":"feed", "clip-path":"feed-clip", "children":[
                        {"name":"scroll", "children":[
                            {"name":"row", "children":[
                                {"type":"text", "content":[{"content":"starred "}, {"content":"codehike"}],
                                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                                    "font-size":16, "translation":[108,49,0], "fill":"black"}
                            ]}
                        ]}
                    ]}
                ],
                "animation":[{"tracks":[{"path":["nodes","scroll","translation"],
                    "keys":[[0,[0,-100,0]],[1,[0,50,0]],[2.9,[0,490,0]],[3,[0,50,0]]]}]}]
            }))?)?;
            let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            for time in [2.9, 1., 0., 3., 2.9] {
                let frame = doc.snapshot(&vork_time::Request::Numeric(time))?;
                doc.plan_snapshot(&frame, &[]).with_context(|| format!("clip {path}, time {time}"))?;
            }
        }
        Ok(())
    }

    #[test]
    fn text_paints_and_geometry_clips_use_the_host_layout() -> Result<()> {
        for fill in ["none", "linear-gradient(to right, red, blue)", "conic-gradient(red, blue)"] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version": 2, "composition": {"space": "2d"},
                "render": {"resolution": [512, 256]},
                "nodes": [
                    {"type": "path", "d": "M0 0H512V256H0Z", "fill": "lime", "clip-path": "title"},
                    {"type": "text", "name": "title", "content": "AB", "font-size": 80,
                        "font": "../vork-std/tests/fonts/LiberationSans-Regular.ttf", "fill": fill,
                        "stroke": "red", "stroke-width": 8, "opacity": 0.2},
                    {"type":"text", "name":"other", "content":"AB", "font-size":80,
                        "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf", "fill":"blue"}
                ],
                "animation": [{"tracks": [
                    {"path": ["nodes", "title", "glyphs", 1, "translation"], "keys": [[0, [300, 0, 0]]]},
                    {"path": ["nodes", "title", "glyphs", 1, "opacity"], "keys": [[0, 0.4]]},
                    {"path": ["nodes", "other", "glyphs", 1, "translation"], "keys": [[0, [-50, 0, 0]]]},
                    {"path": ["nodes", "other", "glyphs", 1, "fill"], "keys": [[0, "green"]]}
                ]}]
            }))?)?;
            let document = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            let frame = document.snapshot(&vork_time::Request::Numeric(0.))?;
            let clip = frame.batches().find(|draw| draw.item == 1 && draw.sub == frame.values[1].texts).unwrap();
            assert_eq!(clip.spec.unit_transforms[1].off, [300., 0.]);
            assert_eq!(clip.spec.fill_override.as_deref(), Some("white"));
            assert!(clip.spec.stroke_color.is_none());
            assert!(clip.spec.unit_transforms.iter().all(|x| x.opacity == 1.));
            let plan = document.plan_snapshot(&frame, &[])?;
            assert!(plan.masks.iter().any(|mask| mask.size[0] > 300.), "moved glyphs must extend the capture beyond the layout box");
        }
        Ok(())
    }

    #[test]
    fn fractional_text_unit_opacity_composes_fill_and_stroke() -> Result<()> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        for level in ["glyphs", "words", "lines"] {
            for content in [serde_json::json!("I"), serde_json::json!([
                {"content":"A", "stroke":"red"}, {"content":"B", "stroke":"blue"}
            ])] {
                for opacity in [0., 0.5, 1.] {
                    let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                        "version":2, "composition":{"space":"2d"},
                        "nodes":[{"type":"text", "name":"title", "content":content,
                            "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                            "fill":"black", "stroke":"black", "stroke-width":10}],
                        "animation":[{"tracks":[{"path":["nodes","title",level,0,"opacity"],
                            "keys":[[0,opacity]]}]}]
                    }))?)?;
                    let doc = Document::new(&scene, base)?;
                    let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                    assert_eq!(!frame.values[0].text_composition.is_empty(), opacity == 0.5);
                    if opacity == 0.5 {
                        use crate::canvas_draw::TextPart;
                        let TextPart::Group { opacity, children } = &frame.values[0].text_composition[0] else {
                            panic!("the faded {level} must compose as a group");
                        };
                        assert_eq!(*opacity, 0.5);
                        assert!(!children.is_empty());
                        assert!(frame.batches().flat_map(|d| d.spec.unit_transforms.iter())
                            .all(|x| x.opacity == 0. || x.opacity == 1.));
                        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
                        let document = crate::canvas_draw::document(&doc.page, &nodes, &frame.values)?;
                        assert!(document["nodes"].as_array().unwrap().iter().any(|node|
                            node["type"] == "group" && node["opacity"] == 0.5));
                        doc.plan_snapshot(&frame, &[])?;
                    }
                }
            }
        }
        for (fill, stroke, width) in [("black", "none", 10), ("none", "black", 10), ("black", "black", 0),
            ("transparent", "black", 10), ("black", "transparent", 10)] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "nodes":[{"type":"text", "name":"title", "content":"I",
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                    "fill":fill, "stroke":stroke, "stroke-width":width}],
                "animation":[{"tracks":[{"path":["nodes","title","glyphs",0,"opacity"],"keys":[[0,0.5]]}]}]
            }))?)?;
            assert!(Document::new(&scene, base)?.snapshot(&vork_time::Request::Numeric(0.))?.values[0].text_composition.is_empty());
        }
        Ok(())
    }

    #[test]
    fn equal_text_unit_opacities_keep_independent_pictures() -> Result<()> {
        use crate::canvas_draw::TextPart;
        for (level, content) in [("glyphs", "HH"), ("words", "H H"), ("lines", "H H")] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "nodes":[{"type":"text", "name":"title", "content":content,
                    "white-space":"pre-line", "font-size":100,
                    "width":if level == "lines" { Some(100) } else { None::<u32> },
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                    "fill":"white", "stroke":"white", "stroke-width":4}],
                "animation":[{"tracks":[
                    {"path":["nodes","title",level,0,"opacity"], "keys":[[0,0.5]]},
                    {"path":["nodes","title",level,1,"opacity"], "keys":[[0,0.5]]},
                    {"path":["nodes","title",level,1,"translation"],
                        "keys":[[0,if level == "lines" { [0,-115,0] } else { [-70,0,0] }]]}
                ]}]
            }))?)?;
            let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            let groups: Vec<_> = frame.values[0].text_composition.iter().filter_map(|part| {
                if let TextPart::Group { opacity, children } = part { Some((*opacity, children)) } else { None }
            }).collect();
            // Two overlapping solid pictures at 0.5 must source-over to 0.75, not 0.5.
            assert_eq!(groups.len(), 2, "separately faded {level} need independent pictures");
            for (opacity, children) in groups {
                assert_eq!(opacity, 0.5);
                assert!(!children.is_empty());
            }
            doc.plan_snapshot(&frame, &[])?;
        }
        Ok(())
    }

    #[test]
    fn unit_opacity_batches_share_layout_payloads() -> Result<()> {
        // Independent opacity pictures must not duplicate the full layout payloads.
        let count = 512;
        for equal in [true, false] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2, "composition":{"space":"2d"},
                "nodes":[{"type":"text", "name":"title", "content":"A".repeat(count),
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf", "font-size":12,
                    "width":1000, "fill":"white", "stroke":"black", "stroke-width":1}],
                "animation":[{"tracks": (0..count).map(|i| serde_json::json!({
                    "path":["nodes","title","glyphs",i,"opacity"],
                    "keys":[[0,if equal || i % 2 == 0 { 0.5 } else { 0.25 }]]
                })).collect::<Vec<_>>()}]
            }))?)?;
            let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            assert_eq!(frame.batches().count(), count);
            let first = &frame.batches().next().unwrap().spec;
            assert_eq!(first.unit_transforms.len(), count);
            for draw in frame.batches() {
                assert!(Arc::ptr_eq(&first.html, &draw.spec.html));
                assert!(Arc::ptr_eq(&first.unit_transforms, &draw.spec.unit_transforms));
                assert!(Arc::ptr_eq(&first.unit_groups, &draw.spec.unit_groups));
                assert!(Arc::ptr_eq(&first.unit_fill, &draw.spec.unit_fill));
            }
            assert_eq!(frame.batches().map(|d| d.spec.unit_range.as_ref().unwrap().len()).sum::<usize>(), count);
        }
        Ok(())
    }

    #[test]
    fn nested_text_opacity_and_seeks_keep_composition() -> Result<()> {
        use crate::canvas_draw::TextPart;
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"},
            "nodes":[{"type":"text", "name":"title", "content":"AB CD", "font-size":80,
                "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                "fill":"white", "stroke":"black", "stroke-width":8},
                {"type":"text", "name":"steady", "content":"AB CD", "font-size":80,
                    "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                    "fill":"blue", "stroke":"red", "stroke-width":4}],
            "animation":[{"tracks":[
                {"path":["nodes","steady","glyphs",0,"translation"], "keys":[[0,[25,0,0]]]},
                {"path":["nodes","steady","words",0,"opacity"], "keys":[[0,0.8]]},
                {"path":["nodes","title","glyphs",0,"translation"], "keys":[[0,[0,0,0]],[1,[10,0,0]]]},
                {"path":["nodes","title","lines",0,"opacity"], "keys":[[0,0.5],[1,1]]},
                {"path":["nodes","title","words",0,"opacity"], "keys":[[0,0.4],[1,1]]},
                {"path":["nodes","title","glyphs",0,"opacity"], "keys":[[0,0.3],[1,1]]},
                {"path":["nodes","title","glyphs",1,"fill"], "keys":[[0,"red"]]},
                {"path":["nodes","title","glyphs",1,"stroke"], "keys":[[0,"blue"]]}
            ]}]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
        let mut steady_spec = None;
        for time in [0., 0., 0.5, 1., 0.5, 0.] {
            let frame = doc.snapshot(&vork_time::Request::Numeric(time))?;
            let steady = frame.batches().find(|draw| draw.item == 1).unwrap();
            if let Some(previous) = &steady_spec { assert!(Arc::ptr_eq(previous, &steady.spec)); }
            steady_spec = Some(steady.spec.clone());
            assert_eq!(frame.values[1].text_revision, 1);
            assert_eq!(steady.spec.unit_transforms[0].off, [25., 0.]);
            let title = frame.batches().find(|draw| draw.item == 0).unwrap();
            assert!((title.spec.unit_transforms[0].off[0] - time as f32 * 10.).abs() < 1e-5);
            if time == 0. {
                let TextPart::Group { opacity: line, children } = &frame.values[0].text_composition[0] else { panic!("line") };
                let TextPart::Group { opacity: word, children } = &children[0] else { panic!("word") };
                let TextPart::Group { opacity: glyph, .. } = &children[0] else { panic!("glyph") };
                for (actual, expected) in [(*line, 0.5), (*word, 0.4), (*glyph, 0.3)] {
                    assert!((actual - expected).abs() < 1e-6);
                }
            }
            assert_eq!(frame.values[0].text_composition.is_empty(), time == 1.);
            doc.plan_snapshot(&frame, &[])?;
            doc.retained.borrow().as_ref().unwrap().assert_fresh(&doc.page, &nodes, &frame.values);
        }
        Ok(())
    }

    #[test]
    fn css_text_paints_reach_moda_as_rgba() -> Result<()> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests");
        for paint in ["rgb(100%, 0%, 0%)", "rgb(255 0 0)", "RGB(255 0 0 / 100%)",
            "hsl(1turn 100% 50%)", "hwb(0 0% 0%)", "red", "#f00"] {
            for content in [serde_json::json!("Hello"), serde_json::json!([{"content":"Hello", "fill":paint,"stroke":paint}])] {
                let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                    "version": 2, "composition": {"space": "2d"}, "nodes": [{
                        "type": "text", "name":"title", "content": content, "font": "fonts/LiberationSans-Regular.ttf",
                        "fill": paint, "stroke": paint
                    }], "animation":[{"tracks":[{"path":["nodes","title","glyphs",0,"fill"],"keys":[[0,paint]]}]}]
                }))?)?;
                let document = Document::new(&scene, &base)?;
                let text = document.items[0].text.as_ref().unwrap();
                assert!(text.fill.iter().all(|p| *p == PaintSource::Color([1., 0., 0., 1.])), "{paint}");
                assert_eq!(text.stroke, text.fill, "{paint}");
                // The colours reach Moda with the draw, not in the layout's HTML.
                assert!(text.spec.run_colors.iter().all(|colour| *colour == [1., 0., 0., 1.]), "{paint}: {:?}", text.spec.run_colors);
                assert!(!text.spec.html.contains("color:"), "{paint}: {}", text.spec.html);
                assert!(document.snapshot(&vork_time::Request::Numeric(0.))?.batches().next().is_some());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod canvas_regressions {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn canvas_png_sequences_finish_with_selected_frames() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-png-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let scene = root.join("scene.json");
        std::fs::write(&scene, serde_json::to_vec(&json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [32, 32], "fps": 30, "duration": 1, "background": "red"},
            "nodes": [], "animation":[{"tracks":[{"path":["render","background"], "keys":[[0,"red"],[1,"blue"]]}]}]
        }))?)?;
        let pixels = |path: PathBuf| -> Result<Vec<u8>> {
            let mut decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?)).read_info()?;
            assert_eq!(decoder.info().srgb, Some(png::SrgbRenderingIntent::Perceptual));
            let mut bytes = vec![0; decoder.output_buffer_size().unwrap()];
            let info = decoder.next_frame(&mut bytes)?;
            bytes.truncate(info.buffer_size());
            Ok(bytes)
        };
        let rendered = crate::shot::Shot { scene: scene.clone(), out: vec![root.join("a").into(), root.join("b").into()],
            frame_start: Some(3), frame_end: Some(10), ..Default::default() }.render()?;
        assert_eq!(rendered.report["frames"], 8);
        for frame in 3..=10 {
            let name = format!("{frame:06}.png");
            assert_eq!(pixels(root.join("a").join(&name))?, pixels(root.join("b").join(&name))?);
        }
        assert_ne!(pixels(root.join("a/000003.png"))?, pixels(root.join("a/000010.png"))?);
        assert!(!root.join("a/000002.png").exists());
        drop(rendered);
        let still = crate::shot::Shot { scene, out: vec![root.join("still.png").into(), root.join("at").into()],
            at: Some("0.2".parse()?), ..Default::default() }.render()?;
        let expected = pixels(root.join("a/000006.png"))?;
        assert_eq!(pixels(root.join("still.png"))?, expected);
        assert_eq!(pixels(root.join("at/000006.png"))?, expected);
        drop(still);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// The precision rule (`Document::float`): only a world, an authored view, an image graph or a linear
    /// working space makes a canvas float.
    #[test]
    fn a_canvas_is_float_only_when_a_world_or_a_view_needs_it() -> Result<()> {
        let float = |scene: Value| -> Result<bool> {
            Ok(Document::new(&Scene::from_json(&serde_json::to_vec(&scene)?)?, Path::new("."))?.float())
        };
        let shape = json!([{"type": "path", "d": "M0 0H16V32H0Z", "fill": "red"}]);
        assert!(!float(json!({"version": 2, "composition": {"space": "2d"}, "render": {"resolution": [32, 32]}, "nodes": shape}))?,
            "a plain canvas keeps the 8-bit picture");
        assert!(float(json!({"version": 2, "composition": {"space": "2d", "view": {"program": "std/standard", "inputs": {}}},
            "render": {"resolution": [32, 32]}, "nodes": shape}))?, "an authored view reads float colour");
        assert!(float(json!({"version": 2, "composition": {"space": "2d", "compositor": ["grade"]}, "render": {"resolution": [32, 32]}, "nodes": shape,
            "materials": {"grade": {"program": "grade"}},
            "programs": {"grade": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_sample(ctx.uv); }"}}}))?,
            "an image graph reads float colour");
        assert!(float(json!({"version": 2, "composition": {"space": "2d"}, "render": {"resolution": [32, 32]},
            "buffers": {"p": {"type": "vec3", "data": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}},
            "geometry": {"g": {"attributes": {"POSITION": "p"}}},
            "nodes": [{"name": "shot", "composition": {"space": "3d"}, "children": [
                {"type": "camera", "name": "cam", "translation": [0, 0, 5]}, {"type": "mesh", "name": "tri", "geometry": "g"}]}]}))?,
            "a world's picture arrives floating point");
        // A working space: linear light needs the precision; an encoded space's code values fit eight bits.
        for (space, expected) in [("sRGB", false), ("Display P3", false), ("Linear Rec.709", true), ("Linear Rec.2020", true), ("ACEScg", true)] {
            assert_eq!(float(json!({"version": 2, "composition": {"space": "2d", "color_space": space}, "render": {"resolution": [32, 32]}, "nodes": shape}))?,
                expected, "{space}");
        }
        Ok(())
    }

    /// A canvas's image graph runs over its picture, on sRGB code values, before its view: an animated input
    /// follows the frame's time, a later entry reads a named half-resolution step, and a result that is not
    /// finite refuses the frame.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_canvas_runs_its_image_graph_before_its_view() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-graph-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let render = |name: &str, at: &str, compositor: Value, programs: Value| -> Result<Vec<u8>> {
            let document = json!({"version": 2, "composition": {"space": "2d", "compositor": compositor},
                "render": {"resolution": [32, 32], "duration": 2, "background": "rgb(128, 64, 32)"}, "nodes": [],
                "materials": {"grade": {"program": "grade"}, "pass": {"program": "pass"}, "add": {"program": "add", "inputs": {"small": {"step": "small"}}}},
                "programs": programs,
                "animation": [{"tracks": [{"path": ["materials", "grade", "inputs", "gain"], "keys": [[0, 0.5], [1, 1.0]]}]}]});
            let scene = root.join(format!("{name}.json"));
            std::fs::write(&scene, serde_json::to_vec(&document)?)?;
            let out = root.join(format!("{name}.png"));
            drop(crate::shot::Shot { scene, out: vec![out.clone().into()], at: Some(at.parse()?), ..Default::default() }.render()?);
            Ok(image::open(&out)?.to_rgba8().into_raw())
        };
        let programs = json!({
            "grade": {"source": "import vork; public struct Inputs { float gain = 1.; }; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(ctx.uv); return float4(c.rgb * inputs.gain, c.a); }"},
            "pass": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_sample(ctx.uv); }"},
            "add": {"source": "import vork; public struct Inputs { Image small; }; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(inputs.small, ctx.uv); return float4(c.rgb * 0.5, c.a); }"}});
        // The graph reads the picture's code values: half of code 128 is code 64, not half its light.
        let half = render("half", "0", json!(["grade"]), programs.clone())?;
        assert!(half.chunks_exact(4).all(|pixel| pixel == [64, 32, 16, 255]), "{:?}", &half[..4]);
        let whole = render("whole", "1", json!(["grade"]), programs.clone())?;
        assert!(whole.chunks_exact(4).all(|pixel| pixel == [128, 64, 32, 255]), "{:?}", &whole[..4]);
        // A named step at half resolution, read by the last entry and resampled to the output.
        let stepped = render("stepped", "1", json!([{"material": "pass", "name": "small", "resolution": 0.5}, "add"]), programs.clone())?;
        assert!(stepped.chunks_exact(4).all(|pixel| pixel == [64, 32, 16, 255]), "{:?}", &stepped[..4]);
        let mut broken = programs;
        broken["grade"] = json!({"source": "import vork; public struct Inputs { float gain = 1.; }; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(ctx.uv); return float4(float3(1. / (c.r - c.r)), c.a); }"});
        let error = format!("{:#}", render("broken", "0", json!(["grade"]), broken).err().context("a non-finite canvas graph rendered")?);
        assert!(error.contains("composition.compositor or composition.view returned a non-finite colour"), "{error}");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// A canvas's EXR is its picture after its view, as a reader takes it. An 8-bit canvas stays 8-bit: its EXR
    /// holds the codes its PNG has. A float canvas's EXR keeps what the 8-bit frame clamps.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_canvas_writes_its_picture_after_its_view_to_exr() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-exr-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let render = |name: &str, composition: Value| -> Result<(Vec<u8>, Vec<f32>)> {
            let document = json!({"version": 2, "composition": composition,
                "render": {"resolution": [32, 32], "background": "rgb(240, 64, 32)"}, "nodes": []});
            let scene = root.join(format!("{name}.json"));
            std::fs::write(&scene, serde_json::to_vec(&document)?)?;
            let (png, exr) = (root.join(format!("{name}.png")), root.join(format!("{name}.%06d.exr")));
            drop(crate::shot::Shot { scene, out: vec![png.clone().into(), exr.into()], at: Some("0".parse()?), ..Default::default() }.render()?);
            Ok((image::open(&png)?.to_rgba8().into_raw(), image::open(root.join(format!("{name}.000000.exr")))?.to_rgba32f().into_raw()))
        };
        let (png, exr) = render("plain", json!({"space": "2d"}))?;
        assert!(png.chunks_exact(4).all(|pixel| pixel == [240, 64, 32, 255]), "{:?}", &png[..4]);
        let codes = [240f32 / 255., 64. / 255., 32. / 255., 1.];
        assert!(exr.chunks_exact(4).all(|pixel| pixel.iter().zip(codes).all(|(a, b)| a.to_bits() == b.to_bits())), "{:?}", &exr[..4]);
        // Two stops up: the frame clamps red at code 255; the EXR keeps the view's code value above one.
        let (png, exr) = render("lifted", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {"exposure": 2}}}))?;
        assert!(png.chunks_exact(4).all(|pixel| pixel[0] == 255), "{:?}", &png[..4]);
        assert!(exr.chunks_exact(4).all(|pixel| pixel[0] > 1.5 && pixel[3] == 1.), "{:?}", &exr[..4]);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// A canvas blends in the encoding its composition names. Its colours, its images and a world's picture are
    /// sRGB code values, converted before they are filtered or blended; its image graph works in that encoding;
    /// its view returns the picture to the display; an EXR stores what a reader takes, in the working space by
    /// default.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_canvas_blends_in_its_working_space() -> Result<()> {
        use v2::WorkingColorSpace as S;
        let root = std::env::temp_dir().join(format!("vork-canvas-spaces-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        image::RgbaImage::from_pixel(8, 8, image::Rgba([255, 0, 0, 128])).save(root.join("red.png"))?;
        // A grey that barely covers, black beside black beside white, a black texel beside a white one, and a
        // one-texel checker of the two.
        image::RgbaImage::from_pixel(8, 8, image::Rgba([128, 128, 128, 1])).save(root.join("faint.png"))?;
        image::RgbaImage::from_fn(3, 1, |x, _| image::Rgba(if x == 2 { [255, 255, 255, 255] } else { [0, 0, 0, 255] })).save(root.join("third.png"))?;
        image::RgbaImage::from_fn(2, 1, |x, _| image::Rgba(if x == 0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] })).save(root.join("ramp.png"))?;
        image::RgbaImage::from_fn(512, 512, |x, y| image::Rgba(if (x + y) % 2 == 0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] })).save(root.join("checker.png"))?;
        // A still at time zero: its PNG's bytes and its EXR's floats.
        let render = |stem: &str, document: &Value, display: Option<v2::OutputDisplay>| -> Result<(Vec<u8>, Vec<f32>)> {
            let scene = root.join(format!("{stem}.json"));
            std::fs::write(&scene, serde_json::to_vec(document)?)?;
            let (png, exr) = (root.join(format!("{stem}.png")), root.join(format!("{stem}.%06d.exr")));
            let frame = v2::OutputDestination { display, ..png.clone().into() };
            drop(crate::shot::Shot { scene, out: vec![frame, exr.into()], at: Some("0".parse()?), ..Default::default() }.render()?);
            Ok((image::open(&png)?.to_rgba8().into_raw(), image::open(root.join(format!("{stem}.000000.exr")))?.to_rgba32f().into_raw()))
        };
        let centre = (16 * 32 + 16) * 4;
        let codes = |colour: [f32; 3]| colour.map(|v| (v.clamp(0., 1.) * 255.).round() as u8);
        let near = |pixel: &[u8], expected: [u8; 3], within: u8| pixel[3] == 255 && pixel[..3].iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= within);
        let decode = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        let encode = |c: f32| if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1. / 2.4) - 0.055 };
        let alpha = 128f32 / 255.;
        for (index, (name, space)) in [("sRGB", S::Srgb), ("Display P3", S::DisplayP3), ("Linear Rec.709", S::LinearRec709),
            ("Linear Rec.2020", S::LinearRec2020), ("ACEScg", S::ACEScg)].into_iter().enumerate() {
            let canvas = |fields: Value| -> Value {
                let mut document = json!({"version": 2, "composition": {"space": "2d", "color_space": name}, "render": {"resolution": [32, 32]}});
                for (field, value) in fields.as_object().expect("fields") {
                    if field == "background" { document["render"]["background"] = value.clone(); } else { document[field.as_str()] = value.clone(); }
                }
                document
            };
            // Red at 128/255 over blue, blended in the working encoding: a colour, and an image of that colour.
            let (red, blue) = (S::Srgb.convert_rgb(space, [1., 0., 0.]), S::Srgb.convert_rgb(space, [0., 0., 1.]));
            let blended: [f32; 3] = std::array::from_fn(|i| alpha * red[i] + (1. - alpha) * blue[i]);
            let shown = codes(space.convert_rgb(S::Srgb, blended));
            for (kind, fill) in [("colour", json!("#ff000080")), ("image", json!({"image": "red", "fit": "fill"}))] {
                let mut document = canvas(json!({"background": "blue", "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": fill}]}));
                if kind == "image" { document["images"] = json!({"red": {"file": "red.png"}}); }
                let (png, exr) = render(&format!("{kind}-{index}"), &document, None)?;
                assert!(near(&png[centre..][..4], shown, 2), "{name} {kind}: {:?} for {shown:?}", &png[centre..][..4]);
                // The EXR stores the reader's picture in the working space: the blend itself.
                let stored = &exr[centre..][..4];
                assert!(stored[3] == 1. && stored[..3].iter().zip(blended).all(|(a, b)| (a - b).abs() < 0.01), "{name} {kind}: {stored:?} for {blended:?}");
                // Written for the display its encoding names, a canvas is its own code values.
                if let Some(own) = space.display().filter(|_| kind == "colour") {
                    let (png, _) = render(&format!("own-{index}"), &document, Some(own))?;
                    assert!(near(&png[centre..][..4], codes(blended), 1), "{name} for its own display: {:?} for {:?}", &png[centre..][..4], codes(blended));
                }
            }
            // An image is resampled in the working encoding. Reduced: a checker of black and white texels drawn
            // sixteen times smaller is their mean. Enlarged: two texels stretched across the canvas mix by the
            // distance between their centres. Each is a mix of code values on an encoded canvas and of light on
            // a linear one. (An sRGB canvas's small still is first enlarged by Moda's bitmap sizing, on its
            // sRGB bytes, which moves the enlarged mix by a code or two.)
            for (image, weight) in [("checker", 0.5f32), ("ramp", (16.5 / 32. - 0.25) / 0.5)] {
                let mut document = canvas(json!({"background": "black",
                    "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": {"image": image, "fit": "fill"}}]}));
                document["images"] = json!({});
                document["images"][image] = json!({"file": format!("{image}.png")});
                let (png, _) = render(&format!("{image}-{index}"), &document, None)?;
                let mixed = codes([if space.is_linear() { encode(weight) } else { weight }; 3]);
                assert!(near(&png[centre..][..4], mixed, 4), "{name} resampling {image}: {:?} for {mixed:?}", &png[centre..][..4]);
            }
            // A canvas outside sRGB takes an image's decoded texels. A texel's colour survives low coverage:
            // grey 128 at alpha 1/255 over black is that grey's light at that coverage, where its 8-bit
            // associated byte would read as white. And a mip level counts every texel of an odd extent: black,
            // black and white drawn one pixel wide are a third white.
            if space.is_linear() {
                let mut document = canvas(json!({"background": "black", "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": {"image": "faint", "fit": "fill"}}]}));
                document["images"] = json!({"faint": {"file": "faint.png"}});
                let (png, _) = render(&format!("faint-{index}"), &document, None)?;
                let faint = codes([encode(decode(128. / 255.) / 255.); 3]);
                assert!(near(&png[centre..][..4], faint, 1), "{name} low coverage: {:?} for {faint:?}", &png[centre..][..4]);
            }
            if space != S::Srgb {
                let mut document = canvas(json!({"background": "black", "nodes": [{"type": "path", "d": "M16 0H17V32H16Z", "fill": {"image": "third", "fit": "fill"}}]}));
                document["images"] = json!({"third": {"file": "third.png"}});
                let (png, _) = render(&format!("third-{index}"), &document, None)?;
                let third = codes([if space.is_linear() { encode(1. / 3.) } else { 1. / 3. }; 3]);
                assert!(near(&png[centre..][..4], third, 4), "{name} odd extent: {:?} for {third:?}", &png[centre..][..4]);
            }
            // The image graph works in the working encoding: half of an sRGB canvas's code values, half of a
            // linear canvas's light.
            if space == S::Srgb || space == S::LinearRec709 {
                let mut document = canvas(json!({"background": "rgb(128, 64, 32)", "nodes": [], "materials": {"half": {"program": "half"}},
                    "programs": {"half": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { float4 c = compositor_sample(ctx.uv); return float4(c.rgb * 0.5, c.a); }"}}}));
                document["composition"]["compositor"] = json!(["half"]);
                let (png, _) = render(&format!("graph-{index}"), &document, None)?;
                let halved = [128f32, 64., 32.].map(|c| c / 255.).map(|c| if space == S::Srgb { c * 0.5 } else { encode(decode(c) * 0.5) });
                assert!(near(&png[centre..][..4], codes(halved), 1), "{name} graph: {:?} for {:?}", &png[centre..][..4], codes(halved));
            }
            // A world's picture is its view's sRGB code values, converted like an image: opaque, it is the same
            // picture in every working space (within the world's own dither).
            let document = canvas(json!({"materials": {"fill": {"program": "fill"}},
                "programs": {"fill": {"source": "import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { return float4(.5, .25, .125, 1.); }"}},
                "nodes": [{"name": "shot", "composition": {"space": "3d", "compositor": ["fill"], "view": {"program": "std/standard", "inputs": {}}},
                    "children": [{"type": "camera", "name": "cam", "translation": [0, 0, 5]}]}]}));
            let (png, _) = render(&format!("world-{index}"), &document, None)?;
            let world = codes([0.5f32, 0.25, 0.125].map(encode));
            assert!(near(&png[centre..][..4], world, 3), "{name} world: {:?} for {world:?}", &png[centre..][..4]);
            // A luminance mask is the luminance of the mask's picture: black under half-covering white is
            // code 0.5 where the canvas blends sRGB code values, and half the light, code 0.735, where it
            // blends linear light. The masked white then shows through that much, blended the same way.
            if space == S::Srgb || space == S::LinearRec709 {
                let document = canvas(json!({"background": "black", "nodes": [
                    {"type": "path", "d": "M0 0H32V32H0Z", "fill": "white", "mask": "cut"},
                    {"name": "cut", "type": "path", "d": "M12 -20H44V52H12Z", "fill": "black", "stroke": "white", "stroke-width": 16, "stroke-opacity": 0.5}]}));
                let (png, _) = render(&format!("mask-{index}"), &document, None)?;
                let through = if space == S::Srgb { 0.5 } else { encode(encode(0.5)) };
                assert!(near(&png[centre..][..4], codes([through; 3]), 4), "{name} mask: {:?} for {:?}", &png[centre..][..4], codes([through; 3]));
            }
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// What a world that reads a canvas takes: the canvas's picture after its view, as sRGB code values, from
    /// any working space. An sRGB canvas's drawn picture is that picture; another encoding's passes its view.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn a_reader_takes_a_canvas_as_srgb_code_values() -> Result<()> {
        use v2::WorkingColorSpace as S;
        let renderer = Renderer::new()?;
        let alpha = 128f32 / 255.;
        for (name, space) in [("sRGB", S::Srgb), ("Display P3", S::DisplayP3), ("Linear Rec.709", S::LinearRec709),
            ("Linear Rec.2020", S::LinearRec2020), ("ACEScg", S::ACEScg)] {
            let scene = Scene::from_json(&serde_json::to_vec(&json!({"version": 2, "composition": {"space": "2d", "color_space": name},
                "render": {"resolution": [32, 32], "background": "blue"},
                "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": "#ff000080"}]}))?)?;
            let document = Document::new(&scene, Path::new("."))?;
            document.install_images()?;
            let engine = crate::canvas_draw::Engine::new_picture(&renderer, document.float(), document.working())?;
            let pictures = Pictures::new(&renderer, &document, Path::new("."), true)?;
            let texture = document.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
            let at = (16 * 32 + 16) * 4;
            let pixel: [f32; 4] = if texture.format() == wgpu::TextureFormat::Rgba16Float {
                let texels = moda_engine::readback::read_texture_rgba16f(&renderer.device, &renderer.queue, &texture, 32, 32)?;
                std::array::from_fn(|c| texels[at + c].to_f32())
            } else {
                let texels = moda_engine::readback::read_texture_rgba(&renderer.device, &renderer.queue, &texture, 32, 32)?;
                std::array::from_fn(|c| f32::from(texels[at + c]) / 255.)
            };
            // Only an sRGB canvas with the default view hands over its drawn 8-bit picture.
            assert_eq!(texture.format() == wgpu::TextureFormat::Rgba8Unorm, space == S::Srgb, "{name}");
            let (red, blue) = (S::Srgb.convert_rgb(space, [1., 0., 0.]), S::Srgb.convert_rgb(space, [0., 0., 1.]));
            let blended: [f32; 3] = std::array::from_fn(|i| alpha * red[i] + (1. - alpha) * blue[i]);
            let expected = space.convert_rgb(S::Srgb, blended);
            assert!(pixel[3] == 1. && pixel[..3].iter().zip(expected).all(|(a, b)| (a - b).abs() < 0.01), "{name}: {pixel:?} for {expected:?}");
        }
        Ok(())
    }

    /// A default canvas is its 8-bit picture copied as it is. A canvas with an authored view or a world in it
    /// keeps a float picture, finished once at the 8-bit output by its view stage. Neither is dithered: a
    /// canvas's flat colour stays flat, and only a world's own picture carries dither.
    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn default_canvases_are_copied_and_float_canvases_are_finished_once() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-float-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let render = |name: &str, composition: Value, extra: Value| -> Result<Vec<u8>> {
            let mut document = json!({"version": 2, "composition": composition,
                "render": {"resolution": [32, 32], "background": "rgb(128, 64, 32)"}, "nodes": []});
            for (field, value) in extra.as_object().expect("fields") { document[field.as_str()] = value.clone(); }
            let scene = root.join(format!("{name}.json"));
            std::fs::write(&scene, serde_json::to_vec(&document)?)?;
            let out = root.join(format!("{name}.png"));
            drop(crate::shot::Shot { scene, out: vec![out.clone().into()], ..Default::default() }.render()?);
            Ok(image::open(&out)?.to_rgba8().into_raw())
        };
        let colour = [128u8, 64, 32, 255];
        let canvas = json!({"space": "2d"});
        let plain = render("plain", canvas.clone(), json!({}))?;
        assert!(plain.chunks_exact(4).all(|p| p == colour), "a default canvas is copied as it is: {:?}", &plain[..4]);
        // std/standard at its defaults is the identity on the canvas's encoding: the same colour through the
        // float picture and the view stage, on every pixel. The stage adds no dither to a canvas.
        let identity = render("identity", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {}}}), json!({}))?;
        assert!(identity.chunks_exact(4).all(|p| p == colour), "{:?}", &identity[..4]);
        // One stop up, in linear light, and still one flat colour.
        let decode = |c: f64| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        let encode = |c: f64| if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1. / 2.4) - 0.055 };
        let lifted = [128., 64., 32.].map(|c: f64| (encode(2. * decode(c / 255.)).clamp(0., 1.) * 255.).round() as u8);
        let viewed = render("lifted", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {"exposure": 1}}}), json!({}))?;
        assert!(viewed[3] == 255 && viewed[..3].iter().zip(&lifted).all(|(a, b)| a.abs_diff(*b) <= 1), "{:?} for {lifted:?}", &viewed[..4]);
        assert!(viewed.chunks_exact(4).all(|p| p == &viewed[..4]), "a viewed flat colour stays flat");
        // A view result that cannot be used refuses the frame, as a world's does.
        let broken = render("broken", json!({"space": "2d", "view": {"program": "broken"}}), json!({"programs": {"broken": {"source":
            "struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i) {return finish_view(float3(1./(ctx.color.x-ctx.color.x)),ctx,1.);}"}}}));
        let error = format!("{:#}", broken.err().context("a non-finite canvas view rendered")?);
        assert!(error.contains("view returned a non-finite colour"), "{error}");
        // A float picture keeps a translucent pixel's straight colour: it is unassociated from float values,
        // where an 8-bit picture has only the rounded associated bytes (128 * 0.02 is 3 of 255, or 153 back).
        let faint = render("faint", json!({"space": "2d", "view": {"program": "std/standard", "inputs": {}}}), json!({
            "render": {"resolution": [32, 32], "background": "transparent"},
            "nodes": [{"type": "path", "d": "M0 0H32V32H0Z", "fill": "rgb(128, 64, 32)", "opacity": 0.02}]}))?;
        let centre = &faint[(16 * 32 + 16) * 4..][..4];
        assert!(centre[3] == 5 && centre[..3].iter().zip(&colour[..3]).all(|(a, b)| a.abs_diff(*b) <= 1), "{centre:?}");
        // A world makes its canvas float without a view: the stage finishes the picture as it is. The
        // canvas's own colour beside the world (the triangle is up and right of the centre) stays exact.
        let world = json!({
            "buffers": {"p": {"type": "vec3", "data": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}},
            "geometry": {"g": {"attributes": {"POSITION": "p"}}},
            "nodes": [{"name": "shot", "composition": {"space": "3d"}, "children": [
                {"type": "camera", "name": "cam", "translation": [0, 0, 5]}, {"type": "mesh", "name": "tri", "geometry": "g"}]}]});
        let still = render("world", canvas.clone(), world.clone())?;
        assert_eq!(still, render("world-again", canvas, world)?);
        assert_eq!(still[31 * 32 * 4..][..4], colour, "the canvas's own colour beside a world");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU, pinned Slang and video encoder"]
    fn canvas_renders_between_encoder_handoff_and_ring_release() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-canvas-overlap-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let scene = root.join("scene.json");
        std::fs::write(&scene, serde_json::to_vec(&json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [192, 108], "fps": 30, "duration": 0.3, "background": "white"},
            "nodes": []
        }))?)?;
        FRAME_EVENTS.with_borrow_mut(|events| *events = Some(Vec::new()));
        let result = crate::shot::Shot { scene, out: vec![root.join("movie.mp4").into()], ..Default::default() }.render();
        let events = FRAME_EVENTS.with_borrow_mut(|events| events.take().unwrap());
        let shot = result?;
        assert_eq!(shot.report["frames"], 9);
        for (end, &(event, frame)) in events.iter().enumerate().filter(|(_, (event, _))| *event == "release") {
            assert_eq!(event, "release");
            let start = events.iter().position(|e| *e == ("push", frame)).expect("release follows handoff");
            assert!(events[start + 1..end].iter().any(|&(event, next)| event == "draw" && next > frame),
                "encoder frame {frame} had no rendering to overlap before release: {events:?}");
        }
        assert!(events.iter().any(|&(event, _)| event == "release"), "exercise ring reuse");
        // Frames stay in flight: a frame reaches the encoder only after its successor was drawn.
        for (at, &(_, frame)) in events.iter().enumerate().filter(|(_, (event, frame))| *event == "push" && frame + 1 < 9) {
            assert!(events[..at].contains(&("draw", frame + 1)), "frame {frame} was accepted before frame {} was drawn: {events:?}", frame + 1);
        }
        drop(shot);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    fn document(nodes: Value, animation: Value) -> Document {
        let scene = Scene::from_json(&serde_json::to_vec(&json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [100, 100]}, "nodes": nodes, "animation": animation
        })).unwrap()).unwrap();
        Document::new(&scene, Path::new(".")).unwrap()
    }

    fn drawing(document: &Document) -> Value {
        let frame = document.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        let nodes: Vec<_> = document.items.iter().map(|item| &item.node).collect();
        crate::canvas_draw::document(&document.page, &nodes, &frame.values).unwrap()
    }

    #[test]
    fn weighted_path_colours_start_from_authored_paint() {
        for field in ["fill", "stroke"] {
            for additive in [false, true] {
                let doc = document(json!([{"name":"p", "type":"path", "d":"M0 0H100V100H0Z",
                    "fill":"red", "stroke":"red"}]), json!([{"weight":0.5, "blend": if additive {"combine"} else {"replace"},
                    "tracks":[{"path":["nodes","p",field], "keys":[[0,[0,0,1,1]],[1,[0,1,0,1]]]}]}]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
                let path = frame.values[0].path.unwrap();
                let colour = if field == "fill" { path.fill } else { path.stroke };
                assert_eq!(colour, Some([if additive {1.} else {0.5}, 0., 0.5, 1.]), "{field}, additive={additive}");
            }
        }
    }

    #[test]
    fn clip_sources_keep_their_own_clip() {
        for group in [false, true] {
            let mut outer = json!({"name":"outer", "type":"path", "d":"M0 0H100V100H0Z", "clip-path":"inner"});
            if group {
                outer = json!({"name":"outer", "clip-path":"inner", "children":[
                    {"type":"path", "d":"M0 0H100V100H0Z"}]});
            }
            let doc = document(json!([
                {"name":"picture", "type":"path", "d":"M0 0H100V100H0Z", "fill":"red", "clip-path":"outer"},
                outer, {"name":"inner", "type":"path", "d":"M0 0H50V100H0Z"}
            ]), json!([]));
            let drawing = drawing(&doc);
            let masks = drawing["nodes"].as_array().unwrap().iter().filter(|node| node.get("clipMaskNodeId").is_some()).count();
            assert_eq!(masks, 2, "clip source group={group}: {drawing}");
        }
    }

    /// The canvas's working encoding reaches Moda's planner with the page, and the nodes it writes are the
    /// same in every encoding: Moda plans a luminance mask by the encoding its engine blends in.
    #[test]
    fn a_canvas_document_is_the_same_in_every_working_space() {
        let document = |space: &str| {
            let scene = Scene::from_json(&serde_json::to_vec(&json!({
                "version": 2, "composition": {"space": "2d", "color_space": space}, "render": {"resolution": [100, 100]},
                "nodes": [{"name":"picture", "type":"path", "d":"M0 0H100V100H0Z", "fill":"white", "mask":"cut"},
                    {"name":"cut", "type":"path", "d":"M0 0H50V100H0Z", "fill":"black", "stroke":"white", "stroke-opacity":0.5}]
            })).unwrap()).unwrap();
            Document::new(&scene, Path::new(".")).unwrap()
        };
        let srgb = document("sRGB");
        assert!(srgb.page.working.is_none());
        for space in ["Display P3", "Linear Rec.709", "Linear Rec.2020", "ACEScg"] {
            let other = document(space);
            assert_eq!(drawing(&other), drawing(&srgb), "{space}");
            let working = other.page.working.expect("a working colour");
            // Authored sRGB white is white in every encoding, and the two conversions invert each other.
            let white = working.from_srgb.apply([1., 1., 1.]);
            assert!(white.iter().all(|c| (c - 1.).abs() < 1e-5), "{space}: {white:?}");
            let back = working.to_srgb.apply(working.from_srgb.apply([0.75, 0.25, 0.5]));
            for (a, b) in back.into_iter().zip([0.75, 0.25, 0.5]) { assert!((a - b).abs() < 1e-4, "{space}: {back:?}"); }
            assert_eq!(working.from_srgb.encode, !other.working().is_linear(), "{space}");
        }
    }

    #[test]
    fn luminance_mask_does_not_paint_referenced_definitions() {
        let doc = document(json!([
            {"name":"picture", "type":"path", "d":"M0 0H100V100H0Z", "fill":"red", "mask":"mask-group"},
            {"name":"mask-group", "children":[
                {"name":"white", "type":"path", "d":"M0 0H100V100H0Z", "fill":"white", "clip-path":"cut"},
                {"name":"cut", "type":"path", "d":"M0 0H50V100H0Z", "fill":"black"}
            ]}
        ]), json!([]));
        let drawing = drawing(&doc);
        let nodes = drawing["nodes"].as_array().unwrap();
        let mask = nodes.iter().find(|node| node["id"] == "n0.luminance").unwrap();
        assert_eq!(mask["children"].as_array().unwrap().len(), 1, "{drawing}");
        assert!(!nodes.iter().any(|node| node["id"] == "n0.luminance/3"), "{drawing}");
    }
}

#[cfg(test)]
mod retained_tests {
    use super::*;

    fn css_document(nodes: serde_json::Value, animation: serde_json::Value) -> Document {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"},
            "render":{"resolution":[100,100]}, "nodes":nodes, "animation":animation
        })).unwrap()).unwrap();
        Document::new(&scene, Path::new(".")).unwrap()
    }

    #[test]
    fn css_blur_refuses_beyond_the_target_pixel_limit() -> Result<()> {
        for group in [false, true] {
            for (sigma, scale, refuses) in [
                (100., 1., false), (100.01, 1., true), (500., 1., true),
                (1e30, 1., true), (50., 2., false), (50.01, 2., true),
                (200., 0.5, false),
            ] {
                let mut node = serde_json::json!({"name":"shape", "type":"path",
                    "d":"M0 0H120V80H0Z", "fill":"red"});
                if group { node = serde_json::json!({"name":"group", "children":[node]}); }
                node["filter"] = serde_json::json!(format!("blur({sigma}px)"));
                node["scale"] = serde_json::json!([scale, scale, 1.]);
                let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                let result = doc.plan_snapshot(&frame, &[]);
                if refuses {
                    let error = result.err().expect("oversized blur must refuse before rendering").to_string();
                    assert!(error.contains("layer blur sigma exceeds 100 target pixels"), "{error}");
                    assert!(!error.contains('\n'), "{error}");
                } else {
                    result?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn css_nested_blurs_keep_every_capture_and_validate_the_parent() -> Result<()> {
        use moda_engine::seams::DrawItem;
        fn captures(items: &[DrawItem]) -> usize {
            items.iter().map(|item| match item {
                DrawItem::LayerBlurComposite { group, .. } => 1 + captures(&group.items),
                _ => 0,
            }).sum()
        }
        for depth in [2, 3, 4] {
            for oversized in [false, true] {
                let mut node = serde_json::json!({"name":"shape", "type":"path",
                    "d":"M20 20H80V80H20Z", "fill":"red", "filter":"blur(1px)"});
                for level in 1..depth {
                    node = serde_json::json!({"name":format!("parent{level}"),
                        "filter":if oversized && level == depth - 1 {"blur(500px)"} else {"blur(2px)"},
                        "children":[node]});
                }
                let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                let result = doc.plan_snapshot(&frame, &[]);
                if oversized {
                    let error = result.err().expect("oversized parent blur must refuse").to_string();
                    assert!(error.contains("layer blur sigma exceeds 100 target pixels"), "{error}");
                    assert!(!error.contains('\n'), "{error}");
                } else {
                    assert_eq!(captures(&result?.items), depth);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn css_blur_uses_small_nonzero_transform_scales() -> Result<()> {
        for scale in [0.000001, 0.00005, 0.0001, 0.0002] {
            for group in [false, true] {
                let extent = 50. / scale;
                let mut node = serde_json::json!({"name":"scaled", "type":"path",
                    "d":format!("M0 0H{extent}V{extent}H0Z"), "fill":"red"});
                if group { node = serde_json::json!({"name":"group", "children":[node]}); }
                node["scale"] = serde_json::json!([scale, scale, 1.]);
                node["filter"] = serde_json::json!("blur(200px)");
                let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                doc.plan_snapshot(&frame, &[])?;
            }
        }
        Ok(())
    }

    #[test]
    fn css_blur_refuses_visible_anisotropy_instead_of_dropping_an_axis() -> Result<()> {
        // Both axis orders, ordinary scales, inherited shear, and the review's
        // tiny-scale input exercise the shared leaf and subtree planner.
        for scale in [[0.00005, 1., 1.], [1., 0.00005, 1.], [0.5, 1., 1.]] {
            for group in [false, true] {
                for sigma in [2., 200.] {
                    let mut node = serde_json::json!({"name":"shape", "type":"path",
                        "d":format!("M0 0H{}V{}H0Z", 50. / scale[0], 50. / scale[1]),
                        "fill":"red"});
                    if group { node = serde_json::json!({"name":"group", "children":[node]}); }
                    node["scale"] = serde_json::json!(scale);
                    node["filter"] = serde_json::json!(format!("blur({sigma}px)"));
                    let doc = css_document(serde_json::json!([node]), serde_json::json!([]));
                    let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                    let error = doc.plan_snapshot(&frame, &[]).err()
                        .expect("visible anisotropic blur must refuse").to_string();
                    assert!(error.contains("layer blur"), "{error}");
                    assert!(error.contains("uniform") || error.contains("reduce"), "{error}");
                    assert!(!error.contains('\n'), "{error}");
                }
            }
        }
        // Rotated nonuniform ancestors produce shear; the leaf's local scale
        // alone cannot identify whether its projected blur is isotropic.
        let doc = css_document(serde_json::json!([{"name":"parent", "scale":[2.,1.,1.],
            "children":[{"name":"child", "rotation":[0.,0.,0.38268343,0.9238795],
                "type":"path", "d":"M0 0H40V40H0Z", "fill":"red", "filter":"blur(2px)"}]}]),
            serde_json::json!([]));
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let error = doc.plan_snapshot(&frame, &[]).err().expect("sheared blur must refuse").to_string();
        assert!(error.contains("uniform"), "{error}");
        Ok(())
    }

    #[test]
    fn css_blur_keeps_conformal_and_invisible_transforms() -> Result<()> {
        for scale in [[0.00005, 0.00005, 1.], [-1., 1., 1.], [1., 1., 1.],
            [0.00005, 0.0001, 1.]] {
            let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
                "d":"M0 0H50V50H0Z", "fill":"red", "scale":scale,
                "rotation":[0.,0.,0.38268343,0.9238795], "filter":"blur(2px)"}]),
                serde_json::json!([]));
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            doc.plan_snapshot(&frame, &[])?;
        }
        Ok(())
    }

    #[test]
    fn css_blur_visibility_uses_the_larger_reach_at_the_cutoff() -> Result<()> {
        // The reaches straddle 0.5px within projection-rounding tolerance.
        // Accepting this near-conformal transform must still retain its blur.
        let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
            "d":"M0 0H300V300H0Z", "fill":"red", "scale":[0.1666,0.1667,1.],
            "filter":"blur(1px)"}]), serde_json::json!([]));
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let plan = doc.plan_snapshot(&frame, &[])?;
        assert!(plan.items.iter().any(|item| matches!(item,
            moda_engine::seams::DrawItem::LayerBlurComposite { .. })),
            "the larger reach is visible, so the blur must remain");
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn held_root_picture_matches_fresh_draws_after_edits() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let reference = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let doc = css_document(serde_json::json!([
            {"name":"shape", "type":"path", "d":"M10 10H60V60H10Z", "fill":"red"},
            {"name":"other", "type":"path", "d":"M40 40H90V90H40Z", "fill":"blue", "opacity":0.5}
        ]), serde_json::json!([]));
        for step in [0, 1, 2, 3, 0] {
            let mut frame = doc.snapshot(&vork_time::Request::Numeric(step as f32))?;
            match step {
                1 => frame.values[0].transform.translation[0] = 13.,
                2 => frame.values[1].opacity = 0.2,
                3 => frame.background = Some([0., 1., 0., 1.]),
                _ => {},
            }
            let plan = doc.plan_snapshot(&frame, &[])?;
            let expected = reference.render_to_texture(&plan)?;
            let expected = moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &expected, 100, 100)?;
            // First render may populate a resource cache. A second draw must
            // establish a stable held picture, and the third must reuse it.
            for _ in 0..2 { engine.render_to_texture(&plan)?; }
            let before = engine.profile_counters();
            let actual = engine.render_to_texture(&plan)?;
            let after = engine.profile_counters();
            assert_eq!(after["picture_hits"].as_u64(), before["picture_hits"].as_u64().map(|v| v + 1));
            assert_eq!(before["bundle_replays"], after["bundle_replays"]);
            let actual = moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &actual, 100, 100)?;
            assert_eq!(actual, expected, "step {step}");
        }
        assert_eq!(reference.profile_counters()["picture_hits"], 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn css_nested_blurs_match_composed_gaussians_and_tiny_scale_renders() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let render = |node: serde_json::Value| -> Result<Vec<u8>> {
            let nodes = if node.is_array() { node } else { serde_json::json!([node]) };
            let doc = css_document(nodes, serde_json::json!([]));
            let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
            let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
            moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &texture, 100, 100)
        };
        let square = || serde_json::json!({"name":"shape", "type":"path",
            "d":"M20 20H80V80H20Z", "fill":"red"});
        // Independent Gaussian convolutions add their variances. Compare associated
        // RGBA with explicit tolerances for finite kernels and capture quantization.
        for (sigmas, combined) in [(vec![3., 4.], 5.), (vec![1., 2., 2.], 3.)] {
            let mut nested = square();
            nested["filter"] = serde_json::json!(format!("blur({}px)", sigmas[0]));
            for (level, sigma) in sigmas.iter().enumerate().skip(1) {
                nested = serde_json::json!({"name":format!("parent{level}"),
                    "filter":format!("blur({sigma}px)"), "children":[nested]});
            }
            let actual = render(nested)?;
            let mut reference = square();
            reference["filter"] = serde_json::json!(format!("blur({combined}px)"));
            let expected = render(reference)?;
            let max = actual.iter().zip(&expected).map(|(a,b)| a.abs_diff(*b)).max().unwrap();
            let rmse = (actual.iter().zip(&expected).map(|(a,b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>() / actual.len() as f64).sqrt();
            assert!(max <= 12 && rmse <= 3., "{sigmas:?}: max={max}, RMSE={rmse}");
        }
        // The child's clip is sampled before the parent blur, while the mask
        // above the parent clips the final filtered result.
        let masked = render(serde_json::json!([
            {"name":"parent", "filter":"blur(4px)", "clip-path":"top", "children":[
                {"name":"child", "filter":"blur(3px)", "clip-path":"left", "children":[square()]}]},
            {"name":"left", "type":"path", "d":"M0 0H50V100H0Z"},
            {"name":"top", "type":"path", "d":"M0 0H100V50H0Z"}
        ]))?;
        let alpha = |x: usize, y: usize| masked[(y * 100 + x) * 4 + 3];
        assert!(alpha(51, 40) > 0, "parent blur spreads past the child's clip");
        assert_eq!(alpha(40, 51), 0, "outer clip masks the parent result");
        assert!(alpha(40, 40) > 200, "interior remains visible");
        let scaled = render(serde_json::json!({"name":"scaled", "type":"path",
            "d":"M0 0H1000000V1000000H0Z", "fill":"red",
            "scale":[0.00005,0.00005,1.], "filter":"blur(200px)"}))?;
        let reference = render(serde_json::json!({"name":"reference", "type":"path",
            "d":"M0 0H50V50H0Z", "fill":"red"}))?;
        assert_eq!(scaled, reference, "0.01 target-pixel sigma preserves the square");
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn css_gradients_ignore_fully_transparent_stop_rgb() -> Result<()> {
        let renderer = Renderer::new()?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        for kind in ["linear-gradient(to right,", "radial-gradient(", "conic-gradient("] {
            let mut reference: Option<Vec<u8>> = None;
            for stop in ["rgba(0,0,255,0)", "rgba(255,0,0,0)", "transparent"] {
                let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
                    "d":"M0 0H100V100H0Z", "fill":format!("{kind}{stop}, blue)")}]), serde_json::json!([]));
                let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
                let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
                let rgba = moda_engine::readback::read_texture_rgba(
                    &renderer.device, &renderer.queue, &texture, 100, 100)?;
                if let Some(expected) = &reference {
                    let max = rgba.iter().zip(expected.iter()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                    assert!(max <= 1, "{kind}{stop}: hidden RGB changed pixels by {max}");
                } else {
                    assert!(rgba.chunks_exact(4).any(|p| p[3] > 32 && p[3] < 224), "exercise translucent pixels");
                    reference = Some(rgba);
                }
            }
        }
        // Nonzero alpha at both ends must also be associated during interpolation.
        let doc = css_document(serde_json::json!([{"name":"shape", "type":"path",
            "d":"M0 0H100V100H0Z",
            "fill":"linear-gradient(to right, rgba(255,0,0,0.25), rgba(0,0,255,0.75))"}]), serde_json::json!([]));
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let texture = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
        let rgba = moda_engine::readback::read_texture_rgba(
            &renderer.device, &renderer.queue, &texture, 100, 100)?;
        let pixel = &rgba[(50 * 100 + 50) * 4..][..4];
        for (actual, expected) in pixel.iter().zip([32u8, 0, 97, 128]) {
            assert!(actual.abs_diff(expected) <= 2, "premultiplied midpoint: {pixel:?}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn retained_picture_outputs_survive_later_draws() -> Result<()> {
        let renderer = Renderer::new()?;
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "render": {"resolution": [32, 32]},
            "nodes": [{"name":"shape", "type":"path", "d":"M0 0H16V32H0Z", "fill":"red"}]
        }))?)?;
        let doc = Document::new(&scene, Path::new("."))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let mut frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let first = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
        let read = |texture: &wgpu::Texture| moda_engine::readback::read_texture_rgba(
            &renderer.device, &renderer.queue, texture, 32, 32);
        let expected = read(&first)?;
        frame.values[doc.by_name["shape"]].transform.translation[0] = 16.;
        let second = engine.render_to_texture(&doc.plan_snapshot(&frame, &[])?)?;
        assert_ne!(read(&second)?, expected, "the second drawing must change");
        assert_eq!(read(&first)?, expected, "a deferred consumer must still see the first drawing");
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn text_filters_follow_glyph_alpha_for_every_paint() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-text-filter-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        let font = dir.join("font.ttf");
        std::fs::write(&font, include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        // Large glyphs: the counters of the O and the gaps of the H must be wider than twice the widest reach
        // probed below plus the shadow's offset, or no pixel is far enough from both the ink and its shadow to
        // be a hole and the oracle probes nothing.
        const W: i32 = 512;
        const H: i32 = 320;
        for fill in [serde_json::json!("red"),
            serde_json::json!("linear-gradient(to right, red, blue)"),
            serde_json::json!({"image":"paint"})] {
            let render = |filter: &str| -> Result<Vec<u8>> {
                let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                    "version":2, "composition":{"space":"2d"},
                    "render":{"resolution":[W,H]},
                    "images":{"paint":{"file":"paint.png"}},
                    "nodes":[{"type":"text", "content":"OHO", "font":font,
                        "font-size":160, "translation":[24,16,0], "fill":fill, "filter":filter}]
                }))?)?;
                let doc = Document::new(&scene, &dir)?;
                doc.install_images()?;
                let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
                let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
                let rgba = moda_engine::readback::read_texture_rgba(
                    &renderer.device, &renderer.queue, &texture, W as u32, H as u32)?;
                Ok(rgba.chunks_exact(4).map(|p| p[3]).collect())
            };
            let source = render("none")?;
            assert!(source.iter().any(|a| *a > 250), "text must render");
            let bounds = source.iter().enumerate().filter(|(_, a)| **a > 0)
                .fold([W, H, 0, 0], |[x0, y0, x1, y1], (i, _)| {
                    let (x, y) = (i as i32 % W, i as i32 / W);
                    [x0.min(x), y0.min(y), x1.max(x), y1.max(y)]
                });
            let sample = |x: i32, y: i32| -> u8 {
                if (0..W).contains(&x) && (0..H).contains(&y) {
                    source[(y * W + x) as usize]
                } else { 0 }
            };
            for (filter, shadow, radius) in [
                ("drop-shadow(12px 8px 0px black)", true, 2),
                ("drop-shadow(12px 8px 4px black)", true, 14),
                ("blur(2px)", false, 8),
                ("drop-shadow(12px 8px 0px black) blur(2px)", true, 8),
            ] {
                let alpha = render(filter)?;
                assert!(alpha.iter().any(|a| *a > 128), "{fill} {filter}: text must remain visible");
                let mut holes = 0;
                let mut shadow_pixels = 0;
                let mut halo_pixels = 0;
                for y in bounds[1]..=bounds[3] + if shadow { 8 } else { 0 } {
                    for x in bounds[0]..=bounds[2] + if shadow { 12 } else { 0 } {
                        // Allow the finite Gaussian reach (3 sigma), plus two
                        // pixels for the different text/mask antialiasing paths.
                        let supported = (-radius..=radius).any(|dy| (-radius..=radius).any(|dx|
                            sample(x + dx, y + dy) > 0
                                || (shadow && sample(x + dx - 12, y + dy - 8) > 0)));
                        let actual = alpha[(y * W + x) as usize];
                        if !supported {
                            holes += 1;
                            assert!(actual <= 3, "{fill} {filter}: glyph hole/background ({x},{y}) has alpha {actual}");
                        }
                        if !shadow && sample(x, y) == 0 && actual > 3 { halo_pixels += 1; }
                        if shadow && sample(x, y) == 0 && sample(x - 12, y - 8) > 250 {
                            shadow_pixels += usize::from(actual > 80);
                        }
                    }
                }
                assert!(holes > 100, "{fill} {filter}: the oracle must probe glyph holes and inter-glyph gaps, and probed {holes}");
                if shadow { assert!(shadow_pixels > 100, "{fill} {filter}: the offset glyphs must cast a visible shadow ({shadow_pixels} pixels)"); }
                else { assert!(halo_pixels > 100, "{fill} {filter}: blur must spread beyond the glyph edges ({halo_pixels} pixels)"); }
            }
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn parent_shadows_follow_glyph_alpha_through_nested_filters() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-nested-text-filter-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        std::fs::write(dir.join("font.ttf"), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        for fill in [serde_json::json!("red"),
            serde_json::json!("linear-gradient(to right, red, blue)"),
            serde_json::json!("radial-gradient(red, blue)"),
            serde_json::json!("conic-gradient(red, blue, red)"),
            serde_json::json!({"image":"paint"})] {
            for depth in [1, 3] {
                let render = |shadow: bool| -> Result<Vec<u8>> {
                    let mut child = serde_json::json!({"name":"text", "type":"text", "content":"OHO",
                        "font":"font.ttf", "font-size":80, "translation":[24,16,0], "fill":fill,
                        "filter":"blur(2px)"});
                    for level in 1..depth {
                        child = serde_json::json!({"name":format!("filter-{level}"),
                            "filter":"blur(0.5px)", "children":[child]});
                    }
                    let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                        "version":2, "composition":{"space":"2d"}, "render":{"resolution":[640,160]},
                        "images":{"paint":{"file":"paint.png"}},
                        "nodes":[{"name":"parent", "filter":if shadow {"drop-shadow(300px 0px 0px black)"} else {"none"},
                            "children":[child]}]
                    }))?)?;
                    let doc = Document::new(&scene, &dir)?;
                    doc.install_images()?;
                    let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
                    let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
                    let rgba = moda_engine::readback::read_texture_rgba(
                        &renderer.device, &renderer.queue, &texture, 640, 160)?;
                    Ok(rgba.chunks_exact(4).map(|p| p[3]).collect())
                };
                let source = render(false)?;
                let shadow = render(true)?;
                let mut holes = 0;
                let mut ink = 0;
                // The far offset isolates the shadow. Probe holes and glyph cores
                // away from antialiasing/blur edges, independently of paint RGB.
                for y in 24..100usize {
                    for x in 24..220usize {
                        let supported = (-2..=2isize).any(|dy| (-2..=2isize).any(|dx| {
                            let (sx, sy) = (x as isize + dx, y as isize + dy);
                            sx >= 0 && sy >= 0 && source[sy as usize * 640 + sx as usize] > 0
                        }));
                        let actual = shadow[y * 640 + x + 300];
                        if !supported {
                            holes += 1;
                            assert!(actual <= 3, "{fill} depth {depth}: shadow fills glyph hole/gap at {x},{y}: {actual}");
                        }
                        // The source is already blurred; its strokes need not
                        // contain fully opaque pixels. Probe 75% coverage.
                        if source[y * 640 + x] > 192 {
                            ink += 1;
                            assert!(actual > 80, "{fill} depth {depth}: missing glyph shadow at {x},{y}");
                        }
                    }
                }
                assert!(holes > 100, "must exercise holes and gaps");
                assert!(ink > 100, "{fill} depth {depth}: only {ink} high-coverage probes; max source alpha {:?}", source.iter().max());
            }
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn rich_text_clip_keeps_exact_shape_with_offscreen_siblings() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-rich-clip-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("font.ttf"), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "render":{"resolution":[240,100]},
            "nodes":[
                {"name":"triangle", "type":"path", "d":"M20 20 H60 L20 60 Z", "fill":"white"},
                {"clip-path":"triangle", "children":[
                    {"type":"text", "content":"H", "font":"font.ttf", "font-size":80, "fill":"red"},
                    {"type":"text", "content":[{"content":"hidden "},{"content":"rich text"}],
                        "font":"font.ttf", "font-size":20, "translation":[0,-100,0]}]},
                {"type":"text", "content":"H", "font":"font.ttf", "font-size":80,
                    "translation":[120,0,0], "fill":"red"}]
        }))?)?;
        let doc = Document::new(&scene, &dir)?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
        for _ in 0..2 {
            let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
            let rgba = moda_engine::readback::read_texture_rgba(
                &renderer.device, &renderer.queue, &texture, 240, 100)?;
            let alpha = |x: usize, y: usize| rgba[(y * 240 + x) * 4 + 3];
            assert!(alpha(46, 25) > 192, "mask removed interior glyph coverage");
            assert!(alpha(46, 55) <= 3, "glyph leaked outside triangle but inside its bounding box");
            assert!(alpha(166, 25) > 192 && alpha(166, 55) > 192, "unclipped reference must contain both probes");
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn ancestor_shadows_follow_masks_at_every_composite_and_paint_depth() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-composite-mask-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        std::fs::write(dir.join("font.ttf"), include_bytes!("../../vork-std/tests/fonts/LiberationSans-Regular.ttf"))?;
        let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
        for text in [false, true] {
        for fill in [serde_json::json!("red"),
            serde_json::json!("linear-gradient(to right, red, blue)"),
            serde_json::json!("radial-gradient(red, blue)"),
            serde_json::json!("conic-gradient(red, blue, red)"),
            serde_json::json!({"image":"paint"})] {
            for paint_depth in 0..4 {
                for leaf_filter in [false, true] {
                    // One tile per subset of masks. Paint may precede deeper
                    // masks; those must not leak back to an earlier sibling.
                    let mut nodes = Vec::new();
                    for placements in 0..16 {
                        let mut child = None;
                        for level in (0..4).rev() {
                            let mut children = Vec::new();
                            if level == paint_depth {
                                let mut paint = if text {
                                    serde_json::json!({"type":"text", "content":"H", "font":"font.ttf", "font-size":80})
                                } else { serde_json::json!({"type":"path", "d":"M0 0H80V80H0Z"}) };
                                paint["fill"] = fill.clone();
                                paint["filter"] = serde_json::json!(if leaf_filter { "blur(2px)" } else { "none" });
                                children.push(paint);
                            }
                            if let Some(child) = child { children.push(child); }
                            let mut group = serde_json::json!({"filter":"blur(0.5px)", "children":children});
                            if placements & (1 << level) != 0 { group["clip-path"] = serde_json::json!("triangle"); }
                            child = Some(group);
                        }
                        nodes.push(serde_json::json!({"translation":[(placements % 4) * 240, (placements / 4) * 100, 0],
                            "filter":"drop-shadow(120px 0px 0px black)", "children":[child.unwrap()]}));
                    }
                    nodes.push(serde_json::json!({"name":"triangle", "type":"path",
                        "d":"M20 20H60L20 60Z", "fill":"white"}));
                    let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                        "version":2, "composition":{"space":"2d"}, "render":{"resolution":[960,400]},
                        "images":{"paint":{"file":"paint.png"}}, "nodes":nodes
                    }))?)?;
                    let doc = Document::new(&scene, &dir)?;
                    doc.install_images()?;
                    let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
                    // A second draw exercises retained plans/resources as well.
                    for _ in 0..2 {
                        let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)
                            .with_context(|| format!("{fill} text {text} depth {paint_depth} leaf {leaf_filter}"))?;
                        let rgba = moda_engine::readback::read_texture_rgba(
                            &renderer.device, &renderer.queue, &texture, 960, 400)?;
                        for placements in 0..16usize {
                            let alpha = |x, y| rgba[(((placements / 4) * 100 + y) * 960
                                + (placements % 4) * 240 + 120 + x) * 4 + 3];
                            assert!(alpha(46, 25) > 192, "missing shadow: {fill} text {text} depth {paint_depth}, masks {placements}");
                            let clipped = placements & ((1 << (paint_depth + 1)) - 1) != 0;
                            if clipped {
                                assert!(alpha(46, 55) <= 3,
                                    "mask lost: {fill} text {text} depth {paint_depth}, masks {placements}, leaf {leaf_filter}");
                            } else {
                                assert!(alpha(46, 55) > 192,
                                    "sibling mask leaked: {fill} text {text} depth {paint_depth}, masks {placements}");
                            }
                        }
                    }
                }
            }
        }
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires real GPU and pinned Slang"]
    fn image_and_conic_strokes_draw_only_the_outline() -> Result<()> {
        let renderer = Renderer::new()?;
        let dir = std::env::temp_dir().join(format!("vork-paint-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        std::fs::create_dir(&dir)?;
        image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255])).save(dir.join("paint.png"))?;
        for paint in [serde_json::json!({"image": "paint"}), serde_json::json!("conic-gradient(red, blue, red)")] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version": 2, "composition": {"space": "2d"}, "render": {"resolution": [128, 128]},
                "images": {"paint": {"file": "paint.png"}},
                "nodes": [{"type": "path", "d": "M32 32H96V96H32Z", "fill": "none", "stroke": paint, "stroke-width": 8}]
            }))?)?;
            let doc = Document::new(&scene, &dir)?;
            doc.install_images()?;
            let engine = crate::canvas_draw::Engine::new_picture(&renderer, false, v2::WorkingColorSpace::Srgb)?;
            let pictures = Pictures::new(&renderer, &doc, &dir, true)?;
            let texture = doc.draw(&engine, &pictures, &renderer, &vork_time::Request::Numeric(0.), None)?;
            let rgba = moda_engine::readback::read_texture_rgba(&renderer.device, &renderer.queue, &texture, 128, 128)?;
            let alpha = |x: usize, y: usize| rgba[(y * 128 + x) * 4 + 3];
            assert!(alpha(32, 64) > 250, "the paint must cover the stroke centre");
            assert_eq!(alpha(64, 64), 0, "a stroke must leave the interior empty");
            assert_eq!(alpha(20, 64), 0, "a stroke must not paint outside its width");
        }
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn retained_canvas_updates_match_fresh_scene() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "nodes": [
                {"name":"solid", "type":"path", "d":"M0 0H30V30H0Z", "fill":"lime", "children":[
                    {"name":"nested-clip", "type":"path", "d":"M0 0H20V20H0Z"}
                ]},
                {"name":"clipped", "type":"path", "d":"M0 0H40V40H0Z", "fill":"blue", "clip-path":"nested-clip"},
                {"name":"clip", "type":"path", "d":"M0 0H50V50H0Z"},
                {"name":"group", "clip-path":"clip", "children":[
                    {"name":"shape", "type":"path", "d":"M0 0H100V100H0Z", "fill":"red",
                     "stroke":"blue", "stroke-width":3, "stroke-dasharray":[4,2], "paint-order":"stroke"},
                    {"name":"text", "type":"text", "content":"Cache", "font-size":20,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"}
                ]}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        for step in 0..20 {
            for (i, value) in values.iter_mut().enumerate() {
                value.transform.translation[0] = (step * (i + 1)) as f32;
                value.transform.scale[0] = if step == 7 { 0. } else { 1. + step as f32 * 0.1 };
                value.opacity = (step % 5) as f32 * 0.2;
                value.visible = step % 4 != 0;
                if let Some(path) = &mut value.path {
                    path.stroke_width = (step % 3) as f32;
                    path.stroke_dashoffset = step as f32 * 0.5;
                    path.fill_opacity = 0.3 + step as f32 * 0.01;
                }
            }
            retained.update(&doc.page, &nodes, &values)?;
            retained.assert_fresh(&doc.page, &nodes, &values);
        }
        retained.update(&doc.page, &nodes, &frame.values)?;
        retained.assert_fresh(&doc.page, &nodes, &frame.values);
        Ok(())
    }

    /// An image or conic stroke is drawn through its coverage, with its picture in a rectangle around it. Its
    /// width and the paint opacities are fields of the nodes it already has: they are written in place, and the
    /// canvas is not lowered again.
    #[test]
    fn coverage_strokes_follow_their_values_without_an_install() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d"},
            "nodes": [
                {"name":"ring", "type":"path", "d":"M32 32H96V96H32Z", "fill":"red",
                 "stroke":"conic-gradient(red, blue, red)", "stroke-width":8},
                {"name":"still", "type":"path", "d":"M0 0H10V10H0Z", "fill":"lime"}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        for step in 1..8 {
            let path = values[doc.by_name["ring"]].path.as_mut().unwrap();
            path.stroke_width = 2. + step as f32;
            path.fill_opacity = 1. - step as f32 * 0.1;
            path.stroke_opacity = 0.2 + step as f32 * 0.1;
            retained.update(&doc.page, &nodes, &values)?;
            retained.assert_fresh(&doc.page, &nodes, &values);
        }
        let counters: serde_json::Value = serde_json::from_str(&retained.profile_counters())?;
        assert!(counters["scene_installs"] == 0 && counters["row_write_frames"] == 7, "values are written into rows: {counters}");
        Ok(())
    }

    #[test]
    fn indexed_canvas_batches_match_fresh_nested_and_clipped_plans() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes":[
                {"name":"g","children":[
                    {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                    {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"linear-gradient(blue, lime)"}]},
                {"name":"mask","type":"path","d":"M0 0H80V80H0Z"},
                {"name":"clipped","type":"path","d":"M0 0H100V100H0Z","fill":"blue","clip-path":"mask"}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new("."))?;
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        let builder: crate::canvas_draw::CanvasTextBuilder = std::rc::Rc::new(|_,_,_,_,_,_,_| Ok(None));
        let mut previous_plan = crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
        let counter = |r: &crate::canvas_draw::Retained, name: &str| -> u64 {
            serde_json::from_str::<serde_json::Value>(&r.profile_counters()).unwrap()[name].as_u64().unwrap()
        };
        for (edits, content, patches) in [
            (vec![("g",11.),("a",3.)], false, true),
            (vec![("a",0.),("g",0.)], false, true),
            // The fade crosses opaque, so the planner prepares `a` again; `b` follows its recipe.
            (vec![("a",5.),("g",2.)], true, true),
            // The generated clip copy is under the translated wrapper, so its
            // mask changes too. The patcher must rebuild unequal masks.
            (vec![("clipped",7.)], false, false),
            (vec![("mask",9.)], false, false),
        ] {
            let changes_mask = edits.iter().any(|(name, _)| matches!(*name, "clipped" | "mask"));
            let before = counter(&retained, "incr_patch_hits");
            if content {
                values[doc.by_name["a"]].opacity = 0.5;
                retained.update(&doc.page, &nodes, &values)?;
            }
            // More than one update before planning must preserve the whole batch.
            for (name, x) in edits {
                values[doc.by_name[name]].transform.translation[0] = x;
                retained.update(&doc.page, &nodes, &values)?;
            }
            let actual = crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
            if changes_mask {
                assert_ne!(actual.masks, previous_plan.masks, "translated clip geometry must update the mask plan");
            }
            assert_eq!(counter(&retained, "incr_patch_hits") - before, u64::from(patches));
            let mut fresh = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
            let expected = crate::canvas_draw::plan(&mut fresh, &doc.page, builder.clone())?;
            assert_eq!(format!("{:?}", *actual), format!("{:?}", *expected));
            previous_plan = actual;
        }
        Ok(())
    }

    /// The oracle for typed placement writes: after every random batch of moves, turns, scales and fades, the
    /// retained scene equals a freshly written one and the retained plan equals a freshly built one.
    /// `uniform` names the nodes whose scale stays uniform (a blurred node refuses any other).
    fn placement_oracle(nodes: serde_json::Value, names: &[&str], leaves: &[&str], uniform: &[&str], group_fades: bool) -> Result<serde_json::Value> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes": nodes}))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let nodes: Vec<_> = doc.items.iter().map(|item| &item.node).collect();
        let fonts = doc.fonts.clone();
        let draws: HashMap<_, _> = frame.batches().map(|d| ((d.item, d.sub), d.clone())).collect();
        let builder: crate::canvas_draw::CanvasTextBuilder = std::rc::Rc::new(move |item, sub, model, z, clip, scale, opacity| {
            let mut placed: Draw = draws.get(&(item, sub)).context("a text batch the frame did not plan")?.clone();
            placed.model = model;
            Arc::make_mut(&mut placed.spec).opacity = opacity;
            fonts.build(&placed, z, clip, scale)
        });
        let mut values = frame.values.clone();
        let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
        crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; (seed >> 40) as f32 / (1u32 << 24) as f32 };
        for step in 0..400 {
            let batches = 1 + (next() * 3.) as usize;
            for _ in 0..batches {
                let name = names[(next() * names.len() as f32) as usize % names.len()];
                let leaf = leaves.contains(&name);
                let value = &mut values[doc.by_name[name]];
                match (next() * 5.) as u32 {
                    0 => value.transform.translation[0] = (next() * 200. - 100.).round() / 4.,
                    1 => value.transform.translation[1] = next() * 60. - 30.,
                    2 => { let half = next() * std::f32::consts::PI; *value.transform.rotation_mut() = [0., 0., half.sin(), half.cos()]; }
                    // Within and across tessellation bands, mirrored now and then.
                    3 => {
                        let scale = 0.4 + next() * 3.;
                        let other = [if next() < 0.2 { -scale } else { scale }, 0.4 + next() * 3., 1.];
                        value.transform.scale = if uniform.contains(&name) { [scale, scale, 1.] } else { other };
                    }
                    // A group that fades composites its children (not a placement): now and then only.
                    _ if leaf => value.opacity = if next() < 0.2 { 1. } else { 0.05 + next() * 0.9 },
                    _ if group_fades => value.opacity = if next() < 0.15 { 0.5 } else { 1. },
                    _ => value.transform.translation[0] = next() * 8.,
                }
                // More than one update before planning must preserve the whole batch.
                retained.update(&doc.page, &nodes, &values)?;
            }
            retained.assert_fresh(&doc.page, &nodes, &values);
            let actual = crate::canvas_draw::plan(&mut retained, &doc.page, builder.clone())?;
            let mut fresh = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
            let expected = crate::canvas_draw::plan(&mut fresh, &doc.page, builder.clone())?;
            assert_eq!(format!("{:?}", *actual), format!("{:?}", *expected), "step {step}");
        }
        Ok(serde_json::from_str(&retained.profile_counters())?)
    }

    #[test]
    fn typed_placements_match_fresh_scenes_and_plans_over_random_edits() -> Result<()> {
        // Plain paths and groups: nearly every frame follows recipes.
        let plain = placement_oracle(serde_json::json!([
            {"name":"g","children":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red"},
                {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"linear-gradient(blue, lime)"},
                {"name":"h","children":[
                    {"name":"c","type":"path","d":"M0 0C10 0 20 10 20 20S0 30 0 20Z","fill":"#336699","opacity":0.5},
                    {"name":"d","type":"path","d":"M5 5H9V9H5Z","fill":"#102030"}]}]},
            {"name":"lone","type":"path","d":"M60 60H90V90H60Z","fill":"#ffcc00cc"}
        ]), &["g", "a", "b", "h", "c", "d", "lone"], &["a", "b", "c", "d", "lone"], &[], false)?;
        assert!(plain["placement_frames"].as_u64().unwrap() > 400 && plain["row_write_frames"] == 0 && plain["scene_installs"] == 0, "placements are written typed: {plain}");
        assert!(plain["incr_placement_nodes"].as_u64().unwrap() > 200, "recipes follow placements: {plain}");
        // Strokes, clips, text and fading groups: recipes where they hold, the planner elsewhere.
        let mixed = placement_oracle(serde_json::json!([
            {"name":"g","children":[
                {"name":"a","type":"path","d":"M0 0H20V20H0Z","fill":"red","stroke":"#00ff0080","stroke-width":2},
                {"name":"b","type":"path","d":"M30 0H50V20H30Z","fill":"linear-gradient(blue, lime)"},
                {"name":"h","children":[
                    {"name":"c","type":"path","d":"M0 0C10 0 20 10 20 20S0 30 0 20Z","fill":"#336699","opacity":0.5}]}]},
            {"name":"lone","type":"path","d":"M60 60H90V90H60Z","fill":"#ffcc00cc"},
            {"name":"mask","type":"path","d":"M0 0H80V80H0Z"},
            {"name":"clipped","type":"path","d":"M0 0H100V100H0Z","fill":"blue","clip-path":"mask"},
            {"name":"text","type":"text","content":"Move","font-size":20,
             "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"}
        ]), &["g", "a", "b", "h", "c", "lone", "clipped", "mask", "text"], &["a", "b", "c", "lone"], &[], true)?;
        assert!(mixed["placement_frames"].as_u64().unwrap() > 100, "placements are written typed: {mixed}");
        assert!(mixed["incr_placement_nodes"].as_u64().unwrap() > 30, "recipes follow placements: {mixed}");
        // A blurred node and a fading group are units: they are planned whole when they or their content move,
        // and the plain nodes beside them keep following recipes.
        let units = placement_oracle(serde_json::json!([
            {"name":"soft","type":"path","d":"M0 0H30V30H0Z","fill":"#cc3366","filter":"blur(3px)"},
            {"name":"fade","opacity":0.5,"children":[
                {"name":"in","type":"path","d":"M40 0H60V20H40Z","fill":"blue"},
                {"name":"in2","type":"path","d":"M50 10H70V30H50Z","fill":"lime"}]},
            {"name":"free","type":"path","d":"M60 60H90V90H60Z","fill":"#ffcc00"},
            {"name":"g","children":[
                {"name":"m","type":"path","d":"M0 40H20V60H0Z","fill":"red"},
                {"name":"n","type":"path","d":"M20 40H40V60H20Z","fill":"#102030"}]}
        ]), &["soft", "fade", "in", "free", "g", "m", "n", "free", "m", "n"], &["free", "m", "n"], &["soft"], false)?;
        assert!(units["incr_placement_nodes"].as_u64().unwrap() > 20, "plain nodes beside units follow recipes: {units}");
        Ok(())
    }

    /// Sparse placement frames through evaluation, the scene write and the plan: unchanged bodies before and
    /// between moved ones, a parent move, frames the recipes decline (a band change, a colour change), and a
    /// frame evaluated but never planned (so the next frame's moved-items list has a stale base).
    #[test]
    fn sparse_placement_frames_plan_like_fresh_documents() -> Result<()> {
        let source = serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"},
            "nodes":[
                {"name":"still0","type":"path","d":"M0 0H10V10H0Z","fill":"#111111"},
                {"name":"g","children":[
                    {"name":"a","type":"path","d":"M20 0H30V10H20Z","fill":"red"},
                    {"name":"still1","type":"path","d":"M40 0H50V10H40Z","fill":"#222222"},
                    {"name":"b","type":"path","d":"M60 0C70 0 70 10 60 10Z","fill":"blue"}]},
                {"name":"still2","type":"path","d":"M80 0H90V10H80Z","fill":"#333333"}],
            "animation":[{"tracks":[
                {"path":["nodes","g","translation"],"keys":[[0,[0,0,0]],[1,[0,0,0]],[2,[30,12,0]]]},
                {"path":["nodes","a","translation"],"keys":[[0,[0,0,0]],[1,[9,3,0]],[3,[9,3,0]]]},
                {"path":["nodes","b","scale"],"keys":[[0,[1,1,1]],[2,[1,1,1]],[3,[5,5,1]]]},
                {"path":["nodes","still2","fill"],"keys":[[0,[0.2,0.2,0.2,1]],[1.4,[0.2,0.2,0.2,1]],[1.6,[1,0,0,1]],[3,[1,0,0,1]]]}
            ]}]
        }))?;
        let document = || Document::new(&Scene::from_json(&source).unwrap(), Path::new("."));
        let kept = document()?;
        // (time, plan it): an unplanned frame leaves the scene one revision behind the next frame's base.
        for (time, planned) in [(0., true), (0.2, true), (0.5, true), (0.5, true), (0.9, true), (1.2, true), (1.3, false),
            (1.35, true), (1.5, true), (1.7, true), (2.2, true), (2.6, true), (2.9, true), (0.4, true), (0.6, true)] {
            let frame = kept.snapshot(&vork_time::Request::Numeric(time))?;
            if !planned { continue; }
            let plan = kept.plan_snapshot(&frame, &[])?;
            let fresh = document()?;
            let fresh_frame = fresh.snapshot(&vork_time::Request::Numeric(time))?;
            let expected = fresh.plan_snapshot(&fresh_frame, &[])?;
            assert_eq!(format!("{:?}", *plan), format!("{:?}", *expected), "time {time}");
        }
        let counters: serde_json::Value = serde_json::from_str(&kept.retained.borrow().as_ref().unwrap().profile_counters())?;
        assert!(counters["placement_frames"].as_u64().unwrap() >= 6, "moves are written sparsely: {counters}");
        assert!(counters["incr_placement_nodes"].as_u64().unwrap() >= 4, "recipes follow them: {counters}");
        Ok(())
    }

    #[test]
    fn retained_shadow_offsets_follow_each_transform_occurrence() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version": 2, "composition": {"space": "2d", "viewBox": [0, 0, 200, 100], "preserveAspectRatio": "none"},
            "render": {"resolution": [400, 300]},
            "nodes": [
                {"name":"mask", "type":"path", "d":"M0 0H50V50H0Z", "fill":"white",
                 "filter":"drop-shadow(3px 5px 2px red) blur(1px)"},
                {"name":"clip", "type":"path", "d":"M0 0H50V50H0Z"},
                {"name":"parent", "children":[
                    {"name":"shape", "type":"path", "d":"M0 0H40V30H0Z", "fill":"red",
                     "filter":"drop-shadow(8px 4px 0px blue)"},
                    {"name":"split", "type":"path", "d":"M0 0H40V30H0Z", "fill":"red",
                     "stroke":"blue", "stroke-width":2, "paint-order":"stroke",
                     "clip-path":"clip", "mask":"mask", "filter":"drop-shadow(8px 4px 0px lime)"},
                    {"name":"text", "type":"text", "content":"A", "font-size":24,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                     "filter":"drop-shadow(8px 4px 0px lime)"},
                    {"name":"batched", "type":"text", "font-size":24, "stroke-width":2,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf",
                     "content":[{"content":"A", "stroke":"red"}, {"content":"B", "stroke":"blue"}],
                     "filter":"drop-shadow(8px 4px 0px lime)"},
                    {"name":"group", "filter":"drop-shadow(8px 4px 0px blue)", "children":[
                        {"type":"path", "d":"M0 0H40V30H0Z", "fill":"red"}
                    ]}
                ]},
                {"name":"other", "scale":[3,2,1], "mask":"mask", "children":[
                    {"type":"path", "d":"M0 0H40V30H0Z", "fill":"blue"}
                ]}
            ]
        }))?)?;
        let doc = Document::new(&scene, Path::new(env!("CARGO_MANIFEST_DIR")))?;
        let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
        let frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        assert!(frame.values[doc.by_name["batched"]].texts > 1);
        // Isolate own, ancestor and mask-instance changes; seek back after each edit.
        for name in ["shape", "split", "text", "batched", "group", "parent", "other", "mask"] {
            let mut values = frame.values.clone();
            let mut retained = crate::canvas_draw::Retained::new(&doc.page, &nodes, &values)?;
            let i = doc.by_name[name];
            for (scale, angle) in [([2., 3., 1.], 0.), ([2., 3., 1.], 1.2), ([-1., 2., 1.], -0.7)] {
                values[i].transform.scale = scale;
                values[i].transform.rotation = Some(glam::Quat::from_rotation_z(angle).to_array());
                retained.update(&doc.page, &nodes, &values)?;
                retained.assert_fresh(&doc.page, &nodes, &values);
            }
            // Rebuilding the hierarchy must refresh the effect bindings as well.
            values[i].visible = false;
            retained.update(&doc.page, &nodes, &values)?;
            values[i].visible = true;
            retained.update(&doc.page, &nodes, &values)?;
            retained.update(&doc.page, &nodes, &frame.values)?;
            retained.assert_fresh(&doc.page, &nodes, &frame.values);
        }
        Ok(())
    }

    #[test]
    fn text_cache_reuses_placement_and_invalidates_content_and_scale() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({
            "version":2, "composition":{"space":"2d"}, "nodes":[
                {"name":"shown", "type":"text", "content":"Cache", "font-size":40,
                 "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"},
                {"visible":false, "children":[
                    {"type":"text", "content":"Hidden", "font-size":40,
                     "font":"../vork-std/tests/fonts/LiberationSans-Regular.ttf"}
                ]}
            ]
        }))?)?;
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let doc = Document::new(&scene, base)?;
        let mut frame = doc.snapshot(&vork_time::Request::Numeric(0.))?;
        let again = doc.snapshot(&vork_time::Request::Numeric(1.))?;
        assert_eq!(frame.batches().count(), 1);
        assert!(Arc::ptr_eq(&frame.batches().next().unwrap().spec, &again.batches().next().unwrap().spec));
        let index = doc.by_name["shown"];
        let texts = |plan: &moda_engine::seams::CoveragePlan| plan.items.iter().filter_map(|item| {
            if let moda_engine::seams::DrawItem::Text(batch) = item { Some(batch.clone()) } else { None }
        }).collect::<Vec<_>>();
        for step in 0..5 {
            match step {
                1 => frame.values[index].transform.translation[0] = 17.,
                2 => {
                    Arc::make_mut(&mut frame.draws[index][0].spec).fill_override = Some("red".into());
                    frame.values[index].text_revision += 1;
                }
                3 => frame.values[index].transform.scale[0] = 0.7,
                4 => frame.values[index].opacity = 0.3,
                _ => {}
            }
            let cached = doc.plan_snapshot(&frame, &[])?;
            let fresh = Document::new(&scene, base)?.plan_snapshot(&frame, &[])?;
            assert_eq!(texts(&cached).len(), 1);
            assert_eq!(texts(&cached), texts(&fresh), "step {step}");
            let (hits, misses) = doc.retained.borrow().as_ref().unwrap().text_cache_counts();
            // Placement and node-group opacity reuse glyphs; content and scale rebuild them.
            assert_eq!(hits, u64::from(step >= 1) + u64::from(step >= 4));
            assert_eq!(misses, if step == 0 { 1 } else { step.min(3) });
        }
        Ok(())
    }

    /// Full animation oracle and CPU-only planning benchmark, with external qualification scenes.
    #[test]
    #[ignore = "requires VORK_CANVAS_CORPUS pointing at the external canvas qualification scenes"]
    fn retained_canvas_corpus() -> Result<()> {
        let root = PathBuf::from(std::env::var("VORK_CANVAS_CORPUS")?);
        let mut ports: Vec<_> = std::fs::read_dir(&root)?.collect::<std::io::Result<Vec<_>>>()?;
        ports.sort_by_key(|p| p.file_name());
        for port in ports {
            let path = port.path().join("port.json");
            if !path.exists() || port.file_name().to_string_lossy().starts_with("05-") { continue; }
            let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            let duration = json["render"]["duration"].as_f64().unwrap_or(1.);
            let fps = json["render"]["fps"].as_f64().unwrap_or(30.);
            let scene = Scene::from_json(&serde_json::to_vec(&json)?)?;
            let doc = Document::new(&scene, &port.path())?;
            let nodes: Vec<_> = doc.items.iter().map(|i| &i.node).collect();
            crate::canvas_draw::install_images(&nodes, &[])?;
            let frames = (duration * fps).round() as usize;
            let mut elapsed = std::time::Duration::ZERO;
            for f in (0..frames).chain((0..frames).rev().step_by(17)) {
                let start = Instant::now();
                let frame = doc.snapshot(&vork_time::Request::Frame(f as u32, scene.render.fps))?;
                let _plan = doc.plan_snapshot(&frame, &[])?;
                elapsed += start.elapsed();
                let nodes: Vec<_> = doc.items.iter().enumerate().map(|(i,item)|frame.nodes[i].as_deref().unwrap_or(&item.node)).collect();
                let mut page=doc.page.clone();page.background=frame.background;
                doc.retained.borrow().as_ref().unwrap().assert_fresh(&page, &nodes, &frame.values);
            }
            eprintln!("{}: {:.3} ms/plan ({frames} forward frames plus reverse seeks)", port.file_name().to_string_lossy(), elapsed.as_secs_f64() * 1000. / (frames + frames.div_ceil(17)) as f64);
        }
        Ok(())
    }
}

#[cfg(test)]
mod style_tests {
    use super::*;
    use serde_json::json;
    fn doc(node:serde_json::Value,tracks:serde_json::Value)->Document {
        let scene=Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},"nodes":[node],"animation":[{"tracks":tracks}]})).unwrap()).unwrap();
        Document::new(&scene,Path::new(".")).unwrap()
    }
    fn drawing(doc:&Document,time:f32)->serde_json::Value {
        let frame=doc.snapshot(&vork_time::Request::Numeric(time)).unwrap();
        let nodes:Vec<_>=doc.items.iter().enumerate().map(|(i,n)|frame.nodes[i].as_deref().unwrap_or(&n.node)).collect();
        crate::canvas_draw::document(&doc.page,&nodes,&frame.values).unwrap()
    }
    /// A group that collapses to zero scale in the frame its text child loses a keyed glyph: the group is
    /// evaluated first whatever the track order, so the child's batches are not built against a parent that
    /// no longer draws. The kept document equals a fresh one.
    #[test]
    fn a_collapsing_parent_is_evaluated_before_its_changed_child() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"g","children":[{"name":"n","type":"text","content":"AB","font":font}]});
        let child=[json!({"path":["nodes","n","content"],"keys":[[0,"AB"],[1,"A"]],"interpolation":"STEP"}),
            json!({"path":["nodes","n","glyphs",1,"opacity"],"keys":[[0,0.25]]})];
        let parent=json!({"path":["nodes","g","scale"],"keys":[[0,[1,1,1]],[1,[0,0,1]]],"interpolation":"STEP"});
        for child_first in [true, false] {
            let tracks=if child_first {json!([child[0],child[1],parent])} else {json!([parent,child[0],child[1]])};
            let kept=doc(node.clone(),tracks.clone());
            assert!(kept.snapshot(&vork_time::Request::Numeric(0.)).unwrap().batches().next().is_some(), "child_first={child_first}");
            let frame=kept.snapshot(&vork_time::Request::Numeric(1.)).expect("a collapsed group's text is not built");
            let fresh=doc(node.clone(),tracks).snapshot(&vork_time::Request::Numeric(1.)).unwrap();
            // The kept document's text cache has a revision and a composition from the frame that drew the text;
            // that identity is not picture content.
            for (kept, fresh) in frame.values.iter().zip(&fresh.values) {
                assert!(kept.transform==fresh.transform && kept.opacity==fresh.opacity && kept.visible==fresh.visible
                    && kept.texts==fresh.texts && kept.path==fresh.path, "child_first={child_first}");
            }
            assert_eq!(frame.batches().count(), 0, "child_first={child_first}");
        }
    }

    #[test]
    fn keyed_layout_piece_bounds_use_sampled_text() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        for (unit, content) in [("glyphs", "AB"), ("words", "A B"), ("lines", "A\nB")] {
            for owned in [false, true] {
                let node=json!({"name":"n","type":"text","content":"A","font":font,"white-space":"pre-line"});
                let prefix=if owned {vec![]} else {vec!["nodes", "n"]};
                let mut content_path=prefix.clone();content_path.push("content");
                let mut piece_path=prefix;piece_path.extend([unit,"1","opacity"]);
                let tracks=json!([
                    {"path":content_path,"keys":[[0,content],[1,"A"]],"interpolation":"STEP"},
                    {"path":piece_path,"keys":[[0,0.25]]}
                ]);
                let animated=if owned {
                    let mut node=node.clone();node["animation"]=json!([{"tracks":tracks}]);doc(node,json!([]))
                } else {doc(node.clone(),tracks)};
                let mut expected=node;expected["content"]=json!(content);
                let expected=doc(expected,json!([{"path":["nodes","n",unit,1,"opacity"],"keys":[[0,0.25]]}]));
                for time in [-1.,0.,0.75] {assert_eq!(drawing(&animated,time),drawing(&expected,time), "{unit}, owned={owned}");}
                let error=animated.snapshot(&vork_time::Request::Numeric(1.)).err().expect("sampled short text must refuse").to_string();
                assert!(error.contains("index 1 is past the end"), "{error}");
                let mut restored = drawing(&animated,0.);
                let mut reference = drawing(&expected,0.);
                // A seek rebuilds text and advances its cache revision; that identity is not picture content.
                for picture in [&mut restored, &mut reference] {
                    for node in picture["nodes"].as_array_mut().unwrap() {
                        if let Some(binding) = node.get_mut("vorkText") { binding[2] = json!(0); }
                    }
                }
                assert_eq!(restored,reference);
            }
        }
    }

    #[test]
    fn frames_are_classified_by_the_largest_changed_track_kind() {
        let doc = doc(serde_json::json!({"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":"#ff0000"}), serde_json::json!([
            {"path":["nodes","p","translation"],"keys":[[0,[0,0,0]],[1,[10,0,0]],[2,[10,0,0]],[3,[10,0,0]],[4,[10,0,0]]]},
            {"path":["nodes","p","opacity"],"keys":[[0,[1]],[1,[1]],[2,[0.5]],[3,[0.5]],[4,[0.5]]]},
            {"path":["nodes","p","fill"],"keys":[[0,[1,0,0,1]],[2,[1,0,0,1]],[3,[0,0,1,1]],[4,[0,0,1,1]]]},
            {"path":["nodes","p","stroke-width"],"keys":[[0,[0]],[3,[0]],[4,[2]]]}
        ]));
        assert_eq!(doc.kinds, [ChangeKind::Placement, ChangeKind::Placement, ChangeKind::PaintValue, ChangeKind::Content].map(Some));
        // First frame, a move, a hold, a fade, a colour change, a stroke change.
        for time in [0., 0.5, 0.5, 1.5, 2.5, 3.5] { doc.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(doc.change_counters(), serde_json::json!({"unchanged":1,"placement":2,"paint_value":1,"content":1,"structure":1}));
    }

    /// A frame that only moves nodes is evaluated for those nodes alone; it equals a full evaluation.
    #[test]
    fn placement_frames_match_full_evaluation() {
        let node = json!({"name":"g","children":[
            {"name":"a","type":"path","d":"M0 0H10V10H0Z","fill":"red"},
            {"name":"b","type":"path","d":"M0 0H10V10H0Z","fill":"blue"},
            {"name":"still","type":"path","d":"M0 0H10V10H0Z","fill":"lime"}]});
        let tracks = json!([
            {"path":["nodes","g","translation"],"keys":[[0,[0,0,0]],[2,[40,10,0]]]},
            {"path":["nodes","a","scale"],"keys":[[0,[1,1,1]],[1,[2,0.5,1]],[2,[1,1,1]]]},
            {"path":["nodes","b","opacity"],"keys":[[0,[1]],[2,[0.2]]]},
            {"path":["nodes","b","translation"],"keys":[[0,[0,0,0]],[1,[5,5,0]],[2,[5,5,0]]]}]);
        let kept = doc(node.clone(), tracks.clone());
        let mut placed = 0;
        for step in [0, 3, 7, 8, 8, 12, 20, 15, 1] {
            let time = step as f32 / 10.;
            let before = kept.change_counters()["placement"].as_u64().unwrap();
            let frame = kept.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            // A placement frame evaluates the items its changed tracks name.
            if kept.change_counters()["placement"].as_u64().unwrap() > before {
                let (items, _) = kept.changed_items(&kept.samples.borrow());
                placed += 1;
                assert!(!items.iter().any(|(item, _)| *item == kept.by_name["still"]), "a node no track moves is not visited");
            }
            let fresh = doc(node.clone(), tracks.clone()).snapshot(&vork_time::Request::Numeric(time)).unwrap();
            assert!(frame.values == fresh.values, "time {time}");
        }
        assert!(placed >= 6, "moves are evaluated sparsely: {placed}");
    }

    /// A text node whose fill or stroke alone changes keeps its text and takes new paints: the kept document's
    /// node, values and batches equal a fresh document's at every time, flat colours and a gradient alike, and
    /// its layout and unit tables are the ones it had (it is not built again).
    #[test]
    fn a_text_fill_change_repaints_the_kept_text() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"n","type":"text","content":[{"content":"A"},{"content":"B","fill":"lime"}],"font":font,"fill":"red","stroke":"black","stroke-width":1});
        let tracks=json!([
            {"path":["nodes","n","fill"],"keys":[[0,"red"],[1,"blue"],[2,"linear-gradient(red, blue)"],[3,"none"]],"interpolation":"STEP"},
            {"path":["nodes","n","stroke"],"keys":[[0,"black"],[1.5,"white"]],"interpolation":"STEP"}]);
        let kept=doc(node.clone(),tracks.clone());
        let mut layout=None;
        for time in [0.,0.5,1.,1.5,2.,3.,1.,0.] {
            let frame=kept.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let fresh_doc=doc(node.clone(),tracks.clone());
            let fresh=fresh_doc.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let specs=|frame:&Snapshot| frame.batches().map(|draw| (draw.sub, draw.spec.run_colors.clone(), draw.spec.fill_override.clone(), draw.spec.stroke_color.clone(), draw.spec.html.clone())).collect::<Vec<_>>();
            assert!(specs(&frame)==specs(&fresh), "time {time}");
            assert!(frame.values[0].text_composition==fresh.values[0].text_composition && frame.values[0].texts==fresh.values[0].texts, "time {time}");
            let (kept_text,fresh_text)=(kept.styles.borrow(),fresh_doc.styles.borrow());
            if let (Some(a),Some(b))=(kept_text[0].as_ref().and_then(|s|s.text.as_ref()),fresh_text[0].as_ref().and_then(|s|s.text.as_ref())) {
                assert!(a.fill==b.fill && a.stroke==b.stroke && a.spec.run_colors==b.spec.run_colors, "time {time}");
                // One layout through every repaint, and the same unit tables: a text built again would have its own.
                let first=layout.get_or_insert_with(||(a.layout.clone(),a.units.clone()));
                assert!(Arc::ptr_eq(&first.0,&a.layout) && Arc::ptr_eq(&first.1,&a.units), "time {time}");
            }
        }
    }

    /// A snapshot planned by another document is that snapshot's state, not the other document's own latest, even
    /// when both have taken as many snapshots.
    #[test]
    fn a_snapshot_is_planned_as_its_own_state_by_any_document() {
        let node = json!({"name":"n","type":"path","d":"M0 0H10V10H0Z","fill":"red"});
        let tracks = json!([{"path":["nodes","n","translation"],"keys":[[0,[0,0,0]],[1,[20,0,0]]]}]);
        let (a, b) = (doc(node.clone(), tracks.clone()), doc(node, tracks));
        let early = a.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        let late = b.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        assert_ne!(early.values[0].transform.translation[0], late.values[0].transform.translation[0]);
        // `b` plans `a`'s snapshot: its scene holds the early state, as a fresh document's would.
        b.plan_snapshot(&early, &[]).unwrap();
        let nodes: Vec<_> = b.items.iter().map(|item| &item.node).collect();
        b.retained.borrow().as_ref().unwrap().assert_fresh(&b.page, &nodes, &early.values);
    }

    /// A world's picture is given by a plan and kept in the evaluated table: a frame that then moves another
    /// item evaluates in place, keeps the picture, and passes the check against a whole evaluation.
    #[test]
    fn a_worlds_picture_stays_in_the_table_across_frames() -> Result<()> {
        let scene = Scene::from_json(&serde_json::to_vec(&json!({"version": 2, "composition": {"space": "2d"}, "render": {"resolution": [32, 32]},
            "buffers": {"p": {"type": "vec3", "data": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}},
            "geometry": {"g": {"attributes": {"POSITION": "p"}}},
            "nodes": [
                {"name": "shot", "composition": {"space": "3d"}, "children": [
                    {"type": "camera", "name": "cam", "translation": [0, 0, 5]}, {"type": "mesh", "name": "tri", "geometry": "g"}]},
                {"name": "box", "type": "path", "d": "M0 0H8V8H0Z", "fill": "red"}],
            "animation": [{"tracks": [{"path": ["nodes", "box", "translation"], "keys": [[0, [0, 0, 0]], [1, [10, 0, 0]]]}]}]}))?)?;
        let doc = Document::new(&scene, Path::new("."))?;
        let world = doc.worlds[0].item;
        doc.snapshot(&vork_time::Request::Numeric(0.))?;
        // What a plan does with the world's picture.
        let picture = Some(("picture".to_string(), [32, 32]));
        doc.evaluated.borrow_mut().as_mut().unwrap().state.values[world].world = picture.clone();
        let moved = doc.snapshot(&vork_time::Request::Numeric(0.5))?;
        assert_eq!(doc.change_counters()["placement"], json!(1), "the box alone moves");
        assert!(moved.values[world].world == picture, "the table keeps the picture");
        assert_eq!(moved.values[doc.by_name["box"]].transform.translation[0], 5.);
        Ok(())
    }

    #[test]
    fn solid_colour_tracks_are_paint_values_however_they_are_spelled() {
        let kinds = |fill: &str, keys: serde_json::Value| doc(serde_json::json!({"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":fill}),
            serde_json::json!([{"path":["nodes","p","fill"],"keys":keys}])).kinds;
        assert_eq!(kinds("red", serde_json::json!([[0,[1,0,0,1]],[1,[0,0,1,1]]])), [Some(ChangeKind::PaintValue)]);
        assert_eq!(kinds("red", serde_json::json!([[0,"red"],[1,"blue"]])), [Some(ChangeKind::PaintValue)]);
        // A key that is not a colour, or a rest paint that is not one, changes the paint's kind.
        assert_eq!(kinds("red", serde_json::json!([[0,"red"],[1,"linear-gradient(red, blue)"]])), [Some(ChangeKind::Content)]);
        assert_eq!(kinds("linear-gradient(red, blue)", serde_json::json!([[0,"red"],[1,"blue"]])), [Some(ChangeKind::Content)]);
        assert_eq!(kinds("none", serde_json::json!([[0,"red"],[1,"blue"]])), [Some(ChangeKind::Content)]);
    }

    #[test]
    fn gradients_of_one_form_are_paint_values() {
        let doc = |fill: &str, tracks: serde_json::Value| doc(serde_json::json!({"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":fill}), tracks);
        let fill = |keys: serde_json::Value, step: bool| if step { serde_json::json!([{"path":["nodes","p","fill"],"keys":keys,"interpolation":"STEP"}]) }
            else { serde_json::json!([{"path":["nodes","p","fill"],"keys":keys}]) };
        let gradient = "linear-gradient(red, blue)";
        // A stop's colour and the gradient's angle change; type, interpolation and stop count do not.
        let stops = doc(gradient, fill(serde_json::json!([[0,gradient],[1,"linear-gradient(green, blue)"]]), false));
        assert_eq!(stops.kinds, [Some(ChangeKind::PaintValue)]);
        for time in [0., 0.25, 0.5] { stops.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(stops.change_counters(), serde_json::json!({"unchanged":0,"placement":0,"paint_value":2,"content":0,"structure":1}));
        assert_eq!(doc(gradient, fill(serde_json::json!([[0,gradient],[1,"linear-gradient(90deg, red, blue)"]]), false)).kinds, [Some(ChangeKind::PaintValue)]);
        // Another type, another stop count, or a colour at rest: the paint's form changes.
        assert_eq!(doc(gradient, fill(serde_json::json!([[0,gradient],[1,"radial-gradient(red, blue)"]]), true)).kinds, [Some(ChangeKind::Content)]);
        assert_eq!(doc(gradient, fill(serde_json::json!([[0,gradient],[1,"linear-gradient(red, green, blue)"]]), true)).kinds, [Some(ChangeKind::Content)]);
        assert_eq!(doc("red", fill(serde_json::json!([[0,gradient],[1,"linear-gradient(green, blue)"]]), false)).kinds, [Some(ChangeKind::Content)]);
        // Opacity scales the paint that is there, whatever its form.
        let opacity = serde_json::json!([{"path":["nodes","p","fill-opacity"],"keys":[[0,[1]],[1,[0]]]}]);
        assert_eq!(doc(gradient, opacity.clone()).kinds, [Some(ChangeKind::PaintValue)]);
        assert_eq!(doc("red", opacity).kinds, [Some(ChangeKind::PaintValue)]);
    }

    #[test]
    fn a_paint_another_layer_can_change_the_form_of_is_content_on_every_track() {
        let scene = Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":"red"}],
            "animation":[
                {"tracks":[{"path":["nodes","p","fill"],"keys":[[0,"red"],[1,"blue"]]}]},
                {"tracks":[{"path":["nodes","p","fill"],"keys":[[0,"red"],[1,"linear-gradient(red, blue)"]],"interpolation":"STEP"}]}]})).unwrap()).unwrap();
        assert_eq!(Document::new(&scene, Path::new(".")).unwrap().kinds, [Some(ChangeKind::Content); 2]);
    }

    #[test]
    fn layer_weight_tracks_keep_the_kind_of_what_they_drive() {
        let scene = |driven: serde_json::Value| {
            let scene = Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},
                "nodes":[{"name":"p","type":"path","d":"M0 0H10V10H0Z","fill":"red"}],
                "animation":[
                    {"tracks":[{"path":["animation","driven","weight"],"keys":[[0,[0]],[1,[1]]]}]},
                    {"name":"driven","tracks":[driven]}]})).unwrap()).unwrap();
            Document::new(&scene, Path::new(".")).unwrap()
        };
        let moved = scene(json!({"path":["nodes","p","translation"],"keys":[[0,[10,0,0]],[1,[10,0,0]]]}));
        assert_eq!(moved.kinds, [None, Some(ChangeKind::Placement)]);
        for time in [0.25, 0.5, 0.75] { moved.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(moved.change_counters(), json!({"unchanged":0,"placement":2,"paint_value":0,"content":0,"structure":1}));
        let widened = scene(json!({"path":["nodes","p","stroke-width"],"keys":[[0,[4]],[1,[4]]]}));
        for time in [0.25, 0.5, 0.75] { widened.snapshot(&vork_time::Request::Numeric(time)).unwrap(); }
        assert_eq!(widened.change_counters(), json!({"unchanged":0,"placement":0,"paint_value":0,"content":2,"structure":1}));
    }

    #[test]
    fn sampled_state_reuses_holds_and_matches_fresh_reverse_seeks() {
        let node=json!({"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"});
        let tracks=json!([
            {"path":["nodes","n","translation"],"keys":[[0,[0,0,0]],[1,[20,0,0]]],"interpolation":"STEP"},
            {"path":["nodes","n","opacity"],"keys":[[0,1],[1,0.4]],"interpolation":"STEP"}
        ]);
        let retained=doc(node.clone(),tracks.clone());
        let first=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        let held=retained.snapshot(&vork_time::Request::Numeric(0.5)).unwrap();
        assert!(first.values == held.values);
        assert_eq!(retained.change_counters()["unchanged"], json!(1), "a held frame evaluates nothing");
        for time in [0.75,1.,2.,0.25,1.,0.] {
            let fresh=doc(node.clone(),tracks.clone());
            assert_eq!(drawing(&retained,time),drawing(&fresh,time),"time {time}");
        }
        // A consumer's edited test frame cannot change another snapshot or
        // trick the retained planner's state identity check.
        let mut edited=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        edited.values[0].transform.translation[0]=17.;
        assert_eq!(first.values[0].transform.translation[0],0.);
    }

    #[test]
    fn sampled_state_tracks_clock_presence_and_recovers_after_errors() {
        let node=json!({"name":"parent","active":[1,3],"children":[
            {"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"}]});
        let retained=doc(node.clone(),json!([]));
        for time in [0.,0.5,1.,2.,3.,2.,0.] {
            assert_eq!(drawing(&retained,time),drawing(&doc(node.clone(),json!([])),time));
        }
        let a=retained.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        let b=retained.snapshot(&vork_time::Request::Numeric(2.)).unwrap();
        assert!(a.values == b.values,"local time alone does not change node values");

        let retained=doc(json!({"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"}),json!([
            {"path":["nodes","n","translation"],"keys":[[0,[0,0,0]],[1,[0,0,10]]],"interpolation":"STEP"}
        ]));
        let good=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        assert!(retained.snapshot(&vork_time::Request::Numeric(1.)).is_err());
        let recovered=retained.snapshot(&vork_time::Request::Numeric(0.)).unwrap();
        assert_eq!(good.values[0].transform,recovered.values[0].transform);
        assert_eq!(retained.change_counters()["structure"], json!(2), "a failed evaluation discards the table: the next frame evaluates every item");
    }

    #[test]
    fn keyed_whitespace_and_width_defer_static_piece_bounds() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        for (field, rest, sampled, unit, index) in [
            ("white-space",json!("pre-line"),json!("normal"),"glyphs",2),
            ("width",json!(1000),json!(20),"lines",1),
        ] {
            let mut node=json!({"name":"n","type":"text","content":"A\nB","font":font});node[field]=rest;
            let animated=doc(node.clone(),json!([
                {"path":["nodes","n",field],"keys":[[0,sampled]]},
                {"path":["nodes","n",unit,index,"opacity"],"keys":[[0,0.5]]}
            ]));
            node[field]=sampled;
            let expected=doc(node,json!([{"path":["nodes","n",unit,index,"opacity"],"keys":[[0,0.5]]}]));
            assert_eq!(drawing(&animated,0.),drawing(&expected,0.),"{field}");
        }
    }

    #[test]
    fn css_background_defaults_to_continuous_interpolation() {
        let scene=Scene::from_json(br##"{"version":2,"composition":{"space":"2d"},"render":{"background":"red"},"nodes":[],"animation":[{"tracks":[{"path":["render","background"],"keys":[[0,"red"],[2,"blue"]]}]}]}"##).unwrap();
        let doc=Document::new(&scene,Path::new(".")).unwrap();
        assert_eq!(doc.snapshot(&vork_time::Request::Numeric(1.)).unwrap().background,Some([0.5,0.,0.5,1.]));
    }

    #[test]
    fn animated_path_filter_and_paint_match_an_authored_midpoint() {
        let node=json!({"name":"n","type":"path","d":"M0 0H10V10H0Z","fill":"red","stroke":"white","stroke-width":2});
        let animated=doc(node.clone(),json!([
            {"path":["nodes","n","d"],"keys":[[0,"M0 0H10V10H0Z"],[2,"M0 0H30V50H0Z"]]},
            {"path":["nodes","n","filter"],"keys":[[0,"none"],[2,"blur(8px)"]]},
            {"path":["nodes","n","fill"],"keys":[[0,"red"],[2,"blue"]]},
            {"path":["nodes","n","stroke-width"],"keys":[[0,2],[2,6]]}
        ]));
        let mut midpoint=node;midpoint["d"]=json!("M0 0H20V30H0Z");midpoint["filter"]=json!("blur(4px)");midpoint["fill"]=json!("rgb(127.5 0 127.5)");midpoint["stroke-width"]=json!(4);
        assert_eq!(drawing(&animated,1.),drawing(&doc(midpoint,json!([])),1.));
        for time in [0.,0.5,1.,1.,2.,0.] {
            let frame=animated.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let nodes:Vec<_>=animated.items.iter().enumerate().map(|(i,n)|frame.nodes[i].as_deref().unwrap_or(&n.node)).collect();
            let mut retained=animated.retained.borrow_mut();
            match retained.as_mut(){Some(r)=>r.update(&animated.page,&nodes,&frame.values).unwrap(),None=>*retained=Some(crate::canvas_draw::Retained::new(&animated.page,&nodes,&frame.values).unwrap())};
            retained.as_ref().unwrap().assert_fresh(&animated.page,&nodes,&frame.values);
        }
    }
    #[test]
    fn keyed_clip_geometry_updates_every_reference() {
        let animated=doc(json!({"name":"root","children":[
            {"name":"clip","type":"path","d":"M0 0H10V10H0Z"},
            {"name":"a","type":"path","d":"M0 0H40V40H0Z","fill":"red","clip-path":"clip"},
            {"name":"b","type":"path","d":"M0 0H40V40H0Z","fill":"blue","clip-path":"clip","translation":[50,0,0]}
        ]}),json!([{"path":["nodes","clip","d"],"keys":[[0,"M0 0H10V10H0Z"],[2,"M0 0H30V30H0Z"]]}]));
        for time in [0.,0.5,1.,2.,0.] {
            let frame=animated.snapshot(&vork_time::Request::Numeric(time)).unwrap();
            let nodes:Vec<_>=animated.items.iter().enumerate().map(|(i,n)|frame.nodes[i].as_deref().unwrap_or(&n.node)).collect();
            let mut retained=animated.retained.borrow_mut();
            match retained.as_mut(){Some(r)=>r.update(&animated.page,&nodes,&frame.values).unwrap(),None=>*retained=Some(crate::canvas_draw::Retained::new(&animated.page,&nodes,&frame.values).unwrap())};
            retained.as_ref().unwrap().assert_fresh(&animated.page,&nodes,&frame.values);
        }
    }

    #[test]
    fn disabled_filter_layer_restores_authored_filter() {
        let scene=Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},"nodes":[{"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red"}],"animation":[
            {"tracks":[{"path":["animation","effect","weight"],"keys":[[0,1],[1,0]]},{"path":["nodes","n","fill-rule"],"keys":[[0,"evenodd"]]}]},
            {"name":"effect","tracks":[{"path":["nodes","n","filter"],"keys":[[0,"blur(4px)"]]}]}
        ]})).unwrap()).unwrap();
        let animated=Document::new(&scene,Path::new(".")).unwrap();
        let rest=doc(json!({"name":"n","type":"path","d":"M0 0H20V20H0Z","fill":"red","fill-rule":"evenodd"}),json!([]));
        let _=drawing(&animated,0.);
        assert_eq!(drawing(&animated,1.),drawing(&rest,1.));
    }
    #[test]
    fn text_settings_start_from_typed_defaults() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"n","type":"text","content":"Keys","font":font});
        let animated=doc(node.clone(),json!([
            {"path":["nodes","n","font-variation-settings"],"keys":[[0,"normal"],[2,"normal"]]},
            {"path":["nodes","n","font-feature-settings"],"keys":[[0,"normal"],[2,"normal"]]}
        ]));
        assert_eq!(drawing(&animated,1.),drawing(&doc(node,json!([])),1.));
    }
    #[test]
    fn text_node_and_glyph_gradients_use_interpolated_stops() {
        let font=Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-std/tests/fonts/LiberationSans-Regular.ttf");
        let node=json!({"name":"n","type":"text","content":"A","font":font,"fill":"linear-gradient(red, blue)"});
        let keys=json!([[0,"linear-gradient(red, blue)"],[2,"linear-gradient(blue, red)"]]);
        let animated=doc(node.clone(),json!([{"path":["nodes","n","fill"],"keys":keys}]));
        let _=animated.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        let styles=animated.styles.borrow();
        let text=styles[0].as_ref().unwrap().text.as_ref().unwrap();
        let expected=crate::canvas_draw::gradient_paint("linear-gradient(rgb(127.5 0 127.5), rgb(127.5 0 127.5))",[0.,0.,text.size[0],text.size[1]]).unwrap();
        assert_eq!(text.fill[0],PaintSource::Moda(expected.clone()));
        let glyph=doc(node,json!([
            {"path":["nodes","n","glyphs",0,"fill"],"keys":[[0,"linear-gradient(red 0px, blue 10px)"],[2,"linear-gradient(blue 0px, red 30px)"]]},
            {"path":["nodes","n","font-size"],"keys":[[0,16],[2,32]]}
        ]));
        let frame=glyph.snapshot(&vork_time::Request::Numeric(1.)).unwrap();
        fn paints(parts:&[crate::canvas_draw::TextPart])->Vec<moda_scene::value::Value> {
            parts.iter().flat_map(|part|match part {
                crate::canvas_draw::TextPart::Paint{paint,..}=>vec![paint.clone()],
                crate::canvas_draw::TextPart::Group{children,..}=>paints(children),
                _=>Vec::new(),
            }).collect()
        }
        let styles=glyph.styles.borrow();let text=styles[0].as_ref().unwrap().text.as_ref().unwrap();
        let expected=crate::canvas_draw::gradient_paint("linear-gradient(rgb(127.5 0 127.5) 0px, rgb(127.5 0 127.5) 20px)",[0.,0.,text.size[0],text.size[1]]).unwrap();
        let mut actual=paints(&frame.values[0].text_composition);
        assert!((actual[0]["stops"][1]["offset"].as_f64().unwrap()-expected["stops"][1]["offset"].as_f64().unwrap()).abs()<1e-12);
        actual[0]["stops"][1]["offset"]=expected["stops"][1]["offset"].clone();
        assert_eq!(actual,vec![expected]);
    }
    #[test]
    fn gradients_interpolate_stops_and_node_clocks_drive_styles() {
        let node=json!({"name":"n","type":"path","time":{"speed":2},"d":"M0 0H100V100H0Z","fill":"linear-gradient(red, blue)",
            "animation":[{"tracks":[{"path":["fill"],"keys":[[0,"linear-gradient(red, blue)"],[2,"linear-gradient(blue, red)"]]}]}]});
        let animated=doc(node,json!([]));
        let static_doc=doc(json!({"name":"n","type":"path","d":"M0 0H100V100H0Z","fill":"linear-gradient(rgb(127.5 0 127.5), rgb(127.5 0 127.5))"}),json!([]));
        assert_eq!(drawing(&animated,0.5),drawing(&static_doc,0.5));
    }
}
