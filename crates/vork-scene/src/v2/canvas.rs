//! The 2D canvas (END-STATE-API §3, §7): `composition: {space: "2d"}` on the scene makes the document a
//! canvas whose nodes paint in tree order, drawn by Moda's 2D engine. The canvas is `render.resolution`
//! pixels, or the SVG `viewBox` placed in it by `preserveAspectRatio`; origin top-left, y down.
//! What is built: group, text and path nodes; SVG 2's path data and paint properties on paths (CSS
//! colours and gradients, an image); the node fields of §3 that move and show them (`translation`,
//! `rotation`, `scale`, `visible`, `opacity`, `clip-path`, `mask`) and the CSS `filter` functions
//! `blur()` and `drop-shadow()`; a CSS colour `render.background`. Tracks on scene seconds (§8) key the
//! numeric and colour fields. Every other 2D field is refused in one line saying it is not built.
use super::*;

/// `2d`: a canvas. `3d`: a closed world (on a node in a canvas).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum Space {
    #[serde(rename = "2d")]
    Canvas,
    #[serde(rename = "3d")]
    World,
}
/// `composition`: on the scene, `{space: "2d"}` makes the document a canvas; on a node, a canvas or a world (compose.rs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Composition {
    #[schemars(extend("x-source" = "END-STATE-API §7", "x-help-path" = "composition.space", "x-help" = "2d|3d; 2d: a canvas, top-left origin, y down, painted in node order; 3d: a standalone or nested closed world"))]
    pub space: Space,
    /// Local view material. Omission selects AgX for worlds, Standard for canvases; never inherited.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "super::output::optional_present")]
    #[schemars(with = "NameOr<Material>", extend("x-source" = "vork colour contract (docs/design/colour.md)", "x-help-path" = "composition.view", "x-help" = "material-name|inline material; local default: worlds std/agx, canvases std/standard; never inherited"))]
    pub view: Option<NameOr<Material>>,
    /// Working RGB encoding; omitted: Linear Rec.709 for worlds, sRGB for canvases.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "super::output::optional_present")]
    #[schemars(with = "WorkingColorSpace", extend("x-source" = "vork colour contract (docs/design/colour.md)", "x-help-path" = "composition.color_space", "x-help" = "Linear Rec.709|Linear Rec.2020|ACEScg|sRGB|Display P3; omitted: worlds Linear Rec.709, canvases sRGB"))]
    pub color_space: Option<WorkingColorSpace>,
    /// This composition's image graph, run in its working space before its view.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(extend("x-source" = "vork colour contract (docs/design/colour.md)", "x-help-path" = "composition.compositor", "x-help" = "(material-name|{material,name?,resolution?})[]; ordered image graph in the working space, before the view; last is the output; material may be inline; a canvas's graph works in the canvas's working encoding (sRGB code values by default)"))]
    pub compositor: Vec<CompositeEntry>,
    /// SVG 2 `viewBox`: min-x, min-y, width, height of the canvas's own coordinates; unsupported on a canvas inside a canvas.
    #[serde(default, rename = "viewBox", skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "SVG 2 viewBox", "x-help-path" = "composition.viewBox", "x-help" = "[min-x,min-y,width,height]|null; the canvas's own coordinates, placed in render.resolution; null = render.resolution px; unsupported on a canvas inside a canvas"))]
    pub view_box: Option<[f32; 4]>,
    /// SVG 2 `preserveAspectRatio`; omitted means `xMidYMid meet`; unsupported on a canvas inside a canvas.
    #[serde(default, rename = "preserveAspectRatio", skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "SVG 2 preserveAspectRatio", "x-help-path" = "composition.preserveAspectRatio", "x-help" = "string|null; none, or x(Min|Mid|Max)Y(Min|Mid|Max) then meet|slice; null = xMidYMid meet; unsupported on a canvas inside a canvas"))]
    pub preserve_aspect_ratio: Option<String>,
}

impl Composition {
    pub fn working_space(&self) -> WorkingColorSpace {
        self.color_space.unwrap_or(match self.space { Space::World => WorkingColorSpace::LinearRec709, Space::Canvas => WorkingColorSpace::Srgb })
    }
    pub fn local_view(&self) -> NameOr<Material> {
        self.view.clone().unwrap_or_else(|| default_view(self.space))
    }
    /// What is built: a canvas blends in any of the five encodings; a world works in Linear Rec.709.
    fn validate_built(&self, path: &str) -> Result<()> {
        ensure!(self.space == Space::Canvas || self.working_space() == WorkingColorSpace::LinearRec709,
            "{path}.color_space: this working space is not built yet; use {}", WorkingColorSpace::LinearRec709.name());
        Ok(())
    }
}
/// A composition's view when it authors none: never its parent's.
fn default_view(space: Space) -> NameOr<Material> {
    NameOr::Inline(Material {
        program: match space { Space::World => "std/agx", Space::Canvas => "std/standard" }.into(),
        inputs: Table::new(), double_sided: true, priority: 0, opacity: None,
    })
}

/// SVG `preserveAspectRatio` (SVG 2 §8.7): how the `viewBox` is placed in the output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AspectRatio {
    /// `None`: `none`, scaled to fill on each axis. Else the alignment (0 min, 0.5 mid, 1 max) per axis.
    pub align: Option<[f32; 2]>,
    /// `slice` (cover) rather than `meet` (contain).
    pub slice: bool,
}
impl AspectRatio {
    pub fn parse(text: Option<&str>) -> std::result::Result<Self, String> {
        let Some(text) = text else { return Ok(Self { align: Some([0.5, 0.5]), slice: false }) };
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let form = || format!("preserveAspectRatio {text:?}: write none, or x(Min|Mid|Max)Y(Min|Mid|Max) optionally followed by meet or slice");
        let (align, rest) = tokens.split_first().ok_or_else(form)?;
        let slice = match rest {
            [] | ["meet"] => false,
            ["slice"] => true,
            _ => return Err(form()),
        };
        if *align == "none" {
            return Ok(Self { align: None, slice });
        }
        let axis = |s: &str| match s { "Min" => Some(0.), "Mid" => Some(0.5), "Max" => Some(1.), _ => None };
        let (x, y) = align.strip_prefix('x').and_then(|a| a.split_once('Y')).ok_or_else(form)?;
        Ok(Self { align: Some([axis(x).ok_or_else(form)?, axis(y).ok_or_else(form)?]), slice })
    }
    /// The canvas-to-output transform for a `viewBox` in a `size` px output: per-axis scale and offset
    /// (output px = canvas units × scale + offset).
    pub fn placement(&self, view_box: [f32; 4], size: [u32; 2]) -> ([f64; 2], [f64; 2]) {
        let [x, y, w, h] = view_box.map(f64::from);
        let (sx, sy) = (f64::from(size[0]) / w, f64::from(size[1]) / h);
        let Some([ax, ay]) = self.align else { return ([sx, sy], [-x * sx, -y * sy]) };
        let s = if self.slice { sx.max(sy) } else { sx.min(sy) };
        let free = [f64::from(size[0]) - w * s, f64::from(size[1]) - h * s];
        ([s, s], [-x * s + free[0] * f64::from(ax), -y * s + free[1] * f64::from(ay)])
    }
}

/// `fill` and `stroke` (§7): `"none"`, a CSS colour, a CSS gradient function, or an image.
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum Paint {
    None(NoPaint),
    /// A CSS colour (`"#fff"`, `"rgb(255 0 0)"`, `"white"`) or a CSS `linear-gradient()`,
    /// `radial-gradient()` or `conic-gradient()` over the shape's bounding box.
    Css(String),
    Image(ImagePaint),
}
/// The string `"none"` (SVG `fill="none"`): nothing is painted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum NoPaint {
    #[default]
    None,
}
impl Paint {
    pub const NONE: Paint = Paint::None(NoPaint::None);
    pub fn is_none(&self) -> bool { matches!(self, Paint::None(_)) }
}
const PAINT_FORM: &str = "a paint is \"none\", a CSS colour (\"#fff\", \"rgb(255 0 0)\", \"white\"), a CSS linear-gradient(), radial-gradient() or conic-gradient(), or {\"image\": name, \"fit\": \"contain\"|\"cover\"|\"fill\"|\"none\"}";
impl<'de> Deserialize<'de> for Paint {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(none) if none == "none" => Ok(Self::NONE),
            serde_json::Value::String(css) => Ok(Self::Css(css)),
            serde_json::Value::Object(map) if map.contains_key("material") || map.contains_key("program") =>
                Err(serde::de::Error::custom("a program paint is not built yet; write \"none\", a CSS colour or gradient, or {\"image\": name}")),
            serde_json::Value::Object(map) => ImagePaint::deserialize(serde_json::Value::Object(map)).map(Self::Image).map_err(serde::de::Error::custom),
            serde_json::Value::Array(_) => Err(serde::de::Error::custom(format!("{PAINT_FORM}; not an RGB array: 2D colours are CSS colours"))),
            _ => Err(serde::de::Error::custom(PAINT_FORM)),
        }
    }
}
/// An image as a paint: bound as every image is (`{image: name}`), drawn in the shape's bounding box
/// and clipped by the shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImagePaint {
    #[schemars(extend("x-source" = "END-STATE-API §2 (a picture is a path filled with an image)", "x-help-path" = "paint.image", "x-help" = "image-name; PNG or JPEG, sRGB; clipped by the shape"))]
    pub image: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS object-fit", "x-help-path" = "paint.fit", "x-help" = "contain|cover|fill|none|null; centred in the shape's box; null = natural size from the box's top-left, repeated as the image's wrap_s and wrap_t say"))]
    pub fit: Option<Fit>,
}
/// CSS `object-fit`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    Contain,
    Cover,
    Fill,
    None,
}

/// SVG `stroke-linecap`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LineCap {
    #[default]
    Butt,
    Round,
    Square,
}
/// SVG `stroke-linejoin`; SVG 2's `miter-clip` and `arcs` are refused as not built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LineJoin {
    #[default]
    Miter,
    Round,
    Bevel,
}
/// SVG `fill-rule`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}
/// SVG `stroke-dasharray`: `"none"` (solid) or the dash and gap lengths in px; an odd count repeats.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum DashArray {
    None(NoDash),
    Lengths(Vec<f32>),
}
/// The string `"none"`: a solid stroke.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum NoDash {
    #[default]
    None,
}
impl Default for DashArray {
    fn default() -> Self { Self::None(NoDash::None) }
}
impl DashArray {
    /// The authored lengths; empty for `"none"`.
    pub fn lengths(&self) -> &[f32] {
        match self { Self::None(_) => &[], Self::Lengths(lengths) => lengths }
    }
}
/// The layers SVG 2 `paint-order` orders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintLayer { Fill, Stroke, Markers }
impl PaintLayer {
    const NORMAL: [PaintLayer; 3] = [PaintLayer::Fill, PaintLayer::Stroke, PaintLayer::Markers];
    fn token(self) -> &'static str {
        match self { Self::Fill => "fill", Self::Stroke => "stroke", Self::Markers => "markers" }
    }
}
/// SVG 2 `paint-order`: `"normal"` (fill, stroke, markers) or the layers to paint first, in order;
/// the layers not named follow in normal order. A path paints no markers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PaintOrder(Vec<PaintLayer>);
impl PaintOrder {
    pub fn parse(text: &str) -> std::result::Result<Self, String> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens == ["normal"] { return Ok(Self::default()); }
        let mut layers = Vec::new();
        for token in &tokens {
            let layer = match *token {
                "fill" => PaintLayer::Fill,
                "stroke" => PaintLayer::Stroke,
                "markers" => PaintLayer::Markers,
                _ => return Err(format!("paint-order: {text:?} is not an SVG paint-order; write \"normal\", or the layers to paint first out of fill, stroke and markers (\"stroke\" paints the stroke under the fill)")),
            };
            if layers.contains(&layer) { return Err(format!("paint-order: {text:?} names {token} twice")); }
            layers.push(layer);
        }
        if layers.is_empty() { return Err("paint-order: write \"normal\" or the layers to paint first (fill, stroke, markers)".into()); }
        Ok(Self(layers))
    }
    /// The painting order after SVG's rule: the named layers first, then the rest in normal order.
    pub fn layers(&self) -> [PaintLayer; 3] {
        let mut order = self.0.clone();
        order.extend(PaintLayer::NORMAL.iter().filter(|layer| !self.0.contains(layer)));
        [order[0], order[1], order[2]]
    }
    /// Whether the stroke is painted before (under) the fill.
    pub fn stroke_first(&self) -> bool {
        let layers = self.layers();
        layers.iter().position(|l| *l == PaintLayer::Stroke) < layers.iter().position(|l| *l == PaintLayer::Fill)
    }
}
impl Serialize for PaintOrder {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        if self.0.is_empty() { return serializer.serialize_str("normal"); }
        serializer.serialize_str(&self.0.iter().map(|layer| layer.token()).collect::<Vec<_>>().join(" "))
    }
}
impl<'de> Deserialize<'de> for PaintOrder {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

/// The default fill of a path: black (SVG); the default stroke: none.
pub(super) fn black() -> Paint { Paint::Css("#000".into()) }
pub(super) fn no_paint() -> Paint { Paint::NONE }
/// The initial value of a path's `d` (SVG): no path.
pub(super) fn no_path() -> String { "none".into() }
pub(super) fn one() -> f32 { 1. }
pub(super) fn is_one(value: &f32) -> bool { *value == 1. }

const NOT_BUILT: &str = "2D is not built yet in a 3D world; put it in a canvas (composition {space: \"2d\"})";

/// The 2D fields that are not built, refused before decoding, each in one line. A node's `composition` changes
/// the space of its children: `3d` in a canvas makes a world, `2d` in a world a canvas (§7).
pub(super) fn refuse_unbuilt(document: &serde_json::Value) -> Result<()> {
    fn composition(value: &serde_json::Value, path: &str) -> Result<()> {
        ensure!(value.get("display").is_none(), "{path}.display: use composition.view; see vork help upgrade");
        ensure!(!value.is_object() || value.get("space").is_some(),
            "{path}.space is required: write \"3d\" for a world (what a scene without a composition is) or \"2d\" for a canvas; see vork help upgrade");
        ensure!(!value.get("view").is_some_and(serde_json::Value::is_null),
            "{path}.view: null is not a view; omit it for the local default");
        if value.get("space").and_then(serde_json::Value::as_str) == Some("3d") {
            ensure!(value.get("viewBox").is_none() && value.get("preserveAspectRatio").is_none(),
                "{path}: a world is framed by its camera; viewBox and preserveAspectRatio belong to a 2D composition");
        }
        Ok(())
    }
    if let Some(value) = document.get("composition") {
        composition(value, "scene.composition")?;
    }
    let canvas = document.pointer("/composition/space").and_then(serde_json::Value::as_str) == Some("2d");
    if canvas {
        for field in document.get("render").and_then(serde_json::Value::as_object).into_iter().flatten().map(|(k, _)| k.as_str()) {
            ensure!(matches!(field, "resolution" | "fps" | "duration" | "background"),
                "render.{field}: not built for a 2D canvas yet; a canvas takes render.resolution, fps, duration and background");
        }
        match document.pointer("/render/background") {
            None | Some(serde_json::Value::String(_)) => {}
            Some(_) => bail!("render.background: a 2D canvas's background is \"transparent\" or a CSS colour"),
        }
        ensure!(document.pointer("/render/background").and_then(serde_json::Value::as_str) != Some("environment"),
            "render.background: a 2D canvas has no environment; write \"transparent\" or a CSS colour");
    }
    /// Where a node is in the document: its index among its parent's children, under where its parent is. It
    /// is written out only into a refusal.
    struct At<'a> { parent: Option<&'a At<'a>>, index: usize }
    impl std::fmt::Display for At<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self.parent {
                Some(parent) => write!(f, "{parent}.children[{}]", self.index),
                None => write!(f, "scene.nodes[{}]", self.index),
            }
        }
    }
    fn nodes(list: &serde_json::Value, parent: Option<&At>, canvas: bool, canvases: &mut Vec<String>) -> Result<()> {
        for (index, node) in list.as_array().into_iter().flatten().enumerate() {
            let here = At { parent, index };
            let kind = node.get("type").and_then(serde_json::Value::as_str).unwrap_or("node");
            let mut inside = canvas;
            if let Some(composition_value) = node.get("composition") {
                composition(composition_value, &format!("{here}.composition"))?;
                let composition = composition_value;
                let space = composition.get("space").and_then(serde_json::Value::as_str);
                ensure!(kind == "node", "{here}.composition: a composition goes on a group node (type \"node\") whose children are the canvas or the world");
                let framed = composition.get("viewBox").is_some() || composition.get("preserveAspectRatio").is_some();
                match (canvas, space) {
                    (true, Some("2d")) => ensure!(!framed, "{here}.composition: viewBox and preserveAspectRatio on a canvas inside a canvas are not built yet"),
                    (true, Some("3d")) => {
                        ensure!(!framed, "{here}.composition: a world is framed by its camera; viewBox and preserveAspectRatio belong to a 2D composition");
                        inside = false;
                    }
                    (false, Some("2d")) => {
                        inside = true;
                        if let Some(name) = node.get("name").and_then(serde_json::Value::as_str) { canvases.push(name.to_string()); }
                    }
                    (false, Some("3d")) => bail!("{here}.composition: a world inside a 3D world is not built yet; its nodes are already in the world"),
                    _ => {}
                }
            }
            if !canvas {
                for field in ["opacity", "clip-path", "mask", "filter"] {
                    ensure!(node.get(field).is_none(), "{here}.{field}: {NOT_BUILT}");
                }
            }
            match kind {
                "path" if !canvas => bail!("{here}: type \"path\": a path in a 3D world is not built yet; put it in a canvas (composition {{space: \"2d\"}}), or extrude it with std extrude"),
                "text" if !canvas => bail!("{here}: type \"text\": text in a 3D world is not built yet; put it in a canvas (composition {{space: \"2d\"}})"),
                "text" | "path" | "node" => {}
                other if canvas => bail!("{here}: type {other:?}: a 2D canvas holds group, text and path nodes; {other} in a canvas is not built yet; put it in a world (composition {{space: \"3d\"}})"),
                _ => {}
            }
            if kind == "text" {
                let runs = node.get("content").and_then(serde_json::Value::as_array).into_iter().flatten().enumerate()
                    .map(|(i, run)| (Some(i), run));
                for (run, value) in std::iter::once((None, node)).chain(runs) {
                    for field in ["fill", "stroke"] {
                        if let Some(value) = value.get(field) {
                            if let Some(paint) = value.as_object() {
                                ensure!(!paint.contains_key("program") && !paint.contains_key("material"),
                                    "{}.{field}: a program paint is not built yet; write a CSS colour or gradient, or {{\"image\": name}}",
                                    match run { Some(i) => format!("{here}.content[{i}]"), None => here.to_string() });
                            }
                        }
                    }
                }
            }
            if kind == "path" {
                for field in ["fill", "stroke"] {
                    if let Some(serde_json::Value::Object(paint)) = node.get(field) {
                        ensure!(!paint.contains_key("material") && !paint.contains_key("program"),
                            "{here}.{field}: a program paint is not built yet; write \"none\", a CSS colour or gradient, or {{\"image\": name}}");
                    }
                }
                if let Some(serde_json::Value::String(join)) = node.get("stroke-linejoin") {
                    ensure!(!matches!(join.as_str(), "miter-clip" | "arcs"), "{here}.stroke-linejoin: {join:?} (SVG 2) is not built yet; use miter, round or bevel");
                }
                if let Some(serde_json::Value::String(dashes)) = node.get("stroke-dasharray") {
                    ensure!(dashes == "none", "{here}.stroke-dasharray: write the dash lengths as an array of px ([4, 2]) or \"none\"");
                }
            }
            if canvas {
                for field in ["target", "pass_index", "is_holdout", "is_shadow_catcher", "light_link_tags", "visible_camera",
                    "visible_shadow", "visible_diffuse", "visible_glossy", "visible_transmission"] {
                    ensure!(node.get(field).is_none(), "{here}.{field}: a 3D field; not in a 2D canvas");
                }
                for field in ["clip-path", "mask"] {
                    if let Some(value) = node.get(field) {
                        ensure!(value.is_string(), "{here}.{field}: name one node whose {} this node; other CSS Masking values are not built",
                            if field == "mask" { "picture masks" } else { "shape clips" });
                    }
                }
            }
            if let Some(children) = node.get("children") {
                nodes(children, Some(&here), inside, canvases)?;
            }
        }
        Ok(())
    }
    let mut canvases = Vec::new();
    nodes(&document["nodes"], None, canvas, &mut canvases)?;
    // A canvas's picture is the canvas, drawn by the 2D engine: it has no camera or backdrop of its own (nor a
    // compositor: `validate_compositions`).
    for (name, image) in document.get("images").and_then(serde_json::Value::as_object).into_iter().flatten() {
        let Some(render) = image.get("render") else { continue };
        if !render.get("root").and_then(serde_json::Value::as_str).is_some_and(|root| canvases.iter().any(|c| c == root)) { continue; }
        for field in ["camera", "background"] {
            ensure!(render.get(field).is_none(), "images.{name}.render.{field}: a canvas's picture has no {field}; it is the canvas, transparent where nothing paints");
        }
        ensure!(image.get("time").is_none(), "images.{name}.time: a canvas's picture shows the canvas node's own clock; set the node's time");
    }
    Ok(())
}

impl Scene {
    /// Whether the document is a 2D canvas (`composition: {space: "2d"}`).
    pub fn is_canvas(&self) -> bool {
        self.composition.as_ref().is_some_and(|c| c.space == Space::Canvas)
    }
    /// This scene's image graph: its own composition's ordered entries.
    pub fn compositor(&self) -> &[CompositeEntry] {
        self.composition.as_ref().map_or(&[][..], |composition| composition.compositor.as_slice())
    }
    /// The scene's image graph for editing; a scene without a composition becomes an explicit world.
    pub fn compositor_mut(&mut self) -> &mut Vec<CompositeEntry> {
        &mut self.composition.get_or_insert_with(|| Composition {
            space: Space::World, view: None, color_space: None, compositor: Vec::new(), view_box: None, preserve_aspect_ratio: None,
        }).compositor
    }
    /// The scene's render with its image graph resolved, as validation and installation read it.
    pub fn main_render(&self) -> Render {
        Render { compositor: self.compositor().to_vec(), ..self.render.clone() }
    }
    /// The view this scene's picture passes through: its own composition's, or the world default.
    pub fn view(&self) -> NameOr<Material> {
        self.composition.as_ref().map_or_else(|| default_view(Space::World), Composition::local_view)
    }
    /// That view's material and, for a named one, its name.
    pub fn view_material(&self) -> Result<(Option<String>, Material)> {
        match self.view() {
            NameOr::Name(name) => {
                let material = self.materials.get(&name).with_context(|| format!("composition.view: unknown material {name}"))?.clone();
                Ok((Some(name), material))
            }
            NameOr::Inline(material) => Ok((None, material)),
        }
    }
    /// Every composition, the scene's and each node's, asks only for what is built.
    pub(super) fn validate_compositions(&self) -> Result<()> {
        fn visit(nodes: &[Node], space: Space) -> Result<()> {
            for node in nodes {
                let mut inside = space;
                if let Some(composition) = &node.composition {
                    composition.validate_built(&format!("node {:?}: composition", node.name))?;
                    // A canvas inside a canvas paints inline in its parent: it has no picture of its own to
                    // view, to run an image graph on, or to blend in an encoding of its own.
                    let inline = space == Space::Canvas && composition.space == Space::Canvas;
                    ensure!(!(inline && composition.color_space.is_some()),
                        "node {:?}: composition.color_space on a canvas inside a canvas is not built yet; it paints inline in its parent", node.name);
                    ensure!(!(inline && composition.view.is_some()),
                        "node {:?}: composition.view on a canvas inside a canvas is not built yet; it paints inline in its parent", node.name);
                    ensure!(!(inline && !composition.compositor.is_empty()),
                        "node {:?}: composition.compositor on a canvas inside a canvas is not built yet; it paints inline in its parent", node.name);
                    inside = composition.space;
                }
                visit(&node.children, inside)?;
            }
            Ok(())
        }
        if let Some(composition) = &self.composition { composition.validate_built("scene.composition")?; }
        visit(&self.nodes, if self.is_canvas() { Space::Canvas } else { Space::World })?;
        // A canvas's picture is the canvas, drawn by the 2D engine: it has no compositor of its own.
        for (image, _) in self.canvas_pictures() {
            ensure!(self.images[image].composition.is_none(), "image {image}: a canvas's picture has no compositor of its own; it is the canvas");
        }
        Ok(())
    }
    /// The canvas's own coordinates (`viewBox`), or `render.resolution` px.
    pub fn canvas_view_box(&self) -> [f32; 4] {
        let [w, h] = self.render.resolution;
        self.composition.as_ref().and_then(|c| c.view_box).unwrap_or([0., 0., w as f32, h as f32])
    }
    /// The canvas's checks after decoding: its text and path nodes' fields, its node references and
    /// its composition.
    pub(super) fn validate_canvas(&self, names: &BTreeMap<&str, &Node>) -> Result<()> {
        if let Background::Css(color) = &self.render.background {
            ensure!(self.is_canvas(), "render.background {color:?}: a CSS colour is a 2D canvas's background; a 3D world takes environment, transparent or RGB");
        }
        if !self.is_canvas() { return Ok(()); }
        let composition = self.composition.as_ref().expect("a canvas has a composition");
        if let Some(view_box) = composition.view_box {
            ensure!(view_box.iter().all(|v| v.is_finite()) && view_box[2] > 0. && view_box[3] > 0.,
                "scene.composition.viewBox: [min-x, min-y, width, height] with a positive width and height");
        }
        AspectRatio::parse(composition.preserve_aspect_ratio.as_deref()).map_err(|e| anyhow::anyhow!("scene.composition.{e}"))?;
        // A canvas's image graph reads the canvas's picture, earlier steps and image files. A world's channels
        // and AOVs do not exist on a canvas, and a rendered image is a world's.
        for (index, entry) in composition.compositor.iter().enumerate() {
            let Some(material) = self.materials.get(entry.material()) else { continue };
            for (input, value) in &material.inputs {
                let at = format!("composition.compositor[{index}]: input {input}");
                match value {
                    Input::Channel(binding) => ensure!(binding.channel == KernelChannel::Color,
                        "{at}: channel {} is a 3D world's; a 2D canvas has color", binding.channel.token()),
                    Input::Aov(binding) => bail!("{at}: AOV {} is a 3D world's; a 2D canvas has color", binding.aov),
                    Input::Texture(binding) => ensure!(self.images.get(&binding.image).is_none_or(|image| image.file.is_some()),
                        "{at}: image {} is rendered; a 2D canvas's compositor reads image files", binding.image),
                    _ => {}
                }
            }
        }
        fn visit(scene: &Scene, nodes: &[Node], names: &BTreeMap<&str, &Node>) -> Result<()> {
            for node in nodes {
                Scene::validate_text(node, &scene.images)?;
                let name = &node.name;
                ensure!(node.opacity.is_finite(), "node {name:?}: opacity must be a finite number");
                for (field, target) in [("clip-path", &node.clip_path), ("mask", &node.mask)] {
                    let Some(target) = target else { continue };
                    ensure!(names.contains_key(target.as_str()), "node {name:?}: {field} names {target:?}, which is no node in the canvas");
                }
                if let NodeKind::Path { fill_opacity, stroke_width, stroke_miterlimit, stroke_dasharray, stroke_dashoffset, stroke_opacity, path_length, stroke, fill, .. } = &node.kind {
                    ensure!(fill_opacity.is_finite() && stroke_opacity.is_finite(), "path {name:?}: fill-opacity and stroke-opacity must be finite numbers");
                    ensure!(stroke_width.is_finite() && *stroke_width >= 0., "path {name:?}: stroke-width must be a nonnegative number of px");
                    ensure!(stroke_miterlimit.is_finite() && *stroke_miterlimit >= 1., "path {name:?}: stroke-miterlimit must be at least 1");
                    ensure!(stroke_dasharray.lengths().iter().all(|v| v.is_finite() && *v >= 0.), "path {name:?}: stroke-dasharray lengths must be nonnegative px");
                    ensure!(stroke_dashoffset.is_finite(), "path {name:?}: stroke-dashoffset must be a finite number of px");
                    if let Some(length) = path_length {
                        ensure!(length.is_finite() && *length > 0., "path {name:?}: pathLength is the author's length of d and must be positive");
                    }
                    for (field, paint) in [("fill", fill), ("stroke", stroke)] {
                        if let Paint::Image(image) = paint {
                            ensure!(scene.images.contains_key(&image.image), "path {name:?}: {field} names image {:?}, which is not in images", image.image);
                        }
                    }
                }
                visit(scene, &node.children, names)?;
            }
            Ok(())
        }
        visit(self, &self.nodes, names)?;
        // Rendering dependencies include children and both kinds of references. Check every
        // definition before hiding referenced roots, including invisible and unused nodes.
        let mut states = BTreeMap::new();
        for &root in names.values() {
            let mut stack = vec![(root, false)];
            while let Some((node, exiting)) = stack.pop() {
                let name = node.name.as_str();
                if exiting {
                    states.insert(name, 2u8);
                    continue;
                }
                match states.get(name) {
                    Some(1) => bail!("clip-path/mask dependency cycle at node {name:?}"),
                    Some(2) => continue,
                    _ => {}
                }
                states.insert(name, 1);
                stack.push((node, true));
                stack.extend(node.children.iter().map(|child| (child, false)));
                for target in [&node.clip_path, &node.mask].into_iter().flatten() {
                    stack.push((names[target.as_str()], false));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_compositor_belongs_to_a_composition() -> Result<()> {
        let grade = serde_json::json!({"materials":{"grade":{"program":"grade"}},
            "programs":{"grade":{"source":"import vork; public struct Inputs {}; public float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_sample(ctx.uv); }"}}});
        let scene = |fields: serde_json::Value| -> Result<Scene> {
            let mut value = grade.clone();
            value["version"] = 2.into();
            for (key, field) in fields.as_object().expect("an object") { value[key] = field.clone(); }
            Scene::from_json(&serde_json::to_vec(&value)?)
        };
        let refusal = |fields: serde_json::Value| format!("{:#}", scene(fields).unwrap_err());
        // A world's graph, at the root and on a world inside a canvas, travels with its composition.
        let root = scene(serde_json::json!({"composition":{"space":"3d","compositor":["grade"]},"nodes":[{"type":"camera"}]}))?;
        assert_eq!(root.compositor().len(), 1);
        assert_eq!(root.main_render().compositor, root.compositor());
        assert!(root.render.compositor.is_empty());
        let nested = scene(serde_json::json!({"composition":{"space":"2d"},"render":{"resolution":[8,8]},
            "nodes":[{"name":"world","composition":{"space":"3d","compositor":["grade"]},"children":[{"type":"camera","name":"camera"}]}]}))?;
        assert!(nested.compositor().is_empty());
        assert_eq!(nested.world_part("world", [8, 8])?.compositor().len(), 1);
        // A rendered image's graph is its own composition's.
        let image = scene(serde_json::json!({"nodes":[{"type":"camera"}],"images":{"capture":{"composition":{"compositor":["grade"]},"render":{}}}}))?;
        assert_eq!(image.images["capture"].resolved_render(&image.render).expect("a rendered image").compositor.len(), 1);
        assert!(refusal(serde_json::json!({"nodes":[{"type":"camera"}],"images":{"file":{"composition":{"compositor":["grade"]},"file":"a.png"}}}))
            .contains("image file: composition is a rendered image's image graph; give the image a render"));
        // The old addresses name the new one.
        assert_eq!(refusal(serde_json::json!({"render":{"compositor":["grade"]}})),
            "render.compositor was removed; use composition.compositor; composition.space is required too: write \"3d\" (a scene without a composition is a world); see vork help upgrade");
        // Every move is named at once.
        assert_eq!(refusal(serde_json::json!({"render":{"compositor":["grade"],"display":"look"},"images":{"capture":{"render":{"compositor":["grade"]}}}})),
            "render.display was removed; use composition.view; render.compositor was removed; use composition.compositor; \
             images.capture.render.compositor was removed; use images.capture.composition.compositor; \
             composition.space is required too: write \"3d\" (a scene without a composition is a world); see vork help upgrade");
        // An inline image is named where the author wrote it.
        assert_eq!(refusal(serde_json::json!({"materials":{"m":{"inputs":{"tex":{"image":{"render":{"compositor":["grade"]}}}}}}})),
            "materials.m.inputs.tex.image.render.compositor was removed; use materials.m.inputs.tex.image.composition.compositor; see vork help upgrade");
        // A resource name is one step of the path whatever it holds, and the refusal stays one line.
        assert_eq!(refusal(serde_json::json!({"materials":{"a.b\nc":{"inputs":{"tex":{"image":{"render":{"compositor":["grade"]}}}}}},
            "nodes":[{"materials":[{"inputs":{"tex":{"image":{"render":{"compositor":["grade"]}}}}}]}]})),
            "materials[\"a.b\\nc\"].inputs.tex.image.render.compositor was removed; use materials[\"a.b\\nc\"].inputs.tex.image.composition.compositor; \
             nodes[0].materials[0].inputs.tex.image.render.compositor was removed; use nodes[0].materials[0].inputs.tex.image.composition.compositor; see vork help upgrade");
        assert_eq!(refusal(serde_json::json!({"composition":{"compositor":["grade"]},"nodes":[{"type":"camera"}]})),
            "scene.composition.space is required: write \"3d\" for a world (what a scene without a composition is) or \"2d\" for a canvas; see vork help upgrade");
        assert_eq!(refusal(serde_json::json!({"images":{"capture":{"render":{"compositor":["grade"]}}}})),
            "images.capture.render.compositor was removed; use images.capture.composition.compositor; see vork help upgrade");
        assert_eq!(refusal(serde_json::json!({"composition":{"space":"2d"},"render":{"compositor":["grade"]}})),
            "render.compositor was removed; use composition.compositor; see vork help upgrade");
        // A canvas runs its own graph: at the root, and on a canvas a world reads, whose document carries it.
        assert_eq!(scene(serde_json::json!({"composition":{"space":"2d","compositor":["grade"]},"nodes":[]}))?.compositor().len(), 1);
        let read = scene(serde_json::json!({"nodes":[{"type":"camera"},{"name":"sign","composition":{"space":"2d","compositor":["grade"]}}],
            "images":{"picture":{"render":{"root":"sign","resolution":[8,8]}}}}))?;
        assert!(read.compositor().is_empty());
        assert_eq!(read.composition_scene("sign", [8, 8])?.compositor().len(), 1);
        // That document keeps the enclosing scene's tracks on its graph's materials and on the images they read.
        let keyed = scene(serde_json::json!({"nodes":[{"type":"camera"},{"name":"sign","composition":{"space":"2d","compositor":["read"]}}],
            "materials":{"read":{"program":"grade","inputs":{"plate":{"image":"clip"}}}},
            "images":{"picture":{"render":{"root":"sign","resolution":[8,8]}},"clip":{"file":"clip.%04d.png"}},
            "animation":[{"tracks":[{"path":["materials","read","inputs","gain"],"keys":[[0,0.5],[1,1.0]]},
                {"path":["images","clip","time"],"keys":[[0,0.0],[1,2.0]]}]}]}))?;
        let part = keyed.composition_scene("sign", [8, 8])?;
        let paths: Vec<Vec<String>> = part.animation.iter().flat_map(|layer| layer.tracks.iter().flatten()).map(|track| track.path.clone()).collect();
        assert_eq!(paths, vec![vec!["materials", "read", "inputs", "gain"], vec!["images", "clip", "time"]]);
        // A canvas inside a canvas paints inline: it has no picture to run a graph on.
        assert!(refusal(serde_json::json!({"composition":{"space":"2d"},"nodes":[{"name":"child","composition":{"space":"2d","compositor":["grade"]}}]}))
            .contains("node \"child\": composition.compositor on a canvas inside a canvas is not built yet; it paints inline in its parent"));
        // A canvas's graph reads no world channel, AOV or rendered image.
        let canvas_graph = |inputs: serde_json::Value, extra: serde_json::Value| {
            let mut document = serde_json::json!({"composition":{"space":"2d","compositor":["read"]},"nodes":[],
                "materials":{"read":{"program":"grade","inputs":inputs}}});
            for (key, value) in extra.as_object().expect("fields") { document[key] = value.clone(); }
            refusal(document)
        };
        assert!(canvas_graph(serde_json::json!({"a":{"channel":"depth"}}), serde_json::json!({}))
            .contains("composition.compositor[0]: input a: channel depth is a 3D world's; a 2D canvas has color"));
        assert!(canvas_graph(serde_json::json!({"a":{"aov":"mask"}}), serde_json::json!({}))
            .contains("composition.compositor[0]: input a: AOV mask is a 3D world's; a 2D canvas has color"));
        assert!(canvas_graph(serde_json::json!({"a":{"image":"shot"}}), serde_json::json!({
            "nodes":[{"name":"world","composition":{"space":"3d"},"children":[{"type":"camera","name":"camera"}]}],
            "images":{"shot":{"render":{"root":"world"}}}}))
            .contains("composition.compositor[0]: input a: image shot is rendered; a 2D canvas's compositor reads image files"));
        // A canvas's picture, named or written inline where it is bound, has no graph of its own.
        let picture = serde_json::json!({"composition":{"compositor":["grade"]},"render":{"root":"sign","resolution":[8,8]}});
        let sign = serde_json::json!([{"type":"camera"},{"name":"sign","composition":{"space":"2d"}}]);
        assert!(refusal(serde_json::json!({"nodes":sign,"images":{"picture":picture}}))
            .contains("image picture: a canvas's picture has no compositor of its own; it is the canvas"));
        assert!(refusal(serde_json::json!({"nodes":sign,"materials":{"grade":{"program":"grade"},"read":{"program":"grade","inputs":{"picture":{"image":picture}}}}}))
            .contains(": a canvas's picture has no compositor of its own; it is the canvas"));
        Ok(())
    }
    #[test]
    fn invalid_and_unbuilt_text_paints_are_refused_in_one_line() {
        for field in ["fill", "stroke"] {
            for paint in ["", "not-a-color", "#1g2733", "ff0000", "rgb(1, 2 3)",
                "red; color: blue", "rgb(1 2 3) trailing", "currentColor", "lab(50% 0 0)",
                "color(display-p3 1 0 0)", "hsv(0, 100%, 100%)"] {
                let mut document = serde_json::json!({"version":2,"composition":{"space":"2d"},
                    "nodes":[{"name":"title","type":"text","content":"Hi","font":"a.ttf"}]});
                document["nodes"][0][field] = paint.into();
                let error = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap_err().root_cause().to_string();
                assert!(error.contains(&format!("node \"title\": {field}")), "{error}");
                assert_eq!(error.lines().count(), 1, "{error}");
                document["nodes"][0].as_object_mut().unwrap().remove(field);
                document["nodes"][0]["content"] = serde_json::json!([{"content":"Hi",field:paint}]);
                let error = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap_err().root_cause().to_string();
                assert!(error.contains(&format!("node \"title\": content[0].{field}")), "{error}");
                assert_eq!(error.lines().count(), 1, "{error}");
                document["nodes"][0]["content"] = "Hi".into();
                document["animation"] = serde_json::json!([{"tracks":[{"path":["nodes","title","glyphs",0,field],"keys":[[0,paint]]}]}]);
                let error = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap_err().root_cause().to_string();
                assert!(error.contains("track ["), "{error}");
                assert_eq!(error.lines().count(), 1, "{error}");
            }
        }
    }
    #[test]
    fn recursive_clip_through_mask_source_child_refuses() {
        // The source is reached through a reference, then its child refers back to it.
        // Tracking only nodes that follow references misses the active source `outer`.
        let document = serde_json::json!({
            "version": 2,
            "composition": {"space": "2d"},
            "render": {"resolution": [100, 100]},
            "nodes": [
                {"name": "picture", "type": "path", "d": "M0 0H100V100H0Z", "fill": "red", "clip-path": "outer"},
                {"name": "outer", "children": [
                    {"name": "child", "type": "path", "d": "M0 0H50V100H0Z", "clip-path": "outer"}
                ]}
            ]
        });
        let error = Scene::from_json(&serde_json::to_vec(&document).unwrap()).err()
            .expect("recursive clip must refuse before reaching the renderer");
        let error = format!("{error:#}");
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(error.contains("cycle") && (error.contains("node \"outer\"") || error.contains("node \"child\"")), "{error}");
    }

    fn load(tracks: serde_json::Value) -> std::result::Result<Scene, String> {
        let document = serde_json::json!({"version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"title","type":"text","content":"Hi there","font":"a.ttf"}],"animation":[{"tracks":tracks}]});
        Scene::from_json(&serde_json::to_vec(&document).unwrap()).map_err(|e| format!("{e:#}"))
    }
    #[test]
    fn node_time_on_a_canvas_node_is_read() {
        let document = serde_json::json!({"version":2,"composition":{"space":"2d"},
            "nodes":[{"name":"card","active":[0.5,2],"time":{"speed":2},"children":[{"name":"title","type":"text","content":"Hi","font":"a.ttf",
                "animation":[{"tracks":[{"path":["glyphs",0,"opacity"],"keys":[[0,0],[1,1]]}]}]}],
                "animation":[{"tracks":[{"path":["children","title","active"],"keys":[[0,false],[0.25,true]]}]}]}]});
        let scene = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap();
        let owned = scene.node_owned_layers().unwrap();
        assert_eq!(owned.len(), 2);
        assert_eq!(owned[1].targets[0], TrackTarget::Text { node: "title".into(), unit: TextUnit::Glyphs, index: 0, property: UnitProperty::Opacity });
        assert!(load(serde_json::json!([{"path":["nodes","title","active"],"keys":[[0,true]]}])).is_ok());
    }
    #[test]
    fn glyph_word_and_line_tracks_resolve_and_refuse_in_one_line() {
        let scene = load(serde_json::json!([
            {"path":["nodes","title","glyphs",3,"opacity"],"keys":[[0,0],[1,1]]},
            {"path":["nodes","title","words",1,"rotation"],"keys":[[0,{"euler":[0,0,1]}]]},
            {"path":["nodes","title","lines",0,"translation"],"keys":[[0,[0,10,0]]]}])).unwrap();
        let targets = scene.track_targets().unwrap();
        assert_eq!(targets[0][0], TrackTarget::Text { node: "title".into(), unit: TextUnit::Glyphs, index: 3, property: UnitProperty::Opacity });
        assert_eq!(targets[0][0].to_string(), r#"["nodes","title","glyphs","3","opacity"]"#);
        // "Hi there" has 8 glyphs (the space is one) and 2 words.
        assert!(load(serde_json::json!([{"path":["nodes","title","glyphs",8,"opacity"],"keys":[[0,1]]}])).unwrap_err().ends_with(
            r#"track ["nodes","title","glyphs","8","opacity"]: node "title" has 8 glyphs; index 8 is past the end"#));
        assert!(load(serde_json::json!([{"path":["nodes","title","words",2,"scale"],"keys":[[0,[1,1,1]]]}])).unwrap_err().ends_with(
            r#"track ["nodes","title","words","2","scale"]: node "title" has 2 words; index 2 is past the end"#));
        assert!(load(serde_json::json!([{"path":["nodes","title","glyphs",0,"fill"],"keys":[[0,1]]}])).unwrap_err().ends_with(
            r#"track ["nodes","title","glyphs","0","fill"]: value 0 is 1; expected arrays of 4 numbers (r, g, b, a from 0 to 1), or CSS colour strings"#));
        // Paint and transform-origin strings interpolate by their CSS value types.
        let scene = load(serde_json::json!([
            {"path":["nodes","title","glyphs",0,"fill"],"keys":[[0,[1,0,0,1]],[1,[0,0,1,0.5]]]},
            {"path":["nodes","title","words",1,"stroke"],"keys":[[0,"red"],[1,"none"]]},
            {"path":["nodes","title","lines",0,"transform-origin"],"keys":[[0,"left bottom"],[1,"25% 4px"]]}])).unwrap();
        assert_eq!(scene.track_targets().unwrap()[0][2],
            TrackTarget::Text { node: "title".into(), unit: TextUnit::Lines, index: 0, property: UnitProperty::TransformOrigin });
        assert!(load(serde_json::json!([{"path":["nodes","title","glyphs",0,"transform-origin"],"keys":[[0,"middle"]]}]))
            .unwrap_err().ends_with(r#"track ["nodes","title","glyphs","0","transform-origin"]: transform-origin "middle": "middle" is not left, center, right, top, bottom, a percentage or a length in px"#));
    }
    #[test]
    fn transform_origin_reads_css() {
        assert_eq!(transform_origin("center").unwrap(), [(0.5, 0.), (0.5, 0.)]);
        assert_eq!(transform_origin("bottom").unwrap(), [(0.5, 0.), (1., 0.)]);
        assert_eq!(transform_origin("top left").unwrap(), [(0., 0.), (0., 0.)]);
        assert_eq!(transform_origin("right 10px").unwrap(), [(1., 0.), (0., 10.)]);
        assert_eq!(transform_origin("25% 0").unwrap(), [(0.25, 0.), (0., 0.)]);
        assert!(transform_origin("left right").is_err());
        assert!(transform_origin("1px 2px 3px").unwrap_err().to_string().contains("z offset"));
    }
    #[test]
    fn runs_collapse_white_space_across_their_boundaries() {
        let content: Content = serde_json::from_value(serde_json::json!([{"content":" Big  "},{"content":" small\n"},{"content":"end "}])).unwrap();
        let runs = laid_out_runs(&content, WhiteSpace::Normal);
        assert_eq!(runs.iter().map(|(c, _)| *c).collect::<String>(), "Big small end");
        // The space after "Big" stands for white space that starts in run 0.
        assert_eq!(runs[3], (' ', 0));
        assert_eq!(runs[9], (' ', 1));
        assert_eq!(laid_out_text(&content, WhiteSpace::PreLine), "Big small\nend");
    }
    #[test]
    fn paths_paints_and_node_fields_load_on_a_canvas() {
        let document = serde_json::json!({"version":2,"composition":{"space":"2d","viewBox":[0,0,100,50],"preserveAspectRatio":"xMinYMax slice"},
            "render":{"resolution":[200,200],"background":"#102030"},
            "images":{"photo":{"file":"photo.png"}},
            "nodes":[
                {"name":"disc","type":"path","d":"M0 0H10V10Z","fill":"radial-gradient(circle, red, blue)","stroke":"white","stroke-width":2,
                 "stroke-dasharray":[4,2],"opacity":0.5,"filter":"blur(4px)","clip-path":"shape"},
                {"name":"shape","type":"path","d":"M0 0H5V5Z"},
                {"name":"pic","type":"path","d":"M0 0H5V5Z","fill":{"image":"photo","fit":"cover"},"mask":"shape"}],
            "animation":[{"tracks":[
                {"path":["nodes","disc","opacity"],"keys":[[0,0],[1,1]]},
                {"path":["nodes","disc","stroke"],"keys":[[0,"red"],[1,"#00f"]]},
                {"path":["nodes","disc","fill"],"keys":[[0,[1,0,0,1]],[1,[0,0,1,0.5]]]}]}]});
        let scene = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap();
        assert_eq!(scene.canvas_view_box(), [0., 0., 100., 50.]);
        let targets = scene.track_targets().unwrap();
        assert_eq!(targets[0][0], TrackTarget::Node { node: "disc".into(), property: NodeProperty::Opacity });
        assert_eq!(targets[0][1], TrackTarget::Node { node: "disc".into(), property: NodeProperty::Stroke });
        // CSS colours and four-number colours both interpolate.
        assert!(matches!(scene.animation[0].tracks.as_ref().unwrap()[1].interpolation, Interpolation::Linear));
        assert!(matches!(scene.animation[0].tracks.as_ref().unwrap()[2].interpolation, Interpolation::Linear));
        let aspect = AspectRatio::parse(Some("xMinYMax slice")).unwrap();
        // 100x50 into 200x200, sliced: scale 4; the box is 400 wide and spills right (xMin).
        assert_eq!(aspect.placement([0., 0., 100., 50.], [200, 200]), ([4., 4.], [0., 0.]));
        let meet = AspectRatio::parse(None).unwrap();
        assert_eq!(meet.placement([0., 0., 100., 50.], [200, 200]), ([2., 2.], [0., 50.]));
    }
    #[test]
    fn two_d_refusals_name_the_field_in_one_line() {
        let canvas = |nodes: serde_json::Value| serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":nodes});
        for (document, line) in [
            (canvas(serde_json::json!([{"type":"path","clip-path":"nobody"}])), "node \"\": clip-path names \"nobody\", which is no node in the canvas"),
            (canvas(serde_json::json!([{"type":"path","fill":{"material":"m"}}])), "a program paint is not built yet"),
            (canvas(serde_json::json!([{"type":"path","stroke-linejoin":"arcs"}])), "scene.nodes[0].stroke-linejoin: \"arcs\" (SVG 2) is not built yet; use miter, round or bevel"),
            (serde_json::json!({"version":2,"composition":{"space":"2d"},"render":{"background":"environment"},"nodes":[]}), "render.background: a 2D canvas has no environment; write \"transparent\" or a CSS colour"),
            (serde_json::json!({"version":2,"composition":{"space":"2d","preserveAspectRatio":"xMidYMid zoom"},"nodes":[]}), "scene.composition.preserveAspectRatio \"xMidYMid zoom\": write none, or x(Min|Mid|Max)Y(Min|Mid|Max) optionally followed by meet or slice"),
        ] {
            let error = Scene::from_json(&serde_json::to_vec(&document).unwrap()).map_err(|e| format!("{e:#}")).unwrap_err();
            assert!(error.contains(line), "{error}");
        }
    }
}
