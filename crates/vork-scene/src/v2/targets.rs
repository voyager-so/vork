//! Track paths: arrays of segments resolved once against the scene into typed targets.
//! Validation and the renderer share this one table of keyable properties.
use super::*;
use anyhow::Context;
use std::collections::BTreeSet;

/// What a track animates.
#[derive(Clone, Debug, PartialEq)]
pub enum TrackTarget {
    /// A property of an authored node.
    Node {
        node: String,
        property: NodeProperty,
    },
    /// A property of a node imported by asset node `asset`: its glTF name or `#index`.
    Imported {
        asset: String,
        node: String,
        property: NodeProperty,
    },
    /// An input of a material.
    Material {
        material: String,
        input: String,
    },
    /// The weight of a named animation layer.
    LayerWeight {
        layer: String,
    },
    ImageTime { image: String },
    Render(RenderProperty),
    /// A field of one glyph, word or line of a text node (END-STATE-API §7).
    Text {
        node: String,
        unit: TextUnit,
        index: usize,
        property: UnitProperty,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub enum NodeProperty {
    Translation,
    Rotation,
    Scale,
    Visibility(Ray),
    MorphWeight(String),
    Color,
    Temperature,
    Energy,
    Angle,
    Radius,
    SpotBlend,
    SpotSize,
    Width,
    Height,
    /// A boolean `active`, keyed STEP by the parent or the scene.
    Active,
    /// `time.speed`, keyed in the owner's seconds and integrated.
    Speed,
    /// Linear amplitude of this node's subtree mix.
    AudioGain,
    /// 2D: CSS `opacity`.
    Opacity,
    /// 2D: a path or text node's `fill` paint.
    Fill,
    /// 2D: a path or text node's `stroke` paint.
    Stroke,
    /// 2D: a path's `stroke-width`.
    StrokeWidth,
    /// 2D: a path's `stroke-dashoffset`.
    StrokeDashoffset,
    /// 2D: a path's `fill-opacity`.
    FillOpacity,
    /// 2D: a path's `stroke-opacity`.
    StrokeOpacity,
    /// A CSS/SVG field, described by the shared style table.
    Style(StyleProperty),
}
/// CSS/SVG interpolation categories. Discrete values switch at half progress (STEP still holds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StyleInterpolation { Number, FontWeight, Origin, Discrete, Path, Filter, Paint, Repeatable, Variations }
/// One table owns field names, applicability and interpolation for canvas style tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StyleProperty(pub usize);
const STYLE_PROPERTIES: &[(&str, &str, StyleInterpolation)] = &[
    ("d", "path", StyleInterpolation::Path),
    ("filter", "node", StyleInterpolation::Filter),
    ("clip-path", "node", StyleInterpolation::Discrete),
    ("mask", "node", StyleInterpolation::Discrete),
    ("font", "text", StyleInterpolation::Discrete),
    ("fill-rule", "path", StyleInterpolation::Discrete),
    ("stroke-linecap", "path", StyleInterpolation::Discrete),
    ("stroke-linejoin", "path", StyleInterpolation::Discrete),
    ("stroke-miterlimit", "path", StyleInterpolation::Number),
    ("stroke-dasharray", "path", StyleInterpolation::Repeatable),
    ("pathLength", "path", StyleInterpolation::Number),
    ("paint-order", "path", StyleInterpolation::Discrete),
    ("content", "text", StyleInterpolation::Discrete),
    ("font-size", "text", StyleInterpolation::Number),
    ("font-weight", "text", StyleInterpolation::FontWeight),
    ("font-style", "text", StyleInterpolation::Discrete),
    ("font-variation-settings", "text", StyleInterpolation::Variations),
    ("font-feature-settings", "text", StyleInterpolation::Discrete),
    ("letter-spacing", "text", StyleInterpolation::Number),
    ("word-spacing", "text", StyleInterpolation::Number),
    ("line-height", "text", StyleInterpolation::Number),
    ("text-align", "text", StyleInterpolation::Discrete),
    ("direction", "text", StyleInterpolation::Discrete),
    ("white-space", "text", StyleInterpolation::Discrete),
    ("word-break", "text", StyleInterpolation::Discrete),
    ("line-break", "text", StyleInterpolation::Discrete),
    ("width", "text", StyleInterpolation::Number),
];
impl StyleProperty {
    pub fn parse(name: &str) -> Option<Self> { STYLE_PROPERTIES.iter().position(|p| p.0 == name).map(Self) }
    pub fn name(self) -> &'static str { STYLE_PROPERTIES[self.0].0 }
    pub fn interpolation(self) -> StyleInterpolation { STYLE_PROPERTIES[self.0].2 }
    fn applies(self, node: &Node) -> bool {
        match STYLE_PROPERTIES[self.0].1 {
            "node" => true,
            "path" => matches!(node.kind, NodeKind::Path { .. }),
            _ => matches!(node.kind, NodeKind::Text { .. }),
        }
    }
}
/// A node visibility flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ray {
    All,
    Camera,
    Shadow,
    Diffuse,
    Glossy,
    Transmission,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderProperty {
    Camera,
    Background,
    MotionBlurShutter,
}

/// Node property segments: (segments, property name for errors).
const NODE_PROPERTIES: &[&str] = &[
    "visible",
    "translation",
    "rotation",
    "scale",
    "visible_camera",
    "visible_shadow",
    "visible_diffuse",
    "visible_glossy",
    "visible_transmission",
    "morph_weights",
    "color",
    "temperature",
    "energy",
    "angle",
    "radius",
    "spot_size",
    "spot_blend",
    "size",
    "size_y",
    "active",
    "time",
    "audio",
    "opacity",
    "fill",
    "stroke",
    "stroke-width",
    "stroke-dashoffset",
    "fill-opacity",
    "stroke-opacity",
];
const RAYS: &[(&str, Ray)] = &[
    ("visible", Ray::All),
    ("visible_camera", Ray::Camera),
    ("visible_shadow", Ray::Shadow),
    ("visible_diffuse", Ray::Diffuse),
    ("visible_glossy", Ray::Glossy),
    ("visible_transmission", Ray::Transmission),
];

impl NodeProperty {
    /// The path segments naming this property.
    fn segments(&self) -> Vec<String> {
        use NodeProperty as P;
        let one = |s: &str| vec![s.to_string()];
        match self {
            P::Style(p) => one(p.name()),
            P::Translation => one("translation"),
            P::Rotation => one("rotation"),
            P::Scale => one("scale"),
            P::Visibility(ray) => one(RAYS.iter().find(|(_, r)| r == ray).unwrap().0),
            P::MorphWeight(name) => vec!["morph_weights".into(), name.clone()],
            P::Color => one("color"),
            P::Temperature => one("temperature"),
            P::Energy => one("energy"),
            P::Angle => one("angle"),
            P::Radius => one("radius"),
            P::SpotBlend => one("spot_blend"),
            P::SpotSize => one("spot_size"),
            P::Width => one("size"),
            P::Height => one("size_y"),
            P::Active => one("active"),
            P::AudioGain => vec!["audio".into(), "gain".into()],
            P::Speed => vec!["time".into(), "speed".into()],
            P::Opacity => one("opacity"),
            P::Fill => one("fill"),
            P::Stroke => one("stroke"),
            P::StrokeWidth => one("stroke-width"),
            P::StrokeDashoffset => one("stroke-dashoffset"),
            P::FillOpacity => one("fill-opacity"),
            P::StrokeOpacity => one("stroke-opacity"),
        }
    }
}
/// A resolved target prints as its track path, e.g. ["nodes","cam","rotation"].
impl std::fmt::Display for TrackTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut path: Vec<String> = match self {
            Self::Node { node, property } => {
                let mut p = vec!["nodes".into(), node.clone()];
                p.extend(property.segments());
                p
            }
            Self::Imported {
                asset,
                node,
                property,
            } => {
                let mut p = vec!["nodes".into(), asset.clone(), "nodes".into(), node.clone()];
                p.extend(property.segments());
                p
            }
            Self::Material { material, input } => {
                vec![
                    "materials".into(),
                    material.clone(),
                    "inputs".into(),
                    input.clone(),
                ]
            }
            Self::ImageTime { image } => vec!["images".into(), image.clone(), "time".into()],
            Self::LayerWeight { layer } => vec!["animation".into(), layer.clone(), "weight".into()],
            Self::Render(_) => vec!["render".into()],
            Self::Text { node, unit, index, property } => {
                vec!["nodes".into(), node.clone(), unit.segment().into(), index.to_string(), property.segment().into()]
            }
        };
        if let Self::Render(property) = self {
            use RenderProperty as R;
            path.extend(
                match property {
                    R::Camera => &["camera"][..],
                    R::Background => &["background"],
                    R::MotionBlurShutter => &["motion_blur", "shutter"],
                }
                .iter()
                .map(|s| s.to_string()),
            );
        }
        f.write_str(&display_path(&path))
    }
}

/// A track path as JSON, for error messages.
pub fn display_path(path: &[String]) -> String {
    serde_json::to_string(path).unwrap_or_default()
}

/// Segment `index` of `path` or an error naming what it should be.
fn segment<'a>(path: &'a [String], index: usize, what: &str) -> Result<&'a str> {
    path.get(index).map(String::as_str).with_context(|| {
        format!(
            "track {}: segment {index} is missing; expected {what}",
            display_path(path)
        )
    })
}
fn end(path: &[String], length: usize) -> Result<()> {
    ensure!(
        path.len() == length,
        "track {}: segment {length} {:?} is past the end of the property",
        display_path(path),
        path.get(length).map(String::as_str).unwrap_or_default()
    );
    Ok(())
}
fn bad_segment<'a>(
    path: &[String],
    index: usize,
    kind: &str,
    valid: impl IntoIterator<Item = &'a str>,
) -> anyhow::Error {
    anyhow::anyhow!(
        "track {}: segment {index}: {}",
        display_path(path),
        diagnostics::unknown(kind, &path[index], valid)
    )
}

impl Scene {
    /// Images referenced by a node's paint/content keys, including ancestor-owned tracks.
    pub(super) fn keyed_images(&self, node: &Node) -> BTreeSet<String> {
        fn images(value: &serde_json::Value, out: &mut BTreeSet<String>) {
            match value {
                serde_json::Value::Object(fields) => {
                    if let Some(name) = fields.get("image").and_then(serde_json::Value::as_str) { out.insert(name.into()); }
                    for value in fields.values() { images(value, out); }
                }
                serde_json::Value::Array(values) => for value in values { images(value, out); },
                _ => {}
            }
        }
        let mut out = BTreeSet::new();
        let mut collect = |track: &Track, target: &TrackTarget| {
            let matches = match target {
                TrackTarget::Node { node: name, property } => name == &node.name &&
                    (matches!(property, NodeProperty::Fill | NodeProperty::Stroke) || matches!(property, NodeProperty::Style(p) if p.name() == "content")),
                TrackTarget::Text { node: name, property, .. } => name == &node.name && matches!(property, UnitProperty::Fill | UnitProperty::Stroke),
                _ => false,
            };
            if matches { if let Values::Inline(values) = &track.values { for value in values { images(value, &mut out); } } }
        };
        if let Ok(targets) = self.track_targets() {
            for (layer, targets) in self.animation.iter().zip(targets) {
                for (track, target) in layer.tracks.iter().flatten().zip(targets) { collect(track, &target); }
            }
        }
        if let Ok(layers) = self.node_owned_layers() {
            for owned in layers {
                for (track, target) in owned.layer.tracks.iter().flatten().zip(owned.targets) { collect(track, &target); }
            }
        }
        out
    }
    /// Every track's target, per layer in track order (resolved once).
    pub fn track_targets(&self) -> Result<Vec<Vec<TrackTarget>>> {
        self.track_targets_in(&self.nodes_by_name())
    }
    /// [`Self::track_targets`] against a name index the caller already holds.
    pub(super) fn track_targets_in(&self, nodes: &BTreeMap<&str, &Node>) -> Result<Vec<Vec<TrackTarget>>> {
        self.animation
            .iter()
            .map(|layer| {
                layer
                    .tracks
                    .iter()
                    .flatten()
                    .map(|track| self.resolve_target(&track.path, nodes))
                    .collect()
            })
            .collect()
    }
    pub(super) fn nodes_by_name(&self) -> BTreeMap<&str, &Node> {
        fn visit<'a>(nodes: &'a [Node], out: &mut BTreeMap<&'a str, &'a Node>) {
            for node in nodes {
                if !node.name.is_empty() {
                    out.insert(&node.name, node);
                }
                visit(&node.children, out);
            }
        }
        let mut out = BTreeMap::new();
        visit(&self.nodes, &mut out);
        out
    }
    fn resolve_target(
        &self,
        path: &[String],
        nodes: &BTreeMap<&str, &Node>,
    ) -> Result<TrackTarget> {
        let area = segment(path, 0, "nodes, materials, animation or render")?;
        match area {
            "nodes" => {
                let name = segment(path, 1, "a node name")?;
                let node = nodes
                    .get(name)
                    .copied()
                    .ok_or_else(|| bad_segment(path, 1, "node", nodes.keys().copied()))?;
                if let NodeKind::Asset { .. } = &node.kind
                    && path.get(2).map(String::as_str) == Some("nodes")
                {
                    // An imported node, as in node.nodes: its glTF name (may contain
                    // '.' or match a property name) or #index, a positional selector.
                    segment(path, 3, "an imported node name or #index")?;
                    let property = node_property(path, 4)?;
                    ensure!(
                        matches!(
                            property,
                            NodeProperty::Translation
                                | NodeProperty::Rotation
                                | NodeProperty::Scale
                                | NodeProperty::MorphWeight(_)
                                | NodeProperty::Visibility(Ray::All)
                        ),
                        "track {}: imported nodes key translation, rotation, scale, morph_weights and visible",
                        display_path(path)
                    );
                    return Ok(TrackTarget::Imported {
                        asset: name.into(),
                        node: path[3].clone(),
                        property,
                    });
                }
                if let (NodeKind::Text { .. }, Some(unit)) = (&node.kind, path.get(2).and_then(|s| TextUnit::parse(s))) {
                    // The index is checked against the node's pieces by `validate_targets`, which counts them once per node.
                    let index: usize = segment(path, 3, "a zero-based index")?.parse().map_err(|_| {
                        anyhow::anyhow!("track {}: segment 3 {:?} is not a zero-based index", display_path(path), path[3])
                    })?;
                    const FIELDS: [&str; 7] = ["translation", "rotation", "scale", "opacity", "fill", "stroke", "transform-origin"];
                    let property = UnitProperty::parse(segment(path, 4, "translation, rotation, scale, opacity, fill, stroke or transform-origin")?)
                        .ok_or_else(|| bad_segment(path, 4, "glyph, word or line field", FIELDS))?;
                    end(path, 5)?;
                    return Ok(TrackTarget::Text { node: name.into(), unit, index, property });
                }
                let property = node_property(path, 2)?;
                self.check_node_property(path, node, &property)?;
                Ok(TrackTarget::Node {
                    node: name.into(),
                    property,
                })
            }
            "images" => {
                let name = segment(path, 1, "an image name")?;
                ensure!(self.images.contains_key(name), "track {}: unknown image {name}", display_path(path));
                ensure!(segment(path, 2, "time")? == "time", "images key time only");
                end(path, 3)?;
                Ok(TrackTarget::ImageTime { image: name.into() })
            }
            "materials" => {
                let name = segment(path, 1, "a material name")?;
                let material = self.materials.get(name).ok_or_else(|| {
                    bad_segment(
                        path,
                        1,
                        "material",
                        self.materials.keys().map(String::as_str),
                    )
                })?;
                if segment(path, 2, "\"inputs\"")? != "inputs" {
                    return Err(bad_segment(path, 2, "material field", ["inputs"]));
                }
                let input = segment(path, 3, "an input name")?;
                end(path, 4)?;
                anyhow::ensure!(!matches!(material.inputs.get(input), Some(Input::Texture(_))), "track {}: image handles are not numeric inputs; animate a program input instead", display_path(path));
                if material.program == "std/openpbr" {
                    validate::validate_openpbr_input(input)
                        .with_context(|| format!("track {}: segment 3", display_path(path)))?;
                }
                Ok(TrackTarget::Material {
                    material: name.into(),
                    input: input.into(),
                })
            }
            "animation" => {
                let name = segment(path, 1, "a layer name")?;
                let layers = self.animation.iter().filter_map(|l| l.name.as_deref());
                if !self
                    .animation
                    .iter()
                    .any(|l| l.name.as_deref() == Some(name))
                {
                    return Err(bad_segment(path, 1, "animation layer", layers));
                }
                if segment(path, 2, "\"weight\"")? != "weight" {
                    return Err(bad_segment(path, 2, "layer property", ["weight"]));
                }
                end(path, 3)?;
                Ok(TrackTarget::LayerWeight { layer: name.into() })
            }
            "render" => {
                let property = match segment(path, 1, "a render property")? {
                    "camera" => RenderProperty::Camera,
                    "background" => {
                        ensure!(matches!(self.render.background, Background::Color(_)) || (self.is_canvas() && matches!(self.render.background,Background::Css(_))), "background track requires a static colour background; environment/transparent are structural modes");
                        RenderProperty::Background
                    },
                    "motion_blur" => {
                        ensure!(segment(path, 2, "shutter")? == "shutter", "motion_blur keys shutter");
                        ensure!(self.render.motion_blur.is_some(), "motion_blur is null; set a static motion_blur to key it");
                        end(path, 3)?;
                        return Ok(TrackTarget::Render(RenderProperty::MotionBlurShutter));
                    }
                    _ => {
                        return Err(bad_segment(
                            path,
                            1,
                            "keyable render property",
                            ["camera", "background", "motion_blur"],
                        ));
                    }
                };
                end(path, 2)?;
                Ok(TrackTarget::Render(property))
            }
            _ => Err(bad_segment(
                path,
                0,
                "track area",
                ["nodes", "images", "materials", "animation", "render"],
            )),
        }
    }
    /// Whether `property` exists on this node and has a static value to key from.
    fn check_node_property(
        &self,
        path: &[String],
        node: &Node,
        property: &NodeProperty,
    ) -> Result<()> {
        use NodeProperty as P;
        let applies = match (&node.kind, property) {
            (_, P::Translation | P::Rotation | P::Scale | P::Visibility(_)) => true,
            (_, P::Active) => {
                ensure!(matches!(node.active, Active::On(_)), "track {}: node {:?}: active is a window, a constant; key a boolean active instead", display_path(path), node.name);
                true
            }
            (_, P::AudioGain) => node.audio.is_some(),
            (_, P::Speed) => {
                let time = node.time.as_ref().with_context(|| format!("track {}: node {:?}: time.speed keys a node with a time", display_path(path), node.name))?;
                ensure!(!time.reverse(), "track {}: a negative speed is a constant; for a changing or reversing speed write a clock program", display_path(path));
                true
            }
            (NodeKind::Mesh { geometry, .. }, P::MorphWeight(name)) => {
                let morphs = &self.geometry[geometry].morphs;
                ensure!(
                    morphs.contains_key(name),
                    "track {}: segment 3: {}",
                    display_path(path),
                    diagnostics::unknown("morph", name, morphs.keys().map(String::as_str))
                );
                true
            }
            (NodeKind::Light { .. }, P::Color | P::Energy) => true,
            (NodeKind::Light { temperature, .. }, P::Temperature) => {
                ensure!(
                    temperature.is_some(),
                    "track {}: temperature keys a light with a temperature",
                    display_path(path)
                );
                true
            }
            (NodeKind::Light { shape, .. }, P::Angle) => matches!(shape, LightShape::Sun),
            (NodeKind::Light { shape, .. }, P::Radius) => matches!(shape, LightShape::Point | LightShape::Spot),
            (NodeKind::Light { shape, .. }, P::SpotBlend | P::SpotSize) => matches!(shape, LightShape::Spot),
            (NodeKind::Light { shape, .. }, P::Width | P::Height) => {
                matches!(shape, LightShape::Rectangle | LightShape::Ellipse)
            }
            (_, P::Style(p)) => self.is_canvas() && p.applies(node),
            (_, P::Opacity) => {
                ensure!(self.is_canvas(), "track {}: opacity is a 2D canvas field", display_path(path));
                true
            }
            (NodeKind::Text { .. }, P::Fill | P::Stroke | P::StrokeWidth) => true,
            (NodeKind::Path { .. }, P::Fill | P::Stroke | P::StrokeWidth | P::StrokeDashoffset | P::FillOpacity | P::StrokeOpacity) => true,
            _ => false,
        };
        ensure!(
            applies,
            "track {}: segment 2: {:?} does not apply to node {:?}",
            display_path(path),
            path[2],
            node.name
        );
        Ok(())
    }

    pub(super) fn validate_targets(&self, nodes: &BTreeMap<&str, &Node>) -> Result<()> {
        let camera = |name: &str| validate::validate_camera(name, nodes);
        // A glyph or word index lies within its text node's pieces, counted once per node and unit. Lines exist after
        // wrapping; the renderer checks their count once the text is laid out.
        let owned = self.node_owned_layers_in(nodes)?;
        let targets = self.track_targets_in(nodes)?;
        let dynamic_pieces: BTreeSet<(&str, TextUnit)> = targets.iter().flatten()
            .chain(owned.iter().flat_map(|layer| &layer.targets))
            .flat_map(|target| [TextUnit::Glyphs, TextUnit::Words].into_iter()
                .filter_map(move |unit| target.changes_text_pieces(unit).map(|node| (node, unit))))
            .collect();
        let mut piece_counts: BTreeMap<(&str, TextUnit), usize> = BTreeMap::new();
        let mut piece_in_range = |path: &[String], target: &TrackTarget| -> Result<()> {
            let TrackTarget::Text { node, unit, index, .. } = target else { return Ok(()) };
            if *unit == TextUnit::Lines || dynamic_pieces.contains(&(node.as_str(), *unit)) { return Ok(()); }
            let Some((&name, &&Node { kind: NodeKind::Text { ref content, white_space, .. }, .. })) = nodes.get_key_value(node.as_str()) else { return Ok(()) };
            let count = *piece_counts.entry((name, *unit)).or_insert_with(|| text::unit_ranges(&text::laid_out_text(content, white_space), *unit).len());
            ensure!(*index < count, "track {}: node {name:?} has {count} {}; index {index} is past the end", display_path(path), unit.segment());
            Ok(())
        };
        let mut layers = BTreeMap::new();
        for (index, layer) in self.animation.iter().enumerate() {
            if let Some(name) = &layer.name {
                validate_name(name)?;
                ensure!(
                    layers.insert(name.as_str(), index).is_none(),
                    "duplicate animation layer name {name:?}; layer names must be unique"
                );
            }
        }
        for (layer_index, layer) in self.animation.iter().enumerate() {
            if let Some(clip) = &layer.clip {
                ensure!(
                    clip.name.is_none() || clip.index.is_none(),
                    "clip requires at most one of name or index"
                );
                let target = &clip.target;
                let node = nodes.get(target.as_str()).with_context(|| {
                    diagnostics::unknown("clip target node", target, nodes.keys().copied())
                })?;
                // External clips retarget onto an asset's skin joints by name.
                ensure!(
                    matches!(node.kind, NodeKind::Asset { .. }),
                    "clip target {target:?} must name an asset node"
                );
            }
            for track in layer.tracks.iter().flatten() {
                let target = self.resolve_target(&track.path, nodes)?;
                piece_in_range(&track.path, &target)?;
                self.validate_keys(track, &target, layer.blend, nodes)?;
                match target {
                    TrackTarget::LayerWeight { layer } => ensure!(
                        layer_index < layers[layer.as_str()],
                        "track {}: layer weight must be keyed from a lower layer (earlier in animation)",
                        display_path(&track.path)
                    ),
                    TrackTarget::Render(RenderProperty::Camera) => {
                        if let Values::Inline(values) = &track.values {
                            for value in values {
                                camera(
                                    value.as_str().context(
                                        "render camera track values must be camera names",
                                    )?,
                                )?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // A3: a property has one owner. Several layers of one owner blend; a scene layer and a
        // node's layer, or two nodes' layers, on one property are refused.
        let mut owners: BTreeMap<String, BTreeSet<Option<Vec<usize>>>> = BTreeMap::new();
        // A clock reads its active and time.speed keys directly, so they sit in a layer that
        // neither retimes nor weighs them.
        let weight_keyed: BTreeSet<&str> = self.animation.iter().flat_map(|l| l.tracks.iter().flatten())
            .filter_map(|t| match t.path.as_slice() { [a, name, w] if a == "animation" && w == "weight" => Some(name.as_str()), _ => None }).collect();
        let plain = |track: &Track, weight: f32, blend: Blend, retimed: bool, keyed_weight: bool| -> Result<()> {
            ensure!(weight == 1. && blend == Blend::Replace && !retimed && !keyed_weight,
                "track {}: active and time.speed are keyed in a plain layer: weight 1, replace, no offset, speed, loop or keyed weight", display_path(&track.path));
            Ok(())
        };
        for layer in &self.animation {
            for track in layer.tracks.iter().flatten() {
                let target = self.resolve_target(&track.path, nodes)?;
                if target.is_clock() {
                    plain(track, layer.weight, layer.blend, layer.offset != 0. || layer.speed != 1. || layer.looping,
                        layer.name.as_deref().is_some_and(|name| weight_keyed.contains(name)))?;
                }
                owners.entry(target.to_string()).or_default().insert(None);
            }
            if let Some(clip) = &layer.clip { owners.entry(format!("clip {:?}", clip.target)).or_default().insert(None); }
        }
        for layer in &owned {
            for (track, target) in layer.layer.tracks.iter().flatten().zip(&layer.targets) {
                piece_in_range(&track.path, target).map_err(|error| anyhow::anyhow!("node {:?}: {error:#}", layer.owner.name))?;
                self.validate_keys(track, target, layer.layer.blend, nodes)
                    .map_err(|error| anyhow::anyhow!("node {:?}: {error:#}", layer.owner.name))?;
                if target.is_clock() {
                    let l = layer.layer;
                    plain(track, l.weight, l.blend, l.offset != 0. || l.speed != 1. || l.looping, false)
                        .map_err(|error| anyhow::anyhow!("node {:?}: {error:#}", layer.owner.name))?;
                }
                owners.entry(target.to_string()).or_default().insert(Some(layer.path.clone()));
            }
            if let Some(clip) = &layer.layer.clip { owners.entry(format!("clip {:?}", clip.target)).or_default().insert(Some(layer.path.clone())); }
        }
        for (target, set) in &owners {
            ensure!(set.len() <= 1, "track {target}: a property has one owner; it is keyed by the scene's animation and a node's, or by two nodes: key it from one");
        }
        // A6: a scene-owned time.speed curve reads the parent's traversal, which exists only
        // when every clock above the node has a constant nonzero speed.
        let keyed_speed: BTreeSet<&str> = owners.keys().filter_map(|t| t.strip_prefix("[\"nodes\",\"")).filter_map(|rest| rest.split_once("\",\"time\",\"speed\"]").map(|(name, _)| name)).collect();
        for layer in &self.animation {
            for track in layer.tracks.iter().flatten() {
                let target = self.resolve_target(&track.path, nodes)?;
                let TrackTarget::Node { node, property: NodeProperty::Speed } = &target else { continue };
                let mut ancestors = Vec::new();
                fn path_to<'a>(list: &'a [Node], name: &str, out: &mut Vec<&'a Node>) -> bool {
                    for node in list {
                        out.push(node);
                        if node.name == name || path_to(&node.children, name, out) { return true; }
                        out.pop();
                    }
                    false
                }
                path_to(&self.nodes, node, &mut ancestors);
                ancestors.pop();
                for ancestor in ancestors {
                    let keyed = keyed_speed.contains(ancestor.name.as_str());
                    let zero = ancestor.time.as_ref().is_some_and(|t| t.speed.as_f64() == 0.);
                    ensure!(!keyed && !zero, "track {}: node {:?}: time.speed is keyed by the scene under {:?}, whose speed is {}; key it from the parent", display_path(&track.path), node, ancestor.name, if keyed { "keyed" } else { "zero" });
                }
            }
        }
        Ok(())
    }
}

/// The value a track target takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackValue {
    /// Numbers with this many components per key.
    Numeric(usize),
    Bool,
    /// A camera node name.
    Camera,
    Token,
    Style,
}
impl TrackTarget {
    /// The node whose piece count can change under this style track.
    /// Static counts remain useful only when no track changes the relevant layout.
    pub fn changes_text_pieces(&self, unit: TextUnit) -> Option<&str> {
        match self {
            Self::Node { node, property: NodeProperty::Style(p) }
                if matches!(p.name(), "content" | "white-space")
                    || (unit == TextUnit::Lines && STYLE_PROPERTIES[p.0].1 == "text") => Some(node),
            _ => None,
        }
    }

    pub fn value(&self) -> TrackValue {
        use NodeProperty as P;
        match self {
            Self::Node { property, .. } | Self::Imported { property, .. } => match property {
                P::Translation | P::Scale | P::Color => TrackValue::Numeric(3),
                P::Rotation => TrackValue::Numeric(4),

                P::Visibility(_) | P::Active => TrackValue::Bool,
                P::Style(p) if matches!(p.interpolation(),StyleInterpolation::Number | StyleInterpolation::FontWeight) => TrackValue::Numeric(1),
                P::Style(_) => TrackValue::Style,
                P::Fill | P::Stroke => TrackValue::Numeric(4),
                _ => TrackValue::Numeric(1),
            },
            Self::Material { input, .. } => match OpenPbrParam::find(input).map(|p| p.kind) {
                Some(OpenPbrKind::Bool) => TrackValue::Bool,
                Some(OpenPbrKind::Color3 | OpenPbrKind::Vector3) => TrackValue::Numeric(3),
                _ => TrackValue::Numeric(1),
            },
            Self::ImageTime { .. } | Self::LayerWeight { .. } => TrackValue::Numeric(1),
            Self::Render(RenderProperty::Background) => TrackValue::Numeric(3),
            Self::Render(RenderProperty::Camera) => TrackValue::Camera,
            Self::Render(_) => TrackValue::Numeric(1),
            Self::Text { property, .. } => match property {
                UnitProperty::Translation | UnitProperty::Scale => TrackValue::Numeric(3),
                UnitProperty::Rotation | UnitProperty::Fill | UnitProperty::Stroke => TrackValue::Numeric(4),
                UnitProperty::Opacity => TrackValue::Numeric(1),
                UnitProperty::TransformOrigin => TrackValue::Token,
            },
        }
    }
}
impl Scene {
    /// Key times, value shapes and buffers for one track against its target.
    fn validate_keys(&self, track: &Track, target: &TrackTarget, blend: Blend, nodes: &BTreeMap<&str, &Node>) -> Result<()> {
        let path = display_path(&track.path);
        let buffer = |name: &str, what: &str| -> Result<&Buffer> {
            self.buffers.get(name).with_context(|| {
                format!(
                    "track {path}: {what}: {}",
                    diagnostics::unknown("buffer", name, self.buffers.keys().map(String::as_str))
                )
            })
        };
        let keys = match &track.times {
            Times::Inline(t) => {
                if matches!(target,TrackTarget::ImageTime {..}|TrackTarget::Render(RenderProperty::Camera)) || target.is_speed() {
                    let Keys::Inline(keys)=&track.keys else {unreachable!()};
                    let times=keys.iter().map(|key|serde_json::from_value::<vork_time::Clock>(key[0].clone()).map(|t|t.exact())).collect::<std::result::Result<Vec<_>,_>>()?;
                    ensure!(!times.is_empty() && times.iter().all(|t|*t>=vork_time::exact::Rational::from_integer(0.into())) && times.windows(2).all(|p|p[0]<p[1]),
                        "track {path}: times must be non-empty, finite, >= 0 and strictly increasing");
                } else {
                ensure!(
                    !t.is_empty()
                        && t.iter().all(|x| x.is_finite() && *x >= 0.)
                        && t.windows(2).all(|w| w[0] < w[1]),
                    "track {path}: times must be non-empty, finite, >= 0 and strictly increasing"
                );
                }
                Some(t.len())
            }
            Times::Buffer(name) => {
                let b = buffer(name, "times")?;
                ensure!(
                    matches!(b.element_type, BufferType::F32),
                    "track {path}: times buffer {name} must be f32"
                );
                b.len()
            }
        };
        if self.is_canvas() && matches!(target,TrackTarget::Render(RenderProperty::Background)) {
            ensure!(matches!(&track.values,Values::Inline(values) if values.iter().all(|v|v.is_string())),"track {path}: a canvas background keys CSS colours");
        }
        if target.style_interpolation().is_some() && !matches!(&track.values, Values::Buffer(_)) {
            let Values::Inline(values) = &track.values else { unreachable!() };
            // Numeric RGBA tracks retain the existing glTF numeric channel.
            let numeric_style = matches!(target.style_interpolation(),Some(StyleInterpolation::Number | StyleInterpolation::FontWeight)) && values.iter().all(|v|v.is_number());
            let rgba = matches!(target, TrackTarget::Render(RenderProperty::Background) | TrackTarget::Node { property: NodeProperty::Fill | NodeProperty::Stroke, .. }
                | TrackTarget::Text { property: UnitProperty::Fill | UnitProperty::Stroke, .. })
                && values.first().is_some_and(|v| !v.is_string() && !v.is_object());
            if !rgba && !numeric_style {
                ensure!(!matches!(track.interpolation, Interpolation::Cubic), "track {path}: CUBICSPLINE for CSS/SVG values is not built; use LINEAR or STEP");
                ensure!(blend != Blend::Add, "track {path}: combine for CSS/SVG values is not built; use replace");
                ensure!(keys == Some(values.len()), "track {path}: keys and values must have the same count");
                // The tracked node, copied once without its children: each key's value is written into the keyed
                // field of that copy, and the copy is checked as the node it then is.
                let mut keyed_node: Option<Node> = None;
                for value in values {
                    match target {
                        TrackTarget::Node { node, property } => {
                            if matches!(property,NodeProperty::Style(p) if matches!(p.interpolation(),StyleInterpolation::Path | StyleInterpolation::Filter)) {
                                ensure!(value.is_string(),"track {path}: expected a CSS/SVG string");
                                continue;
                            }
                            let keyed = keyed_node.get_or_insert_with(|| nodes[node.as_str()].without_children());
                            keyed.set_field(property.segments()[0].as_str(), value).with_context(|| format!("track {path}: invalid key {value}"))?;
                            let keyed = &*keyed;
                            Self::validate_text(keyed, &self.images).with_context(|| format!("track {path}"))?;
                            if let NodeKind::Path {fill,stroke,..} | NodeKind::Text {fill,stroke,..}=&keyed.kind {
                                for paint in [fill,stroke] {if let Paint::Image(image)=paint {ensure!(self.images.contains_key(&image.image),"track {path}: unknown image {:?}",image.image);}}
                            }

                            if let NodeProperty::Style(p) = property {
                                match p.name() {
                                    "clip-path" | "mask" => ensure!(value.is_null() || value.as_str().is_some_and(|name|nodes.contains_key(name)),"track {path}: unknown clip-path or mask node {value}"),
                                    "stroke-miterlimit" => ensure!(value.as_f64().is_some_and(|v| v >= 1.), "track {path}: stroke-miterlimit must be at least 1"),
                                    "pathLength" => ensure!(value.is_null() || value.as_f64().is_some_and(|v| v > 0.), "track {path}: pathLength must be positive"),
                                    "stroke-dasharray" => ensure!(value.as_array().is_none_or(|a| a.iter().all(|n| n.as_f64().is_some_and(|v| v >= 0.))), "track {path}: dash lengths must be nonnegative"),
                                    _ => {},
                                }
                            }
                        }
                        TrackTarget::Render(RenderProperty::Background) => {
                            ensure!(self.is_canvas(),"track {path}: a 3D background keys RGB numbers");
                            ensure!(text::paint(value.as_str().context("background keys are CSS colours")?).with_context(||format!("track {path}"))?.is_some(),"track {path}: use transparent for a transparent background");
                        }
                        TrackTarget::Text { property: UnitProperty::TransformOrigin, .. } => {text::transform_origin(value.as_str().context("transform-origin keys are CSS strings")?).map_err(|e|anyhow::anyhow!("track {path}: {e}"))?;}
                        TrackTarget::Text { .. } => {
                            let paint:Paint=serde_json::from_value(value.clone()).with_context(||format!("track {path}: invalid paint"))?;
                            if let Paint::Css(css)=&paint {if !css.contains("gradient("){text::paint(css).map_err(|e|anyhow::anyhow!("track {path}: {e}"))?;}}
                            if let Paint::Image(image)=&paint {ensure!(self.images.contains_key(&image.image),"track {path}: unknown image {:?}",image.image);}
                        }
                        _ => unreachable!(),
                    }
                }
                return Ok(());
            }
        }
        // Program types are reflected at install. Validate the authored numeric
        // shape here; reflection checks its width and storage type before binding.
        let value = if let TrackTarget::Material { material, .. } = target {
            if self.materials[material].program != "std/openpbr" {
                let width = match &track.values {
                    Values::Inline(values) => values.first().and_then(|v| v.as_array()).map_or(1, Vec::len),
                    Values::Buffer(name) => buffer(name, "values")?.element_type.width(),
                };
                ensure!((1..=4).contains(&width), "track {path}: program inputs require scalar or vector numeric keys");
                match &track.values {
                    Values::Inline(values) if values.first().is_some_and(|v| v.is_string()) => TrackValue::Token,
                    Values::Inline(values) if values.first().is_some_and(|v| v.is_boolean()) => TrackValue::Bool,
                    _ => TrackValue::Numeric(width),
                }
            } else { target.value() }
        } else { target.value() };
        let numeric = |v: &serde_json::Value, width: usize| match (v, width) {
            (serde_json::Value::Number(n), 1) => n.as_f64().is_some_and(|n| (n as f32).is_finite()),
            (serde_json::Value::Array(a), _) => {
                a.len() == width
                    && a.iter()
                        .all(|n| n.as_f64().is_some_and(|n| (n as f32).is_finite()))
            }
            _ => false,
        };
        let expected = match (value, target) {
            (_, TrackTarget::Text { property: UnitProperty::Fill | UnitProperty::Stroke, .. }
                | TrackTarget::Node { property: NodeProperty::Fill | NodeProperty::Stroke, .. }) =>
                "arrays of 4 numbers (r, g, b, a from 0 to 1), or CSS colour strings".to_string(),
            (_, TrackTarget::Text { property: UnitProperty::TransformOrigin, .. }) => "CSS transform-origin strings".to_string(),
            _ => match value {
            TrackValue::Numeric(1) => "numbers".to_string(),
            TrackValue::Numeric(w) => format!("arrays of {w} numbers"),
            TrackValue::Bool => "booleans".into(),
            TrackValue::Camera => "camera names".into(),
            TrackValue::Token => "enum tokens".into(),
            TrackValue::Style => "CSS/SVG values".into(),
            },
        };
        let rotation = matches!(target, TrackTarget::Node { property: NodeProperty::Rotation, .. }
            | TrackTarget::Imported { property: NodeProperty::Rotation, .. }
            | TrackTarget::Text { property: UnitProperty::Rotation, .. });
        let values = match &track.values {
            Values::Inline(values) => {
                // A rotation track is keyed in one form: quaternions or Euler angles.
                let euler = rotation && values.first().is_some_and(|v| v.get("euler").is_some());
                for (i, v) in values.iter().enumerate() {
                    let ok = match value {
                        TrackValue::Numeric(_) if euler => v.as_object().is_some_and(|o| o.len() == 1)
                            && v.get("euler").is_some_and(|angles| numeric(angles, 3)),
                        TrackValue::Numeric(1) if matches!(target,TrackTarget::ImageTime {..}) || target.is_speed() => {
                            let scalar=v.as_array().filter(|v|v.len()==1).map_or(v,|v|&v[0]);
                            serde_json::from_value::<vork_time::Clock>(scalar.clone()).is_ok()
                        }
                        TrackValue::Numeric(w) => numeric(v, w),
                        TrackValue::Style => false,
                        TrackValue::Bool => v.is_boolean(),
                        TrackValue::Camera | TrackValue::Token => v.is_string(),

                    };
                    ensure!(ok, "track {path}: value {i} is {v}; expected {}",
                        if euler { "{\"euler\": [x,y,z]} like the first key".to_string() }
                        else if rotation { format!("{expected}, or {{\"euler\": [x,y,z]}} on every key") } else { expected.clone() });
                }
                if let TrackTarget::Text { property: UnitProperty::TransformOrigin, .. } = target {
                    for value in values.iter().filter_map(|v| v.as_str()) {
                        text::transform_origin(value).map_err(|error| anyhow::anyhow!("track {path}: {error}"))?;
                    }
                }
                Some(values.len())
            }
            Values::Buffer(name) => {
                let b = buffer(name, "values")?;
                let TrackValue::Numeric(width) = value else {
                    anyhow::bail!("track {path}: {expected} must be inline values");
                };
                ensure!(
                    b.element_type.width() == width && !matches!(b.element_type, BufferType::Mat4),
                    "track {path}: values buffer {name} has {} components per key; expected {width}",
                    b.element_type.width()
                );
                b.len()
            }
        };
        if let (Some(keys), Some(values)) = (keys, values) {
            ensure!(
                values == keys * if matches!(track.interpolation, Interpolation::Cubic) { 3 } else { 1 },
                "track {path}: {values} values for {keys} times; CUBICSPLINE requires incoming/value/outgoing triples"
            );
        }
        if let TrackTarget::Node { node, property: NodeProperty::Style(property) } = target {
            if let Values::Inline(values) = &track.values {
                let cubic=matches!(track.interpolation,Interpolation::Cubic);
                // The tracked node, copied once: each key's value is written into the keyed field of the copy.
                let mut keyed=nodes[node.as_str()].without_children();
                for value in values.iter().skip(usize::from(cubic)).step_by(if cubic {3}else{1}) {
                    keyed.set_field(property.name(),value).with_context(||format!("track {path}: invalid key {value}"))?;
                    Self::validate_text(&keyed, &self.images).with_context(||format!("track {path}"))?;
                    if property.name()=="stroke-miterlimit" {ensure!(value.as_f64().is_some_and(|v|v>=1.),"track {path}: stroke-miterlimit must be at least 1");}
                    if property.name()=="pathLength" {ensure!(value.as_f64().is_some_and(|v|v>0.),"track {path}: pathLength must be positive");}
                }
            }
        }
        if target.is_speed() {
            ensure!(blend != Blend::Add, "track {path}: time.speed cannot blend additively");
            if let Values::Inline(values) = &track.values {
                ensure!(speed_curve_nonnegative(track, values), "track {path}: a negative speed is a constant; for a changing or reversing speed write a clock program");
            } else {
                anyhow::bail!("track {path}: time.speed keys must be inline values");
            }
        }
        if !matches!(value, TrackValue::Numeric(_)) {
            ensure!(
                matches!(track.interpolation, Interpolation::Step),
                "track {path}: {expected} require step interpolation"
            );
            ensure!(
                blend != Blend::Add,
                "track {path}: {expected} cannot blend additively"
            );
        }
        Ok(())
    }
}

impl TrackTarget {
    pub fn style_interpolation(&self) -> Option<StyleInterpolation> {
        match self {
            Self::Render(RenderProperty::Background) => Some(StyleInterpolation::Paint),
            Self::Text { property: UnitProperty::TransformOrigin, .. } => Some(StyleInterpolation::Origin),
            Self::Node { property: NodeProperty::Style(p), .. } => Some(p.interpolation()),
            Self::Node { property: NodeProperty::Fill | NodeProperty::Stroke, .. }
                | Self::Text { property: UnitProperty::Fill | UnitProperty::Stroke, .. } => Some(StyleInterpolation::Paint),
            _ => None,
        }
    }
    pub fn is_speed(&self) -> bool {
        matches!(self, Self::Node { property: NodeProperty::Speed, .. } | Self::Imported { property: NodeProperty::Speed, .. })
    }
    pub fn is_clock(&self) -> bool {
        matches!(self, Self::Node { property: NodeProperty::Speed | NodeProperty::Active, .. })
    }
}

/// A keyed speed stays at or above zero everywhere on its curve: at every key, and
/// between keys for CUBICSPLINE, whose Hermite segment may dip below its end values.
fn speed_curve_nonnegative(track: &Track, values: &[serde_json::Value]) -> bool {
    let scalar = |v: &serde_json::Value| -> Option<f64> {
        let v = v.as_array().filter(|v| v.len() == 1).map_or(v, |v| &v[0]);
        serde_json::from_value::<vork_time::Clock>(v.clone()).ok().map(|c| c.as_f64())
    };
    // Exact signs first: an authored -1/10^400 is negative however f64 rounds it.
    let exact = |v: &serde_json::Value| -> Option<vork_time::exact::Rational> {
        let v = v.as_array().filter(|v| v.len() == 1).map_or(v, |v| &v[0]);
        serde_json::from_value::<vork_time::Clock>(v.clone()).ok().map(|c| c.exact())
    };
    let cubic = matches!(track.interpolation, Interpolation::Cubic);
    let stride = if cubic { 3 } else { 1 };
    let zero = vork_time::exact::Rational::from_integer(0.into());
    for value in values.iter().skip(usize::from(cubic)).step_by(stride) {
        match exact(value) { Some(v) if v >= zero => {}, _ => return false }
    }
    let Some(numbers) = values.iter().map(scalar).collect::<Option<Vec<f64>>>() else { return false };
    let keys: Vec<f64> = numbers.iter().skip(usize::from(cubic)).step_by(stride).copied().collect();
    if !cubic { return true; }
    // Key times as authored (exact decimals to f64), not the f32 evaluation copies, so
    // two distinct keys never collapse to one time (xtimerev 3).
    let Keys::Inline(authored) = &track.keys else { return true };
    let Some(times) = authored.iter().map(|key| serde_json::from_value::<vork_time::Clock>(key[0].clone()).ok().map(|c| c.as_f64())).collect::<Option<Vec<f64>>>() else { return false };
    for i in 0..keys.len().saturating_sub(1) {
        let dt = times[i + 1] - times[i];
        let (p0, p3) = (keys[i], keys[i + 1]);
        let (m0, m1) = (numbers[i * 3 + 2] * dt, numbers[(i + 1) * 3] * dt);
        // Bezier control points of the Hermite segment; all nonnegative is sufficient.
        if p0 + m0 / 3. >= 0. && p3 - m1 / 3. >= 0. { continue; }
        // Otherwise the minimum sits at a root of the derivative inside (0, 1); the
        // degeneracy tests are relative to the segment's magnitude.
        let (a, b, c) = (2. * p0 + m0 - 2. * p3 + m1, -3. * p0 - 2. * m0 + 3. * p3 - m1, m0);
        let scale = (a.abs() + b.abs() + c.abs()).max(f64::MIN_POSITIVE);
        let at = |u: f64| ((a * u + b) * u + c) * u + p0;
        let roots: Vec<f64> = if a.abs() <= 1e-12 * scale { if b.abs() <= 1e-12 * scale { vec![] } else { vec![-c / (2. * b)] } } else {
            let d = (2. * b).powi(2) - 12. * a * c;
            if d < 0. { vec![] } else { vec![(-2. * b - d.sqrt()) / (6. * a), (-2. * b + d.sqrt()) / (6. * a)] }
        };
        if roots.into_iter().filter(|u| (0. ..=1.).contains(u)).any(|u| at(u) < 0.) { return false; }
    }
    true
}

/// A node-owned animation layer with its tracks resolved to absolute targets.
pub struct NodeOwnedLayer<'a> {
    /// Child indices from the root: the pre-order address of the owning node.
    pub path: Vec<usize>,
    pub owner: &'a Node,
    pub layer: &'a NodeLayer,
    pub targets: Vec<TrackTarget>,
}
impl Scene {
    /// Every node-owned layer in pre-order, its tracks resolved as the owner's
    /// paths: `["translation"]`, `["children", name, ...]` (A3). Named nodes only.
    pub fn node_owned_layers(&self) -> Result<Vec<NodeOwnedLayer<'_>>> {
        self.node_owned_layers_in(&self.nodes_by_name())
    }
    /// [`Self::node_owned_layers`] against a name index the caller already holds.
    pub(super) fn node_owned_layers_in(&self, nodes: &BTreeMap<&str, &Node>) -> Result<Vec<NodeOwnedLayer<'_>>> {
        let mut out = Vec::new();
        fn visit<'a>(scene: &'a Scene, list: &'a [Node], path: &mut Vec<usize>, nodes: &BTreeMap<&str, &Node>, out: &mut Vec<NodeOwnedLayer<'a>>) -> Result<()> {
            for (index, node) in list.iter().enumerate() {
                path.push(index);
                for layer in &node.animation {
                    ensure!(layer.tracks.is_some() != layer.clip.is_some(), "node {:?}: animation layer requires exactly one of tracks or clip", node.name);
                    if let Some(clip) = &layer.clip {
                        ensure!(scene.descendant_or_self(node, &clip.target).is_some_and(|n| matches!(n.kind, NodeKind::Asset { .. })),
                            "node {:?}: clip target {:?} must be this node or a descendant asset node", node.name, clip.target);
                    }
                    let targets = layer.tracks.iter().flatten().map(|track| scene.resolve_relative(node, &track.path, nodes)).collect::<Result<Vec<_>>>()?;
                    out.push(NodeOwnedLayer { path: path.clone(), owner: node, layer, targets });
                }
                visit(scene, &node.children, path, nodes, out)?;
                path.pop();
            }
            Ok(())
        }
        visit(self, &self.nodes, &mut Vec::new(), nodes, &mut out)?;
        Ok(out)
    }
    fn descendant_or_self<'a>(&self, node: &'a Node, name: &str) -> Option<&'a Node> {
        if node.name == name { return Some(node); }
        node.children.iter().find_map(|child| self.descendant_or_self(child, name))
    }
    /// An owner-relative track path resolved to an absolute target, with rule 12's
    /// ownership: a node never keys its own clock; `active` and `time.speed` are
    /// keyed by the parent (one `children` hop) or the scene.
    fn resolve_relative(&self, owner: &Node, path: &[String], nodes: &BTreeMap<&str, &Node>) -> Result<TrackTarget> {
        let mut node = owner;
        let mut index = 0;
        let mut hops = 0;
        while path.get(index).map(String::as_str) == Some("children") {
            let name = segment(path, index + 1, "a child node name")?;
            node = node.children.iter().find(|child| child.name == name).ok_or_else(|| bad_segment(path, index + 1, "child node", node.children.iter().map(|c| c.as_str_name()).filter(|n| !n.is_empty())))?;
            index += 2;
            hops += 1;
        }
        ensure!(path.first().map(String::as_str) != Some("nodes"), "track {}: a node's paths start at the node: [\"translation\"] or [\"children\", name, ...]", display_path(path));
        ensure!(path.get(index).map(String::as_str) != Some("materials"), "track {}: materials are keyed from the scene's animation: [\"materials\", name, \"inputs\", input]; name the material", display_path(path));
        ensure!(!node.name.is_empty(), "track {}: name the node it keys", display_path(path));
        let mut absolute = vec!["nodes".to_string(), node.name.clone()];
        absolute.extend(path[index..].iter().cloned());
        let target = self.resolve_target(&absolute, nodes).map_err(|error| anyhow::anyhow!("{}", error.to_string().replace(&display_path(&absolute), &display_path(path))))?;
        if target.is_clock() {
            ensure!(hops != 0, "track {}: the clock cannot key itself", display_path(path));
            ensure!(hops == 1, "track {}: active and time.speed are keyed by the parent or the scene", display_path(path));
        }
        Ok(target)
    }
}
impl Node {
    fn as_str_name(&self) -> &str { &self.name }
}

/// The node property starting at segment `index`, with per-segment errors.
fn node_property(path: &[String], index: usize) -> Result<NodeProperty> {
    use NodeProperty as P;
    let name = segment(path, index, "a node property")?;
    let property = match name {
        "target" => anyhow::bail!("track {}: target is structural; aim at a named node and animate that node's translation instead", display_path(path)),
        "translation" => P::Translation,
        "rotation" => P::Rotation,
        "scale" => P::Scale,
        "visible" => P::Visibility(Ray::All),
        "visible_camera" => P::Visibility(Ray::Camera),
        "visible_shadow" => P::Visibility(Ray::Shadow),
        "visible_diffuse" => P::Visibility(Ray::Diffuse),
        "visible_glossy" => P::Visibility(Ray::Glossy),
        "visible_transmission" => P::Visibility(Ray::Transmission),
        "morph_weights" => {
            let morph = segment(path, index + 1, "a morph name")?;
            end(path, index + 2)?;
            return Ok(P::MorphWeight(morph.into()));
        }
        "spot_size" => P::SpotSize,
        "spot_blend" => P::SpotBlend,
        "color" => P::Color,
        "temperature" => P::Temperature,
        "energy" => P::Energy,
        "angle" => P::Angle,
        "radius" => P::Radius,
        "size" => P::Width,
        "size_y" => P::Height,
        "active" => P::Active,
        "opacity" => P::Opacity,
        "fill" => P::Fill,
        "stroke" => P::Stroke,
        "stroke-width" => P::StrokeWidth,
        "stroke-dashoffset" => P::StrokeDashoffset,
        "fill-opacity" => P::FillOpacity,
        "stroke-opacity" => P::StrokeOpacity,
        "audio" => {
            ensure!(
                segment(path, index + 1, "gain")? == "gain",
                "track {}: audio keys gain; file is structural",
                display_path(path)
            );
            end(path, index + 2)?;
            return Ok(P::AudioGain);
        }
        "time" => {
            ensure!(segment(path, index + 1, "speed")? == "speed", "track {}: time keys speed only; start, offset, duration and loop are constants", display_path(path));
            end(path, index + 2)?;
            return Ok(P::Speed);
        }
        name if StyleProperty::parse(name).is_some() => P::Style(StyleProperty::parse(name).unwrap()),
        _ => {
            return Err(bad_segment(
                path,
                index,
                "keyable node property",
                NODE_PROPERTIES.iter().copied().chain(STYLE_PROPERTIES.iter().map(|p| p.0)),
            ));
        }
    };
    end(path, index + 1)?;
    Ok(property)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene() -> Scene {
        crate::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"asset","name":"actor","file":"a.glb"},
                {"type":"light","name":"key","shape":"spot","spot_size":0.7853981633974483,"spot_blend":0.14999999999999986}],
            "materials":{"paint":{}},
            "animation":[{"name":"walk","tracks":[]}]}))
        .unwrap()
    }
    fn path(segments: &[&str]) -> Vec<String> {
        segments.iter().map(|s| s.to_string()).collect()
    }
    #[test]
    fn program_input_css_names_keep_discrete_defaults() {
        for name in ["fill", "stroke", "direction", "width", "filter", "content", "transform-origin"] {
            for values in [serde_json::json!([false, true]), serde_json::json!(["off", "on"])] {
                let document = serde_json::json!({"version":2,
                    "render":{"resolution":[64,64]},"composition":{"space":"3d","view":"grade"},
                    "nodes":[{"type":"camera"}],
                    "materials":{"grade":{"program":"grade","inputs":{name:values[0]}}},
                    "programs":{"grade":{"source":"import vork; struct Inputs { bool fill=false; }; import std.view; ViewResult view(ViewContext c, Inputs i) { return finish_view(i.fill ? float3(1.) : float3(0.),c,1.); }"}},
                    "animation":[{"tracks":[{"path":["materials","grade","inputs",name],"keys":[[0,values[0]],[1,values[1]]]}]}]});
                let scene = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap();
                assert!(matches!(scene.animation[0].tracks.as_ref().unwrap()[0].interpolation, Interpolation::Step), "{name}");
            }
        }
    }

    #[test]
    fn keyed_content_defers_piece_counts_to_sampled_layout() {
        for unit in ["glyphs", "words", "lines"] {
            for owned in [false, true] {
                let prefix = if owned { vec![] } else { vec!["nodes", "title"] };
                let mut content_path = prefix.clone(); content_path.push("content");
                let mut piece_path = prefix; piece_path.extend([unit, "1", "opacity"]);
                let (first, last) = if unit == "glyphs" { ("AB", "CD") } else { ("AB CD", "EF GH") };
                let tracks = serde_json::json!([
                    {"path":content_path,"keys":[[0,first],[1,last]]},
                    {"path":piece_path,"keys":[[0,0],[1,1]]}]);
                let mut document = serde_json::json!({"version":2,"composition":{"space":"2d"},
                    "nodes":[{"name":"title","type":"text","font":"font.ttf","content":"A"}]});
                if owned { document["nodes"][0]["animation"] = serde_json::json!([{"tracks":tracks}]); }
                else { document["animation"] = serde_json::json!([{"tracks":tracks}]); }
                Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap();
            }
        }
    }

    #[test]
    fn movie_resources_include_keyed_paints_and_content() {
        let doc: Scene = serde_json::from_value(serde_json::json!({"version":2,
            "composition":{"space":"2d"}, "images":{"clip":{"file":"clip.mp4"}},
            "nodes":[{"name":"parent","animation":[{"tracks":[
                {"path":["children","path","fill"],"keys":[[0,"red"],[1,{"image":"clip"}]]}
            ]}],"children":[{"name":"path","type":"path","d":"M0 0H10V10Z"}]},
            {"name":"text","type":"text","font":"font.ttf","content":"A"}],
            "animation":[{"tracks":[{"path":["nodes","text","content"],
                "keys":[[0,"A"],[1,[{"content":"B","fill":{"image":"clip"}}]]]}]}]
        })).unwrap();
        assert_eq!(doc.node_media(&doc.nodes[0].children[0]), vec!["clip"]);
        assert_eq!(doc.node_media(&doc.nodes[1]), vec!["clip"]);
    }

    #[test]
    fn target_track_explains_animated_aim() {
        let s = scene();
        let error = s.resolve_target(&path(&["nodes", "cam", "target"]), &s.nodes_by_name()).unwrap_err().to_string();
        assert!(error.contains(r#"["nodes","cam","target"]"#), "{error}");
        assert!(error.contains("aim at a named node"), "{error}");
        assert!(error.contains("animate that node's translation"), "{error}");
        assert!(s.resolve_target(&path(&["nodes", "cam", "translation"]), &s.nodes_by_name()).is_ok());
    }
    #[test]
    fn camera_optics_are_not_node_track_properties() {
        let s=scene();
        for property in ["lens","shift","dof","clip_start","clip_end","ortho_scale"] {
            assert!(s.resolve_target(&path(&["nodes","cam",property]),&s.nodes_by_name()).is_err());
        }
    }
    #[test]
    fn post_processing_inputs_resolve_as_material_tracks() {
        let mut s=scene();
        s.programs.insert("post".into(),crate::v2::from_test_value(serde_json::json!({"source":"struct Inputs {float gain=1.;}; float4 composite(CompositeContext c,Inputs i){return float4(i.gain);}"})).unwrap());
        s.materials.insert("grade".into(),crate::v2::from_test_value(serde_json::json!({"program":"post"})).unwrap());
        s.compositor_mut().push("grade".into());
        let target=s.resolve_target(&path(&["materials","grade","inputs","gain"]),&s.nodes_by_name()).unwrap();
        assert!(matches!(target, TrackTarget::Material { material, input } if material == "grade" && input == "gain"));
    }
    #[test]
    fn render_settings_are_not_animatable() {
        let mut s = scene();
        s.render.settings.integrator.raster_mut().unwrap().fast_gi = Some(Default::default());
        for segments in [&["render", "settings", "fast_gi", "radius"][..], &["render", "integrator", "fast_gi", "radius"]] {
            let error = format!("{:#}", s.resolve_target(&path(segments), &s.nodes_by_name()).unwrap_err());
            assert!(error.contains("keyable render property"), "{error}");
        }
    }
    #[test]
    fn imported_nodes_use_an_explicit_segment() {
        let s = scene();
        // A glTF node named like a property is unambiguous.
        assert_eq!(
            s.resolve_target(&path(&["nodes", "actor", "nodes", "rotation", "rotation"]), &s.nodes_by_name()).unwrap(),
            TrackTarget::Imported { asset: "actor".into(), node: "rotation".into(), property: NodeProperty::Rotation }
        );
        assert_eq!(
            s.resolve_target(&path(&["nodes", "actor", "nodes", "#3", "scale"]), &s.nodes_by_name()).unwrap().to_string(),
            r##"["nodes","actor","nodes","#3","scale"]"##
        );
        let error = format!("{:#}", s.resolve_target(&path(&["nodes", "actor", "Bone.001", "rotation"]), &s.nodes_by_name()).unwrap_err());
        assert!(error.contains("segment 2"), "{error}");
    }
    #[test]
    fn segments_resolve_to_typed_targets() {
        let s = scene();
        assert_eq!(
            s.resolve_target(&path(&["nodes", "actor", "nodes", "Bone.001", "rotation"]), &s.nodes_by_name())
                .unwrap(),
            TrackTarget::Imported {
                asset: "actor".into(),
                node: "Bone.001".into(),
                property: NodeProperty::Rotation
            }
        );
        assert_eq!(
            s.resolve_target(&path(&["nodes", "actor", "translation"]), &s.nodes_by_name())
                .unwrap(),
            TrackTarget::Node {
                node: "actor".into(),
                property: NodeProperty::Translation
            }
        );
        assert_eq!(
            s.resolve_target(&path(&["nodes", "key", "spot_size"]), &s.nodes_by_name())
                .unwrap(),
            TrackTarget::Node {
                node: "key".into(),
                property: NodeProperty::SpotSize
            }
        );
        assert_eq!(
            s.resolve_target(&path(&["materials", "paint", "inputs", "base_color"]), &s.nodes_by_name())
                .unwrap(),
            TrackTarget::Material {
                material: "paint".into(),
                input: "base_color".into()
            }
        );
        assert_eq!(
            s.resolve_target(&path(&["animation", "walk", "weight"]), &s.nodes_by_name())
                .unwrap(),
            TrackTarget::LayerWeight {
                layer: "walk".into()
            }
        );
        assert_eq!(
            s.resolve_target(&path(&["render", "camera"]), &s.nodes_by_name()).unwrap(),
            TrackTarget::Render(RenderProperty::Camera)
        );
        for p in [
            &["nodes", "actor", "nodes", "Bone.001", "rotation"][..],
            &["nodes", "key", "spot_size"],
            &["materials", "paint", "inputs", "base_color"],
            &["animation", "walk", "weight"],
        ] {
            assert_eq!(
                s.resolve_target(&path(p), &s.nodes_by_name()).unwrap().to_string(),
                display_path(&path(p))
            );
        }
    }
    #[test]
    fn errors_name_the_bad_segment() {
        let s = scene();
        for (bad, expect) in [
            (&["nodes", "cmaera", "rotation"][..], "segment 1"),
            (&["nodes", "cam", "rotaton"][..], "segment 2"),
            (&["nodes", "cam", "energy"][..], "segment 2"),
            (&["nodes", "key", "spot_middl"][..], "segment 2"),
            (&["nodes", "cam", "rotation", "x"][..], "segment 3"),
            (&["materials", "paint", "inputs", "colour"][..], "segment 3"),
            (&["render", "fps"][..], "segment 1"),
            (&["layers"][..], "segment 0"),
        ] {
            let error = format!("{:#}", s.resolve_target(&path(bad), &s.nodes_by_name()).unwrap_err());
            assert!(error.contains(expect), "{bad:?}: {error}");
        }
    }
    #[test]
    fn key_shapes_buffers_and_kinds_fail_validation() {
        let check = |tracks: serde_json::Value, blend: &str, expect: &str| {
            let s: Scene = crate::v2::from_test_value(serde_json::json!({"version":2,
                "nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
                "materials":{"paint":{},"custom":{"program":"p"}},"programs":{"p":{"source":"x.slang"}},
                "buffers":{"t":{"type":"f32","data":[0,1]},"v3":{"type":"vec3","data":[[0,0,0],[1,1,1]]}},
                "animation":[{"blend":blend,"tracks":tracks}]}))
            .unwrap();
            let error = format!("{:#}", s.validate().unwrap_err());
            assert!(error.contains(expect), "{tracks}: {error}");
        };
        check(
            serde_json::json!([{"path":["nodes","cam","rotation"],"keys":[[0,[0,0,1]]]}]),
            "replace",
            "expected arrays of 4 numbers",
        );
        check(
            serde_json::json!([{"path":["nodes","cam","rotation"],"keys":{"times":"t","values":"v3"}}]),
            "replace",
            "expected 4",
        );
        check(
            serde_json::json!([{"path":["nodes","cam","translation"],"keys":{"times":"missing","values":"v3"}}]),
            "replace",
            "unknown buffer",
        );
        check(
            serde_json::json!([{"path":["nodes","cam","translation"],"keys":{"times":"t","values":"nope"}}]),
            "replace",
            "unknown buffer",
        );
        check(
            serde_json::json!([{"path":["nodes","cam","translation"],"keys":[[-1,[0,0,0]],[1,[0,0,0]]]}]),
            "replace",
            ">= 0",
        );
        check(
            serde_json::json!([{"path":["nodes","cam","visible_camera"],"keys":[[0,true]],"interpolation":"STEP"}]),
            "combine",
            "additively",
        );
        check(
            serde_json::json!([{"path":["materials","paint","inputs","geometry_thin_walled"],"keys":[[0,1]],"interpolation":"STEP"}]),
            "replace",
            "expected booleans",
        );
        check(
            serde_json::json!([{"path":["materials","custom","inputs","x"],"keys":[[0,"resource"],[1,1]]}]),
            "replace",
            "expected enum tokens",
        );
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    /// The test path: the document path refuses these fields in this release.
    fn open(scene: serde_json::Value) -> std::result::Result<Scene, String> {
        let scene: Scene = crate::v2::from_test_value(scene).map_err(|e| format!("{e:#}"))?;
        scene.validate().map_err(|e| format!("{e:#}"))?;
        Ok(scene)
    }
    fn scene(nodes: serde_json::Value, animation: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"version":2,"nodes":nodes,"animation":animation})
    }
    const CAM: &str = r#"{"type":"camera","name":"cam"}"#;
    fn cam() -> serde_json::Value { serde_json::from_str(CAM).unwrap() }
    fn keys() -> serde_json::Value { serde_json::json!([[0,[0,0,0]],[1,[1,0,0]]]) }

    #[test]
    fn a_property_has_one_owner() {
        // The scene and the node both keying the node's translation: refused (xtimerev 1).
        let both = scene(serde_json::json!([cam(), {"name":"n","animation":[{"tracks":[{"path":["translation"],"keys":keys()}]}]}]),
            serde_json::json!([{"tracks":[{"path":["nodes","n","translation"],"keys":keys()}]}]));
        assert!(open(both).unwrap_err().contains("a property has one owner"));
        // Two scene layers on one property blend: one owner.
        let layers = scene(serde_json::json!([cam(), {"name":"n"}]),
            serde_json::json!([{"tracks":[{"path":["nodes","n","translation"],"keys":keys()}]},{"blend":"combine","tracks":[{"path":["nodes","n","translation"],"keys":keys()}]}]));
        assert!(open(layers).is_ok());
        // Two node layers of one node on one property: one owner too.
        let two = scene(serde_json::json!([cam(), {"name":"n","animation":[{"tracks":[{"path":["translation"],"keys":keys()}]},{"tracks":[{"path":["translation"],"keys":keys()}]}]}]), serde_json::json!([]));
        assert!(open(two).is_ok());
        // Parent and child both keying the child's translation: refused.
        let parent_child = scene(serde_json::json!([cam(), {"name":"p","animation":[{"tracks":[{"path":["children","c","translation"],"keys":keys()}]}],
            "children":[{"name":"c","animation":[{"tracks":[{"path":["translation"],"keys":keys()}]}]}]}]), serde_json::json!([]));
        assert!(open(parent_child).unwrap_err().contains("a property has one owner"));
        // A scene clip layer and the asset's own clip layer on one asset: two owners.
        let clips = scene(serde_json::json!([cam(), {"type":"asset","name":"actor","file":"a.glb","animation":[{"clip":{"target":"actor"}}]}]),
            serde_json::json!([{"clip":{"target":"actor"}}]));
        assert!(open(clips).unwrap_err().contains("a property has one owner"));
    }

    #[test]
    fn clocks_are_keyed_by_the_parent_or_the_scene_only() {
        let own = scene(serde_json::json!([cam(), {"name":"n","time":{},"animation":[{"tracks":[{"path":["time","speed"],"keys":[[0,1],[1,2]]}]}]}]), serde_json::json!([]));
        assert!(open(own).unwrap_err().contains("the clock cannot key itself"));
        let own_active = scene(serde_json::json!([cam(), {"name":"n","animation":[{"tracks":[{"path":["active"],"keys":[[0,true],[1,false]]}]}]}]), serde_json::json!([]));
        assert!(open(own_active).unwrap_err().contains("the clock cannot key itself"));
        let grandparent = scene(serde_json::json!([cam(), {"name":"g","animation":[{"tracks":[{"path":["children","p","children","c","active"],"keys":[[0,true]]}]}],
            "children":[{"name":"p","children":[{"name":"c"}]}]}]), serde_json::json!([]));
        assert!(open(grandparent).unwrap_err().contains("keyed by the parent or the scene"));
        let parent = scene(serde_json::json!([cam(), {"name":"p","animation":[{"tracks":[{"path":["children","c","active"],"keys":[[0,true],[2,false]]}]}],
            "children":[{"name":"c"}]}]), serde_json::json!([]));
        assert!(open(parent).is_ok());
        let window = scene(serde_json::json!([cam(), {"name":"n","active":[0,2]}]),
            serde_json::json!([{"tracks":[{"path":["nodes","n","active"],"keys":[[0,true],[1,false]]}]}]));
        assert!(open(window).unwrap_err().contains("active is a window, a constant"));
        let no_time = scene(serde_json::json!([cam(), {"name":"n"}]), serde_json::json!([{"tracks":[{"path":["nodes","n","time","speed"],"keys":[[0,1],[1,2]]}]}]));
        assert!(open(no_time).unwrap_err().contains("time.speed keys a node with a time"));
        let reversed = scene(serde_json::json!([cam(), {"name":"n","time":{"speed":-1,"duration":2}}]), serde_json::json!([{"tracks":[{"path":["nodes","n","time","speed"],"keys":[[0,1],[1,2]]}]}]));
        assert!(open(reversed).unwrap_err().contains("a negative speed is a constant"));
        let start = scene(serde_json::json!([cam(), {"name":"n","time":{}}]), serde_json::json!([{"tracks":[{"path":["nodes","n","time","start"],"keys":[[0,1]]}]}]));
        assert!(open(start).unwrap_err().contains("time keys speed only"));
        // The fresh agent's spelling for a fade: refused with the scene-level path.
        let material = scene(serde_json::json!([cam(), {"name":"n","animation":[{"tracks":[{"path":["materials","0","inputs","geometry_opacity"],"keys":[[0,0],[1,1]]}]}]}]), serde_json::json!([]));
        assert!(open(material).unwrap_err().contains("materials are keyed from the scene's animation"));
    }

    #[test]
    fn clocks_are_keyed_in_plain_layers() {
        let speed = serde_json::json!([{"path":["nodes","n","time","speed"],"keys":[[0,1],[1,2]]}]);
        let line = "active and time.speed are keyed in a plain layer";
        for layer in [serde_json::json!({"speed":2}), serde_json::json!({"loop":true}), serde_json::json!({"weight":0.5}), serde_json::json!({"name":"w"})] {
            let mut layer = layer;
            layer["tracks"] = speed.clone();
            let mut animation = vec![layer];
            // A layer's weight is keyed from an earlier layer.
            if animation[0]["name"] == "w" { animation.insert(0, serde_json::json!({"tracks":[{"path":["animation","w","weight"],"keys":[[0,1],[1,0]]}]})); }
            let error = open(scene(serde_json::json!([cam(), {"name":"n","time":{}}]), serde_json::Value::Array(animation))).unwrap_err();
            assert!(error.contains(line), "{error}");
        }
        let owned = scene(serde_json::json!([cam(), {"name":"p","children":[{"name":"c"}],
            "animation":[{"offset":1,"tracks":[{"path":["children","c","active"],"keys":[[0,true],[1,false]]}]}]}]), serde_json::json!([]));
        assert!(open(owned).unwrap_err().contains(line));
        assert!(open(scene(serde_json::json!([cam(), {"name":"n","time":{}}]), serde_json::json!([{"tracks":speed}]))).is_ok());
    }

    #[test]
    fn keyed_speeds_stay_nonnegative_everywhere_on_the_curve() {
        let dip = |keys: serde_json::Value, interpolation: &str| scene(serde_json::json!([cam(), {"name":"n","time":{}}]),
            serde_json::json!([{"tracks":[{"path":["nodes","n","time","speed"],"keys":keys,"interpolation":interpolation}]}]));
        assert!(open(dip(serde_json::json!([[0,1],[1,-1]]), "LINEAR")).unwrap_err().contains("a negative speed is a constant"));
        // xtimerev 3: an exact negative that f64 rounds to -0.0 is still negative.
        let tiny = format!("-1/1{}", "0".repeat(400));
        assert!(open(dip(serde_json::json!([[0,1],[1,tiny]]), "LINEAR")).unwrap_err().contains("a negative speed is a constant"));
        // xtimerev 17: rational spellings keep numeric (LINEAR) interpolation.
        let rational = scene(serde_json::json!([cam(), {"name":"n","time":{}}]), serde_json::json!([{"tracks":[{"path":["nodes","n","time","speed"],"keys":[[0,"1/1"],[2,"3/1"]]}]}]));
        let opened = open(rational).unwrap();
        assert!(matches!(opened.animation[0].tracks.as_ref().unwrap()[0].interpolation, Interpolation::Linear));
        // Equal end values with tangents that dip the Hermite segment below zero.
        assert!(open(dip(serde_json::json!([[0,0,0,-10],[1,0,-10,0]]), "CUBICSPLINE")).unwrap_err().contains("a negative speed is a constant"));
        // xtimerev 3: far key times that collapse to one f32 still see the dip.
        assert!(open(dip(serde_json::json!([[100000000,1,0,-10],[100000001,1,10,0]]), "CUBICSPLINE")).unwrap_err().contains("a negative speed is a constant"));
        assert!(open(dip(serde_json::json!([[0,1],[2,1],[4,2],[10,2]]), "LINEAR")).is_ok());
        assert!(open(dip(serde_json::json!([[0,1,0,1],[1,2,1,0]]), "CUBICSPLINE")).is_ok());
    }

    #[test]
    fn scene_owned_speed_needs_constant_nonzero_clocks_above() {
        let under_keyed = scene(serde_json::json!([cam(), {"name":"p","time":{},"children":[{"name":"c","time":{}}]}]),
            serde_json::json!([{"tracks":[{"path":["nodes","p","time","speed"],"keys":[[0,1],[1,2]]},{"path":["nodes","c","time","speed"],"keys":[[0,1],[1,2]]}]}]));
        assert!(open(under_keyed).unwrap_err().contains("key it from the parent"));
        let under_zero = scene(serde_json::json!([cam(), {"name":"p","time":{"speed":0},"children":[{"name":"c","time":{}}]}]),
            serde_json::json!([{"tracks":[{"path":["nodes","c","time","speed"],"keys":[[0,1],[1,2]]}]}]));
        assert!(open(under_zero).unwrap_err().contains("key it from the parent"));
        let under_constant = scene(serde_json::json!([cam(), {"name":"p","time":{"speed":2},"children":[{"name":"c","time":{}}]}]),
            serde_json::json!([{"tracks":[{"path":["nodes","c","time","speed"],"keys":[[0,1],[1,2]]}]}]));
        assert!(open(under_constant).is_ok());
    }
}

#[cfg(test)]
mod style_target_tests {
    use super::*;
    use serde_json::json;
    fn scene(node:serde_json::Value,field:&str,a:serde_json::Value,b:serde_json::Value)->Result<Scene> {
        Scene::from_json(&serde_json::to_vec(&json!({"version":2,"composition":{"space":"2d"},"nodes":[node],
            "animation":[{"tracks":[{"path":["nodes","n",field],"keys":[[0,a],[1,b]]}]}]}))?)
    }
    #[test]
    fn all_path_styles_share_the_target_table() {
        for (field,a,b) in [
            ("d",json!("M0 0H10V10Z"),json!("M0 0H20V30Z")),
            ("filter",json!("none"),json!("blur(8px)")),
            ("fill-rule",json!("nonzero"),json!("evenodd")),
            ("stroke-linecap",json!("butt"),json!("round")),
            ("stroke-linejoin",json!("miter"),json!("bevel")),
            ("stroke-miterlimit",json!(2),json!(5)),
            ("stroke-dasharray",json!([2,4]),json!([4,6,8])),
            ("pathLength",json!(10),json!(20)),
            ("paint-order",json!("normal"),json!("stroke")),
            ("fill",json!("linear-gradient(red, blue)"),json!("linear-gradient(white, black)")),
        ] {
            let s=scene(json!({"type":"path","name":"n"}),field,a,b).unwrap();
            assert!(s.track_targets().unwrap()[0][0].style_interpolation().is_some(),"{field}");
            assert!(matches!(s.animation[0].tracks.as_ref().unwrap()[0].interpolation,Interpolation::Linear));
        }
    }
    #[test]
    fn text_styles_key_and_unbuilt_values_still_refuse() {
        let text=json!({"type":"text","name":"n","font":"font.ttf","content":"Hi"});
        for (field,a,b) in [("font-size",json!(12),json!(24)),("font-weight",json!("normal"),json!("bold")),
            ("font",json!("a.ttf"),json!("b.ttf")),("content",json!("Hi"),json!("Hello")),
            ("letter-spacing",json!(0),json!(2)),("width",json!(null),json!(100)),
            ("line-height",json!("normal"),json!(2)),("font-variation-settings",json!("'wght' 100"),json!("'wght' 900")),
            ("text-align",json!("left"),json!("right")),
            ("fill",json!("linear-gradient(red, blue)"),json!("linear-gradient(blue, red)")),
            ("stroke",json!("linear-gradient(red, blue)"),json!("linear-gradient(blue, red)"))] {
            scene(text.clone(),field,a,b).unwrap();
        }
        for (field,b) in [("font-size",json!(-1)),("word-spacing",json!(2)),("width",json!(-1)),("font-weight",json!(1001))] {
            assert!(scene(text.clone(),field,b.clone(),b).is_err(),"{field}");
        }
        assert!(scene(text,"d",json!("M0 0"),json!("M1 1")).is_err());
    }
}
