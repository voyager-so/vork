use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Node {
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.name", "x-help" = "string; globally unique, including imported nodes"))]
    pub name: String,
    /// glTF node TRS, written directly on the node (`translation`, `rotation`, `scale`).
    #[serde(default, flatten)]
    pub transform: Transform,
    #[schemars(extend("x-source" = "Blender Track To constraint (vork target shorthand)", "x-help-path" = "node.target", "x-help" = "node-name|[x,y,z]|null; aims -Z, world Y up; rotation is an offset on top; structural, not animatable: aim at a named node and animate its translation"))]
    pub target: Option<AimTarget>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.children", "x-help" = "node[]"))]
    pub children: Vec<Node>,
    #[serde(default, flatten)]
    pub visibility: Visibility,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.light_link_tags", "x-help" = "string[]; matched by light illuminates/shadowed_by; inherited"))]
    pub light_link_tags: Vec<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.pass_index", "x-help" = "u32|null; 1..65535; object ID"))]
    pub pass_index: Option<u32>,
    /// Cycles holdout: transparent black on transparent film; opaque black otherwise.
    #[serde(default)]
    #[schemars(extend("x-source" = "Blender Cycles Object.is_holdout", "x-help-path" = "node.is_holdout", "x-help" = "bool; RGB 0; alpha 0 on transparent background, 1 on opaque background"))]
    pub is_holdout: bool,
    /// Cycles shadow catcher. Combined treats this surface as a holdout.
    #[serde(default)]
    #[schemars(extend("x-source" = "Blender Cycles Object.is_shadow_catcher", "x-help-path" = "node.is_shadow_catcher", "x-help" = "bool; shadow_catcher channel is material light with CG / without CG, unclamped; zero baseline gives neutral 1; CG light on an otherwise unlit catcher cannot be represented"))]
    pub is_shadow_catcher: bool,
    /// 2D canvas only: the node's whole picture, children included, placed once (CSS opacity).
    #[serde(default = "canvas::one", skip_serializing_if = "canvas::is_one", deserialize_with = "super::numeric")]
    #[schemars(extend("default" = 1.0, "x-source" = "CSS opacity", "x-help-path" = "node.opacity", "x-help" = "number; 2D; 0..1; the node's picture, children included, placed once"))]
    pub opacity: f32,
    /// 2D canvas only: the named node's shape clips this node, drawn in this node's coordinates (SVG
    /// clip-path, userSpaceOnUse).
    #[serde(default, rename = "clip-path", skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "SVG 2 / CSS Masking clip-path", "x-help-path" = "node.clip-path", "x-help" = "node-name|null; 2D; that node's shape, in this node's coordinates, clips this node; it is not painted itself"))]
    pub clip_path: Option<String>,
    /// 2D canvas only: the named node's picture masks this node by its luminance, drawn in this node's
    /// coordinates (SVG mask, userSpaceOnUse).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "SVG 2 / CSS Masking mask (luminance)", "x-help-path" = "node.mask", "x-help" = "node-name|null; 2D; that node's picture, in this node's coordinates, masks this node by luminance; it is not painted itself"))]
    pub mask: Option<String>,
    /// 2D canvas only: CSS filter functions over the node's picture, in order.
    /// Layer blur refuses sigma above 100 target pixels after accumulated transforms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "CSS Filter Effects filter", "x-help-path" = "node.filter", "x-help" = "string|null; 2D; blur() and drop-shadow(); nonzero lengths require px; blur sigma above 100 target pixels after transforms refuses"))]
    pub filter: Option<String>,
    /// Sound source and gain over this node's complete subtree mix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "Web Audio (gain); vork sound", "x-help-path" = "node.audio", "x-help" = "{file,gain}|null; file relative to scene; gain defaults to 1 and scales source plus children; active mutes, visible does not; node time controls playback and pitch"))]
    pub audio: Option<Audio>,
    /// When the node exists, in parent seconds. Off: no picture, no sound, no evaluation, children included.
    #[serde(default, skip_serializing_if = "Active::is_default")]
    #[schemars(extend("x-source" = "vork (node existence; Unity active)", "x-help-path" = "node.active", "x-help" = "true|false|[start,end); parent seconds; false removes picture and sound, children included; a window starts the clock at start; a boolean is keyable (step)"))]
    pub active: Active,
    /// The clock for node-owned animation and media: local = offset + (parent - start) * speed, wrapped or held in the trimmed section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork (node clock)", "x-help-path" = "node.time", "x-help" = "{start,offset,duration,speed,loop}|null; node-owned animation/media clock; time-reading programs, conflicting movie clocks and scene-time tracks inside retimed compositions refuse; null inherits parent/active-window clock"))]
    pub time: Option<NodeClock>,
    /// Layers evaluated on this node's clock; paths start at this node.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(extend("x-source" = "vork (node-owned animation)", "x-help-path" = "node.animation", "x-help" = "layer[]; tracks on this node's clock in content seconds; paths start here; a property has one owner; the clock cannot key itself"))]
    pub animation: Vec<NodeLayer>,
    /// `{space: "2d"}`: a canvas whose children paint to one picture; `{space: "3d"}`, in a canvas: a closed
    /// 3D world rendered to one picture and placed in the canvas (END-STATE-API §7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "END-STATE-API §7", "x-help-path" = "node.composition", "x-help" = "composition|null; 2d: a canvas, its children painted to one picture; in a 3D world seen through an image's render.root; 3d: in a canvas, a closed world drawn through its one camera to one picture the size of the canvas's box"))]
    pub composition: Option<Composition>,
    #[serde(flatten)]
    pub kind: NodeKind,
}
// Keep the strict derived variant fields; omission alone selects a plain group.
impl<'de> Deserialize<'de> for NodeKind {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = serde_json::Value::deserialize(deserializer)?;
        // Node-kind values are numeric program inputs, not scene clocks.
        super::lower_numeric_json(&mut value).map_err(serde::de::Error::custom)?;
        if let Some(object) = value.as_object_mut() {
            object.entry("type").or_insert_with(|| serde_json::json!("node"));
        }
        NodeKind::deserialize(value).map_err(serde::de::Error::custom)
    }
}
impl Serialize for NodeKind {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        NodeKind::serialize(self, serializer)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
/// Visibility is ANDed with ancestors. Reflections and diffuse transport query
/// scene geometry independently of camera visibility. Diffuse visibility controls AO
/// and indirect occlusion; specular visibility controls reflected appearance.
pub struct Visibility {
    #[schemars(extend("x-source" = "vork visibility primitive", "x-help-path" = "node.visible", "x-help" = "bool; false hides the subtree and disables its lights; animatable (step)"))]
    pub visible: bool,
    #[schemars(extend("x-source" = "Blender Object.visible_camera", "x-help-path" = "node.visible_camera", "x-help" = "bool"))]
    #[serde(rename = "visible_camera")]
    pub camera: bool,
    #[schemars(extend("x-source" = "Blender Object.visible_shadow", "x-help-path" = "node.visible_shadow", "x-help" = "bool; lights must stay true (a light without shadows: shadowed_by [])"))]
    #[serde(rename = "visible_shadow")]
    pub shadow: bool,
    #[schemars(extend("x-source" = "Blender Object.visible_diffuse", "x-help-path" = "node.visible_diffuse", "x-help" = "bool"))]
    #[serde(rename = "visible_diffuse")]
    pub diffuse: bool,
    #[schemars(extend("x-source" = "Blender Object.visible_glossy", "x-help-path" = "node.visible_glossy", "x-help" = "bool"))]
    #[serde(rename = "visible_glossy")]
    pub glossy: bool,
    #[schemars(extend("x-source" = "Blender Object.visible_transmission", "x-help-path" = "node.visible_transmission", "x-help" = "bool; object visibility to transmission rays; lights must stay true"))]
    #[serde(rename = "visible_transmission")]
    pub transmission: bool,
}
impl Default for Visibility {
    fn default() -> Self {
        Self {
            visible: true,
            camera: true,
            shadow: true,
            diffuse: true,
            glossy: true,
            transmission: true,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(remote = "Self", tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[schemars(extend("x-help-path" = "node", "x-help-extra" = "node.type: node|mesh|asset|camera|light|text|path; default \"node\"\nasset (glTF cameras): addressed as <asset>/<node-name|#index>"))]
pub enum NodeKind {
    #[schemars(title = "TransformNode")]
    Node {},
    #[schemars(title = "MeshNode")]
    Mesh {
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.mesh.geometry", "x-help" = "geometry-name|inline geometry"))]
        #[schemars(with = "NameOr<Geometry>")]
        geometry: String,
        #[serde(default)]
        #[schemars(extend("x-source" = "OpenPBR Surface 1.1.1", "x-help-path" = "node.mesh.materials", "x-help" = "(material-name|inline material)[]; empty = default std/openpbr with roughness 0.3"))]
        #[schemars(with = "Vec<NameOr<Material>>")]
        materials: Vec<String>,
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.mesh.instances", "x-help" = "instances|null"))]
        instances: Option<Instances>,
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.mesh.skin", "x-help" = "skin|null"))]
        skin: Option<Skin>,
        #[serde(default)]
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.mesh.morph_weights", "x-help" = "{name:number}"))]
        morph_weights: Table<f32>,
    },
    #[schemars(title = "AssetNode")]
    Asset {
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.asset.file", "x-help" = "file-path"))]
        file: String,
        #[serde(default)]
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.asset.materials", "x-help" = "{imported-name|#index|*: material-name|inline material}; #index = glTF material index; specific beats *"))]
        #[schemars(with = "Table<NameOr<Material>>")]
        materials: Table<String>,
        #[serde(default)]
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.asset.nodes", "x-help" = "{imported-name|#index: overrides}; #index = glTF node index"))]
        nodes: Table<AssetNodeOverrides>,
    },
    #[schemars(title = "CameraNode")]
    Camera {
        #[serde(default)]
        #[schemars(extend("x-source" = "vork (programmable camera and sensor)", "x-help-path" = "camera.material", "x-help" = "material-name|inline material|null; omitted = std/camera; optics are program inputs"))]
        #[schemars(with = "Option<NameOr<Material>>")]
        material: Option<String>,
        #[serde(default)]
        #[schemars(extend("x-source" = "Blender CameraDOFSettings.focus_object", "x-help-path" = "camera.focus_object", "x-help" = "node-name|null; camera-axis depth supplied to the program at sample time"))]
        focus_object: Option<String>,
    },
    #[schemars(title = "LightNode")]
    Light {
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "light.shape", "x-help" = "sun|point|spot|rectangle|ellipse|environment"))]
        shape: LightShape,
        /// None: LightShape::default_color (white, or Blender's world colour for an
        /// environment without an image).
        #[schemars(extend("x-source" = "Blender 5.2.1 Light.color (Add Light default)", "x-help-path" = "light.color", "x-help" = "RGB"))]
        color: Option<[f32; 3]>,
        /// None: `color` alone. Some(kelvin): `color` × std.color blackbody(kelvin).
        #[schemars(extend("x-source" = "Blender 5.2.1 Light.temperature with use_temperature", "x-help-path" = "light.temperature", "x-help" = "number|null; K; multiplies color; not environment"))]
        temperature: Option<f32>,
        #[schemars(extend("x-source" = "Blender 5.2.1 Light.energy (Add Light default)", "x-help-path" = "light.energy", "x-help" = "number|null; sun W/m²; local W; environment multiplier"))]
        energy: Option<f32>,
        #[serde(default = "constants::sun_angle")]
        #[schemars(extend("x-source" = "Blender 5.2.1 SunLight.angle (Add Light default)", "x-help-path" = "light.angle", "x-help" = "number; sun full angular diameter in radians; 0 = sharp shadows"))]
        angle: f32,
        #[serde(default)]
        #[schemars(extend("x-source" = "Blender 5.2.1 Light.shadow_soft_size (Add Light default)", "x-help-path" = "light.radius", "x-help" = "number; point/spot emitter radius (shadow softness)"))]
        radius: f32,
        #[serde(default = "constants::spot_size")]
        #[schemars(extend("x-source" = "Blender 5.2.1 SpotLight.spot_size", "x-help-path" = "light.spot_size", "x-help" = "number; full cone angle"))]
        spot_size: f32,
        #[serde(default = "constants::spot_blend")]
        #[schemars(extend("x-source" = "Blender 5.2.1 SpotLight.spot_blend", "x-help-path" = "light.spot_blend", "x-help" = "number; 0..1"))]
        spot_blend: f32,
        #[serde(default = "constants::area_dimension")]
        #[schemars(extend("x-source" = "Blender 5.2.1 AreaLight.size (Add Light default)", "x-help-path" = "light.size", "x-help" = "number; only rectangle or ellipse lights; rectangle full side, ellipse diameter; size_y sets the second axis"))]
        size: f32,
        #[schemars(extend("x-source" = "Blender 5.2.1 AreaLight square/disk sizing", "x-help-path" = "light.size_y", "x-help" = "number|null; only rectangle or ellipse lights; second side/diameter; omitted = size (square/disk)"))]
        size_y: Option<f32>,
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "light.image", "x-help" = "image-name|inline image|null; LM-63 Type C IES (path only); non-environment lights; emits -Z, no peak normalization"))]
        #[schemars(with = "Option<NameOr<Image>>")]
        image: Option<String>,
        /// Physical world fog material on an environment light.
        #[schemars(extend("x-source" = "vork (physical medium)", "x-help-path" = "light.medium", "x-help" = "material-name|inline material|null; environment only; authored medium program for world fog"))]
        #[schemars(with = "Option<NameOr<Material>>")]
        medium: Option<String>,
        /// Environment radiance material instance.
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "light.material", "x-help" = "material-name|inline material|null; environment program; omitted = uniform color × energy radiance"))]
        #[schemars(with = "Option<NameOr<Material>>")]
        material: Option<String>,
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "light.illuminates", "x-help" = "light-link-tag[]|null; receivers with any tag; null=all; []=none"))]
        illuminates: Option<Vec<String>>,
        #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "light.shadowed_by", "x-help" = "light-link-tag[]|null; casters with any tag; null=all; []=none"))]
        shadowed_by: Option<Vec<String>>,
    },
    /// 2D canvas only (END-STATE-API §7, text): CSS's names for the properties of text; the origin
    /// is the top-left of the text box; the glyph outlines take SVG's fill, stroke and stroke-width.
    #[schemars(title = "TextNode")]
    Text {
        #[schemars(extend("x-source" = "END-STATE-API §7 (content)", "x-help-path" = "node.content (text)", "x-help" = "string|run[]; position by the text box's top-left; switch content with STEP keys; runs set font and paint; white-space controls newlines"))]
        content: Content,
        #[schemars(extend("x-source" = "END-STATE-API §7 (a font file or a list of files)", "x-help-path" = "node.font (text)", "x-help" = "file-path|file-path[]; .ttf/.otf; a list offers faces font-weight and font-style choose from"))]
        font: FontFiles,
        #[serde(rename = "font-size", default = "text::font_size")]
        #[schemars(extend("default" = 16.0, "x-source" = "CSS font-size", "x-help-path" = "node.font-size (text)", "x-help" = "number; px"))]
        font_size: f32,
        #[serde(rename = "font-weight", default)]
        #[schemars(extend("default" = "normal", "x-source" = "CSS font-weight", "x-help-path" = "node.font-weight (text)", "x-help" = "number|normal|bold"))]
        font_weight: FontWeight,
        #[serde(rename = "font-style", default)]
        #[schemars(extend("x-source" = "CSS font-style", "x-help-path" = "node.font-style (text)", "x-help" = "normal|italic|oblique"))]
        font_style: FontStyle,
        #[serde(rename = "font-variation-settings", default = "text::normal", skip_serializing_if = "text::is_normal")]
        #[schemars(extend("default" = "normal", "x-source" = "CSS font-variation-settings", "x-help-path" = "node.font-variation-settings (text)", "x-help" = "string"))]
        font_variation_settings: String,
        #[serde(rename = "font-feature-settings", default = "text::normal", skip_serializing_if = "text::is_normal")]
        #[schemars(extend("default" = "normal", "x-source" = "CSS font-feature-settings", "x-help-path" = "node.font-feature-settings (text)", "x-help" = "string"))]
        font_feature_settings: String,
        #[serde(rename = "letter-spacing", default)]
        #[schemars(extend("x-source" = "CSS letter-spacing", "x-help-path" = "node.letter-spacing (text)", "x-help" = "number; px"))]
        letter_spacing: f32,
        #[serde(rename = "word-spacing", default)]
        #[schemars(extend("x-source" = "CSS word-spacing", "x-help-path" = "node.word-spacing (text)", "x-help" = "number; px; only 0 is built"))]
        word_spacing: f32,
        #[serde(rename = "line-height", default)]
        #[schemars(extend("default" = "normal", "x-source" = "CSS line-height", "x-help-path" = "node.line-height (text)", "x-help" = "normal|number; a multiple of the font size"))]
        line_height: LineHeight,
        #[serde(rename = "text-align", default)]
        #[schemars(extend("x-source" = "CSS text-align", "x-help-path" = "node.text-align (text)", "x-help" = "start|end|left|right|center|justify"))]
        text_align: TextAlign,
        #[serde(default)]
        #[schemars(extend("x-source" = "CSS direction", "x-help-path" = "node.direction (text)", "x-help" = "ltr|rtl"))]
        direction: Direction,
        #[serde(rename = "white-space", default)]
        #[schemars(extend("x-source" = "CSS white-space", "x-help-path" = "node.white-space (text)", "x-help" = "normal|nowrap|pre-line; pre, pre-wrap and break-spaces are not built"))]
        white_space: WhiteSpace,
        #[serde(rename = "word-break", default)]
        #[schemars(extend("x-source" = "CSS word-break", "x-help-path" = "node.word-break (text)", "x-help" = "normal; the others are not built"))]
        word_break: WordBreak,
        #[serde(rename = "line-break", default)]
        #[schemars(extend("x-source" = "CSS line-break", "x-help-path" = "node.line-break (text)", "x-help" = "auto; the others are not built"))]
        line_break: LineBreak,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[schemars(extend("x-source" = "END-STATE-API §7 (width to wrap)", "x-help-path" = "node.width (text)", "x-help" = "number|null; px; lines wrap and align in it; null = no wrapping"))]
        width: Option<f32>,
        #[serde(default = "canvas::black")]
        #[schemars(extend("default" = "#000", "x-source" = "SVG 2 fill; CSS Color 4", "x-help-path" = "node.fill (text)", "x-help" = "\"none\"|CSS colour|CSS gradient|{image, fit}"))]
        fill: Paint,
        #[serde(default = "canvas::no_paint")]
        #[schemars(extend("default" = "none", "x-source" = "SVG 2 stroke; CSS Color 4", "x-help-path" = "node.stroke (text)", "x-help" = "\"none\"|CSS colour|CSS gradient|{image, fit}; the glyph outlines"))]
        stroke: Paint,
        #[serde(rename = "stroke-width", default = "text::one")]
        #[schemars(extend("default" = 1.0, "x-source" = "SVG 2 stroke-width", "x-help-path" = "node.stroke-width (text)", "x-help" = "number; px"))]
        stroke_width: f32,
    },
    /// 2D canvas only (END-STATE-API §7, path): SVG 2's path data and paint properties, with SVG's
    /// names and meanings; colours are CSS colours.
    #[schemars(title = "PathNode")]
    Path {
        #[serde(default = "canvas::no_path")]
        #[schemars(extend("default" = "none", "x-source" = "SVG 2 path d", "x-help-path" = "node.d (path)", "x-help" = "string; SVG path data, px; \"none\" or empty paints nothing"))]
        d: String,
        #[serde(default = "canvas::black")]
        #[schemars(extend("default" = "#000", "x-source" = "SVG 2 fill; CSS colour and gradient", "x-help-path" = "node.fill (path)", "x-help" = "paint; \"none\"|CSS colour|CSS linear-/radial-/conic-gradient()|{image, fit}"))]
        fill: Paint,
        #[serde(rename = "fill-rule", default)]
        #[schemars(extend("x-source" = "SVG 2 fill-rule", "x-help-path" = "node.fill-rule (path)", "x-help" = "nonzero|evenodd"))]
        fill_rule: FillRule,
        #[serde(rename = "fill-opacity", default = "canvas::one")]
        #[schemars(extend("default" = 1.0, "x-source" = "SVG 2 fill-opacity", "x-help-path" = "node.fill-opacity (path)", "x-help" = "number; 0..1"))]
        fill_opacity: f32,
        #[serde(default = "canvas::no_paint")]
        #[schemars(extend("default" = "none", "x-source" = "SVG 2 stroke; CSS colour and gradient", "x-help-path" = "node.stroke (path)", "x-help" = "paint; \"none\"|CSS colour|CSS linear-/radial-/conic-gradient()|{image, fit}"))]
        stroke: Paint,
        #[serde(rename = "stroke-width", default = "constants::stroke_width")]
        #[schemars(extend("default" = 1.0, "x-source" = "SVG 2 stroke-width", "x-help-path" = "node.stroke-width (path)", "x-help" = "number; px; 0 paints nothing"))]
        stroke_width: f32,
        #[serde(rename = "stroke-linecap", default)]
        #[schemars(extend("x-source" = "SVG 2 stroke-linecap", "x-help-path" = "node.stroke-linecap (path)", "x-help" = "butt|round|square"))]
        stroke_linecap: LineCap,
        #[serde(rename = "stroke-linejoin", default)]
        #[schemars(extend("x-source" = "SVG 2 stroke-linejoin", "x-help-path" = "node.stroke-linejoin (path)", "x-help" = "miter|round|bevel"))]
        stroke_linejoin: LineJoin,
        #[serde(rename = "stroke-miterlimit", default = "constants::miter_limit")]
        #[schemars(extend("default" = 4.0, "x-source" = "SVG 2 stroke-miterlimit", "x-help-path" = "node.stroke-miterlimit (path)", "x-help" = "number; at least 1"))]
        stroke_miterlimit: f32,
        #[serde(rename = "stroke-dasharray", default)]
        #[schemars(extend("default" = "none", "x-source" = "SVG 2 stroke-dasharray", "x-help-path" = "node.stroke-dasharray (path)", "x-help" = "\"none\"|number[]; px; an odd count repeats; scaled by pathLength"))]
        stroke_dasharray: DashArray,
        #[serde(rename = "stroke-dashoffset", default)]
        #[schemars(extend("x-source" = "SVG 2 stroke-dashoffset", "x-help-path" = "node.stroke-dashoffset (path)", "x-help" = "number; px"))]
        stroke_dashoffset: f32,
        #[serde(rename = "stroke-opacity", default = "canvas::one")]
        #[schemars(extend("default" = 1.0, "x-source" = "SVG 2 stroke-opacity", "x-help-path" = "node.stroke-opacity (path)", "x-help" = "number; 0..1"))]
        stroke_opacity: f32,
        #[serde(rename = "pathLength", default, skip_serializing_if = "Option::is_none")]
        #[schemars(extend("x-source" = "SVG 2 pathLength", "x-help-path" = "node.pathLength (path)", "x-help" = "number|null; the author's total length of d; dash lengths and offset scale by computed/pathLength"))]
        path_length: Option<f32>,
        #[serde(rename = "paint-order", default)]
        #[schemars(with = "String", extend("default" = "normal", "x-source" = "SVG 2 paint-order", "x-help-path" = "node.paint-order (path)", "x-help" = "string; normal, or the layers to paint first out of fill, stroke, markers (\"stroke\" puts the stroke under the fill)"))]
        paint_order: PaintOrder,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(extend("x-help-extra" = "node.asset.nodes.<name>.visible: bool; default true\nnode.asset.nodes.<name>.visible_camera: bool; default true\nnode.asset.nodes.<name>.visible_shadow: bool; default true\nnode.asset.nodes.<name>.visible_diffuse: bool; default true\nnode.asset.nodes.<name>.visible_glossy: bool; default true\nnode.asset.nodes.<name>.visible_transmission: bool; default true"))]
pub struct AssetNodeOverrides {
    #[serde(default, flatten)]
    pub visibility: Option<Visibility>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.asset.nodes.<name>.light_link_tags", "x-help" = "string[]|null; replaces inherited tags"))]
    pub light_link_tags: Option<Vec<String>>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "node.asset.nodes.<name>.pass_index", "x-help" = "u32|null; 1..65535; object ID"))]
    pub pass_index: Option<u32>,
}
/// Aiming references a node origin or a fixed point in world coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum AimTarget {
    Node(String),
    Point(#[serde(deserialize_with="super::numeric")] [f32; 3]),
}
/// Translation, rotation and scale (glTF node TRS), flat on the node. Rotation is
/// stored as a quaternion; an authored `{"euler": [x,y,z]}` converts losslessly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Transform {
    #[schemars(extend("x-source" = "glTF 2.0", "x-help-path" = "node.translation", "x-help" = "[x,y,z]; parent frame"))]
    #[serde(deserialize_with="super::numeric")]
    pub translation: [f32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "authored_rotation")]
    #[schemars(with = "Rotation", extend("default" = [0., 0., 0., 1.]))]
    #[schemars(extend("x-source" = "glTF 2.0 (quaternion); Blender XYZ Euler", "x-help-path" = "node.rotation", "x-help" = "[x,y,z,w] quaternion | {euler:[x,y,z]} (Blender XYZ)"))]
    // Preserve omission so targets can reject even an authored identity.
    pub rotation: Option<[f32; 4]>,
    #[schemars(extend("x-source" = "glTF 2.0", "x-help-path" = "node.scale", "x-help" = "[x,y,z]; any zero world-scale component hides geometry, including children"))]
    #[serde(deserialize_with="super::numeric")]
    pub scale: [f32; 3],
}
/// An authored rotation: a glTF quaternion or a tagged Euler triple.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum Rotation {
    Quaternion(#[serde(deserialize_with="super::numeric")] [f32; 4]),
    Euler {
        /// Radians about X, then Y, then Z (fixed parent axes; Blender XYZ).
        #[serde(deserialize_with="super::numeric")]
        euler: [f32; 3],
    },
}
impl Rotation {
    /// XYZW quaternion; Euler XYZ is R = Rz * Ry * Rx.
    pub fn quaternion(self) -> [f32; 4] {
        match self {
            Self::Quaternion(q) => q,
            Self::Euler { euler: [x, y, z] } => (glam::Quat::from_rotation_z(z)
                * glam::Quat::from_rotation_y(y)
                * glam::Quat::from_rotation_x(x)).to_array(),
        }
    }
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.; 3],
            rotation: None,
            scale: [1.; 3],
        }
    }
}
fn authored_rotation<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<[f32; 4]>, D::Error> {
    Rotation::deserialize(deserializer).map(|r| Some(r.quaternion())).map_err(|_| serde::de::Error::custom(
        "rotation must be [x,y,z,w] (unit quaternion) or {\"euler\": [x,y,z]} (radians, X then Y then Z)"))
}
impl Transform {
    pub fn resolved_rotation(&self) -> [f32; 4] {
        self.rotation.unwrap_or([0., 0., 0., 1.])
    }
    pub fn rotation_mut(&mut self) -> &mut [f32; 4] {
        self.rotation.get_or_insert([0., 0., 0., 1.])
    }
    pub fn matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            self.scale.into(),
            glam::Quat::from_array(self.resolved_rotation()).normalize(),
            self.translation.into(),
        )
    }
}
/// Testing aid: replace a node document's whole transform (`translation`, `rotation`, `scale`).
#[doc(hidden)]
pub fn set_transform(node: &mut serde_json::Value, trs: serde_json::Value) {
    let Some(node) = node.as_object_mut() else { return };
    for key in ["translation", "rotation", "scale"] { node.remove(key); }
    node.extend(trs.as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.clone())));
}
impl NodeKind {
    /// Whether this light permits shadow casters.
    pub fn casts_shadows(&self) -> bool {
        matches!(self, Self::Light { shadowed_by, .. } if !shadowed_by.as_ref().is_some_and(Vec::is_empty))
    }
}
impl Node {
    /// Write an authored value into one field of this node: a style field a track keys (`fill`, `font-size`,
    /// `d`, `filter`, ...). The value is read as that field reads it from a scene document, numbers included,
    /// and nothing else of the node is touched or read again. `field` is the authored name. A field the node's
    /// kind does not have, or a value the field does not take, is refused and leaves the node as it was.
    pub fn set_field(&mut self, field: &str, value: &serde_json::Value) -> std::result::Result<(), serde_json::Error> {
        // A field of a node kind, read as the kind's reader reads it: the kind's numbers are lowered to `f64`
        // (`NodeKind`'s `Deserialize`), and the kind is a tagged record, whose fields serde reads from a buffer
        // in which a number is already an `f64` or an integer. An `f32` field is therefore that `f64` narrowed,
        // not the nearest `f32` to the number's text; the two differ at a tie. Reading the value as the one
        // field of a tagged record here goes through the same buffer.
        #[derive(Deserialize)]
        #[serde(tag = "record")]
        enum Tagged<T> {
            Field { value: T },
        }
        fn read<T: serde::de::DeserializeOwned>(value: &serde_json::Value) -> std::result::Result<T, serde_json::Error> {
            let mut value = value.clone();
            super::lower_numeric_json(&mut value).map_err(<serde_json::Error as serde::de::Error>::custom)?;
            let mut record = serde_json::Map::new();
            record.insert("record".into(), "Field".into());
            record.insert("value".into(), value);
            let Tagged::Field { value } = serde_json::from_value::<Tagged<T>>(serde_json::Value::Object(record))?;
            Ok(value)
        }
        macro_rules! fields {
            ($($name:literal => $slot:ident),* $(,)?) => {
                match field {
                    $($name => *$slot = read(value)?,)*
                    _ => return Err(<serde_json::Error as serde::de::Error>::custom(format!("unknown field `{field}`"))),
                }
            };
        }
        match field {
            // The node's own fields are read from the document as they stand.
            "filter" => self.filter = serde_json::from_value(value.clone())?,
            "clip-path" => self.clip_path = serde_json::from_value(value.clone())?,
            "mask" => self.mask = serde_json::from_value(value.clone())?,
            _ => match &mut self.kind {
                NodeKind::Text { content, font, font_size, font_weight, font_style, font_variation_settings, font_feature_settings,
                    letter_spacing, word_spacing, line_height, text_align, direction, white_space, word_break, line_break, width,
                    fill, stroke, stroke_width } => fields! {
                    "content" => content, "font" => font, "font-size" => font_size, "font-weight" => font_weight,
                    "font-style" => font_style, "font-variation-settings" => font_variation_settings,
                    "font-feature-settings" => font_feature_settings, "letter-spacing" => letter_spacing,
                    "word-spacing" => word_spacing, "line-height" => line_height, "text-align" => text_align,
                    "direction" => direction, "white-space" => white_space, "word-break" => word_break,
                    "line-break" => line_break, "width" => width, "fill" => fill, "stroke" => stroke,
                    "stroke-width" => stroke_width,
                },
                NodeKind::Path { d, fill, fill_rule, fill_opacity, stroke, stroke_width, stroke_linecap, stroke_linejoin,
                    stroke_miterlimit, stroke_dasharray, stroke_dashoffset, stroke_opacity, path_length, paint_order } => fields! {
                    "d" => d, "fill" => fill, "fill-rule" => fill_rule, "fill-opacity" => fill_opacity, "stroke" => stroke,
                    "stroke-width" => stroke_width, "stroke-linecap" => stroke_linecap, "stroke-linejoin" => stroke_linejoin,
                    "stroke-miterlimit" => stroke_miterlimit, "stroke-dasharray" => stroke_dasharray,
                    "stroke-dashoffset" => stroke_dashoffset, "stroke-opacity" => stroke_opacity, "pathLength" => path_length,
                    "paint-order" => paint_order,
                },
                _ => return Err(<serde_json::Error as serde::de::Error>::custom(format!("unknown field `{field}`"))),
            },
        }
        Ok(())
    }
    /// This node without its subtree (the renderer flattens the tree once).
    pub fn without_children(&self) -> Self {
        Self {
            name: self.name.clone(),
            transform: self.transform.clone(),
            target: self.target.clone(),
            children: Vec::new(),
            visibility: self.visibility.clone(),
            light_link_tags: self.light_link_tags.clone(),
            pass_index: self.pass_index,
            is_holdout: self.is_holdout, is_shadow_catcher: self.is_shadow_catcher,
            opacity: self.opacity,
            clip_path: self.clip_path.clone(),
            mask: self.mask.clone(),
            filter: self.filter.clone(),
            audio: self.audio.clone(),
            active: self.active.clone(),
            time: self.time.clone(),
            animation: self.animation.clone(),
            composition: self.composition.clone(),
            kind: self.kind.clone(),
        }
    }
}


#[cfg(test)]
mod field_write_tests {
    use super::*;
    /// A value written into one field gives the node that reading the whole node with that value gives, and is
    /// refused exactly when that reading is.
    #[test]
    fn a_field_written_alone_is_the_node_read_whole_with_it() {
        use serde_json::json;
        let nodes = [
            json!({"type":"path","name":"p","d":"M0 0H16V16H0Z","fill":"#38bdf8","stroke":"red","stroke-width":2,"translation":[0.1,2,0],"opacity":0.3}),
            json!({"type":"text","name":"t","content":"Hi there","font":"a.ttf","font-size":18.5,"width":120,"filter":"blur(2px)"}),
            json!({"name":"g","clip-path":"p"}),
        ];
        let keys = [
            ("fill", json!("blue")), ("fill", json!("none")), ("fill", json!({"image":"i"})), ("fill", json!(3)), ("fill", json!(null)),
            ("stroke", json!("linear-gradient(red, blue)")), ("stroke-width", json!(0.1)), ("stroke-width", json!(3)), ("stroke-width", json!("x")),
            ("fill-opacity", json!(0.30000001192092896)), ("stroke-opacity", json!(1e-7)), ("fill-rule", json!("evenodd")), ("fill-rule", json!("odd")),
            ("stroke-linecap", json!("round")), ("stroke-linejoin", json!("bevel")), ("stroke-miterlimit", json!(2.5)),
            ("stroke-dasharray", json!([4, 2.5])), ("stroke-dasharray", json!("none")), ("stroke-dashoffset", json!(-3)),
            ("pathLength", json!(null)), ("pathLength", json!(100)), ("paint-order", json!("stroke")), ("paint-order", json!("normal")),
            ("d", json!("M0 0L1 1")), ("d", json!(7)),
            ("content", json!("Other")), ("content", json!([{"content":"a"},{"content":"b","fill":"red","font-size":1.0000000596046448,"letter-spacing":1.0000000596046448}])),
            ("content", json!([{"content":"a","stroke":{"image":"i"}}])), ("content", json!([{"text":"a"},{"text":"b","fill":"red"}])), ("content", json!(5)),
            ("font", json!("b.otf")), ("font", json!(["a.ttf","b.ttf"])), ("font-size", json!(12.5)), ("font-size", json!("big")),
            ("font-weight", json!("bold")), ("font-weight", json!(650)), ("font-style", json!("italic")),
            ("font-variation-settings", json!("\"wght\" 500")), ("font-feature-settings", json!("normal")),
            ("letter-spacing", json!(0.1)), ("word-spacing", json!(0)), ("line-height", json!(1.2)), ("line-height", json!("normal")),
            ("text-align", json!("center")), ("direction", json!("rtl")), ("white-space", json!("nowrap")), ("word-break", json!("normal")),
            ("line-break", json!("auto")), ("width", json!(null)), ("width", json!(64.25)),
            ("filter", json!("blur(4px)")), ("filter", json!(null)), ("filter", json!(2)), ("clip-path", json!("t")), ("mask", json!(null)),
            // A number whose `f64` is the tie between two `f32`s, as text above the tie: the kind's reader narrows
            // the `f64` (to even), where reading the text as an `f32` rounds up.
            ("font-size", json!(1.0000000596046448)), ("stroke-width", json!(1.0000000596046448)), ("width", json!(1.0000000596046448)),
            ("pathLength", json!(1.0000000596046448)), ("stroke-miterlimit", json!(1.0000000596046448)), ("letter-spacing", json!(1.0000000596046448)),
            ("fill-opacity", json!(1.0000000596046448)), ("stroke-dasharray", json!([1.0000000596046448, 3])), ("line-height", json!(1.0000000596046448)),
            ("font-weight", json!(400.00001525878906)),
            ("nope", json!(1)), ("opacity", json!(0.5)),
        ];
        let text = |node: &Node| serde_json::to_string(node).unwrap();
        for node in &nodes {
            let base: Node = crate::v2::from_test_value(node.clone()).unwrap();
            for (field, value) in &keys {
                let mut whole = serde_json::to_value(&base).unwrap();
                whole[*field] = value.clone();
                let whole = serde_json::from_value::<Node>(whole);
                let mut alone = base.clone();
                let written = alone.set_field(field, value);
                // `opacity` and the like are the node's own fields but not style fields: a whole read takes them,
                // this write is for style fields only.
                if *field == "opacity" { assert!(written.is_err()); continue; }
                assert_eq!(written.is_ok(), whole.is_ok(), "{node} {field} {value}: {:?} against {:?}", written.as_ref().err(), whole.as_ref().err());
                match whole {
                    Ok(whole) => assert_eq!(text(&alone), text(&whole), "{node} {field} {value}"),
                    Err(_) => assert_eq!(text(&alone), text(&base), "{node} {field} {value}: a refused write leaves the node"),
                }
            }
        }
    }
}
#[cfg(test)]
mod visibility_tests {
    use super::*;
    #[test]
    fn omitted_type_is_a_strict_group_node() {
        let node: Node = crate::v2::from_test_value(serde_json::json!({"name":"group","children":[{}]})).unwrap();
        assert!(matches!(node.kind, NodeKind::Node {}));
        assert!(matches!(node.children[0].kind, NodeKind::Node {}));
        for value in [serde_json::json!({"geometry":"g"}), serde_json::json!({"lens":50}),
            serde_json::json!({"energy":10}), serde_json::json!({"file":"a.glb"}),
            serde_json::json!({"typo":true}), serde_json::json!({"type":"meshh"}),
            serde_json::json!({"type":null}), serde_json::json!({"type":1})] {
            assert!(crate::v2::from_test_value::<Node>(value.clone()).is_err(), "{value}");
        }
        assert_eq!(serde_json::to_value(node).unwrap()["type"], "node");
        let scene = Scene::from_json(br#"{"version":2,"nodes":[{"name":"group","children":[{"type":"camera","name":"camera"}]}]}"#).unwrap();
        assert!(matches!(scene.nodes[0].kind, NodeKind::Node {}));
    }

    #[test]
    fn flat_rna_visibility_round_trips_on_nodes_and_imported_overrides() {
        let value = serde_json::json!({"version":2,"nodes":[{
            "type":"asset","file":"model.glb","visible_camera":false,
            "nodes":{"Wheel":{"visible_shadow":false}}
        }]});
        let scene: Scene = crate::v2::from_test_value(value).unwrap();
        assert!(!scene.nodes[0].visibility.camera);
        assert!(scene.nodes[0].visibility.shadow);
        let NodeKind::Asset { nodes, .. } = &scene.nodes[0].kind else { panic!() };
        let visibility = nodes["Wheel"].visibility.as_ref().unwrap();
        assert!(!visibility.shadow);
        assert!(visibility.camera);
        let value = serde_json::to_value(scene).unwrap();
        assert_eq!(value["nodes"][0]["visible_camera"], false);
        assert_eq!(value["nodes"][0]["nodes"]["Wheel"]["visible_shadow"], false);
        assert!(value["nodes"][0].get("visibility").is_none());
        assert!(crate::v2::from_test_value::<Scene>(serde_json::json!({"version":2,"nodes":[{
            "type":"node","visibility":{"camera":false}
        }]})).is_err());
    }
}

#[cfg(test)]
mod fit_defaults_tests {
    use super::*;
    #[test]
    fn omitted_area_height_stays_linked_to_size() {
        let scene = Scene::from_json(br#"{"version":2,"nodes":[{"type":"light","shape":"rectangle","size":4},{"type":"camera"}]}"#).unwrap();
        let NodeKind::Light { size, size_y, .. } = &scene.nodes[0].kind else { panic!() };
        assert_eq!(*size, 4.);
        assert_eq!(*size_y, None);
        assert_eq!(size_y.unwrap_or(*size), 4.);
    }
}
/// Sound attached to any node. Omit file for a group gain.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Audio {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork sound source", "x-help-path" = "node.audio.file", "x-help" = "file-path|null; mono/stereo; relative to scene; omitted = children only"))]
    pub file: Option<String>,
    #[serde(default = "canvas::one", deserialize_with = "super::numeric")]
    #[schemars(extend("x-source" = "Web Audio GainNode", "x-help-path" = "node.audio.gain", "x-help" = "number; linear amplitude over source plus children; default 1; keyable"))]
    pub gain: f32,

}

#[cfg(test)]
mod audio_tests {
    use super::*;
    #[test]
    fn clock_pitch_defaults_to_follow_and_round_trips() {
        for clock in [serde_json::json!({"speed":2}), serde_json::json!({"speed":2,"pitch":"follow"})] {
            let scene = super::super::Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2,"composition":{"space":"2d"},"nodes":[{"time":clock,"audio":{"file":"track.wav"}}]
            })).unwrap()).unwrap();
            assert_eq!(scene.nodes[0].time.as_ref().unwrap().pitch, Pitch::Follow);
            let value = serde_json::to_value(scene).unwrap();
            assert_eq!(value["nodes"][0]["time"]["pitch"], "follow");
            assert!(value["nodes"][0]["audio"].get("preservesPitch").is_none());
        }
    }
    #[test]
    fn pitch_keep_refuses_until_built() {
        for node in [
            serde_json::json!({"time":{"speed":2,"pitch":"keep"},"audio":{"file":"track.wav"}}),
            serde_json::json!({"time":{"pitch":"keep"},"children":[{"audio":{"file":"track.wav"}}]}),
        ] {
            let error = super::super::Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                "version":2,"composition":{"space":"2d"},"nodes":[node]
            })).unwrap()).unwrap_err().to_string();
            assert!(error.contains("time.pitch: keep is not built yet"), "{error}");
            assert_eq!(error.lines().count(), 1, "{error}");
        }
    }
    #[test]
    fn old_pitch_spellings_refuse_and_name_clock_pitch() {
        for spelling in ["preservesPitch", "preserves_pitch"] {
            for value in [false, true] {
                let error = super::super::Scene::from_json(&serde_json::to_vec(&serde_json::json!({
                    "version":2,"composition":{"space":"2d"},"nodes":[{"children":[{"audio":{spelling:value}}]}]
                })).unwrap()).unwrap_err().to_string();
                assert!(error.contains("time.pitch: keep"), "{error}");
                assert!(error.contains("vork help upgrade"), "{error}");
                assert_eq!(error.lines().count(), 1, "{error}");
            }
        }
    }
    #[test]
    fn audio_is_a_node_field_and_survives_flattening() {
        for kind in ["node", "camera", "light", "path"] {
            let mut value = serde_json::json!({"type":kind,"audio":{"file":"track.wav"}});
            if kind == "light" {
                value["shape"] = "point".into();
            }
            if kind == "path" {
                value["d"] = "M0 0L1 0L1 1Z".into();
            }
            let node: Node = super::super::from_test_value(value).unwrap();
            let flat = node.without_children();
            assert_eq!(flat.audio.as_ref().unwrap().gain, 1.);
            assert_eq!(flat.audio.unwrap().file.as_deref(), Some("track.wav"));
        }
    }
}
