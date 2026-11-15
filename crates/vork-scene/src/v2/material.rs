use super::*;
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Image {
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "image.file", "x-help" = "file-path; XOR render; png jpg tif hdr exr ies mov mp4 mkv or %04d frame pattern; 8-bit may be block-compressed; 16-bit integer and floating-point inputs use uncompressed half-float precision, not exact source integer codes; ies: LM-63 Type C"))]
    pub file: Option<String>,
    /// Seconds into media; omitted follows scene time. EXR sequences honor
    /// framesPerSecond metadata; other sequences use render.fps.
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (footage clock)", "x-help-path" = "image.time", "x-help" = "number|rational string|null; seconds from first displayed frame; null follows scene time; animatable; subframes hold; out-of-range holds ends"))]
    pub time: Option<vork_time::Clock>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (nested scene-linear render)", "x-help-path" = "image.render", "x-help" = "ImageRender; renders a camera view into the texture; inherits integrator, fps, AOVs; sampling budget follows quality/cost level; closed shutter; scene-linear premultiplied RGBA", "x-help-extra" = "image.render restrictions: shared parent pose; aperture sampling is renderer-selected; geometry_opacity/alpha_cutoff/geometry_normal/environment image inputs unsupported"))]
    pub render: Option<ImageRender>,
    /// A rendered image's composition: its image graph, run on the capture before it is published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork", "x-help-path" = "image.composition", "x-help" = "{compositor}|null; a rendered image's image graph; requires render"))]
    pub composition: Option<ImageComposition>,
    /// The canvas node this image is a picture of, once the document around the canvas is split from it
    /// (`Scene::without_compositions`): the 2D engine draws it. Never authored.
    #[serde(skip)]
    pub picture_of: Option<String>,
    /// Explicit input encoding; omitted uses embedded colour metadata, then the format fallback.
    #[serde(default)]
    #[schemars(extend("x-source" = "Blender 5.2.1 Image.colorspace_settings.name", "x-help-path" = "image.color_space", "x-help" = "sRGB|Display P3|Linear Rec.709|Linear Rec.2020|ACEScg|Non-Color|null; omitted/null: embedded colour description, else hdr/exr/ies linear Rec.709 and other files sRGB"))]
    pub color_space: Option<ColorSpace>,
    #[serde(default)]
    #[schemars(extend("x-source" = "glTF 2.0 sampler.wrapS", "x-help-path" = "image.wrap_s", "x-help" = "repeat|clamp_to_edge|mirrored_repeat"))]
    pub wrap_s: Wrap,
    #[serde(default)]
    #[schemars(extend("x-source" = "glTF 2.0 sampler.wrapT", "x-help-path" = "image.wrap_t", "x-help" = "repeat|clamp_to_edge|mirrored_repeat"))]
    pub wrap_t: Wrap,
    #[serde(default)]
    #[schemars(extend("x-source" = "glTF 2.0 sampler.magFilter (undefined default: vork linear)", "x-help-path" = "image.mag_filter", "x-help" = "nearest|linear"))]
    pub mag_filter: MagFilter,
    #[serde(default)]
    #[schemars(extend("x-source" = "glTF 2.0 sampler.minFilter (undefined default: vork linear_mipmap_linear, anisotropic)", "x-help-path" = "image.min_filter", "x-help" = "nearest|linear|nearest_mipmap_nearest|linear_mipmap_nearest|nearest_mipmap_linear|linear_mipmap_linear"))]
    pub min_filter: MinFilter,
}
impl Image {
    /// This rendered image's graph: its composition's ordered entries.
    pub fn compositor(&self) -> &[CompositeEntry] {
        self.composition.as_ref().map_or(&[][..], |composition| composition.compositor.as_slice())
    }
    /// The render this image runs: its own fields and image graph over the settings it inherits from
    /// `parent`. None for an image that is not rendered.
    pub fn resolved_render(&self, parent: &Render) -> Option<Render> {
        let render = self.render.as_ref()?;
        let integrator = parent.settings.integrator.clone();
        Some(Render {
            camera: render.camera.clone(),
            root: render.root.clone(),
            resolution: render.resolution,
            background: render.background.clone(),
            compositor: self.compositor().to_vec(),
            settings: Settings {
                promotion_denoise: parent.settings.promotion_denoise,
                bounded_secondary_compilation: parent.settings.bounded_secondary_compilation,
                approximate_field_boundaries: parent.settings.approximate_field_boundaries,
                previs: parent.settings.previs,
                reconstruct: parent.settings.reconstruct,
                seed: parent.settings.seed,
                samples: parent.settings.samples,
                ..Settings::with_integrator(integrator)
            },
            fps: parent.fps,
            aovs: parent.aovs.clone(),
            ..Render::default()
        })
    }
    /// Upload encoding after input colour conversion. Metadata is resolved by the decoder.
    pub fn is_srgb(&self) -> bool {
        if self.render.is_some() || self.picture_of.is_some() {
            return false;
        }
        match self.color_space {
            Some(ColorSpace::LinearRec709 | ColorSpace::LinearRec2020 | ColorSpace::ACEScg | ColorSpace::NonColor) => false,
            // Float files upload scene-linear whatever they declare: EXR pixels are associated, and a transfer
            // curve applied to associated colour is wrong. The decoder converts a declared encoded space itself.
            Some(ColorSpace::Srgb | ColorSpace::DisplayP3) | None => !std::path::Path::new(self.file.as_deref().unwrap_or(""))
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| ["hdr", "exr", "ies"].iter().any(|float| e.eq_ignore_ascii_case(float))),
        }
    }
    pub fn sampler(&self) -> Sampler {
        Sampler {
            wrap_s: self.wrap_s,
            wrap_t: self.wrap_t,
            mag_filter: self.mag_filter,
            min_filter: self.min_filter,
        }
    }
}
/// The glTF 2.0 sampler of an image.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Sampler {
    pub wrap_s: Wrap,
    pub wrap_t: Wrap,
    pub mag_filter: MagFilter,
    pub min_filter: MinFilter,
}
impl Sampler {
    /// glTF sampler codes: (wrapS, wrapT, minFilter, magFilter).
    pub fn gltf_codes(&self) -> (u32, u32, u32, u32) {
        let wrap = |w: Wrap| match w {
            Wrap::Repeat => 10497,
            Wrap::ClampToEdge => 33071,
            Wrap::MirroredRepeat => 33648,
        };
        let min = match self.min_filter {
            MinFilter::Nearest => 9728,
            MinFilter::Linear => 9729,
            MinFilter::NearestMipmapNearest => 9984,
            MinFilter::LinearMipmapNearest => 9985,
            MinFilter::NearestMipmapLinear => 9986,
            MinFilter::LinearMipmapLinear => 9987,
        };
        let mag = match self.mag_filter {
            MagFilter::Nearest => 9728,
            MagFilter::Linear => 9729,
        };
        (wrap(self.wrap_s), wrap(self.wrap_t), min, mag)
    }
}
/// Settings that a nested image may author. The parent owns transport and time;
/// a capture has a closed shutter and runs the automatic per-step compositor.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct ImageRender {
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "image.render.camera", "x-help" = "camera-node-name|null; null = the only authored camera, else first imported"))]
    pub camera: Option<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "image.render.root", "x-help" = "node-name|null; only this subtree draws and is lit; objects outside still cast shadows; null = whole scene"))]
    pub root: Option<String>,
    #[schemars(extend("x-source" = "Blender 5.2.1 RenderSettings.resolution_x/resolution_y", "x-help-path" = "image.render.resolution", "x-help" = "uvec2"))]
    pub resolution: [u32; 2],
    #[schemars(extend("x-source" = "vork (Blender RenderSettings.film_transparent plus a constant RGB)", "x-help-path" = "image.render.background", "x-help" = "environment|transparent|RGB"))]
    pub background: Background,
}
/// `images.<name>.composition`: the image graph of a rendered image.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageComposition {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(extend("x-source" = "vork", "x-help-path" = "image.composition.compositor", "x-help" = "same ordered image graph as composition.compositor; names local to this capture"))]
    pub compositor: Vec<CompositeEntry>,
}
impl Default for ImageRender {
    fn default() -> Self {
        let r = Render::default();
        Self {
            camera: r.camera,
            root: r.root,
            resolution: r.resolution,
            background: r.background,
        }
    }
}
/// Blender colour space names, verbatim.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    #[serde(rename = "sRGB")]
    Srgb,
    #[serde(rename = "Linear Rec.709")]
    LinearRec709,
    #[serde(rename = "Linear Rec.2020")]
    LinearRec2020,
    ACEScg,
    #[serde(rename = "Display P3")]
    DisplayP3,
    #[serde(rename = "Non-Color")]
    NonColor,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Wrap {
    #[default]
    Repeat,
    ClampToEdge,
    MirroredRepeat,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MagFilter {
    Nearest,
    #[default]
    Linear,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema, Default, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    #[default]
    LinearMipmapLinear,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Program {
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "program.bindings", "x-help" = "<binding>_load(i), _store(i,v), _count; dense grids also expose _dims; u32 _atomic_min(i,v) returns prior value (OOB: UINT_MAX); mat4 = VorkMat4 {c0..c3 columns}; out-of-range loads 0, stores ignored"))]
    #[schemars(with = "Table<NameOr<Buffer>>")]
    pub bindings: Table<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "program.schedule", "x-help" = "install|frame|simulation; frame uses output FPS; simulation uses scene.simulation pass order; each material instance dispatches with its inputs"))]
    pub schedule: Option<ComputeSchedule>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "program.dispatch", "x-help" = "item count or [x,y,z] threads; install/init dt=0; frame schedule frame 0 dt=0; simulation ticks use fixed dt"))]
    pub dispatch: Option<ComputeDispatch>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "program.file", "x-help" = "file-path; XOR source; Slang module, import vork"))]
    pub file: Option<String>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "program.source", "x-help" = "Slang source; XOR file; import vork", "x-help-extra" = "program enum inputs: lowercased Slang case-name tokens (e.g. fog_glow)"))]
    pub source: Option<String>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    #[schemars(extend("x-source" = "vork program source location", "x-help-path" = "program.source_origin", "x-help" = "virtual filename|null; inline source only; resolves local imports"))]
    pub source_origin: Option<String>,
}
/// Number of threads, independent of the internal workgroup size.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ComputeDispatch {
    Count(u32),
    Dimensions([u32; 3]),
}
impl ComputeDispatch {
    pub fn threads(self) -> [u32; 3] {
        match self {
            Self::Count(n) => [n, 1, 1],
            Self::Dimensions(n) => n,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComputeSchedule {
    Frame,
    Install,
    /// Invoked only by the scene simulation schedule.
    Simulation,
}
/// A deterministic fixed-clock schedule over ordinary typed named buffers.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Simulation {
    /// Time of the initialized state S0, in seconds.
    #[schemars(extend("x-source" = "vork simulation contract", "x-help-path" = "simulation.t0", "x-help" = "number|rational string; seconds; initialized state S0; requests before t0 hold S0"))]
    pub t0: vork_time::Clock,
    /// Positive fixed tick duration, in seconds; independent of render FPS.
    #[schemars(extend("x-source" = "vork simulation contract", "x-help-path" = "simulation.dt", "x-help" = "number|rational string; positive finite seconds; fixed tick independent of output FPS; inputs sample left endpoints"))]
    pub dt: vork_time::Clock,
    /// Simulation randomness is independent of image sampling randomness.
    #[schemars(extend("x-source" = "vork simulation contract", "x-help-path" = "simulation.seed", "x-help" = "u32; independent of image sampling seed"))]
    pub seed: u32,
    /// Complete persistent state and scratch buffers used by these passes.
    #[schemars(extend("x-source" = "vork simulation contract", "x-help-path" = "simulation.state", "x-help" = "buffer-name[]; complete typed persistent and scratch state; exclusive schedule ownership"))]
    pub state: Vec<String>,
    /// Ordered compute program names establishing S0.
    #[schemars(extend("x-source" = "vork simulation contract", "x-help-path" = "simulation.init", "x-help" = "program-name[]; ordered initialization passes; schedule simulation"))]
    pub init: Vec<String>,
    /// Ordered compute program names advancing Sn to Sn+1.
    #[schemars(extend("x-source" = "vork simulation contract", "x-help-path" = "simulation.steps", "x-help" = "program-name[]; ordered tick passes with storage barriers; schedule simulation"))]
    pub steps: Vec<String>,
}
impl Program {
    /// Compute schedules invoke the exported `compute` entry, verified by Slang.
    /// Materials select surface (Closure or curves-only Hair), entries they consume.
    pub fn is_compute(&self) -> bool {
        self.schedule.is_some()
    }
}
fn openpbr() -> String {
    "std/openpbr".into()
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(extend("x-help-extra" = "material (glTF import): mapped to OpenPBR; coat_ior 1.5; occlusion affects indirect only"))]
/// A program instance. Its surface entry shades the assigned geometry.
pub struct Material {
    #[serde(default = "openpbr")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "material.program", "x-help" = "program-name|inline program|<library>/<module>; std/openpbr, std/hair (curves only), or custom", "x-help-extra" = "program.surface emission: authoritative radiance for hits and light samples; lit output is authored shading\nprogram.surface: Closure|Blend|Hair surface(ShadingContext,Inputs); Hair needs curves"))]
    #[schemars(with = "NameOr<Program>")]
    pub program: String,
    #[serde(default)]
    #[schemars(extend("x-source" = "OpenPBR Surface 1.1.1 defaults (spec table)", "x-help-path" = "material.inputs", "x-help" = "{name: number|vector|bool|enum-token|{image}|{attribute}|{grid}|{aov}|{step}|{channel}|{geometry}}", "x-help-extra" = "material.inputs (transmission): transmission_depth 0 = sqrt(tint) per solid interface; thin sheets = full tint; positive-depth solids = Beer absorption\nmaterial.inputs (animation): numeric scalar/vector/colour inputs on surface/environment/compute/compositor/display programs use ordinary tracks; geometry_thin_walled is structural; an omitted animated program input has a zero blend baseline (author inputs for another baseline)\nmaterial.inputs.subsurface: thick: radius × radius_scale; thin: diffuse reflection/transmission split by scatter_anisotropy"))]
    pub inputs: Table<Input>,
    #[serde(default = "material_double_sided")]
    #[schemars(extend("x-source" = "Blender renders both sides; imported glTF retains doubleSided", "x-help-path" = "material.double_sided", "x-help" = "bool; false culls back faces"))]
    pub double_sided: bool,
    /// Highest priority owns an overlapping dielectric region. Equal priorities
    /// use the most recently entered solid; simultaneous entries use solid order.
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (dielectric medium ownership)", "x-help-path" = "material.priority", "x-help" = "integer; highest owns overlapping solid media; equal priorities: last entered wins, simultaneous entries: later solid order wins; lower-priority boundaries change membership only"))]
    pub priority: i32,
    /// Meaning of geometry_opacity; the renderer owns its integration method.
    #[serde(default)]
    #[schemars(extend("x-source" = "vork surface presence semantics", "x-help-path" = "material.opacity", "x-help" = "coverage|transmittance|null; null = coverage, transmittance for transmitting surfaces; coverage = occupied area fraction; transmittance = continuous attenuation by 1 - geometry_opacity; renderer chooses integration"))]
    pub opacity: Option<Opacity>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Opacity { #[default] Coverage, Transmittance }
/// A material input value. Resources are explicit tagged objects; a plain string is
/// only ever an enum token of the program's reflected input type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Input {
    Number(#[serde(deserialize_with="super::numeric")] f32),
    Vector(#[serde(deserialize_with="super::numeric")] Vec<f32>),
    Bool(bool),
    Geometry(GeometryBinding),
    Texture(TextureBinding),
    Attribute(AttributeBinding),
    Grid(GridBinding),
    Aov(AovBinding),
    Channel(ChannelBinding),
    Step(StepBinding),
    /// An enum token of the program's reflected input type.
    Token(String),
}
/// Read-only evaluated geometry of a named mesh object, sampled by the compute clock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GeometryBinding {
    #[schemars(extend("x-source" = "vork geometry chain", "x-help-path" = "material.geometry.geometry", "x-help" = "mesh-node name; read-only geometry at the program's canonical sample time"))]
    pub geometry: String,
}
/// An earlier named compositor entry, local to this graph.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StepBinding {
    #[schemars(extend("x-source" = "vork", "x-help-path" = "material.step.step", "x-help" = "earlier compositor entry name; clamped bilinear image sampling"))]
    pub step: String,
}
/// A volume grid buffer for a FloatGrid or VectorGrid program input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GridBinding {
    #[schemars(with = "NameOr<Buffer>")]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "material.grid.grid", "x-help" = "buffer-name|inline buffer; buffer with a grid"))]
    pub grid: String,
}
/// A render.aovs entry: an Aov handle for surface programs, or the AOV image in compositor programs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AovBinding {
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "material.aov.aov", "x-help" = "render.aovs name"))]
    pub aov: String,
}
/// A kernel image channel read by a compositor program.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChannelBinding {
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "material.channel.channel", "x-help" = "color|depth|view_normal|object_index|motion|shadow_catcher; compositor programs"))]
    pub channel: KernelChannel,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KernelChannel {
    Color,
    Depth,
    ViewNormal,
    ObjectIndex,
    Motion,
    ShadowCatcher,
}
impl KernelChannel {
    pub fn token(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::Depth => "depth",
            Self::ViewNormal => "view_normal",
            Self::ObjectIndex => "object_index",
            Self::Motion => "motion",
            Self::ShadowCatcher => "shadow_catcher",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(extend("x-help-extra" = "material.attribute: binds geometry or instance columns to numeric inputs; custom programs read_attribute(handle,ctx) gives value, dx, dy"))]
/// Install-resolved named float vertex data. Custom programs declare Attribute<float>
/// through Attribute<float4>; read_attribute returns value, dx and dy. Missing or
/// incompatible streams on any bound geometry fail install.
pub struct AttributeBinding {
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "material.attribute.attribute", "x-help" = "string; geometry or instance attribute name; colour values are linear Rec.709 (no sRGB decoding)"))]
    pub attribute: String,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "material.attribute.channel", "x-help" = "r|g|b|a|rgb; omitted: r scalar, rgb colour, exact for vectors"))]
    pub channel: Option<Channel>,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "material.attribute.factor", "x-help" = "number|vector; omitted = 1"))]
    pub factor: Option<Factor>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// An image handle. Sampling and transformations belong to the receiving program.
pub struct TextureBinding {
    #[schemars(extend("x-source" = "vork", "x-help-path" = "material.texture.image", "x-help" = "image-name|inline image; the program owns sampling and UV transforms"))]
    #[schemars(with = "NameOr<Image>")]
    pub image: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Factor {
    Number(#[serde(deserialize_with="super::numeric")] f32),
    Vector(#[serde(deserialize_with="super::numeric")] Vec<f32>),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    R,
    G,
    B,
    A,
    Rgb,
}
/// One OpenPBR Surface parameter: identifier, type, default and allowed range.
#[derive(Clone, Copy, Debug)]
pub struct OpenPbrParam {
    pub name: &'static str,
    pub kind: OpenPbrKind,
    /// Empty for normals and tangents, whose default is the unperturbed frame.
    pub default: &'static [f32],
    pub meaning: &'static str,
    /// Specification unit; length uses scene metres.
    pub unit: &'static str,
    /// Specification spelling, including the unperturbed frame defaults.
    pub default_text: &'static str,
    /// Accepted as a table material input, rather than only a Closure field.
    pub supported: bool,
    /// Present in the evaluated Closure; false for std evaluation inputs.
    pub closure: bool,
    pub min: f32,
    pub max: f32,
    pub open_min: bool,
    pub open_max: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenPbrKind {
    Float,
    Color3,
    Bool,
    Vector3,
}
impl OpenPbrParam {
    pub fn supported() -> impl Iterator<Item = &'static Self> {
        super::constants::OPENPBR.iter().filter(|p| p.supported)
    }
    pub fn find(name: &str) -> Option<&'static Self> {
        super::constants::OPENPBR.iter().find(|p| p.name == name)
    }
    /// Whether a constant channel lies in the parameter's allowed range.
    pub fn allows(&self, x: f32) -> bool {
        x.is_finite()
            && if self.open_min {
                x > self.min
            } else {
                x >= self.min
            }
            && if self.open_max {
                x < self.max
            } else {
                x <= self.max
            }
    }
}

#[cfg(test)]
mod image_tokens {
    #[test]
    fn authored_materials_render_both_sides_by_default() {
        let material: super::Material = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(material.double_sided);
        let material: super::Material = serde_json::from_value(serde_json::json!({"double_sided":false})).unwrap();
        assert!(!material.double_sided);
    }
    use super::*;
    #[test]
    fn blender_colour_spaces_and_gltf_sampler_tokens() {
        let image = |v: serde_json::Value| serde_json::from_value::<Image>(v).unwrap();
        assert!(image(serde_json::json!({"file":"a.png"})).is_srgb());
        assert!(!image(serde_json::json!({"file":"a.exr"})).is_srgb());
        assert!(!image(serde_json::json!({"file":"a.png","color_space":"Non-Color"})).is_srgb());
        assert!(!image(serde_json::json!({"file":"a.png","color_space":"Linear Rec.709"})).is_srgb());
        // A float file uploads linear whatever it declares; the decoder converts a declared encoding itself.
        assert!(!image(serde_json::json!({"file":"a.exr","color_space":"sRGB"})).is_srgb());
        assert!(!image(serde_json::json!({"file":"a.exr","color_space":"Display P3"})).is_srgb());
        assert!(image(serde_json::json!({"file":"a.png","color_space":"Display P3"})).is_srgb());
        let s = image(serde_json::json!({"file":"a.png","wrap_s":"clamp_to_edge","wrap_t":"mirrored_repeat",
            "mag_filter":"nearest","min_filter":"nearest_mipmap_linear"})).sampler();
        assert_eq!(s.gltf_codes(), (33071, 33648, 9986, 9728));
        assert_eq!(Sampler::default().gltf_codes(), (10497, 10497, 9987, 9729));
        for bad in [serde_json::json!({"file":"a.png","color_space":"linear"}), serde_json::json!({"file":"a.png","wrap":"clamp"})] {
            assert!(serde_json::from_value::<Image>(bad).is_err());
        }
    }
}

fn material_double_sided() -> bool { true }

#[cfg(test)]
mod retired_medium_bound_tests {
    #[test]
    fn authored_medium_majorant_is_unknown() {
        let error=serde_json::from_value::<super::Material>(serde_json::json!({"medium_majorant":1.})).unwrap_err();
        assert!(error.to_string().contains("unknown field `medium_majorant`"));
    }
}
