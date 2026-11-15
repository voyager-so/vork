//! Version 2 authoring contract: the one scene model the renderer installs.
mod output;
pub use output::{ColorConversion, OutputDestination, OutputDisplay, WorkingColorSpace};
pub use vork_time::Time;
pub mod animation;
pub mod canvas;
mod compose;
mod constants;
pub mod diagnostics;
pub mod geometry;
mod inline;
mod resources;
mod image_plan;
pub mod light;
pub mod material;
pub mod node;
pub mod profile;
pub mod render;
mod schema;
pub mod targets;
pub mod text;
pub mod time;
mod validate;
pub use animation::*;
pub use canvas::{AspectRatio, Composition, DashArray, FillRule, Fit, ImagePaint, LineCap, LineJoin, NoDash, NoPaint, Paint, PaintLayer, PaintOrder, Space};
pub use text::{Content, Direction, FontFiles, FontStyle, FontWeight, FontWeightKeyword, LineBreak, LineHeight, LineHeightNormal, Run, TextAlign, TextUnit, UnitProperty, WhiteSpace, WordBreak, laid_out_runs, laid_out_text, transform_origin, unit_ranges};
use anyhow::{Context, Result, bail, ensure};
pub use constants::OPENPBR;
pub use geometry::*;
pub use light::*;
pub use material::*;
pub use node::*;
pub use profile::*;
pub use render::*;
pub use time::*;
pub use schema::{api_field, documents, lookup_only_field, schema};
pub use validate::validate_openpbr_input;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
pub use targets::{StyleProperty, StyleInterpolation, NodeProperty, Ray, RenderProperty, TrackTarget, TrackValue, display_path};

// Serde buffers untagged/flattened fields through deserialize_any. Restore
// JSON's exact-number representation before lowering a numeric field to f32/f64.
pub(crate) fn numeric<'de, D: serde::Deserializer<'de>, T: serde::de::DeserializeOwned>(d: D) -> std::result::Result<T, D::Error> {
    serde_json::from_value(serde_json::Value::deserialize(d)?).map_err(serde::de::Error::custom)
}

pub(crate) fn lower_numeric_json(value: &mut serde_json::Value) -> std::result::Result<(), &'static str> {
    match value {
        serde_json::Value::Number(number) if !number.is_i64() && !number.is_u64() => {
            *number = serde_json::Number::from_f64(number.as_f64().ok_or("numeric input must be finite")?)
                .ok_or("numeric input must be finite")?;
        }
        serde_json::Value::Array(values) => for value in values { lower_numeric_json(value)?; },
        serde_json::Value::Object(values) => for value in values.values_mut() { lower_numeric_json(value)?; },
        _ => {}
    }
    Ok(())
}

pub type Table<T> = BTreeMap<String, T>;
/// A resource name or inline definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum NameOr<T> {
    Name(String),
    Inline(T),
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(extend("x-help-extra" = "coordinates: glTF right-handed Y-up; cameras look along local -Z"))]
pub struct Scene {
    /// Authored locations for diagnostics; never scene content or resource identity.
    #[serde(skip)]
    pub origins: std::collections::BTreeMap<String, Vec<String>>,
    /// Editor JSON Schema address; metadata only, never fetched by the renderer.
    #[serde(rename = "$schema", default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-help-path" = "scene.$schema", "x-help" = "string; editor schema URL"))]
    pub schema: Option<String>,
    /// Internal delivery admission/approximation policy; never scene content.
    #[serde(skip)]
    pub delivery: bool,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.version", "x-help" = "2"))]
    pub version: u32,
    /// `{space: "2d"}`: the document is a 2D canvas. Absent: a 3D world.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "END-STATE-API §7", "x-help-path" = "scene.composition", "x-help" = "composition|null; space 2d = canvas, 3d = world; null = 3D world"))]
    pub composition: Option<Composition>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.render", "x-help" = "render"))]
    pub render: Render,
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.nodes", "x-help" = "node[]"))]
    pub nodes: Vec<Node>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.animation", "x-help" = "layer[]"))]
    pub animation: Vec<Layer>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.buffers", "x-help" = "{name:buffer}"))]
    pub buffers: Table<Buffer>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.geometry", "x-help" = "{name:geometry}"))]
    pub geometry: Table<Geometry>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.images", "x-help" = "{name:image}"))]
    pub images: Table<Image>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.programs", "x-help" = "{name:program}"))]
    pub programs: Table<Program>,
    /// Ordered geometry lowerings; compute contains ordinary geometry programs.
    #[serde(default = "constants::geometry_stages")]
    #[schemars(extend("x-source" = "vork geometry chain", "x-help-path" = "scene.geometry_stages", "x-help" = "(deform|compute|displace)[]; ordered, each at most once; default [deform,compute,displace]"))]
    pub geometry_stages: Vec<GeometryStage>,
    /// Fixed-clock retained compute state, independent of output FPS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork simulation contract", "x-help-path" = "scene.simulation", "x-help" = "simulation|null; fixed-clock ordered retained compute"))]
    pub simulation: Option<Simulation>,
    #[serde(default)]
    #[schemars(extend("x-source" = "vork (explicit contract/technique choice)", "x-help-path" = "scene.materials", "x-help" = "{name:material}"))]
    pub materials: Table<Material>,
    /// Authored straight segments inside a world-space box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(extend("x-source" = "vork authored propagation", "x-help-path" = "scene.propagation", "x-help" = "{program,bounds}|null; authored straight ray segments inside world-space bounds"))]
    pub propagation: Option<Propagation>,
}
/// Native deformation (including subdivision), geometry compute, and displacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GeometryStage { Deform, Compute, Displace }
impl GeometryStage {
    pub const DEFAULT: [Self;3] = [Self::Deform,Self::Compute,Self::Displace];
    pub fn name(self)-> &'static str {match self {Self::Deform=>"deform",Self::Compute=>"compute",Self::Displace=>"displace"}}
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Propagation {
    /// Program exporting PropagationStep propagation(PropagationContext, Inputs).
    pub program: String,
    /// World-space minimum and maximum in metres; rays leaving continue straight.
    pub bounds: [[f32; 3]; 2],
}
/// The deepest nesting of arrays and objects a scene document may have (the JSON reader's admission limit).
pub const MAX_JSON_DEPTH: usize = 127;

impl Scene {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read scene {}", path.display()))?;
        Self::from_json_with_source(&bytes, &bytes, Some(path))
            .with_context(|| format!("load scene {}", path.display()))
    }
    /// A scene document; a leading UTF-8 byte-order mark (PowerShell 5.1 writes one) is ignored.
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        Self::from_json_with_source(bytes, bytes, None)
    }
    /// Decode expanded geometry while retaining the original document's locations.
    pub fn from_json_with_source(bytes: &[u8], authored: &[u8], file: Option<&Path>) -> Result<Self> {
        Self::decode_with_source(bytes,authored,file,true)
    }
    /// Offline asset consumer: validate content without requiring a render camera.
    pub fn from_json_for_export(bytes:&[u8],authored:&[u8],file:Option<&Path>)->Result<Self> {
        Self::decode_with_source(bytes,authored,file,false)
    }
    fn decode_with_source(bytes:&[u8],authored:&[u8],file:Option<&Path>,require_camera:bool)->Result<Self> {
        let scene = Self::parse_json_with_source(bytes, authored, file)?;
        scene.validate_content(require_camera).map_err(|error| scene.authored_error(error))?;
        Ok(scene)
    }
    /// Decode resource slots and locations before independent admission checks.
    /// Callers must validate before using references or installing the scene.
    #[doc(hidden)]
    pub fn parse_json_with_source(bytes:&[u8],authored:&[u8],file:Option<&Path>)->Result<Self> {
        let bytes = diagnostics::without_bom(bytes);
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| diagnostics::nesting_error(&error, "scene").unwrap_or_else(|| diagnostics::decode_error(bytes, error)))?;
        Self::parse_value(value, authored, file, |error| diagnostics::decode_error(bytes, error))
    }
    /// Admit an expanded document without encoding and parsing it again. The document is consumed. When a
    /// field fails to decode, `pre_lowering` makes the same document once more, for the diagnostics byte
    /// admission gives: no copy of it is held for an error that does not happen.
    #[doc(hidden)]
    pub fn parse_value_with_source(value: serde_json::Value, authored: &[u8], file: Option<&Path>,
        pre_lowering: impl FnOnce() -> Result<serde_json::Value>) -> Result<Self> {
        // serde_json's byte parser rejects the 128th nested array/object. Expansion
        // can add nesting to otherwise admitted input. Bound this walk and retain
        // byte admission's exact error and precedence for those exceptional values.
        fn reaches_input_limit(value: &serde_json::Value, remaining: u8) -> bool {
            match value {
                serde_json::Value::Array(values) => remaining == 1
                    || values.iter().any(|value| reaches_input_limit(value, remaining - 1)),
                serde_json::Value::Object(values) => remaining == 1
                    || values.values().any(|value| reaches_input_limit(value, remaining - 1)),
                _ => false,
            }
        }
        if reaches_input_limit(&value, MAX_JSON_DEPTH as u8 + 1) {
            return Self::parse_json_with_source(&serde_json::to_vec(&value)?, authored, file);
        }
        Self::parse_value(value, authored, file, |error| {
            match pre_lowering().and_then(|value| Ok(serde_json::to_vec(&value)?)) {
                Ok(bytes) => diagnostics::decode_error(&bytes, error),
                Err(_) => error.into(),
            }
        })
    }
    fn parse_value(mut value: serde_json::Value, authored: &[u8], file: Option<&Path>,
        decode_error: impl FnOnce(serde_json::Error) -> anyhow::Error) -> Result<Self> {
        // Inline resource names and admission diagnostics must not depend on map storage.
        value.sort_all_objects();
        ensure!(value.pointer("/render/settings").is_none(),
            "render.settings: render settings are not scene content; choose them with --profile (vork help profiles)");
        // Tests switch final dither off on a parsed document (Render::dither); authored scenes cannot.
        #[cfg(not(any(test, feature = "test-support")))]
        ensure!(value.pointer("/render/dither").is_none(), "render.dither: output encoding owns dither; it is not scene content");
        ensure!(value.pointer("/render/integrator").is_none() && value.pointer("/render/samples").is_none(),
            "render.integrator and render.samples are render settings, not scene content; choose --profile std/draft, std/final or std/path (vork help profiles)");
        fn refuse_sound(nodes: Option<&serde_json::Value>) -> Result<()> {
            if let Some(nodes) = nodes.and_then(serde_json::Value::as_array) {
                for node in nodes {
                    ensure!(
                        node.get("audio").and_then(|a| a.get("program")).is_none(),
                        "node {}: sound programs are not built yet",
                        node.get("name")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("<unnamed>")
                    );
                    refuse_sound(node.get("children"))?;
                }
            }
            Ok(())
        }
        refuse_sound(value.get("nodes"))?;
        // The view and the image graph belong to a composition: the scene's, or a rendered image's.
        // One line names every move, so an old scene is rewritten in one pass. The authored document is walked
        // before inline resources are hoisted, so each path is the one the author wrote: an image may be inline
        // in any material input. A scene's own composition must also say its space, which an old scene never
        // wrote: a scene without a composition is a world.
        fn moved_fields(value: &serde_json::Value, path: &str, moved: &mut Vec<String>) {
            match value {
                serde_json::Value::Object(fields) => {
                    if let Some(render) = fields.get("render").filter(|render| render.is_object()) {
                        if path.is_empty() {
                            if render.get("display").is_some() { moved.push("render.display was removed; use composition.view".into()); }
                            if render.get("view").is_some() { moved.push("render.view: use composition.view".into()); }
                        }
                        if render.get("compositor").is_some() {
                            let field = |owner: &str| diagnostics::member(&diagnostics::member(path, owner), "compositor");
                            moved.push(format!("{} was removed; use {}", field("render"), field("composition")));
                        }
                    }
                    for (name, field) in fields { moved_fields(field, &diagnostics::member(path, name), moved); }
                }
                serde_json::Value::Array(items) => {
                    for (index, item) in items.iter().enumerate() { moved_fields(item, &format!("{path}[{index}]"), moved); }
                }
                _ => {}
            }
        }
        fn refuse_moved_fields(value: &serde_json::Value) -> Result<()> {
            let mut moved = Vec::new();
            moved_fields(value, "", &mut moved);
            if moved.is_empty() { return Ok(()); }
            let space = if value.get("render").is_some_and(|render| ["display", "view", "compositor"].iter().any(|field| render.get(field).is_some()))
                && value.pointer("/composition/space").is_none() {
                "; composition.space is required too: write \"3d\" (a scene without a composition is a world)"
            } else { "" };
            anyhow::bail!("{}{space}; see vork help upgrade", moved.join("; "))
        }
        ensure!(value.get("output").is_none() && value.get("outputs").is_none(),
            "outputs are render-call destinations; use --out with a path or output descriptor");
        refuse_moved_fields(&value)?;
        canvas::refuse_unbuilt(&value)?;
        let mut origins = inline::lower(&mut value);
        render::refuse_node_bindings(&value)?;
        inline::locate(diagnostics::without_bom(authored), &mut origins);
        if let Some(file) = file {
            for path in origins.values_mut().flatten() { *path = format!("{}: {path}", file.display()); }
        }
        let mut scene: Self = serde_json::from_value(value)
            .map_err(decode_error)?;
        ensure!(!scene.materials.keys().any(|name| name.starts_with("__vork_")),
            "material names starting with __vork_ are reserved for the renderer");
        for material in scene.materials.values() {
            ensure!(!matches!(material.program.as_str(), "std/agx" | "std/standard" | "blender/filmic") || !material.inputs.contains_key("dither_intensity"),
                "{name}: view inputs.dither_intensity was removed; output encoding owns dither: delete the input (1, the default, draws as before; \
                 0 or more than 1 has no setting: dither scales with the output's bit depth, and grain is a compositor program); see vork help upgrade",
                name = material.program);
        }
        scene.origins = origins;
        Ok(scene)
    }
    /// Add authored users to errors without changing normalized resource identity.
    pub fn authored_error(&self, error: anyhow::Error) -> anyhow::Error {
        let text = format!("{error:#}");
        let programs = self.materials.iter().filter(|(name, _)| text.contains(&format!("material {name}:")) || text.contains(&format!("material {name} ")))
            .map(|(_, material)| format!("program {}", material.program)).collect::<Vec<_>>();
        let error = if programs.is_empty() { error } else { error.context(programs.join("; ")) };
        diagnostics::authored_error(&self.origins, error)
    }

    pub fn validate(&self) -> Result<()> { self.validate_content(true) }
    /// Validate graph content; sample-only consumers do not require a camera.
    pub fn validate_content(&self,require_camera:bool)->Result<()> {
        self.validate_compositions()?;
        if self.has_composition_parts() { return self.validate_parts(require_camera); }
        if let Some(p) = &self.propagation {
            ensure!(self.programs.contains_key(&p.program), "unknown propagation program {}", p.program);
            ensure!(p.bounds.iter().flatten().all(|v|v.is_finite()) && (0..3).all(|i|p.bounds[0][i]<p.bounds[1][i]), "propagation.bounds must have finite ordered endpoints");
        }

        ensure!(
            self.version == 2,
            "version {}: only scene version 2 is supported; rewrite the scene using version 2 fields (vork help)",
            self.version
        );
        for names in [
            self.buffers.keys().collect::<Vec<_>>(),
            self.geometry.keys().collect(),
            self.images.keys().collect(),
            self.programs.keys().collect(),
            self.materials.keys().collect(),
        ] {
            for n in names {
                validate_name(n)?;
            }
        }
        let mut names = BTreeMap::new();
        let mut groups = std::collections::HashSet::new();
        let mut cameras = 0;
        let mut has_assets = false;
        fn visit<'a>(
            nodes: &'a [Node],
            names: &mut BTreeMap<&'a str, &'a Node>,
            groups: &mut std::collections::HashSet<&'a str>,
            cameras: &mut usize,
            has_assets: &mut bool,
        ) -> Result<()> {
            for n in nodes {
                if !n.name.is_empty() {
                    validate_name(&n.name)?;
                    ensure!(
                        names.insert(&n.name, n).is_none(),
                        "duplicate node name {:?}; node names must be unique",
                        n.name
                    );
                }
                ensure!(
                    n.pass_index.is_none_or(|v| (1..=65535).contains(&v)),
                    "node {:?}: pass_index 0 is reserved for background; use 1..65535",
                    n.name
                );
                if let NodeKind::Asset {
                    nodes: overrides, ..
                } = &n.kind
                {
                    *has_assets = true;
                    for (name, node) in overrides {
                        groups.extend(node.light_link_tags.iter().flatten().map(String::as_str));
                        ensure!(
                            node.pass_index.is_none_or(|v| (1..=65535).contains(&v)),
                            "asset override {name:?}: pass_index 0 is reserved for background; use 1..65535"
                        );
                    }
                }
                groups.extend(n.light_link_tags.iter().map(String::as_str));
                if matches!(n.kind, NodeKind::Camera { .. }) {
                    *cameras += 1;
                }
                visit(&n.children, names, groups, cameras, has_assets)?;
            }
            Ok(())
        }
        visit(&self.nodes, &mut names, &mut groups, &mut cameras, &mut has_assets)?;
        self.validate_canvas(&names)?;
        ensure!(!require_camera || cameras > 0 || has_assets || self.is_canvas(), "scene requires at least one camera");
        fn validate_links(nodes: &[Node], groups: &std::collections::HashSet<&str>) -> Result<()> {
            for node in nodes {
                if let NodeKind::Light {
                    illuminates,
                    shadowed_by,
                    ..
                } = &node.kind
                {
                    for name in illuminates.iter().chain(shadowed_by.iter()).flatten() {
                        ensure!(
                            groups.contains(name.as_str()),
                            "{}",
                            diagnostics::unknown("light_link_tags entry", name, groups.iter().copied())
                        );
                    }
                }
                validate_links(&node.children, groups)?;
            }
            Ok(())
        }
        validate_links(&self.nodes, &groups)?;
        let mut errors = Vec::new();
        errors.extend(self.buffers.values().map(|b| -> Result<()> {
            ensure!(
                b.dims.is_some() || b.data.is_some() != b.file.is_some(),
                "buffer requires dims or exactly one of data or file"
            );
            Ok(())
        }).filter_map(Result::err));
        errors.extend(self.programs.values().map(|p| -> Result<()> {
            ensure!(
                p.source.is_some() != p.file.is_some(),
                "program requires exactly one of source or file"
            );
            Ok(())
        }).filter_map(Result::err));
        errors.extend(self.geometry.values().map(|g| -> Result<()> {
            ensure!(
                matches!(g.topology, Topology::Volume | Topology::Implicit) || g.attributes.contains_key("POSITION"),
                "geometry requires POSITION"
            );
            for b in g.attributes.values().chain(g.indices.iter()) {
                ensure!(
                    self.buffers.contains_key(b),
                    "{}",
                    diagnostics::unknown("buffer", b, self.buffers.keys().map(String::as_str))
                );
            }
            Ok(())
        }).filter_map(Result::err));
        errors.extend(self.animation.iter().map(|l| -> Result<()> {
            ensure!(
                l.tracks.is_some() != l.clip.is_some(),
                "layer requires exactly one of tracks or clip"
            );
            Ok(())
        }).filter_map(Result::err));
        if let Err(error) = self.validate_image_sources(&names) {errors.push(error);}
        if let Err(error) = self.validate_media_clocks() {errors.push(error);}
        if let Err(error) = self.validate_resources(&names,require_camera) {errors.push(error);}
        diagnostics::joined(errors)
    }
}

pub(super) fn validate_name(s: &str) -> Result<()> {
    ensure!(!s.is_empty(), "name must not be empty");
    // '/' joins an asset name and an imported node name in diagnostics and IDs.
    ensure!(!s.contains('/'), "name {s:?}: '/' is not allowed in names");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expanded_value_admission_preserves_numbers_names_and_locations() {
        let authored = br#"{
            "version":2,
            "render":{"duration":1.2300},
            "nodes":[{"type":"camera","name":"camera","translation":[0.1,-0.0,0.84551240822557006],
                "material":{"program":"std/camera","inputs":{"lens":35,"exact":9007199254740993,"tiny":1e-1000,"exponent":1.234e+20}}}]
        }"#;
        let value: serde_json::Value = serde_json::from_slice(authored).unwrap();
        let bytes = serde_json::to_vec(&value).unwrap();
        for file in [None, Some(Path::new("project/scene.json"))] {
            let from_bytes = Scene::parse_json_with_source(&bytes, authored, file).unwrap();
            let from_value = Scene::parse_value_with_source(value.clone(), authored, file, || Ok(value.clone())).unwrap();
            assert_eq!(serde_json::to_vec(&from_value).unwrap(), serde_json::to_vec(&from_bytes).unwrap());
            assert_eq!(from_value.origins, from_bytes.origins);
            assert!(!from_value.origins.is_empty());
        }
    }

    #[test]
    fn expanded_value_admission_preserves_input_depth_and_error_precedence() {
        let authored = br#"{"version":2,"nodes":[]}"#;
        let file = Some(Path::new("project/scene.json"));
        for depth in [127, 128, 129, 256] {
            let mut nested = serde_json::Value::Null;
            // The scene object contributes the final container level. Alternate
            // arrays and objects to exercise both branches of the bounded walk.
            for level in 1..depth {
                nested = if level % 2 == 0 { serde_json::json!({"nested":nested}) }
                    else { serde_json::Value::Array(vec![nested]) };
            }
            let value = serde_json::json!({"version":2,"render":{"settings":{}},"expanded":nested});
            let bytes = serde_json::to_vec(&value).unwrap();
            let expected = Scene::parse_json_with_source(&bytes, authored, file).unwrap_err();
            let again = value.clone();
            let actual = Scene::parse_value_with_source(value, authored, file, || Ok(again)).unwrap_err();
            assert_eq!(format!("{actual:#}"), format!("{expected:#}"), "depth {depth}");
            assert_eq!(actual.to_string().contains("recursion limit exceeded"), depth >= 128);
            if depth < 128 { assert!(actual.to_string().contains("render.settings")); }
        }
    }

    #[test]
    fn expanded_value_admission_accepts_deep_nodes_below_input_limit() {
        let authored = br#"{"version":2,"nodes":[]}"#;
        // Root object + nodes array + 62 pairs of group/children containers +
        // the camera object reach depth 127, immediately below the parser limit.
        let mut node = serde_json::json!({"type":"camera"});
        for _ in 0..62 { node = serde_json::json!({"type":"node","children":[node]}); }
        let value = serde_json::json!({"version":2,"nodes":[node]});
        let bytes = serde_json::to_vec(&value).unwrap();
        let expected = Scene::parse_json_with_source(&bytes, authored, None).unwrap();
        let again = value.clone();
        let actual = Scene::parse_value_with_source(value, authored, None, || Ok(again)).unwrap();
        assert_eq!(serde_json::to_vec(&actual).unwrap(), serde_json::to_vec(&expected).unwrap());
        assert_eq!(actual.origins, expected.origins);
    }

    #[test]
    fn expanded_value_admission_preserves_pre_lowering_errors() {
        let authored = br#"{"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera"}}]}"#;
        for extra in [
            serde_json::json!({"geometry":{"expanded":{"attributes":{"POSITION":42}}}}),
            serde_json::json!({"nodes":[{"type":"camera","material":{"program":42}}]}),
            serde_json::json!({"animation":[{"blend":"add"}]}),
        ] {
            let mut value: serde_json::Value = serde_json::from_slice(authored).unwrap();
            value.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let bytes = serde_json::to_vec(&value).unwrap();
            let file = Some(Path::new("project/scene.json"));
            let expected = Scene::parse_json_with_source(&bytes, authored, file).unwrap_err();
            let again = value.clone();
            let actual = Scene::parse_value_with_source(value, authored, file, || Ok(again)).unwrap_err();
            assert_eq!(format!("{actual:#}"), format!("{expected:#}"));
        }
    }

    #[test]
    fn object_key_order_does_not_change_admission_diagnostics() {
        let first = br#"{"version":2,"nodes":[{"zzz":1,"aaa":2}]}"#;
        let second = br#"{"nodes":[{"aaa":2,"zzz":1}],"version":2}"#;
        assert_eq!(
            format!("{:#}", Scene::from_json(first).unwrap_err()),
            format!("{:#}", Scene::from_json(second).unwrap_err()),
        );
    }

    #[test]
    fn unbuilt_two_d_fields_are_refused_in_one_line() {
        let camera = serde_json::json!({"type":"camera"});
        let text = serde_json::json!({"type":"text","content":"Hi","font":"a.ttf"});
        for (document, line) in [
            (serde_json::json!({"version":2,"nodes":[camera,text]}), "scene.nodes[1]: type \"text\": text in a 3D world is not built yet; put it in a canvas (composition {space: \"2d\"})"),
            (serde_json::json!({"version":2,"nodes":[camera,{"type":"path","d":"M0 0H1"}]}), "scene.nodes[1]: type \"path\": a path in a 3D world is not built yet; put it in a canvas (composition {space: \"2d\"}), or extrude it with std extrude"),
            (serde_json::json!({"version":2,"nodes":[camera,{"children":[{"opacity":0.5}]}]}), "scene.nodes[1].children[0].opacity: 2D is not built yet in a 3D world; put it in a canvas (composition {space: \"2d\"})"),
            (serde_json::json!({"version":2,"nodes":[camera,{"clip-path":"shape"}]}), "scene.nodes[1].clip-path: 2D is not built yet in a 3D world; put it in a canvas (composition {space: \"2d\"})"),
            (serde_json::json!({"version":2,"nodes":[camera,{"mask":"shape"}]}), "scene.nodes[1].mask: 2D is not built yet in a 3D world; put it in a canvas (composition {space: \"2d\"})"),
            (serde_json::json!({"version":2,"nodes":[camera,{"composition":{"space":"3d"}}]}), "scene.nodes[1].composition: a world inside a 3D world is not built yet; its nodes are already in the world"),
            (serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[camera]}), "scene.nodes[0]: type \"camera\": a 2D canvas holds group, text and path nodes; camera in a canvas is not built yet; put it in a world (composition {space: \"3d\"})"),
            (serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[{"type":"text","content":[{"content":"a","fill":{"program":"p"}}],"font":"a.ttf"}]}), "scene.nodes[0].content[0].fill: a program paint is not built yet; write a CSS colour or gradient, or {\"image\": name}"),
            (serde_json::json!({"version":2,"composition":{"space":"2d"},"render":{"background":[0,0,0]},"nodes":[text]}), "render.background: a 2D canvas's background is \"transparent\" or a CSS colour"),
            (serde_json::json!({"version":2,"composition":{"space":"2d"},"render":{"camera":"c"},"nodes":[text]}), "render.camera: not built for a 2D canvas yet; a canvas takes render.resolution, fps, duration and background"),
            (serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[{"type":"text","content":"a","font":"a.ttf","white-space":"pre"}]}), "node \"\": white-space \"pre\" is not built yet; normal, nowrap and pre-line are"),
        ] {
            let error = Scene::from_json(&serde_json::to_vec(&document).unwrap()).unwrap_err().to_string();
            assert_eq!(error, line);
        }
        let canvas = Scene::from_json(&serde_json::to_vec(&serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[{"name":"group","children":[text]}]})).unwrap()).unwrap();
        assert!(canvas.is_canvas());
    }
    #[test]
    fn an_ordinary_node_serializes_without_node_time() {
        // The release refuses `active` at parse, so a written scene must not gain a default one.
        let scene = Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera","name":"cam"},{"name":"n"}]}"#).unwrap();
        let written = serde_json::to_value(&scene).unwrap();
        assert!(written["nodes"].as_array().unwrap().iter().all(|node| node.get("active").is_none()), "{written}");
        let window: Node = serde_json::from_value(serde_json::json!({"name": "n", "active": [0, 2]})).unwrap();
        assert_eq!(serde_json::to_value(&window).unwrap()["active"], serde_json::json!([0, 2]));
    }
    #[test]
    fn node_time_is_read_from_the_document() {
        // Section 3's `active`, `time` and node `animation`, and scene tracks keying a node's clock (rule 8).
        let scene = Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera","name":"cam"},
            {"name":"shot","active":[1,3],"time":{"offset":2,"speed":2},"animation":[{"tracks":[{"path":["translation"],"keys":[[0,[0,0,0]],[1,[1,0,0]]]}]}]},
            {"name":"n","time":{}}],
            "animation":[{"tracks":[{"path":["nodes","n","active"],"keys":[[0,true],[1,false]]},{"path":["nodes","n","time","speed"],"keys":[[0,1],[2,3]]}]}]}"#).unwrap();
        let shot = scene.nodes.iter().find(|node| node.name == "shot").unwrap();
        assert!(shot.has_clock() && shot.time.is_some() && shot.animation.len() == 1);
        let schema = crate::v2::schema().to_value().to_string();
        for field in ["active", "time", "animation"] {
            assert!(schema.contains(&format!("node.{field}")), "{field} is in the schema");
        }
    }
    #[test]
    fn editor_schema_is_optional_metadata() {
        for url in [crate::SCHEMA_URL, "https://voyager.so/vork/schema/scene.schema.json", "custom editor metadata"] {
            let input = serde_json::json!({"version": 2, "nodes": [], "$schema": url});
            let scene: Scene = serde_json::from_value(input).unwrap();
            assert_eq!(scene.schema.as_deref(), Some(url));
            assert_eq!(serde_json::to_value(scene).unwrap()["$schema"], url);
        }
        assert!(serde_json::from_value::<Scene>(serde_json::json!({"version": 2, "nodes": [], "$schem": "typo"})).is_err());
    }

    #[test]
    fn propagation_is_only_program_and_ordered_bounds() {
        let scene=serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "programs":{"step":{"source":"struct Inputs {};"}},
            "propagation":{"program":"step","bounds":[[-1,-1,-1],[1,1,1]]}});
        assert!(Scene::from_json(&serde_json::to_vec(&scene).unwrap()).is_ok());
        for (field,value) in [("capture_distance",serde_json::json!(1)),("physical_scale",serde_json::json!(1)),
            ("radiance_scale",serde_json::json!(1)),("program",serde_json::json!("missing")),
            ("bounds",serde_json::json!([[1,-1,-1],[1,1,1]]))] {
            let mut invalid=scene.clone();invalid["propagation"][field]=value;
            assert!(Scene::from_json(&serde_json::to_vec(&invalid).unwrap()).is_err(),"{invalid}");
        }
    }

    #[test]
    fn retired_technique_intent_is_not_scene_content() {
        for extra in [serde_json::json!({"requires":["path"]}),serde_json::json!({"materials":{"glass":{"thickness_mode":"slab"}}}),serde_json::json!({"programs":{"p":{"source":"import vork;","passes":[]}}}),serde_json::json!({"render":{"compositor_per_step":[]}})] {
            let mut scene=serde_json::json!({"version":2,"nodes":[{"type":"camera"}]});
            scene.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            assert!(Scene::from_json(&serde_json::to_vec(&scene).unwrap()).is_err(),"{scene}");
        }
    }
    #[cfg(all(test, vork_private_tests))]
    include!("mod_unavailable_path_volume_budget_is_absent_and_rejected_private.rs");

    #[test]
    fn camera_cut_tracks_defer_start_selection_to_animation_evaluation() {
        let text = br#"{"version":2,"nodes":[{"type":"camera","name":"wide"},{"type":"camera","name":"close"}],
            "animation":[{"tracks":[{"path":["render","camera"],"keys":[[0,"close"],[2,"wide"]]}]}]}"#;
        assert_eq!(Scene::from_json(text).unwrap().render.camera, None);
        let without = br#"{"version":2,"nodes":[{"type":"camera","name":"wide"},{"type":"camera","name":"close"}]}"#;
        assert!(Scene::from_json(without).unwrap_err().to_string().contains("requires render.camera"));
    }
    #[test]
    fn inactive_cut_layers_leave_the_unambiguous_camera_default() {
        for weight in [0., 0.25] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({"version":2,
                "nodes":[{"type":"camera","name":"main"},{"type":"asset","name":"rig","file":"imported-camera.gltf"}],
                "animation":[{"weight":weight,"tracks":[{"path":["render","camera"],"keys":[[0,"rig/shot"]]}]}]
            })).unwrap()).unwrap();
            assert_eq!(scene.render.camera, None, "weight={weight}");
        }
    }
    #[test]
    fn offset_cut_layers_still_name_a_camera() {
        for offset in [-2., 2.] {
            let scene = Scene::from_json(&serde_json::to_vec(&serde_json::json!({"version":2,
                "nodes":[{"type":"camera","name":"a"},{"type":"camera","name":"b"}],
                "animation":[{"offset":offset,"tracks":[{"path":["render","camera"],"keys":[[0,"b"]]}]}]
            })).unwrap()).unwrap();
            assert_eq!(scene.render.camera, None);
        }
    }
    #[test]
    fn scene_json_accepts_a_leading_byte_order_mark() {
        let json = br#"{"version":2,"nodes":[{"type":"camera","name":"camera"}]}"#;
        let scene = Scene::from_json(&[b"\xEF\xBB\xBF".as_slice(), json].concat()).unwrap();
        assert_eq!(scene.nodes[0].name, "camera");
    }
    fn parse(v: serde_json::Value) -> Result<Scene> {
        let scene: Scene = crate::v2::from_test_value(v)?;
        scene.validate()?;
        Ok(scene)
    }
    #[test]
    fn probe_world_bounds_are_rejected_in_profiles() {
        for bounds in [serde_json::json!([[-4,-2,-3],[6,5,8]]), serde_json::Value::Null] {
            let error = Profile::parse("test", &serde_json::to_vec(&serde_json::json!({
                "version":1,"level":1,"settings":{"probes":{"volume":{"bounds":bounds}}}
            })).unwrap()).unwrap_err().to_string();
            assert!(error.contains("probes") && error.contains("unknown field"), "{error}");
        }
    }
    #[test]
    fn names_report_the_specific_failed_rule() {
        let check = |names: &[&str]| {
            let nodes: Vec<_> = names
                .iter()
                .map(|name| serde_json::json!({"type":"camera","name":name}))
                .collect();
            parse(serde_json::json!({"version":2,"nodes":nodes}))
                .unwrap_err()
                .to_string()
        };
        assert!(check(&["car/wheel"]).contains("'/' is not allowed"));
        assert!(check(&["camera", "camera"]).contains("duplicate node name"));
        parse(serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"Cube.001","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"materials":{"a.b":{}}}))
            .unwrap();
    }
    #[test]
    fn rejects_unknown_fields_duplicate_names_and_ambiguous_sources() {
        let valid = serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}}]});
        parse(valid.clone()).unwrap();
        for invalid in [
            serde_json::json!({"version":2,"world":{},"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}]}),
            serde_json::json!({"version":2,"nodes":[{"type":"camera","look_att":[0,0,0],"material":{"program":"std/camera","inputs":{"projection":"persp"}}}]}),
            serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"a","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"node","name":"a"}]}),
            serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"buffers":{"p":{"type":"vec3","data":[],"file":"p.bin"}}}),
            serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"programs":{"p":{}}}),
            serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],"animation":[{"tracks":[{"path":["render","camera"],"keys":[[1,"a"],[0,"b"]]}]}]}),
        ] {
            assert!(parse(invalid.clone()).is_err(), "accepted {invalid}");
        }
    }
    #[test]
    fn rejects_unimplemented_pass_fields() {
        let error = parse(serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "render":{"pass_cryptomatte_depth":6}})).unwrap_err().to_string();
        assert!(error.contains("pass_cryptomatte_depth") && error.contains("unknown"), "{error}");
    }
    #[test]
    fn aov_names_never_shadow_built_in_channels() {
        let scene = |name: &str| parse(serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "render":{"aovs":[{"name":name,"type":"value"}]}}));
        scene("heat").unwrap();
        for name in OUTPUT_CHANNELS {
            let error = scene(name).unwrap_err().to_string();
            assert!(error.contains("reserved"), "{name}: {error}");
        }
    }
    #[test]
    fn image_handles_reject_sampling_options_for_every_program() {
        let scene = |program: &str, texture: serde_json::Value| {
            parse(serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
                "images":{"n":{"file":"n.png"}},"programs":{"p":{"source":"import vork;"}},
                "materials":{"m":{"program":program,"inputs":{"specular_roughness":texture}}}}))
        };
        scene("p", serde_json::json!({"image":"n"})).unwrap();
        scene("std/openpbr", serde_json::json!({"image":"n"})).unwrap();
        for option in [serde_json::json!({"factor":0.5}), serde_json::json!({"channel":"r"}), serde_json::json!({"tex_coord":1}),
            serde_json::json!({"scale":[2,2]}), serde_json::json!({"offset":[0.5,0]}), serde_json::json!({"rotation":1})] {
            let mut texture = option.clone();
            texture["image"] = "n".into();
            for program in ["std/openpbr", "p"] {
                assert!(scene(program, texture.clone()).is_err(), "{program}: {texture}");
            }
        }
    }
    #[test]
    fn removed_program_diagnostics_route_to_upgrade() {
        for program in ["std/fog", "blender/principled_volume", "post/vignette", "post/lens_distortion", "post/glare"] {
            // Public builds reject removed programs even in unused definitions.
            assert!(crate::libraries::module_source(program).is_none());
            for field in ["materials", "display"] {
                let mut value = serde_json::json!({"version":2,"nodes":[{"type":"camera"}]});
                if field == "materials" {
                    value["materials"] = serde_json::json!({"unused":{"program":program}});
                } else {
                    value["composition"] = serde_json::json!({"space":"3d","view":{"program":program}});
                }
                let error = parse(value).unwrap_err().to_string();
                assert!(error.contains("vork help upgrade"), "{program} {field}: {error}");
            }
        }
    }
    #[test]
    fn removed_principled_surface_suggests_openpbr() {
        let error = parse(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
            "materials":{"m":{"program":"blender/principled"}}})).unwrap_err().to_string();
        assert!(error.contains("unknown program \"blender/principled\"")
            && error.contains("use std/openpbr"), "{error}");
        assert!(crate::libraries::module_source("blender/principled").is_none());
        assert!(crate::libraries::module_source("blender/principled_volume").is_none());
    }
    #[test]
    fn camera_material_uses_shared_inline_material() {
        let scene=Scene::from_json(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","name":"camera","material":{
                "program":{"source":"struct Inputs {float start=0.;}; CameraRay camera(CameraContext c,Inputs i) {CameraRay r;r.origin=c.origin;r.direction=c.forward;r.t_min=0.;r.t_max=1000.;return r;} float3 sensor(SensorContext c,Inputs i) {return c.path_length>=i.start ? 1.:0.;}"},
                "inputs":{"start":2}}}]}).to_string().as_bytes()).unwrap();
        let NodeKind::Camera {material:Some(name),..}=&scene.nodes[0].kind else {panic!("camera material missing")};
        assert_eq!(scene.materials[name].inputs["start"],Input::Number(2.));
        assert!(parse(serde_json::json!({"version":2,"nodes":[{"type":"camera","material":"missing"}]})).is_err());
        assert!(parse(serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"render":{"path_length_gate":{"start":0,"end":1}}})).is_err());
    }

    #[test]
    fn openpbr_inputs_are_exactly_the_specification_table() {
        assert_eq!(OPENPBR.len(), 39);
        let line = include_str!("reference.txt").lines()
            .find(|l| l.starts_with("material.inputs (std/openpbr): OpenPBR Surface; shaded inputs:")).unwrap();
        let listed = line.split("shaded inputs: ").nth(1).unwrap().split_whitespace().collect::<Vec<_>>();
        assert_eq!(listed, OpenPbrParam::supported().map(|p| p.name).collect::<Vec<_>>());
        for param in OPENPBR {
            let path = format!("material.inputs.{}: ", param.name);
            let records = include_str!("reference.txt").lines().filter(|l| l.starts_with(&path)).collect::<Vec<_>>();
            assert_eq!(records.len(), usize::from(param.supported), "{}", param.name);
            if param.supported && param.kind == OpenPbrKind::Vector3 {
                assert!(records[0].contains("constant or texture; animatable"), "{}", records[0]);
            }
        }
        let p = |n| OpenPbrParam::find(n).unwrap();
        assert_eq!(p("specular_roughness").default, &[0.3]);
        assert_eq!(p("coat_ior").default, &[1.6]);
        assert_eq!(p("transmission_dispersion_scale").default, &[0.]);
        assert_eq!(p("transmission_dispersion_abbe_number").default, &[20.]);
        assert!(!p("transmission_dispersion_abbe_number").allows(0.));
        assert!(!p("specular_ior").allows(0.) && p("specular_ior").allows(4.));
        let scene = |inputs: serde_json::Value| {
            parse(serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
                "images":{"n":{"file":"n.png"}},"materials":{"m":{"inputs":inputs}}}))
        };
        for valid in [
            serde_json::json!({"geometry_coat_normal":{"image":"n"}}),
            serde_json::json!({"fuzz_weight":1,"base_color":[1,0,0],"geometry_thin_walled":true}),
            serde_json::json!({"geometry_normal":{"image":"n"},"emission_luminance":{"image":"n"}}),
        ] {
            scene(valid.clone()).unwrap_or_else(|e| panic!("{valid}: {e}"));
        }
        for invalid in [
            serde_json::json!({"sheen_color":[1,1,1]}),
            serde_json::json!({"geometry_thickness":0.1}),
            serde_json::json!({"specular_anisotropy":0.5}),
            serde_json::json!({"base_color":0.5}),
            serde_json::json!({"base_metalness":2}),
            serde_json::json!({"geometry_thin_walled":1}),
            serde_json::json!({"base_color":"missing"}),
        ] {
            assert!(scene(invalid.clone()).is_err(), "accepted {invalid}");
        }
        for (wrong, meant) in [
            ("roughness", "specular_roughness"),
            ("metallic", "base_metalness"),
            ("sheen_color", "fuzz_color"),
        ] {
            let error = format!("{:#}", scene(serde_json::json!({wrong: 0.5})).unwrap_err());
            assert!(
                error.contains(&format!("did you mean {meant:?}?")),
                "{error}"
            );
        }
    }
}

#[cfg(test)]
mod standard_defaults_tests {
    use super::*;
    #[test]
    fn defaults_match_blender_521_factory_and_add_object_receipts() {
        let b: serde_json::Value = crate::v2::from_test_str(include_str!(
            "../../tests/fixtures/blender-5.2.1-defaults.json"
        ))
        .unwrap();
        let lights: serde_json::Value = crate::v2::from_test_str(include_str!(
            "../../tests/fixtures/blender-5.2.1-light-objects.json"
        ))
        .unwrap();
        let light: Node =
            crate::v2::from_test_value(serde_json::json!({"type":"light","shape":"rectangle"})).unwrap();
        let l = serde_json::to_value(light).unwrap();
        assert_eq!(l["size"], lights["AREA"]["size"]);
        assert!(l.get("shadow").is_none());
        assert_eq!(
            LightShape::Sun.default_energy(),
            lights["SUN"]["energy"].as_f64().unwrap() as f32
        );
        assert_eq!(
            LightShape::Point.default_energy(),
            lights["POINT"]["energy"].as_f64().unwrap() as f32
        );
        for (shape, source) in [("point", "POINT"), ("spot", "SPOT")] {
            let light = serde_json::to_value(crate::v2::from_test_value::<Node>(serde_json::json!({"type":"light","shape":shape})).unwrap()).unwrap();
            assert_eq!(light["radius"], lights[source]["shadow_soft_size"]);
        }
        let spot = serde_json::to_value(crate::v2::from_test_value::<Node>(serde_json::json!({"type":"light","shape":"spot"})).unwrap()).unwrap();
        assert_eq!(LightShape::Spot.default_energy(), lights["SPOT"]["energy"].as_f64().unwrap() as f32);
        assert_eq!(spot["spot_size"].as_f64().unwrap() as f32, b["lights"]["SPOT"]["spot_size"].as_f64().unwrap() as f32);
        assert_eq!(spot["spot_blend"].as_f64().unwrap() as f32, b["lights"]["SPOT"]["spot_blend"].as_f64().unwrap() as f32);
        let r = Render::default();
        assert_eq!(r.fps.as_f32(), b["render"]["fps"].as_f64().unwrap() as f32);
        assert_eq!(r.duration.as_f32(), 250. / r.fps.as_f32());
        assert_eq!((r.duration.exact()*vork_time::exact::decimal(&r.fps.to_string()).unwrap()).ceil().to_integer(),250.into());

        assert_eq!(
            constants::ENVIRONMENT_COLOR[0],
            b["world_background"]["Color"][0].as_f64().unwrap() as f32
        );
    }
}

pub use constants::{
    ENVIRONMENT_COLOR, LIGHT_TEMPERATURE_MAX, LIGHT_TEMPERATURE_MIN, MAX_RENDER_SAMPLES, OUTPUT_CHANNELS,
    RENDER_SAMPLES, MOTION_BLUR_SHUTTER,
};

#[cfg(test)]
mod light_link_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn all_groups_and_link_names_are_checked_before_install() {
        let mut value = json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"node","light_link_tags":(0..130).map(|i|format!("g{i}")).collect::<Vec<_>>()},{"type":"asset","file":"unused.glb","nodes":{"Wheel":{"light_link_tags":["extra"]}}}]});
        let scene: Scene = crate::v2::from_test_value(value.clone()).unwrap();
        scene.validate().unwrap();
        value["nodes"][2] = json!({"type":"light","shape":"point","illuminates":["missing"]});
        let error = crate::v2::from_test_value::<Scene>(value)
            .unwrap()
            .validate()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("light_link_tags entry") && error.contains("valid names"),
            "{error}"
        );
    }
}

#[cfg(test)]
mod launch_api_tests {
    #[test]
    fn world_medium_is_a_public_inline_material_field() {
        let scene=super::Scene::from_json(br#"{"version":2,"nodes":[{"type":"camera"},{"type":"light","shape":"environment","medium":{"program":{"source":"struct Inputs {}; MediumCoefficients medium(MediumContext c, Inputs i) { return MediumCoefficients(); }"}}}]}"#).unwrap();
        assert!(matches!(&scene.nodes[1].kind,super::NodeKind::Light {medium:Some(_),..}));
        let schema=super::schema().to_value().to_string();
        assert!(schema.contains("light.medium"));
        assert!(schema.contains("\"medium\":"));
        assert!(!schema.contains("\"volume\""));
    }
}

#[cfg(test)]
mod geometry_stage_tests {
    use super::*;
    #[test]
    fn stage_list_default_order_roundtrip_and_errors() {
        let empty=serde_json::json!({"version":2,"nodes":[{"type":"camera"}]});
        let baseline=Scene::from_json(&serde_json::to_vec(&empty).unwrap()).unwrap();
        assert_eq!(baseline.geometry_stages,GeometryStage::DEFAULT);
        for names in [vec!["deform","compute","displace"],vec!["displace","deform","compute"],vec![]] {
            let mut value=empty.clone();value["geometry_stages"]=serde_json::json!(names);
            let scene=Scene::from_json(&serde_json::to_vec(&value).unwrap()).unwrap();
            assert_eq!(serde_json::to_value(scene).unwrap()["geometry_stages"],value["geometry_stages"]);
        }
        let mut value=empty.clone();value["geometry_stages"]=serde_json::json!(["compute","compute"]);
        let error=format!("{:#}",Scene::from_json(&serde_json::to_vec(&value).unwrap()).unwrap_err());
        assert!(error.contains("geometry_stages[1]") && error.contains("remove the duplicate"),"{error}");
        value["geometry_stages"]=serde_json::json!(["skin"]);
        let error=format!("{:#}",Scene::from_json(&serde_json::to_vec(&value).unwrap()).unwrap_err());
        assert!(error.contains("deform") && error.contains("compute") && error.contains("displace"),"{error}");
    }
}

#[cfg(test)]
pub(crate) const COPY_COMPOSITOR: &str = "struct Inputs {Image original;Image source;Image texture;}; float4 composite(CompositeContext c,Inputs i){return compositor_sample(c.uv);}";
